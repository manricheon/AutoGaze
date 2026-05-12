from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from autogaze_ext.investigation.quick_start_reference import QuickStartLocation, locate_quick_start
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.scaling import resolve_autogaze_scaling_policy
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import
from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer


COMPONENT_CONFIG_PATHS = {
    "autogaze": ("model", "autogaze"),
    "vision_encoder": ("model", "vision_encoder"),
    "mllm": ("model", "mllm"),
}


@dataclass(frozen=True)
class StageReport:
    name: str
    status: str
    latency_ms: float | None = None
    module_path: str | None = None
    class_or_factory: str | None = None
    output_shape: list[int] | None = None
    skipped_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SmokeInferenceReport:
    experiment_id: str
    mode: str
    status: str
    quick_start_found: bool
    quick_start_path: str | None
    quick_start_reference_found: bool
    quick_start_reference_path: str | None
    input_shape: list[int]
    frame_count: int
    original_resolution: list[int]
    target_resolution: int
    processed_resolution: list[int]
    scaling: dict[str, Any]
    autogaze_enabled: bool
    query_text: str | None
    selected_token_count: int | None
    original_visual_token_count: int | None
    vision_feature_shape: list[int] | None
    mllm_input_shape: list[int] | None
    output_text: str | None
    output_dir: str
    visualization_dir: str | None
    latency_ms: float
    peak_vram_mb: float | str
    stages: list[StageReport]
    skipped_stages: list[dict[str, str]]
    artifacts: dict[str, str]
    device: str
    effective_device: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        resolved = OmegaConf.to_container(config, resolve=True)
    else:
        resolved = dict(config)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved config must be a mapping")
    return resolved


def _get_nested(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _component_node(cfg: Mapping[str, Any], component: str) -> dict[str, Any]:
    value = _get_nested(cfg, COMPONENT_CONFIG_PATHS[component])
    return dict(value) if isinstance(value, Mapping) else {}


def _experiment_id(cfg: Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment")
    if isinstance(experiment, Mapping) and experiment.get("id"):
        return str(experiment["id"])
    return "unknown"


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be one of: float32, float16, bfloat16")


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape"):
        return list(value.shape)
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_object(
    module_path: str | None,
    class_or_factory: str | None,
    *,
    import_module_fn: ImportModuleFn,
) -> tuple[Any | None, str | None]:
    resolution = resolve_import(module_path, class_or_factory, import_module_fn=import_module_fn)
    if not resolution.ready:
        return None, resolution.error
    module = import_module_fn(str(module_path))
    cursor: Any = module
    for part in str(class_or_factory).split("."):
        cursor = getattr(cursor, part)
    return cursor, None


def _construction_prerequisite(
    node: Mapping[str, Any],
    component: str,
    *,
    import_module_fn: ImportModuleFn,
) -> dict[str, Any]:
    module_path = str(node.get("module_path")) if node.get("module_path") else None
    class_or_factory = str(node.get("class_or_factory")) if node.get("class_or_factory") else None
    resolution = resolve_import(module_path, class_or_factory, import_module_fn=import_module_fn)
    return {
        "component": component,
        "construction_level": 1,
        "module_path": module_path,
        "class_or_factory": class_or_factory,
        "passed": resolution.ready,
        "failure_reason": resolution.error,
    }


def _call_from_pretrained(factory: Any, source: str, node: Mapping[str, Any]) -> Any:
    kwargs = dict(node.get("construction_kwargs", {})) if isinstance(node.get("construction_kwargs"), Mapping) else {}
    kwargs.update(dict(node.get("extra_kwargs", {})) if isinstance(node.get("extra_kwargs"), Mapping) else {})
    call_kwargs = {
        "local_files_only": bool(node.get("local_files_only", True)),
        "trust_remote_code": bool(node.get("trust_remote_code", False)),
        **kwargs,
    }
    try:
        return factory.from_pretrained(source, **call_kwargs)
    except TypeError:
        fallback_kwargs = dict(kwargs)
        try:
            return factory.from_pretrained(source, local_files_only=bool(node.get("local_files_only", True)), **fallback_kwargs)
        except TypeError:
            return factory.from_pretrained(source, **fallback_kwargs)


def _load_model_object(
    node: Mapping[str, Any],
    *,
    import_module_fn: ImportModuleFn,
) -> tuple[Any | None, str | None]:
    factory, error = _resolve_object(
        str(node.get("module_path")) if node.get("module_path") else None,
        str(node.get("class_or_factory")) if node.get("class_or_factory") else None,
        import_module_fn=import_module_fn,
    )
    if error:
        return None, error

    source = node.get("checkpoint") or node.get("model_config_path")
    if hasattr(factory, "from_pretrained"):
        if not source:
            return None, "checkpoint or model_config_path is required for from_pretrained construction"
        try:
            return _call_from_pretrained(factory, str(source), node), None
        except Exception as exc:
            return None, f"from_pretrained construction failed for {source}: {exc}"

    kwargs = dict(node.get("construction_kwargs", {})) if isinstance(node.get("construction_kwargs"), Mapping) else {}
    try:
        return factory(**kwargs), None
    except Exception as exc:
        return None, f"model construction failed: {exc}"


def _try_eval_to_device(model: Any, device: torch.device, dtype: torch.dtype) -> None:
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        try:
            model.to(device=device, dtype=dtype)
        except TypeError:
            model.to(device)


def _dummy_video(num_frames: int, resolution: int) -> tuple[torch.Tensor, dict[str, Any]]:
    video = torch.linspace(0, 1, steps=num_frames * 3 * resolution * resolution, dtype=torch.float32)
    video = video.reshape(1, num_frames, 3, resolution, resolution)
    metadata = {
        "source": "dummy",
        "sampled_frame_indices": list(range(num_frames)),
        "original_resolution": [resolution, resolution],
        "video_input_format": "[B, T, C, H, W]",
    }
    return video, metadata


def _tensor_from_raw_video(raw_video: Any) -> torch.Tensor:
    if isinstance(raw_video, torch.Tensor):
        tensor = raw_video
    else:
        tensor = torch.as_tensor(raw_video)
    if tensor.ndim != 4:
        raise ValueError(f"Expected decoded video shape [T, H, W, C] or [T, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] not in {1, 3}:
        raise ValueError(f"Could not infer channel dimension from decoded video shape {tuple(tensor.shape)}")
    return tensor.to(torch.float32) / 255.0 if tensor.max() > 1 else tensor.to(torch.float32)


def _local_video(video_path: str | Path, num_frames: int, resolution: int) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"video path does not exist: {path}")
    try:
        import av  # type: ignore
        from autogaze.datasets.video_utils import read_video_pyav  # type: ignore
    except Exception as exc:
        raise RuntimeError("local video preprocessing requires PyAV and original autogaze.datasets.video_utils") from exc

    container = av.open(str(path))
    try:
        sample_indices = list(range(num_frames))
        raw_video = read_video_pyav(container=container, indices=sample_indices)
    finally:
        container.close()

    video = _tensor_from_raw_video(raw_video).unsqueeze(0)
    original_resolution = [int(video.shape[-2]), int(video.shape[-1])]
    video = _resize_video(video, resolution)
    metadata = {
        "source": str(path),
        "sampled_frame_indices": sample_indices,
        "original_resolution": original_resolution,
        "video_input_format": "[B, T, C, H, W]",
    }
    return video, metadata


def _resize_video(video: torch.Tensor, resolution: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = video.shape
    if height == resolution and width == resolution:
        return video
    flattened = video.reshape(batch * frames, channels, height, width)
    resized = F.interpolate(flattened, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, resolution, resolution)


def _prepare_video(
    *,
    video: str | None,
    video_path: str | Path | None,
    num_frames: int,
    resolution: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if video_path:
        return _local_video(video_path, num_frames, resolution)
    if video in {None, "dummy"}:
        return _dummy_video(num_frames, resolution)
    raise ValueError("Only --video dummy or --video-path are supported")


def _scaling_report(
    *,
    cfg: Mapping[str, Any],
    resolution: int,
    scale_resolution: str | None,
) -> dict[str, Any]:
    autogaze_node = _component_node(cfg, "autogaze")
    vision_node = _component_node(cfg, "vision_encoder")
    a_args = autogaze_node.get("original_cli_args") if isinstance(autogaze_node.get("original_cli_args"), Mapping) else {}
    high_res = {}
    v_args = vision_node.get("original_cli_args") if isinstance(vision_node.get("original_cli_args"), Mapping) else {}
    if isinstance(v_args.get("high_resolution_example"), Mapping):
        high_res = dict(v_args["high_resolution_example"])

    target_scales = autogaze_node.get("target_scales") or a_args.get("target_scales")
    target_patch_size = autogaze_node.get("target_patch_size") or a_args.get("target_patch_size")
    if scale_resolution and not target_scales and high_res:
        target_scales = high_res.get("target_scales")
        target_patch_size = high_res.get("target_patch_size")

    status = "not_requested"
    siglip_scales = None
    policy_details: dict[str, Any] | None = None
    scaling_mode = "resize"
    spatio_temporal_modes = {
        "spatio_temporal_224",
        "spatio_temporal_392",
        "quick_start_spatio_temporal_224",
        "quick_start_spatio_temporal_392",
    }
    if scale_resolution in spatio_temporal_modes:
        scaling_mode = "spatio_temporal"
    if scale_resolution:
        status = "quick_start_target_scales_applied" if target_scales and target_patch_size else "stub_missing_target_scales"
    if scale_resolution in {"spatio_temporal_224", "quick_start_spatio_temporal_224"}:
        target_scales = None
        target_patch_size = None
        status = "spatio_temporal_chunking_utility_supported"
    if scale_resolution in {"spatio_temporal_392", "quick_start_spatio_temporal_392"} and high_res:
        target_scales = high_res.get("target_scales")
        target_patch_size = high_res.get("target_patch_size")
        status = "spatio_temporal_chunking_utility_supported"

    try:
        policy_resolution = resolution
        if scale_resolution in {"spatio_temporal_224", "quick_start_spatio_temporal_224"}:
            policy_resolution = 224
        elif scale_resolution in {"spatio_temporal_392", "quick_start_spatio_temporal_392"}:
            policy_resolution = 392
        patch_size = int(target_patch_size or v_args.get("default_patch_size") or a_args.get("default_patch_size") or 16)
        if target_patch_size:
            patch_size = int(target_patch_size)
        policy = resolve_autogaze_scaling_policy(
            mode=scaling_mode,  # type: ignore[arg-type]
            resolution=policy_resolution,
            patch_size=patch_size,
            target_scales=target_scales,
            target_patch_size=target_patch_size,
            spatial_tile_size=policy_resolution,
        )
        siglip_scales = policy.siglip_scales
        policy_details = policy.to_dict()
    except ValueError as exc:
        if scale_resolution:
            status = "unsupported_scaling_policy"
        policy_details = {"error": str(exc)}

    return {
        "requested": bool(scale_resolution),
        "mode": scale_resolution,
        "preprocess_mode": scaling_mode,
        "status": status,
        "quick_start_behavior": "target_scales_and_target_patch_size" if scale_resolution else "default_224x224_or_requested_resolution",
        "target_scales": _json_safe(target_scales),
        "target_patch_size": _json_safe(target_patch_size),
        "siglip_scales": siglip_scales,
        "any_resolution_chunking": (
            "utility_supported_not_full_pipeline_aggregated" if scaling_mode == "spatio_temporal" else "not_requested"
        ),
        "any_duration_chunking": (
            "utility_supported_not_full_pipeline_aggregated" if scaling_mode == "spatio_temporal" else "not_requested"
        ),
        "target_resolution": resolution,
        "policy": policy_details,
    }


def _autogaze_call_kwargs(node: Mapping[str, Any], scaling: Mapping[str, Any]) -> dict[str, Any]:
    original_args = node.get("original_cli_args") if isinstance(node.get("original_cli_args"), Mapping) else {}
    kwargs: dict[str, Any] = {}
    if original_args.get("gazing_ratio") is not None:
        kwargs["gazing_ratio"] = original_args["gazing_ratio"]
    if original_args.get("task_loss_requirement") is not None:
        kwargs["task_loss_requirement"] = original_args["task_loss_requirement"]
    if scaling.get("target_scales") and scaling.get("target_patch_size"):
        kwargs["target_scales"] = scaling["target_scales"]
        kwargs["target_patch_size"] = scaling["target_patch_size"]
    return kwargs


def _vision_node_with_scaling(node: Mapping[str, Any], scaling: Mapping[str, Any]) -> dict[str, Any]:
    adjusted = dict(node)
    original_args = node.get("original_cli_args") if isinstance(node.get("original_cli_args"), Mapping) else {}
    should_pass_scales = bool(original_args.get("scales_from_autogaze_config")) or str(node.get("variant")) == "modified"
    siglip_scales = scaling.get("siglip_scales")
    if siglip_scales and should_pass_scales:
        construction_kwargs = (
            dict(adjusted.get("construction_kwargs", {}))
            if isinstance(adjusted.get("construction_kwargs"), Mapping)
            else {}
        )
        construction_kwargs["scales"] = str(siglip_scales)
        adjusted["construction_kwargs"] = construction_kwargs
        adjusted["scaling_applied_to_construction"] = {
            "scales": str(siglip_scales),
            "source": "QUICK_START.md AutoGaze/SigLIP scale alignment",
        }
    return adjusted


def _shape_of(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def _extract_hidden_state(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        value = output.last_hidden_state
        return value if isinstance(value, torch.Tensor) else None
    if isinstance(output, Mapping):
        value = output.get("last_hidden_state")
        if value is None:
            value = output.get("visual_features")
        return value if isinstance(value, torch.Tensor) else None
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _extract_generated_text(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if hasattr(output, "generated_text"):
        value = output.generated_text
        if isinstance(value, list):
            return str(value[0]) if value else None
        return str(value) if value is not None else None
    if isinstance(output, Mapping):
        for key in ("generated_text", "text", "answer"):
            if output.get(key) is not None:
                return str(output[key])
    return str(output)


def _token_counts_from_gaze(
    gaze_outputs: Mapping[str, Any],
    *,
    frame_count: int,
    model: Any,
    resolution: int,
) -> tuple[int | None, int | None]:
    if_padded = gaze_outputs.get("if_padded_gazing")
    selected = None
    if isinstance(if_padded, torch.Tensor):
        selected = int((~if_padded.bool()).sum().item())
    elif isinstance(gaze_outputs.get("gazing_pos"), torch.Tensor):
        selected = int(gaze_outputs["gazing_pos"].numel())

    per_frame = getattr(model, "num_vision_tokens_each_frame", None)
    if per_frame is None and hasattr(model, "config"):
        per_frame = getattr(model.config, "num_vision_tokens_each_frame", None)
    original = int(per_frame) * frame_count if per_frame is not None else (resolution // 16) ** 2 * frame_count
    return original, selected


def _save_autogaze_outputs(
    output_dir: Path,
    gaze_outputs: Mapping[str, Any],
    *,
    original_visual_token_count: int | None,
    selected_visual_token_count: int | None,
    selected_scales: Any,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    autogaze_dir = output_dir / "autogaze"
    gazing_pos = gaze_outputs.get("gazing_pos")
    if gazing_pos is not None:
        path = autogaze_dir / "selected_patch_indices.json"
        _write_json(path, {"selected_patch_indices": _json_safe(gazing_pos)})
        artifacts["selected_patch_indices"] = str(path)
    if selected_scales is not None:
        path = autogaze_dir / "selected_scales.json"
        _write_json(path, {"selected_scales": _json_safe(selected_scales)})
        artifacts["selected_scales"] = str(path)
    token_path = autogaze_dir / "token_counts.json"
    _write_json(
        token_path,
        {
            "original_visual_token_count": original_visual_token_count,
            "selected_visual_token_count": selected_visual_token_count,
        },
    )
    artifacts["token_counts"] = str(token_path)
    return artifacts


def _maybe_visualize_autogaze(
    output_dir: Path,
    experiment_id: str,
    video: torch.Tensor,
    gaze_outputs: Mapping[str, Any],
    *,
    resolution: int,
) -> tuple[str | None, dict[str, str]]:
    gazing_pos = gaze_outputs.get("gazing_pos")
    if not isinstance(gazing_pos, torch.Tensor) or gazing_pos.numel() == 0:
        return None, {}
    patch_grid = (max(1, resolution // 16), max(1, resolution // 16))
    grid_size = patch_grid[0] * patch_grid[1]
    selected = gazing_pos.detach().cpu().flatten()[:16] % grid_size
    visualizer = AutoGazeVisualizer(output_root=output_dir.parent, exp_name=output_dir.name)
    paths = visualizer.visualize_selected_patches(video.detach().cpu(), selected, patch_grid, mode="autogaze_only", prefix=experiment_id)
    return str(output_dir / "visualizations" / "autogaze_only"), {"autogaze_visualization": str(paths[0]) if paths else ""}


def _peak_vram_mb(device: str) -> float | str:
    if device == "cuda" and torch.cuda.is_available():
        return round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 3)
    return "N/A"


def run_canonical_smoke_inference(
    *,
    experiment: str,
    mode: str,
    video: str | None = "dummy",
    video_path: str | Path | None = None,
    query_text: str | None = None,
    num_frames: int = 2,
    resolution: int = 224,
    scale_resolution: str | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    max_new_tokens: int = 1,
    allow_mllm_load: bool = False,
    output_dir: str | Path | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    quick_start_path: str | Path | None = None,
    cfg: DictConfig | Mapping[str, Any] | None = None,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> SmokeInferenceReport:
    if experiment not in {"A1_real", "A2_real"}:
        raise ValueError("experiment must be A1_real or A2_real")
    if mode not in {"autogaze_only", "full_pipeline"}:
        raise ValueError("mode must be autogaze_only or full_pipeline")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")

    start_time = time.perf_counter()
    config_dir = Path(config_dir)
    loaded_cfg = cfg if cfg is not None else load_config(config_dir, f"experiment/{experiment}")
    plain_cfg = _plain_config(loaded_cfg)
    experiment_id = _experiment_id(plain_cfg)
    if experiment_id == "unknown":
        experiment_id = experiment

    quick_start_location: QuickStartLocation | None = None
    try:
        quick_start_location = locate_quick_start(quick_start_path, repo_root=config_dir.parent)
    except FileNotFoundError:
        quick_start_location = None
    quick_start_reference = config_dir.parent / "docs" / "QUICK_START_reference.md"

    target_output_dir = Path(output_dir or _get_nested(plain_cfg, ("inference", "output_dir")) or f"outputs/{experiment_id}_{mode}")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    (target_output_dir / "logs").mkdir(parents=True, exist_ok=True)

    requested_dtype = _dtype_from_name(dtype)
    effective_device_name = device if _device_available(device) else "cpu"
    effective_device = torch.device(effective_device_name)
    video_tensor, video_metadata = _prepare_video(
        video=video,
        video_path=video_path,
        num_frames=num_frames,
        resolution=resolution,
    )
    original_resolution = list(video_metadata.get("original_resolution", [resolution, resolution]))
    video_tensor = _resize_video(video_tensor, resolution)
    processed_resolution = [int(video_tensor.shape[-2]), int(video_tensor.shape[-1])]
    video_for_models = video_tensor.to(device=effective_device, dtype=requested_dtype)
    scaling = _scaling_report(cfg=plain_cfg, resolution=resolution, scale_resolution=scale_resolution)

    stages: list[StageReport] = []
    skipped: list[dict[str, str]] = []
    artifacts: dict[str, str] = {}
    autogaze_outputs: Mapping[str, Any] | None = None
    selected_token_count: int | None = None
    original_visual_token_count: int | None = None
    vision_features: torch.Tensor | None = None
    output_text: str | None = None
    mllm_input_shape: list[int] | None = None
    visualization_dir: str | None = None

    def add_skip(stage_name: str, reason: str, node: Mapping[str, Any] | None = None) -> None:
        skipped.append({"stage": stage_name, "reason": reason})
        stages.append(
            StageReport(
                name=stage_name,
                status="skipped",
                module_path=str(node.get("module_path")) if node and node.get("module_path") else None,
                class_or_factory=str(node.get("class_or_factory")) if node and node.get("class_or_factory") else None,
                skipped_reason=reason,
            )
        )

    autogaze_node = _component_node(plain_cfg, "autogaze")
    vision_node = _component_node(plain_cfg, "vision_encoder")
    mllm_node = _component_node(plain_cfg, "mllm")
    vision_node = _vision_node_with_scaling(vision_node, scaling)
    autogaze_enabled = bool(autogaze_node.get("enabled", False))

    if device != effective_device_name:
        add_skip("device", f"requested device {device} is not available; model stages use cpu unless constructed safely")

    if autogaze_enabled:
        prereq = _construction_prerequisite(autogaze_node, "autogaze", import_module_fn=import_module_fn)
        if not prereq["passed"]:
            add_skip("autogaze", f"construction prerequisite failed: {prereq['failure_reason']}", autogaze_node)
        else:
            stage_start = time.perf_counter()
            model, error = _load_model_object(autogaze_node, import_module_fn=import_module_fn)
            if error:
                add_skip("autogaze", error, autogaze_node)
            else:
                _try_eval_to_device(model, effective_device, requested_dtype)
                kwargs = _autogaze_call_kwargs(autogaze_node, scaling)
                try:
                    with torch.inference_mode():
                        raw_output = model({"video": video_for_models}, **kwargs)
                    if not isinstance(raw_output, Mapping):
                        raise TypeError("AutoGaze output must be a mapping with QUICK_START gazing fields")
                    autogaze_outputs = raw_output
                    original_visual_token_count, selected_token_count = _token_counts_from_gaze(
                        raw_output,
                        frame_count=int(video_for_models.shape[1]),
                        model=model,
                        resolution=resolution,
                    )
                    selected_scales = raw_output.get("selected_scales", raw_output.get("scales"))
                    artifacts.update(
                        _save_autogaze_outputs(
                            target_output_dir,
                            raw_output,
                            original_visual_token_count=original_visual_token_count,
                            selected_visual_token_count=selected_token_count,
                            selected_scales=selected_scales,
                        )
                    )
                    vis_dir, vis_artifacts = _maybe_visualize_autogaze(
                        target_output_dir,
                        experiment_id,
                        video_tensor,
                        raw_output,
                        resolution=resolution,
                    )
                    visualization_dir = vis_dir
                    artifacts.update(vis_artifacts)
                    stages.append(
                        StageReport(
                            name="autogaze",
                            status="passed",
                            latency_ms=(time.perf_counter() - stage_start) * 1000,
                            module_path=str(autogaze_node.get("module_path")),
                            class_or_factory=str(autogaze_node.get("class_or_factory")),
                            output_shape=_shape_of(raw_output.get("gazing_pos")),
                            details={"call_kwargs": kwargs},
                        )
                    )
                except Exception as exc:
                    add_skip("autogaze", f"AutoGaze inference failed: {exc}", autogaze_node)
    else:
        patch_count = (resolution // 16) ** 2
        selected_token_count = int(video_for_models.shape[1]) * patch_count
        original_visual_token_count = selected_token_count
        full_indices = torch.arange(patch_count).view(1, 1, patch_count).expand(1, int(video_for_models.shape[1]), patch_count)
        autogaze_outputs = {
            "gazing_pos": full_indices.reshape(1, -1),
            "if_padded_gazing": torch.zeros((1, selected_token_count), dtype=torch.bool),
            "num_gazing_each_frame": torch.full((int(video_for_models.shape[1]),), patch_count, dtype=torch.long),
        }
        artifacts.update(
            _save_autogaze_outputs(
                target_output_dir,
                autogaze_outputs,
                original_visual_token_count=original_visual_token_count,
                selected_visual_token_count=selected_token_count,
                selected_scales=None,
            )
        )
        stages.append(
            StageReport(
                name="autogaze",
                status="disabled_full_token_path",
                output_shape=_shape_of(autogaze_outputs["gazing_pos"]),
                details={"reason": "AutoGaze OFF canonical baseline"},
            )
        )

    if mode == "full_pipeline":
        if autogaze_enabled and autogaze_outputs is None:
            add_skip("vision_encoder", "AutoGaze is enabled but gazing outputs are unavailable", vision_node)
        else:
            prereq = _construction_prerequisite(vision_node, "vision_encoder", import_module_fn=import_module_fn)
            if not prereq["passed"]:
                add_skip("vision_encoder", f"construction prerequisite failed: {prereq['failure_reason']}", vision_node)
            else:
                stage_start = time.perf_counter()
                vision_model, error = _load_model_object(vision_node, import_module_fn=import_module_fn)
                if error:
                    add_skip("vision_encoder", error, vision_node)
                else:
                    _try_eval_to_device(vision_model, effective_device, requested_dtype)
                    try:
                        with torch.inference_mode():
                            if autogaze_outputs is not None:
                                try:
                                    vision_output = vision_model(video_for_models, gazing_info=autogaze_outputs)
                                except TypeError:
                                    vision_output = vision_model(video_for_models)
                            else:
                                vision_output = vision_model(video_for_models)
                        vision_features = _extract_hidden_state(vision_output)
                        if vision_features is None:
                            raise TypeError("vision encoder output did not expose a tensor or last_hidden_state")
                        stages.append(
                            StageReport(
                                name="vision_encoder",
                                status="passed",
                                latency_ms=(time.perf_counter() - stage_start) * 1000,
                                module_path=str(vision_node.get("module_path")),
                                class_or_factory=str(vision_node.get("class_or_factory")),
                                output_shape=_shape_of(vision_features),
                                details={
                                    "gazing_info_used": bool(autogaze_outputs is not None),
                                    "autogaze_enabled": autogaze_enabled,
                                    "scaling_applied_to_construction": vision_node.get("scaling_applied_to_construction"),
                                },
                            )
                        )
                    except Exception as exc:
                        add_skip("vision_encoder", f"vision encoder inference failed: {exc}", vision_node)

        if query_text:
            mllm_input_shape = _shape_of(vision_features)
            if vision_features is None:
                add_skip("mllm", "query text was accepted, but MLLM generation was skipped because visual features are unavailable", mllm_node)
            elif not allow_mllm_load:
                add_skip(
                    "mllm",
                    "query text was accepted, but MLLM generation was skipped because --allow-mllm-load was not set",
                    mllm_node,
                )
            else:
                prereq = _construction_prerequisite(mllm_node, "mllm", import_module_fn=import_module_fn)
                if not prereq["passed"]:
                    add_skip(
                        "mllm",
                        f"query text was accepted, but MLLM generation was skipped because {prereq['failure_reason']}",
                        mllm_node,
                    )
                else:
                    stage_start = time.perf_counter()
                    mllm_model, error = _load_model_object(mllm_node, import_module_fn=import_module_fn)
                    if error:
                        add_skip("mllm", f"query text was accepted, but MLLM generation was skipped because {error}", mllm_node)
                    elif not hasattr(mllm_model, "generate"):
                        add_skip("mllm", "query text was accepted, but MLLM generation was skipped because generate() is unavailable", mllm_node)
                    else:
                        _try_eval_to_device(mllm_model, effective_device, requested_dtype)
                        try:
                            with torch.inference_mode():
                                try:
                                    mllm_output = mllm_model.generate(
                                        visual_features=vision_features,
                                        query_text=query_text,
                                        max_new_tokens=max_new_tokens,
                                    )
                                except TypeError:
                                    mllm_output = mllm_model.generate(vision_features, query_text, max_new_tokens=max_new_tokens)
                            output_text = _extract_generated_text(mllm_output)
                            stages.append(
                                StageReport(
                                    name="mllm",
                                    status="passed",
                                    latency_ms=(time.perf_counter() - stage_start) * 1000,
                                    module_path=str(mllm_node.get("module_path")),
                                    class_or_factory=str(mllm_node.get("class_or_factory")),
                                    output_shape=_shape_of(mllm_output),
                                    details={"max_new_tokens": max_new_tokens},
                                )
                            )
                        except Exception as exc:
                            add_skip("mllm", f"query text was accepted, but MLLM generation failed: {exc}", mllm_node)
        else:
            add_skip("mllm", "query text was not provided, so MLLM generation was skipped", mllm_node)

    if output_text is not None:
        answer_path = target_output_dir / "predictions" / "answer.json"
        _write_json(answer_path, {"answer": output_text, "query_text": query_text})
        artifacts["answer"] = str(answer_path)

    successful_stage_statuses = {"passed", "disabled_full_token_path"}
    status = "passed" if not skipped else (
        "partial" if any(stage.status in successful_stage_statuses for stage in stages) else "skipped"
    )
    report = SmokeInferenceReport(
        experiment_id=experiment_id,
        mode=mode,
        status=status,
        quick_start_found=quick_start_location is not None,
        quick_start_path=str(quick_start_location.path) if quick_start_location else None,
        quick_start_reference_found=quick_start_reference.exists(),
        quick_start_reference_path=str(quick_start_reference) if quick_start_reference.exists() else None,
        input_shape=[int(dim) for dim in video_tensor.shape],
        frame_count=int(video_tensor.shape[1]),
        original_resolution=[int(original_resolution[0]), int(original_resolution[1])],
        target_resolution=resolution,
        processed_resolution=processed_resolution,
        scaling=scaling,
        autogaze_enabled=autogaze_enabled,
        query_text=query_text,
        selected_token_count=selected_token_count,
        original_visual_token_count=original_visual_token_count,
        vision_feature_shape=_shape_of(vision_features),
        mllm_input_shape=mllm_input_shape,
        output_text=output_text,
        output_dir=str(target_output_dir),
        visualization_dir=visualization_dir,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        peak_vram_mb=_peak_vram_mb(effective_device_name),
        stages=stages,
        skipped_stages=skipped,
        artifacts=artifacts,
        device=device,
        effective_device=effective_device_name,
        dtype=dtype,
    )
    summary_path = target_output_dir / "logs" / "inference_summary.json"
    _write_json(summary_path, report.to_dict())
    artifacts["inference_summary"] = str(summary_path)
    return report


def print_smoke_report(report: SmokeInferenceReport) -> None:
    print("Canonical real-path smoke inference")
    print(f"experiment: {report.experiment_id}")
    print(f"mode: {report.mode}")
    print(f"status: {report.status}")
    print(f"device: {report.device}")
    print(f"effective_device: {report.effective_device}")
    print(f"dtype: {report.dtype}")
    print(f"QUICK_START.md found: {report.quick_start_found} ({report.quick_start_path or 'N/A'})")
    print(
        "QUICK_START_reference.md found: "
        f"{report.quick_start_reference_found} ({report.quick_start_reference_path or 'N/A'})"
    )
    print(f"input_shape: {report.input_shape}")
    print(f"frame_count: {report.frame_count}")
    print(f"original_resolution: {report.original_resolution}")
    print(f"processed_resolution: {report.processed_resolution}")
    print(f"scaling: {json.dumps(_json_safe(report.scaling), sort_keys=True)}")
    if report.query_text is not None:
        print(f"query_text: {report.query_text}")
    print(f"selected_token_count: {report.selected_token_count}")
    print(f"original_visual_token_count: {report.original_visual_token_count}")
    print(f"vision_feature_shape: {report.vision_feature_shape}")
    print(f"mllm_input_shape: {report.mllm_input_shape}")
    print(f"output_text: {report.output_text}")
    print(f"output_dir: {report.output_dir}")
    print(f"visualization_dir: {report.visualization_dir or 'N/A'}")
    print(f"latency_ms: {report.latency_ms:.3f}")
    print(f"peak_vram_mb: {report.peak_vram_mb}")
    print()
    for stage in report.stages:
        print(f"- {stage.name}: {stage.status}")
        if stage.module_path:
            print(f"  module: {stage.module_path}")
        if stage.class_or_factory:
            print(f"  class_or_factory: {stage.class_or_factory}")
        if stage.output_shape:
            print(f"  output_shape: {stage.output_shape}")
        if stage.latency_ms is not None:
            print(f"  latency_ms: {stage.latency_ms:.3f}")
        if stage.skipped_reason:
            print(f"  skipped_reason: {stage.skipped_reason}")
        if stage.details:
            print(f"  details: {json.dumps(_json_safe(stage.details), sort_keys=True)}")
    if report.artifacts:
        print()
        print("artifacts:")
        for key, value in sorted(report.artifacts.items()):
            print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal real canonical A1/A2 smoke inference")
    parser.add_argument("--experiment", choices=["A1_real", "A2_real"], required=True)
    parser.add_argument("--mode", choices=["autogaze_only", "full_pipeline"], required=True)
    parser.add_argument("--video", choices=["dummy"], default="dummy")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--scale-resolution", nargs="?", const="quick_start_target_scales", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--allow-mllm-load",
        action="store_true",
        help="Explicitly allow loading the configured MLLM. Disabled by default to avoid large NVILA loads.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--quick-start-path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_canonical_smoke_inference(
        experiment=args.experiment,
        mode=args.mode,
        video=args.video,
        video_path=args.video_path,
        query_text=args.query_text,
        num_frames=args.num_frames,
        resolution=args.resolution,
        scale_resolution=args.scale_resolution,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        allow_mllm_load=args.allow_mllm_load,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        quick_start_path=args.quick_start_path,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print_smoke_report(report)


if __name__ == "__main__":
    main()
