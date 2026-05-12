#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import


PathLike = str | Path


@dataclass(frozen=True)
class PocStage:
    name: str
    status: str
    module_path: str | None = None
    class_or_factory: str | None = None
    latency_ms: float | None = None
    output_shape: list[int] | None = None
    skipped_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PocPathCheck:
    name: str
    path: str | None
    exists: bool
    required: bool = True


@dataclass(frozen=True)
class PocSummary:
    mode: str
    status: str
    experiment_id: str
    output_dir: str
    device: str
    effective_device: str
    dtype: str
    checkpoint_policy: str
    query_text: str | None
    input_video: str
    input_shape: list[int] | None
    selected_token_count: int | None
    original_visual_token_count: int | None
    output_text: str | None
    stages: list[PocStage]
    path_checks: list[PocPathCheck]
    skipped_stages: list[dict[str, str]]
    artifacts: dict[str, str]
    reference: dict[str, str]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


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
        return [int(dim) for dim in value.shape]
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def _plain_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    if not isinstance(data, dict):
        raise TypeError("Config must resolve to a mapping")
    return data


def _load_config_from_path(config: PathLike) -> DictConfig:
    path = Path(config).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    if path.name.endswith(".yaml") and path.parent.name == "experiment" and path.parent.parent.name == "configs":
        return load_config(path.parent.parent, f"experiment/{path.stem}")

    cfg = OmegaConf.load(path)
    defaults = cfg.get("defaults", None)
    if defaults is not None and path.parent.name == "experiment":
        return load_config(path.parent.parent, f"experiment/{path.stem}")
    return cfg


def _get_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _component(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = _get_nested(cfg, "model", name)
    return dict(value) if isinstance(value, Mapping) else {}


def _experiment_id(cfg: Mapping[str, Any]) -> str:
    value = _get_nested(cfg, "experiment", "id")
    return str(value) if value else "unknown"


def _path_exists(path: str | None) -> bool:
    return bool(path and Path(path).expanduser().exists())


def _path_checks(cfg: Mapping[str, Any]) -> list[PocPathCheck]:
    autogaze = _component(cfg, "autogaze")
    vision = _component(cfg, "vision_encoder")
    mllm = _component(cfg, "mllm")
    autogaze_required = _autogaze_required(cfg)
    checks = [
        PocPathCheck(
            "autogaze_checkpoint",
            _as_str(autogaze.get("checkpoint")),
            _path_exists(_as_str(autogaze.get("checkpoint"))),
            required=autogaze_required,
        ),
        PocPathCheck(
            "autogaze_config",
            _as_str(autogaze.get("config_path")),
            _path_exists(_as_str(autogaze.get("config_path"))),
            required=autogaze_required,
        ),
        PocPathCheck(
            "autogaze_processor",
            _as_str(autogaze.get("processor_path") or autogaze.get("tokenizer_or_processor_path")),
            _path_exists(_as_str(autogaze.get("processor_path") or autogaze.get("tokenizer_or_processor_path"))),
            required=autogaze_required,
        ),
        PocPathCheck("siglip_checkpoint", _as_str(vision.get("checkpoint")), _path_exists(_as_str(vision.get("checkpoint")))),
        PocPathCheck("siglip_config", _as_str(vision.get("config_path")), _path_exists(_as_str(vision.get("config_path")))),
        PocPathCheck(
            "siglip_processor",
            _as_str(vision.get("processor_path") or vision.get("tokenizer_or_processor_path")),
            _path_exists(_as_str(vision.get("processor_path") or vision.get("tokenizer_or_processor_path"))),
        ),
        PocPathCheck("nvila_checkpoint", _as_str(mllm.get("checkpoint")), _path_exists(_as_str(mllm.get("checkpoint")))),
        PocPathCheck("nvila_config", _as_str(mllm.get("config_path")), _path_exists(_as_str(mllm.get("config_path")))),
        PocPathCheck("nvila_processor", _as_str(mllm.get("processor_path")), _path_exists(_as_str(mllm.get("processor_path")))),
        PocPathCheck("nvila_tokenizer", _as_str(mllm.get("tokenizer_path")), _path_exists(_as_str(mllm.get("tokenizer_path")))),
    ]
    return checks


def _as_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


def _autogaze_required(cfg: Mapping[str, Any]) -> bool:
    autogaze_enabled = _get_nested(cfg, "model", "autogaze", "enabled")
    mllm_autogaze_enabled = _get_nested(cfg, "model", "mllm", "autogaze_enabled")
    return bool(autogaze_enabled or mllm_autogaze_enabled)


def _check_import_stage(name: str, node: Mapping[str, Any], *, import_module_fn: ImportModuleFn) -> PocStage:
    module_path = _as_str(node.get("module_path"))
    class_or_factory = _as_str(node.get("class_or_factory"))
    resolution = resolve_import(module_path, class_or_factory, import_module_fn=import_module_fn)
    return PocStage(
        name=name,
        status="passed" if resolution.ready else "blocked",
        module_path=module_path,
        class_or_factory=class_or_factory,
        skipped_reason=None if resolution.ready else resolution.error,
        details={
            "module_available": resolution.module_available,
            "class_or_factory_exists": resolution.object_available,
        },
    )


def _check_nvila_processor_stage(mllm: Mapping[str, Any], *, import_module_fn: ImportModuleFn) -> PocStage:
    module_path = _as_str(mllm.get("nvila_hd_video_processor_module_path") or mllm.get("module_path"))
    class_name = _as_str(mllm.get("nvila_hd_video_processor_class_name") or "AutoProcessor")
    resolution = resolve_import(module_path, class_name, import_module_fn=import_module_fn)
    return PocStage(
        name="nvila_processor",
        status="passed" if resolution.ready else "blocked",
        module_path=module_path,
        class_or_factory=class_name,
        skipped_reason=None if resolution.ready else resolution.error,
        details={
            "factory": mllm.get("nvila_hd_video_processor_factory_name", "from_pretrained"),
            "module_available": resolution.module_available,
            "class_or_factory_exists": resolution.object_available,
        },
    )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be float32, float16, or bfloat16")


def _device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    if name == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dummy_video(num_frames: int, resolution: int) -> torch.Tensor:
    values = torch.linspace(0, 1, steps=num_frames * 3 * resolution * resolution, dtype=torch.float32)
    return values.reshape(1, num_frames, 3, resolution, resolution)


def _resize_video(video: torch.Tensor, resolution: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"expected [B, T, C, H, W], got {tuple(video.shape)}")
    bsz, frames, channels, height, width = video.shape
    if height == resolution and width == resolution:
        return video
    flat = video.reshape(bsz * frames, channels, height, width)
    resized = F.interpolate(flat, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(bsz, frames, channels, resolution, resolution)


def _load_local_video(video_path: PathLike, num_frames: int, resolution: int) -> torch.Tensor:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"video path does not exist: {path}")
    try:
        import av  # type: ignore
        from autogaze.datasets.video_utils import read_video_pyav  # type: ignore
    except Exception as exc:
        raise RuntimeError("local AutoGaze preprocessing requires PyAV and autogaze.datasets.video_utils") from exc

    container = av.open(str(path))
    try:
        raw = read_video_pyav(container=container, indices=list(range(num_frames)))
    finally:
        container.close()
    tensor = torch.as_tensor(raw)
    if tensor.ndim != 4:
        raise ValueError(f"decoded video must be [T, H, W, C] or [T, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.to(torch.float32)
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return _resize_video(tensor.unsqueeze(0), resolution)


def _prepare_video(video: str, video_path: str | None, num_frames: int, resolution: int) -> tuple[torch.Tensor | None, str]:
    if video_path:
        return _load_local_video(video_path, num_frames, resolution), str(Path(video_path).expanduser())
    if video == "dummy":
        return _dummy_video(num_frames, resolution), "dummy"
    return None, "none"


def _resolve_object(module_path: str, object_name: str, *, import_module_fn: ImportModuleFn) -> Any:
    module = import_module_fn(module_path)
    cursor: Any = module
    for part in object_name.split("."):
        cursor = getattr(cursor, part)
    return cursor


def _from_pretrained(factory: Any, path: str, node: Mapping[str, Any], **extra_kwargs: Any) -> Any:
    kwargs = dict(node.get("construction_kwargs", {})) if isinstance(node.get("construction_kwargs"), Mapping) else {}
    kwargs.update(extra_kwargs)
    kwargs.setdefault("local_files_only", bool(node.get("local_files_only", True)))
    kwargs.setdefault("trust_remote_code", bool(node.get("trust_remote_code", False)))
    if hasattr(factory, "from_pretrained"):
        try:
            return factory.from_pretrained(path, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            return factory.from_pretrained(path, **kwargs)
    return factory(**kwargs)


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def _extract_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    if isinstance(output, Mapping):
        for key in ("last_hidden_state", "visual_features"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def _autogaze_token_counts(output: Mapping[str, Any], frame_count: int, resolution: int, model: Any) -> tuple[int, int]:
    selected: int | None = None
    padded = output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor):
        selected = int((~padded.bool()).sum().item())
    gaze_pos = output.get("gazing_pos")
    if selected is None and isinstance(gaze_pos, torch.Tensor):
        selected = int(gaze_pos.numel())
    per_frame = getattr(model, "num_vision_tokens_each_frame", None)
    if per_frame is None and hasattr(model, "config"):
        per_frame = getattr(model.config, "num_vision_tokens_each_frame", None)
    original = int(per_frame) * frame_count if per_frame is not None else (resolution // 16) ** 2 * frame_count
    return original, int(selected or 0)


def _save_autogaze_artifacts(
    output_dir: Path,
    gaze_output: Mapping[str, Any],
    *,
    original_count: int,
    selected_count: int,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    autogaze_dir = output_dir / "autogaze"
    if "gazing_pos" in gaze_output:
        path = autogaze_dir / "selected_patch_indices.json"
        _write_json(path, {"selected_patch_indices": _json_safe(gaze_output["gazing_pos"])})
        artifacts["selected_patch_indices"] = str(path)
    token_path = autogaze_dir / "token_counts.json"
    _write_json(
        token_path,
        {
            "original_visual_token_count": original_count,
            "selected_visual_token_count": selected_count,
        },
    )
    artifacts["token_counts"] = str(token_path)
    return artifacts


def _maybe_visualize(output_dir: Path, video: torch.Tensor, gaze_output: Mapping[str, Any], resolution: int) -> dict[str, str]:
    gaze_pos = gaze_output.get("gazing_pos")
    if not isinstance(gaze_pos, torch.Tensor) or gaze_pos.numel() == 0:
        return {}
    try:
        from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer
    except Exception:
        return {}
    patch_grid = (max(1, resolution // 16), max(1, resolution // 16))
    selected = gaze_pos.detach().cpu().flatten()[:16] % (patch_grid[0] * patch_grid[1])
    visualizer = AutoGazeVisualizer(output_root=output_dir.parent, exp_name=output_dir.name)
    paths = visualizer.visualize_selected_patches(video.detach().cpu(), selected, patch_grid, mode="autogaze_only", prefix="nvila_poc")
    if not paths:
        return {}
    return {"autogaze_visualization": str(paths[0])}


def _processor_kwargs(mllm: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "num_video_frames",
        "num_video_frames_thumbnail",
        "max_tiles_video",
        "gazing_ratio_tile",
        "gazing_ratio_thumbnail",
        "task_loss_requirement_tile",
        "task_loss_requirement_thumbnail",
        "max_batch_size_autogaze",
    ]
    return {key: mllm[key] for key in keys if key in mllm}


def _make_summary(
    *,
    mode: str,
    status: str,
    cfg: Mapping[str, Any],
    output_dir: Path,
    device: str,
    effective_device: str,
    dtype: str,
    no_checkpoint_load: bool,
    query_text: str | None,
    input_video: str,
    input_shape: list[int] | None,
    selected_token_count: int | None,
    original_visual_token_count: int | None,
    output_text: str | None,
    stages: list[PocStage],
    path_checks: list[PocPathCheck],
    skipped: list[dict[str, str]],
    artifacts: dict[str, str],
    latency_ms: float,
) -> PocSummary:
    return PocSummary(
        mode=mode,
        status=status,
        experiment_id=_experiment_id(cfg),
        output_dir=str(output_dir),
        device=device,
        effective_device=effective_device,
        dtype=dtype,
        checkpoint_policy="disabled" if no_checkpoint_load else "explicitly_enabled",
        query_text=query_text,
        input_video=input_video,
        input_shape=input_shape,
        selected_token_count=selected_token_count,
        original_visual_token_count=original_visual_token_count,
        output_text=output_text,
        stages=stages,
        path_checks=path_checks,
        skipped_stages=skipped,
        artifacts=artifacts,
        reference={
            "source": "docs/nvila-hd-video-readme.md",
            "extracted_reference": "docs/NVILA_HD_VIDEO_REFERENCE.md",
            "model_id": str(_get_nested(cfg, "model", "mllm", "nvila_hd_video_model_id") or "nvidia/NVILA-8B-HD-Video"),
        },
        latency_ms=latency_ms,
    )


def run_poc(
    *,
    mode: str = "check",
    video: str = "dummy",
    video_path: str | None = None,
    query_text: str | None = None,
    num_frames: int = 2,
    resolution: int = 224,
    device: str = "cpu",
    dtype: str = "float32",
    max_new_tokens: int = 1,
    output_dir: PathLike = "outputs/nvila_hd_video_poc",
    config: PathLike = "configs/experiment/A2_real.yaml",
    no_checkpoint_load: bool = True,
    checkpoint_metadata_only: bool = False,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> PocSummary:
    if mode not in {"check", "autogaze_only", "full_pipeline"}:
        raise ValueError("mode must be check, autogaze_only, or full_pipeline")
    if video != "dummy" and not video_path:
        raise ValueError("only --video dummy is supported unless --video-path is provided")
    if num_frames <= 0 or resolution <= 0 or max_new_tokens <= 0:
        raise ValueError("num_frames, resolution, and max_new_tokens must be > 0")

    start = time.perf_counter()
    cfg = _plain_config(_load_config_from_path(config))
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    autogaze = _component(cfg, "autogaze")
    vision = _component(cfg, "vision_encoder")
    mllm = _component(cfg, "mllm")
    effective_device = _device(device)
    torch_dtype = _dtype(dtype)
    stages: list[PocStage] = []
    skipped: list[dict[str, str]] = []
    artifacts: dict[str, str] = {}
    path_checks = _path_checks(cfg)

    if _autogaze_required(cfg):
        stages.append(_check_import_stage("autogaze_import", autogaze, import_module_fn=import_module_fn))
    else:
        stages.append(
            PocStage(
                "autogaze_import",
                "disabled",
                skipped_reason="AutoGaze is disabled for this experiment",
                details={"autogaze_enabled": False},
            )
        )
    stages.extend(
        [
            _check_import_stage("siglip_import", vision, import_module_fn=import_module_fn),
            _check_import_stage("nvila_model_import", mllm, import_module_fn=import_module_fn),
            _check_nvila_processor_stage(mllm, import_module_fn=import_module_fn),
        ]
    )

    if mode == "check":
        for item in path_checks:
            if item.required and not item.exists:
                skipped.append({"stage": "path_check", "reason": f"{item.name} missing: {item.path}"})
        import_checks_passed = all(stage.status in {"passed", "disabled"} for stage in stages)
        status = "passed" if import_checks_passed and not skipped else "blocked"
        summary = _make_summary(
            mode=mode,
            status=status,
            cfg=cfg,
            output_dir=output_root,
            device=device,
            effective_device=effective_device,
            dtype=dtype,
            no_checkpoint_load=no_checkpoint_load,
            query_text=query_text,
            input_video=video_path or video,
            input_shape=None,
            selected_token_count=None,
            original_visual_token_count=None,
            output_text=None,
            stages=stages,
            path_checks=path_checks,
            skipped=skipped,
            artifacts=artifacts,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        _write_json(output_root / "logs" / "poc_summary.json", summary.to_dict())
        return summary

    try:
        video_tensor, input_video = _prepare_video(video, video_path, num_frames, resolution)
    except Exception as exc:
        video_tensor, input_video = None, video_path or video
        skipped.append({"stage": "video_preprocessing", "reason": str(exc)})
        stages.append(PocStage("video_preprocessing", "skipped", skipped_reason=str(exc)))

    input_shape = [int(dim) for dim in video_tensor.shape] if video_tensor is not None else None
    selected_count: int | None = None
    original_count: int | None = None
    gaze_output: Mapping[str, Any] | None = None
    vision_features: torch.Tensor | None = None
    output_text: str | None = None

    if mode in {"autogaze_only", "full_pipeline"}:
        if video_tensor is None:
            skipped.append({"stage": "autogaze", "reason": "video preprocessing failed"})
            stages.append(PocStage("autogaze", "skipped", skipped_reason="video preprocessing failed"))
        elif no_checkpoint_load or checkpoint_metadata_only:
            reason = "checkpoint loading disabled; pass --allow-checkpoint-load to execute AutoGaze"
            if checkpoint_metadata_only:
                reason = "checkpoint metadata-only mode; AutoGaze execution skipped"
            skipped.append({"stage": "autogaze", "reason": reason})
            stages.append(PocStage("autogaze", "skipped", module_path=_as_str(autogaze.get("module_path")), class_or_factory=_as_str(autogaze.get("class_or_factory")), skipped_reason=reason))
        else:
            stage_start = time.perf_counter()
            try:
                factory = _resolve_object(str(autogaze["module_path"]), str(autogaze["class_or_factory"]), import_module_fn=import_module_fn)
                model = _from_pretrained(factory, str(autogaze.get("checkpoint") or autogaze.get("model_config_path")), autogaze)
                if hasattr(model, "eval"):
                    model.eval()
                if hasattr(model, "to"):
                    try:
                        model.to(device=effective_device, dtype=torch_dtype)
                    except TypeError:
                        model.to(effective_device)
                model_input = video_tensor.to(device=effective_device, dtype=torch_dtype)
                args = autogaze.get("original_cli_args") if isinstance(autogaze.get("original_cli_args"), Mapping) else {}
                kwargs = {
                    "gazing_ratio": args.get("gazing_ratio", 0.75),
                    "task_loss_requirement": args.get("task_loss_requirement", 0.7),
                }
                with torch.inference_mode():
                    result = model({"video": model_input}, **kwargs)
                if not isinstance(result, Mapping):
                    raise TypeError("AutoGaze output must be a mapping")
                gaze_output = result
                original_count, selected_count = _autogaze_token_counts(result, num_frames, resolution, model)
                artifacts.update(_save_autogaze_artifacts(output_root, result, original_count=original_count, selected_count=selected_count))
                artifacts.update(_maybe_visualize(output_root, video_tensor, result, resolution))
                stages.append(
                    PocStage(
                        "autogaze",
                        "passed",
                        module_path=_as_str(autogaze.get("module_path")),
                        class_or_factory=_as_str(autogaze.get("class_or_factory")),
                        latency_ms=(time.perf_counter() - stage_start) * 1000,
                        output_shape=_shape(result.get("gazing_pos")),
                        details={"call_kwargs": kwargs},
                    )
                )
            except Exception as exc:
                reason = f"AutoGaze execution failed: {exc}"
                skipped.append({"stage": "autogaze", "reason": reason})
                stages.append(PocStage("autogaze", "skipped", module_path=_as_str(autogaze.get("module_path")), class_or_factory=_as_str(autogaze.get("class_or_factory")), skipped_reason=reason))

    if mode == "full_pipeline":
        if gaze_output is not None and video_tensor is not None and not (no_checkpoint_load or checkpoint_metadata_only):
            stage_start = time.perf_counter()
            try:
                factory = _resolve_object(str(vision["module_path"]), str(vision["class_or_factory"]), import_module_fn=import_module_fn)
                model = _from_pretrained(factory, str(vision.get("checkpoint") or vision.get("model_config_path")), vision)
                if hasattr(model, "eval"):
                    model.eval()
                model_input = video_tensor.to(device=effective_device, dtype=torch_dtype)
                with torch.inference_mode():
                    try:
                        output = model(model_input, gazing_info=gaze_output)
                    except TypeError:
                        output = model(model_input)
                vision_features = _extract_tensor(output)
                stages.append(
                    PocStage(
                        "siglip_vision_encoder",
                        "passed" if vision_features is not None else "skipped",
                        module_path=_as_str(vision.get("module_path")),
                        class_or_factory=_as_str(vision.get("class_or_factory")),
                        latency_ms=(time.perf_counter() - stage_start) * 1000,
                        output_shape=_shape(vision_features),
                        skipped_reason=None if vision_features is not None else "vision output did not expose features",
                        details={"gazing_info_used": True},
                    )
                )
            except Exception as exc:
                reason = f"SigLIP execution failed: {exc}"
                stages.append(PocStage("siglip_vision_encoder", "skipped", module_path=_as_str(vision.get("module_path")), class_or_factory=_as_str(vision.get("class_or_factory")), skipped_reason=reason))
                skipped.append({"stage": "siglip_vision_encoder", "reason": reason})
        elif no_checkpoint_load or checkpoint_metadata_only:
            reason = "SigLIP execution skipped because checkpoint loading is disabled"
            if checkpoint_metadata_only:
                reason = "SigLIP execution skipped because checkpoint metadata-only mode is active"
            skipped.append({"stage": "siglip_vision_encoder", "reason": reason})
            stages.append(PocStage("siglip_vision_encoder", "skipped", module_path=_as_str(vision.get("module_path")), class_or_factory=_as_str(vision.get("class_or_factory")), skipped_reason=reason))
        else:
            reason = "SigLIP execution skipped because AutoGaze outputs are unavailable"
            skipped.append({"stage": "siglip_vision_encoder", "reason": reason})
            stages.append(PocStage("siglip_vision_encoder", "skipped", module_path=_as_str(vision.get("module_path")), class_or_factory=_as_str(vision.get("class_or_factory")), skipped_reason=reason))

        if not query_text:
            reason = "query text is required for NVILA generation and was not provided"
            skipped.append({"stage": "nvila_generation", "reason": reason})
            stages.append(PocStage("nvila_generation", "skipped", skipped_reason=reason))
        elif no_checkpoint_load or checkpoint_metadata_only:
            reason = "query text was accepted, but NVILA generation was skipped because checkpoint loading is disabled"
            if checkpoint_metadata_only:
                reason = "query text was accepted, but NVILA generation was skipped because checkpoint metadata-only mode is active"
            skipped.append({"stage": "nvila_generation", "reason": reason})
            stages.append(PocStage("nvila_generation", "skipped", module_path=_as_str(mllm.get("module_path")), class_or_factory=_as_str(mllm.get("class_or_factory")), skipped_reason=reason))
        else:
            stage_start = time.perf_counter()
            try:
                processor_class = _resolve_object(
                    str(mllm.get("nvila_hd_video_processor_module_path") or "transformers"),
                    str(mllm.get("nvila_hd_video_processor_class_name") or "AutoProcessor"),
                    import_module_fn=import_module_fn,
                )
                model_class = _resolve_object(str(mllm["module_path"]), str(mllm["class_or_factory"]), import_module_fn=import_module_fn)
                processor_path = str(mllm.get("processor_path") or mllm.get("checkpoint") or mllm.get("nvila_hd_video_model_id"))
                model_path = str(mllm.get("checkpoint") or mllm.get("nvila_hd_video_model_id"))
                processor = _from_pretrained(processor_class, processor_path, mllm, **_processor_kwargs(mllm))
                model = _from_pretrained(model_class, model_path, mllm)
                if hasattr(model, "eval"):
                    model.eval()
                video_token = getattr(getattr(processor, "tokenizer", None), "video_token", "<video>")
                prompt_template = str(mllm.get("prompt_template") or "{video_token}\n\n{prompt}")
                prompt = prompt_template.format(video_token=video_token, prompt=query_text)
                video_input = video_path or ("dummy" if video == "dummy" else video)
                inputs = processor(text=prompt, videos=video_input, return_tensors="pt")
                model_device = getattr(model, "device", effective_device)
                if isinstance(model_device, str):
                    model_device = torch.device(model_device)
                if isinstance(inputs, Mapping):
                    inputs = {
                        key: value.to(model_device) if isinstance(value, torch.Tensor) else value
                        for key, value in inputs.items()
                    }
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
                if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor):
                    decode_input = outputs[:, input_ids.shape[1] :]
                else:
                    decode_input = outputs
                decoded = processor.batch_decode(decode_input, skip_special_tokens=True)
                output_text = str(decoded[0]).strip() if decoded else ""
                answer_path = output_root / "predictions" / "answer.json"
                _write_json(answer_path, {"answer": output_text, "query_text": query_text})
                artifacts["answer"] = str(answer_path)
                stages.append(
                    PocStage(
                        "nvila_generation",
                        "passed",
                        module_path=_as_str(mllm.get("module_path")),
                        class_or_factory=_as_str(mllm.get("class_or_factory")),
                        latency_ms=(time.perf_counter() - stage_start) * 1000,
                        output_shape=_shape(outputs),
                        details={"processor_path": processor_path, "model_path": model_path, "prompt_template": prompt_template},
                    )
                )
            except Exception as exc:
                reason = f"query text was accepted, but NVILA generation failed: {exc}"
                skipped.append({"stage": "nvila_generation", "reason": reason})
                stages.append(PocStage("nvila_generation", "skipped", module_path=_as_str(mllm.get("module_path")), class_or_factory=_as_str(mllm.get("class_or_factory")), skipped_reason=reason))

    successful = any(stage.status == "passed" for stage in stages)
    status = "passed" if successful and not skipped else ("partial" if successful else "blocked")
    summary = _make_summary(
        mode=mode,
        status=status,
        cfg=cfg,
        output_dir=output_root,
        device=device,
        effective_device=effective_device,
        dtype=dtype,
        no_checkpoint_load=no_checkpoint_load,
        query_text=query_text,
        input_video=input_video if "input_video" in locals() else video_path or video,
        input_shape=input_shape,
        selected_token_count=selected_count,
        original_visual_token_count=original_count,
        output_text=output_text,
        stages=stages,
        path_checks=path_checks,
        skipped=skipped,
        artifacts=artifacts,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    summary_path = output_root / "logs" / "poc_summary.json"
    _write_json(summary_path, summary.to_dict())
    summary.artifacts["poc_summary"] = str(summary_path)
    _write_json(summary_path, summary.to_dict())
    return summary


def print_summary(summary: PocSummary) -> None:
    print("NVILA-HD-Video canonical PoC")
    print(f"mode: {summary.mode}")
    print(f"status: {summary.status}")
    print(f"experiment: {summary.experiment_id}")
    print(f"device: {summary.device} (effective: {summary.effective_device})")
    print(f"dtype: {summary.dtype}")
    print(f"checkpoint_policy: {summary.checkpoint_policy}")
    if summary.query_text:
        print(f"query_text: {summary.query_text}")
    print(f"output_dir: {summary.output_dir}")
    for stage in summary.stages:
        print(f"- {stage.name}: {stage.status}")
        if stage.module_path:
            print(f"  module: {stage.module_path}")
        if stage.class_or_factory:
            print(f"  class_or_factory: {stage.class_or_factory}")
        if stage.output_shape:
            print(f"  output_shape: {stage.output_shape}")
        if stage.skipped_reason:
            print(f"  skipped_reason: {stage.skipped_reason}")
    if summary.skipped_stages:
        print("skipped_stages:")
        for item in summary.skipped_stages:
            print(f"  - {item['stage']}: {item['reason']}")
    if summary.artifacts:
        print("artifacts:")
        for key, path in sorted(summary.artifacts.items()):
            print(f"  {key}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated NVILA-HD-Video canonical PoC")
    parser.add_argument("--mode", choices=["check", "autogaze_only", "full_pipeline"], default="check")
    parser.add_argument("--video", choices=["dummy"], default="dummy")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/nvila_hd_video_poc")
    parser.add_argument("--config", default="configs/experiment/A2_real.yaml")
    parser.add_argument("--no-checkpoint-load", action="store_true", default=False)
    parser.add_argument("--allow-checkpoint-load", action="store_true", help="Explicitly allow loading real checkpoints.")
    parser.add_argument("--checkpoint-metadata-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    no_checkpoint_load = not args.allow_checkpoint_load
    if args.no_checkpoint_load:
        no_checkpoint_load = True

    summary = run_poc(
        mode=args.mode,
        video=args.video,
        video_path=args.video_path,
        query_text=args.query_text,
        num_frames=args.num_frames,
        resolution=args.resolution,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        config=args.config,
        no_checkpoint_load=no_checkpoint_load,
        checkpoint_metadata_only=args.checkpoint_metadata_only,
    )
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print_summary(summary)
    return 0 if summary.status in {"passed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
