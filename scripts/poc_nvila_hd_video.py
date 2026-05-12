#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from autogaze_ext.data.frame_selector import FrameSelectionResult, FrameWindow, select_frame_windows
from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.profiling.memory import MemoryTracker
from autogaze_ext.scaling import scale_video_for_autogaze
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import


PathLike = str | Path

DEFAULT_NUM_FRAMES = 16


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
    frame_selection: dict[str, Any] | None
    scaling: dict[str, Any] | None
    autogaze_runtime: dict[str, Any] | None
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


def _dummy_video_hw(num_frames: int, height: int, width: int) -> torch.Tensor:
    values = torch.linspace(0, 1, steps=num_frames * 3 * height * width, dtype=torch.float32)
    return values.reshape(1, num_frames, 3, height, width)


def _resize_video(video: torch.Tensor, resolution: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"expected [B, T, C, H, W], got {tuple(video.shape)}")
    bsz, frames, channels, height, width = video.shape
    if height == resolution and width == resolution:
        return video
    flat = video.reshape(bsz * frames, channels, height, width)
    resized = F.interpolate(flat, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(bsz, frames, channels, resolution, resolution)


def _probe_local_video(video_path: PathLike) -> tuple[int, float | None]:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"video path does not exist: {path}")
    try:
        import av  # type: ignore
    except Exception as exc:
        raise RuntimeError("local video metadata probing requires PyAV") from exc

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        if frames <= 0:
            frames = sum(1 for _ in container.decode(video=0))
    finally:
        container.close()
    if frames <= 0:
        raise ValueError(f"could not determine frame count for video: {path}")
    return frames, fps


def _load_local_video_indices(video_path: PathLike, frame_indices: list[int]) -> torch.Tensor:
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
        raw = read_video_pyav(container=container, indices=frame_indices)
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
    return tensor.unsqueeze(0)


def _prepare_windowed_videos(
    *,
    cfg: Mapping[str, Any],
    video: str,
    video_path: str | None,
    num_frames: int,
    resolution: int,
    frame_selection_mode: str,
    frame_interval: int,
    max_windows: int | None,
    drop_last: bool,
    pad_last: bool,
) -> tuple[list[tuple[FrameWindow, torch.Tensor]], str, FrameSelectionResult]:
    if video_path:
        original_frame_count, original_fps = _probe_local_video(video_path)
        input_video = str(Path(video_path).expanduser())
        full_dummy = None
    elif video == "dummy":
        configured_frames = _get_nested(cfg, "data", "dummy_video", "frames")
        original_frame_count = int(configured_frames or max(num_frames, num_frames * 2))
        original_fps = None
        input_video = "dummy"
        dummy_height = int(_get_nested(cfg, "data", "dummy_video", "height") or resolution)
        dummy_width = int(_get_nested(cfg, "data", "dummy_video", "width") or resolution)
        full_dummy = _dummy_video_hw(original_frame_count, dummy_height, dummy_width)[0]
    else:
        raise ValueError("only --video dummy is supported unless --video-path is provided")

    selection = select_frame_windows(
        original_frame_count=original_frame_count,
        original_fps=original_fps,
        num_frames=num_frames,
        frame_selection_mode=frame_selection_mode,
        frame_interval=frame_interval,
        max_windows=max_windows,
        drop_last=drop_last,
        pad_last=pad_last,
    )

    window_videos: list[tuple[FrameWindow, torch.Tensor]] = []
    for window in selection.windows:
        if full_dummy is not None:
            indices = torch.tensor(window.frame_indices, dtype=torch.long)
            tensor = full_dummy.index_select(0, indices).unsqueeze(0)
        else:
            tensor = _load_local_video_indices(str(video_path), window.frame_indices)
        window_videos.append((window, tensor))
    return window_videos, input_video, selection


def _load_full_reference_video(
    *,
    cfg: Mapping[str, Any],
    video: str,
    video_path: str | None,
    original_frame_count: int,
    resolution: int,
) -> torch.Tensor | None:
    if original_frame_count <= 0:
        return None
    if video_path:
        return _load_local_video_indices(str(video_path), list(range(original_frame_count)))
    if video == "dummy":
        dummy_height = int(_get_nested(cfg, "data", "dummy_video", "height") or resolution)
        dummy_width = int(_get_nested(cfg, "data", "dummy_video", "width") or resolution)
        return _dummy_video_hw(original_frame_count, dummy_height, dummy_width)
    return None


def _parse_scales(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("+", ",").split(",") if part.strip()]
        return [int(part) for part in parts] if parts else None
    return [int(item) for item in value]


def _scale_window_videos(
    windows: list[tuple[FrameWindow, torch.Tensor]],
    *,
    scaling_mode: str,
    resolution: int,
    patch_size: int,
    target_scales: list[int] | None,
    target_patch_size: int | None,
    temporal_chunk_size: int | None,
    spatial_tile_size: int | None,
    max_chops: int | None = None,
    chop_overlap: int = 0,
    chop_stride: int | None = None,
    chop_merge_mode: str = "metadata_only",
) -> tuple[list[tuple[FrameWindow, torch.Tensor, torch.Tensor, dict[str, Any]]], dict[str, Any]]:
    scaled_windows: list[tuple[FrameWindow, torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    combined: dict[str, Any] = {
        "scaling_mode": scaling_mode,
        "resolution": resolution,
        "patch_size": patch_size,
        "target_scales": target_scales,
        "target_patch_size": target_patch_size,
        "temporal_chunk_size": temporal_chunk_size,
        "spatial_tile_size": spatial_tile_size,
        "chop_overlap": chop_overlap,
        "chop_stride": chop_stride,
        "max_chops": max_chops,
        "chop_merge_mode": chop_merge_mode,
        "windows": [],
    }
    for window, original_tensor in windows:
        scaled = scale_video_for_autogaze(
            original_tensor,
            scaling_mode=scaling_mode,  # type: ignore[arg-type]
            resolution=resolution,
            patch_size=patch_size,
            target_scales=target_scales,
            target_patch_size=target_patch_size,
            temporal_chunk_size=temporal_chunk_size,
            spatial_tile_size=spatial_tile_size,
        )
        processed_video = scaled.video
        window_metadata = dict(scaled.metadata)
        if scaling_mode == "chop" and "chop" in window_metadata:
            chop_metadata = dict(window_metadata["chop"])
            records = list(chop_metadata.get("chunk_records", []))
            if max_chops is not None:
                records = records[:max_chops]
                processed_video = processed_video[:max_chops]
                chop_metadata["chunk_records"] = records
                chop_metadata["chunks_shape"] = [int(dim) for dim in processed_video.shape]
                chop_metadata["max_chops_applied"] = max_chops
            chop_metadata["chop_overlap"] = chop_overlap
            chop_metadata["chop_stride"] = chop_stride
            chop_metadata["chop_merge_mode"] = chop_merge_mode
            window_metadata["chop"] = chop_metadata
        window_metadata["window"] = window.to_dict()
        scaled_windows.append((window, original_tensor, processed_video, window_metadata))
        combined["windows"].append(window_metadata)
    if scaled_windows:
        combined["first_processed_shape"] = [int(dim) for dim in scaled_windows[0][2].shape]
        combined["number_of_windows"] = len(scaled_windows)
    return scaled_windows, combined


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


def _autogaze_token_counts(
    output: Mapping[str, Any],
    frame_count: int,
    processed_height: int,
    processed_width: int,
    model: Any,
    *,
    patch_size: int = 16,
) -> tuple[int, int]:
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
    fallback_per_frame = max(1, processed_height // patch_size) * max(1, processed_width // patch_size)
    original = int(per_frame) * frame_count if per_frame is not None else fallback_per_frame * frame_count
    return original, int(selected or 0)


def _save_autogaze_artifacts(
    output_dir: Path,
    gaze_output: Mapping[str, Any],
    *,
    original_count: int,
    selected_count: int,
    patch_grid: tuple[int, int],
    frame_count: int,
    runtime: Mapping[str, Any],
    scaling: Mapping[str, Any] | None = None,
    window: FrameWindow | None = None,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    autogaze_dir = output_dir / "autogaze"
    if window is not None:
        autogaze_dir = autogaze_dir / "windows" / f"window_{window.window_id:03d}"
    if "gazing_pos" in gaze_output:
        path = autogaze_dir / "selected_patch_indices.json"
        _write_json(path, {"selected_patch_indices": _json_safe(gaze_output["gazing_pos"])})
        artifacts["selected_patch_indices" if window is None else f"window_{window.window_id:03d}_selected_patch_indices"] = str(path)
        mask_path = autogaze_dir / "selected_patch_mask.json"
        _write_json(
            mask_path,
            {
                "patch_grid": list(patch_grid),
                "selected_patch_mask": _patch_mask_rows(gaze_output, patch_grid, frame_count),
            },
        )
        artifacts["selected_patch_mask" if window is None else f"window_{window.window_id:03d}_selected_patch_mask"] = str(mask_path)
    for scale_key in ("selected_scales", "scales"):
        if scale_key in gaze_output:
            path = autogaze_dir / "selected_scales.json"
            _write_json(path, {"selected_scales": _json_safe(gaze_output[scale_key])})
            artifacts["selected_scales" if window is None else f"window_{window.window_id:03d}_selected_scales"] = str(path)
            break
    token_path = autogaze_dir / "token_counts.json"
    token_data = {
        "original_visual_token_count": original_count,
        "selected_visual_token_count": selected_count,
        "token_reduction_ratio": 1.0 - (float(selected_count) / float(original_count)) if original_count else None,
        "patch_grid": list(patch_grid),
        "autogaze_runtime": dict(runtime),
    }
    if scaling is not None:
        token_data["scaling"] = dict(scaling)
    if window is not None:
        token_data.update(
            {
                "window_id": window.window_id,
                "frame_indices": window.frame_indices,
                "is_padded": window.is_padded,
                "padded_frame_mask": window.padded_frame_mask,
                "original_frame_count": window.original_frame_count,
                "effective_num_frames": window.effective_num_frames,
            }
        )
    _write_json(token_path, token_data)
    artifacts["token_counts" if window is None else f"window_{window.window_id:03d}_token_counts"] = str(token_path)
    return artifacts


def _patch_mask_rows(gaze_output: Mapping[str, Any], patch_grid: tuple[int, int], frame_count: int) -> list[list[bool]]:
    patches_per_frame = patch_grid[0] * patch_grid[1]
    masks = [[False for _ in range(patches_per_frame)] for _ in range(frame_count)]
    gaze_pos = gaze_output.get("gazing_pos")
    if not isinstance(gaze_pos, torch.Tensor):
        return masks
    selected = gaze_pos.detach().cpu()
    padded = gaze_output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor) and padded.shape == selected.shape:
        selected = selected[~padded.detach().cpu().bool()]
    for raw in selected.flatten().tolist():
        value = int(raw)
        if value < 0:
            continue
        if value >= patches_per_frame:
            frame_idx, patch_idx = divmod(value, patches_per_frame)
            if frame_idx < frame_count:
                masks[frame_idx][patch_idx] = True
        else:
            masks[0][value] = True
    return masks


def _maybe_visualize(output_dir: Path, video: torch.Tensor, gaze_output: Mapping[str, Any], resolution: int) -> dict[str, str]:
    return _maybe_visualize_autogaze(
        output_dir,
        video,
        gaze_output,
        resolution,
        sampled_frame_indices=list(range(int(video.shape[1] if video.ndim == 5 else video.shape[0]))),
    )


def _maybe_visualize_autogaze(
    output_dir: Path,
    video: torch.Tensor,
    gaze_output: Mapping[str, Any],
    resolution: int,
    *,
    sampled_frame_indices: list[int],
    original_video: torch.Tensor | None = None,
    full_original_video: torch.Tensor | None = None,
    original_frame_count: int | None = None,
    original_fps: float | None = None,
    scaling_mode: str = "resize",
    patch_size: int = 16,
    save_overlay_video: bool = False,
    save_side_by_side_video: bool = False,
    save_scale_panel_video: bool = False,
    video_fps: float = 4.0,
    video_export_mode: str = "sampled_only",
    overlay_alpha: float = 0.35,
    overlay_line_width: int = 2,
    overlay_style: str = "mask",
    show_patch_boxes: bool | None = None,
    show_patch_indices: bool = False,
    show_scale_labels: bool = False,
    multi_scale_overlay: bool = True,
    scale_color_mode: str = "gradient",
    scale_panel_layout: str = "2x2",
    comparison_layout: str = "processed_overlay",
    info_panel_mode: str = "external",
    original_visual_token_count: int | None = None,
    selected_visual_token_count: int | None = None,
    visualization_mode: str = "autogaze_only",
    artifact_prefix: str = "nvila_poc",
    output_video_suffix: str | None = None,
) -> dict[str, str]:
    gaze_pos = gaze_output.get("gazing_pos")
    if not isinstance(gaze_pos, torch.Tensor) or gaze_pos.numel() == 0:
        return {}
    try:
        from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer
    except Exception:
        return {}
    frames = video[0] if video.ndim == 5 else video
    processed_height, processed_width = int(frames.shape[-2]), int(frames.shape[-1])
    if original_video is not None:
        original_frames = original_video[0] if original_video.ndim == 5 else original_video
        original_resolution = (int(original_frames.shape[-2]), int(original_frames.shape[-1]))
    else:
        original_resolution = (processed_height, processed_width)
    patch_grid = (max(1, processed_height // patch_size), max(1, processed_width // patch_size))
    selected = gaze_pos.detach().cpu()
    padded = gaze_output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor) and padded.shape == selected.shape:
        selected = selected[~padded.detach().cpu().bool()]
    scales = gaze_output.get("selected_scales")
    if scales is None:
        scales = gaze_output.get("scales")
    visualizer = AutoGazeVisualizer(output_root=output_dir.parent, exp_name=output_dir.name)
    artifacts: dict[str, str] = {}
    paths = visualizer.visualize_selected_patches(
        video.detach().cpu(),
        selected,
        patch_grid,
        scales=scales,
        mode=visualization_mode,
        prefix=artifact_prefix,
        overlay_style=overlay_style,
        overlay_alpha=overlay_alpha,
        show_patch_indices=show_patch_indices,
        show_scale_labels=show_scale_labels,
        multi_scale_overlay=multi_scale_overlay,
        scale_color_mode=scale_color_mode,
    )
    if paths:
        artifacts["autogaze_visualization"] = str(paths[0])
    if save_overlay_video or save_side_by_side_video or save_scale_panel_video:
        video_paths = visualizer.export_autogaze_videos(
            video.detach().cpu(),
            selected,
            patch_grid,
            scales=scales,
            sampled_frame_indices=sampled_frame_indices,
            original_frame_count=original_frame_count or len(sampled_frame_indices),
            original_resolution=original_resolution,
            processed_resolution=(processed_height, processed_width),
            original_video=original_video.detach().cpu() if original_video is not None else None,
            full_original_video=full_original_video.detach().cpu() if full_original_video is not None else None,
            original_fps=original_fps,
            original_visual_token_count=original_visual_token_count,
            selected_visual_token_count=selected_visual_token_count,
            patch_grid_source=f"inferred_from_processed_resolution_and_patch{patch_size}",
            video_export_mode=video_export_mode,
            fps=video_fps,
            overlay_alpha=overlay_alpha,
            overlay_line_width=overlay_line_width,
            save_overlay_video=save_overlay_video,
            save_side_by_side_video=save_side_by_side_video,
            save_scale_panel_video=save_scale_panel_video,
            mode=visualization_mode,
            prefix=artifact_prefix,
            output_video_suffix=output_video_suffix,
            overlay_style=overlay_style,
            show_patch_boxes=show_patch_boxes,
            show_patch_indices=show_patch_indices,
            show_scale_labels=show_scale_labels,
            multi_scale_overlay=multi_scale_overlay,
            scale_color_mode=scale_color_mode,
            scale_panel_layout=scale_panel_layout,
            comparison_layout=comparison_layout,
            info_panel_mode=info_panel_mode,
            scaling_mode=scaling_mode,
        )
        artifacts.update({name: str(path) for name, path in video_paths.items()})
    return artifacts


def _processor_kwargs(
    mllm: Mapping[str, Any],
    *,
    gaze_ratio: float | None = None,
    task_loss_requirement: float | None = None,
) -> dict[str, Any]:
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
    kwargs = {key: mllm[key] for key in keys if key in mllm}
    if gaze_ratio is not None:
        kwargs["gazing_ratio_tile"] = gaze_ratio
    if task_loss_requirement is not None:
        kwargs["task_loss_requirement_tile"] = task_loss_requirement
    return kwargs


def _filtered_patch_indices(gaze_output: Mapping[str, Any], patch_grid: tuple[int, int]) -> list[int]:
    gaze_pos = gaze_output.get("gazing_pos")
    if not isinstance(gaze_pos, torch.Tensor):
        return []
    selected = gaze_pos.detach().cpu()
    padded = gaze_output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor) and padded.shape == selected.shape:
        selected = selected[~padded.detach().cpu().bool()]
    patches_per_frame = patch_grid[0] * patch_grid[1]
    return [int(value) % patches_per_frame for value in selected.flatten().tolist()]


def _write_frame_selection_metadata(
    output_dir: Path,
    selection: FrameSelectionResult,
    *,
    video_export_mode: str,
) -> Path:
    path = output_dir / "autogaze" / "frame_selection_metadata.json"
    data = selection.to_dict()
    data["video_export_mode"] = video_export_mode
    _write_json(path, data)
    return path


def _write_scaling_metadata(output_dir: Path, scaling: Mapping[str, Any]) -> Path:
    path = output_dir / "scaling" / "scaling_metadata.json"
    _write_json(path, scaling)
    return path


def _write_runtime_metadata(output_dir: Path, runtime: Mapping[str, Any]) -> Path:
    path = output_dir / "autogaze" / "runtime_metadata.json"
    _write_json(path, runtime)
    return path


def _scale_counts(gaze_output: Mapping[str, Any]) -> dict[str, int]:
    scales = gaze_output.get("selected_scales")
    if scales is None:
        scales = gaze_output.get("scales")
    if not isinstance(scales, torch.Tensor):
        return {}
    values = scales.detach().cpu()
    padded = gaze_output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor) and padded.shape == values.shape:
        values = values[~padded.detach().cpu().bool()]
    counts: dict[str, int] = {}
    for value in values.flatten().tolist():
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _selected_scale_values(gaze_output: Mapping[str, Any]) -> list[int]:
    scales = gaze_output.get("selected_scales")
    if scales is None:
        scales = gaze_output.get("scales")
    if not isinstance(scales, torch.Tensor):
        return []
    values = scales.detach().cpu()
    padded = gaze_output.get("if_padded_gazing")
    if isinstance(padded, torch.Tensor) and padded.shape == values.shape:
        values = values[~padded.detach().cpu().bool()]
    return [int(value) for value in values.flatten().tolist()]


def _selected_rows_by_batch(
    value: Any,
    padded: Any,
    *,
    batch_count: int,
) -> tuple[list[list[int]], bool]:
    if not isinstance(value, torch.Tensor):
        return [[] for _ in range(batch_count)], False
    tensor = value.detach().cpu()
    pad_tensor = padded.detach().cpu().bool() if isinstance(padded, torch.Tensor) and padded.shape == tensor.shape else None
    exact = True
    if tensor.ndim == 1:
        rows = [tensor]
        pad_rows = [pad_tensor] if pad_tensor is not None else [None]
        exact = batch_count == 1
    elif tensor.ndim >= 2:
        flat = tensor.reshape(tensor.shape[0], -1)
        rows = [flat[index] for index in range(flat.shape[0])]
        if pad_tensor is not None:
            flat_pad = pad_tensor.reshape(pad_tensor.shape[0], -1)
            pad_rows = [flat_pad[index] for index in range(flat_pad.shape[0])]
        else:
            pad_rows = [None for _ in rows]
        exact = len(rows) == batch_count
    else:
        return [[] for _ in range(batch_count)], False

    if len(rows) == 1 and batch_count > 1:
        rows = rows * batch_count
        pad_rows = pad_rows * batch_count
        exact = False
    if len(rows) < batch_count:
        rows = rows + [rows[-1]] * (batch_count - len(rows))
        pad_rows = pad_rows + [pad_rows[-1]] * (batch_count - len(pad_rows))
        exact = False
    output: list[list[int]] = []
    for row, pad_row in zip(rows[:batch_count], pad_rows[:batch_count]):
        if pad_row is not None:
            row = row[~pad_row]
        output.append([int(item) for item in row.flatten().tolist()])
    return output, exact


def _selected_patch_rows_by_batch(
    gaze_output: Mapping[str, Any],
    *,
    batch_count: int,
    patch_grid: tuple[int, int],
) -> tuple[list[list[int]], bool]:
    rows, exact = _selected_rows_by_batch(
        gaze_output.get("gazing_pos"),
        gaze_output.get("if_padded_gazing"),
        batch_count=batch_count,
    )
    patches_per_frame = patch_grid[0] * patch_grid[1]
    return [[value % patches_per_frame for value in row if value >= 0] for row in rows], exact


def _selected_scale_rows_by_batch(
    gaze_output: Mapping[str, Any],
    *,
    batch_count: int,
) -> tuple[list[list[int]], bool]:
    scales = gaze_output.get("selected_scales")
    if scales is None:
        scales = gaze_output.get("scales")
    return _selected_rows_by_batch(scales, gaze_output.get("if_padded_gazing"), batch_count=batch_count)


def _selected_patches_per_frame(
    gaze_output: Mapping[str, Any],
    patch_grid: tuple[int, int],
    frame_count: int,
) -> list[int]:
    return [sum(1 for selected in row if selected) for row in _patch_mask_rows(gaze_output, patch_grid, frame_count)]


def _write_chop_outputs(
    output_dir: Path,
    *,
    window: FrameWindow,
    window_tensor: torch.Tensor,
    gaze_output: Mapping[str, Any],
    patch_grid: tuple[int, int],
    window_scaling: Mapping[str, Any],
    patch_size: int,
    chop_overlap: int,
    chop_stride: int | None,
    chop_merge_mode: str,
    save_chop_frames: bool,
    overlay_alpha: float,
    overlay_line_width: int,
    overlay_style: str,
    show_patch_indices: bool,
    show_scale_labels: bool,
    multi_scale_overlay: bool,
    scale_color_mode: str,
    scale_panel_layout: str,
    info_panel_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    chop_metadata = window_scaling.get("chop")
    if not isinstance(chop_metadata, Mapping):
        return [], {}
    records = chop_metadata.get("chunk_records", [])
    if not isinstance(records, list):
        return [], {}

    batch_count = int(window_tensor.shape[0]) if window_tensor.ndim == 5 else len(records)
    patch_rows, patch_rows_exact = _selected_patch_rows_by_batch(gaze_output, batch_count=batch_count, patch_grid=patch_grid)
    scale_rows, scale_rows_exact = _selected_scale_rows_by_batch(gaze_output, batch_count=batch_count)
    artifacts: dict[str, str] = {}
    output_records: list[dict[str, Any]] = []
    patches_per_frame = patch_grid[0] * patch_grid[1]
    effective_frames = int(window_tensor.shape[1]) if window_tensor.ndim == 5 else window.effective_num_frames
    per_chop_original = patches_per_frame * max(1, effective_frames)
    for item_index, record in enumerate(records):
        if item_index >= int(window_tensor.shape[0]):
            break
        frame_start = int(record.get("frame_start", 0))
        source_frame_index = window.frame_indices[min(frame_start, len(window.frame_indices) - 1)]
        y0 = int(record.get("height_start", 0))
        y1 = int(record.get("height_end_exclusive", window_tensor.shape[-2]))
        x0 = int(record.get("width_start", 0))
        x1 = int(record.get("width_end_exclusive", window_tensor.shape[-1]))
        chop_dir = (
            output_dir
            / "chops"
            / "windows"
            / f"window_{window.window_id:03d}"
            / f"frame_{source_frame_index:03d}"
            / f"chop_{item_index:03d}"
        )
        patches = patch_rows[item_index] if item_index < len(patch_rows) else []
        scales = scale_rows[item_index] if item_index < len(scale_rows) else []
        token_counts = {
            "original_visual_token_count": per_chop_original,
            "selected_visual_token_count": len(patches),
            "token_reduction_ratio": 1.0 - (float(len(patches)) / float(per_chop_original))
            if per_chop_original
            else None,
            "note": "PoC per-chop token count uses chop-local patch grid and selected patch list.",
        }
        _write_json(chop_dir / "selected_patch_indices.json", {"selected_patch_indices": patches})
        _write_json(chop_dir / "selected_scales.json", {"selected_scales": scales})
        _write_json(chop_dir / "token_counts.json", token_counts)

        record_data = {
            "chop_id": item_index,
            "source_frame_index": source_frame_index,
            "window_id": window.window_id,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "chop_width": x1 - x0,
            "chop_height": y1 - y0,
            "chop_overlap": chop_overlap,
            "chop_stride": chop_stride,
            "chop_merge_mode": chop_merge_mode,
            "selected_patch_indices": patches,
            "selected_scales": scales,
            "selected_patch_count": len(patches),
            "token_count": token_counts,
            "coordinate_space": "chop_local",
            "original_space_overlay_supported": False,
            "selection_row_mapping_exact": patch_rows_exact and scale_rows_exact,
        }
        output_records.append(record_data)

        if save_chop_frames:
            sampled_indices = window.frame_indices[
                frame_start : min(frame_start + effective_frames, len(window.frame_indices))
            ]
            if not sampled_indices:
                sampled_indices = [source_frame_index]
            if len(sampled_indices) < effective_frames:
                sampled_indices = sampled_indices + [sampled_indices[-1]] * (effective_frames - len(sampled_indices))
            visual_artifacts = _maybe_visualize_autogaze(
                output_dir,
                window_tensor[item_index : item_index + 1],
                {
                    "gazing_pos": torch.tensor([patches], dtype=torch.long),
                    "selected_scales": torch.tensor([scales], dtype=torch.long) if scales else None,
                },
                int(window_tensor.shape[-1]),
                sampled_frame_indices=sampled_indices,
                patch_size=patch_size,
                scaling_mode="chop",
                save_overlay_video=False,
                save_side_by_side_video=False,
                save_scale_panel_video=False,
                overlay_alpha=overlay_alpha,
                overlay_line_width=overlay_line_width,
                overlay_style=overlay_style,
                show_patch_indices=show_patch_indices,
                show_scale_labels=show_scale_labels,
                multi_scale_overlay=multi_scale_overlay,
                scale_color_mode=scale_color_mode,
                scale_panel_layout=scale_panel_layout,
                info_panel_mode=info_panel_mode,
                original_visual_token_count=per_chop_original,
                selected_visual_token_count=len(patches),
                visualization_mode=(
                    f"autogaze/chops/window_{window.window_id:03d}/"
                    f"frame_{source_frame_index:03d}/chop_{item_index:03d}"
                ),
                artifact_prefix=f"chop_{item_index:03d}",
            )
            artifacts.update(
                {f"window_{window.window_id:03d}_chop_{item_index:03d}_{key}": value for key, value in visual_artifacts.items()}
            )
    return output_records, artifacts


def _write_chop_metadata(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    path = output_dir / "chops" / "chop_metadata.json"
    merge_mode = records[0].get("chop_merge_mode") if records else "metadata_only"
    _write_json(
        path,
        {
            "status": "overlay_union_available" if merge_mode == "overlay_union" else "metadata_only_no_full_frame_merge",
            "coordinate_space": "chop_local",
            "original_space_overlay_supported": merge_mode == "overlay_union",
            "number_of_chops": len(records),
            "merge_mode": merge_mode,
            "chops": records,
        },
    )
    return path


def _write_chop_overlay_metadata(output_dir: Path, metadata: Mapping[str, Any]) -> Path:
    path = output_dir / "visualizations" / "autogaze" / "metadata" / "chop_overlay_metadata.json"
    _write_json(path, metadata)
    return path


def _render_chop_overlay_union(
    output_dir: Path,
    *,
    window: FrameWindow,
    original_tensor: torch.Tensor,
    gaze_output: Mapping[str, Any],
    patch_grid: tuple[int, int],
    window_scaling: Mapping[str, Any],
    patch_size: int,
    overlay_alpha: float,
    overlay_line_width: int,
    overlay_style: str,
    show_patch_indices: bool,
    show_scale_labels: bool,
    multi_scale_overlay: bool,
    scale_color_mode: str,
    info_panel_mode: str,
    original_visual_token_count: int | None,
    selected_visual_token_count: int | None,
) -> tuple[list[Any], dict[str, Any]]:
    chop_metadata = window_scaling.get("chop")
    if not isinstance(chop_metadata, Mapping):
        raise NotImplementedError("overlay_union requires chop metadata")
    records = chop_metadata.get("chunk_records", [])
    if not isinstance(records, list) or not records:
        raise NotImplementedError("overlay_union requires non-empty chop chunk records")

    try:
        from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer
    except Exception as exc:
        raise RuntimeError("AutoGazeVisualizer is required for chop overlay union") from exc

    frames = original_tensor[0].detach().cpu() if original_tensor.ndim == 5 else original_tensor.detach().cpu()
    frame_count = int(frames.shape[0])
    full_h, full_w = int(frames.shape[-2]), int(frames.shape[-1])
    full_grid = (max(1, full_h // patch_size), max(1, full_w // patch_size))
    local_patches_per_frame = patch_grid[0] * patch_grid[1]
    batch_count = min(len(records), int(gaze_output.get("gazing_pos").shape[0]) if isinstance(gaze_output.get("gazing_pos"), torch.Tensor) and gaze_output.get("gazing_pos").ndim >= 2 else len(records))
    raw_patch_rows, patch_rows_exact = _selected_rows_by_batch(
        gaze_output.get("gazing_pos"),
        gaze_output.get("if_padded_gazing"),
        batch_count=len(records),
    )
    scale_rows, scale_rows_exact = _selected_scale_rows_by_batch(gaze_output, batch_count=len(records))

    merged: list[dict[int, int | None]] = [dict() for _ in range(frame_count)]
    conflicts: list[dict[str, Any]] = []
    mapping_exact = patch_rows_exact and scale_rows_exact
    for item_index, record in enumerate(records):
        y0 = int(record.get("height_start", 0))
        x0 = int(record.get("width_start", 0))
        frame_start = int(record.get("frame_start", 0))
        if y0 % patch_size != 0 or x0 % patch_size != 0:
            mapping_exact = False
        row_offset = int(round(y0 / patch_size))
        col_offset = int(round(x0 / patch_size))
        raw_patches = raw_patch_rows[item_index] if item_index < len(raw_patch_rows) else []
        raw_scales = scale_rows[item_index] if item_index < len(scale_rows) else []
        for selected_index, raw_patch in enumerate(raw_patches):
            if raw_patch < 0:
                continue
            frame_offset, local_patch = divmod(int(raw_patch), local_patches_per_frame)
            source_frame = frame_start + frame_offset
            if source_frame >= frame_count:
                mapping_exact = False
                continue
            local_row, local_col = divmod(local_patch, patch_grid[1])
            full_row = row_offset + local_row
            full_col = col_offset + local_col
            if not (0 <= full_row < full_grid[0] and 0 <= full_col < full_grid[1]):
                mapping_exact = False
                continue
            full_patch = full_row * full_grid[1] + full_col
            scale_value = raw_scales[selected_index] if selected_index < len(raw_scales) else None
            existing = merged[source_frame].get(full_patch)
            if existing is not None and existing != scale_value:
                conflicts.append(
                    {
                        "frame_index": source_frame,
                        "patch_index": full_patch,
                        "previous_scale": existing,
                        "new_scale": scale_value,
                        "policy": "last_scale_wins",
                    }
                )
            merged[source_frame][full_patch] = scale_value

    patch_rows: list[list[int]] = []
    merged_scale_rows: list[list[int]] = []
    any_scale = False
    for row in merged:
        ordered = sorted(row)
        patch_rows.append(ordered)
        scales = [int(row[patch]) for patch in ordered if row[patch] is not None]
        if scales:
            any_scale = True
        merged_scale_rows.append(scales)

    visualizer = AutoGazeVisualizer(output_root=output_dir.parent, exp_name=output_dir.name)
    images = visualizer._render_overlay_frames(
        frames,
        patch_rows,
        full_grid,
        scales=merged_scale_rows if any_scale else None,
        sampled_frame_indices=window.frame_indices[:frame_count],
        original_visual_token_count=original_visual_token_count,
        selected_visual_token_count=selected_visual_token_count,
        overlay_style=overlay_style,
        overlay_alpha=overlay_alpha,
        overlay_line_width=overlay_line_width,
        show_patch_indices=show_patch_indices,
        show_scale_labels=show_scale_labels,
        multi_scale_overlay=multi_scale_overlay,
        scale_color_mode=scale_color_mode,
        info_panel_mode=info_panel_mode,
    )
    frames_dir = output_dir / "visualizations" / "autogaze" / "windows" / f"window_{window.window_id:03d}" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for frame_idx, image in enumerate(images):
        source_index = window.frame_indices[min(frame_idx, len(window.frame_indices) - 1)]
        path = frames_dir / f"frame_{source_index:03d}_chop_merged_overlay.png"
        image.save(path)
        image_paths.append(str(path))

    metadata = {
        "window_id": window.window_id,
        "status": "implemented",
        "merge_mode": "overlay_union",
        "chop_count": len(records),
        "chop_coordinates": records,
        "overlapping_region_handling": "last_patch_wins_for_duplicate_patch_indices",
        "scale_conflict_handling": "last_scale_wins",
        "scale_conflicts": conflicts,
        "mapping_exact": mapping_exact and not conflicts,
        "mapping_status": "exact" if mapping_exact and not conflicts else "approximate_or_conflicted",
        "coordinate_space": "full_processed_frame",
        "patch_grid": list(full_grid),
        "patch_size": patch_size,
        "source_frame_indices": window.frame_indices[:frame_count],
        "image_paths": image_paths,
        "batch_row_mapping_exact": batch_count == len(records) and patch_rows_exact and scale_rows_exact,
    }
    return images, metadata


def _write_token_counts_summary(
    output_dir: Path,
    *,
    original_count: int | None,
    selected_count: int | None,
    window_summaries: list[dict[str, Any]],
) -> Path:
    selected_per_scale: dict[str, int] = {}
    selected_per_window: dict[str, int] = {}
    selected_per_frame: list[int] = []
    for item in window_summaries:
        window_id = str(item.get("window_id"))
        selected_per_window[window_id] = int(item.get("selected_visual_token_count") or 0)
        selected_per_frame.extend(int(value) for value in item.get("selected_patches_per_frame", []))
        for scale, count in dict(item.get("selected_patches_per_scale", {})).items():
            selected_per_scale[str(scale)] = selected_per_scale.get(str(scale), 0) + int(count)
    reduction = None
    if original_count:
        reduction = 1.0 - (float(selected_count or 0) / float(original_count))
    path = output_dir / "autogaze" / "token_counts_summary.json"
    _write_json(
        path,
        {
            "original_visual_token_count": original_count,
            "selected_visual_token_count": selected_count,
            "token_reduction_ratio": reduction,
            "selected_patches_per_frame": selected_per_frame,
            "selected_patches_per_scale": selected_per_scale,
            "selected_patches_per_window": selected_per_window,
            "windows": window_summaries,
        },
    )
    return path


def _stage_latency(stages: list[PocStage], *names: str, prefix: str | None = None) -> float | str:
    total = 0.0
    found = False
    for stage in stages:
        if names and stage.name in names or prefix and stage.name.startswith(prefix):
            if stage.latency_ms is not None:
                total += float(stage.latency_ms)
                found = True
    return total if found else "N/A"


def _first_processed_resolution(scaling: Mapping[str, Any] | None) -> list[int] | str:
    if not isinstance(scaling, Mapping):
        return "N/A"
    shape = scaling.get("first_processed_shape")
    if isinstance(shape, list) and len(shape) >= 5:
        return [int(shape[-2]), int(shape[-1])]
    windows = scaling.get("windows")
    if isinstance(windows, list) and windows:
        first = windows[0]
        if isinstance(first, Mapping):
            resolution = first.get("processed_resolution")
            if isinstance(resolution, list):
                return [int(value) for value in resolution]
    return "N/A"


def _first_original_resolution(scaling: Mapping[str, Any] | None) -> list[int] | str:
    if not isinstance(scaling, Mapping):
        return "N/A"
    windows = scaling.get("windows")
    if isinstance(windows, list) and windows:
        first = windows[0]
        if isinstance(first, Mapping):
            resolution = first.get("original_resolution")
            if isinstance(resolution, list):
                return [int(value) for value in resolution]
    return "N/A"


def _build_metrics(summary: PocSummary, memory_snapshot: Any) -> dict[str, Any]:
    token_reduction = None
    if summary.original_visual_token_count:
        token_reduction = 1.0 - (float(summary.selected_token_count or 0) / float(summary.original_visual_token_count))
    frame_selection = summary.frame_selection or {}
    scaling = summary.scaling or {}
    runtime = summary.autogaze_runtime or {}
    memory_unavailable = getattr(memory_snapshot, "peak_vram_mb", "N/A") == "N/A"
    token_summary_path = summary.artifacts.get("token_counts_summary")
    token_summary: dict[str, Any] = {}
    if token_summary_path and Path(token_summary_path).exists():
        token_summary = json.loads(Path(token_summary_path).read_text(encoding="utf-8"))
    chop_metadata_path = summary.artifacts.get("chop_metadata")
    selected_patches_per_chop: Any = "N/A"
    if chop_metadata_path and Path(chop_metadata_path).exists():
        chop_metadata = json.loads(Path(chop_metadata_path).read_text(encoding="utf-8"))
        selected_patches_per_chop = {
            str(item.get("chop_id")): item.get("selected_patch_count")
            for item in chop_metadata.get("chops", [])
        }
    metrics = {
        "mode": summary.mode,
        "status": summary.status,
        "experiment_id": summary.experiment_id,
        "result_label": "real_or_mock_result" if summary.selected_token_count is not None else "stub_or_skipped_result",
        "frame_selection_mode": frame_selection.get("mode", "N/A"),
        "effective_frame_selection_mode": frame_selection.get("effective_mode", "N/A"),
        "scaling_mode": scaling.get("scaling_mode", "N/A"),
        "chop_mode_settings": scaling.get("windows", [{}])[0].get("chop", "N/A") if scaling.get("windows") else "N/A",
        "number_of_frames": frame_selection.get("num_frames", "N/A"),
        "number_of_windows": frame_selection.get("number_of_windows", "N/A"),
        "original_frame_count": frame_selection.get("original_frame_count", "N/A"),
        "original_fps": frame_selection.get("original_fps", "N/A"),
        "original_resolution": _first_original_resolution(scaling),
        "processed_resolution": _first_processed_resolution(scaling),
        "requested_gaze_ratio": runtime.get("requested_gaze_ratio", runtime.get("gaze_ratio")),
        "effective_gaze_ratio": runtime.get("effective_gaze_ratio", runtime.get("gazing_ratio")),
        "requested_task_loss_requirement": runtime.get(
            "requested_task_loss_requirement", runtime.get("task_loss_requirement")
        ),
        "effective_task_loss_requirement": runtime.get(
            "effective_task_loss_requirement", runtime.get("task_loss_requirement")
        ),
        "original_visual_token_count": summary.original_visual_token_count,
        "selected_visual_token_count": summary.selected_token_count,
        "token_reduction_ratio": token_reduction,
        "selected_patches_per_frame": token_summary.get("selected_patches_per_frame", "N/A"),
        "selected_patches_per_scale": token_summary.get("selected_patches_per_scale", "N/A"),
        "selected_patches_per_window": token_summary.get("selected_patches_per_window", "N/A"),
        "selected_patches_per_chop": selected_patches_per_chop,
        "autogaze_latency_ms": _stage_latency(summary.stages, prefix="autogaze_window_"),
        "preprocessing_latency_ms": _stage_latency(summary.stages, "frame_selection_and_decode"),
        "scaling_chop_latency_ms": _stage_latency(summary.stages, "scaling_chop"),
        "visualization_latency_ms": _stage_latency(summary.stages, prefix="visualization_window_"),
        "vision_encoder_latency_ms": _stage_latency(summary.stages, "siglip_vision_encoder"),
        "mllm_prefill_latency_ms": "N/A",
        "mllm_decode_latency_ms": _stage_latency(summary.stages, "nvila_generation"),
        "end_to_end_latency_ms": summary.latency_ms,
        "peak_vram_mb": getattr(memory_snapshot, "peak_vram_mb", "N/A"),
        "memory_metric_unavailable": memory_unavailable,
        "generated_answer": summary.output_text,
        "skipped_stages": summary.skipped_stages,
        "failure_reason": summary.skipped_stages[0]["reason"] if summary.status == "blocked" and summary.skipped_stages else None,
    }
    return _json_safe(metrics)


def _write_metrics_outputs(output_dir: Path, metrics: Mapping[str, Any]) -> dict[str, str]:
    logs_dir = output_dir / "logs"
    json_path = logs_dir / "metrics.json"
    csv_path = logs_dir / "metrics.csv"
    _write_json(json_path, metrics)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        row = {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in metrics.items()
        }
        writer.writerow(row)
    return {"metrics_json": str(json_path), "metrics_csv": str(csv_path)}


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
    frame_selection: dict[str, Any] | None,
    scaling: dict[str, Any] | None,
    autogaze_runtime: dict[str, Any] | None,
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
        frame_selection=frame_selection,
        scaling=scaling,
        autogaze_runtime=autogaze_runtime,
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
    num_frames: int = DEFAULT_NUM_FRAMES,
    frame_selection_mode: str | None = None,
    frame_interval: int | None = None,
    max_windows: int | None = None,
    drop_last: bool | None = None,
    pad_last: bool | None = None,
    scaling_mode: str | None = None,
    resolution: int = 224,
    patch_size: int = 16,
    target_scales: str | list[int] | None = None,
    target_patch_size: int | None = None,
    spatial_tile_size: int | None = None,
    chop_size: int | None = None,
    chop_overlap: int = 0,
    chop_stride: int | None = None,
    max_chops: int | None = None,
    chop_merge_mode: str = "metadata_only",
    save_chop_frames: bool = False,
    save_chop_overlay_video: bool = False,
    gaze_ratio: float | None = None,
    task_loss_requirement: float | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    max_new_tokens: int = 1,
    output_dir: PathLike = "outputs/nvila_hd_video_poc",
    config: PathLike = "configs/experiment/A2_real.yaml",
    no_checkpoint_load: bool = True,
    checkpoint_metadata_only: bool = False,
    save_overlay_video: bool = False,
    save_side_by_side_video: bool = False,
    save_scale_panel_video: bool = False,
    video_fps: float = 4.0,
    video_export_mode: str = "sampled_only",
    overlay_alpha: float = 0.35,
    overlay_line_width: int = 2,
    overlay_style: str = "mask",
    show_patch_boxes: bool | None = None,
    show_patch_indices: bool = False,
    show_scale_labels: bool = False,
    multi_scale_overlay: bool = True,
    scale_color_mode: str = "gradient",
    scale_panel_layout: str = "2x2",
    metadata_placement: str | None = None,
    info_panel_position: str = "bottom",
    info_panel_size: int = 96,
    comparison_layout: str = "processed_overlay",
    info_panel_mode: str = "external",
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> PocSummary:
    if mode not in {"check", "autogaze_only", "full_pipeline"}:
        raise ValueError("mode must be check, autogaze_only, or full_pipeline")
    if video != "dummy" and not video_path:
        raise ValueError("only --video dummy is supported unless --video-path is provided")
    if num_frames <= 0 or resolution <= 0 or max_new_tokens <= 0 or patch_size <= 0:
        raise ValueError("num_frames, resolution, max_new_tokens, and patch_size must be > 0")
    if video_fps <= 0:
        raise ValueError("video_fps must be > 0")
    if gaze_ratio is not None and gaze_ratio <= 0:
        raise ValueError("gaze_ratio must be > 0 when provided")
    if task_loss_requirement is not None and task_loss_requirement <= 0:
        raise ValueError("task_loss_requirement must be > 0 when provided")
    if video_export_mode not in {"sampled_only", "full_length", "hold_last"}:
        raise ValueError("video_export_mode must be sampled_only, full_length, or hold_last")
    if overlay_style not in {"mask", "box", "both"}:
        raise ValueError("overlay_style must be mask, box, or both")
    if scale_color_mode not in {"gradient", "categorical"}:
        raise ValueError("scale_color_mode must be gradient or categorical")
    if scale_panel_layout != "2x2":
        raise NotImplementedError("only --scale-panel-layout 2x2 is currently supported")
    if comparison_layout == "chop_overlay":
        raise NotImplementedError("use --chop-merge-mode overlay_union for chop overlay output; comparison_layout=chop_overlay is not a side-by-side layout")
    if comparison_layout not in {"processed_overlay", "original_overlay", "original_processed_overlay"}:
        raise ValueError("comparison_layout must be processed_overlay, original_overlay, original_processed_overlay, or chop_overlay")
    if metadata_placement is not None:
        if metadata_placement not in {"outside", "inside", "none"}:
            raise ValueError("metadata_placement must be outside, inside, or none")
        info_panel_mode = {"outside": "external", "inside": "inline", "none": "none"}[metadata_placement]
    if info_panel_position != "bottom":
        raise NotImplementedError("only bottom external info panels are currently supported")
    if info_panel_size <= 0:
        raise ValueError("info_panel_size must be > 0")
    if info_panel_mode not in {"external", "inline", "none"}:
        raise ValueError("info_panel_mode must be external, inline, or none")
    if video_export_mode == "hold_last" and (save_overlay_video or save_side_by_side_video or save_scale_panel_video):
        raise NotImplementedError(
            "video_export_mode='hold_last' is not implemented for AutoGaze video export"
        )
    if chop_overlap < 0:
        raise ValueError("chop_overlap must be >= 0")
    if max_chops is not None and max_chops <= 0:
        raise ValueError("max_chops must be > 0 when provided")
    if chop_merge_mode not in {"none", "metadata_only", "overlay_union"}:
        raise ValueError("chop_merge_mode must be none, metadata_only, or overlay_union")

    start = time.perf_counter()
    cfg = _plain_config(_load_config_from_path(config))
    inference_cfg = _get_nested(cfg, "inference")
    inference_cfg = inference_cfg if isinstance(inference_cfg, Mapping) else {}
    resolved_frame_selection_mode = str(frame_selection_mode or inference_cfg.get("frame_selection_mode", "sample"))
    resolved_frame_interval = int(frame_interval if frame_interval is not None else inference_cfg.get("frame_interval", 1))
    resolved_max_windows = max_windows if max_windows is not None else inference_cfg.get("max_windows")
    resolved_max_windows = int(resolved_max_windows) if resolved_max_windows is not None else None
    resolved_drop_last = bool(drop_last if drop_last is not None else inference_cfg.get("drop_last", False))
    resolved_pad_last = bool(pad_last if pad_last is not None else inference_cfg.get("pad_last", False))
    resolved_scaling_mode = str(scaling_mode or inference_cfg.get("scaling_mode", "resize"))
    if comparison_layout in {"original_overlay", "original_processed_overlay"} and resolved_scaling_mode not in {
        "none",
        "resize",
        "fit_short_side",
        "fit_long_side",
        "quickstart",
    }:
        raise NotImplementedError(
            f"{comparison_layout} requires exact processed-to-original affine mapping; "
            f"scaling_mode={resolved_scaling_mode!r} must use --chop-merge-mode overlay_union instead"
        )
    resolved_target_scales = _parse_scales(target_scales if target_scales is not None else inference_cfg.get("target_scales"))
    resolved_target_patch_size = int(
        target_patch_size if target_patch_size is not None else inference_cfg.get("target_patch_size") or patch_size
    )
    resolved_spatial_tile_size = (
        chop_size
        if chop_size is not None
        else spatial_tile_size if spatial_tile_size is not None else inference_cfg.get("chop_size") or inference_cfg.get("spatial_tile_size")
    )
    resolved_spatial_tile_size = int(resolved_spatial_tile_size) if resolved_spatial_tile_size is not None else None
    resolved_chop_overlap = int(chop_overlap if chop_overlap is not None else inference_cfg.get("chop_overlap", 0))
    resolved_chop_stride = chop_stride if chop_stride is not None else inference_cfg.get("chop_stride")
    resolved_chop_stride = int(resolved_chop_stride) if resolved_chop_stride is not None else None
    resolved_max_chops = max_chops if max_chops is not None else inference_cfg.get("max_chops")
    resolved_max_chops = int(resolved_max_chops) if resolved_max_chops is not None else None
    resolved_chop_merge_mode = str(chop_merge_mode or inference_cfg.get("chop_merge_mode", "metadata_only"))
    if save_chop_overlay_video and not (
        resolved_scaling_mode == "chop" and resolved_chop_merge_mode == "overlay_union"
    ):
        raise NotImplementedError("--save-chop-overlay-video requires --scaling-mode chop --chop-merge-mode overlay_union")
    if resolved_scaling_mode == "chop":
        tile = int(resolved_spatial_tile_size or resolution)
        stride = int(resolved_chop_stride or tile)
        if resolved_chop_overlap != 0 or stride != tile:
            raise NotImplementedError("PoC chop mode currently supports non-overlapping chops only")
        resolved_chop_stride = stride
    resolved_gaze_ratio = gaze_ratio if gaze_ratio is not None else inference_cfg.get("gaze_ratio")
    resolved_gaze_ratio = float(resolved_gaze_ratio) if resolved_gaze_ratio is not None else None
    resolved_task_loss_requirement = (
        task_loss_requirement if task_loss_requirement is not None else inference_cfg.get("task_loss_requirement")
    )
    resolved_task_loss_requirement = (
        float(resolved_task_loss_requirement) if resolved_task_loss_requirement is not None else None
    )
    if resolved_gaze_ratio is not None and resolved_gaze_ratio <= 0:
        raise ValueError("gaze_ratio must be > 0 when provided")
    if resolved_task_loss_requirement is not None and resolved_task_loss_requirement <= 0:
        raise ValueError("task_loss_requirement must be > 0 when provided")
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    autogaze = _component(cfg, "autogaze")
    vision = _component(cfg, "vision_encoder")
    mllm = _component(cfg, "mllm")
    effective_device = _device(device)
    torch_dtype = _dtype(dtype)
    memory_tracker = MemoryTracker(effective_device)
    memory_tracker.reset_peak()
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
            frame_selection=None,
            scaling=None,
            autogaze_runtime=None,
            output_text=None,
            stages=stages,
            path_checks=path_checks,
            skipped=skipped,
            artifacts=artifacts,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        metrics = _build_metrics(summary, memory_tracker.snapshot())
        summary.artifacts.update(_write_metrics_outputs(output_root, metrics))
        summary_path = output_root / "logs" / "poc_summary.json"
        summary.artifacts["poc_summary"] = str(summary_path)
        _write_json(summary_path, summary.to_dict())
        return summary

    try:
        preprocess_start = time.perf_counter()
        window_videos, input_video, selection = _prepare_windowed_videos(
            cfg=cfg,
            video=video,
            video_path=video_path,
            num_frames=num_frames,
            resolution=resolution,
            frame_selection_mode=resolved_frame_selection_mode,
            frame_interval=resolved_frame_interval,
            max_windows=resolved_max_windows,
            drop_last=resolved_drop_last,
            pad_last=resolved_pad_last,
        )
        stages.append(
            PocStage(
                "frame_selection_and_decode",
                "passed",
                latency_ms=(time.perf_counter() - preprocess_start) * 1000,
                output_shape=[len(window_videos)],
                details={
                    "frame_selection_mode": selection.mode,
                    "effective_mode": selection.effective_mode,
                    "number_of_windows": len(selection.windows),
                },
            )
        )
        selection_path = _write_frame_selection_metadata(output_root, selection, video_export_mode=video_export_mode)
        artifacts["frame_selection_metadata"] = str(selection_path)
        full_reference_video = None
        if video_export_mode == "full_length":
            full_reference_video = _load_full_reference_video(
                cfg=cfg,
                video=video,
                video_path=video_path,
                original_frame_count=selection.original_frame_count,
                resolution=resolution,
            )
        scaling_start = time.perf_counter()
        scaled_window_videos, scaling_metadata = _scale_window_videos(
            window_videos,
            scaling_mode=resolved_scaling_mode,
            resolution=resolution,
            patch_size=patch_size,
            target_scales=resolved_target_scales,
            target_patch_size=resolved_target_patch_size if resolved_target_scales else None,
            temporal_chunk_size=num_frames if resolved_scaling_mode in {"chop", "spatio_temporal"} else None,
            spatial_tile_size=resolved_spatial_tile_size,
            max_chops=resolved_max_chops,
            chop_overlap=resolved_chop_overlap,
            chop_stride=resolved_chop_stride,
            chop_merge_mode=resolved_chop_merge_mode,
        )
        stages.append(
            PocStage(
                "scaling_chop",
                "passed",
                latency_ms=(time.perf_counter() - scaling_start) * 1000,
                output_shape=list(scaling_metadata.get("first_processed_shape", [])),
                details={
                    "scaling_mode": resolved_scaling_mode,
                    "number_of_windows": len(scaled_window_videos),
                },
            )
        )
        scaling_path = _write_scaling_metadata(output_root, scaling_metadata)
        artifacts["scaling_metadata"] = str(scaling_path)
    except Exception as exc:
        window_videos, scaled_window_videos, input_video, selection, scaling_metadata, full_reference_video = [], [], video_path or video, None, None, None
        skipped.append({"stage": "video_preprocessing", "reason": str(exc)})
        stages.append(PocStage("video_preprocessing", "skipped", skipped_reason=str(exc)))
        if "resolved_scaling_mode" in locals() and resolved_scaling_mode == "quickstart":
            scaling_metadata = {
                "scaling_mode": "quickstart",
                "quickstart_reference_used": "docs/QUICK_START_reference.md",
                "quickstart_exact_match": False,
                "quickstart_differences": [],
                "unsupported_reason": str(exc),
                "status": "unsupported",
            }
            scaling_path = _write_scaling_metadata(output_root, scaling_metadata)
            artifacts["scaling_metadata"] = str(scaling_path)

    video_tensor = scaled_window_videos[0][2] if "scaled_window_videos" in locals() and scaled_window_videos else None
    input_shape = [int(dim) for dim in video_tensor.shape] if video_tensor is not None else None
    selected_count: int | None = None
    original_count: int | None = None
    gaze_output: Mapping[str, Any] | None = None
    vision_features: torch.Tensor | None = None
    output_text: str | None = None
    original_autogaze_args = autogaze.get("original_cli_args") if isinstance(autogaze.get("original_cli_args"), Mapping) else {}
    effective_gaze_ratio = (
        resolved_gaze_ratio if resolved_gaze_ratio is not None else original_autogaze_args.get("gazing_ratio")
    )
    effective_task_loss_requirement = (
        resolved_task_loss_requirement
        if resolved_task_loss_requirement is not None
        else original_autogaze_args.get("task_loss_requirement")
    )
    autogaze_runtime: dict[str, Any] = {
        "gaze_ratio": resolved_gaze_ratio,
        "task_loss_requirement": resolved_task_loss_requirement,
        "requested_gaze_ratio": resolved_gaze_ratio,
        "requested_task_loss_requirement": resolved_task_loss_requirement,
        "effective_gaze_ratio": effective_gaze_ratio,
        "effective_task_loss_requirement": effective_task_loss_requirement,
        "unsupported_runtime_params": [],
        "target_scales": resolved_target_scales,
        "target_patch_size": resolved_target_patch_size if resolved_target_scales else None,
        "scaling_mode": resolved_scaling_mode,
        "visualization": {
            "overlay_style": overlay_style,
            "multi_scale_overlay": multi_scale_overlay,
            "scale_color_mode": scale_color_mode,
            "show_patch_index": show_patch_indices,
            "show_scale_label": show_scale_labels,
            "scale_panel_layout": scale_panel_layout,
            "metadata_placement": metadata_placement or {"external": "outside", "inline": "inside", "none": "none"}[info_panel_mode],
            "info_panel_position": info_panel_position,
            "info_panel_size": info_panel_size,
            "comparison_layout": comparison_layout,
        },
        "chop": {
            "chop_size": resolved_spatial_tile_size if resolved_scaling_mode == "chop" else None,
            "chop_overlap": resolved_chop_overlap if resolved_scaling_mode == "chop" else None,
            "chop_stride": resolved_chop_stride if resolved_scaling_mode == "chop" else None,
            "max_chops": resolved_max_chops if resolved_scaling_mode == "chop" else None,
            "chop_merge_mode": resolved_chop_merge_mode if resolved_scaling_mode == "chop" else None,
            "save_chop_frames": save_chop_frames,
            "save_chop_overlay_video": save_chop_overlay_video,
        },
    }
    window_token_summaries: list[dict[str, Any]] = []
    all_chop_records: list[dict[str, Any]] = []
    all_chop_overlay_frames: list[Any] = []
    all_chop_overlay_metadata: list[dict[str, Any]] = []

    if mode in {"autogaze_only", "full_pipeline"}:
        if not ("scaled_window_videos" in locals() and scaled_window_videos):
            skipped.append({"stage": "autogaze", "reason": "video preprocessing failed"})
            stages.append(PocStage("autogaze", "skipped", skipped_reason="video preprocessing failed"))
        elif no_checkpoint_load or checkpoint_metadata_only:
            reason = "checkpoint loading disabled; pass --allow-checkpoint-load to execute AutoGaze"
            if checkpoint_metadata_only:
                reason = "checkpoint metadata-only mode; AutoGaze execution skipped"
            skipped.append({"stage": "autogaze", "reason": reason})
            stages.append(PocStage("autogaze", "skipped", module_path=_as_str(autogaze.get("module_path")), class_or_factory=_as_str(autogaze.get("class_or_factory")), skipped_reason=reason))
        else:
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
                args = autogaze.get("original_cli_args") if isinstance(autogaze.get("original_cli_args"), Mapping) else {}
                kwargs = {
                    "gazing_ratio": resolved_gaze_ratio if resolved_gaze_ratio is not None else args.get("gazing_ratio", 0.75),
                    "task_loss_requirement": resolved_task_loss_requirement
                    if resolved_task_loss_requirement is not None
                    else args.get("task_loss_requirement", 0.7),
                }
                if resolved_target_scales is not None:
                    kwargs["target_scales"] = resolved_target_scales
                    kwargs["target_patch_size"] = resolved_target_patch_size
                autogaze_runtime.update(kwargs)
                total_original = 0
                total_selected = 0
                combined_videos: list[torch.Tensor] = []
                combined_original_videos: list[torch.Tensor] = []
                combined_patch_rows: list[list[int]] = []
                combined_scale_rows: list[list[int]] = []
                combined_frame_indices: list[int] = []
                for window, original_tensor, window_tensor, window_scaling in scaled_window_videos:
                    stage_start = time.perf_counter()
                    model_input = window_tensor.to(device=effective_device, dtype=torch_dtype)
                    with torch.inference_mode():
                        result = model({"video": model_input}, **kwargs)
                    autogaze_latency_ms = (time.perf_counter() - stage_start) * 1000
                    if not isinstance(result, Mapping):
                        raise TypeError("AutoGaze output must be a mapping")
                    if gaze_output is None:
                        gaze_output = result
                        video_tensor = window_tensor
                    processed_height, processed_width = int(window_tensor.shape[-2]), int(window_tensor.shape[-1])
                    patch_grid = (max(1, processed_height // patch_size), max(1, processed_width // patch_size))
                    window_original, window_selected = _autogaze_token_counts(
                        result,
                        int(window_tensor.shape[1]),
                        processed_height,
                        processed_width,
                        model,
                        patch_size=patch_size,
                    )
                    total_original += window_original
                    total_selected += window_selected
                    selected_per_frame = _selected_patches_per_frame(
                        result,
                        patch_grid,
                        int(window_tensor.shape[1]),
                    )
                    scale_counts = _scale_counts(result)
                    window_token_summaries.append(
                        {
                            "window_id": window.window_id,
                            "frame_indices": window.frame_indices,
                            "original_visual_token_count": window_original,
                            "selected_visual_token_count": window_selected,
                            "token_reduction_ratio": 1.0 - (float(window_selected) / float(window_original))
                            if window_original
                            else None,
                            "selected_patches_per_frame": selected_per_frame,
                            "selected_patches_per_scale": scale_counts,
                            "patch_grid": list(patch_grid),
                            "scaling_mode": window_scaling.get("scaling_mode"),
                        }
                    )
                    artifacts.update(
                        _save_autogaze_artifacts(
                            output_root,
                            result,
                            original_count=window_original,
                            selected_count=window_selected,
                            patch_grid=patch_grid,
                            frame_count=int(window_tensor.shape[1]),
                            runtime=autogaze_runtime,
                            scaling=window_scaling,
                            window=window,
                        )
                    )
                    visualization_start = time.perf_counter()
                    window_visual_artifacts = _maybe_visualize_autogaze(
                        output_root,
                        window_tensor,
                        result,
                        resolution,
                        sampled_frame_indices=window.frame_indices,
                        original_video=original_tensor,
                        full_original_video=full_reference_video if "full_reference_video" in locals() else None,
                        original_frame_count=selection.original_frame_count if isinstance(selection, FrameSelectionResult) else None,
                        original_fps=selection.original_fps if isinstance(selection, FrameSelectionResult) else None,
                        scaling_mode=resolved_scaling_mode,
                        patch_size=patch_size,
                        save_overlay_video=save_overlay_video,
                        save_side_by_side_video=save_side_by_side_video,
                        save_scale_panel_video=save_scale_panel_video,
                        video_fps=video_fps,
                        video_export_mode=video_export_mode,
                        overlay_alpha=overlay_alpha,
                        overlay_line_width=overlay_line_width,
                        overlay_style=overlay_style,
                        show_patch_boxes=show_patch_boxes,
                        show_patch_indices=show_patch_indices,
                        show_scale_labels=show_scale_labels,
                        multi_scale_overlay=multi_scale_overlay,
                        scale_color_mode=scale_color_mode,
                        scale_panel_layout=scale_panel_layout,
                        comparison_layout=comparison_layout,
                        info_panel_mode=info_panel_mode,
                        original_visual_token_count=window_original,
                        selected_visual_token_count=window_selected,
                        visualization_mode=f"{mode}/windows/window_{window.window_id:03d}",
                        artifact_prefix=f"window_{window.window_id:03d}",
                    )
                    canonical_window_artifacts = _maybe_visualize_autogaze(
                        output_root,
                        window_tensor,
                        result,
                        resolution,
                        sampled_frame_indices=window.frame_indices,
                        original_video=original_tensor,
                        full_original_video=full_reference_video if "full_reference_video" in locals() else None,
                        original_frame_count=selection.original_frame_count if isinstance(selection, FrameSelectionResult) else None,
                        original_fps=selection.original_fps if isinstance(selection, FrameSelectionResult) else None,
                        scaling_mode=resolved_scaling_mode,
                        patch_size=patch_size,
                        save_overlay_video=save_overlay_video,
                        save_side_by_side_video=save_side_by_side_video,
                        save_scale_panel_video=save_scale_panel_video,
                        video_fps=video_fps,
                        video_export_mode=video_export_mode,
                        overlay_alpha=overlay_alpha,
                        overlay_line_width=overlay_line_width,
                        overlay_style=overlay_style,
                        show_patch_boxes=show_patch_boxes,
                        show_patch_indices=show_patch_indices,
                        show_scale_labels=show_scale_labels,
                        multi_scale_overlay=multi_scale_overlay,
                        scale_color_mode=scale_color_mode,
                        scale_panel_layout=scale_panel_layout,
                        comparison_layout=comparison_layout,
                        info_panel_mode=info_panel_mode,
                        original_visual_token_count=window_original,
                        selected_visual_token_count=window_selected,
                        visualization_mode=f"autogaze/windows/window_{window.window_id:03d}",
                        artifact_prefix=f"window_{window.window_id:03d}",
                    )
                    visualization_latency_ms = (time.perf_counter() - visualization_start) * 1000
                    artifacts.update({f"window_{window.window_id:03d}_{key}": value for key, value in window_visual_artifacts.items()})
                    artifacts.update(
                        {
                            f"canonical_window_{window.window_id:03d}_{key}": value
                            for key, value in canonical_window_artifacts.items()
                        }
                    )
                    if save_overlay_video or save_side_by_side_video or save_scale_panel_video or window_visual_artifacts:
                        stages.append(
                            PocStage(
                                f"visualization_window_{window.window_id:03d}",
                                "passed" if window_visual_artifacts else "skipped",
                                latency_ms=visualization_latency_ms,
                                skipped_reason=None if window_visual_artifacts else "no visualization artifacts were generated",
                                details={"artifacts": window_visual_artifacts},
                            )
                        )
                    if resolved_scaling_mode == "chop":
                        chop_records, chop_artifacts = _write_chop_outputs(
                            output_root,
                            window=window,
                            window_tensor=window_tensor,
                            gaze_output=result,
                            patch_grid=patch_grid,
                            window_scaling=window_scaling,
                            patch_size=patch_size,
                            chop_overlap=resolved_chop_overlap,
                            chop_stride=resolved_chop_stride,
                            chop_merge_mode=resolved_chop_merge_mode,
                            save_chop_frames=save_chop_frames,
                            overlay_alpha=overlay_alpha,
                            overlay_line_width=overlay_line_width,
                            overlay_style=overlay_style,
                            show_patch_indices=show_patch_indices,
                            show_scale_labels=show_scale_labels,
                            multi_scale_overlay=multi_scale_overlay,
                            scale_color_mode=scale_color_mode,
                            scale_panel_layout=scale_panel_layout,
                            info_panel_mode=info_panel_mode,
                        )
                        all_chop_records.extend(chop_records)
                        artifacts.update(chop_artifacts)
                        if resolved_chop_merge_mode == "overlay_union":
                            overlay_frames, overlay_metadata = _render_chop_overlay_union(
                                output_root,
                                window=window,
                                original_tensor=original_tensor,
                                gaze_output=result,
                                patch_grid=patch_grid,
                                window_scaling=window_scaling,
                                patch_size=patch_size,
                                overlay_alpha=overlay_alpha,
                                overlay_line_width=overlay_line_width,
                                overlay_style=overlay_style,
                                show_patch_indices=show_patch_indices,
                                show_scale_labels=show_scale_labels,
                                multi_scale_overlay=multi_scale_overlay,
                                scale_color_mode=scale_color_mode,
                                info_panel_mode=info_panel_mode,
                                original_visual_token_count=window_original,
                                selected_visual_token_count=window_selected,
                            )
                            all_chop_overlay_frames.extend(overlay_frames)
                            all_chop_overlay_metadata.append(overlay_metadata)
                    patches = _filtered_patch_indices(result, patch_grid)
                    scale_values = _selected_scale_values(result)
                    combined_videos.append(window_tensor[:1])
                    combined_original_videos.append(original_tensor)
                    combined_frame_indices.extend(window.frame_indices)
                    combined_patch_rows.extend([patches for _ in range(int(window_tensor.shape[1]))])
                    combined_scale_rows.extend([scale_values for _ in range(int(window_tensor.shape[1]))])
                    stages.append(
                        PocStage(
                            f"autogaze_window_{window.window_id:03d}",
                            "passed",
                            module_path=_as_str(autogaze.get("module_path")),
                            class_or_factory=_as_str(autogaze.get("class_or_factory")),
                            latency_ms=autogaze_latency_ms,
                            output_shape=_shape(result.get("gazing_pos")),
                            details={
                                "call_kwargs": kwargs,
                                "frame_indices": window.frame_indices,
                                "is_padded": window.is_padded,
                                "scaling": window_scaling,
                                "patch_grid": list(patch_grid),
                            },
                        )
                    )
                original_count = total_original
                selected_count = total_selected
                if (save_overlay_video or save_side_by_side_video or save_scale_panel_video) and combined_videos:
                    combined_tensor = torch.cat(combined_videos, dim=1)
                    combined_original_tensor = torch.cat(combined_original_videos, dim=1)
                    visualization_start = time.perf_counter()
                    artifacts.update(
                        _maybe_visualize_autogaze(
                            output_root,
                            combined_tensor,
                            {
                                "gazing_pos": torch.tensor(combined_patch_rows, dtype=torch.long),
                                "selected_scales": torch.tensor(combined_scale_rows, dtype=torch.long)
                                if combined_scale_rows
                                else None,
                            },
                            resolution,
                            sampled_frame_indices=combined_frame_indices,
                            original_video=combined_original_tensor,
                            full_original_video=full_reference_video if "full_reference_video" in locals() else None,
                            original_frame_count=selection.original_frame_count if isinstance(selection, FrameSelectionResult) else None,
                            original_fps=selection.original_fps if isinstance(selection, FrameSelectionResult) else None,
                            scaling_mode=resolved_scaling_mode,
                            patch_size=patch_size,
                            save_overlay_video=save_overlay_video,
                            save_side_by_side_video=save_side_by_side_video,
                            save_scale_panel_video=save_scale_panel_video,
                            video_fps=video_fps,
                            video_export_mode=video_export_mode,
                            overlay_alpha=overlay_alpha,
                            overlay_line_width=overlay_line_width,
                            overlay_style=overlay_style,
                            show_patch_boxes=show_patch_boxes,
                            show_patch_indices=show_patch_indices,
                            show_scale_labels=show_scale_labels,
                            multi_scale_overlay=multi_scale_overlay,
                            scale_color_mode=scale_color_mode,
                            scale_panel_layout=scale_panel_layout,
                            comparison_layout=comparison_layout,
                            info_panel_mode=info_panel_mode,
                            original_visual_token_count=original_count,
                            selected_visual_token_count=selected_count,
                            visualization_mode=mode,
                            artifact_prefix="combined",
                            output_video_suffix=video_export_mode,
                        )
                    )
                    artifacts.update(
                        {
                            f"canonical_{key}": value
                            for key, value in _maybe_visualize_autogaze(
                                output_root,
                                combined_tensor,
                                {
                                    "gazing_pos": torch.tensor(combined_patch_rows, dtype=torch.long),
                                    "selected_scales": torch.tensor(combined_scale_rows, dtype=torch.long)
                                    if combined_scale_rows
                                    else None,
                                },
                                resolution,
                                sampled_frame_indices=combined_frame_indices,
                                original_video=combined_original_tensor,
                                full_original_video=full_reference_video if "full_reference_video" in locals() else None,
                                original_frame_count=selection.original_frame_count if isinstance(selection, FrameSelectionResult) else None,
                                original_fps=selection.original_fps if isinstance(selection, FrameSelectionResult) else None,
                                scaling_mode=resolved_scaling_mode,
                                patch_size=patch_size,
                                save_overlay_video=save_overlay_video,
                                save_side_by_side_video=save_side_by_side_video,
                                save_scale_panel_video=save_scale_panel_video,
                                video_fps=video_fps,
                                video_export_mode=video_export_mode,
                                overlay_alpha=overlay_alpha,
                                overlay_line_width=overlay_line_width,
                                overlay_style=overlay_style,
                                show_patch_boxes=show_patch_boxes,
                                show_patch_indices=show_patch_indices,
                                show_scale_labels=show_scale_labels,
                                multi_scale_overlay=multi_scale_overlay,
                                scale_color_mode=scale_color_mode,
                                scale_panel_layout=scale_panel_layout,
                                comparison_layout=comparison_layout,
                                info_panel_mode=info_panel_mode,
                                original_visual_token_count=original_count,
                                selected_visual_token_count=selected_count,
                                visualization_mode="autogaze",
                                artifact_prefix="combined",
                            ).items()
                        }
                    )
                    stages.append(
                        PocStage(
                            "visualization_combined_sampled_only",
                            "passed",
                            latency_ms=(time.perf_counter() - visualization_start) * 1000,
                            details={"video_export_mode": video_export_mode},
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
                processor = _from_pretrained(
                    processor_class,
                    processor_path,
                    mllm,
                    **_processor_kwargs(
                        mllm,
                        gaze_ratio=resolved_gaze_ratio,
                        task_loss_requirement=resolved_task_loss_requirement,
                    ),
                )
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
    if mode in {"autogaze_only", "full_pipeline"}:
        runtime_path = _write_runtime_metadata(output_root, autogaze_runtime)
        artifacts["autogaze_runtime_metadata"] = str(runtime_path)
        token_summary_path = _write_token_counts_summary(
            output_root,
            original_count=original_count,
            selected_count=selected_count,
            window_summaries=window_token_summaries,
        )
        artifacts["token_counts_summary"] = str(token_summary_path)
        if all_chop_records:
            chop_metadata_path = _write_chop_metadata(output_root, all_chop_records)
            artifacts["chop_metadata"] = str(chop_metadata_path)
        if all_chop_overlay_metadata:
            chop_overlay_metadata = {
                "status": "implemented",
                "merge_mode": "overlay_union",
                "number_of_windows": len(all_chop_overlay_metadata),
                "windows": all_chop_overlay_metadata,
                "video_path": None,
            }
            if all_chop_overlay_frames and (save_chop_overlay_video or resolved_chop_merge_mode == "overlay_union"):
                from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer

                visualizer = AutoGazeVisualizer(output_root=output_root.parent, exp_name=output_root.name)
                video_path = output_root / "visualizations" / "autogaze" / "videos" / "autogaze_chop_overlay.mp4"
                visualizer._write_mp4(video_path, all_chop_overlay_frames, fps=video_fps)
                artifacts["chop_overlay_video"] = str(video_path)
                chop_overlay_metadata["video_path"] = str(video_path)
            chop_overlay_metadata_path = _write_chop_overlay_metadata(output_root, chop_overlay_metadata)
            artifacts["chop_overlay_metadata"] = str(chop_overlay_metadata_path)
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
        frame_selection=selection.to_dict() if isinstance(selection, FrameSelectionResult) else None,
        scaling=dict(scaling_metadata) if isinstance(scaling_metadata, Mapping) else None,
        autogaze_runtime=autogaze_runtime,
        output_text=output_text,
        stages=stages,
        path_checks=path_checks,
        skipped=skipped,
        artifacts=artifacts,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
    metrics = _build_metrics(summary, memory_tracker.snapshot())
    summary.artifacts.update(_write_metrics_outputs(output_root, metrics))
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
    if summary.frame_selection:
        print(f"frame_selection_mode: {summary.frame_selection.get('mode')}")
        print(f"effective_frame_selection_mode: {summary.frame_selection.get('effective_mode')}")
        print(f"num_frames_per_window: {summary.frame_selection.get('num_frames')}")
        print(f"number_of_windows: {summary.frame_selection.get('number_of_windows')}")
    if summary.scaling:
        print(f"scaling_mode: {summary.scaling.get('scaling_mode')}")
        print(f"processed_shape: {summary.scaling.get('first_processed_shape')}")
    if summary.autogaze_runtime:
        print(f"autogaze_runtime: {summary.autogaze_runtime}")
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
    parser = argparse.ArgumentParser(
        description="Isolated NVILA-HD-Video canonical PoC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["check", "autogaze_only", "full_pipeline"], default="check")
    parser.add_argument("--video", choices=["dummy"], default="dummy")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of frames per model forward pass. Default follows the canonical AutoGaze/NVILA setting.",
    )
    parser.add_argument(
        "--frame-selection-mode",
        choices=["sample", "chunk", "interval", "all"],
        default=None,
        help="How to select frame windows. num-frames is the number of frames per forward pass.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="Frame interval used only by --frame-selection-mode interval.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of non-overlapping inference windows to process.",
    )
    parser.add_argument(
        "--drop-last",
        action="store_true",
        default=None,
        help="Drop incomplete final windows in chunk/all mode.",
    )
    parser.add_argument(
        "--pad-last",
        action="store_true",
        default=None,
        help="Pad incomplete final windows by repeating the last available frame.",
    )
    parser.add_argument(
        "--scaling-mode",
        choices=["none", "resize", "fit_short_side", "fit_long_side", "quickstart", "chop"],
        default=None,
        help="How selected frame windows are scaled before AutoGaze.",
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--target-scales", default=None, help="Comma- or plus-separated AutoGaze target scales.")
    parser.add_argument("--target-patch-size", type=int, default=None)
    parser.add_argument("--spatial-tile-size", type=int, default=None, help="Tile size for --scaling-mode chop.")
    parser.add_argument("--gaze-ratio", type=float, default=None)
    parser.add_argument("--task-loss-requirement", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/nvila_hd_video_poc")
    parser.add_argument("--config", default="configs/experiment/A2_real.yaml")
    parser.add_argument("--no-checkpoint-load", action="store_true", default=False)
    parser.add_argument("--allow-checkpoint-load", action="store_true", help="Explicitly allow loading real checkpoints.")
    parser.add_argument("--checkpoint-metadata-only", action="store_true")
    parser.add_argument("--save-overlay-video", action="store_true")
    parser.add_argument("--save-side-by-side-video", action="store_true")
    parser.add_argument("--save-scale-panel-video", action="store_true")
    parser.add_argument("--save-chop-frames", action="store_true")
    parser.add_argument("--save-chop-overlay-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument(
        "--video-export-mode",
        choices=["sampled_only", "full_length", "hold_last"],
        default="sampled_only",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--overlay-line-width", type=int, default=2)
    parser.add_argument("--overlay-style", choices=["mask", "box", "both"], default="mask")
    parser.add_argument("--multi-scale-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-color-mode", choices=["gradient", "categorical"], default="gradient")
    parser.add_argument("--scale-panel-layout", choices=["2x2"], default="2x2")
    parser.add_argument("--show-patch-index", action="store_true")
    parser.add_argument("--show-scale-label", action="store_true")
    parser.add_argument("--hide-patch-boxes", action="store_true")
    parser.add_argument("--hide-patch-indices", action="store_true")
    parser.add_argument("--metadata-placement", choices=["outside", "inside", "none"], default=None)
    parser.add_argument("--info-panel-position", choices=["bottom", "right"], default="bottom")
    parser.add_argument("--info-panel-size", type=int, default=96)
    parser.add_argument("--info-panel-mode", choices=["external", "inline", "none"], default="external")
    parser.add_argument("--comparison-layout", choices=["processed_overlay", "original_overlay", "original_processed_overlay", "chop_overlay"], default="processed_overlay")
    parser.add_argument("--chop-size", type=int, default=None)
    parser.add_argument("--chop-overlap", type=int, default=0)
    parser.add_argument("--chop-stride", type=int, default=None)
    parser.add_argument("--max-chops", type=int, default=None)
    parser.add_argument("--chop-merge-mode", choices=["none", "metadata_only", "overlay_union"], default="metadata_only")
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
        frame_selection_mode=args.frame_selection_mode,
        frame_interval=args.frame_interval,
        max_windows=args.max_windows,
        drop_last=args.drop_last,
        pad_last=args.pad_last,
        scaling_mode=args.scaling_mode,
        resolution=args.resolution,
        patch_size=args.patch_size,
        target_scales=args.target_scales,
        target_patch_size=args.target_patch_size,
        spatial_tile_size=args.spatial_tile_size,
        chop_size=args.chop_size,
        chop_overlap=args.chop_overlap,
        chop_stride=args.chop_stride,
        max_chops=args.max_chops,
        chop_merge_mode=args.chop_merge_mode,
        save_chop_frames=args.save_chop_frames,
        save_chop_overlay_video=args.save_chop_overlay_video,
        gaze_ratio=args.gaze_ratio,
        task_loss_requirement=args.task_loss_requirement,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        config=args.config,
        no_checkpoint_load=no_checkpoint_load,
        checkpoint_metadata_only=args.checkpoint_metadata_only,
        save_overlay_video=args.save_overlay_video,
        save_side_by_side_video=args.save_side_by_side_video,
        save_scale_panel_video=args.save_scale_panel_video,
        video_fps=args.video_fps,
        video_export_mode=args.video_export_mode,
        overlay_alpha=args.overlay_alpha,
        overlay_line_width=args.overlay_line_width,
        overlay_style=args.overlay_style,
        show_patch_boxes=False if args.hide_patch_boxes else None,
        show_patch_indices=args.show_patch_index and not args.hide_patch_indices,
        show_scale_labels=args.show_scale_label,
        multi_scale_overlay=args.multi_scale_overlay,
        scale_color_mode=args.scale_color_mode,
        scale_panel_layout=args.scale_panel_layout,
        metadata_placement=args.metadata_placement,
        info_panel_position=args.info_panel_position,
        info_panel_size=args.info_panel_size,
        comparison_layout=args.comparison_layout,
        info_panel_mode=args.info_panel_mode,
    )
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    else:
        print_summary(summary)
    return 0 if summary.status in {"passed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
