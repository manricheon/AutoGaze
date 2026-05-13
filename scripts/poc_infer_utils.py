#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]


SCALE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (254, 240, 138),
    1: (251, 146, 60),
    2: (244, 114, 182),
    3: (168, 85, 247),
}


@dataclass(frozen=True)
class FrameWindow:
    window_id: int
    frame_indices: list[int]
    is_padded: bool
    padded_frame_mask: list[bool]
    original_frame_count: int
    effective_num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSelectionResult:
    mode: str
    effective_mode: str
    num_frames: int
    frame_interval: int
    max_windows: int | None
    drop_last: bool
    pad_last: bool
    original_frame_count: int
    original_fps: float | None
    windows: list[FrameWindow]
    unsupported_visualization_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["number_of_windows"] = len(self.windows)
        data["window_frame_indices"] = [window.frame_indices for window in self.windows]
        return data


class FrameSelector:
    SUPPORTED_MODES = {"sample", "chunk", "interval", "all"}

    def __init__(
        self,
        *,
        mode: str,
        num_frames: int,
        frame_interval: int = 1,
        max_windows: int | None = None,
        drop_last: bool = False,
        pad_last: bool = False,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported frame_selection_mode: {mode}")
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be > 0")
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be > 0 when provided")
        if drop_last and pad_last:
            raise ValueError("drop_last and pad_last cannot both be true")
        self.mode = mode
        self.num_frames = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.max_windows = int(max_windows) if max_windows is not None else None
        self.drop_last = bool(drop_last)
        self.pad_last = bool(pad_last)

    def select(self, *, original_frame_count: int, original_fps: float | None = None) -> FrameSelectionResult:
        if original_frame_count <= 0:
            raise ValueError("original_frame_count must be > 0")
        effective_mode = "chunk" if self.mode == "all" else self.mode
        if self.mode == "sample":
            windows = self._sample(original_frame_count)
        elif self.mode == "interval":
            windows = self._interval(original_frame_count)
        else:
            windows = self._chunk(original_frame_count)
        if self.max_windows is not None:
            windows = windows[: self.max_windows]
        windows = [
            FrameWindow(
                window_id=index,
                frame_indices=window.frame_indices,
                is_padded=window.is_padded,
                padded_frame_mask=window.padded_frame_mask,
                original_frame_count=window.original_frame_count,
                effective_num_frames=window.effective_num_frames,
            )
            for index, window in enumerate(windows)
        ]
        return FrameSelectionResult(
            mode=self.mode,
            effective_mode=effective_mode,
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            max_windows=self.max_windows,
            drop_last=self.drop_last,
            pad_last=self.pad_last,
            original_frame_count=original_frame_count,
            original_fps=original_fps,
            windows=windows,
            unsupported_visualization_modes=["hold_last"],
        )

    def _sample(self, total: int) -> list[FrameWindow]:
        if total >= self.num_frames:
            if self.num_frames == 1:
                indices = [0]
            else:
                step = (total - 1) / float(self.num_frames - 1)
                indices = [round(step * idx) for idx in range(self.num_frames)]
            return [self._window(0, indices, total)]
        return self._short_window(0, list(range(total)), total)

    def _interval(self, total: int) -> list[FrameWindow]:
        indices = [idx * self.frame_interval for idx in range(self.num_frames)]
        indices = [idx for idx in indices if idx < total]
        return self._short_window(0, indices, total)

    def _chunk(self, total: int) -> list[FrameWindow]:
        windows: list[FrameWindow] = []
        start = 0
        window_id = 0
        while start < total:
            indices = list(range(start, min(start + self.num_frames, total)))
            maybe_window = self._short_window(window_id, indices, total)
            if maybe_window:
                windows.extend(maybe_window)
                window_id += 1
            start += self.num_frames
        return windows

    def _short_window(self, window_id: int, indices: list[int], total: int) -> list[FrameWindow]:
        if len(indices) == self.num_frames:
            return [self._window(window_id, indices, total)]
        if self.drop_last:
            return []
        if self.pad_last and indices:
            padded = indices + [indices[-1]] * (self.num_frames - len(indices))
            return [self._window(window_id, padded, total, padded_count=self.num_frames - len(indices))]
        if not indices:
            return []
        return [self._window(window_id, indices, total)]

    @staticmethod
    def _window(window_id: int, indices: list[int], total: int, padded_count: int = 0) -> FrameWindow:
        mask = [False] * len(indices)
        if padded_count:
            mask[-padded_count:] = [True] * padded_count
        return FrameWindow(
            window_id=window_id,
            frame_indices=indices,
            is_padded=bool(padded_count),
            padded_frame_mask=mask,
            original_frame_count=total,
            effective_num_frames=len(indices) - padded_count,
        )


def select_frame_windows(
    *,
    original_frame_count: int,
    num_frames: int,
    frame_selection_mode: str = "sample",
    frame_interval: int = 1,
    max_windows: int | None = None,
    drop_last: bool = False,
    pad_last: bool = False,
    original_fps: float | None = None,
) -> FrameSelectionResult:
    return FrameSelector(
        mode=frame_selection_mode,
        num_frames=num_frames,
        frame_interval=frame_interval,
        max_windows=max_windows,
        drop_last=drop_last,
        pad_last=pad_last,
    ).select(original_frame_count=original_frame_count, original_fps=original_fps)


@dataclass(frozen=True)
class PreparedVideo:
    source_video: torch.Tensor
    processed_video: torch.Tensor
    frame_selection: FrameSelectionResult
    frame_records: list[dict[str, Any]]
    scaling_metadata: dict[str, Any]
    chop_metadata: dict[str, Any] | None
    video_source_kind: str
    original_fps: float | None


@dataclass(frozen=True)
class GazeResult:
    status: str
    reason: str | None
    real_model_used: bool
    autogaze_enabled: bool
    original_token_count: int
    selected_token_count: int
    token_reduction_ratio: float | None
    patch_grid: tuple[int, int]
    patch_size: int
    per_frame: list[dict[str, Any]]
    runtime_metadata: dict[str, Any]
    latency_ms: float
    gazing_info_for_vit: Mapping[str, Any] | None = None


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return value


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in json_safe(dict(row)).items()
            }
        )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    data = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    if not isinstance(data, dict):
        raise TypeError(f"config must resolve to a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def checkpoint_exists(value: Any) -> bool:
    if value in {None, ""}:
        return False
    text = str(value)
    if text.startswith((".", "/", "~", "weights/", "checkpoints/")):
        return resolve_path(text).exists()
    # Treat non-local identifiers as potentially loadable model IDs.
    return "/" in text


def model_reference_for_loading(value: Any) -> str:
    text = str(value)
    if text.startswith((".", "/", "~", "weights/", "checkpoints/")):
        return str(resolve_path(text))
    return text


def nested_get(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def cli_or_config(value: Any, cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    return value if value is not None else nested_get(cfg, dotted, default)


def normalize_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def make_dummy_video(num_frames: int = 8, height: int = 64, width: int = 64) -> torch.Tensor:
    yy = torch.linspace(0.0, 1.0, steps=height).view(height, 1).expand(height, width)
    xx = torch.linspace(0.0, 1.0, steps=width).view(1, width).expand(height, width)
    frames = []
    denom = max(1, num_frames - 1)
    for idx in range(num_frames):
        phase = idx / denom
        frame = torch.stack(
            [
                (xx + phase).fmod(1.0),
                (yy * 0.75 + phase * 0.25).clamp(0.0, 1.0),
                torch.full_like(xx, phase),
            ],
            dim=0,
        )
        frames.append(frame)
    return torch.stack(frames, dim=0)


def load_video_frames(video_path: str, *, dummy_frames: int, dummy_resolution: int) -> tuple[torch.Tensor, str, float | None]:
    if video_path == "dummy":
        return make_dummy_video(dummy_frames, dummy_resolution, dummy_resolution), "dummy", None
    path = resolve_path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video file does not exist: {path}")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to decode real video files") from exc

    frames: list[torch.Tensor] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            frames.append(tensor)
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return torch.stack(frames, dim=0), "file", fps


def prepare_video(
    cfg: Mapping[str, Any],
    *,
    video_path: str,
    frame_selection_mode: str,
    num_frames: int,
    frame_interval: int,
    max_windows: int | None,
    scaling_mode: str,
    resolution: int,
    chop_size: int,
    chop_overlap: int,
    max_chops: int | None,
    chop_merge_mode: str,
) -> PreparedVideo:
    dummy_frames = int(nested_get(cfg, "input.dummy_frames", max(num_frames, 8)))
    dummy_resolution = int(nested_get(cfg, "input.dummy_resolution", max(resolution, 64)))
    source_video, source_kind, fps = load_video_frames(
        video_path,
        dummy_frames=dummy_frames,
        dummy_resolution=dummy_resolution,
    )
    selection = select_frame_windows(
        original_frame_count=int(source_video.shape[0]),
        num_frames=num_frames,
        frame_selection_mode=frame_selection_mode,
        frame_interval=frame_interval,
        max_windows=max_windows,
        original_fps=fps,
    )

    processed_windows: list[torch.Tensor] = []
    frame_records: list[dict[str, Any]] = []
    scaling_windows: list[dict[str, Any]] = []
    chop_metadata: dict[str, Any] | None = None
    processed_index = 0
    for window in selection.windows:
        window_video = source_video[window.frame_indices].unsqueeze(0)
        scaled, scaling_record, window_chops = scale_video(
            window_video,
            scaling_mode=scaling_mode,
            resolution=resolution,
            chop_size=chop_size,
            chop_overlap=chop_overlap,
            max_chops=max_chops,
            chop_merge_mode=chop_merge_mode,
        )
        processed_windows.append(scaled)
        scaling_record["window"] = window.to_dict()
        scaling_windows.append(scaling_record)
        if window_chops is not None:
            if chop_metadata is None:
                chop_metadata = {
                    "mode": "chop",
                    "chop_size": chop_size,
                    "chop_overlap": chop_overlap,
                    "chop_merge_mode": chop_merge_mode,
                    "windows": [],
                }
            chop_metadata["windows"].append({"window_id": window.window_id, **window_chops})
        for position, source_index in enumerate(window.frame_indices):
            frame_records.append(
                {
                    "processed_frame_index": processed_index,
                    "source_frame_index": int(source_index),
                    "window_id": int(window.window_id),
                    "position_in_window": int(position),
                    "anchor_frame_index": None,
                    "is_padded": bool(window.padded_frame_mask[position]),
                }
            )
            processed_index += 1

    processed_video = torch.cat(processed_windows, dim=1) if processed_windows else source_video[:0].unsqueeze(0)
    scaling_metadata = {
        "scaling_mode": scaling_mode,
        "resolution": resolution,
        "number_of_windows": len(selection.windows),
        "first_processed_shape": [int(dim) for dim in processed_video.shape],
        "windows": scaling_windows,
    }
    return PreparedVideo(
        source_video=source_video,
        processed_video=processed_video,
        frame_selection=selection,
        frame_records=frame_records,
        scaling_metadata=scaling_metadata,
        chop_metadata=chop_metadata,
        video_source_kind=source_kind,
        original_fps=fps,
    )


def scale_video(
    video: torch.Tensor,
    *,
    scaling_mode: str,
    resolution: int,
    chop_size: int,
    chop_overlap: int,
    max_chops: int | None,
    chop_merge_mode: str,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any] | None]:
    if video.ndim != 5:
        raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = [int(dim) for dim in video.shape]
    notes: list[str] = []
    chop_metadata: dict[str, Any] | None = None
    mode = str(scaling_mode)
    if mode == "none":
        output = video
        notes.append("No scaling was applied.")
    elif mode == "resize":
        output = _resize_video_hw(video, resolution, resolution)
        notes.append("Frames were resized to a square resolution.")
    elif mode == "fit_short_side":
        scale = resolution / float(min(height, width))
        output = _resize_video_hw(video, max(1, round(height * scale)), max(1, round(width * scale)))
        notes.append("Aspect ratio preserved; short side matches resolution.")
    elif mode == "fit_long_side":
        scale = resolution / float(max(height, width))
        output = _resize_video_hw(video, max(1, round(height * scale)), max(1, round(width * scale)))
        notes.append("Aspect ratio preserved; long side matches resolution.")
    elif mode == "quickstart":
        if resolution not in {224, 392}:
            raise NotImplementedError("quickstart scaling is implemented only for 224 and 392 resolution policies")
        output = _resize_video_hw(video, resolution, resolution)
        notes.append("QUICK_START-compatible square resize policy was applied.")
    elif mode == "chop":
        output = _resize_video_hw(video, resolution, resolution)
        chop_metadata = generate_chop_metadata(
            height=height,
            width=width,
            frames=frames,
            chop_size=chop_size,
            chop_overlap=chop_overlap,
            max_chops=max_chops,
            merge_mode=chop_merge_mode,
        )
        notes.append("Chop metadata was generated; Priority 1 keeps visual outputs flat over processed frames.")
    else:
        raise ValueError("scaling_mode must be one of resize, fit_short_side, fit_long_side, chop, quickstart, none")

    processed_h, processed_w = int(output.shape[-2]), int(output.shape[-1])
    record = {
        "status": "ready",
        "scaling_mode": mode,
        "resolution": resolution,
        "original_shape": [batch, frames, channels, height, width],
        "processed_shape": [int(dim) for dim in output.shape],
        "original_resolution": [height, width],
        "processed_resolution": [processed_h, processed_w],
        "notes": notes,
        "coordinate_mapping": {
            "processed_from_original": "chop_metadata_with_resize" if mode == "chop" else mode,
            "scale_x": processed_w / float(width),
            "scale_y": processed_h / float(height),
            "inverse_scale_x": width / float(processed_w),
            "inverse_scale_y": height / float(processed_h),
        },
    }
    return output, record, chop_metadata


def _resize_video_hw(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, frames, channels, old_h, old_w = video.shape
    if int(old_h) == int(height) and int(old_w) == int(width):
        return video
    flat = video.reshape(batch * frames, channels, old_h, old_w)
    resized = F.interpolate(flat, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, int(height), int(width))


def generate_chop_metadata(
    *,
    height: int,
    width: int,
    frames: int,
    chop_size: int,
    chop_overlap: int,
    max_chops: int | None,
    merge_mode: str,
) -> dict[str, Any]:
    if chop_size <= 0:
        raise ValueError("chop_size must be > 0")
    if chop_overlap < 0 or chop_overlap >= chop_size:
        raise ValueError("chop_overlap must be >= 0 and < chop_size")
    stride = chop_size - chop_overlap
    records: list[dict[str, int]] = []
    for frame_index in range(frames):
        for y in range(0, max(1, height), stride):
            y0 = min(y, max(0, height - chop_size))
            y1 = min(height, y0 + chop_size)
            for x in range(0, max(1, width), stride):
                x0 = min(x, max(0, width - chop_size))
                x1 = min(width, x0 + chop_size)
                item = {
                    "chop_index": len(records),
                    "frame_index_within_window": frame_index,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
                if item not in records:
                    records.append(item)
                if max_chops is not None and len(records) >= max_chops:
                    break
            if max_chops is not None and len(records) >= max_chops:
                break
        if max_chops is not None and len(records) >= max_chops:
            break
    return {
        "status": "metadata_only",
        "frame_count": frames,
        "source_resolution": [height, width],
        "chop_size": chop_size,
        "chop_overlap": chop_overlap,
        "stride": stride,
        "max_chops": max_chops,
        "merge_mode": merge_mode,
        "records": records,
    }


def run_autogaze_stage(
    cfg: Mapping[str, Any],
    prepared: PreparedVideo,
    *,
    device: str,
    dtype: str,
    gaze_ratio: float,
    task_loss_requirement: float | None,
    strict_autogaze_params: bool,
    allow_real_model_loading: bool,
) -> GazeResult:
    start = time.perf_counter()
    autogaze_enabled = bool(nested_get(cfg, "autogaze.enabled", False))
    checkpoint = nested_get(cfg, "autogaze.checkpoint_path") or nested_get(cfg, "autogaze.model_id")
    reason: str | None = None
    real_model_used = False
    runtime: dict[str, Any] = {
        "gaze_ratio": gaze_ratio,
        "task_loss_requirement": task_loss_requirement,
        "strict_autogaze_params": strict_autogaze_params,
        "requested_checkpoint": checkpoint,
        "allow_real_model_loading": allow_real_model_loading,
        "patch_size": int(nested_get(cfg, "scaling.patch_size", 16)),
        "scales": configured_scales(cfg),
    }

    if not autogaze_enabled:
        status = "disabled_full_token_path"
        reason = "AutoGaze is disabled by config; all visual tokens are retained."
        result = build_gaze_result(
            prepared,
            autogaze_enabled=False,
            status=status,
            reason=reason,
            real_model_used=False,
            gaze_ratio=1.0,
            task_loss_requirement=task_loss_requirement,
            runtime_metadata=runtime,
            start_time=start,
        )
        return result

    has_checkpoint = checkpoint_exists(checkpoint)
    if allow_real_model_loading and not has_checkpoint:
        reason = f"AutoGaze checkpoint/model is missing: {checkpoint}"
        return build_gaze_result(
            prepared,
            autogaze_enabled=True,
            status="blocked",
            reason=reason,
            real_model_used=False,
            gaze_ratio=gaze_ratio,
            task_loss_requirement=task_loss_requirement,
            runtime_metadata=runtime,
            start_time=start,
        )
    if allow_real_model_loading and has_checkpoint:
        try:
            result = _run_real_autogaze(
                cfg,
                prepared,
                device=device,
                dtype=dtype,
                gaze_ratio=gaze_ratio,
                task_loss_requirement=task_loss_requirement,
                runtime_metadata=runtime,
                start_time=start,
            )
            return result
        except Exception as exc:  # pragma: no cover - real model path is environment-dependent.
            reason = f"real AutoGaze execution failed: {exc}"
            runtime["real_execution_error"] = reason

    if prepared.video_source_kind == "dummy":
        reason = reason or "dummy video requested; using explicit stub AutoGaze metadata, not real model output"
        return build_gaze_result(
            prepared,
            autogaze_enabled=True,
            status="stub_dummy_autogaze",
            reason=reason,
            real_model_used=False,
            gaze_ratio=gaze_ratio,
            task_loss_requirement=task_loss_requirement,
            runtime_metadata=runtime,
            start_time=start,
        )

    if not has_checkpoint:
        reason = f"AutoGaze checkpoint/model is missing: {checkpoint}"
    elif not allow_real_model_loading:
        reason = "real AutoGaze loading is disabled; pass --allow-real-model-loading to execute checkpoints"
    return build_gaze_result(
        prepared,
        autogaze_enabled=True,
        status="blocked",
        reason=reason,
        real_model_used=False,
        gaze_ratio=gaze_ratio,
        task_loss_requirement=task_loss_requirement,
        runtime_metadata=runtime,
        start_time=start,
    )


def _run_real_autogaze(
    cfg: Mapping[str, Any],
    prepared: PreparedVideo,
    *,
    device: str,
    dtype: str,
    gaze_ratio: float,
    task_loss_requirement: float | None,
    runtime_metadata: dict[str, Any],
    start_time: float,
) -> GazeResult:
    from autogaze.models.autogaze import AutoGaze

    checkpoint = nested_get(cfg, "autogaze.checkpoint_path") or nested_get(cfg, "autogaze.model_id")
    model = AutoGaze.from_pretrained(model_reference_for_loading(checkpoint))
    model = model.to(normalize_device(device))
    model.eval()
    video = prepared.processed_video.to(device=normalize_device(device), dtype=dtype_from_name(dtype))
    kwargs: dict[str, Any] = {"gazing_ratio": gaze_ratio}
    if task_loss_requirement is not None:
        kwargs["task_loss_requirement"] = task_loss_requirement
    with torch.inference_mode():
        outputs = model({"video": video}, **kwargs)
    return build_gaze_result(
        prepared,
        autogaze_enabled=True,
        status="real",
        reason=None,
        real_model_used=True,
        gaze_ratio=gaze_ratio,
        task_loss_requirement=task_loss_requirement,
        runtime_metadata={**runtime_metadata, "raw_output_keys": sorted(str(key) for key in outputs.keys())},
        start_time=start_time,
        real_outputs=outputs,
    )


def build_gaze_result(
    prepared: PreparedVideo,
    *,
    autogaze_enabled: bool,
    status: str,
    reason: str | None,
    real_model_used: bool,
    gaze_ratio: float,
    task_loss_requirement: float | None,
    runtime_metadata: dict[str, Any],
    start_time: float,
    real_outputs: Mapping[str, Any] | None = None,
) -> GazeResult:
    video = prepared.processed_video
    _, frame_count, _, height, width = [int(dim) for dim in video.shape]
    patch_size = int(runtime_metadata.get("patch_size") or 16)
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    scale_layout = build_scale_layout(runtime_metadata.get("scales"), patch_size=patch_size, fallback_resolution=max(height, width))
    tokens_per_frame = sum(item["token_count"] for item in scale_layout)
    per_frame: list[dict[str, Any]] = []

    if real_outputs is not None and "gazing_pos" in real_outputs and "num_gazing_each_frame" in real_outputs:
        gazing_pos = real_outputs["gazing_pos"][0].detach().cpu()
        padded = real_outputs.get("if_padded_gazing")
        padded_flat = padded[0].detach().cpu().bool() if isinstance(padded, torch.Tensor) else torch.zeros_like(gazing_pos).bool()
        counts = real_outputs["num_gazing_each_frame"].detach().cpu().tolist()
        tokens_each = int(real_outputs.get("num_vision_tokens_each_frame", tokens_per_frame))
        offset = 0
        for idx in range(frame_count):
            count = int(counts[idx]) if idx < len(counts) else 0
            pos = gazing_pos[offset : offset + count]
            pad = padded_flat[offset : offset + count]
            local = [int((item - idx * tokens_each).item()) % tokens_each for item in pos[~pad]]
            per_frame.append(_frame_gaze_record(prepared.frame_records[idx], local, tokens_each, idx, scale_layout))
            offset += count
    else:
        selected_count = tokens_per_frame if not autogaze_enabled else max(1, min(tokens_per_frame, math.ceil(tokens_per_frame * gaze_ratio)))
        for idx in range(frame_count):
            local_indices = _deterministic_patch_indices(idx, tokens_per_frame, selected_count)
            per_frame.append(_frame_gaze_record(prepared.frame_records[idx], local_indices, tokens_per_frame, idx, scale_layout))

    original = sum(int(item["original_token_count"]) for item in per_frame)
    selected = sum(int(item["selected_token_count"]) for item in per_frame)
    reduction = None if original == 0 else 1.0 - (selected / float(original))
    runtime = {
        **runtime_metadata,
        "status": status,
        "reason": reason,
        "real_model_used": real_model_used,
        "patch_grid": [grid_h, grid_w],
        "patch_size": patch_size,
        "scale_layout": scale_layout,
        "gaze_ratio": gaze_ratio,
        "task_loss_requirement": task_loss_requirement,
        "encoder_side_acceleration_claimed": bool(real_model_used and autogaze_enabled),
    }
    gazing_info_for_vit = None
    if real_outputs is not None:
        required = ("gazing_pos", "num_gazing_each_frame", "if_padded_gazing")
        if all(key in real_outputs for key in required):
            gazing_info_for_vit = {key: real_outputs[key] for key in required}
    return GazeResult(
        status=status,
        reason=reason,
        real_model_used=real_model_used,
        autogaze_enabled=autogaze_enabled,
        original_token_count=original,
        selected_token_count=selected,
        token_reduction_ratio=reduction,
        patch_grid=(grid_h, grid_w),
        patch_size=patch_size,
        per_frame=per_frame,
        runtime_metadata=runtime,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        gazing_info_for_vit=gazing_info_for_vit,
    )


def _deterministic_patch_indices(frame_idx: int, total: int, selected_count: int) -> list[int]:
    if selected_count >= total:
        return list(range(total))
    step = max(1, total // selected_count)
    values = [((frame_idx + offset * step) % total) for offset in range(selected_count)]
    return sorted(set(values))[:selected_count]


def configured_scales(cfg: Mapping[str, Any]) -> list[int]:
    candidates = (
        nested_get(cfg, "autogaze.target_scales"),
        nested_get(cfg, "autogaze.scales"),
        nested_get(cfg, "vision_encoder.from_pretrained_kwargs.scales"),
        nested_get(cfg, "vision_encoder.scales"),
    )
    for value in candidates:
        parsed = parse_scales(value)
        if parsed:
            return parsed
    if bool(nested_get(cfg, "autogaze.enabled", False)):
        return [32, 64, 112, 224]
    return [int(nested_get(cfg, "scaling.resolution", 224))]


def parse_scales(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace(",", "+").split("+")
        return [int(part.strip()) for part in parts if part.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def build_scale_layout(scales: Any, *, patch_size: int, fallback_resolution: int) -> list[dict[str, Any]]:
    parsed = parse_scales(scales) or [int(fallback_resolution)]
    layout: list[dict[str, Any]] = []
    start = 0
    for scale_index, resolution in enumerate(parsed):
        grid_h = max(1, int(resolution) // patch_size)
        grid_w = max(1, int(resolution) // patch_size)
        token_count = grid_h * grid_w
        layout.append(
            {
                "scale": scale_index,
                "scale_resolution": int(resolution),
                "grid_h": grid_h,
                "grid_w": grid_w,
                "start": start,
                "end": start + token_count,
                "token_count": token_count,
            }
        )
        start += token_count
    return layout


def token_record_from_multiscale_index(token_index: int, scale_layout: list[dict[str, Any]]) -> dict[str, Any]:
    for layout in scale_layout:
        if int(layout["start"]) <= token_index < int(layout["end"]):
            local = token_index - int(layout["start"])
            grid_w = int(layout["grid_w"])
            grid_h = int(layout["grid_h"])
            row = local // grid_w
            col = local % grid_w
            return {
                "local_token_index": int(token_index),
                "scale": int(layout["scale"]),
                "scale_resolution": int(layout["scale_resolution"]),
                "scale_patch_index": int(local),
                "scale_grid": [grid_h, grid_w],
                "normalized_box": [
                    col / float(grid_w),
                    row / float(grid_h),
                    (col + 1) / float(grid_w),
                    (row + 1) / float(grid_h),
                ],
            }
    fallback = scale_layout[-1]
    return token_record_from_multiscale_index(int(fallback["end"]) - 1, scale_layout)


def _frame_gaze_record(
    frame_record: Mapping[str, Any],
    local_indices: list[int],
    tokens_per_frame: int,
    processed_frame_index: int,
    scale_layout: list[dict[str, Any]],
) -> dict[str, Any]:
    patch_records = [
        token_record_from_multiscale_index(int(token_index), scale_layout)
        for token_index in local_indices
    ]
    selected_scales = [int(item["scale"]) for item in patch_records]
    return {
        **dict(frame_record),
        "original_token_count": tokens_per_frame,
        "selected_token_count": len(local_indices),
        "token_reduction_ratio": 1.0 - (len(local_indices) / float(tokens_per_frame)) if tokens_per_frame else None,
        "selected_patch_indices": [int(item) for item in local_indices],
        "selected_scales": selected_scales,
        "selected_patch_records": patch_records,
        "global_patch_indices": [processed_frame_index * tokens_per_frame + int(item) for item in local_indices],
        "selected_patch_count_by_scale": {
            str(scale): selected_scales.count(scale) for scale in sorted(SCALE_COLORS)
        },
    }


def write_autogaze_artifacts(output_dir: Path, prepared: PreparedVideo, gaze: GazeResult) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    autogaze_dir = output_dir / "autogaze"
    write_json(autogaze_dir / "frame_selection_metadata.json", prepared.frame_selection.to_dict())
    write_json(autogaze_dir / "runtime_metadata.json", gaze.runtime_metadata)
    write_json(
        autogaze_dir / "token_counts_summary.json",
        {
            "original_token_count": gaze.original_token_count,
            "selected_token_count": gaze.selected_token_count,
            "token_reduction_ratio": gaze.token_reduction_ratio,
            "status": gaze.status,
            "reason": gaze.reason,
        },
    )
    write_json(
        autogaze_dir / "selected_patch_indices.json",
        {
            "frames": [
                {
                    k: item[k]
                    for k in (
                        "processed_frame_index",
                        "source_frame_index",
                        "selected_patch_indices",
                        "selected_patch_records",
                        "global_patch_indices",
                    )
                }
                for item in gaze.per_frame
            ]
        },
    )
    write_json(
        autogaze_dir / "selected_scales.json",
        {"frames": [{k: item[k] for k in ("processed_frame_index", "source_frame_index", "selected_scales")} for item in gaze.per_frame]},
    )
    write_json(
        autogaze_dir / "per_frame_token_counts.json",
        {
            "frames": [
                {
                    "processed_frame_index": item["processed_frame_index"],
                    "source_frame_index": item["source_frame_index"],
                    "original_token_count": item["original_token_count"],
                    "selected_token_count": item["selected_token_count"],
                    "token_reduction_ratio": item["token_reduction_ratio"],
                    "selected_patch_count_by_scale": item["selected_patch_count_by_scale"],
                }
                for item in gaze.per_frame
            ]
        },
    )
    write_json(output_dir / "scaling" / "scaling_metadata.json", prepared.scaling_metadata)
    if prepared.chop_metadata is not None:
        write_json(output_dir / "chops" / "chop_metadata.json", prepared.chop_metadata)
    for name in (
        "frame_selection_metadata",
        "runtime_metadata",
        "token_counts_summary",
        "selected_patch_indices",
        "selected_scales",
        "per_frame_token_counts",
    ):
        artifacts[name] = str(autogaze_dir / f"{name}.json")
    artifacts["scaling_metadata"] = str(output_dir / "scaling" / "scaling_metadata.json")
    if prepared.chop_metadata is not None:
        artifacts["chop_metadata"] = str(output_dir / "chops" / "chop_metadata.json")
    return artifacts


def write_visualizations(
    output_dir: Path,
    prepared: PreparedVideo,
    gaze: GazeResult,
    *,
    overlay_style: str,
    overlay_alpha: float,
    multi_scale_overlay: bool,
    show_patch_index: bool,
    show_scale_label: bool,
    metadata_placement: str,
    info_panel_position: str,
    save_overlay_video: bool,
    save_side_by_side_video: bool,
    save_scale_panel_video: bool,
    video_fps: float,
    video_export_mode: str,
    query_text: str | None = None,
    generation_status: str | None = None,
) -> dict[str, str]:
    base = output_dir / "visualizations" / "autogaze"
    frames_dir = base / "frames"
    panels_dir = base / "scale_panels"
    videos_dir = base / "videos"
    metadata_dir = base / "metadata"
    for directory in (frames_dir, panels_dir, videos_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames = prepared.processed_video[0].detach().cpu()
    overlay_images: list[Image.Image] = []
    side_by_side_images: list[Image.Image] = []
    scale_panel_images: list[Image.Image] = []
    for idx, frame in enumerate(frames):
        record = gaze.per_frame[idx]
        base_image = tensor_to_image(frame)
        overlay = render_overlay(
            base_image,
            record,
            gaze.patch_grid,
            overlay_style=overlay_style,
            overlay_alpha=overlay_alpha,
            multi_scale_overlay=multi_scale_overlay,
            show_patch_index=show_patch_index,
            show_scale_label=show_scale_label,
        )
        overlay_with_panel = add_info_panel(
            overlay,
            record,
            gaze,
            scaling_mode=prepared.scaling_metadata["scaling_mode"],
            metadata_placement=metadata_placement,
            info_panel_position=info_panel_position,
            query_text=query_text,
            generation_status=generation_status,
        )
        overlay_path = frames_dir / f"frame_{idx:06d}_overlay.png"
        overlay_with_panel.save(overlay_path)
        overlay_images.append(overlay_with_panel.convert("RGB"))

        scale_panel = render_scale_panel(base_image, record, gaze.patch_grid)
        scale_panel_path = panels_dir / f"frame_{idx:06d}_scale_panel.png"
        scale_panel.save(scale_panel_path)
        scale_panel_images.append(scale_panel.convert("RGB"))

        side_by_side = combine_side_by_side(base_image, overlay)
        side_by_side_images.append(side_by_side.convert("RGB"))

    artifacts: dict[str, str] = {
        "frames_dir": str(frames_dir),
        "scale_panels_dir": str(panels_dir),
    }
    video_errors: dict[str, str] = {}
    if save_overlay_video:
        path = videos_dir / "autogaze_overlay.mp4"
        error = write_mp4(path, overlay_images, fps=video_fps)
        artifacts["overlay_video"] = str(path) if error is None else f"failed: {error}"
        if error:
            video_errors["overlay_video"] = error
    if save_side_by_side_video:
        path = videos_dir / "autogaze_side_by_side.mp4"
        error = write_mp4(path, side_by_side_images, fps=video_fps)
        artifacts["side_by_side_video"] = str(path) if error is None else f"failed: {error}"
        if error:
            video_errors["side_by_side_video"] = error
    if save_scale_panel_video:
        path = videos_dir / "autogaze_scale_panels.mp4"
        error = write_mp4(path, scale_panel_images, fps=video_fps)
        artifacts["scale_panel_video"] = str(path) if error is None else f"failed: {error}"
        if error:
            video_errors["scale_panel_video"] = error

    write_json(
        metadata_dir / "visualization_metadata.json",
        {
            "status": "ready",
            "flat_output_structure": True,
            "video_export_mode": video_export_mode,
            "frame_count": len(overlay_images),
            "frame_records": prepared.frame_records,
            "scale_colors": {str(key): value for key, value in SCALE_COLORS.items()},
            "scale_layout": gaze.runtime_metadata.get("scale_layout"),
            "video_errors": video_errors,
            "paths": artifacts,
        },
    )
    artifacts["visualization_metadata"] = str(metadata_dir / "visualization_metadata.json")
    return artifacts


def tensor_to_image(frame: torch.Tensor) -> Image.Image:
    array = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def render_overlay(
    image: Image.Image,
    record: Mapping[str, Any],
    patch_grid: tuple[int, int],
    *,
    overlay_style: str,
    overlay_alpha: float,
    multi_scale_overlay: bool,
    show_patch_index: bool,
    show_scale_label: bool,
) -> Image.Image:
    grid_h, grid_w = patch_grid
    width, height = image.size
    patch_w = width / float(grid_w)
    patch_h = height / float(grid_h)
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    indices = list(record.get("selected_patch_indices", []))
    scales = list(record.get("selected_scales", []))
    patch_records = list(record.get("selected_patch_records", []))
    for idx, patch_idx in enumerate(indices):
        patch_record = patch_records[idx] if idx < len(patch_records) and isinstance(patch_records[idx], Mapping) else None
        color = SCALE_COLORS.get(int(scales[idx]) if idx < len(scales) and multi_scale_overlay else 0, SCALE_COLORS[0])
        if patch_record is not None and "normalized_box" in patch_record:
            x0, y0, x1, y1 = [float(value) for value in patch_record["normalized_box"]]
            rect = [x0 * width, y0 * height, x1 * width, y1 * height]
        else:
            row = int(patch_idx) // grid_w
            col = int(patch_idx) % grid_w
            rect = [col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h]
        if overlay_style in {"mask", "both"}:
            draw.rectangle(rect, fill=(*color, int(255 * overlay_alpha)))
        if overlay_style in {"box", "both"}:
            draw.rectangle(rect, outline=(*color, 255), width=2)
        if show_patch_index:
            draw.text((rect[0] + 2, rect[1] + 2), str(patch_idx), fill=(20, 20, 20, 255))
        if show_scale_label:
            label = str(scales[idx]) if idx < len(scales) else "0"
            draw.text((rect[0] + 2, rect[3] - 12), label, fill=(20, 20, 20, 255))
    return Image.alpha_composite(result, overlay).convert("RGB")


def render_scale_panel(image: Image.Image, record: Mapping[str, Any], patch_grid: tuple[int, int]) -> Image.Image:
    panels = []
    selected = list(record.get("selected_patch_indices", []))
    scales = list(record.get("selected_scales", []))
    patch_records = list(record.get("selected_patch_records", []))
    for scale in range(4):
        scale_record = dict(record)
        selected_items = [
            (patch, patch_records[idx] if idx < len(patch_records) else None)
            for idx, (patch, item_scale) in enumerate(zip(selected, scales))
            if item_scale == scale
        ]
        scale_record["selected_patch_indices"] = [patch for patch, _patch_record in selected_items]
        scale_record["selected_scales"] = [scale] * len(scale_record["selected_patch_indices"])
        scale_record["selected_patch_records"] = [
            patch_record if isinstance(patch_record, Mapping) else {}
            for _patch, patch_record in selected_items
        ]
        panels.append(
            render_overlay(
                image,
                scale_record,
                patch_grid,
                overlay_style="both",
                overlay_alpha=0.35,
                multi_scale_overlay=True,
                show_patch_index=False,
                show_scale_label=False,
            )
        )
    width, height = image.size
    canvas = Image.new("RGB", (width * 2, height * 2), (255, 255, 255))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, ((idx % 2) * width, (idx // 2) * height))
    return canvas


def add_info_panel(
    image: Image.Image,
    record: Mapping[str, Any],
    gaze: GazeResult,
    *,
    scaling_mode: str,
    metadata_placement: str,
    info_panel_position: str,
    query_text: str | None,
    generation_status: str | None,
) -> Image.Image:
    if metadata_placement == "none":
        return image
    lines = [
        f"processed_frame_index: {record['processed_frame_index']}",
        f"source_frame_index: {record['source_frame_index']}  window_id: {record['window_id']}",
        f"selected patches: {record['selected_token_count']}/{record['original_token_count']}",
        f"reduction: {record['token_reduction_ratio']:.3f}" if record["token_reduction_ratio"] is not None else "reduction: N/A",
        f"gaze_status: {gaze.status}",
        f"scaling_mode: {scaling_mode}",
    ]
    if query_text is not None:
        lines.append(f"query: {query_text[:80]}")
        lines.append(f"generation: {generation_status or 'not_run'}")
    if metadata_placement == "inside":
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.rectangle([0, 0, image.width, min(image.height, 18 * len(lines) + 8)], fill=(0, 0, 0))
        for idx, line in enumerate(lines):
            draw.text((6, 4 + idx * 16), line, fill=(255, 255, 255))
        return result

    if info_panel_position == "right":
        panel_w = 360
        result = Image.new("RGB", (image.width + panel_w, image.height), (245, 245, 245))
        result.paste(image, (0, 0))
        draw = ImageDraw.Draw(result)
        for idx, line in enumerate(lines):
            draw.text((image.width + 10, 10 + idx * 18), line, fill=(20, 20, 20))
        return result
    panel_h = max(96, 18 * len(lines) + 12)
    result = Image.new("RGB", (image.width, image.height + panel_h), (245, 245, 245))
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    for idx, line in enumerate(lines):
        draw.text((10, image.height + 8 + idx * 18), line, fill=(20, 20, 20))
    return result


def combine_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    right_resized = right.resize(left.size)
    canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right_resized.convert("RGB"), (left.width, 0))
    return canvas


def write_mp4(path: Path, images: list[Image.Image], *, fps: float) -> str | None:
    if not images:
        return "no frames available"
    try:
        import imageio.v3 as iio

        frames = [pad_to_even_image(image).convert("RGB") for image in images]
        arrays = [np.asarray(frame) for frame in frames]
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            iio.imwrite(str(path), arrays, fps=fps, plugin="pyav", codec="h264")
        except TypeError:
            iio.imwrite(str(path), arrays, fps=fps)
        return None
    except Exception as exc:  # pragma: no cover - imageio backend availability varies.
        return str(exc)


def pad_to_even_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_w = width + (width % 2)
    new_h = height + (height % 2)
    if (new_w, new_h) == (width, height):
        return image
    padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded.paste(image.convert("RGB"), (0, 0))
    return padded


def build_metrics(
    *,
    mode: str,
    cfg: Mapping[str, Any],
    video_path: str,
    query_text: str | None,
    prepared: PreparedVideo,
    gaze: GazeResult,
    requested_vision_encoder: str | None,
    actual_vision_encoder: str | None,
    requested_mllm: str | None,
    actual_mllm: str | None,
    generation_status: str | None,
    output_text: str | None,
    skipped_stages: list[dict[str, str]],
    total_latency_ms: float,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "config_path": cfg.get("_config_path"),
        "video_path": video_path,
        "query_text": query_text,
        "frame_selection_mode": prepared.frame_selection.mode,
        "number_of_frames": len(prepared.frame_records),
        "number_of_windows": len(prepared.frame_selection.windows),
        "scaling_mode": prepared.scaling_metadata["scaling_mode"],
        "resolution": prepared.scaling_metadata["resolution"],
        "chop_settings": prepared.chop_metadata,
        "autogaze_enabled": gaze.autogaze_enabled,
        "requested_vision_encoder": requested_vision_encoder,
        "actual_vision_encoder": actual_vision_encoder,
        "requested_mllm": requested_mllm,
        "actual_mllm": actual_mllm,
        "real_stub_blocked_status": gaze.status,
        "generation_status": generation_status,
        "gaze_ratio": gaze.runtime_metadata.get("gaze_ratio"),
        "task_loss_requirement": gaze.runtime_metadata.get("task_loss_requirement"),
        "original_token_count": gaze.original_token_count,
        "selected_token_count": gaze.selected_token_count,
        "token_reduction_ratio": gaze.token_reduction_ratio,
        "selected_patches_per_frame": [item["selected_token_count"] for item in gaze.per_frame],
        "selected_patches_per_scale": _sum_scale_counts(gaze.per_frame),
        "preprocessing_latency_ms": None,
        "autogaze_latency_ms": gaze.latency_ms,
        "vision_encoder_latency_ms": None,
        "mllm_prefill_latency_ms": None,
        "mllm_decode_latency_ms": None,
        "end_to_end_latency_ms": total_latency_ms,
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "memory_unavailable": not torch.cuda.is_available(),
        "output_text": output_text,
        "skipped_stages": skipped_stages,
        "failure_reason": skipped_stages[0]["reason"] if skipped_stages else None,
        "encoder_side_acceleration_claimed": bool(gaze.real_model_used and gaze.autogaze_enabled),
    }


def _sum_scale_counts(per_frame: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {str(scale): 0 for scale in sorted(SCALE_COLORS)}
    for item in per_frame:
        for scale, count in item["selected_patch_count_by_scale"].items():
            counts[str(scale)] = counts.get(str(scale), 0) + int(count)
    return counts


def write_summary_and_metrics(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, str]:
    logs_dir = output_dir / "logs"
    write_json(logs_dir / "poc_summary.json", summary)
    write_json(logs_dir / "metrics.json", metrics)
    write_csv(logs_dir / "metrics.csv", metrics)
    return {
        "poc_summary": str(logs_dir / "poc_summary.json"),
        "metrics_json": str(logs_dir / "metrics.json"),
        "metrics_csv": str(logs_dir / "metrics.csv"),
    }
