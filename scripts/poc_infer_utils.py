#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_DTYPE_DEPRECATION_WARNING = "`torch_dtype` is deprecated! Use `dtype` instead!"


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
        if max_windows is not None and max_windows < 0:
            raise ValueError("max_windows must be >= 0 when provided")
        if drop_last and pad_last:
            raise ValueError("drop_last and pad_last cannot both be true")
        self.mode = mode
        self.num_frames = int(num_frames)
        self.frame_interval = int(frame_interval)
        self.max_windows = _normalize_max_windows(max_windows)
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


class ProgressReporter:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    def warmup(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        runs: int,
        device: str | None = None,
    ) -> None:
        if runs <= 0:
            return
        bar = self._start(f"{label} warm-up", total=runs, unit="run")
        try:
            for _ in range(runs):
                synchronize_device(device)
                fn()
                synchronize_device(device)
                self._update(bar, 1)
        finally:
            self._close(bar)

    def timed(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        device: str | None = None,
    ) -> tuple[Any, float]:
        bar = self._start(label, total=1, unit="run")
        try:
            synchronize_device(device)
            start = time.perf_counter()
            result = fn()
            synchronize_device(device)
            latency_ms = (time.perf_counter() - start) * 1000
            self._update(bar, 1)
            return result, latency_ms
        finally:
            self._close(bar)

    def _start(self, label: str, *, total: int, unit: str) -> Any:
        if not self.enabled:
            return None
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=label, unit=unit, leave=True, dynamic_ncols=True, file=sys.stderr)
        except Exception:
            print(f"{label} ...", file=sys.stderr)
            return None

    def _update(self, bar: Any, amount: int) -> None:
        if bar is not None:
            bar.update(amount)

    def _close(self, bar: Any) -> None:
        if bar is not None:
            bar.close()


def synchronize_device(device: str | None) -> None:
    if device is None:
        return
    device_name = str(device)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_name == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@contextmanager
def suppress_transformers_torch_dtype_warning():
    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return TORCH_DTYPE_DEPRECATION_WARNING not in record.getMessage()

    warning_filter = _Filter()
    loggers = [
        logging.getLogger("transformers"),
        logging.getLogger("transformers.configuration_utils"),
        logging.getLogger("transformers.modeling_utils"),
    ]
    for logger in loggers:
        logger.addFilter(warning_filter)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(warning_filter)


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


def resolve_frame_selection_max_windows(
    *,
    cli_max_windows: Any,
    cfg: Mapping[str, Any],
    frame_selection_mode: str,
    cli_frame_selection_mode: str | None,
) -> int | None:
    if cli_max_windows is not None:
        return _normalize_max_windows(cli_max_windows)
    if cli_frame_selection_mode == "all" and frame_selection_mode == "all":
        return None
    return _normalize_max_windows(nested_get(cfg, "frame_selection.max_windows", None))


def _normalize_max_windows(value: Any) -> int | None:
    if value in {None, "", "none", "None", "null", "Null"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("max_windows must be >= 0 when provided")
    return parsed if parsed > 0 else None


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
    resize_before_chop_threshold: int = 1024,
    resize_before_chop_factor: float = 0.5,
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
        pad_last=scaling_mode in {"chop", "resize_then_chop"},
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
            resize_before_chop_threshold=resize_before_chop_threshold,
            resize_before_chop_factor=resize_before_chop_factor,
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
        if window_chops is not None:
            for chop_record in window_chops["records"]:
                for position, source_index in enumerate(window.frame_indices):
                    frame_records.append(
                        {
                            "processed_frame_index": processed_index,
                            "source_frame_index": int(source_index),
                            "window_id": int(window.window_id),
                            "position_in_window": int(position),
                            "anchor_frame_index": None,
                            "is_padded": bool(window.padded_frame_mask[position]),
                            "chop_index": int(chop_record["chop_index"]),
                            "source_box": [
                                int(chop_record["x0"]),
                                int(chop_record["y0"]),
                                int(chop_record["x1"]),
                                int(chop_record["y1"]),
                            ],
                            "chop_input_box": chop_record.get("chop_input_box"),
                        }
                    )
                    processed_index += 1
        else:
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

    if processed_windows and scaling_mode in {"chop", "resize_then_chop"}:
        processed_video = torch.cat(processed_windows, dim=0)
    else:
        processed_video = torch.cat(processed_windows, dim=1) if processed_windows else source_video[:0].unsqueeze(0)
    scaling_metadata = {
        "scaling_mode": scaling_mode,
        "resolution": resolution,
        "number_of_windows": len(selection.windows),
        "first_processed_shape": [int(dim) for dim in processed_video.shape],
        "temporal_pad_last_applied": bool(scaling_mode in {"chop", "resize_then_chop"} and any(window.is_padded for window in selection.windows)),
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
    resize_before_chop_threshold: int = 1024,
    resize_before_chop_factor: float = 0.5,
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
    elif mode in {"chop", "resize_then_chop"}:
        chop_input = video
        chop_input_h, chop_input_w = height, width
        pre_resize_applied = False
        if mode == "resize_then_chop" and max(height, width) > int(resize_before_chop_threshold):
            if resize_before_chop_factor <= 0:
                raise ValueError("resize_before_chop_factor must be > 0")
            chop_input_h = max(1, round(height * float(resize_before_chop_factor)))
            chop_input_w = max(1, round(width * float(resize_before_chop_factor)))
            chop_input = _resize_video_hw(video, chop_input_h, chop_input_w)
            pre_resize_applied = True
        chop_metadata = generate_chop_metadata(
            height=chop_input_h,
            width=chop_input_w,
            frames=frames,
            chop_size=chop_size,
            chop_overlap=chop_overlap,
            max_chops=max_chops,
            merge_mode=chop_merge_mode,
        )
        _annotate_chop_records_for_source_coordinates(
            chop_metadata,
            source_height=height,
            source_width=width,
            chop_input_height=chop_input_h,
            chop_input_width=chop_input_w,
            pre_resize_applied=pre_resize_applied,
            resize_before_chop_threshold=int(resize_before_chop_threshold),
            resize_before_chop_factor=float(resize_before_chop_factor),
        )
        chunks: list[torch.Tensor] = []
        for record in chop_metadata["records"]:
            input_box = record.get("chop_input_box") or [record["x0"], record["y0"], record["x1"], record["y1"]]
            x0, y0, x1, y1 = [int(value) for value in input_box]
            crop = chop_input[:, :, :, y0:y1, x0:x1]
            chunks.append(_resize_video_hw(crop, resolution, resolution))
        output = torch.cat(chunks, dim=0) if chunks else video[:0]
        if pre_resize_applied:
            notes.append("Frames were resized before spatial chopping, then each crop was resized to the target resolution.")
        else:
            notes.append("Frames were spatially chopped into real model inputs, then each crop was resized to the target resolution.")
    else:
        raise ValueError("scaling_mode must be one of resize, fit_short_side, fit_long_side, chop, resize_then_chop, quickstart, none")

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
            "processed_from_original": "spatial_chop_then_resize" if mode == "chop" else mode,
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
    seen: set[tuple[int, int, int, int]] = set()
    for y in range(0, max(1, height), stride):
        y0 = min(y, max(0, height - chop_size))
        y1 = min(height, y0 + chop_size)
        for x in range(0, max(1, width), stride):
            x0 = min(x, max(0, width - chop_size))
            x1 = min(width, x0 + chop_size)
            box = (x0, y0, x1, y1)
            if box in seen:
                continue
            seen.add(box)
            records.append(
                {
                    "chop_index": len(records),
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )
            if max_chops is not None and len(records) >= max_chops:
                break
        if max_chops is not None and len(records) >= max_chops:
            break
    return {
        "status": "actual_spatial_chops",
        "frame_count": frames,
        "source_resolution": [height, width],
        "chop_size": chop_size,
        "chop_overlap": chop_overlap,
        "stride": stride,
        "max_chops": max_chops,
        "spatial_chop_count": len(records),
        "processed_frame_count": len(records) * frames,
        "merge_mode": merge_mode,
        "records": records,
    }


def _annotate_chop_records_for_source_coordinates(
    chop_metadata: dict[str, Any],
    *,
    source_height: int,
    source_width: int,
    chop_input_height: int,
    chop_input_width: int,
    pre_resize_applied: bool,
    resize_before_chop_threshold: int,
    resize_before_chop_factor: float,
) -> None:
    scale_x = chop_input_width / float(source_width)
    scale_y = chop_input_height / float(source_height)
    chop_metadata["source_resolution"] = [source_height, source_width]
    chop_metadata["chop_input_resolution"] = [chop_input_height, chop_input_width]
    chop_metadata["pre_resize_before_chop"] = {
        "applied": bool(pre_resize_applied),
        "threshold": int(resize_before_chop_threshold),
        "factor": float(resize_before_chop_factor),
        "scale_x": scale_x,
        "scale_y": scale_y,
    }
    for record in chop_metadata["records"]:
        input_box = [int(record["x0"]), int(record["y0"]), int(record["x1"]), int(record["y1"])]
        record["chop_input_box"] = input_box
        if not pre_resize_applied:
            continue
        x0, y0, x1, y1 = input_box
        record["x0"] = max(0, min(source_width, round(x0 / scale_x)))
        record["y0"] = max(0, min(source_height, round(y0 / scale_y)))
        record["x1"] = max(0, min(source_width, round(x1 / scale_x)))
        record["y1"] = max(0, min(source_height, round(y1 / scale_y)))


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
    warmup_runs: int = 0,
    progress: ProgressReporter | None = None,
) -> GazeResult:
    progress = progress or ProgressReporter(enabled=False)
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
            latency_ms=0.0,
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
            latency_ms=0.0,
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
                warmup_runs=warmup_runs,
                progress=progress,
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
            latency_ms=0.0,
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
        latency_ms=0.0,
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
    warmup_runs: int,
    progress: ProgressReporter,
) -> GazeResult:
    from autogaze.models.autogaze import AutoGaze

    checkpoint = nested_get(cfg, "autogaze.checkpoint_path") or nested_get(cfg, "autogaze.model_id")
    with suppress_transformers_torch_dtype_warning():
        model = AutoGaze.from_pretrained(model_reference_for_loading(checkpoint))
    model = model.to(normalize_device(device))
    model.eval()
    video = prepared.processed_video.to(device=normalize_device(device), dtype=dtype_from_name(dtype))
    kwargs: dict[str, Any] = {"gazing_ratio": gaze_ratio}
    if task_loss_requirement is not None:
        kwargs["task_loss_requirement"] = task_loss_requirement

    def forward_once() -> Mapping[str, Any]:
        with torch.inference_mode():
            return model({"video": video}, **kwargs)

    progress.warmup("AutoGaze", forward_once, runs=warmup_runs, device=device)
    outputs, latency_ms = progress.timed("AutoGaze", forward_once, device=device)
    return build_gaze_result(
        prepared,
        autogaze_enabled=True,
        status="real",
        reason=None,
        real_model_used=True,
        gaze_ratio=gaze_ratio,
        task_loss_requirement=task_loss_requirement,
        runtime_metadata={**runtime_metadata, "raw_output_keys": sorted(str(key) for key in outputs.keys())},
        latency_ms=latency_ms,
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
    latency_ms: float,
    real_outputs: Mapping[str, Any] | None = None,
) -> GazeResult:
    video = prepared.processed_video
    batch_size, frame_count, _, height, width = [int(dim) for dim in video.shape]
    patch_size = int(runtime_metadata.get("patch_size") or 16)
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    scale_layout = build_scale_layout(runtime_metadata.get("scales"), patch_size=patch_size, fallback_resolution=max(height, width))
    tokens_per_frame = sum(item["token_count"] for item in scale_layout)
    per_frame: list[dict[str, Any]] = []

    if real_outputs is not None and "gazing_pos" in real_outputs and "num_gazing_each_frame" in real_outputs:
        gazing_pos_all = real_outputs["gazing_pos"].detach().cpu()
        if gazing_pos_all.ndim == 1:
            gazing_pos_all = gazing_pos_all.unsqueeze(0)
        padded = real_outputs.get("if_padded_gazing")
        if isinstance(padded, torch.Tensor):
            padded_all = padded.detach().cpu().bool()
            if padded_all.ndim == 1:
                padded_all = padded_all.unsqueeze(0)
        else:
            padded_all = torch.zeros_like(gazing_pos_all).bool()
        counts_all = real_outputs["num_gazing_each_frame"].detach().cpu()
        tokens_each = int(real_outputs.get("num_vision_tokens_each_frame", tokens_per_frame))
        for batch_idx in range(batch_size):
            gazing_pos = gazing_pos_all[min(batch_idx, gazing_pos_all.shape[0] - 1)]
            padded_flat = padded_all[min(batch_idx, padded_all.shape[0] - 1)]
            counts = _counts_for_batch(counts_all, batch_idx=batch_idx, frame_count=frame_count)
            offset = 0
            for idx in range(frame_count):
                record_index = batch_idx * frame_count + idx
                if record_index >= len(prepared.frame_records):
                    break
                count = int(counts[idx]) if idx < len(counts) else 0
                pos = gazing_pos[offset : offset + count]
                pad = padded_flat[offset : offset + count]
                local = [int((item - idx * tokens_each).item()) % tokens_each for item in pos[~pad]]
                per_frame.append(_frame_gaze_record(prepared.frame_records[record_index], local, tokens_each, record_index, scale_layout))
                offset += count
    else:
        selected_count = tokens_per_frame if not autogaze_enabled else max(1, min(tokens_per_frame, math.ceil(tokens_per_frame * gaze_ratio)))
        for idx in range(len(prepared.frame_records)):
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
        latency_ms=latency_ms,
        gazing_info_for_vit=gazing_info_for_vit,
    )


def _counts_for_batch(counts: torch.Tensor, *, batch_idx: int, frame_count: int) -> list[int]:
    if counts.ndim == 0:
        return [int(counts.item())] * frame_count
    if counts.ndim == 1:
        return [int(item) for item in counts.tolist()]
    selected = counts[min(batch_idx, counts.shape[0] - 1)]
    return [int(item) for item in selected.reshape(-1).tolist()]


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
                        "window_id",
                        "position_in_window",
                        "chop_index",
                        "source_box",
                        "selected_patch_indices",
                        "selected_patch_records",
                        "global_patch_indices",
                    )
                    if k in item
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
    save_frame_images: bool,
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
    directories = [metadata_dir]
    if save_frame_images:
        directories.extend([frames_dir, panels_dir])
    if save_overlay_video or save_side_by_side_video or save_scale_panel_video:
        directories.append(videos_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    render_needed = save_frame_images or save_overlay_video or save_side_by_side_video or save_scale_panel_video
    if prepared.chop_metadata is not None:
        visualization_mode = "merged_chop_source_frames"
        if render_needed:
            overlay_images, side_by_side_images, scale_panel_images, visualization_records = _write_chop_merged_visualization_frames(
                frames_dir,
                panels_dir,
                prepared,
                gaze,
                overlay_style=overlay_style,
                overlay_alpha=overlay_alpha,
                multi_scale_overlay=multi_scale_overlay,
                show_patch_index=show_patch_index,
                show_scale_label=show_scale_label,
                metadata_placement=metadata_placement,
                info_panel_position=info_panel_position,
                query_text=query_text,
                generation_status=generation_status,
                save_frame_images=save_frame_images,
            )
        else:
            grouped_records = _group_chop_records(gaze.per_frame)
            visualization_records = [_merged_chop_record(records, idx) for idx, records in enumerate(grouped_records)]
            overlay_images = []
            side_by_side_images = []
            scale_panel_images = []
    else:
        visualization_mode = "processed_frames"
        visualization_records = prepared.frame_records
        overlay_images = []
        side_by_side_images = []
        scale_panel_images = []
        if render_needed:
            frames = flatten_video_frames(prepared.processed_video).detach().cpu()
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
                if save_frame_images:
                    overlay_path = frames_dir / f"frame_{idx:06d}_overlay.png"
                    overlay_with_panel.save(overlay_path)
                overlay_images.append(overlay_with_panel.convert("RGB"))

                scale_panel = render_scale_panel(base_image, record, gaze.patch_grid)
                if save_frame_images:
                    scale_panel_path = panels_dir / f"frame_{idx:06d}_scale_panel.png"
                    scale_panel.save(scale_panel_path)
                scale_panel_images.append(scale_panel.convert("RGB"))

                side_by_side = combine_side_by_side(base_image, overlay)
                side_by_side_images.append(side_by_side.convert("RGB"))

    artifacts: dict[str, str] = {}
    if save_frame_images:
        artifacts["frames_dir"] = str(frames_dir)
        artifacts["scale_panels_dir"] = str(panels_dir)
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
            "visualization_mode": visualization_mode,
            "video_export_mode": video_export_mode,
            "frame_images_saved": bool(save_frame_images),
            "rendered_frame_count": len(overlay_images),
            "frame_count": len(visualization_records),
            "processed_crop_frame_count": len(gaze.per_frame) if prepared.chop_metadata is not None else None,
            "frame_records": visualization_records,
            "processed_frame_records": prepared.frame_records,
            "scale_colors": {str(key): value for key, value in SCALE_COLORS.items()},
            "scale_layout": gaze.runtime_metadata.get("scale_layout"),
            "video_errors": video_errors,
            "paths": artifacts,
        },
    )
    artifacts["visualization_metadata"] = str(metadata_dir / "visualization_metadata.json")
    return artifacts


def _write_chop_merged_visualization_frames(
    frames_dir: Path,
    panels_dir: Path,
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
    query_text: str | None,
    generation_status: str | None,
    save_frame_images: bool,
) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[dict[str, Any]]]:
    overlay_images: list[Image.Image] = []
    side_by_side_images: list[Image.Image] = []
    scale_panel_images: list[Image.Image] = []
    visualization_records: list[dict[str, Any]] = []
    source_video = prepared.source_video.detach().cpu()
    grouped_records = _group_chop_records(gaze.per_frame)
    for idx, records in enumerate(grouped_records):
        source_index = max(0, min(int(records[0]["source_frame_index"]), int(source_video.shape[0]) - 1))
        base_image = tensor_to_image(source_video[source_index])
        merged_record = _merged_chop_record(records, idx)
        visualization_records.append(merged_record)
        overlay = render_chop_merged_overlay(
            base_image,
            records,
            overlay_style=overlay_style,
            overlay_alpha=overlay_alpha,
            multi_scale_overlay=multi_scale_overlay,
            show_patch_index=show_patch_index,
            show_scale_label=show_scale_label,
        )
        overlay_with_panel = add_info_panel(
            overlay,
            merged_record,
            gaze,
            scaling_mode=prepared.scaling_metadata["scaling_mode"],
            metadata_placement=metadata_placement,
            info_panel_position=info_panel_position,
            query_text=query_text,
            generation_status=generation_status,
        )
        if save_frame_images:
            overlay_path = frames_dir / f"frame_{idx:06d}_overlay.png"
            overlay_with_panel.save(overlay_path)
        overlay_images.append(overlay_with_panel.convert("RGB"))

        scale_panel = render_chop_merged_scale_panel(base_image, records)
        if save_frame_images:
            scale_panel_path = panels_dir / f"frame_{idx:06d}_scale_panel.png"
            scale_panel.save(scale_panel_path)
        scale_panel_images.append(scale_panel.convert("RGB"))

        side_by_side = combine_side_by_side(base_image, overlay)
        side_by_side_images.append(side_by_side.convert("RGB"))
    return overlay_images, side_by_side_images, scale_panel_images, visualization_records


def _group_chop_records(records: list[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            int(record.get("window_id", 0)),
            int(record.get("position_in_window", 0)),
            int(record.get("source_frame_index", 0)),
        )
        grouped.setdefault(key, []).append(record)
    return list(grouped.values())


def _merged_chop_record(records: list[Mapping[str, Any]], processed_frame_index: int) -> dict[str, Any]:
    first = records[0]
    original = sum(int(record.get("original_token_count", 0)) for record in records)
    selected = sum(int(record.get("selected_token_count", 0)) for record in records)
    scale_counts = {str(scale): 0 for scale in sorted(SCALE_COLORS)}
    source_boxes = []
    for record in records:
        source_box = record.get("source_box")
        if source_box is not None:
            source_boxes.append(source_box)
        for scale, count in dict(record.get("selected_patch_count_by_scale", {})).items():
            scale_counts[str(scale)] = scale_counts.get(str(scale), 0) + int(count)
    return {
        "processed_frame_index": processed_frame_index,
        "source_frame_index": int(first.get("source_frame_index", 0)),
        "window_id": int(first.get("window_id", 0)),
        "position_in_window": int(first.get("position_in_window", 0)),
        "anchor_frame_index": first.get("anchor_frame_index"),
        "is_padded": bool(first.get("is_padded", False)),
        "chop_count": len(records),
        "source_boxes": source_boxes,
        "original_token_count": original,
        "selected_token_count": selected,
        "token_reduction_ratio": 1.0 - (selected / float(original)) if original else None,
        "selected_patch_count_by_scale": scale_counts,
    }


def flatten_video_frames(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
    batch, frames, channels, height, width = [int(dim) for dim in video.shape]
    return video.reshape(batch * frames, channels, height, width)


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


def render_chop_merged_overlay(
    image: Image.Image,
    records: list[Mapping[str, Any]],
    *,
    overlay_style: str,
    overlay_alpha: float,
    multi_scale_overlay: bool,
    show_patch_index: bool,
    show_scale_label: bool,
    scale_filter: int | None = None,
) -> Image.Image:
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for record in records:
        source_box = record.get("source_box")
        if not isinstance(source_box, list) or len(source_box) != 4:
            continue
        box_x0, box_y0, box_x1, box_y1 = [float(value) for value in source_box]
        box_w = max(1.0, box_x1 - box_x0)
        box_h = max(1.0, box_y1 - box_y0)
        indices = list(record.get("selected_patch_indices", []))
        scales = list(record.get("selected_scales", []))
        patch_records = list(record.get("selected_patch_records", []))
        for idx, patch_idx in enumerate(indices):
            scale = int(scales[idx]) if idx < len(scales) else 0
            if scale_filter is not None and scale != scale_filter:
                continue
            patch_record = patch_records[idx] if idx < len(patch_records) and isinstance(patch_records[idx], Mapping) else None
            if patch_record is not None and "normalized_box" in patch_record:
                nx0, ny0, nx1, ny1 = [float(value) for value in patch_record["normalized_box"]]
            else:
                nx0, ny0, nx1, ny1 = 0.0, 0.0, 1.0, 1.0
            rect = [
                box_x0 + nx0 * box_w,
                box_y0 + ny0 * box_h,
                box_x0 + nx1 * box_w,
                box_y0 + ny1 * box_h,
            ]
            color = SCALE_COLORS.get(scale if multi_scale_overlay else 0, SCALE_COLORS[0])
            if overlay_style in {"mask", "both"}:
                draw.rectangle(rect, fill=(*color, int(255 * overlay_alpha)))
            if overlay_style in {"box", "both"}:
                draw.rectangle(rect, outline=(*color, 255), width=2)
            if show_patch_index:
                draw.text((rect[0] + 2, rect[1] + 2), str(patch_idx), fill=(20, 20, 20, 255))
            if show_scale_label:
                draw.text((rect[0] + 2, rect[3] - 12), str(scale), fill=(20, 20, 20, 255))
    return Image.alpha_composite(result, overlay).convert("RGB")


def render_chop_merged_scale_panel(image: Image.Image, records: list[Mapping[str, Any]]) -> Image.Image:
    panels = [
        render_chop_merged_overlay(
            image,
            records,
            overlay_style="both",
            overlay_alpha=0.35,
            multi_scale_overlay=True,
            show_patch_index=False,
            show_scale_label=False,
            scale_filter=scale,
        )
        for scale in range(4)
    ]
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
    if "chop_index" in record:
        lines.append(f"chop_index: {record['chop_index']}  source_box: {record.get('source_box')}")
    if "chop_count" in record:
        lines.append(f"merged_chops: {record['chop_count']}")
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
    preprocessing_latency_ms: float | None = None,
    vision_encoder_latency_ms: float | None = None,
    mllm_generation_latency_ms: float | None = None,
    visualization_latency_ms: float | None = None,
    wall_clock_latency_ms: float | None = None,
    warmup_runs: int = 0,
) -> dict[str, Any]:
    module_processing_latency_ms = total_latency_ms
    return {
        "mode": mode,
        "config_path": cfg.get("_config_path"),
        "video_path": video_path,
        "query_text": query_text,
        "frame_selection_mode": prepared.frame_selection.mode,
        "number_of_frames": len(prepared.frame_records),
        "number_of_processed_frames": len(prepared.frame_records),
        "number_of_source_frames": _source_frame_count(prepared.frame_records),
        "number_of_windows": len(prepared.frame_selection.windows),
        "scaling_mode": prepared.scaling_metadata["scaling_mode"],
        "resolution": prepared.scaling_metadata["resolution"],
        "chop_settings": prepared.chop_metadata,
        "spatial_chops_per_window": _spatial_chops_per_window(prepared.chop_metadata),
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
        "full_processed_visual_token_count": gaze.original_token_count,
        "autogaze_selected_visual_token_count": gaze.selected_token_count,
        "estimated_visual_token_savings_ratio": gaze.token_reduction_ratio,
        "selected_patches_per_frame": [item["selected_token_count"] for item in gaze.per_frame],
        "selected_patches_per_scale": _sum_scale_counts(gaze.per_frame),
        "preprocessing_latency_ms": preprocessing_latency_ms,
        "autogaze_latency_ms": gaze.latency_ms,
        "vision_encoder_latency_ms": vision_encoder_latency_ms,
        "mllm_prefill_latency_ms": None,
        "mllm_decode_latency_ms": mllm_generation_latency_ms,
        "mllm_generation_latency_ms": mllm_generation_latency_ms,
        "module_processing_latency_ms": module_processing_latency_ms,
        "visualization_latency_ms": visualization_latency_ms,
        "wall_clock_latency_ms": wall_clock_latency_ms,
        "warmup_runs": warmup_runs,
        "end_to_end_latency_ms": module_processing_latency_ms,
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "memory_unavailable": not torch.cuda.is_available(),
        "output_text": output_text,
        "skipped_stages": skipped_stages,
        "failure_reason": skipped_stages[0]["reason"] if skipped_stages else None,
        "encoder_side_acceleration_claimed": bool(gaze.real_model_used and gaze.autogaze_enabled),
    }


def _source_frame_count(frame_records: list[Mapping[str, Any]]) -> int:
    return len({int(item.get("source_frame_index", idx)) for idx, item in enumerate(frame_records) if not bool(item.get("is_padded", False))})


def _spatial_chops_per_window(chop_metadata: Mapping[str, Any] | None) -> dict[str, int] | None:
    if chop_metadata is None:
        return None
    result: dict[str, int] = {}
    for window in chop_metadata.get("windows", []):
        result[str(window.get("window_id", len(result)))] = int(window.get("spatial_chop_count", len(window.get("records", []))))
    return result


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
