#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import math
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_DTYPE_DEPRECATION_WARNING = "`torch_dtype` is deprecated! Use `dtype` instead!"

NVILA_HD_TILE_GAZE_DEFAULT = [0.2] + [0.06] * 15
NVILA_HD_PROCESSOR_KEYS = (
    "num_video_frames",
    "num_video_frames_thumbnail",
    "max_tiles_video",
    "gazing_ratio_tile",
    "task_loss_requirement_tile",
    "gazing_ratio_thumbnail",
    "task_loss_requirement_thumbnail",
    "max_batch_size_autogaze",
)
NVILA_HD_MODEL_KEYS = ("max_batch_size_siglip",)


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
        selected = list(range(0, total, self.frame_interval))
        windows: list[FrameWindow] = []
        window_id = 0
        for start in range(0, len(selected), self.num_frames):
            indices = selected[start : start + self.num_frames]
            maybe_window = self._short_window(window_id, indices, total)
            if maybe_window:
                windows.extend(maybe_window)
                window_id += 1
        return windows

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
class StreamWindowInput:
    window_id: int
    frames: torch.Tensor
    frame_indices: list[int]
    padded_frame_mask: list[bool]
    original_frame_count: int | None
    original_fps: float | None
    decoded_frame_count: int
    video_source_kind: str
    decode_backend: str
    frame_count_from_metadata: bool

    @property
    def effective_num_frames(self) -> int:
        return len([item for item in self.padded_frame_mask if not item])


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


def tensor_nbytes(value: Any) -> int | None:
    if not isinstance(value, torch.Tensor):
        return None
    return int(value.numel() * value.element_size())


def bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1024 * 1024)


def tensor_memory_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        return {
            "shape": None,
            "dtype": None,
            "numel": None,
            "bytes": None,
            "mib": None,
        }
    nbytes = tensor_nbytes(value)
    return {
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype).replace("torch.", ""),
        "numel": int(value.numel()),
        "bytes": nbytes,
        "mib": bytes_to_mib(nbytes),
    }


def capture_memory_snapshot(label: str, *, device: str | None = None) -> dict[str, Any]:
    process_peak_rss_bytes = _process_peak_rss_bytes()
    snapshot: dict[str, Any] = {
        "label": str(label),
        "timestamp": time.time(),
        "process_peak_rss_bytes": process_peak_rss_bytes,
        "process_peak_rss_mib": bytes_to_mib(process_peak_rss_bytes),
        "process_rss_current_bytes": None,
        "process_rss_current_mib": None,
        "process_rss_current_unavailable": True,
    }
    device_name = str(device or "")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        max_allocated = int(torch.cuda.max_memory_allocated())
        max_reserved = int(torch.cuda.max_memory_reserved())
        snapshot.update(
            {
                "cuda_memory_allocated_bytes": allocated,
                "cuda_memory_allocated_mib": bytes_to_mib(allocated),
                "cuda_memory_reserved_bytes": reserved,
                "cuda_memory_reserved_mib": bytes_to_mib(reserved),
                "cuda_max_memory_allocated_bytes": max_allocated,
                "cuda_max_memory_allocated_mib": bytes_to_mib(max_allocated),
                "cuda_max_memory_reserved_bytes": max_reserved,
                "cuda_max_memory_reserved_mib": bytes_to_mib(max_reserved),
            }
        )
    else:
        snapshot.update(
            {
                "cuda_memory_allocated_bytes": None,
                "cuda_memory_allocated_mib": None,
                "cuda_memory_reserved_bytes": None,
                "cuda_memory_reserved_mib": None,
                "cuda_max_memory_allocated_bytes": None,
                "cuda_max_memory_allocated_mib": None,
                "cuda_max_memory_reserved_bytes": None,
                "cuda_max_memory_reserved_mib": None,
            }
        )
    return snapshot


def prepared_video_memory_metrics(prepared: "PreparedVideo") -> dict[str, Any]:
    source_summary = tensor_memory_summary(prepared.source_video)
    processed_summary = tensor_memory_summary(prepared.processed_video)
    source_frame_count = _source_frame_count(prepared.frame_records)
    processed_frame_count = len(prepared.frame_records)
    source_bytes = source_summary["bytes"]
    processed_bytes = processed_summary["bytes"]
    return {
        "source_video_tensor_shape": source_summary["shape"],
        "source_video_tensor_dtype": source_summary["dtype"],
        "source_video_tensor_numel": source_summary["numel"],
        "source_video_tensor_bytes": source_bytes,
        "source_video_tensor_mib": source_summary["mib"],
        "processed_video_tensor_shape": processed_summary["shape"],
        "processed_video_tensor_dtype": processed_summary["dtype"],
        "processed_video_tensor_numel": processed_summary["numel"],
        "processed_video_tensor_bytes": processed_bytes,
        "processed_video_tensor_mib": processed_summary["mib"],
        "processed_video_clip_count": int(prepared.processed_video.shape[0]) if prepared.processed_video.ndim == 5 else None,
        "processed_video_frames_per_clip": int(prepared.processed_video.shape[1]) if prepared.processed_video.ndim == 5 else None,
        "processed_frame_count": processed_frame_count,
        "source_frame_count": source_frame_count,
        "processed_to_source_frame_expansion_ratio": _safe_ratio(processed_frame_count, source_frame_count),
        "processed_to_source_tensor_byte_ratio": _safe_ratio(processed_bytes, source_bytes),
        "spatial_chop_count_total": _total_spatial_chops(prepared.chop_metadata),
        "scaling_window_count": len(prepared.scaling_metadata.get("windows", [])),
    }


def _process_peak_rss_bytes() -> int | None:
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if raw <= 0:
        return None
    if sys.platform == "darwin":
        return raw
    return raw * 1024


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _total_spatial_chops(chop_metadata: Mapping[str, Any] | None) -> int | None:
    if chop_metadata is None:
        return None
    return sum(int(window.get("spatial_chop_count", len(window.get("records", [])))) for window in chop_metadata.get("windows", []))


def attach_memory_snapshots(metrics: dict[str, Any], snapshots: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics["memory_snapshots"] = list(snapshots)
    if snapshots:
        last = dict(snapshots[-1])
        metrics["process_peak_rss_bytes"] = last.get("process_peak_rss_bytes")
        metrics["process_peak_rss_mib"] = last.get("process_peak_rss_mib")
        metrics["cuda_max_memory_allocated_bytes"] = _max_optional_count(
            [item.get("cuda_max_memory_allocated_bytes") for item in snapshots]
        )
        metrics["cuda_max_memory_allocated_mib"] = bytes_to_mib(metrics["cuda_max_memory_allocated_bytes"])
        metrics["cuda_max_memory_reserved_bytes"] = _max_optional_count(
            [item.get("cuda_max_memory_reserved_bytes") for item in snapshots]
        )
        metrics["cuda_max_memory_reserved_mib"] = bytes_to_mib(metrics["cuda_max_memory_reserved_bytes"])
    return metrics


def format_concise_summary(summary: Mapping[str, Any]) -> str:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    if not isinstance(metrics, Mapping):
        metrics = {}
    lines = [
        (
            f"Status: {summary.get('status')} | mode={summary.get('mode')} | "
            f"output={summary.get('output_dir')}"
        ),
        (
            "Frames: "
            f"source={_fmt_value(metrics.get('number_of_source_frames'))} "
            f"processed={_fmt_value(metrics.get('number_of_processed_frames'))} "
            f"windows={_fmt_value(metrics.get('number_of_windows'))} "
            f"chops={_fmt_value(metrics.get('spatial_chop_count_total') or _sum_chops_from_metrics(metrics))} "
            f"scaling={_fmt_value(metrics.get('scaling_mode'))}"
        ),
        (
            "Tokens: "
            f"original={_fmt_value(metrics.get('original_token_count'))} "
            f"selected={_fmt_value(metrics.get('selected_token_count'))} "
            f"reduction={_fmt_percent(metrics.get('token_reduction_ratio'))}"
        ),
        (
            "Latency ms: "
            f"pre={_fmt_ms(metrics.get('autogaze_preprocessing_latency_ms'))} "
            f"ag_forward={_fmt_forward_latency(metrics)} "
            f"ag_build={_fmt_ms(metrics.get('autogaze_result_build_latency_ms'))} "
            f"autogaze={_fmt_ms(metrics.get('autogaze_latency_ms'))} "
            f"vision={_fmt_ms(metrics.get('vision_encoder_latency_ms'))} "
            f"mllm_prep={_fmt_ms(metrics.get('mllm_processor_latency_ms'))} "
            f"mllm_model={_fmt_ms(metrics.get('mllm_model_generate_latency_ms'))} "
            f"mllm={_fmt_ms(metrics.get('mllm_generation_latency_ms'))} "
            f"module={_fmt_ms(metrics.get('module_processing_latency_ms'))} "
            f"viz={_fmt_ms(metrics.get('visualization_latency_ms'))} "
            f"wall={_fmt_ms(metrics.get('wall_clock_latency_ms'))}"
        ),
        (
            "Per-item ms: "
            f"ag/source_frame={_fmt_ms(metrics.get('autogaze_latency_per_source_frame_ms'))} "
            f"ag/processed_frame={_fmt_ms(metrics.get('autogaze_latency_per_processed_frame_ms'))} "
            f"mllm/processed_frame={_fmt_ms_per_item(metrics.get('mllm_generation_latency_ms'), metrics.get('number_of_processed_frames'))} "
            f"mllm/original_token={_fmt_ms_per_item(metrics.get('mllm_generation_latency_ms'), metrics.get('original_token_count'))} "
            f"mllm/selected_token={_fmt_ms_per_item(metrics.get('mllm_generation_latency_ms'), metrics.get('selected_token_count'))}"
        ),
        (
            "Memory: "
            f"source_tensor={_fmt_mib(metrics.get('source_video_tensor_mib'))} "
            f"processed_tensor={_fmt_mib(metrics.get('processed_video_tensor_mib'))} "
            f"mllm_input={_fmt_mib(metrics.get('mllm_input_tensor_mib'))} "
            f"peak_rss={_fmt_mib(metrics.get('process_peak_rss_mib'))} "
            f"cuda_peak={_fmt_mib(metrics.get('cuda_max_memory_allocated_mib'))}"
        ),
    ]
    return "\n".join(lines)


def _sum_chops_from_metrics(metrics: Mapping[str, Any]) -> int | None:
    chops = metrics.get("spatial_chops_per_window")
    if isinstance(chops, Mapping):
        values = [int(value) for value in chops.values()]
        return sum(values)
    return None


def _fmt_value(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _fmt_forward_latency(metrics: Mapping[str, Any]) -> str:
    value = metrics.get("autogaze_model_forward_latency_ms")
    if value is not None:
        return _fmt_ms(value)
    if metrics.get("autogaze_model_forward_status") == "not_run":
        return "0.00(not_run)"
    return "n/a"


def _fmt_mib(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}MiB"


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_ms_per_item(latency_ms: Any, count: Any) -> str:
    if latency_ms is None or count is None:
        return "n/a"
    try:
        parsed_count = float(count)
    except (TypeError, ValueError):
        return "n/a"
    if parsed_count <= 0:
        return "n/a"
    return _fmt_ms(float(latency_ms) / parsed_count)


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


def add_nvila_hd_cli_args(parser: Any) -> None:
    parser.add_argument("--num-video-frames", type=int, default=None)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=None)
    parser.add_argument("--max-tiles-video", type=int, default=None)
    parser.add_argument("--gazing-ratio-tile", default=None)
    parser.add_argument("--task-loss-requirement-tile", default=None)
    parser.add_argument("--gazing-ratio-thumbnail", default=None)
    parser.add_argument("--task-loss-requirement-thumbnail", default=None)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=None)
    parser.add_argument("--max-batch-size-siglip", type=int, default=None)


def apply_nvila_hd_overrides(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    mllm = cfg.setdefault("mllm", {})
    nvila_hd = cfg.setdefault("nvila_hd", {})
    processor_kwargs = mllm.setdefault("processor_from_pretrained_kwargs", {})
    model_kwargs = mllm.setdefault("from_pretrained_kwargs", {})
    processor_cli_raw = {
        "num_video_frames": getattr(args, "num_video_frames", None),
        "num_video_frames_thumbnail": getattr(args, "num_video_frames_thumbnail", None),
        "max_tiles_video": getattr(args, "max_tiles_video", None),
        "gazing_ratio_tile": getattr(args, "gazing_ratio_tile", None),
        "task_loss_requirement_tile": getattr(args, "task_loss_requirement_tile", None),
        "gazing_ratio_thumbnail": getattr(args, "gazing_ratio_thumbnail", None),
        "task_loss_requirement_thumbnail": getattr(args, "task_loss_requirement_thumbnail", None),
        "max_batch_size_autogaze": getattr(args, "max_batch_size_autogaze", None),
    }
    model_cli = {"max_batch_size_siglip": getattr(args, "max_batch_size_siglip", None)}
    for key, raw in processor_cli_raw.items():
        if raw is not None:
            value = parse_optional_number_or_list(raw)
            processor_kwargs[key] = value
            nvila_hd[key] = value
    for key, value in model_cli.items():
        if value is not None:
            model_kwargs[key] = value
            nvila_hd[key] = value
    if getattr(args, "num_video_frames", None) is not None:
        frame_selection = cfg.setdefault("frame_selection", {})
        streaming = cfg.setdefault("streaming", {})
        frame_selection["num_frames"] = int(args.num_video_frames)
        streaming["window_size"] = int(args.num_video_frames)
    cfg["nvila_hd_effective_settings"] = nvila_hd_effective_settings(cfg)
    return cfg


def nvila_hd_effective_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    processor_kwargs = nested_get(cfg, "mllm.processor_from_pretrained_kwargs", {}) or {}
    model_kwargs = nested_get(cfg, "mllm.from_pretrained_kwargs", {}) or {}
    nvila_hd = nested_get(cfg, "nvila_hd", {}) or {}
    result: dict[str, Any] = {}
    defaults = {
        "num_video_frames": None,
        "num_video_frames_thumbnail": None,
        "max_tiles_video": None,
        "gazing_ratio_tile": None,
        "task_loss_requirement_tile": None,
        "gazing_ratio_thumbnail": None,
        "task_loss_requirement_thumbnail": None,
        "max_batch_size_autogaze": None,
        "max_batch_size_siglip": None,
    }
    for key, default in defaults.items():
        if key in processor_kwargs:
            result[key] = processor_kwargs[key]
        elif key in model_kwargs:
            result[key] = model_kwargs[key]
        elif key in nvila_hd:
            result[key] = nvila_hd[key]
        else:
            result[key] = default
    result["video_read_mode"] = nested_get(cfg, "video_input.read_mode", "full")
    result["decode_backend"] = nested_get(cfg, "video_input.decode_backend", "auto")
    result["max_decode_frames"] = nested_get(cfg, "video_input.max_decode_frames", None)
    result["fail_on_full_video_load"] = nested_get(cfg, "memory.fail_on_full_video_load", True)
    result["cpu_offload_between_windows"] = nested_get(cfg, "streaming.cpu_offload_between_windows", True)
    result["empty_cache_between_windows"] = nested_get(cfg, "streaming.empty_cache_between_windows", False)
    result["autogaze_enabled"] = bool(nested_get(cfg, "autogaze.enabled", False))
    result["official_processor_path"] = bool(nested_get(cfg, "mllm.official_processor_path", False))
    result["video_input_source"] = nested_get(cfg, "mllm.video_input_source", None)
    return json_safe(result)


def nvila_hd_gaze_ratio(cfg: Mapping[str, Any], cli_gaze_ratio: Any = None) -> Any:
    if cli_gaze_ratio is not None:
        return cli_gaze_ratio
    value = nested_get(cfg, "mllm.processor_from_pretrained_kwargs.gazing_ratio_tile", None)
    if value is not None:
        return value
    value = nested_get(cfg, "nvila_hd.gazing_ratio_tile", None)
    if value is not None:
        return value
    return nested_get(cfg, "autogaze.gaze_ratio", 0.75)


def nvila_hd_task_loss_requirement(cfg: Mapping[str, Any], cli_task_loss_requirement: Any = None) -> Any:
    if cli_task_loss_requirement is not None:
        return cli_task_loss_requirement
    value = nested_get(cfg, "mllm.processor_from_pretrained_kwargs.task_loss_requirement_tile", None)
    if value is not None:
        return value
    value = nested_get(cfg, "nvila_hd.task_loss_requirement_tile", None)
    if value is not None:
        return value
    return nested_get(cfg, "autogaze.task_loss_requirement", 0.7)


def parse_optional_number_or_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, list, tuple)):
        return list(value) if isinstance(value, tuple) else value
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return None
    if text.startswith("["):
        return json.loads(text)
    if "," in text:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    return float(text)


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
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
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


def load_video_frames(
    video_path: str,
    *,
    dummy_frames: int,
    dummy_resolution: int,
    max_decode_frames: int | None = None,
) -> tuple[torch.Tensor, str, float | None]:
    if max_decode_frames is not None and int(max_decode_frames) <= 0:
        raise ValueError("max_decode_frames must be > 0 when provided")
    if video_path == "dummy":
        frame_count = min(int(dummy_frames), int(max_decode_frames)) if max_decode_frames is not None else int(dummy_frames)
        return make_dummy_video(frame_count, dummy_resolution, dummy_resolution), "dummy", None
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
            if max_decode_frames is not None and len(frames) >= int(max_decode_frames):
                break
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return torch.stack(frames, dim=0), "file", fps


def iter_stream_windows(
    video_path: str,
    *,
    frame_selection_mode: str,
    num_frames: int,
    frame_interval: int,
    max_windows: int | None,
    stream_window_size: int | None,
    stream_overlap: int,
    max_decode_frames: int | None,
    decode_backend: str,
    decode_fps: float | None,
    dummy_frames: int,
    dummy_resolution: int,
) -> Iterator[StreamWindowInput]:
    window_size = int(stream_window_size or num_frames)
    if window_size <= 0:
        raise ValueError("stream_window_size must be > 0")
    if stream_overlap < 0 or stream_overlap >= window_size:
        raise ValueError("stream_overlap must be >= 0 and < stream_window_size")
    if frame_interval <= 0:
        raise ValueError("frame_interval must be > 0")
    max_window_count = _normalize_max_windows(max_windows)
    mode = str(frame_selection_mode)
    if mode not in FrameSelector.SUPPORTED_MODES:
        raise ValueError(f"Unsupported frame_selection_mode: {mode}")

    if video_path == "dummy":
        frames = make_dummy_video(dummy_frames, dummy_resolution, dummy_resolution)
        metadata = {
            "original_frame_count": int(frames.shape[0]),
            "original_fps": None,
            "frame_count_from_metadata": True,
            "decode_backend": "dummy",
            "video_source_kind": "dummy",
        }
        frame_iter = ((idx, frames[idx]) for idx in range(int(frames.shape[0])))
    else:
        path = resolve_path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video file does not exist: {path}")
        metadata = _stream_video_metadata(path, decode_backend=decode_backend)
        frame_iter = _iter_decoded_file_frames(
            path,
            decode_backend=decode_backend,
            decode_fps=decode_fps,
            original_fps=metadata["original_fps"],
            max_decode_frames=max_decode_frames,
        )

    if mode == "sample":
        yield from _iter_sample_stream_window(
            frame_iter,
            window_size=window_size,
            max_windows=max_window_count,
            metadata=metadata,
        )
        return

    yield from _iter_chunk_like_stream_windows(
        frame_iter,
        frame_selection_mode=mode,
        window_size=window_size,
        stream_overlap=stream_overlap,
        frame_interval=frame_interval,
        max_windows=max_window_count,
        metadata=metadata,
    )


def _stream_video_metadata(path: Path, *, decode_backend: str) -> dict[str, Any]:
    backend = str(decode_backend)
    if backend in {"decord", "torchvision"}:
        raise NotImplementedError(f"streaming decode backend '{backend}' is not implemented; use auto or opencv")
    if backend == "opencv":
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --decode-backend opencv") from exc
        capture = cv2.VideoCapture(str(path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
        finally:
            capture.release()
        return {
            "original_frame_count": total,
            "original_fps": fps,
            "frame_count_from_metadata": total is not None,
            "decode_backend": "opencv",
            "video_source_kind": "file",
        }
    try:
        import av
    except ImportError:
        if backend == "auto":
            return _stream_video_metadata(path, decode_backend="opencv")
        raise RuntimeError("PyAV is required for --decode-backend auto when OpenCV fallback is unavailable")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = int(stream.frames or 0) or None
        fps = float(stream.average_rate) if stream.average_rate else None
    return {
        "original_frame_count": total,
        "original_fps": fps,
        "frame_count_from_metadata": total is not None,
        "decode_backend": "pyav",
        "video_source_kind": "file",
    }


def _iter_decoded_file_frames(
    path: Path,
    *,
    decode_backend: str,
    decode_fps: float | None,
    original_fps: float | None,
    max_decode_frames: int | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    backend = str(decode_backend)
    if backend == "opencv":
        yield from _iter_decoded_file_frames_opencv(
            path,
            decode_fps=decode_fps,
            original_fps=original_fps,
            max_decode_frames=max_decode_frames,
        )
        return
    if backend in {"decord", "torchvision"}:
        raise NotImplementedError(f"streaming decode backend '{backend}' is not implemented; use auto or opencv")
    try:
        yield from _iter_decoded_file_frames_pyav(
            path,
            decode_fps=decode_fps,
            original_fps=original_fps,
            max_decode_frames=max_decode_frames,
        )
    except ImportError:
        if backend == "auto":
            yield from _iter_decoded_file_frames_opencv(
                path,
                decode_fps=decode_fps,
                original_fps=original_fps,
                max_decode_frames=max_decode_frames,
            )
            return
        raise


def _decode_keep_stride(*, decode_fps: float | None, original_fps: float | None) -> int:
    if decode_fps is None or decode_fps <= 0 or original_fps is None or original_fps <= 0:
        return 1
    return max(1, round(float(original_fps) / float(decode_fps)))


def _iter_decoded_file_frames_pyav(
    path: Path,
    *,
    decode_fps: float | None,
    original_fps: float | None,
    max_decode_frames: int | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    import av

    keep_stride = _decode_keep_stride(decode_fps=decode_fps, original_fps=original_fps)
    kept = 0
    with av.open(str(path)) as container:
        for source_idx, frame in enumerate(container.decode(video=0)):
            if source_idx % keep_stride != 0:
                continue
            array = frame.to_ndarray(format="rgb24")
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            yield source_idx, tensor
            kept += 1
            if max_decode_frames is not None and kept >= int(max_decode_frames):
                break


def _iter_decoded_file_frames_opencv(
    path: Path,
    *,
    decode_fps: float | None,
    original_fps: float | None,
    max_decode_frames: int | None,
) -> Iterator[tuple[int, torch.Tensor]]:
    import cv2

    keep_stride = _decode_keep_stride(decode_fps=decode_fps, original_fps=original_fps)
    capture = cv2.VideoCapture(str(path))
    source_idx = 0
    kept = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_idx % keep_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                yield source_idx, tensor
                kept += 1
                if max_decode_frames is not None and kept >= int(max_decode_frames):
                    break
            source_idx += 1
    finally:
        capture.release()


def _iter_sample_stream_window(
    frame_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    window_size: int,
    max_windows: int | None,
    metadata: Mapping[str, Any],
) -> Iterator[StreamWindowInput]:
    total = metadata.get("original_frame_count")
    if total is None:
        raise NotImplementedError("streaming sample mode requires frame-count metadata; use chunk/all/interval or provide a backend with metadata")
    if max_windows == 0:
        return
    total_int = int(total)
    if total_int >= window_size:
        step = (total_int - 1) / float(max(1, window_size - 1))
        targets = [round(step * idx) for idx in range(window_size)] if window_size > 1 else [0]
    else:
        targets = list(range(total_int))
    target_set = set(targets)
    selected: list[torch.Tensor] = []
    selected_indices: list[int] = []
    decoded_count = 0
    for source_idx, frame in frame_iter:
        decoded_count += 1
        if source_idx in target_set:
            selected.append(frame)
            selected_indices.append(source_idx)
        if selected_indices and selected_indices[-1] >= targets[-1]:
            break
    if not selected:
        raise ValueError("no frames decoded for streaming sample window")
    padded_mask = [False] * len(selected)
    while len(selected) < window_size:
        selected.append(selected[-1])
        selected_indices.append(selected_indices[-1])
        padded_mask.append(True)
    yield StreamWindowInput(
        window_id=0,
        frames=torch.stack(selected, dim=0),
        frame_indices=selected_indices,
        padded_frame_mask=padded_mask,
        original_frame_count=total_int,
        original_fps=metadata.get("original_fps"),
        decoded_frame_count=decoded_count,
        video_source_kind=str(metadata.get("video_source_kind", "file")),
        decode_backend=str(metadata.get("decode_backend", "auto")),
        frame_count_from_metadata=bool(metadata.get("frame_count_from_metadata", False)),
    )


def _iter_chunk_like_stream_windows(
    frame_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    frame_selection_mode: str,
    window_size: int,
    stream_overlap: int,
    frame_interval: int,
    max_windows: int | None,
    metadata: Mapping[str, Any],
) -> Iterator[StreamWindowInput]:
    buffer_frames: list[torch.Tensor] = []
    buffer_indices: list[int] = []
    decoded_count = 0
    emitted = 0
    effective_mode = "chunk" if frame_selection_mode == "all" else frame_selection_mode
    for source_idx, frame in frame_iter:
        decoded_count += 1
        if effective_mode == "interval" and source_idx % frame_interval != 0:
            continue
        buffer_frames.append(frame)
        buffer_indices.append(source_idx)
        if len(buffer_frames) < window_size:
            continue
        yield _make_stream_window(
            window_id=emitted,
            frames=buffer_frames,
            indices=buffer_indices,
            padded_count=0,
            decoded_count=decoded_count,
            metadata=metadata,
        )
        emitted += 1
        if max_windows is not None and emitted >= max_windows:
            return
        if stream_overlap:
            buffer_frames = buffer_frames[-stream_overlap:]
            buffer_indices = buffer_indices[-stream_overlap:]
        else:
            buffer_frames = []
            buffer_indices = []
    if buffer_frames and (max_windows is None or emitted < max_windows):
        yield _make_stream_window(
            window_id=emitted,
            frames=buffer_frames,
            indices=buffer_indices,
            padded_count=0,
            decoded_count=decoded_count,
            metadata=metadata,
        )


def _make_stream_window(
    *,
    window_id: int,
    frames: list[torch.Tensor],
    indices: list[int],
    padded_count: int,
    decoded_count: int,
    metadata: Mapping[str, Any],
) -> StreamWindowInput:
    padded_mask = [False] * len(frames)
    if padded_count:
        padded_mask[-padded_count:] = [True] * padded_count
    return StreamWindowInput(
        window_id=window_id,
        frames=torch.stack(frames, dim=0),
        frame_indices=[int(item) for item in indices],
        padded_frame_mask=padded_mask,
        original_frame_count=metadata.get("original_frame_count"),
        original_fps=metadata.get("original_fps"),
        decoded_frame_count=int(decoded_count),
        video_source_kind=str(metadata.get("video_source_kind", "file")),
        decode_backend=str(metadata.get("decode_backend", "auto")),
        frame_count_from_metadata=bool(metadata.get("frame_count_from_metadata", False)),
    )


def prepare_stream_window(
    cfg: Mapping[str, Any],
    *,
    stream_window: StreamWindowInput,
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
    source_video = stream_window.frames
    window = FrameWindow(
        window_id=int(stream_window.window_id),
        frame_indices=list(stream_window.frame_indices),
        is_padded=any(stream_window.padded_frame_mask),
        padded_frame_mask=list(stream_window.padded_frame_mask),
        original_frame_count=int(stream_window.original_frame_count or stream_window.decoded_frame_count),
        effective_num_frames=stream_window.effective_num_frames,
    )
    selection = FrameSelectionResult(
        mode=frame_selection_mode,
        effective_mode="chunk" if frame_selection_mode == "all" else frame_selection_mode,
        num_frames=int(num_frames),
        frame_interval=int(frame_interval),
        max_windows=_normalize_max_windows(max_windows),
        drop_last=False,
        pad_last=any(stream_window.padded_frame_mask),
        original_frame_count=int(stream_window.original_frame_count or stream_window.decoded_frame_count),
        original_fps=stream_window.original_fps,
        windows=[window],
        unsupported_visualization_modes=["hold_last"],
    )
    scaled, scaling_record, window_chops = scale_video(
        source_video.unsqueeze(0),
        scaling_mode=scaling_mode,
        resolution=resolution,
        chop_size=chop_size,
        chop_overlap=chop_overlap,
        max_chops=max_chops,
        chop_merge_mode=chop_merge_mode,
        resize_before_chop_threshold=resize_before_chop_threshold,
        resize_before_chop_factor=resize_before_chop_factor,
    )
    scaling_record["window"] = window.to_dict()
    scaling_record["stream_window_id"] = int(stream_window.window_id)
    frame_records: list[dict[str, Any]] = []
    processed_index = 0
    chop_metadata: dict[str, Any] | None = None
    if window_chops is not None:
        chop_metadata = {
            "mode": "chop",
            "chop_size": chop_size,
            "chop_overlap": chop_overlap,
            "chop_merge_mode": chop_merge_mode,
            "windows": [{"window_id": int(stream_window.window_id), **window_chops}],
        }
        for chop_record in window_chops["records"]:
            for position, source_index in enumerate(window.frame_indices):
                frame_records.append(
                    {
                        "processed_frame_index": processed_index,
                        "source_frame_index": int(source_index),
                        "window_id": int(stream_window.window_id),
                        "stream_window_id": int(stream_window.window_id),
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
                    "window_id": int(stream_window.window_id),
                    "stream_window_id": int(stream_window.window_id),
                    "position_in_window": int(position),
                    "anchor_frame_index": None,
                    "is_padded": bool(window.padded_frame_mask[position]),
                }
            )
            processed_index += 1
    scaling_metadata = {
        "scaling_mode": scaling_mode,
        "resolution": resolution,
        "number_of_windows": 1,
        "first_processed_shape": [int(dim) for dim in scaled.shape],
        "temporal_pad_last_applied": bool(any(stream_window.padded_frame_mask)),
        "temporal_drop_last_applied": False,
        "selected_source_frame_count": stream_window.effective_num_frames,
        "video_read_mode": "streaming",
        "decode_backend": stream_window.decode_backend,
        "decoded_frame_count_so_far": stream_window.decoded_frame_count,
        "windows": [scaling_record],
    }
    return PreparedVideo(
        source_video=source_video,
        processed_video=scaled,
        frame_selection=selection,
        frame_records=frame_records,
        scaling_metadata=scaling_metadata,
        chop_metadata=chop_metadata,
        video_source_kind=stream_window.video_source_kind,
        original_fps=stream_window.original_fps,
    )


def validate_stream_window_memory(
    stream_window: StreamWindowInput,
    *,
    max_frames_in_memory: int | None,
    max_pixels_per_window: int | None,
) -> None:
    frame_count = int(stream_window.frames.shape[0])
    height = int(stream_window.frames.shape[-2])
    width = int(stream_window.frames.shape[-1])
    if max_frames_in_memory is not None and frame_count > int(max_frames_in_memory):
        raise RuntimeError(
            f"streaming memory guard blocked window {stream_window.window_id}: "
            f"{frame_count} frames exceed max_frames_in_memory={max_frames_in_memory}"
        )
    pixels = frame_count * height * width
    if max_pixels_per_window is not None and pixels > int(max_pixels_per_window):
        raise RuntimeError(
            f"streaming memory guard blocked window {stream_window.window_id}: "
            f"{pixels} frame-pixels exceed max_pixels_per_window={max_pixels_per_window}"
        )


def validate_source_video_memory(
    video: torch.Tensor,
    *,
    max_frames_in_memory: int | None,
    max_pixels_per_window: int | None,
) -> None:
    frame_count = int(video.shape[0])
    height = int(video.shape[-2])
    width = int(video.shape[-1])
    if max_frames_in_memory is not None and frame_count > int(max_frames_in_memory):
        raise RuntimeError(
            "full video memory guard blocked input: "
            f"{frame_count} decoded frames exceed max_video_frames_in_memory={max_frames_in_memory}. "
            "Set video_input.max_decode_frames lower, use sample mode, or use --video-read-mode streaming."
        )
    pixels = frame_count * height * width
    if max_pixels_per_window is not None and pixels > int(max_pixels_per_window):
        raise RuntimeError(
            "full video memory guard blocked input: "
            f"{pixels} decoded frame-pixels exceed max_pixels_per_window={max_pixels_per_window}. "
            "Set video_input.max_decode_frames lower, resize before full loading is not available, or use streaming."
        )


def validate_prepared_video_memory(
    prepared: PreparedVideo,
    *,
    max_processed_frames_per_window: int | None,
    max_processed_pixels_per_window: int | None,
) -> None:
    processed = prepared.processed_video
    if processed.ndim != 5:
        raise RuntimeError(f"processed video memory guard expected shape [B,T,C,H,W], got {tuple(processed.shape)}")
    batch = int(processed.shape[0])
    frames = int(processed.shape[1])
    height = int(processed.shape[-2])
    width = int(processed.shape[-1])
    processed_frames = batch * frames
    if max_processed_frames_per_window is not None and processed_frames > int(max_processed_frames_per_window):
        raise RuntimeError(
            "processed video memory guard blocked window: "
            f"{processed_frames} processed frames exceed "
            f"max_processed_frames_per_window={max_processed_frames_per_window}. "
            "This commonly happens when resize_then_chop creates many spatial crops; reduce num_frames, "
            "set --max-chops, use --scaling-mode resize, or increase the limit intentionally."
        )
    processed_pixels = processed_frames * height * width
    if max_processed_pixels_per_window is not None and processed_pixels > int(max_processed_pixels_per_window):
        raise RuntimeError(
            "processed video memory guard blocked window: "
            f"{processed_pixels} processed frame-pixels exceed "
            f"max_processed_pixels_per_window={max_processed_pixels_per_window}. "
            "Use smaller windows, lower resolution, fewer chops, or resize mode."
        )


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
    max_decode_frames: int | None = None,
    max_frames_in_memory: int | None = None,
    max_pixels_per_window: int | None = None,
) -> PreparedVideo:
    dummy_frames = int(nested_get(cfg, "input.dummy_frames", max(num_frames, 8)))
    dummy_resolution = int(nested_get(cfg, "input.dummy_resolution", max(resolution, 64)))
    source_video, source_kind, fps = load_video_frames(
        video_path,
        dummy_frames=dummy_frames,
        dummy_resolution=dummy_resolution,
        max_decode_frames=max_decode_frames,
    )
    validate_source_video_memory(
        source_video,
        max_frames_in_memory=max_frames_in_memory,
        max_pixels_per_window=max_pixels_per_window,
    )
    source_frame_count = int(source_video.shape[0])
    drop_incomplete_chop_windows = (
        scaling_mode in {"chop", "resize_then_chop"}
        and frame_selection_mode in {"all", "chunk"}
        and source_frame_count >= int(num_frames)
    )
    selection = select_frame_windows(
        original_frame_count=int(source_video.shape[0]),
        num_frames=num_frames,
        frame_selection_mode=frame_selection_mode,
        frame_interval=frame_interval,
        max_windows=max_windows,
        drop_last=drop_incomplete_chop_windows,
        pad_last=scaling_mode in {"chop", "resize_then_chop"} and not drop_incomplete_chop_windows,
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
        "temporal_drop_last_applied": bool(
            scaling_mode in {"chop", "resize_then_chop"}
            and selection.drop_last
            and _selected_source_frame_count(selection.windows) < int(source_video.shape[0])
        ),
        "selected_source_frame_count": _selected_source_frame_count(selection.windows),
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


def _selected_source_frame_count(windows: list[FrameWindow]) -> int:
    selected = set()
    for window in windows:
        for index, is_padded in zip(window.frame_indices, window.padded_frame_mask):
            if not is_padded:
                selected.add(int(index))
    return len(selected)


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
    requested_dtype: str | None = None,
    gaze_ratio: Any,
    task_loss_requirement: Any,
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
    requested_dtype = requested_dtype or dtype
    runtime: dict[str, Any] = {
        "gaze_ratio": gaze_ratio,
        "task_loss_requirement": task_loss_requirement,
        "strict_autogaze_params": strict_autogaze_params,
        "requested_checkpoint": checkpoint,
        "allow_real_model_loading": allow_real_model_loading,
        "requested_dtype": requested_dtype,
        "autogaze_execution_dtype": dtype,
        "autogaze_forced_float32": dtype == "float32" and requested_dtype != "float32",
        "patch_size": int(nested_get(cfg, "scaling.patch_size", 16)),
        "scales": configured_scales(cfg),
        "nvila_hd": nvila_hd_effective_settings(cfg),
    }

    if not autogaze_enabled:
        status = "disabled_full_token_path"
        reason = "AutoGaze is disabled by config; all visual tokens are retained."
        build_start = time.perf_counter()
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
        return _finalize_gaze_latency(result, build_start=build_start)

    has_checkpoint = checkpoint_exists(checkpoint)
    if allow_real_model_loading and not has_checkpoint:
        reason = f"AutoGaze checkpoint/model is missing: {checkpoint}"
        build_start = time.perf_counter()
        result = build_gaze_result(
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
        return _finalize_gaze_latency(result, build_start=build_start)
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
        build_start = time.perf_counter()
        result = build_gaze_result(
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
        return _finalize_gaze_latency(result, build_start=build_start)

    if not has_checkpoint:
        reason = f"AutoGaze checkpoint/model is missing: {checkpoint}"
    elif not allow_real_model_loading:
        reason = "real AutoGaze loading is disabled; pass --allow-real-model-loading to execute checkpoints"
    build_start = time.perf_counter()
    result = build_gaze_result(
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
    return _finalize_gaze_latency(result, build_start=build_start)


def _run_real_autogaze(
    cfg: Mapping[str, Any],
    prepared: PreparedVideo,
    *,
    device: str,
    dtype: str,
    gaze_ratio: Any,
    task_loss_requirement: Any,
    runtime_metadata: dict[str, Any],
    warmup_runs: int,
    progress: ProgressReporter,
) -> GazeResult:
    from autogaze.models.autogaze import AutoGaze

    checkpoint = nested_get(cfg, "autogaze.checkpoint_path") or nested_get(cfg, "autogaze.model_id")
    with suppress_transformers_torch_dtype_warning():
        model = AutoGaze.from_pretrained(model_reference_for_loading(checkpoint))
    model = model.to(device=normalize_device(device), dtype=torch.float32)
    model.eval()
    video = prepared.processed_video.to(device=normalize_device(device), dtype=dtype_from_name(dtype))
    kwargs: dict[str, Any] = {"gazing_ratio": gaze_ratio}
    if task_loss_requirement is not None:
        kwargs["task_loss_requirement"] = task_loss_requirement
    max_batch_size = _autogaze_forward_batch_size(cfg, total_batch=int(video.shape[0]))

    def forward_once() -> Mapping[str, Any]:
        return _run_autogaze_forward_batches(
            model,
            video,
            kwargs=kwargs,
            device=device,
            max_batch_size=max_batch_size,
            progress=None,
        )[0]

    progress.warmup("AutoGaze", forward_once, runs=warmup_runs, device=device)
    outputs, latency_ms, forward_metadata = _run_autogaze_forward_batches(
        model,
        video,
        kwargs=kwargs,
        device=device,
        max_batch_size=max_batch_size,
        progress=progress,
    )
    build_start = time.perf_counter()
    result = build_gaze_result(
        prepared,
        autogaze_enabled=True,
        status="real",
        reason=None,
        real_model_used=True,
        gaze_ratio=gaze_ratio,
        task_loss_requirement=task_loss_requirement,
        runtime_metadata={
            **runtime_metadata,
            **forward_metadata,
            "raw_output_keys": sorted(str(key) for key in outputs.keys()),
        },
        latency_ms=latency_ms,
        real_outputs=outputs,
    )
    return _finalize_gaze_latency(result, build_start=build_start, forward_latency_ms=latency_ms)


def _autogaze_forward_batch_size(cfg: Mapping[str, Any], *, total_batch: int) -> int:
    configured = (
        nested_get(cfg, "autogaze.max_batch_size")
        or nested_get(cfg, "autogaze.max_batch_size_autogaze")
        or nvila_hd_effective_settings(cfg).get("max_batch_size_autogaze")
    )
    if configured is None:
        return max(1, int(total_batch))
    return max(1, min(int(configured), int(total_batch)))


def _run_autogaze_forward_batches(
    model: Any,
    video: torch.Tensor,
    *,
    kwargs: Mapping[str, Any],
    device: str,
    max_batch_size: int,
    progress: ProgressReporter | None,
) -> tuple[Mapping[str, Any], float, dict[str, Any]]:
    total_batch = int(video.shape[0])
    frames = int(video.shape[1])
    batch_ranges = [
        (start, min(total_batch, start + int(max_batch_size)))
        for start in range(0, total_batch, int(max_batch_size))
    ]
    outputs: list[Mapping[str, Any]] = []
    batch_latencies: list[float] = []
    progress_bar = progress._start("AutoGaze", total=len(batch_ranges), unit="batch") if progress is not None else None
    try:
        for start, end in batch_ranges:
            batch_video = video[start:end]
            synchronize_device(device)
            batch_start = time.perf_counter()
            with torch.inference_mode():
                batch_output = model({"video": batch_video}, **kwargs)
            synchronize_device(device)
            batch_latency_ms = (time.perf_counter() - batch_start) * 1000
            outputs.append(batch_output)
            batch_latencies.append(batch_latency_ms)
            if progress is not None:
                progress._update(progress_bar, 1)
    finally:
        if progress is not None:
            progress._close(progress_bar)
    metadata = {
        "autogaze_model_forward_call_count": len(batch_ranges),
        "autogaze_model_forward_micro_batch_size": int(max_batch_size),
        "autogaze_model_forward_batch_latencies_ms": batch_latencies,
        "autogaze_model_forward_batch_ranges": [
            {
                "start": int(start),
                "end": int(end),
                "batch_size": int(end - start),
                "frame_count": frames,
                "processed_frame_count": int(end - start) * frames,
            }
            for start, end in batch_ranges
        ],
        "autogaze_model_forward_processed_clip_count": total_batch,
        "autogaze_model_forward_processed_frame_count": total_batch * frames,
    }
    return _merge_autogaze_forward_outputs(outputs), sum(batch_latencies), metadata


def _merge_autogaze_forward_outputs(outputs: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not outputs:
        return {}
    if len(outputs) == 1:
        return outputs[0]
    merged: dict[str, Any] = {}
    keys = sorted({key for output in outputs for key in output.keys()}, key=str)
    for key in keys:
        values = [output[key] for output in outputs if key in output]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if first.ndim == 0:
                merged[key] = first
            elif key in {"num_gazing_each_frame", "num_vision_tokens_each_frame"} and first.ndim <= 1:
                merged[key] = first
            else:
                merged[key] = _cat_autogaze_tensors(key, [value for value in values if isinstance(value, torch.Tensor)])
        else:
            merged[key] = first
    return merged


def _cat_autogaze_tensors(key: Any, tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    shapes = [tuple(tensor.shape) for tensor in tensors]
    if len({shape[1:] for shape in shapes}) == 1:
        return torch.cat(tensors, dim=0)
    if str(key) in {"gazing_pos", "if_padded_gazing"} and all(tensor.ndim == 2 for tensor in tensors):
        total_rows = sum(int(tensor.shape[0]) for tensor in tensors)
        max_cols = max(int(tensor.shape[1]) for tensor in tensors)
        fill_value = True if str(key) == "if_padded_gazing" else 0
        padded = torch.full(
            (total_rows, max_cols),
            fill_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        row_offset = 0
        for tensor in tensors:
            rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
            padded[row_offset : row_offset + rows, :cols] = tensor
            row_offset += rows
        return padded
    return torch.cat(tensors, dim=0)


def _finalize_gaze_latency(
    result: GazeResult,
    *,
    build_start: float,
    forward_latency_ms: float | None = None,
) -> GazeResult:
    build_latency_ms = (time.perf_counter() - build_start) * 1000
    model_forward_latency_ms = float(forward_latency_ms) if forward_latency_ms is not None else None
    stage_latency_ms = float(model_forward_latency_ms or 0.0) + float(build_latency_ms)
    runtime_metadata = dict(result.runtime_metadata)
    runtime_metadata.update(
        {
            "autogaze_stage_latency_ms": stage_latency_ms,
            "autogaze_model_forward_latency_ms": model_forward_latency_ms,
            "autogaze_result_build_latency_ms": build_latency_ms,
            "autogaze_processed_frame_count": len(result.per_frame),
        }
    )
    return replace(result, runtime_metadata=runtime_metadata, latency_ms=stage_latency_ms)


def build_gaze_result(
    prepared: PreparedVideo,
    *,
    autogaze_enabled: bool,
    status: str,
    reason: str | None,
    real_model_used: bool,
    gaze_ratio: Any,
    task_loss_requirement: Any,
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
        for idx in range(len(prepared.frame_records)):
            frame_ratio = gaze_ratio_for_frame(gaze_ratio, idx)
            selected_count = tokens_per_frame if not autogaze_enabled else max(1, min(tokens_per_frame, math.ceil(tokens_per_frame * frame_ratio)))
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
        "encoder_side_acceleration_claimed": False,
    }
    gazing_info_for_vit = None
    if real_outputs is not None:
        required = ("gazing_pos", "num_gazing_each_frame", "if_padded_gazing")
        if all(key in real_outputs for key in required):
            gazing_info_for_vit = {key: real_outputs[key] for key in required}
            runtime["gazing_info_source"] = "real_autogaze_output"
    if gazing_info_for_vit is None and autogaze_enabled and status != "blocked":
        gazing_info_for_vit = _build_gazing_info_for_vit_from_records(
            per_frame,
            batch_size=batch_size,
            frame_count=frame_count,
            tokens_per_frame=tokens_per_frame,
        )
        runtime["gazing_info_source"] = "selected_patch_records"
        runtime["gazing_info_shape"] = {
            key: [int(dim) for dim in value.shape]
            for key, value in gazing_info_for_vit.items()
            if isinstance(value, torch.Tensor)
        }
        runtime["gazing_info_padded_token_count"] = int(gazing_info_for_vit["if_padded_gazing"].sum().item())
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


def _build_gazing_info_for_vit_from_records(
    records: list[Mapping[str, Any]],
    *,
    batch_size: int,
    frame_count: int,
    tokens_per_frame: int,
) -> dict[str, torch.Tensor]:
    selected_by_batch_frame: list[list[list[int]]] = [
        [[] for _frame_idx in range(frame_count)]
        for _batch_idx in range(batch_size)
    ]
    for batch_idx in range(batch_size):
        for frame_idx in range(frame_count):
            record_index = batch_idx * frame_count + frame_idx
            if record_index >= len(records):
                continue
            selected = [
                max(0, min(tokens_per_frame - 1, int(item)))
                for item in records[record_index].get("selected_patch_indices", [])
            ]
            selected_by_batch_frame[batch_idx][frame_idx] = selected

    counts = [
        max(len(selected_by_batch_frame[batch_idx][frame_idx]) for batch_idx in range(batch_size))
        for frame_idx in range(frame_count)
    ]
    total_selected_with_padding = sum(counts)
    if total_selected_with_padding <= 0:
        counts = [1 for _frame_idx in range(frame_count)]
        total_selected_with_padding = frame_count

    gazing_pos = torch.zeros((batch_size, total_selected_with_padding), dtype=torch.long)
    if_padded = torch.ones((batch_size, total_selected_with_padding), dtype=torch.bool)
    for batch_idx in range(batch_size):
        offset = 0
        for frame_idx, count in enumerate(counts):
            selected = selected_by_batch_frame[batch_idx][frame_idx][:count]
            for selected_offset, local_index in enumerate(selected):
                gazing_pos[batch_idx, offset + selected_offset] = frame_idx * tokens_per_frame + int(local_index)
                if_padded[batch_idx, offset + selected_offset] = False
            offset += count

    return {
        "gazing_pos": gazing_pos,
        "num_gazing_each_frame": torch.tensor(counts, dtype=torch.long),
        "if_padded_gazing": if_padded,
        "num_vision_tokens_each_frame": torch.tensor(tokens_per_frame, dtype=torch.long),
    }


def _counts_for_batch(counts: torch.Tensor, *, batch_idx: int, frame_count: int) -> list[int]:
    if counts.ndim == 0:
        return [int(counts.item())] * frame_count
    if counts.ndim == 1:
        return [int(item) for item in counts.tolist()]
    selected = counts[min(batch_idx, counts.shape[0] - 1)]
    return [int(item) for item in selected.reshape(-1).tolist()]


def gaze_ratio_for_frame(value: Any, frame_index: int) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            return 1.0
        return float(value[min(int(frame_index), len(value) - 1)])
    return float(value)


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


class IncrementalVideoWriter:
    def __init__(self, path: Path, *, fps: float) -> None:
        self.path = path
        self.fps = fps
        self._writer: Any = None
        self.error: str | None = None
        self.frame_count = 0

    def append(self, image: Image.Image) -> None:
        if self.error:
            return
        try:
            import imageio.v2 as imageio

            if self._writer is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    self._writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264")
                except Exception:
                    self._writer = imageio.get_writer(str(self.path), fps=self.fps)
            frame = pad_to_even_image(image).convert("RGB")
            self._writer.append_data(np.asarray(frame))
            self.frame_count += 1
        except Exception as exc:  # pragma: no cover - backend availability varies.
            self.error = str(exc)

    def close(self) -> str | None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:  # pragma: no cover - backend availability varies.
                self.error = self.error or str(exc)
        if self.frame_count == 0 and self.error is None:
            self.error = "no frames available"
        return self.error


class StreamingVisualizationSink:
    def __init__(
        self,
        output_dir: Path,
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
    ) -> None:
        if video_export_mode == "full_length" and (save_overlay_video or save_side_by_side_video or save_scale_panel_video):
            raise NotImplementedError(
                "full_length video export is not supported in streaming mode yet; use sampled_only or disable video export"
            )
        self.output_dir = output_dir
        self.base = output_dir / "visualizations" / "autogaze"
        self.frames_dir = self.base / "frames"
        self.panels_dir = self.base / "scale_panels"
        self.videos_dir = self.base / "videos"
        self.metadata_dir = self.base / "metadata"
        self.overlay_style = overlay_style
        self.overlay_alpha = overlay_alpha
        self.multi_scale_overlay = multi_scale_overlay
        self.show_patch_index = show_patch_index
        self.show_scale_label = show_scale_label
        self.metadata_placement = metadata_placement
        self.info_panel_position = info_panel_position
        self.save_frame_images = save_frame_images
        self.save_overlay_video = save_overlay_video
        self.save_side_by_side_video = save_side_by_side_video
        self.save_scale_panel_video = save_scale_panel_video
        self.video_fps = video_fps
        self.video_export_mode = video_export_mode
        self.query_text = query_text
        self.records: list[dict[str, Any]] = []
        self.processed_records: list[dict[str, Any]] = []
        self.video_errors: dict[str, str] = {}
        self.rendered_frame_count = 0
        self.saw_chop = False
        self.scale_layout: Any = None
        for directory in (self.metadata_dir,):
            directory.mkdir(parents=True, exist_ok=True)
        if save_frame_images:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            self.panels_dir.mkdir(parents=True, exist_ok=True)
        if save_overlay_video or save_side_by_side_video or save_scale_panel_video:
            self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_writer = (
            IncrementalVideoWriter(self.videos_dir / "autogaze_overlay.mp4", fps=video_fps)
            if save_overlay_video
            else None
        )
        self.side_by_side_writer = (
            IncrementalVideoWriter(self.videos_dir / "autogaze_side_by_side.mp4", fps=video_fps)
            if save_side_by_side_video
            else None
        )
        self.scale_panel_writer = (
            IncrementalVideoWriter(self.videos_dir / "autogaze_scale_panels.mp4", fps=video_fps)
            if save_scale_panel_video
            else None
        )

    def write_window(
        self,
        prepared: PreparedVideo,
        gaze: GazeResult,
        *,
        processed_frame_offset: int,
        visualization_frame_offset: int,
        generation_status: str | None = None,
    ) -> int:
        render_needed = self.save_frame_images or self.save_overlay_video or self.save_side_by_side_video or self.save_scale_panel_video
        processed_records = [_offset_record(record, processed_frame_offset) for record in prepared.frame_records]
        self.processed_records.extend(processed_records)
        if self.scale_layout is None:
            self.scale_layout = gaze.runtime_metadata.get("scale_layout")
        if prepared.chop_metadata is not None:
            self.saw_chop = True
            grouped_records = _group_chop_records([_offset_record(record, processed_frame_offset) for record in gaze.per_frame])
            for local_idx, records in enumerate(grouped_records):
                global_idx = visualization_frame_offset + local_idx
                merged_record = _merged_chop_record(records, global_idx)
                merged_record["stream_window_id"] = int(records[0].get("stream_window_id", records[0].get("window_id", 0)))
                self.records.append(merged_record)
                if not render_needed:
                    continue
                source_pos = max(0, min(int(records[0].get("position_in_window", 0)), int(prepared.source_video.shape[0]) - 1))
                base_image = tensor_to_image(prepared.source_video.detach().cpu()[source_pos])
                overlay = render_chop_merged_overlay(
                    base_image,
                    records,
                    overlay_style=self.overlay_style,
                    overlay_alpha=self.overlay_alpha,
                    multi_scale_overlay=self.multi_scale_overlay,
                    show_patch_index=self.show_patch_index,
                    show_scale_label=self.show_scale_label,
                )
                overlay_with_panel = add_info_panel(
                    overlay,
                    merged_record,
                    gaze,
                    scaling_mode=prepared.scaling_metadata["scaling_mode"],
                    metadata_placement=self.metadata_placement,
                    info_panel_position=self.info_panel_position,
                    query_text=self.query_text,
                    generation_status=generation_status,
                )
                scale_panel = render_chop_merged_scale_panel(base_image, records)
                side_by_side = combine_side_by_side(base_image, overlay)
                self._write_visual_frame(global_idx, overlay_with_panel, side_by_side, scale_panel)
            return len(grouped_records)

        frames = flatten_video_frames(prepared.processed_video).detach().cpu()
        written = 0
        for local_idx, frame in enumerate(frames):
            global_idx = processed_frame_offset + local_idx
            record = _offset_record(gaze.per_frame[local_idx], processed_frame_offset)
            record["stream_window_id"] = int(record.get("stream_window_id", record.get("window_id", 0)))
            self.records.append(record)
            if render_needed:
                base_image = tensor_to_image(frame)
                overlay = render_overlay(
                    base_image,
                    record,
                    gaze.patch_grid,
                    overlay_style=self.overlay_style,
                    overlay_alpha=self.overlay_alpha,
                    multi_scale_overlay=self.multi_scale_overlay,
                    show_patch_index=self.show_patch_index,
                    show_scale_label=self.show_scale_label,
                )
                overlay_with_panel = add_info_panel(
                    overlay,
                    record,
                    gaze,
                    scaling_mode=prepared.scaling_metadata["scaling_mode"],
                    metadata_placement=self.metadata_placement,
                    info_panel_position=self.info_panel_position,
                    query_text=self.query_text,
                    generation_status=generation_status,
                )
                scale_panel = render_scale_panel(base_image, record, gaze.patch_grid)
                side_by_side = combine_side_by_side(base_image, overlay)
                self._write_visual_frame(global_idx, overlay_with_panel, side_by_side, scale_panel)
            written += 1
        return written

    def _write_visual_frame(self, global_idx: int, overlay: Image.Image, side_by_side: Image.Image, scale_panel: Image.Image) -> None:
        if self.save_frame_images:
            overlay.save(self.frames_dir / f"frame_{global_idx:06d}_overlay.png")
            scale_panel.save(self.panels_dir / f"frame_{global_idx:06d}_scale_panel.png")
        if self.overlay_writer is not None:
            self.overlay_writer.append(overlay.convert("RGB"))
        if self.side_by_side_writer is not None:
            self.side_by_side_writer.append(side_by_side.convert("RGB"))
        if self.scale_panel_writer is not None:
            self.scale_panel_writer.append(scale_panel.convert("RGB"))
        self.rendered_frame_count += 1

    def close(self) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        if self.save_frame_images:
            artifacts["frames_dir"] = str(self.frames_dir)
            artifacts["scale_panels_dir"] = str(self.panels_dir)
        for key, writer, path in (
            ("overlay_video", self.overlay_writer, self.videos_dir / "autogaze_overlay.mp4"),
            ("side_by_side_video", self.side_by_side_writer, self.videos_dir / "autogaze_side_by_side.mp4"),
            ("scale_panel_video", self.scale_panel_writer, self.videos_dir / "autogaze_scale_panels.mp4"),
        ):
            if writer is None:
                continue
            error = writer.close()
            artifacts[key] = str(path) if error is None else f"failed: {error}"
            if error:
                self.video_errors[key] = error
        write_json(
            self.metadata_dir / "visualization_metadata.json",
            {
                "status": "ready",
                "streaming_output": True,
                "flat_output_structure": True,
                "visualization_mode": "merged_chop_source_frames" if self.saw_chop else "processed_frames",
                "streaming_visualization": True,
                "video_export_mode": self.video_export_mode,
                "frame_images_saved": bool(self.save_frame_images),
                "rendered_frame_count": self.rendered_frame_count,
                "frame_count": len(self.records),
                "processed_crop_frame_count": len(self.processed_records) if self.saw_chop else None,
                "frame_records": self.records,
                "processed_frame_records": self.processed_records,
                "scale_colors": {str(key): value for key, value in SCALE_COLORS.items()},
                "scale_layout": self.scale_layout,
                "video_errors": self.video_errors,
                "paths": artifacts,
            },
        )
        artifacts["visualization_metadata"] = str(self.metadata_dir / "visualization_metadata.json")
        return artifacts


def _offset_record(record: Mapping[str, Any], processed_frame_offset: int) -> dict[str, Any]:
    item = dict(record)
    if "processed_frame_index" in item:
        item["processed_frame_index"] = int(item["processed_frame_index"]) + int(processed_frame_offset)
    return item


def new_streaming_aggregate(
    *,
    video_read_mode: str,
    decode_backend: str,
    frame_selection_mode: str,
    frame_interval: int,
    stream_window_size: int,
    stream_overlap: int,
    max_stream_windows: int | None,
    max_decode_frames: int | None,
    decode_fps: float | None,
) -> dict[str, Any]:
    return {
        "video_read_mode": video_read_mode,
        "decode_backend": decode_backend,
        "frame_selection_mode": frame_selection_mode,
        "frame_interval": int(frame_interval),
        "stream_window_size": stream_window_size,
        "stream_overlap": stream_overlap,
        "max_stream_windows": max_stream_windows,
        "max_decode_frames": max_decode_frames,
        "decode_fps": decode_fps,
        "windows": [],
        "frame_records": [],
        "per_frame": [],
        "scaling_windows": [],
        "chop_windows": [],
        "per_window_metrics": [],
        "original_token_count": 0,
        "selected_token_count": 0,
        "processed_frame_count": 0,
        "visualization_frame_count": 0,
        "decoded_frame_count": 0,
        "original_frame_count": None,
        "original_fps": None,
        "frame_count_from_metadata": False,
        "source_kind": None,
        "runtime_metadata": {},
        "gaze_statuses": [],
    }


def add_streaming_window_result(
    aggregate: dict[str, Any],
    *,
    stream_window: StreamWindowInput,
    prepared: PreparedVideo,
    gaze: GazeResult,
    preprocessing_latency_ms: float,
    autogaze_latency_ms: float,
    vision_encoder_latency_ms: float | None = None,
    mllm_generation_latency_ms: float | None = None,
    visualization_latency_ms: float | None = None,
    generation_status: str | None = None,
    output_text: str | None = None,
    generation_metadata: Mapping[str, Any] | None = None,
    skipped_stages: list[dict[str, str]] | None = None,
) -> tuple[int, int]:
    processed_offset = int(aggregate["processed_frame_count"])
    visualization_offset = int(aggregate["visualization_frame_count"])
    autogaze_stage_latency_ms = float(autogaze_latency_ms or 0.0)
    autogaze_total_latency_ms = float(preprocessing_latency_ms or 0.0) + autogaze_stage_latency_ms
    model_forward_latency_ms = gaze.runtime_metadata.get("autogaze_model_forward_latency_ms")
    frame_records = [_offset_record(record, processed_offset) for record in prepared.frame_records]
    source_frame_count = _source_frame_count(frame_records)
    processed_frame_count = len(frame_records)
    window_memory_metrics = prepared_video_memory_metrics(prepared)
    generation_metadata = dict(generation_metadata or {})
    per_frame = [_offset_record(record, processed_offset) for record in gaze.per_frame]
    aggregate["frame_records"].extend(frame_records)
    aggregate["per_frame"].extend(per_frame)
    aggregate["windows"].extend([window.to_dict() for window in prepared.frame_selection.windows])
    aggregate["scaling_windows"].extend(prepared.scaling_metadata.get("windows", []))
    if prepared.chop_metadata is not None:
        aggregate["chop_windows"].extend(prepared.chop_metadata.get("windows", []))
        visualization_count = len(_group_chop_records(per_frame))
    else:
        visualization_count = len(per_frame)
    aggregate["original_token_count"] += int(gaze.original_token_count)
    aggregate["selected_token_count"] += int(gaze.selected_token_count)
    aggregate["processed_frame_count"] += len(frame_records)
    aggregate["visualization_frame_count"] += visualization_count
    aggregate["decoded_frame_count"] = max(int(aggregate["decoded_frame_count"]), int(stream_window.decoded_frame_count))
    aggregate["original_frame_count"] = stream_window.original_frame_count
    aggregate["original_fps"] = stream_window.original_fps
    aggregate["frame_count_from_metadata"] = stream_window.frame_count_from_metadata
    aggregate["source_kind"] = stream_window.video_source_kind
    aggregate["runtime_metadata"] = dict(gaze.runtime_metadata)
    aggregate["gaze_statuses"].append(gaze.status)
    aggregate["per_window_metrics"].append(
        {
            "window_id": stream_window.window_id,
            "source_frame_indices": list(stream_window.frame_indices),
            "effective_num_frames": stream_window.effective_num_frames,
            "source_frame_count": source_frame_count,
            "processed_frame_count": processed_frame_count,
            **window_memory_metrics,
            "visualization_frame_count": visualization_count,
            "original_token_count": gaze.original_token_count,
            "selected_token_count": gaze.selected_token_count,
            "token_reduction_ratio": gaze.token_reduction_ratio,
            "preprocessing_latency_ms": preprocessing_latency_ms,
            "autogaze_preprocessing_latency_ms": preprocessing_latency_ms,
            "autogaze_latency_ms": autogaze_total_latency_ms,
            "autogaze_latency_includes_preprocessing": True,
            "autogaze_latency_scope": "preprocessing_plus_autogaze_stage_over_processed_frames",
            "autogaze_latency_source_frame_count": source_frame_count,
            "autogaze_latency_processed_frame_count": processed_frame_count,
            "autogaze_latency_per_source_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, source_frame_count),
            "autogaze_latency_per_processed_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, processed_frame_count),
            "autogaze_preprocessing_latency_per_processed_frame_ms": _safe_latency_per_item(
                preprocessing_latency_ms,
                processed_frame_count,
            ),
            "autogaze_stage_latency_per_processed_frame_ms": _safe_latency_per_item(
                autogaze_stage_latency_ms,
                processed_frame_count,
            ),
            "autogaze_model_forward_latency_ms": model_forward_latency_ms,
            "autogaze_model_forward_status": "measured" if model_forward_latency_ms is not None else "not_run",
            "autogaze_model_forward_reason": None if model_forward_latency_ms is not None else (gaze.reason or gaze.status),
            "autogaze_non_forward_latency_ms": (
                autogaze_stage_latency_ms - float(model_forward_latency_ms or 0.0)
            ),
            "autogaze_model_forward_call_count": gaze.runtime_metadata.get("autogaze_model_forward_call_count"),
            "autogaze_model_forward_micro_batch_size": gaze.runtime_metadata.get("autogaze_model_forward_micro_batch_size"),
            "autogaze_model_forward_batch_latencies_ms": gaze.runtime_metadata.get("autogaze_model_forward_batch_latencies_ms"),
            "autogaze_model_forward_batch_ranges": gaze.runtime_metadata.get("autogaze_model_forward_batch_ranges"),
            "autogaze_model_forward_processed_clip_count": gaze.runtime_metadata.get("autogaze_model_forward_processed_clip_count"),
            "autogaze_model_forward_processed_frame_count": gaze.runtime_metadata.get("autogaze_model_forward_processed_frame_count"),
            "autogaze_result_build_latency_ms": gaze.runtime_metadata.get("autogaze_result_build_latency_ms"),
            "autogaze_stage_latency_ms": gaze.runtime_metadata.get("autogaze_stage_latency_ms", autogaze_stage_latency_ms),
            "vision_encoder_latency_ms": vision_encoder_latency_ms,
            "mllm_processor_latency_ms": generation_metadata.get("mllm_processor_latency_ms"),
            "mllm_input_move_latency_ms": generation_metadata.get("mllm_input_move_latency_ms"),
            "mllm_model_generate_latency_ms": generation_metadata.get("mllm_model_generate_latency_ms"),
            "mllm_generation_timed_scope": generation_metadata.get("mllm_generation_timed_scope"),
            "nvila_processor_internal_autogaze_timing_status": generation_metadata.get(
                "nvila_processor_internal_autogaze_timing_status"
            ),
            "mllm_generation_latency_ms": mllm_generation_latency_ms,
            "visualization_latency_ms": visualization_latency_ms,
            "generation_status": generation_status,
            "output_text": output_text,
            "skipped_stages": skipped_stages or [],
        }
    )
    return processed_offset, visualization_offset


def write_streaming_autogaze_artifacts(output_dir: Path, aggregate: Mapping[str, Any], *, status: str, reason: str | None = None) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    autogaze_dir = output_dir / "autogaze"
    token_reduction_ratio = (
        None
        if int(aggregate["original_token_count"]) == 0
        else 1.0 - (int(aggregate["selected_token_count"]) / float(int(aggregate["original_token_count"])))
    )
    frame_selection_metadata = {
        "mode": aggregate["frame_selection_mode"],
        "effective_mode": "chunk" if aggregate["frame_selection_mode"] == "all" else aggregate["frame_selection_mode"],
        "num_frames": aggregate["stream_window_size"],
        "frame_interval": aggregate.get("frame_interval"),
        "max_windows": aggregate["max_stream_windows"],
        "drop_last": False,
        "pad_last": any(bool(window.get("is_padded")) for window in aggregate["windows"]),
        "video_read_mode": aggregate["video_read_mode"],
        "decode_backend": aggregate["decode_backend"],
        "original_frame_count": aggregate["original_frame_count"],
        "original_fps": aggregate["original_fps"],
        "decoded_frame_count": aggregate["decoded_frame_count"],
        "processed_frame_count": aggregate["processed_frame_count"],
        "frame_selection_mode": aggregate["frame_selection_mode"],
        "stream_window_size": aggregate["stream_window_size"],
        "stream_overlap": aggregate["stream_overlap"],
        "number_of_windows": len(aggregate["windows"]),
        "window_frame_indices": [window["frame_indices"] for window in aggregate["windows"]],
        "frame_count_from_metadata": aggregate["frame_count_from_metadata"],
        "dropped_or_padded": any(bool(window.get("is_padded")) for window in aggregate["windows"]),
        "windows": aggregate["windows"],
    }
    write_json(autogaze_dir / "frame_selection_metadata.json", frame_selection_metadata)
    write_json(
        autogaze_dir / "runtime_metadata.json",
        {**dict(aggregate.get("runtime_metadata") or {}), "status": status, "reason": reason, "streaming": True},
    )
    write_json(
        autogaze_dir / "token_counts_summary.json",
        {
            "original_token_count": aggregate["original_token_count"],
            "selected_token_count": aggregate["selected_token_count"],
            "token_reduction_ratio": token_reduction_ratio,
            "status": status,
            "reason": reason,
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
                        "stream_window_id",
                        "position_in_window",
                        "chop_index",
                        "source_box",
                        "selected_patch_indices",
                        "selected_patch_records",
                        "global_patch_indices",
                    )
                    if k in item
                }
                for item in aggregate["per_frame"]
            ]
        },
    )
    write_json(
        autogaze_dir / "selected_scales.json",
        {
            "frames": [
                {
                    k: item[k]
                    for k in ("processed_frame_index", "source_frame_index", "stream_window_id", "selected_scales")
                    if k in item
                }
                for item in aggregate["per_frame"]
            ]
        },
    )
    write_json(
        autogaze_dir / "per_frame_token_counts.json",
        {
            "frames": [
                {
                    "processed_frame_index": item["processed_frame_index"],
                    "source_frame_index": item["source_frame_index"],
                    "stream_window_id": item.get("stream_window_id"),
                    "original_token_count": item["original_token_count"],
                    "selected_token_count": item["selected_token_count"],
                    "token_reduction_ratio": item["token_reduction_ratio"],
                    "selected_patch_count_by_scale": item["selected_patch_count_by_scale"],
                }
                for item in aggregate["per_frame"]
            ]
        },
    )
    write_json(
        output_dir / "scaling" / "scaling_metadata.json",
        {
            "video_read_mode": aggregate["video_read_mode"],
            "scaling_mode": aggregate["scaling_windows"][0]["scaling_mode"] if aggregate["scaling_windows"] else None,
            "number_of_windows": len(aggregate["scaling_windows"]),
            "windows": aggregate["scaling_windows"],
        },
    )
    if aggregate["chop_windows"]:
        write_json(
            output_dir / "chops" / "chop_metadata.json",
            {
                "mode": "chop",
                "streaming": True,
                "windows": aggregate["chop_windows"],
            },
        )
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
    if aggregate["chop_windows"]:
        artifacts["chop_metadata"] = str(output_dir / "chops" / "chop_metadata.json")
    return artifacts


def build_streaming_metrics(
    *,
    mode: str,
    cfg: Mapping[str, Any],
    video_path: str,
    query_text: str | None,
    aggregate: Mapping[str, Any],
    requested_vision_encoder: str | None,
    actual_vision_encoder: str | None,
    requested_mllm: str | None,
    actual_mllm: str | None,
    generation_status: str | None,
    output_text: str | None,
    skipped_stages: list[dict[str, str]],
    total_latency_ms: float,
    wall_clock_latency_ms: float,
    visualization_latency_ms: float | None,
    warmup_runs: int,
) -> dict[str, Any]:
    original = int(aggregate["original_token_count"])
    selected = int(aggregate["selected_token_count"])
    token_reduction_ratio = None if original == 0 else 1.0 - (selected / float(original))
    gaze_statuses = list(aggregate.get("gaze_statuses") or [])
    real_stub_blocked_status = gaze_statuses[0] if len(set(gaze_statuses)) <= 1 and gaze_statuses else "mixed"
    runtime_metadata = dict(aggregate.get("runtime_metadata") or {})
    nvila_hd = nvila_hd_effective_settings(cfg)
    source_frame_count = _source_frame_count(list(aggregate["frame_records"]))
    processed_frame_count = int(aggregate["processed_frame_count"])
    autogaze_preprocessing_latency_ms = _sum_optional_metric(aggregate["per_window_metrics"], "autogaze_preprocessing_latency_ms")
    autogaze_stage_latency_ms = sum(float(item["autogaze_stage_latency_ms"] or 0.0) for item in aggregate["per_window_metrics"])
    autogaze_total_latency_ms = sum(float(item["autogaze_latency_ms"] or 0.0) for item in aggregate["per_window_metrics"])
    autogaze_model_forward_latency_ms = _sum_optional_metric(aggregate["per_window_metrics"], "autogaze_model_forward_latency_ms")
    autogaze_non_forward_latency_ms = _sum_optional_metric(aggregate["per_window_metrics"], "autogaze_non_forward_latency_ms")
    source_tensor_bytes_per_window = [item.get("source_video_tensor_bytes") for item in aggregate["per_window_metrics"]]
    processed_tensor_bytes_per_window = [item.get("processed_video_tensor_bytes") for item in aggregate["per_window_metrics"]]
    processed_expansion_per_window = [item.get("processed_to_source_frame_expansion_ratio") for item in aggregate["per_window_metrics"]]
    max_source_tensor_bytes = _max_optional_count(source_tensor_bytes_per_window)
    max_processed_tensor_bytes = _max_optional_count(processed_tensor_bytes_per_window)
    sum_processed_tensor_bytes = _sum_values_as_int(processed_tensor_bytes_per_window)
    spatial_chop_count_total = (
        sum(int(window.get("spatial_chop_count", len(window.get("records", [])))) for window in aggregate.get("chop_windows") or [])
        if aggregate.get("chop_windows")
        else None
    )
    return {
        "mode": mode,
        "config_path": cfg.get("_config_path"),
        "video_path": video_path,
        "query_text": query_text,
        "video_read_mode": aggregate["video_read_mode"],
        "decode_backend": aggregate["decode_backend"],
        "decoded_frame_count": aggregate["decoded_frame_count"],
        "processed_frame_count": aggregate["processed_frame_count"],
        "number_of_frames": aggregate["processed_frame_count"],
        "number_of_processed_frames": aggregate["processed_frame_count"],
        "number_of_source_frames": source_frame_count,
        "number_of_windows": len(aggregate["windows"]),
        "spatial_chop_count_total": spatial_chop_count_total,
        "spatial_chops_per_window": {
            str(window.get("window_id", idx)): int(window.get("spatial_chop_count", len(window.get("records", []))))
            for idx, window in enumerate(aggregate.get("chop_windows") or [])
        }
        or None,
        "window_size": aggregate["stream_window_size"],
        "stream_window_size": aggregate["stream_window_size"],
        "stream_overlap": aggregate["stream_overlap"],
        "max_resident_frames_in_memory": aggregate["stream_window_size"],
        "frame_selection_mode": aggregate["frame_selection_mode"],
        "frame_interval": aggregate.get("frame_interval"),
        "scaling_mode": aggregate["scaling_windows"][0]["scaling_mode"] if aggregate["scaling_windows"] else None,
        "autogaze_enabled": bool(nested_get(cfg, "autogaze.enabled", False)),
        "requested_vision_encoder": requested_vision_encoder,
        "actual_vision_encoder": actual_vision_encoder,
        "requested_mllm": requested_mllm,
        "actual_mllm": actual_mllm,
        "generation_status": generation_status,
        "real_stub_blocked_status": real_stub_blocked_status,
        "requested_runtime_dtype": runtime_metadata.get("requested_dtype"),
        "autogaze_dtype": runtime_metadata.get("autogaze_execution_dtype"),
        "autogaze_forced_float32": runtime_metadata.get("autogaze_forced_float32"),
        "gaze_ratio": runtime_metadata.get("gaze_ratio"),
        "task_loss_requirement": runtime_metadata.get("task_loss_requirement"),
        "nvila_hd": nvila_hd,
        "num_video_frames": nvila_hd.get("num_video_frames"),
        "num_video_frames_thumbnail": nvila_hd.get("num_video_frames_thumbnail"),
        "max_tiles_video": nvila_hd.get("max_tiles_video"),
        "effective_gazing_ratio_tile": nvila_hd.get("gazing_ratio_tile"),
        "effective_task_loss_requirement_tile": nvila_hd.get("task_loss_requirement_tile"),
        "effective_gazing_ratio_thumbnail": nvila_hd.get("gazing_ratio_thumbnail"),
        "effective_task_loss_requirement_thumbnail": nvila_hd.get("task_loss_requirement_thumbnail"),
        "max_batch_size_autogaze": nvila_hd.get("max_batch_size_autogaze"),
        "max_batch_size_siglip": nvila_hd.get("max_batch_size_siglip"),
        "number_of_tiles_processed": _streaming_tile_count(aggregate, nvila_hd),
        "number_of_thumbnail_frames_processed": nvila_hd.get("num_video_frames_thumbnail"),
        "original_token_count": original,
        "selected_token_count": selected,
        "token_reduction_ratio": token_reduction_ratio,
        "full_processed_visual_token_count": original,
        "autogaze_selected_visual_token_count": selected,
        "estimated_visual_token_savings_ratio": token_reduction_ratio,
        "selected_patches_per_window": [item["selected_token_count"] for item in aggregate["per_window_metrics"]],
        "selected_patches_per_frame": [item["selected_token_count"] for item in aggregate["per_frame"]],
        "selected_patches_per_scale": _sum_scale_counts(aggregate["per_frame"]),
        "original_token_count_per_window": [item["original_token_count"] for item in aggregate["per_window_metrics"]],
        "token_reduction_ratio_per_window": [item["token_reduction_ratio"] for item in aggregate["per_window_metrics"]],
        "source_video_tensor_bytes_per_window": source_tensor_bytes_per_window,
        "source_video_tensor_mib_per_window": [item.get("source_video_tensor_mib") for item in aggregate["per_window_metrics"]],
        "processed_video_tensor_bytes_per_window": processed_tensor_bytes_per_window,
        "processed_video_tensor_mib_per_window": [item.get("processed_video_tensor_mib") for item in aggregate["per_window_metrics"]],
        "processed_to_source_frame_expansion_ratio_per_window": processed_expansion_per_window,
        "processed_to_source_tensor_byte_ratio_per_window": [
            item.get("processed_to_source_tensor_byte_ratio") for item in aggregate["per_window_metrics"]
        ],
        "max_source_video_tensor_bytes_per_window": max_source_tensor_bytes,
        "max_source_video_tensor_mib_per_window": bytes_to_mib(max_source_tensor_bytes),
        "max_processed_video_tensor_bytes_per_window": max_processed_tensor_bytes,
        "max_processed_video_tensor_mib_per_window": bytes_to_mib(max_processed_tensor_bytes),
        "sum_processed_video_tensor_bytes_over_windows": sum_processed_tensor_bytes,
        "sum_processed_video_tensor_mib_over_windows": bytes_to_mib(sum_processed_tensor_bytes),
        "source_video_tensor_bytes": max_source_tensor_bytes,
        "source_video_tensor_mib": bytes_to_mib(max_source_tensor_bytes),
        "processed_video_tensor_bytes": max_processed_tensor_bytes,
        "processed_video_tensor_mib": bytes_to_mib(max_processed_tensor_bytes),
        "mllm_input_tensor_bytes": max_processed_tensor_bytes,
        "mllm_input_tensor_mib": bytes_to_mib(max_processed_tensor_bytes),
        "mllm_input_tensor_memory_scope": "max_per_stream_window",
        "max_processed_to_source_frame_expansion_ratio": _max_optional_float(processed_expansion_per_window),
        "preprocessing_latency_per_window_ms": [item["preprocessing_latency_ms"] for item in aggregate["per_window_metrics"]],
        "autogaze_preprocessing_latency_per_window_ms": [
            item.get("autogaze_preprocessing_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_latency_per_window_ms": [item["autogaze_latency_ms"] for item in aggregate["per_window_metrics"]],
        "autogaze_model_forward_latency_per_window_ms": [
            item.get("autogaze_model_forward_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_model_forward_call_count_per_window": [
            item.get("autogaze_model_forward_call_count") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_model_forward_batch_latencies_per_window_ms": [
            item.get("autogaze_model_forward_batch_latencies_ms") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_model_forward_processed_frame_count_per_window": [
            item.get("autogaze_model_forward_processed_frame_count") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_model_forward_status_per_window": [
            item.get("autogaze_model_forward_status") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_model_forward_reason_per_window": [
            item.get("autogaze_model_forward_reason") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_non_forward_latency_per_window_ms": [
            item.get("autogaze_non_forward_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_result_build_latency_per_window_ms": [
            item.get("autogaze_result_build_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "autogaze_stage_latency_per_window_ms": [
            item.get("autogaze_stage_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "vision_encoder_latency_per_window_ms": [item["vision_encoder_latency_ms"] for item in aggregate["per_window_metrics"]],
        "mllm_processor_latency_per_window_ms": [
            item.get("mllm_processor_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "mllm_input_move_latency_per_window_ms": [
            item.get("mllm_input_move_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "mllm_model_generate_latency_per_window_ms": [
            item.get("mllm_model_generate_latency_ms") for item in aggregate["per_window_metrics"]
        ],
        "mllm_latency_per_window_ms": [item["mllm_generation_latency_ms"] for item in aggregate["per_window_metrics"]],
        "autogaze_latency_includes_preprocessing": True,
        "autogaze_latency_scope": "preprocessing_plus_autogaze_stage_over_processed_frames",
        "autogaze_latency_source_frame_count": source_frame_count,
        "autogaze_latency_processed_frame_count": processed_frame_count,
        "autogaze_latency_per_source_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, source_frame_count),
        "autogaze_latency_per_processed_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, processed_frame_count),
        "autogaze_preprocessing_latency_per_processed_frame_ms": _safe_latency_per_item(
            autogaze_preprocessing_latency_ms,
            processed_frame_count,
        ),
        "autogaze_stage_latency_per_processed_frame_ms": _safe_latency_per_item(
            autogaze_stage_latency_ms,
            processed_frame_count,
        ),
        "autogaze_preprocessing_latency_ms": autogaze_preprocessing_latency_ms,
        "autogaze_latency_ms": autogaze_total_latency_ms,
        "autogaze_model_forward_latency_ms": autogaze_model_forward_latency_ms,
        "autogaze_model_forward_status": "measured" if autogaze_model_forward_latency_ms is not None else "not_run",
        "autogaze_model_forward_reason": (
            None
            if autogaze_model_forward_latency_ms is not None
            else next(
                (
                    item.get("autogaze_model_forward_reason")
                    for item in aggregate["per_window_metrics"]
                    if item.get("autogaze_model_forward_reason")
                ),
                real_stub_blocked_status,
            )
        ),
        "autogaze_model_forward_call_count": _sum_optional_count(aggregate["per_window_metrics"], "autogaze_model_forward_call_count"),
        "autogaze_model_forward_processed_frame_count": _sum_optional_count(
            aggregate["per_window_metrics"],
            "autogaze_model_forward_processed_frame_count",
        ),
        "autogaze_result_build_latency_ms": _sum_optional_metric(aggregate["per_window_metrics"], "autogaze_result_build_latency_ms"),
        "autogaze_stage_latency_ms": autogaze_stage_latency_ms,
        "autogaze_non_forward_latency_ms": autogaze_non_forward_latency_ms,
        "mllm_generation_latency_ms": sum(
            float(item["mllm_generation_latency_ms"] or 0.0) for item in aggregate["per_window_metrics"]
        ),
        "mllm_processor_latency_ms": _sum_optional_metric(aggregate["per_window_metrics"], "mllm_processor_latency_ms"),
        "mllm_input_move_latency_ms": _sum_optional_metric(aggregate["per_window_metrics"], "mllm_input_move_latency_ms"),
        "mllm_model_generate_latency_ms": _sum_optional_metric(aggregate["per_window_metrics"], "mllm_model_generate_latency_ms"),
        "mllm_generation_timed_scope": next(
            (
                item.get("mllm_generation_timed_scope")
                for item in aggregate["per_window_metrics"]
                if item.get("mllm_generation_timed_scope")
            ),
            None,
        ),
        "nvila_processor_internal_autogaze_timing_status": next(
            (
                item.get("nvila_processor_internal_autogaze_timing_status")
                for item in aggregate["per_window_metrics"]
                if item.get("nvila_processor_internal_autogaze_timing_status")
            ),
            None,
        ),
        "visualization_write_latency_ms": visualization_latency_ms,
        "visualization_latency_ms": visualization_latency_ms,
        "video_writer_latency_ms": visualization_latency_ms,
        "module_processing_latency_ms": total_latency_ms,
        "wall_clock_latency_ms": wall_clock_latency_ms,
        "end_to_end_latency_ms": total_latency_ms,
        "warmup_runs": warmup_runs,
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "memory_unavailable": not torch.cuda.is_available(),
        "output_text": output_text,
        "skipped_stages": skipped_stages,
        "skipped_windows": [],
        "failure_reason": skipped_stages[0]["reason"] if skipped_stages else None,
        "encoder_side_acceleration_claimed": False,
        "per_window_metrics": aggregate["per_window_metrics"],
    }


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
    autogaze_stage_latency_ms = float(gaze.runtime_metadata.get("autogaze_stage_latency_ms") or gaze.latency_ms or 0.0)
    autogaze_preprocessing_latency_ms = float(preprocessing_latency_ms or 0.0)
    autogaze_total_latency_ms = autogaze_preprocessing_latency_ms + autogaze_stage_latency_ms
    model_forward_latency_ms = gaze.runtime_metadata.get("autogaze_model_forward_latency_ms")
    source_frame_count = _source_frame_count(prepared.frame_records)
    processed_frame_count = len(prepared.frame_records)
    nvila_hd = nvila_hd_effective_settings(cfg)
    memory_metrics = prepared_video_memory_metrics(prepared)
    return {
        "mode": mode,
        "config_path": cfg.get("_config_path"),
        "video_path": video_path,
        "query_text": query_text,
        "frame_selection_mode": prepared.frame_selection.mode,
        "frame_interval": prepared.frame_selection.frame_interval,
        "number_of_frames": processed_frame_count,
        "number_of_processed_frames": processed_frame_count,
        "number_of_source_frames": source_frame_count,
        **memory_metrics,
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
        "requested_runtime_dtype": gaze.runtime_metadata.get("requested_dtype"),
        "autogaze_dtype": gaze.runtime_metadata.get("autogaze_execution_dtype"),
        "autogaze_forced_float32": gaze.runtime_metadata.get("autogaze_forced_float32"),
        "gaze_ratio": gaze.runtime_metadata.get("gaze_ratio"),
        "task_loss_requirement": gaze.runtime_metadata.get("task_loss_requirement"),
        "nvila_hd": nvila_hd,
        "num_video_frames": nvila_hd.get("num_video_frames"),
        "num_video_frames_thumbnail": nvila_hd.get("num_video_frames_thumbnail"),
        "max_tiles_video": nvila_hd.get("max_tiles_video"),
        "effective_gazing_ratio_tile": nvila_hd.get("gazing_ratio_tile"),
        "effective_task_loss_requirement_tile": nvila_hd.get("task_loss_requirement_tile"),
        "effective_gazing_ratio_thumbnail": nvila_hd.get("gazing_ratio_thumbnail"),
        "effective_task_loss_requirement_thumbnail": nvila_hd.get("task_loss_requirement_thumbnail"),
        "max_batch_size_autogaze": nvila_hd.get("max_batch_size_autogaze"),
        "max_batch_size_siglip": nvila_hd.get("max_batch_size_siglip"),
        "number_of_tiles_processed": _prepared_tile_count(prepared, nvila_hd),
        "number_of_thumbnail_frames_processed": nvila_hd.get("num_video_frames_thumbnail"),
        "original_token_count": gaze.original_token_count,
        "selected_token_count": gaze.selected_token_count,
        "token_reduction_ratio": gaze.token_reduction_ratio,
        "full_processed_visual_token_count": gaze.original_token_count,
        "autogaze_selected_visual_token_count": gaze.selected_token_count,
        "estimated_visual_token_savings_ratio": gaze.token_reduction_ratio,
        "selected_patches_per_frame": [item["selected_token_count"] for item in gaze.per_frame],
        "selected_patches_per_scale": _sum_scale_counts(gaze.per_frame),
        "preprocessing_latency_ms": preprocessing_latency_ms,
        "autogaze_preprocessing_latency_ms": preprocessing_latency_ms,
        "autogaze_latency_ms": autogaze_total_latency_ms,
        "autogaze_latency_includes_preprocessing": True,
        "autogaze_latency_scope": "preprocessing_plus_autogaze_stage_over_processed_frames",
        "autogaze_latency_source_frame_count": source_frame_count,
        "autogaze_latency_processed_frame_count": processed_frame_count,
        "autogaze_latency_per_source_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, source_frame_count),
        "autogaze_latency_per_processed_frame_ms": _safe_latency_per_item(autogaze_total_latency_ms, processed_frame_count),
        "autogaze_preprocessing_latency_per_processed_frame_ms": _safe_latency_per_item(
            autogaze_preprocessing_latency_ms,
            processed_frame_count,
        ),
        "autogaze_stage_latency_per_processed_frame_ms": _safe_latency_per_item(
            autogaze_stage_latency_ms,
            processed_frame_count,
        ),
        "autogaze_model_forward_latency_ms": model_forward_latency_ms,
        "autogaze_model_forward_status": "measured" if model_forward_latency_ms is not None else "not_run",
        "autogaze_model_forward_reason": None if model_forward_latency_ms is not None else (gaze.reason or gaze.status),
        "autogaze_non_forward_latency_ms": autogaze_stage_latency_ms - float(model_forward_latency_ms or 0.0),
        "autogaze_model_forward_call_count": gaze.runtime_metadata.get("autogaze_model_forward_call_count"),
        "autogaze_model_forward_micro_batch_size": gaze.runtime_metadata.get("autogaze_model_forward_micro_batch_size"),
        "autogaze_model_forward_batch_latencies_ms": gaze.runtime_metadata.get("autogaze_model_forward_batch_latencies_ms"),
        "autogaze_model_forward_batch_ranges": gaze.runtime_metadata.get("autogaze_model_forward_batch_ranges"),
        "autogaze_model_forward_processed_clip_count": gaze.runtime_metadata.get("autogaze_model_forward_processed_clip_count"),
        "autogaze_model_forward_processed_frame_count": gaze.runtime_metadata.get("autogaze_model_forward_processed_frame_count"),
        "autogaze_result_build_latency_ms": gaze.runtime_metadata.get("autogaze_result_build_latency_ms"),
        "autogaze_stage_latency_ms": autogaze_stage_latency_ms,
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
        "encoder_side_acceleration_claimed": False,
    }


def _source_frame_count(frame_records: list[Mapping[str, Any]]) -> int:
    return len({int(item.get("source_frame_index", idx)) for idx, item in enumerate(frame_records) if not bool(item.get("is_padded", False))})


def _sum_optional_metric(items: list[Mapping[str, Any]], key: str) -> float | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    return sum(float(value or 0.0) for value in values)


def _sum_optional_count(items: list[Mapping[str, Any]], key: str) -> int | None:
    values = [item.get(key) for item in items]
    if not any(value is not None for value in values):
        return None
    return sum(int(value or 0) for value in values)


def _sum_values_as_int(values: list[Any]) -> int | None:
    if not any(value is not None for value in values):
        return None
    return sum(int(value or 0) for value in values)


def _max_optional_count(values: list[Any]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _max_optional_float(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _safe_latency_per_item(latency_ms: float | None, count: int | None) -> float | None:
    if latency_ms is None or count is None or int(count) <= 0:
        return None
    return float(latency_ms) / float(count)


def _prepared_tile_count(prepared: PreparedVideo, nvila_hd: Mapping[str, Any]) -> int | None:
    max_tiles = nvila_hd.get("max_tiles_video")
    if max_tiles is None:
        return None
    chops = _spatial_chops_per_window(prepared.chop_metadata)
    if chops:
        return sum(min(int(max_tiles), int(count)) for count in chops.values())
    return min(int(max_tiles), 1)


def _streaming_tile_count(aggregate: Mapping[str, Any], nvila_hd: Mapping[str, Any]) -> int | None:
    max_tiles = nvila_hd.get("max_tiles_video")
    if max_tiles is None:
        return None
    chop_windows = aggregate.get("chop_windows") or []
    if chop_windows:
        return sum(min(int(max_tiles), int(window.get("spatial_chop_count", 0))) for window in chop_windows)
    return min(int(max_tiles), max(1, len(aggregate.get("windows") or [])))


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
