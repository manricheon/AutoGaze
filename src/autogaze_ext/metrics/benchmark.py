from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NA = "N/A"


@dataclass(frozen=True)
class BenchmarkResult:
    experiment_id: str
    task_type: str
    device: str
    precision: str
    peak_vram_mb: float | str
    inference_latency_ms: float
    throughput_videos_per_sec: float
    fps: float
    visual_token_count_before_autogaze: int
    visual_token_count_after_autogaze: int
    token_reduction_ratio: float
    selected_patches_per_frame: int | float
    selected_patches_per_scale: dict[str, int] | str
    autogaze_latency_ms: float | str = NA
    vit_latency_ms: float | str = NA
    mllm_prefill_latency_ms: float | str = NA
    mllm_decode_latency_ms: float | str = NA
    end_to_end_latency_ms: float | str = NA
    autogaze: str | None = None
    vision_encoder_type: str | None = None
    mllm_type: str | None = None
    integration_mode: str | None = None
    input_frame_count: int | None = None
    input_resolution: str | None = None
    dummy_task_metric: float | None = None
    acceleration_type_note: str | None = None
    stub_status: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_throughput(batch_size: int, latency_ms: float) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if latency_ms <= 0:
        raise ValueError("latency_ms must be > 0")
    return float(batch_size) / (latency_ms / 1000.0)


def compute_fps(batch_size: int, frames_per_video: int, latency_ms: float) -> float:
    if frames_per_video <= 0:
        raise ValueError("frames_per_video must be > 0")
    return compute_throughput(batch_size, latency_ms) * float(frames_per_video)


def write_json_result(result: BenchmarkResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    return output_path


def write_csv_results(results: list[BenchmarkResult], path: str | Path) -> Path:
    if not results:
        raise ValueError("results must not be empty")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if isinstance(row.get("selected_patches_per_scale"), dict):
                row["selected_patches_per_scale"] = json.dumps(row["selected_patches_per_scale"], sort_keys=True)
            if isinstance(row.get("metadata"), dict):
                row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            writer.writerow(row)
    return output_path
