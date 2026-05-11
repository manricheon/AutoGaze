from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from autogaze_ext.metrics import BenchmarkResult, compute_fps, compute_throughput, write_csv_results, write_json_result
from autogaze_ext.profiling import LatencyLogger, MemoryTracker, measure_latency_ms, summarize_tokens, token_reduction_ratio


def test_token_reduction_calculation() -> None:
    assert token_reduction_ratio(100, 25) == 0.75
    summary = summarize_tokens(before=100, after=25, frame_count=5, selected_scales=[224, 224, 448])

    assert summary.visual_token_count_before_autogaze == 100
    assert summary.visual_token_count_after_autogaze == 25
    assert summary.selected_patches_per_frame == 5
    assert summary.selected_patches_per_scale == {"224": 2, "448": 1}


def test_latency_logger_and_warmup() -> None:
    calls = {"count": 0}

    def work() -> int:
        calls["count"] += 1
        return calls["count"]

    result, latency_ms = measure_latency_ms(work, warmup_iters=2, repeat=3)
    logger = LatencyLogger()
    with logger.measure("noop"):
        _ = 1 + 1

    assert result == 5
    assert calls["count"] == 5
    assert latency_ms >= 0
    assert logger.measurements_ms["noop"] >= 0


def test_result_serialization(tmp_path: Path) -> None:
    result = BenchmarkResult(
        experiment_id="dummy",
        task_type="video_vqa",
        device="cpu",
        precision="float32",
        peak_vram_mb="N/A",
        inference_latency_ms=10.0,
        throughput_videos_per_sec=compute_throughput(batch_size=2, latency_ms=10.0),
        fps=compute_fps(batch_size=2, frames_per_video=4, latency_ms=10.0),
        visual_token_count_before_autogaze=100,
        visual_token_count_after_autogaze=100,
        token_reduction_ratio=0.0,
        selected_patches_per_frame=25,
        selected_patches_per_scale="N/A",
        end_to_end_latency_ms=10.0,
        metadata={"source": "test"},
    )

    json_path = write_json_result(result, tmp_path / "benchmark.json")
    csv_path = write_csv_results([result], tmp_path / "benchmark.csv")

    assert json.loads(json_path.read_text())["experiment_id"] == "dummy"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment_id"] == "dummy"
    assert rows[0]["metadata"] == '{"source": "test"}'


def test_cpu_memory_fallback_records_na() -> None:
    snapshot = MemoryTracker("cpu").snapshot()

    assert snapshot.device_type == "cpu"
    assert snapshot.peak_vram_mb == "N/A"
    assert snapshot.allocated_mb == "N/A"


def test_mps_memory_fallback_records_na_where_possible() -> None:
    snapshot = MemoryTracker("mps").snapshot()

    assert snapshot.device_type == "mps"
    assert snapshot.peak_vram_mb == "N/A"
    assert snapshot.allocated_mb == "N/A"


def test_cuda_memory_snapshot_shape_when_available() -> None:
    if not torch.cuda.is_available():
        return
    tracker = MemoryTracker("cuda")
    tracker.reset_peak()
    snapshot = tracker.snapshot()
    assert snapshot.device_type == "cuda"
    assert isinstance(snapshot.peak_vram_mb, float)
