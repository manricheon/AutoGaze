from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autogaze_ext.investigation.canonical_smoke_inference import SmokeInferenceReport, StageReport
from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.pipeline.tiny_real_benchmark import run_tiny_real_benchmark


ROOT = Path(__file__).resolve().parents[1]


def _fake_report(**kwargs: Any) -> SmokeInferenceReport:
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "logs" / "inference_summary.json"
    scales_path = output_dir / "autogaze" / "selected_scales.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    scales_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")
    scales_path.write_text(json.dumps({"selected_scales": [32, 64]}), encoding="utf-8")

    mode = kwargs["mode"]
    experiment = kwargs["experiment"]
    autogaze_enabled = experiment == "A2_real"
    full_pipeline = mode == "full_pipeline"
    before = 530 if autogaze_enabled else 392
    after = 6 if autogaze_enabled else 392
    stages = [
        StageReport(
            name="autogaze",
            status="passed" if autogaze_enabled else "disabled_full_token_path",
            latency_ms=10.0 if autogaze_enabled else None,
            output_shape=[1, 20] if autogaze_enabled else [1, 392],
            details={"autogaze_enabled": autogaze_enabled},
        )
    ]
    skipped = []
    vision_shape = None
    mllm_shape = None
    if full_pipeline:
        vision_shape = [1, 20 if autogaze_enabled else 392, 768]
        mllm_shape = vision_shape
        stages.append(
            StageReport(
                name="vision_encoder",
                status="passed",
                latency_ms=20.0,
                output_shape=vision_shape,
            )
        )
        stages.append(
            StageReport(
                name="mllm",
                status="skipped",
                skipped_reason="query text was accepted, but MLLM generation was skipped because --allow-mllm-load was not set",
            )
        )
        skipped.append(
            {
                "stage": "mllm",
                "reason": "query text was accepted, but MLLM generation was skipped because --allow-mllm-load was not set",
            }
        )

    return SmokeInferenceReport(
        experiment_id=experiment,
        mode=mode,
        status="partial" if skipped else "passed",
        quick_start_found=True,
        quick_start_path=str(ROOT / "QUICK_START.md"),
        quick_start_reference_found=True,
        quick_start_reference_path=str(ROOT / "docs" / "QUICK_START_reference.md"),
        input_shape=[1, int(kwargs["num_frames"]), 3, int(kwargs["resolution"]), int(kwargs["resolution"])],
        frame_count=int(kwargs["num_frames"]),
        original_resolution=[int(kwargs["resolution"]), int(kwargs["resolution"])],
        target_resolution=int(kwargs["resolution"]),
        processed_resolution=[int(kwargs["resolution"]), int(kwargs["resolution"])],
        scaling={
            "status": "not_requested",
            "quick_start_behavior": "default_224x224_or_requested_resolution",
        },
        autogaze_enabled=autogaze_enabled,
        query_text=kwargs.get("query_text"),
        selected_token_count=after,
        original_visual_token_count=before,
        vision_feature_shape=vision_shape,
        mllm_input_shape=mllm_shape,
        output_text=None,
        output_dir=str(output_dir),
        visualization_dir=None,
        latency_ms=100.0,
        peak_vram_mb="N/A",
        stages=stages,
        skipped_stages=skipped,
        artifacts={
            "inference_summary": str(summary_path),
            "selected_scales": str(scales_path),
        },
        device=str(kwargs["device"]),
        effective_device=str(kwargs["device"]),
        dtype=str(kwargs["dtype"]),
    )


def test_tiny_real_benchmark_presets_load() -> None:
    a1 = load_config(ROOT / "configs", "experiment/A1_real")
    a2 = load_config(ROOT / "configs", "experiment/A2_real")
    assert a1.experiment.id == "A1_real"
    assert a2.experiment.id == "A2_real"

    for preset in ["tiny_a1_real", "tiny_a2_real"]:
        path = ROOT / "configs" / "benchmark" / f"{preset}.yaml"
        assert path.exists()


def test_tiny_real_benchmark_writes_json_csv_and_manifest(tmp_path: Path) -> None:
    artifact = run_tiny_real_benchmark(
        config_name="tiny_a2_real",
        mode="full_pipeline",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        warmup_iterations=1,
        benchmark_iterations=3,
        inference_fn=_fake_report,
    )

    assert artifact.json_path.exists()
    assert artifact.csv_path.exists()
    assert artifact.manifest_path.exists()
    assert len(artifact.iteration_summary_paths) == 3

    data = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    assert data["experiment_id"] == "A2_real"
    assert data["visual_token_count_before_autogaze"] == 530
    assert data["visual_token_count_after_autogaze"] == 6
    assert data["token_reduction_ratio"] > 0.98
    assert data["autogaze_latency_ms"] == 10.0
    assert data["vit_latency_ms"] == 20.0
    assert data["metadata"]["benchmark_mode"] == "full_pipeline"
    assert data["metadata"]["executed_stages"] == ["autogaze", "vision_encoder"]
    assert data["metadata"]["skipped_stages"][0]["stage"] == "mllm"
    assert data["metadata"]["model_paths"]["autogaze"]["checkpoint"] == "weights/AutoGaze"


def test_tiny_real_benchmark_autogaze_off_full_token_counts(tmp_path: Path) -> None:
    artifact = run_tiny_real_benchmark(
        config_name="tiny_a1_real",
        mode="autogaze_only",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        warmup_iterations=0,
        benchmark_iterations=1,
        inference_fn=_fake_report,
    )
    data = json.loads(artifact.json_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "A1_real"
    assert data["autogaze"] == "OFF"
    assert data["visual_token_count_before_autogaze"] == 392
    assert data["visual_token_count_after_autogaze"] == 392
    assert data["token_reduction_ratio"] == 0.0
    assert data["metadata"]["executed_stages"] == ["autogaze"]
