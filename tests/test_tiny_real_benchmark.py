from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autogaze_ext.investigation.canonical_smoke_inference import SmokeInferenceReport, StageReport
from autogaze_ext.pipeline.runner import load_config
from autogaze_ext.pipeline.tiny_real_benchmark import (
    CANONICAL_BENCHMARK_CONFIGS,
    run_benchmark_dry_run,
    run_tiny_real_benchmark,
)


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
    assert artifact.stage_latency_path.exists()
    assert artifact.token_counts_path.exists()
    assert artifact.skipped_stages_path.exists()
    assert len(artifact.iteration_summary_paths) == 3

    data = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    stage_rows = json.loads(artifact.stage_latency_path.read_text(encoding="utf-8"))
    token_report = json.loads(artifact.token_counts_path.read_text(encoding="utf-8"))
    skipped_report = json.loads(artifact.skipped_stages_path.read_text(encoding="utf-8"))

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
    assert {row["stage"] for row in stage_rows} == {"autogaze", "vision_encoder", "mllm"}
    assert token_report["visual_token_count_before_autogaze"] == 530
    assert token_report["visual_token_count_after_autogaze"] == 6
    assert skipped_report["skipped_stages"][0]["stage"] == "mllm"


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


def test_tiny_real_benchmark_accepts_conservative_axis_overrides(tmp_path: Path) -> None:
    artifact = run_tiny_real_benchmark(
        config_name="canonical_a2_small",
        mode="full_pipeline",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        warmup_iterations=0,
        benchmark_iterations=1,
        batch_size=1,
        num_frames=2,
        resolution=224,
        max_new_tokens=1,
        inference_fn=_fake_report,
    )
    data = json.loads(artifact.json_path.read_text(encoding="utf-8"))

    assert data["input_frame_count"] == 2
    assert data["metadata"]["benchmark_iterations"] == 1
    assert data["metadata"]["batch_size"] == 1
    assert data["input_resolution"] == "224x224"


def test_canonical_benchmark_dry_run_validates_all_presets(tmp_path: Path) -> None:
    required_paths = [
        ROOT / "weights" / "AutoGaze",
        ROOT / "weights" / "siglip2-base-patch16-224",
        ROOT / "weights" / "NVILA-8B-HD-Video",
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("canonical real-path local weights are not available")

    for config_name in CANONICAL_BENCHMARK_CONFIGS:
        report = run_benchmark_dry_run(
            config_name=config_name,
            mode="full_pipeline",
            config_dir=ROOT / "configs",
            output_dir=tmp_path / config_name,
        )
        data = json.loads(report.report_path.read_text(encoding="utf-8"))

        assert report.safe_to_execute is True
        assert report.failures == []
        assert data["safe_to_execute"] is True
        assert data["no_heavy_model_instantiation"] is True
        assert data["no_checkpoint_tensor_loading"] is True
        assert data["no_inference"] is True
        check_names = {check["name"] for check in data["checks"]}
        assert check_names >= {
            "config_loading",
            "benchmark_axis_values",
            "quick_start_scaling_compatibility",
            "siglip_vit_module_availability",
            "nvila_hd_video_module_availability",
            "nvila_hd_video_processor_module_availability",
            "nvila_hd_video_guide_compatibility",
            "output_directory_writability",
        }
        if data["experiment_id"] == "A2_real":
            assert "autogaze_module_availability" in check_names


def test_canonical_benchmark_dry_run_reports_invalid_dtype(tmp_path: Path) -> None:
    report = run_benchmark_dry_run(
        config_name="canonical_a1_small",
        mode="full_pipeline",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        dtype="float8",
    )

    assert report.safe_to_execute is False
    assert any("unsupported dtype" in failure for failure in report.failures)
    assert report.report_path.exists()


def test_canonical_benchmark_dry_run_validates_local_video_path(tmp_path: Path) -> None:
    report = run_benchmark_dry_run(
        config_name="canonical_a2_small",
        mode="autogaze_only",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        video_path=ROOT / "assets" / "example_input.mp4",
    )

    checks = {check.name: check for check in report.checks}
    assert checks["video_input_availability"].status == "passed"
    assert checks["video_input_availability"].details["source"] == "local_video"


def test_canonical_benchmark_dry_run_accepts_axis_overrides(tmp_path: Path) -> None:
    report = run_benchmark_dry_run(
        config_name="canonical_a2_small",
        mode="full_pipeline",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        num_frames=2,
        resolution=224,
        max_new_tokens=1,
    )

    checks = {check.name: check for check in report.checks}
    assert report.safe_to_execute is True
    assert checks["benchmark_axis_values"].details["num_frames"] == 2
    assert checks["benchmark_axis_values"].details["max_new_tokens"] == 1


def test_canonical_medium_dry_run_rejects_non_policy_resolution(tmp_path: Path) -> None:
    report = run_benchmark_dry_run(
        config_name="canonical_a2_medium",
        mode="full_pipeline",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        resolution=448,
    )

    checks = {check.name: check for check in report.checks}
    assert report.safe_to_execute is False
    assert "largest QUICK_START target scale" in str(checks["quick_start_scaling_compatibility"].message)


def test_canonical_dry_run_accepts_spatio_temporal_policy(tmp_path: Path) -> None:
    report = run_benchmark_dry_run(
        config_name="canonical_a2_small",
        mode="autogaze_only",
        config_dir=ROOT / "configs",
        output_dir=tmp_path,
        scale_resolution="quick_start_spatio_temporal_224",
    )

    checks = {check.name: check for check in report.checks}
    scaling = checks["quick_start_scaling_compatibility"]
    assert scaling.status == "passed"
    assert scaling.details["resolved_policy"]["mode"] == "spatio_temporal"
    assert scaling.details["resolved_policy"]["spatial_tile_size"] == 224
