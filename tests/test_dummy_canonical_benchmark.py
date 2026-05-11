from __future__ import annotations

import json

import pytest

from autogaze_ext.pipeline.benchmark import run_dummy_benchmark


@pytest.mark.parametrize("experiment_id", ["A0", "A1", "A2", "A3"])
def test_dummy_canonical_benchmark_completes(experiment_id: str, tmp_path) -> None:
    artifact = run_dummy_benchmark(experiment_id, output_dir=tmp_path, batch_size=2)

    assert artifact.json_path.exists()
    assert artifact.csv_path.exists()
    data = json.loads(artifact.json_path.read_text())
    assert data["experiment_id"] == experiment_id
    assert data["metadata"]["warning"] == "dummy/stub benchmark only; not a reproduction result"
    assert data["visual_token_count_before_autogaze"] == data["visual_token_count_after_autogaze"]
    assert data["token_reduction_ratio"] == 0.0
    assert data["dummy_task_metric"] == 1.0
    assert data["peak_vram_mb"] == "N/A"


def test_a3_dummy_benchmark_keeps_experimental_stub_status(tmp_path) -> None:
    artifact = run_dummy_benchmark("A3", output_dir=tmp_path, batch_size=2)
    data = json.loads(artifact.json_path.read_text())

    assert data["autogaze"] == "ON"
    assert data["vision_encoder_type"] == "vanilla_siglip"
    assert data["stub_status"] == "stubbed_autogaze_on_no_real_selector_no_token_reduction"
    assert "stubbed AutoGaze ON" in data["acceleration_type_note"]
