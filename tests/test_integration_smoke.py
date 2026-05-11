from __future__ import annotations

import json

from autogaze_ext.pipeline.integration_smoke import run_integration_smoke


def test_integration_smoke_runs_dummy_modules_and_writes_artifacts(tmp_path) -> None:
    artifacts = run_integration_smoke(output_root=tmp_path / "smoke", batch_size=1)

    assert artifacts.output_root.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.manifest_path.exists()

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["warning"] == "dummy/stub integration smoke only; not a real benchmark result"
    assert summary["no_real_model_checkpoint_required"] is True
    assert summary["external_downloads"] is False
    assert summary["dummy_video_vqa"]["generated_text"] == ["dummy"]
    assert summary["dummy_action_recognition"]["logits_shape"] == [1, 4]
    assert sorted(summary["canonical_experiments"]) == ["A0", "A1", "A2", "A3"]

    assert len(artifacts.benchmark_json_paths) == 4
    assert len(artifacts.benchmark_csv_paths) == 4
    for path in artifacts.benchmark_json_paths:
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["metadata"]["warning"] == "dummy/stub benchmark only; not a reproduction result"
        assert data["metadata"]["dummy_execution"]["real_checkpoints_loaded"] is False
    for path in artifacts.benchmark_csv_paths:
        assert path.exists()
        assert "dummy/stub benchmark only" in path.read_text(encoding="utf-8")

    assert artifacts.visualization_paths
    for path in artifacts.visualization_paths:
        assert path.exists()
    required_dirs = {
        artifacts.output_root / "integration_smoke" / "visualizations" / "autogaze_only",
        artifacts.output_root / "integration_smoke" / "visualizations" / "full_pipeline",
        artifacts.output_root / "integration_smoke" / "visualizations" / "video_vqa",
        artifacts.output_root / "integration_smoke" / "visualizations" / "action_recognition",
    }
    assert all(path.exists() and path.is_dir() for path in required_dirs)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["metadata"]["smoke_test"] == "integration_dummy_only"
    assert manifest["metadata"]["real_checkpoints_loaded"] is False
    assert manifest["metadata"]["external_downloads"] is False
    assert "device_information" in manifest
    assert "package_versions" in manifest

