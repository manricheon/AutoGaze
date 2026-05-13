from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_poc_autogaze_impact.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("benchmark_poc_autogaze_impact", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_poc_autogaze_impact"] = module
    spec.loader.exec_module(module)
    return module


def test_poc_autogaze_impact_dry_run_writes_plan(tmp_path: Path) -> None:
    module = _load_script_module()

    rc = module.main(["--output-dir", str(tmp_path)])

    assert rc == 0
    plan_path = tmp_path / "benchmark_plan.json"
    commands_path = tmp_path / "commands.sh"
    assert plan_path.exists()
    assert commands_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["status"] == "dry_run_plan"
    assert plan["axes"]["num_frames"] == 16
    configs = {item["config"] for item in plan["commands"]}
    assert configs == {
        "configs/experiment/A1_real.yaml",
        "configs/experiment/A2_real.yaml",
    }
    assert all("--allow-checkpoint-load" not in item["command"] for item in plan["commands"])
    assert "configs/experiment/A1_real.yaml" in commands_path.read_text(encoding="utf-8")
    assert "configs/experiment/A2_real.yaml" in commands_path.read_text(encoding="utf-8")


def test_poc_autogaze_impact_plan_can_enable_checkpoint_loading(tmp_path: Path) -> None:
    module = _load_script_module()

    args = module.make_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--allow-checkpoint-load",
            "--device",
            "mps",
            "--include-visualization",
        ]
    )
    plan = module.build_plan(args)

    assert plan.axes["device"] == "mps"
    assert plan.axes["include_visualization"] is True
    assert any("--allow-checkpoint-load" in item.command for item in plan.commands)
    assert any("--save-overlay-video" in item.command for item in plan.commands)


def test_poc_autogaze_impact_summarizes_existing_metrics(tmp_path: Path) -> None:
    module = _load_script_module()
    a1_logs = tmp_path / "A1_real" / "run_000" / "logs"
    a2_logs = tmp_path / "A2_real" / "run_000" / "logs"
    a1_logs.mkdir(parents=True)
    a2_logs.mkdir(parents=True)
    base = {
        "mode": "full_pipeline",
        "status": "partial",
        "frame_selection_mode": "sample",
        "effective_frame_selection_mode": "sample",
        "number_of_frames": 16,
        "number_of_windows": 1,
        "scaling_mode": "resize",
        "original_resolution": [224, 224],
        "processed_resolution": [224, 224],
        "mllm_decode_latency_ms": "N/A",
        "peak_vram_mb": "N/A",
        "memory_metric_unavailable": True,
        "skipped_stages": [{"stage": "nvila_generation", "reason": "test skip"}],
        "result_label": "real_or_mock_result",
    }
    a1 = {
        **base,
        "experiment_id": "A1_real",
        "original_visual_token_count": 3136,
        "selected_visual_token_count": 3136,
        "token_reduction_ratio": 0.0,
        "autogaze_latency_ms": "N/A",
        "vision_encoder_latency_ms": 30.0,
        "end_to_end_latency_ms": 100.0,
    }
    a2 = {
        **base,
        "experiment_id": "A2_real",
        "original_visual_token_count": 3136,
        "selected_visual_token_count": 512,
        "token_reduction_ratio": 0.8367,
        "autogaze_latency_ms": 12.0,
        "vision_encoder_latency_ms": 20.0,
        "end_to_end_latency_ms": 90.0,
    }
    (a1_logs / "metrics.json").write_text(json.dumps(a1), encoding="utf-8")
    (a2_logs / "metrics.json").write_text(json.dumps(a2), encoding="utf-8")

    summary = module.summarize_existing_outputs(tmp_path)

    assert summary["valid_internal_comparison"] is True
    assert summary["token_reduction_observed"] is True
    assert summary["generation_skipped"] is True
    assert summary["encoder_side_acceleration_claim_allowed"] is False
    assert (tmp_path / "autogaze_impact_summary.json").exists()
    assert (tmp_path / "autogaze_impact_summary.csv").exists()


def test_poc_autogaze_impact_config_loads() -> None:
    path = ROOT / "configs" / "benchmark" / "poc_autogaze_impact_full_pipeline.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    assert isinstance(cfg, dict)
    benchmark = cfg["benchmark"]
    assert benchmark["script"] == "scripts/benchmark_poc_autogaze_impact.py"
    assert benchmark["execution_engine"] == "scripts/poc_nvila_hd_video.py"
    assert benchmark["experiments"]["baseline"] == "configs/experiment/A1_real.yaml"
    assert benchmark["experiments"]["autogaze"] == "configs/experiment/A2_real.yaml"
    assert benchmark["heavy_benchmark"] is False
    assert benchmark["run_by_default"] is False
