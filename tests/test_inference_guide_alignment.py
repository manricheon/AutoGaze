from __future__ import annotations

from pathlib import Path

from autogaze_ext.investigation import locate_quick_start
from autogaze_ext.pipeline.runner import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_quick_start_can_be_located_from_configured_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "original_autogaze"
    source_dir.mkdir()
    quick_start = source_dir / "QUICK_START.md"
    quick_start.write_text("# Quick Start\n", encoding="utf-8")

    located = locate_quick_start(source_dir, repo_root=tmp_path)

    assert located.path == quick_start.resolve()
    assert located.source == "configured_path"


def test_inference_guide_and_quick_start_reference_exist() -> None:
    inference_guide = ROOT / "docs" / "INFERENCE_GUIDE.md"
    quick_start_reference = ROOT / "docs" / "QUICK_START_reference.md"

    assert inference_guide.exists()
    assert quick_start_reference.exists()

    guide_text = inference_guide.read_text(encoding="utf-8")
    for section in [
        "AutoGaze-Only Inference",
        "Full Pipeline Inference",
        "Query Text Handling",
        "Resolution Scaling",
        "Current Limitations",
    ]:
        assert section in guide_text
    assert "stub-only / future work" in guide_text


def test_quick_start_reference_contains_required_extractions() -> None:
    text = (ROOT / "docs" / "QUICK_START_reference.md").read_text(encoding="utf-8")

    for expected in [
        "AutoGaze.from_pretrained",
        "bfshi/AutoGaze",
        "SiglipVisionModel",
        "gazing_ratio",
        "task_loss_requirement",
        "target_scales",
        "target_patch_size",
        "Query text or prompt arguments are not present",
        "No shell CLI is described",
    ]:
        assert expected in text


def test_a1_a2_real_configs_include_quick_start_alignment_fields() -> None:
    for config_name in ["experiment/A1_real", "experiment/A2_real"]:
        cfg = load_config(ROOT / "configs", config_name)

        assert cfg.inference.input_resolution == 224
        assert cfg.inference.frame_count == 16
        assert "target_scales" in cfg.inference
        assert "target_patch_size" in cfg.inference
        assert "checkpoint_root" in cfg.inference
        assert "inference_script_path" in cfg.inference
        assert "query_text" in cfg.inference
        assert "output_dir" in cfg.inference
        assert "visualization_dir" in cfg.inference

    a2 = load_config(ROOT / "configs", "experiment/A2_real")
    assert a2.model.autogaze.input_resolution == 224
    assert a2.model.autogaze.frame_count == 16
    assert a2.model.autogaze.original_cli_args.gazing_ratio == 0.75
    assert a2.model.autogaze.original_cli_args.task_loss_requirement == 0.7

    a1 = load_config(ROOT / "configs", "experiment/A1_real")
    assert a1.model.vision_encoder.original_cli_args.gazing_info_argument == "gazing_info"
    assert a1.model.vision_encoder.original_cli_args.high_resolution_example.target_patch_size == 14
