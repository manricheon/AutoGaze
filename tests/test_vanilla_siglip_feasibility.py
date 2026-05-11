from __future__ import annotations

import types
from pathlib import Path

import pytest

from autogaze_ext.investigation import check_vanilla_siglip_feasibility
from autogaze_ext.pipeline.runner import load_config


ROOT = Path(__file__).resolve().parents[1]


def _fake_transformers_import(*, include_model: bool = True, include_processor: bool = True):
    module = types.ModuleType("transformers")
    if include_model:
        module.AutoModel = object
    if include_processor:
        module.AutoImageProcessor = object

    def import_module(name: str):
        if name == "transformers":
            return module
        raise ModuleNotFoundError(name)

    return import_module


def test_a0_a3_real_configs_include_vanilla_siglip_feasibility_fields() -> None:
    for config_name in ["experiment/A0_real", "experiment/A3_real"]:
        cfg = load_config(ROOT / "configs", config_name)

        assert cfg.model.vision_encoder.type == "vanilla_siglip"
        assert cfg.model.vision_encoder.module_path == "transformers"
        assert cfg.model.vision_encoder.class_or_factory == "AutoModel"
        assert cfg.model.vision_encoder.processor_class_or_factory == "AutoImageProcessor"
        assert cfg.model.vision_encoder.input_resolution == 224
        assert cfg.model.vision_encoder.patch_size == 16
        assert cfg.model.vision_encoder.output_dim == 768
        assert "scale_resolution" in cfg.model.vision_encoder
        assert "target_scales" in cfg.model.vision_encoder
        assert cfg.inference.inference_script_path == "scripts/check_vanilla_siglip_feasibility.py"

    a0 = load_config(ROOT / "configs", "experiment/A0_real")
    a3 = load_config(ROOT / "configs", "experiment/A3_real")
    assert a0.model.autogaze.enabled is False
    assert a3.model.autogaze.enabled is True
    assert "experimental" in a3.experiment.integration_mode


def test_a0_feasibility_reports_vision_ready_but_nvila_shape_mismatch() -> None:
    report = check_vanilla_siglip_feasibility(
        experiment="A0_real",
        import_module_fn=_fake_transformers_import(),
    )

    assert report.module_import.ready is True
    assert report.processor_import.ready is True
    assert report.patch_grid.patch_grid == (14, 14)
    assert report.patch_grid.visual_tokens_per_frame == 196
    assert report.output_dim == 768
    assert report.nvila_visual_input.expected_visual_dim == 1152
    assert report.nvila_visual_input.expected_tokens_per_frame == 256
    assert report.nvila_visual_input.compatible is False
    assert any("feature dimension mismatch" in item for item in report.nvila_visual_input.issues)
    assert any("token count mismatch" in item for item in report.nvila_visual_input.issues)
    assert report.ready_for_vision_construction_smoke is True
    assert report.ready_for_full_pipeline_construction_smoke is False
    assert report.modes[0].mode == "full_token_vanilla_siglip_baseline"
    assert report.modes[0].true_encoder_side_acceleration is False


def test_a3_feasibility_marks_autogaze_vanilla_modes_experimental() -> None:
    report = check_vanilla_siglip_feasibility(
        experiment="A3_real",
        import_module_fn=_fake_transformers_import(),
    )

    modes = {mode.mode: mode for mode in report.modes}

    assert report.autogaze_enabled is True
    assert report.autogaze_patch_indices.requires_adapter is True
    assert report.ready_for_vision_construction_smoke is True
    assert report.ready_for_full_pipeline_construction_smoke is False
    assert report.ready_for_a3_experimental_construction_smoke is False
    assert set(modes) == {
        "input_level_crop_region_reconstruction",
        "post_patch_embedding_token_masking",
        "compact_token_gathering",
    }
    assert modes["input_level_crop_region_reconstruction"].compatibility_only_path is True
    assert modes["post_patch_embedding_token_masking"].status == "incompatible_without_model_hooks"
    assert modes["compact_token_gathering"].downstream_token_reduction_only is True
    assert all(mode.true_encoder_side_acceleration is False for mode in modes.values())
    assert any("patch-index adapter" in item for item in report.blockers)


def test_vanilla_siglip_missing_hf_factory_blocks_construction_smoke() -> None:
    report = check_vanilla_siglip_feasibility(
        experiment="A0_real",
        import_module_fn=_fake_transformers_import(include_model=False),
    )

    assert report.module_import.module_available is True
    assert report.module_import.class_or_factory_exists is False
    assert report.ready_for_vision_construction_smoke is False
    assert any("does not provide" in item for item in report.blockers)


def test_vanilla_siglip_feasibility_rejects_non_vanilla_real_experiment() -> None:
    with pytest.raises(ValueError, match="A0_real, A3_real"):
        check_vanilla_siglip_feasibility(experiment="A2_real")
