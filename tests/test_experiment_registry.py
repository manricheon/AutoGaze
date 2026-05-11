from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from autogaze_ext.pipeline.experiment_registry import (
    EXPERIMENT_REGISTRY,
    get_experiment_spec,
    validate_experiment_config,
)
from autogaze_ext.pipeline.runner import load_config, summarize_config


def test_registry_contains_canonical_ablation() -> None:
    assert sorted(EXPERIMENT_REGISTRY) == ["A0", "A1", "A2", "A3"]
    assert get_experiment_spec("A0").vision_encoder_type == "vanilla_siglip"
    assert get_experiment_spec("A1").vision_encoder_type == "modified_siglip"
    assert get_experiment_spec("A2").autogaze_enabled is True
    assert get_experiment_spec("A3").compatibility_status.endswith("not_validated")


@pytest.mark.parametrize(
    ("experiment_id", "autogaze", "vision", "integration_mode"),
    [
        ("A0", False, "vanilla_siglip", "full_token_baseline"),
        ("A1", False, "modified_siglip", "modified_siglip_full_token_baseline"),
        ("A2", True, "modified_siglip", "original_autogaze_modified_siglip_nvila"),
        ("A3", True, "vanilla_siglip", "experimental_autogaze_vanilla_siglip_nvila"),
    ],
)
def test_a0_a3_config_loading_and_registry_resolution(
    experiment_id: str,
    autogaze: bool,
    vision: str,
    integration_mode: str,
) -> None:
    cfg = load_config(config_name=f"experiment/{experiment_id}")
    spec = validate_experiment_config(cfg)
    summary = summarize_config(cfg)

    assert spec.experiment_id == experiment_id
    assert cfg.model.autogaze.enabled is autogaze
    assert cfg.model.vision_encoder.type == vision
    assert cfg.model.mllm.type == "nvila"
    assert cfg.task.type == "video_vqa"
    assert summary["integration mode"] == integration_mode
    assert summary["MLLM config"] == "model/mllm/nvila"


def test_registry_validation_rejects_invalid_wiring() -> None:
    cfg = load_config(config_name="experiment/A0")
    cfg = OmegaConf.merge(cfg, {"model": {"vision_encoder": {"type": "modified_siglip"}}})

    with pytest.raises(ValueError, match="Invalid wiring for experiment A0"):
        validate_experiment_config(cfg)
