from __future__ import annotations

from pathlib import Path

from autogaze_ext.pipeline.runner import load_config, summarize_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config_groups() -> None:
    cfg = load_config(ROOT / "configs")

    assert cfg.experiment.id == "smoke_config"
    assert cfg.model.autogaze.enabled is False
    assert cfg.model.vision_encoder.type == "modified_siglip"
    assert cfg.model.mllm.type == "nvila"
    assert cfg.task.type == "video_vqa"
    assert cfg.runtime.device.type == "cpu"
    assert cfg.runtime.precision.dtype == "float32"
    assert cfg.runtime.huggingface.enabled is False


def test_runner_summary_matches_smoke_config() -> None:
    cfg = load_config(ROOT / "configs")
    summary = summarize_config(cfg)

    assert summary == {
        "experiment ID": "smoke_config",
        "AutoGaze": "OFF",
        "vision encoder type": "modified_siglip",
        "MLLM type": "nvila",
        "task type": "video_vqa",
        "device": "cpu",
        "precision": "float32",
        "Hugging Face mode": None,
    }


def test_real_canonical_a1_a2_configs_load() -> None:
    a1 = load_config(ROOT / "configs", "experiment/A1_real")
    a2 = load_config(ROOT / "configs", "experiment/A2_real")

    assert a1.experiment.id == "A1_real"
    assert a1.model.autogaze.enabled is False
    assert a1.model.vision_encoder.module_path == "autogaze.vision_encoders.siglip.modeling_siglip"
    assert a1.model.vision_encoder.class_or_factory == "SiglipVisionModel"
    assert a1.model.mllm.module_path == "transformers"
    assert a1.model.mllm.class_or_factory == "AutoModelForCausalLM"
    assert a1.model.mllm.checkpoint == "weights/NVILA-8B-HD-Video"
    assert a1.model.mllm.local_files_only is True
    assert a1.model.mllm.trust_remote_code is True
    assert a1.model.mllm.strict_checkpoint_loading is True

    assert a2.experiment.id == "A2_real"
    assert a2.model.autogaze.enabled is True
    assert a2.model.autogaze.module_path == "autogaze.models.autogaze"
    assert a2.model.autogaze.class_or_factory == "AutoGaze"
    assert a2.model.autogaze.checkpoint == "weights/AutoGaze"
    assert a2.model.autogaze.local_files_only is True
    assert a2.model.autogaze.strict_checkpoint_loading is True
