from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from autogaze_ext.investigation.model_construction import run_model_construction_check


def _install_mock_real_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    autogaze_module = types.ModuleType("autogaze.models.autogaze")
    siglip_config_module = types.ModuleType("autogaze.vision_encoders.siglip.configuration_siglip")
    siglip_model_module = types.ModuleType("autogaze.vision_encoders.siglip.modeling_siglip")
    nvila_module = types.ModuleType("nvila")
    transformers_module = types.ModuleType("transformers")

    class AutoGazeConfig:
        pass

    class AutoGaze:
        def __init__(self, config=None, **kwargs):
            self.config = config
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, checkpoint_path, **kwargs):
            return cls(config={"checkpoint_path": checkpoint_path}, **kwargs)

    class SiglipVisionConfig:
        pass

    class SiglipVisionModel:
        def __init__(self, config=None, **kwargs):
            self.config = config
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, checkpoint_path, **kwargs):
            return cls(config={"checkpoint_path": checkpoint_path}, **kwargs)

    class NVILA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoModelForCausalLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, checkpoint_path, **kwargs):
            return cls(checkpoint_path=checkpoint_path, **kwargs)

    autogaze_module.AutoGazeConfig = AutoGazeConfig
    autogaze_module.AutoGaze = AutoGaze
    siglip_config_module.SiglipVisionConfig = SiglipVisionConfig
    siglip_model_module.SiglipVisionModel = SiglipVisionModel
    nvila_module.NVILA = NVILA
    transformers_module.AutoModelForCausalLM = AutoModelForCausalLM

    monkeypatch.setitem(sys.modules, "autogaze.models.autogaze", autogaze_module)
    monkeypatch.setitem(sys.modules, "autogaze.vision_encoders.siglip.configuration_siglip", siglip_config_module)
    monkeypatch.setitem(sys.modules, "autogaze.vision_encoders.siglip.modeling_siglip", siglip_model_module)
    monkeypatch.setitem(sys.modules, "nvila", nvila_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)


def test_model_construction_level_0_import_check_reports_quick_start() -> None:
    report = run_model_construction_check(
        experiment="A2_real",
        component="autogaze",
        construction_level=0,
        no_checkpoint_load=True,
        device="cpu",
    )

    assert report.quick_start_found is True
    assert report.quick_start_reference_found is True
    assert report.components[0].module_available is True
    assert report.components[0].model_constructed is False
    assert report.components[0].quick_start_fields_used["input_resolution"] == 224
    assert report.components[0].quick_start_fields_used["frame_count"] == 16


def test_model_construction_level_2_config_only_with_mock_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mock_real_modules(monkeypatch)

    report = run_model_construction_check(
        experiment="A2_real",
        component="autogaze",
        construction_level=2,
        no_checkpoint_load=True,
        device="cpu",
    )

    result = report.components[0]
    assert result.status == "passed"
    assert result.config_constructed is True
    assert result.model_constructed is False
    assert result.checkpoint_load_attempted is False


def test_model_construction_level_3_metadata_uses_configured_local_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mock_real_modules(monkeypatch)

    report = run_model_construction_check(
        experiment="A2_real",
        component="autogaze",
        construction_level=3,
        no_checkpoint_load=True,
        checkpoint_metadata_only=True,
        device="cpu",
    )

    assert report.components[0].status == "passed"
    assert report.components[0].checkpoint_path == "weights/AutoGaze"
    assert report.components[0].checkpoint_exists is True
    assert report.components[0].checkpoint_metadata_checked is True


def test_model_construction_level_4_no_checkpoint_load_constructs_mock_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_mock_real_modules(monkeypatch)

    report = run_model_construction_check(
        experiment="A2_real",
        component="all",
        construction_level=4,
        no_checkpoint_load=True,
        device="cpu",
    )

    by_component = {result.component: result for result in report.components}
    assert by_component["autogaze"].status == "passed"
    assert by_component["autogaze"].model_constructed is True
    assert by_component["vision_encoder"].status == "passed"
    assert by_component["vision_encoder"].model_constructed is True
    assert by_component["mllm"].status == "passed"
    assert by_component["mllm"].model_constructed is True
    assert all(result.checkpoint_load_attempted is False for result in report.components)


def test_model_construction_reports_local_mllm_factory() -> None:
    report = run_model_construction_check(
        experiment="A1_real",
        component="mllm",
        construction_level=1,
        no_checkpoint_load=True,
        device="cpu",
    )

    result = report.components[0]
    assert result.component == "mllm"
    assert result.status == "passed"
    assert result.module_path == "transformers"
    assert result.class_or_factory == "AutoModelForCausalLM"
    assert result.module_available is True
    assert result.class_or_factory_exists is True
