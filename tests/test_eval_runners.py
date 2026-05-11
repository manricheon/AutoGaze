import pytest
import torch

import autogaze.eval.models as models


class DummyRunner(models.BaseMLLMRunner):
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def clear_dummy_calls():
    DummyRunner.calls = []
    yield
    DummyRunner.calls = []


def test_primary_runner_keys_are_registered():
    expected = {
        "nvila",
        "vjepa2_nvila",
        "siglip_qwen25",
        "vjepa2_qwen25",
        "generic_mllm",
        "vjepa2",
        "siglip",
    }

    assert expected.issubset(models.RUNNERS)


def test_register_runner_requires_base_runner_subclass():
    class NotARunner:
        pass

    with pytest.raises(TypeError):
        models.register_runner("bad_runner", NotARunner)


def test_load_runner_applies_primary_default_integration(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "siglip_qwen25", DummyRunner)

    runner = models.load_runner(
        mllm="siglip_qwen25",
        model_path="model",
        autogaze_path="ag",
        gazing_ratio=0.5,
    )

    assert isinstance(runner, DummyRunner)
    assert runner.kwargs["model_path"] == "model"
    assert runner.kwargs["autogaze_path"] == "ag"
    assert runner.kwargs["gazing_ratio"] == 0.5
    assert runner.kwargs["dtype"] is torch.bfloat16
    assert runner.kwargs["integration"] == "hook"


def test_load_runner_allows_integration_override(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "siglip_qwen25", DummyRunner)

    runner = models.load_runner(
        mllm="siglip_qwen25",
        model_path="model",
        autogaze_path="ag",
        gazing_ratio=0.5,
        integration="full",
    )

    assert runner.kwargs["integration"] == "full"


def test_deprecated_qwen25vl_full_alias_sets_full_integration(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "siglip_qwen25", DummyRunner)
    monkeypatch.setitem(models.RUNNERS, "qwen25vl_full", DummyRunner)

    with pytest.warns(DeprecationWarning, match="qwen25vl_full"):
        runner = models.load_runner(
            mllm="qwen25vl_full",
            model_path="model",
            autogaze_path="ag",
            gazing_ratio=0.5,
        )

    assert runner.kwargs["integration"] == "full"


def test_vjepa2_nvila_requires_vjepa2_path(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "vjepa2_nvila", DummyRunner)

    with pytest.raises(ValueError, match="vjepa2_path"):
        models.load_runner(
            mllm="vjepa2_nvila",
            model_path="nvila",
            autogaze_path="ag",
            gazing_ratio=0.75,
        )


def test_vjepa2_nvila_forwards_separate_vjepa2_path(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "vjepa2_nvila", DummyRunner)

    runner = models.load_runner(
        mllm="vjepa2_nvila",
        model_path="nvila",
        vjepa2_path="vjepa2",
        autogaze_path="ag",
        gazing_ratio=0.75,
    )

    assert runner.kwargs["model_path"] == "nvila"
    assert runner.kwargs["vjepa2_path"] == "vjepa2"
    assert runner.kwargs["integration"] == "full"


def test_unknown_runner_error_lists_primary_and_deprecated_keys():
    with pytest.raises(ValueError) as exc_info:
        models.load_runner(
            mllm="unknown_runner",
            model_path="model",
            autogaze_path="ag",
            gazing_ratio=0.5,
        )

    message = str(exc_info.value)
    assert "Primary keys" in message
    assert "Deprecated aliases" in message
    assert "siglip_qwen25" in message


def test_generic_mllm_default_integration_and_required_hook(monkeypatch):
    monkeypatch.setitem(models.RUNNERS, "generic_mllm", DummyRunner)

    runner = models.load_runner(
        mllm="generic_mllm",
        model_path="model",
        autogaze_path="ag",
        gazing_ratio=0.5,
        vision_hook="vision_model.embeddings",
    )

    assert runner.kwargs["integration"] == "hook"
    assert runner.kwargs["vision_hook"] == "vision_model.embeddings"


def test_generic_mask_preserves_cls_token():
    runner = models.GenericHookMLLMRunner.__new__(models.GenericHookMLLMRunner)
    runner.has_cls_token = True

    tensor = torch.ones(1, 5, 2)
    spatial_mask = torch.tensor([0.0, 1.0])

    masked = runner._apply_mask_to_tensor(tensor, spatial_mask)

    assert masked[0, 0].tolist() == [1.0, 1.0]  # CLS preserved
    assert masked[0, 1].tolist() == [0.0, 0.0]
    assert masked[0, 2].tolist() == [1.0, 1.0]
    assert masked[0, 3].tolist() == [0.0, 0.0]
    assert masked[0, 4].tolist() == [1.0, 1.0]


def test_resolve_module_accepts_optional_model_prefix():
    class Leaf:
        pass

    class Root:
        pass

    root = Root()
    root.visual = Root()
    root.visual.patch_embed = Leaf()

    assert models._resolve_module(root, "visual.patch_embed") is root.visual.patch_embed
    assert models._resolve_module(root, "model.visual.patch_embed") is root.visual.patch_embed
