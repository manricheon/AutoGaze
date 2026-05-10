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
