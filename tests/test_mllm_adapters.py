from __future__ import annotations

import pytest
import torch

from autogaze_ext.models.mllm import (
    BaseMLLMAdapter,
    GenericMLLMAdapter,
    HFMLLMAdapter,
    MLLM_REGISTRY,
    NVILAAdapter,
    QwenAdapter,
    build_mllm_adapter,
    get_mllm_adapter_class,
)


def test_mllm_interface_consistency() -> None:
    adapters = [
        GenericMLLMAdapter(),
        NVILAAdapter(),
        QwenAdapter(),
        HFMLLMAdapter(),
    ]

    for adapter in adapters:
        assert isinstance(adapter, BaseMLLMAdapter)
        for method in ["prepare_visual_inputs", "prepare_text_inputs", "forward", "generate", "count_visual_tokens"]:
            assert callable(getattr(adapter, method))


def test_generic_dummy_generation() -> None:
    adapter = GenericMLLMAdapter(answer="ok")
    tokens = torch.zeros(2, 3, 4, 8)

    output = adapter.generate(tokens, questions=["a", "b"])

    assert output.generated_text == ["ok", "ok"]
    assert output.logits is None
    assert output.visual_token_count == 12
    assert output.metadata["mllm_type"] == "generic_mllm"
    assert output.metadata["question_count"] == 2


def test_visual_token_count_reporting() -> None:
    adapter = GenericMLLMAdapter()

    assert adapter.count_visual_tokens(torch.zeros(2, 5, 8)) == 5
    assert adapter.count_visual_tokens(torch.zeros(2, 3, 5, 8)) == 15


def test_qwen_direct_visual_token_injection_error() -> None:
    adapter = QwenAdapter(mode="direct_visual_token_injection")

    with pytest.raises(NotImplementedError, match="direct visual token injection is not assumed supported"):
        adapter.prepare_visual_inputs(torch.zeros(1, 2, 3))


def test_qwen_staged_modes_raise_without_model_processor() -> None:
    for mode in ["official_processor", "input_region_selection", "post_visual_encoder_pruning"]:
        adapter = QwenAdapter(mode=mode)
        with pytest.raises(NotImplementedError, match="requires explicit model/processor"):
            adapter.generate(torch.zeros(1, 2, 3), text_inputs=["question"])


def test_stub_adapter_errors() -> None:
    with pytest.raises(NotImplementedError, match="NVILAAdapter"):
        NVILAAdapter().generate(torch.zeros(1, 2, 3))

    with pytest.raises(NotImplementedError, match="HFMLLMAdapter"):
        HFMLLMAdapter().generate(torch.zeros(1, 2, 3))


def test_registry_resolution() -> None:
    assert sorted(MLLM_REGISTRY) == ["generic_mllm", "hf_mllm", "nvila", "qwen"]
    assert get_mllm_adapter_class("nvila") is NVILAAdapter
    assert isinstance(build_mllm_adapter("generic_mllm"), GenericMLLMAdapter)

    with pytest.raises(ValueError, match="Unknown MLLM adapter"):
        get_mllm_adapter_class("missing")
