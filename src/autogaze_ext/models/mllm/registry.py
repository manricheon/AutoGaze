from __future__ import annotations

from typing import Type

from autogaze_ext.models.mllm.base_mllm_adapter import BaseMLLMAdapter
from autogaze_ext.models.mllm.generic_mllm_adapter import GenericMLLMAdapter
from autogaze_ext.models.mllm.hf_mllm_adapter import HFMLLMAdapter
from autogaze_ext.models.mllm.nvila_adapter import NVILAAdapter
from autogaze_ext.models.mllm.qwen_adapter import QwenAdapter


MLLM_REGISTRY: dict[str, Type[BaseMLLMAdapter]] = {
    "generic_mllm": GenericMLLMAdapter,
    "hf_mllm": HFMLLMAdapter,
    "nvila": NVILAAdapter,
    "qwen": QwenAdapter,
}


def get_mllm_adapter_class(name: str) -> Type[BaseMLLMAdapter]:
    key = name.lower()
    try:
        return MLLM_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(MLLM_REGISTRY))
        raise ValueError(f"Unknown MLLM adapter '{name}'. Known adapters: {known}") from exc


def build_mllm_adapter(name: str, **kwargs):
    return get_mllm_adapter_class(name)(**kwargs)
