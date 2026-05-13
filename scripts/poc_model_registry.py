#!/usr/bin/env python3
from __future__ import annotations

from typing import Mapping

from poc_model_adapters import (
    GenericMLLMAdapter,
    GenericVitAdapter,
    MLLMAdapter,
    ModifiedSiglipAdapter,
    NVILAAdapter,
    QwenAdapter,
    VJEPA2Adapter,
    VanillaSiglipAdapter,
    VisionEncoderAdapter,
)


VISION_ENCODERS: dict[str, type[VisionEncoderAdapter]] = {
    "modified_siglip": ModifiedSiglipAdapter,
    "vanilla_siglip": VanillaSiglipAdapter,
    "vjepa2": VJEPA2Adapter,
    "generic_vit": GenericVitAdapter,
}

MLLMS: dict[str, type[MLLMAdapter]] = {
    "nvila": NVILAAdapter,
    "qwen": QwenAdapter,
    "generic_mllm": GenericMLLMAdapter,
}


def build_vision_encoder(name: str, config: Mapping | None = None) -> VisionEncoderAdapter:
    try:
        cls = VISION_ENCODERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown vision encoder {name!r}; valid names: {sorted(VISION_ENCODERS)}") from exc
    return cls(config)


def build_mllm(name: str, config: Mapping | None = None) -> MLLMAdapter:
    try:
        cls = MLLMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown MLLM {name!r}; valid names: {sorted(MLLMS)}") from exc
    return cls(config)
