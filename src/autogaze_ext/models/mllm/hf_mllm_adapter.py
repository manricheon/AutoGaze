from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.mllm.base_mllm_adapter import BaseMLLMAdapter, MLLMOutput


class HFMLLMAdapter(BaseMLLMAdapter):
    """Placeholder for Hugging Face model-based MLLMs."""

    def __init__(self, model: Any | None = None, processor: Any | None = None, mode: str = "official_processor") -> None:
        self.model = model
        self.processor = processor
        self.mode = mode

    def generate(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        prepared_visual = self.prepare_visual_inputs(visual_inputs)
        self.prepare_text_inputs(text_inputs)
        if self.model is None or self.processor is None:
            raise NotImplementedError(
                "HFMLLMAdapter requires explicit Hugging Face model and processor instances. "
                "Model downloading/loading is outside this placeholder scope."
            )
        raise NotImplementedError("HFMLLMAdapter execution is not implemented in this stub")

    def forward(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        return self.generate(visual_inputs, text_inputs=text_inputs, **kwargs)
