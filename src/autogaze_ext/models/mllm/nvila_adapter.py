from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.mllm.base_mllm_adapter import BaseMLLMAdapter, MLLMOutput


class NVILAAdapter(BaseMLLMAdapter):
    """Adapter boundary for NVILA."""

    def __init__(self, model: Any | None = None, mode: str = "wrapped") -> None:
        self.model = model
        self.mode = mode

    def forward(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        if self.model is None:
            raise NotImplementedError(
                "NVILAAdapter requires an explicit NVILA model instance. "
                "Real NVILA loading/inference is outside this stub scope."
            )
        prepared_visual = self.prepare_visual_inputs(visual_inputs)
        prepared_text = self.prepare_text_inputs(text_inputs)
        output = self.model(prepared_visual["visual_tokens"], prepared_text["texts"], **kwargs)
        if isinstance(output, MLLMOutput):
            return output
        return MLLMOutput(
            generated_text=None,
            logits=output if isinstance(output, torch.Tensor) else None,
            visual_token_count=self.count_visual_tokens(prepared_visual["visual_tokens"]),
            metadata={"mllm_type": "nvila", "mllm_mode": self.mode},
        )

    def generate(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        if self.model is None:
            raise NotImplementedError(
                "NVILAAdapter generation requires an explicit NVILA model instance. "
                "Real NVILA loading/inference is outside this stub scope."
            )
        return self.forward(visual_inputs, text_inputs=text_inputs, **kwargs)
