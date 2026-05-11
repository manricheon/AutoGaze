from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.mllm.base_mllm_adapter import BaseMLLMAdapter, MLLMOutput


class GenericMLLMAdapter(BaseMLLMAdapter):
    """Checkpoint-free dummy MLLM adapter for smoke inference."""

    def __init__(self, mode: str = "dummy", answer: str = "dummy") -> None:
        if mode != "dummy":
            raise NotImplementedError("GenericMLLMAdapter currently supports dummy mode only")
        self.mode = mode
        self.answer = answer

    def generate(
        self,
        visual_tokens: torch.Tensor,
        questions: list[str] | None = None,
        text_inputs: str | list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MLLMOutput:
        prepared_visual = self.prepare_visual_inputs(visual_tokens, metadata=metadata)
        visual_tokens = prepared_visual["visual_tokens"]
        prepared_text = self.prepare_text_inputs(questions if questions is not None else text_inputs)
        batch = visual_tokens.shape[0]
        token_count = self.count_visual_tokens(visual_tokens)
        answers = [self.answer for _ in range(batch)]
        return MLLMOutput(
            generated_text=answers,
            logits=None,
            visual_token_count=token_count,
            metadata={
                **prepared_visual["metadata"],
                "mllm_type": "generic_mllm",
                "mllm_mode": self.mode,
                "question_count": len(prepared_text["texts"]),
            },
        )

    def forward(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        return self.generate(visual_inputs, text_inputs=text_inputs, **kwargs)
