from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.mllm.base_mllm_adapter import BaseMLLMAdapter, MLLMOutput


class QwenAdapter(BaseMLLMAdapter):
    """Qwen-family adapter boundary with staged integration placeholders."""

    SUPPORTED_MODES = {
        "official_processor",
        "input_region_selection",
        "post_visual_encoder_pruning",
        "direct_visual_token_injection",
    }

    def __init__(self, model: Any | None = None, processor: Any | None = None, mode: str = "official_processor") -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported QwenAdapter mode: {mode}")
        self.model = model
        self.processor = processor
        self.mode = mode

    def prepare_visual_inputs(
        self,
        vision_outputs: torch.Tensor | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prepared = super().prepare_visual_inputs(vision_outputs, metadata=metadata)
        prepared["integration_mode"] = self.mode
        if self.mode == "direct_visual_token_injection":
            raise NotImplementedError(
                "Qwen direct visual token injection is not assumed supported. "
                "Use official_processor or verify architecture/API support before enabling this mode."
            )
        return prepared

    def generate(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        prepared_visual = self.prepare_visual_inputs(visual_inputs)
        prepared_text = self.prepare_text_inputs(text_inputs)
        if self.model is None or self.processor is None:
            raise NotImplementedError(
                f"QwenAdapter mode '{self.mode}' requires explicit model/processor instances. "
                "Real Qwen loading/inference is outside this stub scope."
            )
        output = self.model(prepared_visual["visual_tokens"], prepared_text["texts"], **kwargs)
        return MLLMOutput(
            generated_text=output if isinstance(output, list) else None,
            logits=output if isinstance(output, torch.Tensor) else None,
            visual_token_count=self.count_visual_tokens(prepared_visual["visual_tokens"]),
            metadata={"mllm_type": "qwen", "integration_mode": self.mode},
        )

    def forward(
        self,
        visual_inputs: torch.Tensor,
        text_inputs: str | list[str] | None = None,
        **kwargs: Any,
    ) -> MLLMOutput:
        return self.generate(visual_inputs, text_inputs=text_inputs, **kwargs)
