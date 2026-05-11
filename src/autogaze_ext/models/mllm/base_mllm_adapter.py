from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MLLMOutput:
    generated_text: list[str] | None
    logits: torch.Tensor | None
    visual_token_count: int
    metadata: dict[str, Any]


class BaseMLLMAdapter:
    """Common interface for MLLM adapters."""

    def prepare_visual_inputs(
        self,
        vision_outputs: torch.Tensor | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(vision_outputs, dict):
            visual_tokens = vision_outputs.get("visual_tokens")
            merged_metadata = {**dict(vision_outputs.get("metadata", {})), **dict(metadata or {})}
        else:
            visual_tokens = vision_outputs
            merged_metadata = dict(metadata or {})

        if not isinstance(visual_tokens, torch.Tensor):
            raise TypeError("visual inputs must contain a torch.Tensor")
        if visual_tokens.ndim not in {3, 4}:
            raise ValueError(
                f"Expected visual tokens shape [B, N, D] or [B, T, N, D], got {tuple(visual_tokens.shape)}"
            )
        return {"visual_tokens": visual_tokens, "metadata": merged_metadata}

    def prepare_text_inputs(
        self,
        text_inputs: str | list[str] | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if text_inputs is None:
            texts: list[str] = []
        elif isinstance(text_inputs, str):
            texts = [text_inputs]
        else:
            texts = list(text_inputs)
        return {"texts": texts, "metadata": dict(metadata or {})}

    def forward(self, visual_inputs: torch.Tensor, text_inputs: str | list[str] | None = None, **kwargs: Any) -> MLLMOutput:
        raise NotImplementedError

    def generate(self, visual_inputs: torch.Tensor, text_inputs: str | list[str] | None = None, **kwargs: Any) -> MLLMOutput:
        raise NotImplementedError

    def count_visual_tokens(self, visual_inputs: torch.Tensor | dict[str, Any]) -> int:
        prepared = self.prepare_visual_inputs(visual_inputs)
        visual_tokens = prepared["visual_tokens"]
        if visual_tokens.ndim == 4:
            return int(visual_tokens.shape[1] * visual_tokens.shape[2])
        return int(visual_tokens.shape[1])
