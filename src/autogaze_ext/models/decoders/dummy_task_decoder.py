from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TaskDecoderOutput:
    logits: torch.Tensor
    predicted_labels: torch.Tensor
    metadata: dict[str, Any]


class DummyActionRecognitionDecoder:
    """Checkpoint-free action decoder for smoke tests."""

    def __init__(self, num_classes: int = 4) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        self.num_classes = num_classes

    def __call__(self, visual_tokens: torch.Tensor, metadata: dict[str, Any] | None = None) -> TaskDecoderOutput:
        if visual_tokens.ndim not in {3, 4}:
            raise ValueError(
                f"Expected visual_tokens shape [B, N, D] or [B, T, N, D], got {tuple(visual_tokens.shape)}"
            )
        batch = visual_tokens.shape[0]
        logits = torch.zeros(batch, self.num_classes, dtype=visual_tokens.dtype, device=visual_tokens.device)
        predicted_labels = torch.zeros(batch, dtype=torch.long, device=visual_tokens.device)
        logits[:, 0] = 1.0
        return TaskDecoderOutput(
            logits=logits,
            predicted_labels=predicted_labels,
            metadata={**dict(metadata or {}), "decoder_type": "dummy_action_recognition"},
        )
