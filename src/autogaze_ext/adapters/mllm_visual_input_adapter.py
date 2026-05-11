from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter


@dataclass(frozen=True)
class MLLMVisualInput:
    visual_inputs: torch.Tensor
    metadata: dict[str, Any]


class MLLMVisualInputAdapter(BaseAdapter):
    """Minimal standardized visual-input wrapper for future MLLM adapters."""

    def forward(
        self,
        visual_features: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> MLLMVisualInput:
        if visual_features.ndim not in {3, 4}:
            raise ValueError(
                f"Expected visual_features shape [B, N, D] or [B, T, N, D], got {tuple(visual_features.shape)}"
            )
        return MLLMVisualInput(
            visual_inputs=visual_features,
            metadata={**dict(metadata or {}), "visual_input_shape": tuple(visual_features.shape)},
        )
