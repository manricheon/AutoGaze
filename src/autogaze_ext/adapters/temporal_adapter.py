from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter


@dataclass(frozen=True)
class TemporalAdapterOutput:
    features: torch.Tensor
    metadata: dict[str, Any]


class TemporalAdapter(BaseAdapter):
    """Temporal shape adapter for video features."""

    def __init__(self, mode: str) -> None:
        supported = {"frame_wise", "mean_pool", "max_pool", "concat_tokens", "native_autogaze"}
        if mode not in supported:
            raise ValueError(f"Unsupported temporal mode: {mode}")
        self.mode = mode

    def forward(self, features: torch.Tensor, metadata: dict[str, Any] | None = None) -> TemporalAdapterOutput:
        if self.mode == "native_autogaze":
            raise NotImplementedError("native_autogaze temporal mode requires original INTEGRATION.md handling")
        if features.ndim != 4:
            raise ValueError(f"Expected temporal features shape [B, T, N, D], got {tuple(features.shape)}")

        if self.mode == "frame_wise":
            output = features
        elif self.mode == "mean_pool":
            output = features.mean(dim=1)
        elif self.mode == "max_pool":
            output = features.max(dim=1).values
        elif self.mode == "concat_tokens":
            batch, frames, tokens, dim = features.shape
            output = features.reshape(batch, frames * tokens, dim)
        else:
            raise AssertionError(f"Unhandled temporal mode: {self.mode}")

        out_metadata = {
            **dict(metadata or {}),
            "temporal_mode": self.mode,
            "input_shape": tuple(features.shape),
            "output_shape": tuple(output.shape),
        }
        return TemporalAdapterOutput(features=output, metadata=out_metadata)
