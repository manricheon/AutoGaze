from __future__ import annotations

import torch
from torch import nn

from autogaze_ext.adapters.base_adapter import BaseAdapter


class VisionFeatureAdapter(BaseAdapter, nn.Module):
    """Linear projection placeholder for vision feature dimension matching."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        nn.Module.__init__(self)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.projection.in_features:
            raise ValueError(
                f"Expected feature dim {self.projection.in_features}, got {features.shape[-1]}"
            )
        return self.projection(features)
