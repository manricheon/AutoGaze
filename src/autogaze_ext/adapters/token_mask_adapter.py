from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter


@dataclass(frozen=True)
class TokenMaskOutput:
    mask: torch.Tensor
    metadata: dict[str, Any]


class TokenMaskAdapter(BaseAdapter):
    """Convert selected patch indices to boolean token masks."""

    def forward(
        self,
        selected_patch_indices: torch.Tensor,
        num_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> TokenMaskOutput:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if selected_patch_indices.numel() == 0:
            raise ValueError("selected_patch_indices must not be empty")
        if selected_patch_indices.min().item() < 0 or selected_patch_indices.max().item() >= num_tokens:
            raise ValueError("selected_patch_indices contain values outside [0, num_tokens)")

        mask = torch.zeros(
            (*selected_patch_indices.shape[:-1], num_tokens),
            dtype=torch.bool,
            device=selected_patch_indices.device,
        )
        mask.scatter_(-1, selected_patch_indices.to(torch.long), True)
        out_metadata = {
            **dict(metadata or {}),
            "num_tokens": num_tokens,
            "selected_token_count": int(mask.sum(dim=-1).max().item()),
        }
        return TokenMaskOutput(mask=mask, metadata=out_metadata)
