from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter


@dataclass(frozen=True)
class CompactTokenOutput:
    tokens: torch.Tensor
    metadata: dict[str, Any]


class CompactTokenAdapter(BaseAdapter):
    """Gather selected tokens into compact sequences without padding mismatched requests."""

    def forward(
        self,
        tokens: torch.Tensor,
        selected_patch_indices: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> CompactTokenOutput:
        if tokens.ndim not in {3, 4}:
            raise ValueError(f"Expected tokens shape [B, N, D] or [B, T, N, D], got {tuple(tokens.shape)}")
        if selected_patch_indices.numel() == 0:
            raise ValueError("selected_patch_indices must not be empty")

        if tokens.ndim == 3:
            compact = self._gather_3d(tokens, selected_patch_indices)
        else:
            compact = self._gather_4d(tokens, selected_patch_indices)

        out_metadata = {
            **dict(metadata or {}),
            "selected_patch_indices": selected_patch_indices.detach().cpu().tolist(),
            "original_token_shape": tuple(tokens.shape),
            "compact_token_shape": tuple(compact.shape),
        }
        return CompactTokenOutput(tokens=compact, metadata=out_metadata)

    @staticmethod
    def _validate_indices(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        if indices.min().item() < 0 or indices.max().item() >= num_tokens:
            raise ValueError("selected_patch_indices contain values outside token dimension")
        return indices.to(torch.long)

    def _gather_3d(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, dim = tokens.shape
        indices = self._validate_indices(indices, num_tokens).to(tokens.device)
        if indices.ndim == 1:
            gather_idx = indices.view(1, -1, 1).expand(batch, -1, dim)
        elif indices.ndim == 2 and indices.shape[0] == batch:
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, dim)
        else:
            raise ValueError("For [B, N, D] tokens, selected indices must be [K] or [B, K]")
        return torch.gather(tokens, dim=1, index=gather_idx)

    def _gather_4d(self, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, frames, num_tokens, dim = tokens.shape
        indices = self._validate_indices(indices, num_tokens).to(tokens.device)
        if indices.ndim == 1:
            gather_idx = indices.view(1, 1, -1, 1).expand(batch, frames, -1, dim)
        elif indices.ndim == 2 and indices.shape[0] == frames:
            gather_idx = indices.view(1, frames, -1, 1).expand(batch, -1, -1, dim)
        elif indices.ndim == 3 and indices.shape[:2] == (batch, frames):
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        else:
            raise ValueError("For [B, T, N, D] tokens, selected indices must be [K], [T, K], or [B, T, K]")
        return torch.gather(tokens, dim=2, index=gather_idx)
