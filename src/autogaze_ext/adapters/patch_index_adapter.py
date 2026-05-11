from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter
from autogaze_ext.adapters.patch_grid_mapper import PatchGridMapper, PatchGridMapping


class PatchIndexAdapter(BaseAdapter):
    """Adapter wrapper for patch-index remapping."""

    def __init__(self, mapper: PatchGridMapper) -> None:
        self.mapper = mapper

    def forward(
        self,
        selected_patch_indices: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> PatchGridMapping:
        return self.mapper(selected_patch_indices, metadata=metadata)
