from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from autogaze_ext.adapters.base_adapter import BaseAdapter


@dataclass(frozen=True)
class PatchGrid:
    height: int
    width: int

    @property
    def num_patches(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class PatchGridMapping:
    mapped_patch_indices: torch.Tensor
    metadata: dict[str, Any]


class PatchGridMapper(BaseAdapter):
    """Map flattened patch indices between explicit source and target patch grids."""

    def __init__(
        self,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
        source_resolution: tuple[int, int],
        target_resolution: tuple[int, int],
        source_patch_size: int | tuple[int, int] | None = None,
        target_patch_size: int | tuple[int, int] | None = None,
    ) -> None:
        self.source_grid = self._grid(source_grid, "source_grid")
        self.target_grid = self._grid(target_grid, "target_grid")
        self.source_resolution = self._pair(source_resolution, "source_resolution")
        self.target_resolution = self._pair(target_resolution, "target_resolution")
        self.source_patch_size = self._optional_pair(source_patch_size)
        self.target_patch_size = self._optional_pair(target_patch_size)
        self._validate_patch_size("source", self.source_resolution, self.source_grid, self.source_patch_size)
        self._validate_patch_size("target", self.target_resolution, self.target_grid, self.target_patch_size)

    def forward(
        self,
        selected_patch_indices: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> PatchGridMapping:
        if selected_patch_indices.numel() == 0:
            raise ValueError("selected_patch_indices must not be empty")
        if selected_patch_indices.min().item() < 0:
            raise ValueError("selected_patch_indices must be non-negative")
        if selected_patch_indices.max().item() >= self.source_grid.num_patches:
            raise ValueError("selected_patch_indices exceed source patch grid size")

        metadata = dict(metadata or {})
        if metadata.get("scales") is not None and metadata.get("scale_grids") is None:
            raise ValueError("multi-scale patch metadata requires explicit scale_grids")

        source_rows = torch.div(selected_patch_indices, self.source_grid.width, rounding_mode="floor")
        source_cols = selected_patch_indices % self.source_grid.width
        target_rows = self._map_axis(source_rows, self.source_grid.height, self.target_grid.height)
        target_cols = self._map_axis(source_cols, self.source_grid.width, self.target_grid.width)
        mapped = target_rows * self.target_grid.width + target_cols

        out_metadata = {
            **metadata,
            "source_patch_grid_size": (self.source_grid.height, self.source_grid.width),
            "target_patch_grid_size": (self.target_grid.height, self.target_grid.width),
            "source_resolution": self.source_resolution,
            "target_resolution": self.target_resolution,
            "source_patch_size": self.source_patch_size,
            "target_patch_size": self.target_patch_size,
            "patch_size_mismatch": self.source_patch_size != self.target_patch_size,
            "mapping": "nearest_patch_center",
        }
        return PatchGridMapping(mapped_patch_indices=mapped.to(torch.long), metadata=out_metadata)

    @staticmethod
    def _map_axis(values: torch.Tensor, source_size: int, target_size: int) -> torch.Tensor:
        centers = (values.to(torch.float32) + 0.5) / float(source_size)
        mapped = torch.floor(centers * float(target_size)).to(torch.long)
        return mapped.clamp(min=0, max=target_size - 1)

    @classmethod
    def _grid(cls, value: tuple[int, int], name: str) -> PatchGrid:
        height, width = cls._pair(value, name)
        return PatchGrid(height=height, width=width)

    @staticmethod
    def _pair(value: tuple[int, int], name: str) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"{name} must have two values")
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError(f"{name} values must be > 0")
        return first, second

    @staticmethod
    def _optional_pair(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("patch size must be > 0")
            return value, value
        if len(value) != 2:
            raise ValueError("patch size tuple must have two values")
        first, second = int(value[0]), int(value[1])
        if first <= 0 or second <= 0:
            raise ValueError("patch size values must be > 0")
        return first, second

    @staticmethod
    def _validate_patch_size(
        label: str,
        resolution: tuple[int, int],
        grid: PatchGrid,
        patch_size: tuple[int, int] | None,
    ) -> None:
        if patch_size is None:
            return
        expected = (resolution[0] // grid.height, resolution[1] // grid.width)
        if resolution[0] % grid.height != 0 or resolution[1] % grid.width != 0:
            raise ValueError(f"{label} resolution is not divisible by {label} patch grid")
        if patch_size != expected:
            raise ValueError(
                f"{label} patch size {patch_size} does not match resolution/grid-derived patch size {expected}"
            )
