from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VisionEncoderOutput:
    visual_tokens: torch.Tensor
    metadata: dict[str, Any]


class BaseVisionEncoder:
    """Common interface for vision encoder adapters."""

    def __call__(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        return self.forward(video, metadata=metadata)

    def prepare_inputs(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
        return {"video": video, "metadata": dict(metadata or {})}

    def forward(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        raise NotImplementedError

    def count_visual_tokens(self, video_or_tokens: torch.Tensor) -> int:
        if video_or_tokens.ndim == 4:
            return int(video_or_tokens.shape[1] * video_or_tokens.shape[2])
        if video_or_tokens.ndim == 5:
            grid_h, grid_w = self.get_patch_grid((int(video_or_tokens.shape[-2]), int(video_or_tokens.shape[-1])))
            return int(video_or_tokens.shape[1] * grid_h * grid_w)
        raise ValueError(
            f"Expected tokens [B, T, N, D] or video [B, T, C, H, W], got {tuple(video_or_tokens.shape)}"
        )

    def get_patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError


def validate_video_shape(video: torch.Tensor) -> tuple[int, int, int, int, int]:
    if video.ndim != 5:
        raise ValueError(f"Expected video shape [B, T, C, H, W], got {tuple(video.shape)}")
    return tuple(int(dim) for dim in video.shape)  # type: ignore[return-value]


def patch_grid_from_resolution(resolution: tuple[int, int], patch_size: int) -> tuple[int, int]:
    height, width = int(resolution[0]), int(resolution[1])
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if height <= 0 or width <= 0:
        raise ValueError("resolution values must be > 0")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("resolution must be divisible by patch_size")
    return height // patch_size, width // patch_size
