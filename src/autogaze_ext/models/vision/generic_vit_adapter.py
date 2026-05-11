from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from autogaze_ext.models.vision.base_vision_encoder import (
    BaseVisionEncoder,
    VisionEncoderOutput,
    patch_grid_from_resolution,
    validate_video_shape,
)


class GenericViTAdapter(BaseVisionEncoder):
    """Checkpoint-free ViT-shaped dummy adapter.

    This produces patch tokens from average-pooled image patches. It is a shape
    adapter only, not a real ViT implementation.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_dim: int = 8,
        mode: str = "dummy",
        resolution: tuple[int, int] | None = None,
    ) -> None:
        if mode != "dummy":
            raise NotImplementedError("GenericViTAdapter currently supports dummy mode only")
        if patch_size <= 0 or hidden_dim <= 0:
            raise ValueError("patch_size and hidden_dim must be > 0")
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.resolution = resolution

    def prepare_inputs(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        batch, frames, channels, height, width = validate_video_shape(video)
        self.get_patch_grid((height, width))
        return {
            "video": video,
            "metadata": dict(metadata or {}),
            "shape": (batch, frames, channels, height, width),
        }

    def forward(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        prepared = self.prepare_inputs(video, metadata)
        batch, frames, channels, height, width = prepared["shape"]

        flattened = video.reshape(batch * frames, channels, height, width)
        pooled = F.avg_pool2d(flattened, kernel_size=self.patch_size, stride=self.patch_size)
        grid_h, grid_w = pooled.shape[-2:]
        tokens = pooled.flatten(2).transpose(1, 2).reshape(batch, frames, grid_h * grid_w, channels)

        repeat_factor = (self.hidden_dim + channels - 1) // channels
        visual_tokens = tokens.repeat_interleave(repeat_factor, dim=-1)[..., : self.hidden_dim]

        out_metadata = {
            **prepared["metadata"],
            "vision_encoder_type": "generic_vit",
            "vision_encoder_mode": self.mode,
            "patch_size": self.patch_size,
            "patch_grid": (grid_h, grid_w),
            "patch_indices": list(range(grid_h * grid_w)),
            "visual_token_count_per_frame": grid_h * grid_w,
            "visual_token_count": frames * grid_h * grid_w,
            "visual_token_shape": tuple(visual_tokens.shape),
        }
        return VisionEncoderOutput(visual_tokens=visual_tokens, metadata=out_metadata)

    def count_visual_tokens(self, video_or_tokens: torch.Tensor) -> int:
        return super().count_visual_tokens(video_or_tokens)

    def get_patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        resolved = resolution or self.resolution
        if resolved is None:
            raise ValueError("resolution is required when adapter was not configured with one")
        return patch_grid_from_resolution(resolved, self.patch_size)

    def get_output_dim(self) -> int:
        return self.hidden_dim
