from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.vision.base_vision_encoder import BaseVisionEncoder, VisionEncoderOutput, patch_grid_from_resolution
from autogaze_ext.models.vision.generic_vit_adapter import GenericViTAdapter


class VJEPA2Adapter(BaseVisionEncoder):
    """V-JEPA2 adapter boundary with dummy shape behavior and explicit mode checks."""

    SUPPORTED_MODES = {"full", "crop", "mask", "compact"}

    def __init__(
        self,
        model: Any | None = None,
        mode: str = "full",
        patch_size: int = 16,
        hidden_dim: int = 768,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported VJEPA2Adapter mode: {mode}")
        self.model = model
        self.mode = mode
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        self._dummy = GenericViTAdapter(
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            mode="dummy",
            resolution=resolution,
        )

    def prepare_inputs(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        prepared = super().prepare_inputs(video, metadata)
        prepared["vjepa2_mode"] = self.mode
        return prepared

    def forward(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        prepared = self.prepare_inputs(video, metadata)
        if self.model is not None:
            output = self.model(prepared["video"])
            if not isinstance(output, torch.Tensor):
                raise TypeError("Wrapped V-JEPA2 model must return a torch.Tensor in this adapter boundary")
            return VisionEncoderOutput(
                visual_tokens=output,
                metadata={**prepared["metadata"], "vision_encoder_type": "vjepa2", "vision_encoder_mode": self.mode},
            )

        output = self._dummy.forward(prepared["video"], prepared["metadata"])
        return VisionEncoderOutput(
            visual_tokens=output.visual_tokens,
            metadata={
                **output.metadata,
                "vision_encoder_type": "vjepa2",
                "vision_encoder_mode": self.mode,
                "preserves_video_shape": tuple(video.shape),
                "stub_status": "dummy_shape_only_no_real_vjepa2",
            },
        )

    def count_visual_tokens(self, video_or_tokens: torch.Tensor) -> int:
        return self._dummy.count_visual_tokens(video_or_tokens)

    def get_patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        resolved = resolution or self.resolution
        if resolved is None:
            raise ValueError("resolution is required when adapter was not configured with one")
        return patch_grid_from_resolution(resolved, self.patch_size)

    def get_output_dim(self) -> int:
        return self.hidden_dim
