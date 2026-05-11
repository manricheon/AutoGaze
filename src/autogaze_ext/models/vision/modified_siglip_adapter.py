from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.vision.base_vision_encoder import BaseVisionEncoder, VisionEncoderOutput, patch_grid_from_resolution


class ModifiedSigLIPAdapter(BaseVisionEncoder):
    """Stub wrapper boundary for the original modified SigLIP implementation."""

    def __init__(
        self,
        original_model: Any | None = None,
        patch_size: int = 16,
        output_dim: int = 768,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        self.original_model = original_model
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.resolution = resolution

    def prepare_inputs(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        prepared = super().prepare_inputs(video, metadata)
        prepared["adapter"] = "modified_siglip"
        return prepared

    def forward(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        prepared = self.prepare_inputs(video, metadata)
        if self.original_model is None:
            raise NotImplementedError(
                "ModifiedSigLIPAdapter requires an explicit original modified SigLIP model instance. "
                "Checkpoint loading and real SigLIP execution are outside this stub scope."
            )
        output = self.original_model(prepared["video"])
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wrapped modified SigLIP model must return a torch.Tensor in this adapter boundary")
        return VisionEncoderOutput(
            visual_tokens=output,
            metadata={**prepared["metadata"], "vision_encoder_type": "modified_siglip", "vision_encoder_mode": "wrapped"},
        )

    def get_patch_grid(self, resolution: tuple[int, int] | None = None) -> tuple[int, int]:
        resolved = resolution or self.resolution
        if resolved is None:
            raise ValueError("resolution is required when adapter was not configured with one")
        return patch_grid_from_resolution(resolved, self.patch_size)

    def get_output_dim(self) -> int:
        return self.output_dim
