from __future__ import annotations

from typing import Any

import torch

from autogaze_ext.models.vision.generic_vit_adapter import GenericViTAdapter
from autogaze_ext.models.vision.base_vision_encoder import VisionEncoderOutput


class VanillaSigLIPAdapter(GenericViTAdapter):
    """Vanilla SigLIP adapter boundary with dummy mode for shape tests."""

    def __init__(
        self,
        model: Any | None = None,
        patch_size: int = 16,
        hidden_dim: int = 768,
        mode: str = "dummy",
        resolution: tuple[int, int] | None = None,
    ) -> None:
        if mode not in {"dummy", "hf", "local"}:
            raise ValueError(f"Unsupported VanillaSigLIPAdapter mode: {mode}")
        if mode != "dummy" and model is None:
            raise NotImplementedError(
                "VanillaSigLIPAdapter hf/local modes require an explicit model instance. "
                "Hugging Face and local checkpoint loading are not implemented here."
            )
        super().__init__(patch_size=patch_size, hidden_dim=hidden_dim, mode="dummy", resolution=resolution)
        self.model = model
        self.mode = mode

    def forward(self, video: torch.Tensor, metadata: dict[str, Any] | None = None) -> VisionEncoderOutput:
        if self.mode == "dummy":
            output = super().forward(video, metadata)
            return VisionEncoderOutput(
                visual_tokens=output.visual_tokens,
                metadata={
                    **output.metadata,
                    "vision_encoder_type": "vanilla_siglip",
                    "vision_encoder_mode": "dummy",
                },
            )
        prepared = self.prepare_inputs(video, metadata)
        output = self.model(prepared["video"])
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wrapped vanilla SigLIP model must return a torch.Tensor in this adapter boundary")
        return VisionEncoderOutput(
            visual_tokens=output,
            metadata={**prepared["metadata"], "vision_encoder_type": "vanilla_siglip", "vision_encoder_mode": self.mode},
        )
