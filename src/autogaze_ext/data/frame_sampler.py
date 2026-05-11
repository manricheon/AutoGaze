from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrameSamplerOutput:
    video: torch.Tensor
    metadata: dict[str, Any]


class FrameSampler:
    """Uniform frame sampler for videos shaped [T, C, H, W]."""

    def __init__(
        self,
        num_frames: int | None = None,
        mode: str = "fixed",
        max_frames: int | None = None,
    ) -> None:
        if mode not in {"fixed", "max"}:
            raise ValueError(f"Unsupported frame sampling mode: {mode}")
        if mode == "fixed" and (num_frames is None or num_frames <= 0):
            raise ValueError("fixed mode requires num_frames > 0")
        if mode == "max" and (max_frames is None or max_frames <= 0):
            raise ValueError("max mode requires max_frames > 0")

        self.num_frames = num_frames
        self.mode = mode
        self.max_frames = max_frames

    def sample_indices(self, total_frames: int) -> torch.Tensor:
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")

        if self.mode == "fixed":
            target_frames = int(self.num_frames)
        else:
            target_frames = min(total_frames, int(self.max_frames))

        if target_frames == total_frames:
            return torch.arange(total_frames, dtype=torch.long)

        return torch.linspace(0, total_frames - 1, steps=target_frames).round().long()

    def __call__(self, video: torch.Tensor) -> FrameSamplerOutput:
        if video.ndim != 4:
            raise ValueError(f"Expected video shape [T, C, H, W], got {tuple(video.shape)}")

        indices = self.sample_indices(video.shape[0])
        sampled = video.index_select(0, indices.to(video.device))
        metadata = {
            "sampling_mode": self.mode,
            "input_frame_count": int(video.shape[0]),
            "sampled_frame_count": int(sampled.shape[0]),
            "original_frame_indices": indices.tolist(),
        }
        return FrameSamplerOutput(video=sampled, metadata=metadata)
