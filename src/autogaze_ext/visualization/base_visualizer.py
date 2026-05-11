from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image


class BaseVisualizer:
    """Base filesystem and tensor helpers for dummy visualizations."""

    def __init__(self, output_root: str | Path = "outputs", exp_name: str = "default") -> None:
        self.output_root = Path(output_root)
        self.exp_name = exp_name
        self.base_dir = self.output_root / exp_name / "visualizations"

    def ensure_dir(self, mode: str) -> Path:
        path = self.base_dir / mode
        path.mkdir(parents=True, exist_ok=True)
        return path

    def required_dirs(self) -> dict[str, Path]:
        return {
            "autogaze_only": self.ensure_dir("autogaze_only"),
            "full_pipeline": self.ensure_dir("full_pipeline"),
            "video_vqa": self.ensure_dir("video_vqa"),
            "action_recognition": self.ensure_dir("action_recognition"),
        }

    def _frame_to_image(self, frame: torch.Tensor) -> Image.Image:
        if frame.ndim != 3:
            raise ValueError(f"Expected frame shape [C, H, W], got {tuple(frame.shape)}")
        if frame.shape[0] not in {1, 3}:
            raise ValueError("Expected frame channel count 1 or 3")
        frame = frame.detach().cpu().to(torch.float32)
        min_value = float(frame.min().item())
        max_value = float(frame.max().item())
        if max_value > min_value:
            frame = (frame - min_value) / (max_value - min_value)
        frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
        if frame.shape[0] == 1:
            return Image.fromarray(frame.squeeze(0).numpy(), mode="L").convert("RGB")
        return Image.fromarray(frame.permute(1, 2, 0).numpy(), mode="RGB")

    def _first_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 5:
            return video[0]
        if video.ndim == 4:
            return video
        raise ValueError(f"Expected video shape [B, T, C, H, W] or [T, C, H, W], got {tuple(video.shape)}")

    @staticmethod
    def _as_list(values: Any) -> list[Any]:
        if values is None:
            return []
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().flatten().tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]
