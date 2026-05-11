from __future__ import annotations

from pathlib import Path

import torch
from PIL import ImageDraw

from autogaze_ext.visualization.base_visualizer import BaseVisualizer


class TaskOutputVisualizer(BaseVisualizer):
    """Dummy task-output visualizer for Video VQA and Action Recognition."""

    def visualize_video_vqa(
        self,
        video: torch.Tensor,
        answer: str,
        question: str | None = None,
        filename: str = "vqa_overlay.png",
    ) -> Path:
        output_dir = self.ensure_dir("video_vqa")
        frames = self._first_video(video)
        image = self._frame_to_image(frames[0])
        draw = ImageDraw.Draw(image)
        if question:
            draw.rectangle((0, 0, image.width, 36), fill=(0, 0, 0))
            draw.text((6, 4), f"Q: {question}", fill=(255, 255, 255))
        draw.rectangle((0, image.height - 24, image.width, image.height), fill=(0, 0, 0))
        draw.text((6, image.height - 20), f"A: {answer}", fill=(255, 255, 0))
        path = output_dir / filename
        image.save(path)
        return path

    def visualize_action_labels(
        self,
        labels: list[str],
        scores: list[float] | None = None,
        top_k: int = 5,
        filename: str = "action_topk.txt",
    ) -> Path:
        output_dir = self.ensure_dir("action_recognition")
        scores = scores or [1.0 for _ in labels]
        rows = []
        for rank, (label, score) in enumerate(zip(labels[:top_k], scores[:top_k]), start=1):
            rows.append(f"{rank}. {label}\t{float(score):.4f}")
        path = output_dir / filename
        path.write_text("\n".join(rows), encoding="utf-8")
        return path
