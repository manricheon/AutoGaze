from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import ImageDraw

from autogaze_ext.visualization.base_visualizer import BaseVisualizer


class AutoGazeVisualizer(BaseVisualizer):
    """Dummy AutoGaze patch and scale visualizer."""

    def visualize_selected_patches(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        mode: str = "autogaze_only",
        prefix: str = "patches",
    ) -> list[Path]:
        output_dir = self.ensure_dir(mode)
        frames = self._first_video(video)
        selected = [int(idx) for idx in self._as_list(selected_patch_indices)]
        scale_values = self._as_list(scales)
        grid_h, grid_w = int(patch_grid[0]), int(patch_grid[1])
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("patch_grid values must be > 0")

        paths: list[Path] = []
        for frame_idx, frame in enumerate(frames):
            image = self._frame_to_image(frame)
            draw = ImageDraw.Draw(image)
            cell_w = image.width / grid_w
            cell_h = image.height / grid_h
            for i, patch_idx in enumerate(selected):
                if patch_idx < 0 or patch_idx >= grid_h * grid_w:
                    raise ValueError("selected_patch_indices contain values outside patch_grid")
                row, col = divmod(patch_idx, grid_w)
                x0, y0 = col * cell_w, row * cell_h
                x1, y1 = x0 + cell_w, y0 + cell_h
                draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=2)
                label = str(patch_idx)
                if i < len(scale_values):
                    label = f"{patch_idx}@{scale_values[i]}"
                draw.text((x0 + 2, y0 + 2), label, fill=(255, 255, 0))
            path = output_dir / f"{prefix}_frame_{frame_idx:03d}.png"
            image.save(path)
            paths.append(path)
        return paths

    def visualize_scale_indicators(
        self,
        scales: torch.Tensor | list[int],
        mode: str = "autogaze_only",
        filename: str = "scale_indicators.txt",
    ) -> Path:
        output_dir = self.ensure_dir(mode)
        path = output_dir / filename
        counts: dict[str, int] = {}
        for scale in self._as_list(scales):
            key = str(int(scale))
            counts[key] = counts.get(key, 0) + 1
        path.write_text("\n".join(f"{key}: {value}" for key, value in sorted(counts.items())), encoding="utf-8")
        return path

    def visualize(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        **metadata: Any,
    ) -> list[Path]:
        paths = self.visualize_selected_patches(video, selected_patch_indices, patch_grid, scales=scales)
        if scales is not None:
            paths.append(self.visualize_scale_indicators(scales))
        return paths
