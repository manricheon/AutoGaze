from __future__ import annotations

from pathlib import Path

import torch

from autogaze_ext.visualization.autogaze_visualizer import AutoGazeVisualizer
from autogaze_ext.visualization.task_output_visualizer import TaskOutputVisualizer


class FullPipelineVisualizer(AutoGazeVisualizer):
    """Combined dummy visualizer for AutoGaze overlays and task outputs."""

    def __init__(self, output_root: str | Path = "outputs", exp_name: str = "default") -> None:
        super().__init__(output_root=output_root, exp_name=exp_name)
        self.task_visualizer = TaskOutputVisualizer(output_root=output_root, exp_name=exp_name)

    def visualize_full_pipeline(
        self,
        video: torch.Tensor,
        selected_patch_indices: torch.Tensor | list[int],
        patch_grid: tuple[int, int],
        scales: torch.Tensor | list[int] | None = None,
        answer: str | None = None,
        action_labels: list[str] | None = None,
    ) -> list[Path]:
        paths = self.visualize_selected_patches(
            video,
            selected_patch_indices,
            patch_grid,
            scales=scales,
            mode="full_pipeline",
            prefix="full_pipeline",
        )
        if answer is not None:
            paths.append(self.task_visualizer.visualize_video_vqa(video, answer=answer))
        if action_labels is not None:
            paths.append(self.task_visualizer.visualize_action_labels(action_labels))
        return paths
