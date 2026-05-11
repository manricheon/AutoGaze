from __future__ import annotations

import torch

from autogaze_ext.visualization import AutoGazeVisualizer, BaseVisualizer, FullPipelineVisualizer, TaskOutputVisualizer


def _dummy_video() -> torch.Tensor:
    return torch.linspace(0, 1, steps=2 * 3 * 32 * 32).reshape(2, 3, 32, 32)


def test_output_directory_creation(tmp_path) -> None:
    visualizer = BaseVisualizer(output_root=tmp_path, exp_name="exp")

    dirs = visualizer.required_dirs()

    assert sorted(dirs) == ["action_recognition", "autogaze_only", "full_pipeline", "video_vqa"]
    for path in dirs.values():
        assert path.exists()
        assert path.is_dir()


def test_dummy_patch_visualization_save(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    paths = visualizer.visualize_selected_patches(
        _dummy_video(),
        selected_patch_indices=torch.tensor([0, 3, 5]),
        patch_grid=(4, 4),
        scales=[224, 224, 448],
    )
    scale_path = visualizer.visualize_scale_indicators([224, 224, 448])

    assert len(paths) == 2
    assert all(path.exists() and path.suffix == ".png" for path in paths)
    assert scale_path.exists()
    assert "224: 2" in scale_path.read_text()


def test_dummy_vqa_overlay_save(tmp_path) -> None:
    visualizer = TaskOutputVisualizer(output_root=tmp_path, exp_name="exp")

    path = visualizer.visualize_video_vqa(_dummy_video(), question="What?", answer="dummy")

    assert path.exists()
    assert path.parent.name == "video_vqa"
    assert path.suffix == ".png"


def test_dummy_action_label_save(tmp_path) -> None:
    visualizer = TaskOutputVisualizer(output_root=tmp_path, exp_name="exp")

    path = visualizer.visualize_action_labels(["walk", "run"], scores=[0.7, 0.3], top_k=2)

    assert path.exists()
    assert path.parent.name == "action_recognition"
    assert "1. walk" in path.read_text()


def test_full_pipeline_visualization_save(tmp_path) -> None:
    visualizer = FullPipelineVisualizer(output_root=tmp_path, exp_name="exp")

    paths = visualizer.visualize_full_pipeline(
        _dummy_video(),
        selected_patch_indices=[0, 1],
        patch_grid=(4, 4),
        answer="dummy",
        action_labels=["walk"],
    )

    assert any(path.parent.name == "full_pipeline" for path in paths)
    assert any(path.parent.name == "video_vqa" for path in paths)
    assert any(path.parent.name == "action_recognition" for path in paths)
