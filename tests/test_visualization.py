from __future__ import annotations

import json

import pytest
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


def test_autogaze_sampled_only_video_export(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    artifacts = visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0, 3], [5, 7]],
        patch_grid=(4, 4),
        sampled_frame_indices=[10, 3],
        original_frame_count=20,
        original_resolution=(32, 32),
        processed_resolution=(32, 32),
        original_visual_token_count=32,
        selected_visual_token_count=4,
        fps=3.5,
        save_overlay_video=True,
        save_side_by_side_video=True,
    )

    frames_dir = tmp_path / "exp" / "visualizations" / "autogaze_only" / "frames"
    overlay_video = tmp_path / "exp" / "visualizations" / "autogaze_only" / "videos" / "autogaze_overlay.mp4"
    side_by_side_video = tmp_path / "exp" / "visualizations" / "autogaze_only" / "videos" / "autogaze_side_by_side.mp4"
    metadata_path = (
        tmp_path
        / "exp"
        / "visualizations"
        / "autogaze_only"
        / "metadata"
        / "visualization_video_metadata.json"
    )

    assert artifacts["overlay_video"] == overlay_video
    assert artifacts["side_by_side_video"] == side_by_side_video
    assert overlay_video.exists() and overlay_video.stat().st_size > 0
    assert side_by_side_video.exists() and side_by_side_video.stat().st_size > 0
    assert len(list(frames_dir.glob("*_overlay_frame_*.png"))) == 2
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["video_export_mode"] == "sampled_only"
    assert metadata["fps"] == 3.5
    assert metadata["sampled_frame_indices"] == [10, 3]
    assert metadata["processed_frame_count"] == 2
    assert metadata["patch_grid"] == [4, 4]
    assert metadata["patch_grid_source"] == "provided"
    assert metadata["original_visual_token_count"] == 32
    assert metadata["selected_visual_token_count"] == 4
    assert metadata["token_reduction_ratio"] == 0.875


def test_autogaze_scale_panel_video_export(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    artifacts = visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0, 3], [5, 7]],
        patch_grid=(4, 4),
        scales=[[32, 64], [32, 64]],
        sampled_frame_indices=[0, 1],
        save_overlay_video=False,
        save_side_by_side_video=False,
        save_scale_panel_video=True,
        show_patch_boxes=False,
        show_patch_indices=False,
        info_panel_mode="external",
    )

    scale_panel = tmp_path / "exp" / "visualizations" / "autogaze_only" / "videos" / "autogaze_scale_panels.mp4"
    scale_panel_frame = (
        tmp_path
        / "exp"
        / "visualizations"
        / "autogaze_only"
        / "scale_panels"
        / "autogaze_scale_panel_frame_000.png"
    )
    metadata_path = (
        tmp_path
        / "exp"
        / "visualizations"
        / "autogaze_only"
        / "metadata"
        / "visualization_video_metadata.json"
    )

    assert artifacts["scale_panel_video"] == scale_panel
    assert scale_panel.exists() and scale_panel.stat().st_size > 0
    assert scale_panel_frame.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["overlay_style"] == "mask"
    assert metadata["show_patch_indices"] is False
    assert metadata["scale_panel_layout"] == "2x2"
    assert metadata["info_panel_mode"] == "external"


def test_multiscale_metadata_uses_gradient_palette(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0, 1, 2, 3], [4, 5, 6, 7]],
        patch_grid=(4, 4),
        scales=[[0, 1, 2, 3], [0, 1, 2, 3]],
        sampled_frame_indices=[0, 1],
        save_overlay_video=True,
        save_side_by_side_video=False,
        multi_scale_overlay=True,
        scale_color_mode="gradient",
        show_patch_indices=False,
        show_scale_labels=True,
    )

    metadata = json.loads(
        (
            tmp_path
            / "exp"
            / "visualizations"
            / "autogaze_only"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["missing_scale_metadata"] is False
    assert metadata["available_scale_ids"] == [0, 1, 2, 3]
    assert metadata["scale_color_mode"] == "gradient"
    assert metadata["scale_color_map"]["0"] == [254, 240, 138]
    assert metadata["show_patch_indices"] is False
    assert metadata["show_scale_labels"] is True


def test_missing_scale_metadata_falls_back_to_single_color(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0, 1], [4, 5]],
        patch_grid=(4, 4),
        sampled_frame_indices=[0, 1],
        save_overlay_video=True,
        save_side_by_side_video=False,
        multi_scale_overlay=True,
    )

    metadata = json.loads(
        (
            tmp_path
            / "exp"
            / "visualizations"
            / "autogaze_only"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["missing_scale_metadata"] is True
    assert metadata["available_scale_ids"] == []
    assert metadata["scale_color_map"] == {}


def test_autogaze_full_length_video_export_preserves_metadata(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")
    full_original = torch.linspace(0, 1, steps=4 * 3 * 32 * 32).reshape(4, 3, 32, 32)

    artifacts = visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0], [5]],
        patch_grid=(4, 4),
        sampled_frame_indices=[0, 3],
        original_frame_count=4,
        original_fps=7.5,
        full_original_video=full_original,
        save_overlay_video=True,
        save_side_by_side_video=True,
        video_export_mode="full_length",
        fps=2.0,
    )

    metadata = json.loads(
        (
            tmp_path
            / "exp"
            / "visualizations"
            / "autogaze_only"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert artifacts["overlay_video"].name == "autogaze_overlay_full_length.mp4"
    assert artifacts["side_by_side_video"].name == "autogaze_side_by_side_full_length.mp4"
    assert metadata["video_export_mode"] == "full_length"
    assert metadata["original_fps"] == 7.5
    assert metadata["output_fps"] == 7.5
    assert metadata["full_length_export"]["output_frame_count"] == 4
    assert metadata["full_length_export"]["processed_frame_indices"] == [0, 3]
    assert metadata["full_length_export"]["exact"] is True


def test_original_overlay_mapping_metadata_for_resize(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")
    original = torch.linspace(0, 1, steps=2 * 3 * 16 * 32).reshape(2, 3, 16, 32)

    artifacts = visualizer.export_autogaze_videos(
        _dummy_video(),
        selected_patch_indices=[[0], [5]],
        patch_grid=(4, 4),
        sampled_frame_indices=[0, 1],
        original_video=original,
        original_resolution=(16, 32),
        processed_resolution=(32, 32),
        save_overlay_video=False,
        save_side_by_side_video=True,
        comparison_layout="original_overlay",
        scaling_mode="resize",
    )

    metadata = json.loads(
        (
            tmp_path
            / "exp"
            / "visualizations"
            / "autogaze_only"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert artifacts["original_overlay_video"].name == "autogaze_original_overlay.mp4"
    assert metadata["coordinate_mapping"]["mode"] == "processed_to_original_affine"
    assert metadata["coordinate_mapping"]["scale_factors"] == {"x": 1.0, "y": 0.5}
    assert metadata["coordinate_mapping"]["mapping_exact"] is True


@pytest.mark.parametrize(
    ("scaling_mode", "original_resolution", "processed_resolution", "expected"),
    [
        ("fit_short_side", (16, 32), (8, 16), {"x": 2.0, "y": 2.0}),
        ("fit_long_side", (16, 32), (8, 16), {"x": 2.0, "y": 2.0}),
    ],
)
def test_original_overlay_mapping_metadata_for_fit_modes(
    tmp_path,
    scaling_mode: str,
    original_resolution: tuple[int, int],
    processed_resolution: tuple[int, int],
    expected: dict[str, float],
) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name=f"exp_{scaling_mode}")
    original = torch.linspace(0, 1, steps=2 * 3 * original_resolution[0] * original_resolution[1]).reshape(
        2, 3, original_resolution[0], original_resolution[1]
    )
    processed = torch.linspace(0, 1, steps=2 * 3 * processed_resolution[0] * processed_resolution[1]).reshape(
        2, 3, processed_resolution[0], processed_resolution[1]
    )

    visualizer.export_autogaze_videos(
        processed,
        selected_patch_indices=[[0], [1]],
        patch_grid=(2, 4),
        sampled_frame_indices=[0, 1],
        original_video=original,
        original_resolution=original_resolution,
        processed_resolution=processed_resolution,
        save_overlay_video=False,
        save_side_by_side_video=True,
        comparison_layout="original_processed_overlay",
        scaling_mode=scaling_mode,
    )

    metadata = json.loads(
        (
            tmp_path
            / f"exp_{scaling_mode}"
            / "visualizations"
            / "autogaze_only"
            / "metadata"
            / "visualization_video_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["coordinate_mapping"]["scale_factors"] == expected
    assert metadata["coordinate_mapping"]["mapping_exact"] is True


def test_original_overlay_unsupported_for_chop(tmp_path) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    with pytest.raises(NotImplementedError, match="chop"):
        visualizer.export_autogaze_videos(
            _dummy_video(),
            selected_patch_indices=[[0], [1]],
            patch_grid=(4, 4),
            original_video=_dummy_video(),
            save_side_by_side_video=True,
            comparison_layout="original_overlay",
            scaling_mode="chop",
        )


@pytest.mark.parametrize("export_mode", ["hold_last"])
def test_autogaze_video_export_unsupported_modes_raise(tmp_path, export_mode: str) -> None:
    visualizer = AutoGazeVisualizer(output_root=tmp_path, exp_name="exp")

    with pytest.raises(NotImplementedError, match=export_mode):
        visualizer.export_autogaze_videos(
            _dummy_video(),
            selected_patch_indices=[[0], [1]],
            patch_grid=(4, 4),
            video_export_mode=export_mode,
        )


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

    assert any(path.parent.name == "full_pipeline" or path.parent.parent.name == "full_pipeline" for path in paths)
    assert any(path.parent.name == "video_vqa" for path in paths)
    assert any(path.parent.name == "action_recognition" for path in paths)
