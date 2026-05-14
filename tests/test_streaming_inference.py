from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import infer_autogaze
import infer_full
from poc_infer_utils import (
    StreamingVisualizationSink,
    _iter_sample_stream_window,
    iter_stream_windows,
    load_config,
    validate_stream_window_memory,
)


def _cfg(name: str) -> Path:
    return ROOT / "configs" / "poc_inference" / name


def test_streaming_chunk_yields_bounded_windows() -> None:
    windows = list(
        iter_stream_windows(
            "dummy",
            frame_selection_mode="chunk",
            num_frames=4,
            frame_interval=1,
            max_windows=2,
            stream_window_size=4,
            stream_overlap=0,
            max_decode_frames=None,
            decode_backend="auto",
            decode_fps=None,
            dummy_frames=10,
            dummy_resolution=32,
        )
    )
    assert [window.frame_indices for window in windows] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert all(window.frames.shape == (4, 3, 32, 32) for window in windows)


def test_streaming_all_uses_chunked_processing() -> None:
    windows = list(
        iter_stream_windows(
            "dummy",
            frame_selection_mode="all",
            num_frames=4,
            frame_interval=1,
            max_windows=None,
            stream_window_size=4,
            stream_overlap=0,
            max_decode_frames=None,
            decode_backend="auto",
            decode_fps=None,
            dummy_frames=9,
            dummy_resolution=16,
        )
    )
    assert [window.frame_indices for window in windows] == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]


def test_streaming_interval_is_bounded() -> None:
    windows = list(
        iter_stream_windows(
            "dummy",
            frame_selection_mode="interval",
            num_frames=3,
            frame_interval=2,
            max_windows=1,
            stream_window_size=3,
            stream_overlap=0,
            max_decode_frames=None,
            decode_backend="auto",
            decode_fps=None,
            dummy_frames=8,
            dummy_resolution=16,
        )
    )
    assert windows[0].frame_indices == [0, 2, 4]


def test_streaming_sample_without_metadata_fails_clearly() -> None:
    frame_iter = ((idx, torch.zeros(3, 8, 8)) for idx in range(4))
    with pytest.raises(NotImplementedError, match="requires frame-count metadata"):
        list(
            _iter_sample_stream_window(
                frame_iter,
                window_size=4,
                max_windows=None,
                metadata={
                    "original_frame_count": None,
                    "original_fps": None,
                    "decode_backend": "mock",
                    "video_source_kind": "file",
                    "frame_count_from_metadata": False,
                },
            )
        )


def test_streaming_memory_guard_blocks_large_window() -> None:
    window = next(
        iter_stream_windows(
            "dummy",
            frame_selection_mode="chunk",
            num_frames=4,
            frame_interval=1,
            max_windows=1,
            stream_window_size=4,
            stream_overlap=0,
            max_decode_frames=None,
            decode_backend="auto",
            decode_fps=None,
            dummy_frames=4,
            dummy_resolution=32,
        )
    )
    with pytest.raises(RuntimeError, match="max_frames_in_memory"):
        validate_stream_window_memory(window, max_frames_in_memory=2, max_pixels_per_window=None)
    with pytest.raises(RuntimeError, match="max_pixels_per_window"):
        validate_stream_window_memory(window, max_frames_in_memory=None, max_pixels_per_window=1)


def test_infer_autogaze_streaming_dummy_outputs_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "autogaze_stream"
    summary = infer_autogaze.run(
        infer_autogaze.parse_args(
            [
                "--config",
                str(_cfg("A2_modified_siglip_nvila_on.yaml")),
                "--video-path",
                "dummy",
                "--output-dir",
                str(output_dir),
                "--frame-selection-mode",
                "chunk",
                "--num-frames",
                "4",
                "--stream-window-size",
                "4",
                "--max-stream-windows",
                "2",
                "--resolution",
                "64",
                "--no-progress",
            ]
        )
    )
    assert summary["metrics"]["video_read_mode"] == "streaming"
    assert summary["metrics"]["number_of_windows"] == 2
    streaming_metrics = json.loads((output_dir / "logs" / "streaming_metrics.json").read_text())
    assert streaming_metrics["processed_frame_count"] == 8
    selection = json.loads((output_dir / "autogaze" / "frame_selection_metadata.json").read_text())
    assert selection["video_read_mode"] == "streaming"


def test_infer_full_streaming_dummy_window_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "full_stream"
    summary = infer_full.run(
        infer_full.parse_args(
            [
                "--config",
                str(_cfg("A2_modified_siglip_nvila_on.yaml")),
                "--video-path",
                "dummy",
                "--query-text",
                "Describe the video.",
                "--output-dir",
                str(output_dir),
                "--frame-selection-mode",
                "chunk",
                "--num-frames",
                "4",
                "--stream-window-size",
                "4",
                "--max-stream-windows",
                "2",
                "--allow-dummy-weights",
                "--resolution",
                "64",
                "--no-progress",
            ]
        )
    )
    assert summary["metrics"]["streaming_full_pipeline_policy"] == "window_independent_generation"
    answer = json.loads((output_dir / "predictions" / "answer.json").read_text())
    assert len(answer["window_answers"]) == 2
    assert all(item["status"] == "dummy" for item in answer["window_answers"])


def test_streaming_full_length_export_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="full_length video export"):
        StreamingVisualizationSink(
            tmp_path,
            overlay_style="mask",
            overlay_alpha=0.35,
            multi_scale_overlay=True,
            show_patch_index=False,
            show_scale_label=False,
            metadata_placement="outside",
            info_panel_position="bottom",
            save_frame_images=False,
            save_overlay_video=True,
            save_side_by_side_video=False,
            save_scale_panel_video=False,
            video_fps=4.0,
            video_export_mode="full_length",
        )


def test_canonical_configs_declare_streaming_defaults() -> None:
    for name in (
        "A0_vanilla_siglip_nvila_off.yaml",
        "A1_modified_siglip_nvila_off.yaml",
        "A2_modified_siglip_nvila_on.yaml",
        "A3_vanilla_siglip_nvila_on.yaml",
    ):
        cfg = load_config(_cfg(name))
        assert cfg["video_input"]["read_mode"] == "streaming"
        assert cfg["streaming"]["full_pipeline_policy"] == "window_independent_generation"
        assert cfg["memory"]["fail_on_full_video_load"] is True
