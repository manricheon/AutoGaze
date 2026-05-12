from __future__ import annotations

import pytest

from autogaze_ext.data.frame_selector import FrameSelector, select_frame_windows


def test_sample_mode_frame_indices() -> None:
    result = select_frame_windows(
        original_frame_count=10,
        num_frames=4,
        frame_selection_mode="sample",
    )

    assert result.effective_mode == "sample"
    assert len(result.windows) == 1
    assert result.windows[0].frame_indices == [0, 3, 6, 9]


def test_chunk_mode_frame_windows() -> None:
    result = select_frame_windows(
        original_frame_count=10,
        num_frames=4,
        frame_selection_mode="chunk",
    )

    assert [window.frame_indices for window in result.windows] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert result.windows[-1].effective_num_frames == 2


def test_interval_mode_frame_indices() -> None:
    result = select_frame_windows(
        original_frame_count=20,
        num_frames=4,
        frame_selection_mode="interval",
        frame_interval=3,
    )

    assert len(result.windows) == 1
    assert result.windows[0].frame_indices == [0, 3, 6, 9]


def test_all_mode_aliases_to_chunk() -> None:
    result = select_frame_windows(
        original_frame_count=9,
        num_frames=4,
        frame_selection_mode="all",
    )

    assert result.mode == "all"
    assert result.effective_mode == "chunk"
    assert [window.frame_indices for window in result.windows] == [[0, 1, 2, 3], [4, 5, 6, 7], [8]]


def test_max_windows_behavior() -> None:
    result = select_frame_windows(
        original_frame_count=12,
        num_frames=4,
        frame_selection_mode="chunk",
        max_windows=2,
    )

    assert [window.frame_indices for window in result.windows] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_drop_last_behavior() -> None:
    result = select_frame_windows(
        original_frame_count=10,
        num_frames=4,
        frame_selection_mode="chunk",
        drop_last=True,
    )

    assert [window.frame_indices for window in result.windows] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_pad_last_behavior() -> None:
    result = select_frame_windows(
        original_frame_count=10,
        num_frames=4,
        frame_selection_mode="chunk",
        pad_last=True,
    )

    last = result.windows[-1]
    assert last.frame_indices == [8, 9, 9, 9]
    assert last.is_padded is True
    assert last.padded_frame_mask == [False, False, True, True]
    assert last.effective_num_frames == 2


def test_rejects_stride_style_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported frame_selection_mode"):
        FrameSelector(mode="stride", num_frames=4)
