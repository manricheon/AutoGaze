"""Unit tests for the OneVision (per-frame SigLIP) attach adapter.

Verifies `to_onevision_frame_indices` maps Borissal's tubelet-native
`Selection` onto a per-frame SigLIP/OneVision patch order: spatial index
`h*W_grid + w` is passed through unchanged (1:1 raster order), and each
tubelet's spatial selection is duplicated to both of its frames (that is
what selecting a tubelet means physically). No encoder is loaded here.
"""

from types import SimpleNamespace

import pytest
import torch

from autogaze.models.borissal.adapters import to_onevision_frame_indices


def _sel(keep_index, grid_thw, per_frame_keep):
    return SimpleNamespace(
        keep_index=torch.tensor(keep_index),
        grid_thw=torch.tensor(grid_thw),
        per_frame_keep=torch.tensor(per_frame_keep),
    )


def test_onevision_frame_index_mapping():
    # 2 tubelets, 2x2 fine grid (N_pf = 4), tubelet_size 2 -> 4 frames.
    # tubelet 0 keeps fine cells {1, 2}; tubelet 1 keeps {0, 3}.
    sel = _sel([[0 * 4 + 1, 0 * 4 + 2, 1 * 4 + 0, 1 * 4 + 3]], [[2, 2, 2]], [[2, 2]])
    out = to_onevision_frame_indices(sel, tubelet_size=2)

    # frames 0,1 inherit tubelet 0 -> spatial {1,2}; frames 2,3 -> tubelet 1 -> {0,3}
    assert out["frame_keep_index"].tolist() == [[[1, 2], [1, 2], [0, 3], [0, 3]]]
    assert out["num_frames"] == 4
    assert out["num_tokens_each_frame"] == 4
    assert out["num_keep_each_frame"].tolist() == [2, 2, 2, 2]
    assert out["grid_hw"] == (2, 2)


def test_onevision_frame_mask_matches_index():
    sel = _sel([[1, 2, 4, 7]], [[2, 2, 2]], [[2, 2]])
    out = to_onevision_frame_indices(sel, tubelet_size=2)
    fm = out["frame_mask"]
    assert fm.shape == (1, 4, 4)
    # frame 0/1: cells {1,2}; frame 2/3: cells {0,3}
    assert fm[0].tolist() == [
        [False, True, True, False],
        [False, True, True, False],
        [True, False, False, True],
        [True, False, False, True],
    ]
    # index representation agrees with the mask (nonzero, ascending)
    for f in range(4):
        assert out["frame_keep_index"][0, f].tolist() == fm[0, f].nonzero().flatten().tolist()


def test_onevision_index_bounds_and_ascending():
    sel = _sel([[1, 2, 4, 7]], [[2, 2, 2]], [[2, 2]])
    out = to_onevision_frame_indices(sel, tubelet_size=2)
    idx = out["frame_keep_index"]
    assert (idx >= 0).all() and (idx < out["num_tokens_each_frame"]).all()
    assert (idx[..., 1:] > idx[..., :-1]).all(), "per-frame indices must be ascending"


def test_onevision_tubelet_pair_frames_identical():
    # a 3-tubelet case at a 3x3 grid, keep 3 per tubelet
    N = 9
    keep = [[0 * N + 1, 0 * N + 4, 0 * N + 8,
             1 * N + 0, 1 * N + 3, 1 * N + 6,
             2 * N + 2, 2 * N + 5, 2 * N + 7]]
    sel = _sel(keep, [[3, 3, 3]], [[3, 3, 3]])
    out = to_onevision_frame_indices(sel, tubelet_size=2)  # 6 frames
    idx = out["frame_keep_index"][0]
    for tub in range(3):
        f0, f1 = 2 * tub, 2 * tub + 1
        assert idx[f0].tolist() == idx[f1].tolist(), "both frames of a tubelet share the mask"


def test_onevision_guard_padded():
    sel = _sel([[1, 2, -1, -1]], [[2, 2, 2]], [[1, 1]])
    with pytest.raises(NotImplementedError):
        to_onevision_frame_indices(sel, tubelet_size=2)


def test_onevision_guard_nonuniform_batch():
    # two batch rows with differing per-tubelet keep counts
    sel = _sel([[1, 2, 4, 7], [0, 3, 5, 6]], [[2, 2, 2], [2, 2, 2]], [[2, 2], [1, 3]])
    with pytest.raises(NotImplementedError):
        to_onevision_frame_indices(sel, tubelet_size=2)


def test_onevision_guard_spatial_merge_unsupported():
    sel = _sel([[1, 2, 4, 7]], [[2, 2, 2]], [[2, 2]])
    with pytest.raises(NotImplementedError):
        to_onevision_frame_indices(sel, tubelet_size=2, spatial_merge_size=2)
