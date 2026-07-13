"""Bridges from Borissal's grid_thw-native `Selection` to specific consumers.

Borissal's native output (see modeling_borissal.Selection) is intentionally
NOT the legacy AutoGaze `gazing_pos` dict contract -- it is grid_thw-native so
it attaches directly to encoders that already think in (t, h, w) token grids
(V-JEPA2's conv3d-tubelet + 3D-RoPE transformer, Qwen-VL's `video_grid_thw` +
flattened tokens). Each adapter below does the minimal reshaping for one
specific consumer; add new adapters here rather than changing the selector's
native output.
"""

from typing import Optional

import torch

from .modeling_borissal import Selection


def to_canonical_keep_indices(selection: Selection) -> list:
    """Bridge to the canonical `idx = t*N + n` (n = row*W_grid + col) flat
    keep-index-per-video convention used downstream (e.g. a
    `result["keep_indices_per_video"] = [...]` processor step feeding a
    V-JEPA2/Qwen-VL-style sparse encoder). That convention requires each
    video's list to be sorted ascending by (frame, row, col) -- i.e. strictly
    ascending `idx` -- since the encoder's mask-gather + RoPE position
    recovery depends on that order to map each kept token back to its
    original (t, row, col).

    Borissal's `Selection.keep_index` already satisfies this: its flatten
    order is t-major then row-major (`t*(H_grid*W_grid) + h*W_grid + w`,
    exactly the `t*N + n` formula), and the packer that builds it
    (`_pack_gazing_mask`) preserves ascending order among kept indices. So
    this adapter does no reordering -- it only strips the batch's `-1`
    padding, per video, since the downstream convention is a plain
    variable-length list rather than a padded tensor.

    Returns a list of length B, each element a 1-D ascending `LongTensor`
    (length `num_keep[b]`) of that video's kept flat indices.
    """
    return [
        selection.keep_index[b][selection.keep_index[b] >= 0]
        for b in range(selection.keep_index.shape[0])
    ]


def to_vjepa2(selection: Selection) -> dict:
    """Minimal bridge for a V-JEPA2-style encoder that gathers tokens by flat index
    right after the conv3d tubelet embedding and before the transformer.

    Assumes the encoder's own token flatten order is t-major (t, h, w), matching
    Borissal's flatten order. If a target V-JEPA2 implementation flattens
    differently, remap `keep_index`/`keep_coords` here -- do not change
    Selection's native order, since other adapters rely on it too.
    """
    return {
        "keep_index": selection.keep_index,      # (B, K) long, -1 padded
        "keep_coords": selection.keep_coords,     # (B, K, 3) long, -1 padded
        "grid_thw": selection.grid_thw,           # (B, 3) long
        "num_keep": selection.num_keep,           # (B,) long
    }


def to_autogaze_gazing_info(selection: Selection, scale: int, tubelet_size: int) -> dict:
    """Optional compatibility bridge to the legacy AutoGaze gaze-model output
    contract (autogaze/models/autogaze/autogaze.py forward return dict), for
    running Borissal's selection through the existing VideoMAE task/trainer as
    a sanity check.

    Only supports the case where every tubelet keeps the same number of
    patches (Borissal's default `per_frame_allocation="uniform"`), since the
    legacy contract's `num_gazing_each_frame` is shared across the batch.
    """
    B, T_grid, N_pf = selection.per_frame_keep.shape[0], selection.grid_thw[0, 0].item(), selection.grid_thw[0, 1].item() * selection.grid_thw[0, 2].item()

    per_frame_keep = selection.per_frame_keep
    if not torch.equal(per_frame_keep, per_frame_keep[0:1].expand_as(per_frame_keep)):
        raise NotImplementedError(
            "to_autogaze_gazing_info only supports a per-frame keep count shared "
            "across the batch (Borissal's 'uniform' per_frame_allocation); got "
            "differing counts per batch instance (e.g. 'proportional' allocation)."
        )

    num_gazing_each_frame = per_frame_keep[0].to(torch.long)  # (T_grid,)
    gazing_mask = [selection.keep_mask.view(B, T_grid, N_pf)]

    return {
        "gazing_pos": selection.keep_index,
        "if_padded_gazing": selection.keep_index == -1,
        "num_gazing_each_frame": num_gazing_each_frame,
        "gazing_mask": gazing_mask,
        "scales": [scale],
        "frame_sampling_rate": tubelet_size,
        "num_vision_tokens_each_frame": N_pf,
    }
