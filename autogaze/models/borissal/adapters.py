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

from .modeling_borissal import Selection, _pack_gazing_mask


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


def to_videomae_gazing_info(
    selection: Selection,
    tubelet_size: int,
    scales: tuple = (32, 64, 112, 224),
    patch_size: int = 16,
) -> dict:
    """Bridge to the ORIGINAL AutoGaze VideoMAE reconstruction task
    (autogaze/tasks/video_mae_reconstruction), whose released checkpoint is
    multi-scale and per-frame:

    - each frame contributes sum((s/patch)^2 for s in scales) tokens (265
      for 32+64+112+224), ordered scale-ascending, so the finest-scale
      (224 -> 14x14=196) block starts at offset sum of the coarser counts (69);
    - `gazing_pos` holds GLOBAL flat indices `frame*tokens_per_frame + local`,
      frame-major ascending;
    - the task is per-frame (frame_sampling_rate must be 1), while Borissal
      selects per TUBELET -- each tubelet's spatial selection is therefore
      duplicated to both of its frames (that is exactly what selecting a
      tubelet means physically).

    Borissal's selection maps onto the finest scale only; all coarser-scale
    tokens stay unselected (within the checkpoint's training distribution --
    its per-scale allocation was Dirichlet-sampled and could concentrate on
    one scale). Requires an unpadded, batch-uniform per-tubelet keep count
    (uniform allocation).
    """
    if (selection.keep_index < 0).any():
        raise NotImplementedError("to_videomae_gazing_info requires unpadded selection (uniform allocation)")
    B = selection.keep_index.shape[0]
    T_grid, H_grid, W_grid = (int(x) for x in selection.grid_thw[0].tolist())
    N_pf = H_grid * W_grid
    fine = max(scales)
    if N_pf != (fine // patch_size) ** 2:
        raise ValueError(
            f"selection grid {H_grid}x{W_grid} does not match the finest VideoMAE scale "
            f"{fine} (expected {(fine // patch_size)}x{(fine // patch_size)}; run the selector at scale={fine})")
    per_frame_keep = selection.per_frame_keep
    if not torch.equal(per_frame_keep, per_frame_keep[0:1].expand_as(per_frame_keep)):
        raise NotImplementedError("requires a batch-uniform per-tubelet keep count (uniform allocation)")

    tokens_per_frame = sum((s // patch_size) ** 2 for s in scales)
    fine_offset = tokens_per_frame - N_pf
    num_frames = T_grid * tubelet_size

    kept = selection.keep_index                      # (B, K) ascending t*N_pf + n
    t = kept // N_pf                                 # (B, K) tubelet index
    n = kept % N_pf                                  # (B, K) fine-scale spatial index
    j = torch.arange(tubelet_size, device=kept.device)
    frames = t.unsqueeze(-1) * tubelet_size + j      # (B, K, tub)
    gazing = frames * tokens_per_frame + fine_offset + n.unsqueeze(-1)
    gazing = gazing.reshape(B, -1).sort(dim=-1).values  # global ascending = frame-major

    num_gazing_each_frame = per_frame_keep[0].repeat_interleave(tubelet_size).to(torch.long)
    return {
        "gazing_pos": gazing,
        "num_gazing_each_frame": num_gazing_each_frame,          # (num_frames,)
        "if_padded_gazing": torch.zeros_like(gazing, dtype=torch.bool),
        "frame_sampling_rate": 1,
        "num_vision_tokens_each_frame": tokens_per_frame,
        "num_frames": num_frames,
    }


def to_onevision_frame_indices(
    selection: Selection,
    tubelet_size: int,
    spatial_merge_size: int = 1,
) -> dict:
    """Bridge to a LLaVA-OneVision-2 style PER-FRAME vision path (each frame
    encoded independently by a SigLIP tower into H_grid*W_grid raster-order
    patch tokens, later merged/pooled by the Qwen side).

    Borissal's within-tubelet spatial index `n = h*W_grid + w` is the SAME
    raster order a SigLIP tower emits per frame, so it passes through 1:1 with
    NO spatial remap (proven against the SigLIP2 semantic gate's gather path).
    The only bridging needed is temporal: Borissal decides once per TUBELET,
    so each tubelet's spatial selection is DUPLICATED to both of its frames
    (that is what selecting a tubelet means physically) -- the SigLIP grid
    must therefore match `H_grid == W_grid` at the encoder's own patch size.

    Interception is PRE-MERGE: indices address the encoder-native fine grid
    (e.g. 27x27 for patch14-384). Mapping the fine grid through OneVision/Qwen's
    2x2 spatial merge into the LLM token stream is a separate, lossy design step
    (it would force whole-2x2-superpatch selection) and is intentionally out of
    scope here; `spatial_merge_size != 1` raises so callers can't silently
    assume the merged-space remap exists.

    Requires an unpadded, batch-uniform, tubelet-uniform keep count (Borissal's
    default `per_frame_allocation="uniform"`), so the per-frame index tensor has
    a well-defined constant width `k`.

    Returns a dict:
      - `frame_keep_index` (B, num_frames, k) long -- ascending per-frame
        spatial indices into that frame's H_grid*W_grid grid
      - `frame_mask` (B, num_frames, N_pf) bool
      - `num_keep_each_frame` (num_frames,) long -- constant `k`
      - `num_tokens_each_frame` int -- N_pf = H_grid*W_grid
      - `num_frames` int -- T_grid * tubelet_size
      - `grid_hw` (H_grid, W_grid) tuple
    """
    if spatial_merge_size != 1:
        raise NotImplementedError(
            "to_onevision_frame_indices intercepts PRE-merge tokens only; the "
            "fine->merged (2x2 spatial_merge) index remap is future work -- see docstring")
    if (selection.keep_index < 0).any():
        raise NotImplementedError(
            "to_onevision_frame_indices requires unpadded selection (uniform allocation)")
    per_frame_keep = selection.per_frame_keep                # (B, T_grid)
    if not torch.equal(per_frame_keep, per_frame_keep[0:1].expand_as(per_frame_keep)):
        raise NotImplementedError(
            "requires a batch-uniform per-tubelet keep count (uniform allocation)")
    k0 = per_frame_keep[0, 0]
    if not torch.equal(per_frame_keep[0], k0.expand_as(per_frame_keep[0])):
        raise NotImplementedError(
            "requires a constant per-tubelet keep count across tubelets "
            "(uniform allocation) for a fixed per-frame index width")

    B = selection.keep_index.shape[0]
    T_grid, H_grid, W_grid = (int(x) for x in selection.grid_thw[0].tolist())
    N_pf = H_grid * W_grid
    k = int(k0)
    num_frames = T_grid * tubelet_size

    kept = selection.keep_index                              # (B, K) ascending t*N_pf + n
    t = kept // N_pf                                         # (B, K) tubelet index
    n = kept % N_pf                                          # (B, K) per-frame spatial index
    # Ascending t-major order + constant k means each contiguous k-block is one
    # tubelet; verify before relying on the reshape.
    t_tub = t.reshape(B, T_grid, k)
    expected_t = torch.arange(T_grid, device=kept.device).view(1, T_grid, 1)
    if not torch.equal(t_tub, expected_t.expand_as(t_tub)):
        raise AssertionError("keep_index is not contiguous t-major with constant k per tubelet")
    n_tub = n.reshape(B, T_grid, k)                          # (B, T_grid, k) ascending within tubelet

    # duplicate each tubelet's spatial selection to its `tubelet_size` frames
    frame_keep_index = n_tub.repeat_interleave(tubelet_size, dim=1)  # (B, num_frames, k)
    frame_mask = torch.zeros(B, num_frames, N_pf, dtype=torch.bool, device=kept.device)
    frame_mask.scatter_(2, frame_keep_index, True)

    return {
        "frame_keep_index": frame_keep_index,
        "frame_mask": frame_mask,
        "num_keep_each_frame": torch.full((num_frames,), k, dtype=torch.long, device=kept.device),
        "num_tokens_each_frame": N_pf,
        "num_frames": num_frames,
        "grid_hw": (H_grid, W_grid),
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


def to_qwen3vl_video_tokens(
    selection: Selection,
    spatial_merge_size: int = 2,
    partial_blocks: str = "strict",
) -> dict:
    """Bridge to the Qwen3-VL / Qwen3.5 video path (`qwen3_vl`, `qwen3_5`).

    This family's video geometry coincides with Borissal's by construction:
    `patch_size=16`, `temporal_patch_size=2`, `spatial_merge_size=2`, so
    `video_grid_thw == Selection.grid_thw` whenever the selector runs at
    `patch_size=16, tubelet_size=2`. No resampling, no odd-grid problem (the
    OneVision `so400m-patch14` 27x27 headache does not arise at patch16).

    ORDERING (verified against `Qwen3VLVideoProcessor`, not assumed): the
    processor's `permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)` emits patches grouped
    `spatial_merge_size**2` at a time, one group per 2x2 spatial block, with the
    groups raster-ordered over `(t, h//m, w//m)`. The vision tower's patch
    merger folds each consecutive group into one LLM token, so

        llm_token_index = t*(Hm*Wm) + (h//m)*Wm + (w//m),   Hm=H/m, Wm=W/m
        qwen_patch_index = llm_token_index*m**2 + (h%m)*m + (w%m)

    Borissal's own flat order is `t*(H*W) + h*W + w`, so the two differ and the
    remap below is required -- passing Borissal indices straight into Qwen would
    silently scramble the selection.

    PARTIAL BLOCKS. A merged token is one 2x2 patch block, so selection is only
    lossless in this space when whole blocks are kept. Cube coherence
    (`score_coarsen=2`) makes every block's score identical, which is necessary
    but NOT sufficient: top-k stops when the budget runs out, so a per-unit
    budget that is not a multiple of `m**2` cuts its last block. Measured at
    ratio 0.25, 16f, 384: v0.5 and the v0.6 DEFAULT (global allocation, budget
    1152 = 288*4) give 0 partial blocks, while v0.6 with
    `per_frame_allocation="uniform"` gives 8 -- one per tubelet, because the
    keyframe allocation boost hands out per-tubelet counts like 211/121 that are
    not multiples of 4. v0.3 (`score_coarsen=1`) has no block structure at all.
    `partial_blocks`:
      - `"strict"` (default): raise if any block is partially selected -- use
        with `score_coarsen=2` so the mapping is exact.
      - `"any"`: keep the merged token if >=1 of its patches was selected
        (over-keeps: the realised token budget EXCEEDS the requested ratio).
      - `"full"`: keep the merged token only if all m**2 patches were selected
        (under-keeps: the realised budget is BELOW the requested ratio).
    The realised count is always reported as `num_keep_tokens` -- report it, do
    not assume `ratio * n_tokens`.

    Returns a dict:
      - `keep_token_index` (B, K_tok) long -- ascending LLM vision-token indices, -1 padded
      - `token_keep_mask` (B, T_grid*Hm*Wm) bool
      - `token_coords` (B, K_tok, 3) long -- (t, hm, wm) per kept token, -1 padded (for mrope)
      - `qwen_patch_index` (B, K_patch) long -- ascending indices into the
        processor's `pixel_values_videos` patch rows, -1 padded (pre-encoder pruning)
      - `num_keep_tokens` (B,) long
      - `merged_grid` (T_grid, Hm, Wm) tuple
      - `num_tokens_total` int -- T_grid*Hm*Wm (the unpruned LLM vision token count)
      - `n_partial_blocks` int -- how many blocks were partially selected (0 under `score_coarsen=2`)
    """
    if partial_blocks not in ("strict", "any", "full"):
        raise ValueError(f"partial_blocks must be strict|any|full, got {partial_blocks!r}")
    m = int(spatial_merge_size)
    if m < 1:
        raise ValueError(f"spatial_merge_size must be >= 1, got {m}")

    T_grid, H_grid, W_grid = (int(x) for x in selection.grid_thw[0].tolist())
    if H_grid % m or W_grid % m:
        raise ValueError(
            f"grid {H_grid}x{W_grid} must be divisible by spatial_merge_size={m}; "
            f"run the selector at a scale whose patch grid is a multiple of {m}")
    B = selection.keep_mask.shape[0]
    Hm, Wm = H_grid // m, W_grid // m
    device = selection.keep_mask.device

    # (B, T, Hm, m, Wm, m) -> per-block selected-patch count (B, T, Hm, Wm)
    patch_mask = selection.keep_mask.reshape(B, T_grid, Hm, m, Wm, m)
    per_block = patch_mask.sum(dim=(3, 5))
    n_partial = int(((per_block > 0) & (per_block < m * m)).sum())
    if partial_blocks == "strict" and n_partial:
        raise ValueError(
            f"{n_partial} of {T_grid * Hm * Wm} 2x2 blocks are only partially selected, so the "
            f"selection cannot be expressed exactly in Qwen's merged-token space. Use a preset "
            f"with score_coarsen={m} (v0.5/v0.6), or pass partial_blocks='any'/'full' and report "
            f"the realised token count.")
    token_mask = (per_block > 0) if partial_blocks in ("strict", "any") else (per_block == m * m)
    token_mask = token_mask.reshape(B, T_grid * Hm * Wm)

    keep_token_index, _ = _pack_gazing_mask(token_mask)
    num_keep_tokens = token_mask.sum(dim=-1)

    # (t, hm, wm) per kept token, -1 on padding (mrope needs the 3D coordinate)
    valid = keep_token_index >= 0
    safe = keep_token_index.clamp_min(0)
    t_i = safe // (Hm * Wm)
    hm_i = (safe % (Hm * Wm)) // Wm
    wm_i = safe % Wm
    token_coords = torch.stack([t_i, hm_i, wm_i], dim=-1)
    token_coords = token_coords.masked_fill(~valid.unsqueeze(-1), -1)

    # Patch rows in the PROCESSOR's order (pre-encoder pruning). Derived from the
    # patch mask itself rather than from the tokens, so it stays correct under the
    # 'any'/'full' policies where blocks are not whole.
    qwen_order_mask = patch_mask.permute(0, 1, 2, 4, 3, 5).reshape(B, -1)
    if partial_blocks != "strict":
        # keep the patch iff its BLOCK survived, so kept rows stay in intact
        # m**2 groups (the patch merger consumes consecutive groups)
        blk = token_mask.reshape(B, T_grid * Hm * Wm, 1).expand(B, T_grid * Hm * Wm, m * m)
        qwen_order_mask = blk.reshape(B, -1)
    qwen_patch_index, _ = _pack_gazing_mask(qwen_order_mask)

    return {
        "keep_token_index": keep_token_index,
        "token_keep_mask": token_mask,
        "token_coords": token_coords,
        "qwen_patch_index": qwen_patch_index,
        "num_keep_tokens": num_keep_tokens,
        "merged_grid": (T_grid, Hm, Wm),
        "num_tokens_total": T_grid * Hm * Wm,
        "n_partial_blocks": n_partial,
    }


def to_gemma4_video_tokens(selection: Selection, pooling_kernel_size: int = 3) -> dict:
    """Bridge to Gemma 4's video vision tower (`Gemma4VisionModel`, `transformers`
    `models/gemma4`).

    REQUIRES the selector to have run at the POOLED grid directly --
    `patch_size = 16 * pooling_kernel_size` (48 by default: Gemma 4's raw
    `patch_size=16` times its default `pooling_kernel_size=3`) and
    `tubelet_size=1` (Gemma 4 encodes each video frame independently). This is
    not the Qwen pattern (fine 16px grid + `score_coarsen` cube-tie + a
    reshape here to detect partial blocks): v0.8 has no `score_coarsen`
    knob -- its own coarsening (`alpha`, `_soft_coarsen`) is a soft 2x2 mix,
    not a hard k-way tie for arbitrary k -- so at pool_k=3, top-k almost never
    respects 3x3 cell boundaries on its own (measured: 30 of 32 cells partial
    on a random draw). Running the selector one level coarser sidesteps the
    problem instead of detecting it: each of v0.8's own units IS one Gemma 4
    pooling cell, so partial-cell selection is impossible by construction, at
    the cost of computing the selector's saliency signals (A/D/N) at 48px
    resolution instead of its usual 16px.

    Gemma 4 has no merge-before-encoder step like Qwen: pooling happens AFTER
    the 16-layer encoder (`Gemma4VisionPooler`), and it always divides by
    `pooling_kernel_size**2` regardless of how many patches in a cell
    survived (`_avg_pool_by_positions`, modeling_gemma4.py:603) -- a partial
    cell would be a silently biased-low average, not a token-count problem
    like Qwen's partial merge block. Moot here since cells can't be partial.

    Returns a dict:
      - `patch_row_index` (T_grid, max_K) long -- ascending raw-16px-patch row
        index per frame (row-major `h*W_raw+w`, matching Gemma 4's own
        `convert_video_to_patches` + position-id `meshgrid(..., indexing="xy")`
        order -- verified against `video_processing_gemma4.py`, not assumed),
        -1 padded to the batch's per-frame max kept-patch count. Every frame's
        own count is an exact multiple of `pooling_kernel_size**2` (whole
        cells, expanded from v0.8's own kept units), so the -1 padding is
        always whole extra cells.
      - `cell_keep_mask` (T_grid, Hc, Wc) bool -- which pooled cells survived
        (== `Selection.keep_mask` reshaped; Hc, Wc are v0.8's own grid here)
      - `n_kept_per_frame` (T_grid,) long
      - `pooled_grid` (T_grid, Hc, Wc) tuple -- passed to `attach_gemma4.py`
        because `Gemma4VisionPooler`'s own kernel-index math re-derives the
        grid width from whichever patches are physically present in a call,
        which breaks (collides or raises) on a scattered, non-contiguous
        subset -- exactly what a saliency-based selection produces. Passing
        the true grid instead of re-deriving it is the fix; see
        `attach_gemma4.py`'s module docstring.
    """
    k = int(pooling_kernel_size)
    if k < 1:
        raise ValueError(f"pooling_kernel_size must be >= 1, got {k}")

    T_grid, Hc, Wc = (int(x) for x in selection.grid_thw[0].tolist())
    B = selection.keep_mask.shape[0]
    if B != 1:
        raise NotImplementedError("to_gemma4_video_tokens: B=1 only (matches BorissalV08's own batch guard)")

    cell_keep = selection.keep_mask.reshape(B, T_grid, Hc, Wc)   # v0.8's own grid == pooled grid
    n_kept_per_frame = cell_keep.reshape(B, T_grid, -1).sum(dim=-1)[0]

    # Each kept (hc, wc) cell expands to its k*k raw-16px-patch rows. Row-major
    # h*W_raw+w with W_raw = Wc*k, so (Hc, k, Wc, k) reshapes straight to
    # (Hc*k, Wc*k) = (H_raw, W_raw) with no permute -- Gemma 4's pooling reads
    # positions, not sequence order, so the k*k rows don't need to stay
    # contiguous, but the row-major INDEX formula must still match the
    # processor's own patch ordering.
    raw_mask = cell_keep.unsqueeze(3).unsqueeze(5).expand(B, T_grid, Hc, k, Wc, k)
    raw_mask = raw_mask.reshape(B, T_grid, Hc * k * Wc * k)[0]   # (T_grid, H_raw*W_raw)

    patch_row_index, _ = _pack_gazing_mask(raw_mask)              # (T_grid, max_K), -1 padded

    return {
        "patch_row_index": patch_row_index,
        "cell_keep_mask": cell_keep[0],
        "n_kept_per_frame": n_kept_per_frame,
        "pooled_grid": (T_grid, Hc, Wc),
    }
