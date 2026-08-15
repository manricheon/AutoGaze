"""Attach a Borissal v0.8 selection to Gemma 4's video vision tower
(`Gemma4VisionModel`, `transformers.models.gemma4`) by feeding only the
selected raw patches -- an encoder-stage prune, not a mask, and (unlike
`attach_qwen3vl.py`) with a straight-through gate wired all the way through,
since this is the path meant for joint training, not just eval.

WHY THIS WORKS AT ALL (verified against `transformers==5.5.0`'s
`modeling_gemma4.py` source directly, not blog claims). Gemma 4's vision
tower is coordinate-native almost everywhere:

  - `Gemma4VisionPatchEmbedder` embeds each patch independently (flatten +
    Linear, no cross-patch mixing) and looks up its position embedding by the
    patch's own (x, y) value in a table (`_position_embeddings`,
    modeling_gemma4.py:550-561) -- sequence order is irrelevant.
  - `Gemma4VisionEncoder` is full bidirectional attention over every valid
    (non-padding) patch, no window split (unlike Qwen's ViT), and its 2D RoPE
    (`Gemma4VisionRotaryEmbedding`) also keys off `pixel_position_ids`
    directly, not sequence index.
  - the padding sentinel is already `(-1, -1)` on `pixel_position_ids`
    (`Gemma4VisionModel.forward` docstring, :1897-1898) -- the exact
    convention `Selection.keep_coords` already uses.

So `patch_embedder` and `encoder` can be called on an arbitrary SPARSE subset
of patches with zero modification, and every kept patch keeps the exact
embedding/attention it would have had in the dense pass.

THE ONE PLACE THAT BREAKS (found by actually running it, not by reading):
`Gemma4VisionPooler._avg_pool_by_positions` derives the pooled grid's width
from `pixel_position_ids[..., 0].max() + 1` -- the BOUNDING BOX of whatever
patches are physically present in the call, not the true image width. Feed it
a scattered, non-contiguous subset (which is what any saliency-based selector
produces) and two different pooling cells can collide onto the same output
slot -- or the call raises outright: reproduced with two non-adjacent 3x3
cells on what should be a 2x2 pooled grid ("Class values must be smaller than
num_classes"). `_pool_true_grid` below is the same averaging math with the
true `Wc` passed in (known from Borissal's own `Selection.grid_thw` /
`to_gemma4_video_tokens`'s `pooled_grid`, never re-derived from the subset) --
checked against that crash case before this file was written; it no longer
raises and slots land where they should.

Consequence for the selector: pooling cells must be selected WHOLE, and
`to_gemma4_video_tokens` gets this for free rather than detecting violations,
because it REQUIRES v0.8 to run AT the pooled-cell grid
(`patch_size = 16 * pooling_kernel_size`, see its docstring) -- v0.8 has no
`score_coarsen`-style hard k-way tie (only a soft 2x2 mix, `alpha`), so at
`pooling_kernel_size=3` top-k almost never respects 3x3 cells on its own
(measured: 30 of 32 cells partial on one random draw at the fine 16px grid).
Running the selector one level coarser sidesteps the problem: each of v0.8's
own units IS one Gemma 4 pooling cell, so a partial cell is impossible by
construction -- at the cost of computing the selector's saliency signals
(A/D/N) at 48px resolution instead of its usual 16px.

The gate: because of that same grid choice, `st_gate(sel, aux)` already
returns exactly one value per kept CELL, in `Selection.keep_index`'s
t-major-ascending order -- no separate pooling of the gate is needed. That
order lines up with `pooled[pooler_mask]` for a structural reason, not luck:
`_pool_true_grid`'s kernel index is `wc + Wc*hc`, `Selection.keep_index`'s
flat index is `hc*Wc + wc` -- the same sum, so boolean-indexing `pooled` by
`pooler_mask` (row-major over `(T, Hc*Wc)`) and reading `st_gate`'s output
(row-major over the same grid) walk the kept cells in identical order.
`gated_video_features` asserts the two masks are equal before trusting that,
rather than assuming it. Each gate value is 1.0 to float32 precision (an
identical tensor minus its own `.detach()`, and a ratio of two equal
values) -- `torch.allclose`, not bit-exact `torch.equal`; the ~1e-7 gap
comes from unrelated floating-point rounding elsewhere in the chain, same as
`modeling_borissal_v08._demo_train`'s own check.

Batch size must be 1, matching `BorissalV08`'s own guard: per-frame kept-patch
counts differ by design (that's v0.8's rate-control allocation working as
intended), so a batch of clips would need per-row re-padding this file does
not implement.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .modeling_borissal_v08 import Selection, st_gate


@dataclass
class PrunedVideoPatches:
    """Gathered, cross-frame-padded inputs for `gated_video_features`."""

    pixel_values: torch.Tensor        # (T_grid, max_K, 3*patch_size**2)
    pixel_position_ids: torch.Tensor  # (T_grid, max_K, 2) long, (-1,-1) padded
    patch_row_index: torch.Tensor     # (T_grid, max_K) long, -1 padded -- kept for the gate remap


def build_pruned_video_patches(
    pixel_values_dense: torch.Tensor,       # (T_grid, N_dense, 3*patch_size**2), one video, no batch dim
    pixel_position_ids_dense: torch.Tensor,  # (T_grid, N_dense, 2)
    tokens: dict,                            # to_gemma4_video_tokens(...) output
) -> PrunedVideoPatches:
    """Gather the dense per-frame patch tensors down to `tokens["patch_row_index"]`.

    `pixel_values_dense` / `pixel_position_ids_dense` are the Gemma 4 video
    processor's own per-frame patch output for a single video (drop the
    `num_videos` batch dim before calling -- this file is B=1 only). Padding
    rows (`patch_row_index == -1`) gather row 0 and are then overwritten with
    zero pixels / `(-1, -1)` position -- `Gemma4Vision*` already treats that
    combination as inert padding.
    """
    patch_row_index = tokens["patch_row_index"]
    T_grid, max_K = patch_row_index.shape
    pad = patch_row_index < 0
    safe = patch_row_index.clamp(min=0)

    pixel_values = pixel_values_dense.gather(
        1, safe.unsqueeze(-1).expand(-1, -1, pixel_values_dense.shape[-1]))
    pixel_values = pixel_values.masked_fill(pad.unsqueeze(-1), 0.0)

    pixel_position_ids = pixel_position_ids_dense.gather(
        1, safe.unsqueeze(-1).expand(-1, -1, 2))
    pixel_position_ids = pixel_position_ids.masked_fill(pad.unsqueeze(-1), -1)

    return PrunedVideoPatches(
        pixel_values=pixel_values,
        pixel_position_ids=pixel_position_ids,
        patch_row_index=patch_row_index,
    )


def _pool_true_grid(hidden_states, pixel_position_ids, padding_positions, true_Hc, true_Wc, k):
    """`Gemma4VisionPooler._avg_pool_by_positions`'s averaging math, with the
    pooled grid width passed in rather than re-derived from which patches
    happen to be present -- see module docstring for why the original breaks
    on a sparse subset. `hidden_states`: (T, N, D); `pixel_position_ids`,
    `padding_positions`: (T, N, 2) / (T, N).
    """
    hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
    length = true_Hc * true_Wc
    clamped = pixel_position_ids.clamp(min=0)
    kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
    kernel_idxs = kernel_idxs[..., 0] + true_Wc * kernel_idxs[..., 1]   # x//k + Wc*(y//k)
    weights = F.one_hot(kernel_idxs.long(), length).to(hidden_states.dtype) / (k * k)
    pooled = weights.transpose(1, 2) @ hidden_states
    mask = torch.logical_not((weights == 0).all(dim=1))
    return pooled, mask


def gated_video_features(
    vision_tower,          # model.model.vision_tower  (Gemma4VisionModel)
    embed_vision,           # model.model.embed_vision  (Gemma4MultimodalEmbedder)
    pruned: PrunedVideoPatches,
    pooled_grid: tuple,     # (T_grid, Hc, Wc) from to_gemma4_video_tokens
    *,
    sel: Optional[Selection] = None,
    aux: Optional[dict] = None,
    rate_grad: bool = True,
) -> torch.Tensor:
    """Encoder-stage forward over a pruned patch subset, ending at the same
    point `Gemma4Model.get_video_features` does (ready for `masked_scatter`
    into the LLM's `inputs_embeds`) -- but with the true-grid pooling fix, and
    (when `sel`/`aux` are given) the straight-through gate multiplied in right
    before `embed_vision`, mirroring `vjepa2_sparse.py:66`'s insertion point.

    Pass `sel=None` (default) for pure inference -- no gate, cheapest path.
    Pass the `Selection`/`aux` that produced `pruned` to train through this
    encoder: `loss.backward()` then reaches every `BorissalV08` knob that
    `st_gate` covers, exactly as `test_v08_st_gate_knob_coverage` checks for
    the toy encoder.
    """
    _, Hc, Wc = pooled_grid
    k = vision_tower.config.pooling_kernel_size
    pixel_values, pixel_position_ids = pruned.pixel_values, pruned.pixel_position_ids
    padding_positions = (pixel_position_ids == -1).all(dim=-1)

    hidden_states = vision_tower.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
    encoded = vision_tower.encoder(
        inputs_embeds=hidden_states,
        attention_mask=~padding_positions,
        pixel_position_ids=pixel_position_ids,
    ).last_hidden_state

    pooled, pooler_mask = _pool_true_grid(encoded, pixel_position_ids, padding_positions, Hc, Wc, k)
    pooled = pooled * (vision_tower.config.hidden_size ** 0.5)   # Gemma4VisionPooler.forward's root_hidden_size scale
    pooled = pooled[pooler_mask]                                  # (n_kept_cells_total, hidden) -- padding stripped

    if vision_tower.config.standardize:
        pooled = (pooled - vision_tower.std_bias) * vision_tower.std_scale

    if sel is not None:
        T_grid, Hc_sel, Wc_sel = (int(x) for x in sel.grid_thw[0].tolist())
        expect_mask = sel.keep_mask.reshape(T_grid, Hc_sel * Wc_sel)
        if expect_mask.shape != pooler_mask.shape or not torch.equal(expect_mask, pooler_mask):
            raise RuntimeError(
                "gate and feature pooling masks diverged -- sel/aux must be the exact Selection "
                "that produced `pruned` via to_gemma4_video_tokens")
        gate = st_gate(sel, aux, rate_grad=rate_grad)[0]            # (n_kept_cells_total,), ~1.0 forward
        pooled = pooled * gate.unsqueeze(-1)                        # <- the straight-through multiply

    return embed_vision(inputs_embeds=pooled)
