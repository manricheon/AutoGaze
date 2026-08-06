"""Attach a Borissal selection to a Qwen3-VL / Qwen3.5 video path by actually
DROPPING the unselected vision tokens (not masking them).

EVAL-ONLY. This module is deliberately outside the traced/exported core: it
imports nothing at module scope beyond torch, but it pokes at `transformers`
internals (`get_video_features`, `compute_3d_position_ids`, the vision tower's
sub-modules) and is never part of `export_borissal_check.py`.

Two pruning stages, and the difference matters when interpreting results:

  `prune_stage="llm"` (default)
      Run the FULL vision tower, then drop token rows before the language model.
      Saves LLM compute only, and every surviving token has already attended to
      the dropped ones inside the ViT -- information LEAKS in. Answers "can the
      LLM describe the clip from these tokens", not "was the discarded pixel
      content unnecessary".

  `prune_stage="encoder"`
      Feed only the selected patches to the vision tower. Leak-free and saves
      encoder compute too -- this is the configuration that actually tests
      AutoGaze's claim (fewer patches in, no information loss). Requires whole
      2x2 blocks, since the patch merger folds consecutive `merge**2` rows.

Both paths reproduce the model's own forward exactly when nothing is dropped;
`tests/test_borissal_attach_qwen3vl.py` asserts logit equality against the
vanilla forward as the primary correctness gate.

Batch size must be 1: a pruned batch has per-row sequence lengths, which would
need re-padding and a re-derived attention mask for no analytical gain.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PrunedInputs:
    """Assembled inputs for `pruned_forward`, plus the accounting to report."""

    inputs_embeds: torch.Tensor              # (1, S', d)
    attention_mask: torch.Tensor             # (1, S')
    position_ids: torch.Tensor               # (3 or 4, 1, S') mrope
    visual_pos_masks: torch.Tensor           # (1, S') bool -- kept vision positions
    deepstack_visual_embeds: Optional[list]  # list of (n_vision, d)
    input_ids: torch.Tensor                  # (1, S') pruned ids (for labels/debug)
    seq_keep: torch.Tensor                   # (1, S) bool over the DENSE sequence
    n_vision_tokens: int                     # kept vision tokens
    n_vision_tokens_dense: int               # vision tokens before pruning
    prune_stage: str
    meta: dict = field(default_factory=dict)


def _video_token_positions(input_ids: torch.Tensor, video_token_id: int) -> torch.Tensor:
    return input_ids[0] == video_token_id


def _pruned_vision_forward(visual, pixel_values, grid_thw, qwen_patch_index,
                           patches_per_slice: torch.Tensor):
    """Vision tower over a SUBSET of patch rows (leak-free pruning).

    Mirrors `Qwen3VLVisionModel.forward` with three substitutions:
      - `patch_embed` is a Conv3d whose kernel == stride, so it acts per patch
        row; indexing rows before it is exact.
      - both position signals (`fast_pos_embed_interpolate`, `rot_pos_emb`) are
        computed for the FULL grid then gathered, so every kept patch keeps the
        absolute position it had in the dense grid.
      - `cu_seqlens` is rebuilt from the kept patch count per temporal slice
        (attention segments are per-slice in the dense path too).
    The patch merger folds consecutive `merge**2` rows, so `qwen_patch_index`
    must list whole blocks in order -- guaranteed by `to_qwen3vl_video_tokens`.
    """
    hidden_states = visual.patch_embed(pixel_values[qwen_patch_index])

    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)[qwen_patch_index]
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = visual.rot_pos_emb(grid_thw)[qwen_patch_index]
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = patches_per_slice.to(device=hidden_states.device).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(visual.blocks):
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings)
        if layer_num in visual.deepstack_visual_indexes:
            merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)]
            deepstack_feature_lists.append(merger(hidden_states))
    return visual.merger(hidden_states), deepstack_feature_lists


def build_pruned_inputs(
    model,
    inputs: dict,
    keep_token_index: Optional[torch.Tensor] = None,
    *,
    prune_stage: str = "encoder",
    qwen_patch_index: Optional[torch.Tensor] = None,
) -> PrunedInputs:
    """Assemble a forward pass in which only `keep_token_index` vision tokens exist.

    `inputs` is the processor's output (needs `input_ids`, `attention_mask`,
    `pixel_values_videos`, `video_grid_thw`; `mm_token_type_ids` is used if
    present). `keep_token_index` is `to_qwen3vl_video_tokens(...)["keep_token_index"]`
    (1, K) -- ascending merged-token indices; `None` keeps everything (the
    identity case used as the correctness gate).

    `qwen_patch_index` is only needed for `prune_stage="encoder"` when blocks may
    be partial (`partial_blocks="any"/"full"`); with whole blocks it is derived
    from `keep_token_index` here, so the default path needs no extra argument.

    Position ids are NOT recomputed from the pruned sequence. They are computed
    once on the DENSE sequence with the model's own `compute_3d_position_ids`,
    then the dropped columns are deleted. This is exact and avoids
    reimplementing mrope: Qwen advances the post-vision position counter by
    `max(h, w) // merge` (grid-derived, not token-count-derived), so removing
    vision tokens leaves every later text position exactly where the dense pass
    put it -- pruned and dense runs stay positionally comparable, and the
    timestamp/vision_start/vision_end scaffolding is carried through untouched.
    """
    if prune_stage not in ("llm", "encoder"):
        raise ValueError(f"prune_stage must be 'llm' or 'encoder', got {prune_stage!r}")
    inner = model.model if hasattr(model, "model") else model
    cfg = model.config
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            f"batch size must be 1 (got {input_ids.shape[0]}): pruning gives each row its own "
            "sequence length, which would need re-padding and a re-derived attention mask")
    device = input_ids.device
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pixel_values_videos = inputs["pixel_values_videos"]
    video_grid_thw = inputs["video_grid_thw"]
    m = cfg.vision_config.spatial_merge_size

    is_video_tok = _video_token_positions(input_ids, cfg.video_token_id)
    n_dense = int(is_video_tok.sum())
    expected = int((video_grid_thw.prod(-1) // (m * m)).sum())
    if n_dense != expected:
        raise ValueError(
            f"prompt holds {n_dense} video placeholders but video_grid_thw implies {expected} "
            "merged tokens -- processor/grid mismatch")

    if keep_token_index is None:
        keep = torch.arange(n_dense, device=device)
    else:
        keep = keep_token_index.reshape(-1)
        keep = keep[keep >= 0].to(device)
        if keep.numel() and (int(keep.max()) >= n_dense or int(keep.min()) < 0):
            raise ValueError(f"keep_token_index out of range for {n_dense} vision tokens")
        if keep.numel() > 1 and not bool((keep[1:] > keep[:-1]).all()):
            raise ValueError("keep_token_index must be strictly ascending")

    # --- dense position ids, then column-delete ------------------------------
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    dense_embeds = inner.get_input_embeddings()(input_ids)
    position_ids_dense = inner.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=dense_embeds,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )

    seq_keep = torch.ones_like(is_video_tok)
    vid_kept = torch.zeros(n_dense, dtype=torch.bool, device=device)
    vid_kept[keep] = True
    seq_keep[is_video_tok] = vid_kept

    input_ids_p = input_ids[:, seq_keep]
    attention_mask_p = attention_mask[:, seq_keep]
    position_ids_p = None if position_ids_dense is None else position_ids_dense[..., seq_keep]

    # --- vision embeddings ---------------------------------------------------
    if prune_stage == "llm":
        out = inner.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        vis = torch.cat(out.pooler_output, dim=0) if isinstance(out.pooler_output, (list, tuple)) \
            else out.pooler_output
        deep = list(out.deepstack_features or [])
        vis = vis[keep]
        deep = [d[keep] for d in deep]
    else:
        if qwen_patch_index is None:
            # Whole-block case: each kept merged token is backed by exactly its own
            # m**2 consecutive patch rows (see to_qwen3vl_video_tokens). Under the
            # 'any'/'full' policies blocks can be partial, so pass the adapter's
            # `qwen_patch_index` explicitly there.
            qpi = (keep[:, None] * (m * m) + torch.arange(m * m, device=device)).reshape(-1)
        else:
            qpi = qwen_patch_index.reshape(-1)
        qpi = qpi[qpi >= 0].to(device)
        if qpi.numel() != keep.numel() * m * m:
            raise ValueError(
                f"encoder pruning needs whole {m}x{m} blocks: got {qpi.numel()} patch rows for "
                f"{keep.numel()} tokens (expected {keep.numel() * m * m}) -- use a preset with "
                f"score_coarsen={m} and partial_blocks='strict'")
        t_of_tok = keep // int((video_grid_thw[0, 1] // m) * (video_grid_thw[0, 2] // m))
        n_slices = int(video_grid_thw[0, 0])
        toks_per_slice = torch.bincount(t_of_tok, minlength=n_slices)
        vis, deep = _pruned_vision_forward(
            inner.visual, pixel_values_videos, video_grid_thw, qpi, toks_per_slice * (m * m))

    vis = vis.to(dense_embeds.device, dense_embeds.dtype)
    if vis.shape[0] != int(keep.numel()):
        raise ValueError(f"vision embed rows {vis.shape[0]} != kept tokens {int(keep.numel())}")

    inputs_embeds = inner.get_input_embeddings()(input_ids_p)
    video_mask_p = (input_ids_p == cfg.video_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask_p.unsqueeze(-1), vis)

    return PrunedInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask_p,
        position_ids=position_ids_p,
        visual_pos_masks=video_mask_p,
        deepstack_visual_embeds=[d.to(inputs_embeds.dtype) for d in deep] or None,
        input_ids=input_ids_p,
        seq_keep=seq_keep.unsqueeze(0),
        n_vision_tokens=int(keep.numel()),
        n_vision_tokens_dense=n_dense,
        prune_stage=prune_stage,
        meta={"seq_len": int(input_ids_p.shape[1]), "seq_len_dense": int(input_ids.shape[1])},
    )


def pruned_forward(model, pruned: PrunedInputs, labels: Optional[torch.Tensor] = None):
    """Run the language model on pruned inputs and return `(logits, loss)`.

    Calls the text stack directly rather than the top-level forward, because
    `visual_pos_masks` / `deepstack_visual_embeds` are only reachable that way --
    passing `inputs_embeds` to the public forward would silently DROP deepstack
    injection (Qwen3-VL-2B injects vision features at layers 5/11/17, so losing
    it would quietly change the model being measured).
    """
    inner = model.model if hasattr(model, "model") else model
    out = inner.language_model(
        input_ids=None,
        position_ids=pruned.position_ids,
        attention_mask=pruned.attention_mask,
        inputs_embeds=pruned.inputs_embeds,
        visual_pos_masks=pruned.visual_pos_masks,
        deepstack_visual_embeds=pruned.deepstack_visual_embeds,
        use_cache=False,
    )
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    logits = model.lm_head(hidden)
    loss = None
    if labels is not None:
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    return logits, loss
