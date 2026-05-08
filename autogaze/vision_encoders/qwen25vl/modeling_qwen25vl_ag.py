# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoGaze integration for Qwen2.5-VL's visual encoder.

Architecture
------------
Qwen2.5-VL's visual encoder processes each **temporal chunk** independently —
there is NO cross-temporal attention inside the visual encoder.  Temporal
integration is handled entirely by the LLM's 3D RoPE.

Concretely, `Qwen2_5_VisionTransformerPretrainedModel.forward()` builds two
sets of sequence boundaries:

  cu_seqlens          one boundary per temporal chunk  (H_p × W_p patches each)
  cu_window_seqlens   one boundary per spatial window  (window_size² groups each)

Full-attention blocks use cu_seqlens → patches within the same temporal chunk
attend to each other; no cross-chunk attention ever occurs.

AutoGaze integration (follows INTEGRATION.md §Step 1)
-------------------------------------------------------
Since temporal chunks are independent:

1. Run AutoGaze on the video's T input frames.
2. Group the T_ag gaze maps into T_p = T_ag / temporal_patch_size temporal chunks.
   Average the gaze maps within each chunk and bilinearly interpolate to (H_p, W_p).
3. After `patch_embed(pixel_values)` → (N_total, C), zero patches that AutoGaze
   did not select.  This is done BEFORE window reordering so that the mask aligns
   with the (t, h, w) row-major layout that `patch_embed` produces.
4. All remaining processing (window reordering, RoPE, attention blocks, merger,
   reverse reordering) is identical to the original Qwen2.5-VL forward.

Token count is unchanged → the LLM's input format requires no modification.

Comparison vs. zero-shot hook
------------------------------
  Hook mode:  single gaze map (first frame or mean) tiled over ALL temporal chunks.
  Full mode:  per-temporal-chunk gaze maps that respect temporal variation.

Usage
-----
Swap `model.visual` in-place (class monkey-patch, no weight copy needed)::

    from autogaze.vision_encoders.qwen25vl import AutoGazeQwen25VisionTransformer

    model.visual.__class__ = AutoGazeQwen25VisionTransformer
    model.visual._gazing_info = None   # one-time initialisation

Before every `model.generate()` call that uses AutoGaze::

    model.visual._gazing_info = {
        'patch_mask': patch_mask,   # (T_p * H_p * W_p,) float32 — 1.0=keep / 0.0=zero
    }

`_gazing_info` is automatically cleared after each forward pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )
except ImportError as exc:
    raise ImportError(
        "transformers >= 4.45 is required for Qwen2.5-VL support. "
        "Install with: pip install transformers>=4.45"
    ) from exc


class AutoGazeQwen25VisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    """Qwen2.5-VL visual encoder with AutoGaze per-temporal-chunk gaze selection.

    Identical to the original except that, when ``_gazing_info`` is set, the
    forward method zeros non-selected patches immediately after ``patch_embed``
    and before window reordering.

    See module docstring for full explanation and usage instructions.
    """

    # State injected externally before each AutoGaze forward pass.
    _gazing_info: dict | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with optional AutoGaze gaze-masking.

        Args:
            hidden_states: Raw video pixel patches from the processor,
                shape ``(N_input_patches, temporal_patch_size * patch_size² * C_in)``.
            grid_thw: ``(num_videos, 3)`` tensor with (T_p, H_p, W_p) for each video.
            **kwargs: Forwarded to each transformer block.

        Returns:
            Visual token features after the patch merger, shape
            ``(N_total / spatial_merge_size², out_hidden_size)``.
        """
        # ── consume gaze info (cleared after use) ─────────────────────────── #
        gazing_info = getattr(self, '_gazing_info', None)
        self._gazing_info = None

        # ── Step 1: Patch embedding ────────────────────────────────────────── #
        # Output shape: (N_total, C) in row-major (t, h, w) order.
        # N_total = sum(T_p_i * H_p_i * W_p_i) across all videos.
        hidden_states = self.patch_embed(hidden_states)

        # ── Step 2: AutoGaze — zero non-gazed patches ─────────────────────── #
        # Applied BEFORE window reordering so the mask aligns with the
        # (t, h, w) row-major layout that `patch_embed` produces.
        #
        # Since Qwen2.5-VL has no cross-temporal attention in the visual encoder,
        # zeroing patches independently per temporal chunk is equivalent to the
        # full mask_with_gazing() operation in INTEGRATION.md §Step 1.
        if gazing_info is not None:
            patch_mask = gazing_info['patch_mask'].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )  # (N_total,) — 1.0 = keep, 0.0 = zero
            hidden_states = hidden_states * patch_mask.unsqueeze(-1)

        # ── Unchanged Qwen2.5-VL processing ───────────────────────────────── #
        # (exact copy of Qwen2_5_VisionTransformerPretrainedModel.forward)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()

        # Window reordering: groups of spatial_merge_unit (=4) patches move together.
        # This aligns hidden_states with rotary_pos_emb (which uses grouped spatial order).
        hidden_states = (
            hidden_states
            .reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            [window_index, :, :]
            .reshape(seq_len, -1)
        )
        rotary_pos_emb = (
            rotary_pos_emb
            .reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            [window_index, :, :]
            .reshape(seq_len, -1)
        )

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens: one boundary per temporal chunk (H_p × W_p patches each).
        # After window reordering, patches from the same temporal chunk are still
        # contiguous (the window reordering preserves temporal-chunk boundaries).
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = (
                cu_seqlens if layer_num in self.fullatt_block_indexes
                else cu_window_seqlens
            )
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Spatial merger (2×2 → 1 token) and reverse window ordering.
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
