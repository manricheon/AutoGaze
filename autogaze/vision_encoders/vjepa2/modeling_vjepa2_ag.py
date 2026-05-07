# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoGaze integration for V-JEPA2 — proper INTEGRATION.md approach.

INTEGRATION.md Step 1: Patch selection (mask_with_gazing)
----------------------------------------------------------
Only the gazed patches enter the transformer layer stack.
Sequence length is reduced from N_all → N_gazed.

  embeddings()  →  (B, N_all, C)
  mask_with_gazing()  →  (B, N_gazed, C)

  N_all   = T_p × H_p × W_p          (all patches)
  N_gazed = T_p × k                   (k = gazing_ratio × H_p × W_p per temporal group)

INTEGRATION.md Step 2: Attention mask
--------------------------------------
Not applied. V-JEPA2 is a *native video ViT* — trained with full cross-temporal
attention. Unlike SigLIP (image ViT retargeted for video), no causal constraint
is needed. Full bidirectional attention over the selected patch subset is used.

RoPE position correction
-------------------------
After patch selection the sequence indices no longer match the original spatial
positions. The original flat indices are passed as ``position_mask`` to each
transformer layer, so V-JEPA2's RoPE correctly encodes (frame, height, width)
for every selected patch.

Patch layout
------------
Row-major (t, h, w): flat index = t_p * H_p * W_p + h * W_p + w.
V-JEPA2's ``get_position_ids()`` decomposes flat indices using
  grid_size = config.crop_size // config.patch_size
so the flat indices we pass must be computed with the same H_p = W_p = grid_size.

Comparison with the previous zeroing approach
----------------------------------------------
  Old:  embeddings → zero non-gazed tokens → transformer (N_all tokens, most zeroed)
  New:  embeddings → select gazed tokens   → transformer (N_gazed tokens only)

  Advantage: genuine sequence-length reduction; no zero "ghost" tokens in attention.

Usage (unchanged monkey-patch pattern)
---------------------------------------
    from autogaze.vision_encoders.vjepa2 import AutoGazeVJEPA2Encoder

    model.encoder.__class__ = AutoGazeVJEPA2Encoder
    model.encoder._gazing_info = None

Before each AutoGaze forward:
    model.encoder._gazing_info = {
        'gazing_pos':            (B, N_gazed) long   — original flat indices
        'num_gazing_each_frame': (T_p,)       long   — k per temporal group (uniform)
        'if_padded_gazing':      (B, N_gazed) bool   — padding flags (all False for top-k)
    }
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput

try:
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder
except ImportError as exc:
    raise ImportError(
        "transformers >= 4.53 is required for V-JEPA2 support. "
        "Install with: pip install transformers>=4.53"
    ) from exc


class AutoGazeVJEPA2Encoder(VJEPA2Encoder):
    """V-JEPA2 encoder with INTEGRATION.md-style AutoGaze patch selection.

    Replaces the previous zeroing approach with actual token removal:

    * ``mask_with_gazing()`` selects N_gazed tokens from the full N_all sequence,
      mirroring SigLIP's implementation.
    * The original flat indices are forwarded as ``position_mask`` so each layer's
      RoPE uses the correct (frame, height, width) coordinates.
    * Full bidirectional attention is preserved (V-JEPA2 is a video-native model).
    """

    _gazing_info: dict | None = None

    @staticmethod
    def mask_with_gazing(
        sequence: torch.Tensor,
        gazing_pos: torch.Tensor,
        if_padded_gazing: torch.Tensor,
    ) -> torch.Tensor:
        """Select only the gazed patches from the full patch sequence.

        Mirrors SigLIP's ``mask_with_gazing`` (INTEGRATION.md Step 1).

        Args:
            sequence:         ``(B, N_all, C)`` — full patch embeddings.
            gazing_pos:       ``(B, N_gazed)`` long — indices into N_all.
            if_padded_gazing: ``(B, N_gazed)`` bool — True for padding positions.

        Returns:
            ``(B, N_gazed, C)`` — selected patches; padding rows zeroed out.
        """
        B = sequence.shape[0]
        pos = gazing_pos.clone()
        pos[if_padded_gazing] = 0   # redirect padding indices to dummy position 0
        selected = sequence[torch.arange(B, device=sequence.device)[:, None], pos]
        valid = (~if_padded_gazing).to(sequence.dtype).unsqueeze(-1)  # (B, N_gazed, 1)
        return selected * valid

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        """Forward with optional AutoGaze patch selection.

        When ``_gazing_info`` is set:
          - Only N_gazed patches enter the transformer (token-level selection).
          - Original flat indices flow as ``position_mask`` for correct RoPE.
          - Full bidirectional attention (no causal mask).

        When ``_gazing_info`` is None: standard V-JEPA2 forward (all N_all patches).

        Args:
            pixel_values_videos: ``(B, T, C, H, W)`` input video.
            head_mask: Optional per-layer attention head mask.
            output_attentions: Return per-layer attention weights.
            output_hidden_states: Return all intermediate hidden states.

        Returns:
            ``BaseModelOutput`` with ``last_hidden_state``:
              - ``(B, N_gazed, C)`` when AutoGaze is active.
              - ``(B, N_all,   C)`` in baseline mode.
        """
        gazing_info = self._gazing_info
        self._gazing_info = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # ── Step 1a: embed all patches ────────────────────────────────────── #
        hidden_states = self.embeddings(pixel_values_videos)   # (B, N_all, C)

        # ── Step 1b: select only gazed patches ───────────────────────────── #
        if gazing_info is not None:
            gazing_pos = gazing_info['gazing_pos'].to(
                device=hidden_states.device, dtype=torch.long)       # (B, N_gazed)
            if_padded  = gazing_info['if_padded_gazing'].to(
                device=hidden_states.device)                          # (B, N_gazed) bool

            hidden_states = self.mask_with_gazing(hidden_states, gazing_pos, if_padded)
            # hidden_states: (B, N_gazed, C)

            # Position mask carries original flat indices for RoPE in each layer.
            position_mask = gazing_pos                               # (B, N_gazed) long
        else:
            position_mask = None

        # ── Transformer layers ────────────────────────────────────────────── #
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                position_mask=position_mask,   # original indices → correct RoPE
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
