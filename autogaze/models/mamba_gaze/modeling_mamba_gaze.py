# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MambaGaze — Mamba SSM-based drop-in replacement for AutoGaze's LLaMA decoder.

Architecture:
    Video → MambaVisionEncoder → Connector → MambaGazeDecoder → gaze positions

Key differences from AutoGaze:
  - Vision encoder: zigzag-scan Mamba blocks (bidirectional) instead of Conv3D trunk
  - Gaze decoder:   causal Mamba SSM instead of LLaMA transformer
  - Streaming:      constant-size SSM state (96 KB) vs growing KV cache
  - Complexity:     O(N) sequence length vs O(N²) attention

Reference:
  Mamba (Gu & Dao, 2023) · Vision Mamba (Zhu et al., 2024)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .configuration_mamba_gaze import (
    MambaDecoderConfig,
    MambaGazeConfig,
    MambaGazeModelConfig,
    MambaVisionConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Selective SSM (S6) — core Mamba building block
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Mamba Selective State Space Model (S6).

    Unlike classical fixed-parameter SSMs (S4, S5), all SSM matrices
    (B, C, Δ) are functions of the current input — enabling the model to
    "selectively remember or forget" content, like soft attention but O(N).

    Forward returns (output, final_ssm_state) so the state can be cached
    during streaming inference instead of a growing KV cache.

    Shapes:
        in:  (B, L, d_model)
        out: (B, L, d_model),  state: (B, d_inner, d_state)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Split input into SSM branch (u) and gate branch (z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local depthwise conv — gives each position a short window of context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # Selective parameters: project u → (B_par, C_par, dt_raw)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        # Expand scalar dt to d_inner dimensions
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Fixed (but learned) SSM matrix A, parameterized in log space
        # Initialized log-spaced: encourages different time-scales per channel
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1)       # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))            # stored as log(-A)
        self.D = nn.Parameter(torch.ones(self.d_inner))    # skip connection scale

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ---------------------------------------------------------------------- #
    def _selective_scan(
        self,
        u: torch.Tensor,       # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)  — discretization step
        A: torch.Tensor,       # (d_inner, d_state)
        B: torch.Tensor,       # (B, L, d_state)  — selective
        C: torch.Tensor,       # (B, L, d_state)  — selective
        h0: Optional[torch.Tensor] = None,  # (B, d_inner, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential recurrence — educational; replace with parallel scan for speed."""
        B_b, L, D = u.shape

        # Discretize A and B (Zero-Order Hold)
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(B_b, D, self.d_state, device=u.device, dtype=u.dtype) \
            if h0 is None else h0.clone()

        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y, h

    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape

        xz = self.in_proj(x)                    # (B, L, 2·d_inner)
        u, z = xz.chunk(2, dim=-1)              # (B, L, d_inner) each

        # Causal depthwise conv — adds local context before selective scan
        u_conv = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # Selective parameters (all input-dependent)
        params = self.x_proj(u_conv)            # (B, L, 2·d_state + 1)
        B_par = params[..., :self.d_state]
        C_par = params[..., self.d_state:2 * self.d_state]
        dt_raw = params[..., -1:]               # (B, L, 1)
        dt = F.softplus(self.dt_proj(dt_raw))   # (B, L, d_inner) — must be positive

        A = -torch.exp(self.A_log.float())      # (d_inner, d_state) — must be negative

        y, h = self._selective_scan(u_conv, dt, A, B_par, C_par, ssm_state)

        # Skip connection + SiLU gating (Mamba's output gate)
        y = y + u_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y), h


# ─────────────────────────────────────────────────────────────────────────────
# 2. MambaBlock — PreNorm + SSM + residual
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """One Mamba residual block.

    bidirectional=True:  runs SSM in both directions and merges (for vision encoder).
    bidirectional=False: causal SSM only (for gaze decoder).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.ssm_fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        if bidirectional:
            self.ssm_bwd = SelectiveSSM(d_model, d_state, d_conv, expand)
            self.merge = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state=None,
    ) -> Tuple[torch.Tensor, object]:
        """
        x:         (B, L, D)
        ssm_state: for causal — tensor; for bidirectional — (fwd_h, bwd_h) tuple
        Returns: (output (B,L,D), new_state)
        """
        residual = x
        xn = self.norm(x)

        if not self.bidirectional:
            y, new_h = self.ssm_fwd(xn, ssm_state)
            return y + residual, new_h
        else:
            h_fwd = ssm_state[0] if ssm_state is not None else None
            h_bwd = ssm_state[1] if ssm_state is not None else None

            y_fwd, new_fwd = self.ssm_fwd(xn, h_fwd)
            y_bwd, new_bwd = self.ssm_bwd(xn.flip(1), h_bwd)
            y_bwd = y_bwd.flip(1)

            y = self.merge(torch.cat([y_fwd, y_bwd], dim=-1))
            return y + residual, (new_fwd, new_bwd)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MambaVisionEncoder — replaces ShallowVideoConvNet
# ─────────────────────────────────────────────────────────────────────────────

class MambaVisionEncoder(nn.Module):
    """Video vision encoder using Mamba SSM.

    Spatial tokens are processed with bidirectional Mamba in a zigzag scan
    (captures 2D spatial locality without positional encoding).
    Temporal tokens are processed with causal Mamba across frames.

    This replaces ShallowVideoConvNet and its Conv3dBlockForStreaming blocks.
    The 3D patch embedding is kept identical for weight compatibility.
    """

    def __init__(self, config: MambaVisionConfig):
        super().__init__()
        D = config.hidden_dim
        self.temporal_patch_size = config.temporal_patch_size

        # ① 3D patch embedding (same as ShallowVideoConvNet)
        self.temporal_conv = nn.Conv3d(
            3, D,
            kernel_size=(config.temporal_patch_size, config.kernel_size, config.kernel_size),
            stride=(config.temporal_patch_size, config.kernel_size, config.kernel_size),
        )
        self.patch_norm = nn.LayerNorm(D)

        # ② Spatial bidirectional Mamba + temporal causal Mamba, interleaved
        self.spatial_blocks = nn.ModuleList([
            MambaBlock(D, config.d_state, config.d_conv, config.expand, bidirectional=True)
            for _ in range(config.depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            MambaBlock(D, config.d_state, config.d_conv, config.expand, bidirectional=False)
            for _ in range(config.depth)
        ])

        self.out_proj = nn.Conv3d(D, config.out_dim, kernel_size=1)

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _zigzag_indices(H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row-alternating zigzag scan indices and their inverse."""
        idx = []
        for row in range(H):
            cols = range(W) if row % 2 == 0 else range(W - 1, -1, -1)
            idx.extend(row * W + c for c in cols)
        t = torch.tensor(idx, dtype=torch.long, device=device)
        return t, torch.argsort(t)

    # ---------------------------------------------------------------------- #
    def forward(self, x, use_cache=False, past_conv_values=None):
        """
        x: (B, T, C, H, W)
        Returns: (B, out_dim, T', H', W'), None
            None replaces past_conv_values — Mamba uses SSM state instead.
        """
        if x.dim() == 5:
            x = x.permute(0, 2, 1, 3, 4)          # (B, C, T, H, W)

        x = self.temporal_conv(x)                  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        N = H * W

        # Flatten spatial + normalize
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, N, D)
        x = self.patch_norm(x)

        # Precompute zigzag indices
        zz, inv_zz = self._zigzag_indices(H, W, x.device)

        # ③ Interleaved spatial-temporal Mamba
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: bidirectional scan over H×W tokens (per frame)
            x_zz = x[:, zz]                        # (B·T, N, D) zigzag order
            x_zz, _ = s_block(x_zz)
            x = x_zz[:, inv_zz]                    # restore raster order

            # Temporal: causal scan over T frames (per spatial position)
            x = x.reshape(B, T, N, D)
            xt = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B·N, T, D)
            xt, _ = t_block(xt)
            x = xt.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)

        # Output: (B, out_dim, T, H, W)
        x = x.reshape(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        x = self.out_proj(x)
        return x, None   # None = no conv past state (Mamba caches SSM state externally)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MambaGazeDecoder — replaces LlamaForCausalLM_MultiTokenPred
# ─────────────────────────────────────────────────────────────────────────────

class MambaGazeDecoder(nn.Module):
    """Causal Mamba decoder for autoregressive gaze position generation.

    Streaming key insight: instead of a KV cache growing as O(L × d),
    the Mamba SSM state is a constant-size tensor (B, d_inner, d_state)
    per layer — independent of how many tokens have been processed.

    The public API mirrors LlamaForCausalLM sufficiently for AutoGazeModel
    to use it as a drop-in replacement.
    """

    def __init__(self, config: MambaDecoderConfig):
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.n_layers = config.num_hidden_layers

        # Public attribute used by AutoGazeModel.embed() to look up gaze embeddings
        self.model = SimpleNamespace(
            embed_tokens=nn.Embedding(config.vocab_size, D)
        )

        self.blocks = nn.ModuleList([
            MambaBlock(D, config.d_state, config.d_conv, config.expand, bidirectional=False)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(D) if hasattr(nn, "RMSNorm") else nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        # Share embedding weights with lm_head (like LLaMA)
        self.lm_head.weight = self.model.embed_tokens.weight

        # Expose generation_config attribute (used by AutoGazeModel.generate)
        self.generation_config = SimpleNamespace(task_loss_requirement=None)
        self.is_gradient_checkpointing = False

    # ---------------------------------------------------------------------- #
    def _run_blocks(self, x: torch.Tensor, ssm_states=None):
        """Run all Mamba blocks, optionally with cached SSM states.

        Returns (x, new_ssm_states) where new_ssm_states is a list of
        (B, d_inner, d_state) tensors — the streaming state cache.
        """
        new_states = []
        for i, block in enumerate(self.blocks):
            state = ssm_states[i] if ssm_states is not None else None
            x, new_h = block(x, state)
            new_states.append(new_h)
        return x, new_states

    # ---------------------------------------------------------------------- #
    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        """Training forward — processes full sequence in parallel (parallel scan)."""
        x = inputs_embeds
        x, _ = self._run_blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        # Stub task_loss_prediction to match AutoGazeOutput expectations
        task_loss_pred = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)
        return SimpleNamespace(
            logits=logits,
            task_loss_prediction=task_loss_pred,
            past_key_values=None,
            loss=None,
        )

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,          # (B, L, D) — vision + past gaze
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        max_new_tokens: int = 100,
        logits_processor=None,
        eos_token_id: int = None,
        pad_token_id: int = None,
        past_key_values=None,                 # ignored; we use past_ssm_states
        past_ssm_states: list = None,         # list of (B, d_inner, d_state) per layer
        do_sample: bool = False,
        temperature: float = 1.0,
        use_cache: bool = True,
        return_dict_in_generate: bool = True,
        generation_config=None,
        **kwargs,
    ):
        """Autoregressive gaze token generation using SSM state caching.

        Compared to LLaMA's generate():
        - past_key_values → past_ssm_states (constant size!)
        - No position_ids needed (SSM handles position implicitly)
        - Same output format: SimpleNamespace with .sequences, .past_key_values
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        B = inputs_embeds.shape[0]

        # Step 1: Prefill — process all context tokens to get initial SSM state
        # (same as attention's KV cache prefill, but O(N) instead of O(N²))
        x, ssm_states = self._run_blocks(inputs_embeds, past_ssm_states)
        # Use only the last position's hidden state as context summary
        x_last = self.norm(x[:, -1:, :])    # (B, 1, D)

        # Step 2: Autoregressive decode — feed one token at a time
        # SSM state h is updated per token: h_t = Ā·h_{t-1} + B̄·u_t  (O(1) per step!)
        generated = torch.full((B, 1), eos_token_id, dtype=torch.long, device=inputs_embeds.device)
        # The first "token" is a dummy BOS-like state; real generation starts here
        logits_last = self.lm_head(x_last)   # (B, 1, vocab)

        sequences = []
        for step in range(max_new_tokens):
            logits = logits_last[:, -1, :]  # (B, vocab)

            # Apply logit processors (no-repeat, no-eos during forced generation)
            if logits_processor:
                all_ids = torch.cat([inputs_embeds.new_zeros(B, 1, dtype=torch.long),
                                     *([generated] if sequences else [])], dim=1) \
                    if sequences else generated
                # Concat all generated so far for the processor context
                prev_ids = torch.cat(sequences, dim=1) if sequences else generated
                logits = logits_processor(prev_ids, logits)

            # Sample / greedy
            if do_sample and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            sequences.append(next_token)

            # Check if all samples have generated EOS
            if (next_token == eos_token_id).all():
                break

            # Embed next token and run one SSM step
            emb = self.model.embed_tokens(next_token)   # (B, 1, D)
            x_step, ssm_states = self._run_blocks(emb, ssm_states)
            logits_last = self.lm_head(self.norm(x_step))

        # Pad sequences to uniform length with EOS
        seq = torch.cat(sequences, dim=1) if sequences else generated  # (B, N_generated)

        return SimpleNamespace(
            sequences=seq,
            past_key_values=ssm_states,    # reuses the same key for compatibility
        )

    # Gradient checkpointing stubs (called by AutoGazeModel.generate)
    def gradient_checkpointing_enable(self):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False


# ─────────────────────────────────────────────────────────────────────────────
# 5. MambaGazeModel — mirrors AutoGazeModel interface
# ─────────────────────────────────────────────────────────────────────────────

class MambaGazeModel(nn.Module):
    """MambaGaze core model.  Mirrors AutoGazeModel's public interface so it
    can be used as a drop-in replacement inside MambaGaze (analogous to
    AutoGazeModel inside AutoGaze).
    """

    def __init__(self, config: MambaGazeModelConfig):
        super().__init__()
        self.num_vision_tokens_each_frame = config.num_vision_tokens_each_frame
        self.input_img_size = config.input_img_size
        self.frame_sampling_rate = config.vision_config.temporal_patch_size

        vc = config.vision_config
        self.vision_model = MambaVisionEncoder(vc)

        # Connector: learnable positional embedding (same as AutoGaze)
        D = vc.out_dim
        N = config.n_patches
        self.pos_embed = nn.Parameter(torch.randn(N, D) * 0.02)

        dc = config.decoder_config
        self.gaze_decoder = MambaGazeDecoder(dc)
        self.gaze_decoder_config = dc   # attribute accessed by AutoGazeModel-like callers

        # Logit processors (same no-repeat / no-eos logic as AutoGaze)
        from autogaze.models.autogaze.modeling_autogaze import (
            NoRepeatTokensLogitsProcessor, NoEosTokenLogitsProcessor,
        )
        from transformers import LogitsProcessorList
        self.logits_processor = LogitsProcessorList([
            NoRepeatTokensLogitsProcessor(),
            NoEosTokenLogitsProcessor(),
        ])

    # ---------------------------------------------------------------------- #
    def embed(self, video=None, gaze_pos_ids=None, use_cache=False, past_conv_values=None):
        """Mirror of AutoGazeModel.embed() — produces interleaved vision+gaze embeds."""
        B, T = video.shape[:2]

        vision_features, new_past = self.vision_model(video, use_cache=use_cache,
                                                       past_conv_values=past_conv_values)
        # (B, D, T', H, W) → (B, T', N, D)
        vision_features = vision_features.permute(0, 2, 3, 4, 1)     # (B, T', H, W, D)
        B_, T_, H, W, D = vision_features.shape
        vision_features = vision_features.reshape(B_, T_, H * W, D)  # (B, T', N, D)
        vision_features = vision_features + self.pos_embed.unsqueeze(0).unsqueeze(0)
        vision_attn = [torch.ones(B, H * W, device=vision_features.device).long()
                       for _ in range(T_)]

        if gaze_pos_ids is not None:
            num_gazing = [g.shape[1] for g in gaze_pos_ids]
            gaze_ids = torch.cat(gaze_pos_ids, dim=1)
            gaze_attn = (gaze_ids != self.gaze_decoder_config.eos_token_id).long()
            gaze_emb = self.gaze_decoder.model.embed_tokens(gaze_ids)
            gaze_emb = list(gaze_emb.split(num_gazing, dim=1))
            gaze_attn = list(gaze_attn.split(num_gazing, dim=1))

        embeds, gaze_token_mask, gaze_pred_src, attn_mask = [], [], [], []
        for t in range(T_):
            embeds.append(vision_features[:, t])
            gaze_token_mask.append(torch.zeros(H * W, device=vision_features.device).long())
            gaze_pred_src.append(torch.zeros(H * W, device=vision_features.device).long() - 1)
            attn_mask.append(vision_attn[t])
            if gaze_pos_ids is not None:
                embeds.append(gaze_emb[t])
                gaze_token_mask.append(torch.ones(gaze_emb[t].shape[1],
                                                   device=gaze_emb[t].device).long())
                gaze_pred_src.append(-torch.arange(gaze_emb[t].shape[1],
                                                    device=gaze_emb[t].device) % 1 - 1)
                attn_mask.append(gaze_attn[t])

        return embeds, gaze_token_mask, gaze_pred_src, attn_mask, new_past

    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        video,
        max_gaze_tokens_each_frame=100,
        task_loss_requirement=None,
        use_cache=False,
        past_key_values=None,
        past_inputs_embeds=None,
        past_attention_mask=None,
        past_conv_values=None,
        **kwargs,
    ):
        """Mirror of AutoGazeModel.generate() — uses SSM states instead of KV cache."""
        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size,) * 2,
                              mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        video_embeds, _, __, ___, past_conv = self.embed(video=video, use_cache=use_cache,
                                                          past_conv_values=past_conv_values)

        # Retrieve SSM states from "past_key_values" slot (Mamba repurposes this key)
        past_ssm = past_key_values  # None on first call

        gaze_pos_ids_list, inputs_embeds = [], (past_inputs_embeds or [])
        attention_mask = past_attention_mask or []
        num_gazing_each_frame, if_padded_gazing = [], []

        for t in range(len(video_embeds)):
            inputs_embeds.append(video_embeds[t])
            attention_mask.append(torch.ones(B, video_embeds[t].shape[1],
                                             device=video_embeds[t].device).long())

            max_g = (max_gaze_tokens_each_frame
                     if isinstance(max_gaze_tokens_each_frame, int)
                     else max_gaze_tokens_each_frame[t])

            gaze_out = self.gaze_decoder.generate(
                inputs_embeds=torch.cat(inputs_embeds, dim=1),
                attention_mask=torch.cat(attention_mask, dim=1),
                max_new_tokens=max_g,
                logits_processor=self.logits_processor,
                eos_token_id=self.gaze_decoder_config.eos_token_id,
                past_ssm_states=past_ssm,
                **kwargs,
            )

            gaze_ids = gaze_out.sequences
            past_ssm = gaze_out.past_key_values  # SSM states (constant size!)

            gaze_pos_ids_list.append(gaze_ids + self.num_vision_tokens_each_frame * t)
            inputs_embeds.append(self.gaze_decoder.model.embed_tokens(gaze_ids))
            attention_mask.append((gaze_ids != self.gaze_decoder_config.eos_token_id).long())
            num_gazing_each_frame.append(gaze_ids.shape[1])
            if_padded_gazing.append(gaze_ids == self.gaze_decoder_config.eos_token_id)

        gaze_pos = torch.cat(gaze_pos_ids_list, dim=1)
        return {
            "gazing_pos": gaze_pos,
            "num_gazing_each_frame": torch.tensor(num_gazing_each_frame,
                                                   device=gaze_pos.device),
            "if_padded_gazing": torch.cat(if_padded_gazing, dim=1),
            "task_loss_requirement": task_loss_requirement,
            "past_input_embeds": inputs_embeds if use_cache else None,
            "past_attention_mask": attention_mask if use_cache else None,
            "past_key_values": past_ssm if use_cache else None,
            "past_conv_values": past_conv if use_cache else None,
        }

    # ---------------------------------------------------------------------- #
    def forward(self, video, gazing_info, **kwargs):
        """Training forward — parallel scan over full sequence."""
        gaze_pos = gazing_info["gazing_pos"]
        num_gazing = gazing_info["num_gazing_each_frame"]
        if_padded = gazing_info["if_padded_gazing"]

        B, T = video.shape[:2]
        video = rearrange(video, 'b t c h w -> (b t) c h w')
        video = F.interpolate(video, size=(self.input_img_size,) * 2,
                              mode="bicubic", align_corners=False)
        video = rearrange(video, '(b t) c h w -> b t c h w', b=B)

        gaze_split = list(gaze_pos.split(num_gazing.tolist(), dim=1))
        gaze_split = [gaze_split[t] - self.num_vision_tokens_each_frame * t
                      for t in range(len(gaze_split))]
        pad_split = list(if_padded.split(num_gazing.tolist(), dim=1))
        gaze_split = [g * (~p) + self.gaze_decoder_config.eos_token_id * p
                      for g, p in zip(gaze_split, pad_split)]

        embeds, gaze_mask, gaze_src, attn, _ = self.embed(video=video,
                                                            gaze_pos_ids=gaze_split)
        x = torch.cat(embeds, dim=1)
        gaze_mask_t = torch.cat(gaze_mask, dim=0)
        attn_t = torch.cat(attn, dim=1)

        out = self.gaze_decoder(inputs_embeds=x, attention_mask=attn_t, **kwargs)

        # Extract gaze logits at gaze token positions
        gaze_pos_flat = torch.nonzero(gaze_mask_t, as_tuple=True)[0]
        logits_at_gaze = out.logits[:, gaze_pos_flat, :]         # (B, N_gaze, vocab)
        gaze_ids_cat = torch.cat(gaze_split, dim=1)              # (B, N_gaze)

        gaze_probs = F.softmax(logits_at_gaze, dim=-1)
        B_, N = gaze_ids_cat.shape
        gaze_probs_selected = gaze_probs.reshape(B_ * N, -1)[
            torch.arange(B_ * N, device=gaze_probs.device),
            gaze_ids_cat.reshape(-1)
        ].reshape(B_, N)

        from autogaze.models.autogaze.modeling_autogaze import AutoGazeOutput
        return AutoGazeOutput(
            gaze_probs=gaze_probs_selected,
            logits=out.logits,
            task_loss_prediction=out.task_loss_prediction.squeeze(-1),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. MambaGaze — top-level model (mirrors AutoGaze interface)
# ─────────────────────────────────────────────────────────────────────────────

class MambaGaze(nn.Module):
    """Top-level MambaGaze model.  Mirrors the AutoGaze public interface.

    Usage (identical to AutoGaze):
        model = MambaGaze(MambaGazeConfig())
        out = model({'video': video}, gazing_ratio=0.5, generate_only=True)
        gaze_masks = out['gazing_mask']   # list of (B, T, N) tensors per scale
    """

    def __init__(self, config: MambaGazeConfig):
        super().__init__()
        self.config = config
        self.scales = sorted(int(s) for s in str(config.scales).split('+'))
        self.num_vision_tokens_each_frame = config.num_vision_tokens_each_frame
        self.gazing_ratio_config = config.gazing_ratio_config

        self.gazing_model = MambaGazeModel(config.mamba_gaze_model_config)
        self.frame_sampling_rate = (
            config.mamba_gaze_model_config.vision_config.temporal_patch_size
        )

        n_tok = config.num_vision_tokens_each_frame
        self.num_vision_tokens_each_scale_each_frame = [n_tok]

    # ---------------------------------------------------------------------- #
    def get_gazing_ratio(self):
        cfg = self.gazing_ratio_config
        s = cfg.get("sample_strategy_during_training" if self.training
                    else "sample_strategy_during_inference", "fixed")
        if s == "fixed":
            return cfg["fixed"]["gazing_ratio"]
        return random.uniform(cfg["uniform"]["gazing_ratio_min"],
                              cfg["uniform"]["gazing_ratio_max"])

    # ---------------------------------------------------------------------- #
    def get_mask_from_gazing_pos(self, video, gazing_pos, if_padded):
        B, T = video.shape[:2]
        N = self.num_vision_tokens_each_frame
        T_ = T // self.frame_sampling_rate
        mask = torch.zeros(B, N * T_ + 1, device=video.device)
        tmp = gazing_pos.clone()
        tmp[if_padded] = N * T_
        mask[torch.arange(B)[:, None], tmp] = 1
        mask = mask[:, :-1].reshape(B, T_, N)
        return [mask]

    # ---------------------------------------------------------------------- #
    def forward(
        self,
        inputs,
        gazing_ratio=None,
        gazing_info=None,
        temperature=1.0,
        generate_only=False,
        use_cache=False,
        past_key_values=None,
        past_inputs_embeds=None,
        past_attention_mask=None,
        past_conv_values=None,
        **kwargs,
    ):
        video = inputs['video']
        B, T = video.shape[:2]

        if gazing_info is None or len(gazing_info) == 0:
            ratio = gazing_ratio if gazing_ratio is not None else self.get_gazing_ratio()
            T_ = T // self.frame_sampling_rate
            n_gaze = max(1, int(ratio * self.num_vision_tokens_each_frame))
            gazing_info = self.gazing_model.generate(
                video,
                max_gaze_tokens_each_frame=n_gaze,
                do_sample=self.training,
                temperature=temperature,
                use_cache=use_cache,
                past_key_values=past_key_values,
                past_inputs_embeds=past_inputs_embeds,
                past_attention_mask=past_attention_mask,
                past_conv_values=past_conv_values,
            )

        gazing_pos = gazing_info["gazing_pos"]
        if_padded = gazing_info["if_padded_gazing"]
        num_gazing = gazing_info["num_gazing_each_frame"]

        if not generate_only:
            fwd = self.gazing_model(video, gazing_info)
            log_action_probs = torch.log(fwd.gaze_probs + 1e-8)
            task_loss_pred = fwd.task_loss_prediction
        else:
            log_action_probs = task_loss_pred = None

        mask = self.get_mask_from_gazing_pos(video, gazing_pos, if_padded)

        return {
            "gazing_pos": gazing_pos,
            "log_action_probs": log_action_probs,
            "gazing_mask": mask,
            "scales": self.scales,
            "frame_sampling_rate": self.frame_sampling_rate,
            "num_vision_tokens_each_frame": self.num_vision_tokens_each_frame,
            "num_gazing_each_frame": num_gazing,
            "if_padded_gazing": if_padded,
            "task_loss_prediction": task_loss_pred,
            "has_task_loss_requirement": False,
            "task_loss_requirement": None,
            "past_key_values": gazing_info.get("past_key_values") if use_cache else None,
            "past_input_embeds": gazing_info.get("past_input_embeds") if use_cache else None,
            "past_attention_mask": gazing_info.get("past_attention_mask") if use_cache else None,
            "past_conv_values": gazing_info.get("past_conv_values") if use_cache else None,
        }

    # ---------------------------------------------------------------------- #
    def count_parameters(self) -> dict:
        """Parameter count breakdown by component."""
        def _n(m):
            return sum(p.numel() for p in m.parameters())

        gm = self.gazing_model
        return {
            "vision_encoder": _n(gm.vision_model),
            "connector":      gm.pos_embed.numel() if hasattr(gm, "pos_embed") else 0,
            "gaze_decoder":   _n(gm.gaze_decoder),
            "total":          _n(self),
        }
