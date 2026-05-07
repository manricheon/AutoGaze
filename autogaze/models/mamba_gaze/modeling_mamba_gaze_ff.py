# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MambaGaze Feedforward (Branch A, v2) — non-autoregressive visual token selector.

Reference: docs/mamba_gaze_ref_v0.md  (Section 2, Branch A "SRF-Predictor")

Architecture (single forward pass, no generation loop):

  Video (B, T, C, H, W)
      │
      ▼
  LightweightCNNEncoder   (~350K)  — 4× stride-2 conv, 3→192 channels
      │ F_t (B, T, d, h, w)
      ├──→ SaliencyHead    (~200)  — 1×1 conv → sigmoid, S_t ∈ [0,1]^{h×w}
      └──→ TemporalMotion  (0)     — L2(F_t − F_{t−1}), R_t ∈ [0,1]^{h×w}
      │
      ▼ Concat(F, S, R) → (B, T, N, d+2)
  SpatioTemporalMambaAggregator  (~770K)
      │  ∙ in_proj: d+2 → mamba_dim
      │  ∙ for each layer:
      │      Spatial bidirectional Mamba (zigzag scan, 14×14)
      │      Temporal causal Mamba (across T frames)
      │  ∙ score_head: mamba_dim → 1 → sigmoid
      │ Importance scores I_t ∈ [0,1]^N
      ▼
  Top-K Selection
      training: Gumbel-top-k  (differentiable, straight-through)
      inference: Hard top-k   (O(N), no loop)
      │ Binary mask (B, T, N)
      ▼
  AutoGaze-compatible output dict

vs MambaGaze v1 (modeling_mamba_gaze.py):
  v1 still generates tokens autoregressively with a Mamba SSM decoder loop.
  v2 predicts ALL patch importances simultaneously in a single forward pass.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_mamba_gaze_ff import MambaGazeFFConfig
from .modeling_mamba_gaze import MambaBlock, MambaVisionEncoder  # reuse core blocks


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lightweight CNN Encoder  (~350K)
# ─────────────────────────────────────────────────────────────────────────────

class LightweightCNNEncoder(nn.Module):
    """4-stage stride-2 conv: 224×224 → 14×14, ~350K parameters.

    Produces patch-aligned feature maps at 1/16 spatial resolution,
    matching ViT's patch grid for downstream compatibility.
    """

    def __init__(self, channels: list, out_channels: int):
        super().__init__()
        in_ch = 3
        layers = []
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            ]
            in_ch = out_ch
        # Final channel matches out_channels (already in channels list)
        self.stages = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W)  →  (B, d, H/16, W/16)"""
        return self.stages(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatial Saliency Head  (~200 params)
# ─────────────────────────────────────────────────────────────────────────────

class SaliencyHead(nn.Module):
    """1×1 conv → sigmoid: explicit spatial saliency prior."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d, h, w)  →  (B, 1, h, w) ∈ [0, 1]"""
        return torch.sigmoid(self.conv(x))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spatio-Temporal Mamba Aggregator  (~770K)
# ─────────────────────────────────────────────────────────────────────────────

class SpatioTemporalMambaAggregator(nn.Module):
    """Mamba-based importance score predictor.

    Processes tokens with:
      - Spatial bidirectional Mamba (zigzag scan, captures 2D locality)
      - Temporal causal Mamba (captures motion context across frames)

    Returns a per-token importance score in [0, 1].
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        H: int,
        W: int,
    ):
        super().__init__()
        self.H, self.W = H, W

        self.in_proj = nn.Linear(in_dim, d_model)
        self.spatial_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, bidirectional=True)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, bidirectional=False)
            for _ in range(depth)
        ])
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, "RMSNorm") else nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1)

        # Zigzag indices for spatial scan (row-alternating)
        zz, inv_zz = MambaVisionEncoder._zigzag_indices(H, W, device="cpu")
        self.register_buffer("zz", zz)
        self.register_buffer("inv_zz", inv_zz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, in_dim)
        Returns: (B, T, N) importance scores ∈ [0, 1]
        """
        B, T, N, _ = x.shape

        x = self.in_proj(x.reshape(B * T, N, -1))   # (B*T, N, d_model)

        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: bidirectional Mamba over H×W tokens (zigzag order)
            x_zz = x[:, self.zz]
            x_zz, _ = s_block(x_zz)
            x = x_zz[:, self.inv_zz]

            # Temporal: causal Mamba over T frames per spatial position
            x_4d = x.reshape(B, T, N, -1)
            xt = x_4d.permute(0, 2, 1, 3).reshape(B * N, T, -1)   # (B·N, T, d)
            xt, _ = t_block(xt)
            x = xt.reshape(B, N, T, -1).permute(0, 2, 1, 3).reshape(B * T, N, -1)

        x = self.norm(x)
        scores = torch.sigmoid(self.score_head(x)).squeeze(-1)      # (B*T, N)
        return scores.reshape(B, T, N)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MambaGazeFF — top-level feedforward model
# ─────────────────────────────────────────────────────────────────────────────

class MambaGazeFF(nn.Module):
    """Feedforward (non-AR) visual token selector based on Branch A of the
    MambaGaze research proposal (mamba_gaze_ref_v0.md).

    Compatible with AutoGaze's output format:
        out = model({'video': video}, gazing_ratio=0.5)
        mask = out['gazing_mask'][0]   # (B, T, N) binary mask

    Training (distillation from AutoGaze teacher):
        out = model({'video': video}, teacher_mask=teacher_binary_mask)
        loss = out['distill_loss']
        loss.backward()
    """

    def __init__(self, config: MambaGazeFFConfig):
        super().__init__()
        self.config = config
        self.scales = sorted(int(s) for s in str(config.scales).split("+"))
        self.num_vision_tokens_each_frame = config.num_vision_tokens_each_frame
        self.input_img_size = config.input_img_size
        self.frame_sampling_rate = 1
        self.num_vision_tokens_each_scale_each_frame = [config.num_vision_tokens_each_frame]
        self.gazing_ratio_config = config.gazing_ratio_config
        self.gumbel_temperature = config.gumbel_temperature

        h = w = config.input_img_size // 16    # 14 for 224×224
        d_cnn = config.cnn_out_channels

        # ── Components ────────────────────────────────────────────────────────
        self.cnn_encoder = LightweightCNNEncoder(config.cnn_channels, d_cnn)
        self.saliency_head = SaliencyHead(d_cnn)
        # Temporal motion has no parameters — pure L2 diff between feature maps
        self.aggregator = SpatioTemporalMambaAggregator(
            in_dim=d_cnn + 2,        # CNN features + saliency + motion
            d_model=config.mamba_dim,
            depth=config.mamba_depth,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            H=h, W=w,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_gazing_ratio(self) -> float:
        cfg = self.gazing_ratio_config
        s = cfg.get(
            "sample_strategy_during_training" if self.training
            else "sample_strategy_during_inference",
            "fixed",
        )
        if s == "fixed":
            return cfg["fixed"]["gazing_ratio"]
        return random.uniform(cfg["uniform"]["gazing_ratio_min"],
                              cfg["uniform"]["gazing_ratio_max"])

    @staticmethod
    def _compute_motion(features: torch.Tensor) -> torch.Tensor:
        """Frame-to-frame L2 motion magnitude, normalized to [0, 1].

        Args:
            features: (B, T, d, h, w)
        Returns:
            motion: (B, T, 1, h, w)
        """
        B, T, d, h, w = features.shape
        if T == 1:
            return torch.zeros(B, 1, 1, h, w, device=features.device)

        diff = features[:, 1:] - features[:, :-1]              # (B, T-1, d, h, w)
        mag = diff.pow(2).mean(dim=2, keepdim=True)             # (B, T-1, 1, h, w)
        zero = torch.zeros(B, 1, 1, h, w, device=features.device)
        motion = torch.cat([zero, mag], dim=1)                  # (B, T, 1, h, w)

        # Per-frame normalization to [0, 1]
        m_max = motion.reshape(B * T, -1).max(dim=-1).values.clamp(min=1e-6)
        motion = motion / m_max.reshape(B, T, 1, 1, 1)
        return motion.clamp(0.0, 1.0)

    def _scores_to_mask(self, scores: torch.Tensor, gazing_ratio: float) -> torch.Tensor:
        """Convert continuous importance scores → binary (or soft) selection mask.

        Training: Gumbel-top-k with straight-through estimator (differentiable).
        Inference: Hard top-k (no stochasticity, O(N log N) sort).

        Args:
            scores: (B, T, N) ∈ [0, 1]
            gazing_ratio: fraction of tokens to keep
        Returns:
            mask: (B, T, N)  — soft during training, binary at inference
        """
        B, T, N = scores.shape
        k = max(1, int(gazing_ratio * N))

        if self.training:
            # Log-odds + Gumbel noise → perturbed logits
            logits = torch.log(scores.clamp(1e-6, 1 - 1e-6)) \
                   - torch.log1p(-scores.clamp(1e-6, 1 - 1e-6))
            noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
            perturbed = (logits + noise) / self.gumbel_temperature

            # Threshold = k-th largest perturbed logit
            threshold = perturbed.topk(k, dim=-1).values[..., -1:]   # (B, T, 1)
            # Soft mask via sigmoid (straight-through gradient flows through scores)
            soft_mask = torch.sigmoid((perturbed - threshold) / self.gumbel_temperature)
            return soft_mask
        else:
            # Hard top-k
            topk_idx = scores.topk(k, dim=-1).indices   # (B, T, k)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, topk_idx, 1.0)
            return mask

    @torch.no_grad()
    def _mask_to_gazing_pos(
        self,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert (B, T, N) hard binary mask to AutoGaze-compatible gazing_pos format.

        Returns:
            gazing_pos:  (B, sum_k)  — global token indices per frame (t*N + local)
            if_padded:   (B, sum_k)  — True at padding entries
            num_gazing:  (T,)        — number of selected tokens per frame
        """
        B, T, N = mask.shape
        eos = N      # padding sentinel (out of per-frame range [0, N-1])

        all_pos, all_padded, num_per_frame = [], [], []
        for t in range(T):
            m_t = mask[:, t]                         # (B, N) binary
            k = max(1, int(m_t.sum(dim=-1).max().item()))
            num_per_frame.append(k)

            pos_list, flag_list = [], []
            for b in range(B):
                sel = m_t[b].nonzero(as_tuple=True)[0]   # selected local indices
                n_sel = len(sel)
                if n_sel < k:
                    pad = sel.new_full((k - n_sel,), eos)
                    sel = torch.cat([sel, pad])
                elif n_sel > k:
                    sel = sel[:k]
                pos_list.append(sel + t * N)
                padded = torch.zeros(k, dtype=torch.bool, device=mask.device)
                padded[n_sel:] = True
                flag_list.append(padded)

            all_pos.append(torch.stack(pos_list, dim=0))    # (B, k)
            all_padded.append(torch.stack(flag_list, dim=0))

        gazing_pos = torch.cat(all_pos, dim=1)      # (B, sum_k)
        if_padded  = torch.cat(all_padded, dim=1)
        num_gazing = torch.tensor(num_per_frame, device=mask.device)
        return gazing_pos, if_padded, num_gazing

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        inputs: Dict,
        gazing_ratio: Optional[float] = None,
        teacher_mask: Optional[torch.Tensor] = None,   # (B, T, N) for distillation
        **kwargs,
    ) -> Dict:
        """Single feedforward pass — no generation loop.

        Args:
            inputs:        dict with key 'video': (B, T, C, H, W)
            gazing_ratio:  fraction of tokens to keep; uses config default if None
            teacher_mask:  binary teacher mask (B, T, N) for distillation training

        Returns:
            dict compatible with AutoGaze output format, plus:
              'importance_scores': (B, T, N) raw scores ∈ [0, 1]
              'distill_loss':      scalar BCE loss if teacher_mask provided
              'saliency_maps':     (B, T, h, w) explicit spatial saliency
        """
        video = inputs["video"]                       # (B, T, C, H, W)
        B, T, C, H, W = video.shape
        ratio = gazing_ratio if gazing_ratio is not None else self.get_gazing_ratio()

        # ── Resize to input_img_size ──────────────────────────────────────────
        x = video.reshape(B * T, C, H, W)
        if H != self.input_img_size or W != self.input_img_size:
            x = F.interpolate(
                x, size=(self.input_img_size,) * 2,
                mode="bicubic", align_corners=False,
            )

        # ── (1) CNN encoding ──────────────────────────────────────────────────
        features = self.cnn_encoder(x)               # (B*T, d, h, w)
        _, d, h, w = features.shape
        N = h * w

        # ── (2) Saliency ──────────────────────────────────────────────────────
        saliency = self.saliency_head(features)      # (B*T, 1, h, w) ∈ [0,1]

        # ── (3) Temporal motion ───────────────────────────────────────────────
        feat_4d = features.reshape(B, T, d, h, w)
        motion = self._compute_motion(feat_4d)       # (B, T, 1, h, w)
        motion_flat = motion.reshape(B * T, 1, h, w)

        # ── (4) Aggregate: features + saliency + motion → tokens ──────────────
        agg = torch.cat([features, saliency, motion_flat], dim=1)  # (B*T, d+2, h, w)
        agg = agg.permute(0, 2, 3, 1).reshape(B * T, N, d + 2)
        agg = agg.reshape(B, T, N, d + 2)

        # ── (5) Mamba aggregator → importance scores ──────────────────────────
        scores = self.aggregator(agg)                # (B, T, N) ∈ [0, 1]

        # ── (6) Top-K selection ───────────────────────────────────────────────
        mask = self._scores_to_mask(scores, ratio)   # (B, T, N)

        # ── (7) Distillation loss (if teacher provided) ───────────────────────
        distill_loss = None
        if teacher_mask is not None:
            distill_loss = F.binary_cross_entropy(scores, teacher_mask.float().to(scores))

        # ── (8) Gazing pos (AutoGaze-compatible, no gradient needed) ──────────
        with torch.no_grad():
            hard_mask = (mask.detach() > 0.5).float()
            gazing_pos, if_padded, num_gazing = self._mask_to_gazing_pos(hard_mask)

        return {
            # Primary outputs
            "gazing_mask":              [mask],
            "importance_scores":        scores,
            "saliency_maps":            saliency.reshape(B, T, h, w),

            # AutoGaze-compatible keys
            "gazing_pos":               gazing_pos,
            "if_padded_gazing":         if_padded,
            "num_gazing_each_frame":    num_gazing,
            "scales":                   self.scales,
            "frame_sampling_rate":      self.frame_sampling_rate,
            "num_vision_tokens_each_frame": self.num_vision_tokens_each_frame,

            # Training
            "distill_loss":             distill_loss,

            # Stubs for full AutoGaze-API compatibility
            "log_action_probs":         None,
            "task_loss_prediction":     None,
            "has_task_loss_requirement": False,
            "task_loss_requirement":    None,
            "past_key_values":          None,
            "past_input_embeds":        None,
            "past_attention_mask":      None,
            "past_conv_values":         None,
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> Dict:
        def _n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "cnn_encoder":   _n(self.cnn_encoder),
            "saliency_head": _n(self.saliency_head),
            "aggregator":    _n(self.aggregator),
            "total":         _n(self),
        }
