# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monocular depth estimation decoder for AutoGaze (DPT-lite style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import TaskDecoder


class _UpBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )


class DepthDecoder(TaskDecoder):
    """DPT-lite monocular depth estimator.

    Pipeline:
        (B, T, N, C) features
          → temporal mean-pool                → (B, N, C)
          → gaze-scaled features              → gaze-important regions boosted
          → reshape                           → (B, C, 14, 14)
          → 4× UpBlock (×2 each)             → 14→28→56→112→224
          → Conv(16, 1) + Sigmoid × max_depth → (B, 1, 224, 224)

    The gaze mask is used to scale features before decoding: tokens selected
    by AutoGaze get additional importance weighting, concentrating depth
    accuracy on salient scene elements.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        target_size: int = 224,
        max_depth: float = 10.0,
        mid_dims: tuple = (128, 64, 32, 16),
    ):
        super().__init__()
        self.target_size = target_size
        self.max_depth = max_depth

        # Gaze-aware feature scaling
        self.gaze_scale = nn.Linear(feature_dim, feature_dim)

        # Progressive upsampling
        dims = (feature_dim,) + mid_dims
        self.upsample_blocks = nn.ModuleList(
            [_UpBlock(in_d, out_d) for in_d, out_d in zip(dims[:-1], dims[1:])]
        )

        # Depth head: Sigmoid output scaled to [0, max_depth]
        self.depth_head = nn.Sequential(
            nn.Conv2d(mid_dims[-1], 1, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, gaze_mask: torch.Tensor):
        """
        Args:
            features:  (B, T, N, C)
            gaze_mask: (B, T, N) bool

        Returns:
            depth: (B, 1, target_size, target_size)  units determined by max_depth
        """
        B, T, N, C = features.shape
        H = W = int(N ** 0.5)  # 14

        # Temporal pool
        feat = features.mean(dim=1)           # (B, N, C)
        mask = gaze_mask.float().mean(dim=1)  # (B, N) ∈ [0, 1]

        # Gaze feature scaling: add residual scaled by gaze importance
        scale = torch.sigmoid(self.gaze_scale(feat))        # (B, N, C)
        feat = feat + scale * mask.unsqueeze(-1)             # (B, N, C)

        # Reshape to spatial map
        feat_map = rearrange(feat, 'b (h w) c -> b c h w', h=H, w=W)

        # Progressive upsampling
        x = feat_map
        for block in self.upsample_blocks:
            x = block(x)

        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return self.depth_head(x) * self.max_depth  # (B, 1, H, W)

    # ------------------------------------------------------------------ #
    def compute_loss(self, predictions: torch.Tensor, targets) -> dict:
        """Scale-invariant depth loss (log-space).

        Args:
            predictions: (B, 1, H, W)
            targets: (B, H, W) depth in same units as max_depth; 0 = invalid
        """
        if targets is None:
            return {'loss': predictions.new_zeros(1).squeeze()}

        pred = predictions.squeeze(1)  # (B, H, W)

        # Resize targets if shape mismatch
        if targets.shape[-2:] != pred.shape[-2:]:
            targets = F.interpolate(
                targets.float().unsqueeze(1), size=pred.shape[-2:], mode='nearest'
            ).squeeze(1)

        valid = targets > 0
        if not valid.any():
            return {'loss': predictions.new_zeros(1).squeeze()}

        pred_log = torch.log(pred[valid].clamp(min=1e-3))
        tgt_log  = torch.log(targets[valid].clamp(min=1e-3))
        diff = pred_log - tgt_log

        # Scale-invariant loss: E[d²] - 0.5·(E[d])²
        si_loss = diff.pow(2).mean() - 0.5 * diff.mean().pow(2)

        # Smooth L1 in log space for robustness
        smooth_loss = F.smooth_l1_loss(pred_log, tgt_log)

        loss = si_loss + 0.5 * smooth_loss
        return {'loss': loss, 'si_loss': si_loss, 'smooth_loss': smooth_loss}

    def compute_metrics(self, predictions: torch.Tensor, targets) -> dict:
        if targets is None:
            return {}

        pred = predictions.squeeze(1).clamp(min=1e-3)
        if targets.shape[-2:] != pred.shape[-2:]:
            targets = F.interpolate(
                targets.float().unsqueeze(1), size=pred.shape[-2:], mode='nearest'
            ).squeeze(1)

        valid = targets > 0
        if not valid.any():
            return {}

        p, t = pred[valid], targets[valid].clamp(min=1e-3)
        abs_rel = ((p - t).abs() / t).mean().item()
        sq_rel  = (((p - t) ** 2) / t).mean().item()
        rmse    = ((p - t) ** 2).mean().sqrt().item()

        thresh = torch.max(p / t, t / p)
        d1 = (thresh < 1.25).float().mean().item()

        return {
            'abs_rel': abs_rel,
            'sq_rel':  sq_rel,
            'rmse':    rmse,
            'delta1':  d1,
        }
