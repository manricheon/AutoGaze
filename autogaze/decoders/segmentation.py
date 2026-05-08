# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic segmentation decoder for AutoGaze (FPN-lite style)."""

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
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )


class SegmentationDecoder(TaskDecoder):
    """Lightweight FPN-style segmentation head.

    Pipeline:
        (B, T, N, C) features
          → temporal mean-pool   → (B, N, C)
          → reshape              → (B, C, 14, 14)   [N=196=14×14]
          → gaze attention gate  → highlight selected patches
          → 4× UpBlock (×2 each) → 14→28→56→112→224
          → Conv(16, num_classes) → (B, num_classes, 224, 224)

    The gaze mask is turned into a spatial 14×14 weight map that gates the
    feature channels before upsampling, amplifying responses in selected regions.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        num_classes: int = 150,   # ADE20K default; 21 for PASCAL, 80 for COCO
        target_size: int = 224,
        mid_dims: tuple = (128, 64, 32, 16),
    ):
        super().__init__()
        self.target_size = target_size
        self.num_classes = num_classes

        # Gaze attention gate: conv that produces per-channel importance weights
        self.gaze_gate = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # FPN upsampling stack
        dims = (feature_dim,) + mid_dims
        self.upsample_blocks = nn.ModuleList(
            [_UpBlock(in_d, out_d) for in_d, out_d in zip(dims[:-1], dims[1:])]
        )

        # Final prediction head
        self.pred_head = nn.Conv2d(mid_dims[-1], num_classes, 1)

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, gaze_mask: torch.Tensor):
        """
        Args:
            features:  (B, T, N, C)
            gaze_mask: (B, T, N) bool

        Returns:
            logits: (B, num_classes, target_size, target_size)
        """
        B, T, N, C = features.shape
        H = W = int(N ** 0.5)  # 14

        # Temporal pool
        feat = features.mean(dim=1)           # (B, N, C)
        mask = gaze_mask.float().mean(dim=1)  # (B, N)

        # Reshape to spatial maps
        feat_map = rearrange(feat, 'b (h w) c -> b c h w', h=H, w=W)   # (B, C, 14, 14)
        mask_map = rearrange(mask, 'b (h w) -> b 1 h w', h=H, w=W)     # (B, 1, 14, 14)

        # Gaze attention gate: scale features in selected regions
        gate = self.gaze_gate(feat_map)                        # (B, C, 14, 14)
        feat_map = feat_map * (1.0 + mask_map * gate)

        # Progressive upsampling
        x = feat_map
        for block in self.upsample_blocks:
            x = block(x)

        # Ensure exact target size
        if x.shape[-2:] != (self.target_size, self.target_size):
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        return self.pred_head(x)  # (B, num_classes, target_size, target_size)

    # ------------------------------------------------------------------ #
    def compute_loss(self, predictions: torch.Tensor, targets) -> dict:
        """Pixel-wise cross-entropy loss.

        Args:
            predictions: (B, num_classes, H, W)
            targets: (B, H, W) integer class indices; 255 = ignore
        """
        if targets is None:
            return {'loss': predictions.new_zeros(1).squeeze()}

        # Resize targets to match prediction size if needed
        pred_h, pred_w = predictions.shape[-2:]
        if targets.shape[-2:] != (pred_h, pred_w):
            targets = F.interpolate(
                targets.float().unsqueeze(1), size=(pred_h, pred_w), mode='nearest'
            ).squeeze(1).long()

        loss = F.cross_entropy(predictions, targets.long(), ignore_index=255)
        return {'loss': loss, 'seg_loss': loss}

    def compute_metrics(self, predictions: torch.Tensor, targets) -> dict:
        if targets is None:
            return {}
        pred_labels = predictions.argmax(dim=1)

        # Pixel accuracy (ignoring class 255)
        valid = (targets != 255)
        pixel_acc = (pred_labels[valid] == targets[valid]).float().mean()

        # Mean IoU (approximate, not using ignore_index per class)
        iou_list = []
        for cls in range(self.num_classes):
            pred_c = pred_labels == cls
            tgt_c  = (targets == cls) & valid
            inter = (pred_c & tgt_c).sum().float()
            union = (pred_c | tgt_c).sum().float()
            if union > 0:
                iou_list.append((inter / union).item())
        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0

        return {'pixel_acc': pixel_acc.item(), 'mean_iou': mean_iou}
