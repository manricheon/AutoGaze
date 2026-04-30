# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object detection decoder for AutoGaze (DETR-lite style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import TaskDecoder


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DetectionDecoder(TaskDecoder):
    """DETR-lite object detector.

    Architecture:
        1. Temporal-pool encoder features → (B, N, C)
        2. Learnable object queries cross-attend to encoder memory.
           Gaze mask is used as key padding mask so queries focus
           on AutoGaze-selected spatial regions.
        3. Per-query: class head + box MLP (cx, cy, w, h) ∈ [0, 1].

    Loss: simplified (CE on classes + L1 on boxes without Hungarian matching).
    For production, replace with proper DETR bipartite matching loss.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        num_classes: int = 80,        # COCO classes
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        nhead: int = 6,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, feature_dim)

        # Input projection + norm
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers
        )

        # Prediction heads
        self.class_head = nn.Linear(feature_dim, num_classes + 1)  # +1: no-object
        self.box_head = _MLP(feature_dim, ffn_dim, 4, num_layers=3)

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, gaze_mask: torch.Tensor):
        """
        Args:
            features:  (B, T, N, C)
            gaze_mask: (B, T, N) bool

        Returns:
            dict:
                'class_logits': (B, Q, num_classes+1)
                'boxes':        (B, Q, 4)  cx/cy/w/h in [0,1]
        """
        B, T, N, C = features.shape

        # Temporal pool
        feat = features.mean(dim=1)           # (B, N, C)
        mask = gaze_mask.float().mean(dim=1)  # (B, N)  ∈ [0, 1]

        feat = self.feat_proj(feat)

        # Key padding mask: True = position is ignored in cross-attention.
        # We suppress non-gazed tokens so queries attend to selected regions.
        kpm = ~(mask > 0.5)  # (B, N) — True = not gazed

        # Object queries: (Q, C) → (B, Q, C)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        out = self.transformer_decoder(
            tgt=queries,
            memory=feat,
            memory_key_padding_mask=kpm,
        )  # (B, Q, C)

        class_logits = self.class_head(out)          # (B, Q, C+1)
        boxes = self.box_head(out).sigmoid()          # (B, Q, 4)

        return {'class_logits': class_logits, 'boxes': boxes}

    # ------------------------------------------------------------------ #
    def compute_loss(self, predictions: dict, targets) -> dict:
        """Simplified detection loss (no Hungarian matching).

        For a full DETR loss, replace with
        torchvision.ops._loss or detr_matcher from facebookresearch/detr.

        Args:
            predictions: dict from forward()
            targets: list of dicts, each with
                'labels': (num_gt,) int64
                'boxes':  (num_gt, 4) cx/cy/w/h in [0,1]
        """
        cls_logits = predictions['class_logits']   # (B, Q, C+1)
        pred_boxes = predictions['boxes']           # (B, Q, 4)
        B, Q = cls_logits.shape[:2]

        if targets is None:
            return {'loss': cls_logits.new_zeros(1).squeeze()}

        # Background label = num_classes (last logit)
        bg = self.num_classes
        gt_labels = targets.get('labels', None)  # (B, Q) expected for simplified loss
        gt_boxes  = targets.get('boxes',  None)

        cls_loss = box_loss = cls_logits.new_zeros(1).squeeze()

        if gt_labels is not None:
            cls_loss = F.cross_entropy(
                cls_logits.reshape(B * Q, -1),
                gt_labels.reshape(B * Q).long(),
                ignore_index=-1,
            )

        if gt_boxes is not None:
            mask = (gt_labels.reshape(B * Q) >= 0) if gt_labels is not None else \
                   torch.ones(B * Q, dtype=torch.bool, device=pred_boxes.device)
            if mask.any():
                box_loss = F.l1_loss(
                    pred_boxes.reshape(B * Q, 4)[mask],
                    gt_boxes.reshape(B * Q, 4)[mask],
                )

        loss = cls_loss + 5.0 * box_loss
        return {'loss': loss, 'cls_loss': cls_loss, 'box_loss': box_loss}

    def compute_metrics(self, predictions: dict, targets) -> dict:
        # mAP requires separate COCO-API evaluation; return empty here
        return {}
