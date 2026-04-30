# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognition (image/video classification) decoder for AutoGaze."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TaskDecoder


class RecognitionDecoder(TaskDecoder):
    """Gaze-weighted pool → MLP classifier.

    Pipeline:
        (B, T, N, C) features
          → temporal mean-pool          → (B, N, C)
          → gaze-weighted spatial pool  → (B, C)
          → LayerNorm → Linear → GELU → Dropout → Linear
          → (B, num_classes) logits

    The gaze mask is used as soft attention weights during spatial pooling:
    tokens selected by AutoGaze contribute proportionally more to the
    pooled representation.
    """

    def __init__(
        self,
        feature_dim: int = 192,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------ #
    def forward(self, features: torch.Tensor, gaze_mask: torch.Tensor):
        """
        Args:
            features:  (B, T, N, C)
            gaze_mask: (B, T, N) bool

        Returns:
            logits: (B, num_classes)
        """
        # Temporal pool
        feat = features.mean(dim=1)           # (B, N, C)
        mask = gaze_mask.float().mean(dim=1)  # (B, N)  — fraction of frames each token was selected

        # Gaze-weighted spatial pool; fall back to uniform if nothing is selected
        weights = mask / mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, N)
        pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)                # (B, C)

        return self.classifier(pooled)  # (B, num_classes)

    # ------------------------------------------------------------------ #
    def compute_loss(self, predictions: torch.Tensor, targets) -> dict:
        """Cross-entropy loss.

        Args:
            predictions: (B, num_classes) logits
            targets: (B,) integer class labels
        """
        loss = F.cross_entropy(predictions, targets.long())
        return {'loss': loss, 'cls_loss': loss}

    def compute_metrics(self, predictions: torch.Tensor, targets) -> dict:
        pred_labels = predictions.argmax(dim=-1)
        acc1 = (pred_labels == targets).float().mean()

        # Top-5 accuracy
        top5 = predictions.topk(min(5, self.num_classes), dim=-1).indices
        acc5 = (top5 == targets.unsqueeze(-1)).any(dim=-1).float().mean()

        return {'acc1': acc1.item(), 'acc5': acc5.item()}
