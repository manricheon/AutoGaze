# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for all AutoGaze CV task decoders."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TaskDecoder(nn.Module, ABC):
    """Abstract base for task-specific decoders attached to AutoGaze.

    All decoders receive:
        features  (B, T, N, C) — dense spatial tokens from AutoGazeEncoder
        gaze_mask (B, T, N)    — boolean mask; True = token selected by AutoGaze

    T=1 for single-image tasks.  N=196 (14×14 grid).  C=192.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor, gaze_mask: torch.Tensor) -> torch.Tensor:
        """Decode features into task predictions.

        Args:
            features:  (B, T, N, C)
            gaze_mask: (B, T, N) bool

        Returns:
            Task-specific tensor or dict of tensors.
        """

    @abstractmethod
    def compute_loss(self, predictions, targets) -> dict:
        """Compute training loss.

        Returns:
            dict with at least key 'loss' (scalar tensor).
        """

    @abstractmethod
    def compute_metrics(self, predictions, targets) -> dict:
        """Compute evaluation metrics.

        Returns:
            dict of {metric_name: float}.
        """
