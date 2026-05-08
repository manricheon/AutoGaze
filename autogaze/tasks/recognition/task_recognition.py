# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognition task wrapper — mirrors VideoMAEReconstruction interface."""

import torch
from torch import nn

from autogaze.decoders.recognition import RecognitionDecoder


class TaskRecognition(nn.Module):
    """Image / video classification task using AutoGaze-selected features.

    Args:
        num_classes: Number of output classes.
        feature_dim: AutoGaze encoder feature dimension (default 192).
        hidden_dim:  MLP hidden dimension.
        dropout:     Dropout probability.
        scales:      (unused, kept for config parity with other tasks)
    """

    def __init__(
        self,
        num_classes: int = 1000,
        feature_dim: int = 192,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        scales: str = '224',
    ):
        super().__init__()
        self.scales = sorted([int(s) for s in str(scales).split('+')])
        self.decoder = RecognitionDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # kwargs passed to AutoGaze.forward — no target_scales override needed
        self.gaze_model_kwargs = {}

    # ------------------------------------------------------------------ #
    def forward_output(self, inputs, gaze_outputs, encoder_features):
        """Run the decoder.

        Args:
            inputs:           batch dict (must have 'label' for loss)
            gaze_outputs:     dict from AutoGaze.forward()
            encoder_features: (B, T, N, C) from AutoGazeEncoder

        Returns:
            dict with 'logits'
        """
        gazing_mask = torch.cat(gaze_outputs['gazing_mask'], dim=-1).bool()  # (B, T, N)
        logits = self.decoder(encoder_features, gazing_mask)
        return {'logits': logits}

    def loss(self, task_outputs, inputs):
        labels = inputs['label']
        return self.decoder.compute_loss(task_outputs['logits'], labels)

    def reward(self, task_outputs, inputs):
        # Used for GRPO: negative loss (higher = better)
        loss_dict = self.loss(task_outputs, inputs)
        return -loss_dict['loss'].detach()

    def metric(self, task_outputs, inputs):
        labels = inputs['label']
        return self.decoder.compute_metrics(task_outputs['logits'], labels)
