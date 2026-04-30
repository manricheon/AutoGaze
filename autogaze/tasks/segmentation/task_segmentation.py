# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic segmentation task wrapper."""

import torch
from torch import nn

from autogaze.decoders.segmentation import SegmentationDecoder


class TaskSegmentation(nn.Module):
    """Semantic segmentation task using AutoGaze-selected features.

    Args:
        num_classes:  Number of semantic categories.
        target_size:  Output spatial resolution.
        feature_dim:  AutoGaze encoder feature dim (default 192).
        mid_dims:     Upsampling channel progression tuple.
        scales:       (kept for config parity)
    """

    def __init__(
        self,
        num_classes: int = 150,
        target_size: int = 224,
        feature_dim: int = 192,
        mid_dims: tuple = (128, 64, 32, 16),
        scales: str = '224',
    ):
        super().__init__()
        self.scales = sorted([int(s) for s in str(scales).split('+')])
        self.decoder = SegmentationDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            target_size=target_size,
            mid_dims=mid_dims,
        )
        self.gaze_model_kwargs = {}

    # ------------------------------------------------------------------ #
    def forward_output(self, inputs, gaze_outputs, encoder_features):
        gazing_mask = torch.cat(gaze_outputs['gazing_mask'], dim=-1).bool()
        seg_logits = self.decoder(encoder_features, gazing_mask)
        return {'seg_logits': seg_logits}

    def loss(self, task_outputs, inputs):
        targets = inputs.get('seg_mask', None)
        return self.decoder.compute_loss(task_outputs['seg_logits'], targets)

    def reward(self, task_outputs, inputs):
        loss_dict = self.loss(task_outputs, inputs)
        return -loss_dict['loss'].detach()

    def metric(self, task_outputs, inputs):
        targets = inputs.get('seg_mask', None)
        return self.decoder.compute_metrics(task_outputs['seg_logits'], targets)
