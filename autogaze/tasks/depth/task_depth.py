# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monocular depth estimation task wrapper."""

import torch
from torch import nn

from autogaze.decoders.depth import DepthDecoder


class TaskDepth(nn.Module):
    """Depth estimation task using AutoGaze-selected features.

    Args:
        target_size:  Output spatial resolution.
        max_depth:    Maximum depth value (metres).
        feature_dim:  AutoGaze encoder feature dim (default 192).
        mid_dims:     Upsampling channel progression tuple.
        scales:       (kept for config parity)
    """

    def __init__(
        self,
        target_size: int = 224,
        max_depth: float = 10.0,
        feature_dim: int = 192,
        mid_dims: tuple = (128, 64, 32, 16),
        scales: str = '224',
    ):
        super().__init__()
        self.scales = sorted([int(s) for s in str(scales).split('+')])
        self.decoder = DepthDecoder(
            feature_dim=feature_dim,
            target_size=target_size,
            max_depth=max_depth,
            mid_dims=mid_dims,
        )
        self.gaze_model_kwargs = {}

    # ------------------------------------------------------------------ #
    def forward_output(self, inputs, gaze_outputs, encoder_features):
        gazing_mask = torch.cat(gaze_outputs['gazing_mask'], dim=-1).bool()
        depth = self.decoder(encoder_features, gazing_mask)
        return {'depth': depth}

    def loss(self, task_outputs, inputs):
        targets = inputs.get('depth_map', None)
        return self.decoder.compute_loss(task_outputs['depth'], targets)

    def reward(self, task_outputs, inputs):
        loss_dict = self.loss(task_outputs, inputs)
        return -loss_dict['loss'].detach()

    def metric(self, task_outputs, inputs):
        targets = inputs.get('depth_map', None)
        return self.decoder.compute_metrics(task_outputs['depth'], targets)
