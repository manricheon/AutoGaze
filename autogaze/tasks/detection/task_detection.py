# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Object detection task wrapper."""

import torch
from torch import nn

from autogaze.decoders.detection import DetectionDecoder


class TaskDetection(nn.Module):
    """Object detection task using AutoGaze-selected features.

    Args:
        num_classes:       Number of object categories.
        num_queries:       Number of DETR object queries.
        num_decoder_layers: Transformer decoder depth.
        feature_dim:       AutoGaze encoder feature dim (default 192).
        nhead:             Multi-head attention heads.
        ffn_dim:           Feedforward dim in transformer decoder.
        dropout:           Dropout probability.
        scales:            (kept for config parity)
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_queries: int = 100,
        num_decoder_layers: int = 3,
        feature_dim: int = 192,
        nhead: int = 6,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        scales: str = '224',
    ):
        super().__init__()
        self.scales = sorted([int(s) for s in str(scales).split('+')])
        self.decoder = DetectionDecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.gaze_model_kwargs = {}

    # ------------------------------------------------------------------ #
    def forward_output(self, inputs, gaze_outputs, encoder_features):
        gazing_mask = torch.cat(gaze_outputs['gazing_mask'], dim=-1).bool()
        preds = self.decoder(encoder_features, gazing_mask)
        return preds  # {'class_logits': ..., 'boxes': ...}

    def loss(self, task_outputs, inputs):
        targets = inputs.get('detection_targets', None)
        return self.decoder.compute_loss(task_outputs, targets)

    def reward(self, task_outputs, inputs):
        loss_dict = self.loss(task_outputs, inputs)
        return -loss_dict['loss'].detach()

    def metric(self, task_outputs, inputs):
        targets = inputs.get('detection_targets', None)
        return self.decoder.compute_metrics(task_outputs, targets)
