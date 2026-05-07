# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MambaGaze Feedforward (Branch A) configuration."""

from transformers.configuration_utils import PretrainedConfig


class MambaGazeFFConfig(PretrainedConfig):
    """Config for MambaGaze Feedforward — non-AR single-pass token selector.

    Fundamentally different from MambaGazeConfig (v1 AR Mamba decoder):

      v1: Video → MambaVisionEncoder → Connector → MambaGazeDecoder (AR loop)
      v2: Video → CNN → Saliency + Motion → Mamba Aggregator → Top-K (one pass)

    Target: < 10 ms/frame gazing latency (AutoGaze baseline: 193 ms/frame).

    Training: distillation from AutoGaze teacher masks via binary cross-entropy.
    Reference: docs/mamba_gaze_ref_v0.md, Section 2 (Branch A).
    """

    model_type = "mamba_gaze_ff"

    def __init__(
        self,
        scales: str = "224",
        input_img_size: int = 224,
        num_vision_tokens_each_frame: int = 196,    # 14×14 patches

        # ── (1) Lightweight CNN encoder ──────────────────────────────────────
        cnn_channels: list = None,                  # per-stage channel widths
        cnn_out_channels: int = 192,                # final CNN output dim

        # ── (2) Spatio-temporal Mamba aggregator ─────────────────────────────
        mamba_dim: int = 128,                       # SSM hidden dim
        mamba_depth: int = 2,                       # [spatial+temporal] layer pairs
        mamba_d_state: int = 16,                    # SSM state size N
        mamba_d_conv: int = 4,                      # depthwise conv kernel
        mamba_expand: int = 2,                      # inner dim = expand × mamba_dim

        # ── (3) Training ──────────────────────────────────────────────────────
        gumbel_temperature: float = 1.0,            # Gumbel-top-k τ
        gazing_ratio_config: dict = None,
        **kwargs,
    ):
        self.scales = scales
        self.input_img_size = input_img_size
        self.num_vision_tokens_each_frame = num_vision_tokens_each_frame
        self.cnn_channels = cnn_channels or [32, 64, 128, cnn_out_channels]
        self.cnn_out_channels = cnn_out_channels
        self.mamba_dim = mamba_dim
        self.mamba_depth = mamba_depth
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.gumbel_temperature = gumbel_temperature
        self.gazing_ratio_config = gazing_ratio_config or {
            "sample_strategy_during_training": "fixed",
            "sample_strategy_during_inference": "fixed",
            "fixed": {"gazing_ratio": 0.5},
        }
        super().__init__(**kwargs)
