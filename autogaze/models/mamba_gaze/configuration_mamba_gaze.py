# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MambaGaze model configuration."""

from transformers.configuration_utils import PretrainedConfig


class MambaVisionConfig(PretrainedConfig):
    """Vision encoder config (MambaVisionEncoder)."""

    def __init__(
        self,
        hidden_dim: int = 192,
        out_dim: int = 192,
        depth: int = 4,
        kernel_size: int = 16,
        temporal_patch_size: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.temporal_patch_size = temporal_patch_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(**kwargs)


class MambaDecoderConfig(PretrainedConfig):
    """Gaze decoder config (MambaGazeDecoder).

    Replaces GazeDecoderConfig (LLaMA-based) with a Mamba SSM decoder.
    The interface surface is intentionally kept similar so MambaGazeModel
    can serve as a drop-in for AutoGazeModel.
    """

    def __init__(
        self,
        vocab_size: int = 197,       # 196 patch positions + 1 EOS
        eos_token_id: int = 196,
        hidden_size: int = 192,
        num_hidden_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(eos_token_id=eos_token_id, **kwargs)


class MambaGazeModelConfig(PretrainedConfig):
    """Wrapper config for MambaGazeModel."""

    def __init__(
        self,
        input_img_size: int = 224,
        num_vision_tokens_each_frame: int = 196,
        vision_config: dict = None,
        decoder_config: dict = None,
        **kwargs,
    ):
        vision_config = vision_config or {}
        decoder_config = decoder_config or {}

        n_patches = (input_img_size // vision_config.get("kernel_size", 16)) ** 2
        vision_config["out_dim"] = vision_config.get("hidden_dim", 192)

        decoder_config.update({
            "vocab_size": num_vision_tokens_each_frame + 1,
            "eos_token_id": num_vision_tokens_each_frame,
        })

        self.input_img_size = input_img_size
        self.num_vision_tokens_each_frame = num_vision_tokens_each_frame
        self.n_patches = n_patches
        self.vision_config = MambaVisionConfig(**vision_config)
        self.decoder_config = MambaDecoderConfig(**decoder_config)
        super().__init__(**kwargs)


class MambaGazeConfig(PretrainedConfig):
    """Top-level config for MambaGaze (mirrors AutoGazeConfig interface)."""

    model_type = "mamba_gaze"

    def __init__(
        self,
        scales: str = "224",
        num_vision_tokens_each_frame: int = 196,
        gazing_ratio_config: dict = None,
        mamba_gaze_model_config: dict = None,
        **kwargs,
    ):
        self.scales = scales
        self.num_vision_tokens_each_frame = num_vision_tokens_each_frame
        self.gazing_ratio_config = gazing_ratio_config or {
            "sample_strategy_during_training": "fixed",
            "sample_strategy_during_inference": "fixed",
            "fixed": {"gazing_ratio": 0.5},
        }
        cfg = mamba_gaze_model_config or {}
        cfg["num_vision_tokens_each_frame"] = num_vision_tokens_each_frame
        self.mamba_gaze_model_config = MambaGazeModelConfig(**cfg)
        super().__init__(**kwargs)
