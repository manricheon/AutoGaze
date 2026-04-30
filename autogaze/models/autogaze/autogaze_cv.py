# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoGaze CV utilities.

AutoGazeTokenSelector
    Zero-shot token selection for any pretrained ViT-based model.
    AutoGaze computes a 14×14 gaze map; this class interpolates it to
    the target model's patch grid and applies it via a forward hook —
    without touching the pretrained decoder at all.

AutoGazeEncoder (kept for future fine-tuning experiments)
    Exposes AutoGaze's own vision encoder features (B, T, 196, 192).
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .autogaze import AutoGaze


# ══════════════════════════════════════════════════════════════════════════════
# Main class — zero-shot transfer to pretrained models
# ══════════════════════════════════════════════════════════════════════════════

class AutoGazeTokenSelector:
    """Apply AutoGaze gaze selection to any ViT-based pretrained model.

    Usage::

        selector = AutoGazeTokenSelector(ag_model, gazing_ratio=0.5)

        # 1. Compute mask for the target model's patch grid
        mask = selector.compute_gaze_mask(ag_video, target_h=16, target_w=16)
        # mask: (B, 256) bool  [for a 224px / patch16 model]

        # 2. Run target model with mask applied via hook
        with selector.token_mask_context(vit_embed_module, mask, has_cls_token=True):
            output = pretrained_model(**inputs)

    Notes
    -----
    * ``ag_video`` must be preprocessed with :class:`AutoGazeImageProcessor`
      (224 × 224, rescaled to [−1, 1]).
    * ``vit_embed_module`` is the submodule whose **output** is the sequence
      of patch embeddings.  Common locations:

        - DINOv2: ``model.embeddings``
        - Depth Anything V2 (DINOv2 backbone): ``model.backbone.embeddings``
        - YOLOS: ``model.vit.embeddings``
        - ViT (HF): ``model.vit.embeddings``

    * The hook **zeroes out** non-selected tokens (preserves [CLS] if present).
      The pretrained decoder is completely unchanged.
    """

    AG_GRID = 14  # AutoGaze produces 14×14 = 196 gaze scores

    def __init__(self, autogaze_model: AutoGaze, gazing_ratio: float = 0.5):
        self.ag = autogaze_model
        self.gazing_ratio = gazing_ratio

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_gaze_mask(
        self,
        ag_video: torch.Tensor,
        target_h: int,
        target_w: int,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run AutoGaze and return a token mask at the target grid resolution.

        Args:
            ag_video:  (B, T, C, H, W) preprocessed for AutoGaze (224 × 224).
                       For single-image tasks use T = 1.
            target_h:  Patch grid height of the target ViT.
            target_w:  Patch grid width of the target ViT.
            threshold: Threshold applied after bilinear interpolation.

        Returns:
            mask: (B, target_h * target_w) bool tensor.
                  True = selected (gazed), False = suppressed.
        """
        was_training = self.ag.training
        self.ag.eval()

        gaze_outputs = self.ag(
            {'video': ag_video},
            gazing_ratio=self.gazing_ratio,
            generate_only=True,
        )

        if was_training:
            self.ag.train()

        # gazing_mask is a list of (B, T, N_i) at multiple scales (2×2, 4×4, 7×7, 14×14).
        # Use only the finest 14×14 scale (last element = 196 tokens).
        raw = gaze_outputs['gazing_mask'][-1]   # (B, T, 196)
        raw = raw[:, 0].float()                 # (B, 196) — first frame
        raw_2d = raw.reshape(-1, self.AG_GRID, self.AG_GRID)  # (B, 14, 14)

        if target_h != self.AG_GRID or target_w != self.AG_GRID:
            raw_2d = F.interpolate(
                raw_2d.unsqueeze(1),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)  # (B, target_h, target_w)

        return (raw_2d > threshold).reshape(raw_2d.shape[0], -1)  # (B, N_target)

    # ------------------------------------------------------------------ #
    @contextmanager
    def token_mask_context(
        self,
        embed_module: nn.Module,
        mask: torch.Tensor,
        has_cls_token: bool = True,
    ):
        """Context manager that temporarily hooks *embed_module* to apply mask.

        Inside this context the module's forward output has non-selected patch
        tokens zeroed out.  [CLS] (and [REG]) tokens are always preserved.

        Args:
            embed_module: The ViT submodule whose output is the token sequence.
            mask:         (B, N_patches) bool.  True = keep, False = zero.
            has_cls_token: True for most ViTs (first token is [CLS]).
        """
        _mask = mask  # captured by closure
        N_mask = mask.shape[-1]  # number of patch tokens in mask

        def _hook(module, input, output):
            # output shape: (B, N_seq, D)
            # N_seq = [CLS] + N_patches + [suffix tokens, e.g. YOLOS det-tokens]
            B = output.shape[0]
            if has_cls_token:
                prefix = output[:, :1, :]           # [CLS]
                rest = output[:, 1:, :]
                if rest.shape[1] > N_mask:
                    # suffix tokens exist (e.g. YOLOS detection tokens)
                    patches = rest[:, :N_mask, :]
                    suffix  = rest[:, N_mask:, :]
                    w = _mask[:B].float().unsqueeze(-1)
                    return torch.cat([prefix, patches * w, suffix], dim=1)
                else:
                    N = rest.shape[1]
                    w = _mask[:B, :N].float().unsqueeze(-1)
                    return torch.cat([prefix, rest * w], dim=1)
            else:
                if output.shape[1] > N_mask:
                    patches = output[:, :N_mask, :]
                    suffix  = output[:, N_mask:, :]
                    w = _mask[:B].float().unsqueeze(-1)
                    return torch.cat([patches * w, suffix], dim=1)
                else:
                    N = output.shape[1]
                    w = _mask[:B, :N].float().unsqueeze(-1)
                    return output * w

        handle = embed_module.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------------ #
    def patch_grid_size(self, image_size: int, patch_size: int) -> tuple[int, int]:
        """Convenience: compute patch grid (h, w) from image and patch sizes."""
        h = image_size // patch_size
        return h, h


# ══════════════════════════════════════════════════════════════════════════════
# Encoder wrapper (kept for future fine-tuning experiments)
# ══════════════════════════════════════════════════════════════════════════════

class AutoGazeEncoder(nn.Module):
    """Expose AutoGaze's own ShallowVideoConvNet features (B, T, 196, 192).

    Useful when you want to train new lightweight decoders on top of
    AutoGaze's internal representation rather than using a pretrained model.
    """

    def __init__(self, autogaze_model: AutoGaze):
        super().__init__()
        self.ag = autogaze_model

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, T, 196, 192)."""
        B, T = video.shape[:2]
        size = self.ag.gazing_model.input_img_size  # 224
        v2d = rearrange(video, 'b t c h w -> (b t) c h w')
        v2d = F.interpolate(v2d, size=(size, size), mode='bicubic', align_corners=False)
        video_r = rearrange(v2d, '(b t) c h w -> b t c h w', b=B)

        feats, _ = self.ag.gazing_model.vision_model(video_r, use_cache=False, past_conv_values=None)
        feats = feats.transpose(1, 2)
        feats = rearrange(feats, 'b t c h w -> b t (h w) c')
        feats = self.ag.gazing_model.connector(feats)
        return feats  # (B, T, 196, 192)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.encode(video)
