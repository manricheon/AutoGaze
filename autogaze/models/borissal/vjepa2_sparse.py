"""Training-only V-JEPA2 teacher utilities for Borissal v1 SSL (Phase 3).

This is the ONE core-package file allowed to import transformers (pinned
==5.5.0); the selector models themselves stay torch-only. Verified against
the installed 5.5.0 vjepa2 module: `apply_masks` gathers by keep-index
lists, `VJEPA2Layer.forward(hidden_states, position_mask)` routes flat
t-major indices into 3D-RoPE via `get_position_ids`, and the token flatten
order matches Borissal's canonical `idx = t*N + n` exactly.

Design: teacher-agnostic interface -- anything providing `dense_features`,
`sparse_features`, and `predict` with the same shapes can replace
`VJEPA2Teacher` (e.g. a torch.hub vjepa2.1l/b adapter) without touching the
trainer.
"""

from typing import Optional

import torch
import torch.nn as nn

from transformers import VJEPA2Config, VJEPA2Model


def tiny_vjepa2_config(crop_size: int = 128, frames_per_clip: int = 16) -> VJEPA2Config:
    """A drastically reduced, randomly-initialized V-JEPA2 config for Mac smoke
    runs (graph/shape/gradient checks without downloading a checkpoint).
    Keeps patch_size=16 / tubelet_size=2 identical to the real models."""
    return VJEPA2Config(
        crop_size=crop_size,
        frames_per_clip=frames_per_clip,
        patch_size=16,
        tubelet_size=2,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=2.0,
        pred_hidden_size=32,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=2,
        pred_num_mask_tokens=10,
        pred_mlp_ratio=2.0,
    )


def sparse_encoder_forward(
    encoder,                     # transformers VJEPA2Encoder (weights shared, no copy)
    pixel_values_videos: torch.Tensor,   # (B, T, C, H, W)
    keep_index: torch.Tensor,    # (B, K) long, canonical ascending t-major, NO -1 padding
    gate: Optional[torch.Tensor] = None,  # (B, K) float; forward-1.0 ST gate for selector grads
) -> torch.Tensor:
    """Run the V-JEPA2 encoder on a token SUBSET.

    The stock VJEPA2Encoder.forward always runs dense (hardcodes
    position_mask=None). This reimplements its ~8-line body with the two
    changes the sparse path needs: gather embeddings by keep_index, and pass
    keep_index as position_mask so RoPE sees original (t, h, w) positions.
    Returns (B, K, D).
    """
    if (keep_index < 0).any():
        raise ValueError("keep_index must be unpadded (no -1); use uniform allocation for training")

    hidden_states = encoder.embeddings(pixel_values_videos)          # (B, L, D)
    index = keep_index.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
    hidden_states = torch.gather(hidden_states, dim=1, index=index)  # (B, K, D)
    if gate is not None:
        hidden_states = hidden_states * gate.unsqueeze(-1)

    for layer_module in encoder.layer:
        hidden_states = layer_module(hidden_states, keep_index)[0]

    return encoder.layernorm(hidden_states)


class VJEPA2Teacher(nn.Module):
    """Frozen V-JEPA2 wrapped as an SSL teacher for selector training.

    - `dense_features(video)`: full-token encoder features, no_grad (targets).
    - `sparse_features(video, keep_index, gate)`: subset encoder features;
      params frozen but gradients FLOW THROUGH to `gate` (the selector's
      straight-through connection).
    - `predict(sparse_feats, context_idx, target_idx, num_tokens)`: V-JEPA2
      predictor infills target-token features from the sparse context.
    """

    def __init__(self, model: VJEPA2Model):
        super().__init__()
        cfg = model.config
        if cfg.patch_size != 16 or cfg.tubelet_size != 2:
            raise ValueError(
                f"teacher grid mismatch: expected patch_size=16/tubelet_size=2, "
                f"got {cfg.patch_size}/{cfg.tubelet_size}"
            )
        model.eval()
        model.requires_grad_(False)
        self.model = model

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "VJEPA2Teacher":
        return cls(VJEPA2Model.from_pretrained(name_or_path, **kwargs))

    @classmethod
    def tiny_random(cls, **kwargs) -> "VJEPA2Teacher":
        return cls(VJEPA2Model(tiny_vjepa2_config(**kwargs)))

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def dense_features(self, video: torch.Tensor) -> torch.Tensor:
        """video (B, T, C, H, W) -> (B, L, D) full-token features."""
        return self.model.encoder(pixel_values_videos=video).last_hidden_state

    def sparse_features(
        self, video: torch.Tensor, keep_index: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """(B, K, D) features of the kept subset. Differentiable w.r.t. gate."""
        return sparse_encoder_forward(self.model.encoder, video, keep_index, gate)

    def predict(
        self,
        sparse_feats: torch.Tensor,   # (B, K, D) student features at context_idx
        context_idx: torch.Tensor,    # (B, K) long
        target_idx: torch.Tensor,     # (B, K_t) long
        num_tokens: int,              # L, the full token count
    ) -> torch.Tensor:
        """Predict target-token features from the sparse context via the
        built-in V-JEPA2 predictor. Returns (B, K_t, D).

        The stock predictor takes a full-length hidden-state tensor and
        gathers the context itself, so we scatter the sparse features into a
        zeros buffer first -- positions outside context_idx are never read.
        """
        B, K, D = sparse_feats.shape
        full = sparse_feats.new_zeros(B, num_tokens, D)
        full = full.scatter(1, context_idx.unsqueeze(-1).expand(-1, -1, D), sparse_feats)
        out = self.model.predictor(
            encoder_hidden_states=full,
            context_mask=[context_idx],
            target_mask=[target_idx],
        )
        return out.last_hidden_state
