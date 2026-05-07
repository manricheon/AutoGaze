"""
Multi-Scale Selection Head.

AutoGaze vocabulary (265 tokens per frame):
  Scale 32px  → 2×2  =  4 tokens   [global 0..3]
  Scale 64px  → 4×4  = 16 tokens   [global 4..19]
  Scale 112px → 7×7  = 49 tokens   [global 20..68]
  Scale 224px → 14×14=196 tokens   [global 69..264]

For each scale:
  1. AdaptiveAvgPool2d to target spatial size
  2. 1×1 Conv → scalar logit per token
  3. Training:  Gumbel-top-k (temperature annealing 1.0 → 0.1)
  4. Inference: hard top-k

Frame budget: per-frame mask from ReconPredictor zeroes out selections
for frames whose predicted reconstruction loss < ε.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.mask_converter import SCALE_HW, SCALE_PATCHES, N_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Per-scale head
# ─────────────────────────────────────────────────────────────────────────────

class ScaleHead(nn.Module):
    """Pool H to (h_out, w_out) and apply a 1×1 conv to get logits."""

    def __init__(self, embed_dim: int, h_out: int, w_out: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((h_out, w_out))
        self.head = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BT, d, h, w) → (BT, N_s)"""
        x = self.pool(x)             # (BT, d, h_out, w_out)
        return self.head(x).squeeze(1).flatten(1)   # (BT, N_s)


# ─────────────────────────────────────────────────────────────────────────────
# Gumbel-top-k (training) and hard top-k (inference)
# ─────────────────────────────────────────────────────────────────────────────

def gumbel_topk(logits: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """
    Differentiable top-k via Gumbel perturbation + sigmoid thresholding.

    logits: (B, T, N)   — unnormalised
    Returns soft mask (B, T, N) ∈ (0, 1) with ≈ k ones per row.
    Gradients flow through logits via the sigmoid.
    """
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
    perturbed = (logits + noise) / tau
    threshold = perturbed.topk(k, dim=-1).values[..., -1:]      # (B, T, 1)
    return torch.sigmoid((perturbed - threshold) / tau)


def hard_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Hard top-k mask. No gradients."""
    mask = torch.zeros_like(logits)
    idx  = logits.topk(k, dim=-1).indices                        # (B, T, k)
    mask.scatter_(-1, idx, 1.0)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# MultiScaleSelectionHead
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleSelectionHead(nn.Module):
    """
    Applies per-scale ScaleHead to H, then Gumbel (train) or hard (eval) top-k.

    The gazing_ratio applies uniformly to every scale.
    frame_budget_mask (B, T) — 1 = process frame, 0 = skip (easy frame).
    """

    def __init__(
        self,
        embed_dim: int = 192,
        gazing_ratio: float = 0.5,
        gumbel_temp: float = 1.0,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.gazing_ratio = gazing_ratio
        self.temperature  = gumbel_temp

        self.scale_heads = nn.ModuleList([
            ScaleHead(embed_dim, h, w) for h, w in SCALE_HW   # 4 heads
        ])

    def forward(
        self,
        H: torch.Tensor,
        gazing_ratio: Optional[float] = None,
        temperature: Optional[float] = None,
        frame_budget_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        H: (B, T, h, w, d)

        Returns:
            per_scale_logits : list[4] of (B, T, N_s)
            per_scale_masks  : list[4] of (B, T, N_s)
        """
        B, T, h, w, d = H.shape
        ratio = gazing_ratio if gazing_ratio is not None else self.gazing_ratio
        tau   = temperature  if temperature  is not None else self.temperature

        # (BT, d, h, w) for pooling
        x = H.permute(0, 1, 4, 2, 3).reshape(B * T, d, h, w)

        per_scale_logits: List[torch.Tensor] = []
        per_scale_masks:  List[torch.Tensor] = []

        for head, n_s in zip(self.scale_heads, SCALE_PATCHES):
            logits_flat = head(x)                              # (BT, N_s)
            logits = logits_flat.reshape(B, T, n_s)           # (B, T, N_s)
            per_scale_logits.append(logits)

            k = max(1, int(ratio * n_s))

            if self.training:
                mask = gumbel_topk(logits, k, tau)
            else:
                with torch.no_grad():
                    mask = hard_topk(logits, k)

            # Apply frame budget: zero out skipped frames
            if frame_budget_mask is not None:
                budget = frame_budget_mask.unsqueeze(-1)       # (B, T, 1)
                mask = mask * budget

            per_scale_masks.append(mask)

        return per_scale_logits, per_scale_masks

    def set_temperature(self, temp: float) -> None:
        self.temperature = max(temp, 1e-4)
