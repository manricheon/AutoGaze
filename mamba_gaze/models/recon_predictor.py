"""
Reconstruction Loss Predictor.

Predicts per-frame reconstruction loss from Mamba backbone features.
Used as the frame-budget stopping criterion: skip frames whose predicted
loss is below a threshold ε (those frames are "easy" for the decoder).
Supervised with the AutoGaze block-causal VideoMAE reconstruction loss.
"""

import torch
import torch.nn as nn


class ReconPredictor(nn.Module):
    """
    Global average pool over (h, w) → 1-layer MLP → scalar loss prediction.

    Input:  H ∈ (B, T, h, w, d)
    Output: pred_recon ∈ (B, T)  — predicted per-frame reconstruction loss ≥ 0
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.norm = norm_cls(embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, T, h, w, d) → (B, T)"""
        B, T, h, w, d = H.shape
        x = H.reshape(B * T, h * w, d).mean(dim=1)   # global avg pool → (B*T, d)
        x = self.norm(x)
        out = self.head(x).squeeze(-1)                 # (B*T,)
        return torch.relu(out).reshape(B, T)           # relu: loss ≥ 0
