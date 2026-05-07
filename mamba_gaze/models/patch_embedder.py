"""
Patch Embedder: video (B,T,3,H,W) → patch tokens (B,T,h,w,d).

Single-scale 16×16 patch embedding matching ViT convention.
Learnable per-frame spatial positional embedding.
"""

import torch
import torch.nn as nn


class PatchEmbedder(nn.Module):
    """
    2D Conv(k=patch_size, s=patch_size) + learnable spatial PE.

    Input:  (B, T, 3, H, W)   — H = W = img_size
    Output: (B, T, h, w, d)   — h = w = img_size // patch_size = 14
    """

    def __init__(self, embed_dim: int = 192, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.h = self.w = img_size // patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.h, self.w, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 3, H, W)  →  (B, T, h, w, d)"""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)                                  # (B*T, d, h, w)
        x = x.permute(0, 2, 3, 1)                        # (B*T, h, w, d)
        x = x.reshape(B, T, self.h, self.w, self.embed_dim)
        x = x + self.pos_embed                            # broadcast over B, T
        return x
