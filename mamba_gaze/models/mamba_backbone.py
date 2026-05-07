"""
Mamba Backbone: 4–6 layers of temporal causal Mamba + spatial depthwise mixer.

Time-major scan: tokens are ordered (t=0, all spatial), (t=1, all spatial), …
so causal Mamba is automatically causal in time.

Tries mamba_ssm (CUDA-accelerated) first; falls back to a pure-PyTorch
selective SSM for CPU / Apple MPS environments.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaSSM
    _MAMBA_AVAILABLE = True
except (ImportError, RuntimeError):
    _MAMBA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch S6 fallback
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Simplified Mamba S6 (selective state space) in pure PyTorch.

    Uses a sequential recurrent scan — correct but O(L) per sample.
    Replace with CUDA parallel scan for production.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner  = d_inner
        self.d_state  = d_state
        self.d_conv   = d_conv

        self.in_proj   = nn.Linear(d_model, d_inner * 2, bias=False)  # x and z gate
        self.conv1d    = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj    = nn.Linear(d_inner, d_state * 2 + d_inner, bias=False)  # B, C, delta
        self.dt_proj   = nn.Linear(d_inner, d_inner)
        self.out_proj  = nn.Linear(d_inner, d_model, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_inner, -1))
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        nn.init.trunc_normal_(self.x_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # Depthwise conv for local context
        x_conv = self.conv1d(x_in.permute(0, 2, 1))[:, :, :L].permute(0, 2, 1)
        x_conv = F.silu(x_conv)

        # Input-dependent B, C, delta
        bcdt = self.x_proj(x_conv)                        # (B, L, 2*d_state + d_inner)
        B_in = bcdt[..., :self.d_state]
        C_in = bcdt[..., self.d_state: 2 * self.d_state]
        dt   = F.softplus(self.dt_proj(bcdt[..., 2 * self.d_state:]))  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())                 # (d_inner, d_state)

        # ZOH discretization: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)              # (B, L, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * B_in.unsqueeze(2) * x_conv.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Sequential scan
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dBu[:, i]                 # (B, d_inner, d_state)
            y = (h * C_in[:, i].unsqueeze(1)).sum(-1)     # (B, d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)                         # (B, L, d_inner)
        y = y + self.D * x_conv                            # skip connection
        y = y * F.silu(z)                                  # gate
        return self.out_proj(y)


# ─────────────────────────────────────────────────────────────────────────────
# Mamba block (wraps either mamba_ssm or SelectiveSSM)
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.norm = norm_cls(d_model)
        if _MAMBA_AVAILABLE:
            self.ssm = _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """x: (B, L, d)  →  (B, L, d)"""
        return x + self.ssm(self.norm(x)), None


# ─────────────────────────────────────────────────────────────────────────────
# Spatial mixer: depthwise + pointwise conv over the 2D patch grid
# ─────────────────────────────────────────────────────────────────────────────

class SpatialMixer(nn.Module):
    """Lightweight spatial context aggregation via 3×3 DW-conv + 1×1 PW-conv."""

    def __init__(self, d: int, h: int, w: int):
        super().__init__()
        norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.norm = norm_cls(d)
        self.dw   = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
        self.pw   = nn.Conv2d(d, d, 1, bias=False)
        self.act  = nn.GELU()
        self.h, self.w = h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (BT, N, d) → (BT, N, d)"""
        BT, N, d = x.shape
        r = self.norm(x)
        r = r.view(BT, self.h, self.w, d).permute(0, 3, 1, 2)  # (BT, d, h, w)
        r = self.act(self.dw(r))
        r = self.pw(r)
        r = r.permute(0, 2, 3, 1).reshape(BT, N, d)
        return x + r


# ─────────────────────────────────────────────────────────────────────────────
# Full backbone layer: temporal Mamba + spatial mixer
# ─────────────────────────────────────────────────────────────────────────────

class MambaLayer(nn.Module):
    def __init__(self, d: int, d_state: int, d_conv: int, expand: int, h: int, w: int):
        super().__init__()
        self.temporal = MambaBlock(d, d_state, d_conv, expand)
        self.spatial  = SpatialMixer(d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, h, w, d) → (B, T, h, w, d)"""
        B, T, h, w, d = x.shape
        N = h * w

        # Time-major causal Mamba: order tokens as t0·spatial, t1·spatial, …
        seq = x.reshape(B, T * N, d)
        seq, _ = self.temporal(seq)                      # (B, T*N, d)
        x = seq.reshape(B, T, h, w, d)

        # Spatial mixer per-frame
        x_2d = x.reshape(B * T, N, d)
        x_2d = self.spatial(x_2d)
        return x_2d.reshape(B, T, h, w, d)


# ─────────────────────────────────────────────────────────────────────────────
# MambaBackbone
# ─────────────────────────────────────────────────────────────────────────────

class MambaBackbone(nn.Module):
    """
    Stack of MambaLayers.

    Input:  F ∈ (B, T, h, w, d)   — patch embeddings
    Output: H ∈ (B, T, h, w, d)   — contextualised features
    """

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        h: int = 14,
        w: int = 14,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(embed_dim, d_state, d_conv, expand, h, w)
            for _ in range(depth)
        ])
        norm_cls = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.norm = norm_cls(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, h, w, d) → (B, T, h, w, d)"""
        for layer in self.layers:
            x = layer(x)
        B, T, h, w, d = x.shape
        x = self.norm(x.reshape(B * T, h * w, d)).reshape(B, T, h, w, d)
        return x
