"""
Converts between AutoGaze sequential index format and multi-hot tensors.

AutoGaze vocabulary (scales=[32,64,112,224], patch_size=16):
  Scale  32px → (32/16)² =  2×2 =  4 tokens  — global indices [0,   4)
  Scale  64px → (64/16)² =  4×4 = 16 tokens  — global indices [4,  20)
  Scale 112px → (112/16)² = 7×7 = 49 tokens  — global indices [20, 69)
  Scale 224px → (224/16)² =14×14=196 tokens  — global indices [69,265)
  Total: 265 tokens per frame
"""

import itertools
from typing import List, Optional

import torch

SCALES = [32, 64, 112, 224]
PATCH_SIZE = 16
SCALE_HW = [(s // PATCH_SIZE, s // PATCH_SIZE) for s in SCALES]     # [(2,2),(4,4),(7,7),(14,14)]
SCALE_PATCHES = [h * w for h, w in SCALE_HW]                         # [4, 16, 49, 196]
N_TOKENS = sum(SCALE_PATCHES)                                          # 265
SCALE_OFFSETS = [0] + list(itertools.accumulate(SCALE_PATCHES))       # [0,4,20,69,265]


def seq_to_multihot(
    indices: torch.Tensor,
    n: int = N_TOKENS,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """1-D token indices → (n,) binary float tensor."""
    indices = torch.as_tensor(indices, dtype=torch.long)
    dev = device if device is not None else indices.device
    out = torch.zeros(n, device=dev, dtype=torch.float32)
    if indices.numel() > 0:
        out.scatter_(0, indices.clamp(0, n - 1), 1.0)
    return out


def multihot_to_per_scale(multihot: torch.Tensor) -> List[torch.Tensor]:
    """(265,) multi-hot → list of 4 per-scale tensors [(4,),(16,),(49,),(196,)]."""
    return [multihot[SCALE_OFFSETS[i]: SCALE_OFFSETS[i + 1]] for i in range(len(SCALES))]


def seq_to_per_scale(indices: torch.Tensor) -> List[torch.Tensor]:
    """Convenience: global indices → per-scale multi-hot list."""
    return multihot_to_per_scale(seq_to_multihot(indices))


def per_scale_to_multihot(per_scale: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-scale tensors → (265,) multi-hot."""
    return torch.cat(per_scale, dim=0)


def per_scale_to_seq(per_scale: List[torch.Tensor]) -> torch.Tensor:
    """Per-scale multi-hot → global token indices (selected positions only)."""
    return per_scale_to_multihot(per_scale).nonzero(as_tuple=True)[0]


def batch_seq_to_multihot(
    batch_indices: List[torch.Tensor],
    n: int = N_TOKENS,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Variable-length index lists → (B, n) batch of multi-hot tensors."""
    dev = device if device is not None else (batch_indices[0].device if batch_indices else None)
    B = len(batch_indices)
    out = torch.zeros(B, n, device=dev, dtype=torch.float32)
    for b, idx in enumerate(batch_indices):
        idx = torch.as_tensor(idx, dtype=torch.long, device=dev)
        if idx.numel() > 0:
            out[b].scatter_(0, idx.clamp(0, n - 1), 1.0)
    return out


def global_idx_to_scale_local(idx: int):
    """Return (scale_index, local_position) for a global vocabulary index."""
    for s, (lo, hi) in enumerate(zip(SCALE_OFFSETS[:-1], SCALE_OFFSETS[1:])):
        if lo <= idx < hi:
            return s, idx - lo
    raise ValueError(f"Index {idx} out of range [0, {N_TOKENS})")
