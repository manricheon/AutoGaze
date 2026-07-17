"""Borissal v0.3 candidate-bank signal primitives (classical, zero learned weights).

Pure functions over grid- or pixel-resolution maps, `torch`-only (standalone-
portable, same rule as modeling_borissal.py). Each function is one Tier-1
candidate from docs/borissal/v03-design.md; integration points live in
modeling_borissal.py. All ops are mobile-delegate-safe: conv/pool/elementwise/
topk/fixed matmul only -- no FFT, no general sort, no sequential scans.
"""

import math

import torch
import torch.nn.functional as F


def motion_center_surround(motion_p: torch.Tensor, kernel: int) -> torch.Tensor:
    """relu(D - avgpool_large(D)) on the pooled grid-res motion map.

    A uniform ego-motion (pan/zoom) diff field cancels against its own
    surround mean; an independently moving object survives as a local peak
    (Itti 1998 motion conspicuity; Mahadevan & Vasconcelos 2010, simplified).

    Args:
        kernel: pooling kernel size (must be odd for size-preserving avgpool).
    """
    assert kernel % 2 == 1, "motion_cs kernel must be odd (size-preserving avgpool)"
    b, t, h, w = (int(x) for x in motion_p.shape)
    surround = F.avg_pool2d(
        motion_p.reshape(b * t, 1, h, w), kernel_size=kernel, stride=1,
        padding=kernel // 2, count_include_pad=False,
    ).view(b, t, h, w)
    return F.relu(motion_p - surround)
