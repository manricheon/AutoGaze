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


def coherence_gate_map(dx: torch.Tensor, dy: torch.Tensor, kernel: int,
                       gamma: float, eps: float) -> torch.Tensor:
    """(1 - coherence)^gamma texture-suppression gate from the structure tensor.

    Closed form, no eigendecomposition: for the smoothed tensor [a b; b c],
    coherence = ((lam1-lam2)/(lam1+lam2))^2 = ((a-c)^2 + 4b^2) / (a+c)^2.
    Repetitive gratings / long straight edges (lam1 >> lam2) -> gate ~0;
    multi-orientation object micro-structure (lam1 ~ lam2) -> gate ~1
    (Harris 1988; Forstner 1987; Weickert 1999). Box smoothing stands in for
    the classical Gaussian window (cheaper; delegate-native).
    """
    b, t, h, w = (int(x) for x in dx.shape)

    def _smooth(x):
        return F.avg_pool2d(
            x.reshape(b * t, 1, h, w), kernel_size=kernel, stride=1,
            padding=kernel // 2, count_include_pad=False,
        ).view(b, t, h, w)

    a = _smooth(dx * dx)
    c = _smooth(dy * dy)
    bb = _smooth(dx * dy)
    coherence = ((a - c) ** 2 + 4.0 * bb * bb) / ((a + c) ** 2 + eps)
    return (1.0 - coherence).clamp(min=0.0, max=1.0) ** gamma


def dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II basis as a constant (n, n) matrix -- the FFT-free,
    delegate-native (matmul) route to the DCT. Tiny at grid resolution."""
    i = torch.arange(n, device=device, dtype=dtype)
    basis = torch.cos(math.pi * (2.0 * i.view(1, -1) + 1.0) * i.view(-1, 1) / (2.0 * n))
    basis[0] = basis[0] / math.sqrt(2.0)
    return basis * math.sqrt(2.0 / n)


def image_signature(gray_grid: torch.Tensor) -> torch.Tensor:
    """Image-signature saliency (Hou, Harel & Koch, TPAMI 2012) at grid res.

    Reconstruct from only the SIGN of the DCT: energy concentrates on
    spatially sparse foreground, spectrally sparse (periodic-texture)
    background dies. Fires on object support, not just boundaries.
    """
    b, t, h, w = (int(x) for x in gray_grid.shape)
    Dh = dct_matrix(h, gray_grid.device, gray_grid.dtype)
    Dw = dct_matrix(w, gray_grid.device, gray_grid.dtype)
    coef = Dh @ gray_grid @ Dw.t()               # batched: (B, T, h, w)
    recon = Dh.t() @ torch.sign(coef) @ Dw       # inverse DCT of the sign
    sal = recon * recon
    return F.avg_pool2d(
        sal.reshape(b * t, 1, h, w), kernel_size=3, stride=1, padding=1
    ).view(b, t, h, w)
