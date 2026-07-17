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


def color_rarity(rgb_grid: torch.Tensor, num_bins_per_axis: int, sigma: float,
                 eps: float) -> torch.Tensor:
    """Global color rarity (histogram-contrast, HC variant of Cheng et al.
    CVPR 2011 -- no segmentation) over grid-resolution patch mean colors.

    Colors are min-max normalized per clip/channel (input video is
    ImageNet-normalized, so the fixed [0,1] bin lattice needs this),
    soft-binned onto a fixed n^3 RGB lattice, and each patch's saliency is
    its histogram-mass-weighted distance to all bins: rare colors far from
    the color mass score high -- across the OBJECT INTERIOR, not just its
    silhouette. Heavy-tailed by nature: sqrt-compressed here, and the
    caller must normalize it CLIP-GLOBALLY (spec section 3 ordering rules).
    """
    b, t, c, h, w = (int(x) for x in rgb_grid.shape)
    n = num_bins_per_axis
    axis = torch.linspace(0.0, 1.0, n, device=rgb_grid.device, dtype=rgb_grid.dtype)
    centers = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)                                                    # (K, 3)
    pix = rgb_grid.permute(0, 1, 3, 4, 2).reshape(b, t * h * w, 3)      # (B, P, 3)
    mn = pix.amin(dim=1, keepdim=True)
    mx = pix.amax(dim=1, keepdim=True)
    pix = (pix - mn) / (mx - mn + eps)                                  # per-clip [0,1]
    d2 = (pix.unsqueeze(2) - centers.view(1, 1, -1, 3)).pow(2).sum(-1)  # (B, P, K)
    assign = torch.softmax(-d2 / (2.0 * sigma * sigma), dim=-1)
    hist = assign.mean(dim=1)                                           # (B, K) mass
    sal = (d2.sqrt() * hist.unsqueeze(1)).sum(-1)                       # (B, P)
    return sal.sqrt().reshape(b, t, h, w)


def dog_blob(gray_grid: torch.Tensor) -> torch.Tensor:
    """Multi-scale difference-of-boxes blob channel (Lindeberg 1998 substrate)
    at grid resolution: the cheapest interior-filling mechanism. Scale pairs
    (3,7)/(5,11) grid cells bracket typical object sizes on a 24x24 grid."""
    b, t, h, w = (int(x) for x in gray_grid.shape)
    flat = gray_grid.reshape(b * t, 1, h, w)

    def _blur(k):
        return F.avg_pool2d(flat, kernel_size=k, stride=1, padding=k // 2,
                            count_include_pad=False)

    maps = [(_blur(k1) - _blur(k2)).abs() for k1, k2 in ((3, 7), (5, 11))]
    return torch.stack(maps, dim=0).amax(dim=0).view(b, t, h, w)
