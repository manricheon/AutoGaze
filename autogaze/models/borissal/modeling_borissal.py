"""Borissal: a non-learned, feed-forward, single-scale saliency patch selector.

Combines motion (temporal tubelet differencing, a codec-residual proxy) and
spatial (gradient/edge) energy into a per-patch score, then keeps the top-k
patches per tubelet under a `gazing_ratio` budget. Output is grid_thw-native
(video-encoder agnostic: V-JEPA2 / Qwen-VL style), not the AutoGaze
`gazing_pos` dict contract -- see autogaze/models/borissal/adapters.py for an
optional bridge to that contract.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_borissal import BorissalConfig


@dataclass
class Selection:
    """grid_thw-native selector output. All tensors are on the input's device.

    Flatten order for `scores` / `keep_mask` is t-major: flat index
    `i = t * (H_grid * W_grid) + h * W_grid + w`.
    """

    grid_thw: torch.Tensor        # (B, 3) long -- (T_grid, H_grid, W_grid), same for every row in Phase 1
    scores: torch.Tensor          # (B, L) float -- L = T_grid * H_grid * W_grid
    keep_mask: torch.Tensor       # (B, L) bool
    keep_index: torch.Tensor      # (B, K) long -- flat indices of kept patches, -1 padded
    keep_coords: torch.Tensor     # (B, K, 3) long -- (t, h, w) per kept patch, -1 padded
    num_keep: torch.Tensor        # (B,) long -- valid (non-padded) count per instance
    per_frame_keep: torch.Tensor  # (B, T_grid) long -- kept count per tubelet


def _minmax_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per (batch, tubelet) min-max normalize the last two (spatial) dims to [0, 1]."""
    b, t, h, w = x.shape
    flat = x.reshape(b, t, -1)
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    normed = (flat - mn) / (mx - mn + eps)
    return normed.reshape(b, t, h, w)


def _largest_remainder(raw: torch.Tensor, total_budget: int, min_val: int, max_val: int) -> torch.Tensor:
    """Round per-row fractional allocations to integers summing to total_budget (Hamilton's method).

    raw: (B, T_grid) float, each row's values sum to ~total_budget.
    Returns (B, T_grid) long, clamped to [min_val, max_val].
    """
    floor = raw.floor()
    remainder = raw - floor
    base = floor.long()
    deficit = total_budget - base.sum(dim=-1)  # (B,)
    order = remainder.argsort(dim=-1, descending=True)
    rank = order.argsort(dim=-1)
    add = (rank < deficit.unsqueeze(-1)).long()
    counts = base + add
    return counts.clamp(min=min_val, max=max_val)


def _pack_gazing_mask(gazing_mask: torch.Tensor):
    """Pack a (B, N) boolean/0-1 mask into (kept_index, is_padded), both (B, K),
    K = max ones-count over the batch. Kept indices come first in each row, in
    their original ascending order (stable sort), padded with -1.

    Inlined from autogaze/utils.py::get_gazing_pos_from_gazing_mask (same
    logic, ported verbatim) so this module has no dependency outside `torch` --
    see docs/borissal/reference.md's "Standalone" section.
    """
    gazing_mask = gazing_mask.to(torch.long)
    B, N = gazing_mask.shape

    idx = torch.arange(N, device=gazing_mask.device).expand(B, N)
    key = (1 - gazing_mask) * N + idx
    order = key.argsort(dim=1, stable=True)
    sorted_idx = idx.gather(1, order)

    counts = gazing_mask.sum(dim=1)
    K = int(counts.max().item())
    if K == 0:
        empty = sorted_idx[:, :0]
        return empty, empty.to(torch.bool)

    topk = sorted_idx[:, :K]
    pos = torch.arange(K, device=gazing_mask.device).expand(B, K)
    mask = pos < counts.unsqueeze(1)
    kept_index = topk.masked_fill(~mask, -1)
    is_padded = kept_index == -1
    return kept_index, is_padded


class Borissal(nn.Module):
    """Signal-based feed-forward patch selector (Phase 1 of the AutoGaze patch-selector line).

    Non-learned: holds no trainable parameters. Kept as an nn.Module so a
    learned successor (Phase 2) can share the same call surface.
    """

    def __init__(self, config: BorissalConfig):
        super().__init__()
        self.config = config
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("_sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("_sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, video: torch.Tensor, **kwargs) -> Selection:
        return self.select(video, **kwargs)

    @torch.no_grad()
    def select(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float] = None,
        motion_weight: Optional[Union[float, str]] = None,
        per_frame_allocation: Optional[str] = None,
        tubelet_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Selection:
        """video: (B, T, C, H, W) float, already resized/normalized."""
        selection, _ = self._select_impl(
            video, gazing_ratio, motion_weight, per_frame_allocation, tubelet_size, patch_size,
            want_intermediates=False,
        )
        return selection

    @torch.no_grad()
    def select_with_intermediates(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float] = None,
        motion_weight: Optional[Union[float, str]] = None,
        per_frame_allocation: Optional[str] = None,
        tubelet_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ):
        """Same as `select`, but also returns the intermediate (pre-top-k) saliency
        maps -- useful for visualizing the motion/spatial/combined-score stages,
        e.g. in scripts/borissal_dump_outputs.py. Does not change `select`'s
        output or recompute anything differently.

        Returns (Selection, intermediates) where intermediates is a dict with:
            motion_norm       (B, T_grid, H_grid, W_grid) float -- normalized motion map
            spatial_norm      (B, T_grid, H_grid, W_grid) float -- normalized spatial map
            score             (B, T_grid, H_grid, W_grid) float -- combined score S (pre-top-k)
            motion_weight_used (B,) float -- the blend weight actually used per video
                (equal to the fixed config/kwarg value, broadcast, unless
                motion_weight="auto", in which case it's the per-video
                computed value)
        """
        return self._select_impl(
            video, gazing_ratio, motion_weight, per_frame_allocation, tubelet_size, patch_size,
            want_intermediates=True,
        )

    def _select_impl(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float],
        motion_weight: Optional[Union[float, str]],
        per_frame_allocation: Optional[str],
        tubelet_size: Optional[int],
        patch_size: Optional[int],
        want_intermediates: bool,
    ):
        cfg = self.config
        tubelet_size = tubelet_size or cfg.tubelet_size
        patch_size = patch_size or cfg.patch_size
        ratio = cfg.gazing_ratio if gazing_ratio is None else gazing_ratio
        motion_weight_setting = cfg.motion_weight if motion_weight is None else motion_weight
        alloc = per_frame_allocation or cfg.per_frame_allocation
        eps = cfg.eps

        # Explicit int() casts: under torch.jit.trace / torch.export, .shape components
        # can otherwise come back as trace-time symbolic wrappers that break plain
        # Python int arithmetic (e.g. round(), %) used below -- see the "Mobile
        # readiness review" section of docs/borissal/design.md. Harmless in eager mode.
        B, T, C, H, W = (int(x) for x in video.shape)
        if T % tubelet_size != 0:
            raise ValueError(f"num_frames ({T}) must be divisible by tubelet_size ({tubelet_size})")
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"H,W ({H},{W}) must be divisible by patch_size ({patch_size})")

        device = video.device
        gray = video.mean(dim=2)  # (B, T, H, W)

        T_grid = T // tubelet_size
        H_grid, W_grid = H // patch_size, W // patch_size
        N_pf = H_grid * W_grid
        L = T_grid * N_pf

        tub = gray.view(B, T_grid, tubelet_size, H, W).mean(dim=2)  # (B, T_grid, H, W)

        # Motion: forward/backward tubelet differencing (codec-residual proxy).
        if T_grid > 1:
            diff = (tub[:, 1:] - tub[:, :-1]).abs()  # (B, T_grid-1, H, W)
            motion = torch.zeros_like(tub)
            motion[:, 1:] = diff
            motion[:, 0] = diff[:, 0]
        else:
            motion = torch.zeros_like(tub)

        # Spatial: gradient / edge energy.
        if cfg.spatial_op == "grad":
            dy = F.pad(tub[:, :, 1:, :] - tub[:, :, :-1, :], (0, 0, 0, 1))
            dx = F.pad(tub[:, :, :, 1:] - tub[:, :, :, :-1], (0, 1, 0, 0))
        elif cfg.spatial_op == "sobel":
            flat = tub.reshape(B * T_grid, 1, H, W)
            dx = F.conv2d(flat, self._sobel_x, padding=1).view(B, T_grid, H, W)
            dy = F.conv2d(flat, self._sobel_y, padding=1).view(B, T_grid, H, W)
        else:
            raise ValueError(f"unknown spatial_op: {cfg.spatial_op}")
        spatial = torch.sqrt(dx * dx + dy * dy + eps)

        # Pixel -> patch pooling.
        pool = F.avg_pool2d if cfg.pooling == "avg" else F.max_pool2d
        motion_p = pool(motion.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size)
        motion_p = motion_p.view(B, T_grid, H_grid, W_grid)
        spatial_p = pool(spatial.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size)
        spatial_p = spatial_p.view(B, T_grid, H_grid, W_grid)

        if motion_weight_setting == "auto":
            # Content-adaptive blend: derived from the clip's own (pre-normalization,
            # so absolute-magnitude-sensitive) motion vs. spatial energy -- still
            # non-learned, just data-adaptive. A per-tubelet min-max normalized map
            # (motion_n/spatial_n below) can't be used for this since it erases
            # absolute magnitude by construction.
            motion_energy = motion_p.mean(dim=(1, 2, 3), keepdim=True)   # (B, 1, 1, 1)
            spatial_energy = spatial_p.mean(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
            w = motion_energy / (motion_energy + spatial_energy + eps)
        else:
            w = motion_weight_setting

        motion_n = _minmax_norm(motion_p, eps)
        spatial_n = _minmax_norm(spatial_p, eps)
        S = w * motion_n + (1 - w) * spatial_n  # (B, T_grid, H_grid, W_grid)

        # Per-tubelet budget allocation.
        if alloc == "uniform":
            k = min(max(1, round(ratio * N_pf)), N_pf)
            k_per_frame = torch.full((B, T_grid), k, dtype=torch.long, device=device)
        elif alloc == "proportional":
            total_budget = min(max(1, round(ratio * L)), L)
            energy = S.reshape(B, T_grid, -1).sum(dim=-1)
            energy_sum = energy.sum(dim=-1, keepdim=True).clamp_min(eps)
            raw = (energy / energy_sum) * total_budget
            k_per_frame = _largest_remainder(raw, total_budget, min_val=1, max_val=N_pf)
        else:
            raise ValueError(f"unknown per_frame_allocation: {alloc}")

        # Top-k -> keep mask, via torch.topk (bounded by k_max, not a full O(N) sort) --
        # more mobile-runtime-friendly (TFLite TopKV2 / CoreML top_k are first-class ops,
        # unlike general sort/argsort). For "uniform" allocation k_max == k for every
        # tubelet, so this reduces to a single topk(k) + scatter with no extra compare.
        scores_flat = S.reshape(B, T_grid, N_pf)
        k_max = int(k_per_frame.max().item())
        _, topk_idx = scores_flat.topk(k_max, dim=-1)  # (B, T_grid, k_max), sorted descending
        within_topk_rank = torch.arange(k_max, device=device).view(1, 1, k_max).expand(B, T_grid, k_max)
        keep_within_topk = within_topk_rank < k_per_frame.unsqueeze(-1)  # (B, T_grid, k_max) bool
        keep_mask_grid = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
        keep_mask_grid.scatter_(-1, topk_idx, keep_within_topk)  # (B, T_grid, N_pf) bool

        per_frame_keep = keep_mask_grid.sum(dim=-1)  # (B, T_grid)
        num_keep = per_frame_keep.sum(dim=-1)        # (B,)

        keep_mask = keep_mask_grid.reshape(B, L)
        keep_index, is_padded = _pack_gazing_mask(keep_mask)  # (B, K) each

        idx_safe = keep_index.clamp(min=0)
        t_coord = idx_safe // N_pf
        rem = idx_safe % N_pf
        h_coord = rem // W_grid
        w_coord = rem % W_grid
        coords = torch.stack([t_coord, h_coord, w_coord], dim=-1)
        coords = coords.masked_fill(is_padded.unsqueeze(-1), -1)

        grid_thw = torch.tensor([T_grid, H_grid, W_grid], dtype=torch.long, device=device)
        grid_thw = grid_thw.unsqueeze(0).expand(B, 3).clone()

        selection = Selection(
            grid_thw=grid_thw,
            scores=scores_flat.reshape(B, L),
            keep_mask=keep_mask,
            keep_index=keep_index,
            keep_coords=coords,
            num_keep=num_keep,
            per_frame_keep=per_frame_keep,
        )
        intermediates = None
        if want_intermediates:
            if isinstance(w, torch.Tensor):
                motion_weight_used = w.reshape(B)
            else:
                motion_weight_used = torch.full((B,), float(w), device=device)
            intermediates = {
                "motion_norm": motion_n,
                "spatial_norm": spatial_n,
                "score": S,
                "motion_weight_used": motion_weight_used,
            }
        return selection, intermediates
