"""Borissal v0: a non-learned, feed-forward, single-scale saliency patch selector.

Combines motion (temporal tubelet differencing, a codec-residual proxy) and
spatial (gradient/edge) energy into a per-patch score, then keeps the top-k
patches per tubelet under a `gazing_ratio` budget -- with that budget's
per-frame share either uniform or dynamically reallocated to each frame's
own saliency energy (`per_frame_allocation`). Output is grid_thw-native
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
from .signals_v03 import (
    apply_score_ema,
    coherence_gate_map,
    color_rarity,
    dog_blob,
    fused_blend,
    image_signature,
    motion_center_surround,
)


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


def _minmax_norm_global(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-batch clip-GLOBAL min-max over all (tubelet, h, w) jointly --
    preserves cross-tubelet magnitude, unlike the per-tubelet variant."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    normed = (flat - mn) / (mx - mn + eps)
    return normed.reshape_as(x)


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


def _bucket_grid(k_spread: int, t: int, h: int, w: int):
    """Smallest (G_t, G_h, G_w) bucket layout with G_t*G_h*G_w >= k_spread:
    stratify time first (event coverage), then space near-square."""
    G_t = min(t, k_spread)
    q = -(-k_spread // G_t)  # per-time-slice bucket quota (ceil)
    G_h = min(h, max(1, int(q ** 0.5)))
    G_w = min(w, -(-q // G_h))
    while G_t * G_h * G_w < k_spread:
        if G_w < w:
            G_w += 1
        elif G_h < h:
            G_h += 1
        elif G_t < t:
            G_t += 1
        else:
            break
    return G_t, G_h, G_w


def _hybrid_topk(scores: torch.Tensor, k: int, spread_fraction: float, grid: tuple) -> torch.Tensor:
    """Hybrid focus+spread selection over a (t, h, w) grid (the single-scale
    analogue of AutoGaze's multi-scale layout, whose coarse scales dedicate a
    fixed share of every frame's budget to a global gist).

    scores: (R, L) independent selection problems, L == t*h*w for grid.
    Returns a bool keep mask (R, L) with EXACTLY k True per row:
    - focus share (k - k_spread): plain global top-k by score;
    - spread share k_spread = round(spread_fraction * k): the grid is cut
      into >= k_spread spatio-temporal buckets (time stratified first, then
      space) and each of the best k_spread buckets contributes its
      highest-scoring cell -- guaranteed scene/timeline skeleton, with
      saliency still choosing WITHIN each bucket.
    Ops: topk/scatter/gather/scatter_reduce only (mobile-safe); exactness is
    enforced by a final boosted top-k (same bonus pattern as global-alloc's
    per-tubelet floor), which also backfills the rare empty-bucket case.
    """
    R, L = (int(x) for x in scores.shape)
    t, h, w = grid
    device = scores.device
    k_spread = min(k, int(round(spread_fraction * k)))
    k_focus = k - k_spread

    chosen = torch.zeros(R, L, dtype=torch.bool, device=device)
    if k_focus > 0:
        _, fidx = scores.topk(k_focus, dim=-1)
        chosen.scatter_(1, fidx, torch.ones_like(fidx, dtype=torch.bool))

    if k_spread > 0:
        G_t, G_h, G_w = _bucket_grid(k_spread, t, h, w)
        n_b = G_t * G_h * G_w
        tt = torch.arange(t, device=device).view(t, 1, 1)
        hh = torch.arange(h, device=device).view(1, h, 1)
        ww = torch.arange(w, device=device).view(1, 1, w)
        bucket = ((tt * G_t // t) * G_h + (hh * G_h // h)) * G_w + (ww * G_w // w)
        bucket = bucket.reshape(1, L).expand(R, L)

        neg = torch.finfo(scores.dtype).min
        avail = scores.masked_fill(chosen, neg)
        bmax = torch.full((R, n_b), neg, dtype=scores.dtype, device=device)
        bmax.scatter_reduce_(1, bucket, avail, reduce="amax")
        # representative cell per bucket = lowest index attaining the bucket max
        cand = (avail == bmax.gather(1, bucket)) & ~chosen & (avail > neg)
        idx_key = torch.where(cand, (L - torch.arange(L, device=device)).expand(R, L),
                              torch.zeros(1, dtype=torch.long, device=device))
        rep_key = torch.zeros(R, n_b, dtype=torch.long, device=device)
        rep_key.scatter_reduce_(1, bucket, idx_key, reduce="amax")
        has_rep = rep_key > 0
        rep_idx = (L - rep_key).clamp_(0, L - 1)
        rank = torch.where(has_rep, bmax, torch.full_like(bmax, neg))
        # PER-TIME-SLICE quotas (largest-remainder style): guarantees timeline
        # coverage even when scores tie (plain top-k over buckets breaks ties
        # toward low indices and can starve late time slices -- caught by test)
        S_b = G_h * G_w
        base, rem = divmod(k_spread, G_t)
        q_max = min(base + (1 if rem else 0), S_b)
        if q_max > 0:
            _, top_b3 = rank.reshape(R, G_t, S_b).topk(q_max, dim=-1)   # (R, G_t, q_max)
            quota = torch.tensor([min(base + (1 if i < rem else 0), S_b) for i in range(G_t)],
                                 device=device)
            within = torch.arange(q_max, device=device).view(1, 1, q_max) < quota.view(1, G_t, 1)
            flat_b = (top_b3 + torch.arange(G_t, device=device).view(1, G_t, 1) * S_b).reshape(R, -1)
            ok = has_rep.gather(1, flat_b) & within.expand(R, -1, -1).reshape(R, -1)
            # invalid entries scatter into a dummy column (a False src could
            # otherwise CLEAR a colliding True -- scatter overwrite hazard)
            sel_idx = torch.where(ok, rep_idx.gather(1, flat_b), torch.full_like(flat_b, L))
            chosen = torch.cat([chosen, torch.zeros(R, 1, dtype=torch.bool, device=device)], dim=1)
            chosen.scatter_(1, sel_idx, torch.ones_like(sel_idx, dtype=torch.bool))
            chosen = chosen[:, :L]

    # exact-k: chosen cells get a large finite bonus, then one top-k --
    # keeps every chosen cell (they number <= k) and backfills by score.
    boosted = scores + chosen.to(scores.dtype) * 1e4
    _, kidx = boosted.topk(k, dim=-1)
    mask = torch.zeros_like(chosen)
    mask.scatter_(1, kidx, torch.ones_like(kidx, dtype=torch.bool))
    return mask


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
    # keys are UNIQUE (kept -> idx, dropped -> N+idx), so a plain argsort is
    # already deterministic; stable=True would lower to aten::sort.out, which
    # the ONNX exporter cannot represent (found by export_borissal_check.py)
    order = key.argsort(dim=1)
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
        spread_fraction: Optional[float] = None,
        temporal_state: Optional[dict] = None,
        per_frame_counts: Optional[torch.Tensor] = None,
    ) -> Selection:
        """video: (B, T, C, H, W) float, already resized/normalized."""
        selection, _ = self._select_impl(
            video, gazing_ratio, motion_weight, per_frame_allocation, tubelet_size, patch_size,
            want_intermediates=False, spread_fraction=spread_fraction, temporal_state=temporal_state,
            per_frame_counts=per_frame_counts,
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
        spread_fraction: Optional[float] = None,
        temporal_state: Optional[dict] = None,
        per_frame_counts: Optional[torch.Tensor] = None,
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
            temporal_state     dict with "ema" (B, H_grid, W_grid) and
                "prev_keep" (B, N_pf) bool -- feed back into the next clip's
                `select(..., temporal_state=...)` call for cross-clip continuity
        """
        return self._select_impl(
            video, gazing_ratio, motion_weight, per_frame_allocation, tubelet_size, patch_size,
            want_intermediates=True, spread_fraction=spread_fraction, temporal_state=temporal_state,
            per_frame_counts=per_frame_counts,
        )

    def _extra_channels(self, video: torch.Tensor, tub: torch.Tensor,
                        tubelet_size: int, patch_size: int):
        """v0.3 appearance channels at grid resolution (spec section 3).

        Returns (local, global_) lists of (weight, normalized_map); both empty
        when every channel knob is off (the caller then takes the legacy blend
        path). color_rarity is heavy-tailed and uses its clip-global
        normalization in BOTH lists (spec ordering rules).
        """
        cfg = self.config
        eps = cfg.eps
        local, global_ = [], []
        if cfg.signature_weight <= 0.0 and cfg.dog_blob_weight <= 0.0 \
                and cfg.color_rarity_weight <= 0.0:
            return local, global_
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid = T // tubelet_size
        H_grid, W_grid = H // patch_size, W // patch_size
        if cfg.signature_weight > 0.0 or cfg.dog_blob_weight > 0.0:
            gray_grid = F.avg_pool2d(
                tub.reshape(B * T_grid, 1, H, W),
                kernel_size=patch_size, stride=patch_size,
            ).view(B, T_grid, H_grid, W_grid)
        if cfg.signature_weight > 0.0:
            sig = image_signature(gray_grid)
            local.append((cfg.signature_weight, _minmax_norm(sig, eps)))
            global_.append((cfg.signature_weight, _minmax_norm_global(sig, eps)))
        if cfg.dog_blob_weight > 0.0:
            blob = dog_blob(gray_grid)
            local.append((cfg.dog_blob_weight, _minmax_norm(blob, eps)))
            global_.append((cfg.dog_blob_weight, _minmax_norm_global(blob, eps)))
        if cfg.color_rarity_weight > 0.0:
            rgb_tub = video.view(B, T_grid, tubelet_size, C, H, W).mean(dim=2)
            rgb_grid = F.avg_pool2d(
                rgb_tub.reshape(B * T_grid, C, H, W),
                kernel_size=patch_size, stride=patch_size,
            ).view(B, T_grid, C, H_grid, W_grid)
            rar = color_rarity(rgb_grid, cfg.color_bins_per_axis,
                               cfg.color_bin_sigma, eps)
            rar_n = _minmax_norm_global(rar, eps)
            local.append((cfg.color_rarity_weight, rar_n))
            global_.append((cfg.color_rarity_weight, rar_n))
        return local, global_

    def _saliency_scores(
        self,
        video: torch.Tensor,
        tubelet_size: int,
        patch_size: int,
        motion_weight_setting,
    ) -> dict:
        """The full saliency pipeline (luma -> tubelet -> motion/spatial ->
        pool -> noise floor -> auto-weight -> normalize -> blend), shared by
        the fine pass and the v0.2 coarse (resized-input) pass.

        Returns dict(score, motion_norm, spatial_norm, w, noise_floor_tau) --
        maps are (B, T_grid, H_grid, W_grid) for THIS video's resolution.
        """
        cfg = self.config
        eps = cfg.eps
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid = T // tubelet_size
        H_grid, W_grid = H // patch_size, W // patch_size

        gray = video.mean(dim=2)  # (B, T, H, W)
        tub = gray.view(B, T_grid, tubelet_size, H, W).mean(dim=2)  # (B, T_grid, H, W)

        # Motion (codec-residual proxy). Two granularities:
        #  - "tubelet" (v0.1): difference the tubelet means.
        #  - "frame" (v0.2): difference consecutive frames, then aggregate per
        #    tubelet -- catches fast intra-tubelet motion that tubelet
        #    averaging cancels. Both are pure slice ops (fully vectorized).
        if cfg.motion_diff == "frame" and T > 1 and tubelet_size > 1:
            # Frame-rate-aware diff stride (v0.4): consecutive-frame diff
            # (stride 1) UNDER-registers motion when frames are sampled densely
            # -- at 2x the frame rate, adjacent frames are ~2x more similar, so
            # |f_t - f_{t-1}| shrinks (measured: 0.176 -> 0.115 going 16f->32f).
            # "auto" scales the stride with frame count so the effective temporal
            # gap (and thus the motion magnitude) stays constant regardless of
            # how many frames the clip was decoded to; at the 16-frame reference
            # this is stride 1, so 16f behavior is unchanged (bit-identical).
            if cfg.motion_diff_stride == "auto":
                stride = max(1, round(T / cfg.motion_ref_frames))
            else:
                stride = max(1, int(cfg.motion_diff_stride))
            stride = min(stride, T - 1)
            fdiff = (gray[:, stride:] - gray[:, :-stride]).abs()  # (B, T-stride, H, W)
            pad = fdiff[:, :1].expand(B, stride, H, W)            # first `stride` frames <- forward diff
            fdiff = torch.cat([pad, fdiff], dim=1)                # (B, T, H, W)
            if cfg.motion_consistency == "double_diff" and T > 2:
                # Temporal AND (double-difference): real motion persists across
                # consecutive diffs; single-frame spikes (flicker/compression
                # artifacts) vanish under the min. One-sided at the boundary.
                nxt = torch.cat([fdiff[:, 1:], fdiff[:, -1:]], dim=1)
                fdiff = torch.minimum(fdiff, nxt)
            grouped = fdiff.view(B, T_grid, tubelet_size, H, W)
            motion = grouped.mean(dim=2) if cfg.frame_diff_agg == "mean" else grouped.amax(dim=2)
        elif T_grid > 1:
            diff = (tub[:, 1:] - tub[:, :-1]).abs()  # (B, T_grid-1, H, W)
            motion = torch.zeros_like(tub)
            motion[:, 1:] = diff
            motion[:, 0] = diff[:, 0]
        else:
            motion = torch.zeros_like(tub)

        # Optional pixel-level box blur of the motion map (v0.2 knob, default off).
        if cfg.motion_smooth_kernel >= 3:
            ksz = cfg.motion_smooth_kernel
            motion = F.avg_pool2d(
                motion.reshape(B * T_grid, 1, H, W), kernel_size=ksz, stride=1, padding=ksz // 2
            ).view(B, T_grid, H, W)

        # Spatial: gradient / edge energy. Source granularity (v0.3.x knob,
        # mirrors motion_diff): "tubelet" (default) measures the 2-frame MEAN
        # -- a slight motion blur that smears fast movers' edges and halves
        # detail present in only one frame; "frame" measures each raw frame
        # and aggregates per tubelet afterwards ("max" keeps anything sharp
        # in at least one frame). Selection stays tubelet-granular either way
        # (downstream token-grid contract) -- only the SIGNAL granularity
        # changes.
        spatial_src = tub if cfg.spatial_diff == "tubelet" else gray
        Ts = int(spatial_src.shape[1])
        if cfg.spatial_op == "grad":
            dy = F.pad(spatial_src[:, :, 1:, :] - spatial_src[:, :, :-1, :], (0, 0, 0, 1))
            dx = F.pad(spatial_src[:, :, :, 1:] - spatial_src[:, :, :, :-1], (0, 1, 0, 0))
        elif cfg.spatial_op == "sobel":
            flat = spatial_src.reshape(B * Ts, 1, H, W)
            dx = F.conv2d(flat, self._sobel_x, padding=1).view(B, Ts, H, W)
            dy = F.conv2d(flat, self._sobel_y, padding=1).view(B, Ts, H, W)
        else:
            raise ValueError(f"unknown spatial_op: {cfg.spatial_op}")
        spatial = torch.sqrt(dx * dx + dy * dy + eps)
        if cfg.spatial_diff == "frame":
            grouped = spatial.view(B, T_grid, tubelet_size, H, W)
            spatial = grouped.mean(dim=2) if cfg.spatial_agg == "mean" else grouped.amax(dim=2)
            if cfg.coherence_gate:
                # TUNE (2026-07-19 sweep): the gate stays TUBELET-granular
                # even for frame-granular spatial -- coherence is a smoothed
                # regional texture statistic, so recomputing it per frame
                # (~2x coherence cost, ~+6ms) buys nothing; tubelet-mean
                # gradients suffice and keep the frame signal affordable.
                dyg = F.pad(tub[:, :, 1:, :] - tub[:, :, :-1, :], (0, 0, 0, 1))
                dxg = F.pad(tub[:, :, :, 1:] - tub[:, :, :, :-1], (0, 1, 0, 0))
                spatial = spatial * coherence_gate_map(
                    dxg, dyg, cfg.coherence_kernel, cfg.coherence_gamma, eps,
                    downsample=cfg.coherence_downsample)
        elif cfg.coherence_gate:
            spatial = spatial * coherence_gate_map(
                dx, dy, cfg.coherence_kernel, cfg.coherence_gamma, eps,
                downsample=cfg.coherence_downsample)

        # Pixel -> patch pooling.
        pool = F.avg_pool2d if cfg.pooling == "avg" else F.max_pool2d
        motion_p = pool(motion.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size)
        motion_p = motion_p.view(B, T_grid, H_grid, W_grid)
        spatial_p = pool(spatial.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size)
        spatial_p = spatial_p.view(B, T_grid, H_grid, W_grid)

        # v0.3 motion center-surround: BEFORE the noise floor (spec section 3
        # ordering -- center-surround first, then dead-zone shrinkage).
        if cfg.motion_center_surround:
            motion_p = motion_center_surround(motion_p, cfg.motion_cs_kernel)

        # v0.2 noise floor: robust per-tubelet dead-zone shrinkage of the motion
        # map. MUST run before min-max normalization (which would otherwise
        # re-amplify the noise floor to full [0,1] range in low-motion tubelets)
        # and before the "auto" weight energies (so w is noise-corrected).
        # Analogous to codec dead-zone quantization / soft-threshold coring;
        # true motion is spatially sparse, so a median/mean over the tubelet is
        # dominated by non-moving patches and estimates the noise floor.
        tau = None
        if cfg.motion_noise_floor != "none":
            N_pf = H_grid * W_grid
            flat = motion_p.reshape(B, T_grid, N_pf)
            if cfg.motion_noise_floor == "mean":
                tau = flat.mean(dim=-1, keepdim=True)  # (B, T_grid, 1)
            elif cfg.motion_noise_floor == "quantile":
                # q-quantile via topk (mobile-safe: no sort/quantile/kthvalue).
                # k_q is config/shape-derived -> data-independent, trace-safe.
                k_q = min(max(1, int(round((1.0 - cfg.motion_noise_q) * N_pf))), N_pf)
                tau = flat.topk(k_q, dim=-1).values[..., -1:]  # (B, T_grid, 1)
            else:
                raise ValueError(f"unknown motion_noise_floor: {cfg.motion_noise_floor}")
            motion_p = F.relu(motion_p - cfg.motion_noise_scale * tau.reshape(B, T_grid, 1, 1))
            tau = tau.reshape(B, T_grid)

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
        extra_local, extra_global = self._extra_channels(video, tub, tubelet_size, patch_size)
        if not extra_local and cfg.fusion_norm == "none":
            # legacy exact path: bit-identical to v0.2 (the generalized
            # weighted average divides by the weight sum, whose float error
            # would break the all-knobs-off regression guarantee)
            S = w * motion_n + (1 - w) * spatial_n
        else:
            S = fused_blend([(w, motion_n), (1 - w, spatial_n)] + extra_local,
                            cfg.fusion_norm, cfg.fusion_entropy_floor, eps)  # (B, T_grid, H_grid, W_grid)

        # v0.2 local/global normalization blend: per-tubelet min-max (local)
        # equalizes tubelets; a clip-global min-max component preserves
        # cross-tubelet magnitude so high-energy moments can rank higher
        # clip-wide (required for "global" allocation to be meaningful).
        if cfg.score_norm_blend < 1.0:
            motion_g = _minmax_norm_global(motion_p, eps)
            spatial_g = _minmax_norm_global(spatial_p, eps)
            if not extra_global and cfg.fusion_norm == "none":
                S_global = w * motion_g + (1 - w) * spatial_g
            else:
                S_global = fused_blend(
                    [(w, motion_g), (1 - w, spatial_g)] + extra_global,
                    cfg.fusion_norm, cfg.fusion_entropy_floor, eps)
            beta = cfg.score_norm_blend
            S = beta * S + (1.0 - beta) * S_global

        # v0.2 center bias (conditional, off by default): additive Gaussian
        # center prior -- the classical composition prior from saliency
        # benchmarks. Grid-resolution constant map; negligible cost.
        if cfg.center_bias > 0.0:
            hh = torch.linspace(-1.0, 1.0, H_grid, device=S.device)
            ww = torch.linspace(-1.0, 1.0, W_grid, device=S.device)
            g = torch.exp(-(hh.view(-1, 1) ** 2 + ww.view(1, -1) ** 2) / (2 * 0.45 ** 2))
            S = S + cfg.center_bias * g.view(1, 1, H_grid, W_grid)

        return {
            "score": S,
            "motion_norm": motion_n,
            "spatial_norm": spatial_n,
            "w": w,
            "noise_floor_tau": tau,
        }

    def _allocate_and_topk(self, sel_scores, alloc, k_per_frame, m, K_total,
                           spread, B, T_grid, N_pf, H_grid, W_grid, device):
        """Selection stage: (B, T_grid, N_pf) gated scores -> same-shape bool
        keep mask. Extracted verbatim from _select_impl so the hysteresis
        two-pass (v0.3) can run it twice on identically-shaped inputs."""
        L = T_grid * N_pf
        if alloc == "global":
            neg = torch.finfo(sel_scores.dtype).min
            # Optional per-tubelet CAP (v0.3.x allocation knob): each tubelet
            # exposes at most cap = mult x its uniform share to the global
            # top-k, so no single moment can monopolize the free budget.
            # cap >= ceil(K_total / T_grid) keeps the exact budget feasible.
            cap_mult = self.config.max_keep_per_frame_mult
            if cap_mult > 0:
                share = (K_total + T_grid - 1) // T_grid
                cap = min(N_pf, max(share, m, int(round(cap_mult * K_total / T_grid))))
                _, cidx = sel_scores.topk(cap, dim=-1)
                exposed = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
                exposed.scatter_(-1, cidx, torch.ones_like(cidx, dtype=torch.bool))
                sel_scores = sel_scores.masked_fill(~exposed, neg)
            _, gidx = sel_scores.topk(m, dim=-1)
            guaranteed = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
            guaranteed.scatter_(-1, gidx, torch.ones_like(gidx, dtype=torch.bool))
            if spread > 0:
                # hybrid path keeps the single-score-vector bonus form (the
                # stratified bucket helper needs one comparable vector)
                global_scores = (sel_scores + 10.0 * guaranteed.to(sel_scores.dtype)).reshape(B, L)
                keep_mask_flat = _hybrid_topk(global_scores, K_total, spread,
                                              (T_grid, H_grid, W_grid))
                return keep_mask_flat.reshape(B, T_grid, N_pf)
            # Explicit two-step allocation (equivalent to the former
            # +10.0-bonus single top-k, refactored for clarity): (1) every
            # tubelet's guaranteed top-m is kept outright; (2) the remaining
            # free budget K_total - T_grid*m goes to a clip-wide top-k over
            # everything not already guaranteed.
            keep = guaranteed.reshape(B, L).clone()
            k_free = K_total - T_grid * m  # config/shape-derived, trace-safe
            if k_free > 0:
                free = sel_scores.masked_fill(guaranteed, neg).reshape(B, L)
                _, fidx = free.topk(k_free, dim=-1)
                keep.scatter_(1, fidx, torch.ones_like(fidx, dtype=torch.bool))
            return keep.reshape(B, T_grid, N_pf)
        if spread > 0 and alloc == "uniform":
            k_u = int(k_per_frame[0, 0].item())
            return _hybrid_topk(
                sel_scores.reshape(B * T_grid, N_pf), k_u, spread, (1, H_grid, W_grid)
            ).reshape(B, T_grid, N_pf)
        k_max = int(k_per_frame.max().item())
        _, topk_idx = sel_scores.topk(k_max, dim=-1)
        within_topk_rank = torch.arange(k_max, device=device).view(1, 1, k_max).expand(B, T_grid, k_max)
        keep_within_topk = within_topk_rank < k_per_frame.unsqueeze(-1)
        keep_mask_grid = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
        keep_mask_grid.scatter_(-1, topk_idx, keep_within_topk)
        return keep_mask_grid

    def _select_impl(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float],
        motion_weight: Optional[Union[float, str]],
        per_frame_allocation: Optional[str],
        tubelet_size: Optional[int],
        patch_size: Optional[int],
        want_intermediates: bool,
        spread_fraction: Optional[float] = None,
        temporal_state: Optional[dict] = None,
        per_frame_counts: Optional[torch.Tensor] = None,
    ):
        cfg = self.config
        tubelet_size = tubelet_size or cfg.tubelet_size
        patch_size = patch_size or cfg.patch_size
        ratio = cfg.gazing_ratio if gazing_ratio is None else gazing_ratio
        motion_weight_setting = cfg.motion_weight if motion_weight is None else motion_weight
        alloc = per_frame_allocation or cfg.per_frame_allocation
        spread = cfg.spread_fraction if spread_fraction is None else spread_fraction
        if spread > 0 and alloc == "proportional":
            raise ValueError("spread_fraction requires uniform or global allocation")
        if per_frame_counts is not None and spread > 0:
            raise ValueError("per_frame_counts override is incompatible with spread_fraction")
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
        T_grid = T // tubelet_size
        H_grid, W_grid = H // patch_size, W // patch_size
        N_pf = H_grid * W_grid
        L = T_grid * N_pf

        sal = self._saliency_scores(video, tubelet_size, patch_size, motion_weight_setting)
        S, motion_n, spatial_n, w = sal["score"], sal["motion_norm"], sal["spatial_norm"], sal["w"]

        if cfg.score_ema_alpha > 0.0:
            # v0.3 temporal EMA. Applied AFTER _saliency_scores (which already
            # added center_bias): the decay matrix rows sum to 1, so a
            # time-constant additive prior commutes -- this equals the spec's
            # fusion -> EMA -> center_bias order exactly.
            # By design, EMA only smooths this fine-ranking score S: the
            # coarse block-gate pass below recomputes its own saliency on
            # video_small (unsmoothed) and is unaffected by this EMA.
            ema_state = None if temporal_state is None else temporal_state.get("ema")
            S = apply_score_ema(S, cfg.score_ema_alpha, ema_state)

        # v0.5 coarse-cube coherence (inspired by saliency-v3.1): pool the
        # selection score to a 1/c grid and repeat_interleave back, so every
        # c x c block shares one score. Uniform top-k then keeps whole c x c
        # CUBES (coherent object chunks) instead of scattered fine patches --
        # the same "prefer meaningful blobs over isolated micro-patches" goal
        # as the block gate, but enforced harder (full cubes) and computed on
        # the score itself. Token grid stays H_grid x W_grid (downstream
        # positions unchanged). Pair with block_size=1 (the two coherence
        # mechanisms are redundant). c=1 (default) is a no-op / bit-identical.
        if cfg.score_coarsen > 1:
            c = cfg.score_coarsen
            if H_grid % c != 0 or W_grid % c != 0:
                raise ValueError(
                    f"score_coarsen={c} requires grid {H_grid}x{W_grid} divisible by c")
            Sc = F.avg_pool2d(S.reshape(B * T_grid, 1, H_grid, W_grid), c, c)
            S = (Sc.repeat_interleave(c, dim=2).repeat_interleave(c, dim=3)
                 .reshape(B, T_grid, H_grid, W_grid))

        # Per-tubelet budget allocation.
        m = None
        K_total = None
        if per_frame_counts is not None:
            # E5 runtime override: the caller supplies the per-tubelet token
            # counts directly (e.g. a learned/oracle temporal-allocation head).
            # Only the ALLOCATION step is replaced -- patch ranking, block
            # gate, packing and the Selection contract are untouched. Counts
            # are data-dependent by nature: the trace/export caveat of the
            # "proportional" mode (mobile review) applies identically.
            k_per_frame = per_frame_counts.to(device=device, dtype=torch.long)
            if k_per_frame.dim() == 1:
                k_per_frame = k_per_frame.unsqueeze(0).expand(B, T_grid)
            if k_per_frame.shape != (B, T_grid):
                raise ValueError(
                    f"per_frame_counts must be (T_grid,) or (B, T_grid); got {tuple(per_frame_counts.shape)}")
            k_per_frame = k_per_frame.clamp(1, N_pf)
            alloc = "counts"  # falls through to the generic variable-k top-k branch
        elif alloc == "uniform":
            k = min(max(1, round(ratio * N_pf)), N_pf)
            k_per_frame = torch.full((B, T_grid), k, dtype=torch.long, device=device)
        elif alloc == "proportional":
            total_budget = min(max(1, round(ratio * L)), L)
            energy = S.reshape(B, T_grid, -1).sum(dim=-1)
            energy_sum = energy.sum(dim=-1, keepdim=True).clamp_min(eps)
            raw = (energy / energy_sum) * total_budget
            k_per_frame = _largest_remainder(raw, total_budget, min_val=1, max_val=N_pf)
        elif alloc == "global":
            # v0.2: one clip-wide top-K_total with a guaranteed per-tubelet
            # minimum m. The budget concentrates where the action is; the
            # floor preserves temporal coverage (no tubelet ends up empty).
            K_total = min(max(T_grid, round(ratio * L)), L)
            m = max(1, int(round(cfg.min_keep_per_frame_ratio * K_total / T_grid)))
            m = min(m, K_total // T_grid)
            # Coarse-to-fine gate sizing under global allocation: worst-case
            # capacity (K_total - (T-1)m) opens the gate almost fully and
            # neuters coherence (found empirically). Instead allow up to 2x
            # the uniform share per tubelet -- total gated capacity is then
            # 2*K_total >= K_total, so the global topk always has enough
            # finite candidates (exact budget preserved); concentration
            # beyond 2x share spills to other tubelets' gated regions.
            k_gate = min(N_pf, 2 * ((K_total + T_grid - 1) // T_grid))
            k_per_frame = torch.full((B, T_grid), k_gate, dtype=torch.long, device=device)
        else:
            raise ValueError(f"unknown per_frame_allocation: {alloc}")

        scores_flat = S.reshape(B, T_grid, N_pf)

        # v0.2 coherent-region selection (resize-based coarse-to-fine): a
        # low-resolution saliency pass picks top-ceil(k/b^2) blocks per tubelet,
        # then the fine top-k below runs only inside those blocks. Gate capacity
        # ceil(k/b^2)*b^2 >= k guarantees the exact-k budget is unaffected.
        # Fragmentation is hard-bounded to <= ceil(k/b^2) regions per tubelet.
        coarse_score = None
        block_mask = None
        sel_scores = scores_flat
        b = cfg.block_size
        if b > 1:
            if H_grid % b != 0 or W_grid % b != 0 or (H // b) % patch_size != 0 or (W // b) % patch_size != 0:
                raise ValueError(
                    f"block_size={b} incompatible with grid {H_grid}x{W_grid} "
                    f"(needs H_grid,W_grid divisible by b and H/b,W/b divisible by patch_size)"
                )
            Hc, Wc = H_grid // b, W_grid // b
            Nc = Hc * Wc
            A = b * b
            if cfg.block_gate_source == "pool":
                # v0.3.x candidate: derive the coarse signal by block-pooling
                # the FINE per-patch scores instead of recomputing the whole
                # pipeline on a resized clip -- one pipeline pass instead of
                # two. The recompute path's anti-noise role (resize low-pass)
                # is largely covered by the coherence gate since v0.3.
                coarse_score = F.avg_pool2d(
                    scores_flat.reshape(B * T_grid, 1, H_grid, W_grid),
                    kernel_size=b, stride=b,
                ).view(B, T_grid, Hc, Wc)
            else:
                # Coarse signal: the SAME saliency pipeline on a 1/b-resized
                # clip -- the resize's low-pass naturally kills fine noise
                # motion/texture. The resize runs in LUMA space when no color
                # channel is active (mean over channels and bilinear resize
                # commute; 1 channel instead of C to interpolate, and the
                # coarse pass's own luma step becomes a no-op).
                if cfg.color_rarity_weight > 0.0:
                    src, Cs = video, C
                else:
                    src, Cs = video.mean(dim=2, keepdim=True), 1
                video_small = F.interpolate(
                    src.reshape(B * T, Cs, H, W), scale_factor=1.0 / b,
                    mode="bilinear", align_corners=False,
                ).view(B, T, Cs, H // b, W // b)
                coarse_score = self._saliency_scores(
                    video_small, tubelet_size, patch_size, motion_weight_setting
                )["score"]  # (B, T_grid, Hc, Wc)

            kb_pf = ((k_per_frame + A - 1) // A).clamp(max=Nc)  # ceil-div, gate fully open if cap >= capacity
            kb_max = int(kb_pf.max().item())
            coarse_flat = coarse_score.reshape(B, T_grid, Nc)
            _, blk_idx = coarse_flat.topk(kb_max, dim=-1)
            blk_rank = torch.arange(kb_max, device=device).view(1, 1, kb_max).expand(B, T_grid, kb_max)
            blk_keep = blk_rank < kb_pf.unsqueeze(-1)
            block_mask = torch.zeros(B, T_grid, Nc, dtype=torch.bool, device=device)
            block_mask.scatter_(-1, blk_idx, blk_keep)  # (B, T_grid, Nc)

            fine_gate = (
                block_mask.reshape(B, T_grid, Hc, 1, Wc, 1)
                .expand(B, T_grid, Hc, b, Wc, b)
                .reshape(B, T_grid, N_pf)
            )
            # finfo.min (not -inf) for mobile-backend safety.
            sel_scores = scores_flat.masked_fill(~fine_gate, torch.finfo(scores_flat.dtype).min)

        eps_h = cfg.select_hysteresis_eps
        if eps_h > 0.0:
            # v0.3 selection hysteresis -- ONE-STEP VECTORIZED approximation:
            # the continuity bonus comes from the previous tubelet's
            # PRE-hysteresis selection (the true recursive chain would need a
            # sequential loop over tubelets, banned by the vectorization rule).
            base_keep = self._allocate_and_topk(
                sel_scores, alloc, k_per_frame, m, K_total, spread,
                B, T_grid, N_pf, H_grid, W_grid, device)
            prev0 = None if temporal_state is None else temporal_state.get("prev_keep")
            if prev0 is None:
                prev0 = torch.zeros(B, 1, N_pf, dtype=torch.bool, device=device)
            else:
                prev0 = prev0.reshape(B, 1, N_pf)
            prev = torch.cat([prev0, base_keep[:, :-1]], dim=1)
            sel_scores = sel_scores + eps_h * prev.to(sel_scores.dtype)
        keep_mask_grid = self._allocate_and_topk(
            sel_scores, alloc, k_per_frame, m, K_total, spread,
            B, T_grid, N_pf, H_grid, W_grid, device)

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
            intermediates["temporal_state"] = {
                "ema": S[:, -1],
                # Deliberate one-step-approximation asymmetry: this exports the
                # POST-hysteresis final mask (keep_mask_grid, computed after the
                # eps_h bonus above) for cross-clip carry, whereas the in-clip
                # continuity bonus at each tubelet is added w.r.t. the PREVIOUS
                # tubelet's PRE-hysteresis base_keep (see base_keep above).
                "prev_keep": keep_mask_grid[:, -1],
            }
            if sal["noise_floor_tau"] is not None:
                intermediates["noise_floor_tau"] = sal["noise_floor_tau"]
            if coarse_score is not None:
                intermediates["coarse_score"] = coarse_score
                intermediates["block_mask"] = block_mask
        return selection, intermediates
