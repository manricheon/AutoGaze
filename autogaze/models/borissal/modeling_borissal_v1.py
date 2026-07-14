"""Borissal v1: the learned, feed-forward, single-scale patch selector (Phase 2).

A small TSM-style 2D CNN scores patches at grid resolution; inference keeps
the top-k under a `gazing_ratio` budget exactly like v0 (same `Selection`
contract, same canonical ascending keep-index guarantee). Training uses a
fully differentiable straight-through path (`forward_train`) so a
self-supervised objective (see losses.py / train_borissal_v1.py) can reach
the scores.

Mobile constraints inherited from v0's readiness review: 2D convs + topk
only on the inference path, data-independent K (uniform allocation), fixed
input shape per export. Torch-only (standalone), like the rest of the core.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_borissal import BorissalConfig, BorissalV1Config
from .modeling_borissal import (
    Borissal,
    Selection,
    _largest_remainder,
    _pack_gazing_mask,
)


def _selection_from_scores(
    S: torch.Tensor,          # (B, T_grid, H_grid, W_grid) float
    ratio: float,
    alloc: str,
    eps: float,
    min_keep_per_frame_ratio: float = 0.25,
) -> Selection:
    """Budget allocation + hard top-k + canonical packing, from a score grid.

    Mirrors v0's selection tail (modeling_borissal._select_impl) so v1 output
    obeys the identical Selection contract, including ascending keep_index.
    """
    B, T_grid, H_grid, W_grid = S.shape
    N_pf = H_grid * W_grid
    L = T_grid * N_pf
    device = S.device
    scores_flat = S.reshape(B, T_grid, N_pf)

    if alloc == "global":
        # Mirrors v0's global mode: one clip-wide top-K_total with a
        # guaranteed per-tubelet minimum m.
        K_total = min(max(T_grid, round(ratio * L)), L)
        m = max(1, int(round(min_keep_per_frame_ratio * K_total / T_grid)))
        m = min(m, K_total // T_grid)
        _, gidx = scores_flat.topk(m, dim=-1)
        guaranteed = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
        guaranteed.scatter_(-1, gidx, True)
        global_scores = (scores_flat + 10.0 * guaranteed.to(scores_flat.dtype)).reshape(B, L)
        _, kidx = global_scores.topk(K_total, dim=-1)
        keep_mask_flat = torch.zeros(B, L, dtype=torch.bool, device=device)
        keep_mask_flat.scatter_(1, kidx, True)
        keep_mask_grid = keep_mask_flat.reshape(B, T_grid, N_pf)
    else:
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

        k_max = int(k_per_frame.max().item())
        _, topk_idx = scores_flat.topk(k_max, dim=-1)
        within_topk_rank = torch.arange(k_max, device=device).view(1, 1, k_max).expand(B, T_grid, k_max)
        keep_within_topk = within_topk_rank < k_per_frame.unsqueeze(-1)
        keep_mask_grid = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=device)
        keep_mask_grid.scatter_(-1, topk_idx, keep_within_topk)

    per_frame_keep = keep_mask_grid.sum(dim=-1)
    num_keep = per_frame_keep.sum(dim=-1)

    keep_mask = keep_mask_grid.reshape(B, L)
    keep_index, is_padded = _pack_gazing_mask(keep_mask)

    idx_safe = keep_index.clamp(min=0)
    t_coord = idx_safe // N_pf
    rem = idx_safe % N_pf
    h_coord = rem // W_grid
    w_coord = rem % W_grid
    coords = torch.stack([t_coord, h_coord, w_coord], dim=-1)
    coords = coords.masked_fill(is_padded.unsqueeze(-1), -1)

    grid_thw = torch.tensor([T_grid, H_grid, W_grid], dtype=torch.long, device=device)
    grid_thw = grid_thw.unsqueeze(0).expand(B, 3).clone()

    return Selection(
        grid_thw=grid_thw,
        scores=scores_flat.reshape(B, L),
        keep_mask=keep_mask,
        keep_index=keep_index,
        keep_coords=coords,
        num_keep=num_keep,
        per_frame_keep=per_frame_keep,
    )


class TemporalShift(nn.Module):
    """Zero-cost temporal mixing: shift a fraction of channels one tubelet
    forward / one backward (TSM). Input (B, T, C, H, W)."""

    def __init__(self, shift_fraction: float):
        super().__init__()
        self.shift_fraction = shift_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        c = max(1, int(C * self.shift_fraction))
        out = torch.zeros_like(x)
        out[:, 1:, :c] = x[:, :-1, :c]        # shift forward in time
        out[:, :-1, c:2 * c] = x[:, 1:, c:2 * c]  # shift backward in time
        out[:, :, 2 * c:] = x[:, :, 2 * c:]   # rest untouched
        return out


class _TSMBlock(nn.Module):
    def __init__(self, channels: int, shift_fraction: float):
        super().__init__()
        self.shift = TemporalShift(shift_fraction)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, Hg, Wg)
        B, T, C, H, W = x.shape
        residual = x
        x = self.shift(x)
        x = self.conv(x.reshape(B * T, C, H, W))
        x = self.norm(x)
        x = F.gelu(x)
        return residual + x.view(B, T, C, H, W)


class BorissalV1(nn.Module):
    """Learned patch selector: TSM 2D CNN over grid-resolution inputs.

    Inference (`select`) matches v0's call surface and Selection contract.
    Training (`forward_train`) exposes a straight-through differentiable
    selection so gradients reach the score head.
    """

    def __init__(self, config: BorissalV1Config):
        super().__init__()
        self.config = config

        in_channels = {"maps": 2, "pixels": 3, "both": 5}[config.input_mode]
        self.stem = nn.Conv2d(in_channels, config.hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [_TSMBlock(config.hidden_channels, config.shift_fraction) for _ in range(config.num_blocks)]
        )
        self.head = nn.Conv2d(config.hidden_channels, 1, kernel_size=1)

        # Non-learned v0 signal provider (maps input / residual scoring). Not
        # registered parameters; buffers only (sobel kernels), negligible cost.
        # config.v0_preset picks the signal generation: the gate-validated
        # v0.2 preset (default) or the plain v0.1 baseline. Only SIGNAL knobs
        # matter here (the provider's own allocation settings are unused --
        # v1 does its own selection).
        v0_kwargs = dict(
            scale=config.scale,
            patch_size=config.patch_size,
            tubelet_size=config.tubelet_size,
            motion_weight=config.motion_weight,
            spatial_op=config.spatial_op,
            pooling=config.pooling,
            gazing_ratio=config.gazing_ratio,
            eps=config.eps,
        )
        if config.v0_preset == "v0.2":
            # Keep v0.2 SIGNAL knobs (frame-diff, noise floor, score blend);
            # override its selection-stage knobs (global allocation, block
            # gate) to the cheap defaults -- they don't affect the maps v1
            # consumes, only v0's own selection, which is unused here.
            self._v0 = Borissal(BorissalConfig.v0_2(
                **v0_kwargs, per_frame_allocation="uniform", block_size=1,
            ))
        else:
            self._v0 = Borissal(BorissalConfig(**v0_kwargs))

    # ------------------------------------------------------------------ inputs

    def _grid_inputs(self, video: torch.Tensor):
        """Build backbone input (B, T_grid, C_in, H_grid, W_grid) and the v0
        combined score grid (for residual scoring), all without gradients --
        these are fixed functions of the input video, not learned."""
        cfg = self.config
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid = T // cfg.tubelet_size
        H_grid, W_grid = H // cfg.patch_size, W // cfg.patch_size

        with torch.no_grad():
            need_maps = cfg.input_mode in ("maps", "both") or cfg.residual_scoring
            v0_maps = None
            v0_score = None
            if need_maps:
                _, inter = self._v0.select_with_intermediates(video)
                v0_maps = torch.stack([inter["motion_norm"], inter["spatial_norm"]], dim=2)
                # (B, T_grid, 2, H_grid, W_grid)
                v0_score = inter["score"]  # (B, T_grid, H_grid, W_grid)

            pixels = None
            if cfg.input_mode in ("pixels", "both"):
                tub = video.view(B, T_grid, cfg.tubelet_size, C, H, W).mean(dim=2)  # (B,T_grid,C,H,W)
                pixels = F.avg_pool2d(
                    tub.reshape(B * T_grid, C, H, W),
                    kernel_size=cfg.patch_size, stride=cfg.patch_size,
                ).view(B, T_grid, C, H_grid, W_grid)

        if cfg.input_mode == "maps":
            x = v0_maps
        elif cfg.input_mode == "pixels":
            x = pixels
        else:
            x = torch.cat([v0_maps, pixels], dim=2)
        return x, v0_score

    # ------------------------------------------------------------------ scores

    def scores(self, video: torch.Tensor) -> torch.Tensor:
        """Raw score logits (B, T_grid, H_grid, W_grid). Differentiable w.r.t.
        the network parameters (inputs are detached by construction)."""
        x, v0_score = self._grid_inputs(video)
        B, T, C, H, W = x.shape
        h = self.stem(x.reshape(B * T, C, H, W)).view(B, T, -1, H, W)
        for block in self.blocks:
            h = block(h)
        s = self.head(h.reshape(B * T, -1, H, W)).view(B, T, H, W)
        if self.config.residual_scoring:
            s = s + v0_score
        return s

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def select(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float] = None,
        per_frame_allocation: Optional[str] = None,
    ) -> Selection:
        """video: (B, T, C, H, W) float, resized/normalized. Same contract as v0."""
        cfg = self.config
        ratio = cfg.gazing_ratio if gazing_ratio is None else gazing_ratio
        alloc = per_frame_allocation or cfg.per_frame_allocation
        S = self.scores(video)
        return _selection_from_scores(S, ratio, alloc, cfg.eps, cfg.min_keep_per_frame_ratio)

    # ---------------------------------------------------------------- training

    def forward_train(
        self,
        video: torch.Tensor,
        gazing_ratio: Optional[float] = None,
    ) -> dict:
        """Differentiable selection for SSL training.

        Returns a dict:
            scores     (B, T_grid, H_grid, W_grid) -- raw logits (grad-capable)
            probs      (B, T_grid, N_pf)  -- per-tubelet softmax over patches
            hard_keep  (B, T_grid, N_pf) bool -- exact-k top-k of (noised) logits
            st_gate    (B, T_grid, N_pf)  -- forward==hard, backward flows via probs
            keep_index (B, K) long        -- canonical packed indices of hard_keep
            k          int                -- patches kept per tubelet (uniform)

        Training always uses uniform allocation (exact-k, data-independent K),
        matching the mobile-export-safe inference default.
        """
        cfg = self.config
        ratio = cfg.gazing_ratio if gazing_ratio is None else gazing_ratio

        S = self.scores(video)
        B, T_grid, H_grid, W_grid = S.shape
        N_pf = H_grid * W_grid
        logits = S.reshape(B, T_grid, N_pf)

        k = min(max(1, round(ratio * N_pf)), N_pf)

        if self.training and cfg.gumbel_tau > 0:
            # Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1) clamped away from
            # both endpoints so neither log can hit 0/inf. (Parenthesization
            # matters: `-torch.log(x).clamp_min(c)` would clamp BEFORE the
            # unary minus and silently produce NaNs -- caught in smoke.)
            u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-7)
            gumbel = -torch.log(-torch.log(u))
            noised = logits + cfg.gumbel_tau * gumbel
        else:
            noised = logits

        _, topk_idx = noised.topk(k, dim=-1)
        hard_keep = torch.zeros_like(logits, dtype=torch.bool)
        hard_keep.scatter_(-1, topk_idx, True)

        probs = logits.softmax(dim=-1)
        hard_f = hard_keep.to(logits.dtype)
        # Straight-through gate. The soft path is scaled by N_pf so its
        # backward magnitude is O(1) regardless of grid size (raw softmax
        # probs average 1/N_pf, which would shrink selector gradients by
        # ~3 orders of magnitude at the default 24x24 grid). Forward value
        # is exactly hard_f either way (the correction term is zero-valued).
        soft = probs * N_pf
        st_gate = hard_f + soft - soft.detach()

        keep_index, _ = _pack_gazing_mask(hard_keep.reshape(B, T_grid * N_pf))

        return {
            "scores": S,
            "probs": probs,
            "hard_keep": hard_keep,
            "st_gate": st_gate,
            "keep_index": keep_index,
            "k": k,
        }
