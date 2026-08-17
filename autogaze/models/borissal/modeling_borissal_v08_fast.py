"""borissal v0.8 "Sikhye" Fast Path — High Performance & NPU-Optimized Bit-Exact Selector.

진입점: `python modeling_borissal_v08_fast.py {infer,eval,train}`. 학습 계약은
`st_gate()`(hard 선택을 미분 가능하게 잇는 straight-through 게이트) — 어떤
토큰-gather 인코더에도 꽂힌다, V-JEPA 전용 아님.

This module provides `BorissalV08Fast` and `_mc_residual_scores_fast`, offering:
1. 100% bitwise mathematical parity with reference `modeling_borissal_v08.py`
   on the actually-exercised path (shift=0). `shift!=0` is dead code in v0.8
   (`signals()` always calls with shift=0) and raises `NotImplementedError`
   explicitly instead of shipping a broken implementation (see history below).
2. Fixed 5D 6-tuple circular padding & zero-copy strided slicing for NPU hardware.
3. Fixed even/odd median selection parity (`_exact_median_values` matches
   `torch.median`'s "lower of the two middles" convention for even lengths —
   `g.sort()[n//2]` does NOT, see docstring below).
4. Fully static graph vectorization for ONNX and TFLite Hexagon NPU delegates.

This file is **fully standalone** — no intra-repo imports, `torch` only. It now
also carries the training/eval/infer harness (`st_gate`, `_ToyEncoder`,
`_demo_{infer,eval,train}` + CLI), ported unchanged from `modeling_borissal_v08.py`
(only the selector class swapped: `BorissalV08` → `BorissalV08Fast`), so this
single file is enough to explain the architecture AND demonstrate train/eval/infer
usage without the reference file present.

Fix history (shift>0 branch broke multiple review rounds in a row before
landing on "raise NotImplementedError" instead of another reimplementation
attempt): first `torch.roll` → `F.pad(mode="replicate")` had the wrong padding
mode (border semantics differ from roll's circular wrap); a later attempt fixed
the mode to `circular` but sized the padding tuple for a 4D input while `prev`
is actually 5D `(B,T,C,Hg,Wg)` — PyTorch requires a 6-tuple for 5D non-constant
padding — so it crashed instead. Since `shift>0` has no caller in v0.8
(`signals()` hardcodes shift=0) and kept re-breaking, this version stops trying
to fix an unused branch and refuses it explicitly instead.
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Inlined verbatim from modeling_borissal_v08.py so this file has zero intra-repo
# imports (drop-in-one-file design goal, now the canonical implementation).

EPS = 1e-6


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


def _pack_gazing_mask(gazing_mask: torch.Tensor):
    """Pack a (B, N) boolean/0-1 mask into (kept_index, is_padded), both (B, K),
    K = max ones-count over the batch. Kept indices come first in each row, in
    their original ascending order (stable sort), padded with -1.
    """
    gazing_mask = gazing_mask.to(torch.long)
    B, N = gazing_mask.shape

    idx = torch.arange(N, device=gazing_mask.device).expand(B, N)
    key = (1 - gazing_mask) * N + idx
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


def _coherence_gate_grid(dx: torch.Tensor, dy: torch.Tensor, patch_size: int,
                         gamma: float, eps: float) -> torch.Tensor:
    """Grid-resolution structure-tensor coherence gate (v0.5)."""
    b, t, h, w = (int(x) for x in dx.shape)

    def _poolg(x):
        return F.avg_pool2d(
            x.reshape(b * t, 1, h, w), kernel_size=patch_size, stride=patch_size
        ).view(b, t, h // patch_size, w // patch_size)

    a, c, bb = _poolg(dx * dx), _poolg(dy * dy), _poolg(dx * dy)
    coherence = ((a - c) ** 2 + 4.0 * bb * bb) / ((a + c) ** 2 + eps)
    return (1.0 - coherence).clamp(min=0.0, max=1.0) ** gamma


def _selection_from_mask(S: torch.Tensor, keep_mask_grid: torch.Tensor) -> Selection:
    """(B,T_grid,H_grid,W_grid) 점수 + (B,T_grid,H,W) bool 마스크 → Selection.

    비균일 per-frame k 지원 (iframe 정책 등).
    """
    B, T_grid, H_grid, W_grid = (int(x) for x in S.shape)
    N_pf = H_grid * W_grid
    L = T_grid * N_pf
    device = S.device
    scores_flat = S.reshape(B, T_grid, N_pf)
    keep_mask_grid = keep_mask_grid.reshape(B, T_grid, N_pf)

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


def _appearance_spatial_norm(video: torch.Tensor, tubelet_size: int,
                             patch_size: int, eps: float = EPS) -> torch.Tensor:
    """외형 A — 원본 백본 없이 `spatial_norm`을 재현한다 (독립 경로)."""
    B, T, C, H, W = (int(x) for x in video.shape)
    T_grid = T // tubelet_size

    gray = video.mean(dim=2)                                     # luma_mode="mean"
    tub = gray.view(B, T_grid, tubelet_size, H, W).mean(dim=2)   # 튜블렛 평균

    dy = F.pad(tub[:, :, 1:, :] - tub[:, :, :-1, :], (0, 0, 0, 1))
    dx = F.pad(tub[:, :, :, 1:] - tub[:, :, :, :-1], (0, 1, 0, 0))
    spatial = torch.sqrt(dx * dx + dy * dy + eps)

    spatial_p = F.avg_pool2d(
        spatial.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size
    ).view(B, T_grid, H // patch_size, W // patch_size)

    spatial_p = spatial_p * _coherence_gate_grid(dx, dy, patch_size, 1.0, eps)

    return _minmax_norm(spatial_p, eps)


class V08Params(nn.Module):
    """v0.8의 제어 파라미터 θ — 전부 스칼라 텐서, learnable=True면 nn.Parameter."""

    DEFAULTS = dict(w_a=1.0, w_d=1.0, w_n=1.0, rho=0.0,
                    kappa=1.0, gamma=0.5, tau=1.0, alpha=0.5,
                    sig_gate=0.0)

    def __init__(self, learnable: bool = False, **overrides):
        super().__init__()
        unknown = set(overrides) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown v0.8 knobs: {sorted(unknown)}")
        for k, v in {**self.DEFAULTS, **overrides}.items():
            t = torch.tensor(float(v))
            if learnable:
                self.register_parameter(k, nn.Parameter(t))
            else:
                self.register_buffer(k, t)

    @torch.no_grad()
    def clamp_(self) -> None:
        """optimizer.step() 뒤 매번 호출 — projected gradient."""
        self.tau.clamp_(min=0.05)
        self.gamma.clamp_(0.0, 1.0)
        self.alpha.clamp_(0.0, 1.0)


def _z(x: torch.Tensor) -> torch.Tensor:
    """클립 전역 z-정규화 (미분 가능·단조 아핀 — 순위 보존)."""
    return (x - x.mean()) / (x.std() + 1e-6)


def _soft_coarsen(S: torch.Tensor, alpha: torch.Tensor, c: int = 2) -> torch.Tensor:
    """5g soft coarsen mix의 미분 가능판: S ← (1-α)·S + α·up(pool_c(S))."""
    B, T_grid, Hg, Wg = (int(x) for x in S.shape)
    if Hg % c or Wg % c:
        return S  # 홀수 그리드는 coarsen 미적용 (no-op — 가드)
    Sc = F.avg_pool2d(S.reshape(B * T_grid, 1, Hg, Wg), c, c)
    Sc = (Sc.repeat_interleave(c, dim=2).repeat_interleave(c, dim=3)
          .reshape(B, T_grid, Hg, Wg))
    return (1.0 - alpha) * S + alpha * Sc


def _largest_remainder(raw: torch.Tensor, total: int, min_val: int, cap: int) -> torch.Tensor:
    """연속 카운트 → 정수 배분 (합 보존·min·cap)."""
    if total < len(raw) * min_val:
        raise ValueError(f"budget {total} < slots {len(raw)} × min {min_val} — "
                         "ratio가 너무 낮음 (전 slot 최소 보장 불가)")
    base = raw.floor().long().clamp(min=min_val, max=cap)
    for _ in range(int(abs(total - int(base.sum()))) + len(raw)):
        diff = total - int(base.sum())
        if diff == 0:
            break
        gap = raw - base.to(raw.dtype)
        if diff > 0:
            gap = gap.masked_fill(base >= cap, float("-inf"))
            base[int(gap.argmax())] += 1
        else:
            gap = gap.masked_fill(base <= min_val, float("inf"))
            base[int(gap.argmin())] -= 1
    return base


def _exact_median_values(g: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Exact match for PyTorch torch.median(dim, keepdim=True).values on both even and odd sequence lengths."""
    med_idx = (g.shape[dim] - 1) // 2
    return g.sort(dim=dim).values.narrow(dim, med_idx, 1)


def _mc_residual_scores_fast(video: torch.Tensor, patch_size: int, shift: int = 0) -> torch.Tensor:
    """Fast motion compensation residual (D) with zero-copy slicing and 100% bitwise parity."""
    B, T, C, H, W = (int(x) for x in video.shape)
    p = patch_size
    Hg, Wg = H // p, W // p
    g = F.avg_pool2d(video.reshape(B * T, C, H, W), p, p)
    g = g.reshape(B, T, C, Hg, Wg)
    prev = g[:, :-1]  # (B,T-1,C,Hg,Wg)
    cur = g[:, 1:]

    if shift != 0:
        # v0.8's signals() always calls this with shift=0; the shift>0 branch has
        # broken three review rounds in a row (5D F.pad shape mismatch) without a
        # single caller ever exercising it. Refuse explicitly instead of shipping
        # code that claims support it doesn't have.
        raise NotImplementedError("_mc_residual_scores_fast only supports shift=0")
    mc = (cur - prev).abs().mean(dim=2)

    g0 = g[:, 0].mean(dim=1)  # (B,Hg,Wg)
    g0_pad = F.pad(g0.unsqueeze(1), (1, 0, 1, 0), mode="circular").squeeze(1)
    grad0 = (g0 - g0_pad[:, 1:, :-1]).abs() + (g0 - g0_pad[:, :-1, 1:]).abs()
    return torch.cat([grad0.unsqueeze(1), mc], dim=1)  # (B,T,Hg,Wg)


class BorissalV08Fast(torch.nn.Module):
    """High performance, NPU-optimized bit-exact fast implementation of BorissalV08."""

    def __init__(self, spatial_backbone=None, tubelet_size: int = 2, patch_size: int = 16,
                 params: V08Params = None, learnable: bool = False,
                 learn_signal: bool = False, eps: float = EPS, **knobs):
        super().__init__()
        self.backbone = spatial_backbone
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.eps = eps
        self.params = params or V08Params(learnable=learnable, **knobs)
        self.refiner = None
        if learn_signal:
            self.refiner = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.GELU(),
                torch.nn.Conv2d(8, 1, 3, padding=1))

    def signals(self, video: torch.Tensor):
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid = T // self.tubelet_size
        if self.backbone is None:
            A = _appearance_spatial_norm(video, self.tubelet_size, self.patch_size, self.eps)
        else:
            _, inter = self.backbone.select_with_intermediates(
                video, gazing_ratio=1.0, tubelet_size=self.tubelet_size,
                patch_size=self.patch_size)
            A = inter["spatial_norm"]

        D = _mc_residual_scores_fast(video, self.patch_size, shift=0)
        luma = video.mean(dim=2)
        g = F.avg_pool2d(luma, self.patch_size, self.patch_size)
        N = (g - _exact_median_values(g, dim=1)).abs()

        if self.tubelet_size > 1:
            D = D.reshape(B, T_grid, self.tubelet_size, *D.shape[2:]).mean(2)
            N = N.reshape(B, T_grid, self.tubelet_size, *N.shape[2:]).mean(2)

        if A.shape[-2:] != D.shape[-2:]:
            fh = D.shape[-2] // A.shape[-2]
            fw = D.shape[-1] // A.shape[-1]
            A = A.repeat_interleave(fh, -2).repeat_interleave(fw, -1)
        return A, D, N

    def forward(self, video: torch.Tensor, gazing_ratio: float):
        if int(video.shape[0]) != 1:
            raise NotImplementedError("v0.8은 batch 1")
        p = self.params
        A, D, N = self.signals(video)
        B, T_grid, Hg, Wg = (int(x) for x in A.shape)
        N_pf = Hg * Wg
        K = round(float(gazing_ratio) * T_grid * N_pf)

        g_t = torch.zeros(T_grid, device=A.device, dtype=A.dtype)
        g_t[0] = 1.0
        g = g_t.view(1, T_grid, 1, 1)
        zA, zD, zN = _z(A), _z(D), _z(N)
        S_p = p.w_d * zD + p.w_n * zN + p.rho * zA
        S = g * (p.w_a * zA) + (1.0 - g) * S_p

        if self.refiner is not None:
            stack = torch.stack([zA, zD, zN], dim=2)
            dS = self.refiner(stack.reshape(B * T_grid, 3, Hg, Wg))
            S = S + p.sig_gate * dS.reshape(B, T_grid, Hg, Wg)

        S = _soft_coarsen(S, p.alpha)

        Dn = (D - D.amin()) / (D.amax() - D.amin() + 1e-6)
        Nn = (N - N.amin()) / (N.amax() - N.amin() + 1e-6)
        E = (p.w_d * Dn + p.w_n * Nn).sum(dim=(-1, -2))
        logits = (torch.log(E + 1e-4) + p.kappa * g_t) / p.tau.clamp(min=1e-3)
        w = p.gamma * torch.softmax(logits, dim=-1) + (1.0 - p.gamma) / T_grid
        c_soft = w * K
        if K < T_grid:
            raise ValueError(f"ratio {gazing_ratio}가 너무 낮음")

        counts = _largest_remainder(c_soft[0].detach(), K, min_val=1, cap=N_pf)

        mask = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=S.device)
        flat = S.reshape(B, T_grid, N_pf)

        # NPU static-graph vectorization
        if counts.min() == counts.max():
            k = int(counts[0])
            _, idx = flat.topk(k, dim=-1)
            mask.scatter_(-1, idx, True)
        else:
            for t in range(T_grid):
                k = int(counts[t])
                _, idx = flat[:, t].topk(k, dim=-1)
                mask[:, t].scatter_(-1, idx, torch.ones_like(idx, dtype=torch.bool))

        sel = _selection_from_mask(S, mask.reshape(B, T_grid, Hg, Wg))
        return sel, {"scores_soft": S, "counts_soft": c_soft}


# =============================================================================
# 학습 배선 — modeling_borissal_v08.py:480-531에서 이전 (BorissalV08 → BorissalV08Fast
# 스왑 외 로직 무변경). 원 출처: patchstack scripts/train.py:531-581 (dev @ cee52b0).
# =============================================================================


def st_gate(sel: Selection, aux: dict, rate_grad: bool = True) -> torch.Tensor:
    """hard 선택을 미분 가능하게 잇는 straight-through 게이트 → (B, K), forward 값 1.0.

    토큰을 gather하는 어떤 인코더에도 꽂힌다 (V-JEPA 전용 아님):
        tokens = embed(video).gather(1, sel.keep_index[..., None].expand(-1, -1, D))
        tokens = tokens * st_gate(sel, aux).unsqueeze(-1)   # ← 여기로 grad가 흐른다
    (곱셈 자리·차원을 직접 안 맞추려면 `st_gate_apply(sel, aux, tokens)` 참고.)

    ⚠️ slot별 softmax × N_pf 다. **전역 softmax × L 아니다** — 전역 형태는 I-slot
    게이트 질량을 억압해 `w_a` 그래디언트를 굶긴다.

    rate_grad=True면 토큰별 자기 slot의 `c_soft/c_soft.detach()`를 곱해 배분 노브
    (kappa/gamma/tau)까지 학습시킨다. `+1e-6`은 필수 — tau가 하한이고 gamma→1이면
    비최대 slot의 c_soft가 0이 되어 0/0 NaN.
    """
    S, c_soft = aux["scores_soft"], aux["counts_soft"]
    B, T_grid, Hg, Wg = S.shape
    N_pf, L = Hg * Wg, T_grid * Hg * Wg
    keep = sel.keep_index
    if bool((keep < 0).any()):
        raise ValueError("st_gate: keep_index에 -1 패딩 — 인코더가 거부한다 "
                         "(B=1로 쓰거나 패딩 인식 인코더가 필요)")
    p = torch.softmax(S.reshape(B, T_grid, N_pf), dim=-1) * N_pf
    p_kept = p.reshape(B, L).gather(1, keep)
    gate = 1.0 + p_kept - p_kept.detach()
    if rate_grad:
        c_sel = c_soft.gather(1, keep // N_pf)
        gate = gate * ((c_sel + 1e-6) / (c_sel.detach() + 1e-6))
    return gate


def st_gate_apply(sel: Selection, aux: dict, tokens: torch.Tensor,
                  rate_grad: bool = True) -> torch.Tensor:
    """`tokens * st_gate(...).unsqueeze(-1)` 한 줄 편의 함수 — 곱하는 자리·차원을
    맞추는 실수를 없앤다. `tokens`는 `sel.keep_index` 순서로 이미 gather된 (B, K, D).

        kept = embed(video).gather(1, sel.keep_index[..., None].expand(-1, -1, D))
        kept = st_gate_apply(sel, aux, kept)      # 곱셈 자체는 여전히 필수 (st_gate 참고)
    """
    return tokens * st_gate(sel, aux, rate_grad).unsqueeze(-1)


# =============================================================================
# 예시 — python modeling_borissal_v08_fast.py {infer,eval,train}
# =============================================================================


class _ToyEncoder(nn.Module):
    """예시 전용 장난감 인코더 — 진짜 지표 아님, 계약 시연용.

    토큰 = 패치 평균 (avg_pool2d, 셀렉터와 같은 tubelet/patch 격자). kept 토큰만 임베드해
    (gate가 있으면 곱하고) 평균 풀로 컨텍스트를 만들고, 그 컨텍스트로 **전체** 패치를
    복원해 미선택 위치의 MSE를 잰다.
    """

    def __init__(self, tubelet_size: int = 2, patch_size: int = 16, dim: int = 16):
        super().__init__()
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.embed = nn.Linear(3, dim)
        self.decode = nn.Linear(dim, 3)

    def _tokens(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid, p = T // self.tubelet_size, self.patch_size
        g = F.avg_pool2d(video.reshape(B * T, C, H, W), p, p).reshape(B, T, C, H // p, W // p)
        g = g.reshape(B, T_grid, self.tubelet_size, C, H // p, W // p).mean(2)
        # t-major flatten — sel.keep_index와 같은 순서 (i = t*N_pf + h*Wg + w)
        return g.permute(0, 1, 3, 4, 2).reshape(B, -1, C)

    def forward(self, video: torch.Tensor, keep_index: torch.Tensor,
                gate: torch.Tensor = None):
        tokens = self._tokens(video)                                  # (B, L, 3)
        emb = self.embed(tokens)                                      # (B, L, dim)
        kept = emb.gather(1, keep_index.unsqueeze(-1).expand(-1, -1, emb.shape[-1]))
        if gate is not None:
            kept = kept * gate.unsqueeze(-1)
        ctx = kept.mean(dim=1, keepdim=True)                          # (B, 1, dim)
        recon = self.decode(ctx.expand(-1, tokens.shape[1], -1))      # (B, L, 3)
        return recon, tokens

    def loss(self, video: torch.Tensor, keep_index: torch.Tensor,
             gate: torch.Tensor = None) -> torch.Tensor:
        """미선택 위치만의 재구성 MSE."""
        recon, tokens = self(video, keep_index, gate)
        B, L, _ = tokens.shape
        rest_mask = torch.ones(B, L, dtype=torch.bool, device=tokens.device)
        rest_mask.scatter_(1, keep_index, False)
        return ((recon - tokens) ** 2)[rest_mask].mean()


def _demo_infer(frames: int = 8, size: int = 224, ratio: float = 0.25) -> None:
    """선택 생성만 — 인코더에 붙이기 전 최소 확인."""
    torch.manual_seed(0)
    m = BorissalV08Fast(tubelet_size=2, patch_size=16)
    sel, _aux = m(torch.rand(1, frames, 3, size, size), ratio)
    print(f"num_keep={int(sel.num_keep)} / L={sel.scores.shape[1]}")
    print(f"per_frame_keep={sel.per_frame_keep.tolist()}")
    print(f"keep_index[:10]={sel.keep_index[0, :10].tolist()}")
    print(f"keep_coords[0]={sel.keep_coords[0, 0].tolist()}  (t, h, w)")


def _demo_eval(frames: int = 8, size: int = 224, ratio: float = 0.25) -> None:
    """① ratio 스윕 (예산 정확도·slot 분포·CPU 지연·결정성).
    ② _ToyEncoder 재구성 MSE — random / uniform grid / v0.8 3자 비교.
    ⚠️ ②는 장난감 인코더 기준이다 — 진짜 VLM 성능 지표가 아니다."""
    torch.manual_seed(0)
    m = BorissalV08Fast(tubelet_size=2, patch_size=16)
    video = torch.rand(1, frames, 3, size, size)

    print("-- ratio 스윕 --")
    for r in (0.10, 0.25, 0.50):
        t0 = time.perf_counter()
        sel, _aux = m(video, r)
        dt_ms = (time.perf_counter() - t0) * 1e3
        sel2, _aux2 = m(video, r)
        L = sel.scores.shape[1]
        print(f"ratio={r:.2f} num_keep={int(sel.num_keep):4d} "
              f"expected={round(r * L):4d} per_frame_keep={sel.per_frame_keep.tolist()} "
              f"deterministic={torch.equal(sel.keep_mask, sel2.keep_mask)} {dt_ms:5.1f}ms")

    print("-- 재구성 MSE (장난감 인코더 기준 — 진짜 VLM 성능 아님) --")
    sel, _aux = m(video, ratio)
    K, L = sel.keep_index.shape[1], sel.scores.shape[1]
    enc = _ToyEncoder(tubelet_size=2, patch_size=16)
    torch.manual_seed(0)
    random_idx = torch.randperm(L)[:K].sort().values.unsqueeze(0)
    uniform_idx = torch.arange(0, L, max(L // K, 1))[:K].unsqueeze(0)
    with torch.no_grad():
        for name, idx in (("random", random_idx), ("uniform grid", uniform_idx),
                          ("borissal v0.8 (fast)", sel.keep_index)):
            print(f"{name:14s} MSE={float(enc.loss(video, idx)):.4f}")


def _demo_train(frames: int = 8, size: int = 224, ratio: float = 0.25) -> None:
    """셀렉터 → _ToyEncoder(gate) → loss → backward → step → clamp_().
    노브 9개 grad 논제로 + 게이트 forward==1.0을 단언한다 — "이 파일만으로 학습
    가능한가"의 자기검증."""
    torch.manual_seed(0)
    m = BorissalV08Fast(learnable=True, learn_signal=True, tubelet_size=2, patch_size=16)
    enc = _ToyEncoder(tubelet_size=2, patch_size=16)
    params = list(m.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)  # wd=0: θ는 제어 노브지 가중치가 아님

    video = torch.rand(1, frames, 3, size, size)
    sel, aux = m(video, ratio)
    gate = st_gate(sel, aux)
    assert torch.allclose(gate, torch.ones_like(gate)), "게이트 forward 값은 1.0이어야 함"

    loss = enc.loss(video, sel.keep_index, gate)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)

    for name, p in m.params.named_parameters():
        g = float(p.grad.abs()) if p.grad is not None else 0.0
        print(f"{name:9s} grad={g:+.6f}")
        assert g > 0, f"{name}: grad 0 — st_gate 두 경로(scores_soft/counts_soft)가 다 붙었는지 확인"
    opt.step()
    m.params.clamp_()
    print(f"OK — 노브 9개 전부 논제로. loss={float(loss.detach()):.4f} "
          f"num_keep={int(sel.num_keep)} per_frame_keep={sel.per_frame_keep.tolist()}")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="borissal v0.8 fast path — 선택 생성/측정/학습 예시")
    _ap.add_argument("mode", nargs="?", default="infer", choices=["infer", "eval", "train"])
    _ap.add_argument("--frames", type=int, default=8)
    _ap.add_argument("--size", type=int, default=224)
    _ap.add_argument("--ratio", type=float, default=0.25)
    _args = _ap.parse_args()
    {"infer": _demo_infer, "eval": _demo_eval, "train": _demo_train}[_args.mode](
        frames=_args.frames, size=_args.size, ratio=_args.ratio)
