"""borissal v0.8 "Sikhye" — route/코덱 실험의 원리를 자체 신호로 내재화한 차세대 셀렉터.

진입점: `python modeling_borissal_v08.py {infer,eval,train}`. 학습 계약은 `st_gate()`
(hard 선택을 미분 가능하게 잇는 straight-through 게이트) — 어떤 토큰-gather 인코더에도
꽂힌다, V-JEPA 전용 아님.

이 파일은 **완전 독립**이다. 형제 모듈을 import하지 않으며 외부 의존은 `torch`뿐 —
다른 레포에 그대로 떨어뜨려도 동작한다.

이식 출처: patchstack `dev` @ cee52b0, `src/models/borissal/modeling_borissal_v08.py`.
독립화를 위해 아래 조각들을 **본문 무변경**으로 이 파일에 합쳤다:

  _pack_gazing_mask       AutoGaze modeling_borissal.py:239-270
  _minmax_norm            AutoGaze modeling_borissal.py:57-64
  _coherence_gate_grid    AutoGaze signals_v03.py:82-104
  _mc_residual_scores     patchstack src/selector.py:114-146
  _selection_from_mask    patchstack src/selector.py:72-113
  Selection               AutoGaze modeling_borissal.py:41-55 (필드 동일 — 어댑터는
                          isinstance 검사를 하지 않으므로 재정의해도 그대로 먹는다)

원본과 다른 점은 셋:
  1. 런타임 import(`from ...selector import ...`)를 파일 내 private 함수 참조로 교체.
  2. `spatial_backbone`이 필수 → **옵션(기본 None)**. None이면 아래
     `_appearance_spatial_norm`이 외형 A를 만든다(= 독립 경로). 원본 `Borissal`
     인스턴스를 넘기면 그쪽 `spatial_norm`을 쓴다(= patchstack과 동일 경로).
     두 경로가 같은 결과를 내야 한다 — 이식 정확성의 자기검증 지점.
  3. **학습·사용 배선을 파일 안으로**: `st_gate()` · `V08Params.clamp_()` ·
     `_ToyEncoder` · `_demo_{infer,eval,train}` + CLI. patchstack
     `scripts/train.py:531-581`(dev @ cee52b0)에서 이전 — 로직 무변경. patchstack엔
     이미 완성된 학습 하네스(ST+GRPO, θ 체크포인트 14종)가 있고 이 파일은 그중
     인코더까지 미분되는 부분만 가져온다; remote가 없는 로컬 레포라 이전은 수동
     파일 복사다.

---- 이하 원본 독스트링 ----

기원 (Step 6c, 2026-08-08): 코덱 점수를 빌리는 route(6b) 대신, 실측에서 추출한
원리 4개를 미분 가능한 자체 파이프라인으로 재구성 — 코덱 엔진 불요 (선택 ms급).

원리 → 설계 (전부 실측 근거, DEVLOG 5f~6b):
1. 시간 역할 분리 (codec-iframe·route의 본질): I-slot = 외형/커버리지, P-slot =
   변화의 정보량. 연속 게이트 ρ로 hard 분기까지 완화 (ρ=0 → route식 hard 분리).
2. P-점수 = 변화(D, 프레임 차분 — 5f: 보상 잔차는 해로움) + 놀라움(N, 시간 중앙값
   편차) — bit-cost의 "변화량 + 엔트로피" 구조 모사.
3. 예산 적응: I-slot 부스트 κ·coarsen α가 연속 노브 (저예산 몰빵 방지는 배분이
   구조적으로 처리 — softmax + min 1).
4. rate-control lite: slot 예산 = (1-γ)·균등 + γ·에너지 비례 — 컷/사건 slot에 자동
   가중 (6b에서 "후속"으로 남긴 것).

구조 요건 (사용자 지시): ① 전 노브 컨트롤러블 (하드코딩 0 — V08Params 주입),
② 학습 가능 (nn.Module, learnable=True면 전 θ가 nn.Parameter), ③ hard top-k 직전
까지 전 연산 미분 가능 — hard 지점은 v1 st_gate와 동일한 straight-through 브리지
지점으로 명시 (aux의 soft 출력이 학습 손실용).
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# BorissalConfig.eps 기본값 (configuration_borissal.py:292). 독립 파일이므로 상수로 박는다.
EPS = 1e-6


# =============================================================================
# 계약 — AutoGaze modeling_borissal.py:41-55 무변경
# =============================================================================


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


# =============================================================================
# 공유 유틸 — AutoGaze 원본 무변경 복사
# =============================================================================


def _minmax_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per (batch, tubelet) min-max normalize the last two (spatial) dims to [0, 1].

    [출처] AutoGaze modeling_borissal.py:57-64 무변경.
    """
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

    [출처] AutoGaze modeling_borissal.py:239-270 무변경.
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


def _coherence_gate_grid(dx: torch.Tensor, dy: torch.Tensor, patch_size: int,
                         gamma: float, eps: float) -> torch.Tensor:
    """Grid-resolution structure-tensor coherence gate (v0.5).

    The gradient PRODUCTS are pooled straight to the patch grid (the pooling
    window IS the structure-tensor window = patch_size), coherence is computed
    at grid resolution, and the returned (B, T, H//p, W//p) gate multiplies the
    already-pooled `spatial_p`.

    [출처] AutoGaze signals_v03.py:82-104 무변경.
    """
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

    [출처] patchstack src/selector.py:72-113 무변경 (그쪽 패킹 꼬리는 다시
    AutoGaze modeling_borissal_v1.py:102-127에서 온 것 — 같은 canonical 출구).
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


def _mc_residual_scores(video: torch.Tensor, patch_size: int = 16,
                        shift: int = 1) -> torch.Tensor:
    """모션 보상 잔차 점수 (Step 5f — codec bit-cost의 싼 프록시).

    프레임 t의 각 패치를 t-1의 patch-grid ±shift 이웃과 비교해 **최소** 잔차를 점수로.
    입력 [0,1] video (B,T,C,H,W) → (B, T, H/p, W/p). t=0은 자기 프레임 공간 그래디언트
    (I-프레임 상당 — 예측 대상 없음). 순수 torch, 결정적.

    shift=0 = 후보가 zero-shift 하나 = **단순 프레임 차분**. v0.8은 이것만 쓴다
    (5f 실측: 보상 잔차는 해로움).

    [출처] patchstack src/selector.py:114-146 무변경.
    """
    B, T, C, H, W = (int(x) for x in video.shape)
    p = patch_size
    Hg, Wg = H // p, W // p
    # 패치 평균 (C 유지) — patch-grid 표현 (B,T,C,Hg,Wg)
    g = torch.nn.functional.avg_pool2d(video.reshape(B * T, C, H, W), p, p)
    g = g.reshape(B, T, C, Hg, Wg)
    prev = g[:, :-1]                                    # (B,T-1,C,Hg,Wg)
    cur = g[:, 1:]
    residuals = []
    offsets = range(-shift, shift + 1)
    for dh in offsets:
        for dw in offsets:
            shifted = torch.roll(prev, shifts=(dh, dw), dims=(3, 4))
            residuals.append((cur - shifted).abs().mean(dim=2))   # (B,T-1,Hg,Wg)
    mc = torch.stack(residuals).min(dim=0).values               # 최소 잔차 = 보상 후
    # t=0: 공간 그래디언트 (I-프레임 상당)
    g0 = g[:, 0].mean(dim=1)                                    # (B,Hg,Wg)
    grad0 = (g0 - torch.roll(g0, 1, dims=-1)).abs() + (g0 - torch.roll(g0, 1, dims=-2)).abs()
    return torch.cat([grad0.unsqueeze(1), mc], dim=1)           # (B,T,Hg,Wg)


def _appearance_spatial_norm(video: torch.Tensor, tubelet_size: int,
                             patch_size: int, eps: float = EPS) -> torch.Tensor:
    """외형 A — 원본 백본 없이 `spatial_norm`을 재현한다 (독립 경로).

    patchstack은 백본으로 `BorissalConfig.v0_5(score_coarsen=1,
    selection_mode="topk", signal_grid="fine")`를 쓰고 그 `select_with_intermediates`의
    `intermediates["spatial_norm"]`을 가져간다. 프리셋 체인 v0_5 → v0_3 → v0_2 →
    클래스 기본값을 풀면 `spatial_norm`에 실제로 영향을 주는 값은 7개뿐이다:

        luma_mode="mean"        클래스 기본 (체인 어디서도 안 바꿈)
        spatial_diff="tubelet"  v0_5
        spatial_op="grad"       클래스 기본
        pooling="avg"           클래스 기본
        coherence_gate=True     v0_3
        coherence_at_grid=True  v0_5
        coherence_gamma=1.0     클래스 기본

    주의: v0_3의 `dog_blob_weight=0.5`는 여기 안 들어온다 — `_extra_channels`는 최종
    score S의 `fused_blend`에만 쓰이고, `spatial_norm = _minmax_norm(spatial_p, eps)`는
    그 앞 단계다 (modeling_borissal.py:561 vs :562-569).

    아래는 modeling_borissal.py:412-518의 해당 분기만 뽑아 이어붙인 것이다.
    """
    B, T, C, H, W = (int(x) for x in video.shape)
    T_grid = T // tubelet_size

    gray = video.mean(dim=2)                                     # luma_mode="mean" (:417)
    tub = gray.view(B, T_grid, tubelet_size, H, W).mean(dim=2)   # 튜블렛 평균 (:418)

    # spatial_diff="tubelet" → spatial_src = tub ; spatial_op="grad" (:473-484)
    dy = F.pad(tub[:, :, 1:, :] - tub[:, :, :-1, :], (0, 0, 0, 1))
    dx = F.pad(tub[:, :, :, 1:] - tub[:, :, :, :-1], (0, 1, 0, 0))
    spatial = torch.sqrt(dx * dx + dy * dy + eps)

    # pooling="avg" (:507-511)
    spatial_p = F.avg_pool2d(
        spatial.reshape(B * T_grid, 1, H, W), kernel_size=patch_size, stride=patch_size
    ).view(B, T_grid, H // patch_size, W // patch_size)

    # coherence_gate=True + coherence_at_grid=True, gamma=1.0 (:516-518).
    # spatial_diff="tubelet"이므로 dxc, dyc = dx, dy (:497-498).
    spatial_p = spatial_p * _coherence_gate_grid(dx, dy, patch_size, 1.0, eps)

    return _minmax_norm(spatial_p, eps)                          # (:561)


# =============================================================================
# v0.8 본체 — patchstack 원본 무변경 (spatial_backbone 기본값만 None으로)
# =============================================================================


class V08Params(nn.Module):
    """v0.8의 제어 파라미터 θ — 전부 스칼라 텐서, learnable=True면 nn.Parameter.

    노브 (전부 컨트롤러 주입 가능 — CONTROLLER_DESIGN slot-β의 실체):
    - w_a/w_d/w_n: 채널 가중 (외형 A / 변화 D / 놀라움 N)
    - rho: I/P 역할 혼합 (0 = hard 분리 — route식; >0 = P에도 외형 소량)
    - kappa: I-slot 예산 부스트 (softmax logit 가산 — 크면 iframe_full 근사)
    - gamma: 배분 혼합 (0 = 균등, 1 = 에너지 비례 — rate-control 강도)
    - tau: 배분 softmax 온도
    - alpha: coarsen soft-mix (0 = fine, 1 = hard 2×2 공유 — 5g)
    """

    DEFAULTS = dict(w_a=1.0, w_d=1.0, w_n=1.0, rho=0.0,
                    kappa=1.0, gamma=0.5, tau=1.0, alpha=0.5,
                    sig_gate=0.0)  # 학습형 신호 refiner 게이트 (Step 11 — 0 = 현행 동일)

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
        """optimizer.step() 뒤 매번 호출 — projected gradient. 무제약이면 NaN이 실증됐다:
        tau가 하한 밖·gamma가 [0,1] 밖으로 표류하면 st_gate()의 0/0이 재현되고, alpha가
        [0,1] 밖이면 soft-coarsen 외삽이 의미를 잃는다.
        [출처] patchstack scripts/train.py:576-581 — rate_grad 여부와 무관하게 무조건
        적용(조건부면 counts_soft 항을 나중에 추가할 때 재발)."""
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
    """연속 카운트 → 정수 배분 (합 보존·min·cap). hard 지점 — 학습 시 ST 브리지는
    `counts_st = counts + raw - raw.detach()` **가 아니다** (그 형태는 patchstack에도
    구현된 적 없는 주석뿐이고, `counts`가 `topk`의 크기 인자라 곱셈 위치가 아니어서
    이 한 줄만으론 gradient 경로가 안 생긴다). 실제 형태는 `st_gate()`의 `w_slot` 비율
    (`(c_sel + eps) / (c_sel.detach() + eps)`, 아래 `st_gate` 참고)."""
    if total < len(raw) * min_val:
        raise ValueError(f"budget {total} < slots {len(raw)} × min {min_val} — "
                         "ratio가 너무 낮음 (전 slot 최소 보장 불가)")
    base = raw.floor().long().clamp(min=min_val, max=cap)
    # 합 보존 재분배 (결정적): 부족분은 (raw-base) 큰 슬롯부터 +1, 초과분은 작은
    # 슬롯부터 -1 — cap/min 존중, 반드시 수렴 (total ≤ Σcap, ≥ Σmin 전제).
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


class BorissalV08(nn.Module):
    """v0.8 셀렉터 본체 — forward(video, ratio) -> (Selection, aux).

    aux = {"scores_soft": S, "counts_soft": c_soft} — hard top-k 이전의 미분 가능
    출력 (학습 손실·ST 브리지용).

    spatial_backbone (원본과 다른 점):
      None (기본) — 파일 내 `_appearance_spatial_norm`이 외형 A를 만든다. 완전 독립.
      원본 Borissal 인스턴스 — 그것의 `select_with_intermediates(...)["spatial_norm"]`을
        쓴다 (patchstack과 동일 경로). 백본은 frozen — v0_7 계열 fine 설정
        (`BorissalConfig.v0_5(score_coarsen=1, selection_mode="topk",
        signal_grid="fine")`)이어야 두 경로가 일치한다.
    """

    def __init__(self, spatial_backbone=None, tubelet_size: int = 2, patch_size: int = 16,
                 params: V08Params = None, learnable: bool = False,
                 learn_signal: bool = False, eps: float = EPS, **knobs):
        super().__init__()
        self.backbone = spatial_backbone      # None = 독립 경로
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.eps = eps
        self.params = params or V08Params(learnable=learnable, **knobs)
        # 학습형 신호 refiner (Step 11): patch-grid 위 초소형 conv (~300 파라미터,
        # FLOPs 미미 — B1 무관). S ← S + sig_gate·ΔS, sig_gate 초기 0 = 현행 비트
        # 동일 시작. sig_gate=0이면 refiner 가중치 grad도 0 — 2상 학습이 정상 동학.
        self.refiner = None
        if learn_signal:
            self.refiner = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.GELU(),
                nn.Conv2d(8, 1, 3, padding=1))

    def signals(self, video: torch.Tensor):
        """A(외형)·D(변화)·N(놀라움) — 전부 (B, T_grid, Hg, Wg), 미분 가능.

        A는 독립 경로(`_appearance_spatial_norm`) 또는 frozen 백본에서,
        D/N은 순수 torch 자체 계산.
        """
        B, T, C, H, W = (int(x) for x in video.shape)
        T_grid = T // self.tubelet_size
        if self.backbone is None:
            A = _appearance_spatial_norm(video, self.tubelet_size, self.patch_size, self.eps)
        else:
            _, inter = self.backbone.select_with_intermediates(
                video, gazing_ratio=1.0, tubelet_size=self.tubelet_size,
                patch_size=self.patch_size)
            A = inter["spatial_norm"]
        D = _mc_residual_scores(video, self.patch_size, shift=0)     # 프레임 차분
        # N: patch-grid luma의 시간 중앙값 편차 (프레임률 무관 "놀라움")
        luma = video.mean(dim=2)                                     # (B,T,H,W)
        g = F.avg_pool2d(luma, self.patch_size, self.patch_size)     # (B,T,Hg,Wg)
        N = (g - g.median(dim=1, keepdim=True).values).abs()
        if self.tubelet_size > 1:
            D = D.reshape(B, T_grid, self.tubelet_size, *D.shape[2:]).mean(2)
            N = N.reshape(B, T_grid, self.tubelet_size, *N.shape[2:]).mean(2)
        if A.shape[-2:] != D.shape[-2:]:                             # coarsen 정합 (5f)
            fh = D.shape[-2] // A.shape[-2]
            fw = D.shape[-1] // A.shape[-1]
            A = A.repeat_interleave(fh, -2).repeat_interleave(fw, -1)
        return A, D, N

    def forward(self, video: torch.Tensor, gazing_ratio: float):
        if int(video.shape[0]) != 1:
            raise NotImplementedError("v0.8은 batch 1 (slot 배분이 클립 단위 — 7-D 가드)")
        p = self.params
        A, D, N = self.signals(video)
        B, T_grid, Hg, Wg = (int(x) for x in A.shape)
        N_pf = Hg * Wg
        K = round(gazing_ratio * T_grid * N_pf)

        # 점수 — I/P 역할의 연속 분리 (원리 1·2)
        g_t = torch.zeros(T_grid, device=A.device, dtype=A.dtype)
        g_t[0] = 1.0
        g = g_t.view(1, T_grid, 1, 1)
        zA, zD, zN = _z(A), _z(D), _z(N)
        S_p = p.w_d * zD + p.w_n * zN + p.rho * zA                   # P: 변화+놀라움
        S = g * (p.w_a * zA) + (1.0 - g) * S_p                       # I: 외형
        if self.refiner is not None:                                 # Step 11 — coarsen 전
            stack = torch.stack([zA, zD, zN], dim=2)                 # (B,T_grid,3,Hg,Wg)
            dS = self.refiner(stack.reshape(B * T_grid, 3, Hg, Wg))
            S = S + p.sig_gate * dS.reshape(B, T_grid, Hg, Wg)
        S = _soft_coarsen(S, p.alpha)                                # 원리 3 (5g)

        # 배분 — rate-control lite (원리 4), softmax까지 미분 가능
        Dn = (D - D.amin()) / (D.amax() - D.amin() + 1e-6)
        Nn = (N - N.amin()) / (N.amax() - N.amin() + 1e-6)
        E = (p.w_d * Dn + p.w_n * Nn).sum(dim=(-1, -2))              # (B, T_grid) ≥ 0
        logits = (torch.log(E + 1e-4) + p.kappa * g_t) / p.tau.clamp(min=1e-3)
        w = p.gamma * torch.softmax(logits, dim=-1) + (1.0 - p.gamma) / T_grid
        c_soft = w * K                                               # 미분 가능 카운트
        if K < T_grid:
            raise ValueError(f"ratio {gazing_ratio}가 너무 낮음 — 예산 {K} < slot {T_grid}"
                             " (전 slot ≥1 보장 불가)")
        counts = _largest_remainder(c_soft[0].detach(), K, min_val=1, cap=N_pf)

        # slot별 top-k (hard — ST 브리지 지점) → 공유 패킹 꼬리
        mask = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=S.device)
        flat = S.reshape(B, T_grid, N_pf)
        for t in range(T_grid):
            k = int(counts[t])
            _, idx = flat[:, t].topk(k, dim=-1)
            mask[:, t].scatter_(-1, idx, torch.ones_like(idx, dtype=torch.bool))
        sel = _selection_from_mask(S, mask.reshape(B, T_grid, Hg, Wg))
        return sel, {"scores_soft": S, "counts_soft": c_soft}


# =============================================================================
# 학습 배선 — patchstack scripts/train.py:531-581 (dev @ cee52b0)에서 이전, 로직 무변경
# =============================================================================


def st_gate(sel: Selection, aux: dict, rate_grad: bool = True) -> torch.Tensor:
    """hard 선택을 미분 가능하게 잇는 straight-through 게이트 → (B, K), forward 값 1.0.

    토큰을 gather하는 어떤 인코더에도 꽂힌다 (V-JEPA 전용 아님):
        tokens = embed(video).gather(1, sel.keep_index[..., None].expand(-1, -1, D))
        tokens = tokens * st_gate(sel, aux).unsqueeze(-1)   # ← 여기로 grad가 흐른다
    (곱셈 자리·차원을 직접 안 맞추려면 `st_gate_apply(sel, aux, tokens)` 참고.)
    AutoGaze에서는 `sparse_encoder_forward(..., gate=st_gate(sel, aux))`
    (vjepa2_sparse.py:66이 같은 곱을 한다).

    [출처] patchstack scripts/train.py:531-549 — 로직 무변경.

    ⚠️ slot별 softmax × N_pf 다. **전역 softmax × L 아니다** — 전역 형태는 I-slot
    게이트 질량을 ~2.7× 억압해 `w_a` 그래디언트를 굶긴다 (patchstack b910d3c에서
    실측·정정). `_largest_remainder` 독스트링이 한때 말하던
    `counts_st = counts + raw - raw.detach()`는 patchstack에도 구현된 적이 없다 —
    실제 형태는 아래 `w_slot` 비율이다.

    rate_grad=True면 토큰별 자기 slot의 `c_soft/c_soft.detach()`를 곱해 배분 노브
    (kappa/gamma/tau)까지 학습시킨다. `+1e-6`은 필수 — tau가 하한이고 gamma→1이면
    비최대 slot의 c_soft가 0이 되어 0/0 NaN (patchstack 1차 캠페인 step 225 발산 원인).
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

    곱한다는 요구사항은 없앨 수 없다 — hard top-k + 희소 연산(진짜로 토큰을 버려서
    계산량을 줄이는 것)을 동시에 가지려면, gradient가 셀렉터로 돌아갈 유일한 인과
    경로가 "곱해진 값이 loss에 들어간다"뿐이기 때문. 이 함수는 그 지점을 표준화할
    뿐이다.
    """
    return tokens * st_gate(sel, aux, rate_grad).unsqueeze(-1)


# =============================================================================
# 예시 — python modeling_borissal_v08.py {infer,eval,train}
# =============================================================================


class _ToyEncoder(nn.Module):
    """예시 전용 장난감 인코더 — 진짜 지표 아님, 계약 시연용.

    토큰 = 패치 평균 (avg_pool2d, 셀렉터와 같은 tubelet/patch 격자). kept 토큰만 임베드해
    (gate가 있으면 곱하고) 평균 풀로 컨텍스트를 만들고, 그 컨텍스트로 **전체** 패치를
    복원해 미선택 위치의 MSE를 잰다 — patchstack t2의 coverage loss를 축소한 형태.
    `gate.unsqueeze(-1)`을 곱하는 자리가 `sparse_encoder_forward`(vjepa2_sparse.py:66)와
    정확히 같은 자리 — 그게 이 클래스의 존재 이유다.
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
    m = BorissalV08(tubelet_size=2, patch_size=16)
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
    m = BorissalV08(tubelet_size=2, patch_size=16)
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
                          ("borissal v0.8", sel.keep_index)):
            print(f"{name:14s} MSE={float(enc.loss(video, idx)):.4f}")


def _demo_train(frames: int = 8, size: int = 224, ratio: float = 0.25) -> None:
    """셀렉터 → _ToyEncoder(gate) → loss → backward → step → clamp_().
    노브 9개 grad 논제로 + 게이트 forward==1.0을 단언한다 — "이 파일만으로 학습
    가능한가"의 자기검증."""
    torch.manual_seed(0)
    m = BorissalV08(learnable=True, learn_signal=True, tubelet_size=2, patch_size=16)
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
    _ap = argparse.ArgumentParser(description="borissal v0.8 — 선택 생성/측정/학습 예시")
    _ap.add_argument("mode", nargs="?", default="infer", choices=["infer", "eval", "train"])
    _ap.add_argument("--frames", type=int, default=8)
    _ap.add_argument("--size", type=int, default=224)
    _ap.add_argument("--ratio", type=float, default=0.25)
    _args = _ap.parse_args()
    {"infer": _demo_infer, "eval": _demo_eval, "train": _demo_train}[_args.mode](
        frames=_args.frames, size=_args.size, ratio=_args.ratio)
