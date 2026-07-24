"""Borissal v0.3 candidate-bank tests (docs/borissal/v03-design.md)."""
import math

import pytest
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.signals_v03 import motion_center_surround, coherence_gate_map, dct_matrix, image_signature, color_rarity, dog_blob, fused_blend, fusion_multiplier, apply_score_ema


def _v02_uniform(**overrides):
    """semantic-gate convention: v0.2 preset, uniform allocation, no block gate."""
    base = dict(per_frame_allocation="uniform", block_size=1)
    base.update(overrides)
    return BorissalConfig.v0_2(**base)


def _structured_video(B=1, T=8, size=96):
    """Half coherent stripes (texture), half noise, plus a moving red block --
    built so every v0.3 signal knob has something to react to."""
    torch.manual_seed(0)
    v = torch.zeros(B, T, 3, size, size)
    x = torch.arange(size, dtype=torch.float32)
    stripes = ((x // 4) % 2).view(1, 1, 1, 1, size)
    v[:, :, :, : size // 2, :] = stripes.expand(B, T, 3, size // 2, size)
    v[:, :, :, size // 2:, :] = torch.rand(B, T, 3, size // 2, size) * 0.3
    for t in range(T):                                  # test helper: loop OK
        x0 = (12 * t) // 2
        v[:, t, 0, 60:72, x0: x0 + 12] = 1.0
    return v


def _flicker_video(B=1, T=8, size=96):
    """Stimulus for motion_center_surround (v03-design.md §6: a clip designed
    to TRIGGER the knob's mechanism). Static textured background (fixed
    across all T -- no per-pixel motion of its own) plus a GLOBAL per-frame
    brightness flicker (uniform +0.2 on odd frames): the flicker floods the
    frame-diff map identically everywhere, which motion_center_surround's
    relu(D - avgpool(D)) is specifically built to cancel, leaving only the
    one small moving bright block as a surviving local peak -- so the
    selection must shift when the knob is enabled."""
    torch.manual_seed(0)
    bg = torch.rand(1, 1, 3, size, size) * 0.5   # static texture, identical across T
    v = bg.expand(B, T, 3, size, size).clone()
    for t in range(T):                                  # test helper: loop OK
        v[:, t] = v[:, t] + 0.2 * (t % 2)                # global uniform flicker
        x0 = (12 * t) // 2
        v[:, t, 0, 60:72, x0: x0 + 12] = 1.0             # small moving mover
    return v


def _entropy_contrast_video(B=1, T=8, size=96):
    """Stimulus for fusion_entropy (v03-design.md §6). Motion channel
    CONCENTRATED: a static per-pixel texture background (zero motion of its
    own) plus one small fast-sweeping bright block -> a low-entropy motion
    map. Spatial channel DIFFUSE: that same fixed per-pixel random texture
    covers the WHOLE frame -> a high-entropy gradient map everywhere. The
    entropy gate is meant to demote the diffuse spatial channel relative to
    the concentrated motion channel.

    NOTE (see task-8-report.md "fusion_entropy: stop-and-report"): verified
    mathematically that fusion_multiplier's "entropy" mode cannot discriminate
    ANY input bounded to the [0,1] range that _minmax_norm/_minmax_norm_global
    always produce -- even the theoretical maximum-contrast case (a single
    exact 1.0 spike, all else 0) yields a raw (pre-floor) entropy-gate value
    of ~0.007, indistinguishable from a flat map's ~0.00001. This was fixed in
    Task 8 by normalizing entropy against the map's own linear mass (not a
    fixed [0,1] bound), so the knob now discriminates concentrated vs.
    diffuse maps in the actual pipeline. This stimulus is kept as the
    intended, spec-faithful trigger for the fix.
    """
    torch.manual_seed(0)
    texture = torch.rand(1, 1, 3, size, size)            # fixed per-pixel noise, static across T
    v = texture.expand(B, T, 3, size, size).clone()
    for t in range(T):                                  # test helper: loop OK
        x0 = (t * (size - 12)) // (T - 1)                # fast sweep across the full width
        v[:, t, :, 42:54, x0: x0 + 12] = 1.0
    return v


SIGNAL_KNOBS = {
    "motion_center_surround": {"motion_center_surround": True},
    "coherence_gate": {"coherence_gate": True},
    "signature": {"signature_weight": 0.5},
    "color_rarity": {"color_rarity_weight": 0.5},
    "dog_blob": {"dog_blob_weight": 0.5},
    "fusion_peak": {"fusion_norm": "peak"},
    "fusion_entropy": {"fusion_norm": "entropy"},
}

# Per-knob stimulus builders: a clip DESIGNED TO TRIGGER the specific knob's
# mechanism (v03-design.md §6), falling back to the general-purpose
# _structured_video for knobs it already exercises correctly.
SIGNAL_STIMULI = {
    "motion_center_surround": _flicker_video,
    "fusion_entropy": _entropy_contrast_video,
}


def test_all_knobs_off_takes_legacy_blend_path():
    """knobs-off의 score는 정확히 v0.2의 2채널 블렌드여야 한다 (비트 동일 보장)."""
    video = _structured_video()
    cfg = _v02_uniform(scale=96, score_norm_blend=1.0)   # 글로벌 블렌드 없이 화이트박스 검증
    sel, inter = Borissal(cfg).select_with_intermediates(video, gazing_ratio=0.25)
    w = inter["motion_weight_used"].view(-1, 1, 1, 1)
    legacy = w * inter["motion_norm"] + (1 - w) * inter["spatial_norm"]
    assert torch.equal(inter["score"], legacy)


@pytest.mark.parametrize("name", sorted(SIGNAL_KNOBS))
def test_each_signal_knob_changes_selection(name):
    video = SIGNAL_STIMULI.get(name, _structured_video)()
    base = Borissal(_v02_uniform(scale=96)).select(video, gazing_ratio=0.25)
    knob = Borissal(_v02_uniform(scale=96, **SIGNAL_KNOBS[name])).select(
        video, gazing_ratio=0.25)
    assert not torch.equal(base.keep_mask, knob.keep_mask), name
    # 계약 불변식: 예산은 그대로
    assert torch.equal(base.per_frame_keep, knob.per_frame_keep)


def test_motion_cs_suppresses_uniform_pan_keeps_local_mover():
    B, T, H, W = 1, 4, 24, 24
    pan = torch.full((B, T, H, W), 0.8)          # 균일 diff 필드 = 카메라 팬
    local = torch.zeros(B, T, H, W)
    local[:, :, 10:13, 10:13] = 0.8              # 같은 크기의 국소 무버
    out_pan = motion_center_surround(pan, kernel=9)
    out_local = motion_center_surround(local, kernel=9)
    assert out_pan.abs().max() < 1e-6            # 평평한 필드는 완전 상쇄
    assert out_local[0, 0, 11, 11] > 0.5         # 무버는 국소 피크로 생존
    assert (out_local >= 0).all()                # relu 반환 (음수 없음)


@pytest.mark.parametrize("downsample", [1, 4])
def test_coherence_gate_kills_coherent_gradients_spares_isotropic(downsample):
    B, T, H, W = 1, 1, 32, 32
    # 완벽히 일관된 그라디언트 (수직 엣지/격자무늬): coherence ~1 -> 게이트 ~0
    dx = torch.ones(B, T, H, W)
    dy = torch.zeros(B, T, H, W)
    g_coh = coherence_gate_map(dx, dy, kernel=5, gamma=1.0, eps=1e-6, downsample=downsample)
    assert g_coh.max() < 0.05
    # 등방성 랜덤 그라디언트 (다방향 미세구조): coherence 낮음 -> 게이트 큼
    torch.manual_seed(0)
    dx = torch.randn(B, T, H, W)
    dy = torch.randn(B, T, H, W)
    g_iso = coherence_gate_map(dx, dy, kernel=5, gamma=1.0, eps=1e-6, downsample=downsample)
    assert g_iso.mean() > 0.5
    assert (g_iso >= 0).all() and (g_iso <= 1).all()


def test_dct_matrix_is_orthonormal():
    D = dct_matrix(24, torch.device("cpu"), torch.float32)
    assert torch.allclose(D @ D.t(), torch.eye(24), atol=1e-5)


def test_image_signature_fires_on_sparse_foreground():
    B, T, n = 1, 1, 24
    # (a) 상수 배경 + 단일 스파이크: 스파이크 위치가 전역 최대여야 한다
    img = torch.zeros(B, T, n, n)
    img[:, :, 12, 12] = 1.0
    sal = image_signature(img)
    spike = sal[0, 0, 12, 12]
    background = sal[0, 0, :, :8].mean()
    assert spike > 5 * background
    # (b) 주기적 격자무늬 배경 + 희소 blob: blob 영역 평균 > 배경 평균
    xx = torch.arange(n, dtype=torch.float32)
    stripes = 0.5 + 0.5 * torch.cos(2 * math.pi * xx * 6 / n)   # 스펙트럼 희소 배경
    img2 = stripes.view(1, 1, 1, n).expand(B, T, n, n).clone()
    img2[:, :, 10:13, 10:13] = 2.0                              # 공간적으로 희소한 전경
    sal2 = image_signature(img2)
    blob = sal2[:, :, 10:13, 10:13].mean()
    background = sal2[:, :, :, :8].mean()
    assert blob > 1.5 * background


def test_color_rarity_fires_on_rare_color_interior():
    B, T, H, W = 1, 2, 24, 24
    rgb = torch.zeros(B, T, 3, H, W)
    rgb[:, :, 1] = 0.6                       # 지배적 초록 배경
    rgb[:, :, 0, 8:14, 8:14] = 0.9           # 희소한 빨강 사각형
    rgb[:, :, 1, 8:14, 8:14] = 0.1
    sal = color_rarity(rgb, num_bins_per_axis=3, sigma=0.15, eps=1e-6)
    inside = sal[:, :, 10:12, 10:12].mean()  # 사각형 내부 (경계 아님)
    outside = sal[:, :, :, :6].mean()        # 배경
    assert inside > 1.5 * outside            # 내부가 균일하게 발화 (엣지 편향 없음)
    # 내부와 경계가 같은 색 -> 같은 희소성 (interior filling의 증명)
    edge = sal[:, :, 8, 8:14].mean()
    assert torch.allclose(inside, edge, rtol=0.05)


def test_dog_blob_fires_on_flat_interior_where_gradient_is_zero():
    B, T, n = 1, 1, 24
    img = torch.zeros(B, T, n, n)
    img[:, :, 9:15, 9:15] = 1.0              # 6x6 평탄한 사각형
    # 중심 (12,12)의 로컬 그라디언트는 정확히 0 (평탄 영역)
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    assert dy[0, 0, 11:13, 11:13].abs().max() == 0
    assert dx[0, 0, 11:13, 11:13].abs().max() == 0
    # DoG blob은 그 내부에서 발화한다 -- 그라디언트가 못 하는 일
    blob = dog_blob(img)
    assert blob[0, 0, 12, 12] > 0.05


def test_peak_promotion_ranks_single_peak_over_texture():
    B, T, H, W = 1, 1, 24, 24
    single = torch.zeros(B, T, H, W)
    single[:, :, 12, 12] = 1.0                       # 피크 1개짜리 맵
    torch.manual_seed(0)
    many = torch.rand(B, T, H, W)                    # 비슷한 피크 다수 = 텍스처성 맵
    m_single = fusion_multiplier(single, "peak", 0.3, 1e-6)
    m_many = fusion_multiplier(many, "peak", 0.3, 1e-6)
    assert m_single.shape == (B, T)
    assert m_single[0, 0] > m_many[0, 0]


def test_entropy_gate_hits_floor_on_flat_map_and_is_bounded():
    flat = torch.full((1, 1, 24, 24), 0.5)           # 균일 맵 -> 최대 엔트로피
    peaked = torch.zeros(1, 1, 24, 24)
    peaked[:, :, 12, 12] = 8.0
    m_flat = fusion_multiplier(flat, "entropy", 0.3, 1e-6)
    m_peak = fusion_multiplier(peaked, "entropy", 0.3, 1e-6)
    assert abs(m_flat[0, 0].item() - 0.3) < 1e-3     # 하한에 클램프
    assert m_flat[0, 0] < m_peak[0, 0] <= 1.0


def test_fused_blend_two_equal_channels_is_identity_weighted_average():
    torch.manual_seed(0)
    a = torch.rand(2, 4, 6, 6)
    b = torch.rand(2, 4, 6, 6)
    out = fused_blend([(0.5, a), (0.5, b)], "none", 0.3, 1e-6)
    assert torch.allclose(out, (0.5 * a + 0.5 * b) / (1.0 + 1e-6), atol=1e-6)
    # 텐서 가중치 (motion_weight="auto"의 (B,1,1,1) 케이스)도 브로드캐스트된다
    w = torch.full((2, 1, 1, 1), 0.3)
    out_t = fused_blend([(w, a), (1 - w, b)], "none", 0.3, 1e-6)
    assert out_t.shape == a.shape


def test_ema_matches_sequential_recursion():
    torch.manual_seed(0)
    S = torch.rand(2, 8, 6, 6)
    alpha = 0.6
    out = apply_score_ema(S, alpha, None)
    ref = [S[:, 0]]                                 # S_bar_0 = S_0
    for t in range(1, 8):
        ref.append(alpha * ref[-1] + (1 - alpha) * S[:, t])
    assert torch.allclose(out, torch.stack(ref, dim=1), atol=1e-5)


def test_ema_streaming_split_equals_full_run():
    torch.manual_seed(1)
    S = torch.rand(2, 8, 6, 6)
    alpha = 0.6
    full = apply_score_ema(S, alpha, None)
    first = apply_score_ema(S[:, :4], alpha, None)
    second = apply_score_ema(S[:, 4:], alpha, first[:, -1])   # 상태 이월
    assert torch.allclose(torch.cat([first, second], dim=1), full, atol=1e-5)


def test_hysteresis_increases_cross_tubelet_selection_overlap():
    torch.manual_seed(0)
    video = torch.rand(1, 8, 3, 96, 96)        # 노이즈 점수 -> 불안정한 기본 선택

    def overlap(sel):
        T_grid = int(sel.grid_thw[0, 0].item())
        m = sel.keep_mask.reshape(1, T_grid, -1).float()
        inter = (m[:, 1:] * m[:, :-1]).sum()
        return (inter / m[:, 1:].sum()).item()

    base = Borissal(_v02_uniform(scale=96)).select(video, gazing_ratio=0.25)
    hyst = Borissal(_v02_uniform(scale=96, select_hysteresis_eps=0.2)).select(
        video, gazing_ratio=0.25)
    assert overlap(hyst) > overlap(base)
    assert torch.equal(base.per_frame_keep, hyst.per_frame_keep)  # 예산 불변


def test_temporal_state_round_trip():
    video = _structured_video()
    model = Borissal(_v02_uniform(
        scale=96, score_ema_alpha=0.5, select_hysteresis_eps=0.1))
    sel, inter = model.select_with_intermediates(video, gazing_ratio=0.25)
    st = inter["temporal_state"]
    assert st["ema"].shape == (1, 6, 6)        # 96/16 그리드
    assert st["prev_keep"].shape == (1, 36) and st["prev_keep"].dtype == torch.bool
    # 상태를 넣으면 첫 tubelet 선택이 달라진다 (이월 효과)
    sel2 = model.select(video, gazing_ratio=0.25, temporal_state=st)
    m1 = sel.keep_mask.reshape(1, 4, 36)
    m2 = sel2.keep_mask.reshape(1, 4, 36)
    assert not torch.equal(m1[:, 0], m2[:, 0])


def test_auto_weight_composes_with_fusion():
    video = _structured_video(B=2)
    cfg = _v02_uniform(scale=96, motion_weight="auto", fusion_norm="peak",
                       signature_weight=0.5)
    sel = Borissal(cfg).select(video, gazing_ratio=0.25)
    assert torch.isfinite(sel.scores).all()
    assert sel.per_frame_keep.eq(sel.per_frame_keep[0, 0]).all()


def test_temporal_knobs_off_ignore_state_and_match_base():
    video = _structured_video()
    model = Borissal(_v02_uniform(scale=96))
    a = model.select(video, gazing_ratio=0.25)
    b = model.select(video, gazing_ratio=0.25, temporal_state=None)
    assert torch.equal(a.keep_mask, b.keep_mask)


ALL_KNOBS = dict(
    motion_center_surround=True, coherence_gate=True, signature_weight=0.5,
    color_rarity_weight=0.5, dog_blob_weight=0.5, fusion_norm="entropy",
    score_ema_alpha=0.5, select_hysteresis_eps=0.05,
)


def test_all_knobs_on_keeps_selection_contract():
    video = _structured_video()
    for cfg in (_v02_uniform(scale=96, **ALL_KNOBS),          # uniform 변형
                BorissalConfig.v0_2(scale=96, **ALL_KNOBS)):  # 풀 프리셋 (global+block)
        sel = Borissal(cfg).select(video, gazing_ratio=0.25)
        idx = sel.keep_index
        valid = idx[:, 1:] >= 0
        assert ((idx[:, 1:] > idx[:, :-1]) | ~valid).all()    # 오름차순 규약
        assert sel.num_keep.sum() == sel.keep_mask.sum()      # 예산 정합
        assert sel.per_frame_keep.sum(-1).eq(sel.num_keep).all()


def test_all_knobs_on_traces_and_matches_eager_on_fresh_input():
    cfg = _v02_uniform(scale=96, **ALL_KNOBS)
    model = Borissal(cfg).eval()

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, v):
            s = self.m.select(v, gazing_ratio=0.25)
            return s.keep_index, s.keep_mask

    video = _structured_video()
    with torch.no_grad():
        traced = torch.jit.trace(_Wrap(model), video, check_trace=False)
        # Genuinely independent input (not a perturbation of the trace input):
        # a small perturbation of `video` can leave every mask decision
        # unchanged, in which case the traced-vs-eager equality below would
        # pass even with baked-in constants. Self-check this before trusting
        # the comparison.
        g = torch.Generator().manual_seed(123)
        fresh = torch.rand(1, 8, 3, 96, 96, generator=g)
        eager_base = model.select(video, gazing_ratio=0.25)
        eager = model.select(fresh, gazing_ratio=0.25)
        assert not torch.equal(eager.keep_mask, eager_base.keep_mask)
        t_idx, t_mask = traced(fresh)
    assert torch.equal(t_idx, eager.keep_index)      # 상수 고착 없음
    assert torch.equal(t_mask, eager.keep_mask)


def test_v0_3_preset_contract():
    cfg = BorissalConfig.v0_3(scale=96)
    assert cfg.fusion_norm == "peak" and cfg.coherence_gate
    assert cfg.dog_blob_weight > 0 and cfg.coherence_downsample == 4
    assert cfg.block_gate_source == "pool"
    assert cfg.spatial_diff == "frame" and cfg.spatial_agg == "max"
    sel = Borissal(cfg).select(_structured_video(), gazing_ratio=0.25)
    idx = sel.keep_index
    valid = idx[:, 1:] >= 0
    assert ((idx[:, 1:] > idx[:, :-1]) | ~valid).all()   # ascending contract
    assert sel.per_frame_keep.sum(-1).eq(sel.num_keep).all()


def test_max_cap_mult1_equals_uniform_share():
    """cap = 1x uniform share exposes exactly K_total candidates -> every
    tubelet keeps its uniform share (global alloc degenerates to uniform)."""
    cfg = BorissalConfig.v0_3(scale=96, block_size=1, per_frame_allocation="global",
                              max_keep_per_frame_mult=1.0)
    sel = Borissal(cfg).select(_structured_video(), gazing_ratio=0.25)
    assert sel.per_frame_keep.eq(sel.per_frame_keep[0, 0]).all()
    assert int(sel.num_keep[0]) == round(0.25 * 4 * 36)


def test_max_cap_bounds_concentration_keeps_budget():
    cfg = BorissalConfig.v0_3(scale=96, block_size=1, per_frame_allocation="global",
                              max_keep_per_frame_mult=1.5)
    base = BorissalConfig.v0_3(scale=96, block_size=1, per_frame_allocation="global")
    K = round(0.25 * 4 * 36)
    cap = round(1.5 * K / 4)
    sel = Borissal(cfg).select(_structured_video(), gazing_ratio=0.25)
    ref = Borissal(base).select(_structured_video(), gazing_ratio=0.25)
    assert int(sel.num_keep[0]) == K                    # exact budget kept
    assert sel.per_frame_keep.max() <= cap              # cap enforced
    m = max(1, round(0.25 * K / 4))
    assert sel.per_frame_keep.min() >= m                # floor still holds
    assert int(ref.num_keep[0]) == K


def test_block_gate_pool_mode_contract_and_changes_selection():
    video = _structured_video()
    rec = Borissal(BorissalConfig.v0_3(scale=96, block_gate_source="recompute")).select(
        video, gazing_ratio=0.25)
    pool = Borissal(BorissalConfig.v0_3(scale=96)).select(   # preset default = pool
        video, gazing_ratio=0.25)
    assert not torch.equal(rec.keep_mask, pool.keep_mask)
    assert torch.equal(rec.num_keep, pool.num_keep)     # exact budget invariant
    idx = pool.keep_index
    valid = idx[:, 1:] >= 0
    assert ((idx[:, 1:] > idx[:, :-1]) | ~valid).all()  # ascending contract


def test_spatial_frame_max_preserves_single_frame_detail():
    """Two COMPETING texture bands: band A is textured in every frame, band
    B only in one frame of each tubelet pair. The tubelet-mean spatial
    signal halves band B (motion blur), so A outranks B; frame-level max
    sees both at full sharpness, shifting budget toward B -- the selection
    must differ. (A single salient region would NOT discriminate the modes:
    halved magnitude doesn't change the ranking when nothing competes.)"""
    torch.manual_seed(0)
    B, T, size = 1, 8, 96
    v = torch.full((B, T, 3, size, size), 0.5)
    band_a = torch.rand(3, 12, size)
    band_b = torch.rand(3, 12, size) * 2.0 - 0.5   # 2x contrast: sharp-in-one-
    # frame B must OUTRANK steady A under frame-max, but lose to it once the
    # tubelet mean halves its gradient (0.5x contrast-equivalent)
    for t in range(T):                                  # test helper: loop OK
        v[:, t, :, 20:32, :] = band_a                    # steady texture
        if t % 2 == 0:
            v[:, t, :, 60:72, :] = band_b                # blinks: 1 of 2 frames
    # motion_weight=0 isolates the SPATIAL channel: band B blinking also
    # fires frame-diff motion identically in both modes, which would
    # otherwise saturate the budget inside band B regardless of spatial.
    tub_cfg = _v02_uniform(scale=96, motion_weight=0.0)
    frm_cfg = _v02_uniform(scale=96, motion_weight=0.0,
                           spatial_diff="frame", spatial_agg="max")
    sel_t = Borissal(tub_cfg).select(v, gazing_ratio=0.25)
    sel_f = Borissal(frm_cfg).select(v, gazing_ratio=0.25)
    assert not torch.equal(sel_t.keep_mask, sel_f.keep_mask)
    assert torch.equal(sel_t.num_keep, sel_f.num_keep)
    # budget share of the blinking band (grid rows 3..4) must not DROP
    def band_b_count(sel):
        m = sel.keep_mask.reshape(1, 4, 6, 6)
        return int(m[:, :, 3:5, :].sum())
    assert band_b_count(sel_f) >= band_b_count(sel_t)


def test_per_frame_counts_override_respected_and_contract_kept():
    """E5: caller-supplied per-tubelet counts replace the allocation step only."""
    video = _structured_video()
    counts = torch.tensor([4, 20, 8, 4])                 # sums to 36 = 0.25 * 144
    sel = Borissal(BorissalConfig.v0_3(scale=96)).select(
        video, gazing_ratio=0.25, per_frame_counts=counts)
    assert torch.equal(sel.per_frame_keep[0], counts)
    assert int(sel.num_keep[0]) == 36
    idx = sel.keep_index
    valid = idx[:, 1:] >= 0
    assert ((idx[:, 1:] > idx[:, :-1]) | ~valid).all()   # ascending contract
    # spread is incompatible with an explicit counts override
    with pytest.raises(ValueError):
        Borissal(BorissalConfig.v0_3(scale=96)).select(
            video, gazing_ratio=0.25, per_frame_counts=counts, spread_fraction=0.25)


def test_v0_4_identical_to_v0_3_at_16_frames():
    """v0.4 (auto motion stride) must be bit-identical to v0.3 at the 16-frame
    reference (auto stride = round(16/16) = 1)."""
    video = torch.rand(1, 16, 3, 96, 96)
    s3 = Borissal(BorissalConfig.v0_3(scale=96)).select(video, gazing_ratio=0.25)
    s4 = Borissal(BorissalConfig.v0_4(scale=96)).select(video, gazing_ratio=0.25)
    assert torch.equal(s3.keep_mask, s4.keep_mask)


def test_v0_4_changes_selection_at_32_frames():
    """At 32 frames auto stride = 2, so v0.4 differs from v0.3 (which keeps
    stride 1) -- the frame-rate-aware motion fix is active."""
    video = torch.rand(1, 32, 3, 96, 96)
    s3 = Borissal(BorissalConfig.v0_3(scale=96)).select(video, gazing_ratio=0.25)
    s4 = Borissal(BorissalConfig.v0_4(scale=96)).select(video, gazing_ratio=0.25)
    assert not torch.equal(s3.keep_mask, s4.keep_mask)
    assert torch.equal(s3.per_frame_keep, s4.per_frame_keep)   # budget unchanged


def test_motion_stride_recovers_signal_at_32f():
    """auto stride at 32f differences frames 2 apart, restoring motion
    magnitude to ~the stride-1 16f level (a moving block over 32 frames)."""
    T, size = 32, 96
    v = torch.zeros(1, T, 3, size, size)
    for t in range(T):                                    # test helper: loop OK
        x0 = (t * 2) % (size - 12)
        v[:, t, :, 40:52, x0:x0 + 12] = 1.0
    m = Borissal(BorissalConfig.v0_4(scale=96))
    _, inter = m.select_with_intermediates(v, gazing_ratio=0.25)
    m3 = Borissal(BorissalConfig.v0_3(scale=96))
    _, inter3 = m3.select_with_intermediates(v, gazing_ratio=0.25)
    # wider stride sees larger displacement -> stronger pre-norm motion energy
    assert inter["motion_norm"].sum() != inter3["motion_norm"].sum()


def _cube_coherence(sel):
    """4-neighbor selected-fraction of selected patches (1 = perfect cubes)."""
    import torch.nn.functional as F
    Tg = int(sel.grid_thw[0, 0]); Hg = int(sel.grid_thw[0, 1]); Wg = int(sel.grid_thw[0, 2])
    k = sel.keep_mask[0].reshape(Tg, Hg, Wg).float()
    pad = F.pad(k, (1, 1, 1, 1))
    n = (pad[:, :-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, :-2] + pad[:, 1:-1, 2:]) / 4
    return n[k > 0.5].mean().item()


def test_score_coarsen_default_is_noop():
    """score_coarsen=1 (default) leaves selection unchanged."""
    video = _structured_video()
    a = Borissal(_v02_uniform(scale=96)).select(video, gazing_ratio=0.25)
    b = Borissal(_v02_uniform(scale=96, score_coarsen=1)).select(video, gazing_ratio=0.25)
    assert torch.equal(a.keep_mask, b.keep_mask)


def test_v0_5_makes_selection_more_cube_coherent():
    """v0.5 (score_coarsen=2) yields denser cubes than plain v0.3 fine top-k."""
    video = _structured_video()
    scattered = Borissal(BorissalConfig.v0_3(scale=96, block_size=1)).select(
        video, gazing_ratio=0.25)
    cubes = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    assert _cube_coherence(cubes) > _cube_coherence(scattered)
    # contract intact
    idx = cubes.keep_index
    valid = idx[:, 1:] >= 0
    assert ((idx[:, 1:] > idx[:, :-1]) | ~valid).all()
    assert cubes.per_frame_keep.sum(-1).eq(cubes.num_keep).all()


def test_v0_5_motion_weight_knob_shifts_selection():
    """motion_weight is tunable on v0.5 (the intended tuning axis)."""
    video = _structured_video()
    hi = Borissal(BorissalConfig.v0_5(scale=96, motion_weight=0.5)).select(video, gazing_ratio=0.25)
    lo = Borissal(BorissalConfig.v0_5(scale=96, motion_weight=0.0)).select(video, gazing_ratio=0.25)
    assert not torch.equal(hi.keep_mask, lo.keep_mask)
    assert torch.equal(hi.per_frame_keep, lo.per_frame_keep)   # budget unchanged


def test_coherence_at_grid_matches_pixel_closely_and_traces():
    """v0.5's grid-resolution coherence gate selects near-identically to the
    pixel-res gate (regional statistic) and stays trace-safe."""
    video = _structured_video()
    px = Borissal(BorissalConfig.v0_5(scale=96, coherence_at_grid=False)).select(
        video, gazing_ratio=0.25)
    gr = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    iou = (px.keep_mask & gr.keep_mask).sum() / (px.keep_mask | gr.keep_mask).sum()
    assert iou > 0.4   # similar (real 384 clips: 0.92; tiny synthetic grid is coarse). random ~0.14
    assert torch.equal(gr.per_frame_keep, px.per_frame_keep)
    class _W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, v): r = s.m.select(v, gazing_ratio=0.25); return r.keep_index
    tr = torch.jit.trace(_W(Borissal(BorissalConfig.v0_5(scale=96)).eval()), video, check_trace=False)
    assert tr is not None


# ---------------------------------------------------------------------------
# v0.6: Laplacian texture gate + static appearance guard (saliency-v3.1 stage 4/6)
# ---------------------------------------------------------------------------
from autogaze.models.borissal.signals_v03 import (  # noqa: E402
    laplacian_energy, laplacian_texture_gate, static_appearance_guard,
)


def test_laplacian_energy_zero_on_flat_high_on_texture():
    flat = torch.full((1, 2, 8, 8), 0.5)
    assert laplacian_energy(flat).abs().max() < 1e-6
    # checkerboard = densest high-frequency texture -> large laplacian energy
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    chec = ((yy + xx) % 2).float().view(1, 1, 8, 8).expand(1, 2, 8, 8)
    assert laplacian_energy(chec).mean() > laplacian_energy(flat).mean() + 1.0
    assert laplacian_energy(flat).shape == flat.shape


def test_laplacian_texture_gate_suppresses_high_ratio():
    # region A: checkerboard texture, tiny motion -> high R -> suppressed (~0)
    yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    chec = ((yy + xx) % 2).float().view(1, 1, 8, 8) * 0.05 + 0.01  # small motion, high texture
    gate_tex = laplacian_texture_gate(chec, r0=1.0, tau=0.5, eps=1e-6)
    # region B: smooth ramp with real motion magnitude, low texture -> pass (~1)
    ramp = torch.linspace(0.2, 0.8, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8).contiguous()
    gate_smooth = laplacian_texture_gate(ramp, r0=1.0, tau=0.5, eps=1e-6)
    assert (gate_tex >= 0).all() and (gate_tex <= 1).all()
    assert gate_tex.mean() < gate_smooth.mean(), "dense-texture/low-motion must be suppressed more"


def test_static_appearance_guard_fires_only_on_static_tubelets():
    # luma with a clear edge (informative static structure)
    luma = torch.zeros(1, 2, 8, 8)
    luma[:, :, :, 4:] = 1.0  # a vertical edge -> nonzero laplacian
    # tubelet 0 static (motion ~0), tubelet 1 fully moving (motion ~1)
    motion_norm = torch.zeros(1, 2, 8, 8)
    motion_norm[:, 1] = 1.0
    guard = static_appearance_guard(luma, motion_norm, thresh=0.05, tau=0.02)
    assert guard[:, 0].abs().sum() > 0, "static tubelet keeps its edge energy"
    assert guard[:, 1].abs().max() < 1e-4, "moving tubelet is untouched (s_t ~ 0)"


def test_v0_6_default_enables_all_three_knobs():
    """v0.6 DEFAULT = all three saliency-v3.1 knobs ON (2026-07-24 decision,
    matching saliency-v3.1's own config)."""
    cfg = BorissalConfig.v0_6(scale=96)
    assert cfg.static_guard and cfg.laplacian_gate and cfg.center_bias > 0.0


def test_v0_6_all_knobs_off_recovers_v0_5():
    """Explicitly disabling every v0.6 knob must recover exact v0.5 behavior."""
    video = _structured_video()
    s5 = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    s6off = Borissal(BorissalConfig.v0_6(
        scale=96, static_guard=False, laplacian_gate=False, center_bias=0.0
    )).select(video, gazing_ratio=0.25)
    assert torch.equal(s5.scores, s6off.scores)
    assert torch.equal(s5.keep_mask, s6off.keep_mask)


def test_v0_6_static_guard_changes_scores_on_static_clip():
    # a static clip (all frames identical, structured) -> motion ~0 everywhere
    # -> the static guard fires and must move the scores vs v0.5.
    frame = _structured_video()[:, :1]              # (1,1,3,96,96)
    video = frame.expand(1, 16, 3, 96, 96).contiguous()
    base = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    guarded = Borissal(BorissalConfig.v0_6(scale=96, static_guard=True, laplacian_gate=False,
                                           center_bias=0.0, static_guard_weight=1.0)
                       ).select(video, gazing_ratio=0.25)
    assert not torch.equal(base.scores, guarded.scores)


def test_v0_6_laplacian_gate_changes_scores():
    video = _structured_video()
    base = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    gated = Borissal(BorissalConfig.v0_6(scale=96, static_guard=False, laplacian_gate=True,
                                         center_bias=0.0)).select(video, gazing_ratio=0.25)
    assert not torch.equal(base.scores, gated.scores)
    # gate is a (0,1) multiplier -> it can only hold or lower scores
    assert (gated.scores <= base.scores + 1e-6).all()


def test_v0_6_traces_with_both_knobs_on():
    video = _structured_video()
    cfg = BorissalConfig.v0_6(scale=96, static_guard=True, laplacian_gate=True, center_bias=0.2)
    class _W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, v): return s.m.select(v, gazing_ratio=0.25).keep_index
    tr = torch.jit.trace(_W(Borissal(cfg).eval()), video, check_trace=False)
    assert tr is not None


# ---------------------------------------------------------------------------
# v0.6: mechanical-GOP keyframe prior (periodic + soft scene-cut, pixel-only)
# ---------------------------------------------------------------------------
from autogaze.models.borissal.signals_v03 import keyframe_prior  # noqa: E402


def test_keyframe_prior_periodic_fires_on_gop_tubelets():
    # 8 tubelets, gop=8 frames, tubelet_size=2 -> keyframe every 4 tubelets: {0,4}
    luma = torch.rand(1, 8, 6, 6)
    kf = keyframe_prior(luma, gop=8, tubelet_size=2, scene_thresh=99.0, scene_tau=0.5, eps=1e-6)
    energy = kf.reshape(1, 8, -1).sum(-1)  # per-tubelet total guard energy
    # scene_thresh huge -> scene term ~0, so only periodic tubelets carry energy
    fired = (energy[0] > 1e-6).nonzero().flatten().tolist()
    assert fired == [0, 4], f"periodic keyframes should be tubelets 0 and 4, got {fired}"


def test_keyframe_prior_responds_to_scene_cut():
    # tubelet 3 is a hard scene cut (totally different content) off the GOP grid
    luma = torch.zeros(1, 8, 6, 6)
    luma[:, :3] = 0.2
    luma[:, 3:] = 0.9              # abrupt jump at tubelet 3
    kf = keyframe_prior(luma, gop=8, tubelet_size=2, scene_thresh=2.0, scene_tau=0.3, eps=1e-6)
    # even though luma is flat within each side (laplacian ~0), the scene weight
    # is nonzero at t=3; verify the per-tubelet keyframe weight, not the energy.
    # reconstruct weight by probing with a textured luma:
    yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    tex = ((yy + xx) % 2).float().view(1, 1, 6, 6).expand(1, 8, 6, 6).clone()
    lum2 = tex * 0.2
    lum2[:, 3:] = tex[:, 3:] * 0.9 + 0.5   # scene change at t=3, still textured
    kf2 = keyframe_prior(lum2, gop=8, tubelet_size=2, scene_thresh=2.0, scene_tau=0.3, eps=1e-6)
    e = kf2.reshape(1, 8, -1).sum(-1)[0]
    assert e[3] > e[2], "scene-cut tubelet must get more keyframe energy than its static neighbor"


def test_keyframe_prior_trace_safe_shape():
    luma = torch.rand(2, 8, 12, 12)
    kf = keyframe_prior(luma, gop=16, tubelet_size=2, scene_thresh=2.0, scene_tau=0.5, eps=1e-6)
    assert kf.shape == luma.shape


def test_v0_6_keyframe_prior_reallocates_to_keyframes_and_traces():
    video = _structured_video()
    base = Borissal(BorissalConfig.v0_5(scale=96)).select(video, gazing_ratio=0.25)
    kf = Borissal(BorissalConfig.v0_6(scale=96, static_guard=False, laplacian_gate=False,
                                      center_bias=0.0, keyframe_prior=True, keyframe_gop=8)
                  ).select(video, gazing_ratio=0.25)
    assert not torch.equal(base.scores, kf.scores)
    # allocation must actually move tokens toward the periodic keyframe tubelets
    # (base is uniform; keyframe run must NOT be uniform, and same total budget)
    assert torch.equal(base.per_frame_keep, torch.full_like(base.per_frame_keep, base.per_frame_keep[0, 0]))
    assert not torch.equal(kf.per_frame_keep, base.per_frame_keep), "keyframe tubelets must get more tokens"
    assert int(kf.num_keep[0]) == int(base.num_keep[0]), "total budget unchanged"
    # keyframe tubelet 0 (periodic) should hold more than a non-keyframe tubelet
    assert kf.per_frame_keep[0, 0] > kf.per_frame_keep[0, 1]
    # keyframe_prior is OPT-IN, not part of the v0.6 all-on default
    assert BorissalConfig.v0_6(scale=96).keyframe_prior is False
    class _W(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, v): return s.m.select(v, gazing_ratio=0.25).keep_index
    cfg = BorissalConfig.v0_6(scale=96, keyframe_prior=True)
    tr = torch.jit.trace(_W(Borissal(cfg).eval()), video, check_trace=False)
    assert tr is not None
