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
    of ~0.007, indistinguishable from a flat map's ~0.00001. Since
    _saliency_scores always normalizes motion_n/spatial_n to [0,1] BEFORE
    calling fused_blend, "entropy" mode is a structural no-op in the actual
    pipeline for any clip -- not a stimulus-design problem. This stimulus is
    kept as the intended, spec-faithful trigger (and will exercise the
    knob correctly once/if the primitive or its call site is revisited); the
    corresponding parametrized case is expected to stay red until that
    upstream issue is resolved (out of scope here: test-stimulus only).
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


def test_coherence_gate_kills_coherent_gradients_spares_isotropic():
    B, T, H, W = 1, 1, 32, 32
    # 완벽히 일관된 그라디언트 (수직 엣지/격자무늬): coherence ~1 -> 게이트 ~0
    dx = torch.ones(B, T, H, W)
    dy = torch.zeros(B, T, H, W)
    g_coh = coherence_gate_map(dx, dy, kernel=5, gamma=1.0, eps=1e-6)
    assert g_coh.max() < 0.05
    # 등방성 랜덤 그라디언트 (다방향 미세구조): coherence 낮음 -> 게이트 큼
    torch.manual_seed(0)
    dx = torch.randn(B, T, H, W)
    dy = torch.randn(B, T, H, W)
    g_iso = coherence_gate_map(dx, dy, kernel=5, gamma=1.0, eps=1e-6)
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
