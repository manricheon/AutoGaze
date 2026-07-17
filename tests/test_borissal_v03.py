"""Borissal v0.3 candidate-bank tests (docs/borissal/v03-design.md)."""
import math

import pytest
import torch

from autogaze.models.borissal.signals_v03 import motion_center_surround, coherence_gate_map, dct_matrix, image_signature


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
