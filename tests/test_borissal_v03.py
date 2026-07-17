"""Borissal v0.3 candidate-bank tests (docs/borissal/v03-design.md)."""
import math

import pytest
import torch

from autogaze.models.borissal.signals_v03 import motion_center_surround, coherence_gate_map


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
