"""Borissal v0.3 candidate-bank tests (docs/borissal/v03-design.md)."""
import math

import pytest
import torch

from autogaze.models.borissal.signals_v03 import motion_center_surround


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
