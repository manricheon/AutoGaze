"""Adaptive Checkerboard Refresh (ACR, keyframe_refresh) contract tests."""

import pytest
import torch

from autogaze.models.borissal import Borissal, BorissalConfig


def _video(b=1, t=16, size=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, t, 3, size, size, generator=g)


def _cfg(**kw):
    return BorissalConfig.v0_7(scale=kw.pop("scale", 96), **kw)


def _cube_keep(sel, cfg, b, t, size):
    """keep_mask reshaped to the cube grid (T_grid, Hc, Wc) bool."""
    c = cfg.score_coarsen
    hg = size // cfg.patch_size
    t_grid = t // cfg.tubelet_size
    km = sel.keep_mask[b].view(t_grid, hg, hg)
    return km[:, ::c, ::c]  # cubes are c*c blocks with identical values


def test_budget_unchanged_by_keyframes():
    video = _video()
    base = Borissal(_cfg()).select(video, gazing_ratio=0.25)
    acr = Borissal(_cfg(keyframe_refresh=2)).select(video, gazing_ratio=0.25)
    assert int(base.keep_mask.sum()) == int(acr.keep_mask.sum())


def test_checkerboard_present_and_alternating():
    video = _video()
    cfg = _cfg(keyframe_refresh=2, keyframe_keep=0.5, keyframe_dynamic=False)
    sel = Borissal(cfg).select(video, gazing_ratio=0.25)
    cubes = _cube_keep(sel, cfg, 0, 16, 96)             # (8, Hc, Wc)
    t_grid, hc, wc = cubes.shape
    ii = torch.arange(hc).view(hc, 1).expand(hc, wc)
    jj = torch.arange(wc).view(1, wc).expand(hc, wc)
    par = (ii + jj) % 2
    # static placement: window centers of 2 windows over 8 tubelets -> t=1, 5
    kf_ts, k_kf = [1, 5], (hc * wc) // 2   # K_kf = min(round(0.5*Sc), Sc//2)
    for k, t in enumerate(kf_ts):
        on_parity = int((cubes[t] & (par == (k % 2))).sum())
        assert on_parity == k_kf, f"refresh t={t} parity-{k%2} cells {on_parity} != {k_kf}"
    # complementary parity: the two refresh patterns are disjoint, so their
    # union holds 2*K_kf distinct sites (= the full grid when Sc is even).
    union = int((cubes[kf_ts[0]] | cubes[kf_ts[1]]).sum())
    assert union >= 2 * k_kf


def test_low_budget_clamp_keeps_floors():
    video = _video()
    cfg = _cfg(keyframe_refresh=4, keyframe_keep=0.5)
    sel = Borissal(cfg).select(video, gazing_ratio=0.25)  # infeasible at 0.5 -> clamped
    cfg2 = _cfg()
    base = Borissal(cfg2).select(video, gazing_ratio=0.25)
    assert int(sel.keep_mask.sum()) == int(base.keep_mask.sum())
    cubes = _cube_keep(sel, cfg, 0, 16, 96)
    assert bool(cubes.any(dim=(1, 2)).all()), "every tubelet must keep >=1 cube (floor)"


def test_dynamic_placement_deterministic():
    video = _video(seed=3)
    cfg = _cfg(keyframe_refresh=2, keyframe_dynamic=True)
    a = Borissal(cfg).select(video, gazing_ratio=0.25)
    b = Borissal(cfg).select(video, gazing_ratio=0.25)
    assert torch.equal(a.keep_mask, b.keep_mask)


def test_partial_keep_subset_of_checkerboard():
    video = _video()
    cfg = _cfg(keyframe_refresh=2, keyframe_keep=0.375, keyframe_dynamic=False)
    sel = Borissal(cfg).select(video, gazing_ratio=0.25)
    cubes = _cube_keep(sel, cfg, 0, 16, 96)
    _, hc, wc = cubes.shape
    ii = torch.arange(hc).view(hc, 1).expand(hc, wc)
    jj = torch.arange(wc).view(1, wc).expand(hc, wc)
    par = (ii + jj) % 2
    k_kf = round(0.375 * hc * wc)
    for k, t in enumerate([1, 5]):
        on = int((cubes[t] & (par == (k % 2))).sum())
        assert on >= k_kf, f"refresh t={t}: {on} parity cells < K_kf={k_kf}"


def test_topk_mode_rejects_acr():
    cfg = BorissalConfig(selection_mode="topk", keyframe_refresh=2)
    with pytest.raises(ValueError, match="anchor_novelty mechanism"):
        Borissal(cfg).select(_video(t=4, size=32), gazing_ratio=0.25)
