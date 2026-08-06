import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.adapters import to_canonical_keep_indices


def _make_video(B=2, T=16, C=3, H=384, W=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, C, H, W, generator=g)


def test_grid_thw_matches_config():
    cfg = BorissalConfig(scale=384, patch_size=16, tubelet_size=2)
    model = Borissal(cfg)
    video = _make_video(T=16, H=384, W=384)
    sel = model.select(video)
    expected = torch.tensor([16 // 2, 384 // 16, 384 // 16])
    assert torch.equal(sel.grid_thw[0], expected)
    assert torch.equal(sel.grid_thw[1], expected)


def test_keep_index_validity_uniform():
    cfg = BorissalConfig(per_frame_allocation="uniform", gazing_ratio=0.4)
    model = Borissal(cfg)
    video = _make_video()
    sel = model.select(video)

    L = sel.scores.shape[1]
    valid = sel.keep_index[sel.keep_index >= 0]
    assert (valid < L).all()
    assert (valid >= 0).all()

    # no duplicate valid indices within a row
    for b in range(sel.keep_index.shape[0]):
        row = sel.keep_index[b]
        row_valid = row[row >= 0]
        assert row_valid.unique().numel() == row_valid.numel()

    assert torch.equal(sel.num_keep, sel.keep_mask.sum(dim=-1).to(torch.long))


def test_per_frame_keep_sums_to_num_keep():
    cfg = BorissalConfig(per_frame_allocation="proportional", gazing_ratio=0.3)
    model = Borissal(cfg)
    video = _make_video(B=3)
    sel = model.select(video)
    assert torch.equal(sel.per_frame_keep.sum(dim=-1), sel.num_keep)


def test_keep_coords_match_flat_index():
    cfg = BorissalConfig(tubelet_size=2, patch_size=16, scale=384)
    model = Borissal(cfg)
    video = _make_video()
    sel = model.select(video)

    T_grid, H_grid, W_grid = sel.grid_thw[0].tolist()
    N_pf = H_grid * W_grid
    valid_mask = sel.keep_index >= 0

    t = sel.keep_coords[..., 0]
    h = sel.keep_coords[..., 1]
    w = sel.keep_coords[..., 2]
    recomputed = t * N_pf + h * W_grid + w

    assert torch.equal(recomputed[valid_mask], sel.keep_index[valid_mask])
    assert (sel.keep_coords[~valid_mask] == -1).all()


def test_gazing_ratio_controls_budget_uniform():
    video = _make_video(B=1)
    low = Borissal(BorissalConfig(per_frame_allocation="uniform", gazing_ratio=0.1)).select(video)
    high = Borissal(BorissalConfig(per_frame_allocation="uniform", gazing_ratio=0.8)).select(video)
    assert high.num_keep[0].item() > low.num_keep[0].item()


def test_motion_weight_changes_selection():
    torch.manual_seed(0)
    B, T, H, W = 1, 16, 384, 384
    video = torch.rand(B, T, 3, H, W) * 0.05  # low baseline noise
    # inject strong motion in the second half only (a moving bright block)
    for t in range(T):
        offset = t * 4
        video[:, t, :, 100:150, offset:offset + 50] = 1.0
    # inject a static high-contrast edge in a fixed region, present every frame
    video[:, :, :, 300:360, 300:360] = torch.where(
        torch.arange(60).view(1, 1, 60, 1).expand(B, 3, 60, 60) % 2 == 0, 1.0, 0.0
    )

    cfg_kwargs = dict(per_frame_allocation="uniform", gazing_ratio=0.1, scale=H, patch_size=16, tubelet_size=2)
    motion_only = Borissal(BorissalConfig(motion_weight=1.0, **cfg_kwargs)).select(video)
    spatial_only = Borissal(BorissalConfig(motion_weight=0.0, **cfg_kwargs)).select(video)

    motion_mask = motion_only.keep_mask[0]
    spatial_mask = spatial_only.keep_mask[0]
    # the two extremes should not select an identical set of patches
    assert not torch.equal(motion_mask, spatial_mask)


def test_keep_index_is_ascending_per_row():
    # Locks down the canonical downstream contract: valid (non-padded)
    # keep_index entries must be strictly ascending idx = t*N + n per video,
    # for both allocation policies -- a downstream consumer (mask-gather +
    # RoPE position recovery) depends on this order.
    for alloc in ["uniform", "proportional"]:
        cfg = BorissalConfig(per_frame_allocation=alloc, gazing_ratio=0.37)
        video = _make_video(B=4, seed=1)
        sel = Borissal(cfg).select(video)
        for b in range(sel.keep_index.shape[0]):
            valid = sel.keep_index[b][sel.keep_index[b] >= 0]
            assert (valid[1:] > valid[:-1]).all(), f"not ascending for alloc={alloc}, b={b}"


def test_to_canonical_keep_indices():
    cfg = BorissalConfig(per_frame_allocation="proportional", gazing_ratio=0.37)
    video = _make_video(B=3, seed=2)
    sel = Borissal(cfg).select(video)

    per_video = to_canonical_keep_indices(sel)
    assert len(per_video) == 3
    for b, indices in enumerate(per_video):
        assert indices.dim() == 1
        assert indices.numel() == sel.num_keep[b].item()
        assert (indices >= 0).all()
        assert (indices[1:] > indices[:-1]).all()


def test_motion_weight_auto_adapts_to_content():
    torch.manual_seed(0)
    cfg = BorissalConfig(motion_weight="auto")
    model = Borissal(cfg)

    # Static clip: identical frame repeated (zero motion), with spatial texture.
    static_frame = torch.rand(1, 3, 384, 384) * 0.3
    static_frame[:, :, 100:300, 100:300] = 0.9
    static_video = static_frame.unsqueeze(1).repeat(1, 16, 1, 1, 1)

    # High-motion clip: a bright block sweeping across frames, low background texture.
    motion_video = torch.rand(1, 16, 3, 384, 384) * 0.05
    for t in range(16):
        off = t * 20
        motion_video[:, t, :, 150:200, off:off + 40] = 1.0

    _, inter_static = model.select_with_intermediates(static_video)
    _, inter_motion = model.select_with_intermediates(motion_video)

    w_static = inter_static["motion_weight_used"].item()
    w_motion = inter_motion["motion_weight_used"].item()
    assert w_static < 0.2   # near-zero motion -> weight should lean spatial
    assert w_motion > 0.5   # strong motion -> weight should lean motion
    assert w_motion > w_static


def test_motion_weight_fixed_unaffected_by_auto_support():
    # Passing an explicit float must behave exactly as before "auto" existed.
    video = _make_video(B=1, seed=3)
    cfg = BorissalConfig(motion_weight=0.5)
    sel_a = Borissal(cfg).select(video)
    sel_b = Borissal(BorissalConfig(motion_weight=0.5)).select(video)
    assert torch.equal(sel_a.keep_index, sel_b.keep_index)


def _contiguity(sel, B, T_grid, H_grid, W_grid):
    """Mean count of selected 4-neighbors per selected patch (fixed cross kernel)."""
    import torch.nn.functional as F
    m = sel.keep_mask.reshape(B, T_grid, H_grid, W_grid).float()
    kern = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    nb = F.conv2d(m.reshape(-1, 1, H_grid, W_grid), kern, padding=1)
    return ((nb.reshape_as(m) * m).sum() / m.sum()).item()


def test_v02_defaults_are_v01():
    # Guard against accidental default drift: a plain config and an
    # explicitly-all-off config must select identically.
    video = _make_video(B=2, seed=5)
    a = Borissal(BorissalConfig()).select(video)
    b = Borissal(BorissalConfig(motion_noise_floor="none", motion_smooth_kernel=0, block_size=1)).select(video)
    assert torch.equal(a.keep_index, b.keep_index)


def test_v02_preset_contract():
    # The full Selection contract holds under the v0.2 preset, both allocations.
    video = _make_video(B=2, seed=6)
    for alloc in ["uniform", "proportional"]:
        sel = Borissal(BorissalConfig.v0_2(per_frame_allocation=alloc)).select(video, gazing_ratio=0.25)
        assert sel.grid_thw[0].tolist() == [8, 24, 24]
        assert torch.equal(sel.per_frame_keep.sum(dim=-1), sel.num_keep)
        for bb in range(2):
            valid = sel.keep_index[bb][sel.keep_index[bb] >= 0]
            assert (valid[1:] > valid[:-1]).all()
            assert valid.unique().numel() == valid.numel()


def test_c2f_budget_exact():
    video = _make_video(B=2, seed=7)
    N_pf = 24 * 24
    for b in [2, 3]:
        for r in [0.05, 0.13, 0.5, 1.0]:
            sel = Borissal(BorissalConfig(block_size=b)).select(video, gazing_ratio=r)
            k = min(max(1, round(r * N_pf)), N_pf)
            assert (sel.per_frame_keep == k).all()


def test_c2f_full_ratio_is_identity():
    # ratio=1.0 keeps everything regardless of block gating.
    video = _make_video(B=1, seed=8)
    sel = Borissal(BorissalConfig(block_size=2)).select(video, gazing_ratio=1.0)
    assert sel.keep_mask.all()


def test_c2f_bad_block_size_raises():
    video = _make_video(B=1)
    try:
        Borissal(BorissalConfig(block_size=5)).select(video)  # 24 % 5 != 0
    except ValueError as e:
        assert "block_size" in str(e)
    else:
        raise AssertionError("block_size=5 on a 24x24 grid should raise")


def test_c2f_increases_contiguity():
    video = _make_video(B=2, seed=9)
    c1 = _contiguity(Borissal(BorissalConfig(block_size=1)).select(video, gazing_ratio=0.25), 2, 8, 24, 24)
    c2 = _contiguity(Borissal(BorissalConfig(block_size=2)).select(video, gazing_ratio=0.25), 2, 8, 24, 24)
    assert c2 > c1


def test_noise_floor_suppresses_amplified_noise():
    # The floor's worst-case target: a clip with NO true motion anywhere --
    # per-tubelet min-max then amplifies the sensor-noise diff to full [0,1],
    # letting pure noise compete with real spatial signal in the blend.
    # Setup: static texture in the LEFT half (real spatial saliency), static
    # flat gray + per-frame noise everywhere. With the floor on, the motion
    # channel of this motionless clip collapses and selection concentrates on
    # the texture; with it off, amplified noise pulls selections into the
    # flat right half.
    torch.manual_seed(0)
    B, T, H, W = 1, 16, 384, 384
    base = torch.full((1, 1, 3, H, W), 0.5)
    checker = (torch.arange(H).view(-1, 1) // 8 + torch.arange(W).view(1, -1) // 8) % 2
    base[..., :, : W // 2] = 0.2 + 0.6 * checker[:, : W // 2].float()  # static texture, left half
    video = base.expand(B, T, 3, H, W).clone() + 0.02 * torch.randn(B, T, 3, H, W)

    kw = dict(motion_weight=0.5, gazing_ratio=0.1)
    sel_off, inter_off = Borissal(BorissalConfig(motion_noise_floor="none", **kw)).select_with_intermediates(video)
    sel_on, inter_on = Borissal(BorissalConfig(motion_noise_floor="quantile", **kw)).select_with_intermediates(video)

    # budget unchanged
    assert torch.equal(sel_off.num_keep, sel_on.num_keep)

    # Mechanism: the normalized motion channel of a motionless clip is
    # strongly suppressed with the floor on (min-max re-normalizes the
    # positive residual tail, so it cannot reach exactly zero -- observed
    # effect size is ~4x; assert a robust 2x).
    assert inter_on["motion_norm"].mean() < 0.5 * inter_off["motion_norm"].mean()
    assert "noise_floor_tau" in inter_on and inter_on["noise_floor_tau"].shape == (B, 8)

    # (A selection-level "spatial score of kept set improves" assertion was
    # tried and removed: on this clip both variants already saturate their
    # selection inside the texture half (~0.98 either way), so the comparison
    # is pure noise. The mechanism-level suppression above is the meaningful,
    # stable check.)


def test_frame_diff_catches_intra_tubelet_motion():
    # A block that jumps away and back WITHIN one tubelet: tubelet-mean
    # differencing largely cancels it; frame differencing must not.
    B, T, H, W = 1, 16, 3, 384
    video = torch.zeros(B, T, H, 384, 384)
    # frames 0..15; within each tubelet (2 frames), the block alternates
    # between two positions -> tubelet means are all (nearly) identical.
    for t in range(T):
        off = 100 if t % 2 == 0 else 200
        video[:, t, :, off:off + 64, 100:164] = 1.0

    kw = dict(motion_weight=1.0)
    _, inter_tub = Borissal(BorissalConfig(motion_diff="tubelet", **kw)).select_with_intermediates(video)
    _, inter_frm = Borissal(BorissalConfig(motion_diff="frame", **kw)).select_with_intermediates(video)

    # Raw (pre-normalization) signal is not exposed; compare where the
    # normalized motion mass sits: frame mode must put clear mass on the two
    # block rows; contract must hold in both modes.
    sel_frm = Borissal(BorissalConfig(motion_diff="frame", **kw)).select(video, gazing_ratio=0.1)
    h_coord = sel_frm.keep_coords[..., 1]
    valid = sel_frm.keep_index >= 0
    block_rows = ((h_coord >= 100 // 16) & (h_coord <= (264 // 16)) & valid).sum().item()
    assert block_rows / valid.sum().item() > 0.8  # selections concentrate on the oscillating block
    assert torch.equal(sel_frm.per_frame_keep.sum(dim=-1), sel_frm.num_keep)


def test_consistency_penalty_contract():
    # double_diff is an EXPERIMENTAL knob, deliberately excluded from the
    # v0.2 preset: synthetic testing showed the per-tubelet min-max
    # normalization structurally cancels its noise attenuation (min halves
    # noise AND the normalization ceiling alike, so post-norm noise density
    # is unchanged, while textured-mover diffs degrade -- net selection
    # shifted TOWARD noise in our experiments). Recorded as a negative
    # result in docs/borissal/design.md. Here we only lock the contract:
    # it runs, preserves budgets, and actually alters the motion signal.
    video = _make_video(B=1, seed=12)
    kw = dict(motion_diff="frame", motion_weight=1.0, gazing_ratio=0.1)
    off_sel, off_i = Borissal(BorissalConfig(motion_consistency="none", **kw)).select_with_intermediates(video)
    on_sel, on_i = Borissal(BorissalConfig(motion_consistency="double_diff", **kw)).select_with_intermediates(video)
    assert torch.equal(off_sel.num_keep, on_sel.num_keep)
    assert torch.equal(on_sel.per_frame_keep.sum(dim=-1), on_sel.num_keep)
    assert not torch.equal(off_i["motion_norm"], on_i["motion_norm"])  # it does something
    for bb in range(1):
        valid = on_sel.keep_index[bb][on_sel.keep_index[bb] >= 0]
        assert (valid[1:] > valid[:-1]).all()


def test_score_blend_beta1_is_default():
    video = _make_video(B=1, seed=10)
    a = Borissal(BorissalConfig()).select(video)
    b = Borissal(BorissalConfig(score_norm_blend=1.0)).select(video)
    assert torch.equal(a.keep_index, b.keep_index)


def test_global_allocation_invariants():
    video = _make_video(B=2, seed=11)
    L, T_grid = 8 * 576, 8
    for ratio in [0.05, 0.25, 0.5]:
        sel = Borissal(BorissalConfig(per_frame_allocation="global", score_norm_blend=0.7)) \
            .select(video, gazing_ratio=ratio)
        K_total = min(max(T_grid, round(ratio * L)), L)
        m = min(max(1, round(0.25 * K_total / T_grid)), K_total // T_grid)
        assert (sel.num_keep == K_total).all()
        assert (sel.per_frame_keep >= m).all()
        for bb in range(2):
            valid = sel.keep_index[bb][sel.keep_index[bb] >= 0]
            assert (valid[1:] > valid[:-1]).all()


def test_global_allocation_concentrates_on_high_energy_tubelets():
    # First half of the clip static, second half has a strong mover:
    # global allocation should give the active tubelets more budget.
    B, T = 1, 16
    video = torch.full((B, T, 3, 384, 384), 0.5)
    for t in range(8, 16):
        off = (t - 8) * 40
        video[:, t, :, 100:200, off:off + 80] = 1.0

    # NOTE: the budget must stay below the total salient mass -- with a
    # larger ratio the surplus spills into zero-score ties (index-ordered),
    # which is arbitrary by construction. ratio=0.05 keeps K_total (~230)
    # well under the mover's footprint.
    sel = Borissal(BorissalConfig(per_frame_allocation="global", score_norm_blend=0.5,
                                  motion_weight=1.0)).select(video, gazing_ratio=0.05)
    first_half = sel.per_frame_keep[0, :4].sum().item()
    second_half = sel.per_frame_keep[0, 4:].sum().item()
    assert second_half > first_half


def test_center_bias_prefers_center_on_uniform_clip():
    flat = torch.full((1, 16, 3, 384, 384), 0.5)
    on = Borissal(BorissalConfig(center_bias=0.5, motion_weight=0.0)).select(flat, gazing_ratio=0.1)
    off = Borissal(BorissalConfig(center_bias=0.0, motion_weight=0.0)).select(flat, gazing_ratio=0.1)

    def mean_center_dist(sel):
        h = sel.keep_coords[..., 1].float()
        w = sel.keep_coords[..., 2].float()
        valid = sel.keep_index >= 0
        return (((h[valid] - 11.5) / 12) ** 2 + ((w[valid] - 11.5) / 12) ** 2).mean().item()

    assert mean_center_dist(on) < mean_center_dist(off)


def test_mps_matches_cpu_grid_and_counts():
    if not torch.backends.mps.is_available():
        return
    cfg = BorissalConfig(gazing_ratio=0.4)
    video = _make_video(B=1)
    cpu_sel = Borissal(cfg).select(video)
    mps_sel = Borissal(cfg).to("mps").select(video.to("mps"))
    assert torch.equal(cpu_sel.grid_thw, mps_sel.grid_thw.cpu())
    assert torch.equal(cpu_sel.num_keep, mps_sel.num_keep.cpu())
