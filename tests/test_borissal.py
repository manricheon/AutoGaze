import torch

from autogaze.models.borissal import Borissal, BorissalConfig


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


def test_mps_matches_cpu_grid_and_counts():
    if not torch.backends.mps.is_available():
        return
    cfg = BorissalConfig(gazing_ratio=0.4)
    video = _make_video(B=1)
    cpu_sel = Borissal(cfg).select(video)
    mps_sel = Borissal(cfg).to("mps").select(video.to("mps"))
    assert torch.equal(cpu_sel.grid_thw, mps_sel.grid_thw.cpu())
    assert torch.equal(cpu_sel.num_keep, mps_sel.num_keep.cpu())
