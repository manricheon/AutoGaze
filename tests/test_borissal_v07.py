"""Borissal v0.7 "Datdol" anchor-novelty selection tests.

Contract + the adversarial synthetics from the design review: transit
contamination, hard cuts, exact ties, frame-rate robustness of the novelty
channel, and the K_a > Sc boundary that crashed design v1.
"""
import pytest
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.signals_v03 import (
    appearance_novelty, cube_best_time, temporal_median_grid,
)

SIZE = 96          # grid 6x6, cube grid 3x3 -> Sc=9 (small, boundary-crossing)


def _model(**over):
    return Borissal(BorissalConfig.v0_7(scale=SIZE, **over))


def _clip(B=1, T=8, size=SIZE, seed=0):
    torch.manual_seed(seed)
    return torch.rand(B, T, 3, size, size)


# --- M1 primitives -------------------------------------------------------------

def test_temporal_median_odd_and_even():
    x = torch.tensor([[[[1.0]], [[5.0]], [[3.0]], [[9.0]]]])          # T=4
    assert float(temporal_median_grid(x)) == 3.0                      # LOWER median, pinned
    x5 = torch.tensor([[[[1.0]], [[5.0]], [[3.0]], [[9.0]], [[7.0]]]])
    assert float(temporal_median_grid(x5)) == 5.0                     # true median


def test_median_ignores_short_occluder():
    # a cell occluded for 3 of 8 tubelets: canonical = background, novelty
    # fires on the OCCLUDER frames, not on the background before/after
    g = torch.full((1, 8, 1, 1), 0.2)
    g[:, 3:6] = 0.9
    med = temporal_median_grid(g)
    assert float(med) == pytest.approx(0.2)
    nov = appearance_novelty(g, med)[0, :, 0, 0]
    assert (nov[3:6] > 0.6).all() and (nov[:3] < 1e-6).all() and (nov[6:] < 1e-6).all()


def test_cube_best_time_single_index_and_deterministic():
    score = torch.tensor([[[2.0, 1.0], [2.0, 3.0], [0.0, 3.0]]])      # ties at both sites
    v1, i1 = cube_best_time(score)
    v2, i2 = cube_best_time(score)
    assert i1.shape == (1, 2) and torch.equal(i1, i2)                 # one index/site, deterministic
    assert torch.equal(v1, torch.tensor([[2.0, 3.0]]))


def test_best_time_is_found_not_first():
    # sharpest appearance at tubelet 5 of 8 -- the argmax must find it (this
    # is the test that catches per-tubelet-normalized inputs, where every
    # tubelet's max is 1.0 and argmax degenerates to index 0)
    g = torch.full((1, 8, 6, 6), 0.5)
    yy, xx = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    g[:, 5] = ((yy + xx) % 2).float()          # checkerboard: high |lap| at EVERY cell
    from autogaze.models.borissal.modeling_borissal import _minmax_norm_global
    from autogaze.models.borissal.signals_v03 import laplacian_energy
    a = _minmax_norm_global(laplacian_energy(g), 1e-6)
    vals, idx = cube_best_time(a.reshape(1, 8, 36))
    assert (idx == 5).all(), f"argmax must find t=5 everywhere: {idx.tolist()}"


# --- contract -------------------------------------------------------------------

@pytest.mark.parametrize("sg", ["cube", "fine"])
@pytest.mark.parametrize("ratio", [0.15, 0.25, 0.5, 0.75, 1.0])
def test_contract_all_ratios(ratio, sg):
    """Ascending, unique, exact 4*K_cubes budget; ratio sweep crosses the
    K_a > Sc boundary (Sc=9 here) that crashed design v1. Both signal grids:
    "cube" (12x12-native, v0_7 default per user decision) and "fine"."""
    sel = _model(signal_grid=sg).select(_clip(), gazing_ratio=ratio)
    T_grid, Hg, Wg = (int(x) for x in sel.grid_thw[0])
    L = T_grid * Hg * Wg
    n = int(sel.num_keep[0])
    assert n % 4 == 0, "anchor_novelty selects whole 2x2 cubes"
    K_patch = min(max(1, round(ratio * L)), L)
    K_cubes = min(max(T_grid, round(K_patch / 4)), L // 4)
    assert n == 4 * K_cubes
    idx = sel.keep_index[0][:n]
    assert (idx[1:] > idx[:-1]).all()
    assert int(torch.unique(idx).numel()) == n
    assert int(sel.per_frame_keep[0].sum()) == n
    if ratio == 1.0:
        assert torch.equal(idx, torch.arange(L, dtype=idx.dtype))


def test_every_tubelet_keeps_at_least_one_cube():
    sel = _model().select(_clip(), gazing_ratio=0.15)
    assert (sel.per_frame_keep[0] >= 4).all(), "floor: no empty tubelet"


@pytest.mark.parametrize("sg", ["cube", "fine"])
def test_whole_cubes_qwen_strict(sg):
    from autogaze.models.borissal.adapters import to_qwen3vl_video_tokens
    sel = _model(signal_grid=sg).select(_clip(), gazing_ratio=0.25)
    out = to_qwen3vl_video_tokens(sel, 2, "strict")   # raises on partial blocks
    assert out["n_partial_blocks"] == 0


def test_signal_grid_misuse_raises():
    with pytest.raises(ValueError, match="anchor_novelty knob"):
        Borissal(BorissalConfig.v0_5(scale=SIZE, signal_grid="cube")).select(
            _clip(), gazing_ratio=0.25)
    with pytest.raises(ValueError, match="signal_grid"):
        _model(signal_grid="nonsense").select(_clip(), gazing_ratio=0.25)


def test_batch_and_single_tubelet():
    sel = _model().select(_clip(B=2, seed=3), gazing_ratio=0.25)
    assert int(sel.num_keep[0]) == int(sel.num_keep[1])   # same static budget
    # T=2 -> T_grid=1: novelty is zero everywhere, must still satisfy contract
    sel1 = _model().select(_clip(T=2), gazing_ratio=0.5)
    assert int(sel1.num_keep[0]) == int(sel1.per_frame_keep[0].sum())


def test_incompatible_knobs_raise():
    v = _clip()
    with pytest.raises(ValueError, match="score_coarsen"):
        _model(score_coarsen=1).select(v, gazing_ratio=0.25)
    with pytest.raises(ValueError, match="allocation"):
        _model().select(v, gazing_ratio=0.25, per_frame_allocation="global")
    with pytest.raises(ValueError, match="spread"):
        _model().select(v, gazing_ratio=0.25, spread_fraction=0.25)
    with pytest.raises(ValueError, match="per_frame_counts"):
        _model().select(v, gazing_ratio=0.25, per_frame_counts=torch.full((4,), 9))
    with pytest.raises(ValueError, match="hysteresis"):
        _model(select_hysteresis_eps=0.05).select(v, gazing_ratio=0.25)
    with pytest.raises(ValueError, match="block gate"):
        _model(block_size=2).select(v, gazing_ratio=0.25)
    with pytest.raises(ValueError, match="keyframe"):
        _model(keyframe_prior=True).select(v, gazing_ratio=0.25)
    with pytest.raises(ValueError, match="selection_mode"):
        Borissal(BorissalConfig(scale=SIZE, selection_mode="nonsense")).select(v, gazing_ratio=0.25)


def test_preset_contract_pins():
    c = BorissalConfig.v0_7(scale=SIZE)
    assert c.selection_mode == "anchor_novelty"
    assert c.signal_grid == "cube"          # 12x12-native signals are the default
    assert c.motion_weight == 0.0          # the auto blend is REMOVED, not inherited
    assert c.block_size == 1 and c.score_coarsen == 2
    assert c.luma_mode == "bt601" and c.per_frame_allocation == "uniform"
    assert not c.static_guard and not c.laplacian_gate and not c.keyframe_prior
    assert c.center_bias == 0.0


def test_default_config_unaffected():
    assert BorissalConfig().selection_mode == "topk"
    assert BorissalConfig.v0_6(scale=SIZE).selection_mode == "topk"


# --- adversarial synthetics (design-review verdict #3) ---------------------------

def _transit_clip(T=16, size=SIZE):
    """Textured static background; a bright 16px mover crosses the middle row
    fast enough that each cube-column is occupied for only ~1-3 of 8 tubelets
    (the designed-for regime). Returns (video, occupancy) where occupancy maps
    cube-column -> set of tubelets the mover overlaps that column."""
    torch.manual_seed(1)
    bg = 0.25 + 0.5 * torch.rand(1, 1, 3, size, size)
    v = bg.expand(1, T, 3, size, size).clone()
    occ = {c: set() for c in range(3)}
    for f in range(T):
        x0 = (f * (size - 16)) // (T - 1)
        v[0, f, :, 40:56, x0:x0 + 16] = 1.0
        for c in range(3):
            if x0 < (c + 1) * 32 and x0 + 16 > c * 32:
                occ[c].add(f // 2)
    return v, occ


def _anchor_times(m, v):
    """Reconstruct (naive_t, guard_t) anchor choices from the primitives,
    exactly as _anchor_novelty_select ranks them."""
    import torch.nn.functional as F
    from autogaze.models.borissal.modeling_borissal import _minmax_norm_global
    from autogaze.models.borissal.signals_v03 import laplacian_energy, dog_blob
    cfg = m.config
    sal = m._saliency_scores(v, 2, 16, 0.0)
    lg, sp, mp = sal["luma_grid"], sal["spatial_p"], sal["motion_p"]
    A = _minmax_norm_global(sp, cfg.eps) \
        + cfg.dog_blob_weight * _minmax_norm_global(dog_blob(lg), cfg.eps) \
        + cfg.anchor_lap_weight * _minmax_norm_global(laplacian_energy(lg), cfg.eps)
    N = _minmax_norm_global(appearance_novelty(lg, temporal_median_grid(lg)), cfg.eps) \
        + cfg.novelty_shortterm_weight * _minmax_norm_global(mp, cfg.eps)
    Tg, Hg = lg.shape[1], lg.shape[2]
    c = cfg.score_coarsen
    A_c = F.avg_pool2d(A.reshape(-1, 1, Hg, Hg), c, c).view(1, Tg, -1)
    N_c = F.avg_pool2d(N.reshape(-1, 1, Hg, Hg), c, c).view(1, Tg, -1)
    _, guard_t = cube_best_time(A_c - cfg.anchor_novelty_lambda * N_c)
    _, naive_t = cube_best_time(A_c)
    return naive_t[0], guard_t[0]


def test_transit_background_not_anchored_at_transit_time():
    """For cube sites the mover crossed, the guarded anchor time must move OFF
    the occupancy tubelets that the naive appearance argmax picks (transit
    moments have inflated A -- bright block edges -- and belong to the novelty
    pool, not the anchor pool). Asserted where a clean alternative exists;
    when the mover occupies most of a column's tubelets the guard may have no
    clean alternative -- that limitation is deliberate and documented."""
    v, occ = _transit_clip()
    naive_t, guard_t = _anchor_times(_model(), v)
    transit_sites = [(3 + col, col) for col in range(3)]   # cube row 1
    naive_hits = [(s, col) for s, col in transit_sites if int(naive_t[s]) in occ[col]]
    assert len(naive_hits) >= 2, "stimulus must make the naive argmax pick transit moments"
    fixed = sum(int(guard_t[s]) not in occ[col] for s, col in naive_hits)
    assert fixed >= len(naive_hits) - 1, (
        f"guard must move anchors off transit for all but at most one site: "
        f"naive={[int(naive_t[s]) for s, _ in naive_hits]} "
        f"guard={[int(guard_t[s]) for s, _ in naive_hits]} occ={occ}")


def test_hard_cut_novelty_fires_after_cut():
    """Scene A for 5 tubelets, scene B for 3 -- BOTH equally textured (same
    rand distribution, different pattern) so a pure-appearance selector has no
    reason to prefer B. The canonical median = scene A, so post-cut budget
    concentration can only come from the novelty channel."""
    torch.manual_seed(2)
    T = 16
    pat_a = 0.2 + 0.6 * torch.rand(1, 1, 3, SIZE, SIZE)
    pat_b = 0.2 + 0.6 * torch.rand(1, 1, 3, SIZE, SIZE)   # same stats, different pattern
    v = pat_a.expand(1, T, 3, SIZE, SIZE).clone()
    v[0, 10:] = pat_b[0, 0]
    sel = _model().select(v, gazing_ratio=0.25)
    pf = sel.per_frame_keep[0]                            # cut at tubelet 5
    assert pf[5:].float().mean() > 1.3 * pf[:5].float().mean(), \
        f"post-cut tubelets must draw clearly more budget: {pf.tolist()}"


def test_exact_tie_no_duplicates_no_crash():
    """A perfectly constant clip: every score ties everywhere. Contract must
    hold (no duplicate cubes, exact budget) and repeat runs are deterministic."""
    v = torch.full((1, 8, 3, SIZE, SIZE), 0.5)
    s1 = _model().select(v, gazing_ratio=0.25)
    s2 = _model().select(v, gazing_ratio=0.25)
    n = int(s1.num_keep[0])
    assert int(torch.unique(s1.keep_index[0][:n]).numel()) == n
    assert torch.equal(s1.keep_index, s2.keep_index)


# --- frame-rate robustness (the v0.4 problem, solved structurally) ----------------

def test_novelty_magnitude_stable_16f_vs_32f():
    """The median-deviation novelty must NOT shrink when the same clip is
    decoded at 2x the frame rate (consecutive-frame diffs do -- that was v0.4)."""
    torch.manual_seed(4)
    base = torch.rand(1, 8, 3, SIZE, SIZE)
    v16 = base.repeat_interleave(2, dim=1)                 # 16 frames
    v32 = base.repeat_interleave(4, dim=1)                 # 32 frames, same content
    m = _model()
    def novelty_mag(v):
        sal = m._saliency_scores(v, 2, 16, 0.0)
        lg = sal["luma_grid"]
        return float(appearance_novelty(lg, temporal_median_grid(lg)).mean())
    n16, n32 = novelty_mag(v16), novelty_mag(v32)
    assert n32 > 0.5 * n16, f"novelty collapsed with frame rate: {n16:.4f} -> {n32:.4f}"
    assert abs(n32 - n16) / max(n16, 1e-6) < 0.5


# --- nestedness report (not a hard gate) ------------------------------------------

def test_nestedness_report():
    v = _clip(seed=7)
    m = _model()
    prev = None
    worst = 1.0
    for r in (0.15, 0.25, 0.5, 0.75):
        sel = m.select(v, gazing_ratio=r)
        n = int(sel.num_keep[0])
        cur = set(sel.keep_index[0][:n].tolist())
        if prev is not None:
            frac = len(prev & cur) / max(1, len(prev))
            worst = min(worst, frac)
        prev = cur
    # anchors/floors are ratio-stable; the boundary between pools moves, so we
    # REPORT near-nestedness rather than assert strict subset relations
    assert worst > 0.85, f"selection order badly unstable across ratios: {worst:.2f}"


# --- GPU smoke ---------------------------------------------------------------------

def test_gpu_smoke_invariants():
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        pytest.skip("no GPU backend")
    v = _clip(seed=5).to(dev)
    sel = _model().select(v, gazing_ratio=0.25)
    n = int(sel.num_keep[0])
    idx = sel.keep_index[0][:n]
    assert (idx[1:] > idx[:-1]).all() and n % 4 == 0
    assert int(sel.per_frame_keep[0].sum()) == n


# --- tubelet_size support (review round E-C) ------------------------------------

@pytest.mark.parametrize("tub", [1, 2])
def test_tubelet_sizes_contract(tub):
    """tubelet_size=2 is the designed default (V-JEPA tubelet / Qwen temporal
    fold alignment); tubelet_size=1 must also satisfy the full contract --
    it is the per-frame-encoder configuration (OneVision/SigLIP stacks), NOT
    compatible with Qwen's temporal_patch_size=2 grid."""
    torch.manual_seed(0)
    v = torch.rand(1, 8, 3, SIZE, SIZE)
    sel = Borissal(BorissalConfig.v0_7(scale=SIZE, tubelet_size=tub)).select(v, gazing_ratio=0.25)
    T, H, W = (int(x) for x in sel.grid_thw[0])
    assert T == 8 // tub
    n = int(sel.num_keep[0])
    idx = sel.keep_index[0][:n]
    assert n % 4 == 0 and (idx[1:] > idx[:-1]).all()
    assert int(torch.unique(idx).numel()) == n
    assert int(sel.per_frame_keep[0].sum()) == n
    assert (sel.per_frame_keep[0] >= 4).all()          # floor holds per tubelet
    full = Borissal(BorissalConfig.v0_7(scale=SIZE, tubelet_size=tub)).select(v, gazing_ratio=1.0)
    assert int(full.num_keep[0]) == T * H * W          # ratio 1.0 keeps all


def test_tubelet1_novelty_still_framerate_stable():
    """The median-deviation novelty must stay frame-rate stable at tubelet 1
    too (T_grid doubles but the canonical reference is still clip-level)."""
    torch.manual_seed(4)
    base = torch.rand(1, 8, 3, SIZE, SIZE)
    m = Borissal(BorissalConfig.v0_7(scale=SIZE, tubelet_size=1))
    def mag(v):
        sal = m._saliency_scores(v, 1, 16, 0.0)
        lg = sal["luma_grid"]
        return float(appearance_novelty(lg, temporal_median_grid(lg)).mean())
    n8, n16 = mag(base), mag(base.repeat_interleave(2, dim=1))
    assert abs(n16 - n8) / max(n8, 1e-6) < 0.5
