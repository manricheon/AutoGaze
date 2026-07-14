import torch

from autogaze.models.borissal import BorissalV1, BorissalV1Config
from autogaze.models.borissal.losses import (
    LossWeights,
    combine_losses,
    predictor_coverage_loss,
    score_entropy_loss,
    v0_distill_loss,
)
from autogaze.models.borissal.vjepa2_sparse import VJEPA2Teacher


def _make_video(B=2, T=16, C=3, H=384, W=384, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, T, C, H, W, generator=g)


def test_v1_selection_contract_all_input_modes():
    video = _make_video()
    for mode in ["maps", "pixels", "both"]:
        model = BorissalV1(BorissalV1Config(input_mode=mode)).eval()
        sel = model.select(video, gazing_ratio=0.5)
        T_grid, H_grid, W_grid = sel.grid_thw[0].tolist()
        assert [T_grid, H_grid, W_grid] == [8, 24, 24]
        assert torch.equal(sel.per_frame_keep.sum(dim=-1), sel.num_keep)
        # canonical ascending keep_index (same downstream contract as v0)
        for b in range(video.shape[0]):
            valid = sel.keep_index[b][sel.keep_index[b] >= 0]
            assert (valid[1:] > valid[:-1]).all()


def test_v1_residual_scoring_forward():
    video = _make_video(B=1)
    model = BorissalV1(BorissalV1Config(residual_scoring=True)).eval()
    sel = model.select(video, gazing_ratio=0.25)
    assert sel.num_keep[0].item() == sel.keep_mask[0].sum().item()


def test_v1_forward_train_gradients_reach_all_params():
    video = _make_video(B=1)
    model = BorissalV1(BorissalV1Config()).train()
    out = model.forward_train(video, gazing_ratio=0.4)
    loss = (out["st_gate"] * torch.randn_like(out["st_gate"])).sum()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for _ in model.parameters())
    assert n_with_grad == n_params, f"gradients reached only {n_with_grad}/{n_params} param tensors"


def test_v1_st_gate_forward_equals_hard_and_exact_k():
    video = _make_video(B=2)
    model = BorissalV1(BorissalV1Config()).train()
    out = model.forward_train(video, gazing_ratio=0.3)
    assert torch.equal(out["st_gate"] > 0.5, out["hard_keep"])
    assert out["hard_keep"].sum(dim=-1).eq(out["k"]).all()
    assert not out["st_gate"].isnan().any()


def test_v1_gumbel_stochastic_in_train_deterministic_in_eval():
    video = _make_video(B=1, seed=1)
    model = BorissalV1(BorissalV1Config())
    model.train()
    a = model.forward_train(video, gazing_ratio=0.5)["hard_keep"]
    b = model.forward_train(video, gazing_ratio=0.5)["hard_keep"]
    assert not torch.equal(a, b), "gumbel noise should vary selection in train mode"
    model.eval()
    c = model.forward_train(video, gazing_ratio=0.5)["hard_keep"]
    d = model.forward_train(video, gazing_ratio=0.5)["hard_keep"]
    assert torch.equal(c, d), "eval selection must be deterministic"


def test_v1_v0_preset_paths():
    # default is v0.2 signals; v0.1 path must also work; both keep the contract.
    video = _make_video(B=1, seed=20)
    for preset in ["v0.2", "v0.1"]:
        model = BorissalV1(BorissalV1Config(v0_preset=preset)).eval()
        sel = model.select(video, gazing_ratio=0.3)
        assert sel.grid_thw[0].tolist() == [8, 24, 24]
        valid = sel.keep_index[0][sel.keep_index[0] >= 0]
        assert (valid[1:] > valid[:-1]).all()
    # the two presets must actually produce different input signals
    a = BorissalV1(BorissalV1Config(v0_preset="v0.2"))
    assert a._v0.config.motion_noise_floor == "quantile"
    b = BorissalV1(BorissalV1Config(v0_preset="v0.1"))
    assert b._v0.config.motion_noise_floor == "none"


def test_v1_global_allocation_select():
    video = _make_video(B=2, seed=21)
    model = BorissalV1(BorissalV1Config()).eval()
    L, T_grid = 8 * 576, 8
    sel = model.select(video, gazing_ratio=0.25, per_frame_allocation="global")
    K_total = min(max(T_grid, round(0.25 * L)), L)
    m = min(max(1, round(0.25 * K_total / T_grid)), K_total // T_grid)
    assert (sel.num_keep == K_total).all()
    assert (sel.per_frame_keep >= m).all()
    for bb in range(2):
        valid = sel.keep_index[bb][sel.keep_index[bb] >= 0]
        assert (valid[1:] > valid[:-1]).all()


def test_uniqueness_reward_gradient_flow():
    from autogaze.models.borissal.losses import uniqueness_reward_loss
    from autogaze.models.borissal.modeling_borissal import _pack_gazing_mask

    teacher = VJEPA2Teacher.tiny_random(crop_size=128, frames_per_clip=16)
    model = BorissalV1(BorissalV1Config(scale=128)).train()
    video = _make_video(B=1, H=128, W=128, seed=22)

    out = model.forward_train(video, gazing_ratio=0.5)
    B, L = 1, out["hard_keep"].shape[1] * out["hard_keep"].shape[2]
    keep_idx = out["keep_index"]
    tgt_idx, _ = _pack_gazing_mask(~out["hard_keep"].reshape(B, L))
    gate_flat = out["st_gate"].reshape(B, L)
    rest_gate = torch.gather(1.0 - gate_flat, 1, tgt_idx)

    dense = teacher.dense_features(video)
    rest_sparse = teacher.sparse_features(video, tgt_idx, gate=rest_gate)
    pred_sel = teacher.predict(rest_sparse, tgt_idx, keep_idx, num_tokens=L)
    sel_targets = torch.gather(dense, 1, keep_idx.unsqueeze(-1).expand(-1, -1, dense.size(-1)))

    loss = uniqueness_reward_loss(pred_sel, sel_targets)
    assert loss.item() <= 0  # reward form
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), \
        "uniqueness reward must reach the selector through the inverse gate"


def test_gradient_reaches_unselected_and_low_prob_patches():
    # The lock-in worry: patches that are unselected (or low-probability)
    # must still receive score gradients, otherwise they can never recover.
    # Channels: softmax coupling (~p_j), entropy term (all logits), and --
    # when enabled -- the uniqueness inverse gate. This locks in that a
    # standard coverage-style loss through the ST gate reaches UNSELECTED
    # logits with nonzero gradient.
    from autogaze.models.borissal.losses import score_entropy_loss

    model = BorissalV1(BorissalV1Config(scale=128)).train()
    video = _make_video(B=1, H=128, W=128, seed=30)
    out = model.forward_train(video, gazing_ratio=0.3)

    # a loss that only touches the SELECTED side (like coverage does)
    B, L = 1, out["hard_keep"].shape[1] * out["hard_keep"].shape[2]
    gate_flat = out["st_gate"].reshape(B, L)
    gate = torch.gather(gate_flat, 1, out["keep_index"])
    loss = (gate * torch.randn_like(gate)).sum()
    loss.backward(retain_graph=True)

    g = out["logits"].grad.abs()
    unsel = ~out["hard_keep"]
    assert g[unsel].max() > 0, "softmax coupling must reach unselected logits"
    frac_nonzero_unsel = (g[unsel] > 0).float().mean().item()
    assert frac_nonzero_unsel > 0.99, f"only {frac_nonzero_unsel:.2%} of unselected logits got gradient"

    # entropy term reaches ALL logits too (magnitude ~ p log p)
    out["logits"].grad = None
    score_entropy_loss(out["probs"]).backward()
    g2 = out["logits"].grad.abs()
    assert (g2[unsel] > 0).float().mean().item() > 0.99


def test_sparse_teacher_pipeline_and_grad_isolation():
    # tiny random teacher at reduced resolution: full SSL graph in one test
    teacher = VJEPA2Teacher.tiny_random(crop_size=128, frames_per_clip=16)
    model = BorissalV1(BorissalV1Config(scale=128)).train()
    video = _make_video(B=1, H=128, W=128, seed=2)

    out = model.forward_train(video, gazing_ratio=0.5)
    B = 1
    L = out["hard_keep"].shape[1] * out["hard_keep"].shape[2]
    keep_idx = out["keep_index"]
    gate = torch.gather(out["st_gate"].reshape(B, L), 1, keep_idx)

    from autogaze.models.borissal.modeling_borissal import _pack_gazing_mask
    tgt_idx, pad = _pack_gazing_mask(~out["hard_keep"].reshape(B, L))
    assert not pad.any()

    dense = teacher.dense_features(video)
    sparse = teacher.sparse_features(video, keep_idx, gate=gate)
    pred = teacher.predict(sparse, keep_idx, tgt_idx, num_tokens=L)
    tgt_feats = torch.gather(dense, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, dense.size(-1)))

    loss = predictor_coverage_loss(pred, tgt_feats)
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), \
        "selector received no gradient through the frozen teacher"
    assert all(p.grad is None for p in teacher.parameters()), "teacher must stay frozen"


def test_teacher_grid_mismatch_rejected():
    from transformers import VJEPA2Config, VJEPA2Model
    bad = VJEPA2Config(patch_size=14, tubelet_size=2, crop_size=112, frames_per_clip=16,
                       hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
                       pred_hidden_size=32, pred_num_attention_heads=2, pred_num_hidden_layers=1)
    try:
        VJEPA2Teacher(VJEPA2Model(bad))
    except ValueError as e:
        assert "grid mismatch" in str(e)
    else:
        raise AssertionError("patch_size=14 teacher should be rejected")


def test_combine_losses_weights_and_warmup_decay():
    w = LossWeights(predictor_coverage=1.0, v0_distill=1.0, v0_distill_warmup_steps=10)
    make = {
        "predictor_coverage": lambda: torch.tensor(2.0),
        "dense_sparse_match": lambda: torch.tensor(100.0),  # weight 0 -> never evaluated? (lazy)
        "score_entropy": lambda: torch.tensor(100.0),
        "v0_distill": lambda: torch.tensor(4.0),
    }
    total0, logs0 = combine_losses(w, step=0, terms=make)
    assert abs(total0.item() - (2.0 + 4.0)) < 1e-6
    total5, _ = combine_losses(w, step=5, terms=make)
    assert abs(total5.item() - (2.0 + 0.5 * 4.0)) < 1e-6
    total10, logs10 = combine_losses(w, step=10, terms=make)
    assert abs(total10.item() - 2.0) < 1e-6
    assert "dense_sparse_match" not in logs0  # zero-weight terms skipped


def test_v1_block_st_selection():
    video = _make_video(B=2, seed=50)
    model = BorissalV1(BorissalV1Config(train_block_size=2)).train()
    out = model.forward_train(video, gazing_ratio=0.3)
    B, T_grid, N_pf = out["hard_keep"].shape
    k = out["k"]
    assert k % 4 == 0, "budget must snap to whole 2x2 blocks"
    assert out["hard_keep"].sum(dim=-1).eq(k).all()
    # every selected token arrives as a FULL 2x2 block (no scatter by construction)
    m = out["hard_keep"].reshape(B, T_grid, 12, 2, 12, 2)
    assert torch.equal(m.any(dim=5).any(dim=3), m.all(dim=5).all(dim=3))
    assert torch.equal(out["st_gate"] > 0.5, out["hard_keep"])
    # gradients reach every parameter through the block pooling
    loss = (out["st_gate"] * torch.randn_like(out["st_gate"])).sum()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_with_grad == sum(1 for _ in model.parameters())
    # canonical ascending keep_index, eval determinism
    for bb in range(B):
        valid = out["keep_index"][bb][out["keep_index"][bb] >= 0]
        assert (valid[1:] > valid[:-1]).all()
    model.eval()
    a = model.forward_train(video, gazing_ratio=0.3)["hard_keep"]
    c = model.forward_train(video, gazing_ratio=0.3)["hard_keep"]
    assert torch.equal(a, c)


def test_router_z_loss_penalizes_logit_magnitude():
    from autogaze.models.borissal.losses import router_z_loss

    logits = torch.randn(2, 8, 576)
    assert router_z_loss(logits + 5.0) > router_z_loss(logits)  # shift grows logsumexp^2
    l = logits.clone().requires_grad_(True)
    router_z_loss(l).backward()
    assert (l.grad.abs() > 0).float().mean() > 0.99  # reaches (essentially) all logits


def test_cosine_head_bounds_logits():
    video = _make_video(B=1, seed=40)
    model = BorissalV1(BorissalV1Config()).eval()  # cosine_scores default on
    s = model.scores(video)
    temp = model.log_score_temp.exp().clamp(max=100.0).item()
    assert s.abs().max().item() <= temp + 1e-4, "cosine head must bound |logit| by its temperature"
    # plain-conv head path (pre-WP-A behavior) still satisfies the contract
    m2 = BorissalV1(BorissalV1Config(cosine_scores=False)).eval()
    sel = m2.select(video, gazing_ratio=0.3)
    assert sel.grid_thw[0].tolist() == [8, 24, 24]


def test_coverage_floor_gates_gradient():
    pred = torch.randn(1, 4, 8, requires_grad=True)
    tgt = (pred + 0.01).detach()  # mse ~1e-4, far below floor
    loss = predictor_coverage_loss(pred, tgt, floor=1.0)
    assert loss.item() == 0.0
    loss.backward()
    assert pred.grad.abs().sum().item() == 0.0, "below the floor, coverage must exert no pressure"
    # floor=0 keeps the original pure-minimization behavior
    assert predictor_coverage_loss(pred, tgt, floor=0.0).item() > 0.0


def test_hardness_rank_loss_direction():
    from autogaze.models.borissal.losses import hardness_rank_loss

    errors = torch.rand(2, 64)
    torch.manual_seed(1)
    aligned = hardness_rank_loss(errors * 4.0, errors)   # scores rank exactly like errors
    torch.manual_seed(1)
    opposed = hardness_rank_loss(-errors * 4.0, errors)  # anti-ranked
    assert aligned < opposed
    scores = (errors * 4.0).clone().requires_grad_(True)
    hardness_rank_loss(scores, errors).backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0


def test_rloo_microbatch_smoke():
    # WP-C: REINFORCE with leave-one-out baseline against a tiny teacher.
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(__file__).resolve().parent.parent / "scripts" / "train_borissal_v1.py"
    spec = importlib.util.spec_from_file_location("_train_borissal_v1", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    teacher = VJEPA2Teacher.tiny_random(crop_size=128, frames_per_clip=16)
    model = BorissalV1(BorissalV1Config(scale=128)).train()
    video = _make_video(B=1, H=128, W=128, seed=60)
    args = SimpleNamespace(rl_samples=3, rl_cov_weight=1.0, coverage_floor=0.0)
    weights = LossWeights(score_entropy=0.01)

    out, total, logs = mod.rl_microbatch(model, teacher, False, video, 0.4, args, weights)
    assert not torch.isnan(total)
    for key in ("rl_pg", "rl_reward_mean", "rl_adv_std", "total"):
        assert key in logs
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), \
        "policy gradient must reach the selector"
    assert all(p.grad is None for p in teacher.parameters()), "teacher must stay out of the RL graph"
    # REINFORCE reaches ALL logits (log-softmax coupling), selected or not
    g = out["logits"].grad.abs()
    assert (g > 0).float().mean() > 0.99


def test_entropy_and_distill_losses_basic():
    probs_uniform = torch.full((1, 2, 8), 1 / 8)
    probs_peaky = torch.tensor([[[0.99] + [0.01 / 7] * 7] * 2])
    assert score_entropy_loss(probs_uniform) < score_entropy_loss(probs_peaky)

    logits = torch.randn(1, 2, 8)
    same = v0_distill_loss(logits, logits.clone())
    other = v0_distill_loss(logits, -logits)
    assert same < other
