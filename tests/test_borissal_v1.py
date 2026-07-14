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


def test_entropy_and_distill_losses_basic():
    probs_uniform = torch.full((1, 2, 8), 1 / 8)
    probs_peaky = torch.tensor([[[0.99] + [0.01 / 7] * 7] * 2])
    assert score_entropy_loss(probs_uniform) < score_entropy_loss(probs_peaky)

    logits = torch.randn(1, 2, 8)
    same = v0_distill_loss(logits, logits.clone())
    other = v0_distill_loss(logits, -logits)
    assert same < other
