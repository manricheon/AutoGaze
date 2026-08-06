"""Composable SSL losses for Borissal v1 training (Phase 3).

Which SSL objective / loss combination trains a good selector without
ground-truth gazing labels is a core contribution of this line of work --
see docs/borissal/training.md for the rationale behind each term and the
recommended combinations. Torch-only.

All terms return scalar tensors; `combine_losses` applies config weights and
returns (total, {name: detached value}) for logging. Terms with weight 0 are
skipped entirely (no wasted compute).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    predictor_coverage: float = 1.0
    dense_sparse_match: float = 0.0
    score_entropy: float = 0.0
    v0_distill: float = 0.0
    # v0_distill is intended as a warmup: linearly decayed to 0 over this many
    # steps by the trainer (0 = no decay schedule).
    v0_distill_warmup_steps: int = 0
    # Anti-scatter reward (see design.md "Borissal v0.2" Finding 1: pure
    # coverage pressure favors uniform scatter): rewards selections holding
    # information the REST cannot reconstruct. Costs one extra predictor pass
    # when nonzero.
    uniqueness_reward: float = 0.0
    # ST-MoE router z-loss (arXiv:2202.08906): penalizes logit MAGNITUDE at
    # the source, preventing the softmax saturation / "rich-get-richer"
    # lock-in observed in local trend runs (entropy 5.0 -> 2.6, dying grads).
    # 1e-3 is the published coefficient (3/3 unstable runs stabilized WITH a
    # quality gain in the ST-MoE ablation).
    z_loss: float = 1e-3
    # HPM-style hardness ranking (arXiv:2304.05919): score head learns to
    # RANK unselected tokens by the predictor's per-token error (pairwise
    # BCE -- ranking, not absolute MSE, is immune to loss-scale drift).
    hardness_rank: float = 0.0


def predictor_coverage_loss(
    predicted_targets: torch.Tensor,  # (B, K_t, D) predictor output at unselected positions
    teacher_targets: torch.Tensor,    # (B, K_t, D) dense-teacher features at those positions
    floor: float = 0.0,
) -> torch.Tensor:
    """Information-coverage objective: the selected patches, and only them,
    must suffice to reconstruct the features of everything NOT selected.
    Lower = the selection covers the clip's information better.

    floor > 0 turns this from a MAXIMIZED objective into a CONSTRAINT:
    relu(mse - floor) -- zero loss (and zero gradient) once coverage is good
    enough. Rationale (design.md theory notes): coverage MINIMIZATION is a
    facility-location objective whose optimum is uniform scatter + boundary
    anchors -- exactly the measured P1/P3 pathologies -- so below the floor
    the scatter pressure must be switched off and another term (uniqueness /
    hardness) should drive selection."""
    mse = F.mse_loss(predicted_targets, teacher_targets)
    if floor > 0:
        mse = F.relu(mse - floor)
    return mse


def dense_sparse_match_loss(
    sparse_feats: torch.Tensor,   # (B, K, D) student features at selected positions
    teacher_dense: torch.Tensor,  # (B, L, D)
    keep_index: torch.Tensor,     # (B, K)
) -> torch.Tensor:
    """Optional auxiliary: selected tokens' sparse-context features should stay
    close to their dense-context features. NOTE the documented degeneracy risk
    (it can favor context-independent background patches); use only as a
    small-weight regularizer alongside predictor_coverage, never alone."""
    d_at = torch.gather(teacher_dense, 1, keep_index.unsqueeze(-1).expand(-1, -1, teacher_dense.size(-1)))
    return F.mse_loss(sparse_feats, d_at)


def score_entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    """Regularizer on the per-tubelet score distribution (B, T_grid, N_pf).
    Positive weight PENALIZES low entropy (fights early collapse onto a few
    patches); the trainer may anneal it to 0."""
    entropy = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)  # (B, T_grid)
    max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device))
    return (max_entropy - entropy).mean()  # 0 when uniform, grows as it peaks


def uniqueness_reward_loss(
    rest_predicted_selected: torch.Tensor,  # (B, K, D) predictor(unselected -> selected positions)
    teacher_selected: torch.Tensor,         # (B, K, D) dense-teacher features at selected positions
    cap: float = 20.0,
) -> torch.Tensor:
    """Anti-scatter term: NEGATIVE of the error the rest makes when trying to
    reconstruct the selection -- minimizing it maximizes how much unique
    (rest-unexplainable) information the selection holds. Capped so the
    reward cannot diverge by selecting degenerate/unpredictable content."""
    return -torch.clamp(F.mse_loss(rest_predicted_selected, teacher_selected), max=cap)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE z-loss (arXiv:2202.08906, eq. 5): mean over softmax instances of
    logsumexp(logits)^2. Softmax is shift-invariant but its SATURATION is not:
    once logits grow large the distribution peaks and every soft-path gradient
    dies (the P2 pathology). This term shrinks logit magnitude at the source
    without touching relative preferences near the origin.
    logits: (B, T_grid, N_pf) -- one softmax instance per tubelet."""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


def hardness_rank_loss(
    scores_rest: torch.Tensor,  # (B, R) selector logits at UNSELECTED positions
    errors_rest: torch.Tensor,  # (B, R) per-token predictor error at those positions (detached)
) -> torch.Tensor:
    """HPM-style auxiliary (CVPR 2023, arXiv:2304.05919): the score head
    learns to rank tokens by how HARD the predictor finds them -- hard-to-
    predict = discriminative content worth selecting. Pairwise BCE against a
    random within-batch pairing; ranking (not absolute regression) because
    the predictor's error scale drifts over training. Free signal: the
    per-token errors fall out of the coverage pass."""
    errors = errors_rest.detach()
    perm = torch.randperm(scores_rest.shape[-1], device=scores_rest.device)
    margin = scores_rest - scores_rest[:, perm]
    target = (errors > errors[:, perm]).to(scores_rest.dtype)
    per_pair = F.binary_cross_entropy_with_logits(margin, target, reduction="none")
    valid = (errors != errors[:, perm]).to(scores_rest.dtype)  # skip exact ties
    return (per_pair * valid).sum() / valid.sum().clamp_min(1.0)


def v0_distill_loss(
    logits: torch.Tensor,     # (B, T_grid, N_pf) v1 score logits
    v0_scores: torch.Tensor,  # (B, T_grid, N_pf) v0 combined saliency scores
) -> torch.Tensor:
    """Warmup: mimic v0's score ranking (KL between per-tubelet softmax
    distributions) so v1 starts from the proven saliency prior instead of
    random selection."""
    return F.kl_div(
        logits.log_softmax(dim=-1),
        v0_scores.softmax(dim=-1),
        reduction="batchmean",
    )


def combine_losses(
    weights: LossWeights,
    step: int,
    terms: dict,  # name -> zero-arg callable returning a scalar tensor (lazy)
) -> tuple:
    """Weighted sum of lazily-evaluated loss terms. Returns (total, logs)."""
    total = None
    logs = {}
    for name, get_value in terms.items():
        w = getattr(weights, name)
        if name == "v0_distill" and w > 0 and weights.v0_distill_warmup_steps > 0:
            w = w * max(0.0, 1.0 - step / weights.v0_distill_warmup_steps)
        if w == 0:
            continue
        value = get_value()
        logs[name] = value.detach().item()
        total = w * value if total is None else total + w * value
    if total is None:
        raise ValueError("all loss weights are zero")
    logs["total"] = total.detach().item()
    return total, logs
