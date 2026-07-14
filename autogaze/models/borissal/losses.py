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


def predictor_coverage_loss(
    predicted_targets: torch.Tensor,  # (B, K_t, D) predictor output at unselected positions
    teacher_targets: torch.Tensor,    # (B, K_t, D) dense-teacher features at those positions
) -> torch.Tensor:
    """Information-coverage objective: the selected patches, and only them,
    must suffice to reconstruct the features of everything NOT selected.
    Lower = the selection covers the clip's information better."""
    return F.mse_loss(predicted_targets, teacher_targets)


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
