"""Configuration for Borissal v0, the non-learned feed-forward patch selector (Phase 1)."""

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class BorissalConfig:
    """Config for the non-learned saliency selector.

    grid_thw for a clip of shape (T, H, W) is derived as
    (T // tubelet_size, H // patch_size, W // patch_size).
    """

    scale: int = 384
    patch_size: int = 16
    tubelet_size: int = 2

    motion_weight: Union[float, Literal["auto"]] = 0.5
    """Fixed blend weight in [0, 1], or "auto" to derive it per-clip from the
    clip's own (pre-normalization) motion vs. spatial energy -- still
    non-learned, just data-adaptive: motion_weight = motion_energy /
    (motion_energy + spatial_energy)."""
    spatial_op: Literal["grad", "sobel"] = "grad"
    pooling: Literal["avg", "max"] = "avg"

    gazing_ratio: float = 0.5
    # "global" (v0.2): a single clip-wide top-K_total = round(ratio*L) with a
    # guaranteed per-tubelet minimum -- budget concentrates where the action
    # is while every tubelet keeps some temporal context. Pair with
    # score_norm_blend < 1 so cross-tubelet scores are comparable.
    per_frame_allocation: Literal["uniform", "proportional", "global"] = "uniform"
    # global mode: each tubelet is guaranteed at least
    # round(min_keep_per_frame_ratio * K_total / T_grid) patches.
    min_keep_per_frame_ratio: float = 0.25

    # --- v0.2: local/global score-normalization blend ---
    # 1.0 = pure per-tubelet min-max (v0.1: every tubelet competes equally,
    # cross-tubelet magnitude erased). < 1 blends in a clip-global min-max
    # component so high-energy tubelets rank higher clip-wide:
    # S = blend*local + (1-blend)*global.
    score_norm_blend: float = 1.0

    # --- v0.2: center bias (conditional knob, OFF by default) ---
    # Additive Gaussian center prior: S += center_bias * G. The classical
    # center prior from saliency benchmarks; content-dependent, enable
    # per-domain only when composition warrants it.
    center_bias: float = 0.0

    # --- v0.2: motion differencing granularity ---
    # "tubelet" (v0.1): tubelet means are differenced -- |mean(f2,f3)-mean(f0,f1)|.
    # "frame": consecutive FRAMES are differenced, then aggregated per tubelet
    # (frame_diff_agg) -- catches fast intra-tubelet motion that tubelet
    # averaging cancels (e.g. oscillation), at the cost of T-1 diffs vs T_grid-1
    # (still pure slice ops, fully vectorized).
    motion_diff: Literal["tubelet", "frame"] = "tubelet"
    frame_diff_agg: Literal["mean", "max"] = "mean"
    # Consistency penalty (v0.2): temporal double-difference (min of adjacent
    # frame diffs, classic three-frame differencing). What it suppresses,
    # verified empirically: (a) motion GHOSTING -- the leading/trailing
    # revealed-background diffs around an appearing/disappearing object, and
    # (b) temporally-uncorrelated pixel noise (min of two independent |diff|
    # draws is ~half the single-draw expectation). It does NOT remove a
    # genuine single-frame event at its own frame (appear+disappear both
    # produce large adjacent diffs there). KNOWN LIMITATION (found in
    # synthetic testing): an UNTEXTURED uniform-color object moving further
    # than its own edge strip per frame has non-overlapping adjacent diff
    # strips, so the min suppresses it entirely -- real textured objects
    # produce persistent interior diffs and survive. Off by default; gate it
    # per-domain. Requires motion_diff="frame".
    motion_consistency: Literal["none", "double_diff"] = "none"

    # --- v0.2: noise suppression (defaults preserve v0.1 behavior exactly) ---
    # Robust per-tubelet dead-zone shrinkage of the pooled motion map, applied
    # BEFORE min-max normalization and the "auto" weight energy (otherwise
    # normalization re-amplifies the very noise floor this removes).
    motion_noise_floor: Literal["none", "mean", "quantile"] = "none"
    motion_noise_q: float = 0.5       # quantile mode: tau = per-tubelet q-quantile (0.5 = median)
    motion_noise_scale: float = 1.0   # motion_p = relu(motion_p - scale * tau)
    # Optional pixel-level box blur of the motion map before pooling (0 = off).
    motion_smooth_kernel: int = 0

    # --- v0.2: coherent-region (resize-based coarse-to-fine) selection ---
    # b > 1: a low-resolution saliency pass (same pipeline on a 1/b-resized clip)
    # picks top-ceil(k/b^2) blocks, then the fine pass top-k's within them.
    block_size: int = 1

    eps: float = 1e-6

    image_mean: tuple = field(default_factory=lambda: (0.485, 0.456, 0.406))
    image_std: tuple = field(default_factory=lambda: (0.229, 0.224, 0.225))

    @classmethod
    def v0_2(cls, **overrides) -> "BorissalConfig":
        """The tuned Borissal v0.2 preset, finalized against the
        coverage/uniqueness gate (scripts/eval_borissal_coverage.py; numbers
        in docs/borissal/design.md): frame-diff motion + noise floor +
        global allocation with per-tubelet floor + local/global score blend
        + coherent-region gate. Pareto-best at ratio 0.25 on the gate (both
        metrics beat v0.1). Caveat: at very low ratios (<~0.1) the block
        gate over-constrains -- set block_size=1 there. Excluded after
        measurement: motion_consistency (normalization cancels its benefit),
        center_bias (per-domain only). Plain BorissalConfig() defaults stay
        v0.1-identical."""
        base = dict(
            motion_diff="frame",
            motion_noise_floor="quantile",
            motion_noise_q=0.5,
            motion_noise_scale=1.0,
            per_frame_allocation="global",
            score_norm_blend=0.7,
            block_size=2,
        )
        base.update(overrides)
        return cls(**base)


@dataclass
class BorissalV1Config:
    """Config for Borissal v1, the learned selector (Phase 2).

    Shares v0's grid semantics: grid_thw = (T // tubelet_size,
    H // patch_size, W // patch_size). The v0 saliency signal is computed
    internally (non-learned, cheap) whenever `input_mode` includes maps or
    `residual_scoring` is on.
    """

    scale: int = 384
    patch_size: int = 16
    tubelet_size: int = 2

    # What the learned backbone sees, at grid resolution (H_grid x W_grid):
    #   "maps"   -- v0's normalized motion+spatial patch maps only (2 channels)
    #   "pixels" -- grid-downsampled RGB of the tubelet mean only (3 channels)
    #   "both"   -- concat of the two (5 channels)
    input_mode: Literal["maps", "pixels", "both"] = "both"

    hidden_channels: int = 64
    num_blocks: int = 3
    # Fraction of channels shifted forward/backward one tubelet per TSM block.
    shift_fraction: float = 0.25

    # score = v0_score + f_theta(...) instead of score = f_theta(...).
    residual_scoring: bool = False

    # Cosine score head (X-MoE, arXiv:2204.09179): logits = normalized-feature
    # dot normalized-weight, scaled by a LEARNABLE temperature -- bounds logit
    # magnitude by construction, preventing the logit-norm arms race behind
    # score saturation (P2). False = plain 1x1-conv head (pre-WP-A behavior).
    cosine_scores: bool = True

    gazing_ratio: float = 0.5
    per_frame_allocation: Literal["uniform", "proportional", "global"] = "uniform"
    min_keep_per_frame_ratio: float = 0.25  # global mode floor (mirrors BorissalConfig)

    # Training-time block-structured selection (WP-B): b > 1 selects at
    # b x b spatial-block granularity in forward_train (block-mean logits,
    # block-level Gumbel-top-k, gate expanded back to tokens). Rationale:
    # scattered per-token selection is the provable optimum of coverage-style
    # objectives AND deep off-distribution for the multi-block-trained
    # V-JEPA predictor (I-JEPA ablation: scattered 17.6 vs blocks 54.2) --
    # block geometry removes the scatter shortcut by construction. Inference
    # `select()` is unaffected (v0.2's coarse-to-fine gate covers that side).
    train_block_size: int = 1

    # Softmax temperature of the straight-through Gumbel-top-k training path
    # (the canonical Concrete/Gumbel-softmax tau; 2/3 is the literature default
    # -- Maddison et al. 2017 -- and values below 0.5 are the documented
    # gradient-variance blowup zone, so no annealing). 0 disables Gumbel noise
    # AND the temperature (plain softmax backward).
    gumbel_tau: float = 2.0 / 3.0

    eps: float = 1e-6

    # v0 signal settings (used for maps input / residual scoring).
    # v0_preset selects which non-learned signal generation feeds the learned
    # scorer: "v0.2" (default -- frame-diff motion + noise floor etc., the
    # gate-validated preset) or "v0.1" (plain baseline signals).
    v0_preset: Literal["v0.1", "v0.2"] = "v0.2"
    motion_weight: Union[float, Literal["auto"]] = 0.5
    spatial_op: Literal["grad", "sobel"] = "grad"
    pooling: Literal["avg", "max"] = "avg"
