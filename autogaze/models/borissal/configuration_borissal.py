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

    # --- hybrid focus+spread allocation (2026-07-15, description-task review) ---
    # spread_fraction s > 0 dedicates round(s*K) of the budget to a
    # stratified spatio-temporal skeleton (best cell of each of the top
    # buckets; time stratified first), the rest to plain score top-k.
    # The single-scale analogue of AutoGaze's multi-scale coarse share
    # (~26% of its per-frame tokens). 0 = pure top-k (pre-existing
    # behavior). Applies to uniform (per-tubelet 2D buckets) and global
    # (clip-wide 3D buckets) allocation; incompatible with "proportional".
    spread_fraction: float = 0.0

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

    # --- v0.3 candidate bank (docs/borissal/v03-design.md). ALL OFF by
    # default: with every knob at its default the pipeline takes the legacy
    # blend path and is bit-identical to v0.2 (regression-tested). Adoption
    # into a preset happens only through the sweep gates
    # (scripts/sweep_borissal_v03.py); until then these are experimental.
    motion_center_surround: bool = False
    """relu(D - avgpool(D)): cancels uniform ego-motion diff fields (pan/zoom),
    keeps independently moving objects. Runs after pooling, before the noise
    floor."""
    motion_cs_kernel: int = 9

    coherence_gate: bool = False
    """Multiply the gradient channel by (1 - structure-tensor coherence)^gamma:
    suppresses repetitive gratings / long straight edges, spares
    multi-orientation object micro-structure. Pixel-res, closed form."""
    coherence_kernel: int = 5
    coherence_gamma: float = 1.0
    # Sweep TUNE (2026-07-18): pixel-res stride-1 smoothing costs ~45ms at
    # 384^2 (latency-gate FAIL); averaging the gradient PRODUCTS into ds x ds
    # blocks first is valid structure-tensor windowing at ~1/ds^2 the cost.
    coherence_downsample: int = 4

    signature_weight: float = 0.0
    """Image-signature (sign-of-DCT, fixed matmul) appearance channel weight;
    0 = off. Fires on spatially sparse foreground support."""
    color_rarity_weight: float = 0.0
    """Global color-rarity (soft-binned histogram contrast) channel weight;
    0 = off. First use of color in the v0 line; grid-resolution only.
    Heavy-tailed: sqrt-compressed and clip-globally normalized."""
    color_bins_per_axis: int = 3
    color_bin_sigma: float = 0.15
    dog_blob_weight: float = 0.0
    """Multi-scale difference-of-boxes blob channel weight; 0 = off."""

    fusion_norm: Literal["none", "peak", "entropy"] = "none"
    """Content-adaptive per-channel fusion weighting: "peak" = Itti N(.)
    (maps with one decisive peak promoted, everywhere-firing maps demoted),
    "entropy" = bounded inverse-entropy gate (free camera-pan fallback:
    a flooded motion map loses fusion weight)."""
    fusion_entropy_floor: float = 0.3

    # --- v0.3.x follow-up candidates (2026-07-19 review; sweep-gated like the
    # rest of the bank unless marked behavior-preserving) ---
    block_gate_source: Literal["recompute", "pool"] = "recompute"
    """"recompute" (v0.2 behavior): the coarse block-gate signal reruns the
    whole saliency pipeline on a 1/b-resized clip. "pool" (candidate):
    block-pool the fine per-patch scores instead -- one pipeline pass, ~5ms
    cheaper; the recompute path's resize-low-pass noise suppression is
    largely redundant once the coherence gate is on."""
    spatial_diff: Literal["tubelet", "frame"] = "tubelet"
    """Granularity of the spatial/edge signal, mirroring motion_diff:
    "tubelet" differentiates the 2-frame mean (slight motion blur);
    "frame" measures each raw frame then aggregates per tubelet."""
    spatial_agg: Literal["mean", "max"] = "mean"
    """Aggregation for spatial_diff="frame": "max" keeps detail that is
    sharp in at least one frame of the tubelet."""
    max_keep_per_frame_mult: float = 0.0
    """Global-allocation per-tubelet CAP as a multiple of the uniform share
    (0 = off). E.g. 2.0: no tubelet may take more than 2x its uniform share
    of K_total -- bounds free-budget monopolization symmetric to the
    min_keep_per_frame_ratio floor. Clamped so the exact budget stays
    feasible (cap >= ceil(K_total / T_grid) and >= the floor m)."""

    score_ema_alpha: float = 0.0
    """Temporal score EMA over tubelets (0 = off): S_t = a*S_{t-1} + (1-a)*S_t,
    loop-free within a clip; streaming carries one state map via
    select(..., temporal_state=...)."""
    select_hysteresis_eps: float = 0.0
    """Pre-topk additive bonus for patches kept in the previous tubelet
    (0 = off). One-step vectorized approximation (bonus from the
    pre-hysteresis selection, not the recursive chain)."""

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

    @classmethod
    def v0_3(cls, **overrides) -> "BorissalConfig":
        """The Borissal v0.3 preset: v0.2 + the three sweep-gate winners
        (2026-07-19 solo/greedy screening, docs/borissal/design.md "v0.3"
        sections): content-adaptive peak fusion (Itti N(.)), the structure-
        tensor coherence texture gate (products-then-pool ds=4 TUNE), and the
        DoG blob interior channel. Held-out 16-clip semantic recall
        0.325 -> 0.346-0.351 (ratio 0.25); V-JEPA coverage/uniqueness
        Pareto-better than v0.2 (8.174/8.167 vs 8.238/8.106, 4 clips); CPU
        latency ~24.5ms against the 25ms budget. Rejected candidates
        (motion_center_surround, signature, color_rarity, fusion "entropy",
        score_ema, select_hysteresis) remain available as individual
        off-by-default knobs -- verdicts and negative results in design.md."""
        base = dict(
            fusion_norm="peak",
            coherence_gate=True,
            dog_blob_weight=0.5,
            # v0.3.x follow-up round (2026-07-19): pooled block gate --
            # identical selections to the recompute path on the whole held-out
            # set at 384, ~7ms cheaper (single pipeline pass); frame-granular
            # spatial with max aggregation -- +0.003 recall (8W-6L-2T, the
            # weakest accepted margin: re-check at scale), +5ms with the
            # tubelet-granular gate TUNE.
            block_gate_source="pool",
            spatial_diff="frame",
            spatial_agg="max",
        )
        base.update(overrides)
        return cls.v0_2(**base)


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

    # GCNet-lite learned global context (2026-07-15 model review): the plain
    # TSM stack's receptive field is ~9x9 grid cells (measured: far-field
    # perturbation response 200x weaker than local), but coverage/uniqueness
    # objectives ask a CLIP-GLOBAL question ("how does this patch compare to
    # everything else?"). A learned weighted pooling (GCNet, arXiv:1904.11492)
    # provides that signal at O(L) cost: one 1x1 attention conv + softmax +
    # zero-init 1x1 transform, injected before the LAST TSM block so its
    # conv+GELU mixes local x global per position. Mobile-safe (conv/softmax/
    # mul-sum only). False = pre-review local-only behavior.
    global_context: bool = True

    gazing_ratio: float = 0.5
    per_frame_allocation: Literal["uniform", "proportional", "global"] = "uniform"
    min_keep_per_frame_ratio: float = 0.25  # global mode floor (mirrors BorissalConfig)
    spread_fraction: float = 0.0  # hybrid focus+spread allocation (mirrors BorissalConfig)

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
