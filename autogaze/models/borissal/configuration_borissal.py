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
    # --- v0.4: frame-rate-aware motion diff stride ---
    # motion_diff="frame" differences frames `motion_diff_stride` apart. int 1
    # (default) = consecutive frames = current behavior. "auto" = scale the
    # stride with the clip's frame count so the effective temporal gap is
    # constant regardless of how densely the clip was decoded -- fixes the
    # motion signal shrinking at high frame rates (16f->32f: 0.176->0.115 with
    # stride 1). At the reference frame count (motion_ref_frames) "auto" is
    # stride 1, so that operating point is unchanged.
    motion_diff_stride: Union[int, Literal["auto"]] = 1
    motion_ref_frames: int = 16
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

    # --- v0.5: coarse-cube coherence (saliency-v3.1-inspired) ---
    # score_coarsen c > 1: pool the final selection score to a 1/c grid then
    # repeat_interleave back, so each c x c block shares one score and top-k
    # keeps whole c x c CUBES -- dense coherent object chunks instead of
    # scattered fine patches. Same goal as block_size but harder (full cubes)
    # and cheaper (no separate coarse pass). Pair with block_size=1. c=1
    # (default) is a no-op. Grid dims must be divisible by c.
    score_coarsen: int = 1

    # --- v0.7 "Datdol" anchor-novelty selection (docs/borissal/design.md) ---
    # selection_mode="anchor_novelty" replaces the single-score top-k with a
    # codec-style split: motion stops being saliency and becomes "when to
    # update"; appearance is "what to represent". Budget (in score_coarsen
    # cubes) = ANCHOR pool (each spatial site once, at its best-appearance
    # tubelet) + NOVELTY pool (deviation from the clip's temporal-median
    # canonical state) + residual appearance. Requires score_coarsen > 1 and
    # uniform allocation semantics; incompatible knobs raise at select time.
    # "topk" (default) is the legacy path, bit-identical to before.
    selection_mode: str = "topk"
    anchor_fraction: float = 0.5
    """Share of the cube budget offered to the anchor pool. The pool is
    capped at Sc (one candidate per spatial site: 144 at 384/patch16/c=2), so
    the effective K_a = min(round(anchor_fraction*K_cubes), Sc) -- above
    ratio ~0.2 the cap binds and surplus flows to novelty/residual."""
    anchor_novelty_lambda: float = 0.5
    """Transit-contamination guard: anchors are ranked by A_g - lambda*N so a
    background site does NOT anchor at the moment a mover passed through it
    (that moment belongs to the novelty pool)."""
    anchor_lap_weight: float = 0.5
    """|laplacian(luma)| term in the anchor appearance score A_g -- the
    static-text/document signal (same signal static_guard injects, here as a
    ranking component instead of a score addition)."""
    novelty_shortterm_weight: float = 0.3
    """Weight of the short-term motion term (the v0.2 noise-floored frame-diff
    chain) inside N, next to the primary frame-rate-independent
    |luma - temporal_median| deviation."""
    residual_appearance_weight: float = 0.4
    """Appearance weight in the post-anchor ranking R = N + w*A_g: once
    anchors are placed, changed cubes rank by novelty and unchanged cubes by
    appearance -- this is the novelty->residual tier ordering as one
    continuous score (single exact-budget topk)."""
    signal_grid: str = "fine"
    """Where the anchor/novelty SIGNALS are computed (anchor_novelty only).
    "cube": the whole signal pipeline runs at patch_size*score_coarsen (e.g.
    32px -> 12x12) -- the chunky variant, v0_7's DEFAULT (user decision
    2026-07-28); selection unit and the final patch-16 output contract are
    unchanged. "fine": signals at the patch grid (24x24), then cube-averaged
    -- the original Datdol formulation, kept as the comparison knob. NOTE:
    the 2026-07-28 review's "coarse" rejection tested a DIFFERENT variant
    (patch32 + inherited score_coarsen=2 = 64px double-coarsened chunks) and
    does not apply to "cube"."""

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
    # v0.5: compute+apply the coherence gate at the PATCH GRID (structure
    # tensor products pooled straight to the grid, gate multiplies the pooled
    # spatial map) instead of at pixel resolution. Much cheaper (~10ms->~1ms at
    # 384^2, no upsample/pixel-multiply) with a near-identical regional gate.
    # False (default) = pixel-res path (v0.3/v0.4). Ignores coherence_kernel/
    # coherence_downsample (window = patch_size).
    coherence_at_grid: bool = False
    # Sweep TUNE (2026-07-18): pixel-res stride-1 smoothing costs ~45ms at
    # 384^2 (latency-gate FAIL); averaging the gradient PRODUCTS into ds x ds
    # blocks first is valid structure-tensor windowing at ~1/ds^2 the cost.
    coherence_downsample: int = 4

    # --- v0.6: saliency-v3.1-inspired refinements (all OFF by default) ---
    # (1) Laplacian texture gate (stage 4): suppress fine texture that is dense
    # in 2nd-derivative structure but poor in motion. R = |lap(motion)|/motion;
    # gate = sigmoid(-(R - r0)/tau). A DIFFERENT mechanism than coherence_gate
    # (structure-tensor) -- sweep them exclusively, don't stack.
    laplacian_gate: bool = False
    laplacian_gate_r0: float = 1.0
    laplacian_gate_tau: float = 0.5
    # (2) Static appearance guard (stage 6): where a tubelet is ~static (motion
    # ~0), add back appearance edge energy (|lap(luma)|) so static-informative
    # content (text, documents, held outlines) survives top-k. Regime-switched
    # by per-tubelet static weight s_t = sigmoid((thresh - m_t)/tau) on globally-
    # normalized motion; high-motion tubelets untouched. Added like a channel
    # (min-max normalized, weighted by static_guard_weight).
    static_guard: bool = False
    static_guard_weight: float = 0.5
    static_guard_thresh: float = 0.05
    static_guard_tau: float = 0.02
    # (3) center_bias is re-validated in v0.6 (existing knob above), not new code.
    # (4) Mechanical-GOP keyframe prior: the selector gets N already-decoded
    # frames with NO codec metadata, so real I-frame positions are unavailable.
    # Approximate a codec's keyframe structure from the incoming frames: a
    # periodic pseudo-keyframe every `keyframe_gop` frames PLUS soft scene-cut
    # detection (a tubelet whose luma jumps sharply off the GOP grid). Adds
    # luma-edge score there so cleaner keyframe-like frames win a bit more
    # budget. Different gate than static_guard (periodic+scene-cut vs low-motion).
    keyframe_prior: bool = False
    keyframe_gop: int = 8
    keyframe_weight: float = 0.5         # appearance-edge SCORE boost at keyframes
    keyframe_alloc_boost: float = 1.0    # extra token ALLOCATION share at keyframes (uniform only)
    keyframe_scene_thresh: float = 2.0  # scene-cut fires above this x mean luma jump
    keyframe_scene_tau: float = 0.5
    # (5) Luma conversion (saliency-v3.1 stage 1): "mean" = plain channel mean
    # (v0.3-v0.5); "bt601" = 0.299R+0.587G+0.114B perceptual weights (emphasize
    # green/structure, classic CV grayscale). Feeds ALL signals (motion, spatial,
    # static guard, keyframe) since they derive from this luma.
    luma_mode: Literal["mean", "bt601"] = "mean"

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
            # Allocation-policy comparison (2026-07-19, count-agnostic semantic
            # metric, 16 clips): UNIFORM beats global+floor decisively at
            # ratio 0.5 (0.603 vs 0.584, 12 of 16 clips) and ties at 0.25;
            # with the v0.3 signal stack uniform also wins UNIQUENESS
            # (8.191 vs 8.133, cov ~tie) -- the v0.2-era "global+floor
            # strongest" finding does not carry over. Uniform is additionally
            # the trace/export-safe (data-independent) policy per the mobile
            # review. global/proportional/floor/cap stay as control knobs.
            per_frame_allocation="uniform",
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

    @classmethod
    def v0_4(cls, **overrides) -> "BorissalConfig":
        """The Borissal v0.4 preset: v0.3 + frame-rate-aware motion diff
        (`motion_diff_stride="auto"`). v0.3's motion signal shrinks when a clip
        is decoded to more frames (adjacent frames become more similar; measured
        16f->32f: |frame diff| 0.176 -> 0.115, a 35% drop that the noise floor
        then eats further), so selection under-covers action at high frame
        rates -- the observed "32-frame downstream is worse than expected".
        v0.4 scales the diff stride with frame count so the motion magnitude is
        constant regardless of decode density. AT THE 16-FRAME REFERENCE v0.4 IS
        BIT-IDENTICAL TO v0.3 (auto stride = 1 there); it only changes non-16f
        inputs. v0.3 stays the frozen 16f-validated baseline. Token-redundancy
        at high frame counts is a separate, downstream-side concern (frame count
        should track a clip's temporal content, not the token budget)."""
        base = dict(motion_diff_stride="auto")
        base.update(overrides)
        return cls.v0_3(**base)

    @classmethod
    def v0_5(cls, **overrides) -> "BorissalConfig":
        """The Borissal v0.5 preset: v0.3 + coarse-cube coherence
        (`score_coarsen=2`, replacing the block gate with `block_size=1`).

        Motivated by two findings: (1) for a downstream whose encoder is itself
        a temporal/video model (V-JEPA), the selector should preserve coherent
        object/appearance regions rather than scattered patches -- the encoder
        already supplies motion; and (2) the saliency-v3.1 mechanism (compute
        score at 12x12, repeat_interleave to 24x24) enforces dense c x c token
        cubes, which a captioner grounds objects from more reliably. v0.5 makes
        that cube coherence the core selection prior. Built on v0.3 (NOT v0.4,
        which pushed toward motion -- the wrong direction for this downstream).

        motion_weight DEFAULTS TO "auto" (per-clip motion/appearance energy ratio;
        32f mean ~0.34, static clips ~0.03) so the appearance-vs-motion balance
        self-adapts to frame rate and scene. Still validate against the real
        downstream (caption -> QA), not the
        SigLIP-recall proxy (which mispredicted the v0.4 regression). Candidates:
        {0.5, 0.35, 0.25, 0.15, 0.0, "auto"} -- lower / auto emphasize static
        appearance (objects, text) over motion."""
        base = dict(score_coarsen=2, block_size=1, motion_weight="auto",
                    coherence_at_grid=True, spatial_diff="tubelet")
        base.update(overrides)
        return cls.v0_3(**base)

    @classmethod
    def v0_6(cls, **overrides) -> "BorissalConfig":
        """The Borissal v0.6 preset: v0.5 + three saliency-v3.1-inspired knobs,
        ALL OFF by default (so a plain `v0_6()` is bit-identical to `v0_5()` --
        the additions are opt-in and sweep-gated).

        saliency-v3.1 (the user's downstream-validated best) contributed three
        things v0.5 lacked, confirmed against the v0.3-v0.5 stack:
        - `static_guard`: regime-switched static appearance guard. v0.5 blends
          appearance globally (motion_weight); this instead injects |lap(luma)|
          edge energy ONLY where a tubelet is static, so text/documents/held
          shots survive top-k. Aligns with the v0.4-regression lesson (this
          downstream wants appearance, not motion).
        - `laplacian_gate`: Laplacian-to-motion texture suppression -- a
          different mechanism than the structure-tensor `coherence_gate`; sweep
          them exclusively, never stacked.
        - `center_bias`: re-validated (existing knob) -- saliency-v3.1 ships it
          as a winner, but it has been off/untested since v0.2. Enable via
          override to sweep it in the v0.5 signal stack.

        DEFAULT = all three ON (2026-07-24 decision), matching saliency-v3.1's
        own configuration -- its downstream-validated success is the evidence,
        chosen OVER the Mac proxy screen (where laplacian_gate/center_bias
        regressed recall; static_guard won). The proxy is not the arbiter; the
        real caption->QA on CUDA is, and saliency-v3.1 runs all three. To recover
        the exact v0.5 behavior, disable them explicitly:
        `BorissalConfig.v0_6(static_guard=False, laplacian_gate=False, center_bias=0.0,
        keyframe_prior=False, per_frame_allocation="uniform")`.

        ALLOCATION (2026-07-24): v0.6 defaults to `per_frame_allocation="global"`
        (clip-wide top-K + per-tubelet floor) -- CONTENT-ADAPTIVE, matching
        saliency-v3.1's stage-7 (clip-wide top-K + min-1 guarantee), NOT v0.5's
        uniform. Track B found uniform > global on the SigLIP proxy, but
        saliency-v3.1 (global) is much better downstream, so the proxy mis-ranked
        allocation too; global lets the signal boosts (keyframe/static/center)
        flow into content-adaptive per-tubelet counts. Also enables the keyframe
        prior by default (all newly-introduced features on)."""
        base = dict(static_guard=True, laplacian_gate=True, center_bias=0.3,
                    keyframe_prior=True, per_frame_allocation="global",
                    luma_mode="bt601")
        base.update(overrides)
        return cls.v0_5(**base)

    @classmethod
    def v0_7(cls, **overrides) -> "BorissalConfig":
        """The Borissal v0.7 "Datdol" preset: anchor-novelty selection for a
        TEMPORAL downstream encoder (V-JEPA family) + captioner.

        Architectural change, not a knob combo: motion and appearance no
        longer compete inside one saliency score. The cube budget splits into
        an ANCHOR pool (each spatial site selected once, at the tubelet where
        its appearance is best -- a temporal encoder does not need static
        content re-selected every tubelet), a NOVELTY pool (deviation from
        the clip's temporal-median canonical state -- frame-rate independent,
        unlike consecutive-frame diffs), and a residual appearance tier that
        absorbs surplus budget at high ratios (natural multi-anchor).

        Built on v0.5's appearance stack (cube coherence, grid-res coherence
        gate, DoG region channel) with three explicit pins that the mode
        depends on: motion_weight=0.0 (the auto blend is the mechanism this
        design REMOVES -- motion feeds only the novelty pool), block_size=1
        (cube coherence owns spatial grouping), and bt601 luma (all signals
        derive from it). Allocation is architecture-owned: per-tubelet counts
        FOLLOW from where anchors/novelty land, so per_frame_allocation
        stays "uniform" only as the no-op placeholder and incompatible knobs
        (spread/hysteresis/keyframe_prior/block gate/per_frame_counts) raise
        at select time rather than silently composing."""
        base = dict(
            selection_mode="anchor_novelty",
            luma_mode="bt601",
            motion_weight=0.0,
            block_size=1,
            per_frame_allocation="uniform",
            signal_grid="cube",     # 12x12-native signals (user decision 2026-07-28)
        )
        base.update(overrides)
        return cls.v0_5(**base)


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
