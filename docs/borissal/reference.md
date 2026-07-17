# Borissal v0: Reference

A saliency-based, feed-forward patch selector. Non-learned, single-scale,
top-k under a fixed budget — with the per-frame share of that budget either
uniform or dynamically reallocated to each frame's own saliency energy
(`per_frame_allocation`, §3). This is the "what and why" reference; see
[`design.md`](./design.md) for the full engineering rationale and
[`progress.md`](./progress.md) for the session-by-session log.

## 1. Why saliency

```
video ──▶ Borissal (top-k select) ──▶ keep_index / grid_thw ──▶ vision encoder ──▶ LLM ──▶ description
```

A description model doesn't need every patch of every frame — it needs the
patches that carry *information*: a subject moving, a hand gesturing, a line
of on-screen text, the edge of an object entering the frame. Most of a raw
clip is redundant relative to that goal — static background, repeated
texture, frames that barely differ from their neighbors. Feeding all of it
through the encoder and LLM spends compute describing nothing.

This is the same problem video codecs solved decades ago: don't re-encode
every pixel of every frame — encode what *changed* (motion vectors, residuals)
and let flat, unchanging regions cost almost nothing. Borissal borrows that
intuition without needing an actual codec in the loop: it decodes ordinary
RGB frames and computes two cheap, codec-flavored proxies directly —
**motion** (how much a patch changed between frames, a residual/motion-vector
stand-in) and **spatial edge energy** (how much detail a patch holds, right
now). Together they approximate "what would a codec bother to encode" —
and, not coincidentally, approximate "what would be worth describing."

**Why feed-forward, single-scale, top-k — not AutoGaze's approach.**
AutoGaze (this repo's original model) is autoregressive and multi-scale: it
generates patch indices one at a time with a learned decoder. That's
powerful, but it's sequential and comparatively expensive. Borissal is the
other end of the design space on purpose: one forward pass, one resolution,
a hard top-k cut against an explicit `gazing_ratio` budget — because the
priorities here are latency, a deterministic compute budget, and eventual
on-device (mobile) execution, not maximum selection quality per se. See
[`design.md`](./design.md#mobile-readiness-review-2026-07-14-before-starting-phase-2)
for the mobile-specific tradeoffs this drove.

**Where this fits in the bigger picture.** Borissal v0 is Phase 1 of a
three-phase line: a learned selector (Borissal v1, Phase 2) will replace
this hand-built saliency score with a trainable one, and v0's motion/spatial
maps are the natural starting point — either as literal input features or
simply as the baseline it needs to beat. Phase 3 trains that learned
selector self-supervised against V-JEPA2 (dense-vs-sparse feature
comparison). None
of that changes what's described below; it's the reason this exists.

## 2. Algorithm

Given a clip `(B, T, C, H, W)`:

1. **Luma** — average channels: `gray = video.mean(dim=2)`.
2. **Tubelet aggregation** — average every `tubelet_size` consecutive frames
   into one tubelet: `(B, T_grid, H, W)`, `T_grid = T // tubelet_size`.
3. **Motion** — temporal differencing (a residual/motion-vector proxy,
   computed on decoded pixels rather than a real codec's MV field). Two
   granularities (`motion_diff`): `"tubelet"` differences the tubelet
   means; `"frame"` differences consecutive raw frames then aggregates per
   tubelet (`frame_diff_agg: mean|max`) — catches fast intra-tubelet motion
   that tubelet averaging cancels. An experimental `motion_consistency=
   "double_diff"` (temporal min of adjacent frame diffs) exists but is NOT
   in the preset — see the negative result in `design.md`.
4. **Spatial** — gradient magnitude (`spatial_op="grad"`, cheap finite
   differences) or a fixed Sobel kernel (`spatial_op="sobel"`).
5. **Patch pooling** — average- or max-pool both maps down to the patch grid
   `(H_grid, W_grid) = (H, W) // patch_size`.
6. **Noise floor** (v0.2, `motion_noise_floor`) — robust per-tubelet
   dead-zone shrinkage of the pooled motion map:
   `motion = relu(motion − scale·τ)`, τ = per-tubelet mean or a
   topk-computed quantile (median default). Runs BEFORE normalization
   (which would otherwise re-amplify the noise floor to full range in
   motionless tubelets) and before the `"auto"` weight energies.
7. **Normalize & combine** — min-max normalize motion and spatial per
   (instance, tubelet), blend with `motion_weight` (float or `"auto"`).
   With `score_norm_blend < 1` (v0.2) a clip-GLOBAL min-max component is
   mixed in: `S = β·S_local + (1−β)·S_global` — local keeps every tubelet
   internally comparable, global preserves which tubelets carry more
   energy clip-wide.
8. **Center bias** (v0.2, `center_bias`, off by default) — additive
   Gaussian center prior at grid resolution; enable per-domain.
9. **Budget allocation** — turn `gazing_ratio` into patch counts:
   `uniform` (same count every tubelet; exact ratio always), `proportional`
   (counts follow per-tubelet score energy), or `global` (v0.2: one
   clip-wide top-K_total with a guaranteed per-tubelet minimum — budget
   concentrates on high-information moments while the floor preserves
   temporal coverage; pair with `score_norm_blend < 1`).
10. **Coherent-region gate** (v0.2, `block_size=b>1`) — a coarse pass runs
    the SAME saliency pipeline on a 1/b-resized clip; its top-⌈k/b²⌉ blocks
    per tubelet gate the fine selection (out-of-gate scores masked to
    `finfo.min`). Gate capacity ⌈k/b²⌉·b² ≥ k keeps exact budgets intact;
    fragmentation is hard-bounded to ≤⌈k/b²⌉ regions per tubelet.
11. **Top-k → keep mask** — `torch.topk` (chosen over sort/argsort ranks,
    which have weaker mobile-runtime operator support).
12. **Pack to global indices** — flatten to `keep_index`/`keep_coords` in
    `(t, h, w)` grid_thw space via a stable-sort packer (`_pack_gazing_mask`,
    self-contained — see §7) that also guarantees ascending order (§6).

**Why the v0.2 mechanisms are principled (and how they were validated).**
Every v0.2 element carries both a theoretical argument and an empirical
gate (`scripts/eval_borissal_coverage.py`; numbers and negative results in
`design.md`'s "v0.2" section): *frame differencing*
recovers fast intra-tubelet motion that tubelet-mean differencing provably
cancels (an oscillating object's tubelet means are identical). *Noise
floor*: two-frame differencing has a strictly positive expected |diff|
under i.i.d. sensor noise, and per-tubelet min-max then amplifies exactly
that floor to full range in motionless tubelets; subtracting a robust
quantile and soft-shrinking is codec dead-zone quantization / classical
coring, and true motion's spatial sparsity makes the median a valid floor
estimate. *Coherent-region gate*: objects are spatially contiguous, so
informative patches cluster — block gating imposes that smoothness prior
at zero learned cost, low-res-select-then-refine mirrors classical
multi-scale saliency integration and biological foveation, and a
description LLM grounds objects/actions better from coherent chunks than
isolated patches. *Global allocation + floor*: "every moment is equally
informative" (uniform) is false for description — information concentrates
where actions happen; global top-k concentrates budget there while the
per-tubelet floor prevents un-described time gaps. *Local/global blend*
is the scale infrastructure that makes cross-tubelet comparison (and thus
global allocation) meaningful. *Center bias* is the classical composition
prior — powerful on cinematic content, wrong on e.g. screen recordings,
hence a per-domain knob, never a default. (Temporal selection
stabilization was considered and deliberately dropped: for whole-clip
description, cross-tubelet selection variation is not harmful and can even
diversify coverage — a streaming-UI nicety, not a description mechanism.)

## 3. Config knobs

| Knob | Default | What it controls | Why this default |
|---|---|---|---|
| `scale` | `384` | Frame side length (square) | Matches the intended V-JEPA2 input size |
| `patch_size` | `16` | Spatial patch edge | Matches V-JEPA2's patch size |
| `tubelet_size` | `2` | Frames per temporal token | Matches V-JEPA2's tubelet size → `grid_thw=(8,24,24)` |
| `motion_weight` | `0.5` | Motion vs. spatial blend | Neutral fixed default; pass `"auto"` for content-adaptive weighting (see below) |
| `spatial_op` | `"grad"` | Gradient method | Cheapest option (no conv2d call) with no measured quality gain from Sobel |
| `pooling` | `"avg"` | Pixel→patch reduction | Standard, cheap, well-supported everywhere |
| `gazing_ratio` | `0.5` | Fraction of patches kept | Caller-set budget |
| `per_frame_allocation` | `"uniform"` | How the budget splits across tubelets | `"uniform"` exactly hits the ratio every time; `"proportional"` reallocates toward high-energy tubelets (can drift ±1-2 under extreme skew); `"global"` (v0.2) does one clip-wide top-K with a per-tubelet floor — concentrates budget on high-information moments, exact K by construction |
| `min_keep_per_frame_ratio` | `0.25` | Per-tubelet floor in `global` mode | Guarantees temporal coverage (no undescribed gaps); 25% of the uniform share is a conservative floor |
| `score_norm_blend` | `1.0` | Local vs clip-global normalization mix | 1.0 = v0.1 (per-tubelet only); <1 needed for `global` allocation to compare tubelets meaningfully |
| `center_bias` | `0.0` (off) | Additive Gaussian center prior | Classical composition prior; content-dependent → per-domain knob, never a default |
| `motion_diff` | `"tubelet"` | Temporal differencing granularity | `"frame"` catches fast intra-tubelet motion tubelet-averaging cancels; costs T−1 slice diffs (µs) |
| `frame_diff_agg` | `"mean"` | Frame-diff → tubelet aggregation | `"max"` favors transient motion detection |
| `motion_consistency` | `"none"` | Temporal double-difference (experimental) | NOT in preset: per-tubelet min-max structurally cancels its noise attenuation (negative result recorded in design.md); also suppresses untextured fast movers |
| `motion_noise_floor` | `"none"` | Dead-zone shrinkage of motion map | `"quantile"` (median) removes the sensor-noise floor that normalization would otherwise amplify in motionless tubelets |
| `motion_noise_q` / `motion_noise_scale` | `0.5` / `1.0` | Floor quantile / strength | Median assumes true motion is spatially sparse (<50% of patches) |
| `block_size` | `1` | Resize-based coarse-to-fine gate | `2` bounds fragmentation to ≤⌈k/4⌉ regions/tubelet; coarse signal = same pipeline on a 1/b-resized clip (resize's low-pass adds noise robustness) |
| `spread_fraction` | `0.0` | Hybrid focus+spread allocation (v0 AND v1) | `s>0` reserves `round(s·K)` of the budget for a stratified spatio-temporal skeleton (time first, then space; best-scoring cell per bucket), the rest stays pure top-k — the single-scale analogue of AutoGaze's multi-scale coarse share (~26%). Measured sweet spot `s=0.25` at the 0.25–0.5 target budget: improves semantic gist AND recall, coverage, and VideoMAE recon at once (design.md "Description-task alignment"). Works with `uniform` (2D in-tubelet) and `global` (3D clip-wide) allocation; not `proportional`. Inference-only knob — training is untouched |

`spatial_op="grad"`, `pooling="avg"`, and `per_frame_allocation="uniform"`
were each chosen as the faster and/or ratio-safer of their alternatives —
together they're already the fast+exact-budget preset. `motion_weight` was
the one knob whose best value genuinely depends on the clip's content
(static talking-head vs. high-motion footage) — `motion_weight="auto"`
now handles that: `w = motion_energy / (motion_energy + spatial_energy)`,
computed per video from the clip's own **pre-normalization** motion/spatial
magnitude (the per-tubelet min-max-normalized maps used for scoring can't
be used for this, since normalization erases absolute magnitude by
design). Near-zero-cost (two extra `.mean()` calls on already-computed
tensors). Verified on synthetic clips: a fully static clip (identical
repeated frame) resolves to `w=0.000`; a clip with a sweeping bright block
against low-texture background resolves to `w≈0.70`. On a mixed real clip
(`assets/example_input.mp4`, text+diagram content that's both edge-rich and
changing) it resolves to `w≈0.35` — a genuinely intermediate value, not
just defaulting to 0.5.

**Current recommended deploy configuration (2026-07-15)**: the `v0_2()`
preset with `per_frame_allocation="global"` and **`spread_fraction=0.25`**
— at the confirmed 0.25–0.5 target budget it improved ALL measured axes at
once (semantic gist AND recall, V-JEPA coverage, VideoMAE recon; tables in
design.md "Description-task alignment"). Caveat: that sweet spot was
measured on a single clip; it stays a runtime override rather than a baked
preset default until the scale run confirms it on real data.

### v0.3 candidate knobs (experimental, pre-gate)

All OFF by default; with every knob off the pipeline is bit-identical to
v0.2. These are the Tier-1 candidate bank of `v03-design.md` -- they enter
a named preset only after passing the sweep gates
(`scripts/sweep_borissal_v03.py`; verdicts recorded in `design.md`).

| knob | default | what it does |
|---|---|---|
| `motion_center_surround` (+`motion_cs_kernel`) | off | relu(D − avgpool(D)) on the pooled motion map: cancels uniform ego-motion (pan/zoom), keeps independent movers |
| `coherence_gate` (+`coherence_kernel`, `coherence_gamma`) | off | multiplies the gradient channel by (1 − structure-tensor coherence)^γ: suppresses gratings/straight edges, spares isotropic object micro-structure |
| `signature_weight` | 0 | image-signature (sign-of-DCT via fixed matmul) appearance channel: fires on spatially sparse foreground support |
| `color_rarity_weight` (+`color_bins_per_axis`, `color_bin_sigma`) | 0 | global color rarity (soft-binned histogram contrast): object interiors with rare colors fire uniformly; first use of color in the v0 line |
| `dog_blob_weight` | 0 | multi-scale difference-of-boxes blob channel: the cheapest interior filler |
| `fusion_norm` (+`fusion_entropy_floor`) | none | content-adaptive channel fusion: "peak" (Itti N(·)) or "entropy" (bounded inverse-entropy gate; a pan-flooded motion map loses fusion weight automatically) |
| `score_ema_alpha` | 0 | temporal score EMA across tubelets (loop-free); streaming state via `select(..., temporal_state=...)` |
| `select_hysteresis_eps` | 0 | pre-topk bonus for patches kept in the previous tubelet (one-step vectorized approximation) |

## 4. Output: `Selection`

grid_thw-native, not AutoGaze's `gazing_pos` dict contract — see
[`design.md`](./design.md) for why. Flatten order is t-major:
`i = t*(H_grid*W_grid) + h*W_grid + w`.

| Field | Shape | Meaning |
|---|---|---|
| `grid_thw` | `(B, 3)` | `(T_grid, H_grid, W_grid)` |
| `scores` | `(B, L)` | Per-patch saliency, pre-top-k |
| `keep_mask` | `(B, L)` | Boolean, kept/selected |
| `keep_index` | `(B, K)` | Flat indices of kept patches, `-1` padded |
| `keep_coords` | `(B, K, 3)` | `(t, h, w)` per kept patch, `-1` padded |
| `num_keep` | `(B,)` | Valid (non-padded) count |
| `per_frame_keep` | `(B, T_grid)` | Kept count per tubelet |

`autogaze/models/borissal/adapters.py` bridges this to a V-JEPA2-style
gather point (`to_vjepa2`), a canonical flat keep-index-per-video list
(`to_canonical_keep_indices`, see §5), or, optionally, to the legacy AutoGaze
dict contract for sanity-checking against the existing VideoMAE task
(`to_autogaze_gazing_info`).

## 5. Canonical downstream interface

A real downstream pipeline (Qwen-VL-style sparse encoder over V-JEPA2)
expects, per video, a flat list of kept patch indices — `idx = t*N + n`
(`N` = patches per frame, `n` = row-major within-frame index) — **sorted
ascending by (frame, row, col)**, since the encoder's mask-gather step and
its RoPE position recovery both depend on that order to map each surviving
token back to its true `(t, row, col)`.

Borissal's `keep_index` already satisfies this exactly: its native flatten
order (`t*(H_grid*W_grid) + h*W_grid + w`) *is* that formula, and the
packer that builds it preserves ascending order among kept entries — no
reordering needed (locked down by
`tests/test_borissal.py::test_keep_index_is_ascending_per_row`). Use
`adapters.to_canonical_keep_indices(selection) -> list[Tensor]` to get it in
the exact shape a `keep_indices_per_video`-style handoff expects: one
1-D ascending `LongTensor` per video, `-1` padding stripped.

## 6. Standalone

The core — `configuration_borissal.py`, `modeling_borissal.py`,
`adapters.py`, `device.py` — depends on **`torch` only** (no `autogaze.*`
imports); the one packing routine it used to borrow from
`autogaze/utils.py` is now inlined. Copy `autogaze/models/borissal/` into
another project as-is and it works unchanged; drop `video_io.py`
(PyAV) and `viz.py` (matplotlib) if you don't need this repo's demo
scripts. All ops are standard PyTorch (`mean`, `avg_pool2d`/`max_pool2d`,
`conv2d`, `topk`, `argsort`, …) with no custom CUDA/C++ extensions, so it
runs on a Linux/CUDA machine with no code changes — just `.to("cuda")`.
`device.py::resolve_device(mode="auto")` picks `cuda` when available,
otherwise `cpu` (not `mps` — see §7, CPU measured faster for this workload).

## 7. Performance & mobile

Non-learned, so the cost is arithmetic, not FLOPs-vs-an-encoder. Measured
on a Mac: **~7-12ms/clip on CPU**, ~85-150 clips/sec (16 frames, 384×384).
Sort-family ops (the packer's argsort — plain, not stable: its keys are
unique by construction, and the stable variant was removed because the
ONNX exporter cannot represent it — plus `torch.topk`) are under 6% of
self-CPU time; the rest is ordinary elementwise/pooling arithmetic,
universally supported on mobile backends.
An empirical `torch.jit.trace` pass caught and fixed a real tracing bug, and
surfaced a genuine correctness trap in `"proportional"` allocation under
naive tracing (it silently freezes a data-dependent value). Full numbers,
profiler breakdown, and the mobile-export constraints this implies for
Phase 2 are in
[`design.md`](./design.md#mobile-readiness-review-2026-07-14-before-starting-phase-2) —
not duplicated here.

## 8. Borissal v1 (the learned selector) — knob summary

v1 exists (Phase 2/3 built; training guide = `training.md`, architecture
rationale = `design.md`). Same `Selection` contract and canonical
keep-index as v0; `select()` accepts the same runtime overrides
(`gazing_ratio`, `per_frame_allocation`, `spread_fraction`).
`BorissalV1Config` defaults, audited 2026-07-15:

| Knob | Default | Why this default |
|---|---|---|
| `input_mode` | `"both"` | v0 saliency maps (proven prior) + grid pixels (complementary cues); `maps`/`pixels` are ablation arms |
| `hidden_channels` / `num_blocks` | `64` / `3` | ~122K params incl. context path; CPU select() 14.5ms (budget 25ms) |
| `cosine_scores` | `True` | Bounds \|logit\| by construction — the anti-saturation fix validated by the 40-step A/B (entropy mean-reverts instead of collapsing) |
| `global_context` | `True` | GCNet-lite learned weighted frame+clip summary; the local ~9×9-cell stack cannot express the clip-global comparisons the SSL objectives ask for; zero-init = exact no-op at init |
| `gumbel_tau` | `2/3` | Concrete/Gumbel-softmax canonical temperature; fixed (τ<0.5 is the gradient-variance blowup zone) |
| `train_block_size` | `1` | Experimental: `2` = block-structured ST selection (anti-scatter, on-distribution for the multi-block-trained predictor); training-only |
| `v0_preset` | `"v0.2"` | Gate-validated signal preset feeds the scorer |
| `per_frame_allocation` | `"uniform"` | Export-safe default; `"global"` available at inference |
| `spread_fraction` | `0.0` | Backward-compat baseline; see deploy recommendation in §3 |
| `residual_scoring` | `False` | `score = f(x)` not `v0 + f(x)`; residual is an ablation arm |

Still not built: content-adaptive auto-tuning of v0's remaining knobs
(intentionally skipped — v1 IS the learned version of that), on-device
(CoreML/TFLite) export validation (ONNX-17 export passes for v0.2, v1,
and v1+global+spread — see design.md "Mobile-export pre-check"), and the
downstream processor→encoder→LLM attachment (separate track).
