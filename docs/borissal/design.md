# Borissal — Patch Selector Line (design doc)

`Borissal` is the name for the new patch-selector work built on top of this
repo's AutoGaze codebase. This document is the durable, repo-committed
version of the design; it survives across sessions (unlike a local Claude
Code plan file). See `progress.md` in this same directory for the
session-by-session handoff log.

## Why this exists (context)

AutoGaze's own gaze model is **autoregressive**, **multi-scale**, and
**learned** (a shallow 3D-conv encoder feeding a LLaMA-style decoder that
emits patch indices token-by-token). We are building a different axis of
patch selector:

- **single-scale** (one resolution / one patch grid, not a multi-scale tile)
- **feed-forward** (no autoregressive decoding step)
- **top-k under a ratio budget** (`gazing_ratio` sets the total; the
  per-frame share of it is either uniform or dynamically reallocated to
  each frame's own saliency energy — `per_frame_allocation`)

The full effort is split into three phases/versions, all on branch `feat/borissal`:

1. **Phase 1 (this doc, done)** — **Borissal v0**: a non-learned, saliency-based
   feed-forward selector. Mobile-oriented (low latency, no learned weights).
2. **Phase 2** — **Borissal v1**: a trainable selector (TSM or conv3d backbone +
   scoring head + differentiable/straight-through top-k), using v0's
   saliency signal as a baseline/feature input.
3. **Phase 3** — self-supervised training via V-JEPA2.1L: compare V-JEPA2
   features on a dense (full) video vs. a sparse (selector-chosen) video to
   train Borissal v1 without ground-truth gazing labels. New task +
   algorithm.

## Key design decisions (and why they depart from AutoGaze conventions)

| Decision | Rationale |
|---|---|
| Output is **grid_thw-native**, not AutoGaze's `gazing_pos` dict contract | The intended downstream encoders (V-JEPA2, Qwen-VL-style models) already think natively in `(t, h, w)` token grids with flat/gather-based sparsity. Conforming to AutoGaze's dict (`gazing_pos`, `if_padded_gazing`, `gazing_mask` per scale, ...) would have made the *first* integration easy but every *subsequent* encoder integration harder. `adapters.to_canonical_keep_indices` is the real bridge to a verified downstream convention (see "Canonical downstream interface" below); `adapters.to_autogaze_gazing_info` is a separate, optional bridge back to the legacy contract for sanity-checking against the existing VideoMAE task (only supports uniform per-frame allocation). |
| Core package is **`torch`-only, standalone-portable** | `configuration_borissal.py` + `modeling_borissal.py` + `adapters.py` + `device.py` have zero `autogaze.*` imports (the one packer function they used to borrow from `autogaze/utils.py` is now inlined) — the directory can be copied into another project and just works, including unchanged on Linux/CUDA (`device.py::resolve_device`, `cuda` > `cpu`, deliberately not `mps` — see "Mobile readiness review"). `video_io.py`/`viz.py` stay separate, dev-tooling-only, not needed by the model itself. |
| `transformers>=5.5,<6` (repo-wide pin change), not `~=4.51` | Forward-looking: Borissal's own code does not depend on the legacy AutoGaze modeling classes at all, and using a current transformers avoids re-litigating this later when Phase 3 attaches V-JEPA2/Qwen. **Trade-off accepted knowingly**: this pin change is repo-wide (`pyproject.toml`), so the *legacy* AutoGaze custom modeling code (`autogaze/models/autogaze/*`, `autogaze/vision_encoders/siglip/*`, `autogaze/tasks/video_mae_reconstruction/*`) may no longer import cleanly under 5.5.x (large API jump from 4.51). Borissal does not import any of those modules, so this doesn't block Phase 1. Migrating the legacy training path to 5.5.x is out of scope unless/until someone needs to run it again. |
| `flash_attn` moved to an optional `cuda` extra | It doesn't build on macOS. Nothing in Borissal imports it; the legacy AutoGaze code only loads it lazily via `attn_implementation="flash_attention_2"`, which defaults to `sdpa` off anyway. |
| Default grid `scale=384, patch_size=16, tubelet_size=2` → **grid_thw = (8, 24, 24)** | Matches V-JEPA2's native input convention (the intended downstream encoder), and matches the original 24×24×8 framing from the initial brainstorm. All three are config knobs. |
| Repo footprint for Phase 1 is **new files only** | `autogaze/train.py`, `trainer.py`, `datasets/`, and the legacy `tasks/`/`models/autogaze/` are untouched. Borissal lives entirely under `autogaze/models/borissal/`. This keeps the legacy NTP/GRPO training path runnable (on its own transformers version, in its own environment) and makes Borissal trivially removable/relocatable if needed. |
| Development/validation on **Mac (uv, CPU/MPS), standalone** | No DDP/Hydra/trainer coupling for Phase 1 — a plain Python script and pytest suite are enough. Real training-scale work moves to Linux/CUDA later; nothing in Phase 1 assumes CUDA. |
| Saliency-quality evaluation is **qualitative-first** | Per explicit product direction: overlay visualizations (selected patches drawn on frames) are the primary evidence for Phase 1, not a reconstruction-loss number. A quantitative reconstruction-based sanity check (vanilla ViT-MAE or V-JEPA2, dense vs. sparse) is optional/secondary, not required for Phase 1 to be considered done. |

## grid_thw output schema

Flatten order is **t-major**: flat index `i = t*(H_grid*W_grid) + h*W_grid + w`
(`einops` `"(t h w)"`). Any encoder adapter must match this order or remap it
explicitly — do not change the native order in `Selection`, since multiple
adapters rely on it.

`autogaze/models/borissal/modeling_borissal.py::Selection`:

```
grid_thw:       (B, 3)     long   -- (T_grid, H_grid, W_grid)
scores:         (B, L)     float  -- L = T_grid * H_grid * W_grid, per-patch saliency
keep_mask:      (B, L)     bool   -- True = kept/selected
keep_index:     (B, K)     long   -- flat indices of kept patches, -1 padded
keep_coords:    (B, K, 3)  long   -- (t, h, w) per kept patch, -1 padded
num_keep:       (B,)       long   -- valid (non-padded) count per instance
per_frame_keep: (B, T_grid) long  -- kept count per tubelet
```

`T_grid = T // tubelet_size`, `H_grid = H // patch_size`, `W_grid = W // patch_size`.

## Saliency algorithm (`Borissal.select`, non-learned, float32, no autocast)

1. Luma: `gray = video.mean(dim=2)` → `(B, T, H, W)`.
2. Tubelet aggregation: mean over each `tubelet_size`-frame group → `(B, T_grid, H, W)`.
3. **Motion** (codec-residual proxy): `|tub[t] - tub[t-1]|`; `t=0` uses the
   forward diff so the first tubelet isn't starved. (Streaming hook: replace
   `tub[t-1]` with a cached previous tubelet — marked in code for Phase 2/3.)
4. **Spatial** (edge/gradient energy): finite-difference gradients
   (`spatial_op="grad"`) or a fixed non-learnable Sobel kernel
   (`spatial_op="sobel"`); `sqrt(gx^2 + gy^2 + eps)`.
5. Pixel → patch pooling: `avg_pool2d`/`max_pool2d` with `kernel=stride=patch_size`.
6. Per-(instance, tubelet) min-max normalize motion and spatial maps to `[0,1]`,
   combine with a single hyperparameter: `S = w*motion + (1-w)*spatial`
   (`motion_weight = w`, fixed float or `"auto"` — see "Content-adaptive
   `motion_weight`" below).
7. Budget allocation (rule-based, not learned):
   - `per_frame_allocation="uniform"` (default): every tubelet keeps
     `k = clamp(round(gazing_ratio * H_grid*W_grid), 1, H_grid*W_grid)` patches.
   - `"proportional"`: total budget `round(gazing_ratio * L)` distributed across
     tubelets proportional to per-tubelet energy, via largest-remainder
     (Hamilton's method) rounding so counts sum exactly to the budget (mod
     per-tubelet `[1, N_pf]` clamping).
8. Top-k → `keep_mask` via `torch.topk` (bounded by the per-tubelet budget) +
   a boolean rank compare + `scatter_`, no Python loops over batch/frame.
   (Originally a fully vectorized `argsort`-twice rank comparison; swapped to
   `torch.topk` for mobile-op-support reasons — see "Mobile readiness review"
   below.)
9. Packing to `keep_index`/padding uses `_pack_gazing_mask` (stable
   ones-first sort) applied to the flattened `(B, L)` mask. Originally
   imported from `autogaze/utils.py::get_gazing_pos_from_gazing_mask`
   (verified to have no transformers/legacy-model dependency); **inlined
   verbatim into `modeling_borissal.py`** so the package has zero
   `autogaze.*` imports — see "Standalone" below. A pleasant side effect:
   this packer's stable sort keeps kept indices in ascending order, which
   turned out to be exactly the ordering a real downstream consumer needs
   — see "Canonical downstream interface" below.

## Canonical downstream interface (2026-07-14)

The user investigated the actual downstream pipeline (a Qwen-VL-style sparse
encoder over V-JEPA2) and reported back its canonical selector-output
convention: a flat, per-video, ascending list of kept patch indices,
`idx = t*N + n` (`N` = patches/frame, `n` = row-major within-frame),
sorted by (frame, row, col) — required because the encoder's mask-gather
and RoPE position recovery both walk that list assuming that order.

**Checked against `Selection.keep_index` and confirmed to already match,
with zero algorithm changes needed**: Borissal's flatten order (step 9
above) already *is* `t*N+n`, and `_pack_gazing_mask`'s stable sort already
preserves ascending order among kept entries (empirically verified across
`uniform`/`proportional` allocation and a batch of instances before writing
any code). Locked down as an explicit test
(`test_keep_index_is_ascending_per_row`) so a future change to the top-k
mechanism can't silently break it. `adapters.to_canonical_keep_indices`
exposes it in the exact shape the downstream `keep_indices_per_video`
handoff expects (list of 1-D ascending tensors, `-1` padding stripped). See
`docs/borissal/reference.md`'s "Canonical downstream interface" section for
the consumer-facing version of this.

Everything past the selector (processor/model/encoder/LLM internals,
steps 3-6 of what the user's investigation described) is out of scope for
Borissal — that's the downstream pipeline's responsibility, not addressed
here.

## Content-adaptive `motion_weight` (2026-07-14)

`motion_weight` was the one knob whose best fixed value genuinely varies
by clip content (the other allocation/op knobs already default to the
fast+ratio-safe choice regardless of content — see the decisions table
above). This was scoped, designed, and then *deliberately deferred* in an
earlier round ("추후 더 고도화해보자. 학습 기반 모델도 추가로 있을 거니까" —
revisit alongside the learned selector); this round the user asked to
implement it before moving to Phase 2 after all.

Implementation: `motion_weight="auto"` computes
`w = motion_energy / (motion_energy + spatial_energy)` per video, from
`motion_p`/`spatial_p` **before** their per-tubelet min-max normalization
(using the already-normalized maps would erase the absolute-magnitude
signal this needs — min-max normalization always rescales to `[0,1]`
regardless of the original energy level). Two extra `.mean()` reductions
on tensors already computed; no new tensors, no learning. Exposed via a new
`motion_weight_used` field in the `select_with_intermediates` intermediates
dict (not added to `Selection` itself, to avoid touching the
already-tested/downstream-facing output contract) — `(B,)`, equal to the
fixed value broadcast when `motion_weight` isn't `"auto"`.

Verified (`tests/test_borissal.py::test_motion_weight_auto_adapts_to_content`
+ manual runs): a fully static synthetic clip (identical frame repeated)
resolves to `w=0.000`; a synthetic clip with a sweeping bright block against
low-texture background resolves to `w≈0.70`; the real
`assets/example_input.mp4` clip (edge-rich *and* changing content) resolves
to `w≈0.35`. Before/after overlay comparison
(`outputs/borissal/{r50_m50_grad_uniform,r50_mauto_grad_uniform,
auto_demo_static,auto_demo_motion}/04_overlay.png`, gitignored) shows the
synthetic cases most clearly: the static case selects purely the block's
edge, the motion case tracks the moving block's position per tubelet.

## Files

- `autogaze/models/borissal/configuration_borissal.py` — `BorissalConfig`
  (plain dataclass; no HF `PretrainedConfig` needed since Phase 1 has no
  learned weights to checkpoint).
- `autogaze/models/borissal/modeling_borissal.py` — `Borissal(nn.Module)` +
  `Selection` dataclass + the algorithm above.
- `autogaze/models/borissal/adapters.py` — `to_vjepa2` (passthrough/rename for
  a V-JEPA2-style gather-before-transformer attach point),
  `to_canonical_keep_indices` (per-video ascending flat-index list matching a
  real downstream `keep_indices_per_video` convention — see "Canonical
  downstream interface" below), and `to_autogaze_gazing_info` (optional
  bridge to the legacy dict contract, for running Borissal through the
  existing VideoMAE task as a sanity check; requires uniform per-frame
  allocation).
- `autogaze/models/borissal/device.py` — `resolve_device(mode="auto")`
  (`cuda` > `cpu`, deliberately not preferring `mps` — see "Mobile readiness
  review") and `available_devices()` (enumeration, used by the benchmark
  script to test every device present rather than pick one).
- `autogaze/models/borissal/video_io.py` — PyAV decode + uniform frame
  sampling + normalize, no transformers video processor dependency.
- `scripts/eval_borissal_qualitative.py` — standalone Mac/CPU/MPS qualitative
  eval: decodes a video, runs the selector, renders a red-bordered overlay of
  kept patches (ported from
  `autogaze/tasks/video_mae_reconstruction/visualize_video_mae_reconstruction.py`'s
  rendering logic, without the wandb/DDP/task coupling).
- `tests/test_borissal.py` — contract invariants (grid_thw shape, keep_index
  validity/no-duplicates, `per_frame_keep.sum() == num_keep`, coords ↔ flat
  index consistency, ratio controls budget, motion_weight changes selection,
  CPU/MPS parity).
- `autogaze/models/borissal/viz.py` — shared rendering helpers (frame strip,
  overlay, heatmap grid, allocation bar chart) used by both eval scripts below.
- `scripts/borissal_dump_outputs.py` — stage-by-stage dump (input frames,
  motion/spatial/score heatmaps, overlay, allocation bar chart, summary.json)
  to `outputs/borissal/<run>/` (gitignored). Uses
  `Borissal.select_with_intermediates(...)`, which returns the pre-top-k
  motion/spatial/score maps alongside the normal `Selection`.
- `scripts/borissal_benchmark.py` — latency benchmark + `torch.profiler`
  op-breakdown, feeding the "Mobile readiness review" below.
- `docs/borissal/reference.md` — a concise "what and why" reference (algorithm,
  config knobs, output schema) distinct from this file's dev-log rationale.

## Environment

`uv venv --python 3.11 && uv pip install -e .` on macOS installs cleanly with
CPU/MPS torch and `transformers>=5.5,<6`, no `flash_attn` needed. `uv pip
install -e '.[cuda]'` on Linux/CUDA additionally pulls `flash_attn` for the
legacy training path. See `pyproject.toml`.

## What's out of scope for Phase 1

- Any change to `autogaze/train.py`, `trainer.py`, the legacy `tasks/` or
  `models/autogaze/` code, or the Hydra config tree.
- Learned parameters / training of any kind.
- Migrating the legacy NTP/GRPO training path to transformers 5.5.x.
- A real V-JEPA2/Qwen attachment (Phase 2/3) — `adapters.to_vjepa2` is a
  documented stub for the expected call shape, not a working integration.

## Mobile readiness review (2026-07-14, before starting Phase 2)

Borissal v0 has zero learned parameters, so its "burden" isn't FLOPs
(negligible next to any vision encoder) — it's **operator support and shape
behavior on mobile inference runtimes** (CoreML / TFLite / NNAPI / delegates).
This review is empirical, not just theoretical: it measured actual latency
and actually attempted to trace the model, rather than reasoning from op
names alone.

### Latency (measured on this Mac; `scripts/borissal_benchmark.py`)

Clip shape `(B=1, T=16, C=3, H=384, W=384)`, 10 warmup + 50 measured iters,
`Borissal.select()` only (excludes video decode):

| device | gazing_ratio | allocation | mean (ms) | median (ms) | clips/sec |
|---|---|---|---|---|---|
| cpu | 0.5 | uniform | 11.6 | 9.5 | 86 |
| cpu | 0.5 | proportional | 6.7 | 6.5 | 150 |
| cpu | 0.25 | uniform | 7.4 | 6.9 | 136 |
| cpu | 0.25 | proportional | 8.8 | 8.9 | 114 |
| mps | 0.5 | uniform | 207.7 | 174.6 | 4.8 |
| mps | 0.5 | proportional | 165.0 | 152.7 | 6.1 |
| mps | 0.25 | uniform | 154.4 | 152.2 | 6.5 |
| mps | 0.25 | proportional | 175.5 | 169.4 | 5.7 |

Full data: `outputs/borissal/benchmark/latency_{cpu,mps}.json` (gitignored,
regenerate with the script above).

**CPU is faster than MPS here** (a few ms vs ~150-200ms) — a known MPS-backend
quirk where per-op GPU dispatch overhead dominates for many small tensor ops
on small tensors, not a reflection of real mobile SoC behavior. Neither number
is a substitute for an actual on-device (phone CPU delegate / NPU) measurement,
which should happen once real mobile export tooling is used (Phase 3+). Taken
at face value, though, ~6-12ms/clip on a laptop CPU is already utterly
negligible next to any vision-encoder forward pass, and Borissal has no
learned weights to speed up further — the entire cost is in ordinary
elementwise/reduction/pooling arithmetic over the raw pixel grid.

### Operator inventory (`torch.profiler`, CPU, ratio=0.5/uniform; full table in `outputs/borissal/benchmark/profiler_cpu.txt`)

Dominant self-CPU-time contributors: `aten::sum` (26%), `aten::avg_pool2d`
(14%), `aten::fill_`/`aten::div_`/`aten::copy_`/`aten::sub`/`aten::add`/`aten::mul`
(elementwise, ~35% combined), `aten::sqrt`/`aten::abs` (~4.5%), `aten::min`/`aten::max`
(~2%). All of these are standard, universally-supported ops on any mobile
backend. Separately, `aten::mean` (used once, for the initial per-frame luma
`gray = video.mean(dim=2)` over the *full-resolution* pixel grid) accounts for
the single largest chunk of total (non-self) CPU time (~35%) because it
operates on the largest tensor in the whole pipeline (before any patch
pooling shrinks things) — expected and not a concern, just worth knowing
where the time actually goes if this needs future optimization.

**Sort-family ops are now minor**: `aten::sort` (2.9% self-time, 1 call/iter —
this is `_pack_gazing_mask`'s stable argsort, inlined from the originally-
borrowed `autogaze/utils.py::get_gazing_pos_from_gazing_mask`) and
`aten::topk` (2.8% self-time, 1 call/iter — the selector's own top-k, see
next section). Combined, sort/topk is under 6% of self CPU time and
operates on modest tensor sizes (`L=4608` elements at default config).

### Fix applied: argsort-rank top-k → `torch.topk`

`Borissal`'s own top-k step originally used a double-`argsort` rank trick
(`order = scores.argsort(...); rank = order.argsort(...); mask = rank < k`).
This round replaced it with `torch.topk(k_max, dim=-1)` + a boolean
rank-within-topk compare + `scatter_`, keeping identical selection semantics
(both pick exactly the top-k highest-scoring patches per tubelet; only
exact-tie ordering can differ, which this application never depends on —
confirmed by the full `tests/test_borissal.py` suite still passing, 7/7,
plus a dedicated invariant re-check across ratio/allocation combinations).
Rationale: `torch.topk` is a first-class op on TFLite (`TopKV2`) and CoreML
(`top_k`), unlike general `sort`/`argsort`, which have much patchier mobile
support. The one remaining `argsort` (then in the reused legacy
`autogaze/utils.py` packer) was deliberately left algorithmically untouched
at the time — it's a single O(L log L) op on a modest vector, and Phase 1's
own tests depend on its exact tie-order behavior. (It was later inlined verbatim, unchanged, into `modeling_borissal.py` as
`_pack_gazing_mask` for standalone-portability reasons — see the "Core
package is `torch`-only, standalone-portable" row above and the "Canonical
downstream interface" section below; that move didn't touch this
sort/mobile-support tradeoff.)

### Empirical trace check: `torch.jit.trace`, a real bug found and fixed

Rather than reasoning abstractly about export-readiness, this round actually
ran `torch.jit.trace` against `Borissal.select` (wrapped to return plain
tensors). Two concrete findings:

1. **Bug found and fixed**: tracing initially **failed outright**
   (`TypeError: type Tensor doesn't define __round__ method`), because
   `B, T, C, H, W = video.shape` doesn't yield plain Python ints under
   `torch.jit.trace` in this torch version — the components come back as
   trace-time symbolic wrappers, and later code calls Python's `round()` on
   an expression derived from them (`round(ratio * N_pf)`), which only
   `float`/`int` support. **Fixed** with a one-line, behavior-preserving
   change in `modeling_borissal.py`: `B, T, C, H, W = (int(x) for x in
   video.shape)`. Verified: all 7 existing tests still pass (eager-mode
   behavior is identical — `int()` on an already-int-valued eager tensor
   dimension is a no-op), and the traced graph's output now exactly matches
   fresh `select()` calls on new random inputs of the same shape, for
   `per_frame_allocation="uniform"`, across repeated trials.
2. **Real correctness trap, "proportional" allocation only**: because
   `per_frame_allocation="proportional"` computes `k_per_frame` from the
   *actual saliency energy of the traced input*, a naive `torch.jit.trace`
   freezes that data-dependent per-tubelet split as a constant. Verified
   empirically: tracing with one video, then feeding the traced graph a
   **different** video of the identical shape, produces a silently-wrong,
   stale result (`num_keep=2293` from the traced graph vs. the correct,
   freshly-computed `num_keep=2304` from eager `select()` on that same new
   input). `"uniform"` allocation has no such issue — `k_per_frame` there is
   purely config/shape-derived, not data-dependent, so the same experiment
   (fresh random content, same traced graph) reproduced the exact eager
   result every time.

Both TracerWarnings ("Converting a tensor to a Python boolean/integer... will
be treated as a constant") are expected and, for a mobile-export context,
actually describe the *intended* contract: a production export should target
one fixed clip shape per exported artifact (standard practice for
CoreML/TFLite anyway), so baking `T`/`H`/`W`/`patch_size`/`tubelet_size`-derived
constants into the graph is fine. Freezing a *data-dependent* value
(`proportional`'s per-tubelet split) is not fine, and is a correctness bug,
not a shape-genericity nicety.

### Conclusions / constraints carried into Phase 2

- No FLOP/parameter burden concern — confirmed negligible vs. any encoder.
- **Prefer `torch.topk`/bounded-selection over general sort/argsort** in any
  new learned selector code, for the same mobile-op-support reason. Already
  applied to Borissal v0's own top-k step.
- **Treat `"uniform"`-style (data-independent) per-frame/per-tubelet
  allocation as the mobile-export-safe default.** If Phase 2/3 need a
  data-dependent allocation policy on-device, it cannot be naively
  `torch.jit.trace`d — it needs a genuinely dynamic export path (e.g.
  `torch.export` with explicit dynamic-shape/control-flow handling) or must
  be recomputed natively outside any traced/exported subgraph. Don't
  silently reuse a traced graph across different inputs for a data-dependent
  allocation policy.
- **Export one artifact per fixed input shape** (clip length, resolution,
  patch/tubelet size) rather than assuming shape-genericity — consistent with
  how `int()`-casting shape components behaves, and normal for mobile export
  toolchains regardless.
- Real on-device (phone CPU/NPU) latency is still unmeasured — the CPU/MPS
  numbers above are a rough proxy at best (MPS in particular is not
  representative). Get an actual on-device number once mobile export tooling
  is in place (Phase 3+), not before.

## Borissal v1 + SSL training (2026-07-14, Phase 2+3)

Implemented in one pass; full training rationale/recipes live in
`training.md` (canonical), selector docs in `reference.md`. Key engineering
facts recorded here:

- **transformers pinned to ==5.5.0** (user requirement). The vjepa2 sparse
  primitives (`apply_masks`, `VJEPA2Layer(position_mask)`,
  `get_position_ids(masks)`) were re-verified identical at 5.5.0 after
  originally being explored at 5.13.1. V-JEPA2's token flatten order is
  t-major `idx = t*(H'W') + h*W' + w` — exactly Borissal's canonical
  keep_index, so the selector output drives the teacher's RoPE positions
  with no remapping.
- **v1 architecture**: TSM-style 2D CNN (~114K params, 2D convs + a
  zero-cost channel time-shift; mobile-delegate-friendly per the Mobile
  readiness review) over grid-resolution inputs, `input_mode:
  maps|pixels|both` ablation switch. Inference `select()` shares v0's
  Selection contract and the canonical ascending-index guarantee (same
  packing helper; same tests applied).
- **Differentiable selection**: Gumbel-perturbed hard top-k forward +
  straight-through soft gate backward, with an `·N_pf` gradient rescale.
  Two real bugs were caught empirically during smoke: (1) an operator-
  precedence error made the Gumbel term constant-NaN, silently freezing
  selection (fixed, regression-tested via
  `test_v1_gumbel_stochastic_in_train_deterministic_in_eval`); (2) raw
  softmax probs (~1/N_pf) starved the ST gradient by ~3 orders of
  magnitude (fixed by the rescale).
- **Sparse teacher path**: `vjepa2_sparse.py` (the only core file importing
  transformers) — functional sparse-encoder forward over the stock
  encoder's own modules (no weight copies), frozen `VJEPA2Teacher` wrapper
  with `dense_features/sparse_features/predict`, teacher-agnostic so a
  torch.hub V-JEPA2.1 adapter can slot in later.
- **V-JEPA 2.1 checkpoints do NOT load into the native vjepa2 class**
  (empirically confirmed: dual image/video patch embeds, modality embeds,
  distillation norms, predictor proj 1664≠hidden). True-2.1 teachers need a
  torch.hub/custom-code adapter behind the same wrapper interface. Grid
  convention is identical, so selector training against official V-JEPA2
  (`facebook/vjepa2-vitl-fpc64-256`, loads cleanly, patch16/tubelet2
  asserted) validates the mechanism equivalently — user confirmed the
  teacher need not match the downstream encoder.
- **Training loop** (`scripts/train_borissal_v1.py`): DDP-under-torchrun /
  single-device otherwise, per-batch gazing_ratio sampling (rank-synced),
  composable losses (`losses.py`), jsonl logging incl. peak memory,
  checkpoints under `weights/` (gitignored).

## Borissal v0.2 (2026-07-14): mechanisms, quantitative gate, and negative results

Motivated by observed v0.1 failures (scattered selections; budget wasted on
noise motion) with the video-DESCRIPTION task as the arbiter. Seven user-
specified elements were implemented, each fully vectorized (no Python
loops), and — after a skeleton review flagged the risk of hand-tuning 10+
knobs by eyeball — gated with a label-free quantitative metric before
preset admission. `reference.md` §2/§3 documents the mechanisms and their
theoretical justifications; this section records the measurements and what
they changed.

### The adoption gate and what it revealed

`scripts/eval_borissal_coverage.py` scores any selector config with a
frozen V-JEPA2 (+ its predictor) on two axes:
- **coverage** (predict UNSELECTED from selected; lower better),
- **uniqueness** (predict SELECTED from the rest; higher better).

**Finding 1 — coverage alone is scatter-biased (affects Phase 3 too).**
On the example clip, RANDOM selection beat every saliency config on
coverage at every ratio tried (e.g. 8.239 vs v0.1's 8.249 at ratio 0.25):
reconstructing *everything* rewards evenly-scattered anchors for
interpolation, which is precisely not saliency. Consequence for v0.2: the
gate rule is "must not degrade BOTH metrics", never coverage alone.
Consequence for the v1 SSL objective (same math): pure predictor-coverage
training pressure points toward uniform scatter — the training.md
experiment matrix should combine it with an anti-scatter term (uniqueness-
style, or coherence regularization) at scale. Uniqueness, by contrast,
ordered configs sensibly (v0.1 8.289-8.300 >> random 7.873-7.933).

**Finding 2 — the winning combination (preset, ratio 0.25, example clip):**
`frame-diff + quantile noise floor + global allocation(min-floor 25%) +
score blend 0.7 + block gate b=2` is Pareto-best vs v0.1: coverage
8.226 < 8.249 AND uniqueness 8.370 > 8.300, while contiguity rises
2.31 → 2.94 and the overlays show clean object-chunk selections. Global
allocation + blend is the single strongest element (uniqueness +0.07/+0.02
alone). Caveat: at ratio ≲ 0.1 the block gate over-constrains (both
metrics dip) — use `block_size=1` for ultra-low budgets.

**Finding 3 — gate sizing bug found by the metric itself:** under global
allocation, sizing the coarse-to-fine gate by worst-case per-tubelet
capacity opened it fully and silently neutered coherence (identical
metric values with/without b=2 exposed it). Fixed: gate allows up to 2x
the uniform share per tubelet (total capacity 2·K_total ≥ K_total keeps
the exact budget; concentration beyond 2x spills to other tubelets).

**Finding 4 — negative result: `motion_consistency="double_diff"`.**
Classical double-differencing was implemented and then EXCLUDED from the
preset after synthetic testing showed (a) per-tubelet min-max
normalization structurally cancels its noise attenuation (min halves noise
AND the normalization ceiling alike; net selection actually shifted toward
noise), and (b) it entirely suppresses untextured uniform-color movers
whose displacement exceeds their edge-strip width (adjacent diff strips
don't overlap). Kept as an experimental knob with the limitation
documented; a useful reminder that classical mechanisms don't compose
freely with normalization stages.

**Finding 5 — double-diff semantics correction (recorded for honesty):**
the initial design claimed it removes "single-frame spikes"; empirically a
one-frame flash survives at its own frame (appear+disappear both produce
large adjacent diffs) — what it removes is ghosting (leading/trailing
revealed-background diffs) and temporally-uncorrelated pixel noise.

### Latency (CPU, same benchmark protocol as the mobile readiness review)

v0.1 5.9ms → v0.2 preset 12.2ms per clip (dominated by the 1/2-resolution
coarse saliency pass; all four new elements combined add <1ms). Within the
25ms target; still negligible vs. any encoder.

### Roles going forward

v0.2 is (a) the deployable non-learned baseline and (b) v1's input-feature
bank — the hand-tuned combination is deliberately polished only to the
gate's pass line; combination-optimization beyond that is v1's job
(learned). Temporal selection stabilization stays excluded (description-
task analysis: cross-tubelet variation isn't harmful for whole-clip
description; a streaming-UI concern only).

### Gate row added 2026-07-15 — first trained-v1 measurement (local, 60 steps)

`outputs/borissal/gate_v1_recipe_local.json` (HF vitl-256 teacher, example
clip; recipe = warmup 20 + uniqueness 1.0 + floor 8.0, GCNet-lite on):

| selector | ratio 0.25 cov(<) / uniq(>) | ratio 0.5 cov(<) / uniq(>) |
|---|---|---|
| random | 8.239 / 7.933 | 8.065 / 8.055 |
| v0.2 | 8.226 / **8.370** | 8.013 / **8.284** |
| v1 (60-step local) | 8.497 / 7.854 | 8.406 / 8.086 |

Honest read: the 60-step local model does NOT yet beat random on its own
uniqueness objective (7.85 < 7.93 at 0.25) — uniqueness_reward stayed
essentially flat during the run (-8.03→-8.13). Expected at this scale
(1 duplicated clip, zero-init context path barely open at |w|=0.003), but
it sets the bar the scale run must clear early (training.md §7 ladder,
item 4). Notable disagreement: the visually-pleasing selection (screen
diagram content, no edge bands) measured as LOW-uniqueness — human visual
appeal and the predictor metric can conflict; report both, downstream
captioner is the eventual referee.

### v0 vs v1 latency profile (2026-07-15, current architecture)

`scripts/borissal_benchmark.py --model both` (Mac CPU, B=1 T=16 384², 50
iters) + component breakdown (ratio 0.25):

| config | CPU ms/clip | note |
|---|---|---|
| v0 (v0.1 defaults) | 7.1–7.6 | |
| v0.2 preset (deploy baseline) | 15.4 | incl. coarse-to-fine pass |
| **v1 default (both + cosine + gctx)** | **22.0–23.0** | within the 25ms budget, tight |
| ├ `_grid_inputs` (internal v0 maps + pixel downsample) | 13.3 | **the dominant cost** |
| ├ CNN trunk (stem + 3 TSM + gctx) | 7.7 | |
| v1 `input_mode=pixels` (skips v0 maps) | 12.0 | cheapest v1; ablation lever |
| v1 `global_context=False` | 21.8 | **gctx costs ~0.2ms — negligible** |
| v1 `hidden_channels=32` | 19.1 | second lever |

Reading: v1 ≈ 3× v0.1 but only ~1.4× the v0.2 deploy baseline, and v1
REPLACES v0 (the 13.3ms v0-map computation is inside it, not on top). If
the mobile budget tightens on-device, the levers in order are
`input_mode=pixels` (−10ms, needs the §3 input-mode ablation to confirm
quality) then `hidden_channels=32` (−3ms). MPS numbers are pathological
for BOTH models (184ms v0 / 460ms v1 — per-op dispatch overhead on tiny
grid tensors, consistent with the original mobile review; mobile delegates
are a different runtime, CPU is the honest proxy). Raw tables:
`outputs/borissal/benchmark/latency_{cpu,mps}.json`.

### Cross-family gate: VideoMAE reconstruction (2026-07-15)

`scripts/eval_borissal_videomae_recon.py` scores selections with the
ORIGINAL AutoGaze VideoMAE checkpoint (`bfshi/VideoMAE_AutoGaze`,
videomae.pt) — the exact model family behind AutoGaze's own RL reward, and
an axis INDEPENDENT of our V-JEPA training teacher. Wiring notes: multi-
scale per-frame layout (32+64+112+224 → 265 tokens/frame; Borissal maps
onto the finest-scale block via `adapters.to_videomae_gazing_info`,
tubelet→2-frame duplication), DDP "module." prefix stripped, loss_type=l1
only (checkpoint's dinov2/siglip2 heads skipped — need flash-attn + extra
models; irrelevant for comparing selections), transformers-5.5.0 pruning-
API shim installed script-locally. Checkpoint loads with 0 missing keys.

First measurement (example clip, ratio 0.25, recon frames 1..15 odd):

| selector | recon L1 (<) |
|---|---|
| random | **0.183** |
| v0.2 | 0.338 |
| v1 (60-step local recipe) | 1.224 |

Reading: EXACTLY the theory-predicted ordering — a reconstruction-family
metric rewards scatter, so random wins and the deliberately-concentrated
selections lose; the strips confirm the mechanism (random reconstructs the
whole frame faithfully; v1's concentrated selection leaves the background
hallucinated). Two conclusions: (a) the canonical keep-index contract runs
END-TO-END into the original AutoGaze task (first real consumer — the
adapter is the integration point downstream will reuse); (b) this metric
shares coverage's scatter bias and is a CROSS-FAMILY REFERENCE AXIS, not
an adoption gate — do not optimize toward it blindly. Raw:
`outputs/borissal/videomae_recon/results.json` + strips (gitignored).

### Mobile-export pre-check (2026-07-15)

`scripts/export_borissal_check.py`: v0.2 and v1 (cosine + global context)
both PASS torch.jit.trace AND ONNX opset-17 export (v1 → 4.2 MB). Two
real bugs found and fixed by the check: (1) `_pack_gazing_mask`'s
`stable=True` argsort lowered to `aten::sort.out`, unsupported by the ONNX
exporter — removed, safe because the sort keys are UNIQUE by construction
(kept→idx, dropped→N+idx), so plain argsort is already deterministic;
(2) `_selection_from_scores` unpacked `S.shape` without `int()` casts,
breaking Python `round()` budget arithmetic under trace (same pitfall
class as v0's earlier fix). CoreML conversion deliberately deferred to
the Linux/CI side (onnx→coreml); this check's job is catching op/trace
problems on the dev box.

## Description-task alignment (2026-07-15): semantic gate + hybrid allocation

User-driven reframe: the selection exists FOR video description, and both
existing gates are reconstruction-family (random provably/measurably wins
them) — a judgment axis aligned with description was missing. Theory
session conclusions (recorded from the dialogue):

- **"Spread evenly + concentrate on what matters" is AutoGaze's own
  multi-scale essence**: its 3 coarse scales (4+16+49 = 69 of 265 tokens,
  ~26% of every frame's budget) are a STRUCTURAL global-gist reservation;
  fine tokens carry detail. Borissal is single-scale by downstream
  contract (V-JEPA2.1 grid), so the closest analogue is a **stratified
  spread share** — sampled points + RoPE positions instead of per-token
  summaries (honest gap: a coarse token SUMMARIZES its region,
  anti-aliased; a stratified fine token SAMPLES it).
- Target budget confirmed by the user: **gazing_ratio 0.25–0.5** (not
  AutoGaze-RL's 0.02–0.15 ultra-sparse regime) — at 0.25, an s=0.25
  spread share still leaves a 6x6 per-tubelet skeleton plus ~108
  focus tokens/tubelet, arithmetically enough for scene gist + 2-3
  recognizable objects.

**Mechanism — `spread_fraction` hybrid allocation** (`_hybrid_topk`,
config `spread_fraction`, runtime override on both v0/v1 `select()`):
focus share = plain top-k; spread share = per-time-slice quota (largest
remainder) over >= k_spread spatio-temporal buckets, best cell per bucket.
Time is stratified FIRST (event coverage; a tie-broken plain bucket top-k
starved late time slices — caught by test). Inference-only (training path
untouched); uniform (2D in-tubelet) and global (3D clip-wide) modes;
incompatible with proportional. Export: v1+global+spread traces and
exports (found+fixed: scalar-True `scatter_` on bool tensors is
untraceable — tensor src now).

**Semantic gate** (`scripts/eval_borissal_semantic.py`, SigLIP2
`google/siglip2-base-patch16-384` — 24x24 tokens, EXACT 1:1 with the
384/patch16 main-target grid; eval-only, never in the training loop):
- **Metric-design pitfall (first attempt, kept as a warning)**: mean-pool
  gist is won by random (sample mean -> population mean) and
  cosine-to-mean "importance" marks TYPICAL patches — random recall
  landed exactly at chance (=ratio) and saliency scored BELOW chance.
  Both metrics now use the MAP (attention-pooling) head: gist = probe
  pooling of the selected subset vs the full frame; recall = fraction of
  the head's top-10%-attention patches captured.
- **First results where saliency BEATS random** (pre-registered
  expectation met — the design-review alarm does not fire):

| selector (ratio 0.25) | gist(>) | recall(>) |
|---|---|---|
| random | 0.945 | 0.267 |
| v0.1 / v0.2, s=0 | 0.880 / 0.879 | 0.296 / 0.300 |
| **v0.2, s=0.25** | **0.899** | **0.309** |
| v0.2, s=0.5 | 0.899 | 0.283 |
| v1-60step, s=0 → s=0.5 | 0.835 → 0.914 | 0.205 → 0.276 |

- Readings: (a) recall — the description-aligned axis — favors saliency
  over random at both ratios (0.31 vs 0.27 at 0.25; 0.556 vs 0.524 at
  0.5); (b) gist retains a residual spread tilt (subset pooling of
  scattered tokens approximates the full pooling) — treat recall as
  primary, gist as secondary; (c) **s=0.25 is the sweet spot**: for v0.2
  at the target budget it improves BOTH metrics (and V-JEPA coverage
  8.226→8.211 with uniqueness ~held 8.370→8.353); s=0.5 starts costing
  recall. Matches the AutoGaze coarse-share precedent (~26%).
- Cross-family confirmation (VideoMAE recon L1, ratio 0.25): spread
  recovers most of the reconstruction penalty of concentration —
  v0.2 0.338 (s=0) → 0.287 (s=0.25) → 0.254 (s=0.5); v1-60step
  1.224 → 0.684 → 0.685. Together with the semantic table: s=0.25 buys
  large recon/coverage gains at zero-to-positive semantic cost.
- The 60-step local v1 stays behind v0.2 on every axis (consistent with
  all gates; it is barely trained) — v0.2 + s=0.25 is the best currently
  deployable configuration; the scale run remains the deciding step for v1.

## Theory notes (2026-07-14): what the literature says about our measured pathologies

Two parallel surveys (① token selection / differentiable top-k;
② SSL informative masking / selector training mechanics) mapped onto the
three pathologies measured locally: **P1** scatter bias (random beat every
saliency config on coverage MSE), **P2** score saturation (entropy
5.0→2.64 over 40 steps, grad_norm → ~0), **P3** edge/band drift in the
trained selection.

**P1/P3 are the coverage objective's PROVABLE OPTIMUM, not training bugs.**
- Coverage minimization ≡ soft facility-location / column-subset-selection
  / D-optimal design; reconstruction-optimal subsets are space-filling and
  boundary-heavy (DPP-CSS, JMLR 2020 arXiv:1812.09771; leverage-score
  sampling, arXiv:2302.11474; D-optimal design selects extreme points —
  our P3 exactly). Interpolation error scales with fill distance,
  minimized by grid/blue-noise layouts — our P1 exactly.
- Experimentally reproduced across SSL: UM-MAE (arXiv:2205.10063) — evenly
  scattered visible anchors are a low-level interpolation shortcut
  (pretrain loss ↓ while transfer ↓); I-JEPA ablation (arXiv:2301.08243) —
  scattered context 17.6 vs multi-block 54.2 (IN-1k 1% linear); V-JEPA
  ablation — multi-block 72.9 K400 vs random-tube 51.5.
- REAL-X (AISTATS 2021, arXiv:2103.01890) supplies the gradient mechanics:
  a selector trained against a FROZEN evaluator optimizes the evaluator's
  off-distribution inductive bias unless the evaluator was calibrated on
  the selector's mask distribution. The 2.1 predictor was trained at ~90%
  multi-block masking; our 15–75%-keep scattered contexts are deep
  off-distribution, so the coverage gradient points into its interpolation
  prior.
- Consequence adopted in code: coverage demoted to a CONSTRAINT
  (`--coverage-floor`), uniqueness promoted to primary, block-structured
  training selection (`train_block_size`) removes the scatter shortcut by
  construction AND moves the predictor back on-distribution.

**P2 is the MoE-router "rich-get-richer" pathology, with published fixes.**
ST-MoE (arXiv:2202.08906): z-loss 1e-3 stabilized 3/3 unstable runs with
a quality gain → adopted as default. X-MoE (arXiv:2204.09179): cosine
routing bounds the logit norm by construction → adopted as the v1 score
head. Concrete/Gumbel-softmax literature: τ = 2/3 canonical, τ < 0.5 =
gradient-variance blowup → fixed, no annealing. β-DARTS (arXiv:2203.01665)
notes plain L2 on logits is the WRONG regularizer (post-softmax/logsumexp
targets are right) — why z-loss, not weight decay on the head.

**Closest published analogue to our whole setup:** AdaMAE (CVPR 2023,
arXiv:2211.09120) — a tiny selector trained by REINFORCE against a frozen
reconstruction teacher, reward detached from the teacher graph; foreground
saliency EMERGES. Adopted as the optional RL phase (with Kool et al. 2019
leave-one-out baseline). Notably, NO published work trains a token
selector against a frozen JEPA predictor — our measured scatter-bias
result, which matches the facility-location theory exactly, is itself a
citable negative finding for the eventual paper.

**Surveyed and deliberately NOT adopted:**
- DPP / diversity regularizers (CDPruner etc.): our objective is already
  over-spread; diversity pressure is the wrong direction until
  uniqueness-primary training over-concentrates (revisit then).
- SIMPLE (ICLR 2023) / perturbed top-k (Berthet 2020, Cordonnier 2021):
  lower-bias k-subset estimators, but the saturation bundle + RL phase
  cover the same failure modes; SIMPLE's O(nk) DP is heavy at n=4608.
- GradNorm / uncertainty weighting: fixed, coarsely-tuned weights match or
  beat adaptive balancers head-to-head (arXiv:2201.04122).
- SemMAE part-learning / AutoMAE GAN prior: block sampling achieves the
  contiguity prior directly, without the machinery.
- Deferred to Linux compute (training.md §7.8): REAL-X predictor
  calibration adapter, EVAL-X audit gap, Frame-Voyager caption-loss
  model selection.

## Model diagnostics & global context (2026-07-15)

A user-prompted architecture review ("is the model itself the problem — does
it just pass its initial guidance through?"), settled with measurements
(`scripts/borissal_model_diagnostics.py` re-runs all of these):

**Rejected concern — v0 passthrough.** Untrained v1 scores vs v0 saliency:
Spearman ≈ 0.03–0.08 across seeds; after the WP-A 40-step run: −0.08 (and
−0.09 vs the untrained model). v1 neither reproduces v0 nor freezes at its
init. (Init note: a shallow random CNN's scores DO correlate with RGB
brightness at init — measured anywhere from −0.3 to +0.86 depending on
seed/scale; harmless, training reshapes it immediately, but the diagnostics
track it.)

**Confirmed gap — no learnable global pathway.** The TSM stack's receptive
field is ~9×9 grid cells (4 stacked 3×3 convs); a corner perturbation moves
nearby scores ~55–200× more than far-corner scores (the residue is GroupNorm
statistics leakage, not content routing). The v0 input maps carry only a
weak global signal (min-max normalization = 2 scalars per frame). Meanwhile
coverage/uniqueness objectives ask a CLIP-GLOBAL question — a local scorer
is being trained toward something it cannot express, and this worsens under
the uniqueness-primary recipe.

**Related finding — the AutoGaze precedent legitimizes proxy-only training.**
The original pipeline's GRPO stage rewards `-VideoMAE_reconstruction_loss`
only (task_video_mae_reconstruction.py:130-140); the real downstream
(NVILA-8B-HD, HLVid QA) attaches later in a separate repo. And
`gazing_labels.json` is NOT human gaze — it is precomputed
reconstruction-optimal selection orders (another reconstruction-family
proxy). So training Borissal v1 purely against predictor-based proxies,
with downstream attached later, follows the project's own precedent — the
earlier idea of demoting uniqueness to "experimental" was over-conservative
and was withdrawn.

**Fix adopted — GCNet-lite learned global context** (`_GlobalContext` in
modeling_borissal_v1.py, `global_context=True` default):
- One 1×1 attention conv scores positions; softmax within each frame gives
  a per-tubelet context, softmax over all (t, position) gives a clip
  context (weighted — a few high-motion frames aren't diluted like a
  uniform mean); concat → zero-init 1×1 transform → added to features.
- Chosen over (a) plain mean pooling (dilutes sparse important content;
  GCNet showed learned weighted pooling captures most of full attention's
  benefit) and (b) full pairwise attention (~21M pairwise ops per layer at
  L=4608 — threatens the 25ms mobile budget and the raison d'être of the
  feed-forward design).
- Injected BEFORE the last TSM block: the context is per-frame constant, so
  added at the head it would interact with positions only through the
  cosine normalization (a linear head + per-frame softmax would erase it
  entirely via shift invariance); before the last block, its conv+GELU
  mixes local×global per position.
- Zero-init transform = exact no-op at init (preserves WP-A-validated
  early dynamics). Known transient: the attn conv receives zero gradient
  for exactly one step (measured: step 0 = 0, step 1 = healthy).
- Cost: +8.3K params (114K → 122K); CPU select() latency unchanged
  (14.5ms vs 16.0ms off — within noise, budget 25ms). Ops: conv, softmax,
  mul+sum only — mobile constraints intact.
- The canonical Selection contract path (`_selection_from_scores` →
  `_pack_gazing_mask`, ascending `idx = t*N + n`) is untouched (user
  requirement: downstream attachability).

## Open items (updated)

- torch.hub V-JEPA2.1-L/B teacher adapter (three wrapper methods) — when
  large-scale training moves to Linux/CUDA and the team's existing 2.1
  checkpoints should be the teacher.
- Which loss combination / input_mode wins — experiment matrix in
  `training.md` §3, to be run at scale; §8 phased recipe
  (warmup → ST uniqueness-primary → optional RL) is the default plan.
- Predictor fine-tuning (currently frozen) as a later option.
- Inference-side block selection for v1 `select()` if the block-trained
  selector wins the eval gate (training-side only for now).
- **Description-aligned auxiliary distill (E4, designed 2026-07-15, not
  implemented)**: add the patch-attention of a frozen VLM as an auxiliary
  distill target alongside the V-JEPA SSL objective, on the user's
  intuition that description quality is object-driven. Constraints that
  make this safe: (a) the distill VLM must NOT be SigLIP2 — SigLIP2 stays
  the held-out judge or the semantic gate stops measuring anything
  (eval contamination); candidates: a different CLIP-family encoder's
  attention-pool head, or a small frozen captioner's cross-attention
  rollup; (b) never train the selector jointly with any evaluator it is
  scored by (L2X "selection as communication" degeneracy, same reason
  the REAL-X calibration adapter is trained on random masks only);
  (c) keep it a WEIGHTED AUXILIARY (`--w-sem-distill`) so the SSL
  objective stays primary — this is a prior, not a new objective.
  TRIGGER to implement: E0 (pilot extension) and E3 (block2) trend runs
  AND the Linux scale run all leave semantic recall flat — i.e., the
  pure-SSL signal is shown insufficient for the description-aligned
  axis. Until then it stays on the shelf.

## Borissal v0.3 solo screening (2026-07-18): 4 KEEP / 5 KILL / 1 TUNE applied

Stage-1 of the v03-design.md §5 protocol, run with
`scripts/sweep_borissal_v03.py --stage solo --ratio 0.25`. Semantic gate:
held-out `videos/internvid_eval16/` (16 clips), v0.2-base recall 0.325
(exactly reproduces the documented 0.325±0.022 baseline). Coverage gate:
first 4 clips (pilot-gate precedent), HF vitl-256 teacher. Ties resolved by
paired per-clip comparison per the spec rule. Raw tables:
`outputs/borissal/v03_sweep/{solo,cov4,coh_tuned,coh_tuned_cov4}/` (gitignored).

| candidate | recall paired (16 clips) | cov(<)/uniq(>) vs 8.238/8.106 | lat ms | verdict |
|---|---|---|---|---|
| fusion_peak | **13W-3L, +0.015** | 8.214 / 8.105 | 16.3 | **KEEP** (strongest; Itti N(·) delivers) |
| coherence_gate (ds=4 TUNE) | **12W-4L, +0.012** | 8.212 / 7.993 (one-axis trade, allowed) | 18–25 | **KEEP** after TUNE |
| dog_blob | 9W-7L, +0.010 | 8.222 / 8.156 (Pareto-better) | 18.6 | **KEEP** (also best gist +0.02) |
| color_rarity | 9W-6L-1T, +0.004 | 8.233 / **8.220** (largest uniq gain) | 24.5–25.1 | **KEEP** (latency watch) |
| signature | 8W-7L, +0.005 | 8.257 / 8.102 (both degraded) | 18.9 | KILL (coin-flip recall + both-axes rule) |
| fusion_entropy | 1W-0L-15T, +0.000 | ~flat | 16.2 | KILL (practical no-op on real clips — the mass-entropy gate's theoretical camera fallback never fires on this data; knob stays experimental) |
| motion_center_surround | **4W-12L, −0.013** | 8.244 / 8.073 (both degraded) | 16.6 | KILL (InternVid has little ego-motion; the surround subtraction only eats informative motion. Revisit per-domain for pan-heavy footage — the mechanism itself is verified on synthetic flicker) |
| score_ema | 6W-10L, −0.002 | 8.250 / 8.086 (both degraded) | 15.6 | KILL (fails the stability-knob recall-non-degradation admission condition) |
| hysteresis | 4W-9L-3T, −0.002 | 8.242 / 8.084 (both degraded) | 15.6 | KILL (same condition; both stability knobs remain available as off-default deploy knobs for streaming UIs) |

**TUNE record (coherence_gate)**: the solo latency gate failed hard (61ms vs
25ms budget — the three stride-1 pixel-res smooths, ~11ms each, not the
kernel size; count_include_pad is irrelevant, measured). Fix: average the
gradient PRODUCTS into ds×ds blocks (strided pool) before the kernel smooth
— itself valid structure-tensor windowing, and safe for fine gratings
(their dx² stays large at any period; downsampling the SIGNED gradients
would cancel them instead — ordering matters). `coherence_downsample=4`
default; 60.5→18.3ms standalone (24.6–25.0ms inside the sweep process).
Recall actually improved after the TUNE (0.330→0.336; 12W-4L) — the larger
effective window (4px blocks) appears to help.

Greedy stage-2 order (by solo paired recall): fusion_peak →
coherence_gate → dog_blob → color_rarity.

## Borissal v0.3 greedy combination + preset admission (2026-07-19)

Stage-2/3 of the v03-design.md §5 protocol, additions in solo-recall order,
acceptance = all gates pass AND paired recall not degraded vs. the previous
chain. Raw tables: `outputs/borissal/v03_sweep/greedy_*`, `r{1,2}_cov4/`,
`ratio05_spotcheck.json` (gitignored).

| step | chain | recall (16 clips) | paired vs prev | cov(<)/uniq(>) (4 clips) | verdict |
|---|---|---|---|---|---|
| 0 | v0.2 base | 0.325 | — | 8.238 / 8.106 | — |
| 1 | + fusion_peak | 0.339 | 13W-3L, +0.015 | 8.214 / 8.105 | ACCEPT |
| 2 | + coherence_gate (ds=4) | 0.346 | 9W-5L, +0.007 | 8.171 / 8.069 (one-axis trade) | ACCEPT |
| 3 | + dog_blob | 0.344–0.351 | 10W-6L, +0.005 | **8.174 / 8.167 — Pareto-better than v0.2 on BOTH axes** | ACCEPT |
| 4 | + color_rarity | 0.343 | **6W-10L, −0.008** | 8.182 / 8.188 | REJECT |

**color_rarity rejection detail**: its solo signature (largest uniqueness
gain, +0.114) reappears on the chain (uniq 8.167→8.188), but the PRIMARY
axis drops (6W-10L) and its real +7.5ms latency (measured with the coarse
pass isolated; NOT thermal noise) would push the chain to ~32ms vs the
25ms budget. Kept as the top TUNE-later candidate: if a future round wants
the uniqueness, the cost lives in the soft-binning/rgb path, and the
recall drop suggests trying it INSTEAD OF (not on top of) dog_blob.

**Latency note for future sweeps**: in-process latency readings drift
+30-70% after hours of continuous CPU load (thermal) — e.g. the chain read
26-43ms inside sweeps but 24.5ms clean. Solo/chain admission used clean
standalone re-measurement (`borissal_benchmark`-style probe); do the same
before trusting any in-sweep `lat` column.

**Preset admitted**: `BorissalConfig.v0_3()` = v0.2 + `fusion_norm="peak"`
+ `coherence_gate=True` (ds=4) + `dog_blob_weight=0.5`.
- Held-out semantic recall (ratio 0.25): 0.325 → 0.346–0.351 — the first
  non-learned config to clearly beat both v0.2 AND the pilot v1@1000
  (0.315) on the primary axis.
- V-JEPA cov/uniq: Pareto-better than v0.2 (8.174/8.167 vs 8.238/8.106).
- CPU latency ~24.5ms clean (budget 25; tight — the ds=4 coherence TUNE is
  what made it fit).
- Export: jit.trace + ONNX opset-17 PASS (`export_borissal_check.py` v0.3
  case now runs the preset).
- Interaction checks per §5: fusion×channels covered by the chain itself
  (fusion_peak adopted first, every later channel measured on top of it);
  score_ema×select_hysteresis moot (both KILLed solo).
- Ratio-0.5 spot check (16 clips): recall 0.560 -> 0.577 (paired
  +0.017, 10W-6L), gist 0.946 -> 0.950 -- the preset's edge holds at the
  upper end of the target budget range too (`ratio05_spotcheck.json`).

Follow-ups (not blocking): Tier-2 triggers per §5.3 — the texture axis
moved (coherence+dog admitted), camera axis unresolved on this eval set
(InternVid has little ego-motion; `gme`/`motion_cs` need a pan-heavy set
to be judged fairly); v1 retraining on the enriched input bank; E5
(learned cross-frame budget allocation, spec §7.5).

## Borissal v0.3.x efficiency/allocation round (2026-07-19): user-driven review

Four follow-ups from a user review of the v0.3 pipeline ("이중 패스 낭비 /
tubelet 평균 전 프레임 처리 / top-k 할당 개선"), handled as two
behavior-preserving changes + two sweep-gated candidates. Raw tables:
`outputs/borissal/v03_sweep/v03x*/` (gitignored).

**Behavior-preserving (adopted outright):**
- *Allocation clarity refactor*: the global-allocation "+10 bonus" trick
  rewritten as an explicit two-step (per-tubelet guaranteed top-m, then
  free budget by clip-wide top-k over the rest) — equivalent selection,
  regression-covered; plus a `max_keep_per_frame_mult` CAP knob symmetric
  to the floor (mult=1.0 provably degenerates to uniform; boundary tests).
- *Luma-space coarse resize*: the recompute block-gate path resizes 1
  luma channel instead of 3 RGB channels when no color channel is active
  (linear ops commute; verified ~1e-7 coarse-score deltas, gate top-k
  indices identical on real+random clips).

**Sweep-gated (judged on top of the v0.3 preset, 16-clip semantic +
4-clip cov + clean latency):**

| candidate | paired recall | cov/uniq | latency (clean, normalized) | verdict |
|---|---|---|---|---|
| `block_gate_source="pool"` | **0W-0L-16T — selections IDENTICAL** to recompute at 384 | Pareto-better vs v0.2 held | ~24 → **~17ms** (−7ms, single pipeline pass) | **ADOPT** |
| `spatial_diff="frame"`+`agg="max"` (initial) | +0.0043, 10W-4L | — | +12ms → chain ~29ms (over budget) | TUNE |
| same, TUNE: gate stays tubelet-granular | +0.0030, 8W-6L-2T | 8.179/8.133 vs 8.175/8.118 (one-axis trade) | +6ms → chain **~22ms** | **ADOPT (weakest accepted margin — re-check at scale)** |
| `spatial_frame_mean` | +0.0014, 9W-4L-3T | — | — | dominated by max variant |
| `max_keep_per_frame_mult=2.0` | 0W-0L-16T — never binds (block gate already caps exposure ~2x share) | — | free | keep OFF; safety knob for pathological concentration / block_size=1 deployments |

**TUNE record (frame-granular spatial)**: the initial variant recomputed
the coherence gate per frame (~2x coherence cost, +12ms total). Coherence
is a smoothed regional texture statistic, so per-frame recompute buys
nothing: the TUNE computes the gate once from tubelet-mean gradients and
applies it to the frame-aggregated magnitude (+6ms total). The paired
margin softened after the TUNE (10W-4L → 8W-6L-2T) but stays positive;
accepted under the non-degradation rule with an explicit scale-run
re-check flag.

**Updated `v0_3()` preset** (v2): + `block_gate_source="pool"`,
`spatial_diff="frame"`, `spatial_agg="max"`. Recall 0.354 (16-clip,
ratio 0.25) vs v0.2 0.325; clean-normalized latency ~22ms (was ~24.5);
tests 80/80; jit.trace + ONNX PASS. The pool adoption also supersedes the
luma-resize optimization on the preset path (recompute retained for
`block_gate_source="recompute"` users).

## Allocation-policy comparison (2026-07-19, user-driven): uniform wins; preset default flipped

User question: "uniform vs global의 자세한 비교 — description에 맞는 배분을
나중에 제어하고 싶다." This exposed a measurement gap: ALL semantic-recall
numbers to date used the uniform variant (the MAP-head metric's equal-count
requirement); global allocation's semantic effect had never been measured.
Fixed by generalizing the metric to variable per-frame counts (recall is
count-agnostic as-is; gist via per-frame probe pooling), then comparing 6
policies on the v0.3 signal stack (16 clips, both ratios). Raw:
`outputs/borissal/v03_sweep/allocation_policies.json`.

| policy | recall @0.25 (vs unif) | recall @0.5 (vs unif) | uniq/cov @0.25 (4 clips) |
|---|---|---|---|
| **uniform** | **0.3615** | **0.6034** | **8.191** / 8.183 |
| global+floor (0.10/0.25/0.50 identical) | 0.3573 (9W-7L) | 0.5836 (**2W-12L**) | 8.133 / 8.179 |
| proportional | 0.3560 (7W-9L) | 0.5803 (3W-11L) | — |
| global+spread .25 | 0.3556 (8W-8L) | 0.5837 (4W-12L) | — |

Findings:
1. **Uniform wins the primary axis decisively at ratio 0.5** (12/16 clips)
   and ties at 0.25. Interpretation: with a generous budget, concentration
   has nothing left to buy — coverage dominates, and global's drained
   tubelets lose their top-attention patches.
2. **The floor dial is DEAD on this data**: floors 0.10/0.25/0.50 produce
   identical selections — every tubelet naturally wins more than the floor
   in free competition (per-frame std ~29 on mean 144), so the knob never
   binds. The v0.2-era "floor preserves coverage" story was measured on the
   example clip; on real clips the coverage problem it solves doesn't occur.
3. **Uniform also wins uniqueness with the v0.3 stack** (8.191 vs 8.133,
   cov ~tie +0.004): the v0.2-era "global+floor strongest element" finding
   (example clip, v0.1 signals) does NOT carry over to v0.3 signals on
   real data.
4. Consequence: `v0_3()` preset default flipped to
   `per_frame_allocation="uniform"` — also the trace/export-safe
   (data-independent) policy per the mobile review. global/proportional/
   floor/cap/spread all remain as knobs; cap tests pinned to global (the
   only branch where cap exists). 80/80 tests.

Control guidance for description (recorded for the eventual allocation
work): the meaningful dial is NOT floor size but the uniform<->global
choice itself, plus spread. The path to content-adaptive allocation is E5
(learned, oracle-distilled) — rule-level dials measured so far don't move
the semantic axis except to lose.

Also per user review: the research-positioning claim comparing v0.3 to the
PILOT v1@1000 was removed from the dashboard/features doc — a 1K-clip
batch-2 pilot is not a fair learned-model baseline; the scatter-bias
argument stands on coverage-gate measurements alone.
