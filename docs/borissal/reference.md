# Borissal-signal: Reference

A saliency-based, feed-forward patch selector. Non-learned, single-scale,
top-k under a fixed budget. This is the "what and why" reference; see
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

**Where this fits in the bigger picture.** Borissal-signal is Phase 1 of a
three-phase line: a learned selector (Phase 2) will replace this hand-built
saliency score with a trainable one, and Borissal's motion/spatial maps are
the natural starting point — either as literal input features or simply as
the baseline it needs to beat. Phase 3 trains that learned selector
self-supervised against V-JEPA2 (dense-vs-sparse feature comparison). None
of that changes what's described below; it's the reason this exists.

## 2. Algorithm

Given a clip `(B, T, C, H, W)`:

1. **Luma** — average channels: `gray = video.mean(dim=2)`.
2. **Tubelet aggregation** — average every `tubelet_size` consecutive frames
   into one tubelet: `(B, T_grid, H, W)`, `T_grid = T // tubelet_size`.
3. **Motion** — absolute difference between consecutive tubelets (a
   residual/motion-vector proxy, computed directly on pixels rather than a
   real codec's MV field).
4. **Spatial** — gradient magnitude (`spatial_op="grad"`, cheap finite
   differences) or a fixed Sobel kernel (`spatial_op="sobel"`).
5. **Patch pooling** — average- or max-pool both maps down to the patch grid
   `(H_grid, W_grid) = (H, W) // patch_size`.
6. **Normalize & combine** — min-max normalize motion and spatial per
   (instance, tubelet) to `[0,1]`, then blend with one weight:
   `score = motion_weight * motion + (1 - motion_weight) * spatial`.
7. **Budget allocation** — turn `gazing_ratio` into a per-tubelet patch
   count, either `uniform` (same count every tubelet) or `proportional`
   (count follows each tubelet's total score energy).
8. **Top-k → keep mask** — `torch.topk` per tubelet (chosen over a
   sort/argsort-based rank, which has weaker mobile-runtime operator
   support), producing a boolean keep mask.
9. **Pack to global indices** — flatten to `keep_index`/`keep_coords` in
   `(t, h, w)` grid_thw space via a stable-sort packer (`_pack_gazing_mask`,
   self-contained — see §6) that also guarantees ascending order (§5).

## 3. Config knobs

| Knob | Default | What it controls | Why this default |
|---|---|---|---|
| `scale` | `384` | Frame side length (square) | Matches the intended V-JEPA2 input size |
| `patch_size` | `16` | Spatial patch edge | Matches V-JEPA2's patch size |
| `tubelet_size` | `2` | Frames per temporal token | Matches V-JEPA2's tubelet size → `grid_thw=(8,24,24)` |
| `motion_weight` | `0.5` | Motion vs. spatial blend | Neutral default; content-adaptive tuning is future work (Phase 2) |
| `spatial_op` | `"grad"` | Gradient method | Cheapest option (no conv2d call) with no measured quality gain from Sobel |
| `pooling` | `"avg"` | Pixel→patch reduction | Standard, cheap, well-supported everywhere |
| `gazing_ratio` | `0.5` | Fraction of patches kept | Caller-set budget |
| `per_frame_allocation` | `"uniform"` | How the budget splits across tubelets | Only variant that *exactly* hits the requested ratio every time; `"proportional"` is content-adaptive but can drift by a patch or two under extreme energy skew |

`spatial_op="grad"`, `pooling="avg"`, and `per_frame_allocation="uniform"`
were each chosen as the faster and/or ratio-safer of their alternatives —
together they're already the fast+exact-budget preset. `motion_weight` is
the one knob whose best value genuinely depends on the clip's content
(static talking-head vs. high-motion footage); it isn't auto-tuned yet (see
§8).

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
Sort-family ops (the one remaining stable-sort in the packer, plus
`torch.topk`) are under 6% of self-CPU time; the rest is ordinary
elementwise/pooling arithmetic, universally supported on mobile backends.
An empirical `torch.jit.trace` pass caught and fixed a real tracing bug, and
surfaced a genuine correctness trap in `"proportional"` allocation under
naive tracing (it silently freezes a data-dependent value). Full numbers,
profiler breakdown, and the mobile-export constraints this implies for
Phase 2 are in
[`design.md`](./design.md#mobile-readiness-review-2026-07-14-before-starting-phase-2) —
not duplicated here.

## 8. Not yet built

Content-adaptive auto-tuning (starting with `motion_weight`) was scoped and
designed but deliberately deferred — it'll be revisited alongside or after
the learned selector (Phase 2), rather than bolted onto the signal-only
version now.
