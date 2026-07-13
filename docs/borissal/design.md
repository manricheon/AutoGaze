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
- **top-k under a ratio budget** (`gazing_ratio` sets how many patches survive)

The full effort is split into three phases, each its own spec/branch:

1. **Phase 1 (this doc, done)** — Borissal-signal: a non-learned, saliency-based
   feed-forward selector. Mobile-oriented (low latency, no learned weights).
   Branch `feat/selector`.
2. **Phase 2** — Borissal-learned: a trainable selector (TSM or conv3d backbone +
   scoring head + differentiable/straight-through top-k), using the Phase 1
   signal as a baseline/feature input. Branch `feat/selector`.
3. **Phase 3** — self-supervised training via V-JEPA2.1L: compare V-JEPA2
   features on a dense (full) video vs. a sparse (selector-chosen) video to
   train the selector without ground-truth gazing labels. New task +
   algorithm, likely on `feat/train`.

## Key design decisions (and why they depart from AutoGaze conventions)

| Decision | Rationale |
|---|---|
| Output is **grid_thw-native**, not AutoGaze's `gazing_pos` dict contract | The intended downstream encoders (V-JEPA2, Qwen-VL-style models) already think natively in `(t, h, w)` token grids with flat/gather-based sparsity. Conforming to AutoGaze's dict (`gazing_pos`, `if_padded_gazing`, `gazing_mask` per scale, ...) would have made the *first* integration easy but every *subsequent* encoder integration harder. An optional adapter (`adapters.to_autogaze_gazing_info`) bridges back to the legacy contract for sanity-checking against the existing VideoMAE task, but it is not the native format and only supports uniform per-frame allocation. |
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
   (`motion_weight = w`).
7. Budget allocation (rule-based, not learned):
   - `per_frame_allocation="uniform"` (default): every tubelet keeps
     `k = clamp(round(gazing_ratio * H_grid*W_grid), 1, H_grid*W_grid)` patches.
   - `"proportional"`: total budget `round(gazing_ratio * L)` distributed across
     tubelets proportional to per-tubelet energy, via largest-remainder
     (Hamilton's method) rounding so counts sum exactly to the budget (mod
     per-tubelet `[1, N_pf]` clamping).
8. Top-k → `keep_mask` via a fully vectorized rank comparison (`argsort` twice),
   no Python loops over batch/frame.
9. Packing to `keep_index`/padding reuses
   `autogaze/utils.py::get_gazing_pos_from_gazing_mask` (stable ones-first
   sort) applied to the flattened `(B, L)` mask — this is the one piece of
   legacy AutoGaze code Borissal imports, and it has no transformers/legacy-model
   dependency (verified: only `torch`/`numpy`/`omegaconf`/`loguru`/`wandb`).

## Files

- `autogaze/models/borissal/configuration_borissal.py` — `BorissalConfig`
  (plain dataclass; no HF `PretrainedConfig` needed since Phase 1 has no
  learned weights to checkpoint).
- `autogaze/models/borissal/modeling_borissal.py` — `Borissal(nn.Module)` +
  `Selection` dataclass + the algorithm above.
- `autogaze/models/borissal/adapters.py` — `to_vjepa2` (passthrough/rename for
  a V-JEPA2-style gather-before-transformer attach point) and
  `to_autogaze_gazing_info` (optional bridge to the legacy dict contract, for
  running Borissal through the existing VideoMAE task as a sanity check;
  requires uniform per-frame allocation).
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

## Open items for Phase 2/3 (tracked here so they aren't lost)

- Confirm the exact V-JEPA2.1L checkpoint/repo id and its native token flatten
  order (verify it matches Borissal's t-major assumption, or add a remap in
  `adapters.to_vjepa2`).
- Decide whether Phase 2's learned scoring head consumes Borissal's raw
  motion/spatial maps as input features or only its final scores.
- Phase 3: where exactly to apply sparsity relative to V-JEPA2's conv3d
  tubelet embedding (after conv3d, before the transformer — per V-JEPA2
  predictor-based SSL guidance referenced in the original brainstorm).
