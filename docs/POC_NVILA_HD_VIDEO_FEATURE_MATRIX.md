# PoC NVILA-HD-Video Feature Matrix

This document audits the intended feature combinations for `scripts/poc_nvila_hd_video.py`
against the current implementation.

Reference priority:

1. `docs/POC_NVILA_HD_VIDEO_FEATURE_MATRIX_REQUEST.md`
2. `docs/NVILA_HD_VIDEO_REFERENCE.md`
3. `docs/QUICK_START_reference.md`
4. original `QUICK_START.md`
5. original `INTEGRATION.md`

Original AutoGaze source files, original `QUICK_START.md`, original `INTEGRATION.md`,
and `docs/nvila-hd-video-readme.md` must not be modified for this PoC layer.

## Status Vocabulary

| Status | Meaning |
|---|---|
| implemented | Present in code and covered by tests or smoke behavior. |
| partially implemented | Usable for a constrained PoC path, but missing required request details. |
| stub-only | Explicitly guarded, skipped, or raises a clear error. |
| blocked | Requires missing external model code, checkpoints, APIs, or hardware validation. |
| future work | Not implemented and not required for the next minimal batch. |

## Current Script Surface

Current `scripts/poc_nvila_hd_video.py` supports:

```text
--mode check/autogaze_only/full_pipeline
--video dummy
--video-path <path>
--query-text <text>
--frame-selection-mode sample/chunk/interval/all
--num-frames <int>
--frame-interval <int>
--max-windows <int>
--drop-last
--pad-last
--scaling-mode none/resize/fit_short_side/fit_long_side/quickstart/chop
--resolution <int>
--patch-size <int>
--target-scales <comma-or-plus-list>
--target-patch-size <int>
--spatial-tile-size <int>
--gaze-ratio <float>
--task-loss-requirement <float>
--strict-autogaze-params
--device cpu/cuda/mps
--dtype float32/float16/bfloat16
--max-new-tokens <int>
--config <path>
--allow-checkpoint-load
--no-checkpoint-load
--checkpoint-metadata-only
--save-overlay-video
--save-side-by-side-video
--save-scale-panel-video
--video-export-mode sampled_only/full_length/hold_last
--video-fps <float>
--overlay-alpha <float>
--overlay-line-width <int>
--overlay-style mask/box/both
--multi-scale-overlay / --no-multi-scale-overlay
--scale-color-mode gradient/categorical
--scale-panel-layout 2x2
--show-patch-index
--show-scale-label
--hide-patch-boxes
--hide-patch-indices
--metadata-placement outside/inside/none
--info-panel-position bottom/right
--info-panel-size <int>
--info-panel-mode external/inline/none
--comparison-layout processed_overlay/original_overlay/original_processed_overlay/chop_overlay
--chop-size <int>
--chop-overlap <int>
--chop-stride <int>
--max-chops <int>
--chop-merge-mode none/metadata_only/overlay_union
--save-chop-frames
--save-chop-overlay-video
--json
```

Current script does not expose direct per-component override flags; this PoC uses
`--config` as the supported component plug-in boundary so model/module/checkpoint
settings stay together:

```text
--vision-encoder / --vision-encoder-module / --vision-encoder-class
--vision-encoder-ckpt / --vision-encoder-config
--mllm / --mllm-module / --mllm-class
--mllm-ckpt / --mllm-config / --processor-path / --tokenizer-path
```

Some of these are partially represented by existing flags, for example
`--spatial-tile-size`, `--hide-patch-boxes`, `--hide-patch-indices`, and
`--info-panel-mode`.

## Feature Matrix

### Inference Modes

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| `check` mode | implemented | Imports/classes/paths are checked without heavy construction. | None for current check scope. | N/A | N/A |
| `autogaze_only` mode | partially implemented | Runs frame selection, scaling, guarded AutoGaze, metadata, visualization, and metrics/report output when checkpoints are explicitly allowed. | Real execution still depends on configured AutoGaze module/checkpoint. | Yes | N/A |
| `full_pipeline` mode | partially implemented | Runs same AutoGaze path, then guarded modified SigLIP and NVILA generation when checkpoint loading is explicitly allowed. Query text is not silently ignored. | Real NVILA generation is unverified and memory-heavy. Full processor-first canonical path is not fully validated. | N/A | Yes |
| AutoGaze OFF baseline | partially implemented | A1 config disables AutoGaze import/check requirement. | Full-token downstream baseline path needs explicit full-pipeline validation. | Partial | Partial |

### Video Input

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| Dummy video | implemented | Uses generated tensor data for tests and smoke runs. | None. | Yes | Yes |
| Local video path | partially implemented | Uses PyAV and original `read_video_pyav` if available. | Requires PyAV and original AutoGaze video utilities installed. | Yes | Yes |
| Remote video URL | blocked | NVILA reference uses URL input for official processor path, but PoC local preprocessing expects local video unless using processor generation path. | Add explicit URL handling or restrict to processor-first MLLM path. | Blocked | Blocked |

### Frame Selection

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| `sample` | implemented | Uniformly samples `num_frames` from the whole video. | None. | Yes | Yes |
| `chunk` | implemented | Non-overlapping `num_frames` windows. | None. | Yes | Yes |
| `interval` | implemented | Selects one window using fixed `frame_interval`. | Multi-window interval variants are not requested. | Yes | Yes |
| `all` | implemented | Alias to chunked full-video processing. | Full-length export is supported for non-overlapping windows. | Yes | Yes |
| Sliding stride windows | future work | Not implemented by design. | Do not add unless requirements change. | No | No |

### Scaling and Chop

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| `none` | partially implemented | Leaves selected frames at decoded/generated resolution. | Does not yet fail if a real model requires fixed resolution. | Yes | Yes |
| `resize` | implemented | Resizes to square `resolution x resolution`; may distort aspect ratio. | None. | Yes | Yes |
| `fit_short_side` | implemented | Preserves aspect ratio; short side equals `resolution`. | Real AutoGaze compatibility with non-square inputs needs validation. | Yes | Yes |
| `fit_long_side` | implemented | Preserves aspect ratio; long side equals `resolution`. | Real AutoGaze compatibility with non-square inputs needs validation. | Yes | Yes |
| `quickstart` | partially implemented | Exactly supports documented QUICK_START policies: 224/patch16 and 392/patch14 target-scale mode. Unsupported requests raise and write `unsupported_reason`. | Does not cover every NVILA-HD-Video processor setting. | Yes | Yes |
| `chop` | partially implemented | Uses non-overlapping spatio-temporal tile chunks, supports `--chop-*` metadata fields, writes per-chop outputs under `outputs/chops/`, and supports non-overlap `overlay_union` full processed-frame overlay. | Non-zero overlap/custom stride and original decoded-frame chop overlay remain unsupported. | Yes | Yes |
| Chop overlap/stride | stub-only | CLI fields are present, but non-overlap is the only supported mode. Non-zero overlap or custom stride raises clearly. | Full overlap/stride coordinate semantics are Priority 3/future. | Stub | Stub |
| Original-space chop overlay | partially implemented | Non-overlap `overlay_union` maps chop-local patches to full processed-frame coordinates. | Original decoded-frame overlay for arbitrary chop scaling/overlap remains blocked. | Partial | Partial |

### AutoGaze Runtime Controls

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| `--gaze-ratio` | implemented | CLI/config value is mapped to AutoGaze call as `gazing_ratio`; also forwarded to NVILA processor kwargs as `gazing_ratio_tile` if provided. Requested/effective values are saved in runtime metadata and metrics. Optional strict validation checks callable acceptance before execution. | Real behavioral effect still requires checkpoint execution. | Yes | Yes |
| `--task-loss-requirement` | implemented | CLI/config value is passed to AutoGaze and NVILA processor kwargs. Optional strict validation checks callable acceptance before execution. | Do not claim behavior effect beyond passing the parameter unless real AutoGaze execution verifies it. | Yes | Yes |
| Config defaults | partially implemented | A1/A2 configs include `gaze_ratio` and `task_loss_requirement` fields under `inference`. | Request mentions `autogaze_runtime`; not yet represented as a separate group. | Yes | Yes |
| Equivalent parameter detection | partially implemented | Uses original `gazing_ratio` name for AutoGaze call. | Does not detect `gaze_ratio`, `task_requirement`, or `loss_requirement` callable aliases. | Partial | Partial |
| `--strict-autogaze-params` | implemented | Enables signature validation for requested AutoGaze kwargs. If a configured callable lacks required kwargs, the AutoGaze stage fails clearly and records unsupported params in runtime metadata/metrics. | Does not prove semantic use of a parameter inside the implementation. | Yes | Yes |

### Vision Encoder and MLLM Switching

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| Modified SigLIP + NVILA-HD-Video | partially implemented | Config-driven path resolves module/class/checkpoints; full mode attempts modified SigLIP and NVILA when checkpoint loading is allowed. | Real construction and generation require valid local modules/checkpoints and enough memory. | N/A | Partial |
| Vision encoder config override | implemented as config-driven | Can change by editing or passing `--config`; benchmark configs record the expected config sections. | Direct per-component CLI overrides are intentionally not exposed in this PoC layer. | N/A | Yes |
| MLLM config override | implemented as config-driven | Can change by editing or passing `--config`; processor/tokenizer paths are read from `model.mllm`. | Direct `--mllm-*`, `--processor-path`, and `--tokenizer-path` overrides remain future work if needed. | N/A | Yes |
| Vanilla SigLIP + NVILA | blocked | A0/A3 real configs exist elsewhere, but PoC does not claim compatibility. | Requires explicit feasibility/smoke validation. | N/A | Blocked |
| Qwen / generic HF MLLM | future work | Separate adapter stubs exist in project, not this PoC path. | Need official processor baseline and no direct token injection assumption. | N/A | Future |

### Checkpoint and Config Handling

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| Top-level config selection | implemented | `--config` loads experiment/config YAML. | None. | Yes | Yes |
| Path checks | implemented | `check` mode validates module import/class/path existence. | More detailed per-component override reporting would help. | Yes | Yes |
| Explicit checkpoint loading guard | implemented | Heavy loading is disabled unless `--allow-checkpoint-load` is used. | None. | Yes | Yes |
| Per-component CLI checkpoint override | config-driven only | Not exposed as direct flags; use a dedicated experiment YAML through `--config`. | Add direct flags only if the PoC must support ad-hoc runtime overrides. | N/A | Future |

### Visualization

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| Mask overlay | implemented | `--overlay-style mask` draws transparent filled patch masks by default. | None for sampled/chop-local coordinates. | Yes | Yes |
| Box overlay | implemented | `--overlay-style box` draws patch outlines. | None for sampled/chop-local coordinates. | Yes | Yes |
| Mask + box overlay | implemented | `--overlay-style both` draws mask fill and outline. | None for sampled/chop-local coordinates. | Yes | Yes |
| Patch index toggle | implemented | `--show-patch-index` enables patch ID labels; labels are disabled by default. | None. | Yes | Yes |
| Scale label toggle | implemented | `--show-scale-label` renders scale labels when scale metadata exists. | Falls back cleanly when scale metadata is absent. | Yes | Yes |
| Multi-scale overlay | implemented | `--multi-scale-overlay` with `--scale-color-mode gradient/categorical` records color maps and scale IDs in metadata. | Original-space overlays use the same scale color map for affine mappings. | Yes | Yes |
| Scale-wise panel video | implemented | `--save-scale-panel-video` exports 2x2 scale panel video and per-frame panel images. | More layouts are future work. | Yes | Yes |
| External info panel | partially implemented | `--info-panel-mode external` puts basic frame/token text below frames. | Missing position/size controls and many requested fields. | Yes | Yes |
| Chop visualization | partially implemented | `--scaling-mode chop` writes metadata-only chop records, optional chop-local frames, and non-overlap `overlay_union` merged full-frame overlays. | Non-zero overlap/custom stride and original-space merged overlays remain unsupported. | Yes | Yes |

### Video Export

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| sampled-only overlay video | implemented | Exports mode-specific and canonical `visualizations/autogaze/videos/autogaze_overlay.mp4` videos. | None for sampled-only scope. | Yes | Yes |
| sampled-only side-by-side video | implemented | `--comparison-layout processed_overlay` compares processed frame vs processed-frame overlay. | Other comparison layouts raise clearly. | Yes | Yes |
| scale panel video | implemented | Exports `autogaze_scale_panels.mp4` and per-frame panel images. | Layouts beyond `2x2` are future work. | Yes | Yes |
| full-length video | implemented | `full_length` preserves original frame count and FPS when available, inserts processed overlays at sampled/chunk/all/interval frame indices, and marks unprocessed frames. | Unprocessed frames use original-if-available else black placeholder policy. | Yes | Yes |
| hold-last video | stub-only | `hold_last` raises `NotImplementedError`. | Not requested in latest matrix except older path; keep stub-only. | Stub | Stub |
| chop overlay video | partially implemented | Non-overlap `overlay_union` saves `autogaze_chop_overlay.mp4` and per-window merged overlay frames. | Overlap/custom stride and original decoded-frame mapping remain unsupported. | Yes | Yes |
| original-space overlay | implemented for affine modes | `original_overlay` and `original_processed_overlay` support exact affine mapping for none/resize/fit_short_side/fit_long_side/quickstart. | Chop uses explicit overlay-union path instead. | Yes | Yes |

### Reporting and Metrics

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| `poc_summary.json` | implemented | Always saved under `logs/`. | None. | Yes | Yes |
| Frame selection metadata | implemented | Saved under `autogaze/frame_selection_metadata.json`. | None. | Yes | Yes |
| Scaling metadata | implemented | Saved under `scaling/scaling_metadata.json`. | None. | Yes | Yes |
| Per-window token counts | implemented | Saved under `autogaze/windows/window_*/token_counts.json`. | None for current window scope. | Yes | Yes |
| Selected patch masks | implemented | Saved as per-window `selected_patch_mask.json`. | Needs richer mask metadata by scale/chop. | Yes | Yes |
| Runtime metadata | implemented | Saved under `autogaze/runtime_metadata.json`, including requested/effective AutoGaze controls and strict validation results. | None for current PoC scope. | Yes | Yes |
| Metrics JSON/CSV | implemented | Saved under `logs/metrics.json` and `logs/metrics.csv`. | Metrics are PoC-level and use `N/A` for unavailable or skipped stages. | Yes | Yes |
| Visualization skip report | implemented | When AutoGaze output is unavailable, saves `visualizations/autogaze/metadata/visualization_skip_metadata.json` and mirrors the reason in metrics. | None for guarded/no-checkpoint smoke scope. | Yes | Yes |
| Memory metrics | partially implemented | CUDA peak memory is sampled; CPU/MPS metrics are explicitly marked unavailable. | No process-level CPU memory sampling. | Yes | Yes |
| Stage latency breakdown | implemented | Import/check, preprocessing, scaling/chop, AutoGaze, visualization, vision encoder, and NVILA generation stages are recorded when executed. | MLLM prefill/decode separation is not available in the current guarded generation call. | Yes | Yes |
| Token reduction summary | implemented | Top-level `autogaze/token_counts_summary.json` aggregates per-window before/after counts, reduction ratio, and selected patches by frame/window/scale. | Per-chop token summary is still future work. | Yes | Yes |
| Generated answer | partially implemented | Saved when NVILA generation succeeds. Skipped generation is reported. | Real NVILA generation remains unverified. | N/A | Partial |

### Benchmark Config Integration

| Feature | Status | Current behavior | Missing or blocked work | autogaze_only | full_pipeline |
|---|---|---|---|---:|---:|
| Existing tiny/canonical configs | partially implemented | Current repo has tiny and canonical benchmark configs for A1/A2. | They do not cover the full feature matrix combinations. | Partial | Partial |
| `poc_feature_matrix_smoke.yaml` | implemented | Safe dummy/stub smoke config. | None for config-template scope. | Yes | Yes |
| `poc_autogaze_only_visualization.yaml` | implemented | Safe AutoGaze-only visualization smoke config. | None for config-template scope. | Yes | N/A |
| `poc_full_pipeline_visualization.yaml` | implemented | Guarded full-pipeline visualization smoke config with query text. | Real generation remains checkpoint/memory-gated. | N/A | Yes |
| `poc_chop_mode_smoke.yaml` | implemented | Safe non-overlap chop metadata/visualization smoke config. | Overlap/merge remains stub/future. | Yes | Yes |
| `poc_multiscale_visualization.yaml` | implemented | Safe multi-scale overlay smoke config. | None for config-template scope. | Yes | Yes |
| `poc_scale_panel_video.yaml` | implemented | Safe scale-panel video smoke config. | None for config-template scope. | Yes | Yes |
| `poc_high_resolution_chop_smoke.yaml` | implemented | Safe high-resolution chop smoke config with bounded chops. | Config only; not run by default. | Yes | Yes |
| `poc_high_resolution_chop_medium.yaml` | implemented | Medium preparation config for high-resolution chop validation. | Config only; not run by default. | Yes | Yes |
| `poc_full_length_video_export_smoke.yaml` | implemented | Safe full-length video export smoke config. | Config only; not run by default. | Yes | Yes |

## Combination Matrix

### Core Safe Combinations

| Combination | Status | Notes |
|---|---|---|
| dummy + sample + resize + autogaze_only + no checkpoint load | implemented | Produces guarded summary with AutoGaze skipped. |
| dummy + sample + resize + autogaze_only + checkpoint load allowed | partially implemented | Runs if configured AutoGaze module/checkpoint is available. |
| dummy + chunk + resize + autogaze_only + sampled-only videos | implemented for mocked tests, partially implemented for real | Tests validate multiple windows and video export with fake modules. Real depends on AutoGaze assets. |
| dummy + interval + resize + autogaze_only | implemented | Frame metadata preserves interval indices. |
| dummy + all + resize + autogaze_only | implemented | `all` is chunked full-video processing. |
| dummy + sample + resize + full_pipeline + query text + no checkpoint load | implemented as guarded path | Query text is logged; downstream generation is skipped clearly. |
| dummy + sample + resize + full_pipeline + checkpoint load allowed | partially implemented | Attempts AutoGaze, modified SigLIP, NVILA. Real success depends on assets and memory. |
| dummy + chunk + resize + full_pipeline + AutoGaze visualization | partially implemented | AutoGaze visualization is available in full mode; downstream may skip. |

### Scaling Combinations

| Combination | Status | Notes |
|---|---|---|
| resize 224 patch16 | implemented | Closest to QUICK_START default. |
| quickstart 224 patch16 | implemented | Resolves to default QUICK_START policy. |
| quickstart 392 patch14 with target scales | partially implemented | Exact PoC scaling policy exists; real modified SigLIP alignment needs validation. |
| fit_short_side / fit_long_side | partially implemented | Tensor scaling works; real model compatibility is not guaranteed. |
| none | partially implemented | Tensor path works; model fixed-resolution requirements are not enforced. |
| chop non-overlap metadata | partially implemented | Uses spatio-temporal chunks in scaling metadata. |
| chop overlap/stride/merge | partially implemented | Non-overlap `overlay_union` is implemented; non-zero overlap and custom stride remain future work. |

### Visualization Combinations

| Combination | Status | Notes |
|---|---|---|
| mask overlay image frames | implemented | Transparent fill rectangles. |
| box overlay image frames | implemented | Boxes are drawn unless hidden. |
| patch index text | implemented | Disabled by default; enabled with `--show-patch-index`. |
| multi-scale gradient/categorical overlay | implemented | Uses scale metadata where present and records scale color mapping. |
| external info panel | partially implemented | Basic panel only. |
| sampled-only overlay MP4 | implemented | Requires PyAV. |
| sampled-only side-by-side MP4 | implemented | Supports processed-vs-overlay layout. |
| sampled-only scale panel MP4 | implemented | Supports 2x2 layout. |
| full-length MP4 | implemented | Preserves original frame count and records unprocessed-frame policy. |
| chop overlay MP4 | partially implemented | Non-overlap `overlay_union` only. |

## Blocked and Stub Details

| Item | Status | Reason | Missing module/config/checkpoint | Expected fix | Affects autogaze_only | Affects full_pipeline |
|---|---|---|---|---|---:|---:|
| Real AutoGaze execution | blocked when assets unavailable | Requires importable original AutoGaze and checkpoint/config path. | AutoGaze module, checkpoint, processor path. | Run `--mode check`; configure real paths; allow checkpoint load. | Yes | Yes |
| Real modified SigLIP execution | blocked when assets unavailable | Requires original modified SigLIP module and checkpoint/config. | `autogaze.vision_encoders.siglip.SiglipVisionModel` or configured equivalent. | Validate construction before inference. | No | Yes |
| Real NVILA generation | blocked when assets or memory unavailable | NVILA-HD-Video is large and uses trusted remote code. | NVILA checkpoint/cache, processor/tokenizer, enough device memory. | Use guarded construction first; run with explicit checkpoint/model load only. | No | Yes |
| Per-component model override CLI | future work | Script reads config but does not expose requested per-component flags by design. | N/A. | Add override merge layer only if config-driven wiring is insufficient. | No | Yes |
| Hold-last export | stub-only | Requires state carry-forward policy across unprocessed frames. | Original frame access and temporal hold policy. | Keep explicit `NotImplementedError`. | Yes | Yes |
| Custom comparison layouts | partially implemented | `processed_overlay`, `original_overlay`, and `original_processed_overlay` are supported. | `chop_overlay` as side-by-side layout remains unsupported; use `overlay_union`. | Add layouts only with exact mapping. | Yes | Yes |
| Chop overlay video | partially implemented | `--save-chop-overlay-video` works with non-overlap `overlay_union`. | Non-zero overlap/custom stride and original decoded-frame overlay. | Add overlap semantics only after coordinate policy is defined. | Yes | Yes |
| Original-space overlay | implemented for affine modes | Exact affine mapping is recorded for resize/fit modes. | Arbitrary crop/pad/chop original-space mapping. | Fail clearly when mapping is not exact. | Yes | Yes |
| Chop overlay union | partially implemented | Non-overlap union maps chop-local masks to full processed-frame patch grid and records conflict policy. | Overlap/custom stride merge semantics. | Add deterministic overlap policy with tests before enabling. | Yes | Yes |

## Current Output Structure Audit

Current implemented or partially implemented output paths:

```text
outputs/<run_name>/
  autogaze/frame_selection_metadata.json
  autogaze/windows/window_000/selected_patch_indices.json
  autogaze/windows/window_000/selected_scales.json
  autogaze/windows/window_000/selected_patch_mask.json
  autogaze/windows/window_000/token_counts.json
  autogaze/runtime_metadata.json
  autogaze/token_counts_summary.json
  scaling/scaling_metadata.json
  visualizations/autogaze_only/windows/window_000/frames/
  visualizations/autogaze_only/windows/window_000/videos/
  visualizations/autogaze_only/videos/autogaze_overlay_sampled_only.mp4
  visualizations/autogaze_only/videos/autogaze_side_by_side_sampled_only.mp4
  visualizations/autogaze_only/videos/autogaze_scale_panels_sampled_only.mp4
  visualizations/full_pipeline/windows/window_000/frames/
  visualizations/full_pipeline/videos/autogaze_overlay_sampled_only.mp4
  visualizations/autogaze/windows/window_000/frames/
  visualizations/autogaze/windows/window_000/scale_panels/
  visualizations/autogaze/videos/autogaze_overlay.mp4
  visualizations/autogaze/videos/autogaze_side_by_side.mp4
  visualizations/autogaze/videos/autogaze_scale_panels.mp4
  visualizations/autogaze/videos/autogaze_overlay_full_length.mp4
  visualizations/autogaze/videos/autogaze_side_by_side_full_length.mp4
  visualizations/autogaze/videos/autogaze_original_overlay.mp4
  visualizations/autogaze/videos/autogaze_original_processed_overlay.mp4
  visualizations/autogaze/videos/autogaze_chop_overlay.mp4
  visualizations/autogaze/metadata/chop_overlay_metadata.json
  visualizations/autogaze/metadata/visualization_skip_metadata.json
  chops/chop_metadata.json
  chops/windows/window_000/frame_000/chop_000/
  visualizations/autogaze/chops/window_000/frame_000/chop_000/
  predictions/answer.json
  logs/poc_summary.json
  logs/metrics.json
  logs/metrics.csv
```

Requested but not yet implemented exactly:

```text
visualizations/autogaze/metadata/visualization_metadata.json
logs/reproducibility_manifest.json
```

## Minimal Next Implementation Batch

Recommended next batch should stay small and avoid large benchmark jobs:

1. Add per-component full-pipeline override flags only if config-driven wiring proves too slow for iteration:
   - `--vision-encoder-*`
   - `--mllm-*`
   - `--processor-path`
   - `--tokenizer-path`

2. Keep hold-last export, non-zero chop overlap/custom stride, and original decoded-frame chop overlays as explicit future work until coordinate mapping and merge semantics are defined.
