# PoC NVILA-HD-Video Feature Matrix Request

## 0. Purpose

This document defines the requested mid-project audit and implementation scope for `scripts/poc_nvila_hd_video.py`.

`poc_nvila_hd_video.py` should become the all-in-one PoC tester for the canonical AutoGaze + SigLIP ViT + NVILA-HD-Video path and related visualization/benchmark features.

The script should support both:

```text
autogaze_only
full_pipeline
```

and should allow testing combinations of:

```text
video input
-> frame selection
-> scaling / chop preprocessing
-> AutoGaze runtime control
-> AutoGaze inference
-> AutoGaze visualization
-> optional ViT / vision encoder
-> optional MLLM
-> report / benchmark output
```

This request must be implemented based on:

1. the AutoGaze paper
2. the original AutoGaze / NVILA-HD-Video code and guides
3. `docs/nvila-hd-video-readme.md`
4. `docs/NVILA_HD_VIDEO_REFERENCE.md`
5. `docs/QUICK_START_reference.md`
6. the current project implementation
7. the intended role of `poc_nvila_hd_video.py` as a universal test/debug script

Do not run large benchmark jobs.  
Do not claim paper reproduction.  
Do not modify original AutoGaze source files.  
Do not modify original `INTEGRATION.md`, `QUICK_START.md`, or `docs/nvila-hd-video-readme.md`.  
Do not silently treat unsupported paths as supported.

---

## 1. Create the Feature-Combination Matrix First

Before implementation, create:

```text
docs/POC_NVILA_HD_VIDEO_FEATURE_MATRIX.md
```

This document must define all intended combinations for `poc_nvila_hd_video.py`.

Each feature or combination must have one of these statuses:

```text
implemented
partially implemented
stub-only
blocked
future work
```

For every `stub-only` or `blocked` item, document:

- reason
- missing module/config/checkpoint
- expected fix
- whether it affects `autogaze_only` mode
- whether it affects `full_pipeline` mode

The matrix must cover at least:

- inference mode
- frame selection mode
- scaling / chop mode
- AutoGaze runtime controls
- vision encoder selection
- MLLM selection
- checkpoint/config override
- AutoGaze visualization
- multi-scale visualization
- video export
- metrics/reporting
- benchmark config integration

---

## 2. Required Modes

`poc_nvila_hd_video.py` must support these major modes:

```text
--mode autogaze_only
--mode full_pipeline
```

### 2.1 `autogaze_only`

Definition:

```text
video
-> frame selection
-> scaling / chop preprocessing
-> AutoGaze
-> selected patch/token metadata
-> visualization
-> report
```

This mode must not require NVILA generation.

It must still support:

- frame selection mode
- scaling / chop mode
- `gaze_ratio`
- `task_loss_requirement`
- mask overlay visualization
- patch-index visualization
- multi-scale visualization
- video export
- token / latency / memory report

### 2.2 `full_pipeline`

Definition:

```text
video + query text
-> frame selection
-> scaling / chop preprocessing
-> AutoGaze
-> selected patch/token metadata
-> AutoGaze visualization
-> ViT / vision encoder
-> MLLM
-> generated answer, if available
-> report
```

This mode must support everything that `autogaze_only` supports, plus:

- query text
- vision encoder selection
- vision encoder checkpoint/config path
- MLLM selection
- MLLM checkpoint/config path
- tokenizer/processor path
- generation config
- generated answer output, if available

Important:

- AutoGaze visualization must still be saved in `full_pipeline` mode when requested.
- Do not skip AutoGaze visualization just because the downstream model is running.
- If downstream ViT/MLLM fails or is stub-only, AutoGaze results must still be saved and the downstream skip reason must be reported clearly.

---

## 3. Frame Selection Modes

Define and implement or validate these frame selection modes:

```text
sample
chunk
interval
all
```

Do not implement sliding-window stride mode.

The previously discussed “skip every N frames” behavior should be represented by `interval`, not stride.

### 3.1 `sample`

Uniformly sample `num_frames` frames from the entire video.

Inference windows:

```text
1 window
```

### 3.2 `chunk`

Split the video into non-overlapping windows of `num_frames`.

Inference windows:

```text
multiple windows
```

Example:

```text
window 0: [0..15]
window 1: [16..31]
window 2: [32..47]
```

### 3.3 `interval`

Select `num_frames` frames using `frame_interval`.

Example:

```text
num_frames = 16
frame_interval = 2
selected frames = [0, 2, 4, ..., 30]
```

Inference windows:

```text
1 window by default
```

### 3.4 `all`

Process the whole video.

If the entire video cannot fit in one forward pass, internally treat `all` as chunked processing.

Clearly document:

```text
all = chunked full-video processing
```

### Required CLI

```bash
--frame-selection-mode sample/chunk/interval/all
--num-frames <int>
--frame-interval <int>
--max-windows <int>
--drop-last
--pad-last
```

Do not add:

```bash
--frame-stride
```

Do not implement:

```text
overlapping windows
sliding-window stride
```

---

## 4. Scaling and Chop Modes

Define and implement or validate these scaling modes:

```text
none
resize
fit_short_side
fit_long_side
quickstart
chop
```

Note:

The user previously described this as “scaling mode for larger frames,” but based on `QUICK_START.md` this should include `chop` mode.

### 4.1 `none`

No resizing.

If the model requires fixed resolution, fail clearly.

### 4.2 `resize`

Resize to square `resolution x resolution`.

May distort aspect ratio.

### 4.3 `fit_short_side`

Preserve aspect ratio.

Resize so the shorter side equals `resolution`.

### 4.4 `fit_long_side`

Preserve aspect ratio.

Resize so the longer side equals `resolution`.

### 4.5 `quickstart`

Follow the scaling behavior described in:

```text
docs/QUICK_START_reference.md
docs/NVILA_HD_VIDEO_REFERENCE.md
original QUICK_START.md
```

If exact behavior is not implemented, raise `NotImplementedError`.

### 4.6 `chop`

High-resolution chop / tiled processing mode.

This must be based on `QUICK_START.md` if possible.

Conceptual behavior:

```text
large frame
-> split into chops/tiles
-> run AutoGaze or pipeline on each chop
-> save per-chop metadata
-> optionally merge/visualize chop results
```

### Required CLI

```bash
--scaling-mode none/resize/fit_short_side/fit_long_side/quickstart/chop
--resolution <int>
--chop-size <int>
--chop-overlap <int>
--chop-stride <int>
--max-chops <int>
--chop-merge-mode none/metadata_only/overlay_union
--save-chop-frames
--save-chop-overlay-video
```

For chop mode, record:

- chop ID
- source frame index
- window ID
- `x0`, `y0`, `x1`, `y1`
- chop width/height
- overlap/stride
- selected patch indices in chop coordinates
- selected scales
- selected patch count
- token count

Save:

```text
outputs/<run_name>/chops/chop_metadata.json
```

Do not draw chop-relative patch coordinates as full-frame coordinates unless mapping is implemented.

---

## 5. AutoGaze Runtime Controls

`poc_nvila_hd_video.py` must expose AutoGaze runtime controls in both modes.

### Required CLI

```bash
--gaze-ratio <float>
--task-loss-requirement <float>
--strict-autogaze-params
```

Config must also support these values.

Use existing config convention if already present. Otherwise use:

```yaml
autogaze:
  gaze_ratio: null
  task_loss_requirement: null
```

or:

```yaml
autogaze_runtime:
  gaze_ratio: null
  task_loss_requirement: null
```

### Required Behavior

1. CLI values override config.
2. Config values override defaults.
3. If neither is provided, use original AutoGaze defaults.
4. Log requested and effective values.
5. If unsupported by the current AutoGaze callable, do not silently ignore.
6. If `--strict-autogaze-params` is enabled, unsupported params must raise an error.
7. If strict mode is disabled, unsupported params may continue only with a clear warning and metadata record.

Try to detect equivalent original parameter names, including:

```text
gaze_ratio
gazing_ratio
task_loss_requirement
task_requirement
loss_requirement
```

Do not invent behavior.

Do not claim `task_loss_requirement` affects selection unless verified.

---

## 6. Vision Encoder / MLLM Switching in Full Pipeline

In `--mode full_pipeline`, users must be able to select or configure:

```text
vision encoder
MLLM
checkpoint/config paths
processor/tokenizer paths
```

### Required CLI or Config Support

```bash
--vision-encoder <name>
--vision-encoder-module <module_path>
--vision-encoder-class <class_name>
--vision-encoder-ckpt <path>
--vision-encoder-config <path>

--mllm <name>
--mllm-module <module_path>
--mllm-class <class_name>
--mllm-ckpt <path>
--mllm-config <path>
--processor-path <path>
--tokenizer-path <path>
```

If these are already represented in configs, ensure `poc_nvila_hd_video.py` can read and override them cleanly.

Minimum supported full pipeline target:

```text
modified SigLIP + NVILA-HD-Video
```

Optional or future targets:

```text
vanilla SigLIP + NVILA
Qwen / HF MLLM official processor
generic ViT + generic MLLM
```

For each target, record whether it is:

```text
implemented
stub-only
blocked
future work
```

Do not pretend model switching works if only config parsing exists.

---

## 7. Visualization Combinations

Visualization must be available in both:

```text
autogaze_only
full_pipeline
```

Supported visualization types:

```text
mask overlay
box overlay
mask + box overlay
patch index visualization
multi-scale mask overlay
scale-wise panel visualization
side-by-side video
chop visualization
```

### 7.1 Overlay Styles

Required CLI:

```bash
--overlay-style mask/box/both
--overlay-alpha <float>
--show-patch-index
--show-scale-label
```

Default:

```text
overlay_style = mask
overlay_alpha = 0.35
show_patch_index = false
show_scale_label = false
```

Rules:

- Default visualization should be transparent mask overlay.
- Red boxes are optional.
- Patch index text is optional.
- If patch grid metadata is missing, fail clearly.
- Do not guess patch layout silently.

### 7.2 Multi-Scale Overlay

Required CLI:

```bash
--multi-scale-overlay
--scale-color-mode gradient/categorical
```

Default:

```text
multi_scale_overlay = true
scale_color_mode = gradient
```

Requirements:

- use selected scale information if available
- use different colors per scale
- prefer four gradient-like colors when there are four scales
- save scale color mapping in metadata
- if scale metadata is unavailable, fall back to single-color mask and record the fallback

Suggested gradient palette:

```text
scale_0: light yellow
scale_1: orange
scale_2: pink
scale_3: purple
```

### 7.3 Scale-Wise Panel Visualization

Required CLI:

```bash
--save-scale-panel-video
--scale-panel-layout 2x2
```

Create a panel per frame:

```text
panel 0: scale 0 mask
panel 1: scale 1 mask
panel 2: scale 2 mask
panel 3: scale 3 mask
```

Save:

```text
outputs/<run_name>/visualizations/autogaze/videos/autogaze_scale_panels.mp4
```

### 7.4 External Info Panel

Frame metadata should be rendered outside the image by default, not over the image content.

Required CLI:

```bash
--metadata-placement outside/inside/none
--info-panel-position bottom/right
--info-panel-size <int>
```

Default:

```text
metadata_placement = outside
info_panel_position = bottom
```

Info panel must include:

- mode
- visual frame number
- source frame index
- window ID
- anchor frame index
- scaling mode
- chop ID / chop coordinates, if chop mode
- selected patch count
- selected patch count by scale
- original token count
- selected token count
- token reduction ratio
- requested/effective `gaze_ratio`
- requested/effective `task_loss_requirement`
- unsupported AutoGaze runtime params
- processed resolution
- query text summary in full mode
- generation status in full mode

Do not overlay this text on the frame by default.

---

## 8. Video Output Combinations

Video outputs must be available in both modes.

Required CLI:

```bash
--save-overlay-video
--save-side-by-side-video
--save-scale-panel-video
--save-chop-overlay-video
--video-export-mode sampled_only/full_length
--video-fps <float>
--comparison-layout processed_overlay/original_overlay/original_processed_overlay/chop_overlay
```

Initial required support:

```text
sampled_only
processed_overlay
```

Rules:

- `sampled_only` video uses only frames/chops actually processed.
- `full_length` may be stubbed with clear `NotImplementedError`.
- external info panel should appear in exported video frames when enabled.
- side-by-side output should compare processed frame and AutoGaze overlay.
- original-space overlay must not be claimed unless coordinate mapping exists.

Output paths:

```text
outputs/<run_name>/visualizations/autogaze/videos/autogaze_overlay.mp4
outputs/<run_name>/visualizations/autogaze/videos/autogaze_side_by_side.mp4
outputs/<run_name>/visualizations/autogaze/videos/autogaze_scale_panels.mp4
outputs/<run_name>/visualizations/autogaze/videos/autogaze_chop_overlay.mp4
```

---

## 9. Report / Benchmark Metrics for All Combinations

Every inference run should generate a report.

This applies to both:

```text
autogaze_only
full_pipeline
```

Required metrics:

- mode
- frame selection mode
- scaling mode
- chop mode settings, if used
- number of frames
- number of windows
- original frame count
- original FPS
- original resolution
- processed resolution
- requested/effective `gaze_ratio`
- requested/effective `task_loss_requirement`
- original visual token count
- selected visual token count
- token reduction ratio
- selected patches per frame
- selected patches per scale
- selected patches per window
- selected patches per chop, if chop mode
- AutoGaze latency
- preprocessing latency
- scaling/chop latency
- visualization latency, if measured
- vision encoder latency, if full mode
- MLLM prefill latency, if full mode
- MLLM decode latency, if full mode
- end-to-end latency
- peak VRAM, if CUDA
- memory metric unavailable flag, if MPS/CPU
- generated answer, if full mode and generation succeeded
- skipped stages
- failure reason, if any

Save:

```text
outputs/<run_name>/logs/poc_summary.json
outputs/<run_name>/logs/metrics.json
outputs/<run_name>/logs/metrics.csv
```

Do not treat skipped generation as successful full pipeline inference.

Do not claim encoder-side acceleration unless encoder computation is actually reduced.

---

## 10. Output Structure

Use a consistent structure for all combinations:

```text
outputs/<run_name>/
  autogaze/
    frame_selection_metadata.json
    runtime_metadata.json
    token_counts_summary.json
    windows/
      window_000/
        selected_patch_indices.json
        selected_scales.json
        token_counts.json

  scaling/
    scaling_metadata.json
    windows/
      window_000/
        frames_before/
        frames_after/

  chops/
    chop_metadata.json
    windows/
      window_000/
        frame_000/
          chop_000/
            chop.png
            selected_patch_indices.json
            selected_scales.json
            token_counts.json

  visualizations/
    autogaze/
      windows/
        window_000/
          frames/
          scale_panels/
      chops/
        window_000/
          frame_000/
            chop_000/
              frames/
              scale_panels/
      videos/
        autogaze_overlay.mp4
        autogaze_side_by_side.mp4
        autogaze_scale_panels.mp4
        autogaze_chop_overlay.mp4
      metadata/
        visualization_metadata.json

  predictions/
    answer.json

  logs/
    poc_summary.json
    metrics.json
    metrics.csv
    reproducibility_manifest.json
```

For `autogaze_only`, `predictions/answer.json` may be absent or explicitly marked as not applicable.

---

## 11. Benchmark Integration

After defining and implementing the feature matrix, update benchmark support to reflect the same combinations.

Benchmark configs should be able to specify:

- mode: `autogaze_only` or `full_pipeline`
- frame selection mode
- number of frames
- frame interval
- scaling mode
- resolution
- chop settings
- `gaze_ratio`
- `task_loss_requirement`
- overlay settings
- whether to save videos
- vision encoder config, full mode only
- MLLM config, full mode only
- query text, full mode only

Create or update:

```text
configs/benchmark/poc_feature_matrix_smoke.yaml
configs/benchmark/poc_autogaze_only_visualization.yaml
configs/benchmark/poc_full_pipeline_visualization.yaml
configs/benchmark/poc_chop_mode_smoke.yaml
```

Benchmark runner should report which feature combination was used.

Do not run heavy benchmark by default.

---

## 12. Tests

Add tests for the feature matrix and key combinations.

Required tests:

### 12.1 Feature Matrix

- `docs/POC_NVILA_HD_VIDEO_FEATURE_MATRIX.md` exists
- every listed feature has a status
- stub/blocked features have reasons

### 12.2 Frame Selection

- sample
- chunk
- interval
- all as chunk alias

### 12.3 Scaling

- none
- resize
- fit_short_side
- fit_long_side
- quickstart stub or implementation
- chop metadata generation

### 12.4 AutoGaze Runtime Controls

- `gaze_ratio` CLI/config
- `task_loss_requirement` CLI/config
- strict params behavior
- metadata recording

### 12.5 Visualization

- mask overlay
- box overlay
- patch index toggle
- multi-scale overlay
- external info panel
- scale panel video
- side-by-side video
- video output in `autogaze_only`
- video output in `full_pipeline`

### 12.6 Full Pipeline

- query text not silently ignored
- AutoGaze visualization saved even if downstream is skipped
- skipped stages recorded

### 12.7 Reporting

- token count metrics saved
- latency metrics saved
- memory metrics saved or marked unavailable
- benchmark config records feature combination

Do not require:

- real AutoGaze checkpoint
- real NVILA checkpoint
- GPU
- external video files

---

## 13. Documentation Update

Update:

```text
docs/INFERENCE_GUIDE.md
docs/benchmark.md
docs/benchmark_analysis.md
```

Add or update sections:

1. `poc_nvila_hd_video.py` as all-in-one tester
2. supported feature matrix
3. frame selection modes
4. scaling and chop modes
5. AutoGaze runtime controls
6. full pipeline model/ckpt override options
7. visualization modes
8. video export modes
9. metrics and report outputs
10. benchmark feature-combination configs

Do not document unsupported features as runnable.

---

## 14. Example Commands to Document

### AutoGaze-only sample + resize + multi-scale mask

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video-path /path/to/video.mp4 \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 448 \
  --gaze-ratio 0.15 \
  --config configs/experiment/A2_real.yaml \
  --device cuda \
  --output-dir outputs/poc_autogaze_sample_resize \
  --overlay-style mask \
  --multi-scale-overlay \
  --metadata-placement outside \
  --info-panel-position bottom \
  --save-overlay-video \
  --save-side-by-side-video \
  --video-export-mode sampled_only
```

### Full pipeline sample + resize + query text

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 448 \
  --gaze-ratio 0.25 \
  --task-loss-requirement 0.8 \
  --config configs/experiment/A2_real.yaml \
  --device cuda \
  --output-dir outputs/poc_full_sample_resize \
  --overlay-style mask \
  --multi-scale-overlay \
  --metadata-placement outside \
  --info-panel-position bottom \
  --save-overlay-video \
  --save-side-by-side-video \
  --save-scale-panel-video \
  --video-export-mode sampled_only \
  --max-new-tokens 16
```

### Full pipeline chop mode

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode chop \
  --chop-size 448 \
  --chop-overlap 64 \
  --max-chops 16 \
  --chop-merge-mode metadata_only \
  --gaze-ratio 0.25 \
  --task-loss-requirement 0.8 \
  --config configs/experiment/A2_real.yaml \
  --device cuda \
  --output-dir outputs/poc_full_chop \
  --overlay-style mask \
  --multi-scale-overlay \
  --metadata-placement outside \
  --info-panel-position bottom \
  --save-chop-frames \
  --save-overlay-video \
  --save-side-by-side-video \
  --save-scale-panel-video \
  --video-export-mode sampled_only \
  --max-new-tokens 16
```

---

## 15. Implementation Priority

### Priority 1

- define feature matrix
- align implemented / partial / stub / blocked status
- frame selection mode for both modes
- scaling mode for both modes
- `gaze_ratio` / `task_loss_requirement` for both modes
- AutoGaze metadata saving for both modes
- mask overlay and external info panel
- sampled-only overlay video for both modes
- query text handling in full mode
- metrics/report output

### Priority 2

- multi-scale overlay
- patch index visualization
- scale panel video
- chop metadata and per-chop visualization
- side-by-side video
- benchmark configs reflecting combinations

### Priority 3

- quickstart exact scaling
- full-length video
- merged full-frame chop overlay
- original-space overlay
- real high-resolution benchmark

---

## 16. Required Report After Implementation

After implementation, report:

- changed files
- feature matrix path
- implemented combinations
- partially implemented combinations
- stub-only combinations
- blocked combinations
- updated CLI arguments
- updated benchmark configs
- example commands
- output directory examples
- test command
- remaining blockers