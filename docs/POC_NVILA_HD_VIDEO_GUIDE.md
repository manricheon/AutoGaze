# PoC NVILA-HD-Video Script Guide

This document is the focused user guide for:

```text
scripts/poc_nvila_hd_video.py
```

The script is an isolated PoC tester for the canonical path:

```text
video
-> frame selection
-> scaling / chop preprocessing
-> AutoGaze
-> modified SigLIP ViT, optional
-> NVILA-HD-Video / MLLM, optional
-> metadata, visualization, and smoke metrics
```

It is not a public benchmark runner and does not claim paper reproduction.
Large checkpoints are never loaded unless `--allow-checkpoint-load` is passed.

## Reference Priority

Use these references when interpreting script behavior:

1. `docs/NVILA_HD_VIDEO_REFERENCE.md`
2. `docs/QUICK_START_reference.md`
3. `docs/POC_NVILA_HD_VIDEO_FEATURE_MATRIX.md`
4. `docs/INFERENCE_GUIDE.md`
5. Original `QUICK_START.md`, if available
6. Original `INTEGRATION.md`, if available

The original AutoGaze source files, original `QUICK_START.md`, original `INTEGRATION.md`, and `docs/nvila-hd-video-readme.md` must not be modified by this PoC layer.

## Safety Defaults

Default behavior is intentionally conservative:

```text
mode = check
device = cpu
checkpoint loading = disabled
frame_selection_mode = sample
num_frames = 16
scaling_mode = resize
resolution = 224
patch_size = 16
gaze_ratio = 0.75
task_loss_requirement = 0.7
video_export_mode = sampled_only
comparison_layout = processed_overlay
metadata_placement = outside
```

The reference default preset is:

```text
configs/benchmark/poc_default.yaml
```

Smoke and visualization configs may explicitly use fewer frames to keep tests and local checks fast. Those values are local overrides, not the script default.

Real AutoGaze, SigLIP, or NVILA execution requires:

```bash
--allow-checkpoint-load
```

If a module, checkpoint, API, or coordinate mapping is missing, the script should report a blocker or raise a clear error instead of silently falling back.

## Main Modes

| Mode | Purpose | Real checkpoint required? |
|---|---|---:|
| `check` | Validate imports, classes/factories, config paths, checkpoint paths | No |
| `autogaze_only` | Run preprocessing and AutoGaze path, then save AutoGaze metadata/visualization | Only for real AutoGaze execution |
| `full_pipeline` | Run AutoGaze, vision encoder, and NVILA generation when available | Yes for real downstream inference |

## Current CLI Surface

The script options are grouped as follows.
Defaults shown here are the script defaults, not necessarily the values used by every smoke config.

| Group | Options | Default / Notes |
|---|---|---|
| Mode/input | `--mode check|autogaze_only|full_pipeline`, `--video dummy`, `--video-path`, `--query-text` | `check`, dummy input |
| Frame selection | `--frame-selection-mode sample|chunk|interval|all`, `--num-frames`, `--frame-interval`, `--max-windows`, `--drop-last`, `--pad-last` | `num_frames=16`, `sample`; no overlapping stride mode |
| Scaling/chop | `--scaling-mode none|resize|fit_short_side|fit_long_side|quickstart|chop`, `--resolution`, `--patch-size`, `--target-scales`, `--target-patch-size`, `--spatial-tile-size`, `--chop-size`, `--chop-overlap`, `--chop-stride`, `--max-chops`, `--chop-merge-mode none|metadata_only|overlay_union` | `resize`, `resolution=224`, `patch_size=16`; non-zero chop overlap/stride unsupported |
| AutoGaze runtime | `--gaze-ratio`, `--task-loss-requirement`, `--strict-autogaze-params` | Config value if provided; A2 defaults to `0.75` and `0.7`; strict mode validates callable kwargs before AutoGaze execution |
| Runtime safety | `--device cpu|cuda|mps`, `--dtype float32|float16|bfloat16`, `--allow-checkpoint-load`, `--no-checkpoint-load`, `--checkpoint-metadata-only`, `--max-new-tokens` | CPU, float32, checkpoint loading disabled, `max_new_tokens=1` |
| Visualization | `--save-overlay-video`, `--save-side-by-side-video`, `--save-scale-panel-video`, `--save-chop-frames`, `--save-chop-overlay-video`, `--video-fps`, `--video-export-mode sampled_only|full_length|hold_last`, `--overlay-alpha`, `--overlay-line-width`, `--overlay-style mask|box|both` | `sampled_only`; `hold_last` is stub-only |
| Labels/panels | `--multi-scale-overlay/--no-multi-scale-overlay`, `--scale-color-mode gradient|categorical`, `--scale-panel-layout 2x2`, `--show-patch-index`, `--show-scale-label`, `--hide-patch-boxes`, `--hide-patch-indices` | Multi-scale gradient enabled; patch IDs disabled |
| Layout/metadata | `--metadata-placement outside|inside|none`, `--info-panel-position bottom|right`, `--info-panel-size`, `--info-panel-mode external|inline|none`, `--comparison-layout processed_overlay|original_overlay|original_processed_overlay|chop_overlay` | External bottom panel; `chop_overlay` layout unsupported |
| Config/output | `--config`, `--output-dir`, `--json` | `configs/experiment/A2_real.yaml`, `outputs/nvila_hd_video_poc` |

## Full Pipeline Component Plug-In Mode

`full_pipeline` has a config-driven plug-in surface for the vision encoder and MLLM.
The script intentionally does not expose direct CLI overrides such as `--vision-module-path` or `--mllm-module-path`.
Instead, it reads the component wiring from the experiment config passed by `--config`, so module path, class/factory name, checkpoint path, processor path, tokenizer path, and construction kwargs stay synchronized.

Current plug-in status:

| Component | Config section | Runtime behavior |
|---|---|---|
| AutoGaze | `model.autogaze` | Resolves `module_path` and `class_or_factory`, constructs through `from_pretrained` when available, passes `gaze_ratio` and `task_loss_requirement` to runtime calls |
| Vision encoder / ViT | `model.vision_encoder` | Resolves `module_path` and `class_or_factory`, loads `checkpoint` or `model_config_path`, calls `model(video, gazing_info=gaze_output)` when accepted and falls back to `model(video)` on `TypeError` |
| MLLM processor | `model.mllm` | Resolves `nvila_hd_video_processor_module_path` and `nvila_hd_video_processor_class_name`, loads `processor_path`, formats `prompt_template` with `{video_token}` and query text |
| MLLM model | `model.mllm` | Resolves `module_path` and `class_or_factory`, loads `checkpoint`, calls `generate(..., max_new_tokens=N)`, decodes with `processor.batch_decode` |

Required config fields for plug-in construction:

```yaml
model:
  vision_encoder:
    module_path: autogaze.vision_encoders.siglip.modeling_siglip
    class_or_factory: SiglipVisionModel
    checkpoint: weights/siglip2-base-patch16-224
    processor_path: weights/siglip2-base-patch16-224
    construction_kwargs:
      attn_implementation: sdpa
      scales: 32+64+112+224
  mllm:
    module_path: transformers
    class_or_factory: AutoModel
    checkpoint: weights/NVILA-8B-HD-Video
    processor_path: weights/NVILA-8B-HD-Video
    tokenizer_path: weights/NVILA-8B-HD-Video
    nvila_hd_video_processor_module_path: transformers
    nvila_hd_video_processor_class_name: AutoProcessor
    prompt_template: "{video_token}\n\n{prompt}"
    construction_kwargs:
      device_map: auto
```

Full-pipeline plug-in execution is guarded:

```text
--allow-checkpoint-load            required for real AutoGaze, ViT, and MLLM execution
--checkpoint-metadata-only         validates paths/imports but skips model execution
--no-checkpoint-load               forces skip behavior even if checkpoints exist
```

If a component cannot be imported, constructed, or called, the script records a `skipped_stages` entry and a stage-level failure reason in `logs/poc_summary.json`.
It must not silently fall back to a different model path.

Current limitations:

```text
module/class/checkpoint overrides are config-driven, not CLI flags
use a dedicated A1/A2/A0/A3 config YAML rather than mixing component paths through ad-hoc CLI overrides
only the official processor path is implemented for NVILA-HD-Video MLLM input
direct visual token injection into arbitrary MLLMs is not claimed
vision output extraction is limited to tensor, last_hidden_state, visual_features, or first tuple tensor
generic adapter registry integration remains future work
```

Benchmark preset files now include a `full_pipeline_plugin_mode: experiment_config` field and a `component_plugins` block that records the expected AutoGaze, ViT, processor, and MLLM config fields. This is metadata for auditability; the actual construction still comes from `--config`.

Example full-pipeline plug-in command:

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --gaze-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --max-new-tokens 1 \
  --save-overlay-video \
  --save-side-by-side-video \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/full_pipeline_plugin
```

## A1 and A2 Canonical Config Guide

Use A1 and A2 for the first canonical comparison:

| Config | AutoGaze | Vision encoder | MLLM | Intended use |
|---|---:|---|---|---|
| `configs/experiment/A1_real.yaml` | OFF | modified SigLIP ViT | NVILA-HD-Video | Full-token modified-SigLIP baseline |
| `configs/experiment/A2_real.yaml` | ON | modified SigLIP ViT | NVILA-HD-Video | Canonical AutoGaze-enabled path |

Important interpretation rules:

```text
A1_real is the baseline. AutoGaze is disabled, so autogaze_only visualization is not meaningful for A1.
A2_real is the AutoGaze path. Use it for AutoGaze metadata, patch overlays, token reduction, and A2 full-pipeline checks.
Both configs are guarded real paths. Real model construction needs --allow-checkpoint-load.
If checkpoint loading is disabled, query text is accepted but generation is recorded as skipped.
```

### A1 Check Mode

```bash
python scripts/poc_nvila_hd_video.py \
  --mode check \
  --config configs/experiment/A1_real.yaml \
  --output-dir outputs/nvila_hd_video_poc/A1_check
```

Expected behavior:

```text
AutoGaze import stage: disabled
SigLIP import/path checks: reported from model.vision_encoder
NVILA import/path checks: reported from model.mllm
No heavy checkpoint loading
```

### A1 Full-Token Full Pipeline

Use this for the modified-SigLIP + NVILA baseline. It should not report AutoGaze token reduction.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --max-new-tokens 1 \
  --config configs/experiment/A1_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/A1_full_pipeline
```

Expected A1 labeling in outputs:

```text
experiment_id = A1_real
autogaze_enabled = false
selected_visual_token_count should equal the full-token path when reported
skipped_stages must explain any missing NVILA generation
```

### A2 AutoGaze-Only Visualization

Use this for AutoGaze selected patches, scales, token counts, masks, and videos.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --gaze-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --overlay-style both \
  --multi-scale-overlay \
  --save-overlay-video \
  --save-side-by-side-video \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/A2_autogaze_only
```

Expected A2 AutoGaze outputs:

```text
autogaze/runtime_metadata.json
autogaze/token_counts_summary.json
autogaze/windows/window_000/selected_patch_indices.json
autogaze/windows/window_000/selected_scales.json
visualizations/autogaze/videos/autogaze_overlay.mp4
visualizations/autogaze/videos/autogaze_side_by_side.mp4
visualizations/autogaze/metadata/visualization_skip_metadata.json  # only when AutoGaze is skipped
logs/metrics.json
logs/poc_summary.json
```

### A2 Full Pipeline With Query Text

Use this to test the canonical AutoGaze + modified SigLIP + NVILA-HD-Video path.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --gaze-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --max-new-tokens 1 \
  --save-overlay-video \
  --save-side-by-side-video \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/A2_full_pipeline
```

Expected A2 labeling in outputs:

```text
experiment_id = A2_real
autogaze_enabled = true
query_text is saved or used; it must not be silently ignored
visualization outputs are saved under visualizations/full_pipeline/ and visualizations/autogaze/
skipped generation is not treated as successful generation
```

### A1 vs A2 Internal Comparison Checklist

Before comparing A1 and A2, confirm these fields match:

```text
input video
frame_selection_mode
num_frames
scaling_mode
resolution
device
dtype
max_new_tokens
query_text
```

Then compare:

```text
original_visual_token_count
selected_visual_token_count
token_reduction_ratio
autogaze_latency_ms
vision_encoder_latency_ms
mllm_decode_latency_ms
end_to_end latency in logs/poc_summary.json
peak_vram_mb, or N/A on CPU/MPS
skipped_stages
```

Do not claim encoder-side acceleration unless A2 reduces tokens before the intended encoder compute stage.

### A1 vs A2 Benchmark Wrapper

For latency, memory, and token-count comparison, use the PoC benchmark wrapper before running larger benchmark configs:

```bash
python scripts/benchmark_poc_autogaze_impact.py \
  --mode full_pipeline \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 1 \
  --output-dir outputs/poc_autogaze_impact
```

The default is a dry run. It writes the exact A1/A2 PoC commands to:

```text
outputs/poc_autogaze_impact/benchmark_plan.json
outputs/poc_autogaze_impact/commands.sh
```

Execute real guarded PoC runs only when the modules/checkpoints and device are ready:

```bash
python scripts/benchmark_poc_autogaze_impact.py \
  --mode full_pipeline \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --device mps \
  --dtype float32 \
  --max-new-tokens 1 \
  --allow-checkpoint-load \
  --execute \
  --output-dir outputs/poc_autogaze_impact
```

Summarize existing PoC outputs:

```bash
python scripts/benchmark_poc_autogaze_impact.py \
  --summarize-existing \
  --output-dir outputs/poc_autogaze_impact
```

Summary outputs:

```text
outputs/poc_autogaze_impact/autogaze_impact_summary.json
outputs/poc_autogaze_impact/autogaze_impact_summary.csv
```

The summary checks that A1 and A2 used matching axes and reports:

```text
original_visual_token_count
selected_visual_token_count
token_reduction_ratio
autogaze_latency_ms
vision_encoder_latency_ms
mllm_decode_latency_ms
end_to_end_latency_ms
peak_vram_mb
skipped_stages
```

Visualization export is disabled by default in the wrapper so timing is not dominated by MP4 writing. Add `--include-visualization` only when the comparison is for inspectability, not latency.

## Output Type Samples by Mode

This section shows representative output shapes, metadata, and ASCII visualization layouts.
Exact numeric values depend on the model outputs and input video.

### `check` Mode Outputs

`check` mode validates references and writes logs only. It does not run AutoGaze, SigLIP, or NVILA inference.

Output tree:

```text
outputs/<run>/
  logs/
    metrics.json
    metrics.csv
    poc_summary.json
```

Representative `logs/poc_summary.json` excerpt:

```json
{
  "mode": "check",
  "status": "passed",
  "experiment_id": "A2_real",
  "checkpoint_policy": "disabled",
  "selected_token_count": null,
  "original_visual_token_count": null,
  "skipped_stages": []
}
```

If something is missing:

```json
{
  "status": "blocked",
  "skipped_stages": [
    {
      "stage": "path_check",
      "reason": "nvila_checkpoint missing: weights/NVILA-8B-HD-Video"
    }
  ]
}
```

ASCII view:

```text
[config] -> [import checks] -> [path checks] -> logs only
```

### `autogaze_only` Mode Outputs

`autogaze_only` mode runs through AutoGaze and produces selected patch metadata, token counts, optional visualizations, and metrics.
If checkpoint loading is disabled or AutoGaze fails before selected patch metadata is available, the PoC writes `visualizations/autogaze/metadata/visualization_skip_metadata.json` instead of leaving missing images/videos unexplained. `logs/metrics.json` mirrors this as `visualization_status=skipped` and `visualization_skip_reason`.

Output tree:

```text
outputs/<run>/
  autogaze/
    frame_selection_metadata.json
    runtime_metadata.json
    token_counts_summary.json
    windows/
      window_000/
        selected_patch_indices.json
        selected_scales.json
        selected_patch_mask.json
        token_counts.json
  scaling/
    scaling_metadata.json
  visualizations/
    autogaze/
      frames/
      videos/
      metadata/
        visualization_skip_metadata.json  # guarded/no-checkpoint or failed AutoGaze path
  logs/
    metrics.json
    metrics.csv
    poc_summary.json
```

Representative `selected_patch_indices.json`:

```json
{
  "selected_patch_indices": [[0, 1, 18, 31]]
}
```

Representative `selected_scales.json`:

```json
{
  "selected_scales": [[32, 64, 32, 112]]
}
```

Representative `token_counts.json`:

```json
{
  "original_visual_token_count": 392,
  "selected_visual_token_count": 48,
  "token_reduction_ratio": 0.8775510204081632,
  "patch_grid": [14, 14],
  "window_id": 0,
  "frame_indices": [0, 3]
}
```

ASCII patch mask example for one frame:

```text
Patch grid 4 x 4

    c0  c1  c2  c3
r0 [X] [X] [ ] [ ]
r1 [ ] [ ] [ ] [ ]
r2 [ ] [ ] [X] [ ]
r3 [ ] [ ] [ ] [X]

X = AutoGaze-selected patch
```

ASCII multi-scale overlay example:

```text
scale_0 = light yellow
scale_1 = orange
scale_2 = pink
scale_3 = purple

Frame 0 overlay:

+-----------------------+
| Y0 | O1 |    |        |
|    |    |    |        |
|    |    | P2 |        |
|    |    |    | U3     |
+-----------------------+

Y/O/P/U = selected patch colored by scale
number  = patch or scale label if enabled
```

### Full Pipeline Mode Outputs

`full_pipeline` mode saves the same AutoGaze outputs as `autogaze_only`, then attempts the vision encoder and NVILA generation when checkpoint loading is explicitly enabled.

Output tree:

```text
outputs/<run>/
  autogaze/
    ...
  predictions/
    answer.json
  visualizations/
    full_pipeline/
    autogaze/
  logs/
    metrics.json
    metrics.csv
    poc_summary.json
```

Representative `predictions/answer.json` when generation succeeds:

```json
{
  "answer": "B",
  "query_text": "Question: What does the sign say? Please answer directly."
}
```

Representative skipped generation entry:

```json
{
  "stage": "nvila_generation",
  "reason": "query text was accepted, but NVILA generation was skipped because checkpoint loading is disabled"
}
```

ASCII full pipeline flow:

```text
video + query text
      |
      v
[frame selection] -> [scaling/chop] -> [AutoGaze]
                                           |
                                           +--> metadata + visualization
                                           |
                                           v
                                    [SigLIP ViT]
                                           |
                                           v
                                  [NVILA generation]
                                           |
                                           v
                                  predictions/answer.json
```

### Visualization Output Samples

Processed overlay video:

```text
visualizations/autogaze/videos/autogaze_overlay.mp4

+--------------------------------+
| processed frame                |
| +----+                         |
| |mask| selected patch overlay  |
| +----+                         |
|                                |
+--------------------------------+
| frame 0 | source index 0       |
| selected patches: 24           |
| tokens: 48/392                 |
+--------------------------------+
```

Side-by-side video with `processed_overlay`:

```text
visualizations/autogaze/videos/autogaze_side_by_side.mp4

+----------------------+----------------------+
| Original / Processed | AutoGaze Overlay     |
|                      | +----+ +----+        |
| no overlay           | |mask| |box |        |
|                      | +----+ +----+        |
+----------------------+----------------------+
| external info panel below both panes        |
+---------------------------------------------+
```

Scale panel video:

```text
visualizations/autogaze/videos/autogaze_scale_panels.mp4

+-------------------+-------------------+
| scale 0 mask      | scale 1 mask      |
| yellow patches    | orange patches    |
+-------------------+-------------------+
| scale 2 mask      | scale 3 mask      |
| pink patches      | purple patches    |
+-------------------+-------------------+
```

Original-space overlay:

```text
visualizations/autogaze/videos/autogaze_original_overlay.mp4

original frame coordinates
+----------------------------------------+
| selected processed-frame patches        |
| mapped back using recorded scale_x/y    |
| mapping_exact: true                     |
+----------------------------------------+
```

Full-length video export:

```text
Input video frames:      0   1   2   3   4   5
Processed frame indices: 0       2       4

full_length output:
frame 0 = overlay frame
frame 1 = original/unprocessed frame
frame 2 = overlay frame
frame 3 = original/unprocessed frame
frame 4 = overlay frame
frame 5 = original/unprocessed frame
```

Chop overlay union:

```text
Full processed frame
+-----------------------+
| chop_000 | chop_001   |
|          |            |
+-----------------------+
| chop_002 | chop_003   |
|          |            |
+-----------------------+

Each chop has local patch coordinates.
overlay_union maps them back to the full processed-frame grid.
```

Representative `chop_overlay_metadata.json` excerpt:

```json
{
  "status": "implemented",
  "merge_mode": "overlay_union",
  "windows": [
    {
      "coordinate_space": "full_processed_frame",
      "overlapping_region_handling": "last_patch_wins_for_duplicate_patch_indices",
      "scale_conflict_handling": "last_scale_wins",
      "mapping_status": "exact"
    }
  ]
}
```

## Common Commands

### 1. Check Mode

Use this first. It does not instantiate heavy models by default.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode check \
  --config configs/experiment/A2_real.yaml \
  --output-dir outputs/nvila_hd_video_poc/check
```

Sample console output:

```text
NVILA-HD-Video canonical PoC
mode: check
status: passed
experiment: A2_real
checkpoint_policy: disabled
- autogaze_import: passed
- siglip_import: passed
- nvila_model_import: passed
- nvila_processor: passed
artifacts:
  metrics_json: outputs/.../logs/metrics.json
  poc_summary: outputs/.../logs/poc_summary.json
```

If paths or imports are missing, `status` becomes `blocked` and the missing component is listed in `skipped_stages`.

### 2. AutoGaze-Only Smoke With Dummy Video

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --scaling-mode resize \
  --resolution 224 \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/autogaze_only
```

Expected outputs:

```text
outputs/nvila_hd_video_poc/autogaze_only/
  autogaze/frame_selection_metadata.json
  autogaze/runtime_metadata.json
  autogaze/token_counts_summary.json
  autogaze/windows/window_000/selected_patch_indices.json
  autogaze/windows/window_000/selected_scales.json
  autogaze/windows/window_000/selected_patch_mask.json
  autogaze/windows/window_000/token_counts.json
  scaling/scaling_metadata.json
  logs/metrics.json
  logs/metrics.csv
  logs/poc_summary.json
```

### 3. Full Pipeline With Query Text

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --num-frames 2 \
  --scaling-mode resize \
  --resolution 224 \
  --max-new-tokens 1 \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/full_pipeline
```

If NVILA generation is unavailable or too heavy, the query text is still recorded and the generation skip reason is written to:

```text
outputs/<run>/logs/poc_summary.json
outputs/<run>/logs/metrics.json
```

The script must not silently ignore query text.

## Frame Selection

`--num-frames` means frames per model forward pass, not total video length.
The script default is `16`, matching the canonical AutoGaze/NVILA code path and paper-style setting.
Examples below use smaller explicit values such as `--num-frames 4` only to keep the ASCII diagrams compact.
The result of frame selection is one or more inference windows. Each window is passed to AutoGaze independently.

| Mode | Behavior |
|---|---|
| `sample` | Uniformly sample `num_frames` from the whole video and run one window |
| `chunk` | Split into non-overlapping windows of `num_frames` |
| `interval` | Select `num_frames` at fixed `--frame-interval` |
| `all` | Treat whole video as chunked non-overlapping windows |

### Frame Selection ASCII Examples

Assume an input video with 10 decoded frames:

```text
original frames:
idx:  0 1 2 3 4 5 6 7 8 9
      | | | | | | | | | |
```

`sample` with `--num-frames 4` uniformly samples from the whole video and runs one window:

```text
command intent:
--frame-selection-mode sample --num-frames 4

selected frames:
idx:  0 1 2 3 4 5 6 7 8 9
      X     X     X     X

window_000 = [0, 3, 6, 9]

model forward passes:
window_000 -> one AutoGaze call with T=4
```

`chunk` with `--num-frames 4` splits the video into non-overlapping windows:

```text
command intent:
--frame-selection-mode chunk --num-frames 4

windows:
idx:  0 1 2 3 | 4 5 6 7 | 8 9
      W0      | W1      | W2

window_000 = [0, 1, 2, 3]
window_001 = [4, 5, 6, 7]
window_002 = [8, 9]

model forward passes:
window_000 -> AutoGaze call with T=4
window_001 -> AutoGaze call with T=4
window_002 -> AutoGaze call with T=2 unless padded or dropped
```

`chunk` with `--pad-last` keeps the final window at `num_frames` by repeating the last available frame:

```text
command intent:
--frame-selection-mode chunk --num-frames 4 --pad-last

window_002 before padding = [8, 9]
window_002 after padding  = [8, 9, 9, 9]
padded_frame_mask         = [false, false, true, true]

model forward pass:
window_002 -> AutoGaze call with T=4
effective_num_frames = 2
```

`chunk` with `--drop-last` removes the incomplete final window:

```text
command intent:
--frame-selection-mode chunk --num-frames 4 --drop-last

kept:
window_000 = [0, 1, 2, 3]
window_001 = [4, 5, 6, 7]

dropped:
window_002 = [8, 9]
```

`interval` with `--num-frames 4 --frame-interval 2` selects every second frame and runs one window:

```text
command intent:
--frame-selection-mode interval --num-frames 4 --frame-interval 2

selected frames:
idx:  0 1 2 3 4 5 6 7 8 9
      X   X   X   X

window_000 = [0, 2, 4, 6]

model forward passes:
window_000 -> one AutoGaze call with T=4
```

`all` is implemented as chunked full-video processing:

```text
command intent:
--frame-selection-mode all --num-frames 4

effective behavior:
same windowing as chunk mode

window_000 = [0, 1, 2, 3]
window_001 = [4, 5, 6, 7]
window_002 = [8, 9]
```

`--max-windows` caps how many windows are processed:

```text
command intent:
--frame-selection-mode chunk --num-frames 4 --max-windows 2

available windows:
window_000 = [0, 1, 2, 3]
window_001 = [4, 5, 6, 7]
window_002 = [8, 9]

processed windows:
window_000
window_001
```

Representative `frame_selection_metadata.json` for `sample --num-frames 4`:

```json
{
  "mode": "sample",
  "effective_mode": "sample",
  "num_frames": 4,
  "frame_interval": 1,
  "original_frame_count": 10,
  "number_of_windows": 1,
  "window_frame_indices": [[0, 3, 6, 9]],
  "video_export_mode": "sampled_only"
}
```

Representative per-window output layout for `chunk --num-frames 4`:

```text
outputs/<run>/autogaze/windows/
  window_000/
    selected_patch_indices.json
    token_counts.json
  window_001/
    selected_patch_indices.json
    token_counts.json
  window_002/
    selected_patch_indices.json
    token_counts.json
```

Examples:

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy \
  --frame-selection-mode sample --num-frames 4 \
  --config configs/experiment/A2_real.yaml --allow-checkpoint-load
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy \
  --frame-selection-mode chunk --num-frames 4 --max-windows 2 \
  --config configs/experiment/A2_real.yaml --allow-checkpoint-load
```

```bash
python scripts/poc_nvila_hd_video.py --mode autogaze_only --video dummy \
  --frame-selection-mode interval --num-frames 4 --frame-interval 3 \
  --config configs/experiment/A2_real.yaml --allow-checkpoint-load
```

Frame selection metadata is saved to:

```text
outputs/<run>/autogaze/frame_selection_metadata.json
```

## Scaling Modes

| Mode | Status | Meaning |
|---|---|---|
| `none` | guarded | Use decoded resolution as-is; no silent resize |
| `resize` | implemented | Resize to square `resolution x resolution`; aspect ratio may distort |
| `fit_short_side` | implemented | Preserve aspect ratio; short side becomes `resolution` |
| `fit_long_side` | implemented | Preserve aspect ratio; long side becomes `resolution` |
| `quickstart` | guarded exact policies | Supports documented `224/patch16` and `392/patch14` policies |
| `chop` | partial | Non-overlapping spatio-temporal tiles with metadata |

Quickstart example:

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --scaling-mode quickstart \
  --resolution 224 \
  --patch-size 16 \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/quickstart_224
```

Quickstart metadata includes:

```json
{
  "scaling_mode": "quickstart",
  "quickstart_reference_used": "docs/QUICK_START_reference.md",
  "quickstart_exact_match": true,
  "quickstart_differences": [],
  "unsupported_reason": null
}
```

Unsupported quickstart requests are not silently mapped to resize. They report `unsupported_reason`.

## AutoGaze Runtime Controls

Supported controls:

```text
--gaze-ratio <float>
--task-loss-requirement <float>
--target-scales 56,112,196,392
--target-patch-size <int>
```

Example:

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --gaze-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/runtime_controls
```

Runtime metadata is saved to:

```text
outputs/<run>/autogaze/runtime_metadata.json
```

## Visualization

Supported overlay controls:

```text
--save-overlay-video
--save-side-by-side-video
--save-scale-panel-video
--overlay-style mask|box|both
--multi-scale-overlay / --no-multi-scale-overlay
--scale-color-mode gradient|categorical
--show-patch-index
--show-scale-label
--metadata-placement outside|inside|none
```

Patch index labels are disabled by default. Enable them only for inspection:

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --overlay-style both \
  --show-patch-index \
  --show-scale-label \
  --save-overlay-video \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/patch_index_visuals
```

Sample visualization outputs:

```text
outputs/<run>/visualizations/autogaze/videos/autogaze_overlay.mp4
outputs/<run>/visualizations/autogaze/videos/autogaze_side_by_side.mp4
outputs/<run>/visualizations/autogaze/videos/autogaze_scale_panels.mp4
outputs/<run>/visualizations/autogaze/metadata/visualization_video_metadata.json
```

## Full-Length Video Export

`sampled_only` is the default. It exports only processed frames.

`full_length` inserts AutoGaze overlays back into the original timeline for non-overlapping frame selections.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --frame-selection-mode sample \
  --num-frames 2 \
  --save-overlay-video \
  --save-side-by-side-video \
  --video-export-mode full_length \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/full_length_export
```

Expected outputs:

```text
outputs/<run>/visualizations/autogaze/videos/autogaze_overlay_full_length.mp4
outputs/<run>/visualizations/autogaze/videos/autogaze_side_by_side_full_length.mp4
```

Full-length metadata includes:

```json
{
  "video_export_mode": "full_length",
  "original_frame_count": 4,
  "output_fps": 4.0,
  "full_length_export": {
    "status": "implemented",
    "processed_frame_indices": [0, 3],
    "unprocessed_frame_policy": "original_frame_if_available_else_black_frame",
    "output_frame_count": 4,
    "exact": true
  }
}
```

`hold_last` is still stub-only and raises `NotImplementedError`.

## Original-Space Overlay

Supported exact mappings:

```text
none
resize
fit_short_side
fit_long_side
supported quickstart policies
```

Example:

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --scaling-mode fit_short_side \
  --resolution 224 \
  --save-side-by-side-video \
  --comparison-layout original_processed_overlay \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/original_space_overlay
```

Expected outputs:

```text
outputs/<run>/visualizations/autogaze/videos/autogaze_original_overlay.mp4
outputs/<run>/visualizations/autogaze/videos/autogaze_original_processed_overlay.mp4
```

Metadata records scale factors, padding, crop offsets, and `mapping_exact`.

Chop original-space overlay is not guessed. Use `overlay_union` for full processed-frame chop overlays.

## Chop Mode and Merged Overlay

Chop mode is for high-resolution preparation. It currently supports non-overlapping chops.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 4 \
  --scaling-mode chop \
  --resolution 392 \
  --spatial-tile-size 224 \
  --chop-overlap 0 \
  --chop-merge-mode overlay_union \
  --save-chop-overlay-video \
  --config configs/experiment/A2_real.yaml \
  --allow-checkpoint-load \
  --output-dir outputs/nvila_hd_video_poc/chop_overlay_union
```

Expected outputs:

```text
outputs/<run>/chops/chop_metadata.json
outputs/<run>/chops/windows/window_000/frame_000/chop_000/token_counts.json
outputs/<run>/visualizations/autogaze/windows/window_000/frames/frame_000_chop_merged_overlay.png
outputs/<run>/visualizations/autogaze/videos/autogaze_chop_overlay.mp4
outputs/<run>/visualizations/autogaze/metadata/chop_overlay_metadata.json
```

Unsupported:

```text
non-zero chop overlap
custom chop stride
arbitrary full original-frame chop overlay
comparison_layout=chop_overlay
```

These fail clearly because coordinate mapping would otherwise be misleading.

## Metrics and Reports

Every run writes:

```text
outputs/<run>/logs/poc_summary.json
outputs/<run>/logs/metrics.json
outputs/<run>/logs/metrics.csv
```

Important fields:

```text
status
result_label
skipped_stages
frame_selection_mode
scaling_mode
original_visual_token_count
selected_visual_token_count
token_reduction_ratio
autogaze_latency_ms
vision_encoder_latency_ms
mllm_decode_latency_ms
peak_vram_mb
```

CPU and MPS memory metrics that are unavailable are recorded as `N/A`.

## Benchmark Smoke Configs

Safe PoC configs:

| Config | Mode | Main options represented |
|---|---|---|
| `configs/benchmark/poc_default.yaml` | `check` | Canonical defaults, `num_frames=16`, config-driven plug-in metadata |
| `configs/benchmark/poc_autogaze_impact_full_pipeline.yaml` | `full_pipeline` | A1/A2 AutoGaze impact comparison plan using `scripts/benchmark_poc_autogaze_impact.py` |
| `configs/benchmark/poc_feature_matrix_smoke.yaml` | `autogaze_only` | Frame selection, resize scaling, overlay and side-by-side video |
| `configs/benchmark/poc_autogaze_only_visualization.yaml` | `autogaze_only` | AutoGaze image/video visualization |
| `configs/benchmark/poc_full_pipeline_visualization.yaml` | `full_pipeline` | Query text, ViT/MLLM plug-in metadata, overlay, side-by-side, scale panel |
| `configs/benchmark/poc_chop_mode_smoke.yaml` | `autogaze_only` | `scaling_mode=chop`, chop metadata, per-chop frames |
| `configs/benchmark/poc_multiscale_visualization.yaml` | `autogaze_only` | Multi-scale gradient overlay, scale labels |
| `configs/benchmark/poc_scale_panel_video.yaml` | `autogaze_only` | `save_scale_panel_video`, `scale_panel_layout=2x2` |
| `configs/benchmark/poc_high_resolution_chop_smoke.yaml` | `autogaze_only` | Safe high-resolution/chop smoke, `overlay_union`, bounded `max_chops` |
| `configs/benchmark/poc_high_resolution_chop_medium.yaml` | `autogaze_only` | Medium preparation only, `num_frames=16`, bounded iterations/chops |
| `configs/benchmark/poc_full_length_video_export_smoke.yaml` | `autogaze_only` | `video_export_mode=full_length` with tiny input |

All benchmark presets include these audit fields:

```yaml
full_pipeline_plugin_mode: experiment_config
component_plugins:
  status: config_driven_guarded
  autogaze:
    config_section: model.autogaze
  vision_encoder:
    config_section: model.vision_encoder
    input_contract: "[B,T,C,H,W] with optional gazing_info"
  mllm:
    config_section: model.mllm
    official_processor_path: true
```

The audit fields document the options that must be respected by benchmark execution.
They do not replace `--config`; the experiment config remains the source of truth for real module paths and checkpoints.

A1/A2 preset usage:

```text
Use configs/experiment/A1_real.yaml for the full-token modified-SigLIP baseline.
Use configs/experiment/A2_real.yaml for AutoGaze metadata, visualization, and AutoGaze-enabled full-pipeline checks.
```

They are config templates and smoke paths only. Do not treat them as benchmark results.

Safety limits:

```text
batch_size = 1
small num_frames
bounded max_chops
small benchmark_iterations
run_by_default = false for PoC benchmark presets, including high-resolution preparation configs
```

## Unsupported or Stub-Only Paths

| Feature | Status | Reason |
|---|---|---|
| `hold_last` video export | stub-only | Needs a temporal carry-forward policy |
| Non-zero chop overlap | unsupported | Merge semantics are not validated |
| Custom chop stride | unsupported | Could create misleading coordinate mapping |
| `comparison_layout=chop_overlay` | unsupported | Use `--chop-merge-mode overlay_union` instead |
| Real NVILA generation by default | guarded | Large model, requires explicit checkpoint loading and memory |
| Public benchmark reproduction | future work | Requires official datasets, metrics, and validated real inference |

## Debugging Checklist

1. Run `--mode check` first.
2. Confirm module imports and checkpoint/config paths.
3. Use `--video dummy` before local videos.
4. Use `--no-checkpoint-load` or default check mode for path validation.
5. Add `--allow-checkpoint-load` only after check mode passes.
6. Inspect `logs/poc_summary.json` for skipped stages.
7. Inspect `scaling/scaling_metadata.json` before trusting overlays.
8. Inspect visualization metadata before using video outputs for analysis.

## Minimal Test Command

```bash
pytest tests/test_frame_selector.py tests/test_autogaze_scaling.py tests/test_inference_guide_alignment.py tests/test_visualization.py tests/test_poc_nvila_hd_video.py
```
