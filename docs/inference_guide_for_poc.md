# PoC Inference Guide

This branch is now scoped to the lightweight PoC inference surface only. It keeps the runnable inference entry points, A0-A3, E1-E4, and Q0-Q1 configs, local model adapters, flat visualizations, and metrics output. Broader benchmark, HF dataset, profiling, and all-in-one NVILA-HD experiment code were removed from this branch.

## Kept Files

Core scripts:

```text
scripts/infer_autogaze.py
scripts/infer_full.py
scripts/poc_infer_utils.py
scripts/poc_model_adapters.py
scripts/poc_model_registry.py
```

Configs:

```text
configs/poc_inference/A0_vanilla_siglip_nvila_off.yaml
configs/poc_inference/A1_modified_siglip_nvila_off.yaml
configs/poc_inference/A2_modified_siglip_nvila_on.yaml
configs/poc_inference/A3_vanilla_siglip_nvila_on.yaml
configs/poc_inference/E1_vjepa2_encoder.yaml
configs/poc_inference/E2_qwen_mllm.yaml
configs/poc_inference/E3_vjepa2_qwen.yaml
configs/poc_inference/E4_qwen_autogaze_vision_mask.yaml
configs/poc_inference/Q0_qwen_autogaze_off.yaml
configs/poc_inference/Q1_qwen_autogaze_on.yaml
```

Tests:

```text
tests/test_poc_inference_visualizer_priority1.py
```

Reference docs:

```text
INTEGRATION.md
QUICK_START.md
docs/nvila-hd-video-readme.md
docs/INFERENCE_GUIDE.md
docs/inference_guide_for_poc.md
```

## Presets

| Config | AutoGaze | Vision encoder | MLLM | Purpose |
|---|---:|---|---|---|
| `A0_vanilla_siglip_nvila_off.yaml` | off | vanilla SigLIP | NVILA | full-token vanilla baseline |
| `A1_modified_siglip_nvila_off.yaml` | off | modified SigLIP | NVILA | modified-SigLIP baseline |
| `A2_modified_siglip_nvila_on.yaml` | on | modified SigLIP | NVILA | canonical AutoGaze-style path |
| `A3_vanilla_siglip_nvila_on.yaml` | on | vanilla SigLIP | NVILA | experimental compatibility path |
| `E1_vjepa2_encoder.yaml` | off | V-JEPA2 | generic MLLM | V-JEPA2 loading smoke |
| `E2_qwen_mllm.yaml` | off | skipped/generic | Qwen | Qwen official processor smoke |
| `E3_vjepa2_qwen.yaml` | on | V-JEPA2 | Qwen | extension smoke with explicit blockers |
| `E4_qwen_autogaze_vision_mask.yaml` | on | Qwen-owned vision | Qwen | AutoGaze masks Qwen-native vision patches |
| `Q0_qwen_autogaze_off.yaml` | off | Qwen-owned vision | Qwen | explicit Qwen baseline for ON/OFF comparison |
| `Q1_qwen_autogaze_on.yaml` | on | Qwen-owned vision | Qwen | explicit Qwen AutoGaze-mask comparison |

Local defaults point at this workspace cache:

| Component | Default path |
|---|---|
| AutoGaze | `weights/AutoGaze` |
| SigLIP2 base | `weights/siglip2-base-patch16-224` |
| NVILA-HD-Video | `weights/NVILA-8B-HD-Video` |
| V-JEPA2 | `weights/vjepa2-vitl-fpc64-256` |
| Qwen2.5-VL | `weights/Qwen2.5-VL-7B-Instruct` |

Relative paths are resolved from the repo root.

For NVILA, the local `weights/NVILA-8B-HD-Video/preprocessor_config.json` may still contain the upstream default `autogaze_model_id: bfshi/AutoGaze`. The PoC configs override that processor argument to `weights/AutoGaze`, and the adapter resolves it to an absolute local path before loading. This avoids offline failures such as `bfshi/AutoGaze is not a local folder and is not a valid model`.

The same local NVILA processor defaults to `num_video_frames=8`, while the local AutoGaze checkpoint reports `max_num_frames=16`. NVILA's tile processor requires `num_video_frames` to be a positive multiple of AutoGaze `max_num_frames`, so A0-A3 explicitly pass `num_video_frames: 16` and `num_video_frames_thumbnail: 16`. If a custom config sets an invalid value such as `8`, the adapter blocks with a clear message before generation.

AutoGaze always executes in `float32`, even when the CLI or config requests a mixed-precision runtime dtype. The requested dtype is still recorded in `autogaze/runtime_metadata.json` and `logs/metrics.json`, and `autogaze_forced_float32=true` marks when it was overridden for AutoGaze.

For the full pipeline, `--dtype` / `runtime.dtype` remains the default dtype for non-AutoGaze modules. Use `--mllm-dtype` or `runtime.mllm_dtype` to choose the MLLM generation dtype independently, for example `--dtype float32 --mllm-dtype bfloat16`.

The PoC adapters pass Hugging Face model dtype with the current `dtype` keyword instead of deprecated `torch_dtype`. Official processor configs set `use_fast: false` explicitly to preserve the slow image processor behavior saved with the local checkpoints and avoid the Transformers warning about a future default change. Override `processor_from_pretrained_kwargs.use_fast` only when you intentionally want to test the fast processor path.

The local NVILA checkpoint code still reads `config.torch_dtype` internally. That access is inside checkpoint-provided remote code, so the PoC does not edit it. Instead, inference wraps Hugging Face model/processor loading with a targeted filter for the exact deprecation warning while still passing `dtype` from our own code.

## Progress And Latency

Inference commands show tqdm-style progress for the timed module stages:

- `AutoGaze`
- `ViT encoder`, when the full pipeline requires a separate encoder
- `MLLM`

Real module timing uses one warm-up run by default before measurement. The default is configured as `runtime.warmup_runs: 1` and can be overridden with `--warmup-runs 0` or another integer. Warm-up, model loading, artifact writing, and visualization are excluded from the module latency metrics.

The main processing latency in `logs/metrics.json` is `module_processing_latency_ms`, also mirrored to `end_to_end_latency_ms` for compatibility. It is the sum of measured preprocessing and module processing latencies only:

```text
autogaze_latency_ms + vision_encoder_latency_ms + mllm_generation_latency_ms
```

`autogaze_latency_ms` is inclusive: it records preprocessing needed to build the AutoGaze inputs plus the AutoGaze selector/result stage over all processed frames or crops. For `resize_then_chop`, this includes resize/chop expansion time and the work over all processed crop frames. The split fields `autogaze_preprocessing_latency_ms`, `autogaze_stage_latency_ms`, `autogaze_model_forward_latency_ms`, and `autogaze_result_build_latency_ms` are also reported. `visualization_latency_ms` and `wall_clock_latency_ms` are recorded separately. Use `--no-progress` to disable command-line progress bars while keeping the same timing behavior.

## Real Loading Policy

By default, scripts do not load heavy checkpoints. They run guarded smoke paths and record `stub`, `skipped`, or `blocked` adapter status.

Use `--allow-real-model-loading` to request real loading. With that flag:

- missing checkpoints block clearly;
- incomplete Qwen sharded checkpoints block before model construction;
- requested model types are not silently replaced by another model;
- direct visual-token injection into NVILA or Qwen is not claimed unless an adapter explicitly supports it;
- Qwen AutoGaze vision masking is reported separately from visual-token shortening and encoder-side acceleration;
- `predictions/answer.json`, `logs/poc_summary.json`, and `logs/metrics.json` record adapter status and failure reasons.

## AutoGaze-Only Smoke

Dummy run:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path dummy \
  --output-dir /tmp/poc_a2_autogaze_dummy \
  --device cpu \
  --dtype float32 \
  --frame-selection-mode sample \
  --num-frames 4 \
  --scaling-mode resize \
  --resolution 64 \
  --gaze-ratio 0.25 \
  --save-frame-images \
  --save-side-by-side-video \
  --json
```

Real checkpoint attempt:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/video.mp4 \
  --output-dir outputs/poc_inference/a2_autogaze_real \
  --device cuda \
  --dtype float32 \
  --allow-real-model-loading \
  --json
```

If `weights/AutoGaze` or dependencies are missing, this exits as blocked instead of producing fake real output.

## Full Pipeline Smoke

Dummy full-pipeline run:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path dummy \
  --query-text "What is happening in this video?" \
  --output-dir /tmp/poc_a2_full_dummy \
  --device cpu \
  --dtype float32 \
  --frame-selection-mode sample \
  --num-frames 4 \
  --scaling-mode resize \
  --resolution 64 \
  --max-new-tokens 16 \
  --json
```

The full script always preserves query text in `predictions/answer.json` and `logs/metrics.json`. If generation is unavailable, it records the reason and keeps `query_text_used=true` when the adapter consumed the prompt path.

## NVILA Smoke

NVILA uses the official processor-first route represented by `docs/nvila-hd-video-readme.md`: load `AutoProcessor`, load `AutoModel`, format `{video_token}\n\n{prompt}`, pass video through the processor, call `generate`, and decode. The separate AutoGaze stage remains useful for visualization and metrics; direct visual-token injection into NVILA is not claimed.

README-style NVILA-HD presets are available:

```text
configs/poc_inference/nvila_hd_smoke.yaml
configs/poc_inference/nvila_hd_default.yaml
configs/poc_inference/nvila_hd_memory_safe.yaml
```

These expose the official names from `docs/nvila-hd-video-readme.md`: `--num-video-frames`, `--num-video-frames-thumbnail`, `--max-tiles-video`, `--gazing-ratio-tile`, `--task-loss-requirement-tile`, `--gazing-ratio-thumbnail`, `--task-loss-requirement-thumbnail`, `--max-batch-size-autogaze`, and `--max-batch-size-siglip`. See `docs/NVILA_HD_INFERENCE_UPDATE_AUDIT.md` and `docs/POC_INFERENCE_GUIDE.md` for the full matrix and commands.

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --mllm nvila \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float32 \
  --mllm-dtype bfloat16 \
  --max-new-tokens 32 \
  --output-dir outputs/poc_inference/a2_nvila_real \
  --json
```

A0/A1 set AutoGaze off. A2/A3 set AutoGaze on. When `sync_autogaze_controls_from_config` is true, the NVILA processor kwargs are aligned with the config and reported in adapter metadata.

## HLVid Evaluation

Use the isolated HLVid evaluator when you want the NVILA-HD processor setup from `docs/nvila-hd-video-readme.md` and exact multiple-choice scoring.

Config:

```text
configs/poc_inference/hlvid_nvila_hd_eval.yaml
```

Script:

```text
scripts/evaluate_hlvid_nvila.py
```

The default processor settings are the reference HLVid-style values:

```yaml
num_video_frames: 128
num_video_frames_thumbnail: 64
max_tiles_video: 48
gazing_ratio_tile: [0.2, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
task_loss_requirement_tile: 0.6
gazing_ratio_thumbnail: 1
task_loss_requirement_thumbnail: null
max_batch_size_autogaze: 16
max_batch_size_siglip: 32
```

Dry-run with a local HLVid-style JSON/JSONL file:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_eval.yaml \
  --dataset-path /path/to/hlvid.jsonl \
  --video-root /path/to/video/root \
  --max-samples 5 \
  --output-dir outputs/hlvid_nvila_dry_run \
  --dry-run
```

Real local-weight evaluation:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_eval.yaml \
  --dataset-name bfshi/HLVid \
  --model-path weights/NVILA-8B-HD-Video \
  --processor-path weights/NVILA-8B-HD-Video \
  --allow-real-model-loading \
  --local-files-only \
  --max-samples 20 \
  --output-dir outputs/hlvid_nvila_real
```

The evaluator formats each record as a multiple-choice prompt ending with `Please answer directly with the letter of the correct answer.`, decodes the generated answer, extracts `A/B/C/D`, and reports exact option-letter accuracy in `logs/metrics.json`. It does not inject AutoGaze-selected visual tokens manually; the official NVILA processor owns video decoding, tiling, AutoGaze controls, SigLIP batching, and generation input construction.

For `infer_full.py` on one HLVid video, avoid large `num_frames` and uncapped `resize_then_chop`; those can expand one window into many processed frames before NVILA generation. Use these bounded configs instead:

```text
configs/poc_inference/hlvid_infer_full_resize_safe.yaml
configs/poc_inference/hlvid_infer_full_resize_then_chop_safe.yaml
```

Safe resize command:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/hlvid_infer_full_resize_safe.yaml \
  --video-path /path/to/hlvid_video.mp4 \
  --query-text "Question: ... A. ... B. ... C. ... D. ... Please answer directly with the letter of the correct answer." \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --output-dir outputs/hlvid_infer_full_resize_safe
```

Bounded `resize_then_chop` command:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/hlvid_infer_full_resize_then_chop_safe.yaml \
  --video-path /path/to/hlvid_video.mp4 \
  --query-text "Question: ... A. ... B. ... C. ... D. ... Please answer directly with the letter of the correct answer." \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --output-dir outputs/hlvid_infer_full_resize_then_chop_safe
```

The bounded chop preset samples 16 frames, uses at most 4 spatial chops, and blocks if a window expands past 64 processed frames. Override only intentionally with `--max-chops`, `--max-processed-frames-per-window`, or `--max-processed-pixels-per-window`.

## Qwen Smoke

Qwen uses the official Qwen2.5-VL processor path: build a chat-template message with one video item and one text item, call `processor.apply_chat_template(..., add_generation_prompt=True)`, pass the video through the Qwen processor, call `generate`, and decode. Direct visual-token injection is unsupported in this PoC.

The local default config points at `weights/Qwen2.5-VL-7B-Instruct`. Full-pipeline configs use `mllm.video_input_source: processed_tensor`, so the MLLM receives the frames already loaded and scaled by the PoC pipeline instead of re-decoding the original video path. This avoids ffmpeg/backend path errors such as `Resource temporarily unavailable` without a retry/fallback path. Set `mllm.video_input_source: source_video` only when you intentionally want the official processor to decode the source path itself.

AutoGaze OFF baseline:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/Q0_qwen_autogaze_off.yaml \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --mllm qwen \
  --model-id weights/Qwen2.5-VL-7B-Instruct \
  --processor-path weights/Qwen2.5-VL-7B-Instruct \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float32 \
  --mllm-dtype bfloat16 \
  --max-new-tokens 32 \
  --output-dir outputs/poc_inference/q0_qwen_autogaze_off_real \
  --json
```

If any shard referenced by `model.safetensors.index.json` is missing, loading blocks with the missing filenames.

AutoGaze ON comparison with Qwen-native vision patch masking:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/Q1_qwen_autogaze_on.yaml \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --mllm qwen \
  --model-id weights/Qwen2.5-VL-7B-Instruct \
  --processor-path weights/Qwen2.5-VL-7B-Instruct \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float32 \
  --mllm-dtype bfloat16 \
  --frame-selection-mode sample \
  --num-frames 16 \
  --scaling-mode resize \
  --resolution 224 \
  --max-new-tokens 32 \
  --output-dir outputs/poc_inference/q1_qwen_autogaze_on_real \
  --json
```

This mode follows the `dev` branch concept for Qwen: AutoGaze-selected regions are mapped onto Qwen's own `video_grid_thw`, then a forward hook masks non-selected `model.visual.patch_embed` outputs before Qwen's visual encoder continues. The official Qwen processor and generation path are still used. It does not shorten the Qwen visual placeholder sequence and does not claim encoder-side acceleration; `logs/metrics.json` records `qwen_visual_mask_applied`, `qwen_visual_tokens_shortened=false`, and `qwen_encoder_side_acceleration_claimed=false`.

Dummy Qwen processor smoke, useful for checking wiring without a local video file:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/Q0_qwen_autogaze_off.yaml \
  --video-path dummy \
  --query-text "What is visible?" \
  --mllm qwen \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float32 \
  --mllm-dtype bfloat16 \
  --frame-selection-mode sample \
  --num-frames 4 \
  --scaling-mode resize \
  --resolution 224 \
  --max-new-tokens 16 \
  --output-dir outputs/poc_inference/q0_qwen_dummy_real \
  --json
```

`E2_qwen_mllm.yaml` and `E4_qwen_autogaze_vision_mask.yaml` remain supported legacy names for the same two Qwen concepts. Use `Q0` and `Q1` when running a direct Qwen AutoGaze OFF/ON comparison.

MLLM switching summary:

| MLLM | Config | Supported generation path | Direct AutoGaze visual tokens |
|---|---|---|---|
| NVILA | `A0`-`A3` | official NVILA processor, with AutoGaze processor controls in A2/A3 | not directly injected by this PoC |
| Qwen AutoGaze off | `Q0_qwen_autogaze_off.yaml` | official Qwen2.5-VL chat-template processor | unsupported |
| Qwen AutoGaze on | `Q1_qwen_autogaze_on.yaml` | official Qwen2.5-VL processor plus Qwen-native vision patch masking | unsupported; selected patches are masked inside Qwen vision, not injected as external tokens |
| Qwen legacy names | `E2_qwen_mllm.yaml`, `E4_qwen_autogaze_vision_mask.yaml` | same Qwen off/on concepts as Q0/Q1 | unsupported |
| Generic | `E1_vjepa2_encoder.yaml` | stub/status reporting only | unsupported until a model-specific adapter is added |

## External MLLM Adaptation Stubs

The external-MLLM review is maintained in `docs/mllm_adapt_report.md`. The external configs under `configs/poc_inference/external/` are not marked runnable; they are stub/blocker configs for adapter planning and lightweight validation only.

Current external registry keys:

```text
llava_ov
longva
longvila_r1
apollo
videollama3
videochat_flash
internvl3_5
qwen2_5_vl
```

Supported integration mode names are `official_processor`, `autogaze_frame_selection`, `autogaze_chop_selection`, `siglip_sparse_patch`, `post_encoder_pruning`, and `direct_visual_token_injection`. Unsupported modes raise explicit `NotImplementedError`; there is no fallback to NVILA, Qwen, or modified SigLIP. Direct selected-token injection remains disabled unless the adapter verifies positional IDs, attention masks, projector compatibility, visual token count, and placeholder alignment.

`infer_full.py` exposes these through `--integration-mode`. For example, this is a stub-only routing smoke that verifies the requested external adapter is used without loading a checkpoint:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --mllm llava_ov \
  --vision-encoder external \
  --integration-mode autogaze_frame_selection \
  --model-id local/llava-ov-test \
  --num-frames 2 \
  --resolution 32 \
  --no-progress \
  --json
```

This command is not a real LLaVA-OneVision generation run. It validates additive routing, adapter status reporting, and no silent fallback. Real external loading requires `--allow-real-model-loading` plus an explicit model ID/checkpoint and model-specific dependencies.

## V-JEPA2 Smoke

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E1_vjepa2_encoder.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --vision-encoder vjepa2 \
  --vision-encoder-ckpt weights/vjepa2-vitl-fpc64-256 \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --output-dir outputs/poc_inference/e1_vjepa2_real \
  --json
```

This validates requested V-JEPA2 loading. It does not claim AutoGaze patch ids directly map to V-JEPA2 tokens.

## Frame Selection And Scaling

Supported frame modes:

| Mode | Behavior |
|---|---|
| `sample` | uniform sample of `--num-frames` frames |
| `chunk` | non-overlapping windows of `--num-frames` |
| `interval` | fixed `--frame-interval` selection |
| `all` | chunk the whole video into non-overlapping windows |

Supported scaling modes:

| Mode | Behavior |
|---|---|
| `resize` | square resize to `--resolution` |
| `fit_short_side` | aspect-preserving short-side resize |
| `fit_long_side` | aspect-preserving long-side resize |
| `quickstart` | guarded 224/392-style policy from `QUICK_START.md` |
| `chop` | spatially chops selected source frames into real crop tensors, then keeps outputs flat over processed crop-frames |
| `resize_then_chop` | if the max side is above `--resize-before-chop-threshold`, resize by `--resize-before-chop-factor`, then chop and resize each crop |
| `none` | no resize |

`chop` follows the intent of the original `QUICK_START.md` any-resolution guidance: high-resolution frames are split into spatial crops before AutoGaze/ViT processing instead of being represented by one resized frame. For a 1k-ish frame with `--chop-size 224`, the number of processed crop-frames is roughly the number of spatial crops times the selected source frames. Each crop-frame still has the normal per-crop token layout, but aggregate source-frame token counts increase with the crop count.

Use `resize_then_chop` for very large frames when pure chop creates too many crops. The default policy is `--resize-before-chop-threshold 1024` and `--resize-before-chop-factor 0.5`, so full-HD frames are first reduced to half resolution before chopping. This reduces crop count and memory while preserving more local detail than a single whole-frame resize. The crop metadata stores original-frame `source_box` values for visualization and `chop_input_box` values for the actual resized crop coordinates.

For large and long videos, chop mode applies two separate operations. Spatially, each source frame is chopped into crops and each crop is resized to `--resolution`. Temporally, `all`/`chunk` modes split the video into `--num-frames` windows and drop the last incomplete window by default once at least one full window exists. For example, 49 frames with `--num-frames 16` processes 48 frames as three complete windows and drops the final one-frame remainder. Very short videos with fewer than `--num-frames` frames are padded to one valid window instead of being dropped entirely. `scaling/scaling_metadata.json` records dropped remainders as `temporal_drop_last_applied: true`. `sample` and `interval` keep their fixed request behavior and may still pad short selected windows.

The token-saving comparison in chop mode is therefore crop-expanded: `full_processed_visual_token_count` is the full token count across all processed crops, and `autogaze_selected_visual_token_count` is the subset selected by AutoGaze. `estimated_visual_token_savings_ratio` reports the reduction before the MLLM. In full inference, the MLLM adapter consumes the prepared tensor input by default. Metrics record this as `mllm_video_input_source: processed_tensor` or `processed_chop_tensor`.

There is no source-video fallback by default. For NVILA tensor inputs, the adapter passes one nested PIL-frame video to the official processor and pads or samples it to the validated `num_video_frames` setting, normally 16 with the local AutoGaze checkpoint, before calling the official processor. Qwen also receives prepared tensor frames through its official processor path; it does not claim AutoGaze visual-token injection.

Frame PNGs are no longer saved by default. Pass `--save-frame-images` or set `visualization.save_frame_images: true` to write `visualizations/autogaze/frames` and `visualizations/autogaze/scale_panels`. Video exports remain separate opt-in flags: `--save-overlay-video`, `--save-side-by-side-video`, and `--save-scale-panel-video`.

For visualization, chop mode renders merged source-frame views when frame images or videos are requested. Each selected crop overlay is projected back into its `source_box` on the original frame, so the frame/video view matches the original video layout. Crop-frame records remain available in JSON via `processed_frame_records` and `autogaze/selected_patch_indices.json`.

When `--frame-selection-mode all` is passed on the command line, the CLI treats it as an explicit request to process every frame window and ignores the smoke-config `frame_selection.max_windows: 1` cap. Use `--max-windows N` to cap it again, or `--max-windows 0` to request unlimited windows explicitly.

## Output Layout

Outputs are flat by processed frame:

```text
outputs/<run>/
  autogaze/
    frame_selection_metadata.json
    runtime_metadata.json
    selected_patch_indices.json
    selected_scales.json
    per_frame_token_counts.json
    token_counts_summary.json
  scaling/
    scaling_metadata.json
  chops/
    chop_metadata.json
  visualizations/
    autogaze/
      frames/frame_000000_overlay.png              # only with --save-frame-images
      scale_panels/frame_000000_scale_panel.png    # only with --save-frame-images
      videos/autogaze_overlay.mp4                  # only with --save-overlay-video
      videos/autogaze_side_by_side.mp4             # only with --save-side-by-side-video
      videos/autogaze_scale_panels.mp4             # only with --save-scale-panel-video
      metadata/visualization_metadata.json
  predictions/
    answer.json
  logs/
    poc_summary.json
    metrics.json
    metrics.csv
```

The current PoC does not write `visualizations/autogaze/windows/window_*/frames/`.

Scale panels are scale-aware. A low-resolution selected token such as scale `32` renders as a larger processed-frame footprint than scale `64`, `112`, or `224`. The exact normalized box and scale-local grid are saved in `autogaze/selected_patch_indices.json`.

## External Model Asset Workflow

External MLLM and V-JEPA2 smoke tests use a staged, dry-run-first workflow. The manifest is:

```text
configs/poc_inference/model_asset_manifest.yaml
```

Prepare a dry-run report without downloading:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

Verify local checkpoints without loading full weights:

```bash
python scripts/verify_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_ASSET_VERIFY_REPORT.md
```

Inspect local config files without loading weights:

```bash
python scripts/inspect_external_model_configs.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_CONFIG_INSPECTION_REPORT.md
```

Run an external smoke dry-run:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/smoke_qwen2_5_vl.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/smoke_qwen2_5_vl_dry \
  --dry-run \
  --local-files-only
```

Real model loading is disabled by default in every external smoke config. To run a real local smoke, pass `--allow-real-model-loading --local-files-only` and point `--video-path` at a real local video.

Direct sparse visual-token injection remains disabled for external models unless patch/grid mapping, positional IDs, attention masks, projector compatibility, and placeholder alignment are verified. Qwen2.5-VL, InternVL3.5, VideoChat-Flash, and V-JEPA2-to-MLLM projector paths remain input-selection/zero-mask or blocked paths by default.

This external workflow is additive. It must not change A0-A3 AutoGaze ON/OFF + NVILA inference, which remains the canonical `official_processor` path documented above.

## Lightweight Tests

```bash
python -m py_compile \
  scripts/infer_autogaze.py \
  scripts/infer_full.py \
  scripts/poc_infer_utils.py \
  scripts/poc_model_adapters.py \
  scripts/poc_model_registry.py \
  scripts/prepare_external_model_assets.py \
  scripts/verify_external_model_assets.py \
  scripts/inspect_external_model_configs.py \
  scripts/run_external_model_smoke.py

python -m pytest -q tests/test_poc_inference_visualizer_priority1.py
```

These tests use dummy data and fake model factories. They do not download checkpoints or run heavy real inference.

## Current Limitations

- This branch is not a benchmark branch.
- Hugging Face dataset/evaluate support was removed from this cleanup branch.
- The old all-in-one `scripts/poc_nvila_hd_video.py` path was removed from this branch.
- NVILA and Qwen direct visual-token injection remain unsupported.
- Encoder-side acceleration is only valid when real AutoGaze output is passed as `gazing_info` to a compatible modified ViT. Processor-first MLLM generation should not be described as encoder-side acceleration.
- Dummy runs are for smoke testing and visualization only.
