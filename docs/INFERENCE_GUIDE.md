# Inference Guide

The maintained PoC inference guide for this cleaned branch is:

```text
docs/inference_guide_for_poc.md
docs/POC_INFERENCE_GUIDE.md
```

This file is kept because project instructions require `docs/INFERENCE_GUIDE.md` to exist. The detailed guide is intentionally separate and focused on the remaining inference-only branch surface.

External model checkpoint preparation and smoke testing are documented in:

```text
docs/MODEL_ASSET_MANIFEST.md
docs/EXTERNAL_MODEL_SMOKE_PLAN.md
```

Use dry-run first:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

For incomplete external checkpoints such as a partially downloaded LongVILA-R1 folder, use explicit dummy-weight smoke to test routing and output files without loading real shards:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_tier1_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --allow-dummy-weights \
  --local-files-only
```

Dummy smoke is not a model-quality test. Its outputs are deliberately labeled as dummy and do not count as real inference.

Direct sparse token injection is disabled by default for external models. Use official processors, AutoGaze input-level frame/chop selection, or zero-mask probes until positional IDs, dense-grid behavior, projector compatibility, and visual placeholders are verified.

The local dummy wiring smoke for AutoGaze-selected tokens through a ViT adapter is:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E5_autogaze_tokens_modified_siglip_generic_mllm.yaml \
  --video-path dummy \
  --query-text "Describe selected visual tokens." \
  --output-dir outputs/poc_inference/E5_dummy \
  --no-progress
```

This uses metadata-derived `gazing_info` and dummy visual tokens only; it does not claim a verified real MLLM projector path.

The existing A0-A3 AutoGaze ON/OFF + NVILA inference configs remain the canonical path and keep their previous `official_processor` behavior.

NVILA-HD-Video README-style inference presets are:

```text
configs/poc_inference/nvila_hd_smoke.yaml
configs/poc_inference/nvila_hd_default.yaml
configs/poc_inference/nvila_hd_memory_safe.yaml
```

The settings audit is documented in:

```text
docs/NVILA_HD_INFERENCE_UPDATE_AUDIT.md
```

Both `scripts/infer_full.py` and `scripts/infer_autogaze.py` accept `--num-video-frames`, `--num-video-frames-thumbnail`, `--max-tiles-video`, `--gazing-ratio-tile`, `--task-loss-requirement-tile`, `--gazing-ratio-thumbnail`, `--task-loss-requirement-thumbnail`, `--max-batch-size-autogaze`, and `--max-batch-size-siglip`. CLI values override config values and are written to runtime metadata and metrics.

## HLVid Evaluation

HLVid multiple-choice evaluation is available through:

```text
configs/poc_inference/hlvid_nvila_hd_eval.yaml
scripts/evaluate_hlvid_nvila.py
```

The config mirrors the processor setup from `docs/nvila-hd-video-readme.md`: `num_video_frames=128`, `num_video_frames_thumbnail=64`, `max_tiles_video=48`, tile gazing ratio `[0.2] + [0.06] * 15`, tile loss requirement `0.6`, thumbnail gazing ratio `1`, `max_batch_size_autogaze=16`, and `max_batch_size_siglip=32`.

Dry-run with a local HLVid-style JSON/JSONL file:

```bash
python scripts/evaluate_hlvid_nvila.py \
  --config configs/poc_inference/hlvid_nvila_hd_eval.yaml \
  --dataset-path /path/to/hlvid.jsonl \
  --video-root /path/to/video/root \
  --output-dir outputs/hlvid_nvila_dry_run \
  --dry-run
```

Real NVILA-HD evaluation, using local weights only:

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

Outputs are written to `predictions/hlvid_predictions.json`, `predictions/hlvid_predictions.jsonl`, `logs/poc_summary.json`, and `logs/metrics.json`. Accuracy is exact option-letter match against the HLVid answer field. Direct visual-token injection is not used; NVILA owns the video path through its official processor.

For single-video PoC visualization with `infer_full.py`, use the bounded HLVid-safe configs. These are not canonical HLVid reproduction configs; they are OOM-resistant smoke/ablation presets.

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

`resize_then_chop` is intentionally capped by `max_chops: 4` and `max_processed_frames_per_window: 64`:

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

`infer_full.py` now also has processed-tensor memory guards: `--max-processed-frames-per-window` and `--max-processed-pixels-per-window`. These catch chop expansion before the model/processor path can OOM.

## Streaming Video Inference

`scripts/infer_autogaze.py` and `scripts/infer_full.py` default to non-streaming input through `video_input.read_mode: full`. Streaming is an explicit opt-in with `--video-read-mode streaming` or `video_input.read_mode: streaming`. Real file inputs remain blocked in full mode when `memory.fail_on_full_video_load: true`, so long/high-resolution runs should enable streaming instead of disabling the guard.

The streaming path follows the QUICK_START.md guidance for long videos by processing bounded frame windows instead of stacking the whole video. Each window is decoded, scaled or chopped, sent through AutoGaze and the selected downstream path, written to disk, then released before the next window.

Supported frame modes:

- `chunk`: sequential fixed-size windows; best default for long videos.
- `all`: implemented as chunked full-video streaming, not full tensor loading.
- `interval`: keeps frames matching `frame_interval` into bounded windows.
- `sample`: uses frame-count metadata to choose target indices; if metadata is unavailable it fails clearly instead of silently full-loading the video.

Latency metrics use the same scope across `sample`, `chunk`, `interval`, and `all`: `autogaze_latency_ms` includes input preprocessing plus the AutoGaze stage over all processed frames/crops. Use `autogaze_latency_source_frame_count`, `autogaze_latency_processed_frame_count`, `autogaze_latency_per_source_frame_ms`, and `autogaze_latency_per_processed_frame_ms` to compare interval and chop runs.

Full pipeline streaming uses `window_independent_generation` by default. It writes one answer record per window to `predictions/window_answers.json` and a combined list-style `predictions/answer.json`. `aggregate_window_answers` is intentionally not implemented yet. `first_window_only` is available for quick checks, and `blocked_multi_window_generation` allows streaming AutoGaze while skipping MLLM generation.

Visualization in streaming mode writes frame images immediately and uses incremental MP4 writers for `sampled_only` exports. `full_length` video export is blocked until it can be implemented without storing all frames.

Streaming memory safety options:

```yaml
video_input:
  read_mode: full
  decode_backend: auto
  decode_fps: null
  max_decode_frames: null
  resize_before_buffer: true

streaming:
  enabled: false
  window_size: 16
  overlap: 0
  max_windows: null
  output_mode: incremental
  full_pipeline_policy: window_independent_generation
  flush_every_window: true
  cpu_offload_between_windows: true
  empty_cache_between_windows: false
  keep_window_outputs_in_memory: false

memory:
  safe_mode: true
  max_video_frames_in_memory: null
  max_pixels_per_window: null
  fail_on_full_video_load: true
```

For long videos, switch only the input policy:

```yaml
video_input:
  read_mode: streaming

streaming:
  enabled: true
```

AutoGaze-only streaming example:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/long_video.mp4 \
  --video-read-mode streaming \
  --frame-selection-mode chunk \
  --num-frames 16 \
  --stream-window-size 16 \
  --scaling-mode resize \
  --resolution 448 \
  --gaze-ratio 0.25 \
  --output-dir outputs/a2_streaming_autogaze \
  --save-overlay-video \
  --save-side-by-side-video
```

Full pipeline streaming example:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/long_video.mp4 \
  --query-text "Describe the video." \
  --video-read-mode streaming \
  --frame-selection-mode chunk \
  --num-frames 16 \
  --stream-window-size 16 \
  --streaming-full-pipeline-policy window_independent_generation \
  --scaling-mode resize \
  --resolution 448 \
  --gaze-ratio 0.25 \
  --max-new-tokens 32 \
  --output-dir outputs/a2_streaming_full \
  --save-overlay-video
```

For high-resolution `resize_then_chop`, streaming is applied at the temporal window level first. For example, with a 1920x1080 file, `resize_before_chop_factor=0.5`, and `chop_size=448`, each 16-frame stream window is resized to 960x540, chopped into spatial crops, processed, and released before the next temporal window is decoded.
