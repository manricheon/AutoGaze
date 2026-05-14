# Inference Guide

The maintained PoC inference guide for this cleaned branch is:

```text
docs/inference_guide_for_poc.md
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

The existing A0-A3 AutoGaze ON/OFF + NVILA inference configs remain the canonical path and keep their previous `official_processor` behavior.

## Streaming Video Inference

`scripts/infer_autogaze.py` and `scripts/infer_full.py` now default to streaming video input through `video_input.read_mode: streaming`. The full-video loader is still available with `--video-read-mode full`, but real file inputs are blocked by default when `memory.fail_on_full_video_load: true`.

The streaming path follows the QUICK_START.md guidance for long videos by processing bounded frame windows instead of stacking the whole video. Each window is decoded, scaled or chopped, sent through AutoGaze and the selected downstream path, written to disk, then released before the next window.

Supported frame modes:

- `chunk`: sequential fixed-size windows; best default for long videos.
- `all`: implemented as chunked full-video streaming, not full tensor loading.
- `interval`: keeps frames matching `frame_interval` into bounded windows.
- `sample`: uses frame-count metadata to choose target indices; if metadata is unavailable it fails clearly instead of silently full-loading the video.

Full pipeline streaming uses `window_independent_generation` by default. It writes one answer record per window to `predictions/window_answers.json` and a combined list-style `predictions/answer.json`. `aggregate_window_answers` is intentionally not implemented yet. `first_window_only` is available for quick checks, and `blocked_multi_window_generation` allows streaming AutoGaze while skipping MLLM generation.

Visualization in streaming mode writes frame images immediately and uses incremental MP4 writers for `sampled_only` exports. `full_length` video export is blocked until it can be implemented without storing all frames.

Memory safety options:

```yaml
video_input:
  read_mode: streaming
  decode_backend: auto
  decode_fps: null
  max_decode_frames: null
  resize_before_buffer: true

streaming:
  enabled: true
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
