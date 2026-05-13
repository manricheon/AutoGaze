# PoC Inference Guide

This branch is now scoped to the lightweight PoC inference surface only. It keeps the runnable inference entry points, A0-A3 and E1-E3 configs, local model adapters, flat visualizations, and metrics output. Broader benchmark, HF dataset, profiling, and all-in-one NVILA-HD experiment code were removed from this branch.

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

## Real Loading Policy

By default, scripts do not load heavy checkpoints. They run guarded smoke paths and record `stub`, `skipped`, or `blocked` adapter status.

Use `--allow-real-model-loading` to request real loading. With that flag:

- missing checkpoints block clearly;
- incomplete Qwen sharded checkpoints block before model construction;
- requested model types are not silently replaced by another model;
- direct visual-token injection into NVILA or Qwen is not claimed unless an adapter explicitly supports it;
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
  --dtype float16 \
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

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --mllm nvila \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --output-dir outputs/poc_inference/a2_nvila_real \
  --json
```

A0/A1 set AutoGaze off. A2/A3 set AutoGaze on. When `sync_autogaze_controls_from_config` is true, the NVILA processor kwargs are aligned with the config and reported in adapter metadata.

## Qwen Smoke

Qwen uses the official processor path. Direct visual-token injection is unsupported in this PoC.

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E2_qwen_mllm.yaml \
  --video-path dummy \
  --query-text "What is happening in this video?" \
  --mllm qwen \
  --model-id weights/Qwen2.5-VL-7B-Instruct \
  --processor-path weights/Qwen2.5-VL-7B-Instruct \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --output-dir outputs/poc_inference/e2_qwen_real \
  --json
```

If any shard referenced by `model.safetensors.index.json` is missing, loading blocks with the missing filenames.

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
| `chop` | writes chop metadata and keeps outputs flat over processed frames |
| `none` | no resize |

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
      frames/frame_000000_overlay.png
      scale_panels/frame_000000_scale_panel.png
      videos/autogaze_overlay.mp4
      videos/autogaze_side_by_side.mp4
      videos/autogaze_scale_panels.mp4
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

## Lightweight Tests

```bash
python -m py_compile \
  scripts/infer_autogaze.py \
  scripts/infer_full.py \
  scripts/poc_infer_utils.py \
  scripts/poc_model_adapters.py \
  scripts/poc_model_registry.py

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
