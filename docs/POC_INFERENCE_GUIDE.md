# PoC Inference Guide

This guide covers the lightweight Priority 1 inference visualizer branch. It is intentionally smaller than the benchmark framework work: the goal is to run AutoGaze-only inference, run a guarded full-pipeline path, save flat visualizations, and write token/latency/memory reports.

The implementation is isolated under `scripts/` and `configs/poc_inference/`. It does not modify the original AutoGaze core files, `INTEGRATION.md`, `QUICK_START.md`, or `docs/nvila-hd-video-readme.md`.

## Configs

Canonical presets:

| Config | AutoGaze | Vision Encoder | MLLM | Status |
|---|---:|---|---|---|
| `configs/poc_inference/A0_vanilla_siglip_nvila_off.yaml` | off | `vanilla_siglip` | `nvila` | full-token baseline |
| `configs/poc_inference/A1_modified_siglip_nvila_off.yaml` | off | `modified_siglip` | `nvila` | modified-SigLIP baseline |
| `configs/poc_inference/A2_modified_siglip_nvila_on.yaml` | on | `modified_siglip` | `nvila` | canonical AutoGaze path |
| `configs/poc_inference/A3_vanilla_siglip_nvila_on.yaml` | on | `vanilla_siglip` | `nvila` | experimental compatibility path |

Extension presets:

| Config | Purpose |
|---|---|
| `configs/poc_inference/E1_vjepa2_encoder.yaml` | V-JEPA2 video encoder path, preserving `[B,T,C,H,W]` semantics |
| `configs/poc_inference/E2_qwen_mllm.yaml` | Qwen-family MLLM path using the official processor first |
| `configs/poc_inference/E3_vjepa2_qwen.yaml` | V-JEPA2 + Qwen extension path with input-level AutoGaze selection |

Checkpoint/model path fields are configured for this workspace's local `weights/` cache when a matching checkpoint is present. Pass `--allow-real-model-loading` when you want the scripts to try real model loading. Without that flag, stages are marked `stub` or `blocked` and are not presented as real inference.

When `--allow-real-model-loading` is set, blocked adapters do not fall back to dummy visual tokens or another model type. The run records the blocked adapter under `adapter_statuses` and exits with `status=blocked` when that adapter is required for the requested path.

## Local Checkpoints

The branch now uses these local defaults when available:

| Purpose | Local path | Used by |
|---|---|---|
| AutoGaze | `weights/AutoGaze` | A2, A3, E3 AutoGaze stage |
| SigLIP2 base 224 | `weights/siglip2-base-patch16-224` | A0-A3 vision encoder configs |
| NVILA-HD-Video | `weights/NVILA-8B-HD-Video` | A0-A3 MLLM configs |
| V-JEPA2 ViT-L 256 | `weights/vjepa2-vitl-fpc64-256` | E1 and E3 vision encoder configs |
| Qwen2.5-VL-7B-Instruct | `weights/Qwen2.5-VL-7B-Instruct` | E2 and E3 MLLM configs |

The local Qwen directory is wired into E2/E3, but it must contain every shard referenced by `model.safetensors.index.json`. If any shard is missing, the adapter reports `status=blocked` with the missing filenames before attempting model construction.

Relative checkpoint paths such as `weights/AutoGaze` are resolved from the repository root, not from the caller's shell directory.

## AutoGaze-Only

Dummy smoke run:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path dummy \
  --output-dir /tmp/autogaze_a2_dummy \
  --device cpu \
  --dtype float32 \
  --frame-selection-mode sample \
  --num-frames 4 \
  --scaling-mode resize \
  --resolution 64 \
  --gaze-ratio 0.25 \
  --save-side-by-side-video
```

Real checkpoint run:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path /path/to/video.mp4 \
  --output-dir outputs/a2_autogaze_only \
  --device cuda \
  --dtype float16 \
  --allow-real-model-loading
```

This uses `weights/AutoGaze` from the config unless you override the config. If the configured checkpoint is missing, real video runs fail clearly with `status=blocked` in `logs/poc_summary.json` and `logs/metrics.json`.

## Full Pipeline

Dummy smoke run:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/A2_modified_siglip_nvila_on.yaml \
  --video-path dummy \
  --query-text "What is happening in this video?" \
  --output-dir /tmp/autogaze_full_dummy \
  --device cpu \
  --dtype float32 \
  --frame-selection-mode sample \
  --num-frames 4 \
  --scaling-mode resize \
  --resolution 64 \
  --max-new-tokens 16
```

The full script always writes `predictions/answer.json`. If MLLM generation is unavailable, the file records the skipped reason and preserves `query_text` with `query_text_used=true`.

Model override examples:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E1_vjepa2_encoder.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --vision-encoder vjepa2 \
  --mllm generic_mllm \
  --output-dir /tmp/autogaze_e1_dummy
```

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E2_qwen_mllm.yaml \
  --video-path dummy \
  --query-text "What is happening in this video?" \
  --mllm qwen \
  --output-dir /tmp/autogaze_e2_dummy
```

Qwen direct visual token injection is not claimed as supported. The initial Qwen path is official-processor/input-level selection only.

NVILA local loading is configured through `weights/NVILA-8B-HD-Video`, but real NVILA generation is still blocked in this lightweight PoC until a model-specific NVILA generation adapter is implemented. The script reports that adapter blocker instead of silently substituting another MLLM.

## Priority 2 Real Smoke Commands

These commands are intended for local environments where the listed checkpoints or Hugging Face cache entries already exist. Add `--local-files-only` to avoid network access.

V-JEPA2 encoder smoke:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E1_vjepa2_encoder.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --vision-encoder vjepa2 \
  --vision-encoder-ckpt weights/vjepa2-vitl-fpc64-256 \
  --vision-encoder-module transformers \
  --vision-encoder-class AutoModel \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --output-dir outputs/e1_vjepa2_real_smoke
```

Expected behavior:
- If the V-JEPA2 model is available locally, `logs/metrics.json` records `adapter_statuses.vision_encoder.status=real`.
- If it is unavailable, the run is blocked and the reason is recorded. It does not fall back to NVILA, SigLIP, or dummy V-JEPA2 tokens.

Qwen official processor smoke:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E2_qwen_mllm.yaml \
  --video-path dummy \
  --query-text "What is happening in this video?" \
  --mllm qwen \
  --model-id weights/Qwen2.5-VL-7B-Instruct \
  --mllm-module transformers \
  --mllm-class AutoModelForVision2Seq \
  --processor-path weights/Qwen2.5-VL-7B-Instruct \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 32 \
  --output-dir outputs/e2_qwen_real_smoke
```

Expected behavior:
- If the Qwen model and processor are available locally, generation uses the configured official processor path and `predictions/answer.json` records `query_text_used=true`.
- If the model or processor is unavailable, the run is blocked with the exact adapter failure reason.

V-JEPA2 + Qwen smoke:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/E3_vjepa2_qwen.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --vision-encoder vjepa2 \
  --vision-encoder-ckpt weights/vjepa2-vitl-fpc64-256 \
  --mllm qwen \
  --model-id weights/Qwen2.5-VL-7B-Instruct \
  --processor-path weights/Qwen2.5-VL-7B-Instruct \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype bfloat16 \
  --output-dir outputs/e3_vjepa2_qwen_real_smoke
```

This path requires both requested adapters. If either adapter is unavailable, the run is blocked and no fallback model is substituted.

## Frame Selection

Both scripts support:

| Mode | Behavior |
|---|---|
| `sample` | uniform sample of `--num-frames` over the video |
| `chunk` | non-overlapping windows of `--num-frames` |
| `interval` | fixed `--frame-interval` selection |
| `all` | chunk the whole video into non-overlapping windows |

Sliding stride mode is intentionally not implemented.

## Scaling And Chop

Both scripts support:

| Mode | Behavior |
|---|---|
| `resize` | square resize to `--resolution` |
| `fit_short_side` | preserve aspect ratio, short side equals `--resolution` |
| `fit_long_side` | preserve aspect ratio, long side equals `--resolution` |
| `quickstart` | guarded 224/392 square policies from `QUICK_START.md` |
| `chop` | writes chop metadata and keeps visual outputs flat over processed frames |
| `none` | no resize |

Priority 1 writes chop metadata. Advanced chop merging and validated high-resolution aggregation remain Priority 2.

## Output Structure

Outputs are flat by processed frame:

```text
outputs/<run>/
  autogaze/
    frame_selection_metadata.json
    runtime_metadata.json
    token_counts_summary.json
    selected_patch_indices.json
    selected_scales.json
    per_frame_token_counts.json
  scaling/
    scaling_metadata.json
  chops/
    chop_metadata.json
  visualizations/
    autogaze/
      frames/
        frame_000000_overlay.png
      scale_panels/
        frame_000000_scale_panel.png
      videos/
        autogaze_overlay.mp4
        autogaze_side_by_side.mp4
        autogaze_scale_panels.mp4
      metadata/
        visualization_metadata.json
  predictions/
    answer.json
  logs/
    poc_summary.json
    metrics.json
    metrics.csv
```

The scripts do not write `visualizations/autogaze/windows/window_*/frames/`.

## Metrics

Every run writes:

- `logs/poc_summary.json`
- `logs/metrics.json`
- `logs/metrics.csv`

Metrics include mode, config path, video path, query text when applicable, frame selection, scaling/chop settings, AutoGaze status, requested/actual model names, token counts, token reduction ratio, latency, memory availability, skipped stages, and failure reason.

Full-pipeline metrics also include:

- `adapter_statuses.vision_encoder`
- `adapter_statuses.mllm`
- `vision_encoder_required_for_full_pipeline`

Each adapter status contains `name`, `status`, `reason`, and `metadata`. Valid statuses are `real`, `stub`, `skipped`, and `blocked`.

Encoder-side acceleration is only marked when real AutoGaze execution is used with an AutoGaze-enabled config.

## Known Stubs And Blockers

- V-JEPA2 real loading uses the configured `module_path` and `class_name`, defaulting to `transformers.AutoModel`; this workspace config points at `weights/vjepa2-vitl-fpc64-256`.
- Qwen real loading requires both model and official processor availability.
- Qwen is configured to use `weights/Qwen2.5-VL-7B-Instruct`; if any shard referenced by `model.safetensors.index.json` is missing, the adapter blocks before model construction and reports the missing filenames.
- Qwen direct visual token injection is unsupported.
- NVILA checkpoints are configured locally, but real NVILA generation needs a dedicated adapter before it can be marked supported.
- Dummy video runs generate explicit stub AutoGaze metadata. They are useful for smoke tests and visualization checks, but are not real model outputs.
