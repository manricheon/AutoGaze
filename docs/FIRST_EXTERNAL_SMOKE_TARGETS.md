# First External Smoke Targets

This document records the first safe smoke-test targets after the external MLLM and V-JEPA2 compatibility audit. These are smoke scaffolds, not benchmark or paper reproduction runs. They do not use direct visual-token injection, trainable adapters, random Linear projection layers, or fallback to NVILA / modified SigLIP.

## Audit Status

| Item | Status |
|---|---|
| `docs/generic_vit_autogaze_compatibility.md` | Complete enough for smoke selection; includes generic ViT portability, non-training policy, positional risks, and V-JEPA2 implications. |
| `docs/mllm_adapt_report.md` | Complete enough for smoke selection; includes Table-model MLLM review, V-JEPA2 positional/tubelet analysis, decoder recommendations, and V-JEPA2 + MLLM compatibility table. |
| Non-training policy | Present: no trainable adapter, no random Linear projection, no unsupported direct token injection. |
| Canonical NVILA configs | Preserved as A0/A1/A2/A3; smoke scaffolds are additive external configs. |

## Selected Table MLLM Target

| Field | Selection |
|---|---|
| Model | Qwen2.5-VL-7B |
| Adapter key | `qwen2_5_vl` |
| Reason | Local checkpoint exists in `weights/Qwen2.5-VL-7B-Instruct`; official processor path is already represented by the adapter; no direct token injection is required. Higher-preference LongVILA-R1 is still the best future sparse candidate, but no local LongVILA checkpoint is present here. |
| Required model/checkpoint | `weights/Qwen2.5-VL-7B-Instruct` |
| Required processor/tokenizer | `weights/Qwen2.5-VL-7B-Instruct` |
| Selected integration mode | `autogaze_frame_selection` |
| AutoGaze use | Input-level frame/window selection metadata and visualization only; Qwen still uses its official processor path. |
| Direct token injection | Disabled |
| Expected blockers | Large local 7B load, device memory, `qwen_vl_utils` availability for source-video preprocessing, and Qwen M-RoPE/grid requirements if anyone tries sparse token injection. |
| Config | `configs/poc_inference/external/first_external_mllm_smoke.yaml` |
| Status | `stub-only` by default; real loading requires explicit `--allow-real-model-loading --local-files-only`. |
| Fallback target if blocked | LLaVA-OV official-processor input selection if a local checkpoint is supplied, or LongVILA-R1 official-processor input selection when a local LongVILA checkpoint is available. |

### Table MLLM Smoke Command

Stub-only route check:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/external/first_external_mllm_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --integration-mode autogaze_frame_selection \
  --output-dir outputs/poc_inference/external/first_external_mllm_smoke_stub \
  --no-progress
```

Real local smoke attempt:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/external/first_external_mllm_smoke.yaml \
  --video-path /path/to/local_video.mp4 \
  --query-text "Describe the main action in the video." \
  --integration-mode autogaze_frame_selection \
  --allow-real-model-loading \
  --local-files-only \
  --output-dir outputs/poc_inference/external/first_external_mllm_smoke_real
```

The real command is not marked benchmark-runnable in this branch because it was not executed here. It is a prepared local smoke command that may still be blocked by memory or optional processor utilities.

## Selected V-JEPA2 Target

| Field | Selection |
|---|---|
| Model | V-JEPA2 ViT-L FPC64 256 |
| Adapter key | `vjepa2` |
| Reason | Local base checkpoint exists in `weights/vjepa2-vitl-fpc64-256`; feature extraction is safer than classification because no trained local classification head is verified. |
| Required model/checkpoint | `weights/vjepa2-vitl-fpc64-256` |
| Required processor | `weights/vjepa2-vitl-fpc64-256` |
| Selected V-JEPA2 mode | `vjepa2_feature_extraction` |
| Decoder type | `temporal_pooling_feature_probe_stub` |
| AutoGaze use | Disabled for first dense feature smoke; follow-up can use `autogaze_frame_selection_vjepa2` or `autogaze_zero_mask_vjepa2`. |
| Direct MLLM projection | Blocked without a verified frozen projector |
| Expected blockers | Transformers V-JEPA2 support/version, device memory, and no trained action-recognition head in the local base checkpoint. |
| Config | `configs/poc_inference/external/first_vjepa2_smoke.yaml` |
| Status | `stub-only` by default; real loading requires explicit `--allow-real-model-loading --local-files-only`. |
| Fallback target if blocked | `vjepa2_official_dense` status-only run, or VJEPA2ForVideoClassification if a trained classification checkpoint is supplied. |

### V-JEPA2 Smoke Command

Stub-only route check through `infer_full.py`:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/external/first_vjepa2_smoke.yaml \
  --video-path dummy \
  --query-text "Extract visual features for this clip." \
  --output-dir outputs/poc_inference/external/first_vjepa2_smoke_stub \
  --no-progress
```

Real local V-JEPA2 feature-extraction attempt:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/external/first_vjepa2_smoke.yaml \
  --video-path /path/to/local_video.mp4 \
  --query-text "Extract visual features for this clip." \
  --allow-real-model-loading \
  --local-files-only \
  --output-dir outputs/poc_inference/external/first_vjepa2_smoke_real
```

This command routes through `VJEPA2Adapter` and keeps `generic_mllm` as a stub. It is for validating V-JEPA2 loading/feature extraction plumbing, not VQA generation.

## Blocked Commands

Direct token injection remains blocked:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/external/first_external_mllm_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the video." \
  --integration-mode direct_visual_token_injection \
  --output-dir outputs/poc_inference/external/blocked_direct_injection \
  --no-progress
```

V-JEPA2 sparse tubelet remains blocked until source-level sparse 3D-RoPE/tubelet support is verified.

## Staged Asset Workflow

The first smoke targets are now backed by the staged asset workflow:

| Artifact | Purpose |
|---|---|
| `configs/poc_inference/model_asset_manifest.yaml` | Canonical model ID, local path, expected files, adapter, and recommended mode for each candidate. |
| `docs/MODEL_ASSET_MANIFEST.md` | Human-readable manifest summary and dry-run/download policy. |
| `scripts/prepare_external_model_assets.py` | Dry-run-first Hugging Face asset resolver/downloader. No download occurs without `--download`. |
| `scripts/verify_external_model_assets.py` | Local file and adapter-route verification without loading weights. |
| `scripts/inspect_external_model_configs.py` | Config-only architecture inspection without loading weights. |
| `scripts/run_external_model_smoke.py` | Minimal smoke wrapper that dry-runs by default and only loads real models with `--allow-real-model-loading`. |
| `docs/EXTERNAL_MODEL_SMOKE_PLAN.md` | Execution order, storage/gating concerns, and command templates. |

The current local-first order remains Qwen2.5-VL, then V-JEPA2. LongVA, LongVILA-R1, LLaVA-OV, VideoLLaMA3, Apollo, InternVL3.5, and VideoChat-Flash stay blocked until local assets are verified. Direct visual-token injection remains disabled for all external targets.
