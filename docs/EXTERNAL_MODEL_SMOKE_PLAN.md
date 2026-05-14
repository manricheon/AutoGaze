# External Model Smoke Plan

This plan moves external MLLM and V-JEPA2 work from analysis to staged asset verification and minimal smoke tests. It does not claim paper reproduction, direct visual-token injection, encoder-side sparse acceleration, or compatibility through trainable/random adapters.

Canonical A0-A3 AutoGaze ON/OFF + NVILA inference remains the protected baseline. External smoke configs and asset scripts are additive and do not change the default NVILA `official_processor` path.

## Asset Availability

| Model | Local target | Expected current status | First smoke config | First mode |
|---|---|---|---|---|
| Qwen2.5-VL-7B | `weights/Qwen2.5-VL-7B-Instruct` | verified local asset | `configs/poc_inference/external/smoke_qwen2_5_vl.yaml` | `autogaze_frame_selection` |
| V-JEPA2 | `weights/vjepa2-vitl-fpc64-256` | verified local asset | `configs/poc_inference/external/smoke_vjepa2.yaml` | `vjepa2_feature_extraction` |
| LongVA-7B | `weights/longva` | missing until downloaded | `configs/poc_inference/external/smoke_longva.yaml` | `autogaze_frame_selection` |
| LongVILA-R1-7B | `weights/longvila_r1` | missing until downloaded | `configs/poc_inference/external/smoke_longvila_r1.yaml` | `autogaze_frame_selection` |
| LLaVA-OV / OneVision | `weights/llava_ov` | missing until downloaded | `configs/poc_inference/external/smoke_llava_ov.yaml` | `autogaze_frame_selection` |
| VideoLLaMA3-7B | `weights/videollama3` | missing until downloaded | `configs/poc_inference/external/smoke_videollama3.yaml` | `autogaze_frame_selection` |
| Apollo-7B | `weights/apollo` | missing until downloaded | `configs/poc_inference/external/smoke_apollo.yaml` | `autogaze_chop_selection` |
| InternVL3.5-8B | `weights/internvl3_5` | missing until downloaded | `configs/poc_inference/external/smoke_internvl3_5.yaml` | `autogaze_chop_selection` |
| VideoChat-Flash | `weights/videochat_flash` | missing until downloaded | `configs/poc_inference/external/smoke_videochat_flash.yaml` | `autogaze_frame_selection` |

## Recommended Execution Order

1. LongVILA-R1 if downloaded or already local.
2. LongVA if downloaded or already local.
3. LLaVA-OV if downloaded or already local.
4. VideoLLaMA3 if downloaded or already local.
5. Apollo if downloaded or already local.
6. V-JEPA2 if `weights/vjepa2-vitl-fpc64-256` verifies locally.
7. Qwen2.5-VL only as a Tier 2 fallback if `weights/Qwen2.5-VL-7B-Instruct` verifies locally or Tier 1 is blocked.
8. InternVL3.5 if explicitly requested.
9. VideoChat-Flash if explicitly requested.

This order can change after `docs/MODEL_ASSET_VERIFY_REPORT.md` is generated. Tier 1 remains the first direct/sparse investigation priority; already-local Tier 2 models are fallback input-selection/zero-mask targets, not first direct sparse targets.

## Dry-Run Commands

Asset planning:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

Local verification:

```bash
python scripts/verify_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_ASSET_VERIFY_REPORT.md
```

Config inspection:

```bash
python scripts/inspect_external_model_configs.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_CONFIG_INSPECTION_REPORT.md
```

Qwen dry-run smoke:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/smoke_qwen2_5_vl.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/smoke_qwen2_5_vl_dry \
  --dry-run \
  --local-files-only
```

V-JEPA2 dry-run smoke:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/smoke_vjepa2.yaml \
  --video-path dummy \
  --output-dir outputs/poc_inference/external/smoke_vjepa2_dry \
  --dry-run \
  --local-files-only
```

Real smoke requires explicit model loading:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/smoke_qwen2_5_vl.yaml \
  --video-path /path/to/local_video.mp4 \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/smoke_qwen2_5_vl_real \
  --allow-real-model-loading \
  --local-files-only \
  --max-new-tokens 16 \
  --device cuda \
  --dtype bfloat16
```

Dummy-weight smoke is available only as an explicit plumbing check. It assumes the configured local weight directory is the intended target, but does not load real checkpoint shards and does not claim real inference:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_tier1_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/selected_tier1_dummy \
  --allow-dummy-weights \
  --local-files-only \
  --max-new-tokens 8
```

Dummy outputs are marked with `status: dummy`, `dummy_weights: true`, and `real_checkpoint_loaded: false`.

## Expected Blockers

| Blocker | Scope | Handling |
|---|---|---|
| Missing local checkpoint | All non-local models | Report `blocked_missing_assets`; do not fall back to NVILA or modified SigLIP. |
| Gated/private model access | Any model card requiring approval | Require `--allow-gated` and a token env var such as `HF_TOKEN`; never log token value. |
| Storage limits | Large 7B/8B checkpoints | Use `--model <name>` first and `--max-total-gb` for all-model planning. |
| Unsupported direct sparse token path | Qwen, InternVL, VideoChat, V-JEPA2 to MLLM | Keep direct injection disabled until position IDs, masks, projector, and placeholders are verified. |
| Optional remote-code processors | Several external MLLMs | Keep `allow_real_model_loading: false` by default and require explicit real smoke command. |

## Status Semantics

| Status | Meaning |
|---|---|
| theoretical compatibility | Based on public architecture/docs only. |
| config-inspected compatibility | Local `config.json` or processor config was parsed without loading weights. |
| local asset verified | Required config, processor/tokenizer, and weight files exist locally. |
| real model loaded | Adapter loaded a real checkpoint through its explicit path. |
| real inference passed | Minimal smoke produced an answer or feature summary. |
| blocked by missing checkpoint | Manifest entry is known, but local assets are absent. |
| blocked by architecture | Direct sparse/token injection violates positional, projector, or placeholder criteria. |
| blocked by trainable adapter requirement | A new projector/connector would need training, which is outside this branch. |
