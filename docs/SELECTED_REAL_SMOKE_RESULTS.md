# Selected Real Smoke Results

## Summary

| Target | Status | Result |
|---|---|---|
| LongVILA-R1 Tier 1 MLLM | `BLOCKED` for real loading; `PARTIAL` for dummy smoke | Local folder exists, but LLM shards are incomplete. Dummy-weight smoke validates routing, zero-mask metadata, and output files only. |
| V-JEPA2 feature extraction | `PASS` | Local model loaded and produced dense feature tensors. |

No model was downloaded in this run. LongVILA-R1 download was not attempted because dry-run left storage size unknown, local verification failed, and the workspace had about `1.2GiB` free while missing shards are multi-GB files.

## Selected Tier 1 Model

| Field | Value |
|---|---|
| Model | LongVILA-R1 / `longvila_r1` |
| Integration mode | `autogaze_zero_mask` |
| Config | `configs/poc_inference/external/selected_tier1_smoke.yaml` |
| Checkpoint path | `weights/longvila_r1` |
| Verification | `BLOCKED` |
| Config inspection | `PASS`: `model_type=vila`, architecture `VILAForCausalLM`, patch size `14`, image size `448`, hidden size `3584`, projector `mlp_downsample_2x2_fix` |
| Real smoke | `NOT TESTED`: verification did not pass |
| Dummy smoke | `PARTIAL` |
| Output directory | `outputs/poc_inference/external/selected_tier1_dummy_smoke` |
| Generated answer | `[dummy:longvila_r1] placeholder response for: Describe the main action` |

LongVILA-R1 local blockers:

- Missing `weights/longvila_r1/llm/model-00001-of-00004.safetensors`
- Missing `weights/longvila_r1/llm/model-00002-of-00004.safetensors`
- Missing top-level expected processor/tokenizer files according to the manifest
- Workspace free space was about `1.2GiB`

Dummy smoke command:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_tier1_smoke.yaml \
  --video-path dummy \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/selected_tier1_dummy_smoke \
  --allow-dummy-weights \
  --local-files-only \
  --max-new-tokens 8 \
  --device cpu \
  --dtype float32 \
  --warmup-runs 0 \
  --no-progress
```

Dummy smoke validation:

- `logs/poc_summary.json`: present
- `logs/metrics.json`: present
- `predictions/answer.json`: present and labeled `status: dummy`
- `autogaze/frame_selection_metadata.json`: present
- `autogaze/token_counts_summary.json`: present
- `autogaze/zero_mask_metadata.json`: present
- `zero_mask_encoder_compute_reduction`: `false`
- `direct_visual_token_injection`: `false`
- `real_checkpoint_loaded`: `false`
- No fallback to NVILA or modified SigLIP occurred

## Selected V-JEPA2 Target

| Field | Value |
|---|---|
| Model | V-JEPA2 / `vjepa2` |
| Mode | `vjepa2_feature_extraction` |
| Config | `configs/poc_inference/external/selected_vjepa2_smoke.yaml` |
| Checkpoint path | `weights/vjepa2-vitl-fpc64-256` |
| Verification | `PASS` |
| Config inspection | `PASS`: `model_type=vjepa2`, patch size `16`, crop/image size `256`, frames per clip `64`, tubelet size `2`, hidden size `1024` |
| Real smoke | `PASS` |
| Output directory | `outputs/poc_inference/external/selected_vjepa2_real_smoke` |
| Feature shape | `[1, 512, 1024]` |
| Pooled feature shape | `[1, 1024]` |
| Pooling method | `mean_over_visual_tokens` |
| Vision latency | `669.55 ms` |
| Wall-clock latency | `2814.41 ms` |
| Memory | unavailable in this CPU run |

Real smoke command:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_vjepa2_smoke.yaml \
  --video-path assets/example_input.mp4 \
  --output-dir outputs/poc_inference/external/selected_vjepa2_real_smoke \
  --allow-real-model-loading \
  --local-files-only \
  --device cpu \
  --dtype float32 \
  --num-frames 4 \
  --warmup-runs 0 \
  --no-progress
```

V-JEPA2 output validation:

- `logs/poc_summary.json`: present
- `logs/metrics.json`: present
- `features/vjepa2_feature_summary.json`: present
- `predictions/answer.json`: present and labeled MLLM generation skipped
- `autogaze/frame_selection_metadata.json`: present
- `mllm_projector_status`: `blocked_without_verified_frozen_projector`
- `zero_mask_encoder_compute_reduction`: `false`
- `direct visual token injection`: not used
- No fallback to NVILA or modified SigLIP occurred

Implementation note: the local V-JEPA2 folder has `video_preprocessor_config.json`, but `AutoProcessor` cannot instantiate a processor class from it. The real smoke therefore uses the already prepared `[B,T,C,H,W]` tensor path and records `processor_status=unavailable_tensor_input_allowed`. The model itself loads from the local checkpoint and runs `pixel_values_videos` with `skip_predictor=True`.

## Next Recommended Smoke Target

1. Free at least `15-20GB`, complete LongVILA-R1 shards, then rerun selected Tier 1 real smoke.
2. After LongVILA real official-processor smoke passes, inspect VILA media/projector/placeholder code before attempting any sparse patch mode.
3. For the V-JEPA2 branch, next run `autogaze_frame_selection_vjepa2` or `autogaze_zero_mask_vjepa2`; do not connect V-JEPA2 to MLLMs without a verified frozen projector.

The canonical A0-A3 NVILA inference path was not changed. No original AutoGaze source files, `INTEGRATION.md`, or `QUICK_START.md` were modified.
