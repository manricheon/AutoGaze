# External Model Prioritized Plan

This plan prioritizes external AutoGaze integration while protecting the canonical NVILA path. It separates theoretical compatibility, asset availability, config inspection, real loading, and real inference.

## Tier Policy

| Tier | Models | Policy |
|---|---|---|
| Tier 0 | A0, A1, A2, A3 canonical NVILA configs | Regression-protected. Do not change defaults or redirect to external models. |
| Tier 1 | `longvila_r1`, `longva`, `llava_ov`, `videollama3`, `apollo` | Highest-priority external MLLMs because VILA/LLaVA/SigLIP-style paths are most likely to support meaningful AutoGaze adaptation. |
| Tier 1-B | `vjepa2` | Separate video encoder/decoder branch; not a drop-in SigLIP replacement for existing MLLMs. |
| Tier 2 | `qwen2_5_vl`, `internvl3_5`, `videochat_flash` | Input-selection-first. Do not download by default unless already local or explicitly requested. |

## Selected Targets

| Slot | Selection | Reason | Status |
|---|---|---|---|
| Selected Tier 1 MLLM | LongVILA-R1 / `longvila_r1` | Closest VILA/SigLIP-family candidate to the NVILA-style path. Native/light sparse integration is plausible enough to inspect after assets are present. | `BLOCKED` for real loading because LLM shards are incomplete; explicit dummy-weight smoke is `PARTIAL` |
| Selected V-JEPA2 target | V-JEPA2 feature extraction + temporal pooling probe | Local V-JEPA2 assets are verified; feature extraction is the safest no-training decoder branch. | `PASS` for local dense feature extraction smoke |

`configs/poc_inference/external/selected_tier1_smoke.yaml` points to LongVILA-R1 and uses `autogaze_zero_mask` as the first safe fallback mode. `configs/poc_inference/external/selected_vjepa2_smoke.yaml` points to V-JEPA2 feature extraction and keeps MLLM projection blocked.

## Mode Priority Per Selected Target

| Target | First mode | Next mode | Sparse mode status | Direct injection |
|---|---|---|---|---|
| LongVILA-R1 | `autogaze_zero_mask` | `autogaze_frame_selection`, then official processor | `native_sparse_patch` / `light_modified_sparse` need source inspection of VILA SigLIP, TSP pooling, projector, and placeholders | disabled |
| V-JEPA2 | `vjepa2_feature_extraction` | `autogaze_frame_selection_vjepa2`, `autogaze_zero_mask_vjepa2` | `vjepa2_sparse_tubelet` needs source inspection of patchify and 3D-RoPE positions | blocked; not an MLLM visual-token path |

## RoPE Sparse Conclusion

RoPE does not make sparse AutoGaze impossible, but it is not enough by itself. `rope_sparse_patch` is allowed only when the model exposes or can deterministically construct explicit sparse spatial/temporal position IDs, accepts arbitrary sparse token ordering or exact window reconstruction, avoids required dense-grid reshapes before sparse injection, and can align projector plus visual placeholders without training.

Current classification:

| Model | RoPE sparse status | Best fallback |
|---|---|---|
| Qwen2.5-VL | `blocked_architecture`: M-RoPE, `grid_thw`, window attention, visual merger, and processor-owned placeholders | official processor, `autogaze_frame_selection`, `autogaze_chop_selection`, `autogaze_zero_mask` |
| VideoLLaMA3 | `needs_source_inspection`: SigLIP-NaViT/Qwen2.5 path may use RoPE after compression | `autogaze_frame_selection`, `autogaze_zero_mask` |
| V-JEPA2 | `needs_source_inspection`: 3D-RoPE tubelet positions are visible in concept, but sparse tubelet API is not verified | `vjepa2_feature_extraction`, `autogaze_frame_selection_vjepa2`, `autogaze_zero_mask_vjepa2` |
| LLaVA / LongVILA variants | LLM RoPE does not imply vision sparse support; vision path is SigLIP-style absolute/interpolated unless source proves otherwise | official processor, zero-mask, input selection |

When `--integration-mode rope_sparse_patch` is explicitly used, the pipeline writes `autogaze/rope_sparse_mapping_metadata.json` with selected patch coordinates, deterministic token ordering, position IDs, dense-grid/window flags, and support status. It does not run by default.

## Zero-Mask Status

`autogaze_zero_mask` is implemented as a dense-layout compatibility probe:

| Stage | Status | Encoder compute reduction |
|---|---|---|
| `pixel` | implemented for processed `[B,T,C,H,W]` tensors by masking unselected AutoGaze boxes in image space | false |
| `patch_embedding` | metadata-only stub until model-specific patch hooks are verified | false |
| `post_encoder` | metadata-only stub until model-specific encoder outputs are mapped | false |

Metrics explicitly report `zero_mask_encoder_compute_reduction=false` and `zero_mask_expected_speedup=none` unless a future model-specific path proves otherwise.

## Selected Asset Commands

LongVILA-R1 selected dry-run:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model longvila_r1 \
  --dry-run \
  --output-root weights \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

Verification and config inspection:

```bash
python scripts/verify_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model longvila_r1 \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_ASSET_VERIFY_REPORT.md

python scripts/inspect_external_model_configs.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model longvila_r1 \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_CONFIG_INSPECTION_REPORT.md
```

Selected dry-run smoke commands:

```bash
python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_tier1_smoke.yaml \
  --video-path /path/to/local_video.mp4 \
  --query-text "Describe the main action in the video." \
  --output-dir outputs/poc_inference/external/selected_tier1_dry_run \
  --local-files-only \
  --dry-run

python scripts/run_external_model_smoke.py \
  --config configs/poc_inference/external/selected_vjepa2_smoke.yaml \
  --video-path /path/to/local_video.mp4 \
  --output-dir outputs/poc_inference/external/selected_vjepa2_dry_run \
  --local-files-only \
  --dry-run
```

No model is downloaded by these commands.

## Deferred Models

LongVA, LLaVA-OV, VideoLLaMA3, Apollo, InternVL3.5, and VideoChat-Flash remain deferred until assets are explicitly prepared. Qwen2.5-VL is already local but remains Tier 2 because direct sparse/token injection is blocked by architecture and it should be used as an input-selection/zero-mask fallback, not as the first direct sparse target.
