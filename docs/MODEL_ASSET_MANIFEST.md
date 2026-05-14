# Model Asset Manifest

The machine-readable manifest is `configs/poc_inference/model_asset_manifest.yaml`. It is intentionally a dry-run-first manifest: no model is downloaded unless `scripts/prepare_external_model_assets.py` is run with `--download`.

This workflow is additive. It does not change the existing A0-A3 AutoGaze ON/OFF + NVILA configs or their default `official_processor` inference path.

## Local Directory Convention

External model assets live under `weights/`:

| Model key | Local target directory | Adapter | First smoke mode |
|---|---|---|---|
| `llava_ov` | `weights/llava_ov` | `LlavaOVAdapter` | `autogaze_frame_selection` |
| `longva` | `weights/longva` | `LongVAAdapter` | `autogaze_frame_selection` |
| `longvila_r1` | `weights/longvila_r1` | `LongVILAAdapter` | `autogaze_frame_selection` |
| `apollo` | `weights/apollo` | `ApolloAdapter` | `autogaze_chop_selection` |
| `videollama3` | `weights/videollama3` | `VideoLLaMA3Adapter` | `autogaze_frame_selection` |
| `videochat_flash` | `weights/videochat_flash` | `VideoChatFlashAdapter` | `autogaze_frame_selection` |
| `internvl3_5` | `weights/internvl3_5` | `InternVL35Adapter` | `autogaze_chop_selection` |
| `qwen2_5_vl` | `weights/Qwen2.5-VL-7B-Instruct` | `Qwen25VLAdapter` | `autogaze_frame_selection` |
| `vjepa2` | `weights/vjepa2-vitl-fpc64-256` | `VJEPA2Adapter` | `vjepa2_feature_extraction` |

## Initial Asset Status

| Model key | Candidate model ID | Expected local status before verification | Direct token injection |
|---|---|---|---|
| `llava_ov` | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | missing unless user downloads | disabled |
| `longva` | `lmms-lab/LongVA-7B` | missing unless user downloads | disabled |
| `longvila_r1` | `Efficient-Large-Model/LongVILA-R1-7B` | missing unless user downloads | disabled until VILA projector/placeholder compatibility is verified |
| `apollo` | `GoodiesHere/Apollo-LMMs-Apollo-7B-t32` | missing unless user downloads | disabled |
| `videollama3` | `DAMO-NLP-SG/VideoLLaMA3-7B` | missing unless user downloads | disabled until NaViT/compressor compatibility is verified |
| `videochat_flash` | `OpenGVLab/VideoChat-Flash-Qwen2-7B_res448` | missing unless user downloads | disabled |
| `internvl3_5` | `OpenGVLab/InternVL3_5-8B` | missing unless user downloads | disabled |
| `qwen2_5_vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | local folder expected in this workspace | disabled; M-RoPE/grid/window path requires official processor |
| `vjepa2` | `facebook/vjepa2-vitl-fpc64-256` | local folder expected in this workspace | not an MLLM token injection path; MLLM projector blocked without verified frozen connector |

## Workflow Commands

Dry-run download planning:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

Verify local files without loading weights:

```bash
python scripts/verify_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_ASSET_VERIFY_REPORT.md
```

Inspect configs without loading weights:

```bash
python scripts/inspect_external_model_configs.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model all \
  --weights-root weights \
  --local-files-only \
  --write-report docs/MODEL_CONFIG_INSPECTION_REPORT.md
```

Download is explicit only:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model qwen2_5_vl \
  --download \
  --allow-gated \
  --token-env-var HF_TOKEN \
  --output-root weights \
  --skip-existing
```

The token value is read only from the environment and must not be written to reports, configs, or logs.

## Tiered Selection

Tier definitions are stored in the manifest:

| Tier | Models | Download policy |
|---|---|---|
| `tier1` | `longvila_r1`, `longva`, `llava_ov`, `videollama3`, `apollo` | Highest-priority external MLLM candidates. |
| `tier1b` | `vjepa2` | Separate video encoder/decoder branch. |
| `tier2` | `qwen2_5_vl`, `internvl3_5`, `videochat_flash` | Input-selection-first; do not download by default unless explicitly selected. |

Dry-run the top Tier 1 target:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --tier tier1 \
  --select-top-k 1 \
  --dry-run \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```

Download the selected model only after explicit approval:

```bash
python scripts/prepare_external_model_assets.py \
  --manifest configs/poc_inference/model_asset_manifest.yaml \
  --model longvila_r1 \
  --download-selected \
  --output-root weights \
  --skip-existing \
  --write-report docs/MODEL_ASSET_DOWNLOAD_REPORT.md
```
