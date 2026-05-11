# QUICK_START Reference

이 문서는 원본 `QUICK_START.md`에서 inference 관련 정보를 추출한 참조 문서입니다. 원본 `QUICK_START.md`는 수정하지 않습니다.

## Source

| Item | Value |
|---|---|
| Located file | `QUICK_START.md` |
| Source type | repository root |
| Usage style | Python API examples |
| Shell CLI commands | Not present |

## Official Inference Examples

원본 quick start에는 독립 실행형 shell command가 아니라 Python code snippet 형태의 예제가 있습니다.

| Example | Present? | Notes |
|---|---:|---|
| AutoGaze-only model call | Yes | `AutoGaze.from_pretrained("bfshi/AutoGaze")` 후 `autogaze_model({"video": ...}, ...)` 호출 |
| AutoGaze + modified SigLIP | Yes | `SiglipVisionModel(..., scales=..., attn_implementation="sdpa")` 후 `gazing_info=gaze_outputs` 전달 |
| full NVILA pipeline command | No | NVILA 또는 query-text MLLM command는 quick start에 없음 |
| AutoGaze streaming inference | Yes | one-frame-at-a-time + cache example |

## AutoGaze-Only Behavior

Official model/preprocessor loading:

```python
AutoGazeImageProcessor.from_pretrained("bfshi/AutoGaze")
AutoGaze.from_pretrained("bfshi/AutoGaze")
```

Official video loading/preprocessing assumptions:

- `av.open(video_path)` 사용
- `read_video_pyav(container=container, indices=sample_indices)` 사용
- `transform_video_for_pytorch(raw_video, autogaze_transform)` 사용
- tensor shape은 sample 단위에서 `T * C * H * W`, batch 후 `B * T * C * H * W`

Official AutoGaze call arguments:

```python
autogaze_model(
    {"video": video_input_autogaze},
    gazing_ratio=0.75,
    task_loss_requirement=0.7,
)
```

Output fields described by quick start:

| Field | Meaning |
|---|---|
| `gazing_pos` | selected/gazed patch indices, padded positions included |
| `if_padded_gazing` | boolean padded-gazing indicator |
| `num_gazing_each_frame` | number of gazing positions per frame, padding included |
| `past_key_values` | streaming cache output |
| `past_input_embeds` | streaming cache output |
| `past_attention_mask` | streaming cache output |
| `past_conv_values` | streaming cache output |

Important index convention:

- Static video mode counts patch ids from the first patch of the whole video.
- Streaming mode counts patch ids from the first patch within the same frame and must be recalibrated for static comparison.

## Modified SigLIP Integration

Official import:

```python
from autogaze.vision_encoders.siglip import SiglipVisionModel
```

Official baseline SigLIP loading:

```python
AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
SiglipVisionModel.from_pretrained(
    "google/siglip2-base-patch16-224",
    scales=autogaze_model.config.scales,
    attn_implementation="sdpa",
)
```

Official integration argument:

```python
siglip_model(video_input_siglip, gazing_info=gaze_outputs)
```

Padding rule:

- `last_hidden_state` includes dummy features at padded gazing positions.
- Downstream code should filter with `~gaze_outputs["if_padded_gazing"]`.

## Checkpoint and Model Paths

| Component | Official path / model ID | Notes |
|---|---|---|
| AutoGaze | `bfshi/AutoGaze` | Hugging Face `from_pretrained` path |
| AutoGaze image processor | `bfshi/AutoGaze` | Hugging Face `from_pretrained` path |
| SigLIP base | `google/siglip2-base-patch16-224` | 224 resolution, 16 patch size |
| SigLIP SO400M example | `google/siglip2-so400m-patch14-384` | quick start resizes AutoGaze/SigLIP input to 392 for patch divisibility |
| NVILA | Not described | No full NVILA command in quick start |

Local checkpoint paths are not specified in quick start.

## Required Environment Variables

No required environment variables are described in quick start.

## Required CLI Arguments

No shell CLI is described in quick start. The relevant Python API arguments are:

| Argument | Default/example | Scope |
|---|---|---|
| `gazing_ratio` | `0.75` | AutoGaze call |
| `task_loss_requirement` | `0.7` | AutoGaze call |
| `target_scales` | `[56, 112, 196, 392]` in high-res example | AutoGaze call for target encoder grid |
| `target_patch_size` | `14` in high-res example | AutoGaze call for target encoder patch size |
| `gazing_info` | `gaze_outputs` | modified SigLIP call |
| `generate_only` | `True` in streaming example | AutoGaze streaming call |
| `use_cache` | `True` in streaming example | AutoGaze streaming call |

## Query Text / Prompt Behavior

Query text or prompt arguments are not present in quick start. Any PoC full-pipeline command accepting `--query-text` is therefore an extension path and must be labeled stub-only or future work until implemented.

## Resolution and Scaling

Default training/input assumption:

| Item | Value |
|---|---:|
| default input resolution | `224x224` |
| default frame count | `16` |
| default patch size | `16x16` |

High-resolution / patch-size adaptation:

- Quick start says AutoGaze predicts patch ids assuming `224x224` resolution and `16x16` patch size.
- For different encoder resolution or patch size, pass `target_scales` and `target_patch_size` to AutoGaze.
- Example for SigLIP2-SO400M:
  - resize to `392x392` because `384` is not divisible by `14`
  - use target scales `[56, 112, 196, 392]`
  - use `target_patch_size=14`
  - load SigLIP with `scales="56+112+196+392"`

Any-duration / any-resolution behavior:

- Chunk long/high-resolution video into `16`-frame spatial chunks.
- Example shape: `1 x 256 x 3 x 1344 x 1344`
- Example chunking pattern: `16` frames and `224x224` spatial chunks.

## Video Input Format Assumptions

- Video is decoded with PyAV.
- Input video is transformed to PyTorch tensor layout.
- AutoGaze and SigLIP examples use `[B, T, C, H, W]` after adding batch dimension.

## Output Path Conventions

No filesystem output path convention is described in quick start. PoC output paths under `outputs/<exp_name>/...` are extension conventions.

## Visualization Options

Visualization options are not described in quick start.

## AutoGaze ON/OFF and Token Budget Options

Quick start describes AutoGaze ON behavior only.

Selection budget controls:

| Control | Meaning |
|---|---|
| `gazing_ratio` | maximum percentage of patches AutoGaze can gaze per frame |
| `task_loss_requirement` | reconstruction-loss threshold for stopping gazing per frame |

AutoGaze OFF is a PoC/canonical-ablation config convention, not a quick start command.

## Device Assumptions

Quick start examples use PyTorch and `torch.inference_mode()`, but do not specify CUDA, MPS, or CPU requirements.

SigLIP example uses:

```python
attn_implementation="sdpa"
```

## Differences From Current PoC

| Area | Quick Start | Current PoC |
|---|---|---|
| command style | Python snippets | config-driven scripts and stubs |
| full NVILA pipeline | not described | guarded smoke path; NVILA loading/generation skipped by default unless explicitly allowed |
| query text | not described | supported by PoC full-pipeline smoke command as an extension; logged when generation is skipped |
| output directories | not described | `outputs/<exp_name>/...` extension convention |
| visualization | not described | AutoGaze smoke visualization is saved when selected patch outputs are available |
| AutoGaze OFF | not described | canonical ablation config |
