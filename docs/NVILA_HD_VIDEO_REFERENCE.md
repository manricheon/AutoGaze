# NVILA-HD-Video Reference

This file extracts implementation-relevant details from `docs/nvila-hd-video-readme.md`.
Do not edit the source guide when aligning the PoC.

## Source

| Item | Value |
|---|---|
| Source file | `docs/nvila-hd-video-readme.md` |
| Model name | `NVILA-8B-HD-Video` |
| Hugging Face model ID | `nvidia/NVILA-8B-HD-Video` |
| Model size | 8B parameters |
| Usage style | Hugging Face `AutoProcessor` + `AutoModel` Python API |

## Expected Imports and Factories

```python
import torch
from transformers import AutoModel, AutoProcessor
```

| Component | Module | Class/factory |
|---|---|---|
| Processor | `transformers` | `AutoProcessor.from_pretrained(...)` |
| Model | `transformers` | `AutoModel.from_pretrained(...)` |
| Generation | constructed model | `model.generate(**inputs)` |
| Decoding | constructed processor | `processor.batch_decode(...)` |

The guide does not use `AutoModelForCausalLM`.

## Checkpoint, Config, Processor, and Tokenizer Paths

Official Hub path:

```text
nvidia/NVILA-8B-HD-Video
```

PoC local path convention:

```text
weights/NVILA-8B-HD-Video
```

Mapped config fields:

| Field | Expected value |
|---|---|
| `checkpoint` | `weights/NVILA-8B-HD-Video` |
| `checkpoint_root` | `weights/NVILA-8B-HD-Video` |
| `config_path` | `weights/NVILA-8B-HD-Video/config.json` |
| `model_config_path` | `weights/NVILA-8B-HD-Video` |
| `processor_path` | `weights/NVILA-8B-HD-Video` |
| `tokenizer_path` | `weights/NVILA-8B-HD-Video` |
| `tokenizer_or_processor_path` | `weights/NVILA-8B-HD-Video` |

## Official Processor and Model Construction

```python
model_path = "nvidia/NVILA-8B-HD-Video"

processor = AutoProcessor.from_pretrained(
    model_path,
    num_video_frames=128,
    num_video_frames_thumbnail=64,
    max_tiles_video=48,
    gazing_ratio_tile=[0.2] + [0.06] * 15,
    gazing_ratio_thumbnail=1,
    task_loss_requirement_tile=0.6,
    task_loss_requirement_thumbnail=None,
    max_batch_size_autogaze=16,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    max_batch_size_siglip=32,
)
model.eval()
```

## Video Input Format

The official guide passes the video path directly to the processor:

```python
inputs = processor(text=f"{video_token}\n\n{prompt}", videos=video_path, return_tensors="pt")
```

The `video_path` example is a remote MP4 URL:

```text
https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4
```

The PoC smoke scripts currently support `--video dummy` and local `--video-path` preprocessing.
The isolated processor-first PoC entry point is `scripts/poc_nvila_hd_video.py`; by default it runs guarded check/stub stages and does not load heavy checkpoints.

## Prompt Format

The guide uses the processor tokenizer's video token and prepends it to the prompt:

```python
video_token = processor.tokenizer.video_token
inputs = processor(text=f"{video_token}\n\n{prompt}", videos=video_path, return_tensors="pt")
```

Example prompt:

```text
Question: What does the white text on the green road sign say?
 A. Hampden St
 B. Hampden Ave
 C. HampdenBlvd
 D. Hampden Rd
 Please answer directly with the letter of the correct answer.
```

Mapped config field:

```yaml
prompt_template: "{video_token}\n\n{prompt}"
```

## Inference Command

The source guide gives a Python API snippet, not a shell command.
The canonical behavior is:

```python
outputs = model.generate(**inputs)
response = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0].strip()
print(response)
```

## AutoGaze Behavior

The guide presents NVILA-HD-Video as using AutoGaze internally to remove redundant video patches before the vision encoder or LLM.

Processor AutoGaze controls:

| Field | Value from guide |
|---|---|
| `gazing_ratio_tile` | `[0.2] + [0.06] * 15` |
| `task_loss_requirement_tile` | `0.6` |
| `gazing_ratio_thumbnail` | `1` |
| `task_loss_requirement_thumbnail` | `None` |
| `max_batch_size_autogaze` | `16` |

The guide notes that setting `gazing_ratio_thumbnail=None` skips gazing on thumbnails.
It does not describe an AutoGaze OFF baseline. `A1_real` is therefore a PoC ablation, not an official NVILA-HD-Video guide command.

## SigLIP ViT Usage

The guide does not expose a direct SigLIP import or direct `gazing_info` call.
It configures SigLIP-related batching through:

```python
max_batch_size_siglip = 32
```

The current PoC A1/A2 lower-level path still uses the modified SigLIP reference from `QUICK_START.md`:

```python
from autogaze.vision_encoders.siglip import SiglipVisionModel
siglip_model(video_input_siglip, gazing_info=gaze_outputs)
```

This is a lower-level alignment path, while the NVILA-HD-Video guide is processor-first.

## Resolution and Frame Handling

The guide states that NVILA-HD-Video targets up to 4K resolution and 1K frames, but the quick start snippet uses conservative processor settings:

| Setting | Value |
|---|---:|
| `num_video_frames` | 128 |
| `num_video_frames_thumbnail` | 64 |
| `max_tiles_video` | 48 |
| Tile size note | one tile is `392x392` |

The PoC canonical smoke and small benchmark configs remain intentionally conservative and should not run 4K or 1K-frame settings by default.

## Output Format

The guide decodes generated token IDs after the input prompt prefix and prints a stripped response string.

```python
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
```

## Environment Assumptions and Dependencies

Required package usage from the guide:

| Dependency | Purpose |
|---|---|
| `torch` | tensor movement and inference |
| `transformers` | `AutoProcessor`, `AutoModel` |
| AutoGaze repo installation | required before using NVILA-HD-Video |

The official model construction uses:

```python
trust_remote_code=True
device_map="auto"
```

The guide does not document CUDA, MPS, or CPU-specific flags.

## Known Limitations for the Current PoC

- The current A1/A2 smoke path is not yet the official NVILA processor-driven path.
- The PoC currently skips NVILA loading/generation by default to avoid large model construction.
- `A1_real` is an AutoGaze OFF ablation and is not described in the NVILA-HD-Video guide.
- The guide does not provide a shell CLI; current scripts are PoC wrapper commands.
- The guide does not specify local checkpoint layout; the PoC maps the Hub model to `weights/NVILA-8B-HD-Video`.
- Full 4K / 1K-frame behavior is not validated in the PoC.
