# INFERENCE_GUIDE.md

# AutoGaze PoC Inference Guide

이 문서는 AutoGaze 기반 Video Understanding PoC 파이프라인의 inference 실행 방법을 정리한다.

목표는 다음 두 가지 inference 경로를 명확히 분리하는 것이다.

1. **AutoGaze-only inference**
   - 입력 비디오를 AutoGaze까지만 통과시킨다.
   - 선택된 patch/token index, scale 정보, token count, visualization 결과를 저장한다.
   - ViT 또는 MLLM inference는 수행하지 않는다.

2. **Full pipeline inference**
   - 입력 비디오와 query text를 받아 전체 파이프라인을 실행한다.
   - 경로:
     ```text
     video + query text
     -> AutoGaze
     -> vision encoder
     -> MLLM
     -> generated answer
     ```
   - AutoGaze, ViT, MLLM 각 단계의 shape, token count, latency를 기록한다.

---

## 1. Reference Policy

실제 inference 동작은 원본 AutoGaze repository의 다음 문서를 기준으로 맞춘다.

```text
original QUICK_START.md
original INTEGRATION.md
```

역할은 다음과 같이 구분한다.

| Document | Role |
|---|---|
| `INTEGRATION.md` | architecture, integration mode, temporal/video handling |
| `QUICK_START.md` | inference command, checkpoint layout, runtime arguments, resolution scaling |

주의사항:

- 원본 `QUICK_START.md`는 수정하지 않는다.
- 원본 `INTEGRATION.md`는 수정하지 않는다.
- 이 문서의 command가 원본 `QUICK_START.md`와 다르면 차이와 이유를 명시한다.
- 아직 구현되지 않은 command는 `stub` 또는 `future work`로 표시한다.

---

## 2. Inference Modes

### 2.1 AutoGaze-Only Inference

AutoGaze-only mode는 비디오 입력을 AutoGaze selector/router까지만 통과시킨다.

목적:

- AutoGaze가 어떤 frame/patch/scale을 선택하는지 확인
- token reduction ratio 확인
- AutoGaze visualization 저장
- ViT/MLLM 없이 AutoGaze 동작만 smoke test

Conceptual pipeline:

```text
video
-> frame sampling
-> preprocessing
-> AutoGaze
-> selected patch/token metadata
-> visualization
```

Expected outputs:

```text
outputs/<exp_name>/
  autogaze/
    selected_patch_indices.json
    selected_scales.json
    token_counts.json
  visualizations/
    autogaze_only/
      frame_000.png
      frame_001.png
      ...
  logs/
    inference_summary.json
```

---

### 2.2 Full Pipeline Inference

Full pipeline mode는 AutoGaze, vision encoder, MLLM을 모두 연결한다.

Conceptual pipeline:

```text
video + query text
-> frame sampling
-> preprocessing
-> AutoGaze
-> vision encoder
-> MLLM
-> generated answer
```

Expected outputs:

```text
outputs/<exp_name>/
  predictions/
    answer.json
  autogaze/
    selected_patch_indices.json
    selected_scales.json
    token_counts.json
  visualizations/
    full_pipeline/
      frame_000.png
      frame_001.png
      ...
  logs/
    inference_summary.json
```

---

## 3. Supported Inputs

### 3.1 Dummy Video

Dummy video input is used for smoke tests.

Use this when:

- real checkpoints are not available
- real video files are not available
- only shape propagation needs to be tested
- CI/local test should run quickly

Example:

Status:

```text
stub-only / future work: this Hydra-style inference command is not present in original QUICK_START.md
and is not the supported canonical smoke inference command.
```

```bash
python -m autogaze_ext.pipeline.runner \
  experiment=A2_real \
  mode=inference \
  inference.mode=autogaze_only \
  data.source=dummy \
  data.num_frames=2 \
  data.resolution=224
```

Status:

```text
Use scripts/run_canonical_smoke_inference.py for the runnable dummy-video smoke path.
```

---

### 3.2 Local Video

Local video input is used for a small real inference smoke test.

Example:

Status:

```text
runnable smoke script template: scripts/run_canonical_smoke_inference.py exists.
Replace /path/to/video.mp4 with a real local video path.
Real model stages require local/cached checkpoints or model IDs with local_files_only support.
No equivalent shell command is present in original QUICK_START.md.
```

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video-path /path/to/video.mp4 \
  --num-frames 4 \
  --resolution 224 \
  --device cuda \
  --output-dir outputs/a2_autogaze_only
```

Expected behavior:

- decode video
- uniformly sample frames
- resize or scale frames according to config
- run AutoGaze
- save selected patch/token metadata
- save AutoGaze-only visualization

---

### 3.3 Local Video with Query Text

Full pipeline inference should support query text.
The canonical smoke script does not load the configured MLLM by default because NVILA can be large.
Use `--allow-mllm-load` only after visual feature construction is confirmed and the machine has enough memory.

Example:

Status:

```text
runnable smoke script with guarded stages: query text is accepted and logged.
Replace /path/to/video.mp4 with a real local video path.
Original QUICK_START.md does not describe query text or NVILA generation commands,
so MLLM generation may be skipped with a clear reason.
```

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --num-frames 4 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cuda \
  --output-dir outputs/a2_full_pipeline_query
```

Expected behavior:

- process video
- run AutoGaze if enabled
- run vision encoder
- pass visual features and query text to MLLM
- generate answer if MLLM generation is available
- otherwise log that generation was skipped

If generation is unavailable:

```text
The command must not silently ignore query text.
It must report that query text was accepted but MLLM generation was skipped.
```

---

## 4. Canonical Experiments

The canonical experiments are:

| ID | AutoGaze | Vision Encoder | MLLM |
|---|---:|---|---|
| A0 | OFF | vanilla SigLIP ViT | NVILA |
| A1 | OFF | modified SigLIP ViT | NVILA |
| A2 | ON | modified SigLIP ViT | NVILA |
| A3 | ON | vanilla SigLIP ViT | NVILA |

Initial real-path priority:

```text
A1_real:
AutoGaze OFF + modified SigLIP + NVILA

A2_real:
AutoGaze ON + modified SigLIP + NVILA
```

A3 is experimental and should not be assumed compatible until tested.

---

## 5. NVILA-HD-Video Canonical Path

`docs/nvila-hd-video-readme.md` is the concrete reference for the canonical AutoGaze + SigLIP ViT + NVILA-HD-Video usage path.

Official model and processor loading:

```python
from transformers import AutoModel, AutoProcessor

model_path = "nvidia/NVILA-8B-HD-Video"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, ...)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", ...)
```

Official prompt and video processor path:

```python
video_token = processor.tokenizer.video_token
inputs = processor(text=f"{video_token}\n\n{prompt}", videos=video_path, return_tensors="pt")
outputs = model.generate(**inputs)
response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
```

Important alignment points:

- The NVILA-HD-Video guide uses `AutoProcessor` and `AutoModel`, not `AutoModelForCausalLM`.
- The model ID is `nvidia/NVILA-8B-HD-Video`.
- The PoC local path convention is `weights/NVILA-8B-HD-Video`.
- The official processor path accepts `videos=video_path` and prepends `processor.tokenizer.video_token` to the prompt.
- `A1_real` is a PoC AutoGaze OFF ablation; it is not an official NVILA-HD-Video guide command.
- `A2_real` is the closest canonical AutoGaze ON path.

### 5.1 AutoGaze-Only Canonical Check

Status: `runnable smoke script with guarded real model loading`.

This checks AutoGaze behavior for the A2 path without attempting NVILA generation.
It is not the official NVILA-HD-Video processor command, but it uses the lower-level AutoGaze and modified SigLIP references from `QUICK_START.md`.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --resolution 224 \
  --device cpu \
  --output-dir outputs/a2_autogaze_only_dummy
```

### 5.2 Full Pipeline Canonical Inference with Query Text

Status: `runnable partial smoke script with guarded stages`.

The current PoC accepts query text and logs it. NVILA generation is skipped unless explicit checkpoint/model loading is enabled.
The official NVILA-HD-Video processor-first inference behavior is documented in `docs/NVILA_HD_VIDEO_REFERENCE.md` and represented by the guarded isolated PoC script `scripts/poc_nvila_hd_video.py`.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --num-frames 2 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cpu \
  --output-dir outputs/a2_full_pipeline_dummy
```

### 5.3 Current Runnable/Stub Status

| Path | Status | Notes |
|---|---|---|
| Isolated NVILA-HD-Video check | runnable | `scripts/poc_nvila_hd_video.py --mode check`; does not load heavy checkpoints by default |
| Isolated NVILA-HD-Video AutoGaze-only | guarded/stub by default | Runs preprocessing and reports AutoGaze stage; real execution requires explicit checkpoint loading |
| Isolated NVILA-HD-Video full pipeline | guarded/stub by default | Accepts query text and reports skipped generation unless checkpoint loading is explicitly enabled |
| A2 AutoGaze-only smoke | runnable | Requires local/cached AutoGaze for real stage execution |
| A1/A2 modified SigLIP smoke | runnable when local SigLIP assets are present | Uses lower-level `gazing_info` path from `QUICK_START.md` |
| NVILA-HD-Video official processor path | guarded PoC | Implemented in `scripts/poc_nvila_hd_video.py`; real generation requires explicit checkpoint loading and enough memory |
| NVILA generation in current smoke script | guarded/stub by default | Requires `--allow-mllm-load` and enough memory |
| 4K / 1K-frame execution | future work | Not part of smoke or small benchmark defaults |

### 5.4 Isolated PoC Script

Status: `runnable check mode`; no checkpoint tensors are loaded by default.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode check \
  --config configs/experiment/A2_real.yaml \
  --output-dir outputs/nvila_hd_video_poc/check
```

Status: `guarded/stub by default`; AutoGaze execution is skipped unless `--allow-checkpoint-load` is passed.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --resolution 224 \
  --device cpu \
  --config configs/experiment/A2_real.yaml \
  --output-dir outputs/nvila_hd_video_poc/autogaze_only
```

Status: `guarded/stub by default`; query text is accepted and logged, but NVILA generation is skipped unless checkpoint loading is explicitly enabled.

```bash
python scripts/poc_nvila_hd_video.py \
  --mode full_pipeline \
  --video dummy \
  --query-text "Question: What is happening in this video? Please answer directly." \
  --num-frames 2 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cpu \
  --config configs/experiment/A2_real.yaml \
  --output-dir outputs/nvila_hd_video_poc/full_pipeline
```

---

## 6. AutoGaze-Only Commands

### 6.1 A2 AutoGaze-Only Dummy Video

Status: `runnable smoke script with guarded real model loading`. This command is not present in original `QUICK_START.md`.
It does not download models automatically. If `bfshi/AutoGaze` is not cached locally, the AutoGaze stage is marked skipped.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --resolution 224 \
  --device cpu \
  --output-dir outputs/a2_autogaze_only_dummy
```

Expected output:

```text
- selected patch indices
- selected scales, if available
- original visual token count
- selected visual token count
- token reduction ratio
- AutoGaze visualization, if visualizer is available
```

---

### 6.2 A2 AutoGaze-Only Local Video

Status: `runnable smoke script with guarded real model loading`. Original `QUICK_START.md` provides a Python API example instead of this script.
Replace /path/to/video.mp4 with a real local video path.
Local video decoding requires PyAV and the original AutoGaze video utility imports.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video-path /path/to/video.mp4 \
  --num-frames 4 \
  --resolution 224 \
  --device cuda \
  --output-dir outputs/a2_autogaze_only_local
```

Expected output:

```text
outputs/a2_autogaze_only_local/
  autogaze/
  visualizations/autogaze_only/
  logs/
```

---

## 7. Full Pipeline Commands

### 7.1 A1 Full Pipeline without AutoGaze

A1 is the modified SigLIP + NVILA baseline with AutoGaze OFF.

Status: `runnable smoke script with guarded stages`. Original `QUICK_START.md` does not describe AutoGaze OFF or NVILA full-pipeline inference.
Replace /path/to/video.mp4 with a real local video path.
If NVILA is unavailable, query text is logged and generation is skipped with a reason.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A1_real \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "Describe the video." \
  --num-frames 4 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cuda \
  --output-dir outputs/a1_full_pipeline
```

Expected path:

```text
video + query text
-> frame sampling
-> modified SigLIP
-> NVILA
-> generated answer
```

If NVILA generation is unavailable, the script should stop at the last available stage and report the skipped reason.

---

### 7.2 A2 Full Pipeline with AutoGaze

A2 is the canonical AutoGaze ON path.

Status: `runnable smoke script with guarded stages`. Original `QUICK_START.md` describes AutoGaze + modified SigLIP Python API usage, but not NVILA query-text generation.
Replace /path/to/video.mp4 with a real local video path.
If AutoGaze, modified SigLIP, or NVILA construction/inference is unavailable, the script stops at the last available stage and records the blocker.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "What is the main action in the video?" \
  --num-frames 4 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cuda \
  --output-dir outputs/a2_full_pipeline
```

Expected path:

```text
video + query text
-> frame sampling
-> AutoGaze
-> modified SigLIP
-> NVILA
-> generated answer
```

Expected logs:

```text
- input shape
- sampled frame indices
- original resolution
- processed resolution
- AutoGaze ON/OFF
- original visual token count
- selected visual token count
- token reduction ratio
- vision feature shape
- MLLM input shape
- generated answer or skipped reason
```

---

## 8. Query Text Handling

Full pipeline mode should accept query text.

Required CLI argument:

Argument example, not a standalone command:

```text
--query-text "What is happening in this video?"
```

Rules:

- Query text is required for MLLM-style Video VQA inference.
- Query text must not be silently ignored.
- If `--allow-mllm-load` is not set, query text is logged and MLLM generation is skipped by design.
- If the MLLM is unavailable, the system must log:
  ```text
  Query text was provided, but MLLM generation was skipped because <reason>.
  ```
- For action recognition mode, query text may be optional or ignored only if explicitly documented.

---

## 9. Resolution Scaling

Resolution scaling should follow the original `QUICK_START.md` whenever possible.

Required config fields:

```yaml
inference:
  input_resolution: 224
  scale_resolution: null
  max_resolution: null
  target_scales: null
  target_patch_size: null
```

Expected behavior:

| Field | Meaning |
|---|---|
| `input_resolution` | target input resolution for smoke tests; quick start default is 224 |
| `scale_resolution` | PoC extension field; original quick start uses Python API `target_scales` and `target_patch_size` |
| `max_resolution` | upper bound for high-resolution inference |
| `target_scales` | original quick start AutoGaze argument for target encoder scales |
| `target_patch_size` | original quick start AutoGaze argument for target encoder patch size |

Rules:

- Do not invent scaling behavior that conflicts with original `QUICK_START.md`.
- If original scaling uses specific CLI flags, preserve those names where practical.
- Always log:
  - original video resolution
  - target resolution
  - effective processed resolution
  - scaling mode

Example:

Status: `runnable smoke script, partial full-pipeline scaling support`. `--scale-resolution` is a PoC flag, not an original quick start flag.
The script reflects the original Python API by passing `target_scales` and `target_patch_size` when those fields are configured.
See `docs/SCALING_GUIDE.md` for the current policy.

Current support:

- `resize_224`: runnable default path.
- `resize_392_patch14`: runnable guarded high-resolution path; raw `384` should be represented as documented `392`.
- `spatio_temporal_224`: preprocessing utility-supported; full benchmark aggregation remains guarded/stubbed.
- `spatio_temporal_392_patch14`: preprocessing utility-supported; full benchmark aggregation remains guarded/stubbed.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video-path /path/to/video.mp4 \
  --num-frames 4 \
  --resolution 392 \
  --scale-resolution quick_start_target_scales \
  --output-dir outputs/a2_scaled
```

If scaling is not yet implemented:

```text
The command should fail with a clear NotImplementedError or mark scaling as stub-only.
```

---

## 10. Frame Sampling

All video tensors should follow:

```python
[B, T, C, H, W]
```

Frame sampling requirements:

- uniformly sample `N` frames
- preserve original frame indices
- log sampled frame indices
- support dummy video and local video

Example config:

```yaml
data:
  num_frames: 4
  frame_sampling: uniform
```

Expected log:

```json
{
  "num_frames": 4,
  "sampled_frame_indices": [0, 10, 20, 30]
}
```

---

## 11. Device and Precision

Supported devices:

```text
cpu
cuda
mps
```

Supported dtype options:

```text
float32
float16
bfloat16
```

Example:

Status: `runnable smoke script template`; replace /path/to/video.mp4 with a real local video path.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video-path /path/to/video.mp4 \
  --query-text "Describe the video." \
  --device cuda \
  --dtype float16
```

Device policy:

| Device | Purpose |
|---|---|
| CUDA | primary benchmark target |
| MPS | local smoke test |
| CPU | minimal functional test |

MPS notes:

- FlashAttention must not be used.
- Use SDPA or eager attention fallback.
- Some profiling metrics may be `N/A`.

---

## 12. Output Directory Structure

Recommended output structure:

```text
outputs/
  <experiment_name>/
    autogaze/
      selected_patch_indices.json
      selected_scales.json
      token_counts.json

    predictions/
      answer.json
      logits.pt

    visualizations/
      autogaze_only/
      full_pipeline/
      video_vqa/
      action_recognition/

    logs/
      inference_summary.json
      profiling.json
      reproducibility_manifest.json
```

Required summary fields:

```json
{
  "experiment_id": "A2_real",
  "mode": "full_pipeline",
  "autogaze_enabled": true,
  "vision_encoder": "modified_siglip",
  "mllm": "nvila",
  "num_frames": 4,
  "original_resolution": null,
  "processed_resolution": 224,
  "query_text": "What is happening in this video?",
  "original_visual_token_count": null,
  "selected_visual_token_count": null,
  "token_reduction_ratio": null,
  "generated_answer": null,
  "skipped_stages": []
}
```

---

## 13. Visualization Outputs

### 13.1 AutoGaze-Only Visualization

Should show:

- selected patch locations
- selected scales, if available
- frame-wise selection
- token count per frame

Output:

```text
outputs/<exp_name>/visualizations/autogaze_only/
```

### 13.2 Full Pipeline Visualization

Should show:

- selected patches
- predicted answer text
- top-k labels for action recognition, if applicable
- frame-wise output

Output:

```text
outputs/<exp_name>/visualizations/full_pipeline/
```

---

## 14. Hugging Face-Based Inference

Hugging Face inference is optional and should be implemented only when supported.

Initial policy:

- use official processor path first
- do not assume direct AutoGaze token injection
- support AutoGaze-guided frame/region selection only as a separate experimental mode
- support local cache and offline mode
- do not log Hugging Face access tokens

Example future command:

Status: `future work`; run_hf_inference.py does not exist yet.

```bash
python scripts/run_hf_inference.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --video-path /path/to/video.mp4 \
  --query-text "What is happening in this video?" \
  --num-frames 4 \
  --resolution 224 \
  --device cuda \
  --output-dir outputs/hf_qwen_inference
```

---

## 15. Tiny Real Benchmark

Status: `runnable tiny benchmark`; this is a smoke benchmark only, not a paper reproduction result.
It uses local real-path configs and the guarded canonical smoke inference runner.

Default settings:

```text
batch_size: 1
num_frames: 2
resolution: 224
warmup_iterations: 1
benchmark_iterations: 3
max_new_tokens: 1
```

A1 AutoGaze-only:

Status: `runnable tiny benchmark`.

```bash
python scripts/run_tiny_real_benchmark.py \
  --config-name tiny_a1_real \
  --mode autogaze_only \
  --output-dir outputs/tiny_real_benchmarks/A1_real
```

A2 AutoGaze-only:

Status: `runnable tiny benchmark`.

```bash
python scripts/run_tiny_real_benchmark.py \
  --config-name tiny_a2_real \
  --mode autogaze_only \
  --output-dir outputs/tiny_real_benchmarks/A2_real
```

A1 full pipeline:

Status: `runnable tiny benchmark with guarded MLLM stage`.

```bash
python scripts/run_tiny_real_benchmark.py \
  --config-name tiny_a1_real \
  --mode full_pipeline \
  --output-dir outputs/tiny_real_benchmarks/A1_real
```

A2 full pipeline:

Status: `runnable tiny benchmark with guarded MLLM stage`.

```bash
python scripts/run_tiny_real_benchmark.py \
  --config-name tiny_a2_real \
  --mode full_pipeline \
  --output-dir outputs/tiny_real_benchmarks/A2_real
```

Outputs:

```text
outputs/tiny_real_benchmarks/<experiment>/<mode>/
  benchmark_result.json
  benchmark_result.csv
  reproducibility_manifest.json
  iterations/
  warmup/
```

Notes:

- The benchmark logs executed stages and skipped stages.
- NVILA loading/generation is skipped by default; pass `--allow-mllm-load` only when the machine has enough memory.
- QUICK_START scaling behavior is represented through the same smoke inference scaling fields: default `224x224`, or `target_scales` / `target_patch_size` when `--scale-resolution` is used by the underlying smoke path.
- Any-resolution / any-duration chunking is available as a preprocessing utility under `autogaze_ext.scaling`; full benchmark aggregation is still guarded/stubbed.

---

## 16. Troubleshooting

### 16.1 Modified SigLIP import is not detected

Check:

Status: `runnable diagnostic command`.

```bash
python scripts/check_canonical_path.py --experiment-id A1_real
```

Possible causes:

- original AutoGaze repo path is not configured
- Python import path is missing
- modified SigLIP class/factory name is wrong
- modified SigLIP is embedded inside another NVILA module

---

### 16.2 NVILA import is not detected

Check:

Status: `runnable diagnostic command`.

```bash
python scripts/check_canonical_path.py --experiment-id A1_real
```

Possible causes:

- NVILA code is not available
- NVILA is a separate dependency
- module path is wrong
- checkpoint path is missing

---

### 16.3 Checkpoint path is missing

Check real config files:

```text
configs/model/autogaze/real.yaml
configs/model/vision_encoder/modified_siglip_real.yaml
configs/model/mllm/nvila_real.yaml
configs/experiment/A1_real.yaml
configs/experiment/A2_real.yaml
```

Required paths:

```text
AutoGaze checkpoint
modified SigLIP checkpoint or config
NVILA checkpoint or config
tokenizer/processor path, if required
```

---

### 16.4 Query text is ignored

This should not happen.

If query text is provided but MLLM generation is unavailable, the script must report:

```text
Query text was provided, but MLLM generation was skipped because <reason>.
```

---

### 16.5 Resolution scaling does not match QUICK_START.md

Check:

```text
docs/QUICK_START_reference.md
original QUICK_START.md
```

If scaling behavior differs, document:

- original behavior
- current PoC behavior
- reason for difference
- whether it is temporary

---

## 17. Current Limitations

Current expected limitations:

- Local real-path configs currently point to `weights/AutoGaze`, `weights/siglip2-base-patch16-224`, and `weights/NVILA-8B-HD-Video`.
- NVILA-HD-Video reference alignment uses `AutoProcessor` and `AutoModel`; the current generic smoke script is still a lower-level guarded path, not the official processor-first implementation.
- A1/A2 smoke paths can execute real AutoGaze and/or modified SigLIP stages when those local assets are present.
- Full NVILA loading and generation are skipped by default; use `--allow-mllm-load` only when the machine has enough memory.
- Full MLLM generation remains unverified in the tiny smoke path.
- Hugging Face direct visual token injection is not assumed.
- High-resolution target-scale smoke inference is supported for the documented `392` policy; long-video spatio-temporal chunking is currently a preprocessing utility and not yet an official full benchmark path.
- 4K / 1K-frame settings should not be attempted until tiny benchmarks pass.

---

## 18. Recommended Validation Order

Use the following order:

```text
1. check canonical path configuration
2. check non-inference model construction
3. run AutoGaze-only dummy inference
4. run AutoGaze-only local video inference
5. run full pipeline dummy inference with query text
6. run full pipeline local video inference with query text
7. run tiny A1/A2 benchmark
8. run small local video benchmark
```

Commands:

Status: `runnable diagnostic command`.

```bash
python scripts/check_canonical_path.py --experiment-id A2_real
```

Status: `runnable`; `scripts/check_model_construction.py` performs non-inference construction checks only.
It does not run video inference and does not load checkpoint tensors unless explicitly requested.

Construction levels:

```text
0: import check only
1: class/factory resolution only
2: instantiate model config only, no weights
3: checkpoint metadata-only check, no tensor deserialization
4: full model construction, no inference
```

```bash
python scripts/check_model_construction.py \
  --experiment A2_real \
  --component all \
  --construction-level 1 \
  --device cpu
```

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video dummy \
  --num-frames 2 \
  --resolution 224 \
  --device cpu \
  --output-dir outputs/a2_autogaze_only_dummy
```

Status: `runnable smoke script`; real AutoGaze execution requires `bfshi/AutoGaze` to be available locally or in the Hugging Face cache.

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video dummy \
  --query-text "What is happening in this video?" \
  --num-frames 2 \
  --resolution 224 \
  --max-new-tokens 1 \
  --device cpu \
  --output-dir outputs/a2_full_pipeline_dummy
```

Status: `runnable smoke script with guarded stages`; MLLM generation is skipped unless NVILA construction and generation are available.
By default the script also skips MLLM loading to avoid accidentally loading large NVILA checkpoints; add `--allow-mllm-load` only for an explicit MLLM smoke run.

---

## 19. Status Labels

Every command in this guide should be labeled internally as one of:

```text
runnable
stub-only
future work
```

Do not present a command as runnable unless it has been tested.
