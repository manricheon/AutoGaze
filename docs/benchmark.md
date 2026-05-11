# AutoGaze PoC 벤치마크 계획

이 문서는 AutoGaze 확장 PoC의 벤치마크 조사, 재현 계획, 확장 계획, Hugging Face 기반 benchmark path, 측정 방법론, 결과 표 템플릿을 정의합니다.

중요 구분:

- dummy/stub 결과는 pipeline 연결 확인용입니다.
- real benchmark 결과는 실제 checkpoint, dataset, protocol, hardware 정보를 포함해야 합니다.
- 외부 MLLM이 AutoGaze를 사용했다고 명시적으로 보고하지 않은 경우 그렇게 해석하지 않습니다.

## 1. Public Benchmark Survey

AutoGaze paper의 Table 1 역할을 따르는 public survey를 작성합니다. 목적은 기존 video MLLM이 실제로 어느 정도의 frame count, resolution, long-video/high-resolution setting을 처리하는지 비교하는 것입니다.

이 survey는 다른 모델이 AutoGaze를 사용했다는 증거가 아닙니다.

### 조사 항목

| 항목 | 설명 |
|---|---|
| Model | 모델명 |
| Open? | open-source 여부 |
| Model size | parameter 규모 |
| Max #Frames | 보고된 최대 입력 frame 수 |
| Max Resolution | 보고된 최대 resolution |
| VideoMME w/o Sub | subtitle 없는 VideoMME |
| VideoMME w/ Sub | subtitle 있는 VideoMME |
| MVBench | MVBench score |
| NExT-QA | NExT-QA score |
| LongVideoBench / L-VidBench | long-video benchmark |
| EgoSchema | EgoSchema score |
| MLVU | MLVU score |
| HLVid | high-resolution long-video QA |
| Notes | long-video/high-resolution 특성 |

### Survey table template

| Model | Open? | Max #Frames | Max Resolution | VideoMME w/o Sub | VideoMME w/ Sub | MVBench | NExT-QA | L-VidBench / LongVideoBench | EgoSchema | MLVU | HLVid | Result Source | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Gemini 1.5 Pro | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | proprietary baseline |
| GPT-4o | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | proprietary baseline |
| Qwen2.5-VL-7B | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | open-source baseline |
| NVILA-8B-Video | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | original NVILA baseline |
| NVILA-8B-Video + AutoGaze | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | AutoGaze paper | AutoGaze-scaled row |
| Internal PoC A2 | N/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | internal reproduced | 실제 재현 후 입력 |

## 2. Reproduction Benchmark Plan

1차 재현 목표는 canonical SigLIP/NVILA ablation입니다.

| ID | AutoGaze | Vision Encoder | MLLM | 우선순위 | 상태 |
|---|---:|---|---|---:|---|
| A0 | OFF | vanilla SigLIP ViT | NVILA | 2 | vanilla full-token baseline |
| A1 | OFF | modified SigLIP ViT | NVILA | 1 | modified SigLIP effect |
| A2 | ON | modified SigLIP ViT | NVILA | 1 | canonical AutoGaze path |
| A3 | ON | vanilla SigLIP ViT | NVILA | 3 | experimental compatibility ablation |

A3는 구현 및 테스트 전까지 직접 호환된다고 주장하지 않습니다.

### 재현 시 필수 기록

- resolved config
- git commit hash
- package versions
- device information
- CUDA/MPS availability
- precision
- timestamp
- checkpoint paths
- Hugging Face IDs/revisions, 해당 시
- offline/cache mode
- `trust_remote_code`
- token count before/after AutoGaze
- acceleration type note

## 3. Extension Benchmark Plan

Canonical path 이후 다음 축을 확장합니다.

| Axis | 후보 |
|---|---|
| AutoGaze | ON / OFF |
| Vision encoder | modified SigLIP, vanilla SigLIP, V-JEPA2, generic ViT |
| MLLM/decoder | NVILA, Qwen, generic MLLM, task decoder |
| Integration mode | full, hook, native, crop, mask, compact, official_processor |
| Task | Video VQA, Action Recognition |
| Source mode | local, Hugging Face, mixed, offline cache |

주의:

- post-encoder pruning은 encoder-side acceleration이 아닙니다.
- full ViT forward 이후 mask 적용은 ViT acceleration이 아닙니다.
- MLLM prefill token 감소는 downstream acceleration으로 별도 분류합니다.

## 4. Hugging Face-Based Benchmark Plan

지원 benchmark modes:

| Mode | Model Source | Dataset Source | 설명 |
|---|---|---|---|
| `hf_model_only` | HF Hub/cache | dummy/local dataset | public model loading smoke |
| `hf_dataset_only` | local/internal model | HF Hub/local file | public dataset loading smoke |
| `hf_model_and_dataset` | HF Hub/cache | HF Hub/cache | full public path |
| `local_model_hf_dataset` | local/internal checkpoint | HF dataset | internal model on public data |
| `hf_model_local_dataset` | HF model | local/internal dataset | public model on internal data |
| `offline_hf_cache` | local HF cache | local HF cache | offline reproducibility |

기본 원칙:

- dry-run이 기본입니다.
- 큰 public model benchmark는 기본 실행하지 않습니다.
- official processor path를 우선합니다.
- AutoGaze token injection은 지원된다고 가정하지 않습니다.
- model/dataset revision pinning을 사용합니다.

HF benchmark smoke:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.hf_benchmark \
  --config-name hf_benchmark/hf_dataset_only \
  --output-dir outputs/hf_benchmarks \
  --dry-run
```

## 5. Measurement Methodology

### Latency

- warm-up iteration 수는 config로 제어합니다.
- CUDA timing은 `torch.cuda.synchronize()`를 사용합니다.
- data loading 포함 여부를 명시합니다.

Latency breakdown:

| Metric | 설명 |
|---|---|
| AutoGaze latency | AutoGaze selector/router 시간 |
| ViT latency | vision encoder 시간 |
| MLLM prefill latency | visual/text prefill |
| MLLM decode latency | generation decode |
| end-to-end latency | 전체 pipeline 시간 |

### Memory

| Device | Metric |
|---|---|
| CUDA | `torch.cuda.max_memory_allocated()` |
| MPS | unavailable metric은 `N/A` |
| CPU | benchmark timing 해석 제한, memory는 필요 시 별도 기록 |

### Token metrics

- visual token count before AutoGaze
- visual token count after AutoGaze
- token reduction ratio
- selected patches per frame
- selected patches per scale

## 6. Benchmark Result Table Templates

### Efficiency table

| Experiment | AutoGaze | Vision Encoder | MLLM | Frames | Resolution | Before Tokens | After Tokens | Reduction | Latency ms | Throughput | FPS | VRAM MB | Acceleration Type | Notes |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A2 | ON | modified SigLIP | NVILA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | real run only |

### Performance table

| Experiment | Task | Dataset | Metric | Score | Protocol | Result Type | Notes |
|---|---|---|---|---:|---|---|---|
| A2 | Video VQA | TBD | Exact Match | TBD | internal | reproduced | TBD |

### Token reduction table

| Experiment | Frame | Scale | Original Tokens | Selected Tokens | Reduction | Notes |
|---|---:|---|---:|---:|---:|---|
| A2 | TBD | TBD | TBD | TBD | TBD | metadata required |

### Resolution/frame scaling table

| Experiment | Frames | Resolution | Latency | VRAM | Metric | Notes |
|---|---:|---|---:|---:|---:|---|
| A2 | 32 | 224p | TBD | TBD | TBD | baseline |
| A2 | 128 | 720p | TBD | TBD | TBD | scaling |

### Hugging Face benchmark result table

| Experiment | HF Model ID | Model Revision | HF Dataset ID | Dataset Split | Integration Mode | Samples | Metric Source | Metric | Offline | Cache Dir | Notes |
|---|---|---|---|---|---|---:|---|---:|---|---|---|
| hf_dataset_only | N/A | N/A | local.jsonl | validation | official_processor | 2 | internal_fallback | TBD | false | TBD | smoke |

### Dummy/stub result table

| Experiment | Result File | Stub Status | Real Checkpoints Loaded? | Use For |
|---|---|---|---|---|
| A0 dummy | `outputs/dummy_benchmarks/A0.json` | dummy_full_token_baseline | No | wiring smoke |
| A2 dummy | `outputs/dummy_benchmarks/A2.json` | stubbed_autogaze_on_no_real_selector_no_token_reduction | No | wiring smoke |
