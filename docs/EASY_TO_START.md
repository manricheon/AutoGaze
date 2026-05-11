# AutoGaze PoC 빠른 시작 가이드

이 문서는 AutoGaze 확장 PoC를 처음 실행하는 개발자를 위한 시작 문서입니다. 현재 저장소의 확장 코드는 실제 AutoGaze, SigLIP, NVILA, Qwen 체크포인트 없이 동작하는 더미/스텁 경로를 포함합니다. 더미 결과는 기능 연결 확인용이며 실제 재현 결과나 성능 결과가 아닙니다.

## 1. 설치

권장 환경:

- Python 3.10 이상
- CUDA 벤치마크: Linux + NVIDIA GPU
- 로컬 스모크 테스트: macOS + MPS 또는 CPU

기본 설치 예:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

프로젝트 루트에서 실행할 때는 아직 패키지 설치 없이도 다음처럼 `PYTHONPATH`를 사용할 수 있습니다.

```bash
PYTHONPATH=src pytest tests/test_config_loading.py -q
```

## 2. Linux/CUDA 실행

CUDA는 실제 벤치마크의 주 대상입니다. 현재 더미 경로는 CUDA가 없어도 실행됩니다.

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.inference \
  --config-name dummy_video_vqa
```

CUDA 벤치마크를 확장할 때 지켜야 할 원칙:

- 실제 측정 전에 warm-up iteration을 설정합니다.
- CUDA timing은 `torch.cuda.synchronize()`를 사용합니다.
- peak VRAM은 `torch.cuda.max_memory_allocated()` 기준으로 기록합니다.
- FlashAttention 사용 여부와 attention fallback을 명시합니다.

## 3. Mac/MPS 실행

MPS는 로컬 개발과 스모크 테스트 용도입니다.

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.inference \
  --config-name dummy_action_recognition
```

MPS 정책:

- FlashAttention을 사용하지 않습니다.
- PyTorch SDPA 또는 eager attention fallback을 사용합니다.
- MPS에서 제공되지 않는 profiling metric은 `N/A`로 기록합니다.

## 4. 더미 Video VQA 추론 예제

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.inference \
  --config-name dummy_video_vqa
```

출력 예:

```text
task type: video_vqa
input video shape: (2, 4, 3, 224, 224)
sampled frame indices: [0, 2, 5, 7]
AutoGaze: OFF
visual token count before AutoGaze: 784
visual token count after AutoGaze: 784
generated dummy answer: ['dummy', 'dummy']
```

## 5. 더미 Action Recognition 추론 예제

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.inference \
  --config-name dummy_action_recognition
```

현재 더미 decoder는 class 0을 반환합니다. 이는 task pipeline 연결 확인용입니다.

## 6. AutoGaze ON/OFF 예제

AutoGaze OFF 더미 경로:

```yaml
model:
  autogaze:
    enabled: false
    mode: full
```

AutoGaze ON canonical config 예:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.runner --experiment-id A2
```

주의:

- 현재 더미 benchmark에서 A2/A3는 AutoGaze ON wiring만 기록합니다.
- 실제 AutoGaze selector는 로드하지 않습니다.
- A2/A3 더미 결과는 `stubbed_autogaze_on_no_real_selector_no_token_reduction`으로 표시됩니다.

## 7. modified SigLIP / vanilla SigLIP 선택

Canonical experiment:

| ID | AutoGaze | Vision Encoder | MLLM | 상태 |
|---|---:|---|---|---|
| A0 | OFF | vanilla SigLIP ViT | NVILA | vanilla baseline wiring |
| A1 | OFF | modified SigLIP ViT | NVILA | 우선 구현 대상 |
| A2 | ON | modified SigLIP ViT | NVILA | canonical AutoGaze path |
| A3 | ON | vanilla SigLIP ViT | NVILA | experimental compatibility ablation |

출력 확인:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.runner --experiment-id A0
PYTHONPATH=src python -m autogaze_ext.pipeline.runner --experiment-id A1
PYTHONPATH=src python -m autogaze_ext.pipeline.runner --experiment-id A2
PYTHONPATH=src python -m autogaze_ext.pipeline.runner --experiment-id A3
```

A3는 직접 호환된다고 주장하지 않습니다. 실제 구현과 테스트 전까지 experimental ablation입니다.

## 8. NVILA / Qwen / generic MLLM 선택

현재 adapter 상태:

| Adapter | 상태 | 설명 |
|---|---|---|
| `GenericMLLMAdapter` | dummy 구현 | smoke test용 고정 답변 생성 |
| `NVILAAdapter` | wrapper stub | 실제 NVILA model instance 필요 |
| `QwenAdapter` | staged stub | official processor 우선, 직접 visual token injection은 미검증 |
| `HFMLLMAdapter` | placeholder | Hugging Face MLLM용 확장 지점 |

Qwen 통합 단계:

1. `official_processor`
2. `input_region_selection`
3. `post_visual_encoder_pruning`
4. `direct_visual_token_injection`

직접 visual token injection은 architecture/API 지원을 검증하기 전까지 지원된다고 가정하지 않습니다.

## 9. Hugging Face 모델 벤치마크 예제

Dry-run:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.hf_benchmark \
  --config-name hf_benchmark/hf_model_only \
  --output-dir outputs/hf_benchmarks \
  --dry-run
```

실제 로딩은 config에 `model_id`, `revision`, `cache_dir`, `local_files_only` 등을 명시하고 dry-run을 끈 경우에만 수행합니다.

## 10. Hugging Face 데이터셋 벤치마크 예제

Dry-run:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.hf_benchmark \
  --config-name hf_benchmark/hf_dataset_only \
  --output-dir outputs/hf_benchmarks \
  --dry-run
```

로컬 JSONL 메타데이터 smoke test는 `data.huggingface.dataset_id`에 파일 경로를 지정해서 실행할 수 있습니다.

지원 로컬 포맷:

- `.json`
- `.jsonl`
- `.csv`

## 11. Offline Hugging Face Cache 예제

Asset dry-run manifest:

```bash
python scripts/download_hf_assets.py \
  --model-id org/model \
  --dataset-id org/dataset \
  --revision abc123 \
  --cache-dir ./hf_cache \
  --manifest-out ./hf_cache/assets_manifest.json \
  --dry-run
```

Offline benchmark config:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.hf_benchmark \
  --config-name hf_benchmark/offline_hf_cache \
  --output-dir outputs/hf_benchmarks \
  --dry-run
```

Offline mode에서는 다음 환경 변수를 설정합니다.

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

토큰은 `HF_TOKEN` 같은 환경 변수에서만 읽고 manifest, config, output에는 저장하지 않습니다.
