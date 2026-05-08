# AutoGaze 종합 가이드 (Korean)

> **AutoGaze** — CVPR 2026 (NVIDIA)  
> 비디오의 중요한 패치를 자동으로 선택하여 Vision Encoder 및 MLLM의 연산량을 줄이는 자기회귀적(Autoregressive) 가이즈 모델입니다.

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [환경 설정 및 설치](#2-환경-설정-및-설치)
3. [모델 가중치 준비](#3-모델-가중치-준비)
4. [추론 워크플로우 (Inference)](#4-추론-워크플로우)
5. [비디오 QA 벤치마크 (Video QA)](#5-비디오-qa-벤치마크)
6. [벤치마크 및 성능 평가 (CV Tasks)](#6-벤치마크-및-성능-평가)
7. [AutoGaze 통합 및 확장](#7-autogaze-통합-및-확장)
8. [학습 가이드 (Training)](#8-학습-가이드)
9. [자주 묻는 질문 (FAQ)](#9-자주-묻는-질문)

---

## 1. 프로젝트 개요

AutoGaze는 비디오 프레임에서 정보 밀도가 높은 패치를 예측합니다. 선택된 소수의 패치(10~25%)만 Vision Encoder에 전달함으로써, 정확도는 유지하면서 추론 속도와 메모리 효율을 극대화합니다.

### 핵심 아키텍처
- **Encoder**: ConvNeXt 기반 멀티스케일 패치 추출.
- **Decoder**: LLaMA 기반 AR 디코더가 패치 선택 시퀀스 생성.
- **Reward**: VideoMAE 재구성 품질을 보상으로 강화학습(GRPO) 수행.

---

## 2. 환경 설정 및 설치

### 사전 요구사항
- Python 3.11 이상
- CUDA(Linux) 또는 MPS(macOS Apple Silicon) 지원 환경

### 설치 방법
```bash
git clone <repo-url>
cd AutoGaze

# 가상환경 및 패키지 설치
conda create -n autogaze python=3.11
conda activate autogaze
uv pip install -e .
```

---

## 3. 모델 가중치 준비

`scripts/download_models.sh`를 사용하여 HuggingFace에서 가중치를 다운로드합니다.

### 전체 모델 목록

| 키워드 | HF 모델 ID | 역할 | 크기 |
| :--- | :--- | :--- | :--- |
| `autogaze` | `nvidia/AutoGaze` | 패치 선택 모델 (항상 필요) | ~50 MB |
| `videomae` | `bfshi/VideoMAE_AutoGaze` | 학습용 보상 모델 | ~2 GB |
| `nvila` | `nvidia/NVILA-8B-HD-Video` | AutoGaze 통합 MLLM | ~16 GB |
| `vjepa2` | `facebook/vjepa2-vitl-fpc64-256` | Video-native ViT-L | ~2 GB |
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | VL MLLM | ~16 GB |
| `qwen25` | `Qwen/Qwen2.5-7B-Instruct` | LM (`vjepa2_llm` 파이프라인용) | ~15 GB |
| `siglip` | `google/siglip-base-patch16-224` | CV 태스크 | ~400 MB |
| `siglip2` | `google/siglip2-base-patch16-224` | CV 태스크 | ~400 MB |
| `dinov2` | `facebook/dinov2-base-imagenet1k-1-layer` | CV 태스크 | ~350 MB |
| `yolos` | `hustvl/yolos-tiny` | CV 객체 탐지 | ~30 MB |
| `segformer` | `nvidia/segformer-b2-finetuned-ade-512-512` | CV 세그멘테이션 | ~100 MB |
| `depthanything` | `depth-anything/Depth-Anything-V2-Small-hf` | CV 깊이 추정 | ~100 MB |

### 그룹 키워드

| 키워드 | 포함 모델 | 용도 |
| :--- | :--- | :--- |
| `mllm` | nvila + vjepa2 + qwen25vl + qwen25 | MLLM 벤치마크 전체 |
| `cv` | siglip + siglip2 + dinov2 + yolos + segformer + depthanything | CV 태스크 소형 모델 |
| `all` | 전체 | (~52 GB) |

### 다운로드 명령

```bash
# 기본 (autogaze + videomae — 추론·학습에 항상 필요)
bash scripts/download_models.sh

# MLLM 벤치마크용 전체
bash scripts/download_models.sh weights mllm

# V-JEPA2 + NVILA 조합 (nvila_vjepa2 러너)
bash scripts/download_models.sh weights vjepa2 nvila

# CV 태스크용 소형 모델
bash scripts/download_models.sh weights cv

# 전부 다운로드
bash scripts/download_models.sh weights all
```

---

## 4. 추론 워크플로우 (Inference)

추론은 크게 두 가지 스크립트로 나뉩니다. 자세한 내용은 [추론 가이드](inference_guide.md)를 참고하세요.

### 4.1 가이즈 맵 분석 (`infer.py`)
가이즈 선택 결과를 시각화하거나 학습용 레이블을 생성할 때 사용합니다.

```bash
# 25% 선택 vs 100% 전체 패치 비교 시각화
python -m autogaze.infer assets/example_input.mp4 --gazing-ratio 0.25 --compare-autogaze

# 비율별 가이즈 변화 자동 스윕 (0.1 → 1.0)
python -m autogaze.infer assets/example_input.mp4 --sweep-ratio --ratio-step 0.2
```

### 4.2 전체 MLLM 파이프라인 (`infer_full.py`)
AutoGaze를 MLLM에 통합하여 실제 질문 답변 성능을 측정합니다.

```bash
# NVILA 추론 (AutoGaze ON)
python autogaze/infer_full.py assets/example_input.mp4 --mllm nvila

# Qwen2.5-VL 추론
python autogaze/infer_full.py assets/example_input.mp4 --mllm qwen25vl \
    --model-path weights/Qwen2.5-VL-7B-Instruct

# V-JEPA2 + NVILA LLM
python autogaze/infer_full.py assets/example_input.mp4 --mllm nvila_vjepa2 \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256
```

인터랙티브 노트북: **`notebooks/12_inference_full_ko.ipynb`**

---

## 5. 비디오 QA 벤치마크 (Video QA)

표준 벤치마크셋에서 AutoGaze의 성능을 검증합니다. 자세한 내용은 [평가 가이드](eval_guide.md)를 참고하세요.

### 5.1 지원 태스크 및 러너

**태스크 (`--task`)**

| 태스크 | 데이터셋 | 비디오 소스 |
| :--- | :--- | :--- |
| `videomme` / `videomme_w_sub` | `lmms-lab/Video-MME` | HF 임베디드 바이트 |
| `mvbench` | `OpenGVLab/MVBench` | HF 임베디드 바이트 |
| `nextqa` | `lmms-lab/NExTQA` | HF 임베디드 바이트 |
| `egoschema` | `lmms-lab/EgoSchema` | HF 임베디드 바이트 |
| `mlvu` | `MLVU/MLVU` | HF 임베디드 바이트 |
| `longvideobench` | `longvideobench/LongVideoBench` | HF 임베디드 바이트 |
| `hlvid` | `bfshi/HLVid` | 로컬 파일 (`--video-dir` 필수) |

**MLLM 러너 (`--mllm`)**

| 러너 | 모델 | AutoGaze 통합 방식 |
| :--- | :--- | :--- |
| `nvila` | NVILA-8B-HD-Video | 프로세서 통합 (Full) |
| `qwen25vl` | Qwen2.5-VL-7B | Zero-shot 토큰 선택기 |
| `qwen25vl_full` | Qwen2.5-VL-7B | Zero-shot (전체 비디오) |
| `vjepa2_llm` | V-JEPA2 ViT + Qwen2.5-7B | Zero-shot 토큰 선택기 |
| `nvila_vjepa2` | V-JEPA2 ViT + NVILA LLM | Zero-shot 토큰 선택기 |

### 5.2 데이터셋 준비

HF-bytes 태스크의 경우 `download_data_eval.sh`로 사전 다운로드하면 오프라인 실행이 가능합니다.

```bash
# VideoMME + MVBench 다운로드 (~85 GB)
bash scripts/download_data_eval.sh data/eval videomme mvbench

# HF-bytes 전체 (~158 GB)
bash scripts/download_data_eval.sh data/eval hf_bytes

# HLVid (로컬 파일 필요, ~152 GB)
bash scripts/download_data_eval.sh data/eval hlvid

# 전부 (~310 GB)
bash scripts/download_data_eval.sh data/eval all
```

> **HF 바이트 오류 시**: `Video bytes missing — skipping` 경고가 많이 발생하면 위 명령으로 사전 다운로드한 뒤 `--hf-data-dir`을 사용하세요.

### 5.3 기준선 벤치마크 (Baseline VQA)

AutoGaze 없이 모든 패치를 처리하는 "풀 패치 기준선"을 먼저 측정하는 것이 개발 흐름의 핵심입니다. 기준선은 AutoGaze 효과를 측정하는 비교 대상이 됩니다.

```bash
# ── NVILA 기준선 (전체 패치, AutoGaze 비활성) ──────────────────
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --hf-data-dir data/eval/Video-MME \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --no-autogaze \
    --output results/videomme_nvila_baseline.json

# ── Qwen2.5-VL 기준선 ────────────────────────────────────────
python -m autogaze.eval.run_benchmark \
    --task mvbench \
    --hf-data-dir data/eval/MVBench \
    --mllm qwen25vl \
    --model-path weights/Qwen2.5-VL-7B-Instruct \
    --no-autogaze \
    --output results/mvbench_qwen25vl_baseline.json

# ── 여러 태스크 기준선 일괄 실행 ─────────────────────────────
bash scripts/run_benchmarks.sh \
    --tasks videomme,mvbench,nextqa,egoschema \
    --hf-data-dir data/eval \
    --baseline-only \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video
```

**기준선 결과 해석:**

| 태스크 | NVILA 기준선 (참고값) |
| :--- | :---: |
| VideoMME | ~72.0% |
| MVBench | ~76.2% |
| EgoSchema | ~72.5% |

기준선 대비 AutoGaze ON의 정확도 차이가 **±1% 이내**이면 토큰 효율화가 성공적으로 작동하는 것입니다.

### 5.4 AutoGaze ON/OFF 비교 테스트

```bash
# 마스터 스크립트로 ON + OFF 자동 실행 및 요약 테이블 출력
bash scripts/run_benchmarks.sh \
    --tasks videomme,mvbench \
    --hf-data-dir data/eval \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze \
    --max-samples 100    # 스모크 테스트

# AutoGaze ON (75% 토큰)
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --hf-data-dir data/eval/Video-MME \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75 \
    --output results/videomme_nvila_ag075.json

# nvila_vjepa2 러너 (V-JEPA2 ViT + NVILA LLM)
python -m autogaze.eval.run_benchmark \
    --task videomme \
    --hf-data-dir data/eval/Video-MME \
    --mllm nvila_vjepa2 \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256 \
    --autogaze-path weights/AutoGaze \
    --gazing-ratio 0.75
```

**AutoGaze 효과 측정 지표:**

| 지표 | 목표 | 비고 |
| :--- | :--- | :--- |
| 정확도 (%) | 유지 (±1% 이내) | 기준선 대비 |
| 지연 시간 (ms/frame) | 감소 (2–4×) | Full Mode에서만 유효 |
| VRAM (GB) | 감소 | 토큰 수 감소 효과 |

---

## 6. 벤치마크 및 성능 평가 (CV Tasks)

이미지·비디오에서 깊이 추정, 객체 탐지, 세그멘테이션 등 다양한 CV 태스크에 AutoGaze를 적용합니다. 자세한 내용은 [벤치마크 가이드](benchmark_guide.md)를 참고하세요.

### 지원 CV 모델

| 모델 | 태스크 | 키워드 |
| :--- | :--- | :--- |
| SigLIP-base/16 | 특징 추출, Zero-shot 분류 | `siglip` |
| SigLIP2-base/16 | 특징 추출 | `siglip2` |
| DINOv2-base | 특징 추출, 분류 | `dinov2` |
| YOLOS-tiny | 객체 탐지 | `yolos` |
| SegFormer-B2 | 시맨틱 세그멘테이션 | `segformer` |
| Depth-Anything-V2-S | 단안 깊이 추정 | `depthanything` |

```bash
# CV 모델 다운로드
bash scripts/download_models.sh weights cv

# 이미지 전체 태스크 실행
python scripts/run_cv_tasks.py --input assets/sample.jpg --output-dir results/

# 특정 태스크 + ratio 그리드
python scripts/run_cv_tasks.py \
    --input assets/sample.jpg \
    --tasks depth dinov2 yolos \
    --ratios 0.25 0.5 0.75 1.0

# 결과 시각화
python scripts/visualize_cv_results.py results/
```

인터랙티브 노트북: **`notebooks/08_autogaze_cv_tasks_ko.ipynb`**  
상세 메트릭 해석: [벤치마크 가이드](benchmark_guide.md)

---

## 7. AutoGaze 통합 및 확장

기존의 ViT 기반 모델에 AutoGaze를 붙이는 방법은 두 가지입니다.

1.  **Hook Mode (Zero-shot)**: 모델 수정 없이 패치 임베딩 단계에서 마스킹. (정확도 검증용)
2.  **Full Mode (Integrated)**: 토큰을 물리적으로 제거하여 Quadratic Speedup 달성. (속도·메모리 벤치마크용)

기술적 상세 구현은 [통합 가이드](integration_guide.md)에 설명되어 있습니다.

---

## 8. 학습 가이드 (Training)

AutoGaze는 2단계로 학습됩니다.

1.  **Stage 1 (NTP)**: Ground-truth 가이즈 시퀀스 모방 학습.
2.  **Stage 2 (RL)**: GRPO 알고리즘을 사용해 재구성 품질(Reconstruction Reward) 최적화.

```bash
# 학습 데이터 다운로드
bash scripts/download_data.sh data/AutoGaze-Training-Data

# Stage 1 — NTP 단일 GPU
bash scripts/train_ntp_single_gpu.sh data/AutoGaze-Training-Data weights/VideoMAE_AutoGaze/videomae.pt

# Stage 2 — RL 단일 GPU
bash scripts/train_rl_single_gpu.sh data/AutoGaze-Training-Data \
    weights/VideoMAE_AutoGaze/videomae.pt \
    exps/ntp_checkpoint.pt

# 멀티 GPU (torchrun)
bash scripts/train_ntp_multi_gpu.sh data/AutoGaze-Training-Data weights/VideoMAE_AutoGaze/videomae.pt
```

자세한 파라미터 설명은 `TRAIN.md`를 참고하세요.

---

## 9. 자주 묻는 질문 (FAQ)

**Q: Mac에서 `flash_attn` 오류가 발생합니다.**  
A: macOS는 `flash_attn`을 지원하지 않습니다. 시스템이 자동으로 `sdpa`로 전환하므로 무시하셔도 되며, 설정 파일에서 `attn_mode`를 `sdpa`로 명시하면 경고가 사라집니다.

**Q: HF 데이터셋 다운로드 시 `Video bytes missing — skipping` 경고가 발생합니다.**  
A: HF 캐시가 불완전할 때 발생합니다. `bash scripts/download_data_eval.sh data/eval hf_bytes`로 전체 데이터셋을 사전 다운로드한 뒤 `--hf-data-dir data/eval`을 벤치마크 실행 시 전달하세요.

**Q: `gazing_pos` 인덱스는 어떻게 계산되나요?**  
A: `프레임 번호 × 프레임당 토큰 수 + 프레임 내 패치 번호`로 계산되는 전역 인덱스입니다.

**Q: 속도 향상이 체감되지 않습니다.**  
A: `Full Mode` (예: `nvila` 또는 `_full` 접미사 러너)를 사용 중인지 확인하세요. `Hook Mode`는 연산량은 동일하고 결과만 마스킹하므로 속도 향상이 없습니다.

**Q: `nvila_vjepa2` 러너를 사용하려면 무엇이 필요한가요?**  
A: NVILA 가중치(`--model-path`)와 V-JEPA2 가중치(`--vjepa2-path`)가 모두 필요합니다. `bash scripts/download_models.sh weights nvila vjepa2`로 다운로드하세요.

---

*문서 최종 갱신: 2026-05-08*
