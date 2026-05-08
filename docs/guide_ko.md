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
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cuda]"  # Linux CUDA 환경
# 또는
pip install -e ".[dev]"       # macOS/CPU 환경
```

---

## 3. 모델 가중치 준비

HuggingFace에서 공식 가중치를 다운로드합니다.

| 모델명 | 키워드 | 역할 | 크기 |
| :--- | :--- | :--- | :--- |
| **AutoGaze** | `autogaze` | 패치 선택 (필수) | ~50 MB |
| **VideoMAE** | `videomae` | 보상 모델 (학습용) | ~2 GB |
| **NVILA-8B** | `nvila` | 통합 MLLM (데모용) | ~16 GB |

```bash
# 자동 다운로드 (weights/ 디렉토리에 저장)
bash scripts/download_models.sh weights autogaze
bash scripts/download_models.sh weights nvila
```

---

## 4. 추론 워크플로우 (Inference)

추론은 크게 두 가지 스크립트로 나뉩니다. 자세한 내용은 [추론 마스터 가이드](inference_guide.md)를 참고하세요.

### 4.1 가이즈 맵 분석 (`infer.py`)
가이즈 선택 결과를 시각화하거나 학습용 레이블을 생성할 때 사용합니다.

```bash
# 1. 25% 선택 vs 100% 전체 패치 비교 시각화
python -m autogaze.infer assets/example_input.mp4 --gazing-ratio 0.25 --compare-autogaze

# 2. 비율별 가이즈 변화 자동 스윕 (0.1 -> 1.0)
python -m autogaze.infer assets/example_input.mp4 --sweep-ratio --ratio-step 0.2
```

### 4.2 전체 MLLM 파이프라인 (`infer_full.py`)
AutoGaze를 NVILA나 Qwen 같은 모델에 통합하여 실제 질문 답변 성능을 측정합니다.

```bash
# NVILA 추론 (AutoGaze ON)
python autogaze/infer_full.py assets/example_input.mp4 --mllm nvila
```

---

## 5. 비디오 QA 벤치마크 (Video QA)

표준 벤치마크셋(VideoMME, MVBench 등)에서 AutoGaze의 성능을 검증합니다. 자세한 내용은 [평가 가이드](eval_guide.md)를 참고하세요.

### 5.1 ON/OFF 비교 테스트
AutoGaze 사용 여부에 따른 정확도와 속도 차이를 측정합니다.

```bash
# 마스터 스크립트로 ON + OFF 자동 실행
bash scripts/run_benchmarks.sh --tasks videomme --max-samples 100
```

### 5.2 수동 실행 (Python)
```bash
# AutoGaze ON (75% 토큰)
python -m autogaze.eval.run_benchmark --task videomme --mllm nvila --gazing-ratio 0.75

# AutoGaze OFF (기준선)
python -m autogaze.eval.run_benchmark --task videomme --mllm nvila --no-autogaze
```

---

## 6. 벤치마크 및 성능 평가 (CV Tasks)

다양한 하위 태스크(Depth, Detection, Segmentation)에 AutoGaze를 적용하여 효율성을 검증합니다.

### 주요 평가 도구
- **CLI**: `scripts/run_cv_tasks.py`를 통해 다양한 CV 모델 성능 측정.
- **Notebook**: `notebooks/10_autogaze_benchmark_ko.ipynb`에서 인터랙티브하게 실험.

자세한 메트릭 해석은 [벤치마크 가이드](benchmark_guide.md)를 확인하세요.

---

## 7. AutoGaze 통합 및 확장

기존의 ViT 기반 모델에 AutoGaze를 붙이는 방법은 두 가지입니다.

1.  **Hook Mode (Zero-shot)**: 모델 수정 없이 패치 임베딩 단계에서 마스킹. (정확도 테스트용)
2.  **Full Mode (Integrated)**: 토큰을 물리적으로 제거하여 Quadratic Speedup 달성. (성능 벤치마크용)

기술적 상세 구현은 [통합 가이드](integration_guide.md)에 Mermaid 다이어그램과 함께 설명되어 있습니다.

---

## 8. 학습 가이드 (Training)

AutoGaze는 2단계로 학습됩니다.

1.  **Stage 1 (NTP)**: Ground-truth 가이즈 시퀀스 모방 학습.
2.  **Stage 2 (RL)**: GRPO 알고리즘을 사용해 재구성 품질(Reconstruction Reward) 최적화.

자세한 파라미터 설명은 프로젝트 루트의 `TRAIN.md`를 참고하세요.

---

## 9. 자주 묻는 질문 (FAQ)

**Q: Mac에서 `flash_attn` 오류가 발생합니다.**  
A: macOS는 `flash_attn`을 지원하지 않습니다. 시스템이 자동으로 `sdpa`로 전환하므로 무시하셔도 되며, 설정 파일에서 `attn_mode`를 `sdpa`로 명시해주시면 좋습니다.

**Q: `gazing_pos` 인덱스는 어떻게 계산되나요?**  
A: `프레임 번호 * 프레임당 토큰 수 + 프레임 내 패치 번호`로 계산되는 전역 인덱스입니다.

**Q: 속도 향상이 체감되지 않습니다.**  
A: `Full Mode` (예: `nvila` 또는 `_full` 접미사가 붙은 러너)를 사용 중인지 확인하세요. `Hook Mode`는 연산량은 동일하고 결과만 마스킹하므로 속도 향상이 없습니다.

---
*문서 최종 갱신: 2026-05-08*  
*Gemini CLI를 통해 최신 구현 사항을 반영하여 업데이트되었습니다.*
