# AutoGaze Inference Guide (추론 가이드)

This guide covers all inference workflows in the AutoGaze project, from raw gaze map generation to full MLLM pipeline benchmarking.

이 가이드는 AutoGaze 프로젝트의 모든 추론 워크플로우(원시 가이즈 맵 생성부터 전체 MLLM 파이프라인 벤치마킹까지)를 다룹니다.

---

## 1. Overview (개요)

We provide two primary entry points for inference, both supporting comparative analysis.

두 가지 주요 추론 진입점을 제공하며, 두 스크립트 모두 비교 분석 기능을 지원합니다.

| Script | Primary Goal (주요 목적) | Key Outputs (주요 출력) |
| :--- | :--- | :--- |
| `autogaze/infer.py` | **Qualitative Gaze Analysis** (정성적 가이즈 분석) | Gaze maps, MP4 overlays, JSON labels |
| `autogaze/infer_full.py` | **Performance & Accuracy Benchmark** (성능 및 정확도 벤치마크) | MLLM answers, latency (ms), VRAM (GB) |

---

## 2. Standalone Gaze Inference (`infer.py`)

Use this to visualize what AutoGaze is "looking at" across different token budgets.

AutoGaze가 다양한 토큰 예산 내에서 무엇을 "보고 있는지" 시각화할 때 사용합니다.

### Options (옵션)

| Option | Description (설명) |
| :--- | :--- |
| `--compare-autogaze` | Compare the current ratio against the 100% full-patch baseline. (설정된 비율과 100% 기준선 비교) |
| `--sweep-ratio` | Automatically sweep through ratios to see gaze evolution. (비율별 가이즈 변화 시각화) |
| `--ratio-step` | Step size for the sweep (default: 0.25). (스윕 간격) |

### Usage Examples (사용 예시)

```bash
# 25% vs 100% 비교 시각화
python -m autogaze.infer assets/example_input.mp4 --gazing-ratio 0.25 --compare-autogaze

# 0.1 → 1.0 스윕 (간격 0.2)
python -m autogaze.infer assets/example_input.mp4 --sweep-ratio --ratio-step 0.2

# 전체 프레임 처리 (16-frame 청크 단위)
python -m autogaze.infer assets/example_input.mp4 --all-frames --output-format video
```

---

## 3. Full Pipeline MLLM Inference (`infer_full.py`)

This script benchmarks the entire stack: **AutoGaze → ViT → MLLM**.

전체 스택(AutoGaze → ViT → MLLM)의 성능을 측정합니다.

### Supported Runners (지원 러너)

| Runner Key | Model | Integration Mode | Note |
| :--- | :--- | :--- | :--- |
| `nvila` | NVILA-8B-HD-Video | **Full (Native)** | Processor-integrated AutoGaze; `--no-autogaze`로 기준선 실행 가능 |
| `qwen25vl` | Qwen2.5-VL-7B | Hook (Zero-shot) | 빠른 검증용 |
| `qwen25vl_full` | Qwen2.5-VL-7B | **Full** | 최대 효율 벤치마크 |
| `vjepa2_llm` | V-JEPA2 ViT + Qwen2.5-7B LM | **Full** | 대체 ViT 테스트 |
| `nvila_vjepa2` | V-JEPA2 ViT + NVILA LLM | **Full** | V-JEPA2 인코더 + NVILA 언어 모델 조합 |

### Usage Examples (사용 예시)

```bash
# NVILA — AutoGaze ON (비교)
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila \
    --compare-autogaze

# NVILA — ratio 스윕
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila \
    --sweep-ratio --ratio-step 0.25

# Qwen2.5-VL (Full mode)
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm qwen25vl_full \
    --model-path weights/Qwen2.5-VL-7B-Instruct \
    --sweep-ratio --ratio-step 0.25

# V-JEPA2 + NVILA LLM
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila_vjepa2 \
    --model-path weights/NVILA-8B-HD-Video \
    --vjepa2-path weights/vjepa2-vitl-fpc64-256
```

---

## 4. Integration Modes: Hook vs. Full (통합 모드)

| Mode (모드) | Mechanism (작동 방식) | Best Use Case (권장 용도) |
| :--- | :--- | :--- |
| **Hook** | Zeroes tokens ($N_{all}$ remains). (토큰을 0으로 채움) | Accuracy validation for new models. (정확도 검증) |
| **Full** | Removes tokens ($N_{gazed}$ only). (토큰을 물리적으로 제거) | Efficiency benchmarks — speed/VRAM. (효율성 측정) |

Hook Mode에서 정확도가 잘 나온다면 Full Mode에서도 동일하거나 더 좋은 결과가 나옵니다. 속도 향상은 Full Mode에서만 확인 가능합니다.

---

## 5. Interactive Notebooks (인터랙티브 노트북)

| 노트북 | 내용 |
| :--- | :--- |
| `notebooks/12_inference_full_ko.ipynb` | 전체 MLLM 추론 파이프라인 (nvila, qwen25vl, nvila_vjepa2 등) |
| `notebooks/10_autogaze_benchmark_ko.ipynb` | AutoGaze ON/OFF 성능 비교, ratio 스윕 |
| `notebooks/11_video_qa_benchmark_ko.ipynb` | Video QA 벤치마크 결과 분석 |
| `notebooks/autogaze_inference_suite.ipynb` | 백본 교체(SigLIP → V-JEPA2) 실험 |

---

## 6. Key Source Files (주요 소스 파일)

| Purpose | Path |
| :--- | :--- |
| Gaze-only inference | `autogaze/infer.py` |
| Full MLLM inference | `autogaze/infer_full.py` |
| MLLM runner registry | `autogaze/eval/models.py` |
| Benchmark entry point | `autogaze/eval/run_benchmark.py` |
| Task definitions | `autogaze/eval/tasks.py` |
| Main inference notebook | `notebooks/12_inference_full_ko.ipynb` |
