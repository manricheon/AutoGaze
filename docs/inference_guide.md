# AutoGaze 추론 가이드

AutoGaze 관련 두 가지 추론 스크립트와 멀티-MLLM 지원 구조를 설명합니다.

---

## 목차

1. [두 스크립트의 역할 분리](#1-두-스크립트의-역할-분리)
2. [infer.py — AutoGaze 전용 추출](#2-inferpy--autogaze-전용-추출)
3. [infer_full.py — 전체 파이프라인 QA](#3-infer_fullpy--전체-파이프라인-qa)
4. [MLLM 백엔드 선택](#4-mllm-백엔드-선택)
5. [AutoGaze ON / OFF 제어](#5-autogaze-on--off-제어)
6. [비교 모드 & Ratio Sweep](#6-비교-모드--ratio-sweep)
7. [프레임 샘플링](#7-프레임-샘플링)
8. [ViT 통합 방식](#8-vit-통합-방식)
9. [메모리 요건](#9-메모리-요건)
10. [빠른 참조](#10-빠른-참조)

---

## 1. 두 스크립트의 역할 분리

| 스크립트 | 역할 | 출력 |
|---------|------|------|
| `autogaze/infer.py` | AutoGaze 실행 — gaze map 추출·시각화·JSON 저장 | PNG / MP4 / NPZ / JSON |
| `autogaze/infer_full.py` | AutoGaze + ViT + MLLM 전체 파이프라인 — 비디오 QA | 텍스트 답변 + 타이밍 |

```
infer.py 파이프라인
  비디오 → AutoGaze → gaze_map (14×14 per frame) → 시각화/저장

infer_full.py 파이프라인
  비디오 → [AutoGaze →] ViT (SigLIP / Qwen ViT / V-JEPA2) → LLM → 답변
```

---

## 2. infer.py — AutoGaze 전용 추출

### 기본 사용법

```bash
# 단일 비디오, 전체 포맷 (json + viz + npy + video)
python -m autogaze.infer assets/example_input.mp4 --output-dir results/

# 시각화 MP4만
python -m autogaze.infer assets/example_input.mp4 --output-format video

# 디렉토리 전체 → JSON 레이블 생성 (NTP 학습 데이터)
python -m autogaze.infer /data/videos/ --output-format json

# 모든 프레임 처리 (16프레임 청크 분할)
python -m autogaze.infer assets/example_input.mp4 \
    --all-frames --output-format frames,video

# stride 샘플링 (매 10번째 프레임)
python -m autogaze.infer assets/example_input.mp4 \
    --stride 10 --output-format frames,video
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|-----|--------|------|
| `--gazing-ratio` | 0.75 | 프레임당 선택 패치 비율 (0~1) |
| `--output-format` | all | `json,viz,frames,video,npy` 중 선택 |
| `--all-frames` | false | 모든 프레임 처리 (청크 분할) |
| `--stride N` | — | 매 N번째 프레임 추출 |
| `--num-frames N` | 16 | 균일 샘플링 프레임 수 |
| `--model-path` | `weights/AutoGaze` | AutoGaze 가중치 경로 또는 HF ID |

### 출력 포맷

| 포맷 | 파일 | 설명 |
|------|------|------|
| `json` | `gazing_labels.json` | NTP 학습용 레이블 (gazing_pos per frame) |
| `viz` | `{stem}_gaze.png` | 전체 프레임 × 전체 스케일 그리드 |
| `frames` | `{stem}_frames/*.png` | 프레임별 PNG (원본 + 스케일별 오버레이) |
| `video` | `{stem}_gaze.mp4` | 원본 / 오버레이 / 히트맵 3분할 MP4 |
| `npy` | `{stem}_gaze.npz` | raw gaze mask 배열 (numpy) |

---

## 3. infer_full.py — 전체 파이프라인 QA

### 기본 사용법

```bash
# NVILA (기본) — AutoGaze ON
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila \
    --model-path weights/NVILA-8B-HD-Video \
    --autogaze-path weights/AutoGaze

# 질문 직접 지정
python autogaze/infer_full.py assets/example_input.mp4 \
    --question "What is the person doing?"

# AutoGaze OFF (기준선)
python autogaze/infer_full.py assets/example_input.mp4 \
    --no-autogaze

# Gaze map 시각화 PNG 저장
python autogaze/infer_full.py assets/example_input.mp4 \
    --save-gaze --output-dir results/gaze_viz/
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|-----|--------|------|
| `--mllm` | `nvila` | MLLM 백엔드 선택 (아래 표 참조) |
| `--model-path` | `weights/NVILA-8B-HD-Video` | MLLM 가중치 경로 또는 HF ID |
| `--autogaze-path` | `weights/AutoGaze` | AutoGaze 가중치 경로 |
| `--no-autogaze` | false | AutoGaze 비활성화 |
| `--gazing-ratio` | 0.75 | 패치 선택 비율 |
| `--compare-autogaze` | false | ON/OFF 비교 모드 |
| `--sweep-ratio` | false | ratio 단계별 비교 |
| `--ratio-step` | 0.25 | sweep 간격 |
| `--frames N` | 16 | 균일 샘플링 프레임 수 |
| `--stride N` | — | stride 샘플링 |
| `--max-new-tokens` | 256 | 최대 생성 토큰 수 |
| `--save-gaze` | false | Gaze 시각화 PNG 저장 |
| `--output-dir` | `results/infer_full` | 출력 디렉토리 |

---

## 4. MLLM 백엔드 선택

### 지원 백엔드

| `--mllm` 키 | 모델 | ViT 백본 | AutoGaze 통합 | MCQ 지원 |
|------------|------|---------|--------------|---------|
| `nvila` | NVILA-8B-HD-Video | SigLIP-L/14 | NVILAProcessor 내장 | ✓ |
| `qwen25vl` | Qwen2.5-VL-7B | Qwen ViT | zero-shot hook | ✓ |
| `qwen25vl_full` | Qwen2.5-VL-7B | Qwen ViT | class monkey-patch | ✓ |
| `vjepa2` | V-JEPA2 ViT-L | V-JEPA2 ViT | zero-shot hook | 특징 추출만 |
| `vjepa2_full` | V-JEPA2 ViT-L | V-JEPA2 ViT | class monkey-patch | 특징 추출만 |
| `vjepa2_llm` | V-JEPA2 + projector + LLM | V-JEPA2 ViT | class monkey-patch | ✓ (projector 학습 필요) |
| `siglip` | Vanilla HF SigLIP | SigLIP (원본) | zero-shot hook (per-frame) | 특징 추출만 |

> **`siglip` vs `nvila` SigLIP 비교**:
> `nvila`는 `autogaze/vision_encoders/siglip/`의 **수정된 SigLIP**을 사용합니다 — `(B,T,C,H,W)` 입력, 멀티스케일 패치, block-causal inter-frame attention, `gazing_info` 통합.  
> `siglip`은 **순수 HF `SiglipVisionModel`** — 프레임별 독립 처리, 단일 스케일. 
> NVILA 수정 버전과의 특징 추출 비교 연구에 유용합니다.

### 예시: Qwen2.5-VL

```bash
# HuggingFace에서 직접 로드 (hook 방식)
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm qwen25vl \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --autogaze-path weights/AutoGaze

# Full ViT 통합 방식 (per-temporal-chunk gaze map)
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm qwen25vl_full \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --autogaze-path weights/AutoGaze
```

### API로 직접 사용

```python
from autogaze.eval.models import load_runner

runner = load_runner(
    mllm          = "qwen25vl",
    model_path    = "Qwen/Qwen2.5-VL-7B-Instruct",
    autogaze_path = "weights/AutoGaze",
    gazing_ratio  = 0.75,
)

answer = runner.run(frames, "What is happening?", max_new_tokens=128)
```

### 예시: Vanilla SigLIP (`siglip`)

```bash
# 순수 HF SigLIP, AutoGaze 없이 특징 추출
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm siglip \
    --model-path google/siglip-so400m-patch14-224 \
    --no-autogaze

# SigLIP + AutoGaze zero-shot hook
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm siglip \
    --model-path google/siglip-so400m-patch14-224 \
    --autogaze-path weights/AutoGaze
```

```python
# Python API
from autogaze.eval.models import load_runner

# 기준선: AutoGaze 없음
runner = load_runner(
    mllm          = "siglip",
    model_path    = "google/siglip-so400m-patch14-224",
    autogaze_path = None,
    gazing_ratio  = 0.75,
)
feats = runner.encode_video(frames)   # (1, T*N, C)
# T frames × N patches/frame × C=1152

# AutoGaze ON
runner_ag = load_runner(
    mllm          = "siglip",
    model_path    = "google/siglip-so400m-patch14-224",
    autogaze_path = "weights/AutoGaze",
    gazing_ratio  = 0.75,
)
feats_ag = runner_ag.encode_video(frames)   # non-gazed patches zeroed out
```

### 예시: V-JEPA2 + LLM (`vjepa2_llm`)

`vjepa2_llm`은 V-JEPA2 ViT-L → `VJEPA2Projector` → 임의의 causal LLM으로 연결하는
**전체 MCQ 파이프라인**입니다.

**아키텍처**:
```
비디오 프레임
  ↓  AutoGaze  →  per-temporal-group gaze mask
V-JEPA2 ViT-L  →  (B, T_p × H_p × W_p, 1024)
  ↓  VJEPA2Projector
     temporal mean pool  →  (B, T_p, 1024)
     LayerNorm + 2-layer MLP  →  (B, T_p, lm_hidden)
  ↓  causal LLM  (e.g. Qwen2.5-7B-Instruct)
     inputs_embeds = [video_tokens | text_tokens]
MCQ 답변 생성
```

T_p = num_frames / tubelet_size (예: 16프레임 → T_p = 8).

**주의**: projector는 **별도 학습이 필요**합니다. `projector_path=None`이면 랜덤 초기화 projector를 사용하므로 출력이 무의미합니다.

```bash
# CLI — projector 학습 완료 후
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm vjepa2_llm \
    --model-path facebook/vjepa2-vitl-fpc64-256 \
    --autogaze-path weights/AutoGaze \
    --lm-path Qwen/Qwen2.5-7B-Instruct \
    --projector-path weights/vjepa2_projector \
    --prompt "What is the main activity? A) Running B) Cooking C) Reading D) Swimming"
```

```python
# Python API
from autogaze.eval.models import load_runner

runner = load_runner(
    mllm           = "vjepa2_llm",
    model_path     = "facebook/vjepa2-vitl-fpc64-256",
    autogaze_path  = "weights/AutoGaze",
    gazing_ratio   = 0.75,
    lm_path        = "Qwen/Qwen2.5-7B-Instruct",
    projector_path = "weights/vjepa2_projector",   # None → random init
)

answer = runner.run(frames, "What is happening? A) ... B) ...", max_new_tokens=16)
```

**Projector 학습 (개요)**:

```python
from autogaze.vision_encoders.vjepa2 import VJEPA2Projector
from transformers import AutoModelForCausalLM

lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
projector = VJEPA2Projector.new_for_lm(lm, vit_hidden=1024)

# V-JEPA2 + LLM 동결, projector만 학습
optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4)
# ... 학습 루프 ...

projector.save_pretrained("weights/vjepa2_projector/")
```

---

## 5. AutoGaze ON / OFF 제어

### CLI

```bash
# ON (기본)
python autogaze/infer_full.py video.mp4 --gazing-ratio 0.75

# OFF — autogaze_path=None 전달
python autogaze/infer_full.py video.mp4 --no-autogaze
```

### API

```python
# AutoGaze ON
runner_ag   = load_runner("nvila", ..., autogaze_path="weights/AutoGaze", gazing_ratio=0.75)

# AutoGaze OFF (autogaze_path=None)
runner_base = load_runner("nvila", ..., autogaze_path=None, gazing_ratio=1.0)
```

### NVILA 전용: `test_nvila.py` 상세 타이밍

`scripts/test_nvila.py`는 NVILA 전용으로 AutoGaze / ViT / LLM prefill / LLM decode 각 단계별 상세 타이밍을 제공합니다.

```bash
# AutoGaze ON/OFF 비교 + 상세 타이밍
python scripts/test_nvila.py --compare-autogaze

# ratio sweep (0.1 단계)
python scripts/test_nvila.py --sweep-ratio --ratio-step 0.1

# stride 샘플링 비교
python scripts/test_nvila.py --stride 5 --frames 32
```

---

## 6. 비교 모드 & Ratio Sweep

### AutoGaze ON/OFF 비교

```bash
# 두 러너를 자동으로 로드하고 나란히 비교
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila --compare-autogaze

# 출력 예시:
# ──────────────────────────────────────────────────
#   항목               AutoGaze ON    AutoGaze OFF
#   ─────────────────────────────────────────────────
#   AutoGaze              0.42s              —
#   ViT + LLM             3.21s          3.63s
#   전체                  3.63s          3.63s
#   절감률                10.5%
#   ─────────────────────────────────────────────────
#   [ON ]  A
#   [OFF]  A
```

### Ratio Sweep

```bash
# 0.25 단계로 ratio 변경하며 비교
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila --sweep-ratio --ratio-step 0.25

# 더 세밀한 sweep (0.1 단계)
python autogaze/infer_full.py assets/example_input.mp4 \
    --mllm nvila --sweep-ratio --ratio-step 0.1
```

---

## 7. 프레임 샘플링

| 방식 | 옵션 | 설명 | 비고 |
|------|------|------|------|
| 균일 샘플링 | `--frames 16` | 전체 영상에서 N개 linspace | 기본값 |
| Stride 샘플링 | `--stride 10` | 매 N번째 프레임 추출 | 16 배수 truncate |

```bash
# 32프레임 균일 샘플링
python autogaze/infer_full.py video.mp4 --frames 32

# 매 5번째 프레임 (stride)
python autogaze/infer_full.py video.mp4 --stride 5

# 두 방식 비교 (scripts/test_nvila.py 사용)
python scripts/test_nvila.py video.mp4 --frames 16 --stride 5
```

> **AutoGaze 요건**: 입력 프레임 수는 반드시 16의 배수여야 합니다.  
> `load_frames_stride()` 는 추출 후 16 배수로 자동 truncate합니다.

---

## 8. ViT 통합 방식

각 MLLM의 ViT에 AutoGaze를 통합하는 방식은 두 가지입니다:

### Hook 방식 (zero-shot, `qwen25vl` / `vjepa2`)

```
AutoGaze → 전체 프레임의 mean gaze map → patch_embed 출력 후 비선택 패치 zeroing
```
- 모델 수정 없음 — forward hook으로 주입
- 모든 시간 프레임이 동일한 gaze map 사용

### Full 통합 방식 (`qwen25vl_full` / `vjepa2_full`)

```
AutoGaze → 시간별 gaze map → class monkey-patch → 각 temporal chunk별 zeroing
```
- `model.visual.__class__` (Qwen) 또는 `model.encoder.__class__` (V-JEPA2) 교체
- 가중치 복사 없음 — forward() 동작만 오버라이드
- Temporal variation 반영 (시간별로 다른 gaze map 적용)

### NVILA (native 통합)

```
AutoGaze → gazing_pos, num_gazing_each_frame → SigLIP mask_with_gazing() → 선택 패치만 ViT 통과
```
- NVILAProcessor가 내장 처리
- 실제로 선택 패치만 ViT에 통과 (시퀀스 길이 감소)
- Block-causal 어텐션 마스크 적용 (프레임 간 단방향)

### 구현 파일 위치

| 파일 | 역할 |
|------|------|
| `autogaze/vision_encoders/siglip/modeling_siglip.py` | SigLIP + AutoGaze (NVILA용) |
| `autogaze/vision_encoders/qwen25vl/modeling_qwen25vl_ag.py` | Qwen2.5-VL ViT + AutoGaze (full 모드) |
| `autogaze/vision_encoders/vjepa2/modeling_vjepa2_ag.py` | V-JEPA2 인코더 + AutoGaze (full 모드) |
| `autogaze/vision_encoders/vjepa2/projector.py` | VJEPA2Projector (ViT→LLM 임베딩 변환) |
| `autogaze/eval/models.py` | MLLM 러너 레지스트리 (`VJEPA2LLMRunner` 포함) |

자세한 통합 방법은 `INTEGRATION.md` 참조.

---

## 9. 메모리 요건

| 모델 | VRAM (bfloat16) | 권장 GPU |
|------|----------------|---------|
| NVILA-8B | ~18 GB | A100 / H100 |
| Qwen2.5-VL-7B | ~16 GB | A100 / H100 |
| V-JEPA2 ViT-L (특징 추출) | ~4 GB | A40 이상 |
| V-JEPA2 + Qwen2.5-7B (`vjepa2_llm`) | ~18 GB | A100 / H100 |
| AutoGaze | ~50 MB | 어디서나 |

`device_map="auto"` 설정으로 다중 GPU 자동 분산됩니다.

**MPS (Apple Silicon)**:
- M1 Max/Ultra, M2/M3 Ultra (≥ 24 GB Unified Memory) 지원
- `torch.backends.mps.is_available()` 자동 감지

---

## 10. 빠른 참조

### 자주 사용하는 명령어

```bash
# ── AutoGaze 단독 ────────────────────────────────────────────────
# gaze map 추출 + MP4 시각화
python -m autogaze.infer video.mp4 --output-format video

# 모든 프레임 처리
python -m autogaze.infer video.mp4 --all-frames --output-format frames,video

# ── Full pipeline (NVILA) ────────────────────────────────────────
# 기본 추론 (AutoGaze ON)
python autogaze/infer_full.py video.mp4 --mllm nvila

# AutoGaze ON/OFF 비교
python autogaze/infer_full.py video.mp4 --compare-autogaze

# Ratio sweep
python autogaze/infer_full.py video.mp4 --sweep-ratio

# Gaze 시각화 저장
python autogaze/infer_full.py video.mp4 --save-gaze

# ── Full pipeline (Qwen2.5-VL) ──────────────────────────────────
# zero-shot hook 방식
python autogaze/infer_full.py video.mp4 \
    --mllm qwen25vl \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct

# full ViT 통합
python autogaze/infer_full.py video.mp4 \
    --mllm qwen25vl_full \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct

# ── NVILA 상세 타이밍 (test_nvila.py) ───────────────────────────
python scripts/test_nvila.py --compare-autogaze
python scripts/test_nvila.py --sweep-ratio --ratio-step 0.1

# ── 벤치마크 평가 ───────────────────────────────────────────────
bash scripts/run_benchmarks.sh --tasks videomme,mvbench --max-samples 100
python -m autogaze.eval.run_benchmark --task videomme --no-autogaze
```

### 파일 위치 요약

```
autogaze/
├── infer.py           AutoGaze 전용 추출 (gaze map + 시각화)
├── infer_full.py      전체 파이프라인 QA (AutoGaze + ViT + MLLM)
├── eval/
│   ├── models.py      MLLM 러너 레지스트리 (NVILARunner, Qwen25VLRunner, VJEPA2Runner)
│   ├── run_benchmark.py  벤치마크 루프
│   └── tasks.py       벤치마크 태스크 정의
└── vision_encoders/
    ├── siglip/        SigLIP + AutoGaze (NVILA)
    ├── qwen25vl/      Qwen2.5-VL ViT + AutoGaze
    └── vjepa2/        V-JEPA2 + AutoGaze

scripts/
├── test_nvila.py          NVILA 전용 단일 비디오 추론 + 상세 타이밍
├── run_benchmarks.sh      전체 벤치마크 실행 스크립트
└── run_inference.sh       infer.py 래퍼 스크립트

docs/
├── inference_guide.md     이 파일
├── eval_guide.md          벤치마크 평가 가이드
└── INTEGRATION.md         ViT AutoGaze 통합 기술 가이드 (프로젝트 루트)
```
