# AutoGaze 한글 사용 가이드

> **AutoGaze** — CVPR 2026 (NVIDIA)  
> 비디오의 중요한 패치를 자동으로 선택해 Vision Encoder / MLLM의 연산량을 줄이는 Autoregressive Gaze 모델

---

## 목차

1. [개요](#1-개요)
   - 1.1 [전체 파이프라인 다이어그램](#11-전체-파이프라인-다이어그램)
2. [환경 설정](#2-환경-설정)
3. [모델 가중치 다운로드](#3-모델-가중치-다운로드)
4. [인퍼런스 (Inference)](#4-인퍼런스-inference)
   - 4.1 [기본 사용법 — CLI 스크립트](#41-기본-사용법--cli-스크립트)
   - 4.2 [전체 프레임 처리 (--all-frames)](#42-전체-프레임-처리---all-frames)
   - 4.3 [출력 포맷 상세](#43-출력-포맷-상세)
   - 4.4 [Python API로 직접 사용](#44-python-api로-직접-사용)
   - 4.5 [다양한 해상도·패치 크기 지원](#45-다양한-해상도패치-크기-지원)
   - 4.6 [스트리밍 비디오 처리](#46-스트리밍-비디오-처리)
   - 4.7 [Vision Encoder 연동 (SigLIP)](#47-vision-encoder-연동-siglip)
5. [학습 (Training)](#5-학습-training)
   - 5.1 [학습 데이터 다운로드](#51-학습-데이터-다운로드)
   - 5.2 [Stage 1 — NTP 사전학습](#52-stage-1--ntp-사전학습)
   - 5.3 [Stage 2 — GRPO RL 후학습](#53-stage-2--grpo-rl-후학습)
   - 5.4 [단일 GPU에서 테스트 학습 (Mac / 소규모 실험)](#54-단일-gpu에서-테스트-학습-mac--소규모-실험)
   - 5.5 [VideoMAE 도메인 적응 및 Joint Fine-tuning](#55-videomae-도메인-적응-및-joint-fine-tuning)
6. [파라미터 상세 설명](#6-파라미터-상세-설명)
7. [체크포인트 관리](#7-체크포인트-관리)
8. [자주 묻는 질문 / 트러블슈팅](#8-자주-묻는-질문--트러블슈팅)

---

## 1. 개요

### AutoGaze가 하는 일

AutoGaze는 비디오의 각 프레임에서 **어떤 패치(patch)를 봐야 할지**를 Autoregressive하게 예측합니다.  
선택된 패치만 SigLIP / DINOv2 같은 Vision Encoder에 전달하면, 전체 패치를 처리하는 것 대비 **연산량을 대폭 줄이면서** 비슷한 성능을 유지할 수 있습니다.

### 아키텍처 요약

```text
비디오 프레임
    │
    ▼
ConvNeXt Vision Encoder  ← 멀티스케일 패치 추출 (32 / 64 / 112 / 224 px)
    │
    ▼
Connector (MLP)
    │
    ▼
LLaMA 기반 AR Decoder  ← 어떤 패치를 볼지 순서대로 예측
    │
    ▼
gazing_pos / gazing_mask  → 이 인덱스만 Vision Encoder에 전달
```

### 2단계 학습 파이프라인

| 단계 | 방법 | 목적 |
| --- | --- | --- |
| Stage 1 | NTP (Next Token Prediction) | GT 가이즈 시퀀스를 학습해 기본 능력 습득 |
| Stage 2 | GRPO RL (재건 보상) | VideoMAE 재건 품질을 보상으로 삼아 더 나은 가이즈 전략 탐색 |

### 모델 크기 비교

| 모델 | 파라미터 | 파일 크기 | 역할 |
| --- | --- | --- | --- |
| **AutoGaze** | **3M** | ~50 MB | 패치 선택 (inference + training) |
| **VideoMAE** (ViT-L encoder) | ~307M | ~2 GB | 재건 보상 모델 (training 전용) |
| **VideoMAE** (MAE decoder) | ~8M | (포함) | 패치 → 전체 프레임 복원 |

AutoGaze는 VideoMAE의 약 **100분의 1** 크기입니다. 배포(inference) 시에는 AutoGaze만 필요합니다.

### VideoMAE 필요 시점 요약

| 상황 | VideoMAE 필요? | 이유 |
| --- | --- | --- |
| 패치 선택 인퍼런스 | **불필요** | AutoGaze 자체에 `task_loss_prediction_head` 내장 |
| `task_loss_requirement` 조기 종료 | **불필요** | 위와 동일 — 학습 중 내재화된 예측 헤드 사용 |
| 복원 영상 생성 | 필요 | MAE 디코더로 선택 패치 → 전체 프레임 복원 |
| Stage 1 NTP 학습 | 필요 | `task_loss_prediction_head` 학습용 GT loss 제공 (frozen) |
| Stage 2 RL 학습 | **필수** | reward = `-reconstruction_loss` (frozen) |

`task_loss_requirement`가 VideoMAE 없이 동작하는 원리: AutoGaze 내부의 `task_loss_prediction_head`가 "이 패치까지 선택하면 재건 손실이 얼마일지"를 스텝마다 예측합니다. 이 헤드는 Stage 1/2 학습 중 VideoMAE의 실제 loss를 정답으로 학습됩니다.

---

### 1.1 전체 파이프라인 다이어그램

#### 범례

| 기호 | 의미 |
| --- | --- |
| ✏️ | 학습/업데이트 대상 (trainable) |
| 🔒 | 동결 상태 (frozen) |
| — | 해당 단계에서 미사용 |

---

#### Inference (패치 선택 → Vision Encoder → MLLM)

```text
비디오 입력
    │
    ▼
┌─────────────────────────────────────┐
│  AutoGaze 🔒  (3M 파라미터)          │  ← AR 디코더, 패치를 순서대로 선택
│  ConvNeXt → Connector → LLaMA AR   │
└───────────────┬─────────────────────┘
                │  gazing_pos / gazing_mask
                │  (전체 패치의 ~25 % 선택)
                ▼
┌─────────────────────────────────────┐
│  SigLIP / DINOv2 / ViT 🔒           │  ← 선택된 패치만 인코딩 (연산 절감)
└───────────────┬─────────────────────┘
                │  patch features (N_gazed, D)
                ▼
┌─────────────────────────────────────┐
│  MLLM / Downstream Task 🔒          │  ← 비디오 QA, 분류, 캡셔닝 등
└─────────────────────────────────────┘

    * VideoMAE 불필요
    * task_loss_requirement 조기 종료도 내부 task_loss_prediction_head 사용
```

---

#### Stage 1 — NTP 사전학습

```text
비디오 + gazing_labels.json (GT)
    │
    ├─────────────────────────────────────────────────────┐
    ▼                                                     ▼
┌───────────────────────────────┐         ┌──────────────────────────────┐
│  AutoGaze ✏️  (3M)             │         │  VideoMAE 🔒  (~315M)         │
│  → 예측 gazing 시퀀스          │         │  → GT reconstruction loss    │
│  → task_loss_prediction_head  │         │    (각 스텝의 실제 재건 손실)  │
└───────────────┬───────────────┘         └──────────────┬───────────────┘
                │                                        │
                └──────────────┬─────────────────────────┘
                               ▼
                  NTP Cross-Entropy Loss
                  + task_loss_prediction MSE Loss
                               │
                               ▼
                       AutoGaze 가중치 업데이트
```

---

#### Stage 2 — GRPO RL 후학습

```text
비디오  (GT 레이블 불필요)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  AutoGaze ✏️   G=4~12 개의 가이즈 시퀀스 샘플링 (GRPO)    │
└──────────────────────────────┬───────────────────────────┘
                               │  G × gazing_pos 시퀀스
                               ▼
┌──────────────────────────────────────────────────────────┐
│  VideoMAE 🔒   각 시퀀스의 MAE reconstruction loss 계산   │
└──────────────────────────────┬───────────────────────────┘
                               │  reward = -reconstruction_loss
                               ▼
              GRPO Advantage 계산 (그룹 내 상대 보상)
                               │
                               ▼
                 discount_factor γ=0.995 로 스텝별 가중치 부여
                               │
                               ▼
                       AutoGaze 가중치 업데이트
```

---

#### 도메인 적응 — 단계별 권장 전략

```text
Step 1 ─ VideoMAE 도메인 적응  (AutoGaze 미사용)
──────────────────────────────────────────────────
새 도메인 비디오
    │
    ▼
┌──────────────────────────────────────────────────┐
│  VideoMAE ✏️  (train_task=True, train_gaze=False) │
│  → MAE self-supervised reconstruction 학습        │
└──────────────────────────────────────────────────┘
    │
    ▼
exps/videomae_domain_adapt/checkpoint_latest_task


Step 2 ─ AutoGaze RL  (적응된 VideoMAE를 reward 모델로)
──────────────────────────────────────────────────────
새 도메인 비디오
    │
    ▼
┌──────────────────────────────────────────────────┐
│  AutoGaze ✏️   (train_gaze=True)                  │
│  → G개 시퀀스 샘플링                               │
└──────────────────────────────┬───────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────┐
│  VideoMAE 🔒  (Step 1에서 적응된 가중치)           │
│  → 도메인-특화 reconstruction reward 제공         │
└──────────────────────────────────────────────────┘
                               │
                               ▼
                       AutoGaze 가중치 업데이트
```

---

#### 컴포넌트별 역할 요약표

| 컴포넌트 | 파라미터 | Inference | Stage 1 NTP | Stage 2 RL | VideoMAE 적응 Step 1 | VideoMAE 적응 Step 2 |
| --- | ---: | :---: | :---: | :---: | :---: | :---: |
| **AutoGaze** (AR decoder) | 3M | 🔒 | ✏️ | ✏️ | — | ✏️ |
| **VideoMAE** (ViT-L encoder) | ~307M | — | 🔒 | 🔒 | ✏️ | 🔒 |
| **VideoMAE** (MAE decoder) | ~8M | — | 🔒 | 🔒 | ✏️ | 🔒 |
| **SigLIP / DINOv2 / ViT** | 수백M | 🔒 | — | — | — | — |
| **MLLM** | 수십~수백B | 🔒 | — | — | — | — |

> **핵심 요약**: AutoGaze와 VideoMAE는 학습 시 항상 짝을 이루지만, **배포(inference) 시에는 AutoGaze(3M)만 필요**합니다.  
> 복원 영상 생성이 목적일 때만 VideoMAE decoder를 추가로 불러오면 됩니다 (`05_reconstruction_ko.ipynb` 참고).

---

## 2. 환경 설정

### 사전 요구사항

- Python 3.11
- macOS (MPS), Linux (CUDA), 또는 CPU
- Git

### 설치 순서

```bash
# 1. 저장소 클론
git clone <repo-url>
cd AutoGaze

# 2. Python 3.11 가상환경 생성
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 기본 의존성 설치 (CPU / MPS / CUDA 공통)
pip install -e ".[dev]"

# 4. (선택) CUDA 환경에서 flash_attn 설치
pip install -e ".[cuda]"

# 5. HuggingFace 로그인 (모델 다운로드에 필요)
huggingface-cli login
```

> **macOS 주의사항**  
> `flash_attn`은 macOS를 지원하지 않습니다. 이 프로젝트는 Mac에서 자동으로 `sdpa` (Scaled Dot-Product Attention)로 전환됩니다.

---

## 3. 모델 가중치 다운로드

### 한 번에 다운로드 (권장)

```bash
bash scripts/download_models.sh
```

위 스크립트는 아래 두 모델을 `weights/` 디렉터리에 저장합니다.

| 모델 | 저장 경로 | 크기 | 필요 시점 |
| --- | --- | --- | --- |
| `nvidia/AutoGaze` | `weights/AutoGaze/` | ~50 MB | 인퍼런스 + 학습 (항상 필요) |
| `bfshi/VideoMAE_AutoGaze` | `weights/VideoMAE_AutoGaze/` | ~2 GB | 학습 시에만 필요 (인퍼런스 불필요) |

> 패치 선택 인퍼런스만 할 경우 VideoMAE를 다운로드하지 않아도 됩니다.

### 수동 다운로드

```bash
# AutoGaze 모델
huggingface-cli download nvidia/AutoGaze --local-dir weights/AutoGaze

# VideoMAE 재건 모델 (~2 GB)
huggingface-cli download bfshi/VideoMAE_AutoGaze --local-dir weights/VideoMAE_AutoGaze
```

### 다운로드 후 확인

```text
weights/
├── AutoGaze/
│   ├── config.json
│   ├── model.safetensors (또는 pytorch_model.bin)
│   └── preprocessor_config.json
└── VideoMAE_AutoGaze/
    ├── videomae.pt          ← Stage 1/2 학습에 필요한 핵심 가중치
    └── config.yaml
```

---

## 4. 인퍼런스 (Inference)

### 4.1 기본 사용법 — CLI 스크립트

```bash
# 단일 비디오, 모든 출력 포맷 (json + viz + frames + video + npy)
python -m autogaze.infer assets/example_input.mp4 --output-dir results/

# 또는 래퍼 스크립트 사용
bash scripts/run_inference.sh assets/example_input.mp4 results/
```

#### 주요 옵션 요약

```bash
python -m autogaze.infer <입력> \
    --model-path    weights/AutoGaze   # HF ID 또는 로컬 경로 (기본: nvidia/AutoGaze)
    --output-dir    results/           # 결과 저장 디렉터리 (기본: results/)
    --output-format frames,video       # 출력 포맷 (기본: all)
    --gazing-ratio  0.75               # 최대 가이즈 비율 (기본: 0.75)
    --task-loss-requirement 0.7        # 재건 품질 임계값 — 이 이하면 조기 종료 (기본: 0.7)
    --no-task-loss-requirement         # 조기 종료 비활성화 (gazing-ratio만 사용)
    --num-frames    16                 # 샘플링 프레임 수 (기본: 16)
    --all-frames                       # 전체 프레임 처리 (청크 단위, 아래 참고)
    --chunk-size    16                 # --all-frames 시 청크 크기 (기본: 16)
    --video-fps     4.0                # 출력 MP4 FPS (기본: 4.0)
```

---

### 4.2 전체 프레임 처리 (--all-frames)

기본 모드에서는 비디오에서 16프레임을 균등 샘플링합니다.  
`--all-frames`를 사용하면 **비디오의 모든 프레임**에 대해 AutoGaze를 실행합니다.

```bash
# 64프레임 비디오를 4개 청크로 나눠 전부 처리 → per-frame PNG + MP4
python -m autogaze.infer assets/example_input.mp4 \
    --all-frames \
    --output-format frames,video \
    --output-dir results/

# 또는 래퍼 스크립트
bash scripts/run_inference.sh assets/example_input.mp4 results/ \
    --all-frames --output-format frames,video
```

#### 동작 방식

```text
전체 64프레임
    │
    ├── chunk 0: frames 0~15  → AutoGaze 실행 → mask/pos
    ├── chunk 1: frames 16~31 → AutoGaze 실행 → mask/pos (offset 추가)
    ├── chunk 2: frames 32~47 → AutoGaze 실행 → mask/pos (offset 추가)
    └── chunk 3: frames 48~63 → AutoGaze 실행 → mask/pos (offset 추가)
                                                     │
                                             결과 병합 (merge)
                                                     │
                                        gazing_mask: (1, 64, N_scale)
                                        gazing_pos: 전역 인덱스 기준
```

마지막 청크가 `chunk_size`보다 짧으면 자동으로 블랙 프레임으로 패딩 후 처리하고, 결과에서는 실제 프레임만 유지합니다.

> **팁**: 프레임 수가 많을 때 `viz` (전체 그리드 PNG)는 매우 넓어질 수 있습니다. 32프레임 초과 시 경고가 출력되며, `frames,video` 포맷을 권장합니다.

---

### 4.3 출력 포맷 상세

`--output-format` 에 쉼표 구분으로 조합하거나 `all`을 지정합니다.

| 포맷 | 출력 파일 | 설명 |
| --- | --- | --- |
| `json` | `results/gazing_labels.json` | NTP 학습 레이블 포맷. 프레임별 가이즈된 패치 인덱스 저장 |
| `viz` | `results/{stem}_gaze.png` | 전체 그리드 PNG. 행 = [원본, scale32, scale64, scale112, scale224], 열 = 프레임 |
| `frames` | `results/{stem}_frames/frame_NNN.png` | 프레임별 PNG. 1행 = [원본 \| scale32 \| scale64 \| scale112 \| scale224] |
| `video` | `results/{stem}_gaze.mp4` | MP4 영상. 3-패널: [원본 \| 가이즈 오버레이 \| 멀티스케일 히트맵] |
| `npy` | `results/{stem}_gaze.npz` | 압축 numpy 아카이브. 원시 배열 전체 저장 |

#### NPY 파일 불러오기

```python
import numpy as np

data = np.load("results/example_input_gaze.npz")
print(data.files)
# ['gazing_pos', 'if_padded_gazing', 'num_gazing_each_frame', 'scales',
#  'scale_32', 'scale_64', 'scale_112', 'scale_224']

# scale_224: (T, N_patches) 형태의 boolean 마스크
mask_224 = data["scale_224"]    # 예: (16, 196)
print(mask_224.shape, mask_224.sum(axis=1))  # 프레임별 선택된 패치 수
```

#### JSON 포맷 예시 (NTP 학습 레이블)

```json
{
  "dataset/train/video.mp4": {
    "gazing_pos": [[12, 45, 78], [3, 29], ...],
    "task_losses": [[0.0, 0.0, 0.0], [0.0, 0.0], ...]
  }
}
```

각 프레임의 `gazing_pos`는 해당 프레임 내 패치 인덱스 (0-based, 프레임 내 상대 좌표)입니다.

---

### 4.4 Python API로 직접 사용

```python
import av
import torch
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

# 모델 및 프리프로세서 로드
transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze")
model = AutoGaze.from_pretrained("nvidia/AutoGaze")
model.eval()

# 비디오 로드 (처음 16프레임)
container = av.open("assets/example_input.mp4")
indices = list(range(model.config.max_num_frames))   # 기본 max_num_frames = 16
raw_video = read_video_pyav(container, indices)
container.close()

# 전처리
video_input = transform_video_for_pytorch(raw_video, transform)  # (T, C, H, W)
video_input = video_input[None]  # (1, T, C, H, W)

# 인퍼런스
with torch.inference_mode():
    gaze_outputs = model(
        {"video": video_input},
        gazing_ratio=0.75,           # 최대 가이즈 비율
        task_loss_requirement=0.7,   # 재건 품질 임계값
    )

# 결과 확인
print(gaze_outputs['gazing_pos'].shape)         # (1, total_tokens)
print(gaze_outputs['if_padded_gazing'].shape)   # (1, total_tokens) — True=패딩
print((~gaze_outputs['if_padded_gazing']).sum()) # 실제 가이즈된 패치 수
print(gaze_outputs['num_gazing_each_frame'])     # 프레임별 가이즈 수 (패딩 포함)
```

#### 출력 변수 해설

| 변수 | 형태 | 설명 |
| --- | --- | --- |
| `gazing_pos` | `(B, N)` | 가이즈된 패치의 **전역** 토큰 인덱스. 프레임 `t`의 패치 `k`는 `t * num_tokens_per_frame + k` |
| `if_padded_gazing` | `(B, N)` bool | `True`이면 패딩(더미) 가이즈 — 무시해야 함 |
| `num_gazing_each_frame` | `(T,)` | 프레임별 가이즈 토큰 수 (패딩 포함) |
| `gazing_mask` | `list[(B, T, N_scale)]` | 스케일별 per-frame 가이즈 마스크 (boolean) |
| `num_vision_tokens_each_frame` | int | 프레임당 전체 비전 토큰 수 (265 기본) |

---

### 4.5 다양한 해상도·패치 크기 지원

AutoGaze는 224×224 / 16×16 패치를 기준으로 학습되었지만,  
**`target_scales`와 `target_patch_size`**를 넘기면 다른 해상도에도 적용할 수 있습니다.

```python
# 예: SigLIP2-SO400M (384×384, 14×14 패치) 지원
# 384는 14로 나눠지지 않으므로 392로 올림
transform_392 = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze", size=(392, 392))
video_input_392 = transform_video_for_pytorch(raw_video, transform_392)[None]

with torch.inference_mode():
    gaze_outputs_392 = model(
        {"video": video_input_392},
        gazing_ratio=0.75,
        task_loss_requirement=0.7,
        target_scales=[56, 112, 196, 392],  # 4개 스케일 유지 (동일 개수)
        target_patch_size=14,
    )
```

> **규칙**: 스케일 개수는 반드시 학습 시 사용한 것과 같아야 합니다 (기본 4개).  
> 비율만 조정하면 되므로 `[56, 112, 196, 392]`처럼 2배씩 증가하는 구조를 사용합니다.

---

### 4.6 스트리밍 비디오 처리

AutoGaze의 AR 디코더는 프레임 차원에서 **인과적(causal)**으로 동작합니다.  
이를 활용해 실시간 스트리밍 비디오에서 KV 캐시를 사용해 한 프레임씩 처리할 수 있습니다.

```python
past_inputs_embeds = None
past_attention_mask = None
past_key_values = None
past_conv_values = None
streaming_outputs = []

for t in range(video_input.shape[1]):   # 프레임 루프
    frame_t = video_input[:, t:t+1]    # (1, 1, C, H, W)
    out_t = model(
        {"video": frame_t},
        gazing_ratio=0.75,
        generate_only=True,
        use_cache=True,
        past_key_values=past_key_values,
        past_inputs_embeds=past_inputs_embeds,
        past_attention_mask=past_attention_mask,
        past_conv_values=past_conv_values,
    )
    streaming_outputs.append(out_t)
    # 다음 프레임을 위해 캐시 업데이트
    past_key_values = out_t['past_key_values']
    past_inputs_embeds = out_t['past_input_embeds']
    past_attention_mask = out_t['past_attention_mask']
    past_conv_values = out_t['past_conv_values']

# 전체 가이즈 위치 수집 (스트리밍 모드에서는 프레임 내 상대 인덱스 → 전역으로 변환)
streaming_pos = [
    out['gazing_pos'] + model.num_vision_tokens_each_frame * t
    for t, out in enumerate(streaming_outputs)
]
streaming_pos = torch.cat(streaming_pos, dim=1)
```

> **주의**: 스트리밍 모드의 `gazing_pos`는 **프레임 내 상대 인덱스**입니다.  
> 전역 인덱스로 변환하려면 `t * num_vision_tokens_each_frame`을 더해야 합니다.

---

### 4.7 Vision Encoder 연동 (SigLIP)

AutoGaze가 선택한 패치만 SigLIP에 전달해 효율적인 인코딩을 수행합니다.

```python
from transformers import AutoImageProcessor
from autogaze.vision_encoders.siglip import SiglipVisionModel  # AutoGaze 커스텀 버전

# SigLIP 모델 로드 (멀티스케일 지원 버전)
siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
siglip_model = SiglipVisionModel.from_pretrained(
    "google/siglip2-base-patch16-224",
    scales=model.config.scales,         # "32+64+112+224"
    attn_implementation="sdpa",
)

# SigLIP 전처리
video_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None]

# 가이즈된 패치만 인코딩
siglip_out = siglip_model(video_siglip, gazing_info=gaze_outputs)
print(siglip_out.last_hidden_state.shape)  # (1, N_gazed, 768) — 패딩 포함

# 패딩 제거 (실제 가이즈 피처만 남김)
features = [
    feat[~pad]
    for feat, pad in zip(
        siglip_out.last_hidden_state,
        gaze_outputs['if_padded_gazing']
    )
]
# features: 배치별 가변 길이 리스트 [(N_real_0, 768), (N_real_1, 768), ...]
```

---

## 5. 학습 (Training)

### 5.1 학습 데이터 다운로드

#### 전체 데이터 다운로드 (~646 GB)

```bash
bash scripts/download_data.sh
```

#### 일부 서브셋만 다운로드 (테스트용)

```bash
# InternVid만 다운로드 (~130 GB)
bash scripts/download_data.sh InternVid

# 사용 가능한 서브셋: InternVid, 100DoH, Ego4D, scanning_SAM, scanning_idl
```

#### 예상 디렉터리 구조

```text
data/AutoGaze-Training-Data/
├── InternVid_res448_250K/
│   ├── train/  ← .mp4 파일들
│   └── val/
├── 100DoH_res448_250K/
│   ├── train/
│   └── val/
├── Ego4D_res448_250K/
│   ├── train/
│   └── val/
├── scanning_SAM_res448_50K/
│   ├── train/
│   └── val/
├── scanning_idl_res448_50K/
│   ├── train/
│   └── val/
└── gazing_labels.json   ← Stage 1 NTP 학습에 필요한 GT 가이즈 레이블
```

---

### 5.2 Stage 1 — NTP 사전학습

AutoGaze가 GT 가이즈 시퀀스를 모방하도록 Next Token Prediction으로 학습합니다.

#### 단일 GPU (소규모 실험)

```bash
bash scripts/train_ntp_single_gpu.sh \
    "data/AutoGaze-Training-Data/InternVid_res448_250K" \
    weights/VideoMAE_AutoGaze/videomae.pt
```

#### 다중 GPU (8 GPU, 논문 설정)

```bash
bash scripts/train_ntp_multi_gpu.sh \
    "data/AutoGaze-Training-Data/InternVid_res448_250K,\
data/AutoGaze-Training-Data/100DoH_res448_250K,\
data/AutoGaze-Training-Data/Ego4D_res448_250K,\
data/AutoGaze-Training-Data/scanning_SAM_res448_50K,\
data/AutoGaze-Training-Data/scanning_idl_res448_50K" \
    weights/VideoMAE_AutoGaze/videomae.pt
```

#### 핵심 설정 값 (NTP)

| 항목 | 단일 GPU | 8 GPU (논문) |
| --- | --- | --- |
| `batch_size` | 32 | 1024 |
| `per_gpu_max_batch_size` | 4 | 32 |
| `n_epochs` | 150 | 150 |
| `lr` | 5e-4 | 5e-4 |
| `gazing_ratio` | 0.1 | 0.1 |
| 체크포인트 저장 | 100 스텝마다 | 500 스텝마다 |

체크포인트는 `exps/ntp_single_gpu/` 또는 `exps/ntp_8gpu/` 아래에 저장됩니다.

---

### 5.3 Stage 2 — GRPO RL 후학습

Stage 1으로 학습된 체크포인트를 초기 정책으로 삼아, VideoMAE 재건 품질을 보상으로 GRPO RL을 수행합니다.

#### 단일 GPU

```bash
bash scripts/train_rl_single_gpu.sh \
    "data/AutoGaze-Training-Data/InternVid_res448_250K" \
    weights/VideoMAE_AutoGaze/videomae.pt \
    exps/ntp_single_gpu/checkpoint_latest_gaze
```

#### 다중 GPU (논문 설정)

```bash
bash scripts/train_rl_multi_gpu.sh \
    "data/AutoGaze-Training-Data/InternVid_res448_250K,..." \
    weights/VideoMAE_AutoGaze/videomae.pt \
    exps/ntp_8gpu/checkpoint_latest_gaze
```

#### 핵심 설정 값 (RL)

| 항목 | 단일 GPU | 다중 GPU (논문) |
| --- | --- | --- |
| `batch_size` | 8 | 64 |
| `per_gpu_max_batch_size` | 2 | 2 |
| `group_size` | 4 | 12 |
| `n_epochs` | 1 | 1 |
| `gazing_ratio` | 0.75 | 0.75 |
| `discount_factor` | 0.995 | 0.995 |
| 검증 주기 | 200 스텝 | 1000 스텝 |

---

### 5.4 단일 GPU에서 테스트 학습 (Mac / 소규모 실험)

Mac (MPS) 또는 단일 GPU에서 코드가 정상 동작하는지 확인하는 최소 실험입니다.

```bash
# 1. 소규모 데이터로 NTP 테스트
python -m autogaze.train \
    --config-name video_folder_video_mae_reconstruction_ar_gaze_ntp \
    dataset.root="'data/AutoGaze-Training-Data/InternVid_res448_250K'" \
    dataset.gt_gazing_pos_paths.train="'data/AutoGaze-Training-Data/gazing_labels.json'" \
    trainer.batch_size=4 \
    trainer.per_gpu_max_batch_size=2 \
    trainer.n_epochs=1 \
    trainer.val_nsteps=50 \
    trainer.save_nsteps=50 \
    trainer.task_weights=weights/VideoMAE_AutoGaze/videomae.pt \
    trainer.exp_name=test_ntp_mac

# 2. NTP 결과로 RL 테스트
python -m autogaze.train \
    --config-name video_folder_video_mae_reconstruction_ar_gaze_grpo \
    dataset.root="'data/AutoGaze-Training-Data/InternVid_res448_250K'" \
    algorithm.group_size=2 \
    trainer.batch_size=4 \
    trainer.per_gpu_max_batch_size=1 \
    trainer.n_epochs=1 \
    trainer.val_nsteps=20 \
    trainer.save_nsteps=20 \
    trainer.task_weights=weights/VideoMAE_AutoGaze/videomae.pt \
    trainer.gaze_weights=exps/test_ntp_mac/checkpoint_latest_gaze \
    trainer.exp_name=test_rl_mac
```

---

### 5.5 VideoMAE 도메인 적응 및 Joint Fine-tuning

#### VideoMAE도 함께 튜닝할 수 있나요?

기본 학습 설정에서는 VideoMAE가 **동결(frozen)** 상태입니다 (`trainer.train_task=False`).  
하지만 새 도메인(의료·위성·공장 등)에서는 VideoMAE도 함께 학습하면 성능이 향상될 수 있습니다.

#### 전략별 비교

| 전략 | 설정 | 메모리 | 적합 상황 |
| --- | --- | --- | --- |
| AutoGaze만 RL fine-tune (기본) | `train_task=False` | 낮음 | 원본 도메인과 유사한 경우 |
| VideoMAE + AutoGaze 동시 학습 | `train_task=True, detach_task=False` | 매우 높음 (+100배) | 완전한 도메인 특화 |
| **단계별 적응 (권장)** | VideoMAE 먼저 → AutoGaze RL | 단계별 낮음 | 새 도메인, 현실적 선택 |

#### 권장: 단계별 도메인 적응

```bash
# Step 1. VideoMAE를 새 도메인 비디오로 MAE 학습 (AutoGaze 없이)
python -m autogaze.train \
    --config-name video_folder_video_mae_reconstruction_ar_gaze_grpo \
    dataset.root="'<새 도메인 데이터 경로>'" \
    trainer.train_gaze=False \
    trainer.train_task=True \
    trainer.detach_task=False \
    trainer.task_weights=weights/VideoMAE_AutoGaze/videomae.pt \
    trainer.exp_name=videomae_domain_adapt

# Step 2. 적응된 VideoMAE를 reward 모델로 사용해 AutoGaze RL
python -m autogaze.train \
    --config-name video_folder_video_mae_reconstruction_ar_gaze_grpo \
    dataset.root="'<새 도메인 데이터 경로>'" \
    trainer.train_gaze=True \
    trainer.train_task=False \
    trainer.detach_task=True \
    trainer.task_weights=exps/videomae_domain_adapt/checkpoint_latest_task \
    trainer.gaze_weights=weights/AutoGaze \
    trainer.exp_name=autogaze_domain_rl
```

#### 관련 파라미터

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `trainer.train_task` | `False` | `True`로 설정하면 VideoMAE 가중치도 업데이트 |
| `trainer.detach_task` | `True` | `False`로 설정하면 VideoMAE에 gradient 전파 (메모리 증가) |
| `trainer.train_gaze` | `True` | `False`로 설정하면 VideoMAE만 학습 (Step 1용) |

---

## 6. 파라미터 상세 설명

### 데이터셋 파라미터

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `dataset.root` | — | 쉼표 구분 비디오 데이터셋 경로들. 각 경로 아래 `train/`, `val/` 폴더와 `.mp4` 파일 필요 |
| `dataset.gt_gazing_pos_paths.train` | `null` | GT 가이즈 레이블 JSON 경로. **NTP 학습 시에만 필요** |
| `dataset.clip_len` | `16` | 비디오 클립에서 샘플링할 프레임 수 |

### 모델 파라미터 — 가이즈 비율 제어

**전체 비디오 가이즈 비율** (`gazing_ratio`): 전체 패치 중 몇 %를 볼지

| 파라미터 | 설명 |
| --- | --- |
| `model.gazing_ratio_config.sample_strategy_during_training` | 학습 중 샘플링 전략: `fixed` / `uniform` / `exponential` |
| `model.gazing_ratio_config.sample_strategy_during_inference` | 추론 중 샘플링 전략 |
| `model.gazing_ratio_config.fixed.gazing_ratio` | `fixed` 전략 사용 시 고정 비율 (예: `0.75` = 75%) |
| `model.gazing_ratio_config.exponential.*` | 지수 분포 파라미터 (λ=10, min=0.02, max=0.15) |

**프레임별 가이즈 예산 배분** (`gazing_ratio_each_frame`): 각 프레임에 몇 개의 패치를 할당할지

| 파라미터 | 설명 |
| --- | --- |
| `model.gazing_ratio_each_frame_config.sample_strategy_during_training` | `uniform` / `dirichlet` / `self` |
| `model.gazing_ratio_each_frame_config.dirichlet.alpha` | Dirichlet 집중도 파라미터 (프레임 수만큼). `10,3,3,...,3`이면 첫 프레임에 더 많은 예산 |
| `self` 전략 | 모델을 먼저 실행해 자체적으로 결정한 프레임별 비율을 사용 (on-policy) |

**재건 품질 임계값** (`task_loss_requirement`): 충분한 패치를 봤으면 조기 종료

| 파라미터 | 설명 |
| --- | --- |
| `model.has_task_loss_requirement_during_training` | 학습 중 조기 종료 사용 여부 |
| `model.has_task_loss_requirement_during_inference` | 추론 중 조기 종료 사용 여부 |
| `model.task_loss_requirement_config.fixed.task_loss_requirement` | 고정 임계값 (0~1). 낮을수록 더 많은 패치 필요. `0.7` 권장 |
| `model.task_loss_requirement_config.uniform.*` | 학습 시 임계값을 균등 분포로 샘플링 (일반화 향상) |

### 모델 파라미터 — 구조

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `model.scales` | `32+64+112+224` | 멀티스케일 패치 크기 (`+`로 구분). 세밀한 영역은 224, 큰 영역은 32 사용 |
| `model.num_vision_tokens_each_frame` | `265` | 프레임당 전체 비전 토큰 수 (모든 스케일 합산) |
| `model.gaze_model_config.gaze_decoder_config.num_multi_token_pred` | `10` | AR 디코더가 한 스텝에 병렬 예측하는 토큰 수. 높을수록 빠르지만 정확도 하락 |

### 태스크 파라미터 (VideoMAE 재건)

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `task.recon_model` | `facebook/vit-mae-large` | 재건에 사용할 VideoMAE 모델 |
| `task.recon_sample_rate` | `0.125` | 재건 손실 계산에 사용할 프레임 비율 (1/8). 낮을수록 빠름 |
| `task.recon_model_config.loss_type` | `l1+dinov2_reg+siglip2` | 재건 손실 유형 (`+`로 조합). `l1`=픽셀 수준, `dinov2_reg`=DINOv2 피처, `siglip2`=SigLIP2 피처 |
| `task.recon_model_config.loss_weights` | `1+0.3+0.3` | 각 손실의 가중치 |
| `task.scales` | `32+64+112+224` | 태스크 모델이 처리할 스케일 (model.scales와 다를 수 있음) |

### 알고리즘 파라미터 (RL 전용)

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `algorithm.group_size` | `4` (단일) / `12` (멀티) | GRPO에서 입력당 샘플 시퀀스 수. 클수록 안정적이지만 메모리 증가 |
| `algorithm.discount_factor` | `0.995` | 시간 할인 계수. 1.0에 가까울수록 보상이 가이즈 궤적 전체에 고르게 기여 |
| `algorithm.optimize_task_loss_prediction` | `True` | 가이즈 각 스텝에서 재건 손실 예측 학습 여부. 조기 종료 품질 향상에 기여 |

### 트레이너 파라미터

| 파라미터 | 설명 |
| --- | --- |
| `trainer.train_gaze` | 가이즈 모델 학습 여부 (항상 `True`) |
| `trainer.train_task` | VideoMAE 학습 여부 (동결 사용 시 `False`) |
| `trainer.detach_task` | VideoMAE를 `torch.no_grad()`로 실행. 메모리 절약 |
| `trainer.task_weights` | VideoMAE 사전학습 가중치 경로 (`videomae.pt`) |
| `trainer.gaze_weights` | 가이즈 모델 초기화 경로. Stage 2에서 Stage 1 체크포인트 지정 |
| `trainer.lr` | 학습률 |
| `trainer.lr_schedule` | 스케줄: `linear` / `linear_w_warmup` / `constant` |
| `trainer.n_epochs` | 학습 에폭 수 (Stage 1: 150, Stage 2: 1) |
| `trainer.batch_size` | 전체 GPU 합산 배치 크기 |
| `trainer.per_gpu_max_batch_size` | GPU당 최대 배치. `batch_size > per_gpu * n_gpu`이면 자동으로 gradient accumulation |
| `trainer.temp_schedule_args.exp.temp_start/end` | AR 샘플링 온도 스케줄. 높은 값(탐색) → 낮은 값(활용) |
| `trainer.val_nsteps` | 검증 주기 (스텝 단위) |
| `trainer.save_nsteps` | 체크포인트 저장 주기 |
| `trainer.exp_name` | 실험 이름. `exps/<exp_name>/` 아래에 저장 |

---

## 7. 체크포인트 관리

학습 중 체크포인트는 `exps/<exp_name>/` 아래에 저장됩니다.

```text
exps/
└── ntp_single_gpu/
    ├── checkpoint_latest_gaze/     ← 가이즈 모델 최신 체크포인트 (Stage 2에 전달)
    ├── checkpoint_latest_task/     ← 태스크 모델 최신 체크포인트
    ├── checkpoint_step_0500_gaze/  ← 스텝별 체크포인트
    └── ...
```

### Stage 2에서 Stage 1 체크포인트 불러오기

```bash
trainer.gaze_weights=exps/ntp_single_gpu/checkpoint_latest_gaze
```

### 학습 재개 (Resume)

기본적으로 `trainer.resume=auto`이므로 실험 디렉터리에 체크포인트가 있으면 자동으로 이어서 학습합니다.  
처음부터 새로 시작하려면 `trainer.resume=false`를 추가합니다.

---

## 8. 자주 묻는 질문 / 트러블슈팅

### Q. Mac에서 `flash_attn` 관련 오류가 납니다

**A.** macOS는 `flash_attn`을 지원하지 않습니다.  
`autogaze/configs/task/video_mae_reconstruction.yaml`의 `attn_mode`가 `sdpa`로 설정되어 있는지 확인하세요.

```yaml
attn_mode: 'sdpa'   # flash_attention_2 → sdpa 로 변경되어 있어야 함
```

---

### Q. VideoMAE 가중치 로드 시 크기 불일치 오류 (`time_embed`)가 납니다

**A.** 체크포인트가 `max_num_frames=256`으로 학습되었는데, 모델이 16으로 초기화될 때 발생합니다.  
`autogaze/tasks/video_mae_reconstruction/task_video_mae_reconstruction.py`에서 `max_num_frames=256`이 설정되어 있는지 확인하세요.

또한 VideoMAE 가중치에는 DDP prefix (`module.`)가 붙어 있으므로 로드 시 제거가 필요합니다.  
(이미 코드에 반영되어 있습니다.)

---

### Q. `gazing_pos`의 인덱스가 어떻게 계산되나요

**A.** 전역 인덱스 = `프레임 번호(0-based) × num_tokens_per_frame + 프레임 내 패치 번호(0-based)`

예: 5번째 프레임(0-based: 4)의 3번째 패치(0-based: 2), `num_tokens_per_frame=265`이면  
`gazing_pos = 4 × 265 + 2 = 1062`

---

### Q. 단일 GPU에서 메모리가 부족합니다

**A.** 아래 파라미터를 줄여보세요:

```bash
trainer.per_gpu_max_batch_size=1    # GPU당 배치 크기 축소
algorithm.group_size=2              # RL 그룹 크기 축소 (Stage 2)
trainer.detach_task=True            # VideoMAE를 no_grad로 실행
```

---

### Q. 비디오 전체 프레임을 처리하면 시간이 너무 오래 걸립니다

**A.** `--all-frames` 모드는 청크 수만큼 모델을 반복 실행합니다. 64프레임 비디오는 4번 실행됩니다.  
속도가 중요하다면 기본 16-프레임 샘플링을 사용하거나, `--output-format npy`만 저장해 렌더링 오버헤드를 줄이세요.

```bash
# 빠른 전체 프레임 처리 (시각화 없이 마스크만 저장)
python -m autogaze.infer video.mp4 --all-frames --output-format npy
```

---

### Q. NTP 학습 없이 RL만 해도 되나요

**A.** 가능하지만 권장하지 않습니다. NTP로 기본 가이즈 능력을 먼저 학습해야 RL이 의미 있는 보상 신호를 받을 수 있습니다.  
빠른 실험을 원한다면 공개된 `nvidia/AutoGaze` 가중치를 `trainer.gaze_weights`로 사용하고 RL만 수행하세요.

---

### Q. 인퍼런스 시 VideoMAE가 반드시 필요한가요

**A.** **패치 선택 인퍼런스에는 불필요합니다.** VideoMAE는 학습(Stage 1/2)에서만 필요합니다.

`task_loss_requirement` 파라미터로 조기 종료를 사용할 때도 VideoMAE가 필요 없습니다. AutoGaze 내부에 `task_loss_prediction_head`라는 경량 선형 레이어가 있어, 패치를 하나 선택할 때마다 "이 패치까지 선택했을 때 재건 손실이 얼마일지"를 예측합니다. 이 헤드는 학습 중 VideoMAE의 실제 loss를 ground truth로 학습해 해당 기능을 내재화합니다.

VideoMAE가 인퍼런스에 필요한 경우는 선택된 패치로 **실제 복원 영상을 생성**할 때뿐입니다 (`05_reconstruction_ko.ipynb` 참고).

---

### Q. VideoMAE를 새 도메인 데이터로 함께 학습시킬 수 있나요

**A.** 가능합니다. 코드에 이미 지원이 구현되어 있습니다.

기본 설정은 `trainer.train_task=False` (VideoMAE frozen)이지만, `train_task=True, detach_task=False`로 변경하면 VideoMAE 가중치도 업데이트됩니다.

다만 VideoMAE(~315M)는 AutoGaze(3M)보다 약 100배 크므로, 동시 학습 시 메모리와 연산량이 크게 증가합니다. **권장 전략은 단계별 적응**입니다:

1. `trainer.train_gaze=False, train_task=True` — VideoMAE를 새 도메인 비디오로 먼저 적응
2. `trainer.train_gaze=True, train_task=False` — 적응된 VideoMAE를 reward 모델로 AutoGaze RL

자세한 설정은 [5.5 VideoMAE 도메인 적응 및 Joint Fine-tuning](#55-videomae-도메인-적응-및-joint-fine-tuning)을 참고하세요.

---

문서 최종 갱신: 2026-04-26
