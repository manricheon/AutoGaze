# AutoGaze — ViT / MLLM 벤치마크 가이드

> **목적**: 임의의 ViT 또는 MLLM에 AutoGaze를 붙여 토큰 효율성·지연 시간·품질을 측정하는 방법을 설명합니다.

---

## 목차

1. [두 가지 통합 방식](#1-두-가지-통합-방식)
2. [지원 모델 목록](#2-지원-모델-목록)
3. [방식 A: Zero-shot (AutoGazeTokenSelector)](#3-방식-a-zero-shot-autogazetokenselector)
4. [방식 B: 완전 통합 (NVILA 방식)](#4-방식-b-완전-통합-nvila-방식)
5. [새 모델 추가 방법](#5-새-모델-추가-방법)
6. [벤치마크 실행](#6-벤치마크-실행)
7. [결과 해석](#7-결과-해석)

---

## 1. 두 가지 통합 방식

| 구분 | 방식 A: Zero-shot | 방식 B: 완전 통합 |
| :--- | :--- | :--- |
| 모델 수정 | 불필요 | 패치 임베딩 + forward 수정 |
| FLOPs 절감 | ✗ (연산은 그대로, 출력만 마스킹) | ✓ (선택된 토큰만 처리) |
| 지연 시간 절감 | KV cache 절감 (LLM에서 효과적) | ViT 인코딩 속도 직접 향상 |
| 구현 난이도 | 매우 쉬움 (5줄) | 중간~어려움 |
| 참고 구현 | `autogaze_cv.py` | `autogaze/vision_encoders/siglip/` |

**언제 어떤 방식을 써야 하나?**

- **빠른 탐색 / 논문 비교**: 방식 A (Zero-shot). 어떤 ViT도 즉시 테스트 가능.
- **프로덕션 배포 / 최대 속도**: 방식 B. 기존 NVILA SigLIP 통합이 레퍼런스.

---

## 2. 지원 모델 목록

### 방식 A 기준 즉시 사용 가능

| 모델 | HF model ID | 임베딩 모듈 경로 | 패치 크기 | 그리드 |
| :--- | :--- | :--- | :---: | :---: |
| **DINOv2-base** | `facebook/dinov2-base` | `model.embeddings` | 14 px | 16×16 |
| **DINOv2-large** | `facebook/dinov2-large` | `model.embeddings` | 14 px | 16×16 |
| **ViT-B/16** | `google/vit-base-patch16-224` | `model.vit.embeddings` | 16 px | 14×14 |
| **ViT-L/16** | `google/vit-large-patch16-224` | `model.vit.embeddings` | 16 px | 14×14 |
| **YOLOS-tiny** | `hustvl/yolos-tiny` | `model.vit.embeddings` | 16 px | 14×14 |
| **Depth-Anything-V2-S** | `depth-anything/Depth-Anything-V2-Small-hf` | `model.backbone.embeddings` | 14 px | 16×16 |
| **SigLIP-base/16** | `google/siglip-base-patch16-224` | `model.vision_model.embeddings` | 16 px | 14×14 |
| **VideoMAE-base** (pre-training) | `MCG-NJU/videomae-base` | `model.videomae.embeddings` | 16 px | 8×14×14 |
| **VideoMAE-base** (Kinetics-400 cls) | `MCG-NJU/videomae-base-finetuned-kinetics` | `model.videomae.embeddings` | 16 px | 8×14×14 |
| **X-CLIP-base/32** | `microsoft/xclip-base-patch32` | `model.vision_model.vision_model.embeddings` | 32 px | 7×7 |
| **CLIP-ViT-B/32** | `openai/clip-vit-base-patch32` | `model.vision_model.embeddings` | 32 px | 7×7 |

### 방식 A / 방식 B 구현 현황

| 모델 | 방식 A (hook) | 방식 B (완전 통합) | 방식 B 미구현 이유 |
| :--- | :---: | :---: | :--- |
| NVILA-8B-HD-Video (SigLIP) | — | ✅ 프로덕션 | — |
| VideoMAE-CLS (Kinetics-400) | ✅ `run_cv_tasks.py` | ❌ | tubelet 마스킹 수정 필요 (§4.2 참고) |
| X-CLIP-base/32 | ✅ `run_cv_tasks.py` | ❌ | 프레임별 토큰 제거 후 temporal attn 수정 필요 (§4.3 참고) |
| DINOv2 / YOLOS / ViT | ✅ `run_cv_tasks.py` | ❌ | block-causal attention mask 수정 필요 (§4.1 참고) |
| SigLIP (HF) | ✅ `run_cv_tasks.py` | ❌ | block-causal attention mask 수정 필요 (§4.1 참고) |
| SegFormer | ✅ `run_cv_tasks.py` | ❌ | Conv2d 기반 계층 구조 전면 수정 필요 |

> **방식 A** 는 시퀀스 길이를 바꾸지 않고 비선택 토큰을 0으로 채우므로 속도 이득은 없습니다.  실제 지연 시간·VRAM 절감은 **방식 B** 에서만 달성됩니다.

---

## 3. 방식 A: Zero-shot (AutoGazeTokenSelector)

### 3.1 기본 원리

```
입력 비디오/이미지 (224×224)
    │
    ▼
AutoGaze.forward()  →  gazing_mask[-1]: (B, T, 196)  [14×14 gaze map]
    │
    ▼  bilinear 보간
타겟 ViT의 패치 그리드 크기로 resize  →  mask: (B, N_patches) bool
    │
    ▼  forward hook 등록
embed_module.register_forward_hook(_zero_out_hook)
    │
    ▼
pretrained_model(**inputs)  →  선택된 패치만 활성화된 특징 출력
```

### 3.2 코드 예제

```python
import torch
from autogaze.models.autogaze import AutoGaze
from autogaze.models.autogaze.autogaze_cv import AutoGazeTokenSelector
from autogaze.models.autogaze.processing_autogaze import AutoGazeImageProcessor
from transformers import AutoModel, AutoImageProcessor

# ── AutoGaze 로드 ──────────────────────────────────────────
ag_model = AutoGaze.from_pretrained("weights/AutoGaze").eval().to(device)
selector  = AutoGazeTokenSelector(ag_model, gazing_ratio=0.5)
ag_proc   = AutoGazeImageProcessor.from_pretrained("weights/AutoGaze")

# ── 타겟 ViT 로드 (DINOv2 예시) ───────────────────────────
dino = AutoModel.from_pretrained("facebook/dinov2-base").eval().to(device)
dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

# ── 입력 준비 ──────────────────────────────────────────────
# ag_video: AutoGaze 전처리 (B, T, 3, 224, 224), [-1, 1]
# T=1 이면 단일 이미지
ag_video = ag_proc(images=pil_image, return_tensors="pt")["pixel_values"]
ag_video = ag_video.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 224, 224)

# 타겟 ViT 입력
dino_inputs = dino_proc(images=pil_image, return_tensors="pt")
dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}

# ── gaze mask 계산 ─────────────────────────────────────────
# DINOv2: patch_size=14, img_size=224 → 16×16 그리드
mask = selector.compute_gaze_mask(ag_video, target_h=16, target_w=16)
# mask: (1, 256) bool

# ── AutoGaze 적용하여 DINOv2 forward ──────────────────────
embed_module = dino.embeddings  # DINOv2 임베딩 모듈
with selector.token_mask_context(embed_module, mask, has_cls_token=True):
    out_ag = dino(**dino_inputs)

# 비교: AutoGaze 없이
out_full = dino(**dino_inputs)

# 선택된 토큰 수
n_selected = mask[0].sum().item()
n_total    = mask.shape[1]
print(f"선택 토큰: {n_selected}/{n_total} ({100*n_selected/n_total:.1f}%)")
```

### 3.3 임베딩 모듈 경로 찾는 방법

모델 구조를 확인하면 됩니다:

```python
# 모델 서브모듈 이름 출력
for name, module in model.named_modules():
    if "embed" in name.lower() or "patch" in name.lower():
        print(name, type(module).__name__)
```

일반적인 패턴:
- HF ViT 계열: `vit.embeddings` 또는 `embeddings`
- CLIP / SigLIP vision: `vision_model.embeddings`
- VideoMAE (분류 · pre-training 공통): `videomae.embeddings` — 출력 shape `(B, 1568, D)` = 8 tubelet × 196 spatial
- X-CLIP (CLIP 기반 video): `vision_model.vision_model.embeddings`
- DINOv2 backbone 기반 모델: `backbone.embeddings`

---

## 4. 방식 B: 완전 통합 — 모델별 구현 가이드

방식 B는 패치 임베딩 이후 **선택된 토큰만** 트랜스포머 레이어에 전달하므로 실제 FLOPs와 지연 시간이 줄어듭니다.

---

### 4.1 Image ViT (SigLIP / DINOv2 / YOLOS) — NVILA 레퍼런스

NVILA가 채택한 방식입니다. 공간적 패치만 다루며 temporal 차원이 없습니다.

**핵심 수정 3단계**

**① config 추가**

```python
# vision encoder config에 추가
scales = '224'                # 또는 '32+64+112+224'
attn_type = 'block_causal'   # 'block_causal' | 'causal' | 'bidirectional'
frame_independent_encoding = False
```

**② forward에서 토큰 물리적 제거**

```python
def forward(self, pixel_values, gazing_info=None, ...):
    patch_embeds = self.patch_embedding(pixel_values)  # (B, N, D)
    if gazing_info is not None:
        patch_embeds = mask_with_gazing(patch_embeds, gazing_info)
        # mask_with_gazing: gather → (B, k, D)  k = n_selected
    # 이후 transformer는 k-token 시퀀스로 동작
    ...
```

**③ gazing_info 형식**

```python
gazing_info = {
    'gazing_pos':            (B, N),   # 선택된 패치 인덱스
    'num_gazing_each_frame': (T,),     # 프레임별 선택 수
    'if_padded_gazing':      (B, N),   # 패딩 마스크 bool
}
```

레퍼런스: `autogaze/vision_encoders/siglip/modeling_siglip.py` → `mask_with_gazing()`

---

### 4.2 Video ViT — VideoMAE-CLS (tubelet 기반)

VideoMAE는 3D Conv patchify (tubelet_size=2) 로 `(16 frames) → 8×14×14 = 1568` 토큰을 생성합니다. 방식 A는 1568개를 모두 forward하고 비선택 토큰만 0으로 채웁니다.  방식 B는 **tubelet 단위로 물리적 제거**하여 시퀀스를 단축합니다.

```text
방식 A (현재, run_cv_tasks.py)
  VideoMAEEmbeddings 출력: (B, 1568, D)
  hook → non-gaze 토큰 = 0
  transformer layers: 1568 토큰 그대로 처리 (no speedup)

방식 B (미구현 — 아래 체크리스트 참고)
  VideoMAEEmbeddings 출력: (B, 1568, D)
  gather → (B, k, D)  k = ratio × 1568
  transformer layers: k 토큰만 처리  ← real speedup
```

**방식 B 구현 체크리스트 (VideoMAE-CLS)**

- [ ] `VideoMAEEmbeddings.forward()` 출력 직후 `torch.gather`로 k개 선택
- [ ] `VideoMAESelfAttention`에 전달되는 `attention_mask` 제거 (가변 시퀀스 허용)
- [ ] `mean_pool` head를 k-token 시퀀스에서도 올바르게 동작하도록 확인 (`use_mean_pooling=True`)
- [ ] AutoGaze 공간 mask (196,) → 8 tubelet 위치 전체에 broadcast하여 `(1568,)` bool mask 생성
- [ ] `VideoMAEForVideoClassification.forward()`에 `gazing_mask` kwarg 추가

```python
# 방식 B 핵심 패치 예시 (VideoMAEModel.forward 내부)
def forward(self, pixel_values, gazing_mask=None, ...):
    embeddings = self.embeddings(pixel_values)   # (B, 1568, D)
    if gazing_mask is not None:
        # gazing_mask: (B, 1568) bool
        # gather → compact sequence
        idx = gazing_mask[0].nonzero(as_tuple=True)[0]  # (k,)
        embeddings = embeddings[:, idx]          # (B, k, D)
    return self.encoder(embeddings, ...)
```

---

### 4.3 Video ViT — X-CLIP (per-frame CLIP + temporal attention)

X-CLIP 은 공간적 ViT(CLIP)를 각 프레임에 독립적으로 적용한 뒤 temporal transformer로 합칩니다.  방식 B 구현은 두 단계로 나뉩니다.

```text
방식 A (현재, run_cv_tasks.py)
  CLIPVisionEmbeddings (B*T, 197, D) → hook으로 비선택 spatial 토큰 0 처리
  CLIPEncoder: 197 토큰 그대로 처리 (no speedup)
  temporal attn: B×T 프레임 모두 사용

방식 B (미구현)
  CLIPVisionEmbeddings (B*T, 197, D) → gather → (B*T, k+1, D)  (+CLS)
  CLIPEncoder: k+1 토큰만 처리  ← spatial speedup
  temporal attn: 변경 없음 (T 프레임 수는 유지)
```

**방식 B 구현 체크리스트 (X-CLIP)**

- [ ] `CLIPVisionTransformer.forward()` 에서 `patch_embeds` gather 수행 (CLS 토큰은 항상 유지)
- [ ] `CLIPEncoder` 내 `attention_mask` 제거 — k+1 가변 길이 허용
- [ ] `XCLIPVisionModel.forward()` 에 `gazing_mask` kwarg 전달 경로 추가
- [ ] `XCLIPModel.forward()` 에서 per-frame mask를 `(B*T, 196)` 형식으로 reshape하여 전달
- [ ] temporal attention은 수정 불필요 (프레임별 CLS token은 항상 보존됨)

---

## 5. 새 모델 추가 방법

### 방식 A (Zero-shot) 기준 3단계

```python
# 1. 임베딩 모듈 경로 찾기
for name, m in new_model.named_modules():
    if "embed" in name.lower():
        print(name, m.__class__.__name__)

# 2. 패치 크기와 그리드 크기 확인
patch_size = new_model.config.patch_size      # 예: 16
img_size   = new_model.config.image_size      # 예: 224
grid_h = grid_w = img_size // patch_size      # 예: 14

# 3. 임베딩 모듈 가져오기 (점 표기법으로 탐색)
def get_module(model, path):
    m = model
    for part in path.split('.'):
        m = getattr(m, part)
    return m

embed_mod = get_module(new_model, "vit.embeddings")  # 경로에 맞게 수정

# 4. 벤치마크 실행
mask = selector.compute_gaze_mask(ag_video, target_h=grid_h, target_w=grid_w)
with selector.token_mask_context(embed_mod, mask, has_cls_token=True):
    out_ag = new_model(**inputs)
```

### 방식 B (완전 통합) 체크리스트

모델 유형에 따라 §4의 구현 가이드를 참고하세요.

| 모델 유형 | 가이드 |
| :--- | :--- |
| Image ViT (SigLIP / CLIP / DINOv2) | §4.1 — NVILA 레퍼런스 |
| Video ViT — tubelet (VideoMAE) | §4.2 — tubelet gather |
| Video ViT — CLIP+temporal (X-CLIP) | §4.3 — per-frame gather |

공통 절차:

- [ ] 임베딩 출력 직후 `torch.gather`로 선택 토큰만 추출 → `(B, k, D)`
- [ ] 이후 transformer 레이어에 k-token 시퀀스 전달
- [ ] attention mask / position index 수정 (모델별 상이)
- [ ] `AutoGaze.from_pretrained("weights/AutoGaze")` 로드 후 forward에 mask 전달

---

## 6. 벤치마크 실행

### 인터랙티브 노트북

```bash
jupyter notebook notebooks/10_autogaze_benchmark_ko.ipynb
```

### CLI — NVILA 종합 벤치마크

```bash
# 기본 실행 (AutoGaze ON, gazing_ratio=0.75)
python scripts/test_nvila.py assets/example_input.mp4 --frames 16

# AutoGaze ON vs OFF 비교
python scripts/test_nvila.py assets/example_input.mp4 --compare-autogaze

# gazing_ratio 스윕 (0.1 → 1.0)
python scripts/test_nvila.py assets/example_input.mp4 --sweep-ratio --ratio-step 0.1

# 특정 ratio 지정
python scripts/test_nvila.py assets/example_input.mp4 --gazing-ratio 0.5
```

### CLI — CV 태스크 벤치마크

```bash
# 이미지 모든 태스크 (depth, yolos, dinov2, segformer, siglip, videomae_cls, xclip)
python scripts/run_cv_tasks.py --input assets/sample.jpg --output-dir results/

# 특정 태스크만
python scripts/run_cv_tasks.py --input assets/sample.jpg --tasks depth dinov2 yolos

# ratio 그리드
python scripts/run_cv_tasks.py --input assets/sample.jpg --ratios 0.25 0.5 0.75 1.0

# 동작 인식 태스크 (VideoMAE-CLS + X-CLIP)
python scripts/run_cv_tasks.py \
    --input assets/sample.jpg \
    --tasks videomae_cls xclip \
    --ratios 0.75 0.5 0.25

# 비디오 모드 — 청크 단위 동작 인식 결과를 각 프레임에 오버레이
python scripts/run_cv_tasks.py \
    --input assets/example.mp4 \
    --tasks videomae_cls xclip \
    --ag-ratio 0.5 \
    --temporal-window 16
```

지원 태스크 전체 목록:

| 태스크 키 | 모델 | 유형 |
| :--- | :--- | :--- |
| `depth` | Depth-Anything-V2-S | 깊이 추정 |
| `yolos` | YOLOS-tiny | 객체 탐지 |
| `dinov2` | DINOv2-base (ImageNet1k) | 이미지 분류 |
| `segformer` | SegFormer-B2 (ADE20K) | 세그멘테이션 |
| `siglip` | SigLIP-base/16 | Zero-shot 분류 |
| `videomae_cls` | VideoMAE-base (Kinetics-400) | 동작 인식 (supervised) |
| `xclip` | X-CLIP-base/32 | 동작 인식 (zero-shot) |

### MambaGaze 지연 시간 벤치마크

```bash
python -m mamba_gaze.eval.latency \
    --batch-sizes 1 4 8 \
    --n-frames 4 8 16 \
    --ratios 0.25 0.5 0.75 \
    --output results/latency.csv
```

---

## 7. 결과 해석

### 핵심 지표

| 지표 | 설명 | 단위 | 목표 |
| :--- | :--- | :--- | :--- |
| **시각 토큰 수** | 선택된 패치 토큰 수 | count | 비율에 비례 감소 |
| **KV cache 크기** | 토큰×헤드×차원 | MB | 줄수록 LLM 속도↑ |
| **ViT 인코딩 시간** | 방식 B에서만 절감 | ms | 비율에 비례 감소 |
| **LLM 프리필 시간** | KV cache 감소 효과 | ms | 토큰 수에 비례 |
| **PSNR / SSIM** | 마스킹된 재구성 품질 | dB / [0,1] | ↑ 높을수록 좋음 |
| **하위 태스크 정확도** | VideoMME, MLVU 등 | % | AutoGaze OFF 대비 손실 최소화 |

### Gazing Ratio 선택 가이드

| 시나리오 | 추천 ratio |
| :--- | :---: |
| 실시간 스트리밍 (지연 최우선) | 0.25 ~ 0.4 |
| 균형 (속도 + 품질) | 0.5 ~ 0.6 |
| 품질 우선 (긴 비디오) | 0.7 ~ 0.8 |
| AutoGaze 효과 없음 (기준선) | 1.0 |

### 예상 절감 효과 (NVILA 기준)

| ratio | 시각 토큰 | ViT 시간 | LLM 프리필 |
| :---: | :---: | :---: | :---: |
| 0.25 | −75% | −60~70% (방식 B) | −60~70% |
| 0.50 | −50% | −40~50% (방식 B) | −40~50% |
| 0.75 | −25% | −20~30% (방식 B) | −20~30% |

---

## 참고 파일

| 용도 | 경로 |
| :--- | :--- |
| Zero-shot 유틸리티 | `autogaze/models/autogaze/autogaze_cv.py` |
| NVILA 프로세서 | `weights/NVILA-8B-HD-Video/processing_nvila.py` |
| SigLIP 완전 통합 | `autogaze/vision_encoders/siglip/modeling_siglip.py` |
| CV 태스크 스크립트 | `scripts/run_cv_tasks.py` |
| NVILA 벤치마크 스크립트 | `scripts/test_nvila.py` |
| 벤치마크 노트북 | `notebooks/10_autogaze_benchmark_ko.ipynb` |
| MambaGaze 지연 시간 평가 | `mamba_gaze/eval/latency.py` |
