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
|------|-------------------|-------------------|
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
|------|------------|----------------|-----------|--------|
| **DINOv2-base** | `facebook/dinov2-base` | `model.embeddings` | 14 px | 16×16 |
| **DINOv2-large** | `facebook/dinov2-large` | `model.embeddings` | 14 px | 16×16 |
| **ViT-B/16** | `google/vit-base-patch16-224` | `model.vit.embeddings` | 16 px | 14×14 |
| **ViT-L/16** | `google/vit-large-patch16-224` | `model.vit.embeddings` | 16 px | 14×14 |
| **YOLOS-tiny** | `hustvl/yolos-tiny` | `model.vit.embeddings` | 16 px | 14×14 |
| **Depth-Anything-V2-S** | `depth-anything/Depth-Anything-V2-Small-hf` | `model.backbone.embeddings` | 14 px | 16×16 |
| **SigLIP-base/16** | `google/siglip-base-patch16-224` | (local `autogaze/vision_encoders/siglip/`) | 16 px | 14×14 |
| **VideoMAE-base** | `MCG-NJU/videomae-base` | `model.patch_embed` | 16 px | 14×14 |
| **CLIP-ViT-B/32** | `openai/clip-vit-base-patch32` | `model.vision_model.embeddings` | 32 px | 7×7 |

### 방식 B 구현 완료

| 모델 | 상태 |
|------|------|
| NVILA-8B-HD-Video (SigLIP) | ✅ 프로덕션 |

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
- CLIP vision: `vision_model.embeddings`
- VideoMAE: `patch_embed`
- DINOv2 backbone 기반 모델: `backbone.embeddings`

---

## 4. 방식 B: 완전 통합 (NVILA 방식)

방식 B는 패치 임베딩 이후 선택된 토큰만 트랜스포머 레이어에 전달하므로 실제 FLOPs와 지연 시간이 줄어듭니다.

### 4.1 핵심 수정 사항

**① config 추가**
```python
# vision encoder config에 추가
scales = '224'                # 또는 '32+64+112+224'
attn_type = 'block_causal'   # 'block_causal' | 'causal' | 'bidirectional'
frame_independent_encoding = False
```

**② forward 시그니처 수정**
```python
def forward(
    self,
    pixel_values: torch.Tensor,
    gazing_info: Optional[dict] = None,  # ← 추가
    ...
):
    patch_embeds = self.patch_embedding(pixel_values)  # (B, N, D)

    if gazing_info is not None:
        patch_embeds = mask_with_gazing(patch_embeds, gazing_info)
    ...
```

**③ gazing_info 형식**
```python
gazing_info = {
    'gazing_pos':            (B, N),      # 선택된 패치 인덱스
    'num_gazing_each_frame': (T,),        # 프레임별 선택 수
    'if_padded_gazing':      (B, N),      # 패딩 마스크 bool
}
```

**레퍼런스**: `autogaze/vision_encoders/siglip/modeling_siglip.py` → `mask_with_gazing()`

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

- [ ] `config.json`에 `scales`, `attn_type` 필드 추가
- [ ] `patch_embed` forward에서 `gazing_info`를 받아 `mask_with_gazing()` 호출
- [ ] 어텐션 마스크를 `block_causal` 또는 `causal` 형식으로 생성
- [ ] `AutoGaze.from_pretrained("weights/AutoGaze")` 로드 후 processor에 연결

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
# 이미지 모든 태스크
python scripts/run_cv_tasks.py --input assets/sample.jpg --output-dir results/

# 특정 태스크만
python scripts/run_cv_tasks.py --input assets/sample.jpg --tasks depth dinov2 yolos

# ratio 그리드
python scripts/run_cv_tasks.py --input assets/sample.jpg --ratios 0.25 0.5 0.75 1.0
```

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
|------|------|------|------|
| **시각 토큰 수** | 선택된 패치 토큰 수 | count | 비율에 비례 감소 |
| **KV cache 크기** | 토큰×헤드×차원 | MB | 줄수록 LLM 속도↑ |
| **ViT 인코딩 시간** | 방식 B에서만 절감 | ms | 비율에 비례 감소 |
| **LLM 프리필 시간** | KV cache 감소 효과 | ms | 토큰 수에 비례 |
| **PSNR / SSIM** | 마스킹된 재구성 품질 | dB / [0,1] | ↑ 높을수록 좋음 |
| **하위 태스크 정확도** | VideoMME, MLVU 등 | % | AutoGaze OFF 대비 손실 최소화 |

### Gazing Ratio 선택 가이드

| 시나리오 | 추천 ratio |
|----------|-----------|
| 실시간 스트리밍 (지연 최우선) | 0.25 ~ 0.4 |
| 균형 (속도 + 품질) | 0.5 ~ 0.6 |
| 품질 우선 (긴 비디오) | 0.7 ~ 0.8 |
| AutoGaze 효과 없음 (기준선) | 1.0 |

### 예상 절감 효과 (NVILA 기준)

| ratio | 시각 토큰 | ViT 시간 | LLM 프리필 |
|-------|----------|---------|-----------|
| 0.25 | −75% | −60~70% (방식 B) | −60~70% |
| 0.50 | −50% | −40~50% (방식 B) | −40~50% |
| 0.75 | −25% | −20~30% (방식 B) | −20~30% |

---

## 참고 파일

| 용도 | 경로 |
|------|------|
| Zero-shot 유틸리티 | `autogaze/models/autogaze/autogaze_cv.py` |
| NVILA 프로세서 | `weights/NVILA-8B-HD-Video/processing_nvila.py` |
| SigLIP 완전 통합 | `autogaze/vision_encoders/siglip/modeling_siglip.py` |
| CV 태스크 스크립트 | `scripts/run_cv_tasks.py` |
| NVILA 벤치마크 스크립트 | `scripts/test_nvila.py` |
| 벤치마크 노트북 | `notebooks/10_autogaze_benchmark_ko.ipynb` |
| MambaGaze 지연 시간 평가 | `mamba_gaze/eval/latency.py` |
