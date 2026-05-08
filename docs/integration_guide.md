# Integrating AutoGaze into ViTs and MLLMs: A Comparative Guide

This document provides a technical overview and performance comparison of integrating **AutoGaze** into various Vision Transformers (ViTs) and Multimodal Large Language Models (MLLMs).

---

## 1. System Architecture (시스템 아키텍처)

AutoGaze acts as an **intelligent token filter** that sits between the raw video frames and the vision encoder.

AutoGaze는 원본 비디오 프레임과 비전 인코더 사이에서 동작하는 **지능형 토큰 필터** 역할을 합니다.

---

## 2. Integration Modes: Hook vs. Full (통합 모드 비교)

| Feature (기능) | Hook Mode (Zero-shot) | Full Mode (Integrated) |
| :--- | :--- | :--- |
| **Mechanism** | PyTorch Forward Hook | Forward Method Override |
| **Sequence Length** | **Unchanged ($N_{all}$)** | **Reduced ($N_{gazed}$)** |
| **Computation** | Zeroes non-selected tokens | Skips non-selected tokens |
| **Complexity** | $O(N^2)$ attention overhead | **$O(k^2)$ quadratic saving** |
| **Main Use Case** | Accuracy/Task Validation | Efficiency/Latency Benchmark |

### Key Differences in Usage (사용 시 주요 차이점)

#### **Hook Mode (훅 모드)**
- **KOR**: 모델의 소스 코드를 수정할 필요 없이, 인코더의 출력층에 "가이즈 마스크"를 곱해주는 방식입니다. 토큰의 개수 자체는 줄어들지 않지만(0으로 채워짐), 어텐션 층에서 해당 토큰들이 무시되도록 유도합니다.
- **Why use it?**: 새로운 모델에 AutoGaze를 빠르게 적용해보고, **정확도(Accuracy)**가 유지되는지 테스트할 때 가장 좋습니다. (속도 향상은 미미함)

#### **Full Mode (풀 모드)**
- **KOR**: 인코더 내부의 `forward` 함수를 직접 수정하여, 선택되지 않은 토큰을 메모리 상에서 **완전히 제거**하는 방식입니다.
- **Why use it?**: AutoGaze의 진정한 효과인 **추론 속도 향상(Latency)**과 **메모리 절감(VRAM)**을 검증할 때 반드시 필요합니다.

---

## 3. Testing Strategy: Which mode should I use? (테스트 전략)

> **"To test AutoGaze effect, we should use Full mode, isn't it?"**
> **"AutoGaze의 효과를 제대로 보려면 풀 모드를 써야 하나요?"**

**YES and NO.** It depends on what "effect" you are measuring:
어떤 "효과"를 측정하느냐에 따라 달라집니다:

1.  **If testing Accuracy (성능/정확도 테스트)**:
    - **Hook Mode** is sufficient. Since tokens are zeroed, they don't contribute to the output. If the model works well in Hook mode, it will work even better (or identical) in Full mode.
    - **훅 모드**로 충분합니다. 토큰이 0으로 처리되어 결과에 영향을 주지 않으므로, 이 모드에서 정확도가 잘 나온다면 알고리즘적으로 검증된 것입니다.

2.  **If testing Efficiency (효율성/속도 테스트)**:
    - **Full Mode** is mandatory. You cannot see the $O(k^2)$ speedup or VRAM savings in Hook mode because the GPU still allocates memory for all tokens.
    - **풀 모드**가 필수입니다. 훅 모드에서는 GPU가 모든 토큰을 위한 메모리를 여전히 할당하기 때문에, 실제 연산량 감소와 속도 향상을 확인하려면 토큰을 물리적으로 제거하는 풀 모드가 필요합니다.

---

## 4. Compatibility Matrix (호환성 매트릭스)

All ViT variants can technically support both modes, but **Full Mode** requires specific architectural "fixes."

| ViT Variant | Hook Mode | Full Mode | Engineering Requirement |
| :--- | :---: | :---: | :--- |
| **Image-based** (e.g. SigLIP) | ✅ | ✅ | Block-Causal Masking |
| **Video-native** (Absolute PE) | ✅ | ✅ | None (Native awareness) |
| **Video-native** (RoPE) | ✅ | ✅ | **RoPE Position Correction** |
| **Hierarchical** (e.g. Qwen2.5) | ✅ | ✅ | Pre-reordering Masking |

---

## 5. Structural Comparison (구조적 비교)

| Attribute (속성) | SigLIP (Vanilla/NVILA) | V-JEPA2 (Native) |
| :--- | :--- | :--- |
| **Base Domain** | Image (2D) | Video (3D) |
| **Patching** | 2D ($16 \times 16$) | 3D Tubelet ($2 \times 16 \times 16$) |
| **Attention** | Causal (via block-mask) | Full Bidirectional |
| **Positional Enc.** | Absolute (sin/cos or learned) | **Rotary (RoPE)** |
| **AutoGaze Sync** | Flattening + Masking | **Index-based RoPE Mapping** |

---

## 6. Performance Comparison (성능 비교)

Average results across 16-frame 224px video chunks on A100 GPU.

| Model Config | Structure | Token Count | Latency (ms) | VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: |
| **SigLIP (NVILA)** | Baseline | 3,136 | ~320 | ~18.5 |
| **SigLIP (NVILA)** | **Variant 1 (Full)** | **784** | **~145** | **~16.8** |
| **V-JEPA2 (Substitute)** | Baseline | 1,568 | ~55 (ViT) | ~2.1 (ViT) |
| **V-JEPA2 (Substitute)** | **Variant 4 (Full)** | **392** | **~22 (ViT)** | **~1.1 (ViT)** |

---
*Document generated for AutoGaze Architecture Analysis (2026).*
