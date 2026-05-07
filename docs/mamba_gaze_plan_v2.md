# MambaGaze v2: Feedforward 토큰 선택기 설계 문서

> **작성일**: 2026-05-06  
> **버전**: v2 (Feedforward / Non-AR) — Branch A of mamba_gaze_ref_v0.md  
> **v1 파일**: `docs/mamba_gaze_plan.md` (AR Mamba decoder, 비교 참고용)

---

## 1. v1 vs v2 핵심 차이

### v1 (mamba_gaze_plan.md): AR Mamba Decoder

```text
AutoGaze 한계: LLaMA AR 디코더 O(N²)
        ↓ 개선 방향 v1
MambaGaze v1:  Mamba SSM AR 디코더 O(N)
```

v1은 LLaMA를 Mamba SSM으로 교체했지만 **여전히 토큰을 하나씩 순차 생성한다 (AR loop)**. O(N) 복잡도로 KV 캐시를 제거했으나, 생성 루프 자체는 남아 있다.

### v2 (이 문서): Feedforward Non-AR

```text
AutoGaze 한계: AR 디코딩 0.193s/프레임 (목표 < 10ms)
        ↓ 개선 방향 v2
MambaGaze v2:  단일 feedforward pass, 모든 패치 중요도를 병렬 예측
```

v2는 **생성 루프를 완전히 제거**한다. 비디오 전체를 한 번의 forward pass에서 처리하여 이진 마스크를 출력한다.

| 항목 | AutoGaze | v1 (AR Mamba) | v2 (FF Mamba) |
|------|---------|---------------|---------------|
| 디코딩 방식 | AR LLaMA | AR Mamba | Feedforward |
| 시간 복잡도 (패치 N) | O(N) sequential | O(N) sequential | **O(1) parallel** |
| Gazing latency | 193 ms/frame | ~50 ms/frame (예측) | **< 10 ms/frame (목표)** |
| 학습 방법 | NTP + GRPO RL | NTP + GRPO RL | **Distillation (BCE)** |
| 파라미터 수 | ~3.0M | ~4.4M | **~1.1M** |
| 공간 정보 | 암묵적 | 암묵적 | **명시적 saliency head** |

---

## 2. v2 아키텍처: SRF-Predictor (Spatial Relevance Feedforward)

```text
Video (B, T, C, H, W)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ LightweightCNNEncoder  (~315K)                                  │
│   3 → [32→64→128→192] channel, stride-2 × 4                    │
│   224×224 → 14×14 (패치 정렬, ViT 호환)                         │
│   출력: F_t ∈ R^{B×T×192×14×14}                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├──── SaliencyHead  (~200 params)
    │         1×1 Conv → sigmoid
    │         S_t ∈ [0,1]^{14×14}  (명시적 공간 현저성)
    │
    └──── TemporalMotion  (0 params, 연산만)
              R_t = norm(||F_t − F_{t-1}||²)
              R_t ∈ [0,1]^{14×14}  (프레임 간 변화량)
    │
    ▼ Concat(F_t, S_t, R_t) → (B, T, N=196, 194)
┌─────────────────────────────────────────────────────────────────┐
│ SpatioTemporalMambaAggregator  (~769K)                          │
│   in_proj: 194 → 128                                            │
│   Layer × 2:                                                    │
│     Spatial Bi-Mamba (zigzag scan, 14×14, bidirectional)        │
│     Temporal Causal Mamba (T 프레임, causal)                     │
│   score_head: 128 → 1 → sigmoid                                 │
│   출력: I_t ∈ [0,1]^{N=196}  (패치별 중요도)                    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Top-K Selection
    훈련: Gumbel-top-k + straight-through (미분 가능)
    추론: Hard top-k (단순 정렬, O(N log N))
    │
    ▼
Binary Mask (B, T, N)  — AutoGaze 출력과 동일 포맷
```

### 파라미터 분포 (기본 설정)

| 컴포넌트 | 파라미터 | 비율 |
|---------|---------|------|
| LightweightCNNEncoder | 315K | 29% |
| SaliencyHead | < 1K | < 0.1% |
| SpatioTemporalMambaAggregator | 769K | 71% |
| **합계** | **~1.08M** | — |

AutoGaze 대비 **~64% 감소** (3.0M → 1.08M)

---

## 3. 학습 전략 (Distillation)

### A-Step 1: Teacher Mask 생성

```python
# AutoGaze 구동 → 50K 비디오에 대해 binary mask 저장
teacher_mask = autogaze_model({'video': video}, gazing_ratio=0.5)['gazing_mask'][0]
# teacher_mask: (B, T, N) binary
```

### A-Step 2: Distillation 학습

```python
# 학습 손실: BCE(예측 중요도, AutoGaze 마스크)
out = mamba_ff({'video': video}, teacher_mask=teacher_mask)
loss = out['distill_loss']   # F.binary_cross_entropy(scores, teacher_mask)
```

단순 BCE로도 mask IoU > 0.90 달성 가능. 필요 시 추가:
- **Ranking loss**: 선택 vs 비선택 점수 margin 강제
- **Dice loss**: 클래스 불균형(sparse selection) 보정

### A-Step 3: Saliency + Semantic Fine-tuning

- Saliency prior 강화: DHF1K 눈추적 데이터로 saliency head 추가 학습
- Semantic fidelity: 선택된 패치의 ViT 재구성 품질 보조 손실

---

## 4. Gumbel-Top-K 미분 가능 선택

훈련 시 hard selection(argmax)은 기울기가 없다. Gumbel-top-k는 이를 soft selection으로 근사한다:

```
logits = log(I / (1 - I))          # log-odds
noise  = -log(-log(Uniform[0,1]))  # Gumbel noise
perturbed = (logits + noise) / τ   # τ = gumbel_temperature

threshold = kth-largest(perturbed)
soft_mask  = sigmoid((perturbed - threshold) / τ)  # ≈ binary, gradient exists
```

τ를 훈련 중 점차 낮추면(temperature annealing) soft mask가 hard binary에 수렴한다.

---

## 5. 파일 구조

```
autogaze/models/mamba_gaze/
    __init__.py                         # v1 + v2 모두 export
    configuration_mamba_gaze.py         # v1 AR config
    modeling_mamba_gaze.py              # v1 AR implementation
    configuration_mamba_gaze_ff.py      # v2 FF config  ← NEW
    modeling_mamba_gaze_ff.py           # v2 FF implementation  ← NEW
        ├── LightweightCNNEncoder
        ├── SaliencyHead
        ├── SpatioTemporalMambaAggregator  (재사용: MambaBlock from v1)
        └── MambaGazeFF                    (AutoGaze 호환 인터페이스)
```

---

## 6. 예상 성능 (목표)

| 지표 | AutoGaze | MambaGaze v1 (AR) | MambaGaze v2 (FF) |
|------|---------|------------------|------------------|
| Gazing latency | 193 ms/frame | ~50 ms/frame | **< 10 ms/frame** |
| Mask IoU | 1.0 (teacher) | — (랜덤) | > 0.90 (학습 후) |
| 파라미터 수 | ~3.0M | ~4.4M | **~1.1M** |
| 스트리밍 캐시 | 19 MB KV | 98 KB SSM | **없음 (stateless)** |
| 훈련 방식 | NTP+GRPO | NTP+GRPO | **Distillation only** |

v2는 스트리밍 추론 시 상태(캐시)가 없다는 점도 장점이다 — 각 프레임 청크를 독립적으로 처리 가능.

---

## 7. 다음 단계

1. **A-Step 1**: InternVid 50K에서 AutoGaze teacher mask 생성 (A100 × 2, ~2주)
2. **A-Step 2**: Distillation 학습 — Mask IoU > 0.85 달성 (H100 × 4, ~2주)
3. **A-Step 3**: Saliency fine-tuning + latency 프로파일링 (< 10ms 검증)
4. **VideoMME, MLVU** 벤치마크로 video understanding 품질 검증
5. **Branch B**: task-aware fine-tuning (Seg/Depth/Det/Cls frozen 모델 distillation)

---

## 참고문헌

- [Mamba](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [Vision Mamba](https://arxiv.org/abs/2401.13062) (Zhu et al., 2024)
- [VideoMamba](https://arxiv.org/abs/2403.06977) (Li et al., 2024)
- [AutoGaze](https://arxiv.org/abs/2603.12254) (Shi et al., 2026)
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) (Jang et al., 2016)
- [DHF1K](https://mmcheng.net/videosal/) (Wang et al., 2018) — saliency prior
