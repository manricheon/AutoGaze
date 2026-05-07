# MambaGaze: AutoGaze Decoder를 Mamba SSM으로 대체하는 설계 문서

> **작성일**: 2026-05-03  
> **상태**: Prototype / 실험적

---

## 1. 동기 (Why Mamba?)

### 1.1 현재 AutoGaze 아키텍처의 병목

AutoGaze는 3개 구성 요소로 이루어져 있다:

```
Video (B, T, C, H, W)
    ↓
ShallowVideoConvNet      ← 얕은 3D ConvNet, ~1M params
    ↓
Connector                ← learnable positional embedding, ~37K params
    ↓
LlamaForCausalLM         ← LLaMA 기반 AR 디코더, ~2–3M params
    ↓
Gaze position tokens (autoregressive)
```

**LLaMA 디코더의 한계:**

| 문제 | 설명 |
|------|------|
| **O(N²) 복잡도** | 시퀀스 길이에 이차 비례. T=16, 196 토큰/프레임 → 최대 6,272 토큰, 어텐션 = 6,272² ≈ 39M 연산 |
| **KV 캐시 선형 증가** | 스트리밍 시 KV 캐시 크기가 시퀀스 길이 N에 비례해 증가: O(L × d) |
| **위치 인코딩 의존** | RoPE가 절대 위치에 민감, 가변 길이 영상에 취약 |
| **과잉 설계** | gaze 토큰(196+1 vocab)은 단순한 집합 선택 문제 — LLM의 표현력이 필요하지 않음 |

### 1.2 Mamba SSM의 장점

Mamba(Selective State Space Model, S6)는 시퀀스를 선형 점화식으로 처리한다:

```
h_t = Ā_t · h_{t-1} + B̄_t · x_t        # 상태 업데이트 (input-selective)
y_t = C_t · h_t + D · x_t               # 출력
```

여기서 Ā, B̄, C, Δ가 모두 입력 의존(selective) — 기존 SSM과의 핵심 차이.

| 특성 | LLaMA | Mamba |
|------|-------|-------|
| 시퀀스 복잡도 | O(N²) | **O(N)** |
| 스트리밍 캐시 크기 | O(N × d) — 증가 | **O(d × N_state) — 상수** |
| 위치 인코딩 | RoPE 필요 | 불필요 (SSM이 암묵적으로 처리) |
| 병렬 훈련 | Full attention | Parallel associative scan |
| 단계별 추론 | Attention + KV cache | SSM state update (극히 저렴) |

**스트리밍 캐시 크기 비교 (AutoGaze 기본 설정, d=192, 4 layers):**

- LLaMA KV 캐시: `4 × 2 × N × 192 × 2 bytes` = **3,072N bytes** (N은 시퀀스 길이, 가변)
- Mamba SSM 상태: `4 × 384 × 16 × 4 bytes` = **98,304 bytes ≈ 96 KB** (항상 일정)

T=16 최대 시퀀스(6272 토큰) 기준 KV 캐시는 **19 MB**인 반면 SSM 상태는 항상 **96 KB**.

---

## 2. 제안 아키텍처: MambaGaze

### 2.1 전체 구조

```
Video (B, T, C, H, W)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  MambaVisionEncoder  (ShallowVideoConvNet 대체)      │
│                                                     │
│  ① 3D Conv Patch Embedding (동일)                   │
│     kernel=(temporal_patch, 16, 16), stride=same    │
│     → (B, D, T', 14, 14)                            │
│                                                     │
│  ② [Spatial Bi-Mamba + Temporal Causal Mamba] × L   │
│     Spatial: 14×14 zigzag scan, bidirectional       │
│     Temporal: across frames, causal                 │
│                                                     │
│  Output: (B, D, T', 14, 14)                         │
└─────────────────────────────────────────────────────┘
    │
    ▼
Connector (동일 — learnable pos embedding)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  MambaGazeDecoder  (LlamaForCausalLM 대체)          │
│                                                     │
│  ① Token Embedding (vocab=197)                      │
│  ② Causal Mamba Blocks × L                          │
│     SSM 상태만 캐싱 (KV 캐시 없음!)                 │
│  ③ LM Head → gaze position logits                   │
│                                                     │
│  Streaming: past_ssm_states (96 KB, 고정 크기)      │
└─────────────────────────────────────────────────────┘
    │
    ▼
Gaze position tokens (autoregressive)
```

### 2.2 Spatial Scan 패턴

14×14 공간 토큰을 1D 시퀀스로 직렬화하는 방법:

```
Raster scan:      Zigzag scan (구현):
0  1  2  3        0  1  2  3
4  5  6  7        7  6  5  4
8  9  10 11       8  9  10 11
12 13 14 15       15 14 13 12
```

Zigzag은 인접 행의 연결성을 높여 SSM이 공간 경계를 더 잘 모델링한다.
VMamba(2024)의 SS2D처럼 4방향 스캔을 추가하면 더 강력하나, 프로토타입에서는 양방향 zigzag으로 충분.

### 2.3 핵심 알고리즘: Selective Scan (S6)

```
입력 u ∈ R^(L×D)에 대해:

1. 선택적 파라미터 계산 (입력 의존):
   B, C ∈ R^(L×N_state)  ← x_proj(u)
   Δ ∈ R^(L×D)           ← softplus(dt_proj(x_proj(u)[-1:]))

2. A 이산화 (ZOH):
   Ā_t = exp(Δ_t ⊗ A)    ∈ R^(D×N_state)
   B̄_t = Δ_t ⊗ B_t       ∈ R^(D×N_state)

3. 선택적 스캔:
   h_t = Ā_t · h_{t-1} + B̄_t · u_t   ← SSM 상태 업데이트
   y_t = C_t · h_t + D · u_t          ← 출력

4. 게이팅:
   out = y · SiLU(z)     (z는 in_proj의 절반)
```

훈련 시: parallel associative scan으로 O(N log N)  
추론 시: 순차 recurrence로 O(N), 상태 크기 O(d×N_state)

---

## 3. 파라미터 수 비교

| 구성 요소 | AutoGaze (LLaMA) | MambaGaze |
|-----------|-----------------|-----------|
| Vision Encoder | ~1.0M (Conv3D × depth) | ~0.6M (Conv + Mamba) |
| Connector | 37K | 37K |
| Decoder | ~2.0M (LLaMA) | ~1.0M (Mamba) |
| **합계** | **~3.0M** | **~1.6M** |
| 감소율 | — | **~47%** |

기본 설정 (d=192, 4 layers, d_state=16, expand=2):
- d_inner = 384
- SSM 파라미터/레이어: in_proj(192→768) + conv(384) + x_proj(384→33) + dt_proj(1→384) + out_proj(384→192) + A_log/D ≈ **~240K/layer**
- 4 layers ≈ 960K

---

## 4. 구현 파일 구조

```
autogaze/models/mamba_gaze/
    __init__.py
    configuration_mamba_gaze.py    # MambaGazeConfig
    modeling_mamba_gaze.py         # 모델 구현
        ├── SelectiveSSM           # Mamba core (S6)
        ├── MambaBlock             # SSM + norm + residual (causal / bidirectional)
        ├── MambaVisionEncoder     # ShallowVideoConvNet 대체
        ├── MambaGazeDecoder       # LlamaForCausalLM 대체
        ├── MambaGazeModel         # AutoGazeModel 대체
        └── MambaGaze              # AutoGaze 대체 (drop-in 호환)
```

---

## 5. 학습 전략

### 5.1 단계별 접근

**Stage 0: 벤치마크 (skip 가능)**
- 기존 AutoGaze 가중치로 MambaGaze의 초기 gaze 품질 측정
- 비교 기준 설정

**Stage 1: NTP Pre-training (동일)**
- 데이터: VideoMAE pseudo-label gaze 시퀀스
- 손실: Cross-entropy over gaze position tokens
- 차이점: Mamba decoder에는 teacher forcing이 LLaMA와 동일하게 적용됨

**Stage 2: GRPO RL Fine-tuning (동일)**
- 보상: VideoMAE reconstruction loss
- Mamba의 상태 기반 생성이 reward shaping에 더 안정적일 것으로 예상

### 5.2 초기화 전략

- Vision Encoder: AutoGaze의 ShallowVideoConvNet 가중치로 Conv embedding 부분 초기화 가능 (같은 구조)
- Mamba blocks: 표준 초기화 (A_log: log-spaced, D: ones, others: normal)

### 5.3 주의사항

1. **Causal vs Bidirectional**: 시각 인코더는 bidirectional OK (미래 프레임 정보 사용 가능). 디코더는 반드시 causal.
2. **Pure PyTorch 구현**: 프로토타입은 하드웨어 커널 없이 실행 가능하나, 대규모 학습 시 `mamba-ssm` 패키지의 CUDA 커널 사용 권장 (10-20x 속도).
3. **Numerical stability**: A_log에 음수 클리핑 필요 (A = -exp(A_log) < 0 보장).

---

## 6. 예상 효과 (추론 벤치마크 예측)

| 메트릭 | AutoGaze | MambaGaze (예측) |
|--------|---------|-----------------|
| 파라미터 수 | ~3M | ~1.6M |
| 추론 FLOPs (T=16, ratio=0.5) | ~2G | ~0.5G |
| 스트리밍 메모리 (1 청크) | 19 MB (KV) | 0.1 MB (SSM) |
| 첫 토큰 지연 (CPU) | ~50ms | ~15ms (예측) |

*실제 수치는 구현 및 하드웨어에 따라 다름*

---

## 7. 향후 개선 방향

1. **VMamba SS2D 스캔**: 4방향 스캔으로 공간 모델링 강화
2. **Mamba2 (SSD)**: State Space Duality로 더 빠른 학습
3. **하이브리드**: Mamba + 소수의 attention layer (긴 범위 의존성 보완)
4. **VideoMamba 사전학습 가중치 활용**: 사전학습된 temporal Mamba 가중치 전이
5. **Multi-scale Mamba**: 각 해상도에서 독립적 Mamba 인코더

---

## 참고 문헌

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [Vision Mamba (Vim): Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.13062) (Zhu et al., 2024)
- [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166) (Liu et al., 2024)
- [VideoMamba: State Space Model for Efficient Video Understanding](https://arxiv.org/abs/2403.06977) (Li et al., 2024)
- [Mamba2 / Structured State Space Duality](https://arxiv.org/abs/2405.21060) (Dao & Gu, 2024)
- [AutoGaze: CVPR 2026](https://github.com/NVlabs/AutoGaze)
