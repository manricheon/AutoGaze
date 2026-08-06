# Borissal v0.6 — saliency-v3.1 정제 반영 계획

## Context
사용자가 다운스트림 검증된 현재 best(saliency-v3.1)의 7단계 스펙을 공유. 대조 결과
borissal이 못 가졌거나 안 쓴 3가지를 v0.6에 반영(사용자 선택): ① 정적 appearance
가드, ② Laplacian 텍스처 억제, ③ center bias 재검증. 전부 v0.5 위에 얹는 **개별
토글 knob, 기본 OFF**. Mac 프록시로 무회귀 스크린 → 최종은 CUDA QA(프록시 반복
오판 전례). saliency-v3.1의 다운스트림 성공은 이 기능들의 외부 사전증거.

**핵심 통찰(v0.6 헤드라인)**: stage4(모션↑→텍스처 억제)와 stage6(모션0→엣지 보호)는
모순이 아니라 **국면 전환**. borissal은 appearance를 전역 motion_weight로만 섞음;
v0.6은 국소적으로 "모션 있으면 모션, 정지면 appearance-edge"로 외과 처리.

## Global Constraints
- v0.3/v0.4/v0.5 preset·거동 **불변**. 새 knob 전부 기본 OFF.
- Mac/CPU, torch-only, jit.trace+ONNX 통과 유지(정수 캐스팅·동적분기 회피).
- 속도: v0.5 대비 회귀 최소화 — 신호는 **그리드 해상도**에서 계산(v0.5 교훈).
- 프록시 결과는 전부 "CUDA QA 확인" 태그.

---

## 새 신호 primitive (`signals_v03.py`)

### 1. `laplacian_energy(grid_map)` 
3×3 Laplacian 커널 `[[0,1,0],[1,-4,1],[0,1,0]]` conv의 절댓값. 그리드(24×24 또는
튜블렛 그리드) 위에서 계산(값쌈). flat→0, 체커보드→큰 값. 텍스처(②)와 정적 엣지(①)
양쪽의 공용 연산.

### 2. `laplacian_texture_gate(motion_grid, lap_grid, r0, tau, eps)`  → ②
`R = lap_grid / (motion_grid + eps)` (모션 대비 고주파 밀도). 
`gate = sigmoid(-(R - r0)/tau)` ∈ (0,1) — R 높은(모션 없는데 텍스처 촘촘한) 영역을
정비례 억제. score에 곱. **coherence gate와 목적 중복 → 스윕에서 {coherence만 /
laplacian만 / 둘다} 비교, 기본은 겹쳐 쓰지 않음.**

### 3. `static_appearance_guard(luma_grid, motion_energy_t, thresh, tau, eps)` → ①
튜블렛별 정적 가중치 `s_t = sigmoid((thresh - motion_energy_t)/tau)` (모션 낮을수록 →1).
`guard = s_t * laplacian_energy(luma_grid)` (정적 슬롯의 텍스트/문서/인물 외곽).
score에 **가산**(`+ static_guard_weight * guard`). 모션 큰 슬롯은 s_t≈0 → 무영향.

---

## Config knob (`configuration_borissal.py`)
```
laplacian_gate: bool = False            # ② on/off
laplacian_gate_r0: float = 1.0          # 억제 임계 비율
laplacian_gate_tau: float = 0.5         # 시그모이드 온도
static_guard: bool = False              # ① on/off
static_guard_weight: float = 0.5        # 가산 강도
static_guard_thresh: float = 0.05       # 정적 판정 모션 임계(정규화 후)
static_guard_tau: float = 0.02          # 국면 전환 온도
# center_bias: 이미 존재 — 코드 변경 없이 v0_6 스윕에서 재검증
```
`v0_6(cls, **overrides)`: `base = dict(...)` 후 `cls.v0_5(**base)`. 셋 다 OFF가 기본
이므로 **knob 없는 v0_6 == v0_5**(회귀 없음 보장).

## 통합 (`modeling_borissal.py _saliency_scores` / 큐브 경로)
- luma_grid, motion_grid는 이미 계산됨(그리드 coherence 경로 재사용).
- `laplacian_gate`면: motion_grid·lap(motion_grid)로 texture gate → score에 곱
  (coherence gate 적용 지점과 같은 단계, 중복 방지 위해 배타 권장).
- `static_guard`면: 튜블렛 모션 에너지 → s_t → lap(luma_grid) 가산.
- score_coarsen(큐브) **전에** 적용해 12×12 풀링이 정제된 score를 받도록.

---

## 테스트 (`tests/test_borissal_v03.py` 추가)
- `laplacian_energy`: flat=0, 고주파 패턴 큰 값, 커널 홀수 assert.
- `laplacian_texture_gate`: 고R 억제, 저R 통과, 범위 (0,1).
- `static_appearance_guard`: 정적 튜블렛만 가산, 모션 튜블렛 무영향.
- `v0_6` preset: knob 없는 v0_6 == v0_5(scores 동일), 각 knob 개별 on 시 변화.
- `v0_6` jit.trace + ONNX export PASS.

## 스윕·시각화 (Mac 프록시)
- `sweep_borissal_v03.py`(또는 신규 eval): v0.5 vs v0.6 변형
  {static_guard / laplacian_gate / center_bias / 조합} 을 semantic recall +
  V-JEPA coverage로 16 held-out 스크린. 결과 차트 + 선택 오버레이(v0.5 vs 각 knob).
  static_guard는 문서·정적 클립에서 특히 관찰.
- VERDICT: 각 knob이 프록시 무회귀인지 + 어디서 이득/손해, 전부 "CUDA QA 확인".

## 산출물
- `dist/borissal_v06.py` (standalone, v06 preset 노출).
- design.md / v03-features-ko.md v0.6 절.

## 검증
- `uv run pytest tests/` (기존 95 + 신규 유지).
- `export_borissal_check.py` 계열로 v0.6 trace/ONNX.
- 스윕 무회귀 + 시각화.

## 순서
1. primitive 3종 + 단위테스트(TDD) → 2. config knob + v0_6 preset + 무회귀 테스트
→ 3. modeling 통합 + 통합테스트 → 4. trace/ONNX → 5. standalone 빌드 →
6. 스윕·시각화·VERDICT → 7. 문서.
