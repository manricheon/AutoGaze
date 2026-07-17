# Borissal v0.3 — 디자인 스펙 (2026-07-17)

비학습 selector의 다음 이터레이션에 대한 승인된 디자인. 이 스펙은 구현
계획의 계약(contract)이며, 측정 결과와 채택/기각 판정은 v0.2 전례대로
`design.md`(개발 로그)에 기록한다. 동반 문서: `reference.md`(사용자용
노브 설명, 채택 시 갱신), `approach-ko.md`(한글 해설, 채택 시 갱신).

## 1. 포지셔닝과 목표

v0.3은 **단일 메커니즘이 아니라 "후보 뱅크 + 정량 게이트 + 조합 탐색"**
이다 — v0.2를 만든 프로세스(7개 요소를 구현하고 각각 게이트로 판정,
일부는 negative result로 기각)의 확장판. 모든 후보는 독립 config
노브이며, **전부 off면 v0.2와 비트 단위로 동일**하다(회귀 테스트로
잠금). 게이트를 통과한 조합만 v0.3 프리셋에 채택한다.

사용자 지정 우선순위 (2026-07-17), 순서대로:

1. **Semantic recall** — 설명(description) 태스크 정렬 축 (held-out
   16클립 세트에서 SigLIP2 MAP-head recall; v0.2 기준선 0.325 ± 0.022).
2. **엣지/텍스처 편향 완화** — luma 그라디언트는 반복되는 배경
   텍스처에 발화하고, 객체는 경계에만 발화할 뿐 내부는 비어 있다.
3. **시간적 선택 안정성** — 이전에는 스트리밍 UI 관심사로 분류되어
   의도적으로 제외했으나 이제 명시적 타겟. 단, recall 비열화 조건
   하에서만 채택한다(§5).
4. **카메라 모션 강건성** (1에 포섭: 팬/줌이 frame-diff를 화면 전체에
   발화시켜 예산을 낭비하는 문제).

v1의 입력 특징 뱅크가 풍부해지는 것은 기대되는 부수 효과이지 게이트가
아니다.

제약 (v0 계열에서 불변):

- **순수 고전 신호만** — 학습된 가중치 없음, 동결(frozen) 사전학습
  모델도 없음 (하이브리드/동결 경로는 v0.4+ 후보 방향으로, 여기서는
  명시적으로 범위 밖).
- **모바일 안전 연산만**: conv/pool/elementwise/topk. 새 코드 경로에
  FFT 금지, 일반 sort 금지, connected components 금지, 순차 raster
  scan 금지. 고정 소형 matmul(예: 상수 행렬로서의 DCT)은 허용 —
  matmul은 딜리게이트 네이티브 연산이다.
- **지연 예산: 클립당 ≤ 25ms** (개발 머신 CPU 기준; v0.2 프리셋은
  15.4ms → 신규 메커니즘 여유분 ≈ 10ms). v0.3의 비용은 v1의
  `_grid_inputs` 안에도 들어가므로 저렴함이 두 배로 중요하다.
- **`Selection` 출력 규약과 canonical 오름차순 `keep_index` 규약은
  불변** (다운스트림 attachability).
- 완전 벡터화 torch — batch/frame/patch에 대한 Python 루프 금지.

## 2. 후보 뱅크

3개 티어. Tier 1은 무조건 구현; Tier 2는 Tier 1 조합이 정체된 축에만
구현(§5 트리거 규칙); Tier 3은 완결성을 위해 기록만 하고 의도적으로
구현하지 않는다.

### Tier 1 (지금 구현 — 밀리초당 가치 최상)

| id | 메커니즘 | 원리 & 핵심 문헌 | 타겟 | 비용 추정 |
|---|---|---|---|---|
| `motion_cs` | **모션 center-surround**: 풀링된 diff 에너지 `D`를 `relu(D − avgpool_large(D))`로 대체 | 균일한 ego-motion은 평평한 diff 필드를 만들고 그 center-surround 차이는 ≈ 0; 독립적으로 움직이는 객체는 국소 피크로 살아남는다 (Itti motion conspicuity 1998; Mahadevan & Vasconcelos 2010, 단순화판) | 카메라 | <0.5ms |
| `coherence_gate` | **구조텐서 coherence 게이팅**: 기존 그라디언트 채널에 `grad × (1 − coherence)^γ`, coherence는 Gaussian 평활된 그라디언트 곱들로부터 닫힌형 `((a−c)² + 4b²)/(a+c+ε)²` (eig 불필요) | 반복 격자무늬와 긴 직선 엣지는 coherence 최대(λ1≫λ2)라 억제되고, 다방향 미세구조를 가진 객체 내부(λ1≈λ2)는 살아남는다 (Harris 1988; Förstner 1987; Weickert 1999) | 텍스처 | ~1.5ms |
| `signature` | **Image signature** (그리드/저해상도): DCT를 고정 matmul(`D @ X @ D.T`)로, `sign()`, 역 matmul, 제곱, 소형 블러 — *FFT 아님* | sign만으로 재구성하면 에너지가 공간적으로 희소한 전경에 집중되고 스펙트럼이 희소한(주기적) 배경은 죽는다; 경계가 아니라 객체 지지영역(support)에 발화 (Hou, Harel & Koch, TPAMI 2012) | recall, 텍스처 | ~0.5ms |
| `color_rarity` | **전역 색 희소성**: 패치별 평균 색을 K≈32 고정 중심에 soft-binning(matmul 1개), 희소성 = 거리 가중 역 히스토그램 질량; 축퇴 폴백 = 전역 평균색까지의 Mahalanobis 거리 (Achanta 2009) | 희소한 색은 설명에 유의미한 객체를 표시하며 **내부**에 균일하게 발화한다 (빨간 차 전체가 발화하지, 실루엣만 발화하지 않는다) (Cheng et al., CVPR 2011, HC 변형 — 세그멘테이션 불필요) | recall, 텍스처 | ~0.5ms |
| `dog_blob` | **다중 스케일 DoG blob 채널**: tubelet 평균 luma에 avgpool 피라미드 차분, \|·\|, 스케일(그리드 2~6칸)에 걸친 max | 객체는 어떤 스케일에선 blob이다; coarse DoG 극값은 내부에 발화 (Lindeberg 1998) — 가장 싼 내부 채움 장치 | 텍스처 | <1ms |
| `fusion_norm` | **내용 적응형 채널 융합**: Itti의 N(·) 피크 촉진(각 맵을 `(M − m̄_localmax)²`로 스케일) 그리고/또는 하한 있는 엔트로피 게이팅(맵의 질량 정규화 엔트로피 ↑ ⇒ 가중치 ↓, 하한 0.3) (softmax는 정규화된 [0,1] 맵에서 no-op — 구현 중 증명되어 선형 질량 정규화로 대체) | 화면 전체에 발화하는 맵(텍스처 위 그라디언트; 팬 중의 모션)은 블렌딩 전에 *융합 가중치*가 뭉개진다 — 맵 수준 텍스처 억제 + 공짜 카메라 폴백 (Itti 1998; 엔트로피 변형은 vid-TLDR CVPR 2024) | 텍스처, 카메라, 융합 | ~0ms |
| `score_ema` | **시간 점수 EMA**: `S̄_t = α·S̄_{t−1} + (1−α)·S_t`, 클립 안에서는 하삼각 감쇠 행렬 matmul 하나로 언롤(루프 없음); 스트리밍은 그리드 모양 상태 텐서 1개를 이월 | leaky-integrator 증거 누적; 고전적 시간 필터링 | 안정성 | ~0ms |
| `select_hysteresis` | **선택 hysteresis**: 직전 tubelet에서 선택된 패치에 topk 전 가산 보너스 ε | EMA의 이산 사촌 — 점수를 평활하지 않고 선택 집합을 안정화 | 안정성 | ~0ms |

### Tier 2 (Tier 1이 정체된 축에만 구현 — §5 트리거 규칙)

| id | 메커니즘 | 원리 & 문헌 | 타겟 | 비용 추정 |
|---|---|---|---|---|
| `distinctness` | 전역 자기유사성 기반 패치 distinctness: 셀별 기술자(Lab 평균 + 소형 그라디언트 통계), 576×576 Gram matmul, distinctness = 1 − top-k 유사도 평균, 사전계산된 공간 할인 마스크(r칸 이상 떨어진 셀끼리만 비교) | 반복 텍스처는 near-duplicate가 많아 텍스처 전체에서 distinctness ≈ 0; 객체 패치는 전역적으로 희소하다 (Goferman 2010; Margolin 2013) | recall, 텍스처 | ~0.5ms |
| `border_prior` | Boundary-connectivity 배경 prior: 모든 셀과 경계 ~92셀의 유사도(소형 matmul 1개); bgness가 융합 점수를 곱셈으로 게이팅 | 배경 텍스처는 외형상 프레임 경계에 연결된다; *그라디언트가 높아도* 죽인다 (Zhu et al., CVPR 2014, soft 변형 — superpixel 불필요) | 텍스처 | ~0.1ms |
| `mbd` | Minimum-barrier distance 병렬 근사: 경계에서 시드된 셀별 (hi, lo) 경로 극값, 그리드 해상도에서 shift(`roll`/pad) + elementwise min/max의 Bellman–Ford 반복 K≈24회 | 돌출 영역은 경계로 가는 모든 경로의 barrier가 높다; 가장 강력한 고전 내부 채움 장치, 완만한 조명 램프에 강건 (Zhang et al., ICCV 2015 — 발표된 raster-scan 판은 showstopper; 이것은 고정 반복 병렬형) | recall, 텍스처 | ~1–2ms |
| `gme` | 적분 프로젝션 전역 모션 추정 + 보상 diff: 프레임별 행/열 평균 프로파일, shift ∈ [−16, 16]에 대한 conv1d 상관, 승자는 topk-1로, shift 적용은 33-way one-hot pad+slice(정적 그래프 안전, 동적 `roll` 없음) | 정확한 고전 팬/틸트 해법 (Alliney & Morandi 1986; Dufaux & Konrad 2000). 평행이동 전용; 줌/회전은 프로파일을 탈상관시켜 추정치 ~0 → `motion_cs`로 우아하게 폴백 | 카메라 | ~1–2ms |

### Tier 3 (기록만, 의도적 미구현)

- **BMS 근사** (경계 시드 반복 maxpool 전파) — `mbd`와 역할이 겹치고,
  내부 채움 단위 비용은 `mbd`가 더 싸다.
- **Gabor 방향 대비 뱅크** — 채널 비용 4배; anti-texture 역할은
  `coherence_gate`가 훨씬 싼 값에 수행.
- **Coarse 블록 매칭 dominant-motion 필드** — 다운샘플해도 ~5–10ms;
  `gme` + `motion_cs`가 1/4 비용으로 ~80%를 커버.
- **선명도/defocus prior** — bimodality 신뢰 게이팅이 필요하고 도메인
  취약(깊은 심도 폰 영상); `center_bias`처럼 도메인별 재고 대상.
- **Spectral residual / PQFT (DFT-matmul판)** — 원하면 *실험 노브*로
  허용(`double_diff` 전례)하되, 같은 역할을 `signature`가 더 나은
  연산 안전성으로 차지한다.
- **GBVS, RARE2012, Kadir–Brady, radial symmetry, BING, 배경 차분
  GMM, phase correlation** — 조사 후 기각 (각각 중복, scatter 연산
  의존, 학습 가중치 포함, 정적 카메라 가정, FFT 의존).

## 3. 파이프라인 통합

신호 흐름 (대괄호가 신규 노브; 나머지는 v0.2 그대로):

```
디코드 → luma (기존) + 그리드/하프 해상도 RGB 또는 Lab (신규, 색 후보용)

모션 경로:   frame diff → [gme 보상] → pool → [motion_cs]
             → noise floor (순서: floor는 motion_cs 뒤)
공간 경로:   gradient → [coherence_gate] → pool
신규 채널:   [signature] [color_rarity] [dog_blob]      (그리드 해상도)

융합:        채널별 정규화 → [fusion_norm 가중]
             → N채널 블렌드 (motion_weight="auto"를 2~5채널
               에너지 가중으로 일반화)
시간:        [score_ema] → center bias → 할당 (global + floor,
             spread_fraction) → coherent-region 블록 게이트
선택:        [select_hysteresis] → topk → _pack_gazing_mask (불변)
```

노트:

- **유일한 구조 변화는 색이다**: 현재 파이프라인은 luma 전용.
  RGB/Lab은 그리드(24×24) 또는 하프 해상도로만 유지해 디코드/메모리
  추가 비용을 무시 가능한 수준으로 묶는다. Lab 변환은 고정 3×3
  matmul + elementwise(모바일 안전); RGB로 시작하고 rarity 품질이
  요구할 때만 Lab을 채택한다.
- **순서 규칙은 스펙의 일부다** (v0.2의 double_diff negative result가
  가르친 상호작용을 코드화): quantile noise floor는 `motion_cs`
  *뒤에* 실행 (center-surround 먼저, dead-zone 다음); 채널별 정규화는
  `score_ema` *앞에* 실행 (raw 스케일 맵을 EMA하지 않는다);
  `fusion_norm`은 quantile floor 뒤에 실행 (노이즈 스파이크 하나가
  피크 촉진을 이기지 못하게); heavy-tail 채널(`color_rarity`,
  `distinctness`)은 min-max 전에 log/sqrt 압축을 거치고 per-tubelet
  보다 클립 전역 정규화를 선호한다.
- 신규 채널 가중치는 기존 블렌드 장치에 편입: 2채널 `motion_weight`가
  N채널 가중 벡터가 되고, `"auto"` 모드는 현행 에너지 비율 규칙을
  일반화한다 (`w_i ∝ energy_i`, 정규화 전 계산, 선택적으로
  `fusion_norm`이 변조).
- `score_ema`와 `select_hysteresis`는 스트리밍 훅을 유지해야 한다:
  클립 안에서는 루프 없음(감쇠 matmul / shift된 마스크), 클립 사이는
  각각 정확히 상태 텐서 1개를 이월.

## 4. 평가 게이트 (인프라 불변, 판정 규칙 고정)

| 게이트 | 스크립트 | 규칙 |
|---|---|---|
| **Semantic recall (주 지표)** | `scripts/eval_borissal_semantic.py`, held-out `videos/internvid_eval16/`, ratio 0.25 (0.5 스팟체크) | 높을수록 좋음; ±0.02(세트의 관측 표준편차 수준) 이내 차이는 동률 — 동률은 평균이 아니라 클립별 짝지은(paired) 비교로 판정 |
| Coverage / uniqueness | `scripts/eval_borissal_coverage.py`, 동결 V-JEPA2 | **둘 다 나빠지면 안 됨** (v0.2 규칙; coverage 단독 판정 금지 — scatter 편향) |
| 지연 | `scripts/borissal_benchmark.py` | 조합 프리셋 ≤ 25ms CPU; 노브별 증분 보고 |
| Export | `scripts/export_borissal_check.py` | 채택된 노브를 켠 상태로 jit trace + ONNX opset-17 통과 |
| 시각 | 오버레이 덤프 (`borissal_dump_outputs.py`) | 정성 확인용, pass/fail 게이트 아님 |
| VideoMAE recon / gist | 기존 스크립트 | 참조 축 전용 — 보고는 하되 절대 최적화 대상 아님 |

## 5. 조합 탐색 프로토콜

8~12개 이진 노브(+하이퍼파라미터)의 전수 탐색은 불가능하고 불필요하다.
v0.2 프로세스를 반영한 3단계:

1. **솔로 스크리닝.** Tier-1 각 후보를 v0.2 프리셋 위에 단독으로,
   기본 하이퍼파라미터로, 4개 게이트 전부에 대해 평가. 판정: KEEP
   (recall 상승, 또는 동률이면서 다른 축이 뚜렷이 상승), KILL (동률
   범위를 넘는 recall 하락, 또는 cov/uniq 둘 다 열화, 또는 export
   실패), TUNE (명백한 하이퍼파라미터 하나만 바꿔 1회 재시도 — 예:
   surround 반경, K bins, α). KILL은 negative result로 기록하고
   메커니즘은 off 기본값의 실험 노브로 유지 (`double_diff` 전례).
2. **탐욕적 전진 조합.** v0.2 프리셋에서 시작; 생존자를 솔로 recall
   내림차순으로 하나씩 추가; 조합 config가 모든 게이트를 통과하고
   직전 단계 대비 recall이 떨어지지 않을 때만 유지 (동률은 클립별
   비교). 남은 생존자가 도움이 안 되면 종료. 탐욕 순서와 무관하게
   표적 상호작용 체크 2건: (a) `fusion_norm` × 채택된 각 신규 채널
   (융합 가중은 다른 노브의 판정을 뒤집을 가능성이 가장 큰 메커니즘),
   (b) `score_ema` × `select_hysteresis` (중복 안정화 장치 — 둘 다
   독립적으로 recall 조건을 통과하지 않는 한 최대 하나만 채택).
3. **프리셋 승격.** 승리 조합이 다음을 통과하면 v0.3 프리셋(v0.2처럼
   이름 붙은 config)이 된다: v0.2 대비 (recall, cov/uniq) Pareto
   규칙, 지연 ≤ 25ms, export 체크, 오름차순 인덱스 테스트, 오버레이
   검토. 안정성 후보(`score_ema`, `select_hysteresis`)는 **recall이
   비열화일 때만** 채택 (늦게 등장하는 객체의 recall을 안정성과
   맞바꾸는 장치인데, 설명 태스크는 recall에 돈을 낸다). Tier-2
   트리거: Tier-1 조합이 어떤 축에서 무이득일 때만 그 축의 Tier-2
   후보를 구현 (예: 텍스처 편향이 그대로 → `distinctness`/
   `border_prior`/`mbd`; 팬 클립이 여전히 범람 → `gme`).

스윕 러너(`scripts/sweep_borissal_v03.py`)가 1~2단계를 자동화한다:
후보 목록을 받아 config별 게이트를 실행하고 결과 테이블(JSON +
markdown)을 `outputs/borissal/v03_sweep/`(gitignored)에 덤프하되,
**채택 결정은 스스로 내리지 않는다** — 판정은 테이블을 읽은 뒤
`design.md`에 수동으로 기록한다 (v0.2와 같은 규율).

## 6. 산출물

- `autogaze/models/borissal/modeling_borissal.py` — 신규 노브 (전부
  off 기본값); `configuration_borissal.py` — config 필드.
- `scripts/sweep_borissal_v03.py` — 1/2단계 스윕 러너.
- 테스트 (`tests/test_borissal.py` 확장):
  - 전 노브 off 출력이 v0.2와 정확히 동일 (텐서 수준 회귀);
  - 각 노브가 그것을 격발하도록 설계된 합성 클립에서 선택을 바꾸는지
    (예: `motion_cs`용 패닝 텍스처, `coherence_gate`/`color_rarity`용
    말뚝 울타리 + 유색 blob);
  - 모든 노브를 켠 상태에서 오름차순 `keep_index` 불변식;
  - 채택 프리셋의 trace/export 스모크;
  - 스트리밍 상태 등가성: 클립 전체의 EMA/hysteresis = 상태를 이월한
    반클립 2회 실행.
- 채택 시 문서: `reference.md` §2/§3 (메커니즘 + 노브), `design.md`
  (측정, 채택/기각 판정, negative results), `approach-ko.md` (해설
  갱신).

## 7. 리스크와 완화

- **Held-out 평가 분산** (n=16에서 클립별 recall 산포 0.23–0.45):
  §4/§5의 ±0.02 동률 규칙 + 클립별 짝지은 비교가 이를 위해 존재;
  판정이 계속 모호하면 후보 뱅크를 늘리기 전에 평가 세트를 먼저
  늘린다.
- **색은 구조 변화다**: 그리드 해상도로만 유지하고 노브 뒤에 격리;
  전 노브 off 회귀 테스트가 luma 전용 경로 불변을 보장.
- **안정성 vs recall 긴장**: 명시적 채택 조건 (§5.3).
- **노브 폭발**: v0.3은 이미 노브가 많은 config에 ~8개를 더한다.
  완화: 프리셋이 제품이고 노브는 탐색 공간이다. KILL된 것은 전부
  off 기본값의 실험 노브로 문서화되고, Tier-3는 아예 구현하지 않는다.
- **example clip에 대한 하이퍼파라미터 과적합** (v0.2의 교훈):
  1/2단계 판정은 held-out 16클립 세트로만 하고 example clip 단독으로
  절대 하지 않는다; example clip은 오버레이/디버깅용이다.

## 7.5 후속 아이디어 기록 — E5: 학습형 프레임 간 예산 할당 (2026-07-17, 사용자 제안)

토큰 버짓 K_total이 항상 입력으로 주어진다는 전제 하에, 프레임(tubelet) 간
top-k **배분**을 학습하게 하자는 제안. 선택을 (a) 프레임 내 위치(per-patch
점수, v1이 학습)와 (b) 프레임 간 수량(현재 uniform/proportional/global+floor
규칙)으로 분해하면 이것은 (b)의 학습화다. 근거: v0.2에서 global
allocation + floor가 단일 요소로 최대 이득(uniqueness +0.07) — 할당 축은
학습 최적화 여지가 입증됨. 스케치: v1 트렁크에 초소형 할당 헤드(softmax
분율 × K_total → `_largest_remainder` 반올림, 총예산 정확 보존; Selection의
가변 `per_frame_keep`가 이미 지원). 학습 신호 후보: soft top-k 확장 /
REINFORCE(AdaMAE 전례) / **oracle 증류(teacher 한계 이득 기반 탐욕 할당,
권장 출발점)**. 함정: coverage 계열의 scatter 편향(P1)이 할당에도 작용
(uniqueness-primary + floor 제약 필요); 데이터 의존 분할은 mobile review의
proportional trace 고착 버그와 같은 클래스(온디바이스는 동적 export 필요).
v0.3과 상보적 — v0.3 채널들이 할당 헤드의 tubelet별 입력 특징이 된다.
구현은 v1 실험 매트릭스(training.md) 쪽 후속 작업.

## 8. 범위 밖

- 학습 가중치나 동결 사전학습 모델 일체 (v0.4+ 후보 방향, 사용자
  결정: "일단 순수 고전 가자").
- v1 아키텍처나 학습 레시피 변경 (v1은 `input_mode=maps|both`를 통해
  v0.3 맵을 자동으로 받는다; 풍부해진 뱅크로 v1을 재학습/재게이팅하는
  것은 별도 후속 작업).
- `Selection` 규약, 패킹, 어댑터, 다운스트림 인터페이스.
- 다중 스케일 / summary-token 출력 (다운스트림 규약 변경,
  `progress.md` 2026-07-16에 명시적으로 보류됨).
