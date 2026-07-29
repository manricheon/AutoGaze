# v0.8 라운드 사전 등록 (pre-registration)

작성일 2026-07-28. 이 문서는 스윕·채점 **실행 전에** 커밋한다. 실행 후 규칙 변경 금지;
바꿔야 한다면 변경 사실과 이유를 이 파일에 추가 커밋으로 남긴다.

## 목적

selector→(V-JEPA2.1-L | Qwen3-VL/3.5-2B)→description 스택에서 정보 보존(오브젝트·
움직임·상황)을 최대화하는 v0.7 파생 구성을 찾고, **생성-후-심판** 검증 체계로 판정한다.
ratio 0.25 주전장, 0.5 보조.

## 평가셋

- dev-60 / holdout-120: `videos/internvid_pilot` 1K 풀에서 층화 추출
  (`scripts/make_evalset_manifest.py`, seed 20260728; 모션 3분위 × SigLIP2 k-means k=20,
  퇴화 필터). 매니페스트 = `docs/borissal/evalset_manifest.json` (프레임 sha256 포함).
- `videos/internvid_eval16`은 스모크/회귀 전용으로 강등 — 이 라운드의 의사결정에 사용 금지.
- **holdout-120은 Stage C에서 정확히 1회만 채점**. 결과가 어떻든 재방문 금지.

## 생성

- `scripts/eval_mllm_attach.py --generate`, Qwen3-VL-2B-Instruct, greedy, 96 tokens,
  prune_stage=encoder, partial_blocks=strict, 16f/384px, seed 0.
- 사전 재현성 게이트: 동일 구성 2회 실행 캡션 동일률 ≥90% (5클립). 미달 시 CPU 생성 전환.
- attention 백엔드는 전 비교에서 동일(로컬 기본 sdpa). CUDA에서도 sdpa 고정
  (FA2는 별도 진단 트랙에서만).

## 심판 (scripts/eval_caption_judge.py)

- 근거 = 동결 프레임 8장(타임스탬프 포함)이 GT. dense 캡션은 참조로 쓰지 않는다
  (2B 자기 환각 순환 방지). vs-dense 비교는 "드롭 크기" 보조 지표.
- 루브릭 5축: objects / actions / scene-context / temporal-order / hallucination.
- 쌍대 비교, A/B 순서 스왑 2회, 불일치 = tie. dev는 1회(스왑 2콜), holdout은 3회 다수결.
- 심판 백엔드: 1차 Claude(세션 에이전트), 2차 Gemini 무료 티어(승자 교차 검증만).
- **심판 검증 스위트(선행, 20클립) 통과 기준** — 하나라도 실패 시 프롬프트 수정 후 재검증,
  통과 전 본 채점 금지:
  1. dense가 random@0.10을 승률 ≥80%로 이김
  2. 타 클립 캡션(스왑)은 ≥90% 패배
  3. 문장 셔플 dense는 temporal-order 축에서 원본에 패배
  4. 순서 스왑 자기일치 ≥80%
  5. 장황 패딩 캡션이 간결 dense를 이기지 못함
- 게이밍 모니터: 캡션 길이 vs 승률 상관 r>0.4면 심판 오염으로 간주, 프롬프트 수정 후
  해당 라운드 재채점.

## Stage A — 저비용 스크린 (dev-20 = dev-60의 앞 20클립, 이름순)

- 라운드 1 (8구성): `anchor_fraction` {0.25, 0.5, 0.75, 1.0} × `signal_grid` {cube, fine}.
- 라운드 2 (~6구성): 라운드 1 상위 2구성에 `anchor_novelty_lambda` {0.25, 0.5, 0.75} →
  `novelty_shortterm_weight` {0.15, 0.3, 0.5} 좌표하강.
- 지표: SigLIP gist/recall + NLL(@0.25). **탈락 전용**: v0.7 기본 대비 SigLIP recall과
  NLL **양쪽에서** 1σ(NLL 0.0016 nats/tok) 초과 열세인 구성만 탈락. 프록시로 승자 선정 금지.
- Stage B 진출: 생존 구성 중 (NLL@0.25 오름차순) 상위 3.

## Stage B — 심판 판정 (dev-60)

- 대상: Stage A 생존 3 + v0.7 기본 + random. ratio 0.25(주), 0.5(보조).
- 비교: (a) 각 구성 vs random@동일비율 — 주 판정, (b) 각 구성 vs dense — 드롭 측정.
- **승자 규칙**: 0.25에서 vs-random 승률 최고. 동률(±2%p) 시 vs-dense 패배율 최소.
- 검정: 부호검정(양측 α=0.05), tie 제외 n 보고. 0.5에서 tie율 >40%면 0.25 단독 헤드라인
  (사전 등록된 폴백).

## Stage C — holdout-120, 1회

- 대상: Stage B 승자 + v0.7 기본 + random, 양 ratio, 심판 3회 다수결, 이항 CI 보고.
- **성공 기준**: 승자가 0.25에서 random을 유의하게 이기고(p<0.05), vs-dense 패배율이
  v0.7 기본 대비 감소. 충족 시 v0.8 프리셋 후보로 제안(프리셋화는 사용자 승인 후).
- 미충족 시: 결과를 negative로 기록하고 Phase 5(메커니즘 후보: ratio-적응 anchor_fraction,
  무버 지속성 항)로 넘어간다 — 그 역시 본 문서의 심판 체계로 게이트.

## 알려진 한계 (판정 시 고려)

- 심판 프레임 8장은 시간 해상도가 낮다 — 클립 내 모든 후보가 같은 프레임을 공유하므로
  쌍대 설계에서는 공통 핸디캡이지 편향은 아님.
- greedy 디코드는 near-tie에서 불연속 — n=60/120 평균화로 흡수, 재현성 게이트로 감시.
- dev-60 검정력: 승률 ~67% 이상 효과만 검출 가능(α=0.05, power 0.8). holdout-120은 ~62%.

## 수정 기록 (amendments)

### 2026-07-28 A1 — Stage A 라운드 1 af 그리드의 구조적 퇴화 (실측 발견, 규칙 변경 아님)

K_a = min(round(af·K_cubes), Sc=144) 캡 때문에 배치 예산에서 af 그리드가 붕괴한다:
r=0.25(16f)에서는 af ∈ {0.5, 0.75, 1.0}이 전부 앵커 144로 포화되어 **동일 선택**
(dev-20 SigLIP 수치 완전 일치로 확인), r=0.5에서는 af=0.25까지 포화된다. 따라서:
- Stage A NLL 스크린(@0.25)은 중복 제거된 5구성만 실행:
  {v0.7, v0.7,af=0.25, v0.7,fine, v0.7,af=0.25,fine, random}
- af 튜닝은 배치 예산에서는 죽은 knob (af<~0.29@0.25에서만 유효) — 저예산(0.1/0.15)
  전용 knob으로 재분류. 라운드 2(λ, novelty_shortterm_weight)가 실질 레버.
- 이는 커버리지 포화의 재확인이기도 함: r≥0.25에서는 af≥0.5면 전 사이트 커버가
  이미 달성된다 (설계 문서의 커버리지 절과 일치).

### 2026-07-28 A2 — 심판 백엔드에 gemini-cli 추가

Gemini API 무료 티어(키 발급) 대신/병행으로 **Gemini CLI**(OAuth 로그인, 무과금)를
2차 심판 백엔드로 사용 가능하게 함 (`eval_caption_judge.py gemini-cli`). 프로토콜
불변 — verdicts.jsonl 스키마 동일.

### 2026-07-28 A3 — 생성 프롬프트의 구성 요소화

배포에서는 별도의 user/system prompt가 쓰일 예정(사용자 공지). 이에 따라
`eval_mllm_attach.py`에 `--prompt`/`--system-prompt` 옵션을 추가하고 원칙을 명시한다:
**생성 프롬프트는 비교 라운드 내에서 고정**하고 results.json의 args에 그대로 기록한다.
프롬프트 변경은 새 라운드(또는 명시적 프롬프트 축 실험)로 취급 — 라운드 간 원점수
비교는 무효, 심판 프로토콜(프레임=GT)은 프롬프트와 무관하게 유효.

### 2026-07-29 A4 — 치명 버그: mllm_attach 입력 이중 스케일링 (역대 NLL 전부 무효)

심판 검증 스위트 준비 중 QC 캡션 20/20이 "검은 화면"을 묘사 → 동결 프레임은 정상
→ 추적 결과 `_processor_inputs`가 [0,1] float를 넘겨 프로세서 do_rescale(×1/255)이
이중 적용, pixel_values가 상수 -1.0(std 0.002)로 붕괴 — **모델은 검은 영상을 봤다.**
uint8 변환 + std<0.05 시 즉시 예외(tripwire)로 수정, 1클립 검증 통과.

무효화(2026-07-29 이전 eval_mllm_attach 산출 전부):
- v07_gate/v07_sg/v07_review의 NLL 표 전체 (v0.5 vs v0.7, cube-vs-fine tie,
  E-A/E-B NLL, "random이 NLL@0.5에서 승리(P1-echo)" 주장 포함)
- signal_grid="cube" 기본값 결정의 NLL-tie 근거 (SigLIP 근거는 별도 파이프라인이라 유효
  — 결정 자체는 유지, 근거 축소로 기록)
- 재현성 게이트 100%는 검은 입력에서 측정 → 실입력으로 재검증 필요
유효 잔존: SigLIP gist/recall 전 결과, coverage/uniqueness, latency, keep-all
비트 동일성 테스트(입력 무관), 셀렉터 쪽 전부.

교훈(기록): NLL은 검은 입력에서도 그럴듯한 델타를 냈다 — 캡션도 같은 검은 입력에서
생성된 자기일관 폐루프였기 때문. **생성-후-프레임근거-심판이 이 버그를 즉시 잡았다**
— 이 라운드가 검증 고도화를 최우선으로 둔 이유의 실증.

### 2026-07-29 A5 — KV 캐시 디코더를 라운드 고정 생성기로 채택

비캐시 greedy(캡션당 96 전체-forward)로는 Stage B가 클립당 ~38분(총 ~37h)이라
KV 캐시 디코더(`--kv-cache`)를 구현. 게이트 실측:
- 속도 4.8배 (5클립 게이트 2961s → 612s)
- NLL은 비트 동일 (+0.27235/+0.43300 완전 일치 — 수학적 등가 확인)
- 비캐시 대비 캡션 동일률 53% (bf16 누적 순서 차이가 근소 동점 토큰을 뒤집는
  기지의 현상; 품질 저하 아님)
- **캐시 디코더 자체 결정성 100%** (같은 설정 2회, 15/15 동일)

처리: Stage B는 아직 어떤 캡션도 채점하지 않았으므로, 캐시 디코더를 **본 라운드의
고정 생성기**로 채택한다(Stage B와 Stage C 모두 --kv-cache로 통일; 혼용 금지).
"생성" 절의 재현성 게이트는 캐시 디코더 기준으로 충족(100%).

### 2026-07-29 A6 — Stage B 보조 ratio 0.5 판정을 Stage C로 이관

Stage B는 "0.25(주), 0.5(보조)"로 등록했으나, 승자 규칙은 0.25 단독으로
정의되어 있고 0.5 데이터는 어떤 Stage B 결정에도 입력되지 않는다. 0.5의
평가는 Stage C(holdout-120, 승자+기본+random, 양 ratio, 3회 다수결)가
그대로 커버한다. 따라서 dev-60에서의 0.5 생성(~5h)+판정(~1080건)을 생략하고
Stage C의 0.5 결과를 보조 ratio의 유일한 판정으로 삼는다.
이 수정은 Stage B 0.5 데이터가 존재하기 전(어떤 0.5 캡션도 생성 전)에
기록되었다 — 결과를 보고 내린 결정이 아님. 0.5 tie율 >40% 폴백 조항은
Stage C의 0.5 결과에 동일하게 적용한다.
