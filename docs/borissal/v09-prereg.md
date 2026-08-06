# v0.9 라운드 사전 등록 — λ0.75 확인 대결 + ACR(적응 체커보드 리프레시) 탐색

등록일: 2026-07-31 (실행 전 커밋). 검증 체계는 v08-prereg.md의 하네스를
그대로 사용(프레임 근거 블라인드 심판, 순서 스왑, KV-캐시 생성기 고정).

## 주 판정 — fine+λ0.75 확인 대결 (이월 후보)

- 대상: `v0.7,signal_grid=fine,anchor_novelty_lambda=0.75` vs `v0.7` 기본.
- 평가셋: **새 dev-60** (1K 풀에서 기존 dev-60/holdout-120과 서로소로 동일
  층화 추출, 시드 20260731). holdout-120은 재사용하지 않는다.
- 비교: 직접 쌍대(λ0.75 vs 기본) + 각자 vs-dense, 양 ratio(0.25, 0.5),
  스왑 2회.
- **채택 규칙**: 직접 쌍대에서 양 ratio 모두 λ0.75가 부호검정 p<0.05로
  이기면 즉시 v0.9 프리셋 채택. 한쪽 ratio만 이기고 다른 쪽 무승부
  (승률차 ±2%p 이내)여도 채택. 어느 ratio든 유의하게 지면 기각.
- Stage C 관찰 데이터(holdout에서 전 지표 동급-우위)가 사전 확률을 높이지만
  결정은 이 새 dev 대결로만 한다.

## 탐색 arm — ACR (구현: keyframe_refresh/keyframe_keep/keyframe_dynamic)

메커니즘(사용자 제안 GOP 개념의 일반화, 2026-07-31 구현·테스트 완료):
j개 리프레시 튜블렛(윈도우당 1개; dynamic=윈도우 내 novelty argmax,
아니면 중앙)에 패리티 교대 체커보드 큐브를 강제 — 연속 두 리프레시가
전 격자를 절반 밀도로 커버("I-프레임"). 나머지 튜블렛은 v0.7
앵커+novelty("P-프레임"), 리프레시 튜블렛에서는 앵커 제외(이중과세 방지).
고정 GOP는 dynamic=False의 특수 케이스, 회사 인터페이스 {4,8,16}는
16f/튜블렛2 기준 j∈{4,2,1}에 대응.

- arm (dev-60 신규셋, @0.25 주·@0.5 보조):
  1. `v0.7,keyframe_refresh=2` (dynamic, I_keep=0.5)
  2. `v0.7,keyframe_refresh=2,keyframe_dynamic=False` (고정 GOP 등가)
  3. `v0.7,keyframe_refresh=2,keyframe_keep=0.375`
  4. (주 판정 승자에 결합) `<승자>,keyframe_refresh=2`
- **가설(사전 명시)**: ACR은 @0.25 vs-dense의 objects/scene/환각 축 손실을
  줄이고 actions/temporal 축은 해치지 않는다. 축별 판정으로 직접 검증.
- **모션 분위 분해**: 평가셋의 모션 3분위별 승률을 보고, 동적 배치의
  이득이 어느 분위에서 오는지 기록 — 차기 dynamic GOP 스위칭 규칙의
  데이터 근거로 사용.
- 탐색 arm은 이번 라운드 프리셋 결정에 사용하지 않는다(관찰 전용).
  유망하면(주판정 승자 대비 vs-dense 손실 유의 감소) 다음 라운드에서
  holdout 확정 절차를 밟는다.

## 실행·통계

- 생성: eval_mllm_attach --generate --kv-cache (A5 고정), 새 dev-60,
  구성 = {λ0.75, v0.7, ACR arm 1-4, random} @0.25·0.5.
- 심판: 세션 에이전트(sonnet), 검증 스위트는 v0.8에서 통과한 프롬프트
  그대로 사용(프롬프트 변경 시 재검증).
- QC 게이트 동일: 길이 편향 r>0.4 오염, tie율>40% 폴백, 스왑 일치율 보고.
- 검정: 부호검정 양측 α=0.05, Wilson 95% CI.
