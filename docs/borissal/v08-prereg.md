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
