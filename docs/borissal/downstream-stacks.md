# 다운스트림 스택 다양화 (2026-07-26)

## 0. 무엇을 고정하고 무엇을 바꾸는가

**셀렉터(v0.x / v1)는 고정**이다 — 계속 개발하는 대상이고, 이 문서의 변경 대상이 아니다.
바꾸는 것은 **셀렉터 뒤에 오는 스택**이다:

```
[borissal 선택]  →  ( 인코더 ,  LLM )
   고정                 ^^^^^^^^^^^^^  다양화 대상
```

현재 다운스트림은 **V-JEPA 2.1-L + Qwen 캡셔너 → 외부 LLM QA** 하나뿐이다. 모든 v0.4–v0.6
판정이 이 단일 스택에 기대고 있고, 그래서 "v0.6이 좋다/나쁘다"가 **스택 특이적인지 일반적인지
구분할 수 없다**. 다양화의 목적은 그 구분이다.

선택 방식은 어떤 스택에서도 동일하다 — **인코더 앞에서 패치를 고른다**(`encoder-free-attach.md` §0의
세 variant). 인코더가 없는 스택은 그 방법의 극단 케이스일 뿐이다.

## 1. 제약: 임의의 (인코더, LLM) 쌍은 커넥터 학습이 필요하다

LLM은 **자기와 함께 정렬 학습된 인코더의 임베딩만** 이해한다. 따라서 다양화 경로는 둘로 갈린다.

| 방향 | 비용 | 설명 |
|---|---|---|
| **(i) 사전정렬 스택을 통째로 교체** | **학습 0** | 이미 (인코더+LLM)이 함께 학습된 VLM을 가져와 그 인코더 앞에서 자른다. Qwen3-VL 경로가 이것 |
| (ii) V-JEPA를 유지하고 LLM만 교체 | 커넥터(projector) 학습 필요 | LLaVA stage-1 정렬. CUDA 작업이고 데이터·시간이 든다 |

**(i)이 압도적으로 싸므로 다양화의 주력은 (i)이다.** (ii)는 "V-JEPA 표현을 유지해야 하는" 특정
이유가 있을 때만 (§5).

## 2. 핵심 축: 인코더의 **시간성** — 어떤 프리셋이 이길지 예측하는 변수

design.md의 v0.5 판정에 이미 메커니즘이 적혀 있다: *"다운스트림의 인코더 자체가 시간
모델(V-JEPA)이면 모션은 중복이다 — 인코더가 토큰 위치로 dynamics를 추론하므로, 셀렉터의 일은
인코더가 발명할 수 없는 appearance/객체/텍스트를 보존하는 것"*. 그래서 v0.4(모션 강화)가
다운스트림에서 **졌다**.

이 논리를 그대로 밀면 **인코더의 시간성이 약해질수록 모션 선택의 가치가 올라간다**. 소스에서
확인한 시간성 사다리:

| 단계 | 인코더 | 근거(실측) |
|---|---|---|
| **완전 시간적** | V-JEPA 2.1-L | 인코더가 `hidden_states` 하나로 **전체 시공간 집합에 attention**(세그먼트 분할 없음) + tubelet 2 |
| **약한 시간적** | Qwen3-VL / Qwen3.5 자체 ViT | `temporal_patch_size=2`로 2프레임을 conv로 접지만, **`cu_seqlens = repeat_interleave(h*w, t)`이므로 ViT attention은 temporal slice 내부로 제한**된다 → 프레임 간 혼합은 LLM에서만 |
| **순수 프레임별** | SigLIP / InternViT / CLIP / Pixtral / MoonViT | 프레임(또는 타일)을 독립 인코딩. 시간 정보는 **선택된 패치를 통해서만** LLM에 도달 |

### 사전등록 가설 (다양화의 1차 산출물)

1. **v0.4(frame-rate-aware motion)는 순수 프레임별 스택에서 이겨야 한다.** V-JEPA에서 진 이유가
   "모션이 중복"이었다면, 시간 모델이 없는 곳에서는 모션이 **유일한 동작 정보 경로**가 된다.
   → v0.4를 죽은 카드로 두지 말고 프레임별 스택에서 재평가한다.
2. **v0.5의 appearance-first(`motion_weight="auto"`)는 V-JEPA 특화일 수 있다.** 프레임별
   스택에서 v0.3/v0.4에 밀리면 그건 "v0.5가 나쁘다"가 아니라 "v0.5는 시간 모델용 프리셋"이라는
   뜻이고, **프리셋을 다운스트림별로 갈라야 한다**는 결론이 된다(lineage 표에 이미 그 예고가 있다).
3. **v0.6(saliency-v3.1 포트)의 다운스트림 검증은 원래 프레임별 파이프라인에서 나왔다.** 즉
   v0.6은 프레임별 스택에서 더 잘 나올 가능성이 있고, 현재 V-JEPA 스택에서의 프록시 열세와
   충돌하지 않는다.

이 세 가설은 **단일 스택에서는 원리적으로 판정 불가능**하다. 그게 다양화가 필요한 이유다.

## 3. 사전정렬 스택 후보 (학습 0, 인코더 앞 어태치)

params/라이선스는 HF API, 기하는 config 실측. `?`는 미확인(추측하지 않음).

| 스택 | 인코더 | LLM | 시간성 | 기하 | 라이선스 | tf 5.5.0 | 어태치 |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL-2B** | 자체 ViT (24L/1024) | Qwen3 2B | 약한 | patch16 / merge2 / tpatch2 | apache-2.0 | ✅ | **완료** |
| Qwen3-VL-8B/32B | 동일 계열 | Qwen3 | 약한 | 동일 | apache-2.0 | ✅ | 같은 코드 |
| Qwen3.5-2B | 자체 ViT | Qwen3.5 2B | 약한 | 동일(deepstack 없음) | apache-2.0 | ✅ | 같은 코드 |
| **InternVL3.5-8B** | **InternViT-300M** (`intern_vit_6b`, 24L/1024) | **Qwen3-8B** | **프레임별** | patch14 / tile448 / `downsample_ratio=0.5` / 동적타일 1–12 | apache-2.0 | ✅ | 타일 경로 필요 |
| **NVILA-8B-HD-Video** | SigLIP-so400m | Qwen2 | 프레임별 | AutoGaze 규약 | **cc-by-nc-4.0** | ❌(`nvila`) | **`to_autogaze_gazing_info` 이미 있음** |
| Mistral-Small-3.2-24B | Pixtral ViT-400M (24L/1024) | Mistral Small | 프레임별 | patch14 / merge2 / ≤1540 | apache-2.0 | ✅ | patch14 레시피 |
| GLM-4.6V | 자체 ViT | GLM-4.6 MoE (107B) | 프레임별 | patch14 / merge2 | MIT | ✅ | patch14 레시피 |
| Gemma 3 (4B/12B/27B) | SigLIP-400M (미검증 — config gated) | Gemma 3 | 프레임별 | 고정 토큰(`mm_tokens_per_image`) | gemma(gated) | ✅ | B1만(§4 주의) |
| Phi-4-multimodal | SigLIP-400M(?) | Phi-4-mini | 프레임별 | 동적 멀티크롭 + LoRA | MIT | ✅ | 타일 경로 |
| Kimi-VL-A3B | MoonViT | Moonlight 16B-A3B | 프레임별 | ? | MIT | ❌ | remote code |
| LLaVA-OneVision | SigLIP-so400m | Qwen2 | 프레임별 | patch14/384 → **27×27 홀수** | apache-2.0 | ✅ | **막힘**(기존 미해결) |
| **Gemma 4 12B Unified** | **없음** | Gemma 4 | — | 16px→Linear→3×3 pool | apache-2.0 | ❌ | `encoder-free-attach.md` |
| NEO 1.5 2B | `NEOVisionModel`(층수 미확인) | Qwen3 2B | ? | patch16 / `downsample_ratio=0.5` | apache-2.0 | 원격코드 | 동일 |

## 4. 다양화를 **통제된 비교**로 설계하기 (이 문서의 핵심)

무작정 여러 스택을 돌리면 "왜 달라졌는지"를 못 읽는다. 한 축만 움직이는 두 실험으로 쪼갠다.

### 실험 D1 — 인코더만 바꾼다 (LLM ≈ Qwen 고정)
| arm | 인코더 | LLM | 시간성 |
|---|---|---|---|
| 기존 | V-JEPA 2.1-L | Qwen | 완전 |
| A | Qwen3-VL-8B 자체 ViT | Qwen3-8B | 약한 |
| B | InternViT-300M (InternVL3.5-8B) | **Qwen3-8B** | 프레임별 |

**LLM 계열이 Qwen3로 거의 고정**되므로 차이는 인코더의 시간성으로 귀속된다. §2 가설 1·2를
직접 판정하는 실험이고, arm B가 특히 값지다 — InternVL3.5-8B의 LLM이 정확히 Qwen3-8B다.

### 실험 D2 — LLM만 바꾼다 (인코더 ≈ SigLIP 계열 고정)
| arm | 인코더 | LLM |
|---|---|---|
| A | SigLIP-so400m | Qwen2 (NVILA / LLaVA-OneVision) |
| B | SigLIP-400M | Gemma 3 |
| C | SigLIP-400M(?) | Phi-4-mini |

셀렉터 이득이 LLM에 의존하는지를 본다. 이득이 LLM에 둔감하면 "셀렉터는 인코더 계약의 문제"라는
강한 주장이 되고, 프리셋을 LLM별로 튜닝할 필요가 없어진다.

⚠️ **D2의 지표 주의**: Gemma 3처럼 **토큰 수가 고정된 스택**은 인코더 앞에서 잘라도 LLM 입력
개수가 그대로라 캡션 NLL이 거의 안 움직인다. 그런 arm은 **인코더 지연/FLOPs 절감 + 품질
비열화** 쌍으로 판정한다(`borissal_benchmark.py` 계열). 이건 결함이 아니라 이득의 종류가
다른 것이다(`encoder-free-attach.md` §6.5의 B1/B2).

## 5. V-JEPA를 유지하며 LLM만 바꾸는 방향 (비싼 쪽)

인코더 표현을 V-JEPA로 유지해야 할 이유가 있을 때만. 필요한 것:

- **커넥터(projector) 학습**: V-JEPA 특징 → 대상 LLM 임베딩 공간. LLaVA stage-1 정렬 방식.
  동결 인코더 + 동결 LLM + 학습 가능한 projector면 비용이 가장 낮다.
- 주의: 셀렉터를 **평가자와 함께 학습시키지 말 것** (L2X "selection as communication" 축퇴 —
  design.md 이론 노트의 REAL-X 논거와 동일). projector는 **랜덤 마스크**로 학습하고 셀렉터는
  건드리지 않는다.
- 이 레포에 이미 있는 것: `vjepa2_sparse.sparse_encoder_forward`(conv3d tubelet 임베딩
  **직후·트랜스포머 직전**에 `keep_index` gather = 인코더 앞 선택), `adapters.to_vjepa2`,
  `vjepa21_hub`(2.1-L/B torch.hub 로더). **빠진 건 커넥터뿐이다.**

## 6. 어태치 비용 (코드 관점)

인코더 앞 수술에 공통으로 필요한 3가지(`encoder-free-attach.md` §6.5 A축):
(a) 패치 임베딩이 행 단위인가(conv면 `kernel == stride`), (b) 위치 신호를 전체 그리드로 계산 후
인덱싱 가능한가, (c) attention 세그먼트/풀링 단위를 부분집합으로 재구성 가능한가.

| 대상 | 비용 | 메모 |
|---|---|---|
| Qwen3-VL / Qwen3.5 | **완료** | `attach_qwen3vl.py`, keep-all == vanilla forward 비트 일치 |
| Mistral-Small-3.2 / GLM-4.6V | 낮음 | patch14 + merge2 → `patch_size=14, scale=28의 배수`(실측·테스트 잠금). `_pruned_vision_forward` 패턴 이식 |
| InternVL3.5 | 중간 | 동적 타일링(1–12 타일)이 추가 축. `downsample_ratio=0.5` → 선택 단위 2×2(`score_coarsen=2`)로 동일 |
| NVILA-8B-HD-Video | **낮음** | `to_autogaze_gazing_info`가 이미 있다(uniform 할당 전용). 원조 구현과 직접 비교 가능 |
| V-JEPA + 다른 LLM | 높음 | 커넥터 학습(§5) |
| LLaVA-OneVision so400m-384 | 막힘 | 384 % 14 ≠ 0, 우회 시 27×27 홀수 |

## 7. 실행 순서

1. **지금 있는 것으로 D1 arm A를 먼저 끝낸다** — Qwen3-VL로 v0.3/v0.5/v0.6/v0.4/random ×
   ratio 스윕 (`eval_mllm_attach.py --prune-stage encoder`). **v0.4를 반드시 포함**한다:
   §2 가설 1의 첫 시험이고, 지금까지 v0.4는 V-JEPA 한 곳에서만 기각됐다.
2. **D1 arm B(InternVL3.5-8B)를 추가** — LLM이 Qwen3-8B로 고정된 프레임별 인코더. 시간성
   가설을 판정하는 결정적 arm.
3. **NVILA-8B-HD-Video 트랙** — 원조 AutoGaze 통합과의 직접 비교. 어댑터가 이미 있어 싸다.
4. **D2(LLM 축)** — 1–3에서 프리셋 순위가 스택마다 뒤집히는 것이 확인되면 그때 진행. 뒤집히지
   않으면 D2의 정보량은 낮다.
5. Gemma 4 unified(encoder-free)는 별도 트랙(`encoder-free-attach.md`).

**판정 규율(불변)**: 프리셋 순위가 스택마다 다르면 그것은 실패가 아니라 **결과**다 —
"프리셋을 다운스트림별로 고른다"는 lineage 방식이 이미 v0.3/v0.4/v0.5에 예고돼 있다. 스택별
승자를 표로 기록하고, 단일 승자를 억지로 만들지 않는다.
