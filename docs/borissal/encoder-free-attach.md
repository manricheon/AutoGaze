# Encoder-free MLLM 어태치 설계서 (2026-07-26)

Borissal을 **인코더 없는 멀티모달 LLM**에 직접 붙이는 설계. 구현은 하지 않았고
(이 맥에서 검증 불가 — §5), CUDA 머신에서 바로 집행할 수 있도록 확인된 사실만 적는다.
동반 문서: `design.md`(측정·판정 로그), 실행 하네스는 `scripts/eval_mllm_attach.py`.

## 1. 왜 encoder-free가 selector에게 최적인가

지금까지의 모든 어태치 경로(V-JEPA, SigLIP2, VideoMAE, Qwen3-VL)에는 **인코더 계약**이
있었다: 인코더가 기대하는 토큰 수·위치·순서를 맞춰야 하고, 무엇보다 **살아남은 토큰이
ViT self-attention을 통해 버려진 토큰을 이미 봤다**. 그래서 "토큰을 줄였는데 성능이
유지된다"가 정보가 실제로 불필요했다는 증거가 되지 못한다 (`attach_qwen3vl`의
`prune_stage="llm"`이 정확히 이 한계를 갖는다).

encoder-free 모델에는 그 계약이 없다. 패치가 선형사영으로 곧장 LLM 임베딩 공간에
들어가므로 **선택 = LLM에 들어가는 패치 행을 고르는 것**이고, 프루닝은 순수한
정보 제거다. AutoGaze의 원래 주장("다운스트림이 정보 손실 없이 더 적은 패치를 처리한다")을
가장 오염 없이 시험할 수 있는 다운스트림이다.

## 2. Gemma 4 12B "Unified" — 실측 구조

`google/gemma-4-12B`의 config.json과 transformers 5.5.0 소스에서 확인:

| 항목 | 값 |
|---|---|
| `model_type` / architecture | `gemma4_unified` / `Gemma4UnifiedForConditionalGeneration` |
| vision `model_type` | **`gemma4_unified_vision`** |
| `patch_size` | 16 |
| `model_patch_size` | **48** (= 16 × pooling 3) |
| `pooling_kernel_size` | **3** |
| `num_soft_tokens` | **280** (보고서 최대 1120) |
| `mm_embed_dim` / `output_proj_dims` | 3840 |
| `mm_posemb_size` | 1120 |
| **`num_hidden_layers`** | **없음** |
| **`hidden_size`** | **없음** |
| text | 3840 hidden / 48층 / 256K ctx / bf16 |

**핵심: vision_config에 `num_hidden_layers`도 `hidden_size`도 없다 — 비전 트랜스포머가
아예 존재하지 않는다.** 경로는

```
48×48 영역의 16px 패치들 → Linear(3·16², 3840, bias=False)
  → (x,y) 좌표 임베딩 가산 (one-hot @ position_embedding_table, 두 축 합)
  → 3×3 average pooling → 280 soft tokens → LLM
```

따라서 **LLM 이전에 패치 간 attention이 0회**다. 정보 누출이 구조적으로 불가능하고,
`prune_stage="llm"` / `"encoder"` 구분 자체가 사라진다.

주의: transformers 5.5.0의 `Gemma4VisionModel`(16층·hidden 768·~150M)과
`Gemma4VisionEncoder`는 **encoder-based 변형(E2B/E4B/31B/26B-A4B)의 것**이다.
unified 12B와 혼동하지 말 것.

### 부분집합 패치 입력은 이미 1급 시민

`Gemma4VisionPatchEmbedder.forward(pixel_values, pixel_position_ids, padding_positions)`:
- `pixel_values`를 `2*(x-0.5)`로 스케일한 뒤 `input_proj` — **패치별 독립 연산**;
- 좌표는 `pixel_position_ids`(x,y)로 명시 전달, `padding_positions=True`인 패치는
  좌표 임베딩이 0으로 지워진다.

즉 **선택된 패치 + 그 좌표만 넘기면 끝**이다. Qwen 경로에 필요했던 mrope 열 삭제 수술이
여기서는 필요 없다.

## 3. Borissal 매핑

`pooling_kernel_size=3`은 Qwen의 `spatial_merge_size=2`와 **완전히 같은 역할**이다:
soft token 하나 = 16px 패치 3×3 블록. 따라서

> **`score_coarsen=3`으로 선택하면 soft token이 온전하게 유지된다.**

v0.5/v0.6이 `score_coarsen=2`로 Qwen의 2×2에 맞춘 것과 같은 장치를 값만 바꿔 쓰는 것이고,
`score_coarsen`은 이미 임의 정수를 받는 config 노브다(`H_grid % c == 0`만 요구).
대안 2가지도 기록:

- (a) **soft-token 단위 선택**: 3×3 풀링 뒤 280개 중 top-K. 가장 단순하지만 selector가
  풀링된 표현 위에서 점수를 매겨야 해 borissal의 픽셀-레벨 신호가 희석된다.
- (b) **borissal을 `patch_size=48`로 직접 실행**: 그리드가 곧 soft-token 그리드가 되어
  매핑이 항등. 단 모션/그라디언트 신호가 48px 해상도로 뭉개진다(v0.5의 "신호는 그리드
  해상도에서 계산" 교훈과 상충).

권장은 **`score_coarsen=3` + 16px 신호**다: 신호는 미세하게 계산하고 선택 단위만 거칠게.

### 반드시 지켜야 할 3가지 (소스에서 확인한 함정)

1. **`k`는 길이 비율에서 유도된다.** `Gemma4VisionPooler._avg_pool_by_positions`는
   `k = int((input_seq_len // output_length) ** 0.5)`로 k를 역산하고
   `k² * output_length == input_seq_len`이 아니면 **raise**한다. 패치를 P개만 남기면
   `num_soft_tokens`도 **P/9로 함께 줄여야** k=3이 유지된다. 안 줄이면 k가 바뀌어 풀링
   기하가 조용히 달라진다.
2. **`max_x` 선형화 위험.** kernel 인덱스는
   `floor(x/k) + (max_x // k) * floor(y/k)`이고 `max_x`는 **살아남은 패치들의 x 최대값+1**이다.
   프루닝이 가장 오른쪽 열을 통째로 없애면 `max_x`가 줄어 인덱스가 충돌한다.
   → 완화책: 선택되지 않은 패치를 **물리적으로 제거하지 말고 `padding_positions=True`로
   표시**해 시퀀스 길이와 좌표 범위를 보존하는 변형을 먼저 시도한다(풀러가
   `masked_fill(padding, 0)`으로 기여를 0으로 만든다). 토큰 절감은 LLM 단(280 soft token)에서
   얻고, 패치 단 절감은 max_x 보존을 확인한 뒤에 켠다.
3. **부분 3×3 블록은 감쇠된 평균을 만든다.** 풀링 제수는 항상 `k²=9`로 고정이고 padding
   패치는 0을 기여하므로, 9개 중 일부만 남은 soft token은 정규화된 평균이 아니라
   **크기가 줄어든 벡터**가 된다. 이것이 부분 블록을 금지하는 이유다
   (`to_qwen3vl_video_tokens`의 `partial_blocks="strict"`와 같은 규율을 적용할 것).

## 4. 어댑터 스케치 (구현 시)

`adapters.to_gemma4_unified_patches(selection, pooling_kernel_size=3, partial_blocks="strict")`
→ `to_qwen3vl_video_tokens`와 대칭:

- `keep_soft_token_index`: (B, K) — 살아남은 soft token 인덱스
- `keep_patch_index` + `pixel_position_ids` (B, K·9, 2) — 남은 패치의 (x,y)
- `padding_positions` (B, N) — §3.2의 "표시만 하는" 변형용
- `num_soft_tokens` = K — §3.1의 k 보존용으로 프로세서에 되먹임
- `n_partial_blocks` — 0이어야 함

**시간축**: Gemma 4는 비디오를 프레임 시퀀스로 다룬다(보고서: 1 fps 기준 최대 60초).
borissal은 tubelet(2프레임) 단위로 결정하므로 `to_onevision_frame_indices`와 같은
프레임 복제 규칙을 쓴다(튜블렛 마스크를 두 프레임에 복제). Qwen처럼 시간축이
`temporal_patch_size`로 접히지 않으므로 **토큰 예산이 프레임 수에 선형**이라는 점만 주의.

## 5. 왜 지금 이 맥에서 못 하나 (2가지 블로커)

1. **메모리**: 12B bf16 ≈ 24GB > 16GB(Apple M1). "16GB 랩탑에서 돈다"는 보도는
   Q4/llama.cpp 경로이고, 그 경로에서는 패치 단위 주입이 불가능하다(우리가 필요한 건
   torch 레벨 텐서 접근).
2. **transformers 지원**: 5.5.0에 등록된 것은 `gemma4`, `gemma4_text`, `gemma4_vision`,
   `gemma4_audio` — **`gemma4_unified`는 없다**(`Gemma4Unified*` 클래스 0개). 즉
   encoder-free 12B는 **transformers 업그레이드가 선행**돼야 한다.
   `pyproject.toml`이 `transformers>=5.5,<6`이므로 5.x 상향은 핀 변경 없이 가능하지만,
   **업그레이드 후 기존 경로 회귀 확인이 필요하다**(Qwen3-VL 어태치 테스트 13개 +
   export 14개 + 전체 스위트가 그 회귀 게이트다).

## 6. 후보 모델 (오픈소스/오픈웨이트, 2026-07 기준)

파라미터 수·라이선스·아키텍처 문자열은 **HF Hub API에서 직접 조회한 값**이고, transformers
지원 여부는 이 레포의 **설치된 5.5.0에서 실제 확인**했다. "?"는 확인 못 한 항목 — 추측으로
채우지 않는다.

bf16 메모리 ≈ params × 2 GB. 이 맥(16GB)에서 torch로 돌 수 있는 것은 대략 **≤ 5B**.

### Tier A — 진짜 encoder-free (연속 패치 → LLM). 1순위 어태치 대상

패치가 선형사영으로 곧장 들어가므로 **프루닝 = 순수 정보 제거**. 단, "encoder-free"라도
패치 임베딩 뒤에 트랜스포머 층이 있으면(§Tier A 주석) 그 층에서 공간 혼합이 일어나 누출이
생긴다 — Gemma 4 unified만 층이 0개임을 확인했다.

| 모델 | HF repo | params | 라이선스 | arch | tf 5.5.0 | 비디오 | 어태치 메모 |
|---|---|---|---|---|---|---|---|
| **Gemma 4 12B Unified** | `google/gemma-4-12B` | **11.96B** | apache-2.0 | `gemma4_unified` | ❌ | 프레임열 | **ViT 0층 확인.** 16px→Linear→3×3 pool→280 soft tokens → `score_coarsen=3`. §2–3 |
| **NEO 1.5 2B** | `Paranioar/NEO1_5-2B-SFT` | ~2B (LLM: Qwen3, 40층/hidden 2048) | apache-2.0 | `neo_chat` (remote code) | 원격코드 | **multi-image & video**(레포 주장; config엔 video 필드 없음) | **이 맥에서 돌 만한 유일한 encoder-free 후보(bf16 ~4GB).** config 실측: `patch_size=16`, **`downsample_ratio=0.5` → 2×2 병합이므로 `score_coarsen=2`**(Qwen과 동일!), native dynamic resolution(`min/max_pixels`). ⚠️ `vision_config`에 `NEOVisionModel`이 **존재**하고 `num_hidden_layers` 필드가 없어 **층 수 미확인** — 층이 있으면 누출-free가 아니다 |
| NEO 1.5 9B | `Paranioar/NEO1_5-9B-SFT` | ~9B | apache-2.0 | `neo_chat` | 원격코드 | 동일 | CUDA. NEO 1.0(2B/9B, 2025-10, arXiv 2510.14979)도 공개 |
| EVEv2.0 | `BAAI/EVE-7B-HD-v2.0` | **14.16B** | apache-2.0 | `eve-qwen2` | ❌ | 이미지 | **이름은 7B지만 실측 14.2B** — LLM의 모든 linear/norm을 modality별로 분리해 두 배가 됨. 임의 종횡비 |
| SAIL-7B | `ByteDance-Seed/SAIL-7B` | ~7B | apache-2.0 | `mistral` | AutoModel 로드 가능 | 미언급 | raw pixel + 단일 트랜스포머. `vision_patch_size` 존재, 값 미확인 |
| VoRA-7B | `Hon-Wong/VoRA-7B-Instruct` | 7.62B | **라이선스 표기 없음** | `vora` (remote code) | 원격코드 | 이미지 | vision을 LoRA로 LLM에 내재화, 추론 시 병합. 라이선스 없음 = 사용 전 확인 필요 |
| Fuyu-8B | `adept/fuyu-8b` | 9.41B | **cc-by-nc-4.0(비상업)** | `fuyu` | **✅ 내장** | 이미지 | 원조. 실측 `patch_size=30`, `image_size=300` → 10×10=100 패치. transformers 내장이라 **배관 실험용으로는 가장 싸다** |
| Mono-InternVL-2B | `OpenGVLab/Mono-InternVL-2B` | 3.11B | MIT | `internvl_chat` + MoE | 원격코드 | 이미지(프레임별) | LLM 내부 visual expert MoE + 직접 patchify. Mono-InternVL-1.5는 arXiv 2507.12566 |
| HoVLE | `OpenGVLab/HoVLE` | 2.61B | MIT | `internvl_chat` | 원격코드 | 이미지 | "holistic embedding module"이 **트랜스포머 층을 포함** → 공간 혼합 있음, 누출-free 아님 |
| BREEN | ? | ? | ? | ? | ❌ | ? | learnable query 방식 → 패치 통과가 아니라 질의 압축. **선택 대상이 패치가 아니게 되므로 어태치 부적합** |
| NaViL | ? | ? | ? | ? | ❌ | ? | native MLLM 스케일링 연구(arXiv 2510.08565). 웨이트 공개 여부 미확인 |

### Tier B — 이산(VQ) 토크나이저 통합 모델. 어태치는 되지만 누출-free가 아니다

이미지를 **VQ 코드북 토큰**으로 바꾸는 계열. 토큰 시퀀스에서 행을 빼는 것 자체는 쉽지만,
**VQ 토크나이저가 이미지 전체를 보는 CNN**이라 인접 토큰이 receptive field를 공유한다 →
Qwen `prune_stage="llm"`과 같은 누출이 구조적으로 존재하고, 코드북 인덱스라 "패치 일부만
넣기"가 불가능하다. 선택 단위는 VQ 다운샘플 배수(패치 16px이 아님)를 따른다. 게다가 대부분
생성 지향이라 캡션 품질이 주 타깃이 아니다. **Tier A가 막힐 때의 대안으로만 기록.**

| 모델 | HF repo | params | 라이선스 | arch | tf 5.5.0 |
|---|---|---|---|---|---|
| Emu3-Chat | `BAAI/Emu3-Chat` | 8.49B | apache-2.0 | `Emu3` | ✅ (`emu3`) |
| Emu3.5 | `BAAI/Emu3.5` | **34.10B** | apache-2.0 | `Emu3` | ✅ |
| Chameleon-7B | `facebook/chameleon-7b` | 7.04B | other(**gated**) | `chameleon` | ✅ |
| BAGEL-7B-MoT | `ByteDance-Seed/BAGEL-7B-MoT` | 14.69B | apache-2.0 | `bagel` | ❌ (`bagel-mot` 라이브러리) |
| Show-o2-1.5B | `showlab/show-o2-1.5B` | ~1.5B | apache-2.0 | diffusers | ❌ | 
| Janus-Pro-7B | `deepseek-ai/Janus-Pro-7B` | ~7B | MIT | `multi_modality` | `janus` 있음 |

⚠️ **Janus 계열은 encoder-free가 아니다**: 로컬 `JanusConfig` 확인 결과 이해(understanding)
경로에 `vision_config`(24층, patch16, 384)가 붙어 있다 — VQ는 생성 쪽 전용. 즉 "통합 모델"이
곧 "encoder-free"는 아니므로 이 표에 두되 Tier A로 올리지 않는다.

### Tier C — 최소 인코더(작은 ViT). 오늘 당장 붙는 실용 후보

| 모델 | HF repo | params | 라이선스 | arch | tf 5.5.0 | 메모 |
|---|---|---|---|---|---|---|
| **Qwen3-VL-2B** | `Qwen/Qwen3-VL-2B-Instruct` | **2.13B** | apache-2.0 | `qwen3_vl` | ✅ | **이미 구현·검증 완료**(`attach_qwen3vl.py`). 로컬 캐시 4.0GB |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | **2.27B** | apache-2.0 | `qwen3_5` | ✅ | 같은 기하, deepstack 없음. 같은 코드로 동작 |
| gemma-4-E2B-it | `google/gemma-4-E2B-it` | **5.12B** | apache-2.0 | `gemma4` | ✅ | 150M ViT, patch16, 토큰 예산 {70,140,280,560,1120}. bf16 ~10GB로 16GB엔 빡빡 |
| gemma-4-26B-A4B / 31B | `google/gemma-4-26B-A4B` | **26.54B** MoE / 31B | apache-2.0 | `gemma4` | ✅ | 550M ViT. CUDA 전용 |
| SmolVLM-256M | `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | apache-2.0 | `smolvlm` | ✅ | 로컬 캐시 494MB. 아주 싼 배관 실험용 |
| Qwen3.5-Omni | — | 30B MoE (A3B) | — | `qwen3_omni_moe` | ✅ | 맥 경로 아님 |

### 정리 — 실제로 무엇을 할 것인가

1. **CUDA 1순위: Gemma 4 12B Unified.** ViT 0층이 확인된 유일한 모델이라 "버린 패치에
   들어있던 정보량"을 오염 없이 재는 유일한 후보. transformers 상향이 선행.
2. **맥에서 시도해볼 유일한 encoder-free: `NEO1_5-2B-SFT`** (bf16 ~4GB, apache-2.0).
   좋은 소식: `patch_size=16` + `downsample_ratio=0.5`라 **선택 단위가 Qwen과 똑같은 2×2**
   → `to_qwen3vl_video_tokens`의 인덱스 산술을 거의 그대로 재사용할 수 있다.
   확인 필요: (a) `NEOVisionModel`의 층 수(config에 없음 — 층이 있으면 누출-free가 아니라
   Qwen의 `prune_stage="llm"`과 같은 등급이 된다), (b) 비디오 프레임을 실제로 어떻게 넣는지
   (config에 video 필드가 없어 멀티이미지로 처리할 가능성), (c) remote code 신뢰 확인.
3. **배관 검증용 최저비용: Fuyu-8B**(transformers 내장, patch 30). 단 cc-by-nc라
   실험/연구용만.
4. **Tier B는 보류.** 누출-free가 아니고 선택 단위가 VQ 격자에 묶인다 — Qwen 경로로 이미
   같은 성질의 실험(`prune_stage="llm"`)을 할 수 있으므로 추가 가치가 낮다.

각 후보를 실제로 채택하기 전 **§7의 필수 게이트(keep-all == vanilla forward)** 를 먼저
통과시킬 것. Tier A 모델은 대부분 remote code이므로, 어댑터를 쓰기 전에 "패치 임베딩 뒤에
공간 혼합이 있는지"를 소스에서 직접 확인해야 한다(층 수 0인지, attention이 패치 간인지).

## 7. CUDA A/B 계획

`scripts/eval_mllm_attach.py`를 그대로 쓰되 gemma4 어댑터를 추가한 뒤:

- **매트릭스**: `{v0.3, v0.5, v0.6, v0.6-static, random, dense}` × ratio {0.15, 0.25, 0.5, 0.75}
  × `score_coarsen ∈ {2, 3}` (3이 gemma4 정렬, 2는 Qwen 정렬 — 교차 비교로 coarsen 자체의
  비용을 분리).
- **지표**: dense 캡션의 teacher-forced NLL(주), 생성 캡션(보조), 실제 soft-token 수.
  encoder-free에서는 누출이 없으므로 **이 NLL이 곧 "버린 패치에 들어있던 설명 관련 정보량"**이다.
- **교차검증**: 같은 클립·같은 ratio로 Qwen3-VL(`prune_stage=encoder`)과 gemma4를 함께
  돌린다. 두 계열이 같은 순위를 주면 selector의 이득이 모델 특이적이지 않다는 증거가 된다.
- **판정 규율**: 기존 그대로 — 프록시(SigLIP recall)와 어긋나면 **다운스트림이 심판**이다.
  특히 v0.6 all-on은 프록시상 최악(recall 0.2925 vs v0.5 0.3425)이면서 saliency-v3.1의
  다운스트림 증거만으로 채택된 상태이므로, 이 A/B가 그 베팅의 첫 독립 검증이 된다.
- **필수 게이트**: 어떤 새 어댑터든 **keep-all이 vanilla forward와 logit 일치**하는지 먼저
  확인한다(Qwen 경로에서 이 게이트가 배관 오류 전부를 한 번에 잡았다 — 실측 max|diff| 0.0).
