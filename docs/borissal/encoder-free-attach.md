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

## 6. 대안 후보

| 모델 | encoder-free? | 규모 | 비고 |
|---|---|---|---|
| **gemma-4-12B (unified)** | **예** (ViT 0층) | 12B, bf16 24GB | 본 설계의 1순위. transformers 업그레이드 필요 |
| `gemma-4-E2B-it` | 아니오 (150M ViT) | 2.3B eff / 5.1B w-emb, bf16 ~10GB | patch16 + 토큰 예산 {70,140,280,560,1120} → 어태치는 가능. 16GB에선 빡빡 |
| `gemma-4-26B-A4B` / 31B | 아니오 (550M ViT) | MoE / dense | CUDA 전용 |
| `Mono-InternVL-2B` | 사실상 (LLM 내부 visual expert MoE + 직접 patchify) | 1.8B active | **이미지 전용**, remote code. 비디오는 프레임별로만 |
| `Qwen3.5-Omni` | 아니오 | 30B MoE (A3B) | 맥 경로 아님 |

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
