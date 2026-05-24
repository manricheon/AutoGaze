# AutoGaze Reporting / 결과 해석 가이드

이 문서는 single inference, HLVid benchmark, plugin benchmark 결과를 Markdown/chart/trend report로 정리하고 해석하는 방법을 다룹니다.

## 리포트 흐름

```text
runner output JSON/JSONL
  -> summary JSON
  -> gain report JSON/CSV
  -> markdown_report
       Markdown + SVG charts
  -> aggregate_reports
       여러 실험 폴더 trend CSV/Markdown/SVG
```

## 단일 결과 Markdown 변환

```bash
.venv/bin/python -m repro.markdown_report \
  --input-json outputs/autogaze_repro/hlvid_limit3_128f_720/hlvid_autogaze_gain_report.json \
  --output-md outputs/autogaze_repro/hlvid_limit3_128f_720/hlvid_autogaze_gain_report.md
```

기본적으로 chart SVG가 함께 생성됩니다. chart 없이 표만 원하면 `--no-charts`를 붙입니다.

HLVid gain report를 입력하면 Markdown은 기본 비교축을 `keep_all -> single_scale_dense -> autogaze` 순서로 표시합니다. `single_scale_dense`는 raw JSON field 이름이고 chart 라벨은 잘리지 않도록 `single-scale`로 짧게 표시됩니다. 해당 모드를 `--skip-single-scale-dense`로 제외한 경우에는 score/status에는 skipped 또는 missing으로 남고, latency/token 비율은 계산 가능한 축만 채웁니다.

## 여러 실험 trend report

```bash
.venv/bin/python -m repro.aggregate_reports \
  --input-root outputs/autogaze_repro \
  --output-dir outputs/autogaze_repro/trend_report
```

생성물:

```text
aggregate_rows.csv
aggregate_summary.json
aggregate_report.md
assets/latency_by_config.svg
assets/token_reduction_by_config.svg
assets/memory_peak_by_config.svg
assets/status_by_config.svg
```

Plugin HLVid report는 별도 chart도 만듭니다.

```text
plugin_hlvid_report_assets/pairwise_latency_speedup.svg
plugin_hlvid_report_assets/pairwise_token_reduction.svg
```

Qwen sparse처럼 실제 pre-ViT sparse를 목표로 하는 mode는 이 pairwise chart를 먼저 봅니다. VILA/LongVILA sidecar mode는 selector metric만 붙인 상태일 수 있으므로 `integration_level`, `execution_claim`, `actual_pruning_applied`, `vit_latency_reduction_claim`, `mllm_context_reduction_claim`을 같이 확인해야 합니다.

## 실험 폴더 이름 추천

폴더 이름에 비교 축을 넣으면 aggregate report를 읽기 쉽습니다.

```text
hlvid_limit3_128f_t64_tile8_720_autogaze
hlvid_limit3_128f_t64_tile8_720_keepall
hlvid_limit3_256f_t128_tile16_1080_autogaze
plugin_qwen_32f_t8_448_sparse
```

## 핵심 비교 표

리더에게 보여줄 첫 표는 Markdown의 `Key Metrics` 섹션입니다. 이 섹션은 raw field를 그대로 펼치지 않고 두 단계로만 봅니다.

```text
Key Metrics
  Mode Snapshot      # 세 모드의 원값을 나란히 표시. ratio/speedup 없음
  Pairwise Gains     # AutoGaze vs keep-all, AutoGaze vs single-scale만 gain 계산
```

`Mode Snapshot`은 세 모드를 나란히 보여주기 위한 표입니다. 여기에는 reduction ratio나 speedup을 넣지 않습니다. ratio는 분모가 다르면 해석이 바로 섞이기 때문입니다.

| 모드 | status | accuracy | total ms | Decode/read ms | Prep rest ms | Model-side ms | AutoGaze ms | ViT ms | LLM forward ms | peak GiB | encoder tokens | LLM visual token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  |  |  | n/a |  |  |  |  |  |
| single-scale dense |  |  |  |  |  |  | n/a |  |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |  |  |  |  |  |
| paper baseline |  |  |  |  |  |  | n/a |  |  |  | n/a |  |

AutoGaze off/keep-all이 OOM이면 latency는 비어 있을 수 있습니다. 그래도 full patch/LLM visual token 예상치와 failure stage를 남기는 것이 중요합니다.
`single-scale dense`는 기본 HLVid wrapper에서 함께 도는 ablation입니다. `keep-all`이 NVILA-HD multiscale keep-all이라면, single-scale dense는 보통 392px scale 하나만 keep-all로 통과시킨 reference라서 `single_scale_dense_comparison`에서 AutoGaze sparse 결과와 별도로 비교합니다. 제외하려면 `--skip-single-scale-dense`를 사용합니다.

`Pairwise Gains`는 딱 두 줄만 봅니다.

| Pair | 해석 |
| --- | --- |
| `AutoGaze vs keep-all` | HD multiscale keep-all을 분모로 둔 실제 AutoGaze 적용 이득 |
| `AutoGaze vs single-scale` | 392px single-scale dense를 분모로 둔 보수적/참고 비교 |

Markdown의 `Benchmark Score`, `Processing Budget Summary`, latency chart, aggregate rows는 세 모드를 같은 순서로 맞춰 보여줍니다. 그러나 speedup/reduction ratio는 `Pairwise Gains`에서만 리더용으로 해석하세요. raw field 이름과 alias는 `Raw Metric Appendix`로 내려갑니다.

정답 비교는 두 층으로 봅니다. 기존 `counts`와 `paired_rates`는 호환성을 위해 `keep_all vs autogaze` 기준을 유지합니다. 세 모드 비교는 `correctness_comparison.pairwise`에 별도로 들어가며 `keep_all_vs_single_scale_dense`, `single_scale_dense_vs_autogaze`, `keep_all_vs_autogaze`를 각각 보여줍니다. Markdown에는 `Pairwise Correctness Summary`와 `Pairwise Correctness Samples`로 표시됩니다.

## Latency 해석

상위 구조는 다음을 기준으로 봅니다.

```text
total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
```

| 항목 | 해석 |
| --- | --- |
| `video_preprocess_without_autogaze_ms` | decode/sample/resize/tile/tensorize. AutoGaze 제외 |
| `video_decode_read_ms` | 비디오 keyframe scan, seek, decode, frame->PIL 등 읽기/디코드 공통 비용 |
| `preprocess_rest_without_decode_autogaze_ms` | `video_preprocess_without_autogaze_ms - video_decode_read_ms`. decode와 AutoGaze를 뺀 resize/tile/thumbnail/tensorize/packing 비용 |
| `selector_input_build_ms` | `autogaze_total_ms - autogaze_model_forward_ms` residual. AutoGaze 입력 준비/선택 정리 비용 |
| `autogaze_model_forward_ms` | AutoGaze 모델 forward-only. 내부 batch가 여러 번 돌아도 합산 |
| `autogaze_total_ms` | AutoGaze 입력 준비, 모델 실행, patch 선택 정리 전체 |
| `vision_input_build_ms` | `vision_encoder_ms - siglip_vision_ms - mm_projector_ms` residual. vision feature packing/reorder 비용 |
| `siglip_vision_ms` | SigLIP/ViT vision tower forward-only |
| `vision_encoder_ms` | SigLIP/ViT + feature packing/reorder + projector hook을 포함한 vision path wrapper |
| `generate_ms` | preprocessing 이후 전체 `model.generate` 호출. vision path, projector, LLM forward/decode loop와 generation overhead 포함 |
| `llm_generation_ms` | `llm_forward_ms + generation_rest_ms`. vision을 제외한 LLM generation 실전 비용 |
| `llm_forward_ms` | `generate_ms` 안에서 누적된 LLM forward child timer. 전체 generate와 같지 않음 |
| `generation_rest_ms` | `generate_ms - vision_encoder_ms - llm_forward_ms` 기반 residual. child timer 밖의 generation/decode orchestration 비용 |
| `ttft_ms` | 별도 1-token generation으로 잰 time-to-first-token. total에는 보통 별도 포함 |

summary에는 너무 세부적인 field보다 `Decode/read / Prep rest / Selector input / AutoGaze / Vision input / ViT / Projector / Generate total / LLM generation / LLM forward / Generate rest / total` 축을 먼저 보여주는 것이 좋습니다. `video_preprocess_ms`처럼 AutoGaze가 포함된 legacy inclusive field는 appendix에서만 확인하고, 메인 비교/차트에는 쓰지 않습니다.

## 표시명과 정렬

Markdown과 SVG chart는 raw JSON field 대신 짧은 표시명을 우선 사용합니다.

| 표시명 | 의미 |
| --- | --- |
| `Decode/read` | 비디오 읽기, seek, decode, frame 변환. 같은 비디오/sampling이면 on/off 공통 비용에 가까움 |
| `Prep rest` | decode/read와 AutoGaze를 제외한 resize/tile/thumbnail/tensorize/packing |
| `Selector input` | AutoGaze 전체에서 모델 forward를 뺀 입력 준비/선택 정리 residual |
| `AutoGaze` | selector 전체 stage. `Selector input + AutoGaze forward` 관계로 해석 |
| `Vision input` | vision wrapper에서 SigLIP/projector를 뺀 feature packing/reorder residual |
| `ViT` | SigLIP/ViT vision encoder |
| `Generate total` | 전체 `model.generate` 부모 stage |
| `LLM generation` | `LLM forward + Generate rest`. vision path를 제외한 LLM generation 부담 |
| `LLM forward` | generate 안에서 누적한 LLM forward child stage |
| `Generate rest` | generate 부모 stage에서 측정된 vision/LLM child를 뺀 residual |
| `Full patch` | selector 적용 전 patch/token 분모 |
| `Selected patch` | AutoGaze/token selector 이후 남은 patch/token |
| `Patch x` | `Full patch / Selected patch` |
| `Single-scale dense patch` | 392px single-scale dense reference. multiscale keep-all과 분모가 다르므로 별도 축으로 표기 |

Aggregate report의 기본 정렬은 `comparison`입니다. 같은 config 안에서 keep-all/off baseline을 먼저, single-scale dense ablation을 다음, AutoGaze/token selector를 그 다음, probe/sidecar/OOM을 뒤로 배치합니다. 필요하면 `--sort latency|token-reduction|memory|accuracy|status`로 바꿀 수 있습니다.

## Token/Patch 해석

| 기준 | 의미 |
| --- | --- |
| full/off 예상 patch | AutoGaze 없이 encoder에 들어갈 patch budget |
| multiscale candidate patch | AutoGaze가 scale별로 고려한 후보 patch 총량 |
| AutoGaze selected patch | selector가 실제 남긴 patch |
| encoder input token | ViT/SigLIP에 실제 들어간 token |
| LLM visual token | projector/token shuffle 이후 LLM context에 들어간 visual token |

모델별로 patch와 LLM visual token은 1:1이 아닐 수 있습니다. 그래서 report는 “encoder 이전 patch 감소”와 “LLM context 감소”를 분리해서 보여야 합니다.

Markdown의 `Frame, Patch, And Tokenization Info`는 이제 두 표로 단순화됩니다.

| 표 | 목적 |
| --- | --- |
| `Input Shape` | frame 수, thumbnail 수, processor 입력 해상도, multiscale patch space 같은 입력 조건 |
| `Token Boundaries By Mode` | 각 모드가 encoder/LLM 경계에서 실제로 몇 token을 쓰는지 |

`AutoGaze Token And Patch Flow`는 중복 행을 줄이고 아래 기준만 표시합니다.

| Stage | 비교 의미 |
| --- | --- |
| `HD multiscale keep-all -> AutoGaze` | 논문 HD-style multiscale keep-all budget 대비 selected patch |
| `Single-scale dense -> AutoGaze` | 392px single-scale dense reference 대비 selected patch |
| `Main tile patch -> AutoGaze` | thumbnail 제외 main video tile budget |
| `Thumbnail patch -> AutoGaze` | thumbnail이 켜진 경우 overview token budget |
| `LLM visual -> AutoGaze` | TokenShuffle/projector 이후 LLM visual token budget |

같은 숫자가 `Full patch`, `ViT before`, `encoder before` 식으로 반복되어 보이면 raw appendix를 보고 있는 것입니다. 메인 해석은 `Key Metrics`, `Frame/Patch`, `AutoGaze Token And Patch Flow`만 보면 됩니다.

## Memory 해석

| 메모리 | 의미 |
| --- | --- |
| processor peak | decode/tile/AutoGaze processor 구간 peak |
| AutoGaze peak | selector 모델 실행 peak |
| vision peak | ViT/SigLIP encoder 구간 peak |
| LLM peak | prefill/generate 구간 peak |
| overall peak | 전체 실행 중 최대 CUDA allocation |

H100 80GB에서 OOM을 피하려면 전체 peak뿐 아니라 `failure.stage`를 같이 봐야 합니다. SDPA attention/Qwen2 계열 stack trace라면 대개 LLM prefill/context 구간입니다.

## OOM report 규칙

OOM이 나도 가능한 경우 다음 값을 남깁니다.

| 필드 | 이유 |
| --- | --- |
| `failure.kind=oom` | aggregate status chart에 반영 |
| `failure.stage` | processor, AutoGaze, ViT, LLM 중 어디인지 확인 |
| `failure.message` | CUDA stack trace 요약 |
| video/frame/resize/tile config | 어떤 입력 조건에서 터졌는지 재현 |
| full/off expected token | 실행 전 예상 부담 |
| selected token | AutoGaze가 끝난 뒤 터진 경우 reduction 근거 |

## 추천 chart

| chart | 목적 |
| --- | --- |
| stacked latency bar | Decode/read/Prep rest/Selector input/AutoGaze/Vision input/ViT/LLM 중 어디가 병목인지 표시. 같은 stage는 항상 같은 색상 |
| token reduction bar | full patch 대비 selected patch, LLM visual token 감소 표시 |
| memory peak bar | config별 peak GiB 비교 |
| status chart | success/OOM/parse_failed/skipped 분포 |
| accuracy table | keep-all, AutoGaze, paper baseline 점수 비교 |
| pairwise correctness table | keep-all/single-scale/AutoGaze 중 어떤 모드만 정답을 맞췄는지 paired sample 기준으로 비교 |
| plugin pairwise table | Qwen full/chunked/sparse, VILA/LLaVA off/on pair의 latency/token/memory/accuracy delta 비교 |
| plugin integration claim table | `pre_encoder_sparse`, `post_encoder_token_prune`, sidecar/probe 상태를 분리해 “실제 pruning 적용” 여부 표시 |

## 결과 코멘트 템플릿

```text
이번 config는 128f, thumbnail 64f, max_tiles 8, resize longest 720 기준이다.
AutoGaze는 full/off 예상 patch 대비 selected patch를 __x 줄였고,
LLM visual token은 __에서 __로 줄었다.
Latency는 total __ms -> __ms로 변했고,
Selector input/AutoGaze/ViT/LLM은 각각 __/__ /__/__ ms였다.
병목은 __ 단계에 남아 있다.
Accuracy는 keep-all __, AutoGaze __이며,
실패는 OOM __건, parse_failed __건이다.
```
