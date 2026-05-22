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

## 실험 폴더 이름 추천

폴더 이름에 비교 축을 넣으면 aggregate report를 읽기 쉽습니다.

```text
hlvid_limit3_128f_t64_tile8_720_autogaze
hlvid_limit3_128f_t64_tile8_720_keepall
hlvid_limit3_256f_t128_tile16_1080_autogaze
plugin_qwen_32f_t8_448_sparse
```

## 핵심 비교 표

리더에게 보여줄 첫 표는 아래 구조가 좋습니다.

| 모드 | status | accuracy | total ms | preprocess ms | AutoGaze ms | ViT ms | LLM ms | peak GiB | full patch | selected patch | LLM visual token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  | n/a |  |  |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |  |  |  |  |
| paper baseline |  |  |  |  | n/a |  |  |  |  | n/a |  |

AutoGaze off/keep-all이 OOM이면 latency는 비어 있을 수 있습니다. 그래도 full patch/LLM visual token 예상치와 failure stage를 남기는 것이 중요합니다.

## Latency 해석

상위 구조는 다음을 기준으로 봅니다.

```text
total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
```

| 항목 | 해석 |
| --- | --- |
| `video_preprocess_without_autogaze_ms` | decode/sample/resize/tile/tensorize. AutoGaze 제외 |
| `autogaze_total_ms` | AutoGaze 입력 준비, 모델 실행, patch 선택 정리 |
| `vision_encoder_ms` | SigLIP/ViT encoder 구간 |
| `generate_ms` | visual embedding, projector, LLM prefill/decode를 포함한 generation 호출 |
| `ttft_ms` | 별도 1-token generation으로 잰 time-to-first-token. total에는 보통 별도 포함 |

summary에는 너무 세부적인 field보다 `preprocess / AutoGaze / ViT / LLM / total` 5개 축을 먼저 보여주는 것이 좋습니다.

## Token/Patch 해석

| 기준 | 의미 |
| --- | --- |
| full/off 예상 patch | AutoGaze 없이 encoder에 들어갈 patch budget |
| multiscale candidate patch | AutoGaze가 scale별로 고려한 후보 patch 총량 |
| AutoGaze selected patch | selector가 실제 남긴 patch |
| encoder input token | ViT/SigLIP에 실제 들어간 token |
| LLM visual token | projector/token shuffle 이후 LLM context에 들어간 visual token |

모델별로 patch와 LLM visual token은 1:1이 아닐 수 있습니다. 그래서 report는 “encoder 이전 patch 감소”와 “LLM context 감소”를 분리해서 보여야 합니다.

Plugin benchmark에서는 같은 의미를 아래 flatten field로도 남깁니다.

| field | 의미 |
| --- | --- |
| `raw_patch_tokens` | selector/processor 기준 full/off patch token 분모 |
| `selected_patch_tokens` | AutoGaze 또는 sparse plan 이후 남은 patch token |
| `patch_token_reduction_ratio` | `raw_patch_tokens / selected_patch_tokens` |
| `encoder_input_tokens` | ViT/vision encoder에 실제 들어가도록 mapping된 token |
| `llm_visual_tokens` | MLLM context에 들어간 visual token |
| `selector_ms`, `vision_encoder_ms`, `generate_ms` | selector, ViT, MLLM 주요 latency |
| `failure_stage` | OOM/실패가 난 pipeline stage |
| `autogaze_attachment_mode` | sidecar mode인지 실제 prune/sparse mode인지 확인 |
| `visual_pruning_applied` | AutoGaze 결과가 모델 내부 visual token pruning에 적용됐는지 |
| `vision_encoder_latency_reduced` | ViT/vision encoder 계산량 감소를 주장할 수 있는지 |
| `mllm_context_reduced` | LLM context/prefill token 감소를 주장할 수 있는지 |

Plugin mode에서는 status를 먼저 분리해서 봅니다.

| status | 해석 |
| --- | --- |
| `executed` | generation이 실행됨. Qwen sparse/LLaVA prune mode는 token 감소 field를 확인 |
| `executed_dense_with_autogaze_sidecar` | AutoGaze selector는 sidecar로 실행/기록됐지만 dense model generation은 그대로. compute gain 주장의 근거로 쓰면 안 됨 |
| `probe_required` / `probe_collected` | 모델별 visual packing mapping을 확인하는 단계 |
| `failed_missing_dependency` | 외부 CLI 또는 package/model dependency 누락 |
| `oom` | memory failure. `failure_stage`와 stack trace 요약 확인 |

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
| stacked latency bar | preprocess/AutoGaze/ViT/LLM 중 어디가 병목인지 표시 |
| token reduction bar | full patch 대비 selected patch, LLM visual token 감소 표시 |
| memory peak bar | config별 peak GiB 비교 |
| status chart | success/OOM/parse_failed/skipped 분포 |
| accuracy table | keep-all, AutoGaze, paper baseline 점수 비교 |

## 결과 코멘트 템플릿

```text
이번 config는 128f, thumbnail 64f, max_tiles 8, resize longest 720 기준이다.
AutoGaze는 full/off 예상 patch 대비 selected patch를 __x 줄였고,
LLM visual token은 __에서 __로 줄었다.
Latency는 total __ms -> __ms로 변했고,
병목은 __ 단계에 남아 있다.
Accuracy는 keep-all __, AutoGaze __이며,
실패는 OOM __건, parse_failed __건이다.
```
