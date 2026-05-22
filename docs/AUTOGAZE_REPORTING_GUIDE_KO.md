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

| 모드 | status | accuracy | total ms | Decode/read ms | Prep rest ms | AutoGaze ms | ViT ms | LLM ms | peak GiB | full patch | selected patch | LLM visual token |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| keep-all |  |  |  |  |  | n/a |  |  |  |  |  |  |
| AutoGaze |  |  |  |  |  |  |  |  |  |  |  |  |
| paper baseline |  |  |  |  |  | n/a |  |  |  |  | n/a |  |

AutoGaze off/keep-all이 OOM이면 latency는 비어 있을 수 있습니다. 그래도 full patch/LLM visual token 예상치와 failure stage를 남기는 것이 중요합니다.

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
| `autogaze_total_ms` | AutoGaze 입력 준비, 모델 실행, patch 선택 정리 |
| `vision_encoder_ms` | SigLIP/ViT encoder 구간 |
| `generate_ms` | visual embedding, projector, LLM prefill/decode를 포함한 generation 호출 |
| `ttft_ms` | 별도 1-token generation으로 잰 time-to-first-token. total에는 보통 별도 포함 |

summary에는 너무 세부적인 field보다 `Decode/read / Prep rest / AutoGaze / ViT / LLM / total` 축을 먼저 보여주는 것이 좋습니다. `video_preprocess_ms`처럼 AutoGaze가 포함된 legacy inclusive field는 appendix에서만 확인하고, 메인 비교/차트에는 쓰지 않습니다.

## 표시명과 정렬

Markdown과 SVG chart는 raw JSON field 대신 짧은 표시명을 우선 사용합니다.

| 표시명 | 의미 |
| --- | --- |
| `Decode/read` | 비디오 읽기, seek, decode, frame 변환. 같은 비디오/sampling이면 on/off 공통 비용에 가까움 |
| `Prep rest` | decode/read와 AutoGaze를 제외한 resize/tile/thumbnail/tensorize/packing |
| `AutoGaze` | selector 입력 준비와 selector 실행 |
| `ViT` | SigLIP/ViT vision encoder |
| `LLM` | generate/prefill/decode 구간 |
| `Full patch` | selector 적용 전 patch/token 분모 |
| `Selected patch` | AutoGaze/token selector 이후 남은 patch/token |
| `Patch x` | `Full patch / Selected patch` |

Aggregate report의 기본 정렬은 `comparison`입니다. 같은 config 안에서 keep-all/off baseline을 먼저, AutoGaze/token selector를 다음, probe/sidecar/OOM을 뒤로 배치합니다. 필요하면 `--sort latency|token-reduction|memory|accuracy|status`로 바꿀 수 있습니다.

## Token/Patch 해석

| 기준 | 의미 |
| --- | --- |
| full/off 예상 patch | AutoGaze 없이 encoder에 들어갈 patch budget |
| multiscale candidate patch | AutoGaze가 scale별로 고려한 후보 patch 총량 |
| AutoGaze selected patch | selector가 실제 남긴 patch |
| encoder input token | ViT/SigLIP에 실제 들어간 token |
| LLM visual token | projector/token shuffle 이후 LLM context에 들어간 visual token |

모델별로 patch와 LLM visual token은 1:1이 아닐 수 있습니다. 그래서 report는 “encoder 이전 patch 감소”와 “LLM context 감소”를 분리해서 보여야 합니다.

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
| stacked latency bar | Decode/read/Prep rest/AutoGaze/ViT/LLM 중 어디가 병목인지 표시. 같은 stage는 항상 같은 색상 |
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
