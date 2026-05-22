# AutoGaze 재현/벤치마크 문서 허브

이 문서는 이 브랜치에서 추가한 AutoGaze 재현, 벤치마크, 프로파일링, 확장 실험 문서의 입구입니다. 루트의 `README.md`, `QUICK_START.md`, `TRAIN.md`, `INTEGRATION.md`와 `autogaze/` 내부 자료는 upstream 공식 문서로 취급하며 수정하지 않습니다.

## 현재 목표

AutoGaze를 실제 비디오 MLLM 파이프라인에 붙였을 때 다음을 재현 가능하게 측정합니다.

- AutoGaze 적용 전후 latency, token/patch 처리량, memory, accuracy 차이
- HLVid 기준 paper baseline과 NVILA-HD AutoGaze 비교
- Qwen/LongVILA/NVILA-Video 등 다른 MLLM으로 확장 가능한 plugin 실험
- 실행 결과를 Markdown, SVG chart, aggregate trend report로 정리

## 먼저 볼 문서

| 목적 | 문서 |
| --- | --- |
| 전체 실행 순서 | [AUTOGAZE_REPRO_RUNBOOK_KO.md](AUTOGAZE_REPRO_RUNBOOK_KO.md) |
| 영어 실행 요약 | [AUTOGAZE_REPRO_RUNBOOK.md](AUTOGAZE_REPRO_RUNBOOK.md) |
| HLVid benchmark 세부 가이드 | [AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md](AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md) |
| Markdown/chart/trend report 해석 | [AUTOGAZE_REPORTING_GUIDE_KO.md](AUTOGAZE_REPORTING_GUIDE_KO.md) |
| AutoGaze latency 검증 | [AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md](AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md) |
| AutoGaze gazing policy와 NVILA 차이 | [AUTOGAZE_GAZING_POLICY_AND_NVILA_INTEGRATION_KO.md](AUTOGAZE_GAZING_POLICY_AND_NVILA_INTEGRATION_KO.md) |
| Streaming/H100 preflight 추천 | [STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md](STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md) |
| Plugin 구조와 확장 계획 | [AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md](AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md) |
| Token selector / ViT / MLLM 연결성 | [AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md](AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md) |
| CUDA 결과 기록 템플릿 | [CUDA_RESULTS_TEMPLATE.md](CUDA_RESULTS_TEMPLATE.md) |
| 과거 상세 로그/긴 설명 | [temp/](temp/) |

## 목적별 Runner 지도

| 목적 | 권장 entrypoint | 설명 |
| --- | --- | --- |
| 단일 비디오 inference | `python -m repro.nvila_runner --mode single` | NVILA-HD + AutoGaze 안정 경로. 시각화와 상세 timing/token/memory 기록 가능 |
| Direct HLVid 실행 | `python -m repro.nvila_runner --mode hlvid` | 한 가지 `gazing-mode`로 HLVid manifest를 직접 실행 |
| 기본 HLVid benchmark | `python scripts/run_hlvid_folder_benchmark.py` | keep-all/autogaze 비교, paper baseline 비교, H100 preflight, gain report |
| Plugin HLVid benchmark | `python -m repro.plugin_hlvid_benchmark` | Qwen/LongVILA/NVILA-Video 등 확장 실험용 benchmark |
| Plugin single/inspect | `python -m repro.flexible_runner` | token selector / ViT / MLLM 조합을 명시해 실험 |
| Streaming profile | `python -m repro.nvila_runner --mode stream-profile` | LLM 없이 decode/tile/AutoGaze/SigLIP 구간 profile |
| Streaming sweep | `python -m repro.stream_profile_sweep` | 여러 stream config 후보를 비교 |
| Markdown report | `python -m repro.markdown_report` | 단일/benchmark JSON을 표와 SVG chart가 있는 Markdown으로 변환 |
| Trend report | `python -m repro.aggregate_reports` | 여러 실험 폴더의 JSON을 모아 CSV/Markdown/SVG trend 생성 |

## 기본 Benchmark와 Plugin Benchmark 구분

| 구분 | 기준 | 사용처 |
| --- | --- | --- |
| 기본 HLVid benchmark | `scripts/run_hlvid_folder_benchmark.py` / `repro.hlvid_batch_benchmark` | NVILA-HD keep-all/autogaze 비교, 논문 baseline, H100 OOM preflight |
| Direct HLVid runner | `repro.nvila_runner --mode hlvid` | wrapper 없이 한 mode만 직접 실행하거나 debugging할 때 |
| Plugin HLVid benchmark | `repro.plugin_hlvid_benchmark` | Qwen, LongVILA, NVILA-Video, InternVL 등 확장 조합을 같은 HLVid row로 비교 |

`plugin_hlvid_benchmark`는 기본 benchmark가 아니라 확장성 검증용입니다. 리더 설득용 NVILA-HD paper-facing 결과는 기본 HLVid benchmark wrapper를 우선 사용하세요.

## 모델/모드 지원 상태

| 모델/모드 | 상태 | AutoGaze 의미 |
| --- | --- | --- |
| NVILA-HD-Video | 안정 경로 | native processor 안에서 AutoGaze on/off, profiling, visualization, HLVid 가능 |
| NVILA-8B-Video paper baseline | 준비됨 | AutoGaze not applicable. 논문 baseline 재현 후보 |
| NVILA-HD keep-all | ablation | HD 모델에서 AutoGaze selection만 끈 비교용. paper baseline과 혼동 금지 |
| NVILA-Video plugin | off/probe | VILA CLI/off smoke와 feature packing probe 중심 |
| LongVILA plugin | off/probe | VILA CLI/off smoke와 AutoGaze attachment probe 중심 |
| Qwen3-VL | PoC/실험 | full ViT, chunked ViT, chunked ViT + AutoGaze sparse 비교 |
| Qwen2/2.5-VL | adapter 준비 | post-encoder prune/probe 후보 |
| LLaVA-OneVision | adapter 준비 | Qwen 계열 MLLM packing reference 후보 |
| InternVL3 | adapter 준비 | dynamic tile/`num_patches_list` 기반 확장 후보 |

## 확장성 지도

| 축 | 현재 형태 | 다음 확장 방향 |
| --- | --- | --- |
| `token_selector` | keep-all, AutoGaze, PixelPrune reference, external mask 계약 | SparseGazePlan 표준화, selector별 token/latency/memory 비교 |
| `vit_encoder` | NVILA SigLIP, Qwen grid ViT/chunked ViT | V-JEPA2, InternVL dynamic tile, Qwen pre-ViT sparse hook 안정화 |
| `mllm` | NVILA-HD, VILA CLI 계열, Qwen, LLaVA-OneVision, InternVL3 adapter | visual token packing과 position/grid metadata를 모델별로 명확히 기록 |
| benchmark task | HLVid/VideoQA schema | multiple-choice VideoQA 이후 caption/action task adapter 확장 |

## 추천 실행 순서

1. NVILA single smoke로 모델/비디오 path 확인
2. HLVid `--limit 3` 기본 benchmark로 keep-all/autogaze 결과 확인
3. Markdown report로 latency/token/memory/accuracy 표와 chart 확인
4. 여러 해상도/프레임/thumbnail 설정을 돌린 뒤 aggregate trend report 생성
5. 필요한 경우 paper baseline 또는 plugin Qwen/LongVILA 실험으로 확장
6. CUDA 머신 실험 결과는 [CUDA_RESULTS_TEMPLATE.md](CUDA_RESULTS_TEMPLATE.md)에 맞춰 요약

## 결과에서 먼저 볼 것

| 영역 | 핵심 필드 |
| --- | --- |
| Latency | `total_ms`, `video_decode_read_ms`, `preprocess_rest_without_decode_autogaze_ms`, `selector_input_build_ms`, `autogaze_total_ms`, `vision_input_build_ms`, `siglip_vision_ms`, `llm_forward_ms`, `generate_ms` |
| Token/Patch | full/off 예상 patch, AutoGaze selected patch, encoder input patch, LLM visual token |
| Memory | processor/autogaze/ViT/LLM/overall peak memory |
| Benchmark | `accuracy_total`, `accuracy_scored`, `failed`, `parse_failed`, `oom`, `skipped` |
| OOM | `failure.kind`, `failure.stage`, aggregate `status_by_config.svg` |

## 보존 문서

긴 field dictionary, 과거 MPS smoke 수치, command dump, 구현 체크리스트는 [docs/temp](temp/)에 보존합니다. 현재 실행 판단에는 메인 런북과 이 허브 문서를 우선 사용하세요.
