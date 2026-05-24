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
| 기본/Plugin HLVid benchmark | `python scripts/run_hlvid_folder_benchmark.py` | 기본 3모드 keep-all/single-scale dense/autogaze 비교, paper baseline 비교, H100 preflight, `--plugin-suite qwen|vila|llava|expand-smoke` 확장 실험 라우팅 |
| Plugin HLVid 내부 경로 | `python -m repro.plugin_hlvid_benchmark` | Qwen2.5/Qwen3/LongVILA/NVILA-Video/LLaVA 등 확장 실험을 직접 호출할 때 |
| Video task benchmark | `python -m repro.video_task_benchmark` | HLVid 외 VideoQA/captioning/action classification manifest를 plugin runner로 실행 |
| Video task 자산 준비 | `python scripts/prepare_video_task_assets.py` | CUDA 머신에서 HF dataset snapshot과 모델 weight를 local dir로 다운로드 |
| Plugin single/inspect | `python -m repro.flexible_runner` | token selector / ViT / MLLM 조합을 명시해 실험 |
| Qwen sparse preflight | `python -m repro.qwen_sparse_preflight` | CUDA 없이 Qwen grid patch 수, visual token 수, context/H100 risk를 정적 계산 |
| VILA pre-ViT probe | `python -m repro.vila_feature_probe` | NVILA-Video/LongVILA를 external CLI가 아닌 in-process hook으로 옮기기 전 필요한 tensor/position boundary 기록 |
| InternVL dynamic tile probe | `python -m repro.internvl_dynamic_tile_probe` | InternVL3의 dynamic tile order, `num_patches_list`, thumbnail 정책 기록 |
| Streaming profile | `python -m repro.nvila_runner --mode stream-profile` | LLM 없이 decode/tile/AutoGaze/SigLIP 구간 profile |
| Streaming sweep | `python -m repro.stream_profile_sweep` | 여러 stream config 후보를 비교 |
| Markdown report | `python -m repro.markdown_report` | 단일/benchmark JSON을 표와 SVG chart가 있는 Markdown으로 변환 |
| Trend report | `python -m repro.aggregate_reports` | 여러 실험 폴더의 JSON을 모아 CSV/Markdown/SVG trend 생성 |

## 기본 Benchmark와 Plugin Benchmark 구분

| 구분 | 기준 | 사용처 |
| --- | --- | --- |
| 기본 HLVid benchmark | `scripts/run_hlvid_folder_benchmark.py` / `repro.hlvid_batch_benchmark` | 기본 3모드: NVILA-HD keep-all, single-scale dense, AutoGaze 비교. 논문 baseline, H100 OOM preflight |
| Direct HLVid runner | `repro.nvila_runner --mode hlvid` | wrapper 없이 한 mode만 직접 실행하거나 debugging할 때 |
| Plugin HLVid benchmark | `scripts/run_hlvid_folder_benchmark.py --plugin-suite qwen|vila|llava|expand-smoke` / `repro.plugin_hlvid_benchmark` | Qwen, LongVILA, NVILA-Video, LLaVA, InternVL 등 확장 조합을 같은 HLVid row로 비교 |

확장성 검증도 사용 관점에서는 `scripts/run_hlvid_folder_benchmark.py`를 우선 사용합니다. `--plugin-suite qwen`은 Qwen2.5/Qwen3의 full, chunked, AutoGaze sparse 비교를 plugin 경로로 자동 라우팅합니다. `--plugin-suite vila`, `--plugin-suite llava`, `--plugin-suite expand-smoke`는 확장 smoke와 dependency 상태 확인용입니다. 리더 설득용 NVILA-HD paper-facing 결과는 plugin 옵션 없이 기본 HLVid benchmark wrapper를 사용하세요.

## 모델/모드 지원 상태

| 모델/모드 | 상태 | AutoGaze 의미 |
| --- | --- | --- |
| NVILA-HD-Video | 안정 경로 | native processor 안에서 AutoGaze on/off, profiling, visualization, HLVid 가능 |
| NVILA-8B-Video paper baseline | 준비됨 | AutoGaze not applicable. 논문 baseline 재현 후보 |
| NVILA-HD keep-all | ablation | HD 모델에서 AutoGaze selection만 끈 비교용. paper baseline과 혼동 금지 |
| NVILA-Video plugin | off/sidecar generate | external VILA CLI dense generation + AutoGaze selector sidecar metric. 아직 pruning gain 주장 금지 |
| LongVILA plugin | off/sidecar generate | external VILA CLI dense generation + AutoGaze selector sidecar metric. in-process hook 후 actual prune 예정 |
| Qwen3-VL | PoC/실험 | full ViT, chunked ViT, chunked ViT + AutoGaze sparse 비교 |
| Qwen2.5-VL | PoC/실험 | Qwen3와 같은 grid adapter로 full/chunked/AutoGaze sparse 비교 |
| Qwen2-VL | adapter 준비 | 같은 Qwen grid family이나 CUDA smoke 전까지 별도 상태로 분리 |
| LLaVA-OneVision | PoC/실험 | off dense와 post-encoder visual-token prune generate 비교 |
| InternVL3 | adapter 준비 | dynamic tile/`num_patches_list` 기반 확장 후보 |

## 확장성 지도

| 축 | 현재 형태 | 다음 확장 방향 |
| --- | --- | --- |
| `token_selector` | keep-all, AutoGaze, PixelPrune reference, external mask 계약 | SparseGazePlan 표준화, selector별 token/latency/memory 비교 |
| `vit_encoder` | NVILA SigLIP, Qwen grid ViT/chunked ViT | V-JEPA2, InternVL dynamic tile, Qwen pre-ViT sparse hook 안정화 |
| `mllm` | NVILA-HD, VILA CLI 계열, Qwen, LLaVA-OneVision, InternVL3 adapter | visual token packing과 position/grid metadata를 모델별로 명확히 기록 |
| benchmark task | HLVid/VideoQA, captioning, action classification schema | VideoQA는 choice/text scoring, caption은 reference 보존 + overlap hint, action은 exact/choice scoring |

## HLVid 외 Video Task Benchmark

HLVid 이후 task 확장은 `repro.video_task_benchmark`를 사용합니다. 이 경로는 `flexible_runner`를 row별로 호출하므로 Qwen/LongVILA/LLaVA 등 plugin mode와 같은 latency/token/memory/failure logging 구조를 재사용합니다. 우선 선택한 smoke dataset은 captioning용 `VLM2Vec/MSR-VTT`, action classification용 `bitmind/UCF101-Videos`, VideoQA용 `VLM2Vec/EgoSchema`, `VLM2Vec/nextqa`, `vid-modeling/videomme`, `VLM2Vec/ActivityNetQA`입니다.

CUDA 머신에서 dataset/model weight를 먼저 받을 때는 HF snapshot 기반 준비 스크립트를 사용합니다.

```bash
.venv/bin/python scripts/prepare_video_task_assets.py \
  --dry-run \
  --local-root /data/video_tasks \
  --weight-root /models/weight \
  --dataset-preset caption-action-smoke \
  --model-preset qwen-compare
```

VideoQA 후보 dataset은 따로 받을 수 있습니다.

```bash
.venv/bin/python scripts/prepare_video_task_assets.py \
  --dry-run \
  --local-root /data/video_tasks \
  --weight-root /models/weight \
  --dataset-preset videoqa-smoke \
  --model-preset qwen-compare
```

```bash
.venv/bin/python scripts/prepare_video_task_assets.py \
  --local-root /data/video_tasks \
  --weight-root /models/weight \
  --dataset-preset caption-action-smoke \
  --model-preset qwen-compare
```

모델 preset은 `qwen-video-task`(Qwen3-VL + AutoGaze), `qwen-compare`(Qwen2.5-VL + Qwen3-VL + AutoGaze), `expand-smoke`(Qwen/NVILA/LLaVA/InternVL 계열 smoke용)를 지원합니다. 대용량 모델은 CUDA 머신의 디스크/네트워크 정책에 맞춰 `--include`, `--exclude`, `--max-workers`로 조절하세요.

다운로드 후 local metadata를 우리 manifest로 변환합니다.

```bash
.venv/bin/python scripts/convert_video_task_dataset.py \
  --dataset-preset msrvtt-caption \
  --input /data/video_tasks/msrvtt \
  --output /data/video_tasks/manifests/msrvtt_caption.jsonl
```

```bash
.venv/bin/python scripts/convert_video_task_dataset.py \
  --dataset-preset ucf101-action \
  --input /data/video_tasks/ucf101-videos \
  --output /data/video_tasks/manifests/ucf101_action.jsonl
```

```bash
.venv/bin/python scripts/convert_video_task_dataset.py \
  --dataset-preset nextqa-videoqa \
  --input /data/video_tasks/nextqa \
  --output /data/video_tasks/manifests/nextqa_videoqa.jsonl
```

```bash
.venv/bin/python -m repro.video_task_benchmark \
  --task-type captioning \
  --manifest /path/to/caption_manifest.jsonl \
  --video-root /path/to/videos \
  --output-dir outputs/autogaze_repro/video_task_caption_qwen_limit3 \
  --modes qwen3_full_vit,qwen3_chunked_vit,qwen3_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3 \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --video-resize-longest-edge 448 \
  --max-tiles-video 4 \
  --max-new-tokens 32
```

```bash
.venv/bin/python -m repro.video_task_benchmark \
  --task-type action_classification \
  --manifest /path/to/action_manifest.jsonl \
  --video-root /path/to/videos \
  --output-dir outputs/autogaze_repro/video_task_action_qwen_limit3 \
  --modes qwen3_full_vit,qwen3_chunked_vit,qwen3_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3 \
  --num-video-frames 32 \
  --qwen-video-nframes 32 \
  --video-resize-longest-edge 448 \
  --max-tiles-video 4 \
  --max-new-tokens 8
```

Caption manifest는 `video_path`와 `caption` 또는 `references`가 필요합니다. Caption 점수는 기본 `not_scored`이고, reference overlap hint만 별도 기록합니다. Action manifest는 `video_path`와 `label` 또는 `answer`가 필요하며, `choices`가 있으면 multiple-choice letter parsing을 같이 사용합니다.
VideoQA manifest는 `video_path`, `question`, `answer`가 필요합니다. 정답이 A/B/C/D/E 같은 choice letter이면 multiple-choice parsing을 사용하고, open answer이면 normalize된 text containment로 기본 scoring합니다.

## Qwen Sparse Preflight

Qwen plugin sparse mode를 CUDA에서 돌리기 전에 patch/token 규모와 context risk를 정적으로 볼 수 있습니다.

```bash
.venv/bin/python -m repro.qwen_sparse_preflight \
  --model-family qwen3-vl \
  --num-frames 128 \
  --height 720 \
  --width 1280 \
  --autogaze-reduction-ratio 10 \
  --context-limit 32768 \
  --h100-budget-gib 70
```

이 값은 CUDA allocator 실측이 아니라 scheduler preflight입니다. 실제 주장은 `plugin_hlvid_report.md`의 pairwise table, `pairwise_latency_speedup.svg`, `pairwise_token_reduction.svg`, 그리고 CUDA memory log를 함께 보고 판단하세요.

## Pre-ViT Sparse 확장 우선순위

| 순서 | 모델/계열 | 현재 상태 | 의미 |
| --- | --- | --- | --- |
| 1 | Qwen2.5-VL / Qwen3-VL | `implemented_pending_cuda` | `video_grid_thw` 기반 pre-ViT sparse 경로가 코드에 있고 CUDA smoke가 다음 단계 |
| 2 | NVILA-Video / LongVILA | `in_process_probe_required` | external CLI sidecar를 넘어 processor/vision tower/projector hook을 직접 잡아야 함 |
| 3 | InternVL3 | `dynamic_tile_probe_required` | dynamic tile과 `num_patches_list`를 먼저 맞춘 뒤 tile-level prune부터 시도 |
| 4 | LLaVA-OneVision | `candidate_design_required` | pooled video token 구조라 frame/tile-level pre-ViT를 먼저 검토 |
| 5 | Generic SigLIP/V-JEPA2 | `vit_only_adapter_required` | MLLM 연결 전에 ViT 단독 sparse benchmark로 시작 |

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
| Latency | `total_ms`, `video_decode_read_ms`, `preprocess_rest_without_decode_autogaze_ms`, `selector_input_build_ms`, `autogaze_total_ms`, `vision_input_build_ms`, `siglip_vision_ms`, `mm_projector_ms`, `generate_ms`, `llm_generation_ms`, `llm_forward_ms`, `generation_rest_ms` |
| Token/Patch | full/off 예상 patch, AutoGaze selected patch, encoder input patch, LLM visual token |
| Memory | processor/autogaze/ViT/LLM/overall peak memory |
| Benchmark | `accuracy_total`, `accuracy_scored`, `failed`, `parse_failed`, `oom`, `skipped` |
| OOM | `failure.kind`, `failure.stage`, aggregate `status_by_config.svg` |

## 보존 문서

긴 field dictionary, 과거 MPS smoke 수치, command dump, 구현 체크리스트는 [docs/temp](temp/)에 보존합니다. 현재 실행 판단에는 메인 런북과 이 허브 문서를 우선 사용하세요.
