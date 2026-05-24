# AutoGaze 재현 런북

이 런북은 이 브랜치에서 추가한 AutoGaze 재현/벤치마크 실행 순서를 다룹니다. 문서 지도는 [INDEX_KO.md](INDEX_KO.md)를 먼저 보세요. 루트 `README.md`, `QUICK_START.md`, `TRAIN.md`, `INTEGRATION.md`와 `autogaze/` 내부 자료는 upstream 공식 문서이므로 수정하지 않습니다.

## 공식 소스

- AutoGaze 코드: https://github.com/NVlabs/AutoGaze
- AutoGaze 프로젝트 페이지: https://autogaze.github.io/
- AutoGaze 논문: https://arxiv.org/abs/2603.12254
- HLVid 데이터셋: https://huggingface.co/datasets/bfshi/HLVid
- NVILA-HD-Video README: https://github.com/NVlabs/VILA/tree/main/vila_hd/nvila_hd_video

## 현재 지원 범위

| 영역 | 상태 | 주 entrypoint |
| --- | --- | --- |
| NVILA-HD 단일 inference | 안정 경로 | `python -m repro.nvila_runner --mode single` |
| 시각화 | 안정 경로 | `--visualization-output-dir` |
| 기본 HLVid benchmark | 안정 경로 | `python scripts/run_hlvid_folder_benchmark.py` |
| paper baseline 비교 | 준비됨 | `run_hlvid_folder_benchmark.py --paper-baseline --paper-hd-autogaze` |
| stream-profile/H100 preflight | 준비됨 | `repro.nvila_runner`, `repro.hlvid_batch_benchmark` |
| Markdown/chart report | 준비됨 | `python -m repro.markdown_report` |
| aggregate trend report | 준비됨 | `python -m repro.aggregate_reports` |
| plugin 확장 실험 | PoC/probe/actual 일부 | `run_hlvid_folder_benchmark.py --plugin-suite qwen|vila|llava|expand-smoke`, `python -m repro.flexible_runner` |

HLVid 실행은 `scripts/run_hlvid_folder_benchmark.py`를 우선 사용하세요. 옵션을 주지 않으면 NVILA-HD keep-all/single-scale/autogaze 기본 경로로 가고, `--plugin-suite`를 주면 Qwen/VILA/LLaVA plugin HLVid 경로로 라우팅됩니다. `repro.plugin_hlvid_benchmark`는 내부/고급 호출용으로 남겨둡니다.

## 환경 세팅

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements-repro.txt
bash scripts/bootstrap_official_repos.sh
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

Apple MPS/macOS에서는 공식 AutoGaze dependency 중 `flash_attn`이 맞지 않을 수 있으므로 `requirements-repro.txt` 설치 후 editable install은 `--no-deps --no-build-isolation`로 둡니다. CUDA 머신에서는 모델 weight를 로컬 `weight/` 아래에 두는 것을 권장합니다.

Qwen3-VL 실험은 `qwen-vl-utils`가 필요합니다. 기본 requirements에 포함되어 있지만 CUDA 환경에서 누락되면 아래를 다시 실행하세요.

```bash
.venv/bin/python -m pip install -r requirements-repro.txt
```

## 1. NVILA-HD Single Inference Smoke

가장 먼저 단일 비디오로 모델 로드, 비디오 샘플링, AutoGaze, generation, metric logging을 확인합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --model-path nvidia/NVILA-8B-HD-Video \
  --autogaze-model nvidia/AutoGaze \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --prompt "Question: What is happening in the video? Please answer briefly." \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --measure-ttft \
  --warmup-runs 1 \
  --repeat-runs 3 \
  --print-summary \
  --summary-json outputs/autogaze_repro/nvila_single_summary.json \
  --output-json outputs/autogaze_repro/nvila_single.json
```

같은 입력에서 AutoGaze off/keep-all을 보려면 `--gazing-mode keep-all`만 바꿉니다. 392px single-scale dense keep-all은 `--gazing-mode keep-all-single`을 사용합니다. 이 alias는 내부적으로 `keep-all + --autogaze-target-scales 392 + --autogaze-target-patch-size 14`로 정규화됩니다. 상대 latency를 비교할 때는 같은 frame 수, thumbnail 수, resize, max tiles, dtype, batch size를 유지하세요.

| single mode | 의미 |
| --- | --- |
| `--gazing-mode autogaze` | AutoGaze selector on |
| `--gazing-mode keep-all` | NVILA-HD multiscale keep-all |
| `--gazing-mode keep-all-single` | 392px single-scale dense keep-all |

실행 후 `nvila_single_summary.json` 또는 Markdown report의 `Key Metrics`에서 먼저 확인할 값은 다음입니다.

```text
Mode Snapshot:  Keep-all / Single-scale / AutoGaze 원값 나란히 보기
Pairwise Gains: AutoGaze vs keep-all, AutoGaze vs single-scale 두 기준만 speedup/reduction 계산
Token Boundary: Candidate/off patch -> encoder input patch -> LLM visual token
```

`Mode Snapshot`에는 ratio를 넣지 않습니다. ratio는 분모가 `keep-all`인지 `single-scale dense`인지에 따라 의미가 달라지므로 `Pairwise Gains`에서만 봅니다. `Selector input`은 상세 latency view나 raw appendix에서 확인할 수 있으며, 메인 요약에서는 AutoGaze total과 ViT/LLM 경계 값을 우선 봅니다.

## 2. 선택 프레임/AutoGaze Overlay 시각화

AutoGaze가 어떤 프레임과 패치를 남겼는지 확인하려면 visualization 옵션을 켭니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --visualization-output-dir outputs/autogaze_repro/visualizations \
  --visualization-fps 4 \
  --visualization-selected-max-long-side 1280 \
  --output-json outputs/autogaze_repro/nvila_single_with_viz.json
```

저장되는 주요 파일은 selected-frame video, processor-resolution video, AutoGaze overlay video, `gazing_info.json`입니다. `keep-all` 모드에서는 selected/processor frame video만 저장되고 overlay는 skipped 상태로 기록됩니다.

## 3. 기본 HLVid Benchmark

로컬 HLVid 폴더에 manifest와 mp4가 있는 경우 기본 benchmark wrapper를 사용합니다. `video-root`는 mp4가 들어 있는 폴더입니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_batch_limit3 \
  --limit 3 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --measure-ttft \
  --continue-on-error
```

기본 wrapper는 HD multiscale keep-all, 392px single-scale dense keep-all, AutoGaze를 같은 설정으로 각각 실행한 뒤 gain report를 만듭니다. 핵심 산출물은 다음과 같습니다.

```text
hlvid_keep_all_predictions.jsonl
hlvid_single_scale_dense_predictions.jsonl
hlvid_autogaze_predictions.jsonl
hlvid_keep_all_summary.json
hlvid_single_scale_dense_summary.json
hlvid_autogaze_summary.json
hlvid_autogaze_gain_report.json
hlvid_autogaze_gain_report.csv
```

한 mode만 직접 확인할 때는 `repro.nvila_runner --mode hlvid`를 사용합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --manifest /path/to/HLVid/manifest.json \
  --hlvid-video-root /path/to/HLVid/videos \
  --gazing-mode autogaze \
  --limit 3 \
  --continue-on-error \
  --predictions outputs/autogaze_repro/hlvid_autogaze_predictions.jsonl \
  --summary outputs/autogaze_repro/hlvid_autogaze_summary.json \
  --scored-predictions outputs/autogaze_repro/hlvid_autogaze_scored.jsonl
```

필요한 모드만 빼려면 skip 옵션을 사용합니다.

```bash
  --skip-keep-all              # HD multiscale keep-all 제외
  --skip-single-scale-dense    # 392px single-scale dense 제외
  --skip-autogaze              # AutoGaze 제외
```

`--single-scale-dense`는 이전 명령과의 호환을 위해 남아 있지만 이제 기본값이 on입니다. single-scale dense scale을 바꾸고 싶을 때만 `--single-scale-dense-scales 392`처럼 값을 지정하세요. `hlvid_single_scale_dense_*` 파일과 gain report의 `single_scale_dense_comparison`에서 AutoGaze 대비 latency/token 차이를 확인합니다.

## 4. Paper Baseline 비교

AutoGaze 논문 표의 baseline은 HD keep-all이 아니라 `NVILA-8B-Video` 별도 모델로 봅니다. wrapper의 paper comparison 모드를 사용하세요.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_paper_comparison \
  --paper-baseline \
  --paper-hd-autogaze \
  --paper-comparison-report \
  --limit 3 \
  --continue-on-error
```

출력 `hlvid_paper_comparison_report.json`에서 먼저 볼 필드는 `paper_reference_accuracy`, `measured_accuracy`, `delta_from_reference`, `failed`, `oom`, `parse_failed`, `skipped`입니다.

## 5. Plugin 확장 Benchmark

Qwen/LongVILA/NVILA-Video 등 다른 token selector / ViT / MLLM 조합을 HLVid row로 비교할 때는 같은 wrapper에 `--plugin-suite`를 붙입니다. Qwen 검증은 이 명령부터 시작하세요.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_vit_limit3 \
  --plugin-suite qwen \
  --plugin-model qwen2.5-vl=weight/Qwen2.5-VL-7B-Instruct \
  --plugin-model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3 \
  --num-video-frames 32 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 4 \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --qwen-thumbnail-mode append-video \
  --video-resize-longest-edge 448 \
  --max-new-tokens 8
```

`--plugin-suite qwen`은 Qwen2.5-VL과 Qwen3-VL 각각에 대해 full ViT, chunked ViT, chunked ViT + AutoGaze sparse 세 경로를 실행합니다. Qwen frame 수를 따로 주지 않으면 `--qwen-video-nframes`는 `--num-video-frames`와 같게 맞춰집니다. 더 좁은 비교를 원하면 `--plugin-suite custom --plugin-modes qwen3_full_vit,qwen3_chunked_vit`처럼 지정합니다.

Qwen suite mode의 의미는 다음과 같습니다.

| mode | 의미 |
| --- | --- |
| `qwen2.5_full_vit`, `qwen3_full_vit` | Qwen native/full ViT 경로 |
| `qwen2.5_chunked_vit`, `qwen3_chunked_vit` | Qwen ViT를 temporal/spatial chunk로 나눠 실행하되 AutoGaze off |
| `qwen2.5_chunked_vit_autogaze_sparse`, `qwen3_chunked_vit_autogaze_sparse` | AutoGaze selected token만 Qwen ViT/MLLM context에 통과시키는 pre-ViT sparse 경로 |

다른 suite는 다음처럼 호출합니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_expand_smoke_limit3 \
  --plugin-suite expand-smoke \
  --plugin-model qwen2.5-vl=weight/Qwen2.5-VL-7B-Instruct \
  --plugin-model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --plugin-model nvila-video=weight/NVILA-8B-Video \
  --plugin-model longvila=weight/LongVILA \
  --plugin-model llava-onevision=weight/LLaVA-OneVision \
  --external-mllm-command vila-infer \
  --limit 3 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --video-resize-longest-edge 224 \
  --max-tiles-video 4 \
  --max-new-tokens 8 \
  --continue-on-error
```

`--plugin-suite vila`는 `nvila-video-off`, `nvila-video-autogaze-actual`, `longvila-off`, `longvila-autogaze-actual`를 실행합니다. 단, 현재 VILA-family actual entry는 external CLI dense generation에 AutoGaze selector sidecar metric을 붙인 단계라서 ViT/LLM token pruning 성공으로 해석하면 안 됩니다. `--plugin-suite llava`는 LLaVA-OneVision off와 post-encoder visual-token prune generate 경로를 비교합니다.

### Caption/Action Benchmark

HLVid가 아닌 caption/action task는 `repro.video_task_benchmark`를 사용합니다. 이 경로는 같은 plugin mode를 row별 `flexible_runner --mode single`로 실행하고, task별 scoring만 분리합니다. 현재 우선 smoke dataset은 captioning용 `VLM2Vec/MSR-VTT`, action classification용 `bitmind/UCF101-Videos`입니다.

```bash
.venv/bin/python scripts/prepare_video_task_assets.py \
  --local-root /data/video_tasks \
  --weight-root /models/weight \
  --dataset-preset caption-action-smoke \
  --model-preset qwen-compare
```

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
.venv/bin/python -m repro.video_task_benchmark \
  --task-type captioning \
  --manifest /data/video_tasks/manifests/msrvtt_caption.jsonl \
  --video-root /data/video_tasks/msrvtt \
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
  --manifest /data/video_tasks/manifests/ucf101_action.jsonl \
  --video-root /data/video_tasks/ucf101-videos \
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

Caption은 `video_path`와 `caption` 또는 `references`가 필요하고 기본 점수는 `not_scored`입니다. Action은 `video_path`와 `label` 또는 `answer`가 필요하며 exact label/choice parsing으로 accuracy를 계산합니다.

## 6. Stream Profile과 H100 Preflight

긴 4K 비디오에서 LLM을 로드하기 전에 decode/tile/AutoGaze/SigLIP 구간을 확인하려면 stream-profile을 사용합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --video /path/to/video.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --stream-chunk-frames 16 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --gazing-mode autogaze \
  --stream-run-siglip \
  --stream-siglip-mode both \
  --stream-profile-json outputs/autogaze_repro/stream_profile.json
```

HLVid 폴더 전체의 metadata 기반 H100 risk를 보려면 기본 benchmark wrapper의 preflight를 사용합니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /path/to/HLVid \
  --video-root /path/to/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_preflight \
  --h100-preflight \
  --h100-budget-gib 70 \
  --allow-missing-videos
```

## 7. Markdown Chart Report

단일 JSON, HLVid summary/gain report, plugin summary를 공유용 Markdown으로 변환합니다. 기본적으로 SVG chart asset도 같이 생성됩니다.

```bash
.venv/bin/python -m repro.markdown_report \
  --input-json outputs/autogaze_repro/hlvid_batch_limit3/hlvid_autogaze_gain_report.json \
  --output-md outputs/autogaze_repro/hlvid_batch_limit3/hlvid_autogaze_gain_report.md
```

HLVid gain report를 변환하면 `Key Metrics`, `Benchmark Score`, latency chart, token/patch 표가 `keep_all -> single_scale_dense -> autogaze` 순서로 정렬됩니다. `single_scale_dense`는 392px single-scale dense keep-all ablation이고, SVG chart에서는 `single-scale`로 짧게 표시됩니다. 메인 speedup/reduction은 `Pairwise Gains`의 `AutoGaze vs keep-all`, `AutoGaze vs single-scale` 두 줄만 기준으로 해석합니다. `video_preprocess_ms` 같은 legacy inclusive 값과 긴 raw alias는 `Raw Metric Appendix`에서만 디버깅용으로 봅니다.

정답 비교도 같은 관점으로 읽습니다. top-level correctness count는 기존 호환용 `keep_all vs autogaze`이고, 3모드 비교는 Markdown의 `Pairwise Correctness Summary`에서 `keep_all vs single_scale_dense`, `single_scale_dense vs autogaze`, `keep_all vs autogaze`를 각각 확인하세요.

chart가 필요 없으면 `--no-charts`를 붙입니다.

## 8. Aggregate Trend Report

여러 해상도, 프레임 수, thumbnail 수, model mode를 반복 실험한 뒤 전체 경향을 한 번에 보려면 aggregate report를 만듭니다.

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

OOM이 난 실행도 가능한 경우 `failure.kind=oom`, `failure.stage`와 함께 row로 남고, aggregate report의 status chart에 반영됩니다. HLVid direct runner는 model load/row inference failure를 기록하고, 기본 wrapper는 subprocess가 137 등으로 죽어도 partial JSONL/summary를 생성합니다. 단, OS가 프로세스를 즉시 kill한 경우 child 내부 stack trace는 남지 않을 수 있으므로 wrapper의 `failure.stage=subprocess`를 기준으로 보세요.

## 결과 해석 Quick Guide

| 영역 | 먼저 볼 필드 |
| --- | --- |
| 전체 시간 | `total_ms`, `latency_accounting.additive_total_ms` |
| 전처리 | `video_decode_read_ms`, `preprocess_rest_without_decode_autogaze_ms`, `video_preprocess_without_autogaze_ms`, `video_decode_ms`, `video_frame_resize_ms`, `video_tiling_ms` |
| AutoGaze | `autogaze_total_ms`, `selector_input_build_ms`, `autogaze_forward_ms`, `autogaze_model_forward_ms` |
| Vision encoder | `vision_encoder_ms`, `vision_input_build_ms`, `siglip_vision_ms`, `mm_projector_ms`, Qwen의 `qwen_vit_prepare_ms` |
| LLM/generation | `generate_ms`, `llm_generation_ms`, `llm_forward_ms`, `generation_rest_ms`, `ttft_ms` |
| patch/token 감소 | full/off patch, AutoGaze selected patch, encoder input patch, LLM visual token |
| memory | processor/autogaze/vision/LLM/overall peak |
| benchmark | `accuracy_total`, `accuracy_scored`, `failed`, `parse_failed`, `oom`, `skipped` |
| OOM 위치 | `failure.kind`, `failure.stage`, `failure.message` |

Primary latency 공식은 다음입니다.

```text
total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
```

메인 비교 표에서는 `video_preprocess_without_autogaze_ms`를 다시 `Decode/read ms`와 `Prep rest ms`로 나눠 봅니다. 같은 비디오와 같은 sampling이면 decode/read는 on/off 공통 비용에 가깝기 때문에, AutoGaze의 실제 이득은 `Prep rest + Selector input + AutoGaze + Vision input + ViT + LLM` 쪽에서 더 잘 보입니다.

`video_preprocess_ms`는 AutoGaze를 포함한 legacy inclusive field라 primary total에 다시 더하지 않습니다.

`generate_ms`는 LLM forward-only가 아니라 preprocessing 이후 전체 `model.generate` 부모 stage입니다. 리포트에서는 `LLM generation ms = llm_forward_ms + generation_rest_ms`를 vision path 제외 LLM generation 부담으로, `LLM forward ms`를 `llm_forward_ms` child timer로, `Generate rest ms`를 `generate_ms`에서 vision/LLM child timer를 뺀 residual로 분리해 보여줍니다.

## 추천 Config

| 목적 | 설정 |
| --- | --- |
| CUDA smoke | `128f / thumbnail 64 / max_tiles 8 / resize longest 720` |
| HLVid limit 확인 | 위 smoke 설정 + `--limit 3 --continue-on-error` |
| OOM 회피 우선 | `64f / thumbnail 16-32 / max_tiles 4 / resize longest 512-720` |
| paper-facing HD stress | `1024f / thumbnail 512 / 높은 max_tiles`는 H100 preflight 후 시도 |
| Qwen plugin smoke | `32f / thumbnail 8 / max_tiles 4 / resize longest 448` |

## Troubleshooting

| 증상 | 확인 |
| --- | --- |
| CUDA OOM | `failure.stage`, `peak_memory_bytes`, aggregate status chart 확인 |
| 1프레임 divisibility 오류 | AutoGaze chunk frame이 16 단위인지, 비디오 decode/resize path가 정상인지 확인 |
| Qwen `qwen_vl_utils` 누락 | `pip install -r requirements-repro.txt` 재실행 |
| HLVid available video 수 불일치 | `--prepare-only` 또는 layout report로 manifest와 mp4 basename 매칭 확인 |
| paper baseline과 HD keep-all 혼동 | baseline은 `NVILA-8B-Video`, HD keep-all은 ablation |

## 상세 문서

- HLVid benchmark 세부 가이드: [AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md](AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md)
- Markdown/chart/trend report 해석: [AUTOGAZE_REPORTING_GUIDE_KO.md](AUTOGAZE_REPORTING_GUIDE_KO.md)
- Plugin 구조: [AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md](AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md)
- Selector/ViT/MLLM 연결성: [AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md](AUTOGAZE_SELECTOR_VIT_MLLM_CONNECTION_REPORT_KO.md)
- Timing 검증: [AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md](AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md)
- Streaming/H100 config: [STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md](STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO.md)
- Gazing policy와 NVILA integration: [AUTOGAZE_GAZING_POLICY_AND_NVILA_INTEGRATION_KO.md](AUTOGAZE_GAZING_POLICY_AND_NVILA_INTEGRATION_KO.md)
- CUDA 결과 기록 템플릿: [CUDA_RESULTS_TEMPLATE.md](CUDA_RESULTS_TEMPLATE.md)
- 영어 실행 요약: [AUTOGAZE_REPRO_RUNBOOK.md](AUTOGAZE_REPRO_RUNBOOK.md)
- 과거 상세 로그: [temp/](temp/)
