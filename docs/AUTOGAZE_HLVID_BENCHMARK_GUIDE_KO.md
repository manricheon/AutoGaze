# HLVid Benchmark 실행 가이드

이 문서는 HLVid를 기준으로 AutoGaze on/off, paper baseline, plugin 확장 실험을 어떻게 구분해서 실행하고 해석할지 정리합니다. 빠른 실행 순서는 [AUTOGAZE_REPRO_RUNBOOK_KO.md](AUTOGAZE_REPRO_RUNBOOK_KO.md)를 먼저 보세요.

## Benchmark 경로 구분

| 경로 | 엔트리포인트 | 목적 |
| --- | --- | --- |
| 기본 wrapper | `python scripts/run_hlvid_folder_benchmark.py` | NVILA-HD keep-all/autogaze 비교, paper baseline, H100 preflight |
| batch module | `python -m repro.hlvid_batch_benchmark` | wrapper와 같은 동작을 module 형태로 직접 실행 |
| direct runner | `python -m repro.nvila_runner --mode hlvid` | 한 가지 `gazing-mode`만 직접 디버깅 |
| plugin benchmark | `python -m repro.plugin_hlvid_benchmark` | Qwen/LongVILA/NVILA-Video 등 확장 실험 |

리더 설득용 NVILA-HD 결과는 기본 wrapper를 우선 사용합니다. Plugin benchmark는 “다른 MLLM에도 AutoGaze 방식의 selector를 붙일 수 있는가”를 검증하는 별도 경로입니다.

## 데이터 폴더 규칙

CUDA 머신에서 HLVid가 mp4로 풀려 있다면 보통 다음 형태를 기대합니다.

```text
/data/HLVid
  manifest.json
  videos/
    xxx.mp4
    yyy.mp4
```

`--video-root`는 mp4가 들어 있는 폴더입니다. manifest의 `video_path`가 하위 경로를 포함하더라도 runner는 `video_root / video_path`와 `video_root / basename(video_path)`를 모두 확인합니다. 그래서 `videos/` 안에 mp4만 flat하게 있어도 대부분 매칭됩니다.

## 준비 상태 확인

전체 실행 전에 manifest와 mp4 매칭부터 확인합니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_layout_check \
  --prepare-only \
  --allow-missing-videos
```

available video 수가 실제 mp4 수보다 적으면 manifest 안의 row와 mp4 basename이 맞지 않는 것입니다. 이 경우 layout report와 missing 목록을 먼저 확인하세요.

## 기본 limit benchmark

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_limit3_128f_720 \
  --limit 3 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --measure-ttft \
  --warmup-runs 1 \
  --continue-on-error
```

기본 wrapper는 keep-all과 AutoGaze를 각각 실행합니다. 한쪽만 돌리려면 다음 옵션을 사용합니다.

| 목적 | 옵션 |
| --- | --- |
| AutoGaze만 | `--skip-keep-all` |
| keep-all만 | `--skip-autogaze` |
| 실패해도 다음 샘플 계속 | `--continue-on-error` |
| missing video 허용 | `--allow-missing-videos` |

## Paper baseline comparison

논문 table의 baseline은 HD keep-all이 아니라 `NVILA-8B-Video` 별도 row로 봅니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_paper_comparison \
  --paper-baseline \
  --paper-hd-autogaze \
  --paper-comparison-report \
  --limit 3 \
  --continue-on-error
```

| 모드 | 모델 | frame/resolution 기준 | AutoGaze |
| --- | --- | --- | --- |
| paper baseline | `Efficient-Large-Model/NVILA-8B-Video` | 256f / 448 target | not applicable |
| HD AutoGaze | `nvidia/NVILA-8B-HD-Video` | 1024f / high-res target | on |
| HD keep-all optional | `nvidia/NVILA-8B-HD-Video` | HD ablation | off |

HD keep-all은 useful ablation이지만 paper baseline으로 부르지 않습니다.

## Direct HLVid runner

특정 mode 하나의 error나 output을 빠르게 볼 때 사용합니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --manifest /data/HLVid/manifest.json \
  --hlvid-video-root /data/HLVid/videos \
  --gazing-mode autogaze \
  --limit 3 \
  --continue-on-error \
  --predictions outputs/autogaze_repro/direct_hlvid/autogaze_predictions.jsonl \
  --summary outputs/autogaze_repro/direct_hlvid/autogaze_summary.json \
  --scored-predictions outputs/autogaze_repro/direct_hlvid/autogaze_scored.jsonl
```

## Plugin HLVid benchmark

Qwen/LongVILA/NVILA-Video 등 확장 조합을 같은 manifest row로 비교할 때만 사용합니다.

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_limit3 \
  --modes qwen_full_vit,qwen_chunked_vit,qwen_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
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

## 주요 산출물

| 파일 | 의미 |
| --- | --- |
| `hlvid_keep_all_predictions.jsonl` | keep-all row별 prediction |
| `hlvid_autogaze_predictions.jsonl` | AutoGaze row별 prediction |
| `hlvid_keep_all_summary.json` | keep-all accuracy/failure/metric summary |
| `hlvid_autogaze_summary.json` | AutoGaze accuracy/failure/metric summary |
| `hlvid_autogaze_gain_report.json` | keep-all 대비 AutoGaze gain 비교 |
| `hlvid_autogaze_gain_report.csv` | gain report 표 형태 |
| `hlvid_paper_comparison_report.json` | paper reference score와 local score 비교 |

## 점수와 실패 해석

| 필드 | 의미 |
| --- | --- |
| `accuracy_total` | 전체 대상 기준 accuracy |
| `accuracy_scored` | parse/scoring 가능한 샘플 기준 accuracy |
| `failed` | runtime 실패 수 |
| `oom` | OOM으로 분류된 실패 수 |
| `parse_failed` | 답변 파싱 실패 수 |
| `skipped` | missing video 등으로 skip된 수 |
| `failure.kind` | `oom`, `missing_video`, `runtime_error` 등 |
| `failure.stage` | 실패한 pipeline 단계 |

OOM은 accuracy와 별도로 봐야 합니다. H100에서 keep-all이 OOM이고 AutoGaze가 통과한다면, 이것도 AutoGaze의 확장성 이득으로 기록할 수 있습니다.

## Token/latency/memory에서 먼저 볼 것

| 영역 | 비교 기준 |
| --- | --- |
| token/patch | full/off 예상 patch vs AutoGaze selected patch vs encoder input token vs LLM visual token |
| latency | preprocess, AutoGaze, ViT/vision encoder, LLM/generate, total |
| memory | processor/autogaze/ViT/LLM/overall peak |
| benchmark | 정답 여부, failure category, OOM stage |

AutoGaze on만 돌렸더라도 full/off 예상 patch budget은 summary에 남아야 합니다. 이 값이 있어야 keep-all을 실제로 못 돌린 OOM config에서도 token reduction 근거를 유지할 수 있습니다.
