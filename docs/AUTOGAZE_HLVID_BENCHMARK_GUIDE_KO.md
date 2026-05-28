# HLVid Benchmark 실행 가이드

이 문서는 HLVid를 기준으로 AutoGaze on/off, paper baseline, plugin 확장 실험을 어떻게 구분해서 실행하고 해석할지 정리합니다. 빠른 실행 순서는 [AUTOGAZE_REPRO_RUNBOOK_KO.md](AUTOGAZE_REPRO_RUNBOOK_KO.md)를 먼저 보세요.

## Benchmark 경로 구분

| 경로 | 엔트리포인트 | 목적 |
| --- | --- | --- |
| 통합 wrapper | `python scripts/run_hlvid_folder_benchmark.py` | NVILA-HD keep-all/autogaze 비교, paper baseline, H100 preflight, Qwen plugin suite 라우팅 |
| batch module | `python -m repro.hlvid_batch_benchmark` | wrapper와 같은 동작을 module 형태로 직접 실행 |
| direct runner | `python -m repro.nvila_runner --mode hlvid` | 한 가지 `gazing-mode`만 직접 디버깅 |
| plugin benchmark | `python -m repro.plugin_hlvid_benchmark` | Qwen/LongVILA/NVILA-Video 등 확장 실험의 내부/고급 경로 |

리더 설득용 NVILA-HD 결과와 Qwen 확장 smoke 모두 `scripts/run_hlvid_folder_benchmark.py`를 우선 사용합니다. Plugin benchmark는 “다른 MLLM에도 AutoGaze 방식의 selector를 붙일 수 있는가”를 검증하는 내부 경로이며, wrapper의 `--plugin-suite`가 이 경로로 라우팅합니다.

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

기본 wrapper는 세 모드, 즉 HD multiscale keep-all, 392px single-scale dense keep-all, AutoGaze를 각각 실행합니다. 필요한 모드만 빼려면 다음 옵션을 사용합니다.

| 목적 | 옵션 |
| --- | --- |
| AutoGaze만 | `--skip-keep-all --skip-single-scale-dense` |
| HD multiscale keep-all만 | `--skip-single-scale-dense --skip-autogaze` |
| single-scale dense만 | `--skip-keep-all --skip-autogaze` |
| keep-all 두 종류 + AutoGaze 제외 | `--skip-autogaze` |
| HD multiscale keep-all + AutoGaze만 | `--skip-single-scale-dense` |
| 실패해도 다음 샘플 계속 | `--continue-on-error` |
| missing video 허용 | `--allow-missing-videos` |

`--continue-on-error`는 row inference뿐 아니라 direct runner의 model load failure와 wrapper subprocess failure도 가능한 범위에서 JSONL/summary로 남깁니다. 프로세스가 137/SIGKILL 등으로 죽으면 wrapper가 `failure.kind=oom`, `failure.stage=subprocess` row를 만들고, manifest가 있으면 아직 완료되지 않은 row를 실패로 기록합니다.

## Single-scale dense ablation

기본 `keep-all`은 NVILA-HD processor의 multiscale patch space를 keep-all로 통과시키는 ablation입니다. 일반 SigLIP에 가까운 392px single-scale dense 기준은 wrapper에서 기본으로 함께 실행됩니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/hlvid_limit3_single_scale \
  --limit 3 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --video-resize-longest-edge 720 \
  --video-decode-strategy seek \
  --single-scale-dense-scales 392 \
  --continue-on-error
```

이 모드는 single runner의 `--gazing-mode keep-all-single` alias와 같은 의미입니다. 내부적으로는 `keep-all + --autogaze-target-scales 392`로 정규화되고, `hlvid_single_scale_dense_*` 파일과 `single_scale_dense_comparison` 섹션을 추가합니다. 논문 baseline이 아니라 “HD 모델에서 dense single-scale을 태웠을 때 AutoGaze 대비 얼마나 다른가”를 보기 위한 참고 축입니다. 제외하려면 `--skip-single-scale-dense`를 붙입니다.

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

Qwen/LongVILA/NVILA-Video 등 확장 조합을 같은 manifest row로 비교할 때 사용합니다. Qwen은 이제 기본 wrapper에서 `--plugin-suite qwen`으로 바로 실행할 수 있습니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_limit3 \
  --plugin-suite qwen \
  --plugin-model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3 \
  --num-video-frames 32 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 4 \
  --qwen-vit-chunk-frames 16 \
  --qwen-vit-max-spatial-chunks 4 \
  --qwen-thumbnail-mode append-video \
  --video-resize-longest-edge 448 \
  --autogaze-target-scales 112+224+336+448 \
  --autogaze-target-patch-size 16 \
  --autogaze-tile-size 448 \
  --max-new-tokens 8
```

`--plugin-suite qwen`은 기본적으로 다음 세 모드를 같은 HLVid row에서 실행합니다.

```text
qwen_full_vit
qwen_chunked_vit
qwen_chunked_vit_autogaze_sparse
```

`qwen_chunked_vit_autogaze_sparse`는 AutoGaze checkpoint의 4-scale decoder 제약 때문에 AutoGaze target scale도 4개여야 합니다. wrapper는 값을 생략하면 `--video-resize-longest-edge` 기준으로 `112+224+336+448` 같은 patch16 호환 scale을 자동 주입합니다. 실험 재현성을 위해 CUDA benchmark command에는 명시하는 것을 권장합니다.

세부 mode를 직접 지정하려면 `--plugin-modes`를 사용합니다.

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --plugin-suite custom \
  --plugin-modes qwen_full_vit,qwen_chunked_vit \
  --plugin-model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3
```

내부 runner를 직접 호출해야 할 때만 아래 형태를 사용합니다.

```bash
.venv/bin/python -m repro.plugin_hlvid_benchmark \
  --manifest /data/HLVid/manifest.json \
  --video-root /data/HLVid/videos \
  --output-dir outputs/autogaze_repro/plugin_hlvid_qwen_limit3 \
  --modes qwen_full_vit,qwen_chunked_vit,qwen_chunked_vit_autogaze_sparse \
  --model qwen3-vl=weight/Qwen3-VL-8B-Instruct \
  --limit 3
```

## 주요 산출물

| 파일 | 의미 |
| --- | --- |
| `hlvid_keep_all_predictions.jsonl` | keep-all row별 prediction |
| `hlvid_single_scale_dense_predictions.jsonl` | single-scale dense keep-all row별 prediction |
| `hlvid_autogaze_predictions.jsonl` | AutoGaze row별 prediction |
| `hlvid_keep_all_summary.json` | keep-all accuracy/failure/metric summary |
| `hlvid_single_scale_dense_summary.json` | single-scale dense summary |
| `hlvid_autogaze_summary.json` | AutoGaze accuracy/failure/metric summary |
| `hlvid_autogaze_gain_report.json` | keep-all, single-scale dense, AutoGaze의 score/latency/token/memory 비교 |
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
| latency | Decode/read, Prep rest, Selector input, AutoGaze, Vision input, ViT/vision encoder, LLM/generate, total |
| memory | processor/autogaze/ViT/LLM/overall peak |
| benchmark | 정답 여부, failure category, OOM stage |

AutoGaze on만 돌렸더라도 full/off 예상 patch budget은 summary에 남아야 합니다. 이 값이 있어야 keep-all을 실제로 못 돌린 OOM config에서도 token reduction 근거를 유지할 수 있습니다.

HLVid gain report에서는 `readable_summary.key_metrics_median.latency_ms`를 먼저 보고, 더 세부적인 확인은 `readable_summary.latency_ms_detail_median`을 봅니다. 기본 wrapper 결과는 `keep_all -> single_scale_dense -> autogaze` 순서로 Markdown, chart, aggregate에 표시됩니다. `selector_input_build_ms`는 AutoGaze 전체에서 모델 forward만 뺀 residual이고, `vision_input_build_ms`는 vision wrapper에서 SigLIP/projector를 뺀 residual입니다. 둘 다 실제 측정 timer의 차이로 계산되며, total latency에 별도로 더하지 않습니다.

정답 비교는 `correctness_comparison`을 확인합니다. 기존 top-level `counts`는 `keep_all vs autogaze` 호환 필드이고, 세 모드 비교는 `correctness_comparison.pairwise`의 `keep_all_vs_single_scale_dense`, `single_scale_dense_vs_autogaze`, `keep_all_vs_autogaze`를 봅니다. Markdown report에서는 `Pairwise Correctness Summary`가 이 내용을 표로 보여줍니다.
