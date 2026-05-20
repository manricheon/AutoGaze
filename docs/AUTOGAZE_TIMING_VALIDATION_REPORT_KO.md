# AutoGaze 시간 측정 검증 리포트

## 결론 요약

- 비교 스크립트: `repro.autogaze_timing_compare`
- 기본 subprocess Python: 현재 이 스크립트를 실행한 interpreter. CUDA 머신에서는 보통 `.venv/bin/python -m repro.autogaze_timing_compare ...`로 실행하면 하위 Quick Start / stream-profile / single subprocess도 같은 venv를 쓴다.
- 다른 venv를 강제로 쓰려면 `--python /path/to/.venv/bin/python` 또는 `AUTOGAZE_TIMING_PYTHON=/path/to/.venv/bin/python`을 지정한다.
- 기본 AutoGaze repo path: `external/AutoGaze`
- 기본 weights root: `weights`
- 로컬 Mac timing audit처럼 별도 checkout/weights를 쓰려면 `--autogaze-repo /Users/mrc/myresearch/AutoGaze --weights-root /Users/mrc/myresearch/AutoGaze/weights`를 명시한다.
- 현재 이 Codex 세션에서 target venv의 MPS 상태는 `mps_built=True`, `mps_available=False`다. 따라서 새 MPS 실행은 수행하지 못했고, 로컬 weights와 지정 venv를 쓰는 CPU smoke로 스크립트 동작을 검증했다.
- CPU smoke 결과에서 Quick Start direct는 16프레임 224 입력 기준 raw patch `4,240`, 우리 stream-profile은 NVILA 멀티스케일 `[56,112,196,392]` 기준 raw patch `16,960`으로 4배 workload다. AutoGaze 시간이 더 커 보이는 1차 이유는 이 workload 차이를 같이 봐야 한다.

## 비교 스크립트

기본 실행:

```bash
.venv/bin/python -m repro.autogaze_timing_compare
```

위 명령의 하위 subprocess도 기본적으로 같은 `.venv/bin/python`을 사용한다. 예전처럼 Mac 로컬 AutoGaze venv를 직접 쓰고 싶을 때만 아래처럼 `--python`을 추가한다.

```bash
.venv/bin/python -m repro.autogaze_timing_compare \
  --python /Users/mrc/myresearch/AutoGaze/.venv/bin/python \
  --autogaze-repo /Users/mrc/myresearch/AutoGaze \
  --weights-root /Users/mrc/myresearch/AutoGaze/weights
```

MPS가 정상으로 잡히는 터미널에서는 위 명령이 아래 세 경로를 순서대로 실행한다.

1. Quick Start direct
   - `repro.autogaze_bench`
   - 원본 AutoGaze repo의 `assets/example_input.mp4`
   - `/Users/mrc/myresearch/AutoGaze/weights/AutoGaze`
   - `/Users/mrc/myresearch/AutoGaze/weights/siglip2-base-patch16-224`
   - `frames=16`, `gazing_ratio=0.75`, `task_loss_requirement=0.7`
   - 기본 `batch_size=1`
   - 기본은 AutoGaze-only 측정을 위해 SigLIP를 skip한다. SigLIP까지 같이 재려면 `--quickstart-run-siglip`을 쓴다.

2. Current implementation stream-profile
   - `repro.nvila_runner --mode stream-profile`
   - 같은 비디오와 같은 16프레임
   - `max_tiles_video=1`, `stream_chunk_frames=16`, `num_video_frames_thumbnail=0`
   - `stream_gazing_ratio=0.75`, `task_loss_requirement_tile=0.7`
   - NVILA 스타일 target scales 기본값 `[56,112,196,392]`

3. Current implementation single
   - `repro.nvila_runner --mode single`
   - 같은 비디오와 같은 16프레임
   - `gazing_ratio_tile=0.75`, `task_loss_requirement_tile=0.7`
   - full NVILA processor + `model.generate`
   - `measure_ttft` 활성화

산출물:

```text
outputs/autogaze_repro/timing_compare/quickstart_direct_autogaze.json
outputs/autogaze_repro/timing_compare/current_stream_profile_autogaze.json
outputs/autogaze_repro/timing_compare/current_single_autogaze.json
outputs/autogaze_repro/timing_compare/current_single_autogaze_summary.json
outputs/autogaze_repro/timing_compare/quickstart_vs_current_summary.json
outputs/autogaze_repro/timing_compare/quickstart_vs_current_report.md
```

## AutoGaze latency에 영향을 주는 옵션

| 옵션 | Quick Start direct | NVILA stream/single | latency 영향 |
| --- | --- | --- | --- |
| batch size | `--quickstart-batch-size`, 기본 `1` | 직접 대응 없음 | direct AutoGaze에 여러 비디오를 한 번에 넣는다. batch 1은 단일 클립 최저 latency 기준이다. batch를 키우면 총 wall time은 늘 수 있지만 samples/sec throughput은 좋아질 수 있다. |
| AutoGaze tile batch | 없음 | `--max-batch-size-autogaze`, 기본 `16` | NVILA에서 spatial tile sequence들을 몇 개씩 AutoGaze에 넣을지 결정한다. tile이 많을 때 GPU launch/throughput에는 유리할 수 있지만 peak memory가 늘고 단일 1-tile latency를 줄인다는 보장은 없다. |
| gazing ratio | `--gazing-ratio` | `--gazing-ratio-tile`, stream은 `--stream-gazing-ratio` | 더 큰 ratio는 더 많은 patch를 탐색/선택할 수 있어서 AutoGaze forward와 후속 encoder workload가 커질 수 있다. Quick Start는 `0.75`, NVILA 기본은 `[0.2]+[0.06]*15`라서 반드시 맞춰야 한다. |
| task loss threshold | `--task-loss-requirement` | `--task-loss-requirement-tile` | threshold가 낮거나 조건이 더 빡빡하면 더 오래 gaze할 수 있다. |
| target scales | `--autogaze-target-scales`가 comparison에서 direct에는 `--target-scales`로 전달됨 | `--autogaze-target-scales` | `[56,112,196,392]`처럼 multiscale이면 patch budget이 크게 증가한다. Quick Start 기본 224 단일 입력과 직접 비교하면 안 된다. |
| target patch size | `--autogaze-target-patch-size`가 direct에는 `--target-patch-size`로 전달됨 | `--autogaze-target-patch-size` | scale/patch 조합이 patch positions per frame을 결정한다. |
| sampled frames | `--frames` | `--num-video-frames` | 거의 선형적으로 AutoGaze 입력량에 영향을 준다. |
| tiles | 없음 | `--max-tiles-video` | 고해상도에서 tile sequence 수가 늘면 AutoGaze 실행 횟수/입력량이 늘어난다. |
| dtype/device | `--dtype`, `--device` | `--dtype`, `--device` | CUDA fp16/bf16, MPS float32 등 backend 차이가 크다. |

질문한 batch size 영향에 대한 답은 “그렇다, 하지만 의미가 다르다”이다. Quick Start의 batch size 1은 단일 16-frame clip 하나를 넣는 기준이다. batch size를 키우면 한 번의 AutoGaze call에 여러 clip을 넣으므로 total latency는 보통 증가하지만, GPU에서는 samples/sec throughput이 좋아질 수 있다. 반면 NVILA의 `max_batch_size_autogaze`는 사용자 비디오 batch가 아니라 tile sequence batch다. 4K/다중 tile에서는 유의미하지만, 1-tile/16-frame smoke에서는 크게 줄어들 여지가 작다.

## 속도 벤치마크 읽는 순서

AutoGaze 자체가 빠른지/느린지를 보려면 아래 순서로 봐야 한다. 특히 H100에서 “Quick Start는 3초, `nvila_runner`는 300ms”처럼 보이면 거의 항상 측정 경계나 gaze policy가 다르다.

1. 입력량 확인
   - `raw_patch_budget`
   - `autogaze_input_patch_tokens`
   - `frames`, `target_scales`, `target_patch_size`, `max_tiles_video`
   - 이 값이 다르면 같은 AutoGaze latency로 보면 안 된다.

2. 선택량 확인
   - `selected_non_padded_patches`
   - `autogaze_selected_patch_tokens`
   - `total_gaze_slots`
   - token reduction ratio
   - 여기서 실제 90%/95% 감소가 나왔는지 확인한다.

3. AutoGaze forward만 비교
   - Quick Start: `latency_ms.autogaze.median`
   - stream-profile: `timing_ms.tile_autogaze_forward`
   - single: `autogaze_model_forward_ms`
   - 이 세 값이 “모델 forward” 기준의 1차 비교다.

4. AutoGaze 주변 overhead 확인
   - stream-profile: `tile_autogaze_tensorize`, `spatial_tile_build`, `pre_llm_stream_total_measured`
   - single: `autogaze_total_ms`
   - 여기는 tensorization, processor hook, gaze-info 정리 비용이 섞인다.

5. end-to-end 영향 확인
   - single: `video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms = total_ms`
   - `generate_ms`에는 vision encoder/projector/LLM generate 경로가 들어간다.

## AutoGaze 정책 sweep

`gazing_ratio`와 `task_loss_requirement`가 latency에 주는 영향을 바로 보기 위해 comparison script에 sweep 모드를 추가했다.

H100 권장 실행:

```bash
.venv/bin/python -m repro.autogaze_timing_compare \
  --device cuda \
  --dtype float16 \
  --frames 16 \
  --thumbnail-frames 0 \
  --max-tiles-video 1 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --quickstart-batch-size 1 \
  --gazing-ratio-sweep 0.06,0.1,0.2,0.5,0.75,1.0 \
  --task-loss-sweep 0.6,0.7 \
  --autogaze-target-scales 56+112+196+392 \
  --autogaze-target-patch-size 14 \
  --warmup 3 \
  --repeat 10 \
  --output-dir outputs/autogaze_repro/h100_autogaze_policy_sweep
```

산출물:

```text
outputs/autogaze_repro/h100_autogaze_policy_sweep/autogaze_policy_sweep_summary.json
outputs/autogaze_repro/h100_autogaze_policy_sweep/autogaze_policy_sweep_report.md
outputs/autogaze_repro/h100_autogaze_policy_sweep/gazing_0p75__loss_0p7/quickstart_vs_current_summary.json
outputs/autogaze_repro/h100_autogaze_policy_sweep/gazing_0p75__loss_0p7/quickstart_vs_current_report.md
```

sweep summary에서 먼저 볼 핵심 필드:

| Field | 의미 |
| --- | --- |
| `quickstart_autogaze_ms` | 원본 Quick Start 방식 direct AutoGaze forward |
| `stream_autogaze_forward_ms` | 현재 stream-profile의 AutoGaze forward |
| `single_autogaze_forward_ms` | full NVILA processor hook 내부 AutoGaze forward |
| `single_autogaze_total_ms` | full runner에서 gaze-info 생성 전체 비용 |
| `quickstart_raw_patch_budget`, `stream_raw_patch_budget`, `single_raw_patch_budget` | 비교 입력 patch budget |
| `quickstart_selected_patches`, `stream_selected_patches`, `single_selected_patches` | 실제 선택 patch 수 |
| `quickstart_token_reduction_ratio`, `stream_token_reduction_ratio`, `single_token_reduction_ratio` | 감소율 |

dry-run으로 명령만 확인:

```bash
.venv/bin/python -m repro.autogaze_timing_compare \
  --dry-run \
  --skip-single \
  --gazing-ratio-sweep 0.2,0.75 \
  --task-loss-sweep 0.6,0.7 \
  --output-dir outputs/autogaze_repro/timing_compare_dry_policy_sweep
```

## 측정 경계 정의

| Level | Metric | Includes | Excludes | Purpose |
| --- | --- | --- | --- | --- |
| L0 | `latency_ms.autogaze.median` | direct AutoGaze model call | NVILA decode/tile pipeline | Quick Start 기준 sanity check |
| L1 | `tile_autogaze_forward` | stream-profile AutoGaze model forward over tile sequences | decode/tile/tensorize | 현재 구현의 AutoGaze forward 비교 |
| L2 | `pre_llm_stream_total_measured` | decode, frame conversion, tiling, tensorize, AutoGaze, optional SigLIP | NVILA projector/LLM | pre-LLM 전체 overhead |
| L3 | `autogaze_model_forward_ms` | `nvila_runner --mode single`의 `_run_autogaze_batched` hook | parent gaze-info work | Quick Start direct와 가장 먼저 비교할 full-runner AutoGaze forward |
| L4 | `autogaze_total_ms` | `nvila_runner --mode single`의 `_get_gazing_info_from_videos` hook | video decode, generate | full runner에서 실제 AutoGaze processor overhead |
| L5 | `generate_ms` | full NVILA generate path | processor preprocessing | LLM 포함 end-to-end |

`tile_autogaze_forward`와 `pre_llm_stream_total_measured`는 parent-child 관계다. 둘을 더하면 중복 계산이다.
마찬가지로 single mode의 `autogaze_model_forward_ms`와 `autogaze_total_ms`도 parent-child 관계다. 3초 vs 300ms 의심을 볼 때는 먼저 Quick Start direct `latency_ms.autogaze.median`과 single `autogaze_model_forward_ms`를 비교하고, 그 다음 single `autogaze_total_ms`를 확인한다.

## 코드 위치

| 항목 | 코드 |
| --- | --- |
| direct AutoGaze timer | [repro/autogaze_bench.py](</Users/mrc/Documents/New project/repro/autogaze_bench.py:151>) |
| target device sync timer | [repro/common.py](</Users/mrc/Documents/New project/repro/common.py:61>) |
| stream-profile stage timer | [repro/nvila_runner.py](</Users/mrc/Documents/New project/repro/nvila_runner.py:218>) |
| stream-profile AutoGaze forward | [repro/nvila_runner.py](</Users/mrc/Documents/New project/repro/nvila_runner.py:3712>) |
| stream-profile output timing fields | [repro/nvila_runner.py](</Users/mrc/Documents/New project/repro/nvila_runner.py:4323>) |
| comparison script | [repro/autogaze_timing_compare.py](</Users/mrc/Documents/New project/repro/autogaze_timing_compare.py:1>) |

## CPU Smoke 결과

실행:

```bash
.venv/bin/python -m repro.autogaze_timing_compare --device cpu --no-require-mps --warmup 0 --repeat 1 --output-dir outputs/autogaze_repro/timing_compare_cpu_smoke
```

결과 파일:

```text
outputs/autogaze_repro/timing_compare_cpu_smoke/quickstart_vs_current_summary.json
outputs/autogaze_repro/timing_compare_cpu_smoke/quickstart_vs_current_report.md
```

핵심 수치:

| Path | Device | Frames | Raw patches | Selected patches | AutoGaze forward ms | Total measured ms | Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quick Start direct | cpu | 16 | 4,240 | 213 | 718.64 |  | 19.91 |
| Current stream-profile | cpu | 16 | 16,960 | 476 | 2,137.22 | 2,233.95 | 35.63 |

비율:

| Metric | Ratio |
| --- | ---: |
| stream AutoGaze forward / Quick Start AutoGaze | 2.97 |
| stream total / Quick Start AutoGaze | 3.11 |
| stream raw patches / Quick Start raw patches | 4.00 |
| stream selected patches / Quick Start selected patches | 2.23 |

해석: CPU smoke에서는 current stream-profile AutoGaze forward가 direct Quick Start보다 약 2.97배 느리지만, raw patch workload는 4배다. 따라서 단순 wall time만 보면 느려 보여도, workload-normalized 관점에서는 측정 경계가 크게 어긋났다고 보기 어렵다.

## MPS 재실행 명령

MPS가 실제로 `True`로 나오는 터미널에서:

```bash
/Users/mrc/myresearch/AutoGaze/.venv/bin/python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

그 다음:

```bash
.venv/bin/python -m repro.autogaze_timing_compare --device mps --dtype float32 --warmup 1 --repeat 3 --output-dir outputs/autogaze_repro/timing_compare_mps
```

SigLIP stream timing까지 같이 보고 싶으면:

```bash
.venv/bin/python -m repro.autogaze_timing_compare --device mps --dtype float32 --warmup 1 --repeat 3 --stream-run-siglip --stream-siglip-mode both --output-dir outputs/autogaze_repro/timing_compare_mps_siglip
```

MPS/로컬에서 NVILA single이 너무 무겁거나 CUDA 없이 smoke만 보고 싶으면:

```bash
.venv/bin/python -m repro.autogaze_timing_compare --device cpu --no-require-mps --skip-single --warmup 0 --repeat 1 --output-dir outputs/autogaze_repro/timing_compare_cpu_smoke
```

## 다음 조치

- target venv의 MPS availability가 true가 되는 실행 환경에서 위 MPS 명령을 다시 실행한다.
- raw patch ratio가 1에 가까운 비교도 추가하려면 current stream-profile 쪽 target scales를 별도로 맞추는 실험을 추가한다.
- full NVILA LLM까지 포함한 시간은 stream-profile이 아니라 `nvila_runner --mode single` 또는 HLVid benchmark 경로에서 측정한다.

## H100 300ms vs 3s 불일치 체크

현재 `nvila_runner`의 NVILA processor 기본 AutoGaze 정책은 원본 Quick Start와 다르다.

| Path | Default gazing ratio | task loss | 의미 |
| --- | --- | ---: | --- |
| Original Quick Start | `0.75` | `0.7` | 프레임별 최대 75% 패치까지 gaze 가능 |
| NVILA processor default | `[0.2] + [0.06] * 15` | 기본 `0.6` | 첫 프레임은 20%, 이후 프레임은 6%로 훨씬 작은 탐색 예산 |

그래서 H100에서 direct AutoGaze가 3초이고 `nvila_runner --mode single`의 `autogaze_model_forward_ms`가 300ms라면, 가장 먼저 봐야 할 것은 두 결과의 `autogaze_runtime_config`, `autogaze_input_patch_tokens`, `autogaze_selected_patch_tokens`, `total_gaze_slots`다. 특히 `gazing_ratio_tile`이 다르면 AutoGaze forward 시간은 apples-to-apples가 아니다.

Quick Start direct, stream-profile, single을 한 번에 비교하는 권장 명령:

```bash
.venv/bin/python -m repro.autogaze_timing_compare \
  --device cuda \
  --dtype float16 \
  --gazing-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --warmup 3 \
  --repeat 10 \
  --output-dir outputs/autogaze_repro/h100_quickstart_vs_stream_vs_single
```

같은 정책으로 맞춘 NVILA single 재실행 예:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --dtype float16 \
  --model-path /Users/mrc/myresearch/AutoGaze/weights/NVILA-8B-HD-Video \
  --autogaze-repo /Users/mrc/myresearch/AutoGaze \
  --autogaze-model /Users/mrc/myresearch/AutoGaze/weights/AutoGaze \
  --video /Users/mrc/myresearch/AutoGaze/assets/example_input.mp4 \
  --gazing-mode autogaze \
  --gazing-ratio-tile 0.75 \
  --task-loss-requirement-tile 0.7 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --max-batch-size-autogaze 16 \
  --max-new-tokens 1 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/h100_quickstart_ratio_nvila_single.json \
  --summary-json outputs/autogaze_repro/h100_quickstart_ratio_nvila_single_summary.json \
  --print-summary
```

논문/NVILA 기본 정책 그대로 재실행 예:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --dtype float16 \
  --model-path /Users/mrc/myresearch/AutoGaze/weights/NVILA-8B-HD-Video \
  --autogaze-repo /Users/mrc/myresearch/AutoGaze \
  --autogaze-model /Users/mrc/myresearch/AutoGaze/weights/AutoGaze \
  --video /Users/mrc/myresearch/AutoGaze/assets/example_input.mp4 \
  --gazing-mode autogaze \
  --task-loss-requirement-tile 0.6 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --max-batch-size-autogaze 16 \
  --max-new-tokens 1 \
  --measure-ttft \
  --output-json outputs/autogaze_repro/h100_nvila_default_ratio_single.json \
  --summary-json outputs/autogaze_repro/h100_nvila_default_ratio_single_summary.json \
  --print-summary
```

두 결과에서 비교할 필드:

```text
autogaze_runtime_config.gazing_ratio_tile
autogaze_runtime_config.task_loss_requirement_tile
token_metrics.autogaze_input_patch_tokens
token_metrics.autogaze_selected_patch_tokens
token_metrics.autogaze_patch_reduction_ratio
autogaze_model_forward_ms
autogaze_total_ms
stage_timings_ms.processor.autogaze_forward_batched
```
