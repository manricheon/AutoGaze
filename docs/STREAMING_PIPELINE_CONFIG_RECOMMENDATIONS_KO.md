# 스트리밍 파이프라인 추천 설정

목표는 두 가지입니다.

1. 전처리/AutoGaze 단계에서 sampled frame, tile image, AutoGaze tensor를 한 번에 만들지 않아 OOM을 피한다.
2. AutoGaze로 줄어드는 patch/token 수가 downstream SigLIP/MLLM latency를 이길 만큼 충분한 조합을 찾는다.

현재 로컬은 MPS라서 full NVILA-8B MLLM 최종 latency는 여기서 확정할 수 없습니다. 대신 `stream-profile`로 decode, resize, tiling, AutoGaze latency와 patch/token 감소를 실측했고, CUDA 머신에서 같은 matrix를 이어서 돌릴 수 있게 `repro.stream_profile_sweep`와 `configs/repro/streaming_pipeline_profiles.yaml`을 추가했습니다.

## 중요한 결론

- 16:9 영상에서 `--video-resize-shortest-edge`만 낮춰도 `--max-tiles-video 48`이면 여전히 45개 tile이 잡힙니다. 4K latency/OOM을 줄이는 1차 레버는 `max_tiles_video`입니다.
- MPS에서는 `max-batch-size-autogaze=1`이 유리했습니다. 720p/4-tile/16-frame에서 batch 1은 약 11.3초, batch 4는 약 26.3초였습니다.
- CUDA에서는 기존 기본값인 AutoGaze batch 16, SigLIP batch 32부터 시작하는 것이 맞습니다. 단, full NVILA가 느리면 batch보다 먼저 tile/frame 수를 줄여야 합니다.
- thumbnail은 현재 keep-all이라 total patch reduction을 희석합니다. 그래서 리더 설명 시 tile-only reduction과 total reduction을 같이 보여줘야 합니다.
- 4K HLVid 5분 예시는 16프레임만 샘플링해도 끝 프레임까지 decode scan이 필요해 CPU decode가 약 54-68초였습니다. 긴 비디오에서는 decode seeking/샘플링 최적화도 별도 병목입니다.

## 로컬 실측 요약

| 입력 | 설정 | 총 pre-LLM | AutoGaze forward | tile 감소 | total 감소 | keep-all LLM token | AutoGaze LLM lower-bound | peak raw/tile/AG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 896 square | 16f, 8thumb, 1tile, batch4 | 4.64s | 4.27s | 42.6x | 2.87x | 2,832 | 987 | 38.5/7.4/29.5 MB |
| 448 resized | 64f, 32thumb, 1tile, batch4 | 20.48s | 19.12s | 52.8x | 2.89x | 11,328 | 3,912 | 9.6/7.4/29.5 MB |
| 720 resized | 16f, 8thumb, 4tile, batch1 | 11.27s | 10.67s | 58.5x | 7.92x | 8,496 | 1,072 | 24.9/29.5/29.5 MB |
| 720 resized | 16f, 8thumb, 4tile, batch4 | 26.31s | 25.61s | 57.5x | 7.90x | 8,496 | 1,074 | 24.9/29.5/118.0 MB |
| 1080p 16:9 | 16f, 8thumb, 8tile, batch1 | 42.96s | 38.94s | 15.5x | 8.35x | 16,032 | 1,918 | 99.5/59.0/29.5 MB |
| HLVid 4K keep-all | 16f, 8thumb, 1tile | 55.06s | n/a | 1.0x | 1.0x | 2,832 | n/a | 398.1/7.4/0 MB |
| HLVid 4K keep-all | 16f, 8thumb, 8tile | 56.86s | n/a | 1.0x | 1.0x | 16,032 | n/a | 398.1/59.0/0 MB |
| HLVid 4K keep-all | 16f, 8thumb, 45tile | 70.91s | n/a | 1.0x | 1.0x | 85,744 | n/a | 398.1/331.9/0 MB |

실측 파일:

- `outputs/autogaze_repro/stream_profile_security_autogaze_16f_mps.json`
- `outputs/autogaze_repro/stream_sweep/fast_448p_1tile_64f_autogaze.json`
- `outputs/autogaze_repro/security_720p_4tile_16f_batch1_mps.json`
- `outputs/autogaze_repro/security_720p_4tile_16f_batch4_mps.json`
- `outputs/autogaze_repro/bbb_1080p_16f_8tile_batch1_mps.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_16f_8tile_cpu.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_16f_45tile_cpu.json`

## 추천 조합

### 로컬 MPS

MPS는 기능 확인과 patch/token 감소율 확인용으로만 보세요. 성능 claim은 CUDA에서 다시 잡는 것이 안전합니다.

```bash
# 가장 빠른 path check
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device mps \
  --stream-dtype float32 \
  --video inputs/hf_space_autogaze/security.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 64 \
  --num-video-frames-thumbnail 32 \
  --max-tiles-video 1 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --max-batch-size-siglip 16 \
  --video-resize-shortest-edge 448
```

```bash
# MPS에서 tile 증가 효과를 볼 수 있는 상한선에 가까운 probe
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device mps \
  --stream-dtype float32 \
  --video inputs/hf_space_autogaze/security.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 8 \
  --max-tiles-video 4 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 1 \
  --max-batch-size-siglip 16 \
  --video-resize-shortest-edge 720
```

### CUDA latency-first

full NVILA가 AutoGaze overhead 때문에 느리다면 이 조합부터 확인합니다. tile count를 4-8로 제한해서 AutoGaze와 SigLIP workload를 모두 낮춥니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --stream-dtype float16 \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --video-resize-shortest-edge 1080
```

### CUDA balanced quality

latency-first가 통과하면 tile을 15개 수준으로 늘립니다. 16:9에서는 `max_tiles_video=16`이 보통 `5x3=15` tile이 됩니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --stream-dtype float16 \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 16 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --video-resize-shortest-edge 1440
```

### CUDA paper-facing probe

1024프레임 전에 256프레임으로 먼저 봅니다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --stream-dtype float16 \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 256 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 32 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --video-resize-shortest-edge 1440
```

### CUDA paper stress

이건 바로 성능 claim으로 쓰지 말고 stress test로만 시작하세요.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --stream-dtype float16 \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --gazing-mode autogaze \
  --num-video-frames 1024 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32
```

## 자동 sweep

후보 command와 estimate만 만들려면:

```bash
.venv/bin/python -m repro.stream_profile_sweep \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --device cuda \
  --stream-dtype float16 \
  --gazing-mode autogaze \
  --summary-json outputs/autogaze_repro/stream_sweep_hlvid_cuda_dry.json \
  --summary-csv outputs/autogaze_repro/stream_sweep_hlvid_cuda_dry.csv
```

실제로 실행하려면 `--run`을 붙입니다. 시간이 길 수 있으니 처음에는 `--include fast_720`이나 `--include balanced_1080`처럼 후보를 좁혀 실행하세요.

```bash
.venv/bin/python -m repro.stream_profile_sweep \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --device cuda \
  --stream-dtype float16 \
  --gazing-mode autogaze \
  --include balanced_720 \
  --include balanced_1080 \
  --run \
  --timeout-seconds 1800 \
  --summary-json outputs/autogaze_repro/stream_sweep_hlvid_cuda_run.json \
  --summary-csv outputs/autogaze_repro/stream_sweep_hlvid_cuda_run.csv
```

## full NVILA 비교 순서

stream-profile에서 후보를 줄인 뒤 full NVILA는 같은 조합으로 `--mode single` 또는 `--mode hlvid`를 돌립니다.

1. `stream-profile keep-all`: decode/tiling memory와 keep-all token budget 확인
2. `stream-profile autogaze`: AutoGaze forward latency와 patch/token reduction 확인
3. `single keep-all`: full NVILA baseline이 context/OOM 없이 가능한지 확인
4. `single autogaze`: SigLIP/Projector/LLM stage time이 줄어드는지 확인

리더 설득용 지표는 최소 아래를 같이 보고하세요.

- `timing_ms.tile_autogaze_forward`
- `token_metrics.encoder_tile_token_reduction_ratio`
- `token_metrics.encoder_token_reduction_ratio`
- `token_metrics.llm_keep_all_visual_tokens_estimated`
- `token_metrics.llm_autogaze_visual_tokens_lower_bound_estimated`
- full NVILA에서는 `result.siglip_vision_ms`, `result.mm_projector_ms`, `result.llm_forward_ms`, `result.ttft_ms`
