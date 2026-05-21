# Streaming Pipeline / H100 Preflight 추천 설정

이 문서는 긴 4K 비디오와 HLVid benchmark를 CUDA/H100에서 돌릴 때 decode, resize, chunk, AutoGaze, ViT/LLM 메모리 위험을 어떻게 나누어 볼지 정리합니다.

긴 이전 문서는 `docs/temp/STREAMING_PIPELINE_CONFIG_RECOMMENDATIONS_KO_FULL_2026-05-21.md`에 보존했습니다.

## 왜 streaming/profile이 필요한가

HLVid처럼 5분 안팎의 고해상도 비디오는 전체 frame을 순차 decode한 뒤 버리는 방식이면 CPU decode 시간이 커지고, 샘플링된 frame을 한 번에 tensor로 쌓으면 메모리도 커집니다.

현재 정책은 다음을 목표로 합니다.

1. metadata로 전체 frame 수와 duration을 먼저 읽는다.
2. target sample index만 seek/decode한다.
3. chunk 단위로 resize/tile/tensorize/AutoGaze를 수행한다.
4. 처리된 chunk의 중간 tensor는 가능한 빨리 해제한다.
5. 단, full MLLM generate는 모델 특성상 최종 visual sequence를 모아야 할 수 있다.

## Pipeline 도식

```text
Video file
  -> metadata probe
       fps / frame_count / duration / width / height
  -> sample index plan
       1024f or 256f or 128f ...
  -> chunk loop
       seek/decode selected frames only
       resize shortest edge
       tile to max_tiles_video
       build thumbnail frames
       AutoGaze selection
       optional vision encoder profiling
       release chunk tensors
  -> final visual packing
  -> MLLM prefill/generate
  -> JSON + Markdown + aggregate report
```

## 주요 옵션

| 옵션 | 의미 |
| --- | --- |
| `--num-video-frames` | 본 frame 샘플 수 |
| `--num-video-frames-thumbnail` | thumbnail frame 샘플 수 |
| `--stream-chunk-frames` | streaming/profile에서 한 번에 decode/처리할 frame 수 |
| `--max-batch-size-autogaze` | AutoGaze 모델 batch 처리 단위 |
| `--max-tiles-video` | 한 frame을 최대 몇 tile로 자를지 |
| `--video-resize-shortest-edge` | tile 전 비디오 resize 기준 |
| `--thumbnail-resize-shortest-edge` | thumbnail resize 기준 |
| `--h100-preflight` | benchmark 전 token/memory risk 추정 |

## 추천 설정

| 목적 | 권장값 |
| --- | --- |
| 로컬 MPS smoke | `--num-video-frames 16 --num-video-frames-thumbnail 8 --max-tiles-video 1 --video-resize-shortest-edge 384` |
| CUDA smoke | `--num-video-frames 64 --num-video-frames-thumbnail 32 --max-tiles-video 4 --video-resize-shortest-edge 720` |
| HLVid balanced | `--num-video-frames 128 --num-video-frames-thumbnail 64 --max-tiles-video 8 --video-resize-shortest-edge 720` |
| H100 wider sweep | `--num-video-frames 256 --num-video-frames-thumbnail 128 --max-tiles-video 16 --video-resize-shortest-edge 1080` |
| paper-facing HD stress | `--num-video-frames 1024 --num-video-frames-thumbnail 512 --max-tiles-video 48 --video-resize-shortest-edge none` 전 반드시 preflight |

H100 80GB라도 full HD keep-all은 LLM SDPA/KV cache에서 OOM이 날 수 있습니다. AutoGaze가 90-95% patch를 줄여도 projector/visual packing/LLM context가 어떻게 구성되는지 확인해야 합니다.

## 명령 예시

Stream profile:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-model weights/autogaze \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 8 \
  --max-tiles-video 8 \
  --video-resize-shortest-edge 720 \
  --output-json outputs/autogaze_repro/stream_profile_128f.json
```

HLVid preflight:

```bash
.venv/bin/python scripts/run_hlvid_folder_benchmark.py \
  --dataset-dir /data/HLVid \
  --video-root /data/HLVid/videos \
  --limit 0 \
  --h100-preflight \
  --h100-budget-gib 70 \
  --output-dir outputs/autogaze_repro/hlvid_h100_preflight
```

Sweep:

```bash
.venv/bin/python -m repro.stream_profile_sweep \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --frames 64,128,256 \
  --thumbnail-frames 32,64,128 \
  --resize-shortest-edges 720,1080 \
  --max-tiles-video 4,8,16 \
  --output-dir outputs/autogaze_repro/stream_sweep
```

## 로그에서 볼 항목

| 영역 | 핵심 필드 |
| --- | --- |
| decode | `video_decode_sampling_ms`, sample frame count, original resolution |
| preprocess | resize resolution, tile count/frame, tensorize time |
| AutoGaze | candidate patch, multiscale patch, selected patch, `autogaze_total_ms` |
| ViT | full/off expected token, encoder input token, latency, memory |
| LLM | visual token, context length, TTFT, generate time, peak memory |
| failure | `failure.kind`, `failure.stage`, OOM 여부 |

## OOM 대응 우선순위

1. `--num-video-frames`를 줄인다.
2. `--num-video-frames-thumbnail`를 줄인다.
3. `--video-resize-shortest-edge`를 낮춘다.
4. `--max-tiles-video`를 낮춘다.
5. `--max-batch-size-autogaze`를 낮춘다.
6. keep-all 대신 AutoGaze를 먼저 확인한다.
7. 그래도 LLM SDPA/KV cache에서 터지면 context/visual token 수 기준으로 모델 또는 generation 정책을 줄인다.
