# 스트리밍 파이프라인 추천 설정

목표는 두 가지입니다.

1. 전처리/AutoGaze 단계에서 sampled frame, tile image, AutoGaze tensor를 한 번에 만들지 않아 OOM을 피한다.
2. AutoGaze로 줄어드는 patch/token 수가 downstream SigLIP/MLLM latency를 이길 만큼 충분한 조합을 찾는다.

현재 로컬은 MPS라서 full NVILA-8B MLLM 최종 latency는 여기서 확정할 수 없습니다. 대신 `stream-profile`로 decode, resize, tiling, AutoGaze latency, 선택적 SigLIP vision forward, patch/token 감소를 실측했고, CUDA 머신에서 같은 matrix를 이어서 돌릴 수 있게 `repro.stream_profile_sweep`와 `configs/repro/streaming_pipeline_profiles.yaml`을 추가했습니다.

## 중요한 결론

- 16:9 영상에서 `--video-resize-shortest-edge`만 낮춰도 `--max-tiles-video 48`이면 여전히 45개 tile이 잡힙니다. 4K latency/OOM을 줄이는 1차 레버는 `max_tiles_video`입니다.
- MPS에서는 `max-batch-size-autogaze=1`이 유리했습니다. 720p/4-tile/16-frame에서 batch 1은 약 11.3초, batch 4는 약 26.3초였습니다.
- CUDA에서는 기존 기본값인 AutoGaze batch 16, SigLIP batch 32부터 시작하는 것이 맞습니다. 단, full NVILA가 느리면 batch보다 먼저 tile/frame 수를 줄여야 합니다.
- thumbnail은 현재 keep-all이라 total patch reduction을 희석합니다. 그래서 리더 설명 시 tile-only reduction과 total reduction을 같이 보여줘야 합니다.
- `--stream-run-siglip`을 켜면 AutoGaze가 만든 gazing_info를 custom SigLIP vision tower에 바로 넣어 `siglip_gazed_forward`와 선택적 `siglip_keep_all_forward`를 chunk 단위로 측정합니다. projector/LLM은 여전히 full NVILA `single`/`hlvid`에서 확인해야 합니다.
- `google/siglip2-base-patch16-224`로 MPS smoke를 돌릴 때는 `--autogaze-target-scales 32+64+112+224 --autogaze-target-patch-size 16`을 같이 써야 patch 위치가 맞습니다. NVILA-HD에서는 AutoGaze target coordinate의 patch16과 SigLIP vision tower의 patch14가 공존하므로, 두 값을 리포트에서 분리해서 봐야 합니다.
- 4K HLVid 5분 예시는 `--stream-decode-strategy seek`를 써야 합니다. 기존 scan 방식은 16프레임만 샘플링해도 끝 프레임까지 8992프레임을 디코드해서 CPU decode가 약 54-68초였습니다. seek 방식은 packet-level keyframe index를 먼저 읽고 필요한 target frame 근처 keyframe부터만 디코드해서 16프레임 기준 decode 관련 시간이 약 0.94초, 128프레임 기준 약 5.73초였습니다.

## 동작 파이프라인 그림

### 모델/패치 기준

full NVILA-HD-Video + AutoGaze 기준으로 먼저 생각하면 됩니다.

```text
NVILA/AutoGaze target coordinate aligned to SigLIP
  AutoGaze target patch size  = 14
  target scales               = 56 + 112 + 196 + 392
  positions per frame         = (56//14)^2 + (112//14)^2 + (196//14)^2 + (392//14)^2
                              = 4^2 + 8^2 + 14^2 + 28^2
                              = 16 + 64 + 196 + 784
                              = 1060 positions

NVILA/SigLIP vision tower embedding coordinate
  tile image size             = 392 x 392
  vision scales               = 56 + 112 + 196 + 392
  vision patch size           = 14
  embeddings per frame        = 1060 embeddings
  TokenShuffle estimate       = ceil(selected embeddings / 9) visual tokens
```

MPS에서 `google/siglip2-base-patch16-224`로 custom SigLIP smoke를 돌릴 때는 patch 위치가 달라서 별도 probe 설정을 씁니다.

```text
SigLIP patch16 smoke target
  target scales               = 32 + 64 + 112 + 224
  vision patch size           = 16
  patches per frame/sequence  = 2^2 + 4^2 + 7^2 + 14^2
                              = 4 + 16 + 49 + 196
                              = 265 patches
```

### 전체 logical pipeline

```text
Input video
  F source frames, W x H
        |
        | uniform sample over full video
        v
  N sampled video frames                 M thumbnail frames
        |                                      |
        | resize input video if requested      | resize to 392 x 392
        v                                      v
  dynamic spatial tiling                 thumbnail path
  S tiles per sampled frame              currently keep-all
        |
        | each tile sequence = one spatial tile across C temporal frames
        v
  tile sequences
  count = ceil(N / C) temporal chunks * S spatial tiles
        |
        | AutoGaze predicts selected patch positions
        v
  gazing_info per tile sequence
        |
        | custom SigLIP vision tower can consume gazing_info
        v
  reduced visual hidden states
        |
        | TokenShuffle / projector / LLM input assembly
        v
  NVILA MLLM prefill + generation
```

현재 `stream-profile`에서 직접 stream 처리/측정하는 경계는 아래와 같습니다.

```text
STREAMABLE IN THIS BRANCH
  decode scan
    -> sampled frame to PIL
      -> optional resize
        -> chunk spatial tiling
          -> AutoGaze tensorize
            -> AutoGaze forward
              -> optional custom SigLIP gazed/keep-all forward

COLLECTED FOR FULL NVILA GENERATION
  final visual token sequence
    -> projector
      -> LLM prefill/generation
```

즉, AutoGaze와 optional SigLIP까지는 chunk 단위로 재고 버릴 수 있습니다. 하지만 public NVILA generation path는 최종 visual token sequence를 모아 LLM에 넣으므로 projector/LLM의 실제 시간과 peak memory는 `single`/`hlvid`에서 확인해야 합니다.

### HLVid 4K decode scheduling

HLVid처럼 긴 seekable MP4는 scan decode를 쓰면 안 됩니다. 전체 frame 수는 stream metadata에서 얻고, sample index를 timestamp로 바꾼 뒤 keyframe 단위로 묶어서 디코드합니다.

```text
HLVid example metadata
  frames          = 8992
  fps             = 30
  time_base       = 1 / 15360
  pts per frame   = 512
  keyframe period = about 12 frames

Uniform sample target frames, 16f
  [0, 599, 1199, ..., 8991]

Old scan strategy
  decode frame 0
  decode frame 1
  ...
  decode frame 8991
  decoded frames = 8992

Seek strategy
  scan packets only to build keyframe index
  target 599  -> previous keyframe 588 -> decode 588..599
  target 1199 -> previous keyframe 1188 -> decode 1188..1199
  ...
  target 8991 -> previous keyframe 8988 -> decode 8988..8991
  decoded frames = 124 for 16 sampled frames
```

1024프레임처럼 sample target이 촘촘하면 target마다 seek하지 않고 같은 previous keyframe을 공유하는 target들을 한 그룹으로 묶습니다.

```text
Keyframe-grouped seek
  keyframe K0 -> targets [t0, t1, ...] until next keyframe bucket
  seek once to K0
  decode K0..max(targets)
  collect only target frames

This avoids:
  repeated seek to the same GOP
  decoding all frames between far-apart target samples
```

결과 JSON에서 decode 전략은 아래 필드로 확인합니다.

```text
sampling.decode_strategy
sampling.decode_frames_read
sampling.decode_seek_groups
sampling.decode_keyframes_indexed
sampling.decode_packets_scanned_for_keyframes

timing_ms.video_keyframe_index_scan
timing_ms.video_seek
timing_ms.video_decode_seek
timing_ms.video_decode_scan
```

`seek`는 local/seekable MP4와 정상적인 `fps/time_base/pts` metadata가 있을 때 쓰는 경로입니다. 원격 스트림이 range seek를 지원하지 않거나 variable-frame-rate 영상에서 frame index와 PTS 매핑이 불안정하면 `scan`으로 되돌려 확인하세요.

### 스트리밍/배치 동작

`N=128`, `C=16`, `S=45`, `max_batch_size_autogaze=16`이면 실제 작업은 이렇게 나뉩니다.

```text
Video-level request
  num_video_frames N = 128
  stream_chunk_frames C = 16
  spatial tiles S = 45

Temporal chunks
  chunks = ceil(128 / 16) = 8

Each temporal chunk
  sampled raw frames in memory <= 16
  PIL tile images in memory    <= 16 frames * 45 tiles
  tile sequences               = 45

AutoGaze/SigLIP tile batching inside one chunk
  batch 1: tile seq  0..15  -> tensor [16 tile seq, 16 frames, 3, 392, 392]
  batch 2: tile seq 16..31  -> tensor [16 tile seq, 16 frames, 3, 392, 392]
  batch 3: tile seq 32..44  -> tensor [13 tile seq, 16 frames, 3, 392, 392]

After each tile batch
  AutoGaze output gazing_info is summarized in stream-profile
    or kept/stacked in the full NVILA path
  optional SigLIP forward runs on that same batch
  tile tensor is released before next batch

After each temporal chunk
  raw sampled frames are released
  PIL tile images are released
  next 16 sampled frames are decoded
```

`max_batch_size_autogaze`는 한 번에 몇 개의 spatial tile sequence를 AutoGaze tensor로 만들고 forward할지 정합니다. `--stream-siglip-max-embed-batch-size`는 custom SigLIP 내부 embedding mini-batch 크기입니다. CUDA에서는 보통 AutoGaze 16, SigLIP 32로 시작하고, MPS에서는 AutoGaze 1이 더 안정적이었습니다.

thumbnail은 현재 AutoGaze로 줄이지 않고 keep-all로 처리합니다. `M=64`처럼 작게 유지하면 보통 병목은 아니지만, total patch/token 감소율을 희석하므로 리더 설명에서는 tile-only 감소율과 total 감소율을 분리해서 보여주는 것이 좋습니다.

### Latency와 memory를 같이 보는 법

단순히 `pre_llm_stream_total_measured`만 보면 어느 단계가 병목인지 가려집니다. 아래처럼 나눠 봐야 합니다.

```text
End-to-end pre-LLM stream time
  = video_decode_scan or (video_keyframe_index_scan + video_seek + video_decode_seek)
  + video_frame_to_pil
  + video_frame_resize
  + spatial_tile_build
  + tile_autogaze_tensorize
  + tile_autogaze_forward
  + siglip_gazed_forward or siglip_keep_all_forward
  + thumbnail_resize
  + thumbnail_tensorize

Tile-specific processing time
  = spatial_tile_build
  + tile_autogaze_tensorize
  + tile_autogaze_forward
  + optional siglip_*_forward

Memory pressure to compare at the same time
  raw frame buffer peak       = C * effective_width * effective_height * 3
  PIL tile buffer peak        = C * S * 392 * 392 * 3
  AutoGaze tensor peak        = C * min(S, max_batch_size_autogaze) * 3 * 392 * 392 * dtype_bytes
  full chunk tensor reference = C * S * 3 * 392 * 392 * dtype_bytes
  thumbnail tensor            = M * 3 * 392 * 392 * dtype_bytes
```

현재 plan table은 보수적으로 float32 기준 bytes를 잡습니다. 실제 결과 JSON의 `memory_bytes.autogaze_tile_tensor_peak_per_temporal_chunk`는 실제 tensor dtype 기준으로 기록되므로 CUDA float16에서는 더 낮게 나올 수 있습니다.

4K, `C=16`일 때 spatial tile 수별 memory 감각은 아래와 같습니다.

```text
4K source, no resize, C=16, tile=392, AutoGaze batch cap=16

S spatial tiles | raw frame buffer | PIL tile buffer | AG tensor peak | full chunk tensor
--------------- | ---------------- | --------------- | -------------- | -----------------
1 tile          | 398.1 MB         |   7.4 MB        |  29.5 MB       |   29.5 MB
8 tiles         | 398.1 MB         |  59.0 MB        | 236.0 MB       |  236.0 MB
45 tiles        | 398.1 MB         | 331.9 MB        | 472.1 MB       | 1327.7 MB
```

여기서 `AG tensor peak`는 tile batch를 16개 단위로 나누기 때문에 45 tiles에서도 472 MB 수준으로 제한됩니다. 반대로 `PIL tile buffer`는 현재 chunk의 모든 spatial tile image를 만든 뒤 AutoGaze batch로 넘기므로 `S`에 선형으로 늘어납니다. 그래서 4K에서는 latency뿐 아니라 `max_tiles_video`가 memory에도 직접적인 1차 레버입니다.

## 로컬 실측 요약

| 입력 | 설정 | 총 pre-LLM | AutoGaze forward | tile 감소 | total 감소 | keep-all LLM token | AutoGaze LLM lower-bound | peak raw/tile/AG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 896 square | 16f, 8thumb, 1tile, batch4 | 4.64s | 4.27s | 42.6x | 2.87x | 2,832 | 987 | 38.5/7.4/29.5 MB |
| 448 resized | 64f, 32thumb, 1tile, batch4 | 20.48s | 19.12s | 52.8x | 2.89x | 11,328 | 3,912 | 9.6/7.4/29.5 MB |
| 720 resized | 16f, 8thumb, 4tile, batch1 | 11.27s | 10.67s | 58.5x | 7.92x | 8,496 | 1,072 | 24.9/29.5/29.5 MB |
| 720 resized | 16f, 8thumb, 4tile, batch4 | 26.31s | 25.61s | 57.5x | 7.90x | 8,496 | 1,074 | 24.9/29.5/118.0 MB |
| 1080p 16:9 | 16f, 8thumb, 8tile, batch1 | 42.96s | 38.94s | 15.5x | 8.35x | 16,032 | 1,918 | 99.5/59.0/29.5 MB |
| HLVid 4K keep-all seek | 16f, 8thumb, 1tile | 2.36s | n/a | 1.0x | 1.0x | 2,832 | n/a | 398.1/7.4/0 MB |
| HLVid 4K keep-all seek | 128f, 64thumb, 1tile | 16.63s | n/a | 1.0x | 1.0x | 22,656 | n/a | 398.1/7.4/0 MB |
| HLVid 4K keep-all | 16f, 8thumb, 1tile | 55.06s | n/a | 1.0x | 1.0x | 2,832 | n/a | 398.1/7.4/0 MB |
| HLVid 4K keep-all | 16f, 8thumb, 8tile | 56.86s | n/a | 1.0x | 1.0x | 16,032 | n/a | 398.1/59.0/0 MB |
| HLVid 4K keep-all | 16f, 8thumb, 45tile | 70.91s | n/a | 1.0x | 1.0x | 85,744 | n/a | 398.1/331.9/0 MB |

실측 파일:

- `outputs/autogaze_repro/stream_profile_security_autogaze_16f_mps.json`
- `outputs/autogaze_repro/stream_sweep/fast_448p_1tile_64f_autogaze.json`
- `outputs/autogaze_repro/security_720p_4tile_16f_batch1_mps.json`
- `outputs/autogaze_repro/security_720p_4tile_16f_batch4_mps.json`
- `outputs/autogaze_repro/bbb_1080p_16f_8tile_batch1_mps.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_16f_1tile_seek_cpu.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_128f_1tile_seek_cpu.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_16f_8tile_cpu.json`
- `outputs/autogaze_repro/hlvid_4k_keepall_16f_45tile_cpu.json`

SigLIP까지 포함한 MPS smoke:

| 입력 | 설정 | 총 pre-LLM | AutoGaze forward | SigLIP gazed | SigLIP keep-all | tile 감소 | total 감소 | SigLIP hidden peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 896 square | 16f, 8thumb, 1tile, patch16, both | 5.88s | 2.33s | 0.19s | 2.88s | 35.9x | 2.84x | 0.6/13.0 MB |
| 896 square | 64f, 32thumb, 1tile, patch16, both | 25.25s | 7.94s | 1.05s | 15.21s | 41.8x | 2.86x | 0.7/13.0 MB |
| HLVid 4K seek | 16f, 8thumb, 1tile, patch16, gazed | 6.42s | 3.58s | 0.29s | n/a | 15.3x | 2.65x | 0.9 MB |
| HLVid 4K seek | 128f, 64thumb, 1tile, patch16, both | 43.74s* | 10.84s | 1.01s | 14.86s | 16.1x | 2.67x | 0.9/13.0 MB |
| HLVid 720p seek | 128f, 64thumb, 1tile, patch16, both | 41.92s* | 10.03s | 0.74s | 14.83s | 16.1x | 2.67x | 0.9/13.0 MB |

`both` 모드의 총 pre-LLM 시간은 gazed SigLIP와 keep-all SigLIP을 한 실행에서 둘 다 돌린 합산이므로 end-to-end 비교값이 아닙니다. 같은 output에서 branch별로 빼서 보면 아래가 더 해석하기 쉽습니다.

| 입력 | keep-all SigLIP only | AutoGaze forward + gazed SigLIP | 순수 모델 forward speedup | 추정 keep-all stream | 추정 AutoGaze stream | 추정 stream speedup |
|---|---:|---:|---:|---:|---:|---:|
| HLVid 4K 128f | 14.86s | 11.85s | 1.25x | 31.64s | 28.88s | 1.10x |
| HLVid 720p 128f | 14.83s | 10.78s | 1.38x | 30.92s | 27.09s | 1.14x |

여기서 "순수 모델 forward"는 `siglip_keep_all_forward / (tile_autogaze_forward + siglip_gazed_forward)`입니다. AutoGaze tensorize까지 넣으면 4K는 `1.23x`, 720p는 `1.35x`입니다. 즉 HLVid 128프레임/1tile/MPS에서는 SigLIP 자체는 `14.8-19.9x` 빨라지지만, AutoGaze forward가 약 `10-11s` 들어서 vision-only 순이득은 아직 작습니다. 장점은 token/compute 감소입니다: tile patch `16.1x`, LLM visual token lower-bound `2.72x`, SigLIP attention MACs `29.1x`, SigLIP total MACs `4.10x` 감소입니다. full MLLM에서 prefill/KV cache 이득을 확인해야 최종 latency claim을 만들 수 있습니다.

추가 실측 파일:

- `outputs/autogaze_repro/security_16f_1tile_siglip_google_both_mps.json`
- `outputs/autogaze_repro/security_64f_1tile_siglip_google_both_mps.json`
- `outputs/autogaze_repro/hlvid_4k_16f_1tile_siglip_google_gazed_mps.json`
- `outputs/autogaze_repro/hlvid_4k_16f_1tile_siglip_google_gazed_seek_mps.json`
- `outputs/autogaze_repro/hlvid_4k_128f_1tile_siglip_google_both_seek_mps.json`
- `outputs/autogaze_repro/hlvid_720p_128f_1tile_siglip_google_both_seek_mps.json`
- `outputs/autogaze_repro/hlvid_128f_siglip_autogaze_tradeoff_report.json`

MPS timing은 첫 실행의 graph compile/cache 상태에 따라 흔들립니다. 같은 command를 한 번 더 돌리면 특히 SigLIP gazed 시간이 낮아질 수 있으므로, CUDA에서 최종 claim을 만들 때는 warmup 후 반복 측정하세요.

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
# AutoGaze 이후 SigLIP vision tower까지 포함한 MPS path check
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
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --stream-run-siglip \
  --stream-siglip-mode both \
  --stream-siglip-model google/siglip2-base-patch16-224 \
  --stream-siglip-max-embed-batch-size 1 \
  --stream-profile-json outputs/autogaze_repro/security_64f_1tile_siglip_google_both_mps.json
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
  --stream-decode-strategy seek \
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
  --stream-decode-strategy seek \
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
  --stream-decode-strategy seek \
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
  --stream-decode-strategy seek \
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
  --stream-decode-strategy seek \
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
  --stream-decode-strategy seek \
  --gazing-mode autogaze \
  --include balanced_720 \
  --include balanced_1080 \
  --run \
  --timeout-seconds 1800 \
  --summary-json outputs/autogaze_repro/stream_sweep_hlvid_cuda_run.json \
  --summary-csv outputs/autogaze_repro/stream_sweep_hlvid_cuda_run.csv
```

SigLIP forward까지 같이 재려면 sweep에도 같은 옵션을 붙일 수 있습니다. `--stream-siglip-mode gazed`는 AutoGaze 이후만 재고, `both`는 keep-all SigLIP까지 같이 재기 때문에 긴 sequence에서는 메모리와 시간이 크게 늘어납니다.

```bash
.venv/bin/python -m repro.stream_profile_sweep \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --device cuda \
  --stream-dtype float16 \
  --stream-decode-strategy seek \
  --gazing-mode autogaze \
  --include fast_720 \
  --run \
  --autogaze-target-scales 32+64+112+224 \
  --autogaze-target-patch-size 16 \
  --stream-run-siglip \
  --stream-siglip-mode gazed \
  --stream-siglip-model google/siglip2-base-patch16-224 \
  --stream-siglip-max-embed-batch-size 16 \
  --timeout-seconds 1800 \
  --summary-json outputs/autogaze_repro/stream_sweep_hlvid_cuda_siglip_run.json \
  --summary-csv outputs/autogaze_repro/stream_sweep_hlvid_cuda_siglip_run.csv
```

## full NVILA 비교 순서

stream-profile에서 후보를 줄인 뒤 full NVILA는 같은 조합으로 `--mode single` 또는 `--mode hlvid`를 돌립니다.

1. `stream-profile keep-all`: decode/tiling memory와 keep-all token budget 확인
2. `stream-profile autogaze`: AutoGaze forward latency와 patch/token reduction 확인
3. `single keep-all`: full NVILA baseline이 context/OOM 없이 가능한지 확인
4. `single autogaze`: SigLIP/Projector/LLM stage time이 줄어드는지 확인

리더 설득용 지표는 최소 아래를 같이 보고하세요.

- `timing_ms.tile_autogaze_forward`
- `timing_ms.siglip_gazed_forward` / `timing_ms.siglip_keep_all_forward` (`--stream-run-siglip` 사용 시)
- `token_metrics.encoder_tile_token_reduction_ratio`
- `token_metrics.encoder_token_reduction_ratio`
- `token_metrics.llm_keep_all_visual_tokens_estimated`
- `token_metrics.llm_autogaze_visual_tokens_lower_bound_estimated`
- `compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio`
- `compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio`
- `compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio`
- full NVILA에서는 `result.siglip_vision_ms`, `result.mm_projector_ms`, `result.llm_forward_ms`, `result.ttft_ms`
- full NVILA에서는 `compute_metrics.mllm.actual_prefill_context_tokens`, `compute_metrics.mllm.prefill_context_reduction_ratio`, `compute_metrics.mllm.kv_cache_reduction_ratio`, `result.ttft_peak_memory_bytes`, `result.llm_peak_memory_bytes`

코드에서 측정 위치를 확인할 때는 아래를 기준으로 보면 됩니다.

| 측정 항목 | 코드 위치 |
| --- | --- |
| stream stage timer | [StageProfiler:L45](../repro/nvila_runner.py#L45), [run_stream_profile:L1822](../repro/nvila_runner.py#L1822) |
| AutoGaze tensorize/forward | [run_autogaze_on_stream_tile_sequences:L1576](../repro/nvila_runner.py#L1576) |
| gazed/keep-all SigLIP forward | [run_siglip_on_stream_batch:L1527](../repro/nvila_runner.py#L1527) |
| stream token/patch metrics | [build_stream_profile_token_metrics:L313](../repro/nvila_runner.py#L313) |
| stream compute metrics | [build_stream_profile_compute_metrics:L362](../repro/nvila_runner.py#L362) |
| full NVILA SigLIP/projector/LLM timing hook | [ProfilePatches:L81](../repro/nvila_runner.py#L81), [forward hooks:L95-L100](../repro/nvila_runner.py#L95-L100) |
| TTFT/LLM memory | [timed_generate:L1337](../repro/nvila_runner.py#L1337), [generate_one memory fields:L2227-L2302](../repro/nvila_runner.py#L2227-L2302) |
| HLVid batch gain report | [summarize_run:L194](../repro/hlvid_batch_benchmark.py#L194), [build_gain_report:L205](../repro/hlvid_batch_benchmark.py#L205) |
