# AutoGaze Timing 검증 리포트

이 문서는 AutoGaze 자체 latency와 NVILA runner에 붙었을 때의 latency가 왜 다르게 보일 수 있는지, 어떤 경계를 측정해야 공정한지 정리합니다.

긴 이전 리포트는 `docs/temp/AUTOGAZE_TIMING_VALIDATION_REPORT_KO_FULL_2026-05-21.md`에 보존했습니다.

## 비교 목적

우리가 비교하려는 것은 세 가지입니다.

| 비교 | 의미 |
| --- | --- |
| Quick Start AutoGaze-only | 공식 예제와 같은 방식으로 AutoGaze 모델 자체 시간을 확인 |
| NVILA runner AutoGaze-only 구간 | 실제 NVILA-HD processor 안에서 AutoGaze가 쓰일 때 시간 확인 |
| Full pipeline | preprocess + AutoGaze + ViT + LLM까지 end-to-end 이득 확인 |

Quick Start가 느리고 NVILA runner의 AutoGaze 구간이 빠르게 보이면 먼저 다음을 확인해야 합니다.

1. 입력 patch budget이 같은가.
2. scale 목록과 patch size가 같은가.
3. frame/tile/thumbnail 수가 같은가.
4. dtype/device가 같은가.
5. AutoGaze `generate`까지 포함했는가, `forward`만 잰 것인가.
6. batch size와 gazing ratio가 같은가.

## Timing 경계

현재 리포트에서 가장 중요한 상위 구조는 다음입니다.

```text
total_ms
  = video_preprocess_without_autogaze_ms
  + autogaze_total_ms
  + generate_ms
```

| 항목 | 포함 | 미포함 / 주의 |
| --- | --- | --- |
| `video_decode_read_ms` | 비디오 metadata/keyframe scan/seek/decode/frame 변환 | AutoGaze forward, LLM generate |
| `video_frame_resize_ms` | runner-side frame resize가 켜진 경우의 resize | resize를 processor가 수행하면 `Prep rest`에 남을 수 있음 |
| `video_tiling_ms` | NVILA processor의 tile 생성, thumbnail 처리, tensor 변환 | 모델 forward |
| `preprocess_rest_without_decode_autogaze_ms` | `video_preprocess_without_autogaze_ms - video_decode_read_ms` | decode/read와 AutoGaze를 뺀 processor residual |
| `video_preprocess_without_autogaze_ms` | decode + resize/tile/tensorize. AutoGaze 제외 | `autogaze_total_ms`, `generate_ms` |
| `autogaze_model_forward_ms` | AutoGaze 모델 forward batch 실행 합 | 입력 준비/선택 정리, 비디오 decode, ViT, LLM |
| `selector_input_build_ms` | `autogaze_total_ms - autogaze_model_forward_ms` residual | 독립 wrapper가 아니라 실제 측정된 부모/자식 timer 차이 |
| `autogaze_total_ms` | AutoGaze 입력 준비 + 모델 실행 + 선택 결과 정리 | LLM generate |
| `siglip_vision_ms` | SigLIP/ViT vision tower forward-only | projector, LLM |
| `mm_projector_ms` | visual feature -> LLM hidden projection | SigLIP forward, LLM |
| `vision_input_build_ms` | `vision_encoder_ms - siglip_vision_ms - mm_projector_ms` residual | 독립 wrapper가 아니라 feature packing/reorder residual |
| `vision_encoder_ms` | vision path wrapper 전체 | AutoGaze, LLM forward |
| `llm_forward_ms` | LLM forward 호출 누적 | video preprocess, AutoGaze |
| `generate_ms` | MLLM generate/prefill 포함 모델 호출 | preprocess, AutoGaze |

하위 항목은 항상 상위 항목의 완전한 합과 일치하지 않을 수 있습니다. 경계 바깥의 Python overhead, synchronization, processor 내부 bookkeeping이 있을 수 있기 때문입니다. 그래서 report에는 상위 3분할과 핵심 하위 timing을 같이 남깁니다. `selector_input_build_ms`와 `vision_input_build_ms`는 실제 측정된 parent timer에서 child timer를 뺀 residual이며, 이 둘은 total에 다시 더하지 않습니다.

## AutoGaze latency에 영향을 주는 옵션

| 옵션 | 영향 |
| --- | --- |
| `--gazing-ratio` | 선택할 patch budget을 줄임. 낮을수록 후속 decode step/선택 결과는 작아지지만, selector 자체 overhead가 0이 되지는 않음 |
| `--task-loss-threshold` / task loss 관련 설정 | decoder가 더 일찍 멈출 수 있는 조건에 영향 |
| `--max-batch-size-autogaze` | NVILA runner에서 AutoGaze patch/frame batch를 몇 개씩 나누어 돌릴지 결정 |
| Quick Start batch size | 공식 AutoGaze-only 예제에서 모델 입력 batch 크기 결정 |
| `--autogaze-generate-only` | generate 경로만 측정/사용할 수 있는지 확인하는 실험 옵션 |
| `--num-video-frames` | frame 수에 거의 선형적으로 patch 후보 증가 |
| `--num-video-frames-thumbnail` | thumbnail keep/add 비용 증가 |
| `--max-tiles-video` | 고해상도 frame을 몇 개 tile로 나누는지 결정 |
| `--video-resize-shortest-edge` | tile 전 입력 해상도와 patch 후보 수에 영향 |

## 권장 검증 순서

1. Quick Start 예제 비디오로 AutoGaze-only timing을 잰다.
2. 같은 비디오, 같은 frame 수, 같은 scale/patch 설정으로 `nvila_runner --mode single`을 AutoGaze-only 관점에서 잰다.
3. `gazing_ratio` sweep으로 선택 patch 수와 latency가 같이 움직이는지 확인한다.
4. HLVid 긴 영상 128f/256f에서 AutoGaze 시간과 ViT/LLM 시간 감소를 같이 본다.
5. 최종 주장은 full pipeline 기준으로 한다.

## 예시 명령

Quick Start와 NVILA runner 비교용 script:

```bash
.venv/bin/python -m repro.autogaze_timing_compare \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --autogaze-repo external/AutoGaze \
  --weights-root weights \
  --frames 16 \
  --gazing-ratio-sweep 0.1,0.2,0.3,0.5 \
  --quickstart-native \
  --output-dir outputs/autogaze_repro/timing/quickstart_compare
```

NVILA runner single:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --video inputs/hlvid_example/clip_av_video_5_001.mp4 \
  --question "What is happening in the video?" \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-model weights/autogaze \
  --gazing-mode autogaze \
  --gazing-ratio 0.2 \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 8 \
  --output-json outputs/autogaze_repro/timing/nvila_single_16f.json
```

Markdown 변환:

```bash
.venv/bin/python -m repro.markdown_report \
  --input-json outputs/autogaze_repro/timing/nvila_single_16f.json \
  --output-md outputs/autogaze_repro/timing/nvila_single_16f.md
```

## 해석 체크리스트

| 질문 | 봐야 할 값 |
| --- | --- |
| AutoGaze 모델만 얼마 걸렸나 | `autogaze_model_forward_ms` 또는 stage timing의 `processor.autogaze_forward_batched.total_ms` |
| AutoGaze 전체 비용은 얼마인가 | `autogaze_total_ms` |
| AutoGaze 입력/후처리 비용은 얼마인가 | `selector_input_build_ms` |
| 전처리와 중복인가 | `video_preprocess_without_autogaze_ms`와 분리해서 확인 |
| ViT 이득이 있나 | `siglip_vision_ms`, `vision_encoder_ms`, encoder input token 수 |
| vision feature packing 비용이 있나 | `vision_input_build_ms` |
| LLM 이득이 있나 | `generate_ms`, visual token 수, TTFT |
| 선택이 실제로 줄었나 | full/off patch 대비 selected patch ratio |
