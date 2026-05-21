# AutoGaze Gazing Policy and NVILA Integration Notes

이 문서는 두 가지 질문을 코드 기준으로 정리한다.

1. `gazing_ratio`와 `task_loss_requirement`가 AutoGaze decoder의 patch 선택에 어떤 영향을 주는가.
2. AutoGaze만 단독으로 돌릴 때와 NVILA processor에 붙여서 돌릴 때 무엇이 달라지고, 왜 token count 조건을 나눠서 봐야 하는가.

## TL;DR

- `gazing_ratio`는 "선택할 patch 점수 threshold"가 아니라 frame별 최대 gaze token budget이다.
- AutoGaze decoder는 전체 patch를 한 번에 scoring해서 top-k로 자르는 방식이 아니라, autoregressive decoder가 patch id를 한 개씩 생성한다.
- `task_loss_requirement`가 있으면 decoder가 선택 도중 "이 정도면 충분하다"고 판단한 뒤 남은 slot을 padded gaze로 채울 수 있다.
- 첫 번째 frame도 반드시 전체 patch를 선택하는 규칙은 없다. 현재 NVILA runner 기본값은 tile에 대해 `[0.2] + [0.06] * 15`라서 첫 frame도 최대 20% budget이다.
- thumbnail은 별도 경로이며 현재 NVILA kwargs에서는 `gazing_ratio_thumbnail=1`, `task_loss_requirement_thumbnail=None`으로 들어간다. 그래서 thumbnail은 AutoGaze reduction 없이 keep-all로 계산된다.
- AutoGaze standalone과 NVILA-integrated run의 multi-scale gaze output 형식은 같은 계열이지만, 입력 단위와 집계 범위가 다르다. standalone은 "AutoGaze에 넣은 video tensor" 기준이고, NVILA는 tile frames + thumbnail frames + token shuffle 이후 LLM visual token까지 같이 봐야 한다.

## 1. AutoGaze Decoder 동작

### 입력과 출력

AutoGaze model call은 대략 아래 구조다.

```text
video tensor
  shape: B x T x C x H x W
      |
      v
AutoGaze.forward(...)
      |
      +-- optional input_res_adapt(target_scales, target_patch_size)
      |
      v
AutoGazeModel.generate(...)
      |
      +-- frame embedding
      +-- autoregressive gaze decoder
      +-- no-repeat patch id constraint
      +-- optional task-loss early stop
      |
      v
gazing_info
  - gazing_pos
  - num_gazing_each_frame
  - if_padded_gazing
  - gazing_mask
  - task_loss_requirement
```

중요한 output은 세 가지다.

| Field | 의미 |
| --- | --- |
| `gazing_pos` | 선택된 patch id. frame offset이 더해진 global patch position이다. |
| `num_gazing_each_frame` | frame별 gaze slot 수. 여기에는 padded slot도 포함된다. |
| `if_padded_gazing` | 해당 slot이 실제 선택 patch인지, early stop 후 채워진 padding인지 표시한다. |

실제 선택 patch 수는 `(~if_padded_gazing).sum()`으로 세야 한다. `num_gazing_each_frame.sum()`은 slot 수라서 실제 선택 patch 수보다 클 수 있다.

### `gazing_ratio`는 최대 길이 budget이다

사용자가 `gazing_ratio`를 넘기면 AutoGaze는 frame별 최대 gaze token 수를 이렇게 만든다.

```text
num_gaze_tokens_each_frame
  = floor(gazing_ratio_each_frame * num_vision_tokens_each_frame)
  = clamp(min=1)
```

즉 `gazing_ratio=0.06`은 "확률이 0.06보다 큰 patch만 선택"이 아니다. "각 frame에서 최대 약 6% patch slot까지만 autoregressive decoder가 patch id를 생성할 수 있다"에 가깝다.

예를 들어 NVILA HD에서 혼동하기 쉬운 patch budget은 두 가지가 있다.

첫째, NVILA-HD sparse SigLIP alignment 기준:

```text
target_scales = 56 + 112 + 196 + 392
target_patch_size = vision_config.patch_size = 14

56//14  = 4   ->  16 positions
112//14 = 8   ->  64 positions
196//14 = 14  -> 196 positions
392//14 = 28  -> 784 positions

total per frame = 16 + 64 + 196 + 784 = 1060 positions
```

둘째, NVILA-HD weight metadata에 들어 있는 release processor 기준:

```text
preprocessor_config.target_scales = 56 + 112 + 196 + 392
preprocessor_config.target_patch_size = 16

56//16  = 3   ->   9 positions
112//16 = 7   ->  49 positions
196//16 = 12  -> 144 positions
392//16 = 24  -> 576 positions

total per frame = 9 + 49 + 144 + 576 = 778 positions
```

AutoGaze standalone/patch16 SigLIP 설명의 patch size 16은 두 번째 계열에 가깝다. 하지만 NVILA-HD의 실제 vision tower는 `vision_config.patch_size=14`이고, SigLIP sparse path는 `gazing_pos`로 patch sequence를 직접 gather한다. 따라서 runner 기본값은 위치 정렬을 우선해 patch14 aligned 경로로 둔다. patch16 release metadata 경로는 `--autogaze-target-patch-size 16`을 명시한 호환성/ablation 비교로만 해석한다.

이때 scalar ratio를 AutoGaze target coordinate 기준으로 쓰면:

```text
gazing_ratio=0.06 -> floor(1060 * 0.06) = 63 slots/frame
gazing_ratio=0.20 -> floor(1060 * 0.20) = 212 slots/frame
gazing_ratio=0.75 -> floor(1060 * 0.75) = 795 slots/frame
gazing_ratio=1.00 -> 1060 slots/frame
```

AutoGaze 기본 224 단일 scale이면 보통 frame당 196 patches이므로:

```text
gazing_ratio=0.06 -> floor(196 * 0.06) = 11 slots/frame
gazing_ratio=0.75 -> floor(196 * 0.75) = 147 slots/frame
```

### 낮은 `gazing_ratio`일 때 patch 선택 기준

낮은 ratio를 주면 후보 patch 전체를 ranking한 뒤 top-k로 자르는 것이 아니라, decoder가 생성할 수 있는 token 수가 줄어든다.

```text
for each frame t:
  1. 현재 frame embedding을 decoder context에 추가
  2. 이전 frame/video context와 이전 gaze token들을 함께 봄
  3. 다음 gaze patch id를 argmax 또는 sampling으로 생성
  4. 이미 고른 patch id는 다시 못 고르게 막음
  5. max_gaze_tokens_each_frame에 닿거나 task-loss 조건을 만족하면 stop/pad
```

따라서 낮은 ratio에서 선택되는 patch는 "decoder가 현재 context에서 먼저 볼 가치가 높다고 예측한 patch들"이다. saliency heatmap을 전부 계산해서 전역 top-k를 고르는 구현은 아니다.

추론 모드에서는 `do_sample=False`라서 decoder logits의 argmax 선택이 기본이다. 학습 중에는 `do_sample=True`와 temperature가 쓰일 수 있다.

### `task_loss_requirement`는 조기 종료 조건이다

`task_loss_requirement`를 같이 넘기면 ratio budget은 여전히 최대 길이로 쓰이고, task-loss는 그 안에서 멈출 수 있는 조건으로 쓰인다.

```text
max slots/frame = floor(gazing_ratio * num_vision_tokens_each_frame)

while generated slots < max slots/frame:
  next_patch = decoder(...)
  predicted_task_loss = decoder.task_loss_prediction

  if predicted_task_loss <= task_loss_requirement:
      fill current/future slots as padded gaze
      stop this frame
```

해석상 주의점:

- threshold가 높으면 `predicted_task_loss <= threshold`를 더 쉽게 만족하므로 더 일찍 멈출 수 있다.
- threshold가 낮으면 더 낮은 loss까지 요구하므로 일반적으로 더 많은 patch를 보게 될 수 있다.
- 다만 실제 선택 수는 model prediction, video content, frame context, no-repeat constraint에 같이 좌우된다.
- 첫 token은 task-loss 조건으로 바로 pad되지 않도록 막혀 있다. 그래서 ratio가 아주 낮아도 frame당 최소 한 개 slot은 생긴다.

### 첫 번째 frame을 꼭 full로 보지 않는 이유

현재 우리 `nvila_runner`의 기본 tile ratio는 scalar `1.0`이 아니라 list다.

```text
default gazing_ratio_tile = [0.2] + [0.06] * 15
```

즉 16-frame tile chunk 기준으로:

```text
frame 0      -> max 20%
frame 1..15  -> max 6%
```

그래서 첫 frame도 full keep-all이 아니다. full로 만들고 싶으면 ratio list의 첫 값을 `1.0`으로 주면 된다.

```bash
--gazing-ratio-tile 1.0,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06
```

반대로 첫 frame도 낮게 보고 싶으면 첫 값을 낮게 주면 된다.

```bash
--gazing-ratio-tile 0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06
```

keep-all ablation은 다른 의미다.

```text
--gazing-mode keep-all
  -> gazing_ratio_tile = 1
  -> task_loss_requirement_tile = None
  -> AutoGaze selection을 사실상 끄고 tile patch를 전부 통과시키는 비교용 모드
```

### 간단한 시각화

예를 들어 1 frame에 10개 patch가 있고 `gazing_ratio=0.4`라면 최대 slot은 4개다.

```text
raw patches:
[00] [01] [02] [03] [04] [05] [06] [07] [08] [09]

decoder generation with max 4 slots:
step 1 -> patch 03
step 2 -> patch 07
step 3 -> patch 01
step 4 -> patch 05

selected:
[  ] [XX] [  ] [XX] [  ] [XX] [  ] [XX] [  ] [  ]
```

task-loss가 step 2에서 이미 충분하다고 판단하면:

```text
max slots = 4

step 1 -> patch 03
step 2 -> patch 07, predicted_task_loss <= threshold
step 3 -> PAD
step 4 -> PAD

gazing_pos slots:
[03] [07] [PAD] [PAD]

selected_non_padded_patches = 2
total_gaze_slots = 4
padded_gazing_positions = 2
```

## 2. AutoGaze Standalone과 NVILA Integration 차이

### 공통점

AutoGaze standalone과 NVILA-integrated run 모두 최종적으로는 같은 계열의 `gazing_info`를 만든다.

```text
gazing_info
  - gazing_pos
  - num_gazing_each_frame
  - if_padded_gazing
  - gazing_mask
```

multiscale output도 같은 원리다. AutoGaze의 `gazing_mask`는 scale별 mask list이고, SigLIP 쪽에서는 `gazing_pos`와 `if_padded_gazing`을 사용해 raw patch sequence와 position embedding을 같은 위치 기준으로 gather한다.

```text
pixel_values
  |
  +-- scale 56  -> patches + pos_embed
  +-- scale 112 -> patches + pos_embed
  +-- scale 196 -> patches + pos_embed
  +-- scale 392 -> patches + pos_embed
       |
       v
concat multiscale patches
       |
       v
mask_with_gazing(gazing_pos, if_padded_gazing)
       |
       v
selected patches + selected position embeddings
       |
       v
SigLIP encoder
```

즉 "gazed patch에는 position embedding이 같이 따라가느냐"는 질문에는 yes다. patch와 position embedding 둘 다 같은 `gazing_info`로 gather된다.

### 차이 1: 입력 단위

Standalone Quick Start 계열은 보통 하나의 video tensor를 AutoGaze에 직접 넣는다.

```text
example_input.mp4
  -> first/sample frames
  -> transform_video_for_pytorch
  -> B x T x C x H x W
  -> AutoGaze
```

NVILA integration은 MLLM processor 안에서 video를 더 복잡하게 다룬다.

```text
video
  -> sampled frames
  -> optional resize
  -> spatial tiling
  -> temporal chunks, usually 16 frames
  -> tile sequences
  -> AutoGaze over tile sequences
  -> thumbnail sequence, usually keep-all
  -> SigLIP / vision tower
  -> token shuffle
  -> mm projector
  -> LLM prefill/generate
```

그래서 같은 16 frames라 해도 아래 값들이 다르면 latency와 token count가 apples-to-apples가 아니다.

| 조건 | standalone AutoGaze | NVILA integrated |
| --- | --- | --- |
| frame source | 직접 지정한 video tensor | sampled frames + tiles + thumbnails |
| spatial unit | 보통 1 video sequence | tile sequence가 여러 개일 수 있음 |
| scale/patch | AutoGaze model config 또는 CLI target scales | NVILA vision tower target scales와 patch size에 맞춤 |
| thumbnail | 없음 | 보통 별도 keep-all 경로 |
| 후속 처리 | optional SigLIP smoke | SigLIP/vision encoder, projector, LLM까지 연결 |

### 차이 2: thumbnail은 AutoGaze 감소 대상이 아닐 수 있다

현재 NVILA runner에서 AutoGaze kwargs는 tile과 thumbnail을 분리한다.

```text
tile:
  gazing_ratio_tile = user/default
  task_loss_requirement_tile = user/default

thumbnail:
  gazing_ratio_thumbnail = 1
  task_loss_requirement_thumbnail = None
```

이 뜻은 tile patch는 AutoGaze로 줄일 수 있지만, thumbnail patch는 기본적으로 keep-all로 들어간다는 것이다. 그래서 token count는 두 관점으로 나눠야 한다.

```text
AutoGaze-only reduction:
  denominator = raw tile patches
  numerator   = selected tile patches

Encoder total reduction:
  denominator = raw tile patches + raw thumbnail patches
  numerator   = selected tile patches + selected thumbnail patches
```

thumbnail이 많으면 AutoGaze가 tile에서 90%를 줄여도 encoder total reduction은 그보다 낮게 보일 수 있다. 이게 토큰 수 카운트를 조건별로 나눠서 기록한 이유다.

### 차이 3: LLM visual token은 patch token과 1:1이 아닐 수 있다

NVILA에는 vision encoder 이후 token shuffle / projector / packing 단계가 있다. 그래서 "AutoGaze patch token 감소율"과 "LLM에 들어가는 visual token 감소율"은 관련은 있지만 항상 같은 숫자는 아니다.

현재 리포트에서 봐야 하는 계층은 아래 순서다.

```text
1. autogaze_input_patch_tokens
   - AutoGaze가 선택 대상으로 받은 tile patch 수

2. autogaze_selected_patch_tokens
   - AutoGaze가 실제 선택한 tile patch 수

3. encoder_raw_patch_tokens
   - tile raw patches + thumbnail raw patches

4. encoder_autogaze_selected_patch_tokens
   - selected tile patches + thumbnail patches

5. llm_keep_all_visual_tokens_estimated
   - keep-all이면 LLM 쪽으로 갈 것으로 추정되는 visual token 수

6. llm_actual_visual_tokens_after_autogaze
   - 실제 AutoGaze 적용 후 LLM 쪽 visual token 수
```

실험 결과를 볼 때는 아래처럼 읽는 것이 가장 덜 헷갈린다.

```text
AutoGaze selection 자체 효과:
  autogaze_input_patch_tokens / autogaze_selected_patch_tokens

Vision encoder 전체 입력 감소:
  encoder_raw_patch_tokens / encoder_autogaze_selected_patch_tokens

LLM context 감소:
  llm_keep_all_visual_tokens_estimated / llm_actual_visual_tokens_after_autogaze
```

### 왜 standalone과 NVILA에서 raw patch count가 달라질 수 있나

가장 흔한 이유는 네 가지다.

1. scale/patch size가 다름
    - Quick Start 기본은 224 단일 scale에 가까워 frame당 196 patches가 될 수 있다.
    - NVILA-HD weight metadata에는 `target_patch_size=16`이 있지만, 실제 vision tower config에는 conv patch size 14가 기록되어 있다.
   - runner 기본값은 실제 sparse SigLIP gather 정렬을 우선해 `56+112+196+392`, patch size 14이며, frame당 선택 좌표는 `16+64+196+784=1060`개다.

2. tile 수가 다름
   - `max_tiles_video > 1`이면 같은 frame 수라도 AutoGaze 입력 tile-frame instance가 늘어난다.

3. thumbnail 포함 여부가 다름
   - standalone AutoGaze에는 thumbnail 경로가 없다.
   - NVILA에는 thumbnail frames가 encoder/LLM token count에 포함될 수 있다.

4. ratio policy가 다름
   - Quick Start default `gazing_ratio=0.75`, `task_loss_requirement=0.7`
   - NVILA runner default tile ratio `[0.2] + [0.06] * 15`, task loss default `0.6`

따라서 latency를 비교할 때는 반드시 아래를 먼저 맞춰야 한다.

```text
frames
target_scales
target_patch_size
tile count
thumbnail frames
gazing_ratio or gazing_ratio list
task_loss_requirement
batch/chunk setting
```

## 3. 결과 로그에서 어떤 필드를 보면 되나

### AutoGaze standalone / timing compare

| Field | 의미 |
| --- | --- |
| `quickstart_native_autogaze_ms` | README Quick Start code path의 AutoGaze forward only |
| `quickstart_autogaze_ms` | 우리 wrapper의 direct AutoGaze forward only |
| `quickstart_native_raw_patch_budget` | native path에서 AutoGaze가 받은 raw patch budget |
| `quickstart_native_selected_patches` | native path에서 실제 선택한 patch 수 |
| `quickstart_native_token_reduction_ratio` | raw / selected |

### NVILA single/benchmark

| Field | 의미 |
| --- | --- |
| `autogaze_input_patch_tokens` | AutoGaze tile 입력 patch 수 |
| `autogaze_selected_patch_tokens` | AutoGaze가 선택한 tile patch 수 |
| `autogaze_patch_reduction_ratio` | tile 기준 raw / selected |
| `encoder_raw_patch_tokens` | tile + thumbnail raw patch 수 |
| `encoder_autogaze_selected_patch_tokens` | selected tile + thumbnail patch 수 |
| `encoder_token_reduction_ratio` | encoder 전체 기준 raw / selected |
| `llm_keep_all_visual_tokens_estimated` | keep-all 가정의 LLM visual token 추정 |
| `llm_actual_visual_tokens_after_autogaze` | AutoGaze 적용 후 LLM visual token |
| `llm_visual_token_reduction_ratio` | LLM context 기준 keep-all / actual |

### `generate_only`와 AutoGaze latency 경계

AutoGaze forward는 두 부분으로 나눠서 생각하면 쉽다.

```text
video tensor
  -> autoregressive gaze decoder
       -> gazing_pos / if_padded_gazing / num_gazing_each_frame / gazing_mask
  -> optional verification/training forward
       -> logits / task_loss
```

`generate_only=True`이면 첫 번째 단계까지만 수행하고 바로 gaze 정보를 반환한다. 후속 NVILA/SigLIP inference에는 gaze 위치 정보가 필요하므로 이 모드는 “실제 inference에서 AutoGaze selector가 필요한 최소 시간”을 보는 데 유용하다. `generate_only=False`이면 추가로 `gazing_model(video, gazing_info)`가 호출되어 logits/task loss까지 계산되므로, Quick Start 기본 forward와 비교할 때는 이 옵션이 켜졌는지 꺼졌는지를 반드시 같이 봐야 한다.

현재 CLI 대응은 다음과 같다.

| 경로 | 옵션 |
| --- | --- |
| standalone direct timing | `repro.autogaze_bench --generate-only` |
| README-style native timing | `repro.autogaze_quickstart_native --generate-only` |
| NVILA single/HLVid/stream-profile | `repro.nvila_runner --autogaze-generate-only` |
| timing comparison wrapper | `repro.autogaze_timing_compare --autogaze-generate-only` |
| HLVid batch wrapper | `repro.hlvid_batch_benchmark --autogaze-generate-only` |

## 4. 시각화로 확인하는 방법

선택 frame과 AutoGaze overlay는 기존 visualization 옵션으로 저장할 수 있다.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --model-path weights/NVILA-8B-HD-Video \
  --autogaze-repo external/AutoGaze \
  --autogaze-model weights/AutoGaze \
  --video inputs/example_input.mp4 \
  --device cuda \
  --dtype float16 \
  --gazing-mode autogaze \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 1 \
  --max-tiles-video 1 \
  --gazing-ratio-tile 0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06 \
  --task-loss-requirement-tile 0.7 \
  --visualization-output-dir outputs/autogaze_repro/gaze_policy_visualization \
  --max-new-tokens 1 \
  --output-json outputs/autogaze_repro/gaze_policy_visualization/result.json
```

확인 포인트:

- selected frames video: 실제 sampled frame이 어떤 장면인지 확인한다.
- resized selected frames video: AutoGaze/NVILA 입력 resize 이후 기준을 확인한다.
- overlay video: scale별 color mask로 어떤 patch가 선택됐는지 확인한다.
- JSON의 `gazing_info.if_padded_gazing_tiles`: 선택 slot과 padded slot을 구분한다.

## 5. 코드 기준 위치

| 주제 | 코드 |
| --- | --- |
| ratio를 frame별 max token 수로 바꾸는 분기 | `external/AutoGaze/autogaze/models/autogaze/autogaze.py` |
| task-loss early stop | `external/AutoGaze/autogaze/models/autogaze/modeling_llama_multi_token_pred.py` |
| no-repeat gaze token 생성 | `external/AutoGaze/autogaze/models/autogaze/modeling_autogaze.py` |
| multiscale mask 생성 | `external/AutoGaze/autogaze/models/autogaze/autogaze.py` |
| SigLIP patch/position embedding gather | `external/AutoGaze/autogaze/vision_encoders/siglip/modeling_siglip.py` |
| NVILA AutoGaze kwargs 구성 | `repro/nvila_runner.py` |
| NVILA token metric 집계 | `repro/nvila_runner.py` |
| standalone direct timing | `repro/autogaze_bench.py` |
| README-style native timing | `repro/autogaze_quickstart_native.py` |
| NVILA processor에 `generate_only` 주입 | `repro/nvila_runner.py` |
