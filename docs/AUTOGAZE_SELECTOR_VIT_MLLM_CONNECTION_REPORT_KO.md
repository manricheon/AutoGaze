# AutoGaze Token Selector / ViT / MLLM 연결 리포트

## 목적

이 문서는 AutoGaze를 기존 `token selector -> ViT encoder -> MLLM` 파이프라인에 zero-shot 또는 최소 수정으로 붙일 때 생기는 차이를 정리한다. 핵심 질문은 세 가지다.

1. AutoGaze가 sparse multi-scale patch index를 출력할 때, 이 출력을 어떤 형태로 바꿔야 기존 ViT/MLLM에 붙이기 쉬운가?
2. 해상도, patch size, positional encoding, visual feature/token 수가 모델마다 어떻게 달라서 연결 난이도가 달라지는가?
3. 어느 모델/연결 지점이 먼저 PoC하기 좋은가?

결론부터 말하면, **가장 안전한 1차 적용은 post-encoder token prune**이다. 기존 ViT는 dense frame/tile을 그대로 처리하고, ViT output 이후 MLLM에 들어갈 visual token만 AutoGaze mask로 줄인다. 이 방식은 ViT latency를 줄이지 못하지만 MLLM context, prefill, KV cache, TTFT 감소를 먼저 확인할 수 있다. **ViT latency까지 줄이려면 pre-encoder sparse 입력**이 필요하며, 이때는 patch coordinate와 position encoding을 정확히 보존해야 하므로 모델별 probe가 필수다.

---

## 용어 정리: 어디서 줄이는가

token selector를 논의할 때 가장 헷갈리는 지점은 “토큰”이라는 말이 서로 다른 위치를 가리킨다는 점이다. 앞으로는 아래 용어를 고정해서 사용한다.

| 용어 | 위치 | 의미 | 줄이면 빨라지는 곳 |
|---|---|---|---|
| pixel/patch candidate | ViT 입력 전 | 이미지/프레임을 patch grid로 나누었을 때의 후보 | selector 이후 ViT 입력 |
| pre-ViT selector | ViT 입력 전 | ViT가 보기 전에 patch/pixel/tile을 줄임 | ViT encoder + projector + LLM |
| dense ViT token | ViT 내부 | dense patch embedding sequence | 줄이려면 ViT 내부 수정 필요 |
| post-encoder visual feature | ViT 출력 후 | ViT가 만든 visual feature token | projector + LLM |
| post-projector visual token | projector 후 | LLM hidden size에 맞춰진 visual token | LLM prefill + KV cache |
| LLM context token | LLM 입력 | text token + visual token 전체 | LLM attention + KV cache |
| token merge | ViT/LLM 중간 | 여러 token을 하나로 합침 | 주로 LLM, 경우에 따라 ViT |
| token prune | ViT/LLM 중간 | 일부 token을 제거함 | 제거 위치 이후 |

따라서 “토큰을 90% 줄였다”는 말은 반드시 분모와 위치를 같이 써야 한다.

```text
same 90% reduction

pre-ViT patch reduction:
  candidate image patches 10000 -> ViT input patches 1000
  gain: ViT + projector + LLM

post-encoder feature reduction:
  ViT output features 10000 -> MLLM visual features 1000
  gain: projector + LLM
  no gain: ViT encoder

post-projector token reduction:
  projected visual tokens 10000 -> LLM visual tokens 1000
  gain: LLM only
  no gain: ViT encoder, projector
```

---

## PixelPrune / Qwen을 어떻게 이해해야 하나

PixelPrune은 Qwen3-VL/Qwen3.5 계열을 직접 타깃으로 둔 **pre-ViT selector 레퍼런스**로 보는 것이 좋다. PixelPrune 공식 README는 `apply_pixelprune(model="qwen3_vl")`를 모델 로드 전에 호출하는 형태를 제시하고, Qwen3-VL 및 Qwen3.5를 HuggingFace/vLLM 백엔드에서 지원한다고 설명한다. 논문 abstract도 pixel space에서 ViT 전에 patch를 줄이기 때문에 ViT encoder와 downstream LLM 둘 다 가속할 수 있다고 설명한다.

중요한 차이는 다음과 같다.

| 항목 | AutoGaze | PixelPrune |
|---|---|---|
| 주 타깃 | video frame의 중요한 multi-scale patch 선택 | document/GUI 같은 고해상도 이미지의 중복 patch 제거 |
| selector 위치 | 보통 ViT 전 후보 patch 선택이지만, downstream 연결은 mapping 필요 | pre-ViT에서 직접 patch를 줄이는 방향 |
| 기준 | reconstruction/task loss, gazing ratio, autoregressive selection | predictive coding 기반 pixel redundancy |
| Qwen 연결성 | 직접 붙이려면 `video_grid_thw`/position mapping 필요 | Qwen3-VL patch path를 monkey patch하는 레퍼런스 존재 |
| 장점 | video/time-aware selection 가능 | ViT latency까지 줄이는 구조가 명확 |
| 약점 | multi-scale patch index를 downstream token으로 변환해야 함 | 의미 기반 saliency보다 pixel redundancy에 가까움 |

우리 목표가 “새로운 video token selector를 개발하고 기존 모델에 zero-shot으로 붙이는 것”이라면, PixelPrune에서 배울 핵심은 **pre-ViT selector를 기존 Qwen processor/model path에 어떻게 주입했는가**다. 반대로 AutoGaze에서 배울 핵심은 **video에서 어떤 patch를 선택해야 정보 손실이 적은가**다.

### 개념 도식

```text
PixelPrune-style pre-ViT path

image/video frames
  |
  | processor resize / patch grid
  v
pixel-level redundancy selector
  |
  | keep/drop patch map
  v
repacked sparse visual input
  |
  | Qwen ViT with adjusted grid/position
  v
visual features
  |
  v
projector / merger
  |
  v
LLM prefill + generate

gain boundary:
  selector overhead is added
  ViT, projector, LLM can all become cheaper
```

```text
AutoGaze-style video selector path

sampled video frames
  |
  | multi-scale tiling
  v
AutoGaze autoregressive decoder
  |
  | selected multi-scale patch indices
  v
SparseSelectionPlan
  |          |              |
  |          |              |
  v          v              v
dense mask  crop/tile list  token index map
  |          |              |
  v          v              v
zero-fill   zero-shot MLLM  post/pre ViT prune

gain boundary:
  depends on materialization
  dense mask: almost no latency gain
  crop/tile: partial ViT/LLM gain
  post-encoder: LLM gain
  pre-ViT: ViT + LLM gain
```

### Qwen 계열에서 pre-ViT selector가 중요한 이유

Qwen3-VL은 Transformers 경로에서 `pixel_values_videos`, `video_grid_thw`, `get_video_features(...)` 같은 boundary가 비교적 분명하다. 이 구조는 selector 연구에 유리하다. 단, pre-ViT로 patch 수를 줄이면 grid와 position도 같이 바뀌므로, 단순히 pixel tensor만 줄이면 안 된다.

```text
bad sparse injection

drop patches only
  -> old video_grid_thw remains
  -> old position/MRoPE assumption remains
  -> feature position semantics can be wrong

required sparse injection

drop patches
  -> recompute or map grid_thw
  -> preserve original frame/time/xy coordinates
  -> pass correct position ids or equivalent metadata
  -> record exact before/after token counts
```

그래서 Qwen/PixelPrune 축의 PoC는 “빠르게 붙인다”보다 “patch 제거와 grid/position 업데이트가 같이 맞는지 probe한다”가 먼저다.

---

## 범용 Token Selector 설계 방향

궁극적으로 만들 selector는 AutoGaze처럼 특정 모델 안에 묶이지 않고, 아래 모델군에 최대한 zero-shot으로 붙을 수 있어야 한다.

```text
                         +-------------------+
                         | Token Selector    |
                         | AutoGaze-like     |
                         | PixelPrune-like   |
                         | Video selector    |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         | SparseSelection   |
                         | Plan              |
                         +---------+---------+
                                   |
        +--------------------------+--------------------------+
        |                          |                          |
        v                          v                          v
 +-------------+            +-------------+            +-------------+
 | pre-ViT     |            | post-ViT    |            | crop/tile   |
 | patch prune |            | token prune |            | repacking   |
 +------+------+            +------+------+            +------+------+
        |                          |                          |
        v                          v                          v
 +-------------+            +-------------+            +-------------+
 | ViT encoder |            | projector   |            | existing    |
 | cheaper     |            | / MLLM only |            | processor   |
 +------+------+            +------+------+            +------+------+
        |                          |                          |
        +--------------------------+--------------------------+
                                   |
                                   v
                         +-------------------+
                         | MLLM              |
                         +-------------------+
```

이때 selector output은 단순 mask가 아니라 세 계층을 모두 표현해야 한다.

1. **Patch-level plan**: 원본/resize 좌표, frame/time, scale, patch index
2. **Encoder-level plan**: encoder grid, patch size, position id, feature index
3. **MLLM-level plan**: projector 이후 visual token index, LLM context index

이렇게 해야 PixelPrune 같은 pre-ViT selector, AutoGaze 같은 video-aware selector, FrameFusion/PruneVid 같은 post-encoder/post-LLM selector를 같은 벤치마크 표에서 비교할 수 있다.

---

## 현재 AutoGaze Output의 의미

공식 AutoGaze 설명 기준으로 AutoGaze는 비디오에서 frame별 multi-scale patch를 autoregressive하게 선택한다. Hugging Face model card도 output type을 patch index 정수로 설명한다. 프로젝트 페이지는 AutoGaze가 사용자 지정 reconstruction loss threshold 또는 patch budget 내에서 원본 비디오를 복원할 최소 patch set을 고른다고 설명한다.

현재 연결 관점에서 문제는 **patch index만으로는 downstream 모델의 token index가 바로 결정되지 않는다**는 점이다.

필요한 변환은 아래와 같다.

```text
AutoGaze multi-scale patch index
    |
    | needs coordinate mapping
    v
selected patch bbox in resized/tiled frame coordinates
    |
    | needs encoder patch/grid mapping
    v
ViT input patch token index OR ViT output feature token index
    |
    | needs projector/MLLM packing mapping
    v
MLLM visual token index in LLM prefill context
```

따라서 AutoGaze output은 단순 `selected_indices`보다 풍부해야 한다.

---

## 권장 표준 Output: SparseGazePlan

제로샷 연결성을 높이려면 AutoGaze output을 다음 표준 형태로 materialize하는 것이 좋다.

```python
SparseGazePlan = {
    "source_video": {
        "path": str,
        "source_width": int,
        "source_height": int,
        "sampled_frame_indices": list[int],
        "sampled_fps": float | None,
    },
    "preprocess_space": {
        "resize_policy": str,
        "resized_width": int,
        "resized_height": int,
        "tile_grid": [int, int],
        "tile_size": int,
    },
    "patch_space": {
        "autogaze_patch_size": int,
        "encoder_patch_size": int | None,
        "scale_ids": list[int],
        "scale_sizes": list[int],
        "patch_size_mismatch": bool,
    },
    "selected_patches": [
        {
            "frame_index": int,
            "frame_order": int,
            "tile_id": int,
            "scale_id": int,
            "scale_size": int,
            "patch_index": int,
            "bbox_resized_xyxy": [int, int, int, int],
            "bbox_original_xyxy": [float, float, float, float],
            "autoregressive_order": int,
        }
    ],
    "dense_masks": {
        "by_frame_scale": "bool mask or RLE",
    },
    "encoder_mapping": {
        "status": "exact | approximate | not_mapped",
        "encoder_grid_thw": [int, int, int] | None,
        "encoder_patch_indices": list[int] | None,
        "position_ids": object | None,
    },
    "mllm_mapping": {
        "status": "exact | probe_required | not_applicable",
        "visual_feature_indices": list[int] | None,
        "projected_token_indices": list[int] | None,
        "llm_context_indices": list[int] | None,
    },
    "quality_control": {
        "gazing_ratio": float | None,
        "task_loss_requirement": float | None,
        "first_frame_keep_policy": str,
        "reconstruction_loss_estimate": float | None,
    },
}
```

이 형태를 기본으로 두고, downstream별로 네 가지 materialization을 만든다.

| 형태 | 목적 | 장점 | 단점 |
|---|---|---|---|
| `dense_mask_by_scale` | visualization, dense ViT mask, zero-fill ablation | 모델 수정 거의 없음 | ViT compute 감소 없음 |
| `sparse_patch_table` | sparse ViT, patch gather | ViT compute 감소 가능 | position encoding 수정 필요 |
| `selected_crops_or_tiles` | 기존 이미지/비디오 processor에 zero-shot 입력 | 어떤 ViT에도 붙이기 쉬움 | 원본 position/temporal 구조가 약해짐 |
| `post_encoder_token_indices` | ViT output 이후 MLLM token prune | MLLM context 감소 확인 쉬움 | ViT compute 감소 없음 |

---

## 연결 지점별 난이도

### 1. Dense Mask / Zero-fill

선택되지 않은 patch를 0, mean pixel, blur, 또는 background color로 채워서 기존 ViT에 그대로 넣는다.

```text
video frame -> AutoGaze mask -> masked dense frame -> original ViT -> original MLLM
```

- 구현 난이도: 낮음
- 모델 호환성: 높음
- latency gain: 거의 없음
- 용도: accuracy 영향, visualization, mask alignment 검증

이 방식은 “AutoGaze가 고른 영역이 의미 있는가?”를 빠르게 확인하기 좋다. 하지만 dense tensor shape는 그대로라 ViT attention/MLP 계산량은 줄지 않는다.

### 2. Post-Encoder Token Prune

기존 ViT를 그대로 통과시킨 뒤, ViT output feature 또는 projector output token을 AutoGaze mapping으로 줄인다.

```text
video frame -> original ViT -> visual features -> AutoGaze-derived token prune -> projector/MLLM
```

- 구현 난이도: 중간
- 모델 호환성: 중간-높음
- latency gain: MLLM prefill/attention/KV cache 쪽
- ViT gain: 없음

지금 PoC에서 가장 먼저 해야 할 방식이다. 이유는 기존 processor/position encoding을 건드리지 않아도 되고, 사용자가 관심 있는 MLLM context/token 감소를 바로 확인할 수 있기 때문이다.

### 3. Pre-Encoder Sparse Patch

AutoGaze가 고른 patch만 patch embedding으로 만들고, 위치 정보를 같이 넣어 ViT에 통과시킨다.

```text
video frame -> AutoGaze selected patches -> sparse patch embedding + position -> ViT -> MLLM
```

- 구현 난이도: 높음
- 모델 호환성: 낮음-중간
- latency gain: ViT + MLLM 모두 가능
- 핵심 리스크: absolute/2D/3D position encoding, grid shape, RoPE/MRoPE, relative bias, pooling assumptions

일반 dense ViT/SigLIP은 보통 `B x C x H x W` dense image tensor를 받아 내부에서 patchify한다. sparse patch sequence를 그대로 받는 API가 없으면 patch embedding과 position embedding 경로를 직접 열어야 한다. 이때 position이 조금만 어긋나도 feature 의미가 흔들린다.

### 4. Selected Crop/Tile Repacking

선택된 patch들을 원본 좌표와 함께 crop/tile 이미지 묶음으로 만들어 기존 multi-image/video processor에 넣는다.

```text
selected patch bbox -> crop/tile list -> existing image/video processor -> MLLM
```

- 구현 난이도: 낮음-중간
- 모델 호환성: 높음
- latency gain: 입력 crop/tile 개수에 따라 가능
- 핵심 리스크: global context와 원본 위치 손실

제로샷 호환성은 제일 좋다. 단, 모델이 crop의 위치를 모르면 “영상 전체에서 어디인지”를 잃을 수 있다. 따라서 prompt 또는 metadata token으로 frame/time/region 정보를 함께 넣는 방식이 필요하다.

---

## 모델별 연결성 비교

| 모델 계열 | ViT/vision 입력 계약 | position/grid 핵심 | feature/token 수 특성 | 쉬운 적용 | 어려운 적용 |
|---|---|---|---|---|---|
| Qwen3-VL + PixelPrune | Qwen3-VL vision path에 pre-ViT patch pruning 주입 | `video_grid_thw`/MRoPE 정합 필요 | ViT 전 patch 수 자체를 줄이는 레퍼런스 | pre-ViT selector design reference | video-aware sparse grid 일반화 |
| NVILA-HD AutoGaze | native processor 내부 AutoGaze + SigLIP | target scale/patch size 정합 필요 | token shuffle/projector 있음 | native path, keep-all 비교 | 외부 모델에 동일 sparse path 이식 |
| NVILA-Video / LongVILA | 공식 VILA CLI/remote code 경로 | VILA 내부 vision/projector packing probe 필요 | `vila-infer` off부터 확인 | off smoke, post-encoder probe | pre-encoder sparse |
| Qwen2/2.5/3-VL | `pixel_values_videos`, `video_grid_thw` | MRoPE/grid-thw 보존 필요 | `get_video_features` 이후 token insertion | post-encoder token prune | pre-encoder sparse |
| V-JEPA2 ViT | `pixel_values_videos` 기반 video feature extractor | spatiotemporal patch/tubelet position | video-native feature가 강함 | frozen video feature backbone | MLLM projector/adapter 새로 필요 |
| LLaVA-OneVision | processor videos/images | video는 frame별 196 token pooling | pooling 이후 visual token 수가 작음 | post-pool prune | pre-pool sparse 이득 주장 |
| InternVL3 | `pixel_values`, `num_patches_list` | dynamic tiling과 patch count list | tile 단위 동적 resolution | tile/crop-level prune | arbitrary patch sparse |

### Qwen3-VL + PixelPrune

PixelPrune은 지금 가장 중요한 pre-ViT selector reference다. 이유는 단순하다. Qwen3-VL 계열에 대해 모델 로드 전 patch를 적용하는 형태를 제공하고, pruning 위치가 ViT 이전이라 이론적으로 ViT encoder와 LLM 양쪽 gain을 모두 볼 수 있다.

다만 PixelPrune은 document/GUI의 pixel redundancy를 강하게 이용한다. long video understanding에서는 “중복 제거”만으로는 task-relevant event를 충분히 보존하지 못할 수 있다. 따라서 우리 방향은 PixelPrune을 그대로 복제하기보다, 아래 조합으로 확장하는 것이다.

```text
PixelPrune gives:
  pre-ViT injection mechanism
  Qwen3-VL compatibility reference
  patch redundancy baseline

AutoGaze/video selector gives:
  temporal saliency
  event-aware frame/patch choice
  multi-scale patch budget control

our target:
  video-aware pre-ViT selector
  Qwen-style grid/position-safe sparse input
  model-agnostic SparseSelectionPlan
```

PoC 우선순위는 다음과 같다.

1. PixelPrune Qwen3-VL path를 분석해서 “patch 제거 후 grid/position이 어디서 조정되는지” probe한다.
2. 같은 Qwen3-VL에서 AutoGaze/새 selector output을 PixelPrune-style pre-ViT 형태로 materialize할 수 있는지 확인한다.
3. video에서는 temporal coordinate가 필요하므로 `frame_index`, `time_sec`, `grid_t`, `grid_h`, `grid_w`를 selector output 필수 필드로 둔다.

### NVILA-HD / NVILA-Video / LongVILA

NVILA 계열은 “scale-then-compress” 철학이 있고, NVILA-HD + AutoGaze는 이미 native processor에 통합되어 있다. 단, 외부 plugin 구조로 떼어내면 patch size, scale, token shuffle, projector boundary가 모두 명확해야 한다.

권장 순서:

1. `vila-infer` off smoke
2. internal processor input shape / vision feature shape / projector output shape probe
3. post-encoder token prune
4. pre-encoder sparse는 patch position alignment가 검증된 뒤 시도

### Qwen2/2.5/3-VL

Transformers 문서 기준 Qwen 계열은 video input에서 `pixel_values_videos`와 `video_grid_thw`를 사용하고, Qwen3-VL은 `get_video_features(pixel_values_videos, video_grid_thw)` 경로를 제공한다. Qwen2.5-VL도 `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, `rope_deltas`, `second_per_grid_ts` 같은 grid/time 관련 필드가 forward contract에 들어간다.

따라서 Qwen 계열은 post-encoder token prune이 가장 좋은 1차 목표다.

```text
processor -> pixel_values_videos + video_grid_thw
          -> model.get_video_features(...)
          -> selected visual feature indices
          -> visual token insertion into LLM context
```

pre-encoder sparse는 `video_grid_thw`, MRoPE, temporal position이 같이 바뀌므로 별도 sparse-grid 재정의가 필요하다. 이것을 하지 않고 patch만 줄이면 position semantics가 깨질 수 있다.

### V-JEPA2 기반 Video ViT

V-JEPA2는 video-native ViT encoder 후보로 따로 봐야 한다. Hugging Face 문서 기준 V-JEPA2는 `AutoVideoProcessor`와 `AutoModel`로 video feature extraction이 가능하고, config에는 `patch_size`, `crop_size`, `frames_per_clip`, `tubelet_size`가 명시된다. 즉 image-only ViT보다 video temporal structure를 직접 다루기 좋은 축이다.

하지만 V-JEPA2는 일반 MLLM에 바로 연결된 표준 projector가 있는 모델이라기보다, 강한 video feature backbone에 가깝다. 그래서 “기존 MLLM에 zero-shot으로 붙이기” 관점에서는 난이도가 높고, “새 selector를 개발해서 video feature 품질/효율을 검증하기” 관점에서는 가치가 높다.

권장:

- 1차: frozen V-JEPA2 feature extractor로 dense feature/token count와 latency probe
- 2차: V-JEPA2 `context_mask`/`target_mask` 개념을 selector 실험의 reference로 활용
- 3차: MLLM 연결은 lightweight projector 또는 retrieval/caption adapter를 별도 연구 축으로 분리

```text
V-JEPA2 branch

video frames
  |
  v
video-native patch/tubelet encoder
  |
  v
spatiotemporal features
  |
  +--> action/video benchmark probe
  |
  +--> future projector/MLLM adapter

good for:
  selector quality research
  temporal feature retention study

hard for:
  immediate zero-shot MLLM drop-in
```

### LLaVA-OneVision

Transformers 문서 기준 LLaVA-OneVision은 video를 지원하지만, video frame별 token을 196개로 pooling해 memory 효율을 얻는다. 이 구조에서는 이미 token 수가 한번 줄어든 뒤라 AutoGaze의 post-pool gain은 제한적일 수 있다.

권장:

- 1차: post-pool token prune으로 LLM context 감소 확인
- 2차: pre-pool sparse는 큰 이득을 주장하기 어렵고, pooling 전 grid mapping probe가 필요

### InternVL3

InternVL3 계열은 `pixel_values`와 `num_patches_list`를 사용해 dynamic tiling을 표현한다. arbitrary patch sparse보다는 tile/crop level pruning이 자연스럽다.

권장:

1. AutoGaze selected patch를 tile-level coverage로 aggregate
2. selected tile/crop만 `pixel_values`에 넣음
3. `num_patches_list`를 frame/image별로 재계산
4. arbitrary patch sparse는 dynamic tiling 내부 ViT position probe 이후 시도

---

## Sparse Patch를 쉽게 받을 수 있는 ViT/MLLM 조건

Sparse AutoGaze output을 쉽게 적용할 수 있는 모델은 다음 조건을 만족해야 한다.

1. Vision encoder가 dense image tensor뿐 아니라 patch/token embedding sequence를 받을 수 있다.
2. Position embedding 또는 position id를 외부에서 지정하거나 gather할 수 있다.
3. ViT output token과 MLLM visual token insertion 위치가 명확하다.
4. Visual token 수가 variable length여도 projector/MLLM이 shape를 받아들인다.
5. Temporal position이 frame index와 분리되어 metadata로 들어간다.

이 조건을 만족하지 않는 일반 dense ViT는 zero-shot pre-encoder sparse가 어렵다. 대신 다음 중 하나로 우회해야 한다.

- dense mask / zero-fill
- selected crop/tile repacking
- post-encoder token prune

현재 후보 중 쉬운 순서는 다음과 같다.

```text
1. Qwen3-VL + PixelPrune reference 분석
2. Qwen 계열 post-encoder token prune
3. InternVL3 tile/crop-level sparse
4. VILA/NVILA post-encoder token prune
5. LLaVA-OneVision post-pool prune
6. Qwen-style pre-ViT sparse selector prototype
7. V-JEPA2 video ViT selector quality branch
```

---

## AutoGaze Output 형태 추천

현재 multi-scale patch index output은 유지하되, plugin 적용용으로는 아래 세 가지를 반드시 같이 저장해야 한다.

### A. Coordinate-rich patch table

각 selected patch가 어느 frame/tile/scale/원본 좌표인지 명확히 저장한다.

필수 필드:

- `frame_index`
- `frame_order`
- `tile_id`
- `scale_id`
- `scale_size`
- `patch_index`
- `bbox_resized_xyxy`
- `bbox_original_xyxy`
- `autoregressive_order`

### B. Encoder mapping table

AutoGaze patch가 downstream encoder token과 어떻게 대응하는지 저장한다.

필수 필드:

- `encoder_name`
- `encoder_patch_size`
- `encoder_grid_thw`
- `mapping_status`
- `encoder_patch_indices`
- `position_ids`
- `mapping_error_reason`

### C. MLLM visual token mapping

MLLM context에서 어떤 visual token을 줄일 수 있는지 저장한다.

필수 필드:

- `visual_feature_shape_before`
- `visual_feature_shape_after`
- `projected_feature_shape_before`
- `projected_feature_shape_after`
- `llm_context_tokens_before`
- `llm_context_tokens_after`
- `visual_token_indices_in_context`

이렇게 하면 같은 AutoGaze run을 세 방식으로 재사용할 수 있다.

```text
AutoGaze once
  |-- dense mask visualization
  |-- selected crop/tile zero-shot inference
  |-- post-encoder token prune
  `-- pre-encoder sparse prototype
```

---

## Report에 반드시 들어가야 할 비교 항목

### Resolution / Patch / Position

| 항목 | 이유 |
|---|---|
| source resolution | 4K/FHD/resize 여부가 후보 patch 수를 결정 |
| resized resolution | 실제 AutoGaze/ViT 입력 공간 |
| tile grid / tile size | crop/tile repacking과 patch mapping 기준 |
| AutoGaze patch size | multi-scale index의 기본 단위 |
| encoder patch size | ViT token grid와 position embedding 기준 |
| patch size mismatch | AutoGaze index와 encoder token index가 1:1인지 판단 |
| encoder grid THW | Qwen/Video 모델의 RoPE/MRoPE 기준 |
| position encoding type | absolute/2D/3D/RoPE/MRoPE에 따라 sparse 난이도 결정 |

### Feature / Token Count

| 항목 | 이유 |
|---|---|
| AutoGaze candidate patch tokens | selector 입력 분모 |
| AutoGaze selected patch tokens | selector output |
| encoder keep-all patch tokens | ViT 계산량 baseline |
| encoder actual patch tokens | pre-encoder sparse가 성공했는지 |
| ViT output feature tokens | post-encoder prune 분모 |
| projected visual tokens | projector 이후 MLLM 입력 전 |
| LLM context visual tokens | prefill/KV cache 계산 분모 |
| text tokens | visual token 감소와 총 context 감소를 분리 |

### Latency / Memory

| 항목 | 기대 gain |
|---|---|
| AutoGaze forward | selector overhead |
| ViT encoder | pre-encoder sparse에서만 감소 |
| projector | feature/token prune 위치에 따라 감소 |
| LLM prefill / TTFT | post-encoder prune에서도 감소 가능 |
| KV cache | visual context 감소에 따라 감소 |
| peak memory | resident model + activation + KV cache 영향을 분리 |

---

## 구현 태스크 계획

### Phase A: SparseGazePlan Schema

- [x] `repro/plugins/gaze_plan.py` 추가
- [x] `SparseSelectionPlan`, `SelectedPatch`, `EncoderMapping`, `MllmMapping` dataclass 정의
- [x] JSON serialization 테스트
- [ ] 기존 visualization output과 연결

Acceptance:

- AutoGaze output 하나로 dense mask, crop list, encoder mapping, MLLM mapping을 모두 만들 수 있다.

### Phase B: PixelPrune / Qwen Pre-ViT Reference Probe

- [x] Qwen3-VL PixelPrune pre-ViT 실행 gate 추가: hook 성공 시 Qwen native generation, 실패 시 dense 실행 차단
- [ ] PixelPrune Qwen3-VL hook 위치 조사: processor, patch embedding, vision forward, grid metadata 중 어디를 바꾸는지 기록
- [ ] Qwen3-VL에서 PixelPrune on/off token count, ViT latency, LLM prefill latency를 같은 입력으로 비교
- [ ] `video_grid_thw`, MRoPE/position 관련 tensor가 pruning 전후 어떻게 변하는지 shape report 생성
- [ ] image/document 기준 PixelPrune과 video frame 기준 selector의 차이를 `selector_type`으로 분리
- [ ] PixelPrune-style pre-ViT materializer interface 초안 작성

Acceptance:

- Qwen3-VL에서 pre-ViT selector를 붙일 때 반드시 수정해야 하는 metadata와 수정하지 않아도 되는 metadata가 구분된다.

### Phase C: Patch/Position Probe

- [ ] `repro/plugins/connection_probe.py` 추가
- [ ] 모델별 probe 공통 output 정의
- [ ] Qwen: `pixel_values_videos`, `video_grid_thw`, `get_video_features` shape 기록
- [ ] InternVL3: `pixel_values`, `num_patches_list`, dynamic tile shape 기록
- [ ] VILA/NVILA: vision feature shape, projector shape, visual token insertion boundary 기록
- [ ] LLaVA-OneVision: video pooling 전/후 token 수 기록
- [ ] V-JEPA2: `patch_size`, `tubelet_size`, `frames_per_clip`, output feature length 기록

Acceptance:

- 각 모델 run JSON에 `connection_report.resolution`, `connection_report.position`, `connection_report.feature_counts`가 생긴다.

### Phase D: Zero-shot Materialization

- [ ] dense mask / zero-fill mode
- [ ] selected crop/tile list mode
- [ ] metadata prompt mode: frame/time/region index를 text로 추가
- [ ] 동일 sample에서 original, masked, crop/tile 세 결과 비교

Acceptance:

- 모델 수정 없이 AutoGaze-selected regions만 이용한 zero-shot ablation이 가능하다.

### Phase E: Post-Encoder Token Prune

- [x] Qwen3-VL AutoGaze post-encoder attachment PoC 구현: `SparseSelectionPlan`, feature packing probe, visual token estimate 기록
- [x] Qwen 계열 post-encoder feature prune/generate 실험 경로 추가: `--enable-qwen-prune-generate`
- [x] AutoGaze `SparseSelectionPlan.selected_patches`를 Qwen `video_grid_thw` visual feature index로 매핑하는 helper 추가
- [ ] 실제 AutoGaze runner output에서 concrete `selected_patches`를 저장해 Qwen path에 공급
- [ ] `get_video_features` output token index와 AutoGaze patch mapping 연결
- [x] selected feature만 MLLM context에 insert하는 `inputs_embeds` bridge helper 추가
- [x] before/after LLM context와 visual token count 기록
- [ ] TTFT, peak memory를 Qwen prune-generate path에서 CUDA smoke로 검증
- [x] 실패 시 `failed_qwen_prune_generate`로 원인 기록

Acceptance:

- ViT latency는 그대로지만 LLM visual token, prefill context, KV cache estimate가 줄어든다. 현재 구현은 AutoGaze index mapping 전 단계라 `gazing_ratio` 기반 placeholder selection으로 표시된다.

### Phase F: InternVL3 Tile Sparse

- [ ] AutoGaze patch coverage를 tile-level score로 aggregate
- [ ] selected tile만 `pixel_values`에 넣기
- [ ] `num_patches_list` 재계산
- [ ] off vs sparse tile 비교

Acceptance:

- arbitrary patch sparse 전 단계로 tile-level sparse inference를 실행할 수 있다.

### Phase G: Pre-Encoder Sparse ViT Prototype

- [ ] SigLIP/NVILA patch embedding을 직접 호출하는 prototype 작성
- [ ] Qwen3-VL/PixelPrune-style sparse-grid prototype 후보 추가
- [ ] selected patch embedding gather
- [ ] position embedding gather/interpolation
- [ ] full dense ViT output과 sparse reconstructed output 비교
- [ ] patch size mismatch report

Acceptance:

- 특정 encoder 1개에서라도 ViT compute 감소를 실제 측정한다.

### Phase H: V-JEPA2 Video ViT Branch

- [ ] V-JEPA2 dense feature extraction smoke 추가
- [ ] selector 적용 전후 frame/tubelet feature retention report 설계
- [ ] `context_mask`/`target_mask`를 selector 실험 기준으로 사용할 수 있는지 확인
- [ ] MLLM 직접 연결은 별도 projector/adapter 필요 항목으로 기록

Acceptance:

- V-JEPA2는 “즉시 MLLM zero-shot adapter”가 아니라 “video-native selector 품질 검증 backbone”으로 분리해 평가할 수 있다.

### Phase I: HLVid / Long Video Benchmark 연결

- [ ] `plugin_hlvid_benchmark`에 connection report merge
- [x] `--modes`에 `qwen3-vl-pixelprune-pre-vit`, `qwen3-vl-autogaze-prune-generate` 추가
- [ ] `--modes`에 `internvl3-tile-sparse` 추가
- [ ] Markdown report에 resolution/position/feature/token 표 추가
- [ ] failed/probe_required/mapping_failed를 score denominator와 분리

Acceptance:

- HLVid limit3에서 모델별 “왜 붙기 쉬운지/어려운지”를 수치와 함께 볼 수 있다.

---

## 우선순위 제안

1. `SparseGazePlan` schema부터 만든다.
2. PixelPrune/Qwen3-VL pre-ViT reference를 probe한다.
3. Qwen3-VL post-encoder probe를 실제 shape probe로 바꾼다.
4. Qwen3-VL post-encoder token prune을 1개 sample에서 실행한다.
5. InternVL3 tile-level sparse를 시도한다.
6. VILA/NVILA는 `vila-infer` 내부 feature packing boundary가 확인된 뒤 post-encoder prune을 붙인다.
7. Qwen-style pre-ViT sparse selector prototype으로 ViT latency 감소를 확인한다.
8. V-JEPA2는 video-native selector 품질 검증 branch로 분리한다.

이 순서가 좋은 이유는, PixelPrune이 Qwen 계열에서 pre-ViT selector를 붙이는 최신 reference 역할을 하고, Qwen 계열은 공식 Transformers API에서 `video_grid_thw`와 `get_video_features` boundary가 비교적 명확하기 때문이다. InternVL3는 `num_patches_list`라는 dynamic tile metadata가 있어 tile-level sparse가 자연스럽다. 반면 VILA/NVILA는 공식 CLI 경로로 off smoke는 쉽지만, feature packing boundary를 열기 위해 remote code 내부 probe가 필요하다.

---

## 참고 소스

- AutoGaze paper/project: https://arxiv.org/abs/2603.12254, https://autogaze.github.io/
- AutoGaze model card: https://huggingface.co/nvidia/AutoGaze
- NVILA project: https://nvlabs.github.io/VILA/
- PixelPrune paper/code: https://arxiv.org/abs/2604.00886, https://github.com/OPPO-Mente-Lab/PixelPrune
- Qwen3-VL Transformers docs: https://huggingface.co/docs/transformers/v5.3.0/model_doc/qwen3_vl
- Qwen2.5-VL Transformers docs: https://huggingface.co/docs/transformers/v5.7.0/en/model_doc/qwen2_5_vl
- V-JEPA2 Transformers docs/code: https://huggingface.co/docs/transformers/main/model_doc/vjepa2, https://github.com/facebookresearch/vjepa2
- FrameFusion project/code: https://thu-nics.github.io/FrameFusion_Project_Page/, https://github.com/thu-nics/FrameFusion
- PruneVid code: https://github.com/visual-ai/prunevid
- LLaVA-OneVision Transformers docs: https://huggingface.co/docs/transformers/model_doc/llava_onevision
- InternVL3 model card/example path: https://huggingface.co/FriendliAI/InternVL3-8B-Instruct
