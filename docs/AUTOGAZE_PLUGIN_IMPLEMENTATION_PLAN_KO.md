# AutoGaze Plugin Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** AutoGaze를 `NVILA-HD-Video` 전용 processor 의존 구현에서 분리해, token selector / vision encoder / MLLM / benchmark / profiler를 교체 가능한 플러그인 구조로 만든다.

**Architecture:** 기존 `repro/nvila_runner.py`는 NVILA-HD/AutoGaze 재현용 안정 러너로 유지한다. 확장 실험은 `repro/flexible_runner.py`에서 시작하고, 모델 식별, 비디오 전처리, AutoGaze, Vision, projector, MLLM, task, reporter를 작은 adapter 단위로 붙인다. 단기적으로는 `inspect` 모드에서 `NVILA-8B-Video`, `LongVILA`, `Qwen2-VL`류에 대한 AutoGaze on/off 조합을 명확히 기록하고, 장기적으로는 `pre_encoder_sparse`와 `post_encoder_token_prune` 두 integration mode를 실제 실행 경로로 구현한다.

**Tech Stack:** Python, PyTorch, Transformers remote code, PyAV, OmegaConf, pytest, JSON/Markdown reporting.

---

## 현재 상태 요약

현재 구현은 재현 실험에 필요한 기능이 빠르게 붙으면서 대부분의 orchestration이 `repro/nvila_runner.py`에 집중되어 있다. 다만 앞으로의 확장 작업은 이 파일을 계속 키우지 않고, 새 확장 러너와 plugin API로 분리한다.

- stable NVILA runner: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py)
- flexible runner inspect surface: [repro/flexible_runner.py](/Users/mrc/Documents/New%20project/repro/flexible_runner.py)
- plugin API contract: [repro/plugin_api.py](/Users/mrc/Documents/New%20project/repro/plugin_api.py)
- plugin registry: [repro/plugin_registry.py](/Users/mrc/Documents/New%20project/repro/plugin_registry.py)
- `StageProfiler`: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:227)
- run identity / component identity: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:804)
- latency hierarchy: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:911)
- single summary: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:1275)
- processor kwargs: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:2685)
- stream profile: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:4270)
- single inference: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:4856)
- HLVid benchmark: [repro/nvila_runner.py](/Users/mrc/Documents/New%20project/repro/nvila_runner.py:5056)

이미 분리된 보조 모듈은 유지한다.

- HLVid manifest/scoring: [repro/hlvid.py](/Users/mrc/Documents/New%20project/repro/hlvid.py)
- batch benchmark wrapper: [repro/hlvid_batch_benchmark.py](/Users/mrc/Documents/New%20project/repro/hlvid_batch_benchmark.py)
- Markdown report: [repro/markdown_report.py](/Users/mrc/Documents/New%20project/repro/markdown_report.py)
- visualization: [repro/gaze_visualization.py](/Users/mrc/Documents/New%20project/repro/gaze_visualization.py)
- AutoGaze timing 비교: [repro/autogaze_timing_compare.py](/Users/mrc/Documents/New%20project/repro/autogaze_timing_compare.py)
- VideoQA schema: [repro/videoqa_task_schema.py](/Users/mrc/Documents/New%20project/repro/videoqa_task_schema.py)

현재 model family 의미는 다음처럼 분리한다.

| model family | 의미 | AutoGaze 의미 | paper baseline 여부 |
|---|---|---|---|
| `nvila-hd-video-autogaze` | `nvidia/NVILA-8B-HD-Video` remote processor 경로 | native AutoGaze 또는 keep-all ablation | HD AutoGaze row 후보 |
| `nvila-video-baseline` | 논문 table의 `NVILA-8B-Video` baseline 재현 후보 | `not_applicable` | 예 |
| `nvila-video-plugin` | 튜닝되지 않은 `NVILA-8B-Video`에 AutoGaze를 붙이는 실험 | on/off 실험 | 아니오 |
| `longvila` | LongVILA에 AutoGaze를 붙이는 실험 | on/off 실험 | 아니오 |
| `qwen2-vl` | Qwen2-VL류 모델에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |
| `qwen3-vl` | Qwen3-VL dense 계열에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |
| `qwen3-vl-moe` | Qwen3-VL MoE 계열에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |
| `qwen2.5-vl` | Qwen2.5-VL dense 계열에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |
| `llava-onevision` | LLaVA-OneVision SigLIP + Qwen2 계열에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |
| `internvl3` | InternVL3 동적 타일링 계열에 AutoGaze를 붙이는 후속 실험 | on/off 실험 | 아니오 |

Qwen3-VL 검토 메모:

- 공식 Transformers 문서 기준 Qwen3-VL forward는 `pixel_values_videos`, `video_grid_thw`, `mm_token_type_ids`를 받는다.
- 같은 문서에서 `get_video_features(pixel_values_videos, video_grid_thw)` entrypoint를 제공한다.
- PixelPrune은 Qwen3-VL 계열에서 모델 로드 전에 `apply_pixelprune(model="qwen3_vl")`를 호출하고 `PIXELPRUNE_ENABLED=true`로 pre-ViT pruning을 적용하는 선례다.
- 따라서 1차 AutoGaze 부착은 기존 vision encoder를 그대로 둔 `post_encoder_token_prune`가 가장 안전하지만, Qwen3에 한해서는 PixelPrune-style pre-ViT hook을 reference path로 함께 열어둔다.
- AutoGaze 자체의 `pre_encoder_sparse`는 `video_grid_thw`와 temporal/spatial position semantics를 보존해야 하므로 별도 probe가 필요하다.

다른 MLLM 현황:

| family | native/off | 1차 AutoGaze on | pre-encoder sparse | 메모 |
|---|---|---|---|---|
| `qwen2-vl` | native/off runtime ready | `post_encoder_token_prune` probe ready | probe required | AutoGaze on은 `get_video_features` 이후 visual token insertion probe로 기록 |
| `qwen2.5-vl` | native/off runtime ready | `post_encoder_token_prune` probe ready | probe required | Qwen2.5-VL도 `get_video_features(pixel_values_videos, video_grid_thw)` 경로 |
| `qwen3-vl` | native/off runtime ready | `post_encoder_token_prune` probe ready | PixelPrune reference available | PixelPrune pre-ViT hook 선례 있음 |
| `qwen3-vl-moe` | native/off runtime ready | `post_encoder_token_prune` probe ready | PixelPrune reference available | Qwen3-VL 계열 hook 공유 |
| `llava-onevision` | single dry-run ready | `post_encoder_token_prune` | hard | video는 frame당 196 token pooling, pre-sparse 이득 주장 난이도 높음 |
| `nvila-video-plugin` | native/off external CLI ready | `post_encoder_token_prune` | probe required | 공식 VILA `vila-infer` CLI로 off 실행, AutoGaze on은 feature packing probe 필요 |
| `internvl3` | native/off external CLI ready | `post_encoder_token_prune` probe ready | probe required | `repro.internvl3_off_infer` helper로 off 실행, AutoGaze on은 dynamic tiling probe 필요 |
| `longvila` | native/off external CLI ready | `post_encoder_token_prune` | probe required | 공식 VILA `vila-infer` CLI로 off 실행, AutoGaze on은 feature packing probe 필요 |

Planned 계열 adapter의 현재 의미:

- `nvila-video`, `longvila`는 native/off에 한해 공식 `vila-infer` CLI를 호출하는 external CLI adapter다.
- `nvila-video`, `longvila`에서 AutoGaze on을 요청하면 실제 generate 대신 `probe_required` 결과를 만든다.
- `internvl3`는 `repro.internvl3_off_infer` helper CLI를 통해 native/off 실행을 시도한다.
- `internvl3`와 Qwen 계열에서 AutoGaze on을 요청하면 실제 generate 대신 `probe_required` 결과를 만든다.
- 결과 JSON에는 `feature_packing_probe`가 들어가며, CUDA 머신에서 다음 단계로 확인해야 할 processor input, vision output, visual token packing boundary, token accounting target을 기록한다.
- paper baseline 후보인 `nvila-video-baseline` + `nvila-video` 조합은 `autogaze_applicability=not_applicable_for_paper_baseline`으로 표시해 plugin off/on 실험과 섞이지 않게 한다.

---

## 핵심 원칙

1. `paper baseline`과 `plugin experiment`를 절대 섞지 않는다.
2. AutoGaze off는 두 종류로 분리한다.
   - `not_applicable`: paper baseline처럼 AutoGaze가 실험 정의상 없는 경우
   - `plugin_off_native`: 같은 모델에 AutoGaze를 붙이기 전 native 비교군
3. Token 감소를 주장하려면 같은 분모를 쓴다.
   - AutoGaze 입력 후보 patch token
   - vision encoder keep-all patch token
   - 실제 vision encoder 입력 token
   - MLLM visual token
4. Latency는 항상 3-part total을 유지한다.
   - `total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms`
   - `generate_ms`는 vision encoder, mm projector, LLM forward/decode를 포함한다.
   - `autogaze_total_ms`는 generate에 포함하지 않는다.
5. 모델별 remote code 차이를 숨기지 않는다.
   - 가능한 metric은 값을 기록한다.
   - 불가능한 metric은 `null`과 `metric_status.reason`을 함께 기록한다.

---

## 목표 파이프라인

```text
Video Path(s)
    |
    v
+--------------------------+
| VideoSourceAdapter       |
| - metadata               |
| - sampled frame indices  |
| - seek/scan decode       |
+-------------+------------+
              |
              v
+--------------------------+
| FrameTransformAdapter    |
| - resize policy          |
| - tiling policy          |
| - tensorize              |
+-------------+------------+
              |
              v
+--------------------------+       off / not_applicable
| TokenSelectorAdapter     |----------------------+
| - keep-all               |                      |
| - AutoGaze               |                      |
| - external mask          |                      |
+-------------+------------+                      |
              | selected patch positions          |
              v                                   |
+--------------------------+                      |
| VisionEncoderAdapter     |<---------------------+
| - full encode            |
| - pre-encoder sparse     |
| - post-encoder prune     |
+-------------+------------+
              |
              v
+--------------------------+
| MMProjectorAdapter       |
| - model-native projector |
| - token shuffle metadata |
+-------------+------------+
              |
              v
+--------------------------+
| MllmAdapter              |
| - prompt packing         |
| - generate               |
| - TTFT / KV cache        |
+-------------+------------+
              |
              v
+--------------------------+
| TaskAdapter + Reporter   |
| - HLVid scoring          |
| - JSON / Markdown        |
| - visualization assets   |
+--------------------------+
```

---

## 플러그인 계약

### 1. ExperimentSpec

모든 adapter가 같은 실행 정보를 보도록 `ExperimentSpec`을 만든다.

권장 파일:

- Create: `repro/plugin_api.py`
- Test: `tests/test_plugin_api.py`

필수 필드:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ModelFamily = Literal[
    "nvila-hd-video-autogaze",
    "nvila-video-baseline",
    "nvila-video-plugin",
    "longvila",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3-vl-moe",
    "llava-onevision",
    "internvl3",
]

TokenSelectorKind = Literal["none", "keep-all", "autogaze", "external-mask"]
IntegrationLevel = Literal["none", "native_processor", "pre_encoder_sparse", "post_encoder_token_prune", "planned_plugin"]

@dataclass(frozen=True)
class ExperimentSpec:
    model_family: ModelFamily
    model_path: str
    token_selector_kind: TokenSelectorKind
    token_selector_path: str | None
    vision_encoder_kind: str
    vision_encoder_path: str | None
    mllm_kind: str
    mllm_path: str
    integration_level: IntegrationLevel
    num_video_frames: int
    num_thumbnail_frames: int
    max_tiles_video: int
    resize_longest_edge: int | None
    resize_shortest_edge: int | None
    output_dir: Path
```

Acceptance:

- `nvila-video-baseline`은 `token_selector_kind="none"`, `integration_level="none"`이어야 한다.
- `nvila-video-plugin`, `longvila`, `qwen2-vl`, `qwen2.5-vl`, `qwen3-vl`, `qwen3-vl-moe`, `llava-onevision`, `internvl3`에서 AutoGaze on이면 `integration_level`은 `planned_plugin`, `pre_encoder_sparse`, `post_encoder_token_prune` 중 하나여야 한다.

### 2. VideoSourceAdapter

비디오 metadata, frame sampling, decode strategy를 담당한다. 긴 HLVid 4K 영상에서는 전체 scan decode를 피하고 metadata 기반 seek sampling을 우선한다.

권장 파일:

- Create: `repro/plugins/video_source.py`
- Move from: `repro/nvila_runner.py`의 `uniform_sample_indices`, `build_seek_decode_groups`, `stream_pts_per_frame`, `frame_index_to_pts`, `pts_to_frame_index`
- Test: `tests/test_video_source_plugin.py`

출력 계약:

```python
@dataclass
class DecodedFrames:
    frames: list["PIL.Image.Image"]
    source_width: int
    source_height: int
    source_frame_count: int | None
    sampled_indices: list[int]
    decode_strategy: str
    decode_ms: float
```

Acceptance:

- `num_video_frames=128`이면 sample index가 비디오 전체 길이에 균등 분포한다.
- seek decode는 가능한 경우 마지막 프레임 근처까지 전체 decode queue를 만들지 않는다.
- decode time은 `video_decode_ms`로만 기록하고, preprocessing total에는 nested child로만 포함한다.

### 3. FrameTransformAdapter

resize, tiling, tensorize를 담당한다.

권장 파일:

- Create: `repro/plugins/frame_transform.py`
- Move from: `repro/nvila_runner.py`의 resize/tile metadata 관련 함수
- Test: `tests/test_frame_transform_plugin.py`

중요 정책:

- `without resize`: 원본을 tile grid로 chop한 뒤 tile을 model target size로 resize한다.
- `with resize`: 먼저 전체 frame을 downscale한 뒤 chop하고 tile을 target size로 resize한다.
- `max_tiles_video=1`은 고해상도 정보를 보존하는 tile selection이 아니라 실질적으로 single-resized-view에 가깝다.

Acceptance:

- source frame size, resized frame size, tile count, tile size, tensor dtype을 report에 기록한다.
- 224/392/448 입력이 downstream target scale과 어떻게 연결되는지 `patch_space_metadata`에 남긴다.

### 4. TokenSelectorAdapter

AutoGaze, keep-all, none, external mask를 같은 인터페이스로 감싼다.

권장 파일:

- Create: `repro/plugins/token_selector.py`
- Move from: `processor_kwargs`, AutoGaze timing hooks, keep-all gazing info helpers
- Test: `tests/test_token_selector_plugin.py`

입력:

```python
@dataclass
class TokenSelectorInput:
    frames_or_tiles: object
    frame_indices: list[int]
    patch_space: "PatchSpace"
    gazing_ratio: float | list[float] | None
    task_loss_requirement: float | None
```

출력:

```python
@dataclass
class TokenSelectorOutput:
    selected_positions: object | None
    selected_mask_by_scale: dict[int, object] | None
    raw_patch_tokens: int
    selected_patch_tokens: int
    reduction_ratio: float | None
    latency_ms: float
    peak_memory_bytes: int | None
    status: str
```

Integration modes:

- `none`: paper baseline. AutoGaze metric은 `not_applicable`.
- `keep-all`: plugin off 또는 HD keep-all ablation. selected token은 raw token과 같다.
- `autogaze/native_processor`: NVILA-HD remote processor 내부 AutoGaze 경로.
- `autogaze/pre_encoder_sparse`: AutoGaze selected positions만 vision encoder에 입력한다.
- `autogaze/post_encoder_token_prune`: full vision encode 후 selected visual tokens만 MLLM에 넘긴다.

Acceptance:

- `nvila-video-baseline`에서 AutoGaze kwargs가 processor/model load로 들어가지 않는다.
- `nvila-video-plugin`과 `longvila`에서 AutoGaze on/off가 같은 input frames로 비교된다.
- `gazing_ratio` 변경이 selected patch token에 반영된다.
- 첫 프레임 강제 keep-all 여부를 config로 기록한다.

### 5. VisionEncoderAdapter

SigLIP 또는 모델별 vision tower를 감싼다.

권장 파일:

- Create: `repro/plugins/vision_encoder.py`
- Test: `tests/test_vision_encoder_plugin.py`

Adapter 종류:

| adapter | target |
|---|---|
| `nvila-hd-siglip` | NVILA-HD의 SigLIP tower, patch size 14, target scales 56/112/196/392 |
| `nvila-video-vision` | NVILA-8B-Video baseline/plugin의 vision path |
| `longvila-siglip` | LongVILA의 vision tower |
| `mock-vision` | unit/smoke test용 |

출력:

```python
@dataclass
class VisionEncoderOutput:
    visual_features: object
    visual_token_count: int
    raw_patch_tokens: int
    selected_patch_tokens: int
    latency_ms: float
    peak_memory_bytes: int | None
    compute_estimate: dict[str, float | None]
```

Acceptance:

- full vs gazed의 attention/MLP MAC estimate를 같은 patch/token denominator로 비교한다.
- patch size 14/16 mismatch가 있으면 report에 `patch_space_mismatch=true`로 남긴다.
- post-encoder pruning mode에서는 vision encoder latency는 줄지 않고 MLLM context만 줄어든다고 명시한다.

### 6. MMProjectorAdapter

Vision feature가 MLLM token space로 들어가기 전 projector와 token shuffle metadata를 기록한다.

권장 파일:

- Create: `repro/plugins/mm_projector.py`
- Test: `tests/test_mm_projector_plugin.py`

기록 항목:

- input visual feature shape
- output visual token count
- token shuffle factor
- projector latency
- projector memory

Acceptance:

- NVILA-HD의 token shuffle 9 기준을 report에 남긴다.
- token shuffle 전에 몇 token이 있었고, 후에 MLLM에 몇 visual token이 들어갔는지 기록한다.

### 7. MllmAdapter

Prompt packing, generate, TTFT, prefill context, KV cache estimate를 담당한다.

권장 파일:

- Create: `repro/plugins/mllm.py`
- Test: `tests/test_mllm_plugin.py`

Adapter 종류:

| adapter | target |
|---|---|
| `nvila-hd` | `nvidia/NVILA-8B-HD-Video` |
| `nvila-video` | `Efficient-Large-Model/NVILA-8B-Video` 또는 local path |
| `longvila` | LongVILA local/HF path |
| `mock-mllm` | unit/smoke test용 |

Output contract:

```python
@dataclass
class MllmOutput:
    text: str
    prompt: str
    input_token_count: int | None
    visual_token_count: int | None
    prefill_context_length: int | None
    ttft_ms: float | None
    generate_ms: float
    llm_forward_ms: float | None
    peak_memory_bytes: int | None
    kv_cache_estimate_bytes: int | None
```

Acceptance:

- `measure_ttft`가 켜지면 TTFT와 decode-after-TTFT estimate를 분리한다.
- SDPA OOM이 발생하면 `failed_reason`, `oom_stage=llm_attention_or_prefill`로 기록한다.
- context/token limit risk를 preflight와 실제 run summary에 같이 기록한다.

### 8. TaskAdapter

HLVid 이후 다른 VideoQA 계열 task를 붙이기 위한 schema layer다.

권장 파일:

- Existing: [repro/videoqa_task_schema.py](/Users/mrc/Documents/New%20project/repro/videoqa_task_schema.py)
- Extend: `repro/plugins/task_adapter.py`
- Test: `tests/test_task_adapter_plugin.py`

공통 schema:

```python
@dataclass
class VideoQASample:
    video_path: str
    question: str
    answer: str
    question_id: str | None
    choices: list[str] | None
    category: str | None
    duration: float | None
    source: str | None
```

Acceptance:

- HLVid mp4 folder + manifest parquet 조합을 읽는다.
- video root에 mp4만 있는 경우도 manifest의 video id와 매칭한다.
- multiple-choice parse 실패와 model failure를 score denominator에서 분리 기록한다.

### 9. Profiler + Metrics

모든 adapter가 같은 profiler event name을 사용해야 한다.

권장 파일:

- Create: `repro/plugins/profiler.py`
- Move from: `StageProfiler`, latency accounting helpers
- Test: `tests/test_profiler_plugin.py`

표준 latency hierarchy:

```text
total_ms
├─ video_preprocess_without_autogaze_ms
│  ├─ video_decode_ms
│  ├─ frame_resize_ms
│  ├─ video_tiling_ms
│  └─ tensorize_ms
├─ autogaze_total_ms
│  ├─ autogaze_forward_ms
│  └─ autogaze_postprocess_ms
└─ generate_ms
   ├─ vision_encoder_ms
   ├─ mm_projector_ms
   ├─ llm_prefill_ms
   └─ llm_decode_ms
```

표준 token summary:

```text
source frames
sampled frames
tile frame instances
AutoGaze candidate patch tokens
AutoGaze selected patch tokens
vision encoder keep-all patch tokens
vision encoder actual patch tokens
projected visual tokens
LLM text tokens
LLM total context tokens
```

표준 memory summary:

```text
video_preprocess_peak_memory
autogaze_peak_memory
vision_encoder_peak_memory
projector_peak_memory
llm_prefill_peak_memory
overall_peak_memory
```

Acceptance:

- summary JSON에는 “굵직한 5개” latency를 항상 넣는다: preprocess, AutoGaze, vision encoder, LLM, total.
- detailed JSON에는 adapter event별 nested timing을 모두 넣는다.
- AutoGaze off에서도 같은 field가 존재하며 값은 `0`, `null`, 또는 `not_applicable` 중 하나로 채운다.

### 10. Reporter + Visualization

JSON, Markdown, overlay video 생성을 같은 result object에서 만든다.

권장 파일:

- Existing: [repro/markdown_report.py](/Users/mrc/Documents/New%20project/repro/markdown_report.py)
- Existing: [repro/gaze_visualization.py](/Users/mrc/Documents/New%20project/repro/gaze_visualization.py)
- Create: `repro/plugins/reporter.py`
- Test: `tests/test_reporter_plugin.py`

Acceptance:

- inference/benchmark 모두 Markdown report를 생성한다.
- report에는 pipeline ASCII diagram, input video metadata, selected frames, resized frames, patch/token counts, latency, memory, score가 들어간다.
- AutoGaze on이면 selected frame video와 overlay video를 저장한다.
- AutoGaze off이면 selected frame video만 저장하고 overlay field는 `not_applicable`로 기록한다.

### 11. Preflight/OOM Estimator

H100 80GB 기준뿐 아니라 임의 VRAM budget을 넣고 risk를 계산한다.

권장 파일:

- Create: `repro/plugins/preflight.py`
- Move from: H100 estimator helpers in `repro/nvila_runner.py`
- Test: `tests/test_preflight_plugin.py`

입력 축:

- frames: `1024, 512, 256, 128, 64, 32`
- thumbnail frames: `512, 256, 128, 64, 32, 16, 0`
- max tiles: `48, 32, 16, 8, 4, 1`
- resize shortest edge: `None, 1080, 720, 512, 448, 384`
- AutoGaze reduction: measured median, 90%, 95%, synthetic 2x/3x/4x

Acceptance:

- LLM context red, vision encoder memory red, AutoGaze memory red를 별도 band로 낸다.
- AutoGaze 모델이 GPU resident인 경우와 unload-before-generate인 경우를 나눠 계산한다.
- 추천 config는 “paper baseline 재현”, “HD AutoGaze 확장”, “plugin experiment”를 따로 낸다.

---

## 단계별 구현 계획

### Phase 0: 현재 동작 고정

- [ ] `tests/test_nvila_runner.py`에 현재 paper baseline, HD AutoGaze, plugin identity 회귀 테스트를 유지한다.
- [ ] `tests/test_repro_configs.py`에 plugin preset config 4개가 parse되는지 추가한다.
- [ ] `git diff --check`와 `pytest tests/test_nvila_runner.py tests/test_repro_configs.py -q`를 기준선으로 둔다.

### Phase 1: Spec/Registry 분리

- [x] `repro/plugin_api.py`를 만들고 `ExperimentSpec`, `PluginResult`, `MetricStatus` dataclass를 정의한다.
- [x] `repro/plugin_registry.py`를 만들고 string adapter name을 concrete adapter class로 resolve한다.
- [x] `repro/flexible_runner.py`를 만들고 `inspect` 모드에서 `ExperimentSpec.from_args(args)`로 변환한다.
- [ ] 기존 JSON의 `run_identity` schema는 깨지지 않게 유지한다.

### Phase 2: Video/Frame preprocessing 분리

- [ ] `VideoSourceAdapter`를 만들고 frame sampling/decode를 이동한다.
- [ ] `FrameTransformAdapter`를 만들고 resize/tile/tensorize metadata를 이동한다.
- [ ] 기존 stream-profile과 HLVid run이 같은 adapter를 쓰도록 바꾼다.
- [ ] long video seek decode와 scan decode latency를 같은 field로 기록한다.

### Phase 3: Token selector 분리

- [x] `KeepAllTokenSelector`, `NoTokenSelector`, `AutoGazeSelectorPlan` 계약을 만든다.
- [x] `SparseSelectionPlan`, `SelectedPatch`, `EncoderMapping`, `MllmMapping` 표준 schema를 만든다.
- [ ] NVILA-HD native processor path는 우선 wrapper adapter로 감싼다.
- [x] AutoGaze planned output은 표준 `TokenSelectorOutput.sparse_selection_plan`으로 변환한다.
- [x] `gazing_ratio`는 flexible runner spec과 Qwen AutoGaze PoC token estimate에 기록한다.
- [ ] 실제 AutoGaze standalone 실행 output을 concrete selected patch 좌표로 변환한다.
- [ ] `task_loss_requirement`, first-frame keep policy를 output metadata에 기록한다.

### Phase 4: Vision encoder integration

- [ ] `NVILAHDVisionEncoderAdapter`는 기존 remote code hook을 사용한다.
- [ ] `NVILAVideoVisionEncoderAdapter`는 먼저 full encode/off path를 안정화한다.
- [ ] `LongVILAVisionEncoderAdapter`는 model load와 feature extraction entrypoint를 조사한 뒤 full encode/off path를 안정화한다.
- [x] 공통 `post_encoder_token_prune` 결과와 `pre_encoder_sparse` probe 결과 계약을 만든다.
- [x] Qwen3-VL용 PixelPrune pre-ViT adapter hook 계약과 실행 gate를 만든다.
- [x] Qwen3-VL PixelPrune pre-ViT는 hook 성공 시 Qwen native generation까지 내려가고, hook 실패 시 dense 실행을 막는다.
- [ ] `pre_encoder_sparse`가 불가능한 모델은 `post_encoder_token_prune`로 fallback하고 report에 명시한다.
- [ ] patch size 14/16 mismatch check를 adapter 공통 metadata로 이동한다.

### Phase 5: MLLM adapter 분리

- [x] `nvila-video`, `longvila`, `internvl3` planned adapter가 `probe_required` payload를 생성한다.
- [x] `nvila-video` native/off 실행은 공식 VILA `vila-infer` CLI adapter로 분리한다.
- [x] `longvila` native/off 실행도 공식 VILA `vila-infer` CLI adapter로 분리한다.
- [x] `internvl3` native/off 실행은 `repro.internvl3_off_infer` helper CLI adapter로 분리한다.
- [x] Qwen2/2.5/3-VL AutoGaze on은 post-encoder feature packing probe로 분리한다.
- [x] Qwen3-VL AutoGaze post-encoder attachment PoC는 `SparseSelectionPlan`과 before/after visual token estimate를 기록한다.
- [x] VILA 계열 AutoGaze-on 요청은 로컬 `config.json` 기반 static feature packing probe를 `probe_collected`로 기록한다.
- [x] Qwen3-VL은 `--enable-qwen-prune-generate` 실험 플래그에서 `get_video_features` 이후 visual feature를 줄인 `inputs_embeds` generate 경로를 시도한다.
- [ ] `NVILAHDMllmAdapter`, `NVILAVideoMllmAdapter`, `LongVILAMllmAdapter`를 만든다.
- [ ] prompt packing과 visual feature packing을 adapter method로 분리한다.
- [ ] TTFT, LLM prefill, KV cache estimate를 공통 output field로 기록한다.
- [ ] SDPA OOM 위치를 stage-aware exception으로 기록한다.

### Phase 6: Benchmark task 일반화

- [ ] HLVid loader를 `VideoQATaskAdapter` 구현체로 감싼다.
- [ ] 일반 mp4 folder inference는 `SingleVideoTaskAdapter`로 둔다.
- [ ] multiple-choice scoring과 free-form scoring을 분리한다.
- [ ] output row에는 `video_path`, `question`, `prediction`, `answer`, `correct`, `failure_stage`를 항상 넣는다.

### Phase 7: Reporting/Visualization 통합

- [ ] result object 하나에서 JSON summary, JSONL prediction, Markdown report를 만든다.
- [ ] AutoGaze mask overlay는 `TokenSelectorOutput.selected_mask_by_scale`에서 생성한다.
- [ ] selected original frames와 resized frames를 모두 저장한다.
- [ ] benchmark comparison report는 model family별 column을 고정한다.

### Phase 8: H100 preflight와 실제 측정 연결

- [ ] preflight estimator가 measured median AutoGaze reduction을 읽을 수 있게 한다.
- [ ] 실제 benchmark summary에서 reduction median, p50/p90 latency, p50/p90 memory를 뽑아 다음 sweep 입력으로 쓴다.
- [ ] H100 80GB 권장 config는 plugin family별로 분리한다.

---

## 우선순위

1. `NVILA-8B-Video plugin off` 실제 generate 안정화
2. `NVILA-8B-Video plugin on`의 feasible integration mode 결정
3. `LongVILA plugin off` 실제 generate 안정화
4. `LongVILA plugin on`의 feasible integration mode 결정
5. `Qwen2/2.5/3-VL plugin off` 실제 generate 안정화
6. `Qwen2/2.5/3-VL plugin on`은 `post_encoder_token_prune`부터 검증
7. HLVid `--limit 3`으로 on/off score/latency/memory/token report 생성
8. 전체 HLVid benchmark에서 failed/OOM/parse_failed 분리 report
9. LLaVA-OneVision off/post-prune 후보 검증
10. InternVL3 dynamic tiling adapter probe
11. H100 preflight를 measured reduction 기반으로 갱신

---

## NVILA-8B-Video / LongVILA AutoGaze 적용 판단 기준

AutoGaze를 붙이는 방식은 모델 내부 구조에 따라 달라진다.

| 방식 | vision encoder latency 감소 | MLLM latency/memory 감소 | 구현 난이도 | 설명 |
|---|---:|---:|---:|---|
| native processor | 예 | 예 | 낮음 | NVILA-HD처럼 remote processor가 AutoGaze를 이미 알고 있는 경우 |
| pre-encoder sparse | 예 | 예 | 높음 | selected patch만 vision encoder에 넣는다 |
| post-encoder token prune | 아니오 | 예 | 중간 | full vision encode 후 MLLM에 들어갈 visual token만 줄인다 |
| keep-all/native off | 아니오 | 아니오 | 낮음 | 비교군 |

PoC에서 가장 현실적인 순서는 다음과 같다.

1. off path를 먼저 안정화한다.
2. post-encoder token prune으로 MLLM context 감소를 먼저 검증한다.
3. vision encoder의 patch embedding/position encoding contract를 확인한 뒤 pre-encoder sparse를 시도한다.
4. pre-encoder sparse가 가능한 모델만 “vision encoder 계산량 감소”를 주장한다.

---

## Acceptance Criteria

- 같은 HLVid input set에서 `paper baseline`, `HD AutoGaze`, `NVILA-Video plugin off/on`, `LongVILA plugin off/on`이 서로 다른 `run_identity.model_family`로 기록된다.
- `paper baseline`에는 AutoGaze kwargs가 들어가지 않는다.
- plugin off/on은 같은 sampled frames, 같은 resize policy, 같은 prompt/question으로 비교된다.
- summary에는 latency, token, memory 핵심 field가 AutoGaze on/off 양쪽에 모두 존재한다.
- AutoGaze on에서 token reduction 분모가 명확하다.
- OOM은 stage별로 기록된다: `video_decode`, `autogaze`, `vision_encoder`, `mm_projector`, `llm_prefill`, `llm_decode`.
- Markdown report는 리더에게 보여줄 수 있는 형태로 pipeline, 숫자, 실패 원인을 한 화면에서 설명한다.

---

## Verification Commands

기본 회귀:

```bash
.venv/bin/python -m pytest tests/test_nvila_runner.py tests/test_hlvid_batch_benchmark.py tests/test_markdown_report.py tests/test_repro_configs.py -q
```

plugin API 추가 후:

```bash
.venv/bin/python -m pytest tests/test_plugin_api.py tests/test_token_selector_plugin.py tests/test_vision_encoder_plugin.py tests/test_mllm_plugin.py -q
```

CUDA smoke:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter keep-all \
  --video /path/to/video.mp4 \
  --num-video-frames 128 \
  --measure-ttft \
  --print-summary \
  --output-json outputs/autogaze_repro/nvila_video_plugin_off_smoke.json
```

HLVid smoke:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode hlvid \
  --device cuda \
  --model-path weight/NVILA-8B-Video \
  --model-family nvila-video-plugin \
  --token-selector-adapter keep-all \
  --manifest /path/to/HLVid/data/test-00000-of-00001.parquet \
  --hlvid-video-root /path/to/HLVid/videos \
  --limit 3 \
  --measure-ttft \
  --continue-on-error \
  --summary outputs/autogaze_repro/hlvid_nvila_video_plugin_off_limit3_summary.json
```

---

## 실행 시 주의점

- `NVILA-8B-Video`를 `nvila-video-baseline`으로 실행하면 논문 baseline 후보이고 AutoGaze를 붙이면 안 된다.
- 같은 weight를 AutoGaze on/off 실험에 쓰려면 `nvila-video-plugin`을 사용한다.
- LongVILA는 remote code의 video packing 방식이 다를 수 있으므로 off path부터 smoke한다.
- post-encoder token prune은 MLLM token/context 감소를 볼 수 있지만 vision encoder latency 감소를 주장할 수 없다.
- pre-encoder sparse가 성공해야 AutoGaze의 vision encoder 계산량 감소를 직접 주장할 수 있다.
- H100 PoC에서는 AutoGaze, vision tower, MLLM이 모두 GPU resident라고 보는 보수적 계산을 기본으로 한다.
