# AutoGaze PoC 벤치마크 계획

이 문서는 AutoGaze 확장 PoC의 벤치마크 조사, 재현 계획, 확장 계획, Hugging Face 기반 benchmark path, 측정 방법론, 결과 표 템플릿을 정의합니다.

중요 구분:

- dummy/stub 결과는 pipeline 연결 확인용입니다.
- real benchmark 결과는 실제 checkpoint, dataset, protocol, hardware 정보를 포함해야 합니다.
- 외부 MLLM이 AutoGaze를 사용했다고 명시적으로 보고하지 않은 경우 그렇게 해석하지 않습니다.

## 1. Public Benchmark Survey

AutoGaze paper의 Table 1 역할을 따르는 public survey를 작성합니다. 목적은 기존 video MLLM이 실제로 어느 정도의 frame count, resolution, long-video/high-resolution setting을 처리하는지 비교하는 것입니다.

이 survey는 다른 모델이 AutoGaze를 사용했다는 증거가 아닙니다.

### 조사 항목

| 항목 | 설명 |
|---|---|
| Model | 모델명 |
| Open? | open-source 여부 |
| Model size | parameter 규모 |
| Max #Frames | 보고된 최대 입력 frame 수 |
| Max Resolution | 보고된 최대 resolution |
| VideoMME w/o Sub | subtitle 없는 VideoMME |
| VideoMME w/ Sub | subtitle 있는 VideoMME |
| MVBench | MVBench score |
| NExT-QA | NExT-QA score |
| LongVideoBench / L-VidBench | long-video benchmark |
| EgoSchema | EgoSchema score |
| MLVU | MLVU score |
| HLVid | high-resolution long-video QA |
| Notes | long-video/high-resolution 특성 |

### Survey table template

| Model | Open? | Max #Frames | Max Resolution | VideoMME w/o Sub | VideoMME w/ Sub | MVBench | NExT-QA | L-VidBench / LongVideoBench | EgoSchema | MLVU | HLVid | Result Source | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Gemini 1.5 Pro | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | proprietary baseline |
| GPT-4o | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | proprietary baseline |
| Qwen2.5-VL-7B | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | open-source baseline |
| NVILA-8B-Video | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | paper/model-card | original NVILA baseline |
| NVILA-8B-Video + AutoGaze | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | AutoGaze paper | AutoGaze-scaled row |
| Internal PoC A2 | N/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | internal reproduced | 실제 재현 후 입력 |

## 2. Reproduction Benchmark Plan

1차 재현 목표는 canonical SigLIP/NVILA ablation입니다.

| ID | AutoGaze | Vision Encoder | MLLM | 우선순위 | 상태 |
|---|---:|---|---|---:|---|
| A0 | OFF | vanilla SigLIP ViT | NVILA | 2 | vanilla full-token baseline |
| A1 | OFF | modified SigLIP ViT | NVILA | 1 | modified SigLIP effect |
| A2 | ON | modified SigLIP ViT | NVILA | 1 | canonical AutoGaze path |
| A3 | ON | vanilla SigLIP ViT | NVILA | 3 | experimental compatibility ablation |

A3는 구현 및 테스트 전까지 직접 호환된다고 주장하지 않습니다.

### 재현 시 필수 기록

- resolved config
- git commit hash
- package versions
- device information
- CUDA/MPS availability
- precision
- timestamp
- checkpoint paths
- Hugging Face IDs/revisions, 해당 시
- offline/cache mode
- `trust_remote_code`
- token count before/after AutoGaze
- acceleration type note

## 3. Extension Benchmark Plan

Canonical path 이후 다음 축을 확장합니다.

| Axis | 후보 |
|---|---|
| AutoGaze | ON / OFF |
| Vision encoder | modified SigLIP, vanilla SigLIP, V-JEPA2, generic ViT |
| MLLM/decoder | NVILA, Qwen, generic MLLM, task decoder |
| Integration mode | full, hook, native, crop, mask, compact, official_processor |
| Task | Video VQA, Action Recognition |
| Source mode | local, Hugging Face, mixed, offline cache |

주의:

- post-encoder pruning은 encoder-side acceleration이 아닙니다.
- full ViT forward 이후 mask 적용은 ViT acceleration이 아닙니다.
- MLLM prefill token 감소는 downstream acceleration으로 별도 분류합니다.

## 4. Hugging Face-Based Benchmark Plan

지원 benchmark modes:

| Mode | Model Source | Dataset Source | 설명 |
|---|---|---|---|
| `hf_model_only` | HF Hub/cache | dummy/local dataset | public model loading smoke |
| `hf_dataset_only` | local/internal model | HF Hub/local file | public dataset loading smoke |
| `hf_model_and_dataset` | HF Hub/cache | HF Hub/cache | full public path |
| `local_model_hf_dataset` | local/internal checkpoint | HF dataset | internal model on public data |
| `hf_model_local_dataset` | HF model | local/internal dataset | public model on internal data |
| `offline_hf_cache` | local HF cache | local HF cache | offline reproducibility |

기본 원칙:

- dry-run이 기본입니다.
- 큰 public model benchmark는 기본 실행하지 않습니다.
- official processor path를 우선합니다.
- AutoGaze token injection은 지원된다고 가정하지 않습니다.
- model/dataset revision pinning을 사용합니다.

HF benchmark smoke:

```bash
PYTHONPATH=src python -m autogaze_ext.pipeline.hf_benchmark \
  --config-name hf_benchmark/hf_dataset_only \
  --output-dir outputs/hf_benchmarks \
  --dry-run
```

## 5. Measurement Methodology

### Latency

- warm-up iteration 수는 config로 제어합니다.
- CUDA timing은 `torch.cuda.synchronize()`를 사용합니다.
- data loading 포함 여부를 명시합니다.

Latency breakdown:

| Metric | 설명 |
|---|---|
| AutoGaze latency | AutoGaze selector/router 시간 |
| ViT latency | vision encoder 시간 |
| MLLM prefill latency | visual/text prefill |
| MLLM decode latency | generation decode |
| end-to-end latency | 전체 pipeline 시간 |

### Memory

| Device | Metric |
|---|---|
| CUDA | `torch.cuda.max_memory_allocated()` |
| MPS | unavailable metric은 `N/A` |
| CPU | benchmark timing 해석 제한, memory는 필요 시 별도 기록 |

### Token metrics

- visual token count before AutoGaze
- visual token count after AutoGaze
- token reduction ratio
- selected patches per frame
- selected patches per scale

## 6. Benchmark Result Table Templates

### Efficiency table

| Experiment | AutoGaze | Vision Encoder | MLLM | Frames | Resolution | Before Tokens | After Tokens | Reduction | Latency ms | Throughput | FPS | VRAM MB | Acceleration Type | Notes |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A2 | ON | modified SigLIP | NVILA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | real run only |

### Performance table

| Experiment | Task | Dataset | Metric | Score | Protocol | Result Type | Notes |
|---|---|---|---|---:|---|---|---|
| A2 | Video VQA | TBD | Exact Match | TBD | internal | reproduced | TBD |

### Token reduction table

| Experiment | Frame | Scale | Original Tokens | Selected Tokens | Reduction | Notes |
|---|---:|---|---:|---:|---:|---|
| A2 | TBD | TBD | TBD | TBD | TBD | metadata required |

### Resolution/frame scaling table

| Experiment | Frames | Resolution | Latency | VRAM | Metric | Notes |
|---|---:|---|---:|---:|---:|---|
| A2 | 32 | 224p | TBD | TBD | TBD | baseline |
| A2 | 128 | 720p | TBD | TBD | TBD | scaling |

### Hugging Face benchmark result table

| Experiment | HF Model ID | Model Revision | HF Dataset ID | Dataset Split | Integration Mode | Samples | Metric Source | Metric | Offline | Cache Dir | Notes |
|---|---|---|---|---|---|---:|---|---:|---|---|---|
| hf_dataset_only | N/A | N/A | local.jsonl | validation | official_processor | 2 | internal_fallback | TBD | false | TBD | smoke |

### Dummy/stub result table

| Experiment | Result File | Stub Status | Real Checkpoints Loaded? | Use For |
|---|---|---|---|---|
| A0 dummy | `outputs/dummy_benchmarks/A0.json` | dummy_full_token_baseline | No | wiring smoke |
| A2 dummy | `outputs/dummy_benchmarks/A2.json` | stubbed_autogaze_on_no_real_selector_no_token_reduction | No | wiring smoke |

## 7. PoC Visualization Smoke Configs

These configs exercise Priority 2 visualization and metadata wiring only. They use dummy or tiny local inputs by default and must not be interpreted as benchmark-scale results.

| Config | Mode | Purpose | Result Label |
|---|---|---|---|
| `configs/benchmark/poc_default.yaml` | `check` | Default PoC reference preset with canonical `num_frames=16` and full-pipeline plug-in metadata | default config |
| `configs/benchmark/poc_feature_matrix_smoke.yaml` | `autogaze_only` | General feature-matrix smoke path | dummy/stub smoke |
| `configs/benchmark/poc_autogaze_only_visualization.yaml` | `autogaze_only` | Overlay, side-by-side, and scale-panel video export | dummy/stub smoke |
| `configs/benchmark/poc_full_pipeline_visualization.yaml` | `full_pipeline` | Query-text path plus AutoGaze visualization outputs | guarded/stub by default |
| `configs/benchmark/poc_chop_mode_smoke.yaml` | `autogaze_only` | Chop metadata and chop-local visualization | partial chop smoke |
| `configs/benchmark/poc_multiscale_visualization.yaml` | `autogaze_only` | Multi-scale gradient overlay and scale labels | dummy/stub smoke |
| `configs/benchmark/poc_scale_panel_video.yaml` | `autogaze_only` | 2x2 scale-panel video export | dummy/stub smoke |

Supported Priority 2 options include `overlay_style=mask|box|both`, `multi_scale_overlay`, `scale_color_mode=gradient|categorical`, `show_patch_index`, `show_scale_label`, `save_scale_panel_video`, and `comparison_layout=processed_overlay`.

All PoC benchmark presets include:

```yaml
full_pipeline_plugin_mode: experiment_config
component_plugins:
  autogaze:
    config_section: model.autogaze
  vision_encoder:
    config_section: model.vision_encoder
  mllm:
    config_section: model.mllm
```

This records the current config-driven plug-in contract for `scripts/poc_nvila_hd_video.py --mode full_pipeline`.
Actual model construction is still controlled by the experiment config passed through `benchmark.config`, for example `configs/experiment/A2_real.yaml`.
These fields are audit metadata for benchmark presets; they are not separate CLI module overrides.

Use the experiment config to choose the canonical real path:

| Experiment config | Use in PoC benchmarks | AutoGaze interpretation |
|---|---|---|
| `configs/experiment/A1_real.yaml` | Modified-SigLIP + NVILA full-token baseline | AutoGaze OFF; do not use AutoGaze token reduction claims |
| `configs/experiment/A2_real.yaml` | Canonical AutoGaze + modified-SigLIP + NVILA path | AutoGaze ON; token reduction claims require real AutoGaze execution |

For A1/A2 internal comparison, keep the benchmark axes identical:

```text
input video
frame_selection_mode
num_frames
scaling_mode
resolution
device
dtype
max_new_tokens
query_text
benchmark_iterations
```

Full-pipeline plug-in benchmark options currently represented in the presets:

| Option group | Fields |
|---|---|
| ViT plug-in | `module_path_field`, `class_or_factory_field`, `checkpoint_field`, `processor_path_field`, `construction_kwargs_field`, `input_contract`, `output_contract` |
| MLLM plug-in | `module_path_field`, `class_or_factory_field`, `processor_module_path_field`, `processor_class_field`, `processor_path_field`, `tokenizer_path_field`, `prompt_template_field`, `generation_method`, `official_processor_path` |
| Runtime guards | `checkpoint_loading`, `heavy_benchmark`, `run_by_default`, `max_new_tokens`, `device`, `dtype` when applicable |
| AutoGaze controls | `gaze_ratio`, `task_loss_requirement`, frame selection, scaling/chop, and visualization export flags |

Stub-only options remain hold-last video export, non-zero chop overlap/stride, `chop_overlay` as a side-by-side comparison layout, and original decoded-frame chop overlays beyond the non-overlap `overlay_union` path.

## 8. Priority 3 High-Resolution Preparation Configs

These configs prepare high-resolution/chop and full-length visualization checks. They are config templates only and must not be run as paper-scale benchmarks.

| Config | Purpose | Safety Defaults |
|---|---|---|
| `configs/benchmark/poc_high_resolution_chop_smoke.yaml` | Tiny chop + `overlay_union` smoke | `batch_size=1`, `num_frames=4`, `max_chops=4`, `benchmark_iterations=1` |
| `configs/benchmark/poc_high_resolution_chop_medium.yaml` | Medium preparation for chop visualization | `batch_size=1`, `num_frames=16`, `max_chops=16`, `benchmark_iterations=3` |
| `configs/benchmark/poc_full_length_video_export_smoke.yaml` | Full-length export smoke | `batch_size=1`, `num_frames=2`, `video_export_mode=full_length` |

Priority 3 support status:

- `quickstart` exact scaling is implemented only for documented QUICK_START policies: `224/patch16` and `392/patch14` target-scale mode.
- `full_length` video export preserves original frame count and original FPS when available; unprocessed frames are explicitly marked.
- `overlay_union` maps non-overlapping chop-local patch masks back to full processed-frame coordinates.
- `original_overlay` and `original_processed_overlay` are implemented only for exact affine mappings from processed to original frames.
- Non-zero chop overlap/custom stride, hold-last video, 4K runs, 1024-frame runs, and public benchmark-scale jobs remain out of default scope.
