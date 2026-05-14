# AutoGaze-Based Video Understanding PoC Pipeline Request

## 0. Objective

Build an internal PoC pipeline based on the public AutoGaze codebase.

The goals are:

1. Analyze and internalize the main AutoGaze codebase.
2. Preserve the original AutoGaze implementation as much as possible.
3. Modularize AutoGaze as a visual patch/token selector or router.
4. Build a configurable inference, evaluation, benchmark, and visualization pipeline for video understanding tasks.
5. Evaluate the canonical AutoGaze setup first:
   - AutoGaze ON/OFF
   - modified SigLIP ViT / vanilla SigLIP ViT
   - NVILA
6. Extend the pipeline to support other vision backbones and MLLMs:
   - V-JEPA2
   - generic unmodified ViTs
   - Qwen-family MLLMs
   - other future MLLMs
   - task-specific decoders
7. Add Hugging Face-based benchmark support when technically feasible:
   - Hugging Face model loading
   - Hugging Face dataset loading
   - Hugging Face Evaluate-based metric computation
   - local cache / offline mode support
   - reproducible Hub revision pinning

The key design goal is to avoid hardcoding AutoGaze only for the original SigLIP/NVILA path.

AutoGaze should be reusable as a backbone-agnostic visual patch/token selector whenever technically feasible.

---

## 1. Primary References

Before implementation, strictly read the original AutoGaze codebase and its documentation.

Primary reference files:

```text
original AutoGaze codebase
original INTEGRATION.md
original QUICK_START.md
original README
original training scripts
original inference scripts
original evaluation scripts
original docs/nvila-hd-video-readme.md
```

Requirements:
- Use INTEGRATION.md as the primary reference for architecture, integration modes, and image-to-video / temporal alignment.
- Use QUICK_START.md as the primary reference for real inference commands, checkpoint layout, runtime arguments, and resolution scaling behavior.
- Do not modify the original `INTEGRATION.md`.
- Use the original `INTEGRATION.md` as the primary reference for image-to-video integration and temporal handling.
- Preserve the original AutoGaze core implementation as much as possible.
- Prefer wrapper, adapter, inheritance, and extension classes instead of directly rewriting the original implementation.
- If modifying original code is unavoidable, document exactly what was changed and why.
- Do not silently change the original behavior of AutoGaze, modified SigLIP, or NVILA.
- Do not modify the original QUICK_START.md.
- If the implementation behavior differs from QUICK_START.md, document the reason.
- Use docs/nvila-hd-video-readme.md as the concrete canonical-path usage reference for:
  - AutoGaze
  - SigLIP ViT
  - NVILA-HD-Video
  - model loading
  - inference command
  - processor/tokenizer behavior
  - checkpoint layout
  - video input handling
  - query text / prompt handling
  - expected output format
---

## 2. Core Experimental Design

### 2.1 Canonical SigLIP/NVILA Ablation

The first required experiment group is:

```text
AutoGaze ON/OFF
×
modified SigLIP ViT / vanilla SigLIP ViT
×
NVILA
```

This creates a 2×2 ablation.

| ID | AutoGaze | Vision Encoder | MLLM | Purpose |
|---|---:|---|---|---|
| A0 | OFF | vanilla SigLIP ViT | NVILA | Full-token vanilla SigLIP baseline |
| A1 | OFF | modified SigLIP ViT | NVILA | Measure the effect of the modified SigLIP itself |
| A2 | ON | modified SigLIP ViT | NVILA | Canonical path closest to the original AutoGaze paper/codebase |
| A3 | ON | vanilla SigLIP ViT | NVILA | Experimental ablation to test whether AutoGaze can work with vanilla SigLIP |

Implementation priority:

1. Implement A1 and A2 first.
   - This is the closest path to the original AutoGaze setup.
   - This gives the main AutoGaze ON/OFF comparison under the modified SigLIP + NVILA setup.

2. Implement A0.
   - This provides the vanilla SigLIP full-token baseline.

3. Implement A3.
   - Treat AutoGaze ON + vanilla SigLIP as an experimental ablation.
   - Do not assume direct compatibility.
   - Implement explicit adapter modes if needed.

---

## 3. Extension Experiments

After the canonical SigLIP/NVILA ablation is runnable, extend the pipeline to:

```text
AutoGaze ON/OFF
×
target vision backbone
×
target decoder / MLLM
```

Target vision backbones:

- modified SigLIP ViT
- vanilla SigLIP ViT
- V-JEPA2 pretrained video ViT
- generic unmodified ViT
- future vision backbones

Target MLLMs / decoders:

- NVILA
- Qwen-family MLLM
- other MLLMs
- task-specific Video VQA decoder
- task-specific Action Recognition decoder

Important requirements:

- Do not hardcode Qwen as the only non-NVILA target.
- Implement a generic MLLM adapter interface.
- NVILA, Qwen, and future MLLMs should be pluggable through the same interface.
- Clearly distinguish between:
  - true encoder-side acceleration
  - post-patch-embedding token masking
  - post-encoder token pruning
  - downstream token reduction only
  - compatibility-only adapter paths

---

## 4. Benchmark Survey Requirement

Create a benchmark survey section following the purpose of Table 1 in the AutoGaze paper.

The goal is not to find prior works that already applied AutoGaze.

Most prior MLLM papers do not evaluate with AutoGaze.

Instead, the benchmark survey should compare existing video MLLMs by their video scaling capability and benchmark performance.

The survey should include, when available:

- model name
- model size
- whether the model is open-source or proprietary
- maximum number of input frames
- maximum input resolution
- VideoMME score without subtitles
- VideoMME score with subtitles
- MVBench score
- NExT-QA score
- LongVideoBench or L-VidBench score
- EgoSchema score
- MLVU score
- HLVid score, if available
- notes on whether the model targets:
  - general video understanding
  - long-video understanding
  - high-resolution video understanding
  - high-resolution long-form video understanding

Use the AutoGaze paper Table 1 as the primary template.

Important:

- Do not claim that other models used AutoGaze unless explicitly reported.
- Treat other models as external baselines.
- Treat `NVILA-8B-Video + AutoGaze` as the AutoGaze-scaled row.
- Clearly separate:
  1. existing MLLM baseline results
  2. original NVILA baseline
  3. NVILA + AutoGaze result
  4. our internally reproduced results
  5. our extension results with other backbones or MLLMs

---

## 5. Public Benchmark Survey Table Template

Use a table similar to the following for the benchmark survey.

The numbers should be copied from the original paper only after verifying the exact values from the paper.

```markdown
| Model | Open? | Max #Frames | Max Resolution | VideoMME w/o Sub | VideoMME w/ Sub | MVBench | NExT-QA | L-VidBench / LongVideoBench | EgoSchema | MLVU | HLVid | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gemini 1.5 Pro | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | proprietary baseline |
| GPT-4o | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | proprietary baseline |
| Qwen2.5-VL-7B | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | open-source MLLM baseline |
| NVILA-8B-Video | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | original NVILA baseline |
| NVILA-8B-Video + AutoGaze | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | AutoGaze-scaled result |
```

The survey should answer the following questions:

1. Existing MLLMs are strong, but what frame count and resolution do they actually process?
2. Are current video MLLMs mostly limited to low-resolution or short-frame settings?
3. Does AutoGaze allow NVILA to process more frames and higher resolution?
4. Does this scaling improve high-resolution long-form video QA, especially HLVid?
5. Are general video benchmarks always improved by scaling, or is the benefit benchmark-dependent?

---

## 6. Benchmark Survey Interpretation

The Table 1-style survey can support claims such as:

- AutoGaze enables NVILA to process more frames and higher resolution than the original NVILA baseline.
- The AutoGaze-scaled NVILA result is competitive with strong open-source and proprietary MLLMs on several video benchmarks.
- The largest reported gain is expected on high-resolution long-form video QA benchmarks such as HLVid.
- Existing long-video MLLMs do not necessarily solve high-resolution detail understanding.

The Table 1-style survey cannot support claims such as:

- AutoGaze improves all MLLMs.
- AutoGaze improves Qwen, GPT-4o, Gemini, or other MLLMs unless directly tested.
- AutoGaze is directly compatible with every vision encoder.
- Higher resolution and more frames always improve every benchmark.
- Token reduction always preserves accuracy.

Internal reproduction questions:

- Can we reproduce the AutoGaze ON/OFF trend on the canonical modified SigLIP + NVILA path?
- Does vanilla SigLIP preserve the same trend?
- Does the gain come from true encoder-side acceleration or only token reduction after encoding?
- Does AutoGaze still help when connected to non-NVILA MLLMs?
- Which benchmarks actually benefit from longer frame count and higher resolution?

---

## 7. Hugging Face-Based Benchmark Support

Add optional Hugging Face-based benchmark support when technically feasible.

The goal is to make it easy to benchmark public models and public datasets through Hugging Face while still preserving the original AutoGaze/NVILA path.

### 7.1 Supported Hugging Face Components

Support the following Hugging Face components:

- `transformers`
  - model loading
  - processor/tokenizer loading
  - multimodal model inference when available
- `datasets`
  - dataset loading from the Hub
  - dataset loading from local files
  - streaming mode when feasible
- `evaluate`
  - metric loading
  - metric computation
  - standardized result reporting
- `huggingface_hub`
  - checkpoint snapshot download
  - revision pinning
  - cache control
  - offline mode

### 7.2 Hugging Face Benchmark Modes

Implement the following benchmark modes:

```text
hf_model_only
hf_dataset_only
hf_model_and_dataset
local_model_hf_dataset
hf_model_local_dataset
offline_hf_cache
```

Mode definitions:

| Mode | Model Source | Dataset Source | Purpose |
|---|---|---|---|
| `hf_model_only` | Hugging Face Hub | dummy/local dataset | Test public model loading |
| `hf_dataset_only` | local/internal model | Hugging Face Hub | Test public dataset loading |
| `hf_model_and_dataset` | Hugging Face Hub | Hugging Face Hub | Full public benchmark path |
| `local_model_hf_dataset` | local/internal checkpoint | Hugging Face Hub | Evaluate internal model on public dataset |
| `hf_model_local_dataset` | Hugging Face Hub | local/internal dataset | Evaluate public model on internal-style data |
| `offline_hf_cache` | local HF cache | local HF cache | Reproducible offline benchmark |

### 7.3 Hugging Face Configuration

Add config fields for Hugging Face usage.

Example:

```yaml
huggingface:
  enabled: true
  model_id: null
  dataset_id: null
  dataset_config: null
  dataset_split: validation
  revision: null
  trust_remote_code: false
  token_env_var: HF_TOKEN
  cache_dir: ${oc.env:HF_HOME,null}
  local_files_only: false
  offline: false
  streaming: false
  max_samples: null
  num_proc: 1
```

Requirements:

- Do not hardcode Hugging Face model IDs.
- Do not hardcode dataset IDs.
- Always allow `revision` pinning.
- Always allow `local_files_only`.
- Always allow `cache_dir`.
- Do not require authentication unless the selected model/dataset requires it.
- If authentication is needed, read the token from an environment variable.
- Do not write access tokens into logs, configs, or output files.

### 7.4 Hugging Face Model Loading

Implement a generic Hugging Face model loader.

Example interface:

```python
class HFModelLoader:
    def load_model(self, model_id: str, revision=None, device=None, dtype=None, **kwargs):
        raise NotImplementedError

    def load_processor(self, model_id: str, revision=None, **kwargs):
        raise NotImplementedError
```

Requirements:

- Support `AutoModel`, `AutoModelForCausalLM`, or model-specific classes when needed.
- Support `AutoProcessor`, `AutoTokenizer`, and image/video processors when needed.
- Support `trust_remote_code` as a config option.
- Log the exact model ID and revision.
- Log the exact processor/tokenizer ID and revision.
- Avoid direct visual token injection unless the model API clearly supports it.

### 7.5 Hugging Face Dataset Loading

Implement a generic Hugging Face dataset loader.

Example interface:

```python
class HFDatasetLoader:
    def load_dataset(self, dataset_id: str, config=None, split=None, streaming=False, **kwargs):
        raise NotImplementedError
```

Requirements:

- Support public Hub datasets.
- Support local dataset files.
- Support `json`, `jsonl`, `csv`, and folder-based video metadata where feasible.
- Support `max_samples` for smoke tests.
- Support dataset field mapping through config.

Example field mapping:

```yaml
data:
  field_mapping:
    video: video
    question: question
    answer: answer
    label: label
    options: options
```

### 7.6 Hugging Face Evaluate Integration

Use Hugging Face Evaluate when appropriate.

Required metric wrapper:

```python
class HFEvaluateMetric:
    def __init__(self, metric_name: str, config_name=None):
        pass

    def add_batch(self, predictions, references):
        pass

    def compute(self):
        pass
```

Requirements:

- Support built-in metrics when available.
- Support custom local metrics for Video VQA and Action Recognition.
- Do not force Hugging Face Evaluate for metrics that are not supported.
- Allow fallback to internal metric implementations.

Metric examples:

- exact match
- accuracy
- top-k accuracy through internal implementation
- relaxed VQA accuracy through internal implementation
- generation logging

### 7.7 Hugging Face Offline and Cache Mode

Support reproducible offline runs.

Requirements:

- Support `HF_HOME`.
- Support `HF_HUB_CACHE`.
- Support `TRANSFORMERS_CACHE` if needed.
- Support `HF_HUB_OFFLINE=1`.
- Support `TRANSFORMERS_OFFLINE=1`.
- Support `HF_DATASETS_OFFLINE=1`.
- Support pre-downloading model snapshots.
- Support pre-downloading dataset snapshots when feasible.

Add a script:

```text
scripts/download_hf_assets.py
```

Expected behavior:

```bash
python scripts/download_hf_assets.py \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --revision <revision_or_commit_hash> \
  --cache-dir ./hf_cache
```

Requirements:

- Save downloaded model/dataset IDs.
- Save revisions or commit hashes.
- Save cache paths.
- Do not save authentication tokens.

### 7.8 Hugging Face-Based Benchmark Targets

Add benchmark target definitions for Hugging Face-based experiments.

Examples:

```yaml
benchmark_target:
  name: qwen2_5_vl_video_vqa
  source: huggingface
  model_id: Qwen/Qwen2.5-VL-7B-Instruct
  dataset_id: null
  task: video_vqa
  integration_mode: official_processor
```

```yaml
benchmark_target:
  name: hf_dataset_internal_model
  source: mixed
  model_id: null
  dataset_id: some_public_video_vqa_dataset
  task: video_vqa
  integration_mode: internal_model
```

Important:

- Hugging Face model benchmarks should first use the official processor path.
- AutoGaze-guided pre-processing can be added as a separate experimental mode.
- Do not assume that a Hugging Face MLLM accepts arbitrary externally selected visual tokens.
- If the model only supports official video/image processor input, use input-level frame/region selection first.

### 7.9 Hugging Face Benchmark Reporting

Each Hugging Face-based benchmark result must include:

- model ID
- model revision
- processor/tokenizer ID
- processor/tokenizer revision
- dataset ID
- dataset config
- dataset split
- dataset revision, if available
- cache mode
- offline mode status
- `trust_remote_code` status
- number of evaluated samples
- preprocessing details
- frame sampling details
- input resolution
- metric implementation source:
  - Hugging Face Evaluate
  - internal metric
  - custom script
- hardware information
- precision
- latency
- throughput
- peak VRAM when available

### 7.10 Hugging Face Non-goals

Do not:

- assume every public MLLM is compatible with AutoGaze token outputs
- claim AutoGaze improves a Hugging Face model unless directly measured
- automatically download large datasets without explicit config
- automatically use gated models without checking authentication
- log Hugging Face access tokens
- mix reported paper results and Hugging Face reproduction results without labeling them separately

---

## 8. Pipeline Architecture

Implement a config-driven pipeline using Hydra or a YAML-based configuration system.

Required config groups:

```text
configs/
  model/
    autogaze/
    vision_encoder/
    mllm/
    task_decoder/
    adapter/
    huggingface/
  task/
    video_vqa/
    action_recognition/
  data/
    dummy_video/
    video_vqa/
    action_recognition/
    huggingface/
  runtime/
    device/
    precision/
    seed/
    huggingface/
  benchmark/
    profiling/
    visualizer/
    huggingface/
```

The pipeline must support:

- inference
- evaluation
- benchmark
- visualization
- smoke_test
- Hugging Face model benchmark
- Hugging Face dataset benchmark
- offline cached benchmark

Requirements:

- Do not hardcode model names, paths, input sizes, frame counts, patch sizes, or token budgets.
- All experiment combinations should be controlled through config files.
- Save the resolved config for each run.
- Make experiment IDs configurable and reproducible.

---

## 9. Recommended Project Structure

Use or adapt the following structure:

```text
src/
  autogaze_ext/
    pipeline/
      runner.py
      inference.py
      evaluate.py
      benchmark.py

    data/
      frame_sampler.py
      video_dataset.py
      dummy_video_dataset.py
      video_vqa_dataset.py
      action_recognition_dataset.py

      hf_dataset_loader.py
      hf_dataset_adapter.py

    models/
      autogaze_wrapper.py

      vision/
        base_vision_encoder.py
        modified_siglip_adapter.py
        vanilla_siglip_adapter.py
        vjepa2_adapter.py
        generic_vit_adapter.py

      mllm/
        base_mllm_adapter.py
        nvila_adapter.py
        qwen_adapter.py
        generic_mllm_adapter.py
        hf_mllm_adapter.py
        registry.py

      huggingface/
        hf_model_loader.py
        hf_processor_loader.py
        hf_registry.py

      decoders/
        video_vqa_head.py
        action_recognition_head.py

    adapters/
      base_adapter.py
      patch_grid_mapper.py
      patch_index_adapter.py
      token_mask_adapter.py
      compact_token_adapter.py
      vision_feature_adapter.py
      temporal_adapter.py
      mllm_visual_input_adapter.py

    metrics/
      efficiency.py
      video_vqa_metrics.py
      action_recognition_metrics.py
      hf_evaluate_metric.py

    profiling/
      latency.py
      memory.py
      token_counter.py

    visualization/
      base_visualizer.py
      autogaze_visualizer.py
      task_output_visualizer.py
      full_pipeline_visualizer.py

    utils/
      seed.py
      device.py
      reproducibility.py
      logging.py
      hf_cache.py
      hf_offline.py

scripts/
  setup_linux.sh
  setup_mac.sh
  download_hf_assets.py
```

---

## 10. AutoGaze Integration Requirements

Treat AutoGaze as:

```text
AutoGaze = visual patch/token selector/router
```

The AutoGaze wrapper should expose at least the following output structure:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

@dataclass
class AutoGazeOutput:
    selected_patch_indices: torch.Tensor
    selected_scales: Optional[torch.Tensor]
    attention_maps: Optional[torch.Tensor]
    token_budget: Optional[int]
    metadata: Dict[str, Any]
```

Required features:

- AutoGaze ON/OFF switch
- Full patch/token path when AutoGaze is OFF
- Selected patch/token path when AutoGaze is ON
- Preserve patch index, scale, and temporal frame index metadata
- Expose visual token counts before and after AutoGaze

Important:

- Do not assume that AutoGaze patch indices directly match the patch indices of the target vision backbone.
- Patch grid mismatch must be handled explicitly by `PatchGridMapper`.
- Do not silently solve mismatch by zero-padding.
- Do not treat post-encoder token pruning as encoder-side acceleration.

---

## 11. Vision Encoder Integration

### 11.1 Modified SigLIP ViT

Purpose:

- Implement the canonical path closest to the original AutoGaze paper/codebase.
- Build the main NVILA baseline.

Required paths:

```text
AutoGaze OFF -> modified SigLIP ViT -> NVILA
AutoGaze ON  -> modified SigLIP ViT -> NVILA
```

Requirements:

- Follow the original implementation for modified SigLIP multi-scale patch handling.
- Preserve original temporal integration behavior as much as possible.
- Document any difference from the original codebase.

---

### 11.2 Vanilla SigLIP ViT

Purpose:

- Isolate the effect of the modified SigLIP.
- Test whether AutoGaze can be used with a non-modified SigLIP ViT.

Required paths:

```text
AutoGaze OFF -> vanilla SigLIP ViT -> NVILA
AutoGaze ON  -> vanilla SigLIP ViT -> NVILA
```

Treat AutoGaze ON + vanilla SigLIP as an experimental ablation.

Possible integration modes:

1. Input-level crop / region reconstruction
2. Post-patch-embedding token masking
3. Compact token gathering, if feasible

Required logging:

- Whether this provides true encoder-side acceleration
- Whether this is only post-patch-embedding token masking
- Whether this is only post-encoder token pruning
- Whether this is only a compatibility path

---

### 11.3 V-JEPA2 Adapter

Purpose:

- Test whether AutoGaze-style spatial selection transfers to a pretrained video ViT encoder.

Requirements:

- Support video input shape `[B, T, C, H, W]`.
- Preserve the temporal dimension explicitly.
- Isolate any V-JEPA2-specific logic behind an adapter.
- Do not directly modify V-JEPA2 internals unless absolutely necessary.

Supported modes:

1. Full video encoder baseline
2. AutoGaze-guided region/crop input mode
3. Token mask mode
4. Compact token mode, only if technically feasible

Required reporting:

- Whether encoder computation is actually reduced
- Whether tokens are reduced only after patch embedding
- Whether only downstream decoder tokens are reduced

---

### 11.4 Generic / Unmodified ViT Adapter

Purpose:

- Validate model-agnostic ViT integration.

Requirements:

- Support arbitrary patch sizes.
- Support arbitrary image resolutions.
- Remap AutoGaze patch indices to the target ViT patch grid.
- Preserve positional metadata.

Supported modes:

- Full-token baseline
- Selected-token mode
- Masked-token mode
- Compact-token mode

---

## 12. MLLM Integration

### 12.1 General MLLM Interface

MLLMs must not be hardcoded for a single model.

Implement a common interface:

```python
class BaseMLLMAdapter:
    def prepare_visual_inputs(self, vision_outputs, metadata=None):
        raise NotImplementedError

    def prepare_text_inputs(self, text_inputs, metadata=None):
        raise NotImplementedError

    def forward(self, visual_inputs, text_inputs, **kwargs):
        raise NotImplementedError

    def generate(self, visual_inputs, text_inputs, **kwargs):
        raise NotImplementedError

    def count_visual_tokens(self, visual_inputs) -> int:
        raise NotImplementedError
```

Required adapters:

- `NVILAAdapter`
- `QwenAdapter`
- `GenericMLLMAdapter`
- `HFMLLMAdapter`

Future MLLMs should be registered through the same interface.

---

### 12.2 NVILA Adapter

Purpose:

- Implement the canonical AutoGaze/SigLIP/NVILA path.

Required paths:

```text
modified SigLIP -> NVILA
vanilla SigLIP -> NVILA
AutoGaze-selected visual tokens -> NVILA
```

Requirements:

- Handle the expected visual feature shape for NVILA.
- Handle token order and positional information explicitly.
- Adapt differences between modified SigLIP and vanilla SigLIP outputs.
- Clearly document assumptions about NVILA visual input format.

---

### 12.3 Qwen Adapter

Purpose:

- Provide an initial adapter for Qwen-family MLLMs without hardcoding the entire pipeline around Qwen.

Requirements:

- Qwen must follow the same `BaseMLLMAdapter` interface.
- Prefer the official Qwen processor path first.
- Do not assume direct visual token injection is supported.
- Implement staged integration.

Suggested stages:

```text
Stage 0:
  Full Qwen official video/image processor baseline

Stage 1:
  AutoGaze-guided frame or region selection before the Qwen processor

Stage 2:
  Post-visual-encoder token pruning if technically feasible

Stage 3:
  Direct visual token injection only if the architecture/API safely supports it
```

---

### 12.4 Other MLLMs

Support future MLLMs through a registry.

Recommended structure:

```text
mllm/
  base_mllm_adapter.py
  nvila_adapter.py
  qwen_adapter.py
  generic_mllm_adapter.py
  hf_mllm_adapter.py
  registry.py
```

Requirements:

- MLLM should be switchable through config.
- Visual input preparation and text prompt preparation should be separated.
- Visual token count must be logged.
- Generation output should be standardized.

Example output:

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch

@dataclass
class MLLMOutput:
    generated_text: Optional[str]
    logits: Optional[torch.Tensor]
    visual_token_count: int
    metadata: Dict[str, Any]
```

---

## 13. Video Task Scope

For Phase 1, focus strictly on video-related tasks.

Required tasks:

1. Video Question Answering
2. Action Recognition

---

### 13.1 Video Question Answering

Requirements:

- Video input shape: `[B, T, C, H, W]`
- Question text input
- Generated answer or classification-style answer output

Datasets:

- `DummyVideoVQADataset` is required.
- VideoMME-style JSON support is optional.
- ActivityNet-QA-style JSON support is optional.
- Hugging Face dataset support is optional but should be added when feasible.
- Do not automatically download external datasets unless explicitly enabled by config.

Metrics:

- exact match
- relaxed accuracy placeholder
- generated answer logging
- Hugging Face Evaluate metric when applicable
- internal metric fallback when Hugging Face Evaluate is not applicable

---

### 13.2 Action Recognition

Requirements:

- Video input shape: `[B, T, C, H, W]`
- Class label output

Datasets:

- `DummyActionRecognitionDataset` is required.
- Kinetics-style folder layout support is optional.
- Hugging Face dataset support is optional but should be added when feasible.
- Do not automatically download external datasets unless explicitly enabled by config.

Metrics:

- top-1 accuracy
- top-5 accuracy
- Hugging Face Evaluate metric when applicable
- internal metric fallback when Hugging Face Evaluate is not applicable

---

## 14. Video Tensor Handling

All video inputs must use:

```python
[B, T, C, H, W]
```

Required modules:

```python
FrameSampler
TemporalAdapter
TemporalAggregator
```

FrameSampler requirements:

- Uniformly sample N frames.
- Support fixed-N sampling.
- Support max-frame mode.
- Preserve original frame indices.

Temporal modes:

1. `frame_wise`
   - Process each frame independently.

2. `mean_pool`
   - Apply temporal mean pooling over frame-wise features.

3. `max_pool`
   - Apply temporal max pooling over frame-wise features.

4. `concat_tokens`
   - Concatenate frame tokens along the temporal/token dimension.

5. `native_autogaze`
   - Follow the original `INTEGRATION.md` image-to-video integration guideline.

---

## 15. Adapter Requirements

Required adapters:

```python
BaseAdapter
PatchGridMapper
PatchIndexAdapter
TokenMaskAdapter
CompactTokenAdapter
VisionFeatureAdapter
TemporalAdapter
MLLMVisualInputAdapter
```

### 15.1 PatchGridMapper

Purpose:

- Map AutoGaze patch indices to the target backbone patch grid.
- Handle patch-size mismatch.
- Handle input-resolution mismatch.
- Handle multi-scale patch indices.

Important:

- Do not assume that AutoGaze indices and target ViT indices are identical.
- Log and document the remapping strategy.

---

### 15.2 TokenMaskAdapter

Purpose:

- Convert selected patch indices into boolean token masks.
- Support masked-token baselines.

---

### 15.3 CompactTokenAdapter

Purpose:

- Gather selected tokens into a compact token sequence.
- Preserve positional metadata.
- Verify whether the downstream MLLM or decoder can accept compact sequences.

---

### 15.4 VisionFeatureAdapter

Purpose:

- Match output dimensions from the vision backbone to the MLLM or task decoder input dimensions.
- Provide a simple Linear Projection placeholder.

Important:

- Do not claim zero-shot performance from randomly initialized adapters.
- If an adapter is not trained, mark the path as compatibility-only.

---

## 16. Metrics and Profiling

### 16.1 Efficiency Metrics

Required metrics:

- peak VRAM in MB
- inference latency in ms
- throughput in videos/sec
- FPS
- visual token count before AutoGaze
- visual token count after AutoGaze
- token reduction ratio
- number of selected patches per frame
- number of selected patches per scale
- AutoGaze latency
- ViT latency
- MLLM prefill latency
- MLLM decode latency, if applicable
- end-to-end latency

Timing rules:

- Run GPU warm-up iterations before benchmark timing.
- The number of warm-up iterations must be configurable.
- Whether to include data loading time in timing must be configurable.

CUDA requirements:

- Use `torch.cuda.synchronize()` for timing.
- Log `torch.cuda.max_memory_allocated()`.

MPS requirements:

- Do not use FlashAttention.
- Use PyTorch SDPA or eager attention fallback.
- If peak memory measurement is unavailable on MPS, record it as `N/A`.

CPU requirements:

- CPU is for smoke tests only.
- Do not interpret CPU timing as the main benchmark result.

---

### 16.2 Performance Metrics

Video VQA:

- exact match
- relaxed accuracy placeholder
- generated answer logging
- Hugging Face Evaluate metric when applicable
- internal metric fallback

Action Recognition:

- top-1 accuracy
- top-5 accuracy
- Hugging Face Evaluate metric when applicable
- internal metric fallback

---

## 17. Benchmark Axes

Benchmark across multiple video settings when feasible.

| Axis | Example Values |
|---|---|
| Frame count | 8, 16, 32, 64, 128, 256, 512, 1024 |
| Resolution | 224p, 448p, 720p, 1080p, 4K if feasible |
| Token budget | low / medium / high or explicit selected patch counts |
| AutoGaze mode | OFF / ON |
| Vision encoder | modified SigLIP / vanilla SigLIP / V-JEPA2 / generic ViT |
| MLLM | NVILA / Qwen / other MLLM |
| Integration mode | full / hook / native / crop / mask / compact / official_processor |
| Source mode | local / Hugging Face / mixed / offline cache |

Do not claim true acceleration unless the corresponding compute stage is actually reduced.

Examples:

- Masking tokens after the full ViT forward pass is not ViT acceleration.
- Pruning tokens after the visual encoder is not encoder-side acceleration.
- Reducing visual tokens before MLLM prefill may accelerate MLLM prefill but not necessarily the vision encoder.

---

## 18. Visualization

Implement a common visualizer interface.

```python
BaseVisualizer
AutoGazeVisualizer
TaskOutputVisualizer
FullPipelineVisualizer
```

Required features:

- Save selected AutoGaze patch indices.
- Visualize scale indicators.
- Save attention maps if available.
- Save visualization per temporal frame.
- Overlay predicted Video VQA answers when applicable.
- Save top-k Action Recognition labels when applicable.
- Support AutoGaze-only visualization mode.
- Support full-pipeline visualization mode.

Example output structure:

```text
outputs/
  exp_name/
    visualizations/
      autogaze_only/
      full_pipeline/
      video_vqa/
      action_recognition/
```

---

## 19. Documentation

Create the following documents in Korean under `docs/`.

### 19.1 `EASY_TO_START.md`

Include:

- installation guide
- Linux/CUDA execution guide
- Mac/MPS execution guide
- dummy video inference example
- AutoGaze ON/OFF example
- modified SigLIP / vanilla SigLIP selection guide
- NVILA / Qwen / generic MLLM selection guide
- Hugging Face model benchmark example
- Hugging Face dataset benchmark example
- offline Hugging Face cache example

---

### 19.2 `INTEGRATION_extended.md`

Create a new file.

Include:

- summary of the original `INTEGRATION.md`
- extension policy for this project
- how to connect AutoGaze patch indices to other backbones
- modified SigLIP integration
- vanilla SigLIP integration
- V-JEPA2 integration
- generic ViT integration
- NVILA integration
- Qwen integration
- other MLLM integration
- Hugging Face MLLM integration
- temporal dimension handling
- full / hook / native integration modes
- official processor mode for Hugging Face MLLMs

Important:

- Do not copy or modify the original `INTEGRATION.md`.
- Only document the extended integration policy.

---

### 19.3 `benchmark.md`

Create an official benchmark document in Korean.

Include:

1. Public Benchmark Survey
   - Summarize existing video MLLMs as external baselines.
   - Follow the purpose and structure of Table 1 in the AutoGaze paper.
   - Include max frame count, max resolution, and benchmark scores where available.
   - Do not imply that external baselines used AutoGaze.

2. Reproduction Benchmark Plan
   - Define which original AutoGaze settings we will try to reproduce.
   - Include the canonical SigLIP/NVILA ablation:
     - AutoGaze OFF + modified SigLIP + NVILA
     - AutoGaze ON + modified SigLIP + NVILA
     - AutoGaze OFF + vanilla SigLIP + NVILA
     - AutoGaze ON + vanilla SigLIP + NVILA

3. Extension Benchmark Plan
   - Define how to benchmark:
     - V-JEPA2
     - generic ViT
     - Qwen-family MLLM
     - other MLLMs

4. Hugging Face-Based Benchmark Plan
   - Define how to run benchmarks using Hugging Face models.
   - Define how to run benchmarks using Hugging Face datasets.
   - Define how to run offline cache-based benchmarks.
   - Define how to pin model and dataset revisions.
   - Define how to separate official processor inference from AutoGaze-guided inference.

5. Measurement Methodology
   - warm-up iterations
   - CUDA synchronization
   - MPS fallback behavior
   - whether data loading is included
   - latency breakdown:
     - AutoGaze latency
     - ViT latency
     - MLLM prefill latency
     - MLLM decode latency
     - end-to-end latency

6. Benchmark Result Table Templates
   - efficiency table
   - performance table
   - token reduction table
   - resolution/frame scaling table
   - Hugging Face benchmark result table
   - AutoGaze effectiveness analysis table

---

### 19.4 `benchmark_analysis.md`

Create this as a PoC analysis template.

Include:

- experiment setting summary
- AutoGaze ON/OFF comparison table
- modified SigLIP vs vanilla SigLIP comparison table
- NVILA vs other MLLM comparison table
- Hugging Face model comparison table
- Hugging Face dataset comparison table
- token reduction vs accuracy
- latency vs accuracy
- VRAM vs accuracy
- cases where AutoGaze helps
- cases where AutoGaze hurts
- cases where AutoGaze helps without fine-tuning
- cases where adapter fine-tuning is required

### 19.5 `QUICK_START Reference`

Create `docs/QUICK_START_reference.md`.

It should summarize:
- original inference commands
- checkpoint paths
- model config paths
- runtime arguments
- resolution scaling options
- frame count options, if available
- expected input/output formats
- hardware assumptions
- differences between the original quick start and this PoC pipeline

### 19.6 `INFERENCE_GUIDE.md`

Maintain `docs/INFERENCE_GUIDE.md` as the user-facing inference guide.

It should align with the original `QUICK_START.md` for:
- AutoGaze-only inference behavior
- modified SigLIP integration behavior
- checkpoint/model ID conventions
- runtime arguments
- resolution scaling options
- frame count assumptions

Commands not present in the original quick start must be labeled as runnable, stub-only, or future work.


Suggested table:

```markdown
## Public Claim vs Internal Measurement

| Metric | Reported by AutoGaze Paper/Model Card | Our Reproduction | Our Extension Result | Notes |
|---|---:|---:|---:|---|
| Token reduction | TBD | TBD | TBD | Report per dataset/resolution |
| ViT latency speedup | TBD | TBD | TBD | Separate encoder-side vs token-only reduction |
| MLLM latency speedup | TBD | TBD | TBD | Separate prefill/decode |
| VideoMME | TBD | TBD | TBD | Use same evaluation protocol if possible |
| HLVid improvement | TBD | TBD | TBD | Check dataset availability |
| High-resolution long-video support | TBD | TBD | TBD | Record actual hardware and memory limits |
```

Suggested Hugging Face result table:

```markdown
## Hugging Face Benchmark Result

| Experiment | HF Model ID | Model Revision | HF Dataset ID | Dataset Split | AutoGaze | Integration Mode | Samples | Metric | Latency | VRAM | Notes |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | OFF | official_processor | TBD | TBD | TBD | TBD | baseline |
| TBD | TBD | TBD | TBD | TBD | ON | input_region_selection | TBD | TBD | TBD | TBD | AutoGaze-guided |
```

---

## 20. Setup Scripts

Create:

```text
scripts/setup_linux.sh
scripts/setup_mac.sh
scripts/download_hf_assets.py
```

### 20.1 Linux Setup

Requirements:

- CUDA-oriented setup
- optional FlashAttention installation
- graceful fallback if FlashAttention is unsupported
- install Hugging Face dependencies:
  - `transformers`
  - `datasets`
  - `evaluate`
  - `huggingface_hub`
  - `accelerate`, if needed
- include a benchmark smoke-test command

### 20.2 Mac Setup

Requirements:

- MPS-oriented setup
- do not install FlashAttention
- use PyTorch SDPA or eager attention fallback
- install Hugging Face dependencies:
  - `transformers`
  - `datasets`
  - `evaluate`
  - `huggingface_hub`
- include an MPS smoke-test command

### 20.3 Hugging Face Asset Download Script

Create:

```text
scripts/download_hf_assets.py
```

Requirements:

- download model snapshots
- download processor/tokenizer files
- optionally prepare dataset cache
- support revision pinning
- support custom cache directory
- never save authentication tokens
- write an asset manifest file

Example manifest:

```json
{
  "models": [
    {
      "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "revision": "TBD",
      "cache_dir": "./hf_cache"
    }
  ],
  "datasets": [
    {
      "dataset_id": "TBD",
      "revision": "TBD",
      "cache_dir": "./hf_cache"
    }
  ]
}
```

---

## 21. Testing

Implement pytest-based unit tests.

Required tests:

```text
tests/
  test_config_loading.py
  test_dummy_video_dataset.py
  test_frame_sampler.py
  test_autogaze_on_off.py
  test_patch_grid_mapper.py
  test_token_mask_adapter.py
  test_compact_token_adapter.py
  test_temporal_adapter.py
  test_modified_siglip_path.py
  test_vanilla_siglip_path.py
  test_mllm_adapter_interface.py
  test_hf_model_loader.py
  test_hf_dataset_loader.py
  test_hf_evaluate_metric.py
  test_visualizer.py
  test_reproducibility.py
```

Shape mismatch tests must cover:

- `[B, T, C, H, W]` to sampled video
- sampled video to AutoGaze patch indices
- AutoGaze patch indices to target ViT patch grid
- selected tokens to vision encoder
- vision encoder output to MLLM adapter
- MLLM adapter output to task decoder or generation output
- Hugging Face dataset sample to internal video sample format
- Hugging Face model output to standardized `MLLMOutput`

---

## 22. Reproducibility

Implement reproducibility utilities.

Required:

- fix Python random seed
- fix NumPy seed
- fix PyTorch seed
- fix CUDA seed if available
- provide deterministic mode through config
- pin Hugging Face model revisions
- pin Hugging Face dataset revisions when available

For every benchmark run, save:

- resolved config
- git commit hash
- package versions
- device information
- CUDA/MPS availability
- precision setting
- benchmark timestamp
- model checkpoints used
- Hugging Face model IDs and revisions
- Hugging Face dataset IDs and revisions
- cache directory
- offline mode status
- `trust_remote_code` status

---

## 23. Device Compatibility

Supported devices:

1. CUDA
2. MPS
3. CPU

Policy:

- CUDA is the primary benchmark target.
- MPS is for smoke tests and local development.
- CPU is for minimal functional tests only.

MPS-specific requirements:

- Do not use FlashAttention.
- Handle unsupported operator fallback.
- Handle dtype issues explicitly.
- If some profiling metrics are unavailable on MPS, record them as `N/A`.

Hugging Face-specific device requirements:

- Support `device_map` when appropriate.
- Support explicit device placement when `device_map` is not used.
- Support dtype configuration:
  - `float32`
  - `float16`
  - `bfloat16`, if supported
- Do not silently use a different dtype from the config.

---

## 24. Non-goals for Phase 1

Do not:

- rewrite the AutoGaze core from scratch
- train AutoGaze from scratch
- fully integrate every possible MLLM
- fully support every dataset
- automatically download large external datasets unless explicitly enabled by config
- claim zero-shot performance from randomly initialized adapters
- claim encoder acceleration when only token masking is applied after the full encoder computation
- modify the original `INTEGRATION.md`
- claim that external MLLMs used AutoGaze unless explicitly reported
- claim that Hugging Face public models support AutoGaze token injection unless directly verified
- log Hugging Face access tokens

---

## 25. Required Reporting

Each experiment result must include:

- experiment ID
- AutoGaze ON/OFF
- vision encoder type
- MLLM or decoder type
- integration mode
- source mode:
  - local
  - Hugging Face
  - mixed
  - offline cache
- input frame count `T`
- input resolution
- original visual token count
- selected visual token count
- token reduction ratio
- latency
- throughput
- peak VRAM
- task metric
- note on whether acceleration is:
  - true encoder-side acceleration
  - post-patch-embedding token masking
  - post-encoder token pruning
  - downstream token reduction only
  - compatibility-only adapter path

For Hugging Face-based experiments, additionally report:

- model ID
- model revision
- processor/tokenizer ID
- processor/tokenizer revision
- dataset ID
- dataset config
- dataset split
- dataset revision, if available
- cache mode
- offline mode status
- `trust_remote_code` status
- number of evaluated samples
- metric implementation source:
  - Hugging Face Evaluate
  - internal metric
  - custom script

---

## 26. Final Expected Deliverables

Expected deliverables:

1. Config-driven runnable pipeline
2. AutoGaze ON/OFF switch
3. modified SigLIP ViT adapter
4. vanilla SigLIP ViT adapter
5. NVILA adapter
6. generic MLLM adapter
7. Qwen adapter placeholder or initial implementation
8. Hugging Face MLLM adapter
9. V-JEPA2 adapter placeholder or initial implementation
10. generic ViT adapter
11. Video VQA dummy pipeline
12. Action Recognition dummy pipeline
13. Hugging Face model loader
14. Hugging Face dataset loader
15. Hugging Face Evaluate metric wrapper
16. benchmark logger
17. visualizer
18. pytest unit tests
19. Linux setup script
20. Mac setup script
21. Hugging Face asset download script
22. Korean documentation files
23. NVILA-HD-Video canonical PoC script
24. NVILA-HD-Video canonical smoke tests
25. `docs/NVILA_HD_VIDEO_REFERENCE.md`
---

## 27. Summary

The first-stage canonical experiment is:

```text
AutoGaze ON/OFF
×
modified SigLIP ViT / vanilla SigLIP ViT
×
NVILA
```

The extension experiments are:

```text
AutoGaze ON/OFF
×
V-JEPA2 / generic ViT / other vision backbone
×
NVILA / Qwen / other MLLM / task decoder
```

The Hugging Face benchmark path should support:

```text
Hugging Face model loading
Hugging Face dataset loading
Hugging Face Evaluate-based metrics
offline cache mode
revision-pinned reproducibility
```

The benchmark survey should follow the role of Table 1 in the AutoGaze paper:

```text
Compare existing video MLLMs as external baselines by:
- max frame count
- max resolution
- benchmark scores
- long-video capability
- high-resolution video capability
```

The benchmark survey should not be interpreted as:

```text
"Other models used AutoGaze."
```

The most important implementation principle is:

```text
Do not lock AutoGaze to the original SigLIP/NVILA path only.
Modularize AutoGaze as a backbone-agnostic patch/token selector whenever possible.
```

Always distinguish between:

```text
1. true encoder-side acceleration
2. post-patch-embedding token masking
3. post-encoder token pruning
4. downstream token reduction only
5. compatibility-only adapter path
```
