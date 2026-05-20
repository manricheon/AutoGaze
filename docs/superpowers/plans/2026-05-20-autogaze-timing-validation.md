# AutoGaze Timing Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify whether AutoGaze latency is measured correctly in the current NVILA runner, and explain any speed gap against the original AutoGaze repository path.

**Architecture:** Treat this as a measurement audit, not an optimization pass. Use the original Quick Start section **"Running AutoGaze on a Video"** with `external/AutoGaze/assets/example_input.mp4` as the mandatory anchor case, then compare three paths with the same video, frame count, dtype, batch size, gazing settings, and target patch config: direct original AutoGaze forward, current NVILA processor hook, and our stream-profile chunk path. Produce a Markdown report that separates measurement-boundary differences from actual workload differences.

**Tech Stack:** Python, PyTorch, Transformers remote code, PyAV, existing `repro.nvila_runner`, `repro.autogaze_bench`, `repro.hlvid_example_autogaze`, pytest.

---

## Current Measurement Map

The current NVILA runner does not edit AutoProcessor code. It wraps runtime methods in `repro/nvila_runner.py`:

```text
processor(...)
|-- processor_total
|   |-- video_tiling_and_tensorize       -> video_tiling_ms
|   |-- autogaze_total                   -> autogaze_total_ms
|   |   `-- autogaze_forward_batched     -> autogaze_model_forward_ms
|   `-- video_decode_sampling            -> video_decode_ms

model.generate(...)
|-- vision_encode_total                  -> vision_encoder_ms
|   |-- siglip_vision_tower              -> siglip_vision_ms
|   `-- mm_projector                     -> mm_projector_ms
`-- llm_forward                          -> llm_forward_ms
```

Important interpretation:

- `autogaze_total_ms` is a parent stage. It includes AutoGaze forward plus gaze-info construction, padding, splitting, and bookkeeping inside the NVILA processor.
- `autogaze_model_forward_ms` is the wrapped `_run_autogaze_batched` stage. If that method internally loops over batches, `count` may still be `1`, but `total_ms` includes the full internal loop.
- `autogaze_total_ms` and `autogaze_model_forward_ms` must not be added together.
- The required anchor comparison is the original Quick Start **"Running AutoGaze on a Video"** path: `assets/example_input.mp4`, first 16 frames, AutoGaze image processor resize to 224x224, then `autogaze_model({"video": video_input_autogaze}, gazing_ratio=0.75, task_loss_requirement=0.7)`.
- NVILA-HD may run many tile sequences, target scales `[56,112,196,392]`, patch size `14`, and many more tile-frame instances. That realistic path must be compared separately from the Quick Start anchor.

## Primary Hypotheses

1. The timing hook itself is mostly correct because it synchronizes before and after the wrapped method, but the current report may compare different workload boundaries.
2. The original repo path is faster because it uses one 16-frame 224x224 clip, while NVILA-HD runs tiled chunks across selected frames and spatial tiles.
3. `autogaze_total_ms` may look slower than original AutoGaze forward because it includes processor-side gaze-info assembly around the forward call.
4. `autogaze_model_forward_ms` may still be slower than the original quickstart if target scales, target patch size, dtype, frames, tile count, or batch splitting differ.
5. HLVid summary previously hid `stage_timings_ms.processor.autogaze_forward_batched.*`; that is now reported, but old result files will not contain the new summary fields unless regenerated.

## Files

- Inspect: `repro/nvila_runner.py`
- Inspect: `repro/autogaze_bench.py`
- Inspect: `repro/hlvid_example_autogaze.py`
- Inspect: `external/AutoGaze/QUICK_START.md`
- Create: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`
- Optional create: `repro/autogaze_timing_audit.py`
- Optional test: `tests/test_autogaze_timing_audit.py`

## Task 1: Establish Comparable Measurement Boundaries

**Files:**
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Create the report skeleton**

Create `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md` with these sections:

```markdown
# AutoGaze 시간 측정 검증 리포트

## 결론 요약

## 측정 경계 정의

## 현재 NVILA runner 측정 방식

## Original AutoGaze repo 측정 방식

## Apples-to-Apples 실험 설계

## 결과 표

## 차이 원인 분석

## 판정

## 다음 조치
```

- [ ] **Step 2: Document the three latency levels**

Add this table to the report:

```markdown
| Level | Metric | Includes | Excludes | Purpose |
| --- | --- | --- | --- | --- |
| L0 | `autogaze_forward_kernel_ms` | CUDA kernels inside direct AutoGaze model forward | Python/tensor prep | Pure GPU model check |
| L1 | `autogaze_model_forward_ms` | `_run_autogaze_batched` wall time, synced | parent gaze-info work | Compare model call inside NVILA processor |
| L2 | `autogaze_total_ms` | `_get_gazing_info_from_videos` wall time, synced | video decode, tiling if separately hooked | Real AutoGaze processor overhead |
| L3 | `processor_total` | whole NVILA processor call | model.generate | End-to-end preprocessing cost |
```

- [ ] **Step 3: Add the key rule**

Add this text:

```markdown
`autogaze_total_ms`와 `autogaze_model_forward_ms`는 parent-child 관계다. 둘을 더하면 중복 계산이다. Original repo의 quickstart forward time과 직접 비교할 1차 대상은 `autogaze_model_forward_ms` 또는 direct benchmark의 `latency_ms.autogaze.median`이다. NVILA pipeline에서 실제 사용자 체감 overhead를 볼 때는 `autogaze_total_ms`를 본다.
```

## Task 2: Run Quick Start "Running AutoGaze on a Video" Anchor Baseline

**Files:**
- Use: `repro/autogaze_bench.py`
- Use: `external/AutoGaze/assets/example_input.mp4`
- Reference: `external/AutoGaze/QUICK_START.md`
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Record the exact Quick Start contract**

Add this text to the report:

```markdown
Quick Start anchor:
- Source: `external/AutoGaze/QUICK_START.md`, section `Running AutoGaze on a Video`
- Video: `external/AutoGaze/assets/example_input.mp4`
- Sampled frames: `list(range(autogaze_model.config.max_num_frames))`, normally first 16 frames
- Preprocess: `AutoGazeImageProcessor` + `transform_video_for_pytorch`, default 224x224
- Model call: `autogaze_model({"video": video_input_autogaze}, gazing_ratio=0.75, task_loss_requirement=0.7)`
- This is the primary sanity check for whether our AutoGaze forward timing is in the right range.
```

- [ ] **Step 2: Run original-style direct benchmark**

Run on CUDA first; MPS only if CUDA is unavailable.

```bash
.venv/bin/python -m repro.autogaze_bench \
  --autogaze-repo external/AutoGaze \
  --autogaze-model /path/to/local/autogaze-or-hf-id \
  --siglip-model google/siglip2-base-patch16-224 \
  --video external/AutoGaze/assets/example_input.mp4 \
  --device cuda \
  --dtype float16 \
  --frames 16 \
  --gazing-ratio 0.75 \
  --task-loss-requirement 0.7 \
  --warmup 3 \
  --repeat 10 \
  --output-json outputs/autogaze_repro/timing_audit/quickstart_original_example_input_16f.json \
  --output-csv outputs/autogaze_repro/timing_audit/quickstart_original_example_input_16f.csv
```

Expected output fields:

```json
{
  "latency_ms": {
    "autogaze": {"median": "..."},
    "siglip_full": {"median": "..."},
    "siglip_gazed": {"median": "..."}
  },
  "gaze": {
    "raw_patch_budget": "...",
    "selected_non_padded_patches": "...",
    "token_reduction_ratio": "..."
  }
}
```

- [ ] **Step 3: Record workload facts**

Add these fields to the report:

```markdown
| Field | Value |
| --- | --- |
| video | `external/AutoGaze/assets/example_input.mp4` |
| frames | `16` |
| source section | `external/AutoGaze/QUICK_START.md` / `Running AutoGaze on a Video` |
| input tensor | from `shapes.video_input_autogaze` |
| dtype | `float16` |
| raw patch budget | from `gaze.raw_patch_budget` |
| selected patches | from `gaze.selected_non_padded_patches` |
| AutoGaze direct median | from `latency_ms.autogaze.median` |
```

- [ ] **Step 4: Preserve the Quick Start output sanity checks**

Add these checks to the report:

```markdown
The Quick Start text says `gazing_pos` records selected/padded patch indices and gives an example output shape around `1 x 348`. Our exact count can differ by model/checkpoint/version, but the report must record:
- `gaze.total_gaze_slots`
- `gaze.selected_non_padded_patches`
- `gaze.padded_gazing_positions`
- `gaze.token_reduction_ratio`
```

## Task 3: Run Current NVILA Processor Path With Minimal Comparable Workload

**Files:**
- Use: `repro/nvila_runner.py`
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Run NVILA with smallest comparable shape**

Use the same `external/AutoGaze/assets/example_input.mp4` and one tile with 16 video frames so the NVILA path is as close as possible to the original Quick Start anchor while still using the NVILA processor.

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --dtype float16 \
  --model-path /path/to/local/NVILA-8B-HD-Video \
  --autogaze-model /path/to/local/autogaze-or-hf-id \
  --video external/AutoGaze/assets/example_input.mp4 \
  --prompt "Question: What is visible in the video? A. road B. kitchen C. beach D. office Please answer with the letter." \
  --gazing-mode autogaze \
  --num-video-frames 16 \
  --num-video-frames-thumbnail 0 \
  --max-tiles-video 1 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --max-new-tokens 1 \
  --measure-ttft \
  --warmup-runs 3 \
  --repeat-runs 10 \
  --print-summary \
  --output-json outputs/autogaze_repro/timing_audit/nvila_quickstart_example_input_16f_1tile.json \
  --summary-json outputs/autogaze_repro/timing_audit/nvila_quickstart_example_input_16f_1tile_summary.json
```

- [ ] **Step 2: Extract the comparison fields**

Record:

```markdown
| Field | JSON path |
| --- | --- |
| AutoGaze total | `repeat_summary.autogaze_total_ms.median` |
| AutoGaze forward | `repeat_summary.autogaze_model_forward_ms.median` |
| Batched total | `result.stage_timings_ms.processor.autogaze_forward_batched.total_ms` |
| Batched count | `result.stage_timings_ms.processor.autogaze_forward_batched.count` |
| Raw patch budget | `result.token_metrics.autogaze_input_patch_tokens` |
| Selected patches | `result.token_metrics.autogaze_selected_patch_tokens` |
| Tile-frame instances | `result.token_metrics.autogaze_input_tile_frame_instances` |
```

- [ ] **Step 3: Compare against direct original**

Add this decision rule:

```markdown
If Quick Start direct `latency_ms.autogaze.median` and NVILA `autogaze_model_forward_ms` differ by less than 20-30% after matching example video/frame/tile/dtype/patch config, the wrapper timing is likely valid. If `autogaze_total_ms` is much larger but `autogaze_model_forward_ms` is close, the extra cost is processor-side gaze-info assembly. If `autogaze_model_forward_ms` is still much larger on the same `example_input.mp4` anchor case, inspect target scales, target patch size, tensor shape, dtype, and batch splitting before interpreting HLVid-scale runs.
```

## Task 4: Run NVILA Realistic Workload and Normalize by Work Units

**Files:**
- Use: `repro/nvila_runner.py`
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Run the actual workload suspected to be slow**

Use the exact command that looked slow on CUDA. If unknown, use this default:

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode single \
  --device cuda \
  --dtype float16 \
  --model-path /path/to/local/NVILA-8B-HD-Video \
  --autogaze-model /path/to/local/autogaze-or-hf-id \
  --video /path/to/hlvid_or_test_video.mp4 \
  --prompt "Question: <same prompt as the slow run>" \
  --gazing-mode autogaze \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --max-new-tokens 1 \
  --measure-ttft \
  --warmup-runs 3 \
  --repeat-runs 10 \
  --print-summary \
  --output-json outputs/autogaze_repro/timing_audit/nvila_realistic_autogaze.json \
  --summary-json outputs/autogaze_repro/timing_audit/nvila_realistic_autogaze_summary.json
```

- [ ] **Step 2: Normalize timing by work units**

Add these computed rows:

```markdown
| Metric | Formula |
| --- | --- |
| forward ms per tile sequence | `autogaze_model_forward_ms / tile_sequences` |
| forward ms per tile-frame | `autogaze_model_forward_ms / autogaze_input_tile_frame_instances` |
| selected patches per ms | `autogaze_selected_patch_tokens / autogaze_model_forward_ms` |
| raw patches per ms | `autogaze_input_patch_tokens / autogaze_model_forward_ms` |
| processor overhead over forward | `autogaze_total_ms - autogaze_model_forward_ms` |
| overhead percent | `(autogaze_total_ms - autogaze_model_forward_ms) / autogaze_total_ms * 100` |
```

- [ ] **Step 3: Decide whether the speed gap is workload-driven**

Use this rule:

```markdown
If realistic workload has N times more tile-frame instances than the original direct benchmark and latency grows roughly with N, the difference is expected workload scaling. If normalized ms per tile-frame is much worse, investigate batching, dtype, device placement, target scales, and processor-side Python overhead.
```

## Task 5: Cross-Check With Stream-Profile AutoGaze-Only Path

**Files:**
- Use: `repro/nvila_runner.py`
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Run stream-profile for the same video/config**

```bash
.venv/bin/python -m repro.nvila_runner \
  --mode stream-profile \
  --device cuda \
  --stream-dtype float16 \
  --autogaze-model /path/to/local/autogaze-or-hf-id \
  --video /path/to/hlvid_or_test_video.mp4 \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 8 \
  --stream-chunk-frames 16 \
  --max-batch-size-autogaze 16 \
  --stream-siglip-mode none \
  --stream-profile-json outputs/autogaze_repro/timing_audit/stream_profile_autogaze_only.json
```

- [ ] **Step 2: Compare stream-profile and NVILA processor forward**

Compare:

```markdown
| Source | Field |
| --- | --- |
| NVILA single | `result.autogaze_model_forward_ms` |
| NVILA single nested | `result.stage_timings_ms.processor.autogaze_forward_batched.total_ms` |
| stream-profile | `timing_ms.tile_autogaze_forward` |
```

Expected:

```markdown
The two forward-only values should be in the same ballpark after matching frame count, tiles, batch size, dtype, and target scales. If stream-profile is much faster, NVILA remote processor likely adds overhead inside `_run_autogaze_batched` or uses a different data packing path. If stream-profile is similar but `autogaze_total_ms` is much larger, the difference is gaze-info bookkeeping outside the model forward.
```

## Task 6: Add Missing Diagnostic Logging If Needed

**Files:**
- Optional modify: `repro/nvila_runner.py`
- Optional test: `tests/test_nvila_runner.py`

- [ ] **Step 1: Add a debug flag only if existing logs are insufficient**

Add a CLI flag:

```python
parser.add_argument("--debug-autogaze-timing", action="store_true")
```

- [ ] **Step 2: Log method call shapes without changing timing boundaries**

Extend `ProfilePatches._patch_method` so that when `stage == "autogaze_forward_batched"` and `--debug-autogaze-timing` is enabled, the result JSON includes:

```json
{
  "autogaze_forward_batched_calls": [
    {
      "call_index": 0,
      "elapsed_ms": 123.4,
      "input_shapes": ["..."],
      "output_keys": ["gazing_pos", "if_padded_gazing", "num_gazing_each_frame"]
    }
  ]
}
```

Do not implement this unless the first five tasks cannot explain the speed difference.

## Task 7: Write The Final Report

**Files:**
- Modify: `docs/AUTOGAZE_TIMING_VALIDATION_REPORT_KO.md`

- [ ] **Step 1: Fill the result table**

Use this exact table:

```markdown
| Run | Frames | Tiles | Tile-frame instances | Dtype | Batch | Raw patches | Selected patches | AutoGaze forward median | AutoGaze total median | Forward count | Notes |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Quick Start direct `example_input.mp4` | 16 | n/a | n/a | | | | | | n/a | n/a | Original `Running AutoGaze on a Video` anchor |
| NVILA `example_input.mp4` comparable | 16 | 1 | | | | | | | | | Same video, closest NVILA processor path |
| NVILA realistic | | | | | | | | | | |
| stream-profile | | | | | | | | | | |
```

- [ ] **Step 2: Fill the judgment section**

Use one of these conclusions:

```markdown
판정 A: 측정은 정상이고 workload가 달랐다.
근거: direct original과 NVILA comparable의 forward-only latency가 유사하고, realistic run의 latency 증가는 tile-frame/patch 수 증가와 비례한다.
```

```markdown
판정 B: forward-only는 정상이나 processor overhead가 크다.
근거: `autogaze_model_forward_ms`는 direct/stream-profile과 유사하지만 `autogaze_total_ms - autogaze_model_forward_ms`가 크다.
```

```markdown
판정 C: `_run_autogaze_batched` 내부 경로가 느리다.
근거: 같은 tile-frame/patch/dtype 조건에서도 NVILA `autogaze_model_forward_ms`가 direct/stream-profile보다 크게 느리다. 다음 조치는 `_run_autogaze_batched` 내부 packing/batching/device 이동 확인이다.
```

## Verification

Run:

```bash
.venv/bin/python -m pytest tests/test_autogaze_bench.py tests/test_hlvid.py tests/test_hlvid_batch_benchmark.py tests/test_nvila_runner.py
git diff --check
```

Expected:

```text
pytest exits 0
git diff --check exits 0
```

## Self-Review

- This plan distinguishes measurement correctness from workload mismatch.
- It requires the original Quick Start `Running AutoGaze on a Video` example video as the anchor comparison.
- It compares parent stage, forward-only stage, and direct original model forward separately.
- It avoids changing original AutoGaze code.
- It uses existing scripts first and only proposes new instrumentation if the current evidence is insufficient.
