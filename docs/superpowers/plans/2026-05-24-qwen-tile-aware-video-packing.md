# Qwen Tile-Aware Video Packing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experimental Qwen tile-aware packing modes that approximate NVILA-HD high-resolution spatial tiling while preserving clear reporting that this is zero-shot experimental Qwen packing.

**Architecture:** Keep the existing Qwen native-grid sparse path unchanged. Add two Qwen modes: dense tile-packed and AutoGaze sparse tile-packed. The new path decodes sampled frames, creates NVILA-style spatial tiles, packs tiles as tile-major temporal chunks in a Qwen video sequence, maps AutoGaze tile-local selected patch indices to Qwen visual indices, then reuses the existing chunked Qwen ViT feature path.

**Tech Stack:** Python, Transformers Qwen processor/model APIs, existing `SparseSelectionPlan`, existing Qwen chunked ViT helpers, pytest.

---

### Task 1: Lock CLI And Routing Contract

**Files:**
- Modify: `repro/flexible_runner.py`
- Modify: `repro/plugins/mllm_adapters.py`
- Test: `tests/test_flexible_runner.py`
- Test: `tests/test_mllm_adapters.py`

- [ ] Add `qwen_tile_packed_vit` and `qwen_tile_packed_vit_autogaze_sparse` to `QWEN_VIT_MODE_CHOICES`.
- [ ] Add adapter routing so the dense mode calls `_run_qwen_tile_packed_vit_generate()` and sparse mode calls `_run_qwen_tile_packed_vit_autogaze_sparse_generate()`.
- [ ] Test parse/routing with monkeypatched methods.

### Task 2: Build Tile-Packed Qwen Inputs

**Files:**
- Modify: `repro/plugins/mllm_adapters.py`
- Test: `tests/test_mllm_adapters.py`

- [ ] Add `_qwen_tile_packed_video_frames()` that reuses runner-side sampled frames, computes NVILA-style `spatial_tile_grid`, and emits tile-major temporal chunk order followed by optional thumbnail tail.
- [ ] Add `_build_qwen_tile_packed_grid_inputs()` that feeds the tile-packed frame list to the Qwen processor.
- [ ] Record metadata: source frames, tiles per frame, tile size, packed main frames, thumbnail frames, packed total frames, and `position_semantics=spatial_tiles_encoded_as_temporal_sequence`.

### Task 3: Map AutoGaze Tile Indices To Packed Qwen Visual Indices

**Files:**
- Modify: `repro/plugins/mllm_adapters.py`
- Test: `tests/test_mllm_adapters.py`

- [ ] Add `_qwen_tile_packed_mapping_from_sparse_plan_path()` and helpers that map `(frame_order, tile_id, patch_index, scale_size)` to packed temporal index and Qwen local row/col.
- [ ] Add thumbnail keep-all indices for the appended global-context tail.
- [ ] Test exact small grids and temporal packing math.

### Task 4: Execute Dense And Sparse Tile-Packed Modes

**Files:**
- Modify: `repro/plugins/mllm_adapters.py`
- Modify: `repro/plugin_hlvid_benchmark.py`
- Modify: `scripts/run_hlvid_folder_benchmark.py`
- Test: `tests/test_plugin_hlvid_benchmark.py`
- Test: `tests/test_run_hlvid_folder_benchmark_wrapper.py`

- [ ] Dense mode: tile-packed input -> chunked Qwen ViT -> original placeholder packing -> generate.
- [ ] Sparse mode: tile-packed input -> AutoGaze packed index mapping -> selected Qwen ViT features -> selected placeholders -> generate.
- [ ] Add optional plugin modes/suite entries without replacing the safer native-grid Qwen sparse path.

### Task 5: Docs And Verification

**Files:**
- Modify: `docs/AUTOGAZE_PLUGIN_IMPLEMENTATION_PLAN_KO.md`
- Modify: `docs/AUTOGAZE_REPRO_RUNBOOK_KO.md`
- Modify: `docs/AUTOGAZE_HLVID_BENCHMARK_GUIDE_KO.md`

- [ ] Document that tile-aware Qwen packs spatial tiles on Qwen's temporal axis and is zero-shot experimental.
- [ ] Run targeted tests, compile, then full pytest.
