# NVILA-HD Inference Update Audit

This audit compares the PoC inference surface with `docs/nvila-hd-video-readme.md` and `QUICK_START.md`. Original AutoGaze source, `QUICK_START.md`, `INTEGRATION.md`, and `docs/nvila-hd-video-readme.md` were not modified.

## Support Matrix

| Item | Status | Current behavior / update |
|---|---|---|
| `num_video_frames` | implemented | Added `--num-video-frames`; it updates NVILA processor kwargs, effective metadata, and stream window size when provided from CLI. Presets set 16 / 128 / 64. |
| `num_video_frames_thumbnail` | implemented | Added `--num-video-frames-thumbnail`; passed to official NVILA processor and reported in metadata/metrics. |
| `max_tiles_video` | implemented | Added `--max-tiles-video`; passed to official NVILA processor and reported. PoC reports expected tile budget; exact internal tile count remains processor-owned. |
| `gazing_ratio_tile` | implemented | Added `--gazing-ratio-tile` with comma-list or JSON-list parsing. Passed to NVILA processor. PoC AutoGaze-only uses it as the per-frame gazing ratio when running/stubbing the standalone AutoGaze stage. |
| `task_loss_requirement_tile` | implemented | Added `--task-loss-requirement-tile`; passed to NVILA processor and standalone AutoGaze stage metadata. |
| `gazing_ratio_thumbnail` | implemented | Added `--gazing-ratio-thumbnail`; passed to NVILA processor and reported. |
| `task_loss_requirement_thumbnail` | implemented | Added `--task-loss-requirement-thumbnail`; supports explicit `none`; passed to processor and reported. |
| `max_batch_size_autogaze` | implemented | Added `--max-batch-size-autogaze`; passed to official processor and reported. |
| `max_batch_size_siglip` | implemented | Added `--max-batch-size-siglip`; passed to `AutoModel.from_pretrained(..., max_batch_size_siglip=...)` and reported. |
| Streaming video reading | implemented | Streaming is supported but opt-in. The new NVILA-HD presets default to `video_input.read_mode: full` per branch preference, while full-video loading remains blocked by default for real files unless `memory.fail_on_full_video_load=false` is set. |
| Tile-based high-resolution handling | partially implemented | Full NVILA-HD tile logic is owned by the official NVILA processor. The new presets pass the README tile budget to the processor. The generic PoC `resize` / `resize_then_chop` path is not claimed as a replacement for NVILA-HD tiling. |
| Thumbnail path | partially implemented | `num_video_frames_thumbnail`, thumbnail gazing ratio, and thumbnail loss requirement are passed to the official processor. Standalone `infer_autogaze.py` does not reproduce the full NVILA thumbnail path; it records the settings and runs the bounded AutoGaze stage. |
| AutoGaze ON/OFF | implemented | A0/A1 remain OFF; A2/A3 and NVILA-HD presets remain ON. OFF mode sets processor gazing controls to `None` through the existing adapter path when no explicit processor gazing kwargs are supplied. |
| Full pipeline inference | implemented | `infer_full.py` routes NVILA through the official processor path, preserves query text, saves generation status, and records skipped/failure reasons. |
| AutoGaze-only inference | partially implemented | `infer_autogaze.py` accepts and reports NVILA-HD controls, supports streaming input, saves token counts/patch indices/scales, and can visualize. It does not implement the full NVILA-HD processor tile + thumbnail pipeline without NVILA. |
| Token / latency / memory reporting | implemented | Metrics now include NVILA-HD fields, token counts, selected patches per frame/scale, latencies, VRAM if CUDA, and skipped/failure reason. |
| OOM-safe long/high-resolution behavior | implemented | Streaming remains available as an explicit OOM-safe mode. Added processed-tensor memory guards and NVILA-HD memory-safe preset. Visualization is disabled by default for performance presets. |

## Effective Settings Precedence

The scripts use this precedence for NVILA-HD settings:

1. CLI values such as `--num-video-frames`
2. config values under `mllm.processor_from_pretrained_kwargs` / `mllm.from_pretrained_kwargs`
3. config values under `nvila_hd`
4. model defaults

CLI `--num-video-frames` also updates `frame_selection.num_frames` and `streaming.window_size` so the PoC streaming window does not silently stay at the generic 16-frame setting when an NVILA-HD frame count is requested.

## Presets Added

| Config | Purpose | Key settings |
|---|---|---|
| `configs/poc_inference/nvila_hd_smoke.yaml` | minimal path validation | 16 frames, 8 thumbnail frames, 8 tiles, AutoGaze batch 2, SigLIP batch 4, 8 new tokens |
| `configs/poc_inference/nvila_hd_default.yaml` | README-like settings | 128 frames, 64 thumbnail frames, 48 tiles, AutoGaze batch 16, SigLIP batch 32, 32 new tokens |
| `configs/poc_inference/nvila_hd_memory_safe.yaml` | reduced long/high-res setting | 64 frames, 32 thumbnail frames, 24 tiles, AutoGaze batch 4, SigLIP batch 8, 16 new tokens |

## Known Gaps

- `infer_full.py` can pass the README NVILA-HD controls to the official processor, but exact internal tile and thumbnail counts are controlled by NVILA remote code and are not introspected without real processor execution.
- `infer_autogaze.py` is a standalone AutoGaze visualizer/selector. It does not implement the full NVILA-HD processor's two-path tile + thumbnail inference by itself. It records thumbnail settings and uses tile controls for the standalone AutoGaze stage.
- Streaming `infer_full.py` with `mllm.video_input_source: source_video` delegates video decoding/tiling to the NVILA processor for generation. The PoC streaming window is still used for AutoGaze metadata/visualization and memory guards.
- Encoder-side acceleration is not claimed by the generic PoC metrics. NVILA remote code may reduce processor/encoder work internally when official AutoGaze settings are active, but this branch reports that as processor-owned behavior rather than asserting a measured encoder-side speedup.

## Regression Status

- A0/A1/A2/A3 configs were not changed in this update.
- A1 and A2 still share the same NVILA processor frame/batch settings except AutoGaze ON/OFF behavior.
- Original AutoGaze source files and original docs were not modified.
