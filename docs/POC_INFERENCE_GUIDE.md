# PoC Inference Guide

This guide is the user-facing entry point for the PoC inference branch. `docs/INFERENCE_GUIDE.md` remains as the maintained index; this file focuses on runnable PoC commands.

## NVILA-HD-Video README-Based Inference

Use these configs when you want NVILA-HD-Video settings that match the README-style processor interface:

| Config | Purpose |
|---|---|
| `configs/poc_inference/nvila_hd_smoke.yaml` | minimal path validation with small frame/tile/batch budgets |
| `configs/poc_inference/nvila_hd_default.yaml` | README-like default: 128 frames, 64 thumbnail frames, 48 tiles |
| `configs/poc_inference/nvila_hd_memory_safe.yaml` | lower-memory preset: 64 frames, 32 thumbnail frames, 24 tiles |

Meaning of key settings:

- `num_video_frames`: sampled frames for the main tiled video path.
- `num_video_frames_thumbnail`: sampled frames for thumbnail/global context.
- `max_tiles_video`: maximum spatial tile budget used by the official NVILA-HD processor.
- `gazing_ratio_tile` / `task_loss_requirement_tile`: AutoGaze controls for the tile path.
- `gazing_ratio_thumbnail` / `task_loss_requirement_thumbnail`: AutoGaze controls for thumbnails.
- `max_batch_size_autogaze`: AutoGaze processor micro-batch size.
- `max_batch_size_siglip`: SigLIP / vision encoder micro-batch size in `AutoModel.from_pretrained`.

AutoGaze-only smoke:

```bash
python scripts/infer_autogaze.py \
  --config configs/poc_inference/nvila_hd_default.yaml \
  --video-path /path/to/video.mp4 \
  --video-read-mode streaming \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --gazing-ratio-tile "0.2,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06" \
  --task-loss-requirement-tile 0.6 \
  --max-batch-size-autogaze 16 \
  --device cuda \
  --dtype float16 \
  --output-dir outputs/nvila_hd_autogaze_only \
  --overlay-style mask \
  --multi-scale-overlay \
  --save-overlay-video
```

Full pipeline default:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/nvila_hd_default.yaml \
  --video-path /path/to/video.mp4 \
  --query-text "Describe the important events in this video." \
  --video-read-mode streaming \
  --num-video-frames 128 \
  --num-video-frames-thumbnail 64 \
  --max-tiles-video 48 \
  --gazing-ratio-tile "0.2,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06" \
  --task-loss-requirement-tile 0.6 \
  --max-batch-size-autogaze 16 \
  --max-batch-size-siglip 32 \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --max-new-tokens 32 \
  --output-dir outputs/nvila_hd_full
```

Memory-safe full pipeline:

```bash
python scripts/infer_full.py \
  --config configs/poc_inference/nvila_hd_memory_safe.yaml \
  --video-path /path/to/long_or_highres_video.mp4 \
  --query-text "Describe the important events in this video." \
  --video-read-mode streaming \
  --num-video-frames 64 \
  --num-video-frames-thumbnail 32 \
  --max-tiles-video 24 \
  --max-batch-size-autogaze 4 \
  --max-batch-size-siglip 8 \
  --allow-real-model-loading \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --max-new-tokens 16 \
  --output-dir outputs/nvila_hd_memory_safe
```

OOM mitigation:

- Default presets keep `video_input.read_mode: full`; for long or high-resolution videos, enable `--video-read-mode streaming`.
- Use `nvila_hd_memory_safe.yaml` first for long or high-resolution videos.
- Reduce `--num-video-frames`, `--num-video-frames-thumbnail`, `--max-tiles-video`, `--max-batch-size-autogaze`, and `--max-batch-size-siglip`.
- Leave visualization disabled for timing/performance runs. Overlay video export adds CPU and disk overhead.
- Use `--max-processed-frames-per-window` and `--max-processed-pixels-per-window` when testing `resize_then_chop`.
- Do not disable `memory.fail_on_full_video_load` for large real videos unless you intentionally want a full-load run.

A1 vs A2 comparison guidance:

- Use identical NVILA-HD processor settings for both runs.
- Only change AutoGaze ON/OFF.
- Keep query text, video path, frame counts, thumbnail counts, tile budget, and batch sizes fixed.

Limitations:

- The full NVILA-HD tile and thumbnail internals are owned by the official NVILA processor. `infer_full.py` passes the README controls and reports them, but does not reimplement the remote processor.
- `infer_autogaze.py` records thumbnail settings and uses tile gazing controls for the standalone AutoGaze stage; it is not a full NVILA thumbnail-path reproduction.
