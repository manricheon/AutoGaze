# AutoGaze Scaling Guide

This guide summarizes the scaling behavior from the original `QUICK_START.md` and how the current PoC represents it.

## What the Original Repo Supports

The original AutoGaze quick start has three relevant paths.

| Path | Original behavior | Current PoC status |
|---|---|---|
| Default resize | `16` frames at `224x224`, patch size `16` | Runnable in A1/A2 smoke and benchmark presets |
| High-resolution target scale | For patch14 SigLIP, use `392x392`, `target_scales=[56,112,196,392]`, `target_patch_size=14`, and SigLIP `scales="56+112+196+392"` | Runnable as guarded smoke/benchmark config wiring; still not paper reproduction |
| Spatio-temporal chunking | Chunk long/high-resolution video into `16`-frame spatial tiles, for example `224x224` tiles | Utility-supported in `autogaze_ext.scaling`; full benchmark aggregation remains guarded/stubbed |

Important resolution rule: the quick start does not use raw `384x384` for patch14. It rounds the working image size to `392x392` because `384` is not divisible by `14`.

## Config Presets

Scaling examples live under:

```text
configs/scaling/resize_224.yaml
configs/scaling/resize_392_patch14.yaml
configs/scaling/spatio_temporal_224.yaml
configs/scaling/spatio_temporal_392_patch14.yaml
```

Benchmark presets now include explicit scaling fields:

```yaml
scaling_mode: resize
temporal_chunk_size: 16
spatial_tile_size: 224  # or 392 for the patch14 high-resolution path
```

The medium canonical presets use the documented high-resolution target-scale policy:

```yaml
resolution: 392
scale_resolution: quick_start_target_scales
target_scales: [56, 112, 196, 392]
target_patch_size: 14
```

## Programmatic Use

Resize/default policy:

```python
from autogaze_ext.scaling import resolve_autogaze_scaling_policy, resize_video

policy = resolve_autogaze_scaling_policy(mode="resize", resolution=224, patch_size=16)
video_224 = resize_video(video, policy.effective_resolution)
```

High-resolution patch14 policy:

```python
policy = resolve_autogaze_scaling_policy(mode="resize", resolution=384, patch_size=14)

assert policy.effective_resolution == 392
assert policy.autogaze_call_kwargs == {
    "target_scales": [56, 112, 196, 392],
    "target_patch_size": 14,
}
assert policy.siglip_scales == "56+112+196+392"
```

Spatio-temporal chunking:

```python
from autogaze_ext.scaling import chunk_video_spatio_temporal

chunks = chunk_video_spatio_temporal(
    video,  # [B, T, C, H, W]
    temporal_chunk_size=16,
    spatial_tile_size=224,
)

assert chunks.chunks.ndim == 5
print(chunks.metadata["chunk_records"][0])
```

For a video shaped `[1, 32, 3, 448, 448]`, the 224-tile chunker returns `[8, 16, 3, 224, 224]`.

## Smoke Commands

Default 224 path:

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode full_pipeline \
  --video dummy \
  --num-frames 16 \
  --resolution 224 \
  --device cpu \
  --output-dir outputs/scaling_smoke/a2_224
```

High-resolution target-scale path:

```bash
python scripts/run_canonical_smoke_inference.py \
  --experiment A2_real \
  --mode autogaze_only \
  --video dummy \
  --num-frames 16 \
  --resolution 392 \
  --scale-resolution quick_start_target_scales \
  --device cpu \
  --output-dir outputs/scaling_smoke/a2_392
```

Dry-run benchmark validation:

```bash
python scripts/run_tiny_real_benchmark.py \
  --config-name canonical_a2_medium \
  --mode full_pipeline \
  --dry-run \
  --output-dir outputs/scaling_smoke/dry_run_a2_medium
```

## Current Boundaries

- `224` is valid for the default AutoGaze path.
- `384` should be represented as the documented `392` policy for patch14 SigLIP.
- `448` is not a current canonical policy unless a new, explicitly validated target-scale config is added.
- Spatio-temporal chunking is now available as a preprocessing utility, but the generic benchmark runner still does not merge per-chunk outputs back into an official NVILA-HD result.
- Encoder-side acceleration can only be claimed when AutoGaze-selected chunks/tokens are actually passed into modified SigLIP so the encoder computes fewer visual tokens.
