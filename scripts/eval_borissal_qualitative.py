#!/usr/bin/env python
"""Mac-friendly standalone qualitative evaluation for the Borissal saliency selector.

No DDP / trainer / Hydra / transformers video processor required -- decodes a
video with PyAV, runs Borissal feed-forward, and renders an overlay of the
selected (kept) patches on top of the original frames.

Example:
    uv run python scripts/eval_borissal_qualitative.py \
        --video assets/example_input.mp4 --gazing-ratio 0.5 --motion-weight 0.5 \
        --out /tmp/sal_r50_m50.png
"""

import argparse

from autogaze.models.borissal import Borissal, BorissalConfig, resolve_device
from autogaze.models.borissal.video_io import load_video, unnormalize
from autogaze.models.borissal.viz import render_overlay


def _motion_weight_type(s):
    return s if s == "auto" else float(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="path to an input video file")
    p.add_argument("--out", required=True, help="path to save the overlay PNG")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--gazing-ratio", type=float, default=0.5)
    p.add_argument("--motion-weight", type=_motion_weight_type, default=0.5, help="float in [0,1], or 'auto'")
    p.add_argument("--per-frame-allocation", choices=["uniform", "proportional"], default="uniform")
    p.add_argument("--spatial-op", choices=["grad", "sobel"], default="grad")
    p.add_argument("--pooling", choices=["avg", "max"], default="avg")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    video = load_video(args.video, num_frames=args.num_frames, size=args.scale).to(device)

    config = BorissalConfig(
        scale=args.scale,
        patch_size=args.patch,
        tubelet_size=args.tubelet_size,
        gazing_ratio=args.gazing_ratio,
        motion_weight=args.motion_weight,
        per_frame_allocation=args.per_frame_allocation,
        spatial_op=args.spatial_op,
        pooling=args.pooling,
    )
    model = Borissal(config).to(device)
    selection, intermediates = model.select_with_intermediates(video)

    grid_thw = selection.grid_thw[0].tolist()
    T_grid, H_grid, W_grid = grid_thw
    keep_mask_grid = selection.keep_mask[0].reshape(T_grid, H_grid, W_grid).cpu()

    print(f"grid_thw = {grid_thw}")
    print(f"motion_weight = {args.motion_weight} (resolved = {intermediates['motion_weight_used'][0].item():.3f})")
    print(f"num_keep = {selection.num_keep[0].item()} / {selection.scores.shape[1]}")
    print(f"per_frame_keep = {selection.per_frame_keep[0].tolist()}")

    video_disp = unnormalize(video[0]).cpu()  # (T, C, H, W) in [0,1]
    render_overlay(video_disp, keep_mask_grid, args.tubelet_size, args.out)
    print(f"saved overlay to {args.out}")


if __name__ == "__main__":
    main()
