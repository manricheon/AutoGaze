#!/usr/bin/env python
"""Dump every stage of the Borissal saliency selector to outputs/borissal/<run_name>/.

Mac-friendly, standalone (no DDP/Hydra/trainer). Decodes a video, runs the
selector, and saves one file per stage so the pipeline can be inspected
end-to-end:

    00_input_frames.png  -- all decoded input frames (thumbnail strip)
    01_motion.png         -- per-tubelet normalized motion heatmap
    02_spatial.png        -- per-tubelet normalized spatial/edge heatmap
    03_score.png          -- per-tubelet combined score heatmap (pre-top-k)
    04_overlay.png         -- final selected-patch overlay
    05_allocation.png      -- per-tubelet kept-patch-count bar chart
    summary.json           -- config + grid_thw/num_keep/per_frame_keep

`outputs/` is gitignored -- nothing this script writes is meant to be committed.

Example:
    uv run python scripts/borissal_dump_outputs.py \
        --video assets/example_input.mp4 --gazing-ratio 0.5 --motion-weight 0.5
"""

import argparse
import json
from pathlib import Path

from autogaze.models.borissal import Borissal, BorissalConfig, resolve_device
from autogaze.models.borissal.video_io import load_video, unnormalize
from autogaze.models.borissal.viz import (
    render_allocation_bar,
    render_frame_strip,
    render_heatmap_grid,
    render_overlay,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _motion_weight_type(s):
    return s if s == "auto" else float(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="path to an input video file")
    p.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "borissal"))
    p.add_argument("--run-name", default=None, help="defaults to a config-derived name")
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


def default_run_name(args) -> str:
    r = int(round(args.gazing_ratio * 100))
    m = "auto" if args.motion_weight == "auto" else int(round(args.motion_weight * 100))
    return f"r{r}_m{m}_{args.spatial_op}_{args.per_frame_allocation}"


def main():
    args = parse_args()
    device = resolve_device(args.device)
    run_name = args.run_name or default_run_name(args)
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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
    per_frame_keep = selection.per_frame_keep[0].tolist()
    num_keep = selection.num_keep[0].item()

    video_disp = unnormalize(video[0]).cpu()  # (T, C, H, W) in [0,1]
    keep_mask_grid = selection.keep_mask[0].reshape(T_grid, H_grid, W_grid).cpu()

    render_frame_strip(video_disp, str(run_dir / "00_input_frames.png"), title=f"Input frames ({args.num_frames})")
    render_heatmap_grid(
        intermediates["motion_norm"][0], args.tubelet_size, str(run_dir / "01_motion.png"),
        suptitle="Motion (normalized)",
    )
    render_heatmap_grid(
        intermediates["spatial_norm"][0], args.tubelet_size, str(run_dir / "02_spatial.png"),
        suptitle="Spatial / edge (normalized)",
    )
    motion_weight_used = intermediates["motion_weight_used"][0].item()
    render_heatmap_grid(
        intermediates["score"][0], args.tubelet_size, str(run_dir / "03_score.png"),
        suptitle=f"Combined score (motion_weight={args.motion_weight}, resolved={motion_weight_used:.3f})",
    )
    render_overlay(video_disp, keep_mask_grid, args.tubelet_size, str(run_dir / "04_overlay.png"))
    render_allocation_bar(
        per_frame_keep, str(run_dir / "05_allocation.png"),
        title=f"per_frame_allocation={args.per_frame_allocation}",
    )

    summary = {
        "video": str(Path(args.video).resolve()),
        "device": str(device),
        "config": {
            "scale": args.scale,
            "patch_size": args.patch,
            "tubelet_size": args.tubelet_size,
            "gazing_ratio": args.gazing_ratio,
            "motion_weight": args.motion_weight,
            "per_frame_allocation": args.per_frame_allocation,
            "spatial_op": args.spatial_op,
            "pooling": args.pooling,
            "num_frames": args.num_frames,
        },
        "motion_weight_used": motion_weight_used,
        "grid_thw": grid_thw,
        "num_keep": num_keep,
        "L": selection.scores.shape[1],
        "per_frame_keep": per_frame_keep,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"grid_thw = {grid_thw}")
    print(f"motion_weight = {args.motion_weight} (resolved = {motion_weight_used:.3f})")
    print(f"num_keep = {num_keep} / {selection.scores.shape[1]}")
    print(f"per_frame_keep = {per_frame_keep}")
    print(f"wrote stage outputs to {run_dir}")


if __name__ == "__main__":
    main()
