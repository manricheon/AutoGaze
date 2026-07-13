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

import matplotlib.pyplot as plt
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.video_io import load_video, unnormalize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="path to an input video file")
    p.add_argument("--out", required=True, help="path to save the overlay PNG")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--gazing-ratio", type=float, default=0.5)
    p.add_argument("--motion-weight", type=float, default=0.5)
    p.add_argument("--per-frame-allocation", choices=["uniform", "proportional"], default="uniform")
    p.add_argument("--spatial-op", choices=["grad", "sobel"], default="grad")
    p.add_argument("--pooling", choices=["avg", "max"], default="avg")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_overlay(video_disp, keep_mask_grid, tubelet_size, out_path):
    """video_disp: (T, C, H, W) float in [0,1]. keep_mask_grid: (T_grid, H_grid, W_grid) bool."""
    T_grid, H_grid, W_grid = keep_mask_grid.shape
    H, W = video_disp.shape[-2], video_disp.shape[-1]
    patch_h, patch_w = H // H_grid, W // W_grid

    fig, axes = plt.subplots(2, T_grid, figsize=(3 * T_grid, 6), squeeze=False)
    for t in range(T_grid):
        rep_frame_idx = min(t * tubelet_size, video_disp.shape[0] - 1)
        frame = video_disp[rep_frame_idx].permute(1, 2, 0).numpy()  # H, W, C

        axes[0, t].imshow(frame)
        axes[0, t].set_title(f"Original t={t}")
        axes[0, t].axis("off")

        mask = keep_mask_grid[t].float().numpy()
        mask_full = mask.repeat(patch_h, axis=0).repeat(patch_w, axis=1)
        dimmed = frame * (0.8 * mask_full[..., None] + 0.2)
        axes[1, t].imshow(dimmed.clip(0, 1))
        for i in range(H_grid):
            for j in range(W_grid):
                if mask[i, j] > 0.5:
                    rect = plt.Rectangle(
                        (j * patch_w - 0.5, i * patch_h - 0.5), patch_w, patch_h,
                        linewidth=1, edgecolor="red", facecolor="none",
                    )
                    axes[1, t].add_patch(rect)
        axes[1, t].set_title(f"Selected t={t}")
        axes[1, t].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
    selection = model.select(video)

    grid_thw = selection.grid_thw[0].tolist()
    T_grid, H_grid, W_grid = grid_thw
    keep_mask_grid = selection.keep_mask[0].reshape(T_grid, H_grid, W_grid).cpu()

    print(f"grid_thw = {grid_thw}")
    print(f"num_keep = {selection.num_keep[0].item()} / {selection.scores.shape[1]}")
    print(f"per_frame_keep = {selection.per_frame_keep[0].tolist()}")

    video_disp = unnormalize(video[0]).cpu()  # (T, C, H, W) in [0,1]
    render_overlay(video_disp, keep_mask_grid, args.tubelet_size, args.out)
    print(f"saved overlay to {args.out}")


if __name__ == "__main__":
    main()
