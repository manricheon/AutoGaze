"""Shared rendering helpers for Borissal qualitative outputs.

Used by both scripts/eval_borissal_qualitative.py (single overlay) and
scripts/borissal_dump_outputs.py (full stage-by-stage dump). Kept dependency-
light (matplotlib + numpy/torch only) and DDP/Hydra/wandb-free so it runs
standalone on Mac/CPU/MPS.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def tubelet_title(t: int, tubelet_size: int) -> str:
    start = t * tubelet_size
    end = start + tubelet_size - 1
    frame_range = f"frame {start}" if tubelet_size == 1 else f"frames {start}-{end}"
    return f"t={t} ({frame_range})"


def render_frame_strip(video_disp: torch.Tensor, out_path: str, title: str = "Input frames"):
    """video_disp: (T, C, H, W) float in [0,1]. One row, all T frames, no gaps."""
    T = video_disp.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(2 * T, 2.2), squeeze=False)
    for i in range(T):
        frame = video_disp[i].permute(1, 2, 0).numpy()
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"frame {i}", fontsize=9)
        axes[0, i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_overlay(video_disp: torch.Tensor, keep_mask_grid: torch.Tensor, tubelet_size: int, out_path: str):
    """video_disp: (T, C, H, W) float in [0,1]. keep_mask_grid: (T_grid, H_grid, W_grid) bool.

    Row 0: representative frame per tubelet. Row 1: same frame with non-kept
    patches dimmed and kept patches outlined in red.
    """
    T_grid, H_grid, W_grid = keep_mask_grid.shape
    H, W = video_disp.shape[-2], video_disp.shape[-1]
    patch_h, patch_w = H // H_grid, W // W_grid

    fig, axes = plt.subplots(2, T_grid, figsize=(3 * T_grid, 6), squeeze=False)
    for t in range(T_grid):
        rep_frame_idx = min(t * tubelet_size, video_disp.shape[0] - 1)
        frame = video_disp[rep_frame_idx].permute(1, 2, 0).numpy()

        axes[0, t].imshow(frame)
        axes[0, t].set_title(f"Original {tubelet_title(t, tubelet_size)}", fontsize=9)
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
        axes[1, t].set_title(f"Selected {tubelet_title(t, tubelet_size)}", fontsize=9)
        axes[1, t].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_heatmap_grid(heatmap: torch.Tensor, tubelet_size: int, out_path: str, suptitle: str, cmap: str = "magma"):
    """heatmap: (T_grid, H_grid, W_grid) float, already normalized to a fixed range for display."""
    T_grid = heatmap.shape[0]
    hm = heatmap.cpu().numpy()
    vmin, vmax = float(hm.min()), float(hm.max())

    fig, axes = plt.subplots(1, T_grid, figsize=(2.2 * T_grid, 2.6), squeeze=False)
    im = None
    for t in range(T_grid):
        im = axes[0, t].imshow(hm[t], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, t].set_title(tubelet_title(t, tubelet_size), fontsize=9)
        axes[0, t].axis("off")
    fig.suptitle(suptitle)
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_allocation_bar(per_frame_keep: torch.Tensor, out_path: str, title: str = "Per-tubelet kept patch count"):
    """per_frame_keep: (T_grid,) long/int."""
    counts = per_frame_keep.cpu().numpy() if isinstance(per_frame_keep, torch.Tensor) else np.asarray(per_frame_keep)
    T_grid = len(counts)

    fig, ax = plt.subplots(figsize=(max(4, 0.8 * T_grid), 3))
    ax.bar(np.arange(T_grid), counts, color="steelblue")
    ax.set_xlabel("tubelet index (t)")
    ax.set_ylabel("kept patch count")
    ax.set_xticks(np.arange(T_grid))
    ax.set_title(title)
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
