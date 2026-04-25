# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoGaze inference script.

Runs AutoGaze on a single video or a directory of videos and saves results.

Output formats (--output-format accepts comma-separated list or 'all'):
  json   — gazing_labels.json compatible with NTP pre-training data format.
  viz    — Single PNG grid: all frames × all scales in one image.
  frames — Per-frame PNGs: one image per frame (original + all-scale overlays).
  video  — MP4: animated side-by-side (original | gaze overlay) for easy viewing.
  npy    — Compressed numpy archive of raw gazing masks.

Usage examples:
  # Single video, all formats
  python -m autogaze.infer assets/example_input.mp4 --output-dir results/

  # Video output only
  python -m autogaze.infer assets/example_input.mp4 --output-format video

  # Per-frame images + video
  python -m autogaze.infer assets/example_input.mp4 --output-format frames,video

  # Directory → JSON labels only (NTP training data generation)
  python -m autogaze.infer /data/my_videos/ --output-format json

  # Use local model
  python -m autogaze.infer video.mp4 --model-path weights/AutoGaze
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F
import imageio.v3 as iio

from autogaze.datasets.video_utils import (
    read_video_pyav,
    sample_frame_indices,
    process_video_frames,
    transform_video_for_pytorch,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.utils import UnNormalize, get_device


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

VALID_FORMATS = {"json", "viz", "frames", "video", "npy"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_video_paths(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    videos = sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No video files found under {input_path}")
    return videos


def load_video(video_path: Path, num_frames: int) -> np.ndarray:
    container = av.open(str(video_path))
    total = container.streams.video[0].frames
    indices = sample_frame_indices(
        clip_len=num_frames, frame_sample_rate=1,
        seg_len=total, random_sample_frame=False,
    )
    raw = read_video_pyav(container, indices)
    container.close()
    return process_video_frames(raw, num_frames)


def _unnorm_video(raw_video: np.ndarray, transform, normalize_mean,
                  normalize_std, normalize_rescale) -> np.ndarray:
    """Return float32 (T, C, H, W) in [0,1] after un-normalising."""
    video_tensor = transform_video_for_pytorch(raw_video, transform)
    unnorm = UnNormalize(normalize_mean, normalize_std, normalize_rescale)
    return unnorm(video_tensor).cpu().float().numpy()


def _overlay_gaze_on_frame(frame_chw: np.ndarray, mask_hw: np.ndarray,
                            patch_grid: int, scale: int,
                            dim_factor: float = 0.25) -> np.ndarray:
    """
    Render a gaze-overlay image for one frame at one scale.
    Returns uint8 HWC RGB (scale × scale).
    """
    frame_t = torch.from_numpy(frame_chw).unsqueeze(0)
    frame_s = F.interpolate(frame_t, size=(scale, scale),
                            mode="bicubic", align_corners=False)
    frame_s = frame_s.squeeze().clamp(0, 1).numpy()      # C H W

    pm_up = F.interpolate(
        torch.from_numpy(mask_hw).unsqueeze(0).unsqueeze(0).float(),
        size=(scale, scale), mode="nearest"
    ).squeeze().numpy()

    display = frame_s * (dim_factor + (1 - dim_factor) * pm_up[None])  # C H W
    return (np.clip(display.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def _draw_patch_borders(ax, mask_hw: np.ndarray, patch_grid: int, scale: int,
                        color: str = "red", lw: float = 0.8):
    patch_px = scale // patch_grid
    for pi in range(patch_grid):
        for pj in range(patch_grid):
            if mask_hw[pi, pj] > 0.5:
                ax.add_patch(mpatches.Rectangle(
                    (pj * patch_px - 0.5, pi * patch_px - 0.5),
                    patch_px, patch_px,
                    linewidth=lw, edgecolor=color, facecolor="none",
                ))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_json(result: dict, video_path: Path, all_results: dict):
    parts = video_path.parts
    key = str(Path(*parts[-3:])) if len(parts) >= 3 else str(video_path)

    num_tokens_per_frame = result["num_vision_tokens_each_frame"]
    gazing_pos_flat = result["gazing_pos"][0]
    if_padded_flat  = result["if_padded_gazing"][0]
    num_each        = result["num_gazing_each_frame"]

    gazing_pos_per_frame, task_losses_per_frame = [], []
    offset = 0
    for t, count in enumerate(num_each.tolist()):
        frame_pos = gazing_pos_flat[offset: offset + count]
        frame_pad = if_padded_flat[offset: offset + count]
        within = (frame_pos[~frame_pad] - t * num_tokens_per_frame).tolist()
        gazing_pos_per_frame.append([int(x) for x in within])
        task_losses_per_frame.append([0.0] * len(within))
        offset += count

    all_results[key] = {
        "gazing_pos": gazing_pos_per_frame,
        "task_losses": task_losses_per_frame,
    }


def save_viz(result: dict, raw_video: np.ndarray, out_dir: Path, video_path: Path,
             transform, normalize_mean, normalize_std, normalize_rescale) -> Path:
    """Single PNG grid: rows = [original, scale32, scale64, scale112, scale224], cols = frames."""
    scales = result["scales"]
    gazing_mask = result["gazing_mask"]
    T = gazing_mask[0].shape[1]
    num_scales = len(scales)

    video_np = _unnorm_video(raw_video, transform, normalize_mean, normalize_std, normalize_rescale)

    fig, axes = plt.subplots(num_scales + 1, T,
                             figsize=(max(2 * T, 4), 2.5 * (num_scales + 1)),
                             squeeze=False)

    for t in range(T):
        axes[0, t].imshow(np.clip(video_np[t].transpose(1, 2, 0), 0, 1))
        axes[0, t].set_title(f"Frame {t+1}", fontsize=7)
        axes[0, t].axis("off")

    for si, scale in enumerate(scales):
        mask_scale = gazing_mask[si][0]           # (T, N_scale)
        patch_grid = int(mask_scale.shape[-1] ** 0.5)
        for t in range(T):
            mask_hw = mask_scale[t].reshape(patch_grid, patch_grid).cpu().float().numpy()
            img = _overlay_gaze_on_frame(video_np[t], mask_hw, patch_grid, scale)
            axes[si + 1, t].imshow(img)
            _draw_patch_borders(axes[si + 1, t], mask_hw, patch_grid, scale)
            axes[si + 1, t].set_title(f"Scale {scale} F{t+1}", fontsize=7)
            axes[si + 1, t].axis("off")

    plt.tight_layout(pad=0.3)
    out_path = out_dir / f"{video_path.stem}_gaze.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_frames(result: dict, raw_video: np.ndarray, out_dir: Path, video_path: Path,
                transform, normalize_mean, normalize_std, normalize_rescale) -> list[Path]:
    """One PNG per frame: columns = [original, scale32, scale64, scale112, scale224]."""
    scales = result["scales"]
    gazing_mask = result["gazing_mask"]
    T = gazing_mask[0].shape[1]
    num_scales = len(scales)

    video_np = _unnorm_video(raw_video, transform, normalize_mean, normalize_std, normalize_rescale)
    num_each = result["num_gazing_each_frame"]

    frames_dir = out_dir / f"{video_path.stem}_frames"
    frames_dir.mkdir(exist_ok=True)

    out_paths = []
    for t in range(T):
        n_gazed = int((~result["if_padded_gazing"][0]).sum().item())  # total across frames shown in title
        n_this = int(num_each[t].item())

        cols = 1 + num_scales
        fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 3.5), squeeze=False)

        # Original frame (from raw_video at full resolution)
        orig_hw = raw_video[t]                          # H W 3  uint8
        axes[0, 0].imshow(orig_hw)
        axes[0, 0].set_title(f"Frame {t+1}  ({n_this} gazed)", fontsize=9)
        axes[0, 0].axis("off")

        for si, scale in enumerate(scales):
            mask_scale = gazing_mask[si][0]
            patch_grid = int(mask_scale.shape[-1] ** 0.5)
            mask_hw = mask_scale[t].reshape(patch_grid, patch_grid).cpu().float().numpy()
            img = _overlay_gaze_on_frame(video_np[t], mask_hw, patch_grid, scale)
            axes[0, si + 1].imshow(img)
            _draw_patch_borders(axes[0, si + 1], mask_hw, patch_grid, scale)
            gazed_count = int(mask_hw.sum())
            axes[0, si + 1].set_title(f"Scale {scale}  ({gazed_count} patches)", fontsize=9)
            axes[0, si + 1].axis("off")

        plt.suptitle(f"{video_path.name} — Frame {t+1}/{T}", fontsize=10, y=1.01)
        plt.tight_layout(pad=0.4)
        out_path = frames_dir / f"frame_{t+1:03d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def save_video(result: dict, raw_video: np.ndarray, out_dir: Path, video_path: Path,
               transform, normalize_mean, normalize_std, normalize_rescale,
               fps: float = 4.0) -> Path:
    """
    MP4 video.
    Each frame is a 3-panel layout:
      [Original (full-res) | Scale-224 overlay | Multi-scale patch heatmap]

    fps is kept low (default 4) so individual frames are easy to inspect.
    """
    scales = result["scales"]
    gazing_mask = result["gazing_mask"]
    T = gazing_mask[0].shape[1]
    num_each = result["num_gazing_each_frame"]

    video_np = _unnorm_video(raw_video, transform, normalize_mean, normalize_std, normalize_rescale)

    # Use the largest scale for the main overlay panel
    largest_si = len(scales) - 1
    largest_scale = scales[largest_si]
    largest_mask = gazing_mask[largest_si][0]          # (T, N_largest)
    largest_grid = int(largest_mask.shape[-1] ** 0.5)

    # Build combined heatmap across all scales (upsampled to largest_scale)
    all_masks_upsampled = []
    for si, scale in enumerate(scales):
        mask_s = gazing_mask[si][0]                    # (T, N_s)
        pg = int(mask_s.shape[-1] ** 0.5)
        for t in range(T):
            pass
        all_masks_upsampled.append(mask_s)             # kept per scale, upsampled per frame

    render_size = 224   # each panel rendered at this pixel resolution
    out_frames = []

    for t in range(T):
        n_gazed_this = int(num_each[t].item())

        # ---- Panel 1: original (resized to render_size) ----
        orig = raw_video[t]                             # H W 3  uint8
        orig_t = torch.from_numpy(orig).permute(2, 0, 1).float() / 255.0
        orig_resized = F.interpolate(orig_t.unsqueeze(0), size=(render_size, render_size),
                                     mode="bicubic", align_corners=False)
        orig_resized = (orig_resized.squeeze().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # ---- Panel 2: largest-scale overlay ----
        mask_hw = largest_mask[t].reshape(largest_grid, largest_grid).cpu().float().numpy()
        overlay_img = _overlay_gaze_on_frame(video_np[t], mask_hw, largest_grid, render_size)

        # Draw red borders on overlay (by rendering with matplotlib then extracting canvas)
        fig_p, ax_p = plt.subplots(1, 1, figsize=(render_size / 100, render_size / 100))
        ax_p.imshow(overlay_img)
        patch_px = render_size // largest_grid
        for pi in range(largest_grid):
            for pj in range(largest_grid):
                if mask_hw[pi, pj] > 0.5:
                    ax_p.add_patch(mpatches.Rectangle(
                        (pj * patch_px - 0.5, pi * patch_px - 0.5),
                        patch_px, patch_px,
                        linewidth=1.0, edgecolor="red", facecolor="none",
                    ))
        ax_p.axis("off")
        ax_p.set_position([0, 0, 1, 1])
        fig_p.canvas.draw()
        fig_p.canvas.draw()
        buf = np.frombuffer(fig_p.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig_p.canvas.get_width_height()
        overlay_rendered = buf.reshape(h, w, 4)[:, :, :3]
        overlay_rendered = (torch.from_numpy(overlay_rendered).float() / 255.0)
        overlay_rendered = F.interpolate(overlay_rendered.permute(2, 0, 1).unsqueeze(0),
                                         size=(render_size, render_size), mode="bilinear",
                                         align_corners=False)
        overlay_rendered = (overlay_rendered.squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        plt.close(fig_p)

        # ---- Panel 3: multi-scale heatmap (sum of upsampled masks) ----
        heat = np.zeros((render_size, render_size), dtype=np.float32)
        for si in range(len(scales)):
            ms = gazing_mask[si][0][t]
            pg = int(ms.shape[-1] ** 0.5)
            m_hw = ms.reshape(pg, pg).cpu().float()
            m_up = F.interpolate(m_hw.unsqueeze(0).unsqueeze(0),
                                 size=(render_size, render_size), mode="nearest").squeeze().numpy()
            heat += m_up
        heat = heat / heat.max() if heat.max() > 0 else heat

        cmap = plt.get_cmap("hot")
        heat_rgb = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)

        # Blend heatmap with original
        orig_float = orig_resized.astype(np.float32) / 255.0
        heat_float = heat_rgb.astype(np.float32) / 255.0
        alpha = 0.55
        blend = np.clip(orig_float * (1 - alpha * heat[:, :, None]) + heat_float * alpha * heat[:, :, None], 0, 1)
        heatmap_panel = (blend * 255).astype(np.uint8)

        # ---- Assemble 3-panel row with header bar ----
        panel_w = render_size
        panel_h = render_size
        bar_h   = 28
        total_w = panel_w * 3
        total_h = panel_h + bar_h

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:bar_h, :, :] = 30          # dark header

        canvas[bar_h:, :panel_w]            = orig_resized
        canvas[bar_h:, panel_w:2*panel_w]   = overlay_rendered
        canvas[bar_h:, 2*panel_w:]          = heatmap_panel

        # Text labels via matplotlib (bake into canvas)
        fig_c, ax_c = plt.subplots(figsize=(total_w / 100, total_h / 100), dpi=100)
        ax_c.imshow(canvas)
        labels = [
            (panel_w // 2,        12, "Original"),
            (panel_w + panel_w//2, 12, f"Gaze overlay  ({n_gazed_this} patches)"),
            (2*panel_w + panel_w//2, 12, "Multi-scale heatmap"),
        ]
        for lx, ly, txt in labels:
            ax_c.text(lx, ly, txt, ha="center", va="center",
                      fontsize=8, color="white", fontweight="bold")
        ax_c.text(total_w - 6, 12, f"Frame {t+1}/{T}",
                  ha="right", va="center", fontsize=8, color="#aaaaaa")
        ax_c.axis("off")
        ax_c.set_position([0, 0, 1, 1])
        fig_c.canvas.draw()
        buf = np.frombuffer(fig_c.canvas.buffer_rgba(), dtype=np.uint8)
        cw, ch = fig_c.canvas.get_width_height()
        frame_rgb = buf.reshape(ch, cw, 4)[:, :, :3]
        plt.close(fig_c)

        out_frames.append(frame_rgb)

    out_path = out_dir / f"{video_path.stem}_gaze.mp4"
    iio.imwrite(str(out_path), out_frames, fps=fps, plugin="pyav", codec="h264")
    return out_path


def save_npy(result: dict, out_dir: Path, video_path: Path) -> Path:
    out_path = out_dir / f"{video_path.stem}_gaze.npz"
    masks = {f"scale_{s}": result["gazing_mask"][i][0].cpu().numpy()
             for i, s in enumerate(result["scales"])}
    np.savez_compressed(
        out_path,
        gazing_pos=result["gazing_pos"][0].cpu().numpy(),
        if_padded_gazing=result["if_padded_gazing"][0].cpu().numpy(),
        num_gazing_each_frame=result["num_gazing_each_frame"].cpu().numpy(),
        scales=np.array(result["scales"]),
        **masks,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_formats(fmt_str: str) -> set[str]:
    if fmt_str == "all":
        return {"json", "viz", "frames", "video", "npy"}
    fmts = {f.strip() for f in fmt_str.split(",")}
    unknown = fmts - VALID_FORMATS
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown format(s): {unknown}. Valid: {VALID_FORMATS | {'all'}}"
        )
    return fmts


def parse_args():
    p = argparse.ArgumentParser(description="AutoGaze inference")
    p.add_argument("input", help="Video file or directory of videos")
    p.add_argument("--model-path", default="nvidia/AutoGaze",
                   help="HuggingFace model ID or local path (default: nvidia/AutoGaze)")
    p.add_argument("--output-dir", default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--output-format", default="all",
                   help=(
                       "Comma-separated output formats or 'all'. "
                       f"Valid: {', '.join(sorted(VALID_FORMATS))}. "
                       "  json=gazing_labels.json (NTP training labels), "
                       "  viz=grid PNG (all frames × all scales), "
                       "  frames=per-frame PNGs (original + all-scale overlays), "
                       "  video=MP4 (original | overlay | heatmap), "
                       "  npy=compressed numpy arrays. "
                       "(default: all)"
                   ))
    p.add_argument("--gazing-ratio", type=float, default=0.75,
                   help="Max fraction of patches to gaze at (default: 0.75)")
    p.add_argument("--task-loss-requirement", type=float, default=0.7,
                   help="Reconstruction quality threshold for early stopping (default: 0.7)")
    p.add_argument("--no-task-loss-requirement", action="store_true",
                   help="Disable early stopping; use --gazing-ratio only")
    p.add_argument("--num-frames", type=int, default=16,
                   help="Number of frames to sample per video (default: 16)")
    p.add_argument("--video-fps", type=float, default=4.0,
                   help="FPS for output video (default: 4.0; low fps = easier to inspect)")
    return p.parse_args()


def main():
    args = parse_args()
    fmts = parse_formats(args.output_format)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading AutoGaze from '{args.model_path}'...")
    transform = AutoGazeImageProcessor.from_pretrained(args.model_path)
    model = AutoGaze.from_pretrained(args.model_path).to(device)
    model.eval()
    num_frames = getattr(model.config, "max_num_frames", args.num_frames)
    print(f"  Device: {device}  |  max_num_frames: {num_frames}  |  formats: {sorted(fmts)}")

    normalize_mean    = transform.image_mean
    normalize_std     = transform.image_std
    normalize_rescale = getattr(transform, "rescale_factor", 1.0 / 255.0)

    # -----------------------------------------------------------------------
    # Collect videos
    # -----------------------------------------------------------------------
    video_paths = collect_video_paths(args.input)
    print(f"Found {len(video_paths)} video(s)\n")

    task_loss_req = None if args.no_task_loss_requirement else args.task_loss_requirement
    all_json_results: dict = {}
    saved_files: list[str] = []

    # -----------------------------------------------------------------------
    # Per-video inference
    # -----------------------------------------------------------------------
    for idx, video_path in enumerate(video_paths):
        print(f"[{idx+1}/{len(video_paths)}] {video_path.name}")

        try:
            raw_video = load_video(video_path, num_frames)
        except Exception as e:
            print(f"  WARNING: could not load — {e}")
            continue

        video_input = transform_video_for_pytorch(raw_video, transform)
        video_input = video_input.unsqueeze(0).to(device)

        with torch.inference_mode():
            gaze_outputs = model(
                {"video": video_input},
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=task_loss_req,
            )

        n_real  = int((~gaze_outputs["if_padded_gazing"]).sum().item())
        n_total = gaze_outputs["num_vision_tokens_each_frame"] * num_frames
        per_frame = gaze_outputs["num_gazing_each_frame"].tolist()
        print(f"  Gazed: {n_real}/{n_total} ({100*n_real/n_total:.1f}%)  |  per frame: {per_frame}")

        viz_kwargs = dict(
            transform=transform,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            normalize_rescale=normalize_rescale,
        )

        if "json" in fmts:
            save_json(gaze_outputs, video_path, all_json_results)

        if "viz" in fmts:
            p = save_viz(gaze_outputs, raw_video, out_dir, video_path, **viz_kwargs)
            saved_files.append(str(p))
            print(f"  viz    → {p}")

        if "frames" in fmts:
            paths = save_frames(gaze_outputs, raw_video, out_dir, video_path, **viz_kwargs)
            saved_files.extend(str(p) for p in paths)
            print(f"  frames → {paths[0].parent}/  ({len(paths)} files)")

        if "video" in fmts:
            p = save_video(gaze_outputs, raw_video, out_dir, video_path,
                           fps=args.video_fps, **viz_kwargs)
            saved_files.append(str(p))
            print(f"  video  → {p}")

        if "npy" in fmts:
            p = save_npy(gaze_outputs, out_dir, video_path)
            saved_files.append(str(p))
            print(f"  npy    → {p}")

    # -----------------------------------------------------------------------
    # Write aggregated JSON
    # -----------------------------------------------------------------------
    if "json" in fmts and all_json_results:
        json_path = out_dir / "gazing_labels.json"
        existing: dict = {}
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
        existing.update(all_json_results)
        with open(json_path, "w") as f:
            json.dump(existing, f)
        saved_files.append(str(json_path))
        print(f"\n  json   → {json_path}  ({len(all_json_results)} entries)")

    print(f"\nDone. {len(saved_files)} file(s) written to '{out_dir}'.")


if __name__ == "__main__":
    main()
