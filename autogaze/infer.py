# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AutoGaze inference script.

Runs AutoGaze on a single video or a directory of videos and saves results.

Output formats (can be combined with --output-format):
  json  — gazing_pos per frame as JSON, compatible with gazing_labels.json
           used in NTP pre-training.
  viz   — PNG visualization: original frames + multi-scale gaze mask overlays.
  npy   — numpy arrays of gazing masks (one .npz per video).

Usage examples:
  # Single video → all output formats
  python -m autogaze.infer assets/example_input.mp4 --output-dir results/

  # Directory of videos → only JSON labels (for NTP training data generation)
  python -m autogaze.infer /data/my_videos/ --output-dir results/ --output-format json

  # Custom gazing ratio
  python -m autogaze.infer assets/example_input.mp4 --gazing-ratio 0.5

  # Use local model checkpoint instead of HuggingFace
  python -m autogaze.infer assets/example_input.mp4 --model-path weights/AutoGaze
"""

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

from autogaze.datasets.video_utils import (
    read_video_pyav,
    sample_frame_indices,
    process_video_frames,
    transform_video_for_pytorch,
)
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.utils import UnNormalize, get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def collect_video_paths(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    videos = sorted(
        f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found under {input_path}")
    return videos


def load_video(video_path: Path, num_frames: int) -> np.ndarray:
    container = av.open(str(video_path))
    total = container.streams.video[0].frames
    indices = sample_frame_indices(
        clip_len=num_frames,
        frame_sample_rate=1,
        seg_len=total,
        random_sample_frame=False,
    )
    raw = read_video_pyav(container, indices)
    container.close()
    return process_video_frames(raw, num_frames)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_json(result: dict, out_dir: Path, video_path: Path, all_results: dict):
    """Accumulate result into the shared all_results dict (written at the end)."""
    # Key = last 3 path components, matching gazing_labels.json convention
    parts = video_path.parts
    key = str(Path(*parts[-3:])) if len(parts) >= 3 else str(video_path)

    # gazing_pos: list[list[int]] — one list per frame, each entry is a
    # *within-frame* patch index (0-based, not offset by frame number).
    num_tokens = result["num_vision_tokens_each_frame"]
    gazing_pos_per_frame = []
    task_losses_per_frame = []

    gazing_pos_flat = result["gazing_pos"][0]       # (N,)
    if_padded_flat = result["if_padded_gazing"][0]  # (N,)
    num_each = result["num_gazing_each_frame"]       # (T,)

    offset = 0
    for t, count in enumerate(num_each.tolist()):
        frame_pos = gazing_pos_flat[offset: offset + count]
        frame_pad = if_padded_flat[offset: offset + count]
        # convert global index → within-frame index
        within_frame = (frame_pos[~frame_pad] - t * num_tokens).tolist()
        gazing_pos_per_frame.append([int(x) for x in within_frame])
        task_losses_per_frame.append([0.0] * len(within_frame))  # placeholder
        offset += count

    all_results[key] = {
        "gazing_pos": gazing_pos_per_frame,
        "task_losses": task_losses_per_frame,
    }


def save_viz(result: dict, raw_video: np.ndarray, out_dir: Path, video_path: Path,
             transform, normalize_mean, normalize_std, normalize_rescale):
    """Save a PNG grid: original frames + per-scale gaze mask overlays."""
    scales = result["scales"]
    gazing_mask = result["gazing_mask"]  # list[Tensor(1, T, N_scale)]
    num_tokens_each_frame = result["num_vision_tokens_each_frame"]
    T = len(scales[0:1]) and gazing_mask[0].shape[1]
    num_scales = len(scales)

    # Reconstruct normalised video tensor for display
    video_tensor = transform_video_for_pytorch(raw_video, transform)  # (T, C, H, W)
    unnorm = UnNormalize(normalize_mean, normalize_std, normalize_rescale)
    video_np = unnorm(video_tensor).cpu().float().numpy()  # (T, C, H, W)

    rows = num_scales + 1
    fig, axes = plt.subplots(rows, T, figsize=(max(2 * T, 4), 2.5 * rows),
                             squeeze=False)

    # Row 0: original frames
    for t in range(T):
        frame = video_np[t].transpose(1, 2, 0)  # H W C
        axes[0, t].imshow(np.clip(frame, 0, 1))
        axes[0, t].set_title(f"Frame {t + 1}", fontsize=7)
        axes[0, t].axis("off")

    # Rows 1…: per-scale gaze overlay
    for si, scale in enumerate(scales):
        mask_scale = gazing_mask[si][0]  # (T, N_scale)  — batch index 0
        patch_grid = int(mask_scale.shape[-1] ** 0.5)

        for t in range(T):
            # Resize original frame to this scale for context
            frame_t = torch.from_numpy(video_np[t]).unsqueeze(0)  # 1 C H W
            frame_scaled = F.interpolate(frame_t, size=(scale, scale),
                                         mode="bicubic", align_corners=False)
            frame_scaled = frame_scaled.squeeze().clamp(0, 1).numpy()  # C H W

            pm = mask_scale[t].reshape(patch_grid, patch_grid).cpu().float().numpy()
            pm_up = F.interpolate(
                torch.from_numpy(pm).unsqueeze(0).unsqueeze(0),
                size=(scale, scale), mode="nearest"
            ).squeeze().numpy()

            # Dimmed background + bright foreground for gazed patches
            display = frame_scaled * (0.25 + 0.75 * pm_up[None])
            axes[si + 1, t].imshow(display.transpose(1, 2, 0))

            # Red rectangle borders around gazed patches
            patch_px = scale // patch_grid
            for pi in range(patch_grid):
                for pj in range(patch_grid):
                    if pm[pi, pj] > 0.5:
                        rect = patches.Rectangle(
                            (pj * patch_px - 0.5, pi * patch_px - 0.5),
                            patch_px, patch_px,
                            linewidth=0.8, edgecolor="red", facecolor="none",
                        )
                        axes[si + 1, t].add_patch(rect)

            axes[si + 1, t].set_title(f"Scale {scale} F{t + 1}", fontsize=7)
            axes[si + 1, t].axis("off")

    plt.tight_layout(pad=0.3)
    stem = video_path.stem
    out_path = out_dir / f"{stem}_gaze.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_npy(result: dict, out_dir: Path, video_path: Path):
    """Save raw gazing data as a compressed numpy archive."""
    stem = video_path.stem
    out_path = out_dir / f"{stem}_gaze.npz"

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

def parse_args():
    p = argparse.ArgumentParser(description="AutoGaze inference")
    p.add_argument("input", help="Video file or directory of videos")
    p.add_argument("--model-path", default="nvidia/AutoGaze",
                   help="HuggingFace model ID or local path (default: nvidia/AutoGaze)")
    p.add_argument("--output-dir", default="results",
                   help="Directory to write outputs (default: results/)")
    p.add_argument("--output-format", default="all",
                   choices=["all", "json", "viz", "npy"],
                   help="Output format (default: all)")
    p.add_argument("--gazing-ratio", type=float, default=0.75,
                   help="Fraction of patches to gaze at (default: 0.75)")
    p.add_argument("--task-loss-requirement", type=float, default=0.7,
                   help="Reconstruction loss threshold for early stopping (default: 0.7). "
                        "Set to None to disable.")
    p.add_argument("--no-task-loss-requirement", action="store_true",
                   help="Disable task-loss-based early stopping")
    p.add_argument("--num-frames", type=int, default=16,
                   help="Number of frames to sample per video (default: 16)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Inference batch size (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json_fmt = args.output_format in ("all", "json")
    save_viz_fmt  = args.output_format in ("all", "viz")
    save_npy_fmt  = args.output_format in ("all", "npy")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"Loading AutoGaze from '{args.model_path}'...")
    transform = AutoGazeImageProcessor.from_pretrained(args.model_path)
    model = AutoGaze.from_pretrained(args.model_path).to(device)
    model.eval()
    num_frames = getattr(model.config, "max_num_frames", args.num_frames)
    print(f"  Device: {device} | max_num_frames: {num_frames}")

    # Normalisation params for visualisation
    normalize_mean = transform.image_mean
    normalize_std  = transform.image_std
    normalize_rescale = getattr(transform, "rescale_factor", 1.0 / 255.0)

    # -----------------------------------------------------------------------
    # Collect videos
    # -----------------------------------------------------------------------
    video_paths = collect_video_paths(args.input)
    print(f"Found {len(video_paths)} video(s) under '{args.input}'")

    task_loss_req = (
        None if args.no_task_loss_requirement
        else args.task_loss_requirement
    )

    all_json_results: dict = {}
    saved_files: list[str] = []

    # -----------------------------------------------------------------------
    # Per-video inference
    # -----------------------------------------------------------------------
    for idx, video_path in enumerate(video_paths):
        print(f"[{idx + 1}/{len(video_paths)}] {video_path.name}")

        try:
            raw_video = load_video(video_path, num_frames)
        except Exception as e:
            print(f"  WARNING: could not load video — {e}")
            continue

        video_input = transform_video_for_pytorch(raw_video, transform)
        video_input = video_input.unsqueeze(0).to(device)  # (1, T, C, H, W)

        with torch.inference_mode():
            gaze_outputs = model(
                {"video": video_input},
                gazing_ratio=args.gazing_ratio,
                task_loss_requirement=task_loss_req,
            )

        n_real = (~gaze_outputs["if_padded_gazing"]).sum().item()
        n_total = gaze_outputs["num_vision_tokens_each_frame"] * num_frames
        print(f"  Gazed patches: {n_real} / {n_total} "
              f"({100 * n_real / n_total:.1f}%)")

        if save_json_fmt:
            save_json(gaze_outputs, out_dir, video_path, all_json_results)

        if save_viz_fmt:
            out_png = save_viz(
                gaze_outputs, raw_video, out_dir, video_path,
                transform, normalize_mean, normalize_std, normalize_rescale,
            )
            saved_files.append(str(out_png))
            print(f"  Viz  → {out_png}")

        if save_npy_fmt:
            out_npz = save_npy(gaze_outputs, out_dir, video_path)
            saved_files.append(str(out_npz))
            print(f"  NPZ  → {out_npz}")

    # -----------------------------------------------------------------------
    # Write aggregated JSON
    # -----------------------------------------------------------------------
    if save_json_fmt and all_json_results:
        json_path = out_dir / "gazing_labels.json"
        existing = {}
        if json_path.exists():
            with open(json_path) as f:
                existing = json.load(f)
        existing.update(all_json_results)
        with open(json_path, "w") as f:
            json.dump(existing, f)
        saved_files.append(str(json_path))
        print(f"  JSON → {json_path}  ({len(all_json_results)} entries)")

    print(f"\nDone. {len(saved_files)} file(s) written to '{out_dir}'.")


if __name__ == "__main__":
    main()
