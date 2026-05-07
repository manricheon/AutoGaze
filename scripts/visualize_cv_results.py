#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Visualize saved results from run_cv_tasks.py.

Usage:
    # show latest run
    python scripts/visualize_cv_results.py --results-dir results/cv_tasks/20250430_120000

    # list all runs
    python scripts/visualize_cv_results.py --list --results-dir results/cv_tasks

    # show only specific images
    python scripts/visualize_cv_results.py --results-dir results/cv_tasks/20250430_120000 \
        --show depth segmentation

    # save to file instead of interactive display
    python scripts/visualize_cv_results.py --results-dir results/cv_tasks/20250430_120000 \
        --save overview.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize AutoGaze CV task results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", required=True,
                   help="Path to a specific run directory OR root results dir (with --list)")
    p.add_argument("--list", action="store_true",
                   help="List all run directories under results-dir")
    p.add_argument("--latest", action="store_true",
                   help="Auto-select the most recent run under results-dir")
    p.add_argument("--show", nargs="*",
                   choices=["gaze", "depth", "detection", "recognition",
                             "segmentation", "siglip", "summary", "metrics"],
                   default=None,
                   help="Which panels to display (default: all found)")
    p.add_argument("--save", default=None,
                   help="Save combined overview to this path instead of showing interactively")
    p.add_argument("--dpi", type=int, default=110)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

FILE_MAP = {
    "gaze":        "gaze_map.png",
    "depth":       "depth_comparison.png",
    "detection":   "detection_comparison.png",
    "recognition": "recognition_comparison.png",
    "segmentation":"segmentation_comparison.png",
    "siglip":      "siglip_comparison.png",
    "summary":     "summary.png",
}


def list_runs(root):
    root = Path(root)
    runs = sorted(root.glob("*/metrics.json"), reverse=True)
    if not runs:
        runs = sorted([d for d in root.iterdir() if d.is_dir()], reverse=True)
    else:
        runs = [r.parent for r in runs]
    return runs


def find_run_dir(args):
    d = Path(args.results_dir)
    if args.list:
        runs = list_runs(d)
        print(f"Found {len(runs)} run(s) under {d}:")
        for r in runs:
            metrics_f = r / "metrics.json"
            tasks = list(json.load(open(metrics_f)).keys()) if metrics_f.exists() else []
            print(f"  {r.name}  tasks={tasks}")
        sys.exit(0)
    if args.latest:
        runs = list_runs(d)
        if not runs:
            sys.exit(f"No runs found under {d}")
        return runs[0]
    if (d / "metrics.json").exists() or any(d.glob("*.png")):
        return d
    # maybe a root dir with one or more runs — pick latest
    runs = list_runs(d)
    if runs:
        print(f"Auto-selecting latest run: {runs[0].name}")
        return runs[0]
    sys.exit(f"No run directory found at {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def _load_png(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def show_metrics(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("\n" + "─" * 60)
    print("metrics.json")
    print("─" * 60)
    for task, vals in metrics.items():
        print(f"  {task}:")
        for k, v in vals.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
    print("─" * 60)


def display_images(run_dir, show_keys, save_path, dpi):
    import matplotlib.pyplot as plt
    import platform
    if platform.system() == "Darwin":
        import matplotlib
        matplotlib.rc("font", family="AppleGothic")
    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False

    found = {}
    for key, fname in FILE_MAP.items():
        p = run_dir / fname
        if p.exists():
            found[key] = p

    if show_keys:
        found = {k: v for k, v in found.items() if k in show_keys}

    if not found:
        print("No result images found.")
        return

    titles = {
        "gaze":        "AutoGaze Gaze Map",
        "depth":       "Depth Estimation 비교",
        "detection":   "Object Detection 비교",
        "recognition": "Recognition 비교",
        "segmentation":"Segmentation 비교",
        "siglip":      "SigLIP Zero-shot 비교",
        "summary":     "종합 요약",
    }

    n = len(found)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5.5 * n))
    if n == 1:
        axes = [axes]

    for ax, (key, img_path) in zip(axes, found.items()):
        img = _load_png(img_path)
        ax.imshow(img)
        ax.set_title(titles.get(key, key), fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    run_name = run_dir.name
    fig.suptitle(f"AutoGaze CV Task Results — {run_name}",
                 fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved overview: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def show_video_list(run_dir):
    videos = sorted(run_dir.glob("*_video.mp4"))
    if not videos:
        return
    print("\nVideo outputs:")
    for v in videos:
        size_mb = v.stat().st_size / 1024 / 1024
        print(f"  {v.name}  ({size_mb:.1f} MB)")
    print("  (open with: open <file>  or  vlc <file>)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_dir = find_run_dir(args)
    print(f"Run directory: {run_dir}")

    show_keys = set(args.show) if args.show else None
    if show_keys and "metrics" in show_keys:
        show_keys.discard("metrics")
        metrics_f = run_dir / "metrics.json"
        if metrics_f.exists():
            show_metrics(metrics_f)

    if not show_keys or show_keys - {"metrics"}:
        display_images(run_dir, show_keys, args.save, args.dpi)

    if not args.show or "metrics" in (args.show or []):
        metrics_f = run_dir / "metrics.json"
        if metrics_f.exists() and not (show_keys and "metrics" in show_keys):
            show_metrics(metrics_f)

    show_video_list(run_dir)


if __name__ == "__main__":
    main()
