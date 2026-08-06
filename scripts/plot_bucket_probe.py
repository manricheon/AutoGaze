#!/usr/bin/env python
"""Charts for the content-adaptivity probe (scripts/borissal_bucket_probe.py).

Reads outputs/borissal/bucket_probe/results.json and renders, per axis:
  1. grouped-bar: mean recall by bucket x allocator (one panel per judge)
  2. Delta-vs-uniform heatmap (allocator x bucket) -- the adaptivity fingerprint
  3. longform_deadtime allocation profile: uniform vs motion_prop vs oracle
     per-tubelet counts (mean across clips), frozen-tail region shaded
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# categorical palette (fixed order, CVD-aware): uniform / oracle / motion
ALLOC_COLORS = {"uniform": "#5B6B7B", "oracle_sig": "#E08A1E", "motion_prop": "#2E8B8B"}


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def plot_axis(axis, res, out_dir):
    allocs = res["alloc_names"]
    judges = res["judges"]
    buckets = list(res["recalls"].keys())

    # 1. grouped bars, one panel per judge
    fig, axes = plt.subplots(1, len(judges), figsize=(5.5 * len(judges), 4), squeeze=False)
    x = np.arange(len(buckets))
    w = 0.8 / len(allocs)
    for ji, jn in enumerate(judges):
        ax = axes[0, ji]
        for ai, a in enumerate(allocs):
            means = [_mean(res["recalls"][b][a][jn]) for b in buckets]
            ax.bar(x + ai * w - 0.4 + w / 2, means, w, label=a,
                   color=ALLOC_COLORS.get(a, "#888"))
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("recall (proxy)")
        ax.set_title(f"judge: {jn}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if ji == 0:
            ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(f"[{axis}] proxy recall by bucket x allocator", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f"{axis}_recall_bars.png", dpi=130)
    plt.close(fig)

    # 2. Delta-vs-uniform heatmap (avg over judges), rows=alloc(non-uniform), cols=bucket
    nonunif = [a for a in allocs if a != "uniform"]
    mat = np.zeros((len(nonunif), len(buckets)))
    for ai, a in enumerate(nonunif):
        for bi, b in enumerate(buckets):
            deltas = []
            for jn in judges:
                base = _mean(res["recalls"][b]["uniform"][jn])
                deltas.append(_mean(res["recalls"][b][a][jn]) - base)
            mat[ai, bi] = np.mean(deltas)
    vmax = max(1e-4, np.abs(mat).max())
    fig, ax = plt.subplots(figsize=(1.6 * len(buckets) + 1, 1.1 * len(nonunif) + 1.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(nonunif)))
    ax.set_yticklabels(nonunif, fontsize=9)
    for i in range(len(nonunif)):
        for j in range(len(buckets)):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                    color="black")
    ax.set_title(f"[{axis}] recall delta vs uniform (avg judges)\n"
                 "positive = adaptivity helps in that bucket", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_dir / f"{axis}_delta_heatmap.png", dpi=130)
    plt.close(fig)

    # 3. longform allocation profile (length axis only)
    if "longform_deadtime" in res["profiles"]:
        prof = res["profiles"]["longform_deadtime"]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        any_alloc = next(iter(prof.values()))
        T_grid = len(any_alloc[0]) if any_alloc else 0
        xg = np.arange(T_grid)
        for a in allocs:
            arr = np.array(prof[a])  # (n_clips, T_grid)
            ax.plot(xg, arr.mean(0), marker="o", label=a, color=ALLOC_COLORS.get(a, "#888"))
        ax.axvspan(T_grid / 2 - 0.5, T_grid - 0.5, color="gray", alpha=0.15,
                   label="frozen dead-time tail")
        ax.set_xlabel("tubelet index (t)")
        ax.set_ylabel("kept patches / tubelet")
        ax.set_title("[length] longform_deadtime allocation profile\n"
                     "adaptive allocators should pull budget OFF the shaded tail", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig.savefig(out_dir / "length_deadtime_profile.png", dpi=130)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(REPO_ROOT / "outputs" / "borissal" / "bucket_probe" / "results.json"))
    args = p.parse_args()
    data = json.loads(Path(args.results).read_text())
    out_dir = Path(args.results).parent
    for axis, res in data["axes"].items():
        plot_axis(axis, res, out_dir)
        print(f"plotted axis: {axis}")
    print(f"saved charts under {out_dir}")


if __name__ == "__main__":
    main()
