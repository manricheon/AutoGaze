#!/usr/bin/env python
"""Render a Borissal v1 train_log.jsonl into a one-page trend dashboard PNG.

The fastest way to eyeball whether a (local or scale) run is healthy:

    uv run python scripts/plot_borissal_training.py \
        weights/borissal_v1_trend_check/train_log.jsonl

Healthy signatures (see docs/borissal/training.md §2/§7):
- loss/predictor_coverage trending down (the actual learning signal)
- grad_norm NOT decaying to ~0 (score saturation)
- probe_overlap_prev NOT pinned at 1.0 (frozen selection); ~0.3 = random-ish
- score_entropy_mean stable or rising (saturation counter-indicator)
- lgrad ratios: unselected/low-decile staying within ~an order of magnitude
  of selected (gradient reach; collapse = score lock-in)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="path to train_log.jsonl")
    p.add_argument("--out", default=None, help="output PNG (default: alongside the log)")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.log)]
    if not rows:
        raise SystemExit("empty log")
    steps = [r["step"] for r in rows]

    def series(key):
        return [r.get(key) for r in rows]

    loss_keys = sorted({k for r in rows for k in r if k.startswith("loss/") and k != "loss/total"})

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    for k in loss_keys:
        ax.plot(steps, series(k), label=k.removeprefix("loss/"))
    ax.plot(steps, series("loss/total"), "k--", label="total")
    ax.set_title("loss terms")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, series("grad_norm"))
    ax.set_yscale("log")
    ax.set_title("grad_norm (log scale) — dying to ~0 = score saturation")
    ax.set_xlabel("step")

    ax = axes[0, 2]
    vals = [(s, v) for s, v in zip(steps, series("probe_overlap_prev")) if v is not None]
    if vals:
        ax.plot(*zip(*vals), marker="o", ms=3)
    ax.axhline(1.0, color="r", ls=":", label="1.0 = frozen selection")
    ax.axhline(0.3, color="gray", ls=":", label="~random (ratio 0.3)")
    ax.set_ylim(0, 1.05)
    ax.set_title("probe selection IoU vs previous log point")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, series("score_entropy_mean"))
    ax.set_title("score entropy (falling fast = sharpening/saturation)")
    ax.set_xlabel("step")

    ax = axes[1, 1]
    for k, lbl in [("lgrad_sel_mean", "selected"),
                   ("lgrad_unsel_mean", "unselected"),
                   ("lgrad_low_decile_mean", "lowest-prob 10%")]:
        vals = [(s, v) for s, v in zip(steps, series(k)) if v is not None]
        if vals:
            ax.plot(*zip(*vals), label=lbl)
    ax.set_yscale("log")
    ax.set_title("|dL/dlogit| by group — low-decile collapse = lock-in")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(steps, series("v0_overlap"), label="v0 overlap")
    r = [(s, v) for s, v in zip(steps, series("ratio")) if v is not None]
    ax.plot(*zip(*r), color="gray", alpha=0.5, label="sampled ratio")
    ax.set_ylim(0, 1.0)
    ax.set_title("v0-selection overlap & sampled ratio")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)

    fig.suptitle(Path(args.log).parent.name)
    fig.tight_layout()
    out = args.out or str(Path(args.log).with_name("training_dashboard.png"))
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
