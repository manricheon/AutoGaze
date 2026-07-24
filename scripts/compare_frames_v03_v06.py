#!/usr/bin/env python
"""16f vs 32f: v0.3 vs v0.6, same clips, ratios 0.25 & 0.5. SigLIP2 variable-k
recall (v0.6 global allocation gives non-uniform per-frame counts).

Asks: does frame count (16 vs 32) change the v0.3-vs-v0.6 picture? (16f was the
sweet spot for v0.3/v0.5; 32f diluted.) PROXY-level; arbiter = downstream QA.

Usage: uv run python scripts/compare_frames_v03_v06.py --limit 24
Outputs: outputs/borissal/frames_v03_v06/{results.json, recall_bars.png}
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_borissal_semantic as sem  # noqa: E402
from compare_v03_v06 import recall_vark  # noqa: E402
from autogaze.models.borissal import Borissal, BorissalConfig  # noqa: E402
from autogaze.models.borissal.video_io import IMAGENET_MEAN, IMAGENET_STD, load_video  # noqa: E402

SELECTORS = {"v0.3": lambda s: BorissalConfig.v0_3(scale=s),
             "v0.6": lambda s: BorissalConfig.v0_6(scale=s)}
FRAMES = [16, 32]
RATIOS = [0.25, 0.5]
# color by selector, hatch/alpha by frame count
COLORS = {"v0.3": "#5B6B7B", "v0.6": "#C0392B"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "frames_v03_v06"))
    args = p.parse_args()

    device = torch.device(args.device)
    encoder, processor = sem.build_encoder(device)
    scale = encoder.config.image_size
    grid = scale // encoder.config.patch_size
    videos = sorted(Path(args.videos_dir).glob("*.mp4"))[: args.limit]
    assert videos, f"no clips under {args.videos_dir}"

    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    enc_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    enc_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)

    models = {n: Borissal(fn(scale)) for n, fn in SELECTORS.items()}
    # recalls[selector][frames][ratio] = list per clip
    rec = {n: {f: {r: [] for r in RATIOS} for f in FRAMES} for n in SELECTORS}

    with torch.no_grad():
        for vi, path in enumerate(videos):
            for f in FRAMES:
                video = load_video(str(path), num_frames=f, size=scale)
                frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)
                tokens = encoder(pixel_values=frames).last_hidden_state
                for n, model in models.items():
                    for r in RATIOS:
                        sel = model.select(video, gazing_ratio=r)
                        T_grid = sel.grid_thw[0, 0].item()
                        mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
                        fm = mask_tub.repeat_interleave(f // T_grid, dim=0).to(device)
                        rec[n][f][r].append(recall_vark(encoder, tokens, fm))
            print(f"clip {vi + 1}/{len(videos)} done", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {n: {str(f): {str(r): float(np.mean(rec[n][f][r])) for r in RATIOS} for f in FRAMES}
               for n in SELECTORS}
    (out_dir / "results.json").write_text(json.dumps(
        {"args": vars(args), "n_clips": len(videos), "summary": summary}, indent=2))

    # table
    print(f"\n{'condition':>16} | " + " | ".join(f"gr={r}" for r in RATIOS))
    print("-" * 40)
    for n in SELECTORS:
        for f in FRAMES:
            cells = " | ".join(f"{np.mean(rec[n][f][r]):6.4f}" for r in RATIOS)
            print(f"{n+' @'+str(f)+'f':>16} | {cells}")

    # grouped bars: one subplot per ratio, bars = 4 conditions (sel x frames)
    conds = [(n, f) for n in SELECTORS for f in FRAMES]
    labels = [f"{n}\n{f}f" for n, f in conds]
    fig, axes = plt.subplots(1, len(RATIOS), figsize=(5.2 * len(RATIOS), 4.2), squeeze=False)
    for ax, r in zip(axes[0], RATIOS):
        vals = [np.mean(rec[n][f][r]) for n, f in conds]
        errs = [np.std(rec[n][f][r]) for n, f in conds]
        bars = ax.bar(range(len(conds)), vals, yerr=errs, capsize=3,
                      color=[COLORS[n] for n, _ in conds],
                      alpha=[1.0 if f == 16 else 0.55 for _, f in conds])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("SigLIP2 recall (proxy)")
        ax.set_title(f"gazing ratio {r}", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"16f vs 32f — v0.3 vs v0.6 ({len(videos)} clips) — solid=16f, faded=32f — PROXY",
                 fontsize=12)
    plt.tight_layout(); fig.savefig(out_dir / "recall_bars.png", dpi=130); plt.close(fig)
    print(f"\nsaved {out_dir}/results.json, recall_bars.png")


if __name__ == "__main__":
    main()
