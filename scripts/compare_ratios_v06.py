#!/usr/bin/env python
"""Compare the v0.6 default (all saliency-v3.1 knobs ON) ACROSS gazing ratios.

Two views (Mac/CPU, SigLIP2 proxy):
  1. recall + gist vs gazing_ratio (line, 16 held-out clips) -- proxy quality/
     budget trade-off; watch for the non-monotonic 0.75 region seen downstream.
  2. selection overlays at each ratio on representative clips -- the nested
     growth (0.15 subset 0.25 subset ... subset 1.0) made visual.

PROXY-LEVEL (recall mis-ranked v0.4/motion_weight before). Arbiter = CUDA QA.

Usage: uv run python scripts/compare_ratios_v06.py
Outputs: outputs/borissal/v06_ratios/{results.json, recall_vs_ratio.png, overlays/}
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
from autogaze.models.borissal import Borissal, BorissalConfig  # noqa: E402
from autogaze.models.borissal.video_io import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, load_video, unnormalize,
)

RATIOS = [0.15, 0.25, 0.5, 0.75, 1.0]


def render_ratio_grid(disp, masks_by_ratio, tubelet_size, sample_tubelets, out_path):
    """rows = sampled tubelets, cols = ratios. Each cell = that tubelet's rep
    frame with kept patches outlined red -- shows nested growth across ratio."""
    ratios = list(masks_by_ratio)
    H, W = disp.shape[-2], disp.shape[-1]
    any_mask = masks_by_ratio[ratios[0]]
    Hg, Wg = any_mask.shape[-2], any_mask.shape[-1]
    ph, pw = H // Hg, W // Wg
    nr, nc = len(sample_tubelets), len(ratios)
    fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.6 * nr), squeeze=False)
    for ri, t in enumerate(sample_tubelets):
        frame = disp[min(t * tubelet_size, disp.shape[0] - 1)].permute(1, 2, 0).numpy()
        for ci, r in enumerate(ratios):
            m = masks_by_ratio[r][t].float().numpy()
            m_full = m.repeat(ph, axis=0).repeat(pw, axis=1)
            ax = axes[ri, ci]
            ax.imshow((frame * (0.82 * m_full[..., None] + 0.18)).clip(0, 1))
            for i in range(Hg):
                for j in range(Wg):
                    if m[i, j] > 0.5:
                        ax.add_patch(plt.Rectangle((j * pw - 0.5, i * ph - 0.5), pw, ph,
                                                   lw=0.5, edgecolor="red", facecolor="none"))
            ax.axis("off")
            if ri == 0:
                ax.set_title(f"ratio {r}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"t={t}", fontsize=9)
    fig.suptitle("v0.6 (all knobs ON) — selection vs gazing ratio (nested growth)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-overlay-clips", type=int, default=2)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "v06_ratios"))
    args = p.parse_args()

    device = torch.device(args.device)
    encoder, processor = sem.build_encoder(device)
    scale = encoder.config.image_size
    grid = scale // encoder.config.patch_size
    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    assert videos, f"no clips under {args.videos_dir}"

    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    enc_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    enc_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)

    model = Borissal(BorissalConfig.v0_6(scale=scale))   # DEFAULT = all knobs ON
    recalls = {r: [] for r in RATIOS}
    gists = {r: [] for r in RATIOS}

    with torch.no_grad():
        for vi, path in enumerate(videos):
            video = load_video(str(path), num_frames=args.num_frames, size=scale)
            frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)
            tokens = encoder(pixel_values=frames).last_hidden_state
            for r in RATIOS:
                sel = model.select(video, gazing_ratio=r)
                T_grid = sel.grid_thw[0, 0].item()
                mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
                fm = mask_tub.repeat_interleave(args.num_frames // T_grid, dim=0).to(device)
                # gist/recall need a uniform per-frame count; ratio 1.0 keeps all
                if int(fm[0].sum()) in (0, grid * grid):
                    # all-or-nothing: recall trivially 1.0 / gist 1.0 at r=1.0
                    gist = 1.0 if int(fm[0].sum()) == grid * grid else 0.0
                    recall = 1.0 if int(fm[0].sum()) == grid * grid else 0.0
                else:
                    gist, recall = sem.semantic_metrics(encoder, tokens, fm)
                recalls[r].append(recall)
                gists[r].append(gist)
            print(f"clip {vi + 1}/{len(videos)} done", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = {"ratios": RATIOS, "recall_mean": [float(np.mean(recalls[r])) for r in RATIOS],
           "gist_mean": [float(np.mean(gists[r])) for r in RATIOS],
           "recalls": recalls, "gists": gists}
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # line chart: recall + gist vs ratio
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rm = [np.mean(recalls[r]) for r in RATIOS]
    gm = [np.mean(gists[r]) for r in RATIOS]
    ax.plot(RATIOS, rm, marker="o", color="#2E8B8B", label="recall")
    ax.plot(RATIOS, gm, marker="s", color="#E08A1E", label="gist")
    for r, y in zip(RATIOS, rm):
        ax.annotate(f"{y:.3f}", (r, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xlabel("gazing ratio"); ax.set_ylabel("SigLIP2 proxy metric")
    ax.set_title("v0.6 (all knobs ON) — proxy quality vs gazing ratio\nPROXY, confirm on CUDA QA", fontsize=11)
    ax.set_xticks(RATIOS); ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); fig.savefig(out_dir / "recall_vs_ratio.png", dpi=130); plt.close(fig)

    # overlays: nested growth across ratios on the first n clips
    ov_dir = out_dir / "overlays"; ov_dir.mkdir(exist_ok=True)
    for vi in range(min(args.n_overlay_clips, len(videos))):
        video = load_video(str(videos[vi]), num_frames=args.num_frames, size=scale)
        disp = unnormalize(video[0]).cpu()
        T_grid = args.num_frames // model.config.tubelet_size
        masks = {}
        for r in RATIOS:
            sel = model.select(video, gazing_ratio=r)
            masks[r] = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
        sample_t = sorted({0, T_grid // 2, T_grid - 1})
        render_ratio_grid(disp, masks, model.config.tubelet_size, sample_t,
                          str(ov_dir / f"clip{vi}_{videos[vi].stem[:20]}.png"))

    print("\nratio :", RATIOS)
    print("recall:", [f"{v:.3f}" for v in rm])
    print("gist  :", [f"{v:.3f}" for v in gm])
    print(f"saved {out_dir}/results.json, recall_vs_ratio.png, overlays/")


if __name__ == "__main__":
    main()
