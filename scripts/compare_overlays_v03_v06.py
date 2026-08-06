#!/usr/bin/env python
"""Side-by-side selection overlays: v0.3 vs v0.6 (maximal saliency-v3.1 port),
one figure per clip, rows = [Original, v0.3, v0.6] over sampled tubelets, so the
two selectors line up under the same frames for an easy visual diff. No encoder
needed (selection + render only) -> fast, many examples.

Usage: uv run python scripts/compare_overlays_v03_v06.py --limit 12 --ratio 0.25
Outputs: outputs/borissal/v03_vs_v06/side_by_side/clipNN.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.video_io import load_video, unnormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
ROWS = [("v0.3", lambda s: BorissalConfig.v0_3(scale=s)),
        ("v0.6", lambda s: BorissalConfig.v0_6(scale=s))]  # v0.6 default = global


def render_pair(disp, masks, tubelet_size, sample_t, ratio, out_path, clip_name):
    """disp (T,C,H,W) in [0,1]; masks = {name: (T_grid,Hg,Wg) bool}. Rows:
    Original / each selector; cols: sampled tubelets."""
    H, W = disp.shape[-2], disp.shape[-1]
    any_m = next(iter(masks.values()))
    Hg, Wg = any_m.shape[-2], any_m.shape[-1]
    ph, pw = H // Hg, W // Wg
    names = list(masks)
    nrow, ncol = 1 + len(names), len(sample_t)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.25 * nrow), squeeze=False)
    row_labels = ["Original"] + names
    colors = {"v0.3": "#3AA0FF", "v0.6": "#FF3B30"}
    for ci, t in enumerate(sample_t):
        frame = disp[min(t * tubelet_size, disp.shape[0] - 1)].permute(1, 2, 0).numpy()
        axes[0, ci].imshow(frame)
        axes[0, ci].set_title(f"t={t}", fontsize=9)
        for ri, name in enumerate(names, start=1):
            m = masks[name][t].float().numpy()
            m_full = m.repeat(ph, axis=0).repeat(pw, axis=1)
            ax = axes[ri, ci]
            ax.imshow((frame * (0.82 * m_full[..., None] + 0.18)).clip(0, 1))
            for i in range(Hg):
                for j in range(Wg):
                    if m[i, j] > 0.5:
                        ax.add_patch(plt.Rectangle((j * pw - 0.5, i * ph - 0.5), pw, ph,
                                                   lw=0.5, edgecolor=colors.get(name, "red"),
                                                   facecolor="none"))
    for ri in range(nrow):
        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=11, rotation=90, labelpad=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    for ri, name in enumerate(names, start=1):
        kept = int(masks[name].sum())
        axes[ri, 0].set_ylabel(f"{name}\n({kept} tok)", fontsize=10)
    fig.suptitle(f"{clip_name}  —  v0.3 vs v0.6 selection @ ratio {ratio}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=125)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--n-tubelets", type=int, default=6)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "v03_vs_v06" / "side_by_side"))
    args = p.parse_args()

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))[: args.limit]
    assert videos, f"no clips under {args.videos_dir}"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = {name: Borissal(fn(args.scale)) for name, fn in ROWS}
    tub = models["v0.3"].config.tubelet_size
    T_grid = args.num_frames // tub
    sample_t = sorted(set(np.linspace(0, T_grid - 1, args.n_tubelets).round().astype(int).tolist()))
    grid = args.scale // models["v0.3"].config.patch_size

    with torch.no_grad():
        for vi, path in enumerate(videos):
            video = load_video(str(path), num_frames=args.num_frames, size=args.scale)
            disp = unnormalize(video[0]).cpu()
            masks = {}
            for name, model in models.items():
                sel = model.select(video, gazing_ratio=args.ratio)
                masks[name] = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
            out = out_dir / f"clip{vi:02d}_{path.stem[:22]}.png"
            render_pair(disp, masks, tub, sample_t, args.ratio, str(out), path.stem[:22])
            print(f"clip {vi + 1}/{len(videos)} -> {out.name}", flush=True)
    print(f"saved {len(videos)} side-by-side figures under {out_dir}")


if __name__ == "__main__":
    main()
