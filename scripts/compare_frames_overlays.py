#!/usr/bin/env python
"""Visual 16f-vs-32f selection overlays: rows = [v0.3@16f, v0.3@32f, v0.6@16f,
v0.6@32f], columns = the SAME time positions across the clip (so the 16f and 32f
rows line up in time). One figure per (clip, ratio). Encoder-free -> fast.

Usage: uv run python scripts/compare_frames_overlays.py --limit 6 --ratios 0.25 0.5
Outputs: outputs/borissal/frames_v03_v06/overlays/clipNN_rRR.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.video_io import load_video, unnormalize

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTORS = {"v0.3": lambda s: BorissalConfig.v0_3(scale=s),
             "v0.6": lambda s: BorissalConfig.v0_6(scale=s)}
FRAMES = [16, 32]
ROW_COLOR = {"v0.3": "#3AA0FF", "v0.6": "#FF3B30"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--scale", type=int, default=384)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5])
    p.add_argument("--n-cols", type=int, default=6)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "frames_v03_v06" / "overlays"))
    args = p.parse_args()

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))[: args.limit]
    assert videos, f"no clips under {args.videos_dir}"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = {n: Borissal(fn(args.scale)) for n, fn in SELECTORS.items()}
    tub = models["v0.3"].config.tubelet_size
    grid = args.scale // models["v0.3"].config.patch_size
    ph = pw = args.scale // grid
    rows = [(n, f) for n in SELECTORS for f in FRAMES]

    with torch.no_grad():
        for vi, path in enumerate(videos):
            # decode once per frame count; keep display frames + masks per (sel, f)
            data = {}   # f -> (disp, {sel: mask_grid})
            for f in FRAMES:
                video = load_video(str(path), num_frames=f, size=args.scale)
                disp = unnormalize(video[0]).cpu()
                data[f] = (disp, video)
            for ratio in args.ratios:
                fig, axes = plt.subplots(len(rows), args.n_cols,
                                         figsize=(2.05 * args.n_cols, 2.15 * len(rows)), squeeze=False)
                for ri, (name, f) in enumerate(rows):
                    disp, video = data[f]
                    T_grid = f // tub
                    sel = models[name].select(video, gazing_ratio=ratio)
                    mg = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
                    # sample n_cols tubelets by TIME fraction (aligns 16f & 32f)
                    fracs = np.linspace(0, 1, args.n_cols)
                    t_idx = (fracs * (T_grid - 1)).round().astype(int)
                    for ci, t in enumerate(t_idx):
                        frame = disp[min(t * tub, disp.shape[0] - 1)].permute(1, 2, 0).numpy()
                        m = mg[t].float().numpy()
                        m_full = m.repeat(ph, axis=0).repeat(pw, axis=1)
                        ax = axes[ri, ci]
                        ax.imshow((frame * (0.82 * m_full[..., None] + 0.18)).clip(0, 1))
                        for i in range(grid):
                            for j in range(grid):
                                if m[i, j] > 0.5:
                                    ax.add_patch(plt.Rectangle((j * pw - 0.5, i * ph - 0.5), pw, ph,
                                                               lw=0.4, edgecolor=ROW_COLOR[name], facecolor="none"))
                        ax.set_xticks([]); ax.set_yticks([])
                        if ri == 0:
                            ax.set_title(f"t≈{fracs[ci]:.0%}", fontsize=9)
                    axes[ri, 0].set_ylabel(f"{name} @{f}f\n({int(mg.sum())} tok)", fontsize=9)
                fig.suptitle(f"{path.stem[:24]}  —  16f vs 32f selection @ ratio {ratio}", fontsize=12)
                plt.tight_layout()
                out = out_dir / f"clip{vi:02d}_r{int(ratio*100)}.png"
                fig.savefig(str(out), dpi=120); plt.close(fig)
                print(f"clip {vi+1}/{len(videos)} ratio {ratio} -> {out.name}", flush=True)
    print(f"saved figures under {out_dir}")


if __name__ == "__main__":
    main()
