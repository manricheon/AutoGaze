#!/usr/bin/env python
"""v0.6 proxy screen: do the saliency-v3.1-inspired knobs regress v0.5, and
where do they move the selection? (Mac/CPU, SigLIP2 description-aligned recall.)

Compares v0.5 vs v0.6 variants (static_guard / laplacian_gate / center_bias) on
held-out clips with the SigLIP2 gist+recall gate (encode once per clip, re-score
each variant's mask). PROXY-LEVEL: recall has mis-ranked vs the real caption->QA
(v0.4, motion_weight); this is a no-regression + direction screen only. The
knobs' downstream value is externally evidenced by saliency-v3.1 (which ships
them); the arbiter is the CUDA QA run.

Usage:
    uv run python scripts/sweep_borissal_v06.py --videos-dir videos/internvid_eval16
Outputs: outputs/borissal/v06_sweep/{results.json, recall_bars.png, overlays/}
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
from autogaze.models.borissal.viz import render_overlay  # noqa: E402

VARIANTS = {
    "v0.5": lambda s: BorissalConfig.v0_5(scale=s),
    "v0.6+static": lambda s: BorissalConfig.v0_6(scale=s, static_guard=True, static_guard_weight=0.5),
    "v0.6+laplacian": lambda s: BorissalConfig.v0_6(scale=s, laplacian_gate=True),
    "v0.6+center": lambda s: BorissalConfig.v0_6(scale=s, center_bias=0.3),
}
COLORS = {"v0.5": "#5B6B7B", "v0.6+static": "#2E8B8B",
          "v0.6+laplacian": "#E08A1E", "v0.6+center": "#9B5BA5"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "v06_sweep"))
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

    models = {name: Borissal(fn(scale)) for name, fn in VARIANTS.items()}
    recalls = {n: [] for n in VARIANTS}
    gists = {n: [] for n in VARIANTS}
    motion_energy = []   # per clip, to pick the most-static clip for overlays

    with torch.no_grad():
        for vi, path in enumerate(videos):
            video = load_video(str(path), num_frames=args.num_frames, size=scale)
            frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)
            tokens = encoder(pixel_values=frames).last_hidden_state
            # crude per-clip motion energy for the static-clip pick
            gray = video[0].mean(dim=1)
            motion_energy.append((gray[1:] - gray[:-1]).abs().mean().item())
            for name, model in models.items():
                sel = model.select(video, gazing_ratio=args.ratio)
                T_grid = sel.grid_thw[0, 0].item()
                mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
                frame_mask = mask_tub.repeat_interleave(args.num_frames // T_grid, dim=0).to(device)
                gist, recall = sem.semantic_metrics(encoder, tokens, frame_mask)
                recalls[name].append(recall)
                gists[name].append(gist)
            print(f"clip {vi + 1}/{len(videos)} done", flush=True)

    # table + W-L vs v0.5
    out = {"args": vars(args), "recalls": recalls, "gists": gists,
           "motion_energy": motion_energy, "summary": {}}
    hdr = f"{'variant':16} {'recall':>8} {'gist':>8}  {'recall W-L vs v0.5':>20}"
    print("\n" + hdr); print("-" * len(hdr))
    for name in VARIANTS:
        mr, mg = np.mean(recalls[name]), np.mean(gists[name])
        wl = ""
        if name != "v0.5":
            d = [x - y for x, y in zip(recalls[name], recalls["v0.5"])]
            wl = f"{sum(v>1e-6 for v in d)}W-{sum(v<-1e-6 for v in d)}L"
        out["summary"][name] = {"recall": mr, "gist": mg, "recall_wl_vs_v05": wl}
        print(f"{name:16} {mr:8.4f} {mg:8.4f}  {wl:>20}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # bar chart: recall + gist per variant
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, data in ((axes[0], "recall", recalls), (axes[1], "gist", gists)):
        names = list(VARIANTS)
        means = [np.mean(data[n]) for n in names]
        ax.bar(range(len(names)), means, color=[COLORS[n] for n in names])
        ax.axhline(np.mean(data["v0.5"]), ls="--", lw=1, color="#333", alpha=0.6)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"{metric} (proxy)"); ax.set_title(f"SigLIP2 {metric} vs v0.5 (dashed)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        lo = min(means) * 0.98; ax.set_ylim(lo, max(means) * 1.01)
    fig.suptitle(f"v0.6 knobs — proxy screen ({len(videos)} clips, ratio {args.ratio}) — PROXY, confirm on CUDA QA")
    plt.tight_layout(); fig.savefig(out_dir / "recall_bars.png", dpi=130); plt.close(fig)

    # overlays on the MOST STATIC clip (where static_guard should differ most)
    ov_dir = out_dir / "overlays"; ov_dir.mkdir(exist_ok=True)
    static_i = int(np.argmin(motion_energy))
    video = load_video(str(videos[static_i]), num_frames=args.num_frames, size=scale)
    disp = unnormalize(video[0]).cpu()
    for name, model in models.items():
        sel = model.select(video, gazing_ratio=args.ratio)
        T_grid = sel.grid_thw[0, 0].item()
        km = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
        render_overlay(disp, km, model.config.tubelet_size, str(ov_dir / f"static_clip_{name}.png"))
    print(f"\nmost-static clip = {videos[static_i].name} (motion {motion_energy[static_i]:.4f})")
    print(f"saved {out_dir}/results.json, recall_bars.png, overlays/")


if __name__ == "__main__":
    main()
