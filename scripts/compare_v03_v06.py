#!/usr/bin/env python
"""Compare borissal v0.3 vs v0.6 (maximal saliency-v3.1 port) across gazing
ratios, at 32 frames, over many clips. SigLIP2 description-aligned recall.

v0.6 uses GLOBAL (content-adaptive) allocation -> per-frame keep counts vary, so
the recall metric here is the variable-k robust form (top-attention recall works
per frame regardless of count), unlike eval_borissal_semantic's uniform-only one.

HONEST FRAMING: every v0.6 change (bt601 luma, static/laplacian/center/keyframe,
global allocation) OPPOSES the SigLIP-recall proxy's earlier preferences, so v0.6
may score LOWER than v0.3 here -- expected, and NOT the verdict. The proxy has
mis-ranked vs the real caption->action/risk-QA repeatedly; this run shows the
selection-behavior difference + confirms v0.6 runs at 32f across ratios. Arbiter
is the downstream pipeline.

Usage: uv run python scripts/compare_v03_v06.py --limit 24
Outputs: outputs/borissal/v03_vs_v06/{results.json, recall_vs_ratio.png, overlays/}
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

RATIOS = [0.15, 0.25, 0.5, 0.75, 1.0]
SELECTORS = {"v0.3": lambda s: BorissalConfig.v0_3(scale=s),
             "v0.6": lambda s: BorissalConfig.v0_6(scale=s)}
COLORS = {"v0.3": "#5B6B7B", "v0.6": "#C0392B"}


def recall_vark(encoder, tokens, frame_mask, top_frac=0.1):
    """Variable-per-frame-count robust recall: fraction of each frame's
    top-`top_frac` MAP-attention patches that the selection kept. Works for any
    per-frame count (v0.6 global allocation), unlike the uniform-only gate."""
    T, N, D = tokens.shape
    _, attn = sem.probe_pool(encoder, tokens, need_weights=True)  # (T, N)
    n_top = max(1, round(top_frac * N))
    top_idx = attn.topk(n_top, dim=-1).indices                    # (T, n_top)
    return frame_mask.gather(1, top_idx).float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--num-frames", type=int, default=32)
    p.add_argument("--center-crop", action="store_true", help="saliency-v3.1 resize+center-crop preprocessing")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "v03_vs_v06"))
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
    # recalls[selector][ratio] = list per clip
    recalls = {n: {r: [] for r in RATIOS} for n in SELECTORS}

    with torch.no_grad():
        for vi, path in enumerate(videos):
            video = load_video(str(path), num_frames=args.num_frames, size=scale,
                               center_crop=args.center_crop)
            frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)
            tokens = encoder(pixel_values=frames).last_hidden_state
            for n, model in models.items():
                for r in RATIOS:
                    sel = model.select(video, gazing_ratio=r)
                    T_grid = sel.grid_thw[0, 0].item()
                    mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
                    fm = mask_tub.repeat_interleave(args.num_frames // T_grid, dim=0).to(device)
                    recalls[n][r].append(recall_vark(encoder, tokens, fm))
            print(f"clip {vi + 1}/{len(videos)} done", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {n: {str(r): float(np.mean(recalls[n][r])) for r in RATIOS} for n in SELECTORS}
    out = {"args": vars(args), "n_clips": len(videos), "recalls": recalls, "summary": summary}
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # print table
    print(f"\n{'ratio':>7} | " + " | ".join(f"{n:>8}" for n in SELECTORS) + " |   v0.6-v0.3   W-L")
    print("-" * 56)
    for r in RATIOS:
        cells = " | ".join(f"{np.mean(recalls[n][r]):8.4f}" for n in SELECTORS)
        d = [a - b for a, b in zip(recalls["v0.6"][r], recalls["v0.3"][r])]
        wl = f"{sum(x>1e-6 for x in d)}W-{sum(x<-1e-6 for x in d)}L"
        print(f"{r:>7} | {cells} | {np.mean(d):+8.4f}  {wl}")

    # line chart: recall vs ratio, both selectors, with per-clip spread band
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in SELECTORS:
        means = np.array([np.mean(recalls[n][r]) for r in RATIOS])
        stds = np.array([np.std(recalls[n][r]) for r in RATIOS])
        ax.plot(RATIOS, means, marker="o", color=COLORS[n], label=n, lw=2)
        ax.fill_between(RATIOS, means - stds, means + stds, color=COLORS[n], alpha=0.12)
    ax.plot(RATIOS, RATIOS, ls=":", color="#888", lw=1, label="random (=ratio)")
    ax.set_xlabel("gazing ratio"); ax.set_ylabel("SigLIP2 recall (proxy)")
    ax.set_title(f"borissal v0.3 vs v0.6 (maximal saliency-v3.1 port)\n"
                 f"{args.num_frames}f, {len(videos)} clips, band=±1 std -- PROXY (arbiter=downstream QA)",
                 fontsize=11)
    ax.set_xticks(RATIOS); ax.grid(alpha=0.25); ax.legend()
    plt.tight_layout(); fig.savefig(out_dir / "recall_vs_ratio.png", dpi=130); plt.close(fig)

    # overlays: v0.3 vs v0.6 at ratio 0.25 and 0.5 on the first 2 clips (32f -> 16 tubelets)
    ov_dir = out_dir / "overlays"; ov_dir.mkdir(exist_ok=True)
    for vi in range(min(2, len(videos))):
        video = load_video(str(videos[vi]), num_frames=args.num_frames, size=scale,
                           center_crop=args.center_crop)
        disp = unnormalize(video[0]).cpu()
        for r in (0.25, 0.5):
            for n, model in models.items():
                sel = model.select(video, gazing_ratio=r)
                T_grid = sel.grid_thw[0, 0].item()
                km = sel.keep_mask[0].reshape(T_grid, grid, grid).cpu()
                render_overlay(disp, km, model.config.tubelet_size,
                               str(ov_dir / f"clip{vi}_r{int(r*100)}_{n}.png"))
    print(f"\nsaved {out_dir}/results.json, recall_vs_ratio.png, overlays/")


if __name__ == "__main__":
    main()
