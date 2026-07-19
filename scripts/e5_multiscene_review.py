#!/usr/bin/env python
"""E5 revisit-condition check: does temporal allocation matter on MULTI-SCENE video?

The Phase-0 verdict (e5_teacher_review.py) found the oracle-allocation
ceiling ~= uniform on 2-7s single-scene clips. This script tests the
recorded revisit condition by CONSTRUCTING multi-scene videos: hard-cut
concatenations of held-out clips (2-clip splices -> 1 cut at tubelet 4;
4-clip splices -> 3 cuts). Cut positions are exactly known, so the
"different scenes carry different amounts of content" regime is guaranteed.

Compared allocations (all on frozen v0.3 patch scores):
  uniform        every tubelet gets K/T_grid
  global         v0.2-era default: clip-wide top-K + per-tubelet floor
  oracle         SigLIP2 MAP attention mass per tubelet (teacher sees ALL
                 frames densely; only the per-tubelet counts are taken)

Judges: CLIP ViT-L + DINOv2 recall (SigLIP2 excluded -- it feeds the oracle).

Usage:
    uv run python scripts/e5_multiscene_review.py --ratio 0.25
Outputs: outputs/borissal/e5_multiscene/{results.json,matrix.md}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from e5_teacher_review import (
    ClipCandidate, Dinov2Candidate, Siglip2Candidate, judge_recall, oracle_counts,
)


def judge_recall_clip_global(imp, frame_mask, top_frac: float = 0.1):
    """Clip-GLOBAL importance: targets = top-`top_frac` of ALL T*N patches
    (variable count per frame -- busy moments own more targets). Under this
    scope a temporal allocator CAN in principle beat uniform; the per-frame
    scope cannot (equal target counts make uniform structurally optimal)."""
    T, N = imp.shape
    n_top = max(1, round(top_frac * T * N))
    flat = imp.reshape(-1)
    top_idx = flat.topk(n_top).indices
    return frame_mask.reshape(-1)[top_idx].float().mean().item()
from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.video_io import IMAGENET_MEAN, IMAGENET_STD, load_video


def build_composites(videos, num_frames):
    """8 two-clip splices (cut at frame num_frames/2) + 4 four-clip splices."""
    comps = []
    n = len(videos)
    half, quarter = num_frames // 2, num_frames // 4
    for i in range(n // 2):
        a = load_video(str(videos[i]), num_frames=half, size=384)
        b = load_video(str(videos[i + n // 2]), num_frames=half, size=384)
        comps.append((f"pair{i}", torch.cat([a, b], dim=1)))
    for i in range(n // 4):
        parts = [load_video(str(videos[4 * i + j]), num_frames=quarter, size=384)
                 for j in range(4)]
        comps.append((f"quad{i}", torch.cat(parts, dim=1)))
    return comps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "e5_multiscene"))
    p.add_argument("--judge-scope", choices=["frame", "clip"], default="frame",
                   help="importance target scope: per-frame top 10%% (equal "
                        "targets/frame) or clip-global top 10%% (variable)")
    args = p.parse_args()

    device = torch.device(args.device)
    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    comps = build_composites(videos, args.num_frames)

    teacher = Siglip2Candidate(device)
    judges = [ClipCandidate(device), Dinov2Candidate(device)]
    jnames = [j.name for j in judges]

    sel_uni = Borissal(BorissalConfig.v0_3(scale=384))
    sel_glo = Borissal(BorissalConfig.v0_3(scale=384, per_frame_allocation="global"))
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    n_pf = 24 * 24

    alloc_names = ["uniform", "global", "oracle"]
    recalls = {a: {j: [] for j in jnames} for a in alloc_names}
    profiles = {a: [] for a in alloc_names}     # per-tubelet counts, for viz
    with torch.no_grad():
        for ci, (tag, video) in enumerate(comps):
            video01 = (video[0] * std[0] + mean[0]).clamp(0, 1)
            T_grid = args.num_frames // sel_uni.config.tubelet_size
            K_total = round(args.ratio * T_grid * n_pf)

            t_imp = teacher.importance(video01)
            counts = oracle_counts(t_imp, sel_uni.config.tubelet_size, K_total, n_pf)
            sels = {
                "uniform": sel_uni.select(video, gazing_ratio=args.ratio),
                "global": sel_glo.select(video, gazing_ratio=args.ratio),
                "oracle": sel_uni.select(video, gazing_ratio=args.ratio,
                                         per_frame_counts=counts),
            }
            j_imps = {j.name: j.importance(video01) for j in judges}
            rep = args.num_frames // T_grid
            for a, sel in sels.items():
                profiles[a].append(sel.per_frame_keep[0].tolist())
                fm = sel.keep_mask[0].reshape(T_grid, n_pf).repeat_interleave(rep, dim=0)
                jr = judge_recall if args.judge_scope == "frame" else judge_recall_clip_global
                for jn in jnames:
                    recalls[a][jn].append(jr(j_imps[jn], fm))
            print(f"composite {ci + 1}/{len(comps)} ({tag}) done", flush=True)

    out = {"args": vars(args), "composites": [t for t, _ in comps],
           "profiles": profiles, "matrix": {}}
    lines = ["| allocation \\ judge | " + " | ".join(jnames) + " |",
             "|" + "---|" * (len(jnames) + 1)]
    for a in alloc_names:
        row = [a]
        out["matrix"][a] = {}
        for jn in jnames:
            vals = recalls[a][jn]
            m = sum(vals) / len(vals)
            cell = f"{m:.4f}"
            if a != "uniform":
                diffs = [x - y for x, y in zip(vals, recalls["uniform"][jn])]
                w = sum(1 for x in diffs if x > 1e-9)
                l = sum(1 for x in diffs if x < -1e-9)
                cell += f" ({w}W-{l}L vs unif)"
            out["matrix"][a][jn] = {"mean": m, "per_clip": vals}
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.md").write_text("\n".join(lines) + "\n")
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(lines))
    print(f"saved {out_dir}/results.json and matrix.md")


if __name__ == "__main__":
    main()
