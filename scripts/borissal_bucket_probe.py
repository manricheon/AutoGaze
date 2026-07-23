#!/usr/bin/env python
"""Content-adaptivity probe: does the selector need to allocate DIFFERENTLY by
(A) content length and (B) scene-transition density?

Extends the E5 negative result (oracle temporal allocation ~= uniform on short
single-scene clips; e5_teacher_review.py / e5_multiscene_review.py) into its two
un-retired axes, reusing that machinery verbatim:
  - candidates/judges (SigLIP2 as oracle source; CLIP-L + DINOv2 as foreign
    recall judges), `oracle_counts`, `judge_recall`  (e5_teacher_review)
  - hard-cut scene composites  (e5_multiscene.build_composites)

Buckets:
  LENGTH  short(16f raw) / mid(32f raw, redundancy-dilution regime) /
          longform_deadtime(32f = 16f active ++ 16f frozen last frame = the
          genuine dead-time case E5 never tested)
  SCENE   single(0 cuts) / two_scene(1 cut) / four_scene(3 cuts)

Allocators (all on FROZEN v0.3 patch scores, via `per_frame_counts`):
  uniform      baseline (K/T_grid per tubelet)
  oracle_sig   SigLIP2 MAP-attention-mass proportional -- the deployable-teacher
               CEILING (needs a teacher at inference)
  motion_prop  Borissal's OWN per-tubelet motion energy proportional -- the
               training-free adaptive rule we could actually ship

DECISIVE CELL: does oracle/motion_prop beat uniform ONLY in longform_deadtime
(and NOT in short/single)? If yes -> first crack in the E5 negative for real
dead-time. If ~uniform there too -> E5 negative extends to long-form; close it.

CAVEAT (hard): recall here is a PROXY that has repeatedly mis-ranked vs the real
caption->QA (v0.4 recall-neutral but downstream-worse; motion_weight inverted).
Every verdict below is a PROXY-LEVEL HYPOTHESIS to confirm on the CUDA QA run.

Usage:
    uv run python scripts/borissal_bucket_probe.py --limit-length 8
Outputs: outputs/borissal/bucket_probe/{results.json, VERDICT.md, overlays/}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from e5_teacher_review import (  # noqa: E402
    ClipCandidate, Dinov2Candidate, Siglip2Candidate, judge_recall, oracle_counts,
)
from e5_multiscene_review import build_composites  # noqa: E402
from autogaze.models.borissal import Borissal, BorissalConfig  # noqa: E402
from autogaze.models.borissal.modeling_borissal import _largest_remainder  # noqa: E402
from autogaze.models.borissal.video_io import IMAGENET_MEAN, IMAGENET_STD, load_video, unnormalize  # noqa: E402
from autogaze.models.borissal.viz import render_overlay  # noqa: E402

GRID = 24
N_PF = GRID * GRID
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)


# ----------------------------- bucket builders -----------------------------

def build_length_buckets(videos, limit):
    """Same source clips at 16f (short) and 32f (mid), plus a synthetic
    dead-time clip (16f active ++ 16f frozen last frame)."""
    short, mid, deadtime = [], [], []
    for i, path in enumerate(videos[:limit]):
        v16 = load_video(str(path), num_frames=16, size=384)      # (1,16,3,H,W)
        v32 = load_video(str(path), num_frames=32, size=384)
        active = v16                                              # 16 real frames
        tail = v16[:, -1:].expand(-1, 16, -1, -1, -1)            # 16 frozen frames
        dead = torch.cat([active, tail], dim=1)                  # (1,32,...)
        short.append((f"short{i}", v16))
        mid.append((f"mid{i}", v32))
        deadtime.append((f"dead{i}", dead))
    return {"short": short, "mid": mid, "longform_deadtime": deadtime}


def build_scene_buckets(videos):
    """single (raw 16f) + 2-clip/4-clip hard-cut composites (build_composites)."""
    single = [(f"single{i}", load_video(str(p), num_frames=16, size=384))
              for i, p in enumerate(videos)]
    comps = build_composites(videos, num_frames=16)              # pair* + quad*
    two = [(t, v) for t, v in comps if t.startswith("pair")]
    four = [(t, v) for t, v in comps if t.startswith("quad")]
    return {"single": single, "two_scene": two, "four_scene": four}


# ----------------------------- allocators -----------------------------

def motion_counts(video, tubelet_size, K_total, n_pf):
    """Borissal-native, training-free adaptive allocation: per-tubelet counts
    proportional to that tubelet's motion energy (mean abs temporal gradient of
    grayscale), largest-remainder rounded. Dead/static tubelets -> ~floor."""
    gray = (video[0] * STD[0] + MEAN[0]).clamp(0, 1).mean(dim=1)  # (T,H,W) in [0,1]
    T = gray.shape[0]
    diff = (gray[1:] - gray[:-1]).abs().flatten(1).mean(dim=1)    # (T-1,)
    energy_f = torch.cat([diff[:1], diff])                        # (T,) pad first
    energy_t = energy_f.view(T // tubelet_size, tubelet_size).mean(-1)
    frac = energy_t / energy_t.sum().clamp_min(1e-8)
    return _largest_remainder((frac * K_total).unsqueeze(0), K_total,
                              min_val=1, max_val=n_pf)[0]


def fix_total(counts, target, n_pf):
    """`_largest_remainder`'s final min-clamp can push the sum PAST the budget
    on skewed distributions (e.g. dead-time: 8 static tubelets floored to 1
    each). Trim the excess from the largest tubelets (add to the largest below
    max) so every allocator keeps EXACTLY `target` tokens -- otherwise an
    allocator that keeps more tokens wins recall trivially. Floors stay >=1."""
    c = counts.clone().long()
    over = int(c.sum()) - int(target)
    while over > 0:
        c[int(c.argmax())] -= 1
        over -= 1
    while over < 0:
        cand = (c < n_pf).nonzero().flatten()
        c[int(cand[c[cand].argmax()])] += 1
        over += 1
    return c


# ----------------------------- probe -----------------------------

def run_axis(axis_name, buckets, selector, teacher, judges, ratio, out_dir, save_overlays):
    jnames = [j.name for j in judges]
    alloc_names = ["uniform", "oracle_sig", "motion_prop"]
    # recalls[bucket][alloc][judge] = list per clip
    recalls = {b: {a: {j: [] for j in jnames} for a in alloc_names} for b in buckets}
    profiles = {b: {a: [] for a in alloc_names} for b in buckets}
    deadtail = {b: {a: [] for a in alloc_names} for b in buckets}  # frac budget in dead tail

    tub = selector.config.tubelet_size
    with torch.no_grad():
        for bucket, clips in buckets.items():
            for ci, (tag, video) in enumerate(clips):
                T = video.shape[1]
                T_grid = T // tub
                video01 = (video[0] * STD[0] + MEAN[0]).clamp(0, 1)   # (T,3,H,W)

                # uniform first -> its exact token total is the shared budget so
                # all three allocators keep the SAME number of tokens (fair recall)
                sel_uniform = selector.select(video, gazing_ratio=ratio)
                K_ref = int(sel_uniform.num_keep[0])
                t_imp = teacher.importance(video01)                   # (T,576)
                oc = fix_total(oracle_counts(t_imp, tub, K_ref, N_PF), K_ref, N_PF)
                mc = fix_total(motion_counts(video, tub, K_ref, N_PF), K_ref, N_PF)
                sels = {
                    "uniform": sel_uniform,
                    "oracle_sig": selector.select(video, gazing_ratio=ratio, per_frame_counts=oc),
                    "motion_prop": selector.select(video, gazing_ratio=ratio, per_frame_counts=mc),
                }
                assert int(sels["oracle_sig"].num_keep[0]) == K_ref
                assert int(sels["motion_prop"].num_keep[0]) == K_ref
                j_imps = {j.name: j.importance(video01) for j in judges}
                rep = T // T_grid
                for a, sel in sels.items():
                    pfk = sel.per_frame_keep[0]
                    profiles[bucket][a].append(pfk.tolist())
                    if bucket == "longform_deadtime":
                        # tail = second half tubelets (frozen frames)
                        tail_frac = pfk[T_grid // 2:].sum().item() / max(1, pfk.sum().item())
                        deadtail[bucket][a].append(tail_frac)
                    fm = sel.keep_mask[0].reshape(T_grid, N_PF).repeat_interleave(rep, dim=0)
                    for jn in jnames:
                        recalls[bucket][a][jn].append(judge_recall(j_imps[jn], fm))
                    if save_overlays and ci == 0:
                        _overlay(video, sels, bucket, tub, T_grid, out_dir)
                print(f"[{axis_name}] {bucket} {ci + 1}/{len(clips)} ({tag}) done", flush=True)
    return {"alloc_names": alloc_names, "judges": jnames, "recalls": recalls,
            "profiles": profiles, "deadtail": deadtail}


def _overlay(video, sels, bucket, tub, T_grid, out_dir):
    ov_dir = Path(out_dir) / "overlays"
    ov_dir.mkdir(parents=True, exist_ok=True)
    disp = unnormalize(video[0]).cpu()
    for a in ("uniform", "motion_prop"):
        km = sels[a].keep_mask[0].reshape(T_grid, GRID, GRID).cpu()
        render_overlay(disp, km, tub, str(ov_dir / f"{bucket}_{a}.png"))


def summarize(axis_res):
    """Build a markdown table + wins-vs-uniform per bucket/alloc/judge."""
    lines = []
    rec = axis_res["recalls"]
    for bucket, per_alloc in rec.items():
        lines.append(f"\n**{bucket}**\n")
        lines.append("| allocation | " + " | ".join(axis_res["judges"]) + " |")
        lines.append("|" + "---|" * (len(axis_res["judges"]) + 1))
        for a in axis_res["alloc_names"]:
            row = [a]
            for jn in axis_res["judges"]:
                vals = per_alloc[a][jn]
                m = sum(vals) / len(vals)
                cell = f"{m:.4f}"
                if a != "uniform":
                    base = per_alloc["uniform"][jn]
                    diffs = [x - y for x, y in zip(vals, base)]
                    w = sum(1 for d in diffs if d > 1e-9)
                    l = sum(1 for d in diffs if d < -1e-9)
                    cell += f" ({w}W-{l}L)"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length-dir", default=str(REPO_ROOT / "videos" / "internvid_pilot"))
    p.add_argument("--scene-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--limit-length", type=int, default=8)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--axes", nargs="+", default=["length", "scene"], choices=["length", "scene"])
    p.add_argument("--judges", nargs="+", default=["dinov2"], choices=["dinov2", "clip"],
                   help="foreign recall judges (SigLIP excluded -- it feeds the oracle). "
                        "clip is language-aligned but VERY slow on CPU (eager attention); "
                        "dinov2 is the cheap language-unaligned control.")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "bucket_probe"))
    args = p.parse_args()

    device = torch.device(args.device)
    selector = Borissal(BorissalConfig.v0_3(scale=384))
    teacher = Siglip2Candidate(device)
    _judge_ctor = {"dinov2": Dinov2Candidate, "clip": ClipCandidate}
    judges = [_judge_ctor[j](device) for j in args.judges]  # SigLIP excluded (feeds oracle)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"args": vars(args), "axes": {}}

    length_videos = sorted(Path(args.length_dir).glob("*.mp4"))
    scene_videos = sorted(Path(args.scene_dir).glob("*.mp4"))

    if "length" in args.axes:
        buckets = build_length_buckets(length_videos, args.limit_length)
        results["axes"]["length"] = run_axis("length", buckets, selector, teacher, judges,
                                              args.ratio, out_dir, save_overlays=True)
    if "scene" in args.axes:
        buckets = build_scene_buckets(scene_videos)
        results["axes"]["scene"] = run_axis("scene", buckets, selector, teacher, judges,
                                            args.ratio, out_dir, save_overlays=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # console + verdict tables
    md = ["# Content-adaptivity probe — proxy recall by bucket\n",
          "> PROXY-LEVEL: recall has mis-ranked vs caption->QA before. "
          "Every signal below is a HYPOTHESIS to confirm on the CUDA QA run.\n"]
    for axis, res in results["axes"].items():
        md.append(f"\n## Axis: {axis}")
        md.append(summarize(res))
        # decisive cell for length axis
        if axis == "length" and "longform_deadtime" in res["recalls"]:
            dt = res["deadtail"]["longform_deadtime"]
            md.append("\n**Dead-time tail budget fraction** (lower = more budget "
                      "moved off the frozen tail):")
            for a in res["alloc_names"]:
                vals = dt[a]
                if vals:
                    md.append(f"- {a}: {sum(vals)/len(vals):.3f}")
    (out_dir / "VERDICT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved {out_dir}/results.json, VERDICT.md, overlays/")


if __name__ == "__main__":
    main()
