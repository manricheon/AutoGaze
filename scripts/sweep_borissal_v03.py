#!/usr/bin/env python
"""Borissal v0.3 candidate sweep runner (stages 1-2, docs/borissal/v03-design.md §5).

Runs the gate battery for a base config plus candidate knobs and dumps a
results table (JSON + markdown). It NEVER decides adoption -- read the table,
record verdicts in design.md (v0.2 discipline). Conventions:

- semantic gate: UNIFORM-allocation variant of each config (the MAP-head
  recall metric needs equal per-frame keep counts), scale 384 -- same as the
  measurement convention in design.md.
- coverage/uniqueness gate: full preset (global allocation), scale 256
  (HF vitl-256 teacher), rule "must not degrade BOTH".
- latency: CPU median over 20 iters at (1, 16, 3, 384, 384).

Usage:
  # stage 1 (solo screening) on the held-out set
  uv run python scripts/sweep_borissal_v03.py --stage solo \
      --videos-dir videos/internvid_eval16 --ratio 0.25
  # stage 2 (greedy): base + adopted knobs + each remaining candidate
  uv run python scripts/sweep_borissal_v03.py --stage greedy \
      --adopted motion_center_surround coherence_gate --ratio 0.25
  # quick smoke (2 clips, no teacher download)
  uv run python scripts/sweep_borissal_v03.py --stage solo \
      --include motion_center_surround --limit-videos 2 --skip-coverage
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.video_io import IMAGENET_MEAN, IMAGENET_STD, load_video

CANDIDATES = {
    "motion_center_surround": {"motion_center_surround": True},
    "coherence_gate": {"coherence_gate": True},
    "signature": {"signature_weight": 0.5},
    "color_rarity": {"color_rarity_weight": 0.5},
    "dog_blob": {"dog_blob_weight": 0.5},
    "fusion_peak": {"fusion_norm": "peak"},
    "fusion_entropy": {"fusion_norm": "entropy"},
    "score_ema": {"score_ema_alpha": 0.5},
    "hysteresis": {"select_hysteresis_eps": 0.05},
}


def build_cfg(overrides: dict, scale: int, uniform: bool) -> BorissalConfig:
    kw = dict(overrides)
    if uniform:
        kw.setdefault("per_frame_allocation", "uniform")
        kw.setdefault("block_size", 1)
    return BorissalConfig.v0_2(scale=scale, **kw)


def semantic_scores(encoder, processor, video_paths, overrides, ratio,
                    num_frames, device):
    import eval_borissal_semantic as sem
    scale = encoder.config.image_size
    grid = scale // encoder.config.patch_size
    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    enc_mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
    enc_std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
    selector = Borissal(build_cfg(overrides, scale, uniform=True))
    gists, recalls = [], []
    for path in video_paths:
        video = load_video(str(path), num_frames=num_frames, size=scale)
        frames = (((video * ours_std + ours_mean) - enc_mean) / enc_std)[0].to(device)
        with torch.no_grad():
            tokens = encoder(pixel_values=frames).last_hidden_state
        sel = selector.select(video, gazing_ratio=ratio)
        T_grid = int(sel.grid_thw[0, 0].item())
        mask_tub = sel.keep_mask[0].reshape(T_grid, grid * grid)
        frame_mask = mask_tub.repeat_interleave(num_frames // T_grid, dim=0).to(device)
        gist, recall = sem.semantic_metrics(encoder, tokens, frame_mask)
        gists.append(gist)
        recalls.append(recall)
    return gists, recalls


def coverage_scores(teacher, video_paths, overrides, ratio, num_frames,
                    cov_scale, device):
    from eval_borissal_coverage import score_selection
    selector = Borissal(build_cfg(overrides, cov_scale, uniform=False)).to(device)
    covs, uniqs = [], []
    for path in video_paths:
        video = load_video(str(path), num_frames=num_frames, size=cov_scale).to(device)
        s = score_selection(teacher, video, selector, ratio)
        covs.append(s["coverage_mse"])
        uniqs.append(s["uniqueness_mse"])
    return covs, uniqs


def latency_ms(overrides: dict, scale: int = 384, iters: int = 20) -> float:
    g = torch.Generator().manual_seed(0)
    video = torch.rand(1, 16, 3, scale, scale, generator=g)
    model = Borissal(build_cfg(overrides, scale, uniform=False))
    for _ in range(5):
        model.select(video, gazing_ratio=0.25)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.select(video, gazing_ratio=0.25)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["solo", "greedy"], required=True)
    p.add_argument("--adopted", nargs="*", default=[],
                   help="greedy: candidate names already adopted into the base")
    p.add_argument("--include", nargs="*", default=None,
                   help="subset of candidate names to evaluate")
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--skip-coverage", action="store_true")
    p.add_argument("--cov-scale", type=int, default=256)
    p.add_argument("--teacher", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "v03_sweep"))
    args = p.parse_args()

    device = torch.device(args.device)
    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    assert videos, f"no .mp4 under {args.videos_dir}"
    unknown = [n for n in args.adopted + (args.include or []) if n not in CANDIDATES]
    assert not unknown, f"unknown candidates: {unknown} (choose from {sorted(CANDIDATES)})"

    adopted_over = {}
    for n in args.adopted:
        adopted_over.update(CANDIDATES[n])
    names = args.include or [n for n in CANDIDATES if n not in args.adopted]
    base_name = "base+" + "+".join(args.adopted) if args.adopted else "v0.2-base"
    rows = [(base_name, dict(adopted_over))]
    for n in names:
        over = dict(adopted_over)
        over.update(CANDIDATES[n])
        rows.append((n, over))

    import eval_borissal_semantic as sem
    encoder, processor = sem.build_encoder(device)
    teacher = None
    if not args.skip_coverage:
        from autogaze.models.borissal.vjepa2_sparse import VJEPA2Teacher
        teacher = VJEPA2Teacher.from_pretrained(args.teacher).to(device)

    results = []
    for name, over in rows:
        gists, recalls = semantic_scores(encoder, processor, videos, over,
                                         args.ratio, args.num_frames, device)
        row = {
            "name": name, "overrides": over,
            "recall_mean": sum(recalls) / len(recalls),
            "gist_mean": sum(gists) / len(gists),
            "recall_per_clip": recalls,       # 동률 판정용 클립별 짝 비교 데이터
            "latency_ms": latency_ms(over),
        }
        if teacher is not None:
            covs, uniqs = coverage_scores(teacher, videos, over, args.ratio,
                                          args.num_frames, args.cov_scale, device)
            row["coverage_mean"] = sum(covs) / len(covs)
            row["uniqueness_mean"] = sum(uniqs) / len(uniqs)
        results.append(row)
        extra = (f"  cov {row['coverage_mean']:.3f}  uniq {row['uniqueness_mean']:.3f}"
                 if teacher is not None else "")
        print(f"{name:26} recall {row['recall_mean']:.3f}  gist {row['gist_mean']:.3f}"
              f"  lat {row['latency_ms']:.1f}ms{extra}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.stage + ("_" + "_".join(args.adopted) if args.adopted else "")
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump({"args": {k: v for k, v in vars(args).items()},
                   "results": results}, f, indent=2)
    md = ["| config | recall(>) | gist | cov(<) | uniq(>) | lat ms |",
          "|---|---|---|---|---|---|"]
    for r in results:
        md.append(
            f"| {r['name']} | {r['recall_mean']:.3f} | {r['gist_mean']:.3f} "
            f"| {r.get('coverage_mean', float('nan')):.3f} "
            f"| {r.get('uniqueness_mean', float('nan')):.3f} "
            f"| {r['latency_ms']:.1f} |")
    (out_dir / f"{tag}.md").write_text("\n".join(md) + "\n")
    print(f"saved {out_dir / (tag + '.json')} and {out_dir / (tag + '.md')}")


if __name__ == "__main__":
    main()
