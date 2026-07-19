#!/usr/bin/env python
"""E5 Phase 0 — teacher-candidate review for learned temporal budget allocation.

Three frozen VLM/SSL encoders each produce a per-frame patch-importance map;
each map is turned into an ORACLE per-tubelet allocation (fractions of the
clip budget), applied on top of v0.3's non-learned patch scores via the
`per_frame_counts` override, and the resulting selections are cross-judged
with the OTHER candidates' recall metrics (never self-judged). The candidate
whose oracle allocation helps foreign judges most is the best TEACHER for the
E5 distillation; the strongest remaining candidate becomes the new JUDGE.
If no oracle beats uniform on any foreign judge, E5 is dead on arrival
(training could at best reach the oracle ceiling) -- record and stop.

Candidates:
  siglip2  google/siglip2-base-patch16-384  MAP attention-pool head (24x24)
  clip     openai/clip-vit-large-patch14-336  last-layer CLS attention (24x24)
  dinov2   facebook/dinov2-base @336        last-layer CLS attention (24x24)
           (language-UNaligned control -- expected weaker for description)

Usage:
    uv run python scripts/e5_teacher_review.py --ratio 0.25 \
        --videos-dir videos/internvid_eval16
Outputs: outputs/borissal/e5_teacher_review/{results.json,matrix.md}
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from autogaze.models.borissal import Borissal, BorissalConfig
from autogaze.models.borissal.modeling_borissal import _largest_remainder
from autogaze.models.borissal.video_io import IMAGENET_MEAN, IMAGENET_STD, load_video

GRID = 24  # all three candidates are configured to a 24x24 patch grid


def _renorm(video01: torch.Tensor, mean, std) -> torch.Tensor:
    """video01: (T, 3, H, W) in [0,1] -> normalized for a given processor."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return (video01 - mean) / std


class Siglip2Candidate:
    name = "siglip2"

    def __init__(self, device):
        import eval_borissal_semantic as sem
        self.sem = sem
        self.encoder, self.processor = sem.build_encoder(device)

    def importance(self, video01: torch.Tensor) -> torch.Tensor:
        """video01 (T, 3, 384, 384) in [0,1] -> (T, 576) MAP-head attention."""
        frames = _renorm(video01, self.processor.image_mean, self.processor.image_std)
        tokens = self.encoder(pixel_values=frames).last_hidden_state
        _, attn = self.sem.probe_pool(self.encoder, tokens, need_weights=True)
        return attn  # (T, 576), rows sum to 1


class ClipCandidate:
    name = "clip"

    def __init__(self, device):
        from transformers import CLIPVisionModel, AutoProcessor
        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", attn_implementation="eager"
        ).eval().requires_grad_(False).to(device)
        proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.mean, self.std = proc.image_processor.image_mean, proc.image_processor.image_std

    def importance(self, video01: torch.Tensor) -> torch.Tensor:
        frames = torch.nn.functional.interpolate(
            video01, size=(336, 336), mode="bilinear", align_corners=False)
        frames = _renorm(frames, self.mean, self.std)
        out = self.model(pixel_values=frames, output_attentions=True)
        # last layer, CLS -> patch attention, mean over heads: (T, 1+576, 1+576)
        attn = out.attentions[-1].mean(dim=1)[:, 0, 1:]
        return attn / attn.sum(dim=-1, keepdim=True)


class Dinov2Candidate:
    name = "dinov2"

    def __init__(self, device):
        from transformers import Dinov2Model, AutoImageProcessor
        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-base", attn_implementation="eager"
        ).eval().requires_grad_(False).to(device)
        proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.mean, self.std = proc.image_mean, proc.image_std

    def importance(self, video01: torch.Tensor) -> torch.Tensor:
        frames = torch.nn.functional.interpolate(
            video01, size=(336, 336), mode="bilinear", align_corners=False)
        frames = _renorm(frames, self.mean, self.std)
        out = self.model(pixel_values=frames, output_attentions=True)
        attn = out.attentions[-1].mean(dim=1)[:, 0, 1:]  # CLS -> 24x24 patches
        return attn / attn.sum(dim=-1, keepdim=True)


def oracle_counts(imp: torch.Tensor, tubelet_size: int, K_total: int, n_pf: int,
                  top_frac: float = 0.1) -> torch.Tensor:
    """(T, N) importance -> (T_grid,) oracle counts: fractions proportional to
    per-tubelet top-`top_frac` attention mass, largest-remainder rounded."""
    T, N = imp.shape
    n_top = max(1, round(top_frac * N))
    mass_f = imp.topk(n_top, dim=-1).values.sum(dim=-1)         # (T,)
    mass_t = mass_f.view(T // tubelet_size, tubelet_size).sum(-1)
    frac = mass_t / mass_t.sum().clamp_min(1e-8)
    return _largest_remainder((frac * K_total).unsqueeze(0), K_total,
                              min_val=1, max_val=n_pf)[0]


def judge_recall(imp: torch.Tensor, frame_mask: torch.Tensor,
                 top_frac: float = 0.1) -> float:
    """Recall of a judge's top-`top_frac` importance patches (count-agnostic)."""
    T, N = imp.shape
    n_top = max(1, round(top_frac * N))
    top_idx = imp.topk(n_top, dim=-1).indices
    return frame_mask.gather(1, top_idx).float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=str(REPO_ROOT / "videos" / "internvid_eval16"))
    p.add_argument("--limit-videos", type=int, default=None)
    p.add_argument("--ratio", type=float, default=0.25)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "borissal" / "e5_teacher_review"))
    args = p.parse_args()

    device = torch.device(args.device)
    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    assert videos, f"no clips under {args.videos_dir}"

    candidates = [Siglip2Candidate(device), ClipCandidate(device), Dinov2Candidate(device)]
    names = [c.name for c in candidates]
    selector = Borissal(BorissalConfig.v0_3(scale=384))
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)

    n_pf = GRID * GRID
    # per clip: importance maps per candidate; selections for uniform + each oracle;
    # then judge every selection with every candidate's metric (self-judging
    # recorded but EXCLUDED from the verdict).
    alloc_names = ["uniform"] + [f"oracle_{n}" for n in names] + ["random"]
    recalls = {a: {j: [] for j in names} for a in alloc_names}
    kept_counts_example = {}

    with torch.no_grad():
        for vi, path in enumerate(videos):
            video = load_video(str(path), num_frames=args.num_frames, size=384)
            video01 = (video[0] * std[0] + mean[0]).clamp(0, 1)   # (T, 3, H, W)
            imps = {c.name: c.importance(video01) for c in candidates}

            T_grid = args.num_frames // selector.config.tubelet_size
            K_total = round(args.ratio * T_grid * n_pf)
            sels = {"uniform": selector.select(video, gazing_ratio=args.ratio)}
            for n in names:
                counts = oracle_counts(imps[n], selector.config.tubelet_size, K_total, n_pf)
                sels[f"oracle_{n}"] = selector.select(
                    video, gazing_ratio=args.ratio, per_frame_counts=counts)
                if vi == 0:
                    kept_counts_example[n] = counts.tolist()
            g = torch.Generator().manual_seed(vi)
            rand_scores = torch.rand(1, T_grid, n_pf, generator=g)
            _, ridx = rand_scores.topk(K_total // T_grid, dim=-1)
            rmask = torch.zeros(1, T_grid, n_pf, dtype=torch.bool)
            rmask.scatter_(-1, ridx, True)

            rep = args.num_frames // T_grid
            for a, sel in sels.items():
                fm = sel.keep_mask[0].reshape(T_grid, n_pf).repeat_interleave(rep, dim=0)
                for j in names:
                    recalls[a][j].append(judge_recall(imps[j], fm))
            fm_r = rmask[0].repeat_interleave(rep, dim=0)
            for j in names:
                recalls["random"][j].append(judge_recall(imps[j], fm_r))
            print(f"clip {vi + 1}/{len(videos)} done", flush=True)

    # verdict table: mean recall + paired wins vs uniform, self-judging flagged
    out = {"args": vars(args), "kept_counts_example": kept_counts_example, "matrix": {}}
    lines = ["| allocation \\ judge | " + " | ".join(names) + " |",
             "|" + "---|" * (len(names) + 1)]
    for a in alloc_names:
        row = [a]
        out["matrix"][a] = {}
        for j in names:
            vals = recalls[a][j]
            m = sum(vals) / len(vals)
            cell = f"{m:.4f}"
            if a.startswith("oracle_"):
                diffs = [x - y for x, y in zip(vals, recalls["uniform"][j])]
                w = sum(1 for x in diffs if x > 1e-9)
                l = sum(1 for x in diffs if x < -1e-9)
                cell += f" ({w}W-{l}L)"
                if a == f"oracle_{j}":
                    cell += " [self]"
            out["matrix"][a][j] = {"mean": m, "per_clip": vals}
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
