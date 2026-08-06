#!/usr/bin/env python
"""Model-health diagnostics for Borissal v1 (run before/after scale training).

Three checks, born from the 2026-07-15 architecture review (design.md
"Model diagnostics & global context"):

(a) v0-correlation -- Spearman between v1 scores and the v0 saliency map.
    ~0 for an untrained model (no input passthrough); if a TRAINED model
    pins near 1.0, it has collapsed onto reproducing the hand-crafted prior.
(b) Perturbation receptive field -- black out a corner region, measure the
    score response near the perturbation vs mid-grid vs the far corner.
    Local-only models show a ~200x near/far decay; the global-context path
    should shrink that gap once trained (note: it is ZERO-INIT, so an
    untrained model still measures as local -- expected).
(c) RGB-brightness init correlation -- a shallow random CNN tracks
    brightness at init (~-0.3 observed); flagged only if it grows extreme.

Usage:
    uv run python scripts/borissal_model_diagnostics.py                  # untrained defaults
    uv run python scripts/borissal_model_diagnostics.py \
        --checkpoint weights/<run>/checkpoint_last.pt                    # trained model

Prints one JSON line to stdout.
"""

import argparse
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

from autogaze.models.borissal import BorissalV1, BorissalV1Config  # noqa: E402
from autogaze.models.borissal.video_io import load_video  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = a.flatten().argsort().argsort().float()
    br = b.flatten().argsort().argsort().float()
    ac, bc = ar - ar.mean(), br - br.mean()
    return ((ac * bc).sum() / (ac.norm() * bc.norm() + 1e-9)).item()


def build_model(args) -> BorissalV1:
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        # pre-upgrade checkpoints predate these fields (same compat pattern
        # as borissal_dump_outputs.py)
        cfg_dict.setdefault("cosine_scores", False)
        cfg_dict.setdefault("global_context", False)
        model = BorissalV1(BorissalV1Config(**cfg_dict))
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = BorissalV1(BorissalV1Config(scale=args.scale))
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(REPO_ROOT / "assets" / "example_input.mp4"))
    p.add_argument("--checkpoint", default=None, help="v1 checkpoint .pt (omit = untrained defaults)")
    p.add_argument("--scale", type=int, default=384, help="ignored when --checkpoint provides a config")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = build_model(args)
    scale = model.config.scale
    patch = model.config.patch_size
    video = load_video(args.video, num_frames=args.num_frames, size=scale)

    out = {"checkpoint": args.checkpoint, "scale": scale,
           "global_context": model.config.global_context}

    with torch.no_grad():
        x, v0_score = model._grid_inputs(video)
        s = model.scores(video)

        # (a) v0 passthrough / collapse check
        out["v0_spearman"] = round(spearman(s, v0_score), 4)

        # (c) brightness correlation (pixels channels exist for pixels/both)
        if model.config.input_mode in ("pixels", "both"):
            luma = x[:, :, -3:].mean(dim=2)  # RGB channels are last
            out["brightness_spearman"] = round(spearman(s, luma), 4)

        # (b) perturbation receptive field: black out a 3x3-grid-cell corner
        Hg, Wg = s.shape[-2:]
        px = 3 * patch
        vid_p = video.clone()
        vid_p[:, :, :, :px, :px] = 0.0
        d = (model.scores(vid_p) - s).abs().mean(dim=(0, 1))  # (Hg, Wg)
        mid_h, mid_w = Hg // 2, Wg // 2
        out["perturb_near"] = round(d[:3, :3].mean().item(), 6)
        out["perturb_mid"] = round(d[mid_h - 1:mid_h + 1, mid_w - 1:mid_w + 1].mean().item(), 6)
        out["perturb_far"] = round(d[-3:, -3:].mean().item(), 6)
        far = max(out["perturb_far"], 1e-12)
        out["perturb_near_over_far"] = round(out["perturb_near"] / far, 1)

    print(json.dumps(out))


if __name__ == "__main__":
    main()
