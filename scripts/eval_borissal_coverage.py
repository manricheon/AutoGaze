#!/usr/bin/env python
"""Label-free quantitative gate for selector configurations.

Two complementary metrics from the same frozen V-JEPA2 + predictor:

- **coverage** (lower = better): MSE of reconstructing the UNSELECTED
  tokens from the SELECTED ones. CAUTION -- measured on the example clip,
  this metric FAVORS spatially uniform scatter (every region gets a nearby
  anchor for interpolation): random selection beat saliency selection at
  every ratio tried. Do not use it alone as a quality gate, and note the
  same bias applies to the v1 SSL training objective (see
  docs/borissal/training.md).
- **uniqueness** (higher = better): MSE of reconstructing the SELECTED
  tokens from the UNSELECTED rest -- how much information the selection
  holds that the rest cannot explain. Empirically discriminative in the
  right direction (v0.1 > v0.2 > random on the example clip).

Gate rule for v0.2 elements: an element may trade a little of one metric
for the other, but must not degrade BOTH at equal budget.

Examples:
    # v0.1 vs v0.2 preset on the example clip at two ratios
    uv run python scripts/eval_borissal_coverage.py \
        --video assets/example_input.mp4 --ratios 0.25 0.5 \
        --configs v0.1 v0.2

    # single-knob ablation against v0.1
    uv run python scripts/eval_borissal_coverage.py \
        --video assets/example_input.mp4 --ratios 0.25 \
        --configs v0.1 "v0.1,block_size=2" "v0.1,motion_noise_floor=quantile"

    # trained v1 checkpoint vs random and v0.2 (the post-training gate)
    uv run python scripts/eval_borissal_coverage.py \
        --video assets/example_input.mp4 --ratios 0.25 0.5 \
        --configs random v0.2 v1:weights/<run>/checkpoint_final.pt

    # measure the random coverage baseline for a hub 2.1 teacher (sets
    # --coverage-floor for the scale run; oracle-reference targets applied)
    uv run python scripts/eval_borissal_coverage.py \
        --video assets/example_input.mp4 --ratios 0.25 \
        --teacher hub:vjepa2_1_vit_large_384 --scale 384 --configs random
"""

import argparse
import json
from pathlib import Path

import torch

from autogaze.models.borissal import Borissal, BorissalConfig, resolve_device
from autogaze.models.borissal.video_io import load_video
from autogaze.models.borissal.vjepa2_sparse import VJEPA2Teacher
from autogaze.models.borissal.modeling_borissal import _pack_gazing_mask

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_config_spec(spec: str) -> BorissalConfig:
    """"v0.1" | "v0.2" optionally followed by ,key=value overrides."""
    parts = spec.split(",")
    base = parts[0].strip()
    overrides = {}
    for kv in parts[1:]:
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.replace(".", "", 1).replace("-", "", 1).isdigit():
            v = float(v) if "." in v else int(v)
        overrides[k] = v
    if base == "v0.2":
        return BorissalConfig.v0_2(**overrides)
    if base == "v0.1":
        return BorissalConfig(**overrides)
    raise ValueError(f"unknown base config: {base}")


def random_keep_mask(video: torch.Tensor, cfg: BorissalConfig, ratio: float, seed: int = 0):
    """Uniform-random per-tubelet exact-k selection -- the sanity anchor every
    saliency config must beat for the metric to mean anything."""
    B, T, C, H, W = video.shape
    T_grid = T // cfg.tubelet_size
    N_pf = (H // cfg.patch_size) * (W // cfg.patch_size)
    k = min(max(1, round(ratio * N_pf)), N_pf)
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(B, T_grid, N_pf, generator=g).to(video.device)
    _, idx = scores.topk(k, dim=-1)
    mask = torch.zeros(B, T_grid, N_pf, dtype=torch.bool, device=video.device)
    mask.scatter_(-1, idx, True)
    return mask.reshape(B, T_grid * N_pf)


def _predict_mse(teacher, video, ctx_idx, tgt_idx, L, hub: bool = False) -> float:
    with torch.no_grad():
        dense = teacher.dense_features(video)
        ctx = teacher.sparse_features(video, ctx_idx)
        pred = teacher.predict(ctx, ctx_idx, tgt_idx, num_tokens=L)
        if hub:
            # 2.1 predictors project into the distillation-teacher space, so
            # the target is an oracle-reference pass through the SAME head
            # with full context (training.md §5 -- same wiring as the trainer)
            all_idx = torch.arange(L, device=video.device).unsqueeze(0).expand(video.shape[0], -1)
            tgt = teacher.predict(dense, all_idx, tgt_idx, num_tokens=L)
        else:
            tgt = torch.gather(dense, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, dense.size(-1)))
        return torch.nn.functional.mse_loss(pred, tgt).item()


def score_selection(teacher, video: torch.Tensor, selector, ratio: float, cfg=None, hub: bool = False) -> dict:
    """Both metrics for one clip at one budget."""
    if selector == "random":
        keep_mask = random_keep_mask(video, cfg, ratio)
        keep_index, pad_k = _pack_gazing_mask(keep_mask)
        assert not pad_k.any()
    else:
        sel = selector.select(video, gazing_ratio=ratio)
        keep_mask = sel.keep_mask
        keep_index = sel.keep_index
        if keep_index.lt(0).any():
            raise ValueError("eval requires unpadded selection (uniform/global exact budgets)")
    L = keep_mask.shape[1]
    rest_idx, pad = _pack_gazing_mask(~keep_mask)
    assert not pad.any()

    return {
        # reconstruct the rest FROM the selection (lower better; scatter-biased)
        "coverage_mse": _predict_mse(teacher, video, keep_index, rest_idx, L, hub=hub),
        # reconstruct the selection FROM the rest (higher better)
        "uniqueness_mse": _predict_mse(teacher, video, rest_idx, keep_index, L, hub=hub),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", nargs="+", required=True, help="one or more clips")
    p.add_argument("--configs", nargs="+", required=True,
                   help='config specs, e.g. v0.1 v0.2 "v0.1,block_size=2"')
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25])
    p.add_argument("--teacher", default="facebook/vjepa2-vitl-fpc64-256",
                   help="HF id/path (crop must match --scale)")
    p.add_argument("--scale", type=int, default=256,
                   help="clip resolution fed to selector AND teacher (default matches HF vitl-256)")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="optional json output path")
    args = p.parse_args()

    device = resolve_device(args.device)
    # "hub:<entrypoint>" -> torch.hub V-JEPA 2.1 adapter with oracle-reference
    # targets; anything else -> HF (same convention as the trainer).
    hub = args.teacher.startswith("hub:")
    if hub:
        from autogaze.models.borissal.vjepa21_hub import VJEPA21HubTeacher
        teacher = VJEPA21HubTeacher.from_hub(args.teacher[len("hub:"):]).to(device)
    else:
        teacher = VJEPA2Teacher.from_pretrained(args.teacher).to(device)

    videos = {v: load_video(v, num_frames=args.num_frames, size=args.scale).to(device)
              for v in args.video}

    results = []
    header = f"{'config':40} {'ratio':6} {'coverage(<)':>12} {'uniqueness(>)':>14}"
    print(header)
    print("-" * len(header))
    for spec in args.configs:
        if spec == "random":
            cfg = BorissalConfig(scale=args.scale)
            selector = "random"
        elif spec.startswith("v1:"):
            # trained learned-selector checkpoint (same Selection contract)
            from autogaze.models.borissal import BorissalV1, BorissalV1Config
            ckpt = torch.load(spec[len("v1:"):], map_location="cpu", weights_only=False)
            ckpt_cfg = dict(ckpt["config"])
            ckpt_cfg.setdefault("cosine_scores", False)   # pre-upgrade checkpoints
            ckpt_cfg.setdefault("global_context", False)
            cfg = BorissalV1Config(**ckpt_cfg)
            selector = BorissalV1(cfg)
            selector.load_state_dict(ckpt["state_dict"])
            selector = selector.to(device).eval()
        else:
            cfg = parse_config_spec(spec)
            cfg.scale = args.scale
            selector = Borissal(cfg).to(device)
        for ratio in args.ratios:
            per_clip = {v: score_selection(teacher, vid, selector, ratio, cfg=cfg, hub=hub)
                        for v, vid in videos.items()}
            cov = sum(s["coverage_mse"] for s in per_clip.values()) / len(per_clip)
            uni = sum(s["uniqueness_mse"] for s in per_clip.values()) / len(per_clip)
            results.append({"config": spec, "ratio": ratio, "coverage_mse": cov,
                            "uniqueness_mse": uni, "per_clip": per_clip})
            print(f"{spec:40} {ratio:<6} {cov:12.5f} {uni:14.5f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
