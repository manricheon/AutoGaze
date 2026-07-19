#!/usr/bin/env python
"""Cross-family reconstruction gate: Borissal selections scored by the
ORIGINAL AutoGaze VideoMAE (the same model family that produced AutoGaze's
own RL reward — an evaluation axis INDEPENDENT of our V-JEPA training
teacher, which mitigates the circular-validation concern).

For each selector, the selected patches (mapped to the checkpoint's
multi-scale per-frame token layout via adapters.to_videomae_gazing_info —
this doubles as the first real consumer of the canonical keep-index
contract) are fed to VideoMAE, which reconstructs the chosen frames; we
report the per-frame reconstruction loss (lower = the selection preserves
more reconstructable content) and save original-vs-reconstruction strips.

CAUTION for interpretation: this is a RECONSTRUCTION-family metric, so it
shares coverage's documented scatter bias (spread selections reconstruct
backgrounds well). Use it as a cross-family comparison axis — the same
axis AutoGaze's own reward used — not as a sole adoption gate.

Setup notes (found empirically, recorded in design.md):
- videomae.pt keys carry a DDP "module." prefix (stripped here);
- loss is evaluated with loss_type=l1 only (the checkpoint's training
  config also had dinov2+siglip2 heads; those need extra models and
  flash-attn — irrelevant for comparing selections);
- transformers==5.5.0 removed find_pruneable_heads_and_indices; a local
  shim is installed before importing the legacy module (never called at
  inference — pruning-only API).

Usage:
    uv run python scripts/eval_borissal_videomae_recon.py \
        --video assets/example_input.mp4 --ratios 0.25 \
        --selectors random v0.2 v1:weights/<run>/checkpoint_final.pt
"""

import argparse
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_transformers_shim():
    """transformers 5.x dropped two pruning helpers the vendored legacy
    model imports (but never calls at inference). Shim them in WITHOUT
    touching the legacy files."""
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(pu, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim=0):
            raise NotImplementedError("pruning is not supported under the 5.5.0 shim")
        pu.prune_linear_layer = prune_linear_layer


def build_task(videomae_ckpt: str, device: torch.device):
    _install_transformers_shim()
    from omegaconf import OmegaConf
    from autogaze.tasks.video_mae_reconstruction.task_video_mae_reconstruction import (
        VideoMAEReconstruction,
    )
    cfg = OmegaConf.create({
        "scale_embed": True, "max_num_frames": 256, "time_embed": True, "causal": True,
        "loss_type": "l1", "loss_weights": "1", "l1_loss_config": None,
    })
    task = VideoMAEReconstruction(
        recon_model="facebook/vit-mae-large", recon_model_config=cfg,
        scales="32+64+112+224", recon_sample_rate=0.125, attn_mode="sdpa",
    )
    sd = torch.load(videomae_ckpt, map_location="cpu", weights_only=False)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, _ = task.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"videomae checkpoint left {len(missing)} params uninitialized: {missing[:5]}")
    return task.to(device).eval()


def build_selection(spec: str, video: torch.Tensor, ratio: float, spread: float = 0.0):
    """spec: random | v0.1 | v0.2 | v1:<checkpoint>. Returns a Selection."""
    from autogaze.models.borissal import Borissal, BorissalConfig, BorissalV1, BorissalV1Config
    scale = video.shape[-1]
    if spec == "random":
        # random per-tubelet exact-k: score with pure noise, reuse the
        # shared selection tail (exact budget + canonical packing)
        from autogaze.models.borissal.modeling_borissal_v1 import _selection_from_scores
        cfg = BorissalConfig(scale=scale)
        g = torch.Generator().manual_seed(0)
        S = torch.rand(video.shape[0], video.shape[1] // cfg.tubelet_size,
                       scale // cfg.patch_size, scale // cfg.patch_size, generator=g)
        return _selection_from_scores(S, ratio, "uniform", cfg.eps)
    if spec.startswith("v1:"):
        ckpt = torch.load(spec[len("v1:"):], map_location="cpu", weights_only=False)
        ckpt_cfg = dict(ckpt["config"])
        ckpt_cfg.setdefault("cosine_scores", False)
        ckpt_cfg.setdefault("global_context", False)
        model = BorissalV1(BorissalV1Config(**ckpt_cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval().select(video, gazing_ratio=ratio, per_frame_allocation="uniform",
                                   spread_fraction=spread)
    if spec == "v0.2":
        return Borissal(BorissalConfig.v0_2(scale=scale, per_frame_allocation="uniform",
                                            block_size=1)).select(video, gazing_ratio=ratio,
                                                                  spread_fraction=spread)
    if spec == "v0.3":
        return Borissal(BorissalConfig.v0_3(scale=scale, per_frame_allocation="uniform",
                                            block_size=1)).select(video, gazing_ratio=ratio,
                                                                  spread_fraction=spread)
    if spec == "v0.1":
        return Borissal(BorissalConfig(scale=scale)).select(video, gazing_ratio=ratio,
                                                            spread_fraction=spread)
    raise ValueError(f"unknown selector spec: {spec}")


def save_strip(orig, recon, frame_idx, selection, patch_size, tubelet_size, mean, std, path):
    """Three rows per reconstructed frame: input / selection-mask overlay /
    reconstruction."""
    import matplotlib.pyplot as plt
    n = len(frame_idx)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6.9))
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    T_grid, H_grid, W_grid = (int(x) for x in selection.grid_thw[0].tolist())
    mask_grid = selection.keep_mask[0].reshape(T_grid, H_grid, W_grid).float().cpu()

    def denorm(img):
        return (img * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    for i, f in enumerate(frame_idx):
        f = int(f)
        img = orig[0, f].cpu()
        up = mask_grid[f // tubelet_size].repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
        overlay = img * (0.25 + 0.75 * up.unsqueeze(0))
        for row, (label, im) in enumerate([("input", img), ("selected", overlay),
                                           ("recon", recon[0, i].float().cpu())]):
            ax = axes[row, i]
            ax.imshow(denorm(im))
            ax.set_axis_off()
            ax.set_title(f"{label} f{f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(REPO_ROOT / "assets" / "example_input.mp4"))
    p.add_argument("--videomae-ckpt", default=str(REPO_ROOT / "weights" / "VideoMAE_AutoGaze" / "videomae.pt"))
    p.add_argument("--selectors", nargs="+", default=["random", "v0.2"],
                   help="random | v0.1 | v0.2 | v0.3 | v1:<checkpoint.pt>")
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25])
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--spread", type=float, default=0.0,
                   help="spread_fraction for hybrid allocation (non-random selectors)")
    p.add_argument("--recon-frames", default="1,3,5,7,9,11,13,15",
                   help="comma-separated frame indices to reconstruct (fixed for comparability)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "borissal" / "videomae_recon"))
    args = p.parse_args()

    device = torch.device(args.device)
    task = build_task(args.videomae_ckpt, device)

    # video at the checkpoint's finest scale, renormalized to ITS stats
    from autogaze.models.borissal.video_io import load_video, IMAGENET_MEAN, IMAGENET_STD
    from autogaze.models.borissal.adapters import to_videomae_gazing_info
    scale = task.scales[-1]
    video = load_video(args.video, num_frames=args.num_frames, size=scale)
    ours_mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    ours_std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    task_mean = torch.tensor(task.transform.image_mean).view(1, 1, 3, 1, 1)
    task_std = torch.tensor(task.transform.image_std).view(1, 1, 3, 1, 1)
    video_task = ((video * ours_std + ours_mean) - task_mean) / task_std

    frame_idx = torch.tensor([int(x) for x in args.recon_frames.split(",")], device=device)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    header = f"{'selector':44} {'ratio':6} {'recon_l1(<)':>12}  per-frame"
    print(header)
    print("-" * len(header))
    for spec in args.selectors:
        for ratio in args.ratios:
            sel = build_selection(spec, video, ratio,
                                  spread=0.0 if spec == "random" else args.spread)
            gaze = to_videomae_gazing_info(sel, tubelet_size=2, scales=tuple(task.scales))
            with torch.no_grad():
                out = task.forward_output({"video_for_task": video_task.to(device)},
                                          gaze, frame_idx_to_reconstruct=frame_idx)
            per_frame = out["reconstruction_loss_each_reconstruction_frame"][0].float().tolist()
            mean_loss = sum(per_frame) / len(per_frame)
            tag = spec.replace(":", "_").replace("/", "_")
            png = out_root / f"{tag}_r{int(ratio * 100)}.png"
            save_strip(video_task, out["reconstruction"], frame_idx.tolist(),
                       sel, patch_size=16, tubelet_size=2,
                       mean=task.transform.image_mean, std=task.transform.image_std, path=png)
            results.append({"selector": spec, "ratio": ratio, "recon_l1_mean": mean_loss,
                            "recon_l1_per_frame": per_frame, "strip": str(png)})
            print(f"{spec:44} {ratio:<6} {mean_loss:12.5f}  "
                  + " ".join(f"{v:.3f}" for v in per_frame))

    with open(out_root / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_root}/results.json (+ per-selector strips)")


if __name__ == "__main__":
    main()
