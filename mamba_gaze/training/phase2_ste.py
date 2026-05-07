"""
Phase 2: Straight-Through Estimator fine-tuning.

Loads a Phase-1 checkpoint and continues training with:
  - Hard top-k selection during forward pass
  - STE (straight-through estimator) gradient through the threshold
  - Low fixed temperature (0.1) so soft mask ≈ binary
  - Smaller lr=1e-4, fewer epochs=30

The STE trick: we do forward with hard selection (no gradient), but we
copy gradients from the hard mask back to the logits as if the selection
were differentiable (equivalent to identity in the backward pass).

Launch:
    torchrun --nproc_per_node=8 mamba_gaze/training/phase2_ste.py \
        --config mamba_gaze/configs/default.yaml \
        --resume checkpoints/phase1_epochXXXX.pt
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import yaml
except ImportError:
    yaml = None

from .phase1_bce import compute_loss, cosine_with_warmup, save_checkpoint


# ── STE selection override ────────────────────────────────────────────────────

class STETopK(torch.autograd.Function):
    """Hard top-k in forward; identity (pass-through) gradient in backward."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, k: int):
        mask = torch.zeros_like(logits)
        idx  = logits.topk(k, dim=-1).indices
        mask.scatter_(-1, idx, 1.0)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None   # straight-through: pass gradient unchanged


def ste_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    return STETopK.apply(logits, k)


def patch_model_for_ste(model: nn.Module, gazing_ratio: float) -> None:
    """Replace the selection head's forward to use STE top-k."""
    from ..models.selection_head import SCALE_PATCHES

    head = model.selection_head if not isinstance(model, DDP) else model.module.selection_head

    _orig_forward = head.forward

    def ste_forward(H, gazing_ratio_=gazing_ratio, temperature=None, frame_budget_mask=None):
        B, T, h, w, d = H.shape
        x = H.permute(0, 1, 4, 2, 3).reshape(B * T, d, h, w)

        per_scale_logits, per_scale_masks = [], []
        for sh, n_s in zip(head.scale_heads, SCALE_PATCHES):
            logits_flat = sh(x).reshape(B, T, n_s)
            per_scale_logits.append(logits_flat)

            k    = max(1, int(gazing_ratio_ * n_s))
            mask = ste_topk(logits_flat, k)

            if frame_budget_mask is not None:
                mask = mask * frame_budget_mask.unsqueeze(-1)
            per_scale_masks.append(mask)

        return per_scale_logits, per_scale_masks

    head.forward = ste_forward


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str, local_rank: int = 0):
    distributed = dist.is_available() and dist.is_initialized()
    is_main = (not distributed) or (local_rank == 0)
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from ..models.mamba_gaze import MambaGaze
    model = MambaGaze.from_config(cfg).to(device)

    # Load phase-1 weights
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if is_main:
            print(f"Loaded phase-1 checkpoint: {resume_path}")
    elif is_main:
        print("Warning: no phase-1 checkpoint found; starting from scratch.")

    t_cfg = cfg.get("training", {}).get("phase2_ste", {})
    gazing_ratio = cfg.get("selection", {}).get("default_gazing_ratio", 0.5)
    ste_temp = t_cfg.get("ste_temperature", 0.1)

    # Override selection with STE
    patch_model_for_ste(model, gazing_ratio)
    model.set_gumbel_temperature(ste_temp)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    from ..data.autogaze_dataset import build_dataloader
    d_cfg = cfg.get("data", {})
    train_loader = build_dataloader(
        "train",
        batch_size=t_cfg.get("batch_size", 128),
        num_workers=d_cfg.get("num_workers", 8),
        pin_memory=d_cfg.get("pin_memory", True),
        distributed=distributed,
        num_frames=d_cfg.get("num_frames", 16),
        img_size=d_cfg.get("img_size", 224),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 0.01),
    )
    epochs       = t_cfg.get("epochs", 30)
    warmup_epochs= t_cfg.get("warmup_epochs", 3)
    steps_per_ep = len(train_loader)
    total_steps  = epochs * steps_per_ep
    warmup_steps = warmup_epochs * steps_per_ep

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, warmup_steps, total_steps),
    )

    log_every   = cfg.get("logging", {}).get("log_every_n_steps", 50)
    save_every  = cfg.get("checkpoint", {}).get("save_every_n_epochs", 5)
    grad_clip   = t_cfg.get("grad_clip", 1.0)
    rl_w        = cfg.get("training", {}).get("phase1", {}).get("recon_loss_weight", 0.1)
    fa          = cfg.get("training", {}).get("phase1", {}).get("focal_alpha", 0.25)
    fg          = cfg.get("training", {}).get("phase1", {}).get("focal_gamma", 2.0)

    global_step = 0
    avg = 0.0
    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            video         = batch["video"].to(device)
            gazing_mh     = batch["gazing_multihot"].to(device)
            recon_loss_gt = batch["recon_loss"].to(device)

            outputs = model({"video": video})
            losses  = compute_loss(outputs, gazing_mh, recon_loss_gt, rl_w, fa, fg)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss  += losses["total"].item()

            if is_main and global_step % log_every == 0:
                print(
                    f"[phase2-ste] ep {epoch}/{epochs}  step {global_step}"
                    f"  loss={losses['total']:.4f}"
                    f"  sel={losses['selection']:.4f}"
                    f"  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg = epoch_loss / max(steps_per_ep, 1)
        if is_main:
            print(f"[phase2-ste] epoch {epoch} done | avg_loss={avg:.4f} | {time.time()-t0:.1f}s")

        if epoch % save_every == 0:
            # reuse save_checkpoint but with "phase2ste" prefix by monkeypatching save dir
            save_checkpoint(model, optimizer, scheduler, epoch, avg,
                            {**cfg, "checkpoint": {
                                **cfg.get("checkpoint", {}),
                                "save_dir": str(Path(cfg.get("checkpoint", {}).get("save_dir", "checkpoints")) / "phase2_ste"),
                            }}, is_main)

    save_checkpoint(model, optimizer, scheduler, epochs, avg,
                    {**cfg, "checkpoint": {
                        **cfg.get("checkpoint", {}),
                        "save_dir": str(Path(cfg.get("checkpoint", {}).get("save_dir", "checkpoints")) / "phase2_ste"),
                    }}, is_main)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Phase-1 checkpoint path")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
