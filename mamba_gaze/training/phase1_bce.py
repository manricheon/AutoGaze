"""
Phase 1: Focal-BCE distillation + L2 reconstruction-loss supervision.

Loss:
    L = focal_bce(logits, target_multihot, α=0.25, γ=2.0)
      + 0.1 × mse(pred_recon_loss, target_recon_loss)

Optimizer: AdamW, lr=5e-4, weight_decay=0.05
Scheduler: cosine with warmup
Batch size: 256  (per GPU; scale with DDP)

Launch (single node, 8 GPUs):
    torchrun --nproc_per_node=8 mamba_gaze/training/phase1_bce.py \
        --config mamba_gaze/configs/default.yaml
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
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    yaml = None


# ── loss ─────────────────────────────────────────────────────────────────────

def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy.

    logits:  (B, T, N) — raw logits (pre-sigmoid)
    targets: (B, T, N) — binary float [0, 1]
    """
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt    = torch.where(targets == 1, probs, 1 - probs)
    alpha_t = torch.where(targets == 1,
                          torch.full_like(targets, alpha),
                          torch.full_like(targets, 1 - alpha))
    focal = alpha_t * (1 - pt) ** gamma * bce
    return focal.mean()


def compute_loss(
    outputs: dict,
    gazing_multihot: torch.Tensor,   # (B, T, 265)
    recon_loss_gt: torch.Tensor,     # (B, T)
    recon_loss_weight: float = 0.1,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict:
    from ..data.mask_converter import SCALE_PATCHES, SCALE_OFFSETS

    # Concatenate per-scale logits → (B, T, 265)
    per_scale_logits = outputs["per_scale_logits"]
    all_logits = torch.cat(per_scale_logits, dim=-1)

    sel_loss  = focal_bce(all_logits, gazing_multihot, focal_alpha, focal_gamma)
    pred_recon = outputs["pred_recon_loss"]
    recon_loss = F.mse_loss(pred_recon, recon_loss_gt)

    total = sel_loss + recon_loss_weight * recon_loss
    return {"total": total, "selection": sel_loss, "recon": recon_loss}


# ── LR schedule ──────────────────────────────────────────────────────────────

def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# ── Gumbel temperature annealing ─────────────────────────────────────────────

def anneal_temperature(
    epoch: int,
    temp_init: float,
    temp_final: float,
    anneal_epochs: int,
) -> float:
    t = min(epoch / max(anneal_epochs, 1), 1.0)
    return temp_final + (temp_init - temp_final) * (1 - t)


# ── checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, cfg, is_main: bool):
    if not is_main:
        return
    save_dir = Path(cfg.get("checkpoint", {}).get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"phase1_epoch{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }, path)
    # Prune old checkpoints
    keep = cfg.get("checkpoint", {}).get("keep_last_n", 5)
    ckpts = sorted(save_dir.glob("phase1_epoch*.pt"))
    for old in ckpts[:-keep]:
        old.unlink(missing_ok=True)


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, local_rank: int = 0):
    distributed = dist.is_available() and dist.is_initialized()
    is_main = (not distributed) or (local_rank == 0)
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── model ─────────────────────────────────────────────────────────────────
    from ..models.mamba_gaze import MambaGaze
    model = MambaGaze.from_config(cfg).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── data ──────────────────────────────────────────────────────────────────
    from ..data.autogaze_dataset import build_dataloader
    d_cfg = cfg.get("data", {})
    t_cfg = cfg.get("training", {}).get("phase1", {})

    train_loader = build_dataloader(
        "train",
        batch_size=t_cfg.get("batch_size", 256),
        num_workers=d_cfg.get("num_workers", 8),
        pin_memory=d_cfg.get("pin_memory", True),
        distributed=distributed,
        num_frames=d_cfg.get("num_frames", 16),
        img_size=d_cfg.get("img_size", 224),
    )

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.get("lr", 5e-4),
        weight_decay=t_cfg.get("weight_decay", 0.05),
    )
    epochs        = t_cfg.get("epochs", 150)
    warmup_epochs = t_cfg.get("warmup_epochs", 10)
    steps_per_ep  = len(train_loader)
    total_steps   = epochs * steps_per_ep
    warmup_steps  = warmup_epochs * steps_per_ep

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, warmup_steps, total_steps),
    )

    s_cfg       = cfg.get("selection", {})
    temp_init   = s_cfg.get("gumbel_temp_init", 1.0)
    temp_final  = s_cfg.get("gumbel_temp_final", 0.1)
    temp_anneal = s_cfg.get("gumbel_anneal_epochs", 100)
    log_every   = cfg.get("logging", {}).get("log_every_n_steps", 50)
    save_every  = cfg.get("checkpoint", {}).get("save_every_n_epochs", 10)
    grad_clip   = t_cfg.get("grad_clip", 1.0)
    rl_w        = t_cfg.get("recon_loss_weight", 0.1)
    fa          = t_cfg.get("focal_alpha", 0.25)
    fg          = t_cfg.get("focal_gamma", 2.0)

    global_step = 0
    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        # Anneal Gumbel temperature
        temp = anneal_temperature(epoch, temp_init, temp_final, temp_anneal)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.set_gumbel_temperature(temp)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            video         = batch["video"].to(device)           # (B, T, 3, H, W)
            gazing_mh     = batch["gazing_multihot"].to(device) # (B, T, 265)
            recon_loss_gt = batch["recon_loss"].to(device)      # (B, T)

            outputs = model({"video": video})
            losses  = compute_loss(outputs, gazing_mh, recon_loss_gt, rl_w, fa, fg)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += losses["total"].item()
            if is_main and global_step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[phase1] ep {epoch}/{epochs}  step {global_step}"
                    f"  loss={losses['total']:.4f}"
                    f"  sel={losses['selection']:.4f}"
                    f"  recon={losses['recon']:.4f}"
                    f"  lr={lr_now:.2e}  τ={temp:.3f}"
                )

        avg = epoch_loss / max(steps_per_ep, 1)
        if is_main:
            elapsed = time.time() - t0
            print(f"[phase1] epoch {epoch} done | avg_loss={avg:.4f} | {elapsed:.1f}s")

        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg, cfg, is_main)

    save_checkpoint(model, optimizer, scheduler, epochs, avg, cfg, is_main)
    return model


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(cfg, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
