"""
Phase 2 (ablation): Mask-GRPO — policy gradient on binary gaze masks.

GRPO (Group Relative Policy Optimization) treats mask selection as a
stochastic policy π_θ(mask | video).  Reward = negative reconstruction
loss of the downstream VideoMAE decoder given the selected patches.

This is an *optional* ablation comparing GRPO to STE (phase2_ste.py).

Reference: DeepSeek-R1 / GRPO — Shao et al., 2024

Launch:
    torchrun --nproc_per_node=8 mamba_gaze/training/phase2_maskgrpo.py \
        --config mamba_gaze/configs/default.yaml \
        --resume checkpoints/phase1_epochXXXX.pt \
        --recon_model checkpoints/videomae_recon.pt
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import yaml
except ImportError:
    yaml = None


# ── reward (reconstruction quality) ──────────────────────────────────────────

def compute_reward(
    video: torch.Tensor,              # (B, T, 3, H, W)
    gazing_mask: list,                # per-scale (B, T, N_s)
    recon_model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Reward = negative mean reconstruction loss (higher = better).
    If no recon_model is provided, falls back to −mask_sparsity as proxy reward.
    """
    if recon_model is None:
        # Proxy: reward = negative fraction of selected tokens (encourage efficiency)
        total = sum(m.sum(dim=-1).mean() for m in gazing_mask)
        n_s   = sum(m.shape[-1] for m in gazing_mask)
        return -(total / n_s).detach()

    with torch.no_grad():
        recon_loss = recon_model(video, gazing_mask)   # scalar or (B,)
        if recon_loss.dim() > 0:
            recon_loss = recon_loss.mean()
    return -recon_loss   # negative: lower loss = higher reward


# ── GRPO-style policy gradient loss ──────────────────────────────────────────

def grpo_loss(
    log_probs: torch.Tensor,          # (B, T, N) log P(mask | video)
    masks: torch.Tensor,              # (B, T, N) sampled binary masks
    rewards: torch.Tensor,            # (B,) per-sample rewards
    kl_coef: float = 0.01,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    REINFORCE with group-relative baseline.

    rewards:  normalised within the group: R̂ = (R − mean(R)) / (std(R) + ε)
    kl_coef:  KL penalty to keep policy close to reference (phase-1 model)
    """
    if rewards.numel() > 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    advantage = rewards.detach().view(-1, 1, 1)   # (B, 1, 1)

    # log π(mask | video) = sum over selected tokens of log P = log_probs * mask
    #                      + sum over unselected of log(1 - P)
    log_pi = (log_probs * masks + (1 - masks) * torch.log1p(-log_probs.exp().clamp(max=1-1e-6))).sum(dim=(-1, -2))  # (B,)
    policy_loss = -(log_pi * advantage.squeeze()).mean()

    # Entropy bonus (encourage exploration)
    p    = log_probs.exp().clamp(1e-6, 1 - 1e-6)
    entropy = -(p * log_probs + (1 - p) * torch.log1p(-p)).mean()

    return policy_loss - kl_coef * entropy


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_mask(
    logits: torch.Tensor,             # (B, T, N)
    k: int,
) -> tuple:
    """Bernoulli sample from logits, forced to exactly k selections via Gumbel trick."""
    noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-8)))
    perturbed = logits + noise
    idx = perturbed.topk(k, dim=-1).indices
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, idx, 1.0)
    log_probs = F.logsigmoid(logits)
    return mask, log_probs


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    cfg: dict,
    resume_path: str,
    recon_model_path: Optional[str],
    local_rank: int = 0,
):
    distributed = dist.is_available() and dist.is_initialized()
    is_main = (not distributed) or (local_rank == 0)
    device  = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from ..models.mamba_gaze import MambaGaze
    model = MambaGaze.from_config(cfg).to(device)

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if is_main:
            print(f"Loaded checkpoint: {resume_path}")

    # Load optional reconstruction model
    recon_model = None
    if recon_model_path and Path(recon_model_path).exists():
        recon_model = torch.load(recon_model_path, map_location=device)
        recon_model.eval()
        if is_main:
            print(f"Loaded recon model: {recon_model_path}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    from ..data.autogaze_dataset import build_dataloader
    d_cfg = cfg.get("data", {})
    # GRPO uses smaller batch due to rollout overhead
    train_loader = build_dataloader(
        "train",
        batch_size=32,
        num_workers=d_cfg.get("num_workers", 8),
        distributed=distributed,
        num_frames=d_cfg.get("num_frames", 16),
        img_size=d_cfg.get("img_size", 224),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    s_cfg = cfg.get("selection", {})
    gazing_ratio = s_cfg.get("default_gazing_ratio", 0.5)
    epochs       = 20
    log_every    = cfg.get("logging", {}).get("log_every_n_steps", 50)
    global_step  = 0
    from ..data.mask_converter import SCALE_PATCHES

    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            video = batch["video"].to(device)
            B = video.shape[0]

            # Forward to get per-scale logits
            outputs = model({"video": video})
            per_scale_logits = outputs["per_scale_logits"]   # list[4] of (B, T, N_s)

            # Sample masks and compute log-probs
            all_masks, all_logprobs = [], []
            for logits, n_s in zip(per_scale_logits, SCALE_PATCHES):
                k = max(1, int(gazing_ratio * n_s))
                mask, lp = sample_mask(logits, k)
                all_masks.append(mask)
                all_logprobs.append(lp)

            # Reward from downstream recon model
            reward = compute_reward(video, all_masks, recon_model, device)
            if reward.dim() == 0:
                reward = reward.expand(B)

            # GRPO loss (concatenate all scales)
            all_lp_cat = torch.cat(all_logprobs, dim=-1)   # (B, T, 265)
            all_m_cat  = torch.cat(all_masks, dim=-1)
            loss = grpo_loss(all_lp_cat, all_m_cat, reward)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            epoch_loss  += loss.item()

            if is_main and global_step % log_every == 0:
                print(
                    f"[grpo] ep {epoch}/{epochs}  step {global_step}"
                    f"  loss={loss.item():.4f}"
                    f"  reward={reward.mean().item():.4f}"
                )

        if is_main:
            avg = epoch_loss / max(len(train_loader), 1)
            print(f"[grpo] epoch {epoch} done | avg_loss={avg:.4f} | {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--recon_model", default=None)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.init_process_group("nccl")

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(cfg, args.resume, args.recon_model, local_rank=args.local_rank)


if __name__ == "__main__":
    main()
