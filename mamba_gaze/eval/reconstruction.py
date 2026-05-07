"""
Reconstruction quality evaluation: PSNR / SSIM / LPIPS.

Compares MambaGaze vs AutoGaze on the block-causal VideoMAE reconstruction task.
Requires the AutoGaze reconstruction model weights (block-causal VideoMAE).

Usage:
    python -m mamba_gaze.eval.reconstruction \
        --config mamba_gaze/configs/default.yaml \
        --ckpt   checkpoints/phase2_ste/phase2ste_epoch0030.pt \
        --recon_model checkpoints/videomae_recon.pt \
        --split  val
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    yaml = None


# ── metric utilities ──────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. pred/target: (B, C, H, W) in [0, 1]."""
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1)).item()
    if mse < 1e-10:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def ssim_single(pred: torch.Tensor, target: torch.Tensor, win: int = 11) -> float:
    """Simplified SSIM for (C, H, W) tensors in [0, 1]."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # Gaussian kernel
    coords = torch.arange(win, dtype=torch.float32, device=pred.device) - win // 2
    kernel = torch.exp(-0.5 * (coords / (win / 6)) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, 1, win).expand(1, 1, win, win)

    def _mu(x):
        c = x.shape[0]
        k = kernel.expand(c, 1, win, win)
        return F.conv2d(x.unsqueeze(0), k, padding=win // 2, groups=c).squeeze(0)

    mu_x, mu_y   = _mu(pred), _mu(target)
    mu_xx, mu_yy = mu_x ** 2, mu_y ** 2
    mu_xy        = mu_x * mu_y
    sig_xx = _mu(pred ** 2) - mu_xx
    sig_yy = _mu(target ** 2) - mu_yy
    sig_xy = _mu(pred * target) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
    return (num / den.clamp(min=1e-8)).mean().item()


def lpips_score(pred: torch.Tensor, target: torch.Tensor, net="vgg") -> float:
    """LPIPS using lpips library. Returns float; 0 = identical."""
    try:
        import lpips as lpips_lib
        _fn = getattr(lpips_score, "_cache", None)
        if _fn is None or lpips_score._cache_net != net:
            lpips_score._cache = lpips_lib.LPIPS(net=net).to(pred.device)
            lpips_score._cache_net = net
        with torch.no_grad():
            return lpips_score._cache(
                pred.clamp(-1, 1).unsqueeze(0),
                target.clamp(-1, 1).unsqueeze(0),
            ).item()
    except ImportError:
        return float("nan")


# ── main evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    recon_model: Optional[nn.Module],
    data_loader,
    device: torch.device,
    gazing_ratio: float = 0.5,
) -> dict:
    model.eval()
    metrics = {"psnr": [], "ssim": [], "lpips": [], "mask_sparsity": []}

    for batch in data_loader:
        video = batch["video"].to(device)              # (B, T, 3, H, W)
        B, T  = video.shape[:2]

        outputs      = model({"video": video}, gazing_ratio=gazing_ratio)
        gazing_mask  = outputs["gazing_mask"]          # list[4] of (B, T, N_s)

        # Sparsity: fraction of tokens selected
        total_sel = sum(m.sum().item() for m in gazing_mask)
        total_tok = sum(m.numel() for m in gazing_mask)
        metrics["mask_sparsity"].append(total_sel / max(total_tok, 1))

        if recon_model is None:
            continue

        # Run reconstruction model with selected tokens
        try:
            recon_frames = recon_model(video, gazing_mask)   # (B, T, 3, H, W)
            for b in range(B):
                for t in range(T):
                    p = recon_frames[b, t]
                    g = video[b, t]
                    metrics["psnr"].append(psnr(p.unsqueeze(0), g.unsqueeze(0)))
                    metrics["ssim"].append(ssim_single(p, g))
                    metrics["lpips"].append(lpips_score(p, g))
        except Exception as e:
            print(f"Reconstruction error: {e}")

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {k: _avg(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--ckpt",        required=True, help="MambaGaze checkpoint")
    parser.add_argument("--recon_model", default=None,  help="VideoMAE recon weights")
    parser.add_argument("--split",       default="val")
    parser.add_argument("--gazing_ratio", type=float, default=0.5)
    parser.add_argument("--max_samples",  type=int,   default=None)
    args = parser.parse_args()

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ..models.mamba_gaze import MambaGaze
    from ..data.autogaze_dataset import build_dataloader

    model = MambaGaze.from_config(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded {args.ckpt}")

    recon_model = None
    if args.recon_model and Path(args.recon_model).exists():
        recon_model = torch.load(args.recon_model, map_location=device)
        recon_model.eval()

    loader = build_dataloader(
        args.split,
        batch_size=cfg.get("eval", {}).get("batch_size", 64),
        num_workers=4, distributed=False,
        max_samples=args.max_samples,
    )

    results = evaluate(model, recon_model, loader, device, args.gazing_ratio)

    print("\n── Reconstruction Eval ──────────────────────────")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
