"""
Gazing latency benchmark — replicates AutoGaze Fig. 8 format.

Measures mean per-frame gazing latency across:
  - Batch sizes: 1, 4, 8
  - T (frames): 4, 8, 16
  - Gazing ratios: 0.25, 0.50, 0.75
  - Model variants: MambaGaze (ours) vs AutoGaze baseline

Outputs a CSV + optional ASCII table.

Usage:
    python -m mamba_gaze.eval.latency \
        --config mamba_gaze/configs/default.yaml \
        --ckpt   checkpoints/phase2_ste/phase2ste_epoch0030.pt \
        --autogaze_ckpt weights/autogaze.pt \
        --out    results/latency.csv
"""

import argparse
import csv
import time
from itertools import product
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None


# ── timing utility ─────────────────────────────────────────────────────────────

def timed_forward(
    model: nn.Module,
    inputs: dict,
    gazing_ratio: float,
    n_runs: int,
    warmup: int,
    device: torch.device,
) -> float:
    """Returns mean latency in milliseconds per frame."""
    model.eval()
    T = inputs["video"].shape[1]

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs, gazing_ratio=gazing_ratio)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(inputs, gazing_ratio=gazing_ratio)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return elapsed_ms / (n_runs * T)   # ms per frame


# ── benchmark grid ─────────────────────────────────────────────────────────────

def run_benchmark(
    models: dict,         # {"MambaGaze": model, "AutoGaze": model, ...}
    cfg: dict,
    device: torch.device,
) -> list:
    """
    Returns list of result dicts for CSV export.
    """
    e_cfg   = cfg.get("eval", {})
    n_runs  = e_cfg.get("latency_runs",   200)
    warmup  = e_cfg.get("latency_warmup",  20)
    img_size= cfg.get("model", {}).get("img_size", 224)

    batch_sizes   = [1, 4, 8]
    frame_counts  = [4, 8, 16]
    gazing_ratios = [0.25, 0.50, 0.75]

    rows = []
    for model_name, model in models.items():
        model = model.to(device)
        for B, T, ratio in product(batch_sizes, frame_counts, gazing_ratios):
            video = torch.randn(B, T, 3, img_size, img_size, device=device)
            inputs = {"video": video}
            try:
                ms = timed_forward(model, inputs, ratio, n_runs, warmup, device)
            except Exception as e:
                ms = float("nan")
                print(f"  Error ({model_name}, B={B}, T={T}, r={ratio}): {e}")

            row = {
                "model":        model_name,
                "batch_size":   B,
                "num_frames":   T,
                "gazing_ratio": ratio,
                "ms_per_frame": round(ms, 3),
            }
            rows.append(row)
            print(f"  {model_name:15s}  B={B}  T={T}  r={ratio:.2f}  → {ms:.2f} ms/frame")

    return rows


def save_csv(rows: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = ["model", "batch_size", "num_frames", "gazing_ratio", "ms_per_frame"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved latency results → {path}")


def print_table(rows: list) -> None:
    header = f"{'Model':15s}  {'B':>3}  {'T':>3}  {'ratio':>5}  {'ms/frame':>10}"
    print("\n── Latency Benchmark ──────────────────────────────")
    print(header)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r['model']:15s}  {r['batch_size']:>3}  {r['num_frames']:>3}"
            f"  {r['gazing_ratio']:>5.2f}  {r['ms_per_frame']:>10.2f}"
        )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         default="mamba_gaze/configs/default.yaml")
    parser.add_argument("--ckpt",           required=True, help="MambaGaze checkpoint")
    parser.add_argument("--autogaze_ckpt",  default=None,  help="AutoGaze checkpoint (optional)")
    parser.add_argument("--out",            default="results/latency.csv")
    args = parser.parse_args()

    cfg = {}
    if yaml and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from ..models.mamba_gaze import MambaGaze
    mamba_model = MambaGaze.from_config(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    mamba_model.load_state_dict(ckpt.get("model", ckpt))

    models = {"MambaGaze": mamba_model}

    if args.autogaze_ckpt and Path(args.autogaze_ckpt).exists():
        try:
            from autogaze.models.autogaze.autogaze import AutoGaze
            ag = AutoGaze.__new__(AutoGaze)
            ag_ckpt = torch.load(args.autogaze_ckpt, map_location="cpu")
            ag.load_state_dict(ag_ckpt.get("model", ag_ckpt))
            models["AutoGaze"] = ag
        except Exception as e:
            print(f"Could not load AutoGaze model: {e}")

    rows = run_benchmark(models, cfg, device)
    print_table(rows)
    save_csv(rows, args.out)


if __name__ == "__main__":
    main()
