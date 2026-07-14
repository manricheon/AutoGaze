#!/usr/bin/env python
"""Latency benchmark for Borissal.select() (selector forward only, no video decode).

Measures wall-clock latency across devices, gazing_ratio, and per_frame_allocation
for the default clip shape (B=1, T=16, C=3, H=384, W=384). Also captures a
torch.profiler per-op breakdown for one representative config, which doubles as
input to the mobile operator/burden review (docs/borissal/design.md).

Results are written to outputs/borissal/benchmark/ (gitignored) and printed to
stdout as a summary table.

Example:
    uv run python scripts/borissal_benchmark.py
"""

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from autogaze.models.borissal import (
    Borissal,
    BorissalConfig,
    BorissalV1,
    BorissalV1Config,
    available_devices,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "borissal" / "benchmark"

B, T, C, H, W = 1, 16, 3, 384, 384
WARMUP = 10
ITERS = 50
RATIOS = [0.5, 0.25]
ALLOCATIONS = ["uniform", "proportional"]


def build_model(which: str, ratio: float, alloc: str, checkpoint: str = None):
    if which == "v0":
        return Borissal(BorissalConfig(gazing_ratio=ratio, per_frame_allocation=alloc))
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = BorissalV1(BorissalV1Config(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = BorissalV1(BorissalV1Config(gazing_ratio=ratio, per_frame_allocation=alloc))
    return model.eval()


def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    # ru_maxrss: bytes on macOS, KB on Linux
    import sys
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def bench_one(which: str, device: torch.device, ratio: float, alloc: str, checkpoint: str = None):
    model = build_model(which, ratio, alloc, checkpoint).to(device)
    video = torch.rand(B, T, C, H, W, device=device)

    sync = (lambda: torch.mps.synchronize()) if device.type == "mps" else (
        (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    )

    for _ in range(WARMUP):
        model.select(video)
    sync()
    reset_peak_memory(device)

    times_ms = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        model.select(video)
        sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms)
    return {
        "model": which,
        "device": str(device),
        "gazing_ratio": ratio,
        "per_frame_allocation": alloc,
        "iters": ITERS,
        "mean_ms": t.mean().item(),
        "median_ms": t.median().item(),
        "std_ms": t.std().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
        "throughput_clips_per_sec": 1000.0 / t.mean().item(),
        "peak_mem_mb": peak_memory_mb(device),
    }


def run_profiler(device: torch.device):
    cfg = BorissalConfig(gazing_ratio=0.5, per_frame_allocation="uniform")
    model = Borissal(cfg).to(device)
    video = torch.rand(B, T, C, H, W, device=device)
    for _ in range(5):
        model.select(video)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(10):
            model.select(video)

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["v0", "v1", "both"], default="v0")
    p.add_argument("--checkpoint", default=None, help="v1 checkpoint .pt (optional; random init if omitted)")
    p.add_argument("--skip-profiler", action="store_true")
    args = p.parse_args()
    models = ["v0", "v1"] if args.model == "both" else [args.model]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    devices = available_devices()
    print(f"devices available: {devices}")
    print(f"clip shape: (B={B}, T={T}, C={C}, H={H}, W={W}); warmup={WARMUP}, iters={ITERS}")
    print()

    all_results = []
    header = (f"{'model':5} {'device':6} {'ratio':6} {'alloc':12} {'mean(ms)':9} {'median(ms)':11} "
              f"{'std(ms)':8} {'min(ms)':8} {'max(ms)':8} {'clips/s':9} {'peakMB':8}")
    print(header)
    print("-" * len(header))
    for device_name in devices:
        device = torch.device(device_name)
        for which in models:
            for ratio in RATIOS:
                for alloc in ALLOCATIONS:
                    r = bench_one(which, device, ratio, alloc, args.checkpoint)
                    all_results.append(r)
                    print(
                        f"{r['model']:5} {r['device']:6} {r['gazing_ratio']:<6} {r['per_frame_allocation']:12} "
                        f"{r['mean_ms']:9.3f} {r['median_ms']:11.3f} {r['std_ms']:8.3f} "
                        f"{r['min_ms']:8.3f} {r['max_ms']:8.3f} {r['throughput_clips_per_sec']:9.1f} "
                        f"{r['peak_mem_mb']:8.1f}"
                    )
        out_path = OUT_DIR / f"latency_{device_name}.json"
        with open(out_path, "w") as f:
            json.dump([r for r in all_results if r["device"] == str(device)], f, indent=2)
        print(f"-> saved {out_path}")

    if not args.skip_profiler:
        print()
        print("running torch.profiler on cpu (representative config: ratio=0.5, uniform)...")
        table = run_profiler(torch.device("cpu"))
        profiler_path = OUT_DIR / "profiler_cpu.txt"
        with open(profiler_path, "w") as f:
            f.write(table)
        print(f"-> saved {profiler_path}")
        print()
        print(table)


if __name__ == "__main__":
    main()
