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

import json
import time
from pathlib import Path

import torch

from autogaze.models.borissal import Borissal, BorissalConfig, available_devices

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "borissal" / "benchmark"

B, T, C, H, W = 1, 16, 3, 384, 384
WARMUP = 10
ITERS = 50
RATIOS = [0.5, 0.25]
ALLOCATIONS = ["uniform", "proportional"]


def bench_one(device: torch.device, ratio: float, alloc: str):
    cfg = BorissalConfig(gazing_ratio=ratio, per_frame_allocation=alloc)
    model = Borissal(cfg).to(device)
    video = torch.rand(B, T, C, H, W, device=device)

    sync = (lambda: torch.mps.synchronize()) if device.type == "mps" else (
        (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    )

    for _ in range(WARMUP):
        model.select(video)
    sync()

    times_ms = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        model.select(video)
        sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms)
    return {
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    devices = available_devices()
    print(f"devices available: {devices}")
    print(f"clip shape: (B={B}, T={T}, C={C}, H={H}, W={W}); warmup={WARMUP}, iters={ITERS}")
    print()

    all_results = []
    header = f"{'device':6} {'ratio':6} {'alloc':12} {'mean(ms)':9} {'median(ms)':11} {'std(ms)':8} {'min(ms)':8} {'max(ms)':8} {'clips/s':9}"
    print(header)
    print("-" * len(header))
    for device_name in devices:
        device = torch.device(device_name)
        for ratio in RATIOS:
            for alloc in ALLOCATIONS:
                r = bench_one(device, ratio, alloc)
                all_results.append(r)
                print(
                    f"{r['device']:6} {r['gazing_ratio']:<6} {r['per_frame_allocation']:12} "
                    f"{r['mean_ms']:9.3f} {r['median_ms']:11.3f} {r['std_ms']:8.3f} "
                    f"{r['min_ms']:8.3f} {r['max_ms']:8.3f} {r['throughput_clips_per_sec']:9.1f}"
                )
        out_path = OUT_DIR / f"latency_{device_name}.json"
        with open(out_path, "w") as f:
            json.dump([r for r in all_results if r["device"] == str(device)], f, indent=2)
        print(f"-> saved {out_path}")

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
