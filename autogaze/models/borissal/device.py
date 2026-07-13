"""Device selection for Borissal.

Static, evidence-informed priority: cuda > cpu > mps. cuda is the obvious
choice when present. Between cpu and mps, this is NOT the usual "prefer any
accelerator" default -- measured on this project's benchmark
(scripts/borissal_benchmark.py, see docs/borissal/design.md's "Mobile
readiness review"), MPS was consistently slower than CPU for Borissal's
small-tensor workload (per-op GPU dispatch overhead dominates), so CPU is
tried first. Pass mode="mps" explicitly if you want it anyway.
"""

import torch


def resolve_device(mode: str = "auto") -> torch.device:
    if mode != "auto":
        return torch.device(mode)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def available_devices() -> list:
    """All device types actually present on this machine (for benchmarking/enumeration,
    not for picking one to run on -- see resolve_device for that)."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices
