from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when model deps are absent
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SimpleDevice:
    type: str


def resolve_device(name: str):
    if torch is None:
        if name in {"auto", "cpu"}:
            return SimpleDevice("cpu")
        raise RuntimeError(f"{name.upper()} was requested but PyTorch is not installed.")

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def _device_type(device: str | Any) -> str:
    if isinstance(device, str):
        return device
    return getattr(device, "type", str(device))


def synchronize(device: str | Any) -> None:
    if torch is None:
        return
    device_type = _device_type(device)
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


class BenchmarkTimer:
    def __init__(self, device: str | Any) -> None:
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.elapsed_ms: list[float] = []

    @contextmanager
    def measure(self):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.elapsed_ms.append((time.perf_counter() - start) * 1000.0)


def compute_stats(values: Iterable[float]) -> dict[str, float | int]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("")
        return
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_revision(path: str | Path) -> str | None:
    repo = Path(path)
    if not (repo / ".git").exists():
        return None
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def environment_metadata(device: Any, external_root: str | Path = "external") -> dict[str, Any]:
    external = Path(external_root)
    device_type = _device_type(device)
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "device": device_type,
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "mps_available": bool(torch and torch.backends.mps.is_available()),
        "autogaze_revision": git_revision(external / "AutoGaze"),
        "vila_revision": git_revision(external / "VILA"),
    }
    if torch is not None and device_type == "cuda":
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    return metadata
