from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, TypeVar

import torch


T = TypeVar("T")


def synchronize_if_cuda(device: str | torch.device | None = None) -> None:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
            return
        resolved = torch.device(device)
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)


@dataclass
class LatencyLogger:
    device: str = "cpu"
    measurements_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        synchronize_if_cuda(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize_if_cuda(self.device)
            self.measurements_ms[name] = (time.perf_counter() - start) * 1000.0

    def record(self, name: str, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        self.measurements_ms[name] = float(latency_ms)


def measure_latency_ms(
    fn: Callable[[], T],
    *,
    device: str | torch.device = "cpu",
    warmup_iters: int = 0,
    repeat: int = 1,
) -> tuple[T, float]:
    if warmup_iters < 0 or repeat <= 0:
        raise ValueError("warmup_iters must be >= 0 and repeat must be > 0")

    result: T | None = None
    for _ in range(warmup_iters):
        result = fn()
    synchronize_if_cuda(device)

    start = time.perf_counter()
    for _ in range(repeat):
        result = fn()
    synchronize_if_cuda(device)
    latency_ms = (time.perf_counter() - start) * 1000.0 / float(repeat)

    return result, latency_ms
