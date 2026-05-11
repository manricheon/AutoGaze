from __future__ import annotations

from dataclasses import dataclass

import torch


NA = "N/A"


@dataclass(frozen=True)
class MemorySnapshot:
    device_type: str
    peak_vram_mb: float | str
    allocated_mb: float | str


class MemoryTracker:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def reset_peak(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> MemorySnapshot:
        if self.device.type == "cuda" and torch.cuda.is_available():
            return MemorySnapshot(
                device_type="cuda",
                peak_vram_mb=torch.cuda.max_memory_allocated(self.device) / (1024**2),
                allocated_mb=torch.cuda.memory_allocated(self.device) / (1024**2),
            )
        if self.device.type == "mps":
            return MemorySnapshot(device_type="mps", peak_vram_mb=NA, allocated_mb=NA)
        return MemorySnapshot(device_type=self.device.type, peak_vram_mb=NA, allocated_mb=NA)
