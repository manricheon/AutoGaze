"""Profiling utilities for the AutoGaze extension PoC."""

from autogaze_ext.profiling.latency import LatencyLogger, measure_latency_ms, synchronize_if_cuda
from autogaze_ext.profiling.memory import MemorySnapshot, MemoryTracker
from autogaze_ext.profiling.token_counter import (
    TokenCountSummary,
    count_selected_patches_per_scale,
    summarize_tokens,
    token_reduction_ratio,
)

__all__ = [
    "LatencyLogger",
    "MemorySnapshot",
    "MemoryTracker",
    "TokenCountSummary",
    "count_selected_patches_per_scale",
    "measure_latency_ms",
    "summarize_tokens",
    "synchronize_if_cuda",
    "token_reduction_ratio",
]
