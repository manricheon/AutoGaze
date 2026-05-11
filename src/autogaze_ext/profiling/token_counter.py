from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TokenCountSummary:
    visual_token_count_before_autogaze: int
    visual_token_count_after_autogaze: int
    token_reduction_ratio: float
    selected_patches_per_frame: int | float
    selected_patches_per_scale: dict[str, int] | str


def token_reduction_ratio(before: int, after: int) -> float:
    if before <= 0:
        raise ValueError("before token count must be > 0")
    if after < 0:
        raise ValueError("after token count must be >= 0")
    if after > before:
        raise ValueError("after token count cannot exceed before token count")
    return 1.0 - (float(after) / float(before))


def count_selected_patches_per_scale(
    selected_scales: torch.Tensor | list[int] | None,
) -> dict[str, int] | str:
    if selected_scales is None:
        return "N/A"
    tensor = torch.as_tensor(selected_scales).flatten()
    counts: dict[str, int] = {}
    for scale in tensor.detach().cpu().tolist():
        counts[str(int(scale))] = counts.get(str(int(scale)), 0) + 1
    return counts


def summarize_tokens(
    *,
    before: int,
    after: int,
    frame_count: int | None = None,
    selected_scales: torch.Tensor | list[int] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TokenCountSummary:
    if frame_count is None:
        frame_count = int((metadata or {}).get("sampled_frame_count", 1))
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    return TokenCountSummary(
        visual_token_count_before_autogaze=int(before),
        visual_token_count_after_autogaze=int(after),
        token_reduction_ratio=token_reduction_ratio(int(before), int(after)),
        selected_patches_per_frame=int(after) / float(frame_count),
        selected_patches_per_scale=count_selected_patches_per_scale(selected_scales),
    )
