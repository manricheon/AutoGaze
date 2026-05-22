from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from repro.plugins.gaze_plan import SparseSelectionPlan


@dataclass(frozen=True)
class TokenSelectorInput:
    raw_patch_tokens: int
    frame_indices: list[int]
    patch_space: dict[str, Any]
    gazing_ratio: float | list[float] | None
    task_loss_requirement: float | None


@dataclass(frozen=True)
class TokenSelectorOutput:
    selected_positions: Any
    selected_mask_by_scale: dict[int, Any] | None
    raw_patch_tokens: int
    selected_patch_tokens: int | None
    reduction_ratio: float | None
    latency_ms: float
    peak_memory_bytes: int | None
    status: str
    metric_status: dict[str, str | None]
    sparse_selection_plan: SparseSelectionPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_positions": self.selected_positions,
            "selected_mask_by_scale": self.selected_mask_by_scale,
            "raw_patch_tokens": self.raw_patch_tokens,
            "selected_patch_tokens": self.selected_patch_tokens,
            "reduction_ratio": self.reduction_ratio,
            "latency_ms": self.latency_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
            "status": self.status,
            "metric_status": self.metric_status,
            "sparse_selection_plan": (
                self.sparse_selection_plan.to_dict() if self.sparse_selection_plan is not None else None
            ),
        }


class NoTokenSelector:
    name = "none"

    def select(self, selector_input: TokenSelectorInput) -> TokenSelectorOutput:
        return TokenSelectorOutput(
            selected_positions=None,
            selected_mask_by_scale=None,
            raw_patch_tokens=selector_input.raw_patch_tokens,
            selected_patch_tokens=None,
            reduction_ratio=None,
            latency_ms=0.0,
            peak_memory_bytes=None,
            status="not_applicable",
            metric_status={"value": "not_applicable", "reason": "No token selector is defined for this run."},
        )


class KeepAllTokenSelector:
    name = "keep-all"

    def select(self, selector_input: TokenSelectorInput) -> TokenSelectorOutput:
        return TokenSelectorOutput(
            selected_positions="keep_all",
            selected_mask_by_scale=None,
            raw_patch_tokens=selector_input.raw_patch_tokens,
            selected_patch_tokens=selector_input.raw_patch_tokens,
            reduction_ratio=1.0,
            latency_ms=0.0,
            peak_memory_bytes=None,
            status="keep_all",
            metric_status={"value": "native_off", "reason": "All visual tokens are preserved."},
        )


class AutoGazeSelectorPlan:
    name = "autogaze"

    def select(self, selector_input: TokenSelectorInput) -> TokenSelectorOutput:
        selected_patch_tokens = _estimate_selected_patch_tokens(
            selector_input.raw_patch_tokens,
            selector_input.gazing_ratio,
        )
        return TokenSelectorOutput(
            selected_positions=None,
            selected_mask_by_scale=None,
            raw_patch_tokens=selector_input.raw_patch_tokens,
            selected_patch_tokens=selected_patch_tokens,
            reduction_ratio=_safe_ratio(selector_input.raw_patch_tokens, selected_patch_tokens),
            latency_ms=0.0,
            peak_memory_bytes=None,
            status="probe_required",
            metric_status={
                "value": "probe_required",
                "reason": "AutoGaze selector has not emitted concrete patch coordinates.",
            },
            sparse_selection_plan=SparseSelectionPlan.placeholder(
                selector_name="autogaze",
                source_path=None,
                raw_patch_tokens=selector_input.raw_patch_tokens,
                selected_patch_tokens=selected_patch_tokens,
                frame_indices=selector_input.frame_indices,
                reason="AutoGaze selector has not emitted concrete patch coordinates.",
            ),
        )


def _estimate_selected_patch_tokens(raw_patch_tokens: int, gazing_ratio: float | list[float] | None) -> int:
    if gazing_ratio is None:
        return raw_patch_tokens
    if isinstance(gazing_ratio, list):
        ratio = sum(float(value) for value in gazing_ratio) / len(gazing_ratio) if gazing_ratio else 1.0
    else:
        ratio = float(gazing_ratio)
    ratio = max(0.0, min(1.0, ratio))
    return max(1, int(round(raw_patch_tokens * ratio)))


def _safe_ratio(raw_patch_tokens: int, selected_patch_tokens: int | None) -> float | None:
    if selected_patch_tokens in (None, 0):
        return None
    return raw_patch_tokens / selected_patch_tokens
