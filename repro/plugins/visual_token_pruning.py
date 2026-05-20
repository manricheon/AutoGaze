from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def prune_sequence_by_indices(features: Sequence[Any], selected_indices: Sequence[int]) -> list[Any]:
    return [features[index] for index in selected_indices]


@dataclass(frozen=True)
class PostEncoderPruneResult:
    raw_visual_tokens: int
    selected_visual_tokens: int
    reduction_ratio: float | None
    vision_encoder_latency_reduced: bool
    mllm_context_reduced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_visual_tokens": self.raw_visual_tokens,
            "selected_visual_tokens": self.selected_visual_tokens,
            "reduction_ratio": self.reduction_ratio,
            "vision_encoder_latency_reduced": self.vision_encoder_latency_reduced,
            "mllm_context_reduced": self.mllm_context_reduced,
            "expected_gain": "mllm_context_only",
        }


@dataclass(frozen=True)
class PreEncoderSparseProbe:
    model_family: str
    raw_patch_tokens: int
    selected_patch_tokens: int
    reduction_ratio: float | None
    required_semantics: list[str]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "raw_patch_tokens": self.raw_patch_tokens,
            "selected_patch_tokens": self.selected_patch_tokens,
            "reduction_ratio": self.reduction_ratio,
            "required_semantics": self.required_semantics,
            "status": self.status,
        }


def build_post_encoder_prune_result(raw_visual_tokens: int, selected_indices: Sequence[int]) -> PostEncoderPruneResult:
    selected_visual_tokens = len(selected_indices)
    return PostEncoderPruneResult(
        raw_visual_tokens=raw_visual_tokens,
        selected_visual_tokens=selected_visual_tokens,
        reduction_ratio=_safe_ratio(raw_visual_tokens, selected_visual_tokens),
        vision_encoder_latency_reduced=False,
        mllm_context_reduced=True,
    )


def build_pre_encoder_sparse_probe(
    *,
    model_family: str,
    raw_patch_tokens: int,
    selected_indices: Sequence[int],
    position_grid_fields: Sequence[str],
) -> PreEncoderSparseProbe:
    selected_patch_tokens = len(selected_indices)
    return PreEncoderSparseProbe(
        model_family=model_family,
        raw_patch_tokens=raw_patch_tokens,
        selected_patch_tokens=selected_patch_tokens,
        reduction_ratio=_safe_ratio(raw_patch_tokens, selected_patch_tokens),
        required_semantics=list(position_grid_fields),
        status="requires_model_specific_probe",
    )


def _safe_ratio(raw_tokens: int, selected_tokens: int) -> float | None:
    if selected_tokens == 0:
        return None
    return raw_tokens / selected_tokens
