from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class SourceVideo:
    path: str | None
    source_width: int | None = None
    source_height: int | None = None
    sampled_frame_indices: list[int] = field(default_factory=list)
    sampled_fps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "sampled_frame_indices": list(self.sampled_frame_indices),
            "sampled_fps": self.sampled_fps,
        }


@dataclass(frozen=True)
class PreprocessSpace:
    resize_policy: str | None = None
    resized_width: int | None = None
    resized_height: int | None = None
    tile_grid: list[int] | None = None
    tile_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resize_policy": self.resize_policy,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "tile_grid": list(self.tile_grid) if self.tile_grid is not None else None,
            "tile_size": self.tile_size,
        }


@dataclass(frozen=True)
class PatchSpace:
    autogaze_patch_size: int | None = None
    encoder_patch_size: int | None = None
    scale_ids: list[int] = field(default_factory=list)
    scale_sizes: list[int] = field(default_factory=list)
    patch_size_mismatch: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        mismatch = self.patch_size_mismatch
        if mismatch is None and self.autogaze_patch_size is not None and self.encoder_patch_size is not None:
            mismatch = int(self.autogaze_patch_size) != int(self.encoder_patch_size)
        return {
            "autogaze_patch_size": self.autogaze_patch_size,
            "encoder_patch_size": self.encoder_patch_size,
            "scale_ids": list(self.scale_ids),
            "scale_sizes": list(self.scale_sizes),
            "patch_size_mismatch": mismatch,
        }


@dataclass(frozen=True)
class SelectedPatch:
    frame_index: int
    frame_order: int
    tile_id: int
    scale_id: int
    scale_size: int
    patch_index: int
    bbox_resized_xyxy: list[int]
    bbox_original_xyxy: list[float]
    autoregressive_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "frame_order": int(self.frame_order),
            "tile_id": int(self.tile_id),
            "scale_id": int(self.scale_id),
            "scale_size": int(self.scale_size),
            "patch_index": int(self.patch_index),
            "bbox_resized_xyxy": [int(value) for value in self.bbox_resized_xyxy],
            "bbox_original_xyxy": [float(value) for value in self.bbox_original_xyxy],
            "autoregressive_order": int(self.autoregressive_order),
        }


@dataclass(frozen=True)
class EncoderMapping:
    status: str
    encoder_grid_thw: list[int] | None = None
    encoder_patch_indices: list[int] | None = None
    position_ids: Any = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "encoder_grid_thw": list(self.encoder_grid_thw) if self.encoder_grid_thw is not None else None,
            "encoder_patch_indices": (
                [int(value) for value in self.encoder_patch_indices]
                if self.encoder_patch_indices is not None
                else None
            ),
            "position_ids": _json_ready(self.position_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MllmMapping:
    status: str
    visual_feature_indices: list[int] | None = None
    projected_token_indices: list[int] | None = None
    llm_context_indices: list[int] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "visual_feature_indices": _optional_int_list(self.visual_feature_indices),
            "projected_token_indices": _optional_int_list(self.projected_token_indices),
            "llm_context_indices": _optional_int_list(self.llm_context_indices),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SparseSelectionPlan:
    selector_name: str
    source_video: SourceVideo
    preprocess_space: PreprocessSpace
    patch_space: PatchSpace
    selected_patches: list[SelectedPatch]
    encoder_mapping: EncoderMapping
    mllm_mapping: MllmMapping
    raw_patch_tokens: int | None = None
    selected_patch_tokens: int | None = None
    dense_masks: dict[str, Any] | None = None
    quality_control: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def placeholder(
        cls,
        *,
        selector_name: str,
        source_path: str | None,
        raw_patch_tokens: int | None,
        selected_patch_tokens: int | None,
        frame_indices: list[int],
        reason: str,
    ) -> "SparseSelectionPlan":
        return cls(
            selector_name=selector_name,
            source_video=SourceVideo(path=source_path, sampled_frame_indices=list(frame_indices)),
            preprocess_space=PreprocessSpace(),
            patch_space=PatchSpace(),
            selected_patches=[],
            encoder_mapping=EncoderMapping(status="not_mapped", reason=reason),
            mllm_mapping=MllmMapping(status="not_mapped", reason=reason),
            raw_patch_tokens=raw_patch_tokens,
            selected_patch_tokens=selected_patch_tokens,
            quality_control={"reason": reason},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_name": self.selector_name,
            "source_video": self.source_video.to_dict(),
            "preprocess_space": self.preprocess_space.to_dict(),
            "patch_space": self.patch_space.to_dict(),
            "selected_patches": [patch.to_dict() for patch in self.selected_patches],
            "dense_masks": _json_ready(self.dense_masks),
            "encoder_mapping": self.encoder_mapping.to_dict(),
            "mllm_mapping": self.mllm_mapping.to_dict(),
            "token_accounting": {
                "raw_patch_tokens": self.raw_patch_tokens,
                "selected_patch_tokens": self.selected_patch_tokens,
                "reduction_ratio": _safe_ratio(self.raw_patch_tokens, self.selected_patch_tokens),
            },
            "quality_control": _json_ready(self.quality_control),
        }


def sparse_selection_plan_from_dict(payload: dict[str, Any]) -> SparseSelectionPlan:
    return SparseSelectionPlan(
        selector_name=str(payload.get("selector_name") or "unknown"),
        source_video=SourceVideo(**_source_video_kwargs(payload.get("source_video") or {})),
        preprocess_space=PreprocessSpace(**_preprocess_space_kwargs(payload.get("preprocess_space") or {})),
        patch_space=PatchSpace(**_patch_space_kwargs(payload.get("patch_space") or {})),
        selected_patches=[
            SelectedPatch(**_selected_patch_kwargs(item))
            for item in payload.get("selected_patches") or []
        ],
        encoder_mapping=EncoderMapping(**_encoder_mapping_kwargs(payload.get("encoder_mapping") or {})),
        mllm_mapping=MllmMapping(**_mllm_mapping_kwargs(payload.get("mllm_mapping") or {})),
        raw_patch_tokens=(payload.get("token_accounting") or {}).get("raw_patch_tokens")
        if "token_accounting" in payload
        else payload.get("raw_patch_tokens"),
        selected_patch_tokens=(payload.get("token_accounting") or {}).get("selected_patch_tokens")
        if "token_accounting" in payload
        else payload.get("selected_patch_tokens"),
        dense_masks=payload.get("dense_masks"),
        quality_control=payload.get("quality_control") or {},
    )


def qwen_visual_indices_from_sparse_plan(
    plan: SparseSelectionPlan,
    *,
    video_grid_thw: Any,
    spatial_merge_size: int = 1,
) -> MllmMapping:
    t, h, w = _grid_thw(video_grid_thw)
    merge = max(1, int(spatial_merge_size or 1))
    merged_h = max(1, math.ceil(h / merge))
    merged_w = max(1, math.ceil(w / merge))
    indices: list[int] = []
    statuses: set[str] = set()
    for patch in sorted(plan.selected_patches, key=lambda item: item.autoregressive_order):
        frame_order = _map_frame_order_to_qwen_temporal_index(plan, int(patch.frame_order), t)
        row_col_status = _qwen_row_col_for_patch(plan, patch, h=h, w=w)
        if row_col_status is None:
            continue
        row, col, status = row_col_status
        statuses.add(status)
        merged_row = min(max(row // merge, 0), merged_h - 1)
        merged_col = min(max(col // merge, 0), merged_w - 1)
        index = frame_order * merged_h * merged_w + merged_row * merged_w + merged_col
        if index not in indices:
            indices.append(index)
    if not indices:
        return MllmMapping(
            status="mapping_failed",
            visual_feature_indices=[],
            reason="no selected AutoGaze patch could be mapped to Qwen visual feature indices",
        )
    if statuses == {"exact_grid"} and merge == 1:
        status = "exact_grid"
        reason = f"mapped {len(plan.selected_patches)} AutoGaze patches to {len(indices)} Qwen visual feature indices"
    elif "approximate_bbox" in statuses:
        status = "approximate_bbox"
        reason = (
            f"mapped {len(plan.selected_patches)} AutoGaze patches to {len(indices)} Qwen visual feature "
            "indices using bbox center overlap"
        )
    else:
        status = "approximate_grid"
        reason = f"mapped {len(plan.selected_patches)} AutoGaze patches to {len(indices)} merged Qwen visual feature indices"
    return MllmMapping(status=status, visual_feature_indices=indices, reason=reason)


def _map_frame_order_to_qwen_temporal_index(plan: SparseSelectionPlan, frame_order: int, qwen_t: int) -> int:
    if qwen_t <= 1:
        return 0
    sampled_count = len(plan.source_video.sampled_frame_indices)
    if sampled_count > 1 and sampled_count != qwen_t:
        return min(max(int(round(frame_order * (qwen_t - 1) / (sampled_count - 1))), 0), qwen_t - 1)
    return min(max(int(frame_order), 0), qwen_t - 1)


def _qwen_row_col_for_patch(
    plan: SparseSelectionPlan,
    patch: SelectedPatch,
    *,
    h: int,
    w: int,
) -> tuple[int, int, str] | None:
    autogaze_patch_size = plan.patch_space.autogaze_patch_size
    scale_size = patch.scale_size or _first_or_none(plan.patch_space.scale_sizes)
    if autogaze_patch_size and scale_size:
        grid = max(1, int(scale_size) // int(autogaze_patch_size))
        patch_row = int(patch.patch_index) // grid
        patch_col = int(patch.patch_index) % grid
        patch_size_mismatch = (
            plan.patch_space.patch_size_mismatch
            if plan.patch_space.patch_size_mismatch is not None
            else (
                plan.patch_space.autogaze_patch_size is not None
                and plan.patch_space.encoder_patch_size is not None
                and int(plan.patch_space.autogaze_patch_size) != int(plan.patch_space.encoder_patch_size)
            )
        )
        if grid == h and grid == w and patch_size_mismatch is not True:
            return min(patch_row, h - 1), min(patch_col, w - 1), "exact_grid"
        mapped_row = min(max(int((patch_row + 0.5) * h / grid), 0), h - 1)
        mapped_col = min(max(int((patch_col + 0.5) * w / grid), 0), w - 1)
        if _has_resized_bbox(plan, patch):
            return _qwen_row_col_from_bbox(plan, patch, h=h, w=w)
        return mapped_row, mapped_col, "approximate_grid"
    if _has_resized_bbox(plan, patch):
        return _qwen_row_col_from_bbox(plan, patch, h=h, w=w)
    return None


def _qwen_row_col_from_bbox(
    plan: SparseSelectionPlan,
    patch: SelectedPatch,
    *,
    h: int,
    w: int,
) -> tuple[int, int, str]:
    width = max(float(plan.preprocess_space.resized_width or 1), 1.0)
    height = max(float(plan.preprocess_space.resized_height or 1), 1.0)
    x1, y1, x2, y2 = [float(value) for value in patch.bbox_resized_xyxy]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    col = min(max(int(center_x / width * w), 0), w - 1)
    row = min(max(int(center_y / height * h), 0), h - 1)
    return row, col, "approximate_bbox"


def _grid_thw(value: Any) -> tuple[int, int, int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"Expected Qwen video_grid_thw with three values, got {value!r}")
    return int(value[0]), int(value[1]), int(value[2])


def _has_resized_bbox(plan: SparseSelectionPlan, patch: SelectedPatch) -> bool:
    return (
        plan.preprocess_space.resized_width is not None
        and plan.preprocess_space.resized_height is not None
        and len(patch.bbox_resized_xyxy) == 4
    )


def _first_or_none(values: list[int]) -> int | None:
    return values[0] if values else None


def _source_video_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": payload.get("path"),
        "source_width": payload.get("source_width"),
        "source_height": payload.get("source_height"),
        "sampled_frame_indices": list(payload.get("sampled_frame_indices") or []),
        "sampled_fps": payload.get("sampled_fps"),
    }


def _preprocess_space_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "resize_policy": payload.get("resize_policy"),
        "resized_width": payload.get("resized_width"),
        "resized_height": payload.get("resized_height"),
        "tile_grid": payload.get("tile_grid"),
        "tile_size": payload.get("tile_size"),
    }


def _patch_space_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "autogaze_patch_size": payload.get("autogaze_patch_size"),
        "encoder_patch_size": payload.get("encoder_patch_size"),
        "scale_ids": list(payload.get("scale_ids") or []),
        "scale_sizes": list(payload.get("scale_sizes") or []),
        "patch_size_mismatch": payload.get("patch_size_mismatch"),
    }


def _selected_patch_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": payload.get("frame_index", 0),
        "frame_order": payload.get("frame_order", 0),
        "tile_id": payload.get("tile_id", 0),
        "scale_id": payload.get("scale_id", 0),
        "scale_size": payload.get("scale_size", 0),
        "patch_index": payload.get("patch_index", 0),
        "bbox_resized_xyxy": list(payload.get("bbox_resized_xyxy") or [0, 0, 0, 0]),
        "bbox_original_xyxy": list(payload.get("bbox_original_xyxy") or [0.0, 0.0, 0.0, 0.0]),
        "autoregressive_order": payload.get("autoregressive_order", 0),
    }


def _encoder_mapping_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status", "not_mapped"),
        "encoder_grid_thw": payload.get("encoder_grid_thw"),
        "encoder_patch_indices": payload.get("encoder_patch_indices"),
        "position_ids": payload.get("position_ids"),
        "reason": payload.get("reason"),
    }


def _mllm_mapping_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status", "not_mapped"),
        "visual_feature_indices": payload.get("visual_feature_indices"),
        "projected_token_indices": payload.get("projected_token_indices"),
        "llm_context_indices": payload.get("llm_context_indices"),
        "reason": payload.get("reason"),
    }


def _optional_int_list(values: list[int] | None) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def _safe_ratio(before: int | None, after: int | None) -> float | None:
    if before is None or after in (None, 0):
        return None
    return float(before) / float(after)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
