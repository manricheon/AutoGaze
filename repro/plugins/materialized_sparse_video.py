from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from repro.gaze_visualization import write_video
from repro.nvila_runner import load_sampled_video_frames
from repro.plugins.gaze_plan import SparseSelectionPlan, sparse_selection_plan_from_dict


@dataclass(frozen=True)
class MaterializedSparseFrames:
    frames: list[Image.Image]
    metadata: dict[str, Any]


def build_materialized_sparse_frames(
    frames: list[Image.Image],
    plan: SparseSelectionPlan,
    *,
    crop_to_selection: bool = True,
) -> MaterializedSparseFrames:
    if not frames:
        raise ValueError("materialized sparse video requires at least one sampled frame")
    selected_orders = _selected_frame_orders(plan, len(frames))
    fallback_reason = None
    if not selected_orders:
        selected_orders = [0]
        fallback_reason = "no_selected_patches"

    crop_boxes = [_union_crop_box(plan, order, frames[order].size) for order in selected_orders]
    selected_frames: list[Image.Image] = []
    for order, crop_box in zip(selected_orders, crop_boxes):
        frame = frames[order].convert("RGB")
        if crop_to_selection and crop_box is not None:
            frame = frame.crop(tuple(crop_box))
        selected_frames.append(frame)

    output_size = _common_output_size(selected_frames)
    normalized_frames = [
        frame if frame.size == output_size else frame.resize(output_size)
        for frame in selected_frames
    ]
    sampled_indices = plan.source_video.sampled_frame_indices
    metadata: dict[str, Any] = {
        "integration_claim": "materialized_sparse_video",
        "coarse_pre_vit_input_reduced": len(selected_orders) < len(frames) or any(crop_boxes),
        "original_sampled_frame_count": len(frames),
        "kept_frame_count": len(normalized_frames),
        "kept_frame_orders": selected_orders,
        "kept_source_frame_indices": [
            int(sampled_indices[order]) if order < len(sampled_indices) else int(order)
            for order in selected_orders
        ],
        "crop_to_selection": bool(crop_to_selection),
        "crop_boxes_resized_xyxy": crop_boxes,
        "output_width": int(output_size[0]),
        "output_height": int(output_size[1]),
        "note": (
            "This is diagnostic input materialization: selected frames and optional union crops "
            "are written to a new video before the downstream MLLM processor runs. It does not "
            "preserve sparse patch layout and is not exact patch-level sparse attention inside "
            "the model."
        ),
    }
    if fallback_reason:
        metadata["fallback_reason"] = fallback_reason
    return MaterializedSparseFrames(frames=normalized_frames, metadata=metadata)


def materialize_sparse_video(
    *,
    plan_path: str | Path,
    source_video: str,
    output_path: str | Path | None = None,
    sample_count: int | None = None,
    resize: dict[str, Any] | None = None,
    fps: float = 2.0,
    crop_to_selection: bool = True,
) -> dict[str, Any]:
    plan = _load_plan(plan_path)
    frame_count = int(
        sample_count
        or len(plan.source_video.sampled_frame_indices)
        or max((patch.frame_order for patch in plan.selected_patches), default=0) + 1
        or 1
    )
    frames, decode_stats = load_sampled_video_frames(
        source_video,
        frame_count,
        resize or {},
        decode_strategy="auto",
    )
    materialized = build_materialized_sparse_frames(frames, plan, crop_to_selection=crop_to_selection)
    path = Path(output_path) if output_path is not None else _default_output_path(plan_path)
    write_video(materialized.frames, path, fps=fps)
    return {
        "status": "executed",
        "path": str(path),
        "source_video": source_video,
        "sparse_selection_plan_path": str(plan_path),
        "decode_stats": decode_stats,
        **materialized.metadata,
    }


def _load_plan(path: str | Path) -> SparseSelectionPlan:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return sparse_selection_plan_from_dict(json.load(handle))


def _default_output_path(plan_path: str | Path) -> Path:
    path = Path(plan_path)
    return path.with_name(f"{path.stem}.materialized_sparse.mp4")


def _selected_frame_orders(plan: SparseSelectionPlan, frame_count: int) -> list[int]:
    orders = sorted({int(patch.frame_order) for patch in plan.selected_patches})
    return [order for order in orders if 0 <= order < frame_count]


def _union_crop_box(
    plan: SparseSelectionPlan,
    frame_order: int,
    frame_size: tuple[int, int],
) -> list[int] | None:
    boxes = [
        patch.bbox_resized_xyxy
        for patch in plan.selected_patches
        if int(patch.frame_order) == int(frame_order) and len(patch.bbox_resized_xyxy) == 4
    ]
    if not boxes:
        return None
    source_width = int(plan.preprocess_space.resized_width or frame_size[0])
    source_height = int(plan.preprocess_space.resized_height or frame_size[1])
    target_width, target_height = frame_size
    x_scale = target_width / max(source_width, 1)
    y_scale = target_height / max(source_height, 1)
    x0 = max(0, min(int(min(box[0] for box in boxes) * x_scale), target_width - 1))
    y0 = max(0, min(int(min(box[1] for box in boxes) * y_scale), target_height - 1))
    x1 = max(x0 + 1, min(int(max(box[2] for box in boxes) * x_scale), target_width))
    y1 = max(y0 + 1, min(int(max(box[3] for box in boxes) * y_scale), target_height))
    return [x0, y0, x1, y1]


def _common_output_size(frames: list[Image.Image]) -> tuple[int, int]:
    width = max(frame.size[0] for frame in frames)
    height = max(frame.size[1] for frame in frames)
    return max(width, 2), max(height, 2)
