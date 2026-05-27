from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from repro.plugins.gaze_plan import SparseSelectionPlan


@dataclass(frozen=True)
class VjepaGridConfig:
    frames_per_clip: int
    tubelet_size: int = 2
    crop_size: int = 224
    patch_size: int = 16

    @property
    def grid_t(self) -> int:
        if self.tubelet_size <= 0:
            raise ValueError("tubelet_size must be positive")
        if self.frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be positive")
        if self.frames_per_clip % self.tubelet_size != 0:
            raise ValueError("frames_per_clip must be divisible by tubelet_size")
        return self.frames_per_clip // self.tubelet_size

    @property
    def grid_h(self) -> int:
        return _spatial_grid(self.crop_size, self.patch_size)

    @property
    def grid_w(self) -> int:
        return _spatial_grid(self.crop_size, self.patch_size)

    @property
    def grid_thw(self) -> list[int]:
        return [self.grid_t, self.grid_h, self.grid_w]

    @property
    def raw_token_count(self) -> int:
        return self.grid_t * self.grid_h * self.grid_w

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "frames_per_clip": int(self.frames_per_clip),
            "tubelet_size": int(self.tubelet_size),
            "crop_size": int(self.crop_size),
            "patch_size": int(self.patch_size),
            "grid_thw": self.grid_thw,
            "raw_token_count": self.raw_token_count,
        }


@dataclass(frozen=True)
class VjepaTokenSelection:
    status: str
    grid_config: VjepaGridConfig
    selected_token_indices: list[int]
    selected_tokens_by_scale: dict[str, int] = field(default_factory=dict)
    mapping_policy: dict[str, Any] = field(default_factory=dict)
    token_records: list[dict[str, Any]] = field(default_factory=list)
    scale_passes: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_token_count_override: int | None = None
    reason: str | None = None

    @property
    def grid_thw(self) -> list[int]:
        return self.grid_config.grid_thw

    @property
    def raw_token_count(self) -> int:
        if self.raw_token_count_override is not None:
            return int(self.raw_token_count_override)
        return self.grid_config.raw_token_count

    @property
    def selected_token_count(self) -> int:
        return len(self.selected_token_indices)

    @property
    def reduction_ratio(self) -> float | None:
        if not self.selected_token_indices:
            return None
        return float(self.raw_token_count) / float(len(self.selected_token_indices))

    def to_dict(self) -> dict[str, Any]:
        return {
            "vjepa": {
                "status": self.status,
                "grid_config": self.grid_config.to_dict(),
                "grid_thw": self.grid_thw,
                "raw_token_count": self.raw_token_count,
                "selected_token_count": self.selected_token_count,
                "selected_token_indices": [int(value) for value in self.selected_token_indices],
                "selected_tokens_by_scale": dict(self.selected_tokens_by_scale),
                "reduction_ratio": self.reduction_ratio,
                "mapping_policy": _json_ready(self.mapping_policy),
                "token_records": _json_ready(self.token_records),
                "scale_passes": _json_ready(self.scale_passes),
                "reason": self.reason,
            }
        }


def dense_vjepa_token_selection(grid_config: VjepaGridConfig) -> VjepaTokenSelection:
    selected = list(range(grid_config.raw_token_count))
    return VjepaTokenSelection(
        status="dense_keep_all",
        grid_config=grid_config,
        selected_token_indices=selected,
        selected_tokens_by_scale={"dense": len(selected)},
        mapping_policy={
            "selector": "off",
            "temporal": "all_tubelets",
            "spatial": "all_grid_cells",
            "multiscale": "not_applicable",
        },
        reason=f"AutoGaze disabled; keeping all {len(selected)} V-JEPA tokens",
    )


def vjepa_token_selection_from_sparse_plan(
    plan: SparseSelectionPlan,
    grid_config: VjepaGridConfig,
    *,
    overlap_threshold: float = 0.0,
) -> VjepaTokenSelection:
    selected_indices: set[int] = set()
    scale_to_indices: dict[str, set[int]] = {}
    records: list[dict[str, Any]] = []
    skipped = 0

    for patch in sorted(plan.selected_patches, key=lambda item: int(item.autoregressive_order)):
        tubelet_index = _tubelet_index_for_patch(plan, int(patch.frame_order), grid_config)
        cells = _vjepa_cells_for_patch_bbox(plan, patch.bbox_resized_xyxy, grid_config, overlap_threshold)
        if not cells:
            skipped += 1
            continue
        patch_indices: list[int] = []
        for row, col in cells:
            index = tubelet_index * grid_config.grid_h * grid_config.grid_w + row * grid_config.grid_w + col
            selected_indices.add(index)
            patch_indices.append(index)
            scale_to_indices.setdefault(str(int(patch.scale_id)), set()).add(index)
        records.append(
            {
                "frame_order": int(patch.frame_order),
                "tubelet_index": tubelet_index,
                "scale_id": int(patch.scale_id),
                "scale_size": int(patch.scale_size),
                "patch_index": int(patch.patch_index),
                "vjepa_token_indices": patch_indices,
            }
        )

    sorted_indices = sorted(selected_indices)
    status = "mapped" if sorted_indices else "mapping_failed"
    reason = None
    if not sorted_indices:
        reason = "no AutoGaze selected patch overlapped the V-JEPA token grid"
    elif skipped:
        reason = f"mapped {len(sorted_indices)} V-JEPA tokens and skipped {skipped} patches without overlap"
    else:
        reason = f"mapped {len(sorted_indices)} V-JEPA tokens from {len(plan.selected_patches)} AutoGaze patches"

    return VjepaTokenSelection(
        status=status,
        grid_config=grid_config,
        selected_token_indices=sorted_indices,
        selected_tokens_by_scale={scale: len(indices) for scale, indices in sorted(scale_to_indices.items())},
        mapping_policy={
            "temporal": "tubelet",
            "tubelet": "any_frame_selected",
            "spatial": "bbox_overlap_union",
            "multiscale": "scale_bbox_expansion",
            "overlap_threshold": float(overlap_threshold),
            "coordinate_space": "plan.preprocess_space resized bbox -> square V-JEPA crop",
        },
        token_records=records,
        reason=reason,
    )


def scale_aware_vjepa_selection_from_sparse_plan(
    plan: SparseSelectionPlan,
    *,
    frames_per_clip: int,
    tubelet_size: int = 2,
    patch_size: int = 16,
    overlap_threshold: float = 0.0,
) -> VjepaTokenSelection:
    scale_sizes = _scale_sizes_for_plan(plan)
    selected_indices: list[int] = []
    selected_by_scale: dict[str, int] = {}
    scale_passes: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    raw_offset = 0

    for scale_id, scale_size in scale_sizes:
        scale_key = str(int(scale_id))
        scale_plan = _plan_subset_for_scale(plan, scale_id)
        grid_config = VjepaGridConfig(
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            crop_size=int(scale_size),
            patch_size=patch_size,
        )
        selection = vjepa_token_selection_from_sparse_plan(
            scale_plan,
            grid_config,
            overlap_threshold=overlap_threshold,
        )
        offset_indices = [raw_offset + index for index in selection.selected_token_indices]
        selected_indices.extend(offset_indices)
        selected_by_scale[scale_key] = len(offset_indices)
        scale_passes[scale_key] = {
            "scale_size": int(scale_size),
            "grid_thw": grid_config.grid_thw,
            "raw_token_count": grid_config.raw_token_count,
            "selected_token_count": len(offset_indices),
            "selected_token_indices": offset_indices,
            "local_selected_token_indices": selection.selected_token_indices,
        }
        records.extend(
            {
                **record,
                "scale_pass_offset": raw_offset,
                "scale_aware_token_indices": [raw_offset + index for index in record["vjepa_token_indices"]],
            }
            for record in selection.token_records
        )
        raw_offset += grid_config.raw_token_count

    status = "mapped" if selected_indices else "mapping_failed"
    return VjepaTokenSelection(
        status=status,
        grid_config=VjepaGridConfig(
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            crop_size=max((scale for _, scale in scale_sizes), default=patch_size),
            patch_size=patch_size,
        ),
        selected_token_indices=sorted(set(selected_indices)),
        selected_tokens_by_scale=selected_by_scale,
        mapping_policy={
            "temporal": "tubelet",
            "tubelet": "any_frame_selected",
            "spatial": "scale_local_bbox_overlap_union",
            "multiscale": "separate_vjepa_pass_per_autogaze_scale",
            "overlap_threshold": float(overlap_threshold),
            "coordinate_space": "plan resized bbox -> each AutoGaze scale crop",
        },
        token_records=records,
        scale_passes=scale_passes,
        raw_token_count_override=raw_offset,
        reason=(
            f"mapped {len(set(selected_indices))} scale-aware V-JEPA tokens across {len(scale_passes)} scale passes"
            if selected_indices
            else "no scale-aware V-JEPA tokens were selected"
        ),
    )


def _tubelet_index_for_patch(plan: SparseSelectionPlan, frame_order: int, grid_config: VjepaGridConfig) -> int:
    sampled_count = len(plan.source_video.sampled_frame_indices)
    if sampled_count > 1 and sampled_count != grid_config.frames_per_clip:
        clip_frame_index = int(round(frame_order * (grid_config.frames_per_clip - 1) / (sampled_count - 1)))
    else:
        clip_frame_index = int(frame_order)
    clip_frame_index = min(max(clip_frame_index, 0), grid_config.frames_per_clip - 1)
    return min(clip_frame_index // grid_config.tubelet_size, grid_config.grid_t - 1)


def _scale_sizes_for_plan(plan: SparseSelectionPlan) -> list[tuple[int, int]]:
    scale_ids_by_size: dict[int, int] = {}
    for patch in plan.selected_patches:
        scale_ids_by_size.setdefault(int(patch.scale_size), int(patch.scale_id))
    for fallback_id, scale_size in enumerate(plan.patch_space.scale_sizes):
        scale_ids_by_size.setdefault(int(scale_size), int(fallback_id))
    return sorted((scale_id, scale_size) for scale_size, scale_id in scale_ids_by_size.items())


def _plan_subset_for_scale(plan: SparseSelectionPlan, scale_id: int) -> SparseSelectionPlan:
    return SparseSelectionPlan(
        selector_name=plan.selector_name,
        source_video=plan.source_video,
        preprocess_space=plan.preprocess_space,
        patch_space=plan.patch_space,
        selected_patches=[patch for patch in plan.selected_patches if int(patch.scale_id) == int(scale_id)],
        encoder_mapping=plan.encoder_mapping,
        mllm_mapping=plan.mllm_mapping,
        raw_patch_tokens=plan.raw_patch_tokens,
        selected_patch_tokens=plan.selected_patch_tokens,
        dense_masks=plan.dense_masks,
        quality_control=plan.quality_control,
    )


def _vjepa_cells_for_patch_bbox(
    plan: SparseSelectionPlan,
    bbox_resized_xyxy: list[int],
    grid_config: VjepaGridConfig,
    overlap_threshold: float,
) -> list[tuple[int, int]]:
    if len(bbox_resized_xyxy) != 4:
        return []
    resized_width = max(float(plan.preprocess_space.resized_width or grid_config.crop_size), 1.0)
    resized_height = max(float(plan.preprocess_space.resized_height or grid_config.crop_size), 1.0)
    scale_x = float(grid_config.crop_size) / resized_width
    scale_y = float(grid_config.crop_size) / resized_height
    x1, y1, x2, y2 = [float(value) for value in bbox_resized_xyxy]
    crop_bbox = [
        _clamp(x1 * scale_x, 0.0, float(grid_config.crop_size)),
        _clamp(y1 * scale_y, 0.0, float(grid_config.crop_size)),
        _clamp(x2 * scale_x, 0.0, float(grid_config.crop_size)),
        _clamp(y2 * scale_y, 0.0, float(grid_config.crop_size)),
    ]
    if crop_bbox[2] <= crop_bbox[0] or crop_bbox[3] <= crop_bbox[1]:
        return []

    cells: list[tuple[int, int]] = []
    cell = float(grid_config.patch_size)
    cell_area = cell * cell
    for row in range(grid_config.grid_h):
        cy1 = row * cell
        cy2 = cy1 + cell
        for col in range(grid_config.grid_w):
            cx1 = col * cell
            cx2 = cx1 + cell
            overlap = _intersection_area(crop_bbox, [cx1, cy1, cx2, cy2])
            if overlap <= 0:
                continue
            if overlap_threshold > 0 and overlap / cell_area < overlap_threshold:
                continue
            cells.append((row, col))
    return cells


def _spatial_grid(crop_size: int, patch_size: int) -> int:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if crop_size % patch_size != 0:
        raise ValueError("crop_size must be divisible by patch_size")
    return crop_size // patch_size


def _intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    return value
