from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from functools import wraps
from typing import Any

import numpy as np
from PIL import Image

from repro.autogaze_bench import (
    add_external_autogaze,
    autogaze_forward_kwargs,
    autogaze_transform_kwargs,
    move_model_to_device_dtype,
    parse_int_sequence,
    summarize_gaze,
    tensor_bytes,
)
from repro.common import resolve_device, synchronize, write_json
from repro.hlvid_example_autogaze import read_video_metadata, uniform_sample_indices
from repro.nvila_runner import apply_resize_to_dimensions, load_sampled_video_frames, spatial_tile_grid
from repro.plugins.gaze_plan import (
    EncoderMapping,
    MllmMapping,
    PatchSpace,
    PreprocessSpace,
    SelectedPatch,
    SourceVideo,
    SparseSelectionPlan,
)


@dataclass(frozen=True)
class AutogazeSelectorRuntimeConfig:
    video: str
    output_json: str
    autogaze_repo: str = "."
    autogaze_model: str = "nvidia/AutoGaze"
    device: str = "auto"
    dtype: str = "auto"
    num_video_frames: int = 16
    num_video_frames_thumbnail: int = 0
    qwen_thumbnail_mode: str = "none"
    chunk_frames: int = 16
    max_tiles_video: int = 1
    tile_size: int = 224
    max_batch_size: int = 16
    gazing_ratio: float | None = None
    task_loss_requirement: float | None = None
    target_scales: list[int] | None = None
    target_patch_size: int | None = 16
    encoder_patch_size: int | None = None
    generate_only: bool = False
    video_decode_strategy: str = "auto"
    video_resize_shortest_edge: int | None = None
    video_resize_longest_edge: int | None = None
    video_resize_width: int | None = None
    video_resize_height: int | None = None


def autogaze_selector_resize_enabled(config: AutogazeSelectorRuntimeConfig) -> bool:
    return any(
        value is not None
        for value in (
            config.video_resize_shortest_edge,
            config.video_resize_longest_edge,
            config.video_resize_width,
            config.video_resize_height,
        )
    )


def build_autogaze_selector_video_plan(
    config: AutogazeSelectorRuntimeConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    source_width = int(metadata["width"])
    source_height = int(metadata["height"])
    resize = apply_resize_to_dimensions(
        width=source_width,
        height=source_height,
        shortest_edge=config.video_resize_shortest_edge,
        longest_edge=config.video_resize_longest_edge,
        exact_width=config.video_resize_width,
        exact_height=config.video_resize_height,
    )
    effective_width = int(resize["width"])
    effective_height = int(resize["height"])
    grid = spatial_tile_grid(
        width=effective_width,
        height=effective_height,
        max_tiles_video=int(config.max_tiles_video),
        image_size=int(config.tile_size),
    )
    canvas_width = int(grid["cols"]) * int(config.tile_size)
    canvas_height = int(grid["rows"]) * int(config.tile_size)
    return {
        "source_width": source_width,
        "source_height": source_height,
        "effective_width": effective_width,
        "effective_height": effective_height,
        "autogaze_canvas_width": canvas_width,
        "autogaze_canvas_height": canvas_height,
        "resize": {
            "enabled": autogaze_selector_resize_enabled(config),
            "shortest_edge": config.video_resize_shortest_edge,
            "longest_edge": config.video_resize_longest_edge,
            "width": config.video_resize_width,
            "height": config.video_resize_height,
            "effective": resize,
        },
        "grid": grid,
    }


def build_sparse_selection_plan_from_autogaze_outputs(
    outputs: dict[str, Any],
    *,
    source_path: str | None,
    frame_indices: list[int],
    target_scales: list[int],
    target_patch_size: int,
    encoder_patch_size: int | None,
    resized_width: int,
    resized_height: int,
    source_width: int | None = None,
    source_height: int | None = None,
    tile_id_offset: int = 0,
    tile_grid: list[int] | None = None,
    tile_size: int | None = None,
    selector_name: str = "autogaze-direct",
    autoregressive_order_offset: int = 0,
) -> SparseSelectionPlan:
    patch_counts = _patch_counts_by_scale(target_scales, target_patch_size)
    patches_per_frame = sum(patch_counts)
    positions = _ensure_2d(outputs.get("gazing_pos"))
    padded = _ensure_2d(outputs.get("if_padded_gazing"))
    selected_patches: list[SelectedPatch] = []
    skipped_invalid_positions = 0

    for batch_index, row in enumerate(positions):
        padded_row = padded[batch_index] if batch_index < len(padded) else []
        for order_in_row, position_value in enumerate(row):
            is_padded = bool(padded_row[order_in_row]) if order_in_row < len(padded_row) else False
            if is_padded:
                continue
            absolute_position = int(position_value)
            if absolute_position < 0 or absolute_position >= len(frame_indices) * patches_per_frame:
                skipped_invalid_positions += 1
                continue

            frame_order = absolute_position // patches_per_frame
            per_frame_position = absolute_position % patches_per_frame
            scale_id, scale_size, scale_patch_index = _decode_multiscale_patch_index(
                per_frame_position,
                target_scales,
                target_patch_size,
            )
            tile_id = int(tile_id_offset + batch_index)
            bbox_resized = _bbox_for_patch(
                scale_size=scale_size,
                patch_size=target_patch_size,
                patch_index=scale_patch_index,
                resized_width=resized_width,
                resized_height=resized_height,
            )
            if tile_grid is not None and tile_size is not None:
                cols = max(1, int(tile_grid[0]))
                rows = max(1, int(tile_grid[1])) if len(tile_grid) > 1 else 1
                col = tile_id % cols
                row = tile_id // cols
                offset_x = col * int(tile_size)
                offset_y = row * int(tile_size)
                bbox_resized = [
                    int(bbox_resized[0] + offset_x),
                    int(bbox_resized[1] + offset_y),
                    int(bbox_resized[2] + offset_x),
                    int(bbox_resized[3] + offset_y),
                ]
                bbox_space_width = cols * int(tile_size)
                bbox_space_height = rows * int(tile_size)
            else:
                bbox_space_width = resized_width
                bbox_space_height = resized_height
            bbox_original = _scale_bbox_to_original(
                bbox_resized,
                resized_width=bbox_space_width,
                resized_height=bbox_space_height,
                source_width=source_width,
                source_height=source_height,
            )
            selected_patches.append(
                SelectedPatch(
                    frame_index=int(frame_indices[frame_order]),
                    frame_order=int(frame_order),
                    tile_id=tile_id,
                    scale_id=int(scale_id),
                    scale_size=int(scale_size),
                    patch_index=int(scale_patch_index),
                    bbox_resized_xyxy=bbox_resized,
                    bbox_original_xyxy=bbox_original,
                    autoregressive_order=int(autoregressive_order_offset + len(selected_patches)),
                )
            )

    raw_patch_tokens = len(positions) * len(frame_indices) * patches_per_frame
    return SparseSelectionPlan(
        selector_name=selector_name,
        source_video=SourceVideo(
            path=source_path,
            source_width=source_width,
            source_height=source_height,
            sampled_frame_indices=list(frame_indices),
        ),
        preprocess_space=PreprocessSpace(
            resize_policy=f"autogaze_largest_scale={max(target_scales)}",
            resized_width=int(resized_width),
            resized_height=int(resized_height),
            tile_grid=list(tile_grid) if tile_grid is not None else None,
            tile_size=tile_size,
        ),
        patch_space=PatchSpace(
            autogaze_patch_size=int(target_patch_size),
            encoder_patch_size=int(encoder_patch_size) if encoder_patch_size is not None else None,
            scale_ids=list(range(len(target_scales))),
            scale_sizes=[int(scale) for scale in target_scales],
        ),
        selected_patches=selected_patches,
        encoder_mapping=EncoderMapping(status="not_mapped", reason="direct AutoGaze selector emits source patch indices"),
        mllm_mapping=MllmMapping(status="not_mapped", reason="Qwen mapping is resolved after processor video_grid_thw is known"),
        raw_patch_tokens=raw_patch_tokens,
        selected_patch_tokens=len(selected_patches),
        quality_control={
            "patches_per_frame": patches_per_frame,
            "autogaze_output_keys": sorted(str(key) for key in outputs.keys()),
            "skipped_invalid_positions": skipped_invalid_positions,
        },
    )


def run_direct_autogaze_selector(config: AutogazeSelectorRuntimeConfig) -> dict[str, Any]:
    add_external_autogaze(config.autogaze_repo)

    import av
    import torch
    from autogaze.datasets.video_utils import transform_video_for_pytorch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    ensure_transformers_tied_weight_compat(AutoGaze)
    target_scales = config.target_scales or [config.tile_size]
    target_patch_size = int(config.target_patch_size or 16)
    tile_size = int(config.tile_size or target_scales[-1])
    dtype = _torch_dtype(config.dtype, config.device)
    device = resolve_device(config.device)
    metadata = read_video_metadata(config.video)
    if metadata["frames"] is None:
        raise ValueError("Video frame count is required for direct AutoGaze selector sampling")
    if config.num_video_frames <= 0:
        raise ValueError("num_video_frames must be positive")
    if config.chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if config.num_video_frames % config.chunk_frames != 0:
        raise ValueError("num_video_frames must be divisible by direct AutoGaze chunk_frames")

    sampled_indices = uniform_sample_indices(int(metadata["frames"]), int(config.num_video_frames))
    video_plan = build_autogaze_selector_video_plan(config, metadata)
    grid = video_plan["grid"]

    total_start = time.perf_counter()
    start = time.perf_counter()
    transform = AutoGazeImageProcessor.from_pretrained(
        config.autogaze_model,
        **autogaze_transform_kwargs(target_scales),
    )
    processor_load_ms = _elapsed_ms(start)
    start = time.perf_counter()
    model = move_model_to_device_dtype(AutoGaze.from_pretrained(config.autogaze_model), device, dtype)
    patch_autogaze_inputs_embeds_generate_compat(model)
    model.eval()
    model_load_ms = _elapsed_ms(start)

    decode_ms = 0.0
    tile_build_ms = 0.0
    tensorize_ms = 0.0
    forward_ms = 0.0
    peak_tensor_bytes = 0
    selected_patches: list[SelectedPatch] = []
    raw_patch_tokens = 0
    selected_patch_tokens = 0
    padded_positions = 0
    total_slots = 0
    chunk_summaries: list[dict[str, Any]] = []

    current_frames: list[Image.Image] = []
    current_indices: list[int] = []
    chunk_number = 0

    if autogaze_selector_resize_enabled(config) or config.video_decode_strategy != "auto":
        start = time.perf_counter()
        sampled_frames, decode_stats = load_sampled_video_frames(
            config.video,
            int(config.num_video_frames),
            video_plan["resize"]["effective"],
            decode_strategy=config.video_decode_strategy,
        )
        decode_ms += _elapsed_ms(start)
        for frame_index, image in zip(sampled_indices, sampled_frames):
            current_frames.append(image.copy())
            current_indices.append(int(frame_index))
            if len(current_frames) == config.chunk_frames:
                chunk_number += 1
                chunk_result = _run_autogaze_selector_chunk(
                    frames=current_frames,
                    frame_indices=current_indices,
                    chunk_number=chunk_number,
                    grid=grid,
                    tile_size=tile_size,
                    transform=transform,
                    model=model,
                    device=device,
                    dtype=dtype,
                    max_batch_size=int(config.max_batch_size),
                    gazing_ratio=None if config.gazing_ratio is None else float(config.gazing_ratio),
                    task_loss_requirement=config.task_loss_requirement,
                    target_scales=target_scales,
                    target_patch_size=target_patch_size,
                    encoder_patch_size=config.encoder_patch_size,
                    source_path=config.video,
                    source_width=int(metadata["width"]),
                    source_height=int(metadata["height"]),
                    autoregressive_order_offset=len(selected_patches),
                    generate_only=bool(config.generate_only),
                )
                selected_patches.extend(chunk_result["selected_patches"])
                raw_patch_tokens += int(chunk_result["raw_patch_tokens"])
                selected_patch_tokens += int(chunk_result["selected_patch_tokens"])
                padded_positions += int(chunk_result["padded_gazing_positions"])
                total_slots += int(chunk_result["total_gaze_slots"])
                tensorize_ms += float(chunk_result["autogaze_tensorize_ms"])
                forward_ms += float(chunk_result["autogaze_forward_ms"])
                tile_build_ms += float(chunk_result["tile_build_ms"])
                peak_tensor_bytes = max(peak_tensor_bytes, int(chunk_result["tensor_bytes_peak"]))
                chunk_summaries.append(chunk_result["summary"])
                current_frames = []
                current_indices = []
        video_plan["decode"] = decode_stats
    else:
        frame_counts = _counts_by_index(sampled_indices)
        selected_index_set = set(frame_counts)
        container = av.open(config.video)
        try:
            decoder = container.decode(video=0)
            for frame_index, frame in enumerate(decoder):
                if frame_index > sampled_indices[-1]:
                    break
                if frame_index not in selected_index_set:
                    continue
                start = time.perf_counter()
                image = frame.to_image().convert("RGB")
                decode_ms += _elapsed_ms(start)
                for _ in range(frame_counts[frame_index]):
                    current_frames.append(image.copy())
                    current_indices.append(frame_index)
                    if len(current_frames) == config.chunk_frames:
                        chunk_number += 1
                        chunk_result = _run_autogaze_selector_chunk(
                            frames=current_frames,
                            frame_indices=current_indices,
                            chunk_number=chunk_number,
                            grid=grid,
                            tile_size=tile_size,
                            transform=transform,
                            model=model,
                            device=device,
                            dtype=dtype,
                            max_batch_size=int(config.max_batch_size),
                            gazing_ratio=None if config.gazing_ratio is None else float(config.gazing_ratio),
                            task_loss_requirement=config.task_loss_requirement,
                            target_scales=target_scales,
                            target_patch_size=target_patch_size,
                            encoder_patch_size=config.encoder_patch_size,
                            source_path=config.video,
                            source_width=int(metadata["width"]),
                            source_height=int(metadata["height"]),
                            autoregressive_order_offset=len(selected_patches),
                            generate_only=bool(config.generate_only),
                        )
                        selected_patches.extend(chunk_result["selected_patches"])
                        raw_patch_tokens += int(chunk_result["raw_patch_tokens"])
                        selected_patch_tokens += int(chunk_result["selected_patch_tokens"])
                        padded_positions += int(chunk_result["padded_gazing_positions"])
                        total_slots += int(chunk_result["total_gaze_slots"])
                        tensorize_ms += float(chunk_result["autogaze_tensorize_ms"])
                        forward_ms += float(chunk_result["autogaze_forward_ms"])
                        tile_build_ms += float(chunk_result["tile_build_ms"])
                        peak_tensor_bytes = max(peak_tensor_bytes, int(chunk_result["tensor_bytes_peak"]))
                        chunk_summaries.append(chunk_result["summary"])
                        current_frames = []
                        current_indices = []
        finally:
            container.close()

    if current_frames:
        raise RuntimeError(f"Unprocessed partial direct AutoGaze chunk with {len(current_frames)} frames")

    plan = SparseSelectionPlan(
        selector_name="autogaze-direct",
        source_video=SourceVideo(
            path=config.video,
            source_width=int(metadata["width"]),
            source_height=int(metadata["height"]),
            sampled_frame_indices=sampled_indices,
            sampled_fps=metadata.get("fps"),
        ),
        preprocess_space=PreprocessSpace(
            resize_policy=f"autogaze_largest_scale={max(target_scales)}",
            resized_width=int(video_plan["autogaze_canvas_width"]),
            resized_height=int(video_plan["autogaze_canvas_height"]),
            tile_grid=[int(grid["cols"]), int(grid["rows"])],
            tile_size=tile_size,
        ),
        patch_space=PatchSpace(
            autogaze_patch_size=target_patch_size,
            encoder_patch_size=config.encoder_patch_size,
            scale_ids=list(range(len(target_scales))),
            scale_sizes=target_scales,
        ),
        selected_patches=selected_patches,
        encoder_mapping=EncoderMapping(status="not_mapped", reason="Qwen ViT mapping is resolved by video_grid_thw later"),
        mllm_mapping=MllmMapping(status="not_mapped", reason="Qwen visual placeholder mapping is resolved inside the MLLM adapter"),
        raw_patch_tokens=raw_patch_tokens,
        selected_patch_tokens=selected_patch_tokens,
        quality_control={
            "chunk_frames": int(config.chunk_frames),
            "temporal_chunks": len(chunk_summaries),
            "spatial_tiles": int(grid["tiles"]),
            "generate_only": bool(config.generate_only),
            "runner_resize": video_plan["resize"],
        },
    )
    write_json(config.output_json, plan.to_dict())
    total_ms = _elapsed_ms(total_start)
    return {
        "status": "executed",
        "sparse_selection_plan_json": str(config.output_json),
        "runtime_config": {
            "autogaze_model": config.autogaze_model,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "target_scales": target_scales,
            "target_patch_size": target_patch_size,
            "encoder_patch_size": config.encoder_patch_size,
            "num_video_frames": int(config.num_video_frames),
            "num_video_frames_thumbnail": int(config.num_video_frames_thumbnail),
            "qwen_thumbnail_mode": config.qwen_thumbnail_mode,
            "chunk_frames": int(config.chunk_frames),
            "max_tiles_video": int(config.max_tiles_video),
            "max_batch_size": int(config.max_batch_size),
            "gazing_ratio": config.gazing_ratio,
            "task_loss_requirement": config.task_loss_requirement,
            "generate_only": bool(config.generate_only),
            "video_resize": video_plan["resize"],
        },
        "source_metadata": metadata,
        "preprocess_video_plan": video_plan,
        "tokens": {
            "raw_patch_tokens": raw_patch_tokens,
            "selected_patch_tokens": selected_patch_tokens,
            "padded_gazing_positions": padded_positions,
            "total_gaze_slots": total_slots,
            "reduction_ratio": raw_patch_tokens / selected_patch_tokens if selected_patch_tokens else None,
        },
        "latency_ms": {
            "processor_load": processor_load_ms,
            "model_load": model_load_ms,
            "video_decode_selected_frames": decode_ms,
            "tile_build": tile_build_ms,
            "autogaze_tensorize": tensorize_ms,
            "autogaze_forward": forward_ms,
            "total": total_ms,
        },
        "memory_bytes": {
            "tile_tensor_peak": peak_tensor_bytes,
        },
        "temporal_chunk_summaries": chunk_summaries,
    }


def ensure_transformers_tied_weight_compat(model_cls: Any) -> None:
    """Patch older local AutoGaze classes for newer Transformers loaders."""
    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = {}
    if not hasattr(model_cls, "_tied_weights_keys"):
        model_cls._tied_weights_keys = []


def patch_autogaze_inputs_embeds_generate_compat(model: Any) -> bool:
    """Patch AutoGaze decoder generation for newer Transformers inputs_embeds handling."""
    gaze_decoder = getattr(getattr(model, "gazing_model", None), "gaze_decoder", None)
    if gaze_decoder is None or getattr(gaze_decoder, "_autogaze_inputs_embeds_generate_compat", False):
        return False
    original_generate = gaze_decoder.generate

    @wraps(original_generate)
    def generate_with_inputs_embeds_prefix(*args: Any, **kwargs: Any) -> Any:
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None or kwargs.get("input_ids") is not None:
            return original_generate(*args, **kwargs)
        prefix_len = int(inputs_embeds.shape[1])
        if prefix_len <= 0:
            return original_generate(*args, **kwargs)
        max_new_tokens = kwargs.get("max_new_tokens")
        kwargs["input_ids"] = _dummy_generation_input_ids(inputs_embeds)
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = prefix_len + _int_scalar(max_new_tokens)
        output = original_generate(*args, **kwargs)
        sequences = getattr(output, "sequences", None)
        if sequences is not None and int(sequences.shape[1]) >= prefix_len:
            output.sequences = sequences[:, prefix_len:]
        return output

    gaze_decoder.generate = generate_with_inputs_embeds_prefix
    gaze_decoder._autogaze_inputs_embeds_generate_compat = True
    return True


def _dummy_generation_input_ids(inputs_embeds: Any) -> Any:
    import torch

    return torch.zeros(
        (int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])),
        dtype=torch.long,
        device=inputs_embeds.device,
    )


def _int_scalar(value: Any) -> int:
    if hasattr(value, "detach"):
        return int(value.detach().cpu().item())
    return int(value)


def runtime_config_from_args(args: Any) -> AutogazeSelectorRuntimeConfig:
    output_json = getattr(args, "autogaze_selector_output_json", None) or getattr(args, "sparse_selection_plan_json", None)
    if not output_json:
        output_path = Path(getattr(args, "output_json", "outputs/autogaze_repro/flexible_runner.json"))
        output_json = str(output_path.with_name(f"{output_path.stem}_autogaze_sparse_plan.json"))
    target_scales = parse_int_sequence(getattr(args, "autogaze_target_scales", None)) or [224]
    return AutogazeSelectorRuntimeConfig(
        video=str(getattr(args, "video", "") or ""),
        output_json=str(output_json),
        autogaze_repo=str(getattr(args, "autogaze_repo", ".")),
        autogaze_model=str(getattr(args, "autogaze_model", None) or getattr(args, "token_selector_path", None) or "nvidia/AutoGaze"),
        device=str(getattr(args, "autogaze_device", "auto")),
        dtype=str(getattr(args, "autogaze_dtype", "auto")),
        num_video_frames=int(getattr(args, "num_video_frames", 16)),
        num_video_frames_thumbnail=int(getattr(args, "num_video_frames_thumbnail", 0)),
        qwen_thumbnail_mode=str(getattr(args, "qwen_thumbnail_mode", "none")),
        chunk_frames=int(getattr(args, "autogaze_chunk_frames", 16)),
        max_tiles_video=int(getattr(args, "max_tiles_video", 1)),
        tile_size=int(getattr(args, "autogaze_tile_size", None) or max(target_scales)),
        max_batch_size=int(getattr(args, "max_batch_size_autogaze", 16)),
        gazing_ratio=getattr(args, "gazing_ratio", None),
        task_loss_requirement=getattr(args, "task_loss_requirement", None),
        target_scales=target_scales,
        target_patch_size=int(getattr(args, "autogaze_target_patch_size", 16)),
        encoder_patch_size=(
            int(getattr(args, "autogaze_encoder_patch_size"))
            if getattr(args, "autogaze_encoder_patch_size", None) is not None
            else None
        ),
        generate_only=bool(getattr(args, "autogaze_generate_only", False)),
        video_decode_strategy=str(getattr(args, "video_decode_strategy", "auto")),
        video_resize_shortest_edge=(
            int(getattr(args, "video_resize_shortest_edge"))
            if getattr(args, "video_resize_shortest_edge", None) is not None
            else None
        ),
        video_resize_longest_edge=(
            int(getattr(args, "video_resize_longest_edge"))
            if getattr(args, "video_resize_longest_edge", None) is not None
            else None
        ),
        video_resize_width=(
            int(getattr(args, "video_resize_width")) if getattr(args, "video_resize_width", None) is not None else None
        ),
        video_resize_height=(
            int(getattr(args, "video_resize_height")) if getattr(args, "video_resize_height", None) is not None else None
        ),
    )


def _run_autogaze_selector_chunk(
    *,
    frames: list[Image.Image],
    frame_indices: list[int],
    chunk_number: int,
    grid: dict[str, int],
    tile_size: int,
    transform: Any,
    model: Any,
    device: Any,
    dtype: Any,
    max_batch_size: int,
    gazing_ratio: float,
    task_loss_requirement: float | None,
    target_scales: list[int],
    target_patch_size: int,
    encoder_patch_size: int | None,
    source_path: str,
    source_width: int,
    source_height: int,
    autoregressive_order_offset: int,
    generate_only: bool,
) -> dict[str, Any]:
    import torch
    from autogaze.datasets.video_utils import transform_video_for_pytorch

    start = time.perf_counter()
    tile_sequences = _build_spatial_tile_sequences(
        frames,
        cols=int(grid["cols"]),
        rows=int(grid["rows"]),
        tile_size=tile_size,
    )
    tile_build_ms = _elapsed_ms(start)

    patch_counts = _patch_counts_by_scale(target_scales, target_patch_size)
    patches_per_frame = sum(patch_counts)
    raw_patch_tokens = 0
    selected_patch_tokens = 0
    padded_positions = 0
    total_slots = 0
    tensorize_ms = 0.0
    forward_ms = 0.0
    tensor_bytes_peak = 0
    selected_patches: list[SelectedPatch] = []

    with torch.inference_mode():
        for start_index in range(0, len(tile_sequences), max_batch_size):
            batch_sequences = tile_sequences[start_index : start_index + max_batch_size]
            start = time.perf_counter()
            flat_tiles = [np.array(frame) for sequence in batch_sequences for frame in sequence]
            transformed = transform_video_for_pytorch(np.stack(flat_tiles), transform)
            transformed = transformed.reshape(len(batch_sequences), len(batch_sequences[0]), *transformed.shape[1:])
            tensorize_ms += _elapsed_ms(start)
            tensor_bytes_peak = max(tensor_bytes_peak, tensor_bytes(transformed))

            batch = transformed.to(device=device, dtype=dtype)
            raw_budget = int(batch.shape[0] * batch.shape[1] * patches_per_frame)
            synchronize(device)
            start = time.perf_counter()
            outputs = model(
                {"video": batch},
                **autogaze_forward_kwargs(
                    gazing_ratio=gazing_ratio,
                    task_loss_requirement=task_loss_requirement,
                    target_scales=target_scales,
                    target_patch_size=target_patch_size,
                    generate_only=generate_only,
                ),
            )
            synchronize(device)
            forward_ms += _elapsed_ms(start)
            summary = summarize_gaze(outputs, raw_budget)
            raw_patch_tokens += raw_budget
            selected_patch_tokens += int(summary["selected_non_padded_patches"])
            padded_positions += int(summary["padded_gazing_positions"])
            total_slots += int(summary["total_gaze_slots"])
            plan = build_sparse_selection_plan_from_autogaze_outputs(
                outputs,
                source_path=source_path,
                frame_indices=frame_indices,
                target_scales=target_scales,
                target_patch_size=target_patch_size,
                encoder_patch_size=encoder_patch_size,
                resized_width=tile_size,
                resized_height=tile_size,
                source_width=source_width,
                source_height=source_height,
                tile_id_offset=start_index,
                tile_grid=[int(grid["cols"]), int(grid["rows"])],
                tile_size=tile_size,
                autoregressive_order_offset=autoregressive_order_offset + len(selected_patches),
            )
            selected_patches.extend(plan.selected_patches)

    return {
        "selected_patches": selected_patches,
        "raw_patch_tokens": raw_patch_tokens,
        "selected_patch_tokens": selected_patch_tokens,
        "padded_gazing_positions": padded_positions,
        "total_gaze_slots": total_slots,
        "tile_build_ms": tile_build_ms,
        "autogaze_tensorize_ms": tensorize_ms,
        "autogaze_forward_ms": forward_ms,
        "tensor_bytes_peak": tensor_bytes_peak,
        "summary": {
            "chunk_number": chunk_number,
            "sampled_frame_start": frame_indices[0],
            "sampled_frame_end": frame_indices[-1],
            "tile_sequences": len(tile_sequences),
            "raw_patch_tokens": raw_patch_tokens,
            "selected_patch_tokens": selected_patch_tokens,
            "token_reduction_ratio": raw_patch_tokens / selected_patch_tokens if selected_patch_tokens else None,
            "autogaze_forward_ms": forward_ms,
        },
    }


def _build_spatial_tile_sequences(
    frames: list[Image.Image],
    *,
    cols: int,
    rows: int,
    tile_size: int,
) -> list[list[Image.Image]]:
    sequences: list[list[Image.Image]] = [[] for _ in range(cols * rows)]
    target_size = (cols * tile_size, rows * tile_size)
    for frame in frames:
        resized = frame.resize(target_size)
        for tile_idx in range(cols * rows):
            col = tile_idx % cols
            row = tile_idx // cols
            box = (col * tile_size, row * tile_size, (col + 1) * tile_size, (row + 1) * tile_size)
            sequences[tile_idx].append(resized.crop(box))
    return sequences


def _patch_counts_by_scale(scales: list[int], patch_size: int) -> list[int]:
    counts: list[int] = []
    for scale in scales:
        if int(scale) % int(patch_size) != 0:
            raise ValueError(f"AutoGaze scale {scale} is not divisible by patch size {patch_size}")
        grid = int(scale) // int(patch_size)
        counts.append(grid * grid)
    return counts


def _decode_multiscale_patch_index(
    per_frame_position: int,
    scales: list[int],
    patch_size: int,
) -> tuple[int, int, int]:
    offset = 0
    for scale_id, (scale, count) in enumerate(zip(scales, _patch_counts_by_scale(scales, patch_size))):
        if per_frame_position < offset + count:
            return scale_id, int(scale), int(per_frame_position - offset)
        offset += count
    raise ValueError(f"Patch index {per_frame_position} is outside scales={scales}")


def _bbox_for_patch(
    *,
    scale_size: int,
    patch_size: int,
    patch_index: int,
    resized_width: int,
    resized_height: int,
) -> list[int]:
    grid = int(scale_size) // int(patch_size)
    row = int(patch_index) // grid
    col = int(patch_index) % grid
    bbox_at_scale = [
        col * patch_size,
        row * patch_size,
        (col + 1) * patch_size,
        (row + 1) * patch_size,
    ]
    x_scale = float(resized_width) / float(scale_size)
    y_scale = float(resized_height) / float(scale_size)
    return [
        int(round(bbox_at_scale[0] * x_scale)),
        int(round(bbox_at_scale[1] * y_scale)),
        int(round(bbox_at_scale[2] * x_scale)),
        int(round(bbox_at_scale[3] * y_scale)),
    ]


def _scale_bbox_to_original(
    bbox: list[int],
    *,
    resized_width: int,
    resized_height: int,
    source_width: int | None,
    source_height: int | None,
) -> list[float]:
    if not source_width or not source_height:
        return [float(value) for value in bbox]
    x_scale = float(source_width) / float(resized_width)
    y_scale = float(source_height) / float(resized_height)
    return [
        float(bbox[0]) * x_scale,
        float(bbox[1]) * y_scale,
        float(bbox[2]) * x_scale,
        float(bbox[3]) * y_scale,
    ]


def _ensure_2d(value: Any) -> list[list[Any]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value is None:
        return []
    if not isinstance(value, list):
        return [[value]]
    if not value:
        return []
    if isinstance(value[0], list):
        if value and value[0] and isinstance(value[0][0], list):
            return [[item for sub in row for item in sub] for row in value]
        return value
    return [value]


def _counts_by_index(indices: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for index in indices:
        counts[int(index)] = counts.get(int(index), 0) + 1
    return counts


def _torch_dtype(dtype_name: str, device_name: str) -> Any:
    import torch

    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "auto":
        resolved = resolve_device(device_name)
        return torch.float16 if getattr(resolved, "type", "") == "cuda" else torch.float32
    raise ValueError(f"Unsupported AutoGaze dtype: {dtype_name}")


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
