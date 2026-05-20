from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import torch
from PIL import Image

from repro.autogaze_bench import add_external_autogaze, move_model_to_device_dtype, summarize_gaze, tensor_bytes
from repro.common import resolve_device, synchronize, write_json
from repro.nvila_runner import spatial_tile_grid

DEFAULT_HLVID_EXAMPLE_VIDEO = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"


def uniform_sample_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if total_frames == 1:
        return [0] * sample_count
    if sample_count == 1:
        return [0]
    return [int(round(i * (total_frames - 1) / (sample_count - 1))) for i in range(sample_count)]


def chunked(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def nvila_thumbnail_indices(sampled_frame_indices: list[int], thumbnail_count: int) -> list[int]:
    if thumbnail_count <= 0:
        return []
    if len(sampled_frame_indices) > thumbnail_count:
        step = len(sampled_frame_indices) // thumbnail_count
        return sampled_frame_indices[::step][:thumbnail_count]
    return list(sampled_frame_indices)


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    raw_patch_budget = sum(int(chunk["raw_patch_budget"]) for chunk in chunks)
    selected = sum(int(chunk["selected_non_padded_patches"]) for chunk in chunks)
    padded = sum(int(chunk["padded_gazing_positions"]) for chunk in chunks)
    total_slots = sum(int(chunk["total_gaze_slots"]) for chunk in chunks)
    tensorize_ms = sum(float(chunk.get("autogaze_tensorize_ms", 0.0)) for chunk in chunks)
    forward_ms = sum(float(chunk["autogaze_forward_ms"]) for chunk in chunks)
    tile_sequences = sum(int(chunk["tile_sequences"]) for chunk in chunks)
    return {
        "tile_sequences": tile_sequences,
        "raw_patch_budget": raw_patch_budget,
        "selected_non_padded_patches": selected,
        "padded_gazing_positions": padded,
        "total_gaze_slots": total_slots,
        "token_reduction_ratio": raw_patch_budget / selected if selected else 0.0,
        "autogaze_tensorize_ms": tensorize_ms,
        "autogaze_forward_ms": forward_ms,
    }


def read_video_metadata(video: str) -> dict[str, Any]:
    container = av.open(video)
    try:
        stream = container.streams.video[0]
        return {
            "width": int(stream.width),
            "height": int(stream.height),
            "frames": int(stream.frames) if stream.frames else None,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "duration_seconds": float(stream.duration * stream.time_base) if stream.duration else None,
            "codec": stream.codec_context.name,
        }
    finally:
        container.close()


def build_spatial_tile_sequences(
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
            box = (
                col * tile_size,
                row * tile_size,
                (col + 1) * tile_size,
                (row + 1) * tile_size,
            )
            sequences[tile_idx].append(resized.crop(box))
    return sequences


def run_autogaze_on_tile_sequences(
    *,
    tile_sequences: list[list[Image.Image]],
    transform: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    gazing_ratio: float,
    task_loss_requirement: float | None,
) -> tuple[dict[str, Any], int]:
    from autogaze.datasets.video_utils import transform_video_for_pytorch

    flat_tiles = [np.array(frame) for sequence in tile_sequences for frame in sequence]
    start = time.perf_counter()
    transformed = transform_video_for_pytorch(np.stack(flat_tiles), transform)
    transformed = transformed.reshape(len(tile_sequences), len(tile_sequences[0]), *transformed.shape[1:])
    tensorize_ms = (time.perf_counter() - start) * 1000.0

    raw_patch_budget = 0
    selected = 0
    padded = 0
    total_slots = 0
    forward_ms = 0.0

    with torch.inference_mode():
        for batch in torch.split(transformed, max_batch_size):
            batch = batch.to(device=device, dtype=dtype)
            raw_budget = int(batch.shape[0] * batch.shape[1] * model.num_vision_tokens_each_frame)
            synchronize(device)
            start = time.perf_counter()
            outputs = model(
                {"video": batch},
                gazing_ratio=gazing_ratio,
                task_loss_requirement=task_loss_requirement,
            )
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            summary = summarize_gaze(outputs, raw_budget)
            raw_patch_budget += raw_budget
            selected += int(summary["selected_non_padded_patches"])
            padded += int(summary["padded_gazing_positions"])
            total_slots += int(summary["total_gaze_slots"])
            forward_ms += elapsed_ms

    return (
        {
            "tile_sequences": len(tile_sequences),
            "raw_patch_budget": raw_patch_budget,
            "selected_non_padded_patches": selected,
            "padded_gazing_positions": padded,
            "total_gaze_slots": total_slots,
            "token_reduction_ratio": raw_patch_budget / selected if selected else 0.0,
            "autogaze_tensorize_ms": tensorize_ms,
            "autogaze_forward_ms": forward_ms,
        },
        tensor_bytes(transformed),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    add_external_autogaze(args.autogaze_repo)

    from autogaze.datasets.video_utils import transform_video_for_pytorch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    metadata = read_video_metadata(args.video)
    if metadata["frames"] is None:
        raise ValueError("Video frame count is required for uniform sampling")
    if args.num_video_frames % args.chunk_frames != 0:
        raise ValueError("--num-video-frames must be divisible by --chunk-frames")

    sampled_indices = uniform_sample_indices(metadata["frames"], args.num_video_frames)
    thumbnail_indices = set(nvila_thumbnail_indices(sampled_indices, args.num_video_frames_thumbnail))
    grid = spatial_tile_grid(
        width=metadata["width"],
        height=metadata["height"],
        max_tiles_video=args.max_tiles_video,
        image_size=args.tile_size,
    )

    transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    model = move_model_to_device_dtype(AutoGaze.from_pretrained(args.autogaze_model), device, dtype)
    model.eval()

    target_counts = Counter(sampled_indices)
    selected_index_set = set(target_counts)
    end_index = max(selected_index_set)
    temporal_chunks: list[dict[str, Any]] = []
    current_frames: list[Image.Image] = []
    thumbnails: list[Image.Image] = []
    thumbnail_resize_ms = 0.0
    decode_scan_ms = 0.0
    tile_build_ms = 0.0
    tile_tensor_bytes_peak = 0
    decoded_selected_frames = 0

    print(
        json.dumps(
            {
                "event": "start",
                "video": args.video,
                "source_frames": metadata["frames"],
                "num_video_frames": args.num_video_frames,
                "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
                "spatial_tiles": grid["tiles"],
                "expected_tile_sequences": (args.num_video_frames // args.chunk_frames) * grid["tiles"],
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )

    container = av.open(args.video)
    try:
        decoder = container.decode(video=0)
        frame_index = 0
        while frame_index <= end_index:
            start = time.perf_counter()
            try:
                frame = next(decoder)
            except StopIteration:
                break
            decode_scan_ms += (time.perf_counter() - start) * 1000.0

            if frame_index in selected_index_set:
                image = frame.to_image().convert("RGB")
                for _ in range(target_counts[frame_index]):
                    frame_copy = image.copy()
                    current_frames.append(frame_copy)
                    decoded_selected_frames += 1
                    if frame_index in thumbnail_indices:
                        start_thumb = time.perf_counter()
                        thumbnails.append(frame_copy.resize((args.tile_size, args.tile_size)))
                        thumbnail_resize_ms += (time.perf_counter() - start_thumb) * 1000.0

                    if len(current_frames) == args.chunk_frames:
                        start_tiles = time.perf_counter()
                        tile_sequences = build_spatial_tile_sequences(
                            current_frames,
                            cols=grid["cols"],
                            rows=grid["rows"],
                            tile_size=args.tile_size,
                        )
                        tile_build_ms += (time.perf_counter() - start_tiles) * 1000.0
                        chunk_summary, tile_tensor_bytes = run_autogaze_on_tile_sequences(
                            tile_sequences=tile_sequences,
                            transform=transform,
                            model=model,
                            device=device,
                            dtype=dtype,
                            max_batch_size=args.max_batch_size_autogaze,
                            gazing_ratio=args.gazing_ratio,
                            task_loss_requirement=args.task_loss_requirement,
                        )
                        chunk_summary["sampled_frame_start"] = sampled_indices[
                            len(temporal_chunks) * args.chunk_frames
                        ]
                        chunk_summary["sampled_frame_end"] = sampled_indices[
                            len(temporal_chunks) * args.chunk_frames + args.chunk_frames - 1
                        ]
                        temporal_chunks.append(chunk_summary)
                        print(
                            json.dumps(
                                {
                                    "event": "chunk_done",
                                    "chunk": len(temporal_chunks),
                                    "sampled_frame_start": chunk_summary["sampled_frame_start"],
                                    "sampled_frame_end": chunk_summary["sampled_frame_end"],
                                    "tile_sequences": chunk_summary["tile_sequences"],
                                    "selected_non_padded_patches": chunk_summary["selected_non_padded_patches"],
                                    "token_reduction_ratio": chunk_summary["token_reduction_ratio"],
                                    "autogaze_forward_ms": chunk_summary["autogaze_forward_ms"],
                                },
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        tile_tensor_bytes_peak = max(tile_tensor_bytes_peak, tile_tensor_bytes)
                        current_frames = []
            frame_index += 1
    finally:
        container.close()

    if current_frames:
        raise RuntimeError(f"Unprocessed partial chunk with {len(current_frames)} frames")

    start = time.perf_counter()
    thumbnail_tensor = (
        transform_video_for_pytorch(np.stack([np.array(frame) for frame in thumbnails]), transform)
        if thumbnails
        else None
    )
    thumbnail_tensorize_ms = (time.perf_counter() - start) * 1000.0
    thumbnail_tensor_bytes = tensor_bytes(thumbnail_tensor) if thumbnail_tensor is not None else 0

    tile_summary = summarize_chunks(temporal_chunks)
    result = {
        "video": args.video,
        "source_metadata": metadata,
        "device": str(device),
        "dtype": args.dtype,
        "autogaze_model": args.autogaze_model,
        "sampling": {
            "policy": "nvila_round_linspace_over_full_video",
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "chunk_frames": args.chunk_frames,
            "sampled_frame_start": sampled_indices[0],
            "sampled_frame_end": sampled_indices[-1],
            "decoded_selected_frames": decoded_selected_frames,
            "thumbnail_frames_processed": len(thumbnails),
        },
        "tiling": {
            "tile_size": args.tile_size,
            "cols": grid["cols"],
            "rows": grid["rows"],
            "spatial_tiles": grid["tiles"],
            "temporal_chunks": len(temporal_chunks),
            "tile_sequences": tile_summary["tile_sequences"],
        },
        "gaze": tile_summary,
        "thumbnail_keep_all": {
            "frames": len(thumbnails),
            "raw_patch_budget": len(thumbnails) * model.num_vision_tokens_each_frame,
            "tensor_shape": list(thumbnail_tensor.shape) if thumbnail_tensor is not None else None,
        },
        "timing_ms": {
            "video_decode_scan_to_last_sample": decode_scan_ms,
            "spatial_tile_build": tile_build_ms,
            "tile_autogaze_tensorize": tile_summary["autogaze_tensorize_ms"],
            "tile_autogaze_forward": tile_summary["autogaze_forward_ms"],
            "thumbnail_resize": thumbnail_resize_ms,
            "thumbnail_tensorize": thumbnail_tensorize_ms,
            "total_measured": decode_scan_ms
            + tile_build_ms
            + tile_summary["autogaze_tensorize_ms"]
            + tile_summary["autogaze_forward_ms"]
            + thumbnail_resize_ms
            + thumbnail_tensorize_ms,
        },
        "memory_bytes": {
            "tile_tensor_peak_per_temporal_chunk": tile_tensor_bytes_peak,
            "thumbnail_tensor": thumbnail_tensor_bytes,
        },
        "temporal_chunk_summaries": temporal_chunks,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AutoGaze-only NVILA-like smoke on the HLVid example video")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--video", default=DEFAULT_HLVID_EXAMPLE_VIDEO)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--tile-size", type=int, default=392)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--gazing-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--output-json", default="outputs/autogaze_repro/hlvid_example_autogaze_only_128f.json")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
