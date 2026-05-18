from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import av
import torch
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf
from transformers import AutoModel, AutoProcessor

from repro.common import append_jsonl, environment_metadata, resolve_device, synchronize, write_json, write_jsonl
from repro.hlvid import load_hlvid_manifest, parse_choice, score_predictions

DEFAULT_MODEL = "nvidia/NVILA-8B-HD-Video"
DEFAULT_EXAMPLE_VIDEO = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
DEFAULT_PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\n"
    "B. Hampden Ave\n"
    "C. HampdenBlvd\n"
    "D. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)
NVILA_IMAGE_SIZE = 392
NVILA_TARGET_SCALES = [56, 112, 196, 392]
NVILA_TARGET_PATCH_SIZE = 14
NVILA_TOKEN_SHUFFLE = 9
NVILA_CONTEXT_LIMIT = 40960
AUTOGAZE_CHUNK_FRAMES = 16


class StageProfiler:
    def __init__(self, device: torch.device | None = None):
        self.device = device
        self._totals: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, name: str):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self._totals[name] += (time.perf_counter() - start) * 1000.0
            self._counts[name] += 1

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_ms": self._totals[name],
                "count": self._counts[name],
                "mean_ms": self._totals[name] / self._counts[name],
            }
            for name in sorted(self._totals)
        }


class ProfilePatches:
    def __init__(self, model, processor, profiler: StageProfiler):
        self.model = model
        self.processor = processor
        self.profiler = profiler
        self._patches: list[tuple[Any, str, Any]] = []

    def __enter__(self):
        self._patch_method(self.processor, "_preprocess_videos", "video_tiling_and_tensorize")
        self._patch_method(self.processor, "_get_gazing_info_from_videos", "autogaze_total")
        self._patch_method(self.processor, "_run_autogaze_batched", "autogaze_forward_batched")
        self._patch_module_function("_load_video_frames", "video_decode_sampling")
        self._patch_method(self.model, "_encode_vision", "vision_encode_total")
        self._patch_method(self.model, "_run_vision_tower_batched", "siglip_vision_tower")
        mm_projector = getattr(self.model, "mm_projector", None)
        if mm_projector is not None:
            self._patch_method(mm_projector, "forward", "mm_projector")
        llm = getattr(self.model, "llm", None)
        if llm is not None:
            self._patch_method(llm, "forward", "llm_forward")
        return self

    def __exit__(self, exc_type, exc, tb):
        for obj, attr, original in reversed(self._patches):
            setattr(obj, attr, original)

    def _patch_method(self, obj: Any, attr: str, stage: str) -> None:
        original = getattr(obj, attr, None)
        if not callable(original):
            return

        def wrapped(*args, **kwargs):
            with self.profiler.measure(stage):
                return original(*args, **kwargs)

        setattr(obj, attr, wrapped)
        self._patches.append((obj, attr, original))

    def _patch_module_function(self, attr: str, stage: str) -> None:
        module = sys.modules.get(self.processor.__class__.__module__)
        if module is None:
            return
        self._patch_method(module, attr, stage)


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def closest_aspect_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]], width: int, height: int, image_size: int) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def spatial_tile_grid(width: int, height: int, max_tiles_video: int, image_size: int = NVILA_IMAGE_SIZE) -> dict[str, int]:
    max_spatial_tiles = max(max_tiles_video, 1)
    target_ratios = {
        (i, j)
        for n in range(1, max_spatial_tiles + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if 1 <= i * j <= max_spatial_tiles
    }
    sorted_ratios = sorted(target_ratios, key=lambda item: item[0] * item[1])
    cols, rows = closest_aspect_ratio(width / height, sorted_ratios, width, height, image_size)
    return {"cols": cols, "rows": rows, "tiles": cols * rows}


def patches_per_frame(scales: list[int] | None = None, patch_size: int = NVILA_TARGET_PATCH_SIZE) -> int:
    active_scales = scales or NVILA_TARGET_SCALES
    return sum((scale // patch_size) ** 2 for scale in active_scales)


def estimate_nvila_preflight(
    *,
    width: int,
    height: int,
    source_frames: int | None,
    num_video_frames: int,
    num_video_frames_thumbnail: int,
    max_tiles_video: int,
    image_size: int = NVILA_IMAGE_SIZE,
    context_limit: int = NVILA_CONTEXT_LIMIT,
) -> dict[str, Any]:
    grid = spatial_tile_grid(width, height, max_tiles_video, image_size)
    spatial_tiles = grid["tiles"]
    temporal_chunks = math.ceil(num_video_frames / AUTOGAZE_CHUNK_FRAMES)
    tile_sequences = temporal_chunks * spatial_tiles
    padded_sampled_frames = temporal_chunks * AUTOGAZE_CHUNK_FRAMES
    tile_images = padded_sampled_frames * spatial_tiles
    thumbnail_frames = min(num_video_frames, num_video_frames_thumbnail)
    per_frame_patches = patches_per_frame()
    tokens_per_frame_tile = math.ceil(per_frame_patches / NVILA_TOKEN_SHUFFLE)
    keep_all_tile_tokens = num_video_frames * spatial_tiles * tokens_per_frame_tile
    keep_all_thumbnail_tokens = thumbnail_frames * tokens_per_frame_tile
    keep_all_projected_tokens = keep_all_tile_tokens + keep_all_thumbnail_tokens

    sampled_frame_rgb_bytes = num_video_frames * width * height * 3
    resized_tile_pil_rgb_bytes = tile_images * image_size * image_size * 3
    siglip_tile_tensor_bytes = tile_images * 3 * image_size * image_size * 4
    autogaze_tile_tensor_bytes = siglip_tile_tensor_bytes
    siglip_thumbnail_tensor_bytes = thumbnail_frames * 3 * image_size * image_size * 4
    autogaze_thumbnail_tensor_bytes = siglip_thumbnail_tensor_bytes
    estimated_cpu_preprocess_bytes = (
        sampled_frame_rgb_bytes
        + resized_tile_pil_rgb_bytes
        + siglip_tile_tensor_bytes
        + autogaze_tile_tensor_bytes
        + siglip_thumbnail_tensor_bytes
        + autogaze_thumbnail_tensor_bytes
    )

    risk_flags: list[str] = []
    if source_frames is not None and num_video_frames > source_frames:
        risk_flags.append("requested_frames_exceed_source_frames")
    if num_video_frames % AUTOGAZE_CHUNK_FRAMES != 0:
        risk_flags.append("frame_count_not_divisible_by_16")
    if keep_all_projected_tokens > context_limit:
        risk_flags.append("context")
    if estimated_cpu_preprocess_bytes > 32 * 1024**3:
        risk_flags.append("cpu_memory")
    if tile_sequences > 1024:
        risk_flags.append("many_tile_sequences")

    return {
        "video": {
            "width": width,
            "height": height,
            "source_frames": source_frames,
        },
        "sampling": {
            "requested_frames": num_video_frames,
            "thumbnail_frames": thumbnail_frames,
            "policy": "uniform_total_frames",
        },
        "tiling": {
            "cols": grid["cols"],
            "rows": grid["rows"],
            "spatial_tiles": spatial_tiles,
            "tile_size": image_size,
        },
        "chunking": {
            "chunk_frames": AUTOGAZE_CHUNK_FRAMES,
            "temporal_chunks": temporal_chunks,
            "padded_sampled_frames": padded_sampled_frames,
            "tile_sequences": tile_sequences,
        },
        "counts": {
            "tile_images": tile_images,
            "siglip_autogaze_tile_tensor_items": tile_images,
        },
        "tokens": {
            "patches_per_frame_tile": per_frame_patches,
            "tokens_per_frame_tile_after_shuffle": tokens_per_frame_tile,
            "keep_all_tile_tokens": keep_all_tile_tokens,
            "keep_all_thumbnail_tokens": keep_all_thumbnail_tokens,
            "keep_all_projected_tokens": keep_all_projected_tokens,
            "llm_context_limit": context_limit,
            "keep_all_exceeds_context": keep_all_projected_tokens > context_limit,
        },
        "memory": {
            "sampled_frame_rgb_bytes": sampled_frame_rgb_bytes,
            "resized_tile_pil_rgb_bytes": resized_tile_pil_rgb_bytes,
            "siglip_tile_tensor_bytes": siglip_tile_tensor_bytes,
            "autogaze_tile_tensor_bytes": autogaze_tile_tensor_bytes,
            "estimated_cpu_preprocess_bytes": estimated_cpu_preprocess_bytes,
            "estimated_cpu_preprocess_gib": bytes_to_gib(estimated_cpu_preprocess_bytes),
            "note": "Lower-bound estimate for current public processor path before model forward; Python/PIL overhead and intermediate arrays are not included.",
        },
        "risk_flags": risk_flags,
        "recommendation": (
            "Use chunked preprocessing/vision encoding or reduce num_video_frames/max_tiles_video before full generation."
            if risk_flags
            else "No obvious preflight risk detected for current thresholds."
        ),
    }


def processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.gazing_mode == "keep-all":
        gazing_ratio_tile: float | list[float] = 1
        task_loss_requirement_tile = None
    else:
        gazing_ratio_tile = [0.2] + [0.06] * 15
        task_loss_requirement_tile = args.task_loss_requirement_tile

    return {
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
        "autogaze_model_id": args.autogaze_model,
        "gazing_ratio_tile": gazing_ratio_tile,
        "gazing_ratio_thumbnail": 1,
        "task_loss_requirement_tile": task_loss_requirement_tile,
        "task_loss_requirement_thumbnail": None,
        "max_batch_size_autogaze": args.max_batch_size_autogaze,
        "trust_remote_code": True,
    }


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs(args))
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map=args.device_map,
        max_batch_size_siglip=args.max_batch_size_siglip,
    )
    model.eval()
    return model, processor


def read_video_metadata(video: str) -> dict[str, int | float | str | None]:
    container = av.open(video)
    try:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        fps = float(stream.average_rate) if stream.average_rate else None
        return {
            "width": int(stream.width),
            "height": int(stream.height),
            "frames": int(stream.frames) if stream.frames else None,
            "fps": fps,
            "duration_seconds": duration,
            "codec": stream.codec_context.name,
        }
    finally:
        container.close()


def input_device(model, fallback: torch.device) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def resolve_video(video: str, args: argparse.Namespace) -> str:
    if video.startswith("http://") or video.startswith("https://"):
        return video
    local = Path(video)
    if local.exists():
        return str(local)
    cached = Path(args.hlvid_video_root) / video
    if cached.exists():
        return str(cached)
    return hf_hub_url(repo_id=args.hlvid_repo, filename=video, repo_type="dataset")


def tensor_shapes(payload: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            shapes[key] = list(value.shape)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in value):
            shapes[key] = [list(item.shape) for item in value]
    return shapes


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _count_padding_masks(value: Any) -> tuple[int, int, int]:
    selected = 0
    padded_count = 0
    total = 0
    for tensor in _iter_tensors(value):
        padded_bool = tensor.bool()
        selected += int((~padded_bool).sum().item())
        padded_count += int(padded_bool.sum().item())
        total += int(padded_bool.numel())
    return selected, padded_count, total


def _gazing_padding_payloads(payload: dict[str, Any]) -> list[Any]:
    masks: list[Any] = []
    gazing_info = payload.get("gazing_info")
    if isinstance(gazing_info, dict):
        for key in (
            "if_padded_gazing",
            "if_padded_gazing_tiles",
            "if_padded_gazing_thumbnails",
        ):
            if key in gazing_info:
                masks.append(gazing_info[key])

    for value in payload.values():
        if isinstance(value, dict) and "if_padded_gazing" in value:
            masks.append(value["if_padded_gazing"])
    return masks


def extract_gaze_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "autogaze_selected_patches": None,
        "autogaze_padded_patches": None,
        "autogaze_total_gaze_slots": None,
        "autogaze_token_reduction_ratio": None,
        "available_input_keys": sorted(payload.keys()),
    }
    selected, padded_count, total = _count_padding_masks(_gazing_padding_payloads(payload))
    if total:
        metrics.update(
            {
                "autogaze_selected_patches": selected,
                "autogaze_padded_patches": padded_count,
                "autogaze_total_gaze_slots": total,
                "autogaze_token_reduction_ratio": None,
            }
        )
    return metrics


def _tensor_sequence(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, torch.Tensor)]
    return []


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        output: list[int] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                output.extend(_int_list(item))
            else:
                output.append(int(item))
        return output
    return [int(value)]


def _safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _visual_frame_count(tensor: torch.Tensor) -> int:
    if tensor.ndim >= 5:
        return int(tensor.shape[0]) * int(tensor.shape[1])
    if tensor.ndim >= 1:
        return int(tensor.shape[0])
    return 0


def compute_visual_token_metrics(
    payload: dict[str, Any],
    *,
    video_token_id: int | None,
    patches_per_frame_value: int = patches_per_frame(),
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    tile_tensors = _tensor_sequence(payload.get("pixel_values_videos_tiles"))
    thumbnail_tensors = _tensor_sequence(payload.get("pixel_values_videos_thumbnails"))

    raw_tile_patches = sum(_visual_frame_count(tensor) * patches_per_frame_value for tensor in tile_tensors)
    raw_thumbnail_patches = sum(_visual_frame_count(tensor) * patches_per_frame_value for tensor in thumbnail_tensors)
    raw_encoder_patches = raw_tile_patches + raw_thumbnail_patches

    selected, padded_count, total_slots = _count_padding_masks(_gazing_padding_payloads(payload))
    selected_encoder_patches = selected if total_slots else raw_encoder_patches

    num_spatial_tiles = _int_list(payload.get("num_spatial_tiles_each_video"))
    keep_all_tile_tokens = 0
    for index, tensor in enumerate(tile_tensors):
        if tensor.ndim < 2:
            continue
        spatial_tiles = num_spatial_tiles[index] if index < len(num_spatial_tiles) else 1
        spatial_tiles = max(spatial_tiles, 1)
        tile_sequences = int(tensor.shape[0])
        frames_per_sequence = int(tensor.shape[1])
        temporal_chunks = math.ceil(tile_sequences / spatial_tiles)
        total_frames = temporal_chunks * frames_per_sequence
        keep_all_tile_tokens += total_frames * math.ceil(spatial_tiles * patches_per_frame_value / token_shuffle)

    keep_all_thumbnail_tokens = 0
    thumbnail_token_per_frame = math.ceil(patches_per_frame_value / token_shuffle)
    for tensor in thumbnail_tensors:
        if tensor.ndim >= 5:
            keep_all_thumbnail_tokens += int(tensor.shape[0]) * int(tensor.shape[1]) * thumbnail_token_per_frame
        elif tensor.ndim >= 1:
            keep_all_thumbnail_tokens += int(tensor.shape[0]) * thumbnail_token_per_frame

    llm_actual_visual_tokens = None
    input_ids = payload.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and video_token_id is not None:
        llm_actual_visual_tokens = int((input_ids == video_token_id).sum().item())

    keep_all_projected_tokens = keep_all_tile_tokens + keep_all_thumbnail_tokens
    return {
        "patches_per_frame": patches_per_frame_value,
        "token_shuffle": token_shuffle,
        "encoder_raw_tile_patch_tokens": raw_tile_patches,
        "encoder_raw_thumbnail_patch_tokens": raw_thumbnail_patches,
        "encoder_raw_patch_tokens": raw_encoder_patches,
        "encoder_autogaze_selected_patch_tokens": selected_encoder_patches,
        "encoder_autogaze_padded_patch_tokens": padded_count if total_slots else 0,
        "encoder_autogaze_total_gaze_slots": total_slots,
        "encoder_token_reduction_ratio": _safe_ratio(raw_encoder_patches, selected_encoder_patches),
        "llm_keep_all_tile_visual_tokens_estimated": keep_all_tile_tokens,
        "llm_keep_all_thumbnail_visual_tokens_estimated": keep_all_thumbnail_tokens,
        "llm_keep_all_visual_tokens_estimated": keep_all_projected_tokens,
        "llm_actual_visual_tokens": llm_actual_visual_tokens,
        "llm_visual_token_reduction_ratio": _safe_ratio(keep_all_projected_tokens, llm_actual_visual_tokens),
    }


def resolve_video_token_id(model, processor) -> int | None:
    config_token_id = getattr(getattr(model, "config", None), "video_token_id", None)
    if config_token_id is not None:
        return int(config_token_id)
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_token_id = getattr(tokenizer, "video_token_id", None)
    if tokenizer_token_id is not None:
        return int(tokenizer_token_id)
    video_token = getattr(tokenizer, "video_token", None)
    if video_token is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(video_token)
        if token_id is not None:
            return int(token_id)
    return None


def model_patches_per_frame(model) -> int:
    vision_tower = getattr(model, "vision_tower", None)
    config = getattr(vision_tower, "config", None)
    scales = getattr(config, "scales", None)
    patch_size = getattr(config, "patch_size", NVILA_TARGET_PATCH_SIZE)
    if isinstance(scales, str):
        clean = scales.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        parsed_scales = [int(part.strip()) for part in clean.split(",") if part.strip()]
    elif isinstance(scales, int):
        parsed_scales = [int(scales)]
    elif scales is None:
        parsed_scales = NVILA_TARGET_SCALES
    else:
        parsed_scales = [int(scale) for scale in scales]
    return patches_per_frame(parsed_scales, int(patch_size))


def move_tensors(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move_value(item) for item in value)
        if isinstance(value, dict):
            return {key: move_value(item) for key, item in value.items()}
        return value

    return {key: move_value(value) for key, value in payload.items()}


def timed_generate(model, inputs: dict[str, Any], processor, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    synchronize(device)
    generate_ms = (time.perf_counter() - start) * 1000.0
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "raw_output": response,
        "parsed_answer": parse_choice(response),
        "generate_ms": generate_ms,
        "generated_tokens": int(generated.shape[1]),
        "peak_memory_bytes": peak_memory_bytes,
    }


def stage_total(timings: dict[str, dict[str, float | int]], stage: str) -> float | None:
    value = timings.get(stage)
    if value is None:
        return None
    return float(value["total_ms"])


def generate_one(model, processor, video: str, prompt: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    video_token = processor.tokenizer.video_token
    resolved_video = resolve_video(video, args)

    profiler = StageProfiler(device)
    with ProfilePatches(model, processor, profiler):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with profiler.measure("processor_total"):
            inputs = processor(text=f"{video_token}\n\n{prompt}", videos=resolved_video, return_tensors="pt")
        processor_peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        processor_timings = profiler.as_dict()
        profiler.reset()

        inputs = dict(inputs)
        token_metrics = compute_visual_token_metrics(
            inputs,
            video_token_id=resolve_video_token_id(model, processor),
            patches_per_frame_value=model_patches_per_frame(model),
            token_shuffle=NVILA_TOKEN_SHUFFLE,
        )
        gaze_metrics = extract_gaze_metrics(inputs)

        target_device = input_device(model, device)
        inputs = move_tensors(inputs, target_device)

        ttft_ms = None
        ttft_timings = None
        if args.measure_ttft:
            one_token = timed_generate(model, inputs, processor, device, max_new_tokens=1)
            ttft_ms = one_token["generate_ms"]
            ttft_timings = profiler.as_dict()
            profiler.reset()

        result = timed_generate(model, inputs, processor, device, max_new_tokens=args.max_new_tokens)
        generate_timings = profiler.as_dict()
        profiler.reset()

    preprocess_ms = stage_total(processor_timings, "processor_total") or 0.0
    decode_estimated_ms = max(result["generate_ms"] - ttft_ms, 0.0) if ttft_ms is not None else None

    return {
        **result,
        **gaze_metrics,
        "video_input": video,
        "video_resolved": resolved_video,
        "gazing_mode": args.gazing_mode,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "input_shapes": tensor_shapes(inputs),
        "token_metrics": token_metrics,
        "video_preprocess_ms": preprocess_ms,
        "video_decode_ms": stage_total(processor_timings, "video_decode_sampling"),
        "video_tiling_ms": stage_total(processor_timings, "video_tiling_and_tensorize"),
        "autogaze_ms": stage_total(processor_timings, "autogaze_total"),
        "autogaze_forward_ms": stage_total(processor_timings, "autogaze_forward_batched"),
        "processor_peak_memory_bytes": processor_peak_memory_bytes,
        "ttft_ms": ttft_ms,
        "ttft_stage_timings_ms": ttft_timings,
        "decode_estimated_ms": decode_estimated_ms,
        "total_ms": preprocess_ms + result["generate_ms"],
        "stage_timings_ms": {
            "processor": processor_timings,
            "ttft": ttft_timings,
            "generate": generate_timings,
        },
        "vision_encoder_ms": stage_total(generate_timings, "vision_encode_total"),
        "siglip_vision_ms": stage_total(generate_timings, "siglip_vision_tower"),
        "mm_projector_ms": stage_total(generate_timings, "mm_projector"),
        "llm_forward_ms": stage_total(generate_timings, "llm_forward"),
        "mllm_prefill_ms": ttft_ms,
        "metric_note": (
            "Stage timings are collected by wrapping public NVILA remote-code methods at runtime. "
            "video_tiling_ms includes tile creation and image tensorization; vision_encoder_ms includes "
            "SigLIP, feature reordering, and projector prep unless more specific sub-stages are present."
        ),
    }


def run_single(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, processor = load_model_and_processor(args)
    result = generate_one(model, processor, args.video, args.prompt, device, args)
    payload = {
        "metadata": environment_metadata(device),
        "model_path": args.model_path,
        "autogaze_model": args.autogaze_model,
        "gazing_mode": args.gazing_mode,
        "video": args.video,
        "prompt": args.prompt,
        "result": result,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_preflight(args: argparse.Namespace) -> None:
    resolved_video = resolve_video(args.video, args)
    metadata: dict[str, Any]
    if args.preflight_width and args.preflight_height:
        metadata = {
            "width": args.preflight_width,
            "height": args.preflight_height,
            "frames": args.preflight_source_frames,
            "fps": None,
            "duration_seconds": None,
            "codec": None,
        }
    elif resolved_video.startswith("http://") or resolved_video.startswith("https://"):
        raise ValueError(
            "Preflight needs a local video path or explicit --preflight-width and --preflight-height for remote videos."
        )
    else:
        metadata = read_video_metadata(resolved_video)

    estimate = estimate_nvila_preflight(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        source_frames=metadata.get("frames"),
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
    )
    payload = {
        "model_path": args.model_path,
        "autogaze_model": args.autogaze_model,
        "gazing_mode": args.gazing_mode,
        "video": args.video,
        "video_resolved": resolved_video,
        "source_metadata": metadata,
        "estimate": estimate,
    }
    write_json(args.preflight_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def completed_question_ids(path: Path) -> set[Any]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text().splitlines() if line.strip()}


def run_hlvid(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    rows = load_hlvid_manifest(split=args.split, limit=args.limit, config=args.config)
    model, processor = load_model_and_processor(args)
    output_path = Path(args.predictions)
    completed_ids = completed_question_ids(output_path) if args.resume else set()

    for row in rows:
        if row["question_id"] in completed_ids:
            continue
        result = generate_one(model, processor, row["video_path"], row["question"], device, args)
        prediction = {
            **row,
            **result,
            "model_path": args.model_path,
            "autogaze_model": args.autogaze_model,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "gazing_mode": args.gazing_mode,
            "task_loss_requirement_tile": args.task_loss_requirement_tile,
        }
        append_jsonl(output_path, [prediction])

    all_rows = []
    if output_path.exists():
        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary, scored = score_predictions(all_rows)
    write_json(args.summary, summary)
    write_jsonl(args.scored_predictions, scored)
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_preset_defaults(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    preset = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(preset, dict):
        raise ValueError(f"Preset config must be a mapping: {path}")
    runner = preset.get("nvila_runner", {})
    if not isinstance(runner, dict):
        raise ValueError(f"Preset config 'nvila_runner' must be a mapping: {path}")
    args = runner.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Preset config 'nvila_runner.args' must be a mapping: {path}")
    return dict(args)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NVILA-HD-Video quickstart and HLVid benchmark")
    parser.add_argument("--preset-config")
    parser.add_argument("--mode", choices=["single", "hlvid", "preflight"], default="single")
    parser.add_argument("--model-path", "--nvila-model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--video", default=DEFAULT_EXAMPLE_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--gazing-mode", choices=["autogaze", "keep-all"], default="autogaze")
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.6)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--hlvid-repo", default="bfshi/HLVid")
    parser.add_argument("--hlvid-video-root", default="data/hlvid/videos")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-width", type=int)
    parser.add_argument("--preflight-height", type=int)
    parser.add_argument("--preflight-source-frames", type=int)
    parser.add_argument("--preflight-json", default="outputs/autogaze_repro/nvila_preflight.json")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/nvila_single.json")
    parser.add_argument("--predictions", default="outputs/autogaze_repro/hlvid_predictions.jsonl")
    parser.add_argument("--summary", default="outputs/autogaze_repro/hlvid_summary.json")
    parser.add_argument("--scored-predictions", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--preset-config")
    known, _ = bootstrap.parse_known_args(argv)
    defaults = load_preset_defaults(known.preset_config)
    return build_parser(defaults).parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "single":
        run_single(args)
    else:
        run_hlvid(args)


if __name__ == "__main__":
    main()
