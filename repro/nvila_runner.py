from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_url
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoModel, AutoProcessor

from repro.common import (
    append_jsonl,
    compute_stats,
    environment_metadata,
    resolve_device,
    synchronize,
    write_json,
    write_jsonl,
)
from repro.failure_logging import classify_exception, minimal_runner_failure_payload
from repro.hlvid import (
    latency_hierarchy_summary,
    load_hlvid_manifest,
    parse_choice,
    read_manifest_file,
    score_predictions,
)
from repro.gaze_visualization import safe_label, write_gaze_visualization_artifacts

DEFAULT_MODEL = "nvidia/NVILA-8B-HD-Video"
DEFAULT_HD_MODEL = DEFAULT_MODEL
DEFAULT_BASELINE_MODEL = "Efficient-Large-Model/NVILA-8B-Video"
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
NVILA_VISION_SCALES = [56, 112, 196, 392]
NVILA_VISION_PATCH_SIZE = 14
NVILA_AUTOGAZE_TARGET_SCALES = [56, 112, 196, 392]
NVILA_AUTOGAZE_TARGET_PATCH_SIZE = NVILA_VISION_PATCH_SIZE
NVILA_TARGET_SCALES = NVILA_VISION_SCALES
NVILA_TARGET_PATCH_SIZE = NVILA_VISION_PATCH_SIZE
NVILA_TOKEN_SHUFFLE = 9
NVILA_CONTEXT_LIMIT = 40960
AUTOGAZE_CHUNK_FRAMES = 16
MODEL_FAMILY_AUTO = "auto"
MODEL_FAMILY_HD_AUTOGAZE = "nvila-hd-video-autogaze"
MODEL_FAMILY_VIDEO_BASELINE = "nvila-video-baseline"
MODEL_FAMILY_CHOICES = (
    MODEL_FAMILY_AUTO,
    MODEL_FAMILY_HD_AUTOGAZE,
    MODEL_FAMILY_VIDEO_BASELINE,
)
TOKEN_SELECTOR_ADAPTER_CHOICES = ("auto", "none", "keep-all", "autogaze")
VISION_ENCODER_ADAPTER_CHOICES = ("auto", "nvila-hd-siglip", "nvila-video-vision")
MLLM_ADAPTER_CHOICES = ("auto", "nvila-hd", "nvila-video")
PAPER_PRESET_BASELINE = "autogaze-hlvid-baseline"
PAPER_PRESET_HD = "autogaze-hlvid-hd"
PAPER_PRESET_CHOICES = (PAPER_PRESET_BASELINE, PAPER_PRESET_HD)
PAPER_PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    PAPER_PRESET_BASELINE: {
        "model_path": DEFAULT_BASELINE_MODEL,
        "model_family": MODEL_FAMILY_VIDEO_BASELINE,
        "num_video_frames": 256,
        "num_video_frames_thumbnail": 0,
        "max_tiles_video": 1,
        "video_resize_longest_edge": 448,
        "gazing_mode": "keep-all",
        "paper_reference_score": 42.5,
        "paper_reference_frames": 256,
        "paper_reference_max_resolution": 448,
        "autogaze_applicability": "not_applicable",
    },
    PAPER_PRESET_HD: {
        "model_path": DEFAULT_HD_MODEL,
        "model_family": MODEL_FAMILY_HD_AUTOGAZE,
        "num_video_frames": 1024,
        "num_video_frames_thumbnail": 128,
        "max_tiles_video": 48,
        "video_resize_longest_edge": 3584,
        "gazing_mode": "autogaze",
        "task_loss_requirement_tile": 0.7,
        "paper_reference_score": 52.6,
        "paper_reference_frames": 1024,
        "paper_reference_max_resolution": 3584,
        "autogaze_applicability": "enabled",
    },
}
PAPER_PRESET_FIELD_OPTIONS: dict[str, tuple[str, ...]] = {
    "model_path": ("--model-path", "--nvila-model", "--mllm-path"),
    "model_family": ("--model-family",),
    "num_video_frames": ("--num-video-frames",),
    "num_video_frames_thumbnail": ("--num-video-frames-thumbnail",),
    "max_tiles_video": ("--max-tiles-video",),
    "video_resize_longest_edge": (
        "--video-resize-longest-edge",
        "--video-resize-shortest-edge",
        "--video-resize-width",
        "--video-resize-height",
    ),
    "gazing_mode": ("--gazing-mode",),
    "task_loss_requirement_tile": ("--task-loss-requirement-tile",),
    "token_selector_adapter": ("--token-selector-adapter",),
    "token_selector_name": ("--token-selector-name",),
    "token_selector_path": ("--token-selector-path",),
    "vision_encoder_adapter": ("--vision-encoder-adapter",),
    "vision_encoder_name": ("--vision-encoder-name",),
    "vision_encoder_path": ("--vision-encoder-path",),
    "mllm_adapter": ("--mllm-adapter",),
    "mllm_name": ("--mllm-name",),
}
AUTOGAZE_PROCESSOR_KWARGS = {
    "autogaze_model_id",
    "gazing_ratio_tile",
    "gazing_ratio_thumbnail",
    "task_loss_requirement_tile",
    "task_loss_requirement_thumbnail",
    "max_batch_size_autogaze",
    "target_scales",
    "target_patch_size",
}
H100_DEFAULT_BUDGET_GIB = 70.0
H100_GREEN_GIB = 55.0
H100_SWEEP_FRAMES = (1024, 512, 256, 128, 64, 32)
H100_SWEEP_THUMBNAIL_FRAMES = (512, 256, 128, 64, 32, 16)
H100_SWEEP_MAX_TILES = (48, 32, 16, 8, 4, 1)
H100_SWEEP_RESIZE_SHORTEST_EDGE = (None, 1080, 720, 512, 448, 384)
REPEAT_SUMMARY_FIELDS = (
    "total_ms",
    "generate_ms",
    "video_preprocess_ms",
    "video_preprocess_without_autogaze_ms",
    "autogaze_total_ms",
    "video_decode_ms",
    "video_tiling_ms",
    "autogaze_ms",
    "gazing_info_total_ms",
    "autogaze_forward_ms",
    "autogaze_model_forward_ms",
    "vision_encoder_ms",
    "siglip_vision_ms",
    "mm_projector_ms",
    "llm_forward_ms",
    "ttft_ms",
    "decode_estimated_ms",
    "generation_decode_after_ttft_estimated_ms",
    "processor_peak_memory_bytes",
    "ttft_peak_memory_bytes",
    "llm_peak_memory_bytes",
    "peak_memory_bytes",
    "token_metrics.video_sampled_frames",
    "token_metrics.thumbnail_sampled_frames",
    "token_metrics.encoder_raw_patch_tokens",
    "token_metrics.encoder_autogaze_selected_patch_tokens",
    "token_metrics.encoder_token_reduction_ratio",
    "token_metrics.encoder_raw_tile_patch_tokens",
    "token_metrics.encoder_autogaze_selected_tile_patch_tokens",
    "token_metrics.autogaze_input_tile_frame_instances",
    "token_metrics.autogaze_input_patch_tokens",
    "token_metrics.autogaze_selected_patch_tokens",
    "token_metrics.autogaze_removed_patch_tokens",
    "token_metrics.autogaze_patch_reduction_ratio",
    "token_metrics.encoder_tile_token_reduction_ratio",
    "token_metrics.encoder_raw_thumbnail_patch_tokens",
    "token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens",
    "token_metrics.llm_visual_token_reduction_ratio",
    "token_metrics.llm_actual_visual_tokens",
    "token_metrics.llm_keep_all_visual_tokens_estimated",
    "compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio",
    "compute_metrics.mllm.prefill_context_reduction_ratio",
    "compute_metrics.mllm.kv_cache_reduction_ratio",
    "compute_metrics.mllm.prefill_attention_pair_reduction_ratio",
    "compute_metrics.mllm.prefill_total_macs_reduction_ratio",
)
TOKEN_BUDGET_SUMMARY_FIELDS = (
    "token_metrics.video_sampled_frames",
    "token_metrics.thumbnail_sampled_frames",
    "token_metrics.encoder_raw_tile_patch_tokens",
    "token_metrics.encoder_autogaze_selected_tile_patch_tokens",
    "token_metrics.autogaze_input_tile_frame_instances",
    "token_metrics.autogaze_input_patch_tokens",
    "token_metrics.autogaze_selected_patch_tokens",
    "token_metrics.autogaze_removed_patch_tokens",
    "token_metrics.autogaze_patch_reduction_ratio",
    "token_metrics.encoder_raw_thumbnail_patch_tokens",
    "token_metrics.encoder_autogaze_selected_thumbnail_patch_tokens",
    "token_metrics.encoder_raw_patch_tokens",
    "token_metrics.encoder_autogaze_selected_patch_tokens",
    "token_metrics.encoder_token_reduction_ratio",
    "token_metrics.llm_keep_all_visual_tokens_estimated",
    "token_metrics.llm_actual_visual_tokens",
    "token_metrics.llm_visual_token_reduction_ratio",
)


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

    def add(self, name: str, elapsed_ms: float) -> None:
        self._totals[name] += elapsed_ms
        self._counts[name] += 1

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


def patch_positions_by_scale(scales: list[int], patch_size: int) -> dict[str, int]:
    return {str(scale): (scale // patch_size) ** 2 for scale in scales}


def patches_per_frame(scales: list[int] | None = None, patch_size: int = NVILA_TARGET_PATCH_SIZE) -> int:
    active_scales = scales or NVILA_TARGET_SCALES
    return sum(patch_positions_by_scale(active_scales, patch_size).values())


def parse_int_sequence(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = [int(part) for part in re.findall(r"\d+", value)]
    return parsed or None


def parse_float_sequence(value: str | list[float] | tuple[float, ...] | None) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    parsed = [float(part) for part in re.findall(r"\d+(?:\.\d+)?", value)]
    return parsed or None


def effective_stream_gazing_ratio(args: argparse.Namespace) -> float | list[float]:
    parsed = parse_float_sequence(getattr(args, "stream_gazing_ratio", None))
    if parsed is None:
        parsed = parse_float_sequence(getattr(args, "gazing_ratio_tile", None))
    if parsed is None:
        return [0.2] + [0.06] * 15
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def effective_gazing_ratio_tile(args: argparse.Namespace) -> float | list[float]:
    parsed = parse_float_sequence(getattr(args, "gazing_ratio_tile", None))
    if parsed is None:
        return [0.2] + [0.06] * 15
    if len(parsed) == 1:
        return parsed[0]
    return parsed


def autogaze_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "gazing_mode", None) == "keep-all":
        ratio: float | list[float] | str = 1
        task_loss = None
    else:
        ratio = effective_gazing_ratio_tile(args)
        task_loss = getattr(args, "task_loss_requirement_tile", None)
    return {
        "gazing_mode": getattr(args, "gazing_mode", None),
        "gazing_ratio_tile": ratio,
        "stream_gazing_ratio": effective_stream_gazing_ratio(args),
        "task_loss_requirement_tile": task_loss,
        "target_scales": effective_autogaze_target_scales(args),
        "target_patch_size": effective_autogaze_target_patch_size(args),
        "max_batch_size_autogaze": getattr(args, "max_batch_size_autogaze", None),
        "generate_only": (
            bool(getattr(args, "autogaze_generate_only", False))
            if getattr(args, "gazing_mode", None) == "autogaze"
            else None
        ),
        "note": (
            "NVILA processor default is [0.2] + [0.06] * 15, while the original AutoGaze Quick Start uses 0.75. "
            "Compare AutoGaze latency only when this config and the raw patch budget match."
        ),
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def provided_cli_options(argv: list[str] | None) -> set[str]:
    tokens = sys.argv[1:] if argv is None else list(argv)
    options: set[str] = set()
    for token in tokens:
        if not token.startswith("--"):
            continue
        options.add(token.split("=", 1)[0])
    return options


def _field_was_provided(field: str, provided_options: set[str]) -> bool:
    return any(option in provided_options for option in PAPER_PRESET_FIELD_OPTIONS.get(field, ()))


def apply_paper_preset_defaults(args: argparse.Namespace, provided_options: set[str] | None = None) -> argparse.Namespace:
    preset = getattr(args, "paper_preset", None)
    if not preset:
        return args
    if preset not in PAPER_PRESET_CONFIGS:
        raise ValueError(f"Unsupported paper preset: {preset}")
    provided_options = provided_options or set()
    for field, value in PAPER_PRESET_CONFIGS[preset].items():
        if field.startswith("paper_reference_") or field == "autogaze_applicability":
            continue
        if _field_was_provided(field, provided_options):
            continue
        setattr(args, field, value)
    return args


def apply_pipeline_preset_alias(args: argparse.Namespace) -> argparse.Namespace:
    pipeline_preset = getattr(args, "pipeline_preset", None)
    paper_preset = getattr(args, "paper_preset", None)
    if pipeline_preset and not paper_preset:
        args.paper_preset = pipeline_preset
    elif paper_preset and not pipeline_preset:
        args.pipeline_preset = paper_preset
    return args


def effective_model_family(args: argparse.Namespace) -> str:
    explicit = getattr(args, "model_family", MODEL_FAMILY_AUTO)
    if explicit and explicit != MODEL_FAMILY_AUTO:
        return str(explicit)
    preset = getattr(args, "paper_preset", None)
    if preset in PAPER_PRESET_CONFIGS:
        return str(PAPER_PRESET_CONFIGS[preset]["model_family"])
    model_path = str(getattr(args, "model_path", DEFAULT_MODEL))
    if "NVILA-8B-HD-Video" in model_path:
        return MODEL_FAMILY_HD_AUTOGAZE
    if "NVILA-8B-Video" in model_path:
        return MODEL_FAMILY_VIDEO_BASELINE
    return MODEL_FAMILY_HD_AUTOGAZE


def effective_autogaze_target_scales(args: argparse.Namespace) -> list[int] | None:
    explicit = parse_int_sequence(getattr(args, "autogaze_target_scales", None))
    if explicit is not None:
        return explicit
    if effective_model_family(args) == MODEL_FAMILY_HD_AUTOGAZE:
        return list(NVILA_AUTOGAZE_TARGET_SCALES)
    return None


def effective_autogaze_target_patch_size(args: argparse.Namespace) -> int | None:
    explicit = getattr(args, "autogaze_target_patch_size", None)
    if explicit is not None:
        return int(explicit)
    if effective_model_family(args) == MODEL_FAMILY_HD_AUTOGAZE:
        return NVILA_AUTOGAZE_TARGET_PATCH_SIZE
    return None


def effective_token_selector_adapter(args: argparse.Namespace) -> str:
    explicit = getattr(args, "token_selector_adapter", "auto")
    if explicit and explicit != "auto":
        return str(explicit)
    family = effective_model_family(args)
    if family == MODEL_FAMILY_VIDEO_BASELINE:
        return "none"
    if getattr(args, "gazing_mode", None) == "keep-all":
        return "keep-all"
    return "autogaze"


def effective_vision_encoder_adapter(args: argparse.Namespace) -> str:
    explicit = getattr(args, "vision_encoder_adapter", "auto")
    if explicit and explicit != "auto":
        return str(explicit)
    if effective_model_family(args) == MODEL_FAMILY_VIDEO_BASELINE:
        return "nvila-video-vision"
    return "nvila-hd-siglip"


def effective_mllm_adapter(args: argparse.Namespace) -> str:
    explicit = getattr(args, "mllm_adapter", "auto")
    if explicit and explicit != "auto":
        return str(explicit)
    if effective_model_family(args) == MODEL_FAMILY_VIDEO_BASELINE:
        return "nvila-video"
    return "nvila-hd"


def effective_mllm_path(args: argparse.Namespace) -> str:
    return str(getattr(args, "mllm_path", None) or getattr(args, "model_path", DEFAULT_MODEL))


def effective_token_selector_path(args: argparse.Namespace) -> str | None:
    adapter = effective_token_selector_adapter(args)
    explicit = getattr(args, "token_selector_path", None)
    if explicit is not None:
        return str(explicit)
    if adapter == "autogaze":
        return str(getattr(args, "autogaze_model", "nvidia/AutoGaze"))
    return None


def effective_component_names(args: argparse.Namespace) -> dict[str, str]:
    token_adapter = effective_token_selector_adapter(args)
    vision_adapter = effective_vision_encoder_adapter(args)
    mllm_adapter = effective_mllm_adapter(args)
    token_name = getattr(args, "token_selector_name", None)
    vision_name = getattr(args, "vision_encoder_name", None)
    mllm_name = getattr(args, "mllm_name", None)
    if token_name is None:
        if token_adapter == "none":
            token_name = "not_applicable"
        elif token_adapter == "keep-all":
            token_name = "keep_all"
        else:
            token_name = str(getattr(args, "autogaze_model", "nvidia/AutoGaze"))
    if vision_name is None:
        vision_name = "nvila-8b-video-vision" if vision_adapter == "nvila-video-vision" else "nvila-hd-siglip"
    if mllm_name is None:
        mllm_name = effective_mllm_path(args)
    return {
        "token_selector": str(token_name),
        "vision_encoder": str(vision_name),
        "mllm": str(mllm_name),
    }


def build_component_identity(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    names = effective_component_names(args)
    return {
        "token_selector": {
            "adapter": effective_token_selector_adapter(args),
            "name": names["token_selector"],
            "path": effective_token_selector_path(args),
            "applicability": autogaze_applicability(args),
        },
        "vision_encoder": {
            "adapter": effective_vision_encoder_adapter(args),
            "name": names["vision_encoder"],
            "path": str(getattr(args, "vision_encoder_path", None) or "auto"),
        },
        "mllm": {
            "adapter": effective_mllm_adapter(args),
            "name": names["mllm"],
            "path": effective_mllm_path(args),
        },
    }


def apply_component_defaults(args: argparse.Namespace) -> argparse.Namespace:
    mllm_path = getattr(args, "mllm_path", None)
    if mllm_path is not None:
        args.model_path = str(mllm_path)
    args.mllm_path = str(getattr(args, "model_path", DEFAULT_MODEL))

    args.token_selector_adapter = effective_token_selector_adapter(args)
    args.vision_encoder_adapter = effective_vision_encoder_adapter(args)
    args.mllm_adapter = effective_mllm_adapter(args)
    if args.token_selector_adapter == "autogaze":
        args.gazing_mode = "autogaze"
    elif args.token_selector_adapter in {"keep-all", "none"} and effective_model_family(args) == MODEL_FAMILY_HD_AUTOGAZE:
        args.gazing_mode = "keep-all"
    token_path = effective_token_selector_path(args)
    args.token_selector_path = token_path
    if token_path is not None and args.token_selector_adapter == "autogaze":
        args.autogaze_model = token_path
    if getattr(args, "vision_encoder_path", None) is None:
        args.vision_encoder_path = "auto"
    names = effective_component_names(args)
    if getattr(args, "token_selector_name", None) is None:
        args.token_selector_name = names["token_selector"]
    if getattr(args, "vision_encoder_name", None) is None:
        args.vision_encoder_name = names["vision_encoder"]
    if getattr(args, "mllm_name", None) is None:
        args.mllm_name = names["mllm"]
    return args


def validate_thumbnail_compatibility(args: argparse.Namespace) -> None:
    if getattr(args, "mode", None) not in {"single", "hlvid"}:
        return
    if effective_model_family(args) != MODEL_FAMILY_HD_AUTOGAZE:
        return
    thumbnail_frames = int(getattr(args, "num_video_frames_thumbnail", 0) or 0)
    if thumbnail_frames <= 0:
        raise ValueError(
            "Public NVILA-HD single/hlvid generate paths require "
            "--num-video-frames-thumbnail >= 1. The remote processor samples thumbnail frames "
            "with integer division and the model concatenates thumbnail tensors, so 0 thumbnails "
            "can fail before generation. Use --num-video-frames-thumbnail 1 for full NVILA-HD "
            "runs, or use --mode stream-profile for AutoGaze-only/streamed preprocessing checks "
            "with 0 thumbnails."
        )


def paper_reference_for_args(args: argparse.Namespace) -> dict[str, Any]:
    preset = getattr(args, "paper_preset", None)
    if preset in PAPER_PRESET_CONFIGS:
        reference_applies = True
        if preset == PAPER_PRESET_BASELINE:
            reference_applies = effective_model_family(args) == MODEL_FAMILY_VIDEO_BASELINE
        elif preset == PAPER_PRESET_HD:
            reference_applies = (
                effective_model_family(args) == MODEL_FAMILY_HD_AUTOGAZE
                and effective_token_selector_adapter(args) == "autogaze"
            )
        if not reference_applies:
            return {
                "paper_reference_score": None,
                "paper_reference_frames": None,
                "paper_reference_max_resolution": None,
            }
        config = PAPER_PRESET_CONFIGS[preset]
        return {
            "paper_reference_score": config.get("paper_reference_score"),
            "paper_reference_frames": config.get("paper_reference_frames"),
            "paper_reference_max_resolution": config.get("paper_reference_max_resolution"),
        }
    family = effective_model_family(args)
    if family == MODEL_FAMILY_VIDEO_BASELINE:
        config = PAPER_PRESET_CONFIGS[PAPER_PRESET_BASELINE]
    elif family == MODEL_FAMILY_HD_AUTOGAZE and getattr(args, "gazing_mode", None) == "autogaze":
        config = PAPER_PRESET_CONFIGS[PAPER_PRESET_HD]
    else:
        return {
            "paper_reference_score": None,
            "paper_reference_frames": None,
            "paper_reference_max_resolution": None,
        }
    return {
        "paper_reference_score": config.get("paper_reference_score"),
        "paper_reference_frames": config.get("paper_reference_frames"),
        "paper_reference_max_resolution": config.get("paper_reference_max_resolution"),
    }


def autogaze_applicability(args: argparse.Namespace) -> str:
    family = effective_model_family(args)
    token_selector_adapter = effective_token_selector_adapter(args)
    if family == MODEL_FAMILY_VIDEO_BASELINE:
        return "not_applicable"
    if family == MODEL_FAMILY_HD_AUTOGAZE and token_selector_adapter == "autogaze":
        return "enabled"
    if family == MODEL_FAMILY_HD_AUTOGAZE and token_selector_adapter in {"keep-all", "none"}:
        return "hd_keep_all_ablation"
    return "unknown"


def build_adapter_identity(args: argparse.Namespace) -> dict[str, Any]:
    family = effective_model_family(args)
    applicability = autogaze_applicability(args)
    if applicability == "not_applicable":
        token_selector = {
            "name": "not_applicable",
            "description": "NVILA-8B-Video paper baseline path; no AutoGaze processor kwargs are injected.",
        }
    elif applicability == "enabled":
        token_selector = {
            "name": "autogaze",
            "description": "AutoGaze token selector active before SigLIP/MLLM.",
        }
    else:
        token_selector = {
            "name": "keep_all",
            "description": "HD ablation path keeps all patches while preserving NVILA-HD processor family.",
        }

    if family == MODEL_FAMILY_VIDEO_BASELINE:
        vision_encoder = {
            "name": "nvila_video_baseline_vision_metadata",
            "model_family": family,
            "metric_status": "best_effort_from_model_inputs",
        }
        mllm = {"name": "nvila_video_baseline", "model_family": family}
    else:
        vision_encoder = {
            "name": "nvila_hd_siglip",
            "model_family": family,
            "metric_status": "measured_when_remote_code_hooks_are_present",
        }
        mllm = {"name": "nvila_hd", "model_family": family}

    return {
        "token_selector": token_selector,
        "vision_encoder": vision_encoder,
        "mllm": mllm,
        "extension_note": (
            "This is a lightweight model-family adapter layer. LongVILA/Qwen2-VL can be added later "
            "behind the same token_selector / vision_encoder / mllm identity fields."
        ),
    }


def build_run_identity(args: argparse.Namespace) -> dict[str, Any]:
    family = effective_model_family(args)
    reference = paper_reference_for_args(args)
    preset = getattr(args, "paper_preset", None)
    return {
        "model_family": family,
        "paper_preset": preset,
        "model_path": effective_mllm_path(args),
        "paper_reference_score": reference["paper_reference_score"],
        "paper_reference_metric": "HLVid accuracy percent",
        "paper_reference_frames": reference["paper_reference_frames"],
        "paper_reference_max_resolution": reference["paper_reference_max_resolution"],
        "is_paper_baseline_candidate": family == MODEL_FAMILY_VIDEO_BASELINE,
        "autogaze_applicability": autogaze_applicability(args),
        "adapters": build_adapter_identity(args),
        "components": build_component_identity(args),
        "note": (
            "NVILA-8B-Video is the AutoGaze paper baseline candidate. "
            "NVILA-HD keep-all is an HD ablation, not the paper baseline."
            if family == MODEL_FAMILY_VIDEO_BASELINE or getattr(args, "gazing_mode", None) == "keep-all"
            else "NVILA-HD AutoGaze run."
        ),
    }


def metric_value(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def summarize_repeat_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for field in REPEAT_SUMMARY_FIELDS:
        values: list[float] = []
        for result in results:
            value = metric_value(result, field)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        summary[field] = compute_stats(values)
    return summary


def summary_metric(payload: dict[str, Any], field: str, stat: str = "median") -> Any:
    repeat_summary = payload.get("repeat_summary")
    if isinstance(repeat_summary, dict) and isinstance(repeat_summary.get(field), dict):
        return repeat_summary[field].get(stat)
    return metric_value(payload.get("result", {}), field)


def _first_metric(metrics: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = metrics.get(field)
        if value is not None:
            return value
    return None


def _sum_if_present(*values: Any) -> float | None:
    total = 0.0
    for value in values:
        if value is None:
            return None
        try:
            total += float(value)
        except (TypeError, ValueError):
            return None
    return total


def _subtract_if_present(total: Any, part: Any) -> float | None:
    if total is None:
        return None
    try:
        total_value = float(total)
    except (TypeError, ValueError):
        return None
    if part is None:
        return total_value
    try:
        part_value = float(part)
    except (TypeError, ValueError):
        return None
    difference = total_value - part_value
    if difference < 0:
        return None
    return difference


def build_latency_accounting(metrics: dict[str, Any]) -> dict[str, Any]:
    total_ms = metrics.get("total_ms")
    preprocess_ms = metrics.get("video_preprocess_ms")
    generate_ms = metrics.get("generate_ms")
    ttft_ms = metrics.get("ttft_ms")
    gazing_info_ms = _first_metric(metrics, "gazing_info_total_ms", "autogaze_ms")
    autogaze_total_ms = _first_metric(metrics, "autogaze_total_ms", "gazing_info_total_ms", "autogaze_ms")
    preprocess_without_autogaze_ms = _first_metric(metrics, "video_preprocess_without_autogaze_ms")
    if preprocess_without_autogaze_ms is None:
        preprocess_without_autogaze_ms = _subtract_if_present(preprocess_ms, autogaze_total_ms)
    autogaze_forward_ms = _first_metric(metrics, "autogaze_model_forward_ms", "autogaze_forward_ms")
    generation_decode_estimated_ms = _first_metric(
        metrics,
        "generation_decode_after_ttft_estimated_ms",
        "decode_estimated_ms",
    )
    recomputed_total = _sum_if_present(preprocess_without_autogaze_ms, autogaze_total_ms, generate_ms)
    delta = _numeric_delta(total_ms, recomputed_total) if recomputed_total is not None else None
    legacy_recomputed_total = _sum_if_present(preprocess_ms, generate_ms)
    legacy_delta = _numeric_delta(total_ms, legacy_recomputed_total) if legacy_recomputed_total is not None else None
    hierarchy_metrics = {
        **metrics,
        "video_preprocess_without_autogaze_ms": preprocess_without_autogaze_ms,
        "autogaze_total_ms": autogaze_total_ms,
    }

    return {
        "additive_total_ms": {
            "formula": "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms",
            "total_ms": total_ms,
            "video_preprocess_without_autogaze_ms": preprocess_without_autogaze_ms,
            "autogaze_total_ms": autogaze_total_ms,
            "generate_ms": generate_ms,
            "recomputed_total_ms": recomputed_total,
            "delta_ms": delta,
            "ttft_ms_excluded_from_total": ttft_ms,
        },
        "legacy_inclusive_total_ms": {
            "formula": "total_ms = video_preprocess_ms + generate_ms",
            "total_ms": total_ms,
            "video_preprocess_ms": preprocess_ms,
            "generate_ms": generate_ms,
            "recomputed_total_ms": legacy_recomputed_total,
            "delta_ms": legacy_delta,
            "ttft_ms_excluded_from_total": ttft_ms,
        },
        "nested_preprocess_breakdown_ms": {
            "video_preprocess_ms": {
                "value": preprocess_ms,
                "relationship": "legacy inclusive field",
                "includes": ["video_preprocess_without_autogaze_ms", "autogaze_total_ms"],
                "add_to_total_ms": False,
            },
            "video_preprocess_without_autogaze_ms": {
                "value": preprocess_without_autogaze_ms,
                "included_in": "total_ms",
                "add_to_total_ms": True,
            },
            "video_decode_ms": {
                "value": metrics.get("video_decode_ms"),
                "included_in": "video_preprocess_without_autogaze_ms",
                "add_to_total_ms": False,
            },
            "video_tiling_ms": {
                "value": metrics.get("video_tiling_ms"),
                "included_in": "video_preprocess_without_autogaze_ms",
                "add_to_total_ms": False,
            },
            "autogaze_total_ms": {
                "value": autogaze_total_ms,
                "included_in": "total_ms",
                "add_to_total_ms": True,
            },
            "gazing_info_total_ms": {
                "value": gazing_info_ms,
                "included_in": "autogaze_total_ms",
                "add_to_total_ms": False,
            },
            "autogaze_model_forward_ms": {
                "value": autogaze_forward_ms,
                "included_in": "autogaze_total_ms",
                "add_to_total_ms": False,
            },
        },
        "nested_generate_breakdown_ms": {
            "vision_encoder_ms": {
                "value": metrics.get("vision_encoder_ms"),
                "included_in": "generate_ms",
                "add_to_total_ms": False,
            },
            "siglip_vision_ms": {
                "value": metrics.get("siglip_vision_ms"),
                "included_in": "vision_encoder_ms",
                "add_to_total_ms": False,
            },
            "mm_projector_ms": {
                "value": metrics.get("mm_projector_ms"),
                "included_in": "vision_encoder_ms",
                "add_to_total_ms": False,
            },
            "llm_forward_ms": {
                "value": metrics.get("llm_forward_ms"),
                "included_in": "generate_ms",
                "add_to_total_ms": False,
            },
            "generation_decode_after_ttft_estimated_ms": {
                "value": generation_decode_estimated_ms,
                "included_in": "generate_ms",
                "add_to_total_ms": False,
                "note": "This is generation decode time estimated as generate_ms - ttft_ms, not video decode time.",
            },
        },
        "do_not_sum_with_total_ms": [
            "video_preprocess_ms",
            "video_decode_ms",
            "video_tiling_ms",
            "autogaze_ms",
            "gazing_info_total_ms",
            "autogaze_forward_ms",
            "autogaze_model_forward_ms",
            "vision_encoder_ms",
            "siglip_vision_ms",
            "mm_projector_ms",
            "llm_forward_ms",
            "ttft_ms",
            "decode_estimated_ms",
            "generation_decode_after_ttft_estimated_ms",
        ],
        "note": (
            "Only additive_total_ms fields are intended for recomputing total latency. "
            "video_preprocess_ms is kept as a legacy inclusive field; the primary breakdown separates "
            "preprocess_without_autogaze, autogaze_total, and generate."
        ),
        "hierarchy": latency_hierarchy_summary(hierarchy_metrics),
    }


def _numeric_delta(before: Any, after: Any) -> float | int | None:
    if before is None or after is None:
        return None
    try:
        before_value = float(before)
        after_value = float(after)
    except (TypeError, ValueError):
        return None
    delta = before_value - after_value
    if isinstance(before, int) and isinstance(after, int):
        return int(delta)
    return delta


def _reduction_percent(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    try:
        before_value = float(before)
        after_value = float(after)
    except (TypeError, ValueError):
        return None
    if before_value == 0:
        return None
    return round((1.0 - (after_value / before_value)) * 100.0, 6)


def build_autogaze_token_summary(token_metrics: dict[str, Any]) -> dict[str, Any]:
    autogaze_input = token_metrics.get("autogaze_input_patch_tokens")
    autogaze_selected = token_metrics.get("autogaze_selected_patch_tokens")
    raw_tile = token_metrics.get("encoder_raw_tile_patch_tokens")
    selected_tile = token_metrics.get("encoder_autogaze_selected_tile_patch_tokens")
    raw_thumbnail = token_metrics.get("encoder_raw_thumbnail_patch_tokens")
    selected_thumbnail = token_metrics.get("encoder_autogaze_selected_thumbnail_patch_tokens")
    raw_total = token_metrics.get("encoder_raw_patch_tokens")
    selected_total = token_metrics.get("encoder_autogaze_selected_patch_tokens")
    keep_all_visual = token_metrics.get("llm_keep_all_visual_tokens_estimated")
    actual_visual = token_metrics.get("llm_actual_visual_tokens")
    return {
        "frame_basis": {
            "video_sampled_frames": token_metrics.get("video_sampled_frames"),
            "thumbnail_sampled_frames": token_metrics.get("thumbnail_sampled_frames"),
            "spatial_tiles_per_video": token_metrics.get("spatial_tiles_per_video"),
            "temporal_chunks_per_video": token_metrics.get("temporal_chunks_per_video"),
            "encoder_patches_per_frame_multiscale": token_metrics.get("encoder_patches_per_frame_multiscale"),
        },
        "patch_space_basis": {
            "autogaze_target_scales": token_metrics.get("autogaze_target_scales"),
            "autogaze_target_patch_size": token_metrics.get("autogaze_target_patch_size"),
            "autogaze_coordinate_patches_per_frame_multiscale": token_metrics.get(
                "autogaze_coordinate_patches_per_frame_multiscale"
            ),
            "autogaze_coordinate_patches_per_frame_by_scale": token_metrics.get(
                "autogaze_coordinate_patches_per_frame_by_scale"
            ),
            "vision_encoder_scales": token_metrics.get("vision_encoder_scales"),
            "vision_encoder_patch_size": token_metrics.get("vision_encoder_patch_size"),
            "vision_encoder_patches_per_frame_multiscale": token_metrics.get(
                "vision_encoder_patches_per_frame_multiscale"
            ),
            "vision_encoder_patches_per_frame_by_scale": token_metrics.get(
                "vision_encoder_patches_per_frame_by_scale"
            ),
            "patch_space_mismatch": token_metrics.get("patch_space_mismatch"),
            "note": token_metrics.get("patch_space_note"),
        },
        "autogaze_input_breakdown": build_autogaze_input_breakdown(token_metrics),
        "autogaze_selection_patch_tokens": {
            "input_patch_tokens": autogaze_input,
            "selected_patch_tokens": autogaze_selected,
            "removed_patch_tokens": token_metrics.get("autogaze_removed_patch_tokens")
            if token_metrics.get("autogaze_removed_patch_tokens") is not None
            else _numeric_delta(autogaze_input, autogaze_selected),
            "reduction_ratio": token_metrics.get("autogaze_patch_reduction_ratio")
            or token_metrics.get("encoder_tile_token_reduction_ratio"),
            "reduction_percent": _reduction_percent(autogaze_input, autogaze_selected),
            "scope": (
                "Tiled-video encoder patch positions passed to AutoGaze before selection. "
                "Thumbnail patches are not included because thumbnails are keep-all in this runner."
            ),
        },
        "encoder_patch_tokens_before_siglip": {
            "raw_tile_patch_tokens": raw_tile,
            "selected_tile_patch_tokens": selected_tile,
            "removed_tile_patch_tokens": _numeric_delta(raw_tile, selected_tile),
            "raw_thumbnail_patch_tokens": raw_thumbnail,
            "selected_thumbnail_patch_tokens": selected_thumbnail,
            "removed_thumbnail_patch_tokens": _numeric_delta(raw_thumbnail, selected_thumbnail),
            "raw_total_patch_tokens": raw_total,
            "selected_total_patch_tokens": selected_total,
            "removed_total_patch_tokens": _numeric_delta(raw_total, selected_total),
            "reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
            "reduction_percent": _reduction_percent(raw_total, selected_total),
            "selected_token_definition": (
                "Non-padded AutoGaze-selected encoder patch positions before TokenShuffle, "
                "SigLIP, and the MLLM projector. Thumbnails are keep-all in this runner."
            ),
        },
        "llm_visual_tokens_after_token_shuffle": {
            "keep_all_visual_tokens_estimated": keep_all_visual,
            "actual_visual_tokens": actual_visual,
            "removed_visual_tokens_estimated": _numeric_delta(keep_all_visual, actual_visual),
            "reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
            "reduction_percent": _reduction_percent(keep_all_visual, actual_visual),
            "token_definition": (
                "Visual placeholder tokens consumed by the LLM after TokenShuffle/projector input "
                "packing; this is the token count that drives LLM prefill/KV cache estimates."
            ),
        },
    }


def _single_or_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
        return list(value)
    return value


def _first_int(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_single_scale_dense_vision_budget(
    *,
    video_frames: Any,
    thumbnail_frames: Any,
    spatial_tiles: Any,
    tile_frame_instances: Any,
    selected_total_patch_tokens: Any,
    multiscale_total_patch_tokens: Any,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    patches_per_tile = (NVILA_IMAGE_SIZE // NVILA_VISION_PATCH_SIZE) ** 2
    frame_count = _first_int(video_frames)
    thumbnail_count = _first_int(thumbnail_frames) or 0
    tile_count = _first_int(spatial_tiles)
    tile_frame_count = _first_int(tile_frame_instances)
    if tile_frame_count is None and frame_count is not None and tile_count is not None:
        tile_frame_count = frame_count * tile_count

    tile_patch_tokens = tile_frame_count * patches_per_tile if tile_frame_count is not None else None
    thumbnail_patch_tokens = thumbnail_count * patches_per_tile
    total_patch_tokens = (
        tile_patch_tokens + thumbnail_patch_tokens if tile_patch_tokens is not None else None
    )

    tile_visual_tokens = None
    if frame_count is not None and tile_count is not None:
        tile_visual_tokens = frame_count * math.ceil(tile_count * patches_per_tile / max(token_shuffle, 1))
    elif tile_patch_tokens is not None:
        tile_visual_tokens = math.ceil(tile_patch_tokens / max(token_shuffle, 1))
    thumbnail_visual_tokens = thumbnail_count * math.ceil(patches_per_tile / max(token_shuffle, 1))
    llm_visual_tokens = (
        tile_visual_tokens + thumbnail_visual_tokens if tile_visual_tokens is not None else None
    )

    return {
        "comparison_scope": "siglip_392px_single_scale_reference",
        "reference_scale_px": NVILA_IMAGE_SIZE,
        "reference_patch_size": NVILA_VISION_PATCH_SIZE,
        "patch_positions_per_tile_frame": patches_per_tile,
        "spatial_tiles_per_frame": tile_count,
        "tile_frame_instances": tile_frame_count,
        "thumbnail_frames": thumbnail_count,
        "tile_patch_tokens": tile_patch_tokens,
        "thumbnail_patch_tokens": thumbnail_patch_tokens,
        "total_patch_tokens": total_patch_tokens,
        "llm_visual_tokens_estimated": llm_visual_tokens,
        "token_shuffle": token_shuffle,
        "ratio_over_autogaze_selected_total_patch_tokens": _safe_ratio(
            total_patch_tokens,
            selected_total_patch_tokens,
        ),
        "ratio_over_hd_multiscale_keep_all_total_patch_tokens": _safe_ratio(
            total_patch_tokens,
            multiscale_total_patch_tokens,
        ),
        "note": (
            "Reference-only dense SigLIP baseline using only the 392px scale: "
            "392/14 = 28, so 28*28 = 784 patch positions per 392x392 tile-frame. "
            "NVILA-HD AutoGaze keep-all uses the multiscale 56+112+196+392 space instead."
        ),
    }


def build_processing_budget_summary(
    *,
    video_input_summary: dict[str, Any],
    token_metrics: dict[str, Any],
    runner: str,
) -> dict[str, Any]:
    spatial_tiles = _single_or_list(token_metrics.get("spatial_tiles_per_video"))
    temporal_chunks = _single_or_list(token_metrics.get("temporal_chunks_per_video"))
    thumbnail_frames = token_metrics.get("thumbnail_sampled_frames")
    raw_total = token_metrics.get("encoder_raw_patch_tokens")
    selected_total = token_metrics.get("encoder_autogaze_selected_patch_tokens")
    raw_tile = token_metrics.get("encoder_raw_tile_patch_tokens")
    selected_tile = token_metrics.get("encoder_autogaze_selected_tile_patch_tokens")
    raw_thumbnail = token_metrics.get("encoder_raw_thumbnail_patch_tokens")
    selected_thumbnail = token_metrics.get("encoder_autogaze_selected_thumbnail_patch_tokens")
    keep_all_visual = token_metrics.get("llm_keep_all_visual_tokens_estimated")
    actual_visual = token_metrics.get("llm_actual_visual_tokens")
    single_scale_dense = build_single_scale_dense_vision_budget(
        video_frames=token_metrics.get("video_sampled_frames") or video_input_summary.get("actual_video_frames"),
        thumbnail_frames=thumbnail_frames or video_input_summary.get("actual_thumbnail_frames"),
        spatial_tiles=spatial_tiles,
        tile_frame_instances=token_metrics.get("autogaze_input_tile_frame_instances"),
        selected_total_patch_tokens=selected_total,
        multiscale_total_patch_tokens=raw_total,
        token_shuffle=int(token_metrics.get("token_shuffle") or NVILA_TOKEN_SHUFFLE),
    )
    return {
        "runner": runner,
        "video": {
            "source_resolution": video_input_summary.get("source_resolution"),
            "source_width": video_input_summary.get("source_width"),
            "source_height": video_input_summary.get("source_height"),
            "processor_input_resolution": video_input_summary.get("processor_input_resolution"),
            "processor_input_width": video_input_summary.get("processor_input_width"),
            "processor_input_height": video_input_summary.get("processor_input_height"),
            "resize_enabled": video_input_summary.get("runner_resize_enabled"),
            "resize_request": video_input_summary.get("runner_resize_request"),
            "requested_video_frames": video_input_summary.get("requested_video_frames"),
            "actual_video_frames": token_metrics.get("video_sampled_frames")
            or video_input_summary.get("actual_video_frames"),
        },
        "model_processing_unit": {
            "name": "nvila_392px_spatial_tile_sequence",
            "tile_size_px": NVILA_IMAGE_SIZE,
            "patch_space": "multiscale_patch_positions_before_siglip",
            "token_shuffle": token_metrics.get("token_shuffle"),
        },
        "tiling": {
            "spatial_tiles_per_frame": spatial_tiles,
            "temporal_chunks": temporal_chunks,
            "tile_frame_instances": token_metrics.get("autogaze_input_tile_frame_instances"),
            "tile_sequences": token_metrics.get("tile_sequences"),
            "interpretation": "video frames are spatially tiled, then each tile-frame contributes multiscale patch positions",
        },
        "thumbnail": {
            "enabled": bool(thumbnail_frames),
            "requested_frames": video_input_summary.get("requested_thumbnail_frames"),
            "actual_frames": thumbnail_frames or video_input_summary.get("actual_thumbnail_frames"),
            "raw_patch_tokens": raw_thumbnail,
            "selected_patch_tokens": selected_thumbnail,
            "policy": "keep_all" if thumbnail_frames else "not_applicable",
        },
        "single_scale_dense_vision_budget": single_scale_dense,
        "multiscale_patch_space": {
            "patch_positions_per_tile_frame": token_metrics.get("encoder_patches_per_frame_multiscale"),
            "patch_positions_by_scale": token_metrics.get("encoder_patches_per_frame_by_scale"),
            "autogaze_target_scales": token_metrics.get("autogaze_target_scales"),
            "autogaze_target_patch_size": token_metrics.get("autogaze_target_patch_size"),
            "vision_encoder_scales": token_metrics.get("vision_encoder_scales"),
            "vision_encoder_patch_size": token_metrics.get("vision_encoder_patch_size"),
            "patch_space_mismatch": token_metrics.get("patch_space_mismatch"),
            "note": token_metrics.get("patch_space_note"),
        },
        "patch_budget_before_siglip": {
            "keep_all_tile_patch_tokens": raw_tile,
            "autogaze_selected_tile_patch_tokens": selected_tile,
            "removed_tile_patch_tokens": _numeric_delta(raw_tile, selected_tile),
            "keep_all_thumbnail_patch_tokens": raw_thumbnail,
            "autogaze_selected_thumbnail_patch_tokens": selected_thumbnail,
            "removed_thumbnail_patch_tokens": _numeric_delta(raw_thumbnail, selected_thumbnail),
            "keep_all_total_patch_tokens": raw_total,
            "autogaze_selected_total_patch_tokens": selected_total,
            "removed_total_patch_tokens": _numeric_delta(raw_total, selected_total),
            "autogaze_input_tile_patch_tokens": token_metrics.get("autogaze_input_patch_tokens"),
            "autogaze_selected_tile_input_patch_tokens": token_metrics.get("autogaze_selected_patch_tokens"),
            "tile_patch_reduction_ratio": token_metrics.get("encoder_tile_token_reduction_ratio")
            or token_metrics.get("autogaze_patch_reduction_ratio"),
            "total_patch_reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
            "total_patch_reduction_percent": _reduction_percent(raw_total, selected_total),
            "thumbnail_policy": "keep_all" if thumbnail_frames else "not_applicable",
            "unit": "encoder patch positions before SigLIP/TokenShuffle/projector",
        },
        "llm_visual_budget": {
            "keep_all_visual_tokens_estimated": keep_all_visual,
            "actual_visual_tokens": actual_visual,
            "removed_visual_tokens_estimated": _numeric_delta(keep_all_visual, actual_visual),
            "visual_token_reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
            "visual_token_reduction_percent": _reduction_percent(keep_all_visual, actual_visual),
            "unit": "LLM visual placeholder tokens after TokenShuffle/projector packing",
        },
    }


def _integer_division_or_float(numerator: Any, denominator: Any) -> int | float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    try:
        numerator_value = int(numerator)
        denominator_value = int(denominator)
    except (TypeError, ValueError):
        return None
    if denominator_value == 0:
        return None
    if numerator_value % denominator_value == 0:
        return numerator_value // denominator_value
    return numerator_value / denominator_value


def build_autogaze_input_breakdown(token_metrics: dict[str, Any]) -> dict[str, Any]:
    autogaze_input = token_metrics.get("autogaze_input_patch_tokens")
    patches_per_tile_frame = token_metrics.get("encoder_patches_per_frame_multiscale")
    tile_frame_instances = token_metrics.get("autogaze_input_tile_frame_instances")
    if tile_frame_instances is None:
        tile_frame_instances = _integer_division_or_float(autogaze_input, patches_per_tile_frame)

    expanded_formula = None
    if tile_frame_instances is not None and patches_per_tile_frame is not None and autogaze_input is not None:
        expanded_formula = (
            f"{tile_frame_instances} tile-frame instances * "
            f"{patches_per_tile_frame} multiscale patch positions = {autogaze_input}"
        )

    return {
        "formula": "tile_frame_instances * multiscale_patch_positions_per_tile_frame",
        "expanded_formula": expanded_formula,
        "video_sampled_frames": token_metrics.get("video_sampled_frames"),
        "spatial_tiles_per_frame": _single_or_list(token_metrics.get("spatial_tiles_per_video")),
        "temporal_chunks": _single_or_list(token_metrics.get("temporal_chunks_per_video")),
        "tile_frame_instances": tile_frame_instances,
        "multiscale_patch_positions_per_tile_frame": patches_per_tile_frame,
        "patch_positions_by_scale": token_metrics.get("encoder_patches_per_frame_by_scale"),
        "input_patch_tokens": autogaze_input,
        "unit_note": (
            "These are encoder patch positions before SigLIP/TokenShuffle/MLLM, not final LLM visual tokens."
        ),
        "why_it_can_be_large": (
            "A resized video can still be split into multiple spatial tiles. "
            "For example, 128 frames * 8 tiles/frame * multiscale patch positions can exceed one million positions."
        ),
    }


def _token_budget_summary_key(field: str) -> str:
    return field.split(".")[-1]


def summarize_token_budget_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_token_metrics = [
        row for row in rows if isinstance(row.get("token_metrics"), dict)
    ]
    values_by_key: dict[str, list[float]] = {
        _token_budget_summary_key(field): [] for field in TOKEN_BUDGET_SUMMARY_FIELDS
    }
    values_by_key["encoder_removed_patch_tokens"] = []
    values_by_key["autogaze_removed_patch_tokens"] = []
    values_by_key["llm_removed_visual_tokens_estimated"] = []
    for row in rows_with_token_metrics:
        for field in TOKEN_BUDGET_SUMMARY_FIELDS:
            value = metric_value(row, field)
            if value is None:
                continue
            try:
                values_by_key[_token_budget_summary_key(field)].append(float(value))
            except (TypeError, ValueError):
                continue
        removed_patch_tokens = _numeric_delta(
            metric_value(row, "token_metrics.encoder_raw_patch_tokens"),
            metric_value(row, "token_metrics.encoder_autogaze_selected_patch_tokens"),
        )
        if removed_patch_tokens is not None:
            values_by_key["encoder_removed_patch_tokens"].append(float(removed_patch_tokens))
        removed_autogaze_patch_tokens = _numeric_delta(
            metric_value(row, "token_metrics.autogaze_input_patch_tokens"),
            metric_value(row, "token_metrics.autogaze_selected_patch_tokens"),
        )
        if removed_autogaze_patch_tokens is not None:
            values_by_key["autogaze_removed_patch_tokens"].append(float(removed_autogaze_patch_tokens))
        removed_visual_tokens = _numeric_delta(
            metric_value(row, "token_metrics.llm_keep_all_visual_tokens_estimated"),
            metric_value(row, "token_metrics.llm_actual_visual_tokens"),
        )
        if removed_visual_tokens is not None:
            values_by_key["llm_removed_visual_tokens_estimated"].append(float(removed_visual_tokens))

    stats_by_key = {
        key: compute_stats(values) for key, values in values_by_key.items()
    }
    return {
        "rows_with_token_metrics": len(rows_with_token_metrics),
        "median": {key: stats["median"] for key, stats in stats_by_key.items()},
        "mean": {key: stats["mean"] for key, stats in stats_by_key.items()},
        "stats": stats_by_key,
        "selected_token_definition": (
            "encoder_autogaze_selected_* counts non-padded AutoGaze-selected encoder patch "
            "positions before TokenShuffle/SigLIP/projector. llm_actual_visual_tokens is the "
            "visual placeholder count consumed by the LLM after TokenShuffle/projector packing."
        ),
    }


def build_single_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result", {})
    token_metrics = result.get("token_metrics", {}) if isinstance(result, dict) else {}
    compute_metrics = result.get("compute_metrics", {}) if isinstance(result, dict) else {}
    siglip_metrics = compute_metrics.get("siglip_encoder", {}) if isinstance(compute_metrics, dict) else {}
    mllm_metrics = compute_metrics.get("mllm", {}) if isinstance(compute_metrics, dict) else {}
    encoder_raw_patch_tokens = token_metrics.get("encoder_raw_patch_tokens")
    encoder_selected_patch_tokens = token_metrics.get("encoder_autogaze_selected_patch_tokens")
    llm_keep_all_visual_tokens = token_metrics.get("llm_keep_all_visual_tokens_estimated")
    llm_actual_visual_tokens = token_metrics.get("llm_actual_visual_tokens")
    gazing_info_total_median = summary_metric(payload, "gazing_info_total_ms")
    if gazing_info_total_median is None:
        gazing_info_total_median = summary_metric(payload, "autogaze_ms")
    autogaze_total_median = summary_metric(payload, "autogaze_total_ms")
    if autogaze_total_median is None:
        autogaze_total_median = gazing_info_total_median
    preprocess_total_median = summary_metric(payload, "video_preprocess_ms")
    preprocess_without_autogaze_median = summary_metric(payload, "video_preprocess_without_autogaze_ms")
    if preprocess_without_autogaze_median is None:
        preprocess_without_autogaze_median = _subtract_if_present(
            preprocess_total_median,
            autogaze_total_median,
        )
    autogaze_model_forward_median = summary_metric(payload, "autogaze_model_forward_ms")
    if autogaze_model_forward_median is None:
        autogaze_model_forward_median = summary_metric(payload, "autogaze_forward_ms")
    generation_decode_after_ttft_estimated_median = summary_metric(
        payload,
        "generation_decode_after_ttft_estimated_ms",
    )
    if generation_decode_after_ttft_estimated_median is None:
        generation_decode_after_ttft_estimated_median = summary_metric(payload, "decode_estimated_ms")
    module_latency = {
        "total_median": summary_metric(payload, "total_ms"),
        "generate_median": summary_metric(payload, "generate_ms"),
        "preprocess_without_autogaze_median": preprocess_without_autogaze_median,
        "preprocess_total_median": preprocess_total_median,
        "autogaze_median": summary_metric(payload, "autogaze_ms"),
        "autogaze_total_median": autogaze_total_median,
        "gazing_info_total_median": gazing_info_total_median,
        "autogaze_model_forward_median": autogaze_model_forward_median,
        "vit_encoder_median": summary_metric(payload, "siglip_vision_ms"),
        "llm_median": summary_metric(payload, "llm_forward_ms"),
    }
    latency_accounting = build_latency_accounting(
        {
            "total_ms": summary_metric(payload, "total_ms"),
            "video_preprocess_ms": preprocess_total_median,
            "video_preprocess_without_autogaze_ms": preprocess_without_autogaze_median,
            "autogaze_total_ms": autogaze_total_median,
            "generate_ms": summary_metric(payload, "generate_ms"),
            "ttft_ms": summary_metric(payload, "ttft_ms"),
            "video_decode_ms": summary_metric(payload, "video_decode_ms"),
            "video_tiling_ms": summary_metric(payload, "video_tiling_ms"),
            "gazing_info_total_ms": gazing_info_total_median,
            "autogaze_model_forward_ms": autogaze_model_forward_median,
            "vision_encoder_ms": summary_metric(payload, "vision_encoder_ms"),
            "siglip_vision_ms": summary_metric(payload, "siglip_vision_ms"),
            "mm_projector_ms": summary_metric(payload, "mm_projector_ms"),
            "llm_forward_ms": summary_metric(payload, "llm_forward_ms"),
            "generation_decode_after_ttft_estimated_ms": generation_decode_after_ttft_estimated_median,
        }
    )
    key_token_summary = {
        "video_sampled_frames": token_metrics.get("video_sampled_frames"),
        "thumbnail_sampled_frames": token_metrics.get("thumbnail_sampled_frames"),
        "encoder_patch_tokens_before_keep_all_or_raw": encoder_raw_patch_tokens,
        "encoder_patch_tokens_after_autogaze": encoder_selected_patch_tokens,
        "encoder_token_reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
        "autogaze_input_tile_patch_tokens": token_metrics.get("autogaze_input_patch_tokens"),
        "autogaze_selected_tile_patch_tokens": token_metrics.get("autogaze_selected_patch_tokens"),
        "autogaze_patch_reduction_ratio": token_metrics.get("autogaze_patch_reduction_ratio"),
        "llm_visual_tokens_before_keep_all_estimated": llm_keep_all_visual_tokens,
        "llm_visual_tokens_after_actual": llm_actual_visual_tokens,
        "llm_visual_token_reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
    }
    key_memory_summary = {
        "processor_peak_median": summary_metric(payload, "processor_peak_memory_bytes"),
        "ttft_peak_median": summary_metric(payload, "ttft_peak_memory_bytes"),
        "llm_peak_median": summary_metric(payload, "llm_peak_memory_bytes"),
        "overall_peak_median": summary_metric(payload, "peak_memory_bytes"),
    }
    question = payload.get("question")
    if question is None and isinstance(result, dict):
        question = result.get("question")
    return {
        "model_path": payload.get("model_path"),
        "run_identity": payload.get("run_identity") or result.get("run_identity"),
        "autogaze_runtime_config": payload.get("autogaze_runtime_config") or result.get("autogaze_runtime_config"),
        "gazing_mode": payload.get("gazing_mode"),
        "video": payload.get("video"),
        "prompt": payload.get("prompt"),
        "question": question,
        "video_input_summary": result.get("video_input_summary") if isinstance(result, dict) else None,
        "processing_budget_summary": result.get("processing_budget_summary") if isinstance(result, dict) else None,
        "autogaze_token_summary": build_autogaze_token_summary(token_metrics),
        "key_autogaze_effect": {
            "gazing_mode": payload.get("gazing_mode"),
            "total_ms_median": summary_metric(payload, "total_ms"),
            "ttft_ms_median": summary_metric(payload, "ttft_ms"),
            "autogaze_total_ms_median": autogaze_total_median,
            "autogaze_forward_ms_median": summary_metric(payload, "autogaze_forward_ms"),
            "gazing_info_total_ms_median": gazing_info_total_median,
            "autogaze_model_forward_ms_median": autogaze_model_forward_median,
            "siglip_vision_ms_median": summary_metric(payload, "siglip_vision_ms"),
            "llm_forward_ms_median": summary_metric(payload, "llm_forward_ms"),
            "encoder_patch_tokens_before_keep_all": encoder_raw_patch_tokens,
            "encoder_patch_tokens_after_actual": encoder_selected_patch_tokens,
            "encoder_patch_tokens_removed": _numeric_delta(encoder_raw_patch_tokens, encoder_selected_patch_tokens),
            "encoder_patch_reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
            "encoder_patch_reduction_percent": _reduction_percent(
                encoder_raw_patch_tokens,
                encoder_selected_patch_tokens,
            ),
            "llm_visual_tokens_before_keep_all_estimated": llm_keep_all_visual_tokens,
            "llm_visual_tokens_after_actual": llm_actual_visual_tokens,
            "llm_visual_tokens_removed_estimated": _numeric_delta(
                llm_keep_all_visual_tokens,
                llm_actual_visual_tokens,
            ),
            "llm_visual_token_reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
            "llm_visual_token_reduction_percent": _reduction_percent(
                llm_keep_all_visual_tokens,
                llm_actual_visual_tokens,
            ),
            "siglip_total_macs_reduction_ratio": siglip_metrics.get("keep_all_to_actual_total_macs_ratio"),
            "mllm_prefill_context_reduction_ratio": mllm_metrics.get("prefill_context_reduction_ratio"),
            "mllm_kv_cache_reduction_ratio": mllm_metrics.get("kv_cache_reduction_ratio"),
            "llm_peak_memory_bytes_median": summary_metric(payload, "llm_peak_memory_bytes"),
        },
        "answer": result.get("raw_output"),
        "parsed_answer": result.get("parsed_answer"),
        "generated_tokens": result.get("generated_tokens"),
        "module_latency_ms": {
            **module_latency,
            "field_note": (
                "Summary-level module latency is intentionally coarse. "
                "preprocess_without_autogaze=video_preprocess_without_autogaze_ms, "
                "preprocess_total=legacy inclusive video_preprocess_ms, autogaze=autogaze_total_ms, "
                "gazing_info_total=gazing_info_total_ms, "
                "autogaze_model_forward=autogaze_model_forward_ms, "
                "vit_encoder=siglip_vision_ms, llm=llm_forward_ms. "
                "The primary additive formula is preprocess_without_autogaze + autogaze_total + generate."
            ),
        },
        "latency_accounting": latency_accounting,
        "key_metrics_summary": {
            "latency_ms": module_latency,
            "latency_accounting": latency_accounting,
            "tokens": key_token_summary,
            "memory_bytes": key_memory_summary,
        },
        "latency_ms": {
            "total_median": summary_metric(payload, "total_ms"),
            "generate_median": summary_metric(payload, "generate_ms"),
            "ttft_median": summary_metric(payload, "ttft_ms"),
            "decode_estimated_median": summary_metric(payload, "decode_estimated_ms"),
            "generation_decode_after_ttft_estimated_median": generation_decode_after_ttft_estimated_median,
            "preprocess_without_autogaze_median": preprocess_without_autogaze_median,
            "preprocess_total_median": preprocess_total_median,
            "video_decode_median": summary_metric(payload, "video_decode_ms"),
            "video_tiling_median": summary_metric(payload, "video_tiling_ms"),
            "autogaze_total_median": autogaze_total_median,
            "autogaze_forward_median": summary_metric(payload, "autogaze_forward_ms"),
            "gazing_info_total_median": gazing_info_total_median,
            "autogaze_model_forward_median": autogaze_model_forward_median,
            "siglip_vision_median": summary_metric(payload, "siglip_vision_ms"),
            "mm_projector_median": summary_metric(payload, "mm_projector_ms"),
            "llm_forward_median": summary_metric(payload, "llm_forward_ms"),
        },
        "memory_bytes": {
            "processor_peak_median": summary_metric(payload, "processor_peak_memory_bytes"),
            "ttft_peak_median": summary_metric(payload, "ttft_peak_memory_bytes"),
            "llm_peak_median": summary_metric(payload, "llm_peak_memory_bytes"),
            "overall_peak_median": summary_metric(payload, "peak_memory_bytes"),
        },
        "tokens": {
            "encoder_raw_patch_tokens": token_metrics.get("encoder_raw_patch_tokens"),
            "encoder_selected_patch_tokens": token_metrics.get("encoder_autogaze_selected_patch_tokens"),
            "encoder_token_reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
            "encoder_tile_token_reduction_ratio": token_metrics.get("encoder_tile_token_reduction_ratio"),
            "llm_keep_all_visual_tokens_estimated": token_metrics.get("llm_keep_all_visual_tokens_estimated"),
            "llm_actual_visual_tokens": token_metrics.get("llm_actual_visual_tokens"),
            "llm_visual_token_reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
        },
        "compute": {
            "siglip_attention_macs_reduction_ratio": siglip_metrics.get(
                "keep_all_to_actual_attention_macs_ratio"
            ),
            "siglip_mlp_macs_reduction_ratio": siglip_metrics.get("keep_all_to_actual_mlp_macs_ratio"),
            "siglip_total_macs_reduction_ratio": siglip_metrics.get("keep_all_to_actual_total_macs_ratio"),
            "mllm_prefill_context_reduction_ratio": mllm_metrics.get("prefill_context_reduction_ratio"),
            "mllm_kv_cache_reduction_ratio": mllm_metrics.get("kv_cache_reduction_ratio"),
            "mllm_prefill_total_macs_reduction_ratio": mllm_metrics.get(
                "prefill_total_macs_reduction_ratio"
            ),
        },
    }


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


def stream_pts_per_frame(*, average_rate: Fraction | None, time_base: Fraction | None) -> Fraction | None:
    if average_rate is None or time_base is None or average_rate == 0 or time_base == 0:
        return None
    return Fraction(average_rate.denominator, average_rate.numerator) / time_base


def frame_index_to_pts(frame_index: int, *, pts_per_frame: Fraction, start_time: int | None = 0) -> int:
    return int(round(Fraction(start_time or 0, 1) + frame_index * pts_per_frame))


def pts_to_frame_index(pts: int, *, pts_per_frame: Fraction, start_time: int | None = 0) -> int:
    return int(round(Fraction(pts - (start_time or 0), 1) / pts_per_frame))


def build_seek_decode_groups(
    *,
    target_indices: list[int],
    keyframe_indices: list[int],
) -> list[dict[str, Any]]:
    if not target_indices:
        return []
    targets = sorted(set(int(index) for index in target_indices))
    keyframes = sorted(set(int(index) for index in keyframe_indices))
    groups_by_seek: dict[int, list[int]] = {}
    for target in targets:
        key_pos = bisect_right(keyframes, target) - 1
        seek_index = keyframes[key_pos] if key_pos >= 0 else target
        groups_by_seek.setdefault(seek_index, []).append(target)
    return [
        {"seek_frame_index": seek_index, "target_indices": groups_by_seek[seek_index]}
        for seek_index in sorted(groups_by_seek)
    ]


def nvila_thumbnail_indices(sampled_frame_indices: list[int], thumbnail_count: int) -> list[int]:
    if thumbnail_count <= 0:
        return []
    if len(sampled_frame_indices) > thumbnail_count:
        step = len(sampled_frame_indices) // thumbnail_count
        return sampled_frame_indices[::step][:thumbnail_count]
    return list(sampled_frame_indices)


def estimate_stream_profile_plan(
    *,
    width: int,
    height: int,
    source_frames: int | None,
    num_video_frames: int,
    num_video_frames_thumbnail: int,
    max_tiles_video: int,
    chunk_frames: int,
    max_batch_size_autogaze: int | None = None,
    scales: list[int] | None = None,
    patch_size: int = NVILA_TARGET_PATCH_SIZE,
    image_size: int = NVILA_IMAGE_SIZE,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    if num_video_frames <= 0:
        raise ValueError("num_video_frames must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    grid = spatial_tile_grid(width, height, max_tiles_video, image_size)
    temporal_chunks = math.ceil(num_video_frames / chunk_frames)
    thumbnail_frames = min(num_video_frames, max(num_video_frames_thumbnail, 0))
    per_frame_patches = patches_per_frame(scales, patch_size)
    patch_breakdown = {str(scale): (scale // patch_size) ** 2 for scale in (scales or NVILA_TARGET_SCALES)}
    raw_tile_patches = num_video_frames * grid["tiles"] * per_frame_patches
    raw_thumbnail_patches = thumbnail_frames * per_frame_patches
    keep_all_tile_tokens = num_video_frames * math.ceil(grid["tiles"] * per_frame_patches / token_shuffle)
    keep_all_thumbnail_tokens = thumbnail_frames * math.ceil(per_frame_patches / token_shuffle)
    autogaze_batch_tile_sequences = min(grid["tiles"], max_batch_size_autogaze or grid["tiles"])
    autogaze_tensor_bytes_per_batch = chunk_frames * autogaze_batch_tile_sequences * 3 * image_size * image_size * 4
    autogaze_tensor_bytes_full_chunk = chunk_frames * grid["tiles"] * 3 * image_size * image_size * 4

    return {
        "video": {
            "width": width,
            "height": height,
            "source_frames": source_frames,
        },
        "sampling": {
            "requested_frames": num_video_frames,
            "thumbnail_frames": thumbnail_frames,
            "policy": "nvila_round_linspace_over_full_video",
        },
        "tiling": {
            "cols": grid["cols"],
            "rows": grid["rows"],
            "spatial_tiles": grid["tiles"],
            "tile_size": image_size,
        },
        "chunking": {
            "chunk_frames": chunk_frames,
            "temporal_chunks": temporal_chunks,
            "tile_sequences": temporal_chunks * grid["tiles"],
            "last_chunk_frames": num_video_frames - (temporal_chunks - 1) * chunk_frames,
        },
        "tokens": {
            "encoder_patches_per_frame_multiscale": per_frame_patches,
            "encoder_patches_per_frame_by_scale": patch_breakdown,
            "encoder_raw_tile_patch_tokens": raw_tile_patches,
            "encoder_raw_thumbnail_patch_tokens": raw_thumbnail_patches,
            "encoder_raw_patch_tokens": raw_tile_patches + raw_thumbnail_patches,
            "llm_keep_all_tile_visual_tokens_estimated": keep_all_tile_tokens,
            "llm_keep_all_thumbnail_visual_tokens_estimated": keep_all_thumbnail_tokens,
            "llm_keep_all_visual_tokens_estimated": keep_all_tile_tokens + keep_all_thumbnail_tokens,
        },
        "memory": {
            "streaming_raw_frame_buffer_bytes": chunk_frames * width * height * 3,
            "streaming_tile_pil_buffer_bytes": chunk_frames * grid["tiles"] * image_size * image_size * 3,
            "autogaze_batch_tile_sequences": autogaze_batch_tile_sequences,
            "streaming_autogaze_tile_tensor_bytes": autogaze_tensor_bytes_per_batch,
            "streaming_autogaze_tile_tensor_bytes_per_batch": autogaze_tensor_bytes_per_batch,
            "streaming_autogaze_tile_tensor_bytes_full_chunk": autogaze_tensor_bytes_full_chunk,
            "thumbnail_tensor_bytes": thumbnail_frames * 3 * image_size * image_size * 4,
            "note": "Streaming estimates only keep one temporal chunk of raw frames and tiles before AutoGaze/SigLIP-style work; final NVILA generation still needs collected visual tokens.",
        },
        "streaming_boundary": {
            "pre_llm_stages_can_stream": True,
            "llm_generation_requires_collected_visual_tokens": True,
            "note": "Decode, resize, spatial tiling, AutoGaze tensorization, and AutoGaze forward can be measured per chunk. NVILA LLM prefill/generation consumes the accumulated visual token sequence.",
        },
    }


def build_stream_profile_token_metrics(plan: dict[str, Any], tile_summary: dict[str, Any]) -> dict[str, Any]:
    raw_tile_patches = int(plan["tokens"]["encoder_raw_tile_patch_tokens"])
    raw_thumbnail_patches = int(plan["tokens"]["encoder_raw_thumbnail_patch_tokens"])
    raw_total = raw_tile_patches + raw_thumbnail_patches
    patches_per_tile_frame = int(plan["tokens"]["encoder_patches_per_frame_multiscale"])
    tile_frame_instances = raw_tile_patches // patches_per_tile_frame if patches_per_tile_frame else 0
    selected_tile_patches = int(tile_summary.get("selected_non_padded_patches", raw_tile_patches))
    selected_thumbnail_patches = raw_thumbnail_patches
    selected_total = selected_tile_patches + selected_thumbnail_patches
    padded_tile = int(tile_summary.get("padded_gazing_positions", 0))
    total_tile_slots = int(tile_summary.get("total_gaze_slots", 0))
    token_shuffle = NVILA_TOKEN_SHUFFLE

    return {
        "video_sampled_frames": int(plan["sampling"]["requested_frames"]),
        "thumbnail_sampled_frames": int(plan["sampling"]["thumbnail_frames"]),
        "tile_sequences": int(plan["chunking"]["tile_sequences"]),
        "spatial_tiles_per_video": [int(plan["tiling"]["spatial_tiles"])],
        "temporal_chunks_per_video": [int(plan["chunking"]["temporal_chunks"])],
        "encoder_patches_per_frame_multiscale": int(plan["tokens"]["encoder_patches_per_frame_multiscale"]),
        "encoder_patches_per_frame_by_scale": plan["tokens"]["encoder_patches_per_frame_by_scale"],
        "token_shuffle": token_shuffle,
        "encoder_raw_tile_patch_tokens": raw_tile_patches,
        "encoder_autogaze_selected_tile_patch_tokens": selected_tile_patches,
        "autogaze_input_tile_frame_instances": tile_frame_instances,
        "autogaze_input_patch_tokens": raw_tile_patches,
        "autogaze_selected_patch_tokens": selected_tile_patches,
        "autogaze_removed_patch_tokens": raw_tile_patches - selected_tile_patches,
        "autogaze_patch_reduction_ratio": _safe_ratio(raw_tile_patches, selected_tile_patches),
        "encoder_autogaze_padded_tile_patch_tokens": padded_tile,
        "encoder_autogaze_total_tile_gaze_slots": total_tile_slots,
        "encoder_tile_token_reduction_ratio": _safe_ratio(raw_tile_patches, selected_tile_patches),
        "encoder_raw_thumbnail_patch_tokens": raw_thumbnail_patches,
        "encoder_autogaze_selected_thumbnail_patch_tokens": selected_thumbnail_patches,
        "encoder_autogaze_padded_thumbnail_patch_tokens": 0,
        "encoder_autogaze_total_thumbnail_gaze_slots": 0,
        "encoder_thumbnail_token_reduction_ratio": _safe_ratio(raw_thumbnail_patches, selected_thumbnail_patches),
        "encoder_raw_patch_tokens": raw_total,
        "encoder_raw_total_patch_tokens": raw_total,
        "encoder_autogaze_selected_patch_tokens": selected_total,
        "encoder_autogaze_selected_total_patch_tokens": selected_total,
        "encoder_autogaze_padded_patch_tokens": padded_tile,
        "encoder_autogaze_total_gaze_slots": total_tile_slots,
        "encoder_token_reduction_ratio": _safe_ratio(raw_total, selected_total),
        "llm_keep_all_tile_visual_tokens_estimated": plan["tokens"]["llm_keep_all_tile_visual_tokens_estimated"],
        "llm_keep_all_thumbnail_visual_tokens_estimated": plan["tokens"]["llm_keep_all_thumbnail_visual_tokens_estimated"],
        "llm_keep_all_visual_tokens_estimated": plan["tokens"]["llm_keep_all_visual_tokens_estimated"],
        "llm_autogaze_visual_tokens_lower_bound_estimated": math.ceil(selected_total / token_shuffle)
        if selected_total
        else 0,
        "llm_actual_visual_tokens": None,
        "llm_actual_visual_tokens_after_autogaze": None,
        "llm_visual_token_reduction_ratio": None,
    }


def build_stream_profile_compute_metrics(
    plan: dict[str, Any],
    tile_summary: dict[str, Any],
    token_metrics: dict[str, Any],
    *,
    siglip_info: dict[str, Any] | None = None,
    dtype_bytes: int = 4,
) -> dict[str, Any]:
    siglip_info = siglip_info or {}
    hidden_size = int(siglip_info.get("hidden_size") or 1152)
    intermediate_size = int(siglip_info.get("intermediate_size") or 4304)
    num_hidden_layers = int(siglip_info.get("num_hidden_layers") or 27)
    num_attention_heads = int(siglip_info.get("num_attention_heads") or 16)
    patch_tokens = int(plan["tokens"]["encoder_patches_per_frame_multiscale"])
    tile_sequences = int(plan["chunking"]["tile_sequences"])
    chunk_frames = int(plan["chunking"]["chunk_frames"])
    thumbnail_frames = int(plan["sampling"]["thumbnail_frames"])
    keep_all_tile_sequence_tokens = tile_sequences * chunk_frames * patch_tokens
    keep_all_tile_attention_pairs = tile_sequences * (chunk_frames * patch_tokens) ** 2
    thumbnail_sequence_tokens = thumbnail_frames * patch_tokens
    thumbnail_attention_pairs = thumbnail_frames * patch_tokens * patch_tokens

    actual_tile_sequence_tokens = int(
        tile_summary.get("siglip_gazed_sequence_slots_sum")
        or tile_summary.get("total_gaze_slots")
        or keep_all_tile_sequence_tokens
    )
    actual_tile_attention_pairs = int(
        tile_summary.get("siglip_gazed_sequence_slots_squared_sum")
        or (
            tile_sequences
            * math.ceil(actual_tile_sequence_tokens / max(tile_sequences, 1))
            * math.ceil(actual_tile_sequence_tokens / max(tile_sequences, 1))
        )
    )

    keep_all = estimate_siglip_encoder_compute_from_sums(
        sequence_count=tile_sequences + thumbnail_frames,
        sequence_tokens=keep_all_tile_sequence_tokens + thumbnail_sequence_tokens,
        dense_attention_pairs=keep_all_tile_attention_pairs + thumbnail_attention_pairs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        dtype_bytes=dtype_bytes,
    )
    actual = estimate_siglip_encoder_compute_from_sums(
        sequence_count=tile_sequences + thumbnail_frames,
        sequence_tokens=actual_tile_sequence_tokens + thumbnail_sequence_tokens,
        dense_attention_pairs=actual_tile_attention_pairs + thumbnail_attention_pairs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        dtype_bytes=dtype_bytes,
    )

    return {
        "metric_type": "estimated_macs_and_bytes_from_stream_token_counts",
        "note": (
            "stream-profile does not run NVILA projector/LLM. SigLIP compute fields are analytical estimates; "
            "timing_ms.siglip_* fields are measured only when --stream-run-siglip is enabled."
        ),
        "siglip_encoder": {
            "keep_all": keep_all,
            "actual": actual,
            "keep_all_to_actual_token_ratio": _safe_ratio(
                keep_all["sequence_tokens"], actual["sequence_tokens"]
            ),
            "keep_all_to_actual_dense_attention_pair_ratio": _safe_ratio(
                keep_all["dense_attention_pairs"], actual["dense_attention_pairs"]
            ),
            "keep_all_to_actual_attention_macs_ratio": _safe_ratio(
                keep_all["attention_quadratic_macs_estimated"],
                actual["attention_quadratic_macs_estimated"],
            ),
            "keep_all_to_actual_mlp_macs_ratio": _safe_ratio(
                keep_all["mlp_macs_estimated"], actual["mlp_macs_estimated"]
            ),
            "keep_all_to_actual_total_macs_ratio": _safe_ratio(
                keep_all["total_macs_estimated"], actual["total_macs_estimated"]
            ),
        },
        "mllm": {
            "full_llm_not_run_in_stream_profile": True,
            "keep_all_visual_tokens_estimated": token_metrics["llm_keep_all_visual_tokens_estimated"],
            "autogaze_visual_tokens_lower_bound_estimated": token_metrics[
                "llm_autogaze_visual_tokens_lower_bound_estimated"
            ],
            "note": (
                "Run single/hlvid mode with --measure-ttft to get actual prefill context, KV cache estimate, "
                "TTFT, and LLM peak memory."
            ),
        },
    }


def apply_resize_to_dimensions(
    *,
    width: int,
    height: int,
    shortest_edge: int | None,
    longest_edge: int | None,
    exact_width: int | None,
    exact_height: int | None,
) -> dict[str, int | str]:
    exact_requested = exact_width is not None or exact_height is not None
    if exact_requested and (exact_width is None or exact_height is None):
        raise ValueError("--video-resize-width and --video-resize-height must be provided together.")
    active_modes = sum(value is not None for value in (shortest_edge, longest_edge)) + int(exact_requested)
    if active_modes > 1:
        raise ValueError("Use only one video resize mode: exact size, shortest edge, or longest edge.")
    if exact_requested:
        return {"width": int(exact_width), "height": int(exact_height), "mode": "exact"}
    if shortest_edge is not None:
        scale = shortest_edge / min(width, height)
        return {
            "width": max(1, int(round(width * scale))),
            "height": max(1, int(round(height * scale))),
            "mode": "shortest_edge",
        }
    if longest_edge is not None:
        scale = longest_edge / max(width, height)
        return {
            "width": max(1, int(round(width * scale))),
            "height": max(1, int(round(height * scale))),
            "mode": "longest_edge",
        }
    return {"width": width, "height": height, "mode": "none"}


def has_video_resize(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None
        for name in (
            "video_resize_shortest_edge",
            "video_resize_longest_edge",
            "video_resize_width",
            "video_resize_height",
        )
    )


def resize_frame(frame: Image.Image, resize: dict[str, int | str]) -> Image.Image:
    if resize["mode"] == "none":
        return frame
    return frame.resize((int(resize["width"]), int(resize["height"])))


def _new_decode_stats(requested_strategy: str) -> dict[str, Any]:
    return {
        "requested_decode_strategy": requested_strategy,
        "decode_strategy": None,
        "decode_strategy_fallback_error": None,
        "decode_frames_read": 0,
        "decode_seek_groups": 0,
        "decode_keyframes_indexed": None,
        "decode_packets_scanned_for_keyframes": None,
    }


def _target_frame_counts(total_frames: int, sample_count: int) -> Counter[int]:
    return Counter(uniform_sample_indices(total_frames, sample_count))


def _load_sampled_video_frames_scan(
    video: str,
    target_counts: Counter[int],
    resize: dict[str, int | str],
    *,
    requested_strategy: str,
) -> tuple[list[Image.Image], dict[str, Any]]:
    stats = _new_decode_stats(requested_strategy)
    stats["decode_strategy"] = "scan"
    max_index = max(target_counts)

    frames: list[Image.Image] = []
    container = av.open(video)
    try:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index > max_index:
                break
            stats["decode_frames_read"] += 1
            count = target_counts.get(frame_index, 0)
            if count == 0:
                continue
            image = resize_frame(frame.to_image().convert("RGB"), resize)
            frames.extend(image.copy() for _ in range(count))
            if len(frames) >= sum(target_counts.values()):
                break
    finally:
        container.close()
    return frames, stats


def _load_sampled_video_frames_seek(
    video: str,
    target_counts: Counter[int],
    resize: dict[str, int | str],
    *,
    requested_strategy: str,
) -> tuple[list[Image.Image], dict[str, Any]]:
    stats = _new_decode_stats(requested_strategy)
    stats["decode_strategy"] = "seek"
    keyframe_indices, keyframe_metadata = read_video_keyframe_indices(video)
    stats["decode_keyframes_indexed"] = keyframe_metadata["keyframes"]
    stats["decode_packets_scanned_for_keyframes"] = keyframe_metadata["packets_scanned"]
    groups = build_seek_decode_groups(
        target_indices=sorted(target_counts),
        keyframe_indices=keyframe_indices,
    )
    stats["decode_seek_groups"] = len(groups)

    frames_by_index: dict[int, Image.Image] = {}
    processed_targets: set[int] = set()
    container = av.open(video)
    try:
        stream = container.streams.video[0]
        pts_per_frame = stream_pts_per_frame(average_rate=stream.average_rate, time_base=stream.time_base)
        if pts_per_frame is None:
            raise ValueError("Seek decode requires video average_rate and time_base metadata.")
        start_time = int(stream.start_time) if stream.start_time is not None else 0
        for group in groups:
            seek_pts = frame_index_to_pts(
                int(group["seek_frame_index"]),
                pts_per_frame=pts_per_frame,
                start_time=start_time,
            )
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            group_targets = set(int(index) for index in group["target_indices"])
            group_last_target = max(group_targets)
            decoder = container.decode(video=0)
            while True:
                try:
                    frame = next(decoder)
                except StopIteration:
                    break
                stats["decode_frames_read"] += 1
                if frame.pts is None:
                    continue
                frame_index = pts_to_frame_index(frame.pts, pts_per_frame=pts_per_frame, start_time=start_time)
                if frame_index in group_targets and frame_index not in processed_targets:
                    frames_by_index[frame_index] = resize_frame(frame.to_image().convert("RGB"), resize)
                    processed_targets.add(frame_index)
                if frame_index >= group_last_target:
                    break
    finally:
        container.close()

    frames: list[Image.Image] = []
    for index in sorted(target_counts):
        image = frames_by_index.get(index)
        if image is None:
            continue
        frames.extend(image.copy() for _ in range(target_counts[index]))
    return frames, stats


def load_sampled_video_frames(
    video: str,
    sample_count: int,
    resize: dict[str, int | str],
    *,
    decode_strategy: str = "auto",
) -> tuple[list[Image.Image], dict[str, Any]]:
    metadata = read_video_metadata(video)
    total_frames = metadata.get("frames")
    if total_frames is None:
        raise ValueError("Video frame count is required for runner-side resize sampling.")
    target_counts = _target_frame_counts(int(total_frames), sample_count)

    if decode_strategy not in {"auto", "seek", "scan"}:
        raise ValueError(f"Unsupported video decode strategy: {decode_strategy}")
    if decode_strategy in {"auto", "seek"}:
        try:
            frames, stats = _load_sampled_video_frames_seek(
                video,
                target_counts,
                resize,
                requested_strategy=decode_strategy,
            )
            if len(frames) >= sample_count:
                while len(frames) < sample_count:
                    frames.append(frames[-1].copy())
                return frames, stats
            if decode_strategy == "seek":
                if frames:
                    while len(frames) < sample_count:
                        frames.append(frames[-1].copy())
                    return frames, stats
                raise ValueError("Seek decode returned no frames")
        except Exception as exc:
            if decode_strategy == "seek":
                raise
            fallback_error = repr(exc)
        else:
            fallback_error = f"seek decoded {len(frames)} of {sample_count} requested frames"

        frames, stats = _load_sampled_video_frames_scan(
            video,
            target_counts,
            resize,
            requested_strategy=decode_strategy,
        )
        stats["decode_strategy_fallback_error"] = fallback_error
    else:
        frames, stats = _load_sampled_video_frames_scan(
            video,
            target_counts,
            resize,
            requested_strategy=decode_strategy,
        )

    if not frames:
        raise ValueError(f"Could not extract any frames from video: {video}")
    while len(frames) < sample_count:
        frames.append(frames[-1].copy())
    return frames, stats


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


def h100_risk_band(
    estimated_vram_gib: float,
    *,
    context_exceeded: bool = False,
    h100_budget_gib: float = H100_DEFAULT_BUDGET_GIB,
) -> str:
    if context_exceeded:
        return "context_red"
    if estimated_vram_gib >= h100_budget_gib:
        return "red"
    if estimated_vram_gib >= H100_GREEN_GIB:
        return "yellow"
    return "green"


def _estimate_siglip_peak_memory(
    *,
    sequence_count: int,
    max_sequence_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    dtype_bytes: int,
    max_batch_size_siglip: int,
) -> dict[str, int]:
    batch_sequences = min(max(sequence_count, 1), max(max_batch_size_siglip, 1))
    hidden_state_bytes = int(batch_sequences * max_sequence_tokens * hidden_size * dtype_bytes)
    attention_score_bytes = int(
        batch_sequences * max_sequence_tokens * max_sequence_tokens * num_attention_heads * dtype_bytes
    )
    mlp_intermediate_bytes = int(batch_sequences * max_sequence_tokens * intermediate_size * dtype_bytes)
    return {
        "batch_sequences": batch_sequences,
        "max_sequence_tokens_per_batch": int(max_sequence_tokens),
        "hidden_state_peak_bytes_estimated": hidden_state_bytes,
        "attention_score_peak_bytes_estimated": attention_score_bytes,
        "mlp_intermediate_peak_bytes_estimated": mlp_intermediate_bytes,
        "peak_working_bytes_estimated": hidden_state_bytes + attention_score_bytes + mlp_intermediate_bytes,
    }


def estimate_h100_vision_encoder_bottleneck(
    *,
    preflight: dict[str, Any],
    token_reduction_ratio: float,
    max_batch_size_siglip: int,
    hidden_size: int = 1152,
    intermediate_size: int = 4304,
    num_hidden_layers: int = 27,
    num_attention_heads: int = 16,
    dtype_bytes: int = 2,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    spatial_tiles = int(preflight["tiling"]["spatial_tiles"])
    temporal_chunks = int(preflight["chunking"]["temporal_chunks"])
    tile_sequences = int(preflight["chunking"]["tile_sequences"])
    chunk_frames = int(preflight["chunking"]["chunk_frames"])
    thumbnail_frames = int(preflight["sampling"]["thumbnail_frames"])
    patches_per_frame_value = int(preflight["tokens"]["patches_per_frame_tile"])
    full_tile_sequence_tokens = chunk_frames * patches_per_frame_value
    ratio = max(float(token_reduction_ratio or 1.0), 1.0)
    actual_tile_sequence_tokens = max(1, int(math.ceil(full_tile_sequence_tokens / ratio)))
    thumbnail_sequence_tokens = patches_per_frame_value if thumbnail_frames > 0 else 0

    keep_all_sequence_tokens = tile_sequences * full_tile_sequence_tokens + thumbnail_frames * patches_per_frame_value
    keep_all_attention_pairs = (
        tile_sequences * full_tile_sequence_tokens * full_tile_sequence_tokens
        + thumbnail_frames * patches_per_frame_value * patches_per_frame_value
    )
    actual_sequence_tokens = tile_sequences * actual_tile_sequence_tokens + thumbnail_frames * patches_per_frame_value
    actual_attention_pairs = (
        tile_sequences * actual_tile_sequence_tokens * actual_tile_sequence_tokens
        + thumbnail_frames * patches_per_frame_value * patches_per_frame_value
    )
    sequence_count = tile_sequences + thumbnail_frames

    keep_all = estimate_siglip_encoder_compute_from_sums(
        sequence_count=sequence_count,
        sequence_tokens=keep_all_sequence_tokens,
        dense_attention_pairs=keep_all_attention_pairs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        dtype_bytes=dtype_bytes,
    )
    actual = estimate_siglip_encoder_compute_from_sums(
        sequence_count=sequence_count,
        sequence_tokens=actual_sequence_tokens,
        dense_attention_pairs=actual_attention_pairs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        dtype_bytes=dtype_bytes,
    )

    keep_all.update(
        {
            "tile_sequence_tokens": full_tile_sequence_tokens,
            "thumbnail_sequence_tokens": thumbnail_sequence_tokens,
            **_estimate_siglip_peak_memory(
                sequence_count=sequence_count,
                max_sequence_tokens=max(full_tile_sequence_tokens, thumbnail_sequence_tokens),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                dtype_bytes=dtype_bytes,
                max_batch_size_siglip=max_batch_size_siglip,
            ),
        }
    )
    actual.update(
        {
            "tile_sequence_tokens": actual_tile_sequence_tokens,
            "thumbnail_sequence_tokens": thumbnail_sequence_tokens,
            **_estimate_siglip_peak_memory(
                sequence_count=sequence_count,
                max_sequence_tokens=max(actual_tile_sequence_tokens, thumbnail_sequence_tokens),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                dtype_bytes=dtype_bytes,
                max_batch_size_siglip=max_batch_size_siglip,
            ),
        }
    )

    actual_visual_tokens_from_vit = int(math.ceil(actual_sequence_tokens / max(token_shuffle, 1)))
    keep_all_visual_tokens_from_vit = int(math.ceil(keep_all_sequence_tokens / max(token_shuffle, 1)))
    return {
        "metric_type": "estimated_siglip_dense_attention_and_mlp_from_patch_slots",
        "assumptions": {
            "spatial_tiles": spatial_tiles,
            "temporal_chunks": temporal_chunks,
            "tile_sequences": tile_sequences,
            "chunk_frames": chunk_frames,
            "thumbnail_frames_keep_all": thumbnail_frames,
            "token_reduction_ratio_applies_to": "tile_patch_slots_only",
            "token_shuffle": token_shuffle,
            "max_batch_size_siglip": max_batch_size_siglip,
        },
        "keep_all": keep_all,
        "actual": actual,
        "keep_all_to_actual_token_ratio": _safe_ratio(
            keep_all["sequence_tokens"], actual["sequence_tokens"]
        ),
        "keep_all_to_actual_dense_attention_pair_ratio": _safe_ratio(
            keep_all["dense_attention_pairs"], actual["dense_attention_pairs"]
        ),
        "keep_all_to_actual_total_macs_ratio": _safe_ratio(
            keep_all["total_macs_estimated"], actual["total_macs_estimated"]
        ),
        "keep_all_llm_visual_tokens_from_vit_patch_slots": keep_all_visual_tokens_from_vit,
        "actual_llm_visual_tokens_from_vit_patch_slots": actual_visual_tokens_from_vit,
    }


def estimate_llm_context_capacity(
    *,
    preflight: dict[str, Any],
    text_tokens_estimated: int,
    context_limit: int,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    tile_sequences = int(preflight["chunking"]["tile_sequences"])
    chunk_frames = int(preflight["chunking"]["chunk_frames"])
    thumbnail_frames = int(preflight["sampling"]["thumbnail_frames"])
    patches_per_frame_value = int(preflight["tokens"]["patches_per_frame_tile"])
    full_tile_sequence_tokens = chunk_frames * patches_per_frame_value
    thumbnail_sequence_tokens = thumbnail_frames * patches_per_frame_value
    available_visual_tokens = max(int(context_limit) - int(text_tokens_estimated), 0)
    max_sequence_tokens_before_shuffle = available_visual_tokens * max(token_shuffle, 1)
    max_tile_tokens_total = max_sequence_tokens_before_shuffle - thumbnail_sequence_tokens
    max_tile_sequence_tokens = (
        math.floor(max_tile_tokens_total / max(tile_sequences, 1))
        if max_tile_tokens_total > 0 and tile_sequences > 0
        else 0
    )
    min_tile_reduction_ratio = (
        full_tile_sequence_tokens / max_tile_sequence_tokens if max_tile_sequence_tokens > 0 else None
    )
    return {
        "context_limit": int(context_limit),
        "text_tokens_estimated": int(text_tokens_estimated),
        "available_visual_tokens": available_visual_tokens,
        "token_shuffle": token_shuffle,
        "max_sequence_tokens_before_shuffle": max_sequence_tokens_before_shuffle,
        "thumbnail_sequence_tokens_keep_all": thumbnail_sequence_tokens,
        "tile_sequences": tile_sequences,
        "full_tile_sequence_tokens": full_tile_sequence_tokens,
        "max_tile_sequence_tokens_for_context": max_tile_sequence_tokens,
        "min_tile_reduction_ratio_for_context": (
            round(float(min_tile_reduction_ratio), 2) if min_tile_reduction_ratio is not None else None
        ),
    }


def estimate_h100_preflight_config(
    *,
    width: int,
    height: int,
    source_frames: int | None,
    model_family: str,
    num_video_frames: int,
    num_video_frames_thumbnail: int,
    max_tiles_video: int,
    resize_shortest_edge: int | None,
    token_reduction_ratio: float,
    resize_longest_edge: int | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    h100_budget_gib: float = H100_DEFAULT_BUDGET_GIB,
    context_limit: int = NVILA_CONTEXT_LIMIT,
    text_tokens_estimated: int = 256,
    stream_chunk_frames: int | None = None,
    max_batch_size_autogaze: int | None = None,
    max_batch_size_siglip: int = 32,
    autogaze_residency_policy: str = "resident",
    autogaze_model_resident_gib: float = 0.0,
) -> dict[str, Any]:
    effective = apply_resize_to_dimensions(
        width=width,
        height=height,
        shortest_edge=resize_shortest_edge,
        longest_edge=resize_longest_edge,
        exact_width=resize_width,
        exact_height=resize_height,
    )
    preflight = estimate_nvila_preflight(
        width=int(effective["width"]),
        height=int(effective["height"]),
        source_frames=source_frames,
        num_video_frames=num_video_frames,
        num_video_frames_thumbnail=num_video_frames_thumbnail,
        max_tiles_video=max_tiles_video,
        context_limit=context_limit,
    )
    keep_all_visual_tokens = int(preflight["tokens"]["keep_all_projected_tokens"])
    ratio = max(float(token_reduction_ratio or 1.0), 1.0)
    vision_encoder = estimate_h100_vision_encoder_bottleneck(
        preflight=preflight,
        token_reduction_ratio=ratio,
        max_batch_size_siglip=max_batch_size_siglip,
        dtype_bytes=2,
    )
    context_capacity = estimate_llm_context_capacity(
        preflight=preflight,
        text_tokens_estimated=text_tokens_estimated,
        context_limit=context_limit,
    )
    actual_visual_tokens = int(vision_encoder["actual_llm_visual_tokens_from_vit_patch_slots"])
    actual_context_tokens = int(text_tokens_estimated + actual_visual_tokens)
    keep_all_context_tokens = int(text_tokens_estimated + keep_all_visual_tokens)
    context_exceeded = actual_context_tokens > context_limit

    dtype_bytes = 2
    llm_prefill = estimate_llm_prefill_compute(
        context_tokens=actual_context_tokens,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        dtype_bytes=dtype_bytes,
    )
    model_static_gib = 20.0 if model_family == MODEL_FAMILY_VIDEO_BASELINE else 22.0
    autogaze_resident_gib = (
        max(float(autogaze_model_resident_gib or 0.0), 0.0)
        if model_family == MODEL_FAMILY_HD_AUTOGAZE and autogaze_residency_policy == "resident"
        else 0.0
    )
    keep_all_prefill = estimate_llm_prefill_compute(
        context_tokens=keep_all_context_tokens,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        dtype_bytes=dtype_bytes,
    )
    vision_encoder_peak_bytes = int(vision_encoder["actual"]["peak_working_bytes_estimated"])
    autogaze_full_video_tensor_bytes = int(preflight["memory"]["autogaze_tile_tensor_bytes"])
    autogaze_tensor_residency_bytes = autogaze_full_video_tensor_bytes
    autogaze_forward_batch_tensor_bytes = autogaze_full_video_tensor_bytes
    autogaze_working_mode = "full_video_tensor"
    if stream_chunk_frames is not None and int(stream_chunk_frames) > 0:
        stream_plan = estimate_stream_profile_plan(
            width=int(effective["width"]),
            height=int(effective["height"]),
            source_frames=source_frames,
            num_video_frames=num_video_frames,
            num_video_frames_thumbnail=num_video_frames_thumbnail,
            max_tiles_video=max_tiles_video,
            chunk_frames=int(stream_chunk_frames),
            max_batch_size_autogaze=max_batch_size_autogaze,
        )
        autogaze_tensor_residency_bytes = int(
            stream_plan["memory"]["streaming_autogaze_tile_tensor_bytes_full_chunk"]
        )
        autogaze_forward_batch_tensor_bytes = int(
            stream_plan["memory"]["streaming_autogaze_tile_tensor_bytes_per_batch"]
        )
        autogaze_working_mode = "stream_chunk"
    autogaze_working_bytes = (
        int(autogaze_tensor_residency_bytes)
        if model_family == MODEL_FAMILY_HD_AUTOGAZE
        else 0
    )
    working_bytes = (
        int(llm_prefill["kv_cache_bytes_after_prefill_estimated"])
        + int(llm_prefill["attention_score_bytes_estimated"])
        + int(vision_encoder_peak_bytes)
        + int(autogaze_working_bytes)
    )
    estimated_vram_gib = model_static_gib + autogaze_resident_gib + bytes_to_gib(working_bytes) * 1.15
    band = h100_risk_band(
        estimated_vram_gib,
        context_exceeded=context_exceeded,
        h100_budget_gib=h100_budget_gib,
    )
    mllm_prefill_memory_bytes = int(llm_prefill["kv_cache_bytes_after_prefill_estimated"]) + int(
        llm_prefill["attention_score_bytes_estimated"]
    )
    stage_memory_gib = {
        "autogaze": bytes_to_gib(autogaze_working_bytes),
        "vision_encoder": bytes_to_gib(vision_encoder_peak_bytes),
        "mllm_prefill": bytes_to_gib(mllm_prefill_memory_bytes),
    }
    stage_compute_macs = {
        "vision_encoder": int(vision_encoder["actual"]["total_macs_estimated"]),
        "mllm_prefill": int(llm_prefill["total_macs_estimated"]),
    }
    recommended_role = (
        "paper_baseline_reproduction_config"
        if model_family == MODEL_FAMILY_VIDEO_BASELINE
        else "hd_autogaze_scaling_config"
    )
    return {
        "model_family": model_family,
        "recommended_role": recommended_role,
        "config": {
            "num_video_frames": num_video_frames,
            "num_video_frames_thumbnail": num_video_frames_thumbnail,
            "max_tiles_video": max_tiles_video,
            "resize_shortest_edge": resize_shortest_edge,
            "resize_longest_edge": resize_longest_edge,
            "resize_width": resize_width,
            "resize_height": resize_height,
            "token_reduction_ratio": ratio,
            "stream_chunk_frames": stream_chunk_frames,
            "max_batch_size_autogaze": max_batch_size_autogaze,
            "max_batch_size_siglip": max_batch_size_siglip,
            "autogaze_residency_policy": autogaze_residency_policy,
            "autogaze_model_resident_gib": autogaze_model_resident_gib,
        },
        "source_video": {
            "width": width,
            "height": height,
            "source_frames": source_frames,
        },
        "effective_video": {
            "width": int(effective["width"]),
            "height": int(effective["height"]),
            "resize_mode": effective["mode"],
        },
        "tokens": {
            "text_tokens_estimated": text_tokens_estimated,
            "keep_all_llm_visual_tokens_estimated": keep_all_visual_tokens,
            "actual_llm_visual_tokens_estimated": actual_visual_tokens,
            "keep_all_context_tokens_estimated": keep_all_context_tokens,
            "actual_context_tokens_estimated": actual_context_tokens,
            "context_limit": context_limit,
            "context_exceeded": context_exceeded,
            "context_fits": not context_exceeded,
            "context_margin_tokens": int(context_limit - actual_context_tokens),
            "context_utilization_percent": round((actual_context_tokens / max(context_limit, 1)) * 100.0, 2),
            "context_capacity": context_capacity,
        },
        "memory": {
            "model_static_gib_assumed": model_static_gib,
            "working_set_bytes_estimated": working_bytes,
            "working_set_gib_estimated": bytes_to_gib(working_bytes),
            "estimated_vram_gib": estimated_vram_gib,
            "h100_budget_gib": h100_budget_gib,
            "autogaze_residency_policy": autogaze_residency_policy,
            "autogaze_model_resident_gib_assumed": autogaze_resident_gib,
            "llm_kv_cache_bytes_estimated": llm_prefill["kv_cache_bytes_after_prefill_estimated"],
            "llm_attention_score_bytes_estimated": llm_prefill["attention_score_bytes_estimated"],
            "siglip_peak_working_bytes_estimated": vision_encoder_peak_bytes,
            "siglip_hidden_bytes_estimated": vision_encoder["actual"]["hidden_state_peak_bytes_estimated"],
            "siglip_attention_score_bytes_capped_estimated": vision_encoder["actual"][
                "attention_score_peak_bytes_estimated"
            ],
            "siglip_attention_score_peak_bytes_estimated": vision_encoder["actual"][
                "attention_score_peak_bytes_estimated"
            ],
            "siglip_mlp_intermediate_peak_bytes_estimated": vision_encoder["actual"][
                "mlp_intermediate_peak_bytes_estimated"
            ],
            "autogaze_working_bytes_estimated": autogaze_working_bytes,
            "autogaze_working_mode": autogaze_working_mode,
            "autogaze_tile_tensor_full_video_bytes_estimated": autogaze_full_video_tensor_bytes,
            "autogaze_tensor_residency_bytes_estimated": autogaze_tensor_residency_bytes,
            "autogaze_forward_batch_tensor_bytes_estimated": autogaze_forward_batch_tensor_bytes,
            "note": (
                "This is a conservative planning estimate for H100 scheduling. "
                "It combines assumed 8B fp16 model residency, estimated LLM prefill KV/attention, "
                "SigLIP activation pressure, and AutoGaze tensor pressure when applicable."
            ),
        },
        "risk": {
            "band": band,
            "green_lt_gib": H100_GREEN_GIB,
            "yellow_until_gib": h100_budget_gib,
            "red_gte_gib": h100_budget_gib,
            "context_red": context_exceeded,
        },
        "vision_encoder": vision_encoder,
        "mllm": {
            "metric_type": "estimated_qwen2_prefill_attention_kv_and_mlp",
            "actual": llm_prefill,
            "keep_all_estimated": keep_all_prefill,
            "prefill_context_reduction_ratio": _safe_ratio(keep_all_context_tokens, actual_context_tokens),
            "kv_cache_reduction_ratio": _safe_ratio(
                keep_all_prefill["kv_cache_bytes_after_prefill_estimated"],
                llm_prefill["kv_cache_bytes_after_prefill_estimated"],
            ),
            "attention_score_reduction_ratio": _safe_ratio(
                keep_all_prefill["attention_score_bytes_estimated"],
                llm_prefill["attention_score_bytes_estimated"],
            ),
            "total_macs_reduction_ratio": _safe_ratio(
                keep_all_prefill["total_macs_estimated"],
                llm_prefill["total_macs_estimated"],
            ),
        },
        "bottlenecks": {
            "stage_memory_gib_estimated": stage_memory_gib,
            "stage_compute_macs_estimated": stage_compute_macs,
            "dominant_memory_stage_estimated": max(stage_memory_gib, key=stage_memory_gib.get),
            "dominant_compute_stage_estimated": (
                "vision_encoder"
                if stage_compute_macs["vision_encoder"] >= stage_compute_macs["mllm_prefill"]
                else "mllm_prefill"
            ),
            "note": (
                "AutoGaze can be streamed before the LLM, so the key OOM risks are the actual SigLIP "
                "sequence batch and the collected MLLM prefill context. SigLIP estimates assume dense "
                "attention score materialization; memory-efficient kernels can lower this peak."
            ),
        },
        "nvila_preflight": preflight,
    }


def _resolution_label(video: dict[str, Any]) -> str:
    return f"{int(video['width'])}x{int(video['height'])}"


def build_h100_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    bottlenecks = row.get("bottlenecks", {})
    stage_memory = bottlenecks.get("stage_memory_gib_estimated", {})
    return {
        "risk_band": row["risk"]["band"],
        "source_resolution": _resolution_label(row["source_video"]),
        "effective_resolution": _resolution_label(row["effective_video"]),
        "resize_mode": row["effective_video"].get("resize_mode"),
        "frames": row["config"]["num_video_frames"],
        "thumbnail_frames": row["config"]["num_video_frames_thumbnail"],
        "max_tiles_video": row["config"]["max_tiles_video"],
        "spatial_tiles": row["nvila_preflight"]["tiling"]["spatial_tiles"],
        "token_reduction_ratio": row["config"]["token_reduction_ratio"],
        "llm_keep_all_visual_tokens": row["tokens"]["keep_all_llm_visual_tokens_estimated"],
        "llm_actual_visual_tokens": row["tokens"]["actual_llm_visual_tokens_estimated"],
        "llm_actual_context_tokens": row["tokens"]["actual_context_tokens_estimated"],
        "context_limit": row["tokens"]["context_limit"],
        "llm_context_fits": row["tokens"].get("context_fits"),
        "llm_context_margin_tokens": row["tokens"].get("context_margin_tokens"),
        "llm_context_utilization_percent": row["tokens"].get("context_utilization_percent"),
        "min_tile_reduction_ratio_for_context": row["tokens"].get("context_capacity", {}).get(
            "min_tile_reduction_ratio_for_context"
        ),
        "max_tile_sequence_tokens_for_context": row["tokens"].get("context_capacity", {}).get(
            "max_tile_sequence_tokens_for_context"
        ),
        "estimated_vram_gib": round(float(row["memory"]["estimated_vram_gib"]), 2),
        "autogaze_residency_policy": row["memory"].get("autogaze_residency_policy"),
        "autogaze_model_resident_gib": row["memory"].get("autogaze_model_resident_gib_assumed"),
        "autogaze_memory_gib": round(float(stage_memory.get("autogaze", 0.0)), 2),
        "vision_encoder_memory_gib": round(float(stage_memory.get("vision_encoder", 0.0)), 2),
        "mllm_prefill_memory_gib": round(float(stage_memory.get("mllm_prefill", 0.0)), 2),
        "dominant_memory_stage": bottlenecks.get("dominant_memory_stage_estimated"),
        "dominant_compute_stage": bottlenecks.get("dominant_compute_stage_estimated"),
        "vit_tile_sequence_tokens": row.get("vision_encoder", {}).get("actual", {}).get("tile_sequence_tokens"),
        "vit_max_sequence_tokens_per_batch": row.get("vision_encoder", {}).get("actual", {}).get(
            "max_sequence_tokens_per_batch"
        ),
        "stream_chunk_frames": row["config"].get("stream_chunk_frames"),
        "max_batch_size_autogaze": row["config"].get("max_batch_size_autogaze"),
        "max_batch_size_siglip": row["config"].get("max_batch_size_siglip"),
    }


def build_h100_decision_summary(
    *,
    requested_rows: list[dict[str, Any]],
    sweep: dict[str, Any],
    limit: int = 20,
) -> dict[str, Any]:
    return {
        "how_to_read": (
            "Check effective_resolution, frames, spatial_tiles, llm_actual_visual_tokens, "
            "llm_actual_context_tokens, estimated_vram_gib, and dominant_memory_stage first. "
            "Full detailed rows are stored in the JSON file."
        ),
        "risk_band_counts": sweep.get("risk_band_counts", {}),
        "requested_config_table": [build_h100_decision_row(row) for row in requested_rows],
        "sweep_decision_table": [
            build_h100_decision_row(row)
            for row in sweep.get("recommended_configs", [])[:limit]
        ],
    }


def estimate_h100_preflight_sweep(
    *,
    width: int,
    height: int,
    source_frames: int | None,
    model_family: str,
    token_reduction_ratios: list[float] | tuple[float, ...],
    h100_budget_gib: float = H100_DEFAULT_BUDGET_GIB,
    stream_chunk_frames: int | None = None,
    max_batch_size_autogaze: int | None = None,
    max_batch_size_siglip: int = 32,
    autogaze_residency_policy: str = "resident",
    autogaze_model_resident_gib: float = 0.0,
) -> dict[str, Any]:
    rows = []
    for num_frames in H100_SWEEP_FRAMES:
        for thumbnail_frames in H100_SWEEP_THUMBNAIL_FRAMES:
            if thumbnail_frames > num_frames:
                continue
            for max_tiles in H100_SWEEP_MAX_TILES:
                for resize_shortest_edge in H100_SWEEP_RESIZE_SHORTEST_EDGE:
                    for reduction_ratio in token_reduction_ratios:
                        rows.append(
                            estimate_h100_preflight_config(
                                width=width,
                                height=height,
                                source_frames=source_frames,
                                model_family=model_family,
                                num_video_frames=num_frames,
                                num_video_frames_thumbnail=thumbnail_frames,
                                max_tiles_video=max_tiles,
                                resize_shortest_edge=resize_shortest_edge,
                                resize_longest_edge=None,
                                token_reduction_ratio=reduction_ratio,
                                h100_budget_gib=h100_budget_gib,
                                stream_chunk_frames=stream_chunk_frames,
                                max_batch_size_autogaze=max_batch_size_autogaze,
                                max_batch_size_siglip=max_batch_size_siglip,
                                autogaze_residency_policy=autogaze_residency_policy,
                                autogaze_model_resident_gib=autogaze_model_resident_gib,
                            )
                        )
    by_band = Counter(row["risk"]["band"] for row in rows)
    safe_rows = [row for row in rows if row["risk"]["band"] in {"green", "yellow"}]
    safe_rows.sort(
        key=lambda row: (
            row["risk"]["band"] != "green",
            -int(row["config"]["num_video_frames"]),
            -int(row["config"]["max_tiles_video"]),
            float(row["memory"]["estimated_vram_gib"]),
        )
    )
    return {
        "model_family": model_family,
        "h100_budget_gib": h100_budget_gib,
        "rows": rows,
        "risk_band_counts": dict(by_band),
        "recommended_configs": safe_rows[:20],
        "decision_table": [build_h100_decision_row(row) for row in safe_rows[:20]],
        "recommendation_note": (
            "Paper baseline recommendations prioritize NVILA-8B-Video 256f/448-style configs; "
            "HD recommendations prioritize the widest green/yellow AutoGaze config before full CUDA runs."
        ),
    }


def processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if effective_model_family(args) == MODEL_FAMILY_VIDEO_BASELINE:
        kwargs = {
            "num_video_frames": args.num_video_frames,
            "trust_remote_code": True,
        }
        thumbnail_frames = int(getattr(args, "num_video_frames_thumbnail", 0) or 0)
        if thumbnail_frames > 0:
            kwargs["num_video_frames_thumbnail"] = thumbnail_frames
        return kwargs

    if args.gazing_mode == "keep-all":
        gazing_ratio_tile: float | list[float] = 1
        task_loss_requirement_tile = None
    else:
        gazing_ratio_tile = effective_gazing_ratio_tile(args)
        task_loss_requirement_tile = args.task_loss_requirement_tile

    kwargs = {
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
        "autogaze_model_id": effective_token_selector_path(args) or args.autogaze_model,
        "gazing_ratio_tile": gazing_ratio_tile,
        "gazing_ratio_thumbnail": 1,
        "task_loss_requirement_tile": task_loss_requirement_tile,
        "task_loss_requirement_thumbnail": None,
        "max_batch_size_autogaze": args.max_batch_size_autogaze,
        "trust_remote_code": True,
    }
    target_scales = effective_autogaze_target_scales(args)
    if target_scales is not None:
        kwargs["target_scales"] = target_scales
    target_patch_size = effective_autogaze_target_patch_size(args)
    if target_patch_size is not None:
        kwargs["target_patch_size"] = int(target_patch_size)
    return kwargs


def model_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
    }
    torch_dtype = requested_torch_dtype(args)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if effective_model_family(args) == MODEL_FAMILY_HD_AUTOGAZE:
        kwargs["max_batch_size_siglip"] = args.max_batch_size_siglip
    return kwargs


def requested_torch_dtype(args: argparse.Namespace) -> torch.dtype | None:
    value = getattr(args, "dtype", None)
    if value in (None, "auto"):
        return None
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def apply_processor_autogaze_generate_only(processor: Any, *, enabled: bool) -> bool:
    setattr(processor, "autogaze_generate_only", bool(enabled))
    if not enabled:
        return False
    autogaze_model = getattr(processor, "_autogaze_model", None)
    if autogaze_model is None:
        return False
    if getattr(autogaze_model, "_nvila_generate_only_patch_applied", False):
        return True

    original_forward = autogaze_model.forward

    def forward_with_generate_only(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("generate_only", True)
        return original_forward(*args, **kwargs)

    autogaze_model.forward = forward_with_generate_only
    autogaze_model._nvila_generate_only_patch_applied = True
    autogaze_model._nvila_original_forward_before_generate_only_patch = original_forward
    return True


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs(args))
    apply_processor_autogaze_generate_only(
        processor,
        enabled=bool(getattr(args, "autogaze_generate_only", False)),
    )
    model = AutoModel.from_pretrained(args.model_path, **model_load_kwargs(args))
    model.eval()
    return model, processor


def read_video_metadata(video: str) -> dict[str, int | float | str | None]:
    container = av.open(video)
    try:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        fps = float(stream.average_rate) if stream.average_rate else None
        pts_per_frame = stream_pts_per_frame(average_rate=stream.average_rate, time_base=stream.time_base)
        return {
            "width": int(stream.width),
            "height": int(stream.height),
            "frames": int(stream.frames) if stream.frames else None,
            "fps": fps,
            "duration_seconds": duration,
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "start_time": int(stream.start_time) if stream.start_time is not None else None,
            "pts_per_frame": float(pts_per_frame) if pts_per_frame is not None else None,
        }
    finally:
        container.close()


def read_video_keyframe_indices(video: str) -> tuple[list[int], dict[str, Any]]:
    container = av.open(video)
    try:
        stream = container.streams.video[0]
        pts_per_frame = stream_pts_per_frame(average_rate=stream.average_rate, time_base=stream.time_base)
        if pts_per_frame is None:
            raise ValueError("Seek decode requires video average_rate and time_base metadata.")
        start_time = int(stream.start_time) if stream.start_time is not None else 0
        keyframes: list[int] = []
        packets_scanned = 0
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            packets_scanned += 1
            if packet.is_keyframe:
                keyframes.append(pts_to_frame_index(packet.pts, pts_per_frame=pts_per_frame, start_time=start_time))
        return keyframes, {
            "packets_scanned": packets_scanned,
            "keyframes": len(keyframes),
            "pts_per_frame": float(pts_per_frame),
            "start_time": start_time,
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


def _gazing_padding_payload(payload: dict[str, Any], key: str) -> Any:
    gazing_info = payload.get("gazing_info")
    if isinstance(gazing_info, dict) and key in gazing_info:
        return gazing_info[key]
    return None


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


def _config_int(config: Any, key: str, default: int = 0) -> int:
    value = getattr(config, key, default)
    if value is None:
        return default
    return int(value)


def _model_dtype_bytes(model: Any, default: int = 2) -> int:
    try:
        return int(next(model.parameters()).element_size())
    except Exception:
        pass

    dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    dtype_name = str(dtype).lower()
    if "float32" in dtype_name or "fp32" in dtype_name:
        return 4
    if "float64" in dtype_name or "fp64" in dtype_name:
        return 8
    if "int8" in dtype_name or "float8" in dtype_name:
        return 1
    return default


def _visual_frame_count(tensor: torch.Tensor) -> int:
    if tensor.ndim >= 5:
        return int(tensor.shape[0]) * int(tensor.shape[1])
    if tensor.ndim >= 1:
        return int(tensor.shape[0])
    return 0


def _sequence_lengths_from_padding_masks(value: Any) -> list[int]:
    lengths: list[int] = []
    for tensor in _iter_tensors(value):
        if tensor.numel() == 0:
            continue
        if tensor.ndim == 1:
            lengths.append(int(tensor.shape[0]))
            continue
        rows = tensor.reshape(-1, int(tensor.shape[-1]))
        lengths.extend([int(rows.shape[-1])] * int(rows.shape[0]))
    return lengths


def _keep_all_siglip_sequence_lengths(payload: dict[str, Any], patches_per_frame_value: int) -> list[int]:
    lengths: list[int] = []
    for tensor in _tensor_sequence(payload.get("pixel_values_videos_tiles")):
        if tensor.ndim >= 5:
            lengths.extend([int(tensor.shape[1]) * patches_per_frame_value] * int(tensor.shape[0]))
    for tensor in _tensor_sequence(payload.get("pixel_values_videos_thumbnails")):
        if tensor.ndim >= 5:
            lengths.extend([int(tensor.shape[1]) * patches_per_frame_value] * int(tensor.shape[0]))
        elif tensor.ndim >= 1:
            lengths.extend([patches_per_frame_value] * int(tensor.shape[0]))
    return lengths


def _actual_siglip_sequence_lengths(payload: dict[str, Any], patches_per_frame_value: int) -> list[int]:
    lengths = _sequence_lengths_from_padding_masks(_gazing_padding_payload(payload, "if_padded_gazing_tiles"))
    lengths.extend(_sequence_lengths_from_padding_masks(_gazing_padding_payload(payload, "if_padded_gazing_thumbnails")))
    if lengths:
        return lengths
    return _keep_all_siglip_sequence_lengths(payload, patches_per_frame_value)


def estimate_siglip_encoder_compute(
    *,
    sequence_lengths: list[int],
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    dtype_bytes: int,
) -> dict[str, int]:
    """Estimate dense SigLIP transformer MACs and major activation sizes."""

    sequence_count = len(sequence_lengths)
    sequence_tokens = int(sum(sequence_lengths))
    dense_attention_pairs = int(sum(length * length for length in sequence_lengths))
    return estimate_siglip_encoder_compute_from_sums(
        sequence_count=sequence_count,
        sequence_tokens=sequence_tokens,
        dense_attention_pairs=dense_attention_pairs,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        dtype_bytes=dtype_bytes,
    )


def estimate_siglip_encoder_compute_from_sums(
    *,
    sequence_count: int,
    sequence_tokens: int,
    dense_attention_pairs: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    dtype_bytes: int,
) -> dict[str, int]:
    attention_projection_macs = int(num_hidden_layers * 4 * sequence_tokens * hidden_size * hidden_size)
    attention_quadratic_macs = int(num_hidden_layers * 2 * dense_attention_pairs * hidden_size)
    mlp_macs = int(num_hidden_layers * 2 * sequence_tokens * hidden_size * intermediate_size)
    total_macs = attention_projection_macs + attention_quadratic_macs + mlp_macs

    return {
        "sequence_count": sequence_count,
        "sequence_tokens": sequence_tokens,
        "dense_attention_pairs": dense_attention_pairs,
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_hidden_layers": int(num_hidden_layers),
        "num_attention_heads": int(num_attention_heads),
        "dtype_bytes": int(dtype_bytes),
        "attention_projection_macs_estimated": attention_projection_macs,
        "attention_quadratic_macs_estimated": attention_quadratic_macs,
        "mlp_macs_estimated": mlp_macs,
        "total_macs_estimated": total_macs,
        "hidden_state_bytes_estimated": int(sequence_tokens * hidden_size * dtype_bytes),
        "attention_score_bytes_estimated": int(dense_attention_pairs * num_attention_heads * dtype_bytes),
        "mlp_intermediate_bytes_estimated": int(sequence_tokens * intermediate_size * dtype_bytes),
    }


def estimate_llm_prefill_compute(
    *,
    context_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    dtype_bytes: int,
) -> dict[str, int]:
    head_dim = hidden_size // max(num_attention_heads, 1)
    kv_hidden_size = num_key_value_heads * head_dim
    causal_attention_pairs = context_tokens * (context_tokens + 1) // 2
    attention_projection_macs = int(
        num_hidden_layers
        * context_tokens
        * (
            hidden_size * hidden_size
            + 2 * hidden_size * kv_hidden_size
            + hidden_size * hidden_size
        )
    )
    attention_quadratic_macs = int(num_hidden_layers * 2 * causal_attention_pairs * hidden_size)
    mlp_macs = int(num_hidden_layers * 3 * context_tokens * hidden_size * intermediate_size)
    total_macs = attention_projection_macs + attention_quadratic_macs + mlp_macs
    kv_cache_bytes = int(num_hidden_layers * 2 * context_tokens * num_key_value_heads * head_dim * dtype_bytes)

    return {
        "context_tokens": int(context_tokens),
        "causal_attention_pairs": int(causal_attention_pairs),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_hidden_layers": int(num_hidden_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "dtype_bytes": int(dtype_bytes),
        "attention_projection_macs_estimated": attention_projection_macs,
        "attention_quadratic_macs_estimated": attention_quadratic_macs,
        "mlp_macs_estimated": mlp_macs,
        "total_macs_estimated": total_macs,
        "kv_cache_bytes_after_prefill_estimated": kv_cache_bytes,
        "attention_score_bytes_estimated": int(causal_attention_pairs * num_attention_heads * dtype_bytes),
    }


def build_autogaze_effect_metrics(
    payload: dict[str, Any],
    *,
    model: Any,
    token_metrics: dict[str, Any],
    input_token_count: int,
    dtype_bytes: int | None = None,
    patches_per_frame_value: int | None = None,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    dtype_bytes = int(dtype_bytes or _model_dtype_bytes(model))
    patches_value = int(patches_per_frame_value or model_patches_per_frame(model))

    vision_config = getattr(getattr(model, "vision_tower", None), "config", None)
    hidden_size = _config_int(vision_config, "hidden_size", 1152)
    intermediate_size = _config_int(vision_config, "intermediate_size", 4304)
    vision_layers = _config_int(vision_config, "num_hidden_layers", 27)
    vision_heads = _config_int(vision_config, "num_attention_heads", 16)

    keep_all_sequence_lengths = _keep_all_siglip_sequence_lengths(payload, patches_value)
    actual_sequence_lengths = _actual_siglip_sequence_lengths(payload, patches_value)
    keep_all_siglip = estimate_siglip_encoder_compute(
        sequence_lengths=keep_all_sequence_lengths,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=vision_layers,
        num_attention_heads=vision_heads,
        dtype_bytes=dtype_bytes,
    )
    actual_siglip = estimate_siglip_encoder_compute(
        sequence_lengths=actual_sequence_lengths,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=vision_layers,
        num_attention_heads=vision_heads,
        dtype_bytes=dtype_bytes,
    )

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    text_hidden_size = _config_int(text_config, "hidden_size", 4096)
    text_intermediate_size = _config_int(text_config, "intermediate_size", 11008)
    text_layers = _config_int(text_config, "num_hidden_layers", 32)
    text_heads = _config_int(text_config, "num_attention_heads", 32)
    text_kv_heads = _config_int(text_config, "num_key_value_heads", text_heads)

    actual_visual_tokens = token_metrics.get("llm_actual_visual_tokens")
    keep_all_visual_tokens = int(token_metrics.get("llm_keep_all_visual_tokens_estimated") or 0)
    text_tokens = None
    keep_all_context = None
    if actual_visual_tokens is not None:
        actual_visual_tokens = int(actual_visual_tokens)
        text_tokens = max(int(input_token_count) - actual_visual_tokens, 0)
        keep_all_context = text_tokens + keep_all_visual_tokens
    actual_prefill = estimate_llm_prefill_compute(
        context_tokens=int(input_token_count),
        hidden_size=text_hidden_size,
        intermediate_size=text_intermediate_size,
        num_hidden_layers=text_layers,
        num_attention_heads=text_heads,
        num_key_value_heads=text_kv_heads,
        dtype_bytes=dtype_bytes,
    )
    keep_all_prefill = (
        estimate_llm_prefill_compute(
            context_tokens=int(keep_all_context),
            hidden_size=text_hidden_size,
            intermediate_size=text_intermediate_size,
            num_hidden_layers=text_layers,
            num_attention_heads=text_heads,
            num_key_value_heads=text_kv_heads,
            dtype_bytes=dtype_bytes,
        )
        if keep_all_context is not None
        else None
    )

    return {
        "metric_type": "estimated_macs_and_bytes_from_token_counts",
        "note": (
            "MAC/byte fields are analytical estimates from model config and token counts. "
            "Use stage timings and CUDA peak memory fields for measured runtime values."
        ),
        "siglip_encoder": {
            "token_shuffle": int(token_shuffle),
            "keep_all": keep_all_siglip,
            "actual": actual_siglip,
            "keep_all_to_actual_token_ratio": _safe_ratio(
                keep_all_siglip["sequence_tokens"], actual_siglip["sequence_tokens"]
            ),
            "keep_all_to_actual_dense_attention_pair_ratio": _safe_ratio(
                keep_all_siglip["dense_attention_pairs"], actual_siglip["dense_attention_pairs"]
            ),
            "keep_all_to_actual_attention_macs_ratio": _safe_ratio(
                keep_all_siglip["attention_quadratic_macs_estimated"],
                actual_siglip["attention_quadratic_macs_estimated"],
            ),
            "keep_all_to_actual_mlp_macs_ratio": _safe_ratio(
                keep_all_siglip["mlp_macs_estimated"], actual_siglip["mlp_macs_estimated"]
            ),
            "keep_all_to_actual_total_macs_ratio": _safe_ratio(
                keep_all_siglip["total_macs_estimated"], actual_siglip["total_macs_estimated"]
            ),
        },
        "mllm": {
            "prefill_explanation": (
                "Prefill is the first LLM forward over the entire prompt plus visual token sequence. "
                "It builds the KV cache used by subsequent token-by-token decode."
            ),
            "actual_prefill_context_tokens": int(input_token_count),
            "actual_visual_tokens": actual_visual_tokens,
            "text_tokens_estimated": text_tokens,
            "keep_all_visual_tokens_estimated": keep_all_visual_tokens,
            "keep_all_prefill_context_tokens_estimated": keep_all_context,
            "prefill_context_reduction_ratio": _safe_ratio(keep_all_context, int(input_token_count)),
            "actual": actual_prefill,
            "keep_all_estimated": keep_all_prefill,
            "actual_kv_cache_bytes_after_prefill_estimated": actual_prefill[
                "kv_cache_bytes_after_prefill_estimated"
            ],
            "keep_all_kv_cache_bytes_after_prefill_estimated": (
                keep_all_prefill["kv_cache_bytes_after_prefill_estimated"] if keep_all_prefill else None
            ),
            "kv_cache_reduction_ratio": _safe_ratio(
                keep_all_prefill["kv_cache_bytes_after_prefill_estimated"] if keep_all_prefill else None,
                actual_prefill["kv_cache_bytes_after_prefill_estimated"],
            ),
            "prefill_attention_pair_reduction_ratio": _safe_ratio(
                keep_all_prefill["causal_attention_pairs"] if keep_all_prefill else None,
                actual_prefill["causal_attention_pairs"],
            ),
            "prefill_total_macs_reduction_ratio": _safe_ratio(
                keep_all_prefill["total_macs_estimated"] if keep_all_prefill else None,
                actual_prefill["total_macs_estimated"],
            ),
        },
    }


def compute_visual_token_metrics(
    payload: dict[str, Any],
    *,
    video_token_id: int | None,
    patches_per_frame_value: int = patches_per_frame(),
    patches_per_frame_by_scale: dict[str, int] | None = None,
    token_shuffle: int = NVILA_TOKEN_SHUFFLE,
) -> dict[str, Any]:
    tile_tensors = _tensor_sequence(payload.get("pixel_values_videos_tiles"))
    thumbnail_tensors = _tensor_sequence(payload.get("pixel_values_videos_thumbnails"))
    patch_breakdown = patches_per_frame_by_scale or {"multiscale_total": patches_per_frame_value}

    tile_frame_instances = sum(_visual_frame_count(tensor) for tensor in tile_tensors)
    raw_tile_patches = tile_frame_instances * patches_per_frame_value
    raw_thumbnail_patches = sum(_visual_frame_count(tensor) * patches_per_frame_value for tensor in thumbnail_tensors)
    raw_encoder_patches = raw_tile_patches + raw_thumbnail_patches

    tile_selected, tile_padded_count, tile_slots = _count_padding_masks(
        _gazing_padding_payload(payload, "if_padded_gazing_tiles")
    )
    thumbnail_selected, thumbnail_padded_count, thumbnail_slots = _count_padding_masks(
        _gazing_padding_payload(payload, "if_padded_gazing_thumbnails")
    )
    selected, padded_count, total_slots = _count_padding_masks(_gazing_padding_payloads(payload))
    selected_tile_patches = tile_selected if tile_slots else raw_tile_patches
    selected_thumbnail_patches = thumbnail_selected if thumbnail_slots else raw_thumbnail_patches
    selected_encoder_patches = (
        selected_tile_patches + selected_thumbnail_patches
        if tile_slots or thumbnail_slots
        else selected if total_slots else raw_encoder_patches
    )

    num_spatial_tiles = _int_list(payload.get("num_spatial_tiles_each_video"))
    keep_all_tile_tokens = 0
    video_sampled_frames = 0
    tile_sequences = 0
    temporal_chunks_per_video: list[int] = []
    for index, tensor in enumerate(tile_tensors):
        if tensor.ndim < 2:
            continue
        spatial_tiles = num_spatial_tiles[index] if index < len(num_spatial_tiles) else 1
        spatial_tiles = max(spatial_tiles, 1)
        video_tile_sequences = int(tensor.shape[0])
        frames_per_sequence = int(tensor.shape[1])
        temporal_chunks = math.ceil(video_tile_sequences / spatial_tiles)
        total_frames = temporal_chunks * frames_per_sequence
        tile_sequences += video_tile_sequences
        video_sampled_frames += total_frames
        temporal_chunks_per_video.append(temporal_chunks)
        keep_all_tile_tokens += total_frames * math.ceil(spatial_tiles * patches_per_frame_value / token_shuffle)

    keep_all_thumbnail_tokens = 0
    thumbnail_sampled_frames = 0
    thumbnail_token_per_frame = math.ceil(patches_per_frame_value / token_shuffle)
    for tensor in thumbnail_tensors:
        if tensor.ndim >= 5:
            frame_count = int(tensor.shape[0]) * int(tensor.shape[1])
        elif tensor.ndim >= 1:
            frame_count = int(tensor.shape[0])
        else:
            frame_count = 0
        thumbnail_sampled_frames += frame_count
        keep_all_thumbnail_tokens += frame_count * thumbnail_token_per_frame

    llm_actual_visual_tokens = None
    input_ids = payload.get("input_ids")
    if isinstance(input_ids, torch.Tensor) and video_token_id is not None:
        llm_actual_visual_tokens = int((input_ids == video_token_id).sum().item())

    keep_all_projected_tokens = keep_all_tile_tokens + keep_all_thumbnail_tokens
    return {
        "video_sampled_frames": video_sampled_frames,
        "thumbnail_sampled_frames": thumbnail_sampled_frames,
        "tile_sequences": tile_sequences,
        "spatial_tiles_per_video": num_spatial_tiles,
        "temporal_chunks_per_video": temporal_chunks_per_video,
        "patches_per_frame": patches_per_frame_value,
        "encoder_patches_per_frame_multiscale": patches_per_frame_value,
        "encoder_patches_per_frame_by_scale": patch_breakdown,
        "token_shuffle": token_shuffle,
        "encoder_raw_tile_patch_tokens": raw_tile_patches,
        "encoder_autogaze_selected_tile_patch_tokens": selected_tile_patches,
        "autogaze_input_tile_frame_instances": tile_frame_instances,
        "autogaze_input_patch_tokens": raw_tile_patches,
        "autogaze_selected_patch_tokens": selected_tile_patches,
        "autogaze_removed_patch_tokens": raw_tile_patches - selected_tile_patches,
        "autogaze_patch_reduction_ratio": _safe_ratio(raw_tile_patches, selected_tile_patches),
        "encoder_autogaze_padded_tile_patch_tokens": tile_padded_count if tile_slots else 0,
        "encoder_autogaze_total_tile_gaze_slots": tile_slots,
        "encoder_tile_token_reduction_ratio": _safe_ratio(raw_tile_patches, selected_tile_patches),
        "encoder_raw_thumbnail_patch_tokens": raw_thumbnail_patches,
        "encoder_autogaze_selected_thumbnail_patch_tokens": selected_thumbnail_patches,
        "encoder_autogaze_padded_thumbnail_patch_tokens": thumbnail_padded_count if thumbnail_slots else 0,
        "encoder_autogaze_total_thumbnail_gaze_slots": thumbnail_slots,
        "encoder_thumbnail_token_reduction_ratio": _safe_ratio(raw_thumbnail_patches, selected_thumbnail_patches),
        "encoder_raw_patch_tokens": raw_encoder_patches,
        "encoder_raw_total_patch_tokens": raw_encoder_patches,
        "encoder_autogaze_selected_patch_tokens": selected_encoder_patches,
        "encoder_autogaze_selected_total_patch_tokens": selected_encoder_patches,
        "encoder_autogaze_padded_patch_tokens": padded_count if total_slots else 0,
        "encoder_autogaze_total_gaze_slots": total_slots,
        "encoder_token_reduction_ratio": _safe_ratio(raw_encoder_patches, selected_encoder_patches),
        "llm_keep_all_tile_visual_tokens_estimated": keep_all_tile_tokens,
        "llm_keep_all_thumbnail_visual_tokens_estimated": keep_all_thumbnail_tokens,
        "llm_keep_all_visual_tokens_estimated": keep_all_projected_tokens,
        "llm_actual_visual_tokens": llm_actual_visual_tokens,
        "llm_actual_visual_tokens_after_autogaze": llm_actual_visual_tokens,
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
    return sum(model_patches_per_frame_by_scale(model).values())


def model_patches_per_frame_by_scale(model) -> dict[str, int]:
    vision_tower = getattr(model, "vision_tower", None)
    config = getattr(vision_tower, "config", None)
    scales = getattr(config, "scales", None)
    patch_size = getattr(config, "patch_size", NVILA_TARGET_PATCH_SIZE)
    if isinstance(scales, str):
        parsed_scales = parse_int_sequence(scales) or NVILA_TARGET_SCALES
    elif isinstance(scales, int):
        parsed_scales = [int(scales)]
    elif scales is None:
        parsed_scales = NVILA_TARGET_SCALES
    else:
        parsed_scales = [int(scale) for scale in scales]
    return patch_positions_by_scale(parsed_scales, int(patch_size))


def processor_autogaze_target_scales(processor) -> list[int] | None:
    scales = getattr(processor, "target_scales", None)
    if scales is None:
        if hasattr(processor, "target_patch_size"):
            return list(NVILA_AUTOGAZE_TARGET_SCALES)
        return None
    return [int(scale) for scale in scales]


def processor_autogaze_target_patch_size(processor) -> int | None:
    patch_size = getattr(processor, "target_patch_size", None)
    if patch_size is None:
        if hasattr(processor, "target_scales"):
            return NVILA_AUTOGAZE_TARGET_PATCH_SIZE
        return None
    return int(patch_size)


def processor_autogaze_patches_per_frame_by_scale(processor) -> dict[str, int] | None:
    scales = processor_autogaze_target_scales(processor)
    patch_size = processor_autogaze_target_patch_size(processor)
    if scales is None or patch_size is None:
        return None
    return patch_positions_by_scale(scales, patch_size)


def build_patch_space_metadata(model, processor) -> dict[str, Any]:
    vision_config = getattr(getattr(model, "vision_tower", None), "config", None)
    vision_scales = [int(scale) for scale in model_patches_per_frame_by_scale(model).keys()]
    vision_patch_size = int(getattr(vision_config, "patch_size", NVILA_VISION_PATCH_SIZE))
    vision_by_scale = model_patches_per_frame_by_scale(model)

    autogaze_scales = processor_autogaze_target_scales(processor)
    autogaze_patch_size = processor_autogaze_target_patch_size(processor)
    autogaze_by_scale = processor_autogaze_patches_per_frame_by_scale(processor)

    metadata: dict[str, Any] = {
        "vision_encoder_scales": vision_scales,
        "vision_encoder_patch_size": vision_patch_size,
        "vision_encoder_patches_per_frame_by_scale": vision_by_scale,
        "vision_encoder_patches_per_frame_multiscale": sum(vision_by_scale.values()),
    }
    if autogaze_scales is not None and autogaze_patch_size is not None and autogaze_by_scale is not None:
        metadata.update(
            {
                "autogaze_target_scales": autogaze_scales,
                "autogaze_target_patch_size": autogaze_patch_size,
                "autogaze_coordinate_patches_per_frame_by_scale": autogaze_by_scale,
                "autogaze_coordinate_patches_per_frame_multiscale": sum(autogaze_by_scale.values()),
                "patch_space_mismatch": (
                    autogaze_patch_size != vision_patch_size
                    or autogaze_scales != vision_scales
                ),
                "patch_space_note": (
                    "AutoGaze target coordinates and NVILA SigLIP embedding coordinates can differ; "
                    "use AutoGaze fields for gaze slot budgets and vision_encoder fields for SigLIP full-token estimates."
                ),
            }
        )
    return metadata


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


def video_resize_config(args: argparse.Namespace, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    source_width = int(metadata["width"]) if metadata and metadata.get("width") is not None else None
    source_height = int(metadata["height"]) if metadata and metadata.get("height") is not None else None
    effective = None
    if source_width is not None and source_height is not None:
        effective = apply_resize_to_dimensions(
            width=source_width,
            height=source_height,
            shortest_edge=getattr(args, "video_resize_shortest_edge", None),
            longest_edge=getattr(args, "video_resize_longest_edge", None),
            exact_width=getattr(args, "video_resize_width", None),
            exact_height=getattr(args, "video_resize_height", None),
        )
    return {
        "enabled": has_video_resize(args),
        "shortest_edge": getattr(args, "video_resize_shortest_edge", None),
        "longest_edge": getattr(args, "video_resize_longest_edge", None),
        "width": getattr(args, "video_resize_width", None),
        "height": getattr(args, "video_resize_height", None),
        "effective": effective,
    }


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolution(width: int | None, height: int | None) -> str | None:
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def read_video_metadata_if_local(video: str) -> dict[str, Any] | None:
    if video.startswith("http://") or video.startswith("https://"):
        return None
    path = Path(video)
    if not path.exists():
        return None
    try:
        return read_video_metadata(str(path))
    except Exception:
        return None


def build_video_input_summary(
    *,
    args: argparse.Namespace,
    resolved_video: str,
    source_metadata: dict[str, Any] | None,
    video_input_info: dict[str, Any],
    token_metrics: dict[str, Any],
) -> dict[str, Any]:
    source_width = _maybe_int(source_metadata.get("width")) if source_metadata else None
    source_height = _maybe_int(source_metadata.get("height")) if source_metadata else None
    source_frames = _maybe_int(source_metadata.get("frames")) if source_metadata else None
    requested_video_frames = _maybe_int(getattr(args, "num_video_frames", None))
    requested_thumbnail_frames = _maybe_int(getattr(args, "num_video_frames_thumbnail", None))
    resize = video_input_info.get("resize") if isinstance(video_input_info, dict) else None
    decode = video_input_info.get("decode") if isinstance(video_input_info, dict) else None
    if not isinstance(decode, dict):
        decode = {}
    if not isinstance(resize, dict):
        resize = video_resize_config(args, source_metadata)
    effective = resize.get("effective") if isinstance(resize, dict) else None
    processor_width = None
    processor_height = None
    if isinstance(effective, dict):
        processor_width = _maybe_int(effective.get("width"))
        processor_height = _maybe_int(effective.get("height"))
    if processor_width is None:
        processor_width = source_width
    if processor_height is None:
        processor_height = source_height
    sampled_frame_start = None
    sampled_frame_end = None
    if (
        source_frames is not None
        and source_frames > 0
        and requested_video_frames is not None
        and requested_video_frames > 0
    ):
        sampled_indices = uniform_sample_indices(source_frames, requested_video_frames)
        sampled_frame_start = sampled_indices[0]
        sampled_frame_end = sampled_indices[-1]
    return {
        "resolved_video": resolved_video,
        "source_frames": source_frames,
        "source_resolution": _resolution(source_width, source_height),
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": _maybe_float(source_metadata.get("fps")) if source_metadata else None,
        "source_duration_seconds": _maybe_float(source_metadata.get("duration_seconds")) if source_metadata else None,
        "source_codec": source_metadata.get("codec") if source_metadata else None,
        "requested_video_frames": requested_video_frames,
        "actual_video_frames": _maybe_int(token_metrics.get("video_sampled_frames")),
        "requested_thumbnail_frames": requested_thumbnail_frames,
        "actual_thumbnail_frames": _maybe_int(token_metrics.get("thumbnail_sampled_frames")),
        "sampled_frame_start": sampled_frame_start,
        "sampled_frame_end": sampled_frame_end,
        "runner_resize_enabled": bool(resize.get("enabled")) if isinstance(resize, dict) else False,
        "runner_resize_request": {
            "shortest_edge": resize.get("shortest_edge") if isinstance(resize, dict) else None,
            "longest_edge": resize.get("longest_edge") if isinstance(resize, dict) else None,
            "width": resize.get("width") if isinstance(resize, dict) else None,
            "height": resize.get("height") if isinstance(resize, dict) else None,
        },
        "processor_input_width": processor_width,
        "processor_input_height": processor_height,
        "processor_input_resolution": _resolution(processor_width, processor_height),
        "processor_video_input_mode": video_input_info.get("mode"),
        "frames_loaded_for_processor": _maybe_int(video_input_info.get("frames_loaded")),
        "video_decode_requested_strategy": decode.get("requested_decode_strategy"),
        "video_decode_strategy": decode.get("decode_strategy"),
        "video_decode_strategy_fallback_error": decode.get("decode_strategy_fallback_error"),
        "video_decode_frames_read": _maybe_int(decode.get("decode_frames_read")),
        "video_decode_seek_groups": _maybe_int(decode.get("decode_seek_groups")),
        "video_decode_keyframes_indexed": _maybe_int(decode.get("decode_keyframes_indexed")),
        "video_decode_packets_scanned_for_keyframes": _maybe_int(
            decode.get("decode_packets_scanned_for_keyframes")
        ),
        "spatial_tiles_per_video": token_metrics.get("spatial_tiles_per_video"),
        "temporal_chunks_per_video": token_metrics.get("temporal_chunks_per_video"),
    }


def prepare_video_for_processor(video: str, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if not has_video_resize(args):
        return video, {"mode": "path_or_url", "resize": video_resize_config(args)}
    metadata = read_video_metadata(video)
    resize = apply_resize_to_dimensions(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        shortest_edge=getattr(args, "video_resize_shortest_edge", None),
        longest_edge=getattr(args, "video_resize_longest_edge", None),
        exact_width=getattr(args, "video_resize_width", None),
        exact_height=getattr(args, "video_resize_height", None),
    )
    frames, decode_stats = load_sampled_video_frames(
        video,
        args.num_video_frames,
        resize,
        decode_strategy=getattr(args, "video_decode_strategy", "auto"),
    )
    return frames, {
        "mode": "preloaded_resized_frames",
        "source_metadata": metadata,
        "resize": video_resize_config(args, metadata),
        "frames_loaded": len(frames),
        "decode": decode_stats,
    }


def processor_videos_argument(video_payload: Any, video_input_info: dict[str, Any]) -> Any:
    if video_input_info.get("mode") == "preloaded_resized_frames":
        return [video_payload]
    return video_payload


def _model_patch_size(model: Any, args: argparse.Namespace) -> int:
    target_patch_size = getattr(args, "autogaze_target_patch_size", None)
    if target_patch_size is not None:
        return int(target_patch_size)
    vision_tower = getattr(model, "vision_tower", None)
    config = getattr(vision_tower, "config", None)
    return int(getattr(config, "patch_size", NVILA_TARGET_PATCH_SIZE) or NVILA_TARGET_PATCH_SIZE)


def _model_scales(model: Any, args: argparse.Namespace) -> list[int]:
    target_scales = parse_int_sequence(getattr(args, "autogaze_target_scales", None))
    if target_scales is not None:
        return target_scales
    try:
        return [int(scale) for scale in model_patches_per_frame_by_scale(model).keys()]
    except Exception:
        return list(NVILA_TARGET_SCALES)


def _visualization_selected_resize(
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, int | str]:
    max_long_side = getattr(args, "visualization_selected_max_long_side", None)
    if max_long_side is None:
        return apply_resize_to_dimensions(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            shortest_edge=None,
            longest_edge=None,
            exact_width=None,
            exact_height=None,
        )
    return apply_resize_to_dimensions(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        shortest_edge=None,
        longest_edge=int(max_long_side),
        exact_width=None,
        exact_height=None,
    )


def _visualization_label(args: argparse.Namespace, video: str, gazing_mode: str) -> str:
    label = getattr(args, "_visualization_run_label", None)
    if label:
        return safe_label(str(label))
    return safe_label(f"single_{Path(video).stem}_{gazing_mode}")


def maybe_write_generation_visualization(
    *,
    args: argparse.Namespace,
    model: Any,
    resolved_video: str,
    source_metadata: dict[str, Any] | None,
    video_input_summary: dict[str, Any],
    processor_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    output_dir = getattr(args, "visualization_output_dir", None)
    if not output_dir:
        return None
    if resolved_video.startswith("http://") or resolved_video.startswith("https://"):
        return {
            "status": "skipped",
            "reason": "visualization_requires_local_video_path",
            "video": resolved_video,
        }
    if not Path(resolved_video).exists():
        return {
            "status": "skipped",
            "reason": "resolved_video_not_found",
            "video": resolved_video,
        }
    if not source_metadata or source_metadata.get("frames") is None:
        return {
            "status": "skipped",
            "reason": "source_frame_count_unavailable",
            "video": resolved_video,
        }

    try:
        sampled_frame_indices = uniform_sample_indices(
            int(source_metadata["frames"]),
            int(args.num_video_frames),
        )
        selected_resize = _visualization_selected_resize(source_metadata, args)
        selected_frames, selected_decode = load_sampled_video_frames(
            resolved_video,
            int(args.num_video_frames),
            selected_resize,
            decode_strategy=getattr(args, "video_decode_strategy", "auto"),
        )

        overlay_resize = {
            "width": int(video_input_summary["processor_input_width"]),
            "height": int(video_input_summary["processor_input_height"]),
            "mode": "visualization_processor_input",
        }
        overlay_base_frames, overlay_decode = load_sampled_video_frames(
            resolved_video,
            int(args.num_video_frames),
            overlay_resize,
            decode_strategy=getattr(args, "video_decode_strategy", "auto"),
        )

        grid = spatial_tile_grid(
            width=int(video_input_summary["processor_input_width"]),
            height=int(video_input_summary["processor_input_height"]),
            max_tiles_video=int(args.max_tiles_video),
            image_size=NVILA_IMAGE_SIZE,
        )
        manifest = write_gaze_visualization_artifacts(
            selected_frames=selected_frames,
            overlay_base_frames=overlay_base_frames,
            output_dir=output_dir,
            label=_visualization_label(args, resolved_video, args.gazing_mode),
            video=resolved_video,
            sampled_frame_indices=sampled_frame_indices,
            gazing_mode=args.gazing_mode,
            gazing_info=processor_inputs.get("gazing_info"),
            spatial_tiles=int(grid["tiles"]),
            grid_cols=int(grid["cols"]),
            grid_rows=int(grid["rows"]),
            scales=_model_scales(model, args),
            patch_size=_model_patch_size(model, args),
            tile_size=NVILA_IMAGE_SIZE,
            fps=float(getattr(args, "visualization_fps", 4.0)),
            alpha=float(getattr(args, "visualization_alpha", 0.35)),
        )
        manifest.update(
            {
                "selected_video_decode": selected_decode,
                "overlay_video_decode": overlay_decode,
                "processor_video_decode": overlay_decode,
                "selected_video_resize": selected_resize,
                "overlay_video_resize": overlay_resize,
                "processor_video_resize": overlay_resize,
                "processor_tile_grid": grid,
            }
        )
        return manifest
    except Exception as exc:
        return {
            "status": "failed",
            "error": repr(exc),
            "video": resolved_video,
        }


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


def repeat_last_stream_samples_after_eof(
    *,
    current_frames: list[Image.Image],
    thumbnails: list[Image.Image],
    last_selected_frame: Image.Image | None,
    missing_sampled_frames: int,
    missing_thumbnail_frames: int,
    tile_size: int,
) -> dict[str, int]:
    if missing_sampled_frames <= 0:
        return {"padded_sampled_frames_after_eof": 0, "padded_thumbnail_frames_after_eof": 0}
    if last_selected_frame is None:
        raise RuntimeError("Cannot pad missing video samples because no sampled frame was decoded.")

    thumbnail_padding = min(missing_sampled_frames, max(missing_thumbnail_frames, 0))
    for index in range(missing_sampled_frames):
        frame_copy = last_selected_frame.copy()
        current_frames.append(frame_copy)
        if index < thumbnail_padding:
            thumbnails.append(frame_copy.resize((tile_size, tile_size)))
    return {
        "padded_sampled_frames_after_eof": missing_sampled_frames,
        "padded_thumbnail_frames_after_eof": thumbnail_padding,
    }


def _measure_elapsed(device: torch.device, fn):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def _tensor_bytes(tensor: Any) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def summarize_stream_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    raw_patch_budget = sum(int(chunk["raw_patch_budget"]) for chunk in chunks)
    selected = sum(int(chunk["selected_non_padded_patches"]) for chunk in chunks)
    padded = sum(int(chunk.get("padded_gazing_positions", 0)) for chunk in chunks)
    total_slots = sum(int(chunk.get("total_gaze_slots", 0)) for chunk in chunks)
    tile_sequences = sum(int(chunk["tile_sequences"]) for chunk in chunks)
    summary = {
        "tile_sequences": tile_sequences,
        "raw_patch_budget": raw_patch_budget,
        "selected_non_padded_patches": selected,
        "padded_gazing_positions": padded,
        "total_gaze_slots": total_slots,
        "token_reduction_ratio": raw_patch_budget / selected if selected else None,
    }
    for key in (
        "siglip_gazed_forward_ms",
        "siglip_keep_all_forward_ms",
        "siglip_gazed_sequence_slots_sum",
        "siglip_gazed_sequence_slots_squared_sum",
    ):
        values = [float(chunk.get(key, 0.0) or 0.0) for chunk in chunks]
        summary[key] = sum(values)
    for key in (
        "siglip_gazed_hidden_bytes_peak",
        "siglip_keep_all_hidden_bytes_peak",
    ):
        values = [int(chunk.get(key, 0) or 0) for chunk in chunks]
        summary[key] = max(values) if values else 0
    return summary


def build_keep_all_gazing_info(
    *,
    batch_size: int,
    frames: int,
    patches_per_frame_value: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    total_patches = frames * patches_per_frame_value
    positions = torch.arange(total_patches, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    return {
        "gazing_pos": positions,
        "if_padded_gazing": torch.zeros_like(positions, dtype=torch.bool),
        "num_gazing_each_frame": torch.full((frames,), patches_per_frame_value, device=device, dtype=torch.long),
    }


def siglip_hidden_summary(output: Any) -> tuple[list[int] | None, int]:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        return None, 0
    return list(hidden.shape), _tensor_bytes(hidden)


def run_siglip_on_stream_batch(
    *,
    siglip_model: Any,
    batch: torch.Tensor,
    gazing_info: dict[str, Any],
    mode: str,
    patches_per_frame_value: int,
    profiler: StageProfiler,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with torch.inference_mode():
        if mode in {"gazed", "both"}:
            output, elapsed_ms = _measure_elapsed(
                batch.device,
                lambda: siglip_model(batch, gazing_info=gazing_info),
            )
            shape, hidden_bytes = siglip_hidden_summary(output)
            profiler.add("siglip_gazed_forward", elapsed_ms)
            summary.update(
                {
                    "siglip_gazed_forward_ms": elapsed_ms,
                    "siglip_gazed_last_hidden_shape": shape,
                    "siglip_gazed_hidden_bytes_peak": hidden_bytes,
                }
            )

        if mode in {"keep-all", "both"}:
            keep_all_info = build_keep_all_gazing_info(
                batch_size=int(batch.shape[0]),
                frames=int(batch.shape[1]),
                patches_per_frame_value=patches_per_frame_value,
                device=batch.device,
            )
            output, elapsed_ms = _measure_elapsed(
                batch.device,
                lambda: siglip_model(batch, gazing_info=keep_all_info),
            )
            shape, hidden_bytes = siglip_hidden_summary(output)
            profiler.add("siglip_keep_all_forward", elapsed_ms)
            summary.update(
                {
                    "siglip_keep_all_forward_ms": elapsed_ms,
                    "siglip_keep_all_last_hidden_shape": shape,
                    "siglip_keep_all_hidden_bytes_peak": hidden_bytes,
                }
            )
    return summary


def run_autogaze_on_stream_tile_sequences(
    *,
    tile_sequences: list[list[Image.Image]],
    transform: Any,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    gazing_ratio: float | list[float],
    task_loss_requirement: float | None,
    target_scales: list[int],
    target_patch_size: int,
    patches_per_frame_value: int,
    profiler: StageProfiler,
    generate_only: bool = False,
    siglip_model: Any | None = None,
    siglip_mode: str = "gazed",
) -> tuple[dict[str, Any], int]:
    from autogaze.datasets.video_utils import transform_video_for_pytorch
    from repro.autogaze_bench import summarize_gaze

    raw_patch_budget = 0
    selected = 0
    padded = 0
    total_slots = 0
    tensorize_ms_total = 0.0
    forward_ms = 0.0
    tensor_bytes_peak = 0
    siglip_gazed_ms = 0.0
    siglip_keep_all_ms = 0.0
    siglip_gazed_hidden_bytes_peak = 0
    siglip_keep_all_hidden_bytes_peak = 0
    siglip_gazed_last_hidden_shape = None
    siglip_keep_all_last_hidden_shape = None
    siglip_gazed_sequence_slots_sum = 0
    siglip_gazed_sequence_slots_squared_sum = 0

    with torch.inference_mode():
        for start in range(0, len(tile_sequences), max_batch_size):
            batch_sequences = tile_sequences[start : start + max_batch_size]

            def tensorize():
                flat_tiles = [np.array(frame) for sequence in batch_sequences for frame in sequence]
                transformed = transform_video_for_pytorch(np.stack(flat_tiles), transform)
                return transformed.reshape(len(batch_sequences), len(batch_sequences[0]), *transformed.shape[1:])

            transformed, tensorize_ms = _measure_elapsed(device, tensorize)
            profiler.add("tile_autogaze_tensorize", tensorize_ms)
            tensorize_ms_total += tensorize_ms
            tensor_bytes_peak = max(tensor_bytes_peak, _tensor_bytes(transformed))

            batch = transformed.to(device=device, dtype=dtype)
            raw_budget = int(batch.shape[0] * batch.shape[1] * patches_per_frame_value)

            def forward():
                return model(
                    {"video": batch},
                    gazing_ratio=gazing_ratio,
                    task_loss_requirement=task_loss_requirement,
                    target_scales=target_scales,
                    target_patch_size=target_patch_size,
                    generate_only=generate_only,
                )

            outputs, elapsed_ms = _measure_elapsed(device, forward)
            profiler.add("tile_autogaze_forward", elapsed_ms)
            summary = summarize_gaze(outputs, raw_budget)
            raw_patch_budget += raw_budget
            selected += int(summary["selected_non_padded_patches"])
            padded += int(summary["padded_gazing_positions"])
            total_slots += int(summary["total_gaze_slots"])
            if isinstance(outputs.get("if_padded_gazing"), torch.Tensor):
                slots = int(outputs["if_padded_gazing"].shape[-1])
                siglip_gazed_sequence_slots_sum += int(batch.shape[0]) * slots
                siglip_gazed_sequence_slots_squared_sum += int(batch.shape[0]) * slots * slots
            forward_ms += elapsed_ms

            if siglip_model is not None:
                siglip_summary = run_siglip_on_stream_batch(
                    siglip_model=siglip_model,
                    batch=batch,
                    gazing_info=outputs,
                    mode=siglip_mode,
                    patches_per_frame_value=patches_per_frame_value,
                    profiler=profiler,
                )
                siglip_gazed_ms += float(siglip_summary.get("siglip_gazed_forward_ms", 0.0) or 0.0)
                siglip_keep_all_ms += float(siglip_summary.get("siglip_keep_all_forward_ms", 0.0) or 0.0)
                siglip_gazed_hidden_bytes_peak = max(
                    siglip_gazed_hidden_bytes_peak,
                    int(siglip_summary.get("siglip_gazed_hidden_bytes_peak", 0) or 0),
                )
                siglip_keep_all_hidden_bytes_peak = max(
                    siglip_keep_all_hidden_bytes_peak,
                    int(siglip_summary.get("siglip_keep_all_hidden_bytes_peak", 0) or 0),
                )
                siglip_gazed_last_hidden_shape = (
                    siglip_summary.get("siglip_gazed_last_hidden_shape") or siglip_gazed_last_hidden_shape
                )
                siglip_keep_all_last_hidden_shape = (
                    siglip_summary.get("siglip_keep_all_last_hidden_shape") or siglip_keep_all_last_hidden_shape
                )

    return (
        {
            "tile_sequences": len(tile_sequences),
            "raw_patch_budget": raw_patch_budget,
            "selected_non_padded_patches": selected,
            "padded_gazing_positions": padded,
            "total_gaze_slots": total_slots,
            "token_reduction_ratio": raw_patch_budget / selected if selected else None,
            "autogaze_tensorize_ms": tensorize_ms_total,
            "autogaze_forward_ms": forward_ms,
            "siglip_gazed_forward_ms": siglip_gazed_ms,
            "siglip_keep_all_forward_ms": siglip_keep_all_ms,
            "siglip_gazed_hidden_bytes_peak": siglip_gazed_hidden_bytes_peak,
            "siglip_keep_all_hidden_bytes_peak": siglip_keep_all_hidden_bytes_peak,
            "siglip_gazed_last_hidden_shape": siglip_gazed_last_hidden_shape,
            "siglip_keep_all_last_hidden_shape": siglip_keep_all_last_hidden_shape,
            "siglip_gazed_sequence_slots_sum": siglip_gazed_sequence_slots_sum,
            "siglip_gazed_sequence_slots_squared_sum": siglip_gazed_sequence_slots_squared_sum,
            "generate_only": generate_only,
        },
        tensor_bytes_peak,
    )


def build_keep_all_stream_chunk_summary(
    *,
    tile_sequences: list[list[Image.Image]],
    patches_per_frame_value: int,
    profiler: StageProfiler,
) -> tuple[dict[str, Any], int]:
    def build_summary():
        raw_budget = len(tile_sequences) * len(tile_sequences[0]) * patches_per_frame_value
        sequence_slots = len(tile_sequences[0]) * patches_per_frame_value
        return {
            "tile_sequences": len(tile_sequences),
            "raw_patch_budget": raw_budget,
            "selected_non_padded_patches": raw_budget,
            "padded_gazing_positions": 0,
            "total_gaze_slots": 0,
            "token_reduction_ratio": 1.0,
            "autogaze_tensorize_ms": 0.0,
            "autogaze_forward_ms": 0.0,
            "siglip_gazed_sequence_slots_sum": len(tile_sequences) * sequence_slots,
            "siglip_gazed_sequence_slots_squared_sum": len(tile_sequences) * sequence_slots * sequence_slots,
        }

    summary, elapsed_ms = _measure_elapsed(profiler.device or torch.device("cpu"), build_summary)
    profiler.add("keep_all_mask_build", elapsed_ms)
    return summary, 0


def stream_profile_dtype(args: argparse.Namespace) -> torch.dtype:
    if getattr(args, "stream_dtype", "float32") == "float16":
        return torch.float16
    return torch.float32


def scales_to_string(scales: list[int]) -> str:
    return "+".join(str(int(scale)) for scale in scales)


def build_stream_siglip_model(
    *,
    args: argparse.Namespace,
    target_scales: list[int],
    target_patch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any | None, dict[str, Any]]:
    if not getattr(args, "stream_run_siglip", False):
        return None, {"enabled": False}
    if getattr(args, "gazing_mode", "autogaze") != "autogaze":
        raise ValueError("--stream-run-siglip currently requires --gazing-mode autogaze.")

    from autogaze.vision_encoders.siglip import SiglipVisionConfig, SiglipVisionModel

    scales = scales_to_string(target_scales)
    model_path = getattr(args, "stream_siglip_model", None)
    max_embed_batch_size = getattr(args, "stream_siglip_max_embed_batch_size", None)
    attn_implementation = getattr(args, "stream_siglip_attn_implementation", "sdpa")
    if model_path:
        model = SiglipVisionModel.from_pretrained(
            model_path,
            scales=scales,
            max_embed_batch_size=max_embed_batch_size,
            attn_implementation=attn_implementation,
        )
        source = "pretrained"
        random_init = False
    else:
        config = SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_channels=3,
            image_size=int(target_scales[-1]),
            patch_size=target_patch_size,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            frame_independent_encoding=False,
            scales=scales,
            max_embed_batch_size=max_embed_batch_size,
            attn_type="block_causal",
        )
        config._attn_implementation = attn_implementation
        model = SiglipVisionModel(config)
        source = "random_nvila_hd_vision_config"
        random_init = True

    model_patch_size = int(getattr(model.config, "patch_size", target_patch_size))
    if model_patch_size != int(target_patch_size):
        raise ValueError(
            f"SigLIP patch size ({model_patch_size}) must match --autogaze-target-patch-size "
            f"({target_patch_size}). For google/siglip2-base-patch16-224, use "
            "--autogaze-target-patch-size 16 and matching patch16 scales such as 32+64+112+224."
        )

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, {
        "enabled": True,
        "mode": getattr(args, "stream_siglip_mode", "gazed"),
        "model": model_path,
        "source": source,
        "random_init": random_init,
        "scales": target_scales,
        "patch_size": target_patch_size,
        "max_embed_batch_size": max_embed_batch_size,
        "attn_implementation": attn_implementation,
        "hidden_size": int(getattr(model.config, "hidden_size", 0) or 0),
        "intermediate_size": int(getattr(model.config, "intermediate_size", 0) or 0),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", 0) or 0),
        "num_attention_heads": int(getattr(model.config, "num_attention_heads", 0) or 0),
    }


def autogaze_processor_size_kwargs(target_scales: list[int]) -> dict[str, dict[str, int]]:
    largest_scale = int(target_scales[-1])
    return {
        "size": {"height": largest_scale, "width": largest_scale},
        "crop_size": {"height": largest_scale, "width": largest_scale},
    }


def run_stream_profile(args: argparse.Namespace) -> None:
    from repro.autogaze_bench import add_external_autogaze

    if args.num_video_frames % args.stream_chunk_frames != 0:
        raise ValueError("--num-video-frames must be divisible by --stream-chunk-frames for AutoGaze chunk profiling.")

    device = resolve_device(args.device)
    resolved_video = resolve_video(args.video, args)
    metadata = read_video_metadata(resolved_video)
    if metadata["frames"] is None:
        raise ValueError("Video frame count is required for stream-profile uniform sampling.")

    effective = apply_resize_to_dimensions(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        shortest_edge=getattr(args, "video_resize_shortest_edge", None),
        longest_edge=getattr(args, "video_resize_longest_edge", None),
        exact_width=getattr(args, "video_resize_width", None),
        exact_height=getattr(args, "video_resize_height", None),
    )
    target_scales = parse_int_sequence(getattr(args, "autogaze_target_scales", None)) or NVILA_TARGET_SCALES
    target_patch_size = int(getattr(args, "autogaze_target_patch_size", None) or NVILA_TARGET_PATCH_SIZE)
    patches_per_frame_value = patches_per_frame(target_scales, target_patch_size)
    plan = estimate_stream_profile_plan(
        width=int(effective["width"]),
        height=int(effective["height"]),
        source_frames=int(metadata["frames"]),
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        chunk_frames=args.stream_chunk_frames,
        max_batch_size_autogaze=args.max_batch_size_autogaze,
        scales=target_scales,
        patch_size=target_patch_size,
    )

    transform = None
    model = None
    siglip_model = None
    siglip_info: dict[str, Any] = {"enabled": False}
    dtype = stream_profile_dtype(args)
    if args.gazing_mode == "autogaze" or getattr(args, "stream_run_siglip", False):
        from repro.autogaze_bench import move_model_to_device_dtype

        add_external_autogaze(args.autogaze_repo)
    if args.gazing_mode == "autogaze":
        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

        transform = AutoGazeImageProcessor.from_pretrained(
            args.autogaze_model,
            **autogaze_processor_size_kwargs(target_scales),
        )
        model = move_model_to_device_dtype(AutoGaze.from_pretrained(args.autogaze_model), device, dtype)
        model.eval()
    siglip_model, siglip_info = build_stream_siglip_model(
        args=args,
        target_scales=target_scales,
        target_patch_size=target_patch_size,
        device=device,
        dtype=dtype,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    profiler = StageProfiler(device)
    sampled_indices = uniform_sample_indices(int(metadata["frames"]), args.num_video_frames)
    thumbnail_counts = Counter(nvila_thumbnail_indices(sampled_indices, args.num_video_frames_thumbnail))
    target_counts = Counter(sampled_indices)
    selected_index_set = set(target_counts)
    end_index = max(selected_index_set)
    grid = plan["tiling"]
    resize = {
        "width": int(effective["width"]),
        "height": int(effective["height"]),
        "mode": effective["mode"],
    }

    temporal_chunks: list[dict[str, Any]] = []
    current_frames: list[Image.Image] = []
    thumbnails: list[Image.Image] = []
    raw_frame_buffer_peak_bytes = 0
    tile_pil_buffer_peak_bytes = 0
    autogaze_tile_tensor_peak_bytes = 0
    siglip_gazed_hidden_peak_bytes = 0
    siglip_keep_all_hidden_peak_bytes = 0
    decoded_selected_frames = 0
    eof_padding_summary = {"padded_sampled_frames_after_eof": 0, "padded_thumbnail_frames_after_eof": 0}
    last_selected_frame: Image.Image | None = None
    decode_stats: dict[str, Any] = {
        "decode_strategy": args.stream_decode_strategy,
        "decode_frames_read": 0,
        "decode_seek_groups": 0,
        "decode_keyframes_indexed": None,
        "decode_packets_scanned_for_keyframes": None,
    }

    def process_current_stream_chunk() -> None:
        nonlocal current_frames, tile_pil_buffer_peak_bytes, autogaze_tile_tensor_peak_bytes
        nonlocal siglip_gazed_hidden_peak_bytes, siglip_keep_all_hidden_peak_bytes
        tile_sequences, tile_build_ms = _measure_elapsed(
            device,
            lambda current_frames=current_frames: build_spatial_tile_sequences(
                current_frames,
                cols=int(grid["cols"]),
                rows=int(grid["rows"]),
                tile_size=NVILA_IMAGE_SIZE,
            ),
        )
        profiler.add("spatial_tile_build", tile_build_ms)
        tile_pil_buffer_peak_bytes = max(
            tile_pil_buffer_peak_bytes,
            len(current_frames) * int(grid["spatial_tiles"]) * NVILA_IMAGE_SIZE * NVILA_IMAGE_SIZE * 3,
        )

        if args.gazing_mode == "autogaze":
            chunk_summary, tile_tensor_bytes = run_autogaze_on_stream_tile_sequences(
                tile_sequences=tile_sequences,
                transform=transform,
                model=model,
                device=device,
                dtype=dtype,
                max_batch_size=args.max_batch_size_autogaze,
                gazing_ratio=effective_stream_gazing_ratio(args),
                task_loss_requirement=args.task_loss_requirement_tile,
                target_scales=target_scales,
                target_patch_size=target_patch_size,
                patches_per_frame_value=patches_per_frame_value,
                profiler=profiler,
                generate_only=bool(getattr(args, "autogaze_generate_only", False)),
                siglip_model=siglip_model,
                siglip_mode=args.stream_siglip_mode,
            )
        else:
            chunk_summary, tile_tensor_bytes = build_keep_all_stream_chunk_summary(
                tile_sequences=tile_sequences,
                patches_per_frame_value=patches_per_frame_value,
                profiler=profiler,
            )

        chunk_start_pos = len(temporal_chunks) * args.stream_chunk_frames
        chunk_summary["sampled_frame_start"] = sampled_indices[chunk_start_pos]
        chunk_summary["sampled_frame_end"] = sampled_indices[chunk_start_pos + args.stream_chunk_frames - 1]
        chunk_summary["spatial_tile_build_ms"] = tile_build_ms
        temporal_chunks.append(chunk_summary)
        autogaze_tile_tensor_peak_bytes = max(autogaze_tile_tensor_peak_bytes, tile_tensor_bytes)
        siglip_gazed_hidden_peak_bytes = max(
            siglip_gazed_hidden_peak_bytes,
            int(chunk_summary.get("siglip_gazed_hidden_bytes_peak", 0) or 0),
        )
        siglip_keep_all_hidden_peak_bytes = max(
            siglip_keep_all_hidden_peak_bytes,
            int(chunk_summary.get("siglip_keep_all_hidden_bytes_peak", 0) or 0),
        )
        current_frames = []

    def process_selected_frame(frame_index: int, frame: Any) -> None:
        nonlocal decoded_selected_frames, last_selected_frame, raw_frame_buffer_peak_bytes
        image, to_pil_ms = _measure_elapsed(device, lambda: frame.to_image().convert("RGB"))
        profiler.add("video_frame_to_pil", to_pil_ms)
        if resize["mode"] != "none":
            image, resize_ms = _measure_elapsed(device, lambda image=image: resize_frame(image, resize))
            profiler.add("video_frame_resize", resize_ms)

        for _ in range(target_counts[frame_index]):
            frame_copy = image.copy()
            current_frames.append(frame_copy)
            last_selected_frame = frame_copy
            decoded_selected_frames += 1
            raw_frame_buffer_peak_bytes = max(
                raw_frame_buffer_peak_bytes,
                len(current_frames) * int(effective["width"]) * int(effective["height"]) * 3,
            )

            if thumbnail_counts[frame_index] > 0:
                thumb, thumb_resize_ms = _measure_elapsed(
                    device,
                    lambda frame_copy=frame_copy: frame_copy.resize((NVILA_IMAGE_SIZE, NVILA_IMAGE_SIZE)),
                )
                profiler.add("thumbnail_resize", thumb_resize_ms)
                thumbnails.append(thumb)
                thumbnail_counts[frame_index] -= 1

            if len(current_frames) == args.stream_chunk_frames:
                process_current_stream_chunk()

    if args.stream_decode_strategy == "scan":
        container = av.open(resolved_video)
        try:
            decoder = container.decode(video=0)
            frame_index = 0
            while frame_index <= end_index:
                frame, decode_ms = _measure_elapsed(device, lambda: next(decoder))
                profiler.add("video_decode_scan", decode_ms)
                decode_stats["decode_frames_read"] += 1

                if frame_index in selected_index_set:
                    process_selected_frame(frame_index, frame)
                frame_index += 1
        except StopIteration:
            pass
        finally:
            container.close()
    else:
        keyframe_result, keyframe_ms = _measure_elapsed(
            torch.device("cpu"),
            lambda: read_video_keyframe_indices(resolved_video),
        )
        keyframe_indices, keyframe_metadata = keyframe_result
        profiler.add("video_keyframe_index_scan", keyframe_ms)
        decode_stats["decode_keyframes_indexed"] = keyframe_metadata["keyframes"]
        decode_stats["decode_packets_scanned_for_keyframes"] = keyframe_metadata["packets_scanned"]
        groups = build_seek_decode_groups(
            target_indices=sorted(selected_index_set),
            keyframe_indices=keyframe_indices,
        )
        decode_stats["decode_seek_groups"] = len(groups)

        container = av.open(resolved_video)
        try:
            stream = container.streams.video[0]
            pts_per_frame = stream_pts_per_frame(average_rate=stream.average_rate, time_base=stream.time_base)
            if pts_per_frame is None:
                raise ValueError("Seek decode requires video average_rate and time_base metadata.")
            start_time = int(stream.start_time) if stream.start_time is not None else 0
            processed_targets: set[int] = set()
            for group in groups:
                seek_pts = frame_index_to_pts(
                    int(group["seek_frame_index"]),
                    pts_per_frame=pts_per_frame,
                    start_time=start_time,
                )
                _, seek_ms = _measure_elapsed(
                    device,
                    lambda seek_pts=seek_pts: container.seek(
                        seek_pts,
                        stream=stream,
                        backward=True,
                        any_frame=False,
                    ),
                )
                profiler.add("video_seek", seek_ms)
                group_targets = set(int(index) for index in group["target_indices"])
                group_last_target = max(group_targets)
                decoder = container.decode(video=0)
                while True:
                    try:
                        frame, decode_ms = _measure_elapsed(device, lambda: next(decoder))
                    except StopIteration:
                        break
                    decode_stats["decode_frames_read"] += 1
                    profiler.add("video_decode_seek", decode_ms)
                    if frame.pts is None:
                        continue
                    frame_index = pts_to_frame_index(frame.pts, pts_per_frame=pts_per_frame, start_time=start_time)
                    if frame_index in group_targets and frame_index not in processed_targets:
                        process_selected_frame(frame_index, frame)
                        processed_targets.add(frame_index)
                    if frame_index >= group_last_target:
                        break
        finally:
            container.close()

    if decoded_selected_frames < args.num_video_frames:
        missing_sampled_frames = args.num_video_frames - decoded_selected_frames
        missing_thumbnail_frames = sum(thumbnail_counts.values())
        eof_padding_summary, eof_padding_ms = _measure_elapsed(
            device,
            lambda: repeat_last_stream_samples_after_eof(
                current_frames=current_frames,
                thumbnails=thumbnails,
                last_selected_frame=last_selected_frame,
                missing_sampled_frames=missing_sampled_frames,
                missing_thumbnail_frames=missing_thumbnail_frames,
                tile_size=NVILA_IMAGE_SIZE,
            ),
        )
        profiler.add("eof_sample_padding", eof_padding_ms)
        decoded_selected_frames += eof_padding_summary["padded_sampled_frames_after_eof"]
        raw_frame_buffer_peak_bytes = max(
            raw_frame_buffer_peak_bytes,
            len(current_frames) * int(effective["width"]) * int(effective["height"]) * 3,
        )
        if len(current_frames) == args.stream_chunk_frames:
            process_current_stream_chunk()

    if current_frames:
        raise RuntimeError(f"Unprocessed partial stream chunk with {len(current_frames)} frames.")
    if decoded_selected_frames != args.num_video_frames:
        raise RuntimeError(f"Decoded {decoded_selected_frames} sampled frames, expected {args.num_video_frames}.")

    thumbnail_tensor = None
    if thumbnails:
        if transform is not None:
            from autogaze.datasets.video_utils import transform_video_for_pytorch

            thumbnail_tensor, thumb_tensorize_ms = _measure_elapsed(
                device,
                lambda: transform_video_for_pytorch(np.stack([np.array(frame) for frame in thumbnails]), transform),
            )
        else:
            thumbnail_tensor, thumb_tensorize_ms = _measure_elapsed(
                device,
                lambda: torch.from_numpy(np.stack([np.array(frame) for frame in thumbnails])).permute(0, 3, 1, 2),
            )
        profiler.add("thumbnail_tensorize", thumb_tensorize_ms)

    tile_summary = summarize_stream_chunks(temporal_chunks)
    token_metrics = build_stream_profile_token_metrics(plan, tile_summary)
    stream_video_input_summary = {
        "source_resolution": _resolution(metadata.get("width"), metadata.get("height")),
        "source_width": _maybe_int(metadata.get("width")),
        "source_height": _maybe_int(metadata.get("height")),
        "processor_input_resolution": _resolution(effective["width"], effective["height"]),
        "processor_input_width": int(effective["width"]),
        "processor_input_height": int(effective["height"]),
        "runner_resize_enabled": bool(video_resize_config(args, metadata).get("enabled")),
        "runner_resize_request": {
            "shortest_edge": getattr(args, "video_resize_shortest_edge", None),
            "longest_edge": getattr(args, "video_resize_longest_edge", None),
            "width": getattr(args, "video_resize_width", None),
            "height": getattr(args, "video_resize_height", None),
        },
        "requested_video_frames": args.num_video_frames,
        "actual_video_frames": token_metrics.get("video_sampled_frames"),
        "requested_thumbnail_frames": args.num_video_frames_thumbnail,
        "actual_thumbnail_frames": token_metrics.get("thumbnail_sampled_frames"),
        "spatial_tiles_per_video": token_metrics.get("spatial_tiles_per_video"),
        "temporal_chunks_per_video": token_metrics.get("temporal_chunks_per_video"),
    }
    processing_budget_summary = build_processing_budget_summary(
        video_input_summary=stream_video_input_summary,
        token_metrics=token_metrics,
        runner="nvila_stream_profile",
    )
    compute_metrics = build_stream_profile_compute_metrics(
        plan,
        tile_summary,
        token_metrics,
        siglip_info=siglip_info,
        dtype_bytes=int(torch.empty((), dtype=dtype).element_size()),
    )
    stage_timings = profiler.as_dict()
    total_measured_ms = sum(float(value["total_ms"]) for value in stage_timings.values())
    if siglip_info.get("enabled"):
        metric_note = (
            "stream-profile measures decode/resize/tile/AutoGaze and customized SigLIP vision forward one temporal "
            "chunk at a time. Raw frames, tile images, and tile tensors are released after each chunk. NVILA projector "
            "and LLM timings remain in single/hlvid mode because the public NVILA generate path consumes the collected "
            "visual token sequence."
        )
    else:
        metric_note = (
            "stream-profile measures decode/resize/tile/AutoGaze work one temporal chunk at a time and releases "
            "raw frames/tiles after each chunk. NVILA vision encoding, projector, and LLM timings remain in single/hlvid "
            "mode because the public NVILA generate path consumes the collected visual token sequence."
        )
    payload = {
        "metadata": environment_metadata(device),
        "mode": "stream-profile",
        "model_path": args.model_path,
        "run_identity": build_run_identity(args),
        "autogaze_runtime_config": autogaze_runtime_config(args),
        "autogaze_model": args.autogaze_model,
        "gazing_mode": args.gazing_mode,
        "autogaze_generate_only": bool(getattr(args, "autogaze_generate_only", False)),
        "video": args.video,
        "video_resolved": resolved_video,
        "source_metadata": metadata,
        "effective_video": {
            "width": int(effective["width"]),
            "height": int(effective["height"]),
            "resize_mode": effective["mode"],
        },
        "video_resize": video_resize_config(args, metadata),
        "autogaze_target_scales": target_scales,
        "autogaze_target_patch_size": target_patch_size,
        "stream_siglip": siglip_info,
        "stream_plan": plan,
        "streaming_boundary": plan["streaming_boundary"],
        "sampling": {
            "policy": "nvila_round_linspace_over_full_video",
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "stream_chunk_frames": args.stream_chunk_frames,
            "sampled_frame_start": sampled_indices[0],
            "sampled_frame_end": sampled_indices[-1],
            "decoded_selected_frames": decoded_selected_frames,
            "thumbnail_frames_processed": len(thumbnails),
            **decode_stats,
            **eof_padding_summary,
        },
        "gaze": tile_summary,
        "token_metrics": token_metrics,
        "autogaze_token_summary": build_autogaze_token_summary(token_metrics),
        "processing_budget_summary": processing_budget_summary,
        "compute_metrics": compute_metrics,
        "timing_ms": {
            "video_decode_scan": stage_total(stage_timings, "video_decode_scan"),
            "video_keyframe_index_scan": stage_total(stage_timings, "video_keyframe_index_scan"),
            "video_seek": stage_total(stage_timings, "video_seek"),
            "video_decode_seek": stage_total(stage_timings, "video_decode_seek"),
            "video_frame_to_pil": stage_total(stage_timings, "video_frame_to_pil"),
            "video_frame_resize": stage_total(stage_timings, "video_frame_resize"),
            "spatial_tile_build": stage_total(stage_timings, "spatial_tile_build"),
            "tile_autogaze_tensorize": stage_total(stage_timings, "tile_autogaze_tensorize"),
            "tile_autogaze_forward": stage_total(stage_timings, "tile_autogaze_forward"),
            "siglip_gazed_forward": stage_total(stage_timings, "siglip_gazed_forward"),
            "siglip_keep_all_forward": stage_total(stage_timings, "siglip_keep_all_forward"),
            "keep_all_mask_build": stage_total(stage_timings, "keep_all_mask_build"),
            "thumbnail_resize": stage_total(stage_timings, "thumbnail_resize"),
            "thumbnail_tensorize": stage_total(stage_timings, "thumbnail_tensorize"),
            "eof_sample_padding": stage_total(stage_timings, "eof_sample_padding"),
            "pre_llm_stream_total_measured": total_measured_ms,
        },
        "stage_timings_ms": stage_timings,
        "memory_bytes": {
            "raw_frame_buffer_peak": raw_frame_buffer_peak_bytes,
            "tile_pil_buffer_peak": tile_pil_buffer_peak_bytes,
            "autogaze_tile_tensor_peak_per_temporal_chunk": autogaze_tile_tensor_peak_bytes,
            "siglip_gazed_hidden_peak": siglip_gazed_hidden_peak_bytes,
            "siglip_keep_all_hidden_peak": siglip_keep_all_hidden_peak_bytes,
            "thumbnail_tensor": _tensor_bytes(thumbnail_tensor),
            "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        },
        "temporal_chunk_summaries": temporal_chunks,
        "metric_note": metric_note,
    }
    write_json(args.stream_profile_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def generate_one(
    model,
    processor,
    video: str,
    prompt: str,
    device: torch.device,
    args: argparse.Namespace,
    *,
    enable_visualization: bool = True,
) -> dict[str, Any]:
    video_token = processor.tokenizer.video_token
    resolved_video = resolve_video(video, args)

    profiler = StageProfiler(device)
    with ProfilePatches(model, processor, profiler):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        video_payload: Any = resolved_video
        video_input_info: dict[str, Any] = {"mode": "path_or_url", "resize": video_resize_config(args)}
        if has_video_resize(args):
            with profiler.measure("video_decode_sampling"):
                video_payload, video_input_info = prepare_video_for_processor(resolved_video, args)
        with profiler.measure("processor_total"):
            inputs = processor(
                text=f"{video_token}\n\n{prompt}",
                videos=processor_videos_argument(video_payload, video_input_info),
                return_tensors="pt",
            )
        processor_peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        processor_timings = profiler.as_dict()
        profiler.reset()

        inputs = dict(inputs)
        processor_inputs = inputs
        token_metrics = compute_visual_token_metrics(
            inputs,
            video_token_id=resolve_video_token_id(model, processor),
            patches_per_frame_value=model_patches_per_frame(model),
            patches_per_frame_by_scale=model_patches_per_frame_by_scale(model),
            token_shuffle=NVILA_TOKEN_SHUFFLE,
        )
        token_metrics.update(build_patch_space_metadata(model, processor))
        input_token_count = int(inputs["input_ids"].shape[1])
        compute_metrics = build_autogaze_effect_metrics(
            inputs,
            model=model,
            token_metrics=token_metrics,
            input_token_count=input_token_count,
            dtype_bytes=_model_dtype_bytes(model),
            patches_per_frame_value=model_patches_per_frame(model),
            token_shuffle=NVILA_TOKEN_SHUFFLE,
        )
        gaze_metrics = extract_gaze_metrics(inputs)
        source_metadata = video_input_info.get("source_metadata")
        if source_metadata is None:
            source_metadata = read_video_metadata_if_local(resolved_video)
        video_input_summary = build_video_input_summary(
            args=args,
            resolved_video=resolved_video,
            source_metadata=source_metadata,
            video_input_info=video_input_info,
            token_metrics=token_metrics,
        )
        processing_budget_summary = build_processing_budget_summary(
            video_input_summary=video_input_summary,
            token_metrics=token_metrics,
            runner="nvila_runner",
        )

        target_device = input_device(model, device)
        inputs = move_tensors(inputs, target_device)

        ttft_ms = None
        ttft_timings = None
        ttft_peak_memory_bytes = None
        if args.measure_ttft:
            one_token = timed_generate(model, inputs, processor, device, max_new_tokens=1)
            ttft_ms = one_token["generate_ms"]
            ttft_peak_memory_bytes = one_token["peak_memory_bytes"]
            ttft_timings = profiler.as_dict()
            profiler.reset()

        result = timed_generate(model, inputs, processor, device, max_new_tokens=args.max_new_tokens)
        generate_timings = profiler.as_dict()
        profiler.reset()

        visualization = None
        if enable_visualization:
            visualization = maybe_write_generation_visualization(
                args=args,
                model=model,
                resolved_video=resolved_video,
                source_metadata=source_metadata,
                video_input_summary=video_input_summary,
                processor_inputs=processor_inputs,
            )

    preprocess_ms = stage_total(processor_timings, "processor_total") or 0.0
    if video_input_info["mode"] == "preloaded_resized_frames":
        preprocess_ms += stage_total(processor_timings, "video_decode_sampling") or 0.0
    decode_estimated_ms = max(result["generate_ms"] - ttft_ms, 0.0) if ttft_ms is not None else None
    video_decode_ms = stage_total(processor_timings, "video_decode_sampling")
    video_tiling_ms = stage_total(processor_timings, "video_tiling_and_tensorize")
    gazing_info_total_ms = stage_total(processor_timings, "autogaze_total")
    autogaze_total_ms = gazing_info_total_ms if gazing_info_total_ms is not None else 0.0
    video_preprocess_without_autogaze_ms = max(preprocess_ms - autogaze_total_ms, 0.0)
    autogaze_model_forward_ms = stage_total(processor_timings, "autogaze_forward_batched")
    vision_encoder_ms = stage_total(generate_timings, "vision_encode_total")
    siglip_vision_ms = stage_total(generate_timings, "siglip_vision_tower")
    mm_projector_ms = stage_total(generate_timings, "mm_projector")
    llm_forward_ms = stage_total(generate_timings, "llm_forward")
    latency_accounting = build_latency_accounting(
        {
            "total_ms": preprocess_ms + result["generate_ms"],
            "video_preprocess_ms": preprocess_ms,
            "video_preprocess_without_autogaze_ms": video_preprocess_without_autogaze_ms,
            "autogaze_total_ms": autogaze_total_ms,
            "generate_ms": result["generate_ms"],
            "ttft_ms": ttft_ms,
            "video_decode_ms": video_decode_ms,
            "video_tiling_ms": video_tiling_ms,
            "gazing_info_total_ms": gazing_info_total_ms,
            "autogaze_model_forward_ms": autogaze_model_forward_ms,
            "vision_encoder_ms": vision_encoder_ms,
            "siglip_vision_ms": siglip_vision_ms,
            "mm_projector_ms": mm_projector_ms,
            "llm_forward_ms": llm_forward_ms,
            "generation_decode_after_ttft_estimated_ms": decode_estimated_ms,
        }
    )

    return {
        **result,
        **gaze_metrics,
        "run_identity": build_run_identity(args),
        "autogaze_runtime_config": autogaze_runtime_config(args),
        "video_input": video,
        "video_resolved": resolved_video,
        "video_input_info": video_input_info,
        "video_input_summary": video_input_summary,
        "processing_budget_summary": processing_budget_summary,
        "gazing_mode": args.gazing_mode,
        "autogaze_generate_only": bool(getattr(args, "autogaze_generate_only", False)),
        "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
        "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
        "input_token_count": input_token_count,
        "input_shapes": tensor_shapes(inputs),
        "token_metrics": token_metrics,
        "autogaze_token_summary": build_autogaze_token_summary(token_metrics),
        "compute_metrics": compute_metrics,
        "visualization": visualization,
        "video_preprocess_ms": preprocess_ms,
        "video_preprocess_without_autogaze_ms": video_preprocess_without_autogaze_ms,
        "autogaze_total_ms": autogaze_total_ms,
        "video_decode_ms": video_decode_ms,
        "video_tiling_ms": video_tiling_ms,
        "autogaze_ms": gazing_info_total_ms,
        "gazing_info_total_ms": gazing_info_total_ms,
        "autogaze_forward_ms": autogaze_model_forward_ms,
        "autogaze_model_forward_ms": autogaze_model_forward_ms,
        "processor_peak_memory_bytes": processor_peak_memory_bytes,
        "ttft_ms": ttft_ms,
        "ttft_peak_memory_bytes": ttft_peak_memory_bytes,
        "llm_peak_memory_bytes": result["peak_memory_bytes"],
        "ttft_stage_timings_ms": ttft_timings,
        "decode_estimated_ms": decode_estimated_ms,
        "generation_decode_after_ttft_estimated_ms": decode_estimated_ms,
        "total_ms": preprocess_ms + result["generate_ms"],
        "latency_accounting": latency_accounting,
        "stage_timings_ms": {
            "processor": processor_timings,
            "ttft": ttft_timings,
            "generate": generate_timings,
        },
        "vision_encoder_ms": vision_encoder_ms,
        "siglip_vision_ms": siglip_vision_ms,
        "mm_projector_ms": mm_projector_ms,
        "llm_forward_ms": llm_forward_ms,
        "mllm_prefill_ms": ttft_ms,
        "metric_note": (
            "Stage timings are collected by wrapping public NVILA remote-code methods at runtime. "
            "video_tiling_ms includes tile creation and image tensorization; vision_encoder_ms includes "
            "SigLIP, feature reordering, and projector prep unless more specific sub-stages are present."
        ),
    }


def run_single(args: argparse.Namespace) -> None:
    try:
        device = resolve_device(args.device)
        model, processor = load_model_and_processor(args)
        warmup_runs = int(args.warmup_runs)
        repeat_runs = int(args.repeat_runs)
        for index in range(warmup_runs):
            print(f"Warmup run {index + 1}/{warmup_runs}", file=sys.stderr)
            generate_one(model, processor, args.video, args.prompt, device, args, enable_visualization=False)
        repeat_results: list[dict[str, Any]] = []
        for index in range(repeat_runs):
            print(f"Measured run {index + 1}/{repeat_runs}", file=sys.stderr)
            run_result = generate_one(
                model,
                processor,
                args.video,
                args.prompt,
                device,
                args,
                enable_visualization=index == repeat_runs - 1,
            )
            run_result["repeat_index"] = index
            repeat_results.append(run_result)
        result = repeat_results[-1]
        payload = {
            "metadata": environment_metadata(device),
            "model_path": args.model_path,
            "run_identity": build_run_identity(args),
            "autogaze_runtime_config": autogaze_runtime_config(args),
            "autogaze_model": args.autogaze_model,
            "gazing_mode": args.gazing_mode,
            "video_resize": video_resize_config(args),
            "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
            "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
            "video": args.video,
            "video_input_summary": result.get("video_input_summary"),
            "processing_budget_summary": result.get("processing_budget_summary"),
            "prompt": args.prompt,
            "result": result,
        }
        if warmup_runs or repeat_runs > 1:
            payload["warmup_runs"] = warmup_runs
            payload["repeat_runs"] = repeat_runs
            payload["repeat_results"] = repeat_results
            payload["repeat_summary"] = summarize_repeat_results(repeat_results)
            payload["repeat_metric_note"] = (
                "result is the last measured run for backward compatibility. "
                "Use repeat_summary median/mean/min/max fields for warmup-aware latency and memory claims."
            )
        summary = build_single_summary(payload)
        payload["summary"] = summary
        if args.summary_json:
            write_json(args.summary_json, summary)
        write_json(args.output_json, payload)
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except Exception as exc:
        failure = classify_exception(exc, stage="nvila_single")
        if failure["kind"] != "oom":
            raise
        payload = minimal_runner_failure_payload(args, failure)
        payload["metadata"] = {"device": getattr(args, "device", None)}
        payload["run_identity"] = build_run_identity(args)
        payload["summary"] = {"status": "oom", "failure": failure}
        if args.summary_json:
            write_json(args.summary_json, payload["summary"])
        write_json(args.output_json, payload)
        print(json.dumps(payload["summary"] if args.print_summary else payload, indent=2, sort_keys=True))


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

    effective = apply_resize_to_dimensions(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        shortest_edge=getattr(args, "video_resize_shortest_edge", None),
        longest_edge=getattr(args, "video_resize_longest_edge", None),
        exact_width=getattr(args, "video_resize_width", None),
        exact_height=getattr(args, "video_resize_height", None),
    )
    estimate = estimate_nvila_preflight(
        width=int(effective["width"]),
        height=int(effective["height"]),
        source_frames=metadata.get("frames"),
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
    )
    payload = {
        "model_path": args.model_path,
        "run_identity": build_run_identity(args),
        "autogaze_model": args.autogaze_model,
        "gazing_mode": args.gazing_mode,
        "video": args.video,
        "video_resolved": resolved_video,
        "source_metadata": metadata,
        "effective_video": {
            "width": int(effective["width"]),
            "height": int(effective["height"]),
            "resize_mode": effective["mode"],
        },
        "video_resize": video_resize_config(args, metadata),
        "estimate": estimate,
    }
    write_json(args.preflight_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_h100_preflight_sweep(args: argparse.Namespace) -> None:
    resolved_video = resolve_video(args.video, args)
    if args.preflight_width and args.preflight_height:
        metadata = {
            "width": args.preflight_width,
            "height": args.preflight_height,
            "frames": args.preflight_source_frames,
        }
    elif resolved_video.startswith("http://") or resolved_video.startswith("https://"):
        raise ValueError(
            "H100 preflight sweep needs a local video path or explicit --preflight-width and --preflight-height."
        )
    else:
        metadata = read_video_metadata(resolved_video)

    ratios = parse_float_sequence(args.h100_reduction_ratios)
    if ratios is None:
        ratios = [1.0, 2.0, 3.0, 4.0]
    model_family = effective_model_family(args)
    requested_ratios = [1.0] if model_family == MODEL_FAMILY_VIDEO_BASELINE else ratios
    stream_chunk_frames = (
        int(args.stream_chunk_frames)
        if getattr(args, "stream_chunk_frames", None) is not None and int(args.stream_chunk_frames) > 0
        else None
    )
    max_batch_size_siglip = int(getattr(args, "max_batch_size_siglip", 32) or 32)
    autogaze_residency_policy = getattr(args, "autogaze_residency_policy", "resident")
    autogaze_model_resident_gib = float(getattr(args, "autogaze_model_resident_gib", 0.0) or 0.0)
    requested_config_estimates = [
        estimate_h100_preflight_config(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            source_frames=metadata.get("frames"),
            model_family=model_family,
            num_video_frames=args.num_video_frames,
            num_video_frames_thumbnail=args.num_video_frames_thumbnail,
            max_tiles_video=args.max_tiles_video,
            resize_shortest_edge=getattr(args, "video_resize_shortest_edge", None),
            resize_longest_edge=getattr(args, "video_resize_longest_edge", None),
            resize_width=getattr(args, "video_resize_width", None),
            resize_height=getattr(args, "video_resize_height", None),
            token_reduction_ratio=ratio,
            h100_budget_gib=float(args.h100_budget_gib),
            stream_chunk_frames=stream_chunk_frames,
            max_batch_size_autogaze=getattr(args, "max_batch_size_autogaze", None),
            max_batch_size_siglip=max_batch_size_siglip,
            autogaze_residency_policy=autogaze_residency_policy,
            autogaze_model_resident_gib=autogaze_model_resident_gib,
        )
        for ratio in requested_ratios
    ]
    sweep = estimate_h100_preflight_sweep(
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        source_frames=metadata.get("frames"),
        model_family=model_family,
        token_reduction_ratios=([1.0] if model_family == MODEL_FAMILY_VIDEO_BASELINE else ratios),
        h100_budget_gib=float(args.h100_budget_gib),
        stream_chunk_frames=stream_chunk_frames,
        max_batch_size_autogaze=getattr(args, "max_batch_size_autogaze", None),
        max_batch_size_siglip=max_batch_size_siglip,
        autogaze_residency_policy=autogaze_residency_policy,
        autogaze_model_resident_gib=autogaze_model_resident_gib,
    )
    payload = {
        "model_path": args.model_path,
        "run_identity": build_run_identity(args),
        "video": args.video,
        "video_resolved": resolved_video,
        "source_metadata": metadata,
        "requested_config_estimates": requested_config_estimates,
        "sweep": sweep,
    }
    payload["summary"] = build_h100_decision_summary(
        requested_rows=requested_config_estimates,
        sweep=sweep,
    )
    write_json(args.h100_sweep_json, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def completed_question_ids(path: Path) -> set[Any]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text().splitlines() if line.strip()}


def run_hlvid(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if args.manifest:
        rows = read_manifest_file(args.manifest)
        if args.limit is not None:
            rows = rows[: args.limit]
    else:
        rows = load_hlvid_manifest(split=args.split, limit=args.limit, config=args.config)
    model, processor = load_model_and_processor(args)
    output_path = Path(args.predictions)
    completed_ids = completed_question_ids(output_path) if args.resume else set()

    warmup_rows = [row for row in rows if row["question_id"] not in completed_ids] or rows[:1]
    if warmup_rows and args.warmup_runs:
        warmup_row = warmup_rows[0]
        for index in range(args.warmup_runs):
            print(f"HLVid warmup run {index + 1}/{args.warmup_runs}", file=sys.stderr)
            generate_one(
                model,
                processor,
                warmup_row["video_path"],
                warmup_row["question"],
                device,
                args,
                enable_visualization=False,
            )

    for row in rows:
        if row["question_id"] in completed_ids:
            continue
        try:
            previous_label = getattr(args, "_visualization_run_label", None)
            args._visualization_run_label = (
                f"hlvid_{row['question_id']}_{Path(str(row['video_path'])).stem}_{args.gazing_mode}"
            )
            try:
                result = generate_one(model, processor, row["video_path"], row["question"], device, args)
            finally:
                if previous_label is None:
                    if hasattr(args, "_visualization_run_label"):
                        delattr(args, "_visualization_run_label")
                else:
                    args._visualization_run_label = previous_label
        except Exception as exc:
            if not args.continue_on_error:
                raise
            prediction = {
                **row,
                "status": "failed",
                "error": repr(exc),
                "model_path": args.model_path,
                "run_identity": build_run_identity(args),
                "autogaze_model": args.autogaze_model,
                "num_video_frames": args.num_video_frames,
                "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
                "max_tiles_video": args.max_tiles_video,
                "gazing_mode": args.gazing_mode,
                "autogaze_generate_only": bool(getattr(args, "autogaze_generate_only", False)),
                "video_resize": video_resize_config(args),
                "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
                "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
                "autogaze_runtime_config": autogaze_runtime_config(args),
                "task_loss_requirement_tile": args.task_loss_requirement_tile,
            }
            append_jsonl(output_path, [prediction])
            continue
        prediction = {
            **row,
            **result,
            "status": "ok",
            "model_path": args.model_path,
            "run_identity": build_run_identity(args),
            "autogaze_model": args.autogaze_model,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "gazing_mode": args.gazing_mode,
            "autogaze_generate_only": bool(getattr(args, "autogaze_generate_only", False)),
            "video_resize": video_resize_config(args),
            "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
            "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
            "autogaze_runtime_config": autogaze_runtime_config(args),
            "task_loss_requirement_tile": args.task_loss_requirement_tile,
        }
        append_jsonl(output_path, [prediction])

    all_rows = []
    if output_path.exists():
        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary, scored = score_predictions(all_rows)
    summary["token_budget_summary"] = summarize_token_budget_rows(all_rows)
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
    parser.add_argument(
        "--mode",
        choices=["single", "hlvid", "preflight", "stream-profile", "h100-preflight-sweep"],
        default="single",
    )
    parser.add_argument("--model-path", "--nvila-model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, default=MODEL_FAMILY_AUTO)
    parser.add_argument("--paper-preset", choices=PAPER_PRESET_CHOICES)
    parser.add_argument("--pipeline-preset", choices=PAPER_PRESET_CHOICES)
    parser.add_argument("--token-selector-adapter", choices=TOKEN_SELECTOR_ADAPTER_CHOICES, default="auto")
    parser.add_argument("--token-selector-name")
    parser.add_argument("--token-selector-path")
    parser.add_argument("--vision-encoder-adapter", choices=VISION_ENCODER_ADAPTER_CHOICES, default="auto")
    parser.add_argument("--vision-encoder-name")
    parser.add_argument("--vision-encoder-path")
    parser.add_argument("--mllm-adapter", choices=MLLM_ADAPTER_CHOICES, default="auto")
    parser.add_argument("--mllm-name")
    parser.add_argument("--mllm-path")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--video", default=DEFAULT_EXAMPLE_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
    parser.add_argument("--video-decode-strategy", choices=["auto", "seek", "scan"], default="auto")
    parser.add_argument("--gazing-mode", choices=["autogaze", "keep-all"], default="autogaze")
    parser.add_argument("--autogaze-target-scales", "--autogaze-resize-scales", dest="autogaze_target_scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--gazing-ratio-tile")
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.6)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--autogaze-generate-only", action="store_true")
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--warmup-runs", type=non_negative_int, default=0)
    parser.add_argument("--repeat-runs", type=positive_int, default=1)
    parser.add_argument("--visualization-output-dir")
    parser.add_argument("--visualization-fps", type=float, default=4.0)
    parser.add_argument("--visualization-alpha", type=float, default=0.35)
    parser.add_argument("--visualization-selected-max-long-side", type=int)
    parser.add_argument("--stream-chunk-frames", type=int, default=AUTOGAZE_CHUNK_FRAMES)
    parser.add_argument("--stream-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--stream-decode-strategy", choices=["scan", "seek"], default="scan")
    parser.add_argument("--stream-gazing-ratio")
    parser.add_argument("--stream-run-siglip", action="store_true")
    parser.add_argument("--stream-siglip-mode", choices=["gazed", "keep-all", "both"], default="gazed")
    parser.add_argument("--stream-siglip-model")
    parser.add_argument("--stream-siglip-max-embed-batch-size", type=int, default=1)
    parser.add_argument("--stream-siglip-attn-implementation", default="sdpa")
    parser.add_argument("--hlvid-repo", default="bfshi/HLVid")
    parser.add_argument("--hlvid-video-root", default="data/hlvid/videos")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight-width", type=int)
    parser.add_argument("--preflight-height", type=int)
    parser.add_argument("--preflight-source-frames", type=int)
    parser.add_argument("--preflight-json", default="outputs/autogaze_repro/nvila_preflight.json")
    parser.add_argument("--h100-budget-gib", type=float, default=H100_DEFAULT_BUDGET_GIB)
    parser.add_argument("--h100-reduction-ratios", default="1,2,3,4")
    parser.add_argument(
        "--autogaze-residency-policy",
        choices=["resident", "unload-before-generate"],
        default="resident",
    )
    parser.add_argument("--autogaze-model-resident-gib", type=float, default=0.0)
    parser.add_argument("--h100-sweep-json", default="outputs/autogaze_repro/h100_preflight_sweep.json")
    parser.add_argument("--stream-profile-json", default="outputs/autogaze_repro/nvila_stream_profile.json")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/nvila_single.json")
    parser.add_argument("--summary-json")
    parser.add_argument("--print-summary", action="store_true")
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
    args = build_parser(defaults).parse_args(argv)
    apply_pipeline_preset_alias(args)
    apply_paper_preset_defaults(args, provided_cli_options(argv))
    apply_component_defaults(args)
    validate_thumbnail_compatibility(args)
    return args


def main() -> None:
    try:
        args = parse_args()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "stream-profile":
        run_stream_profile(args)
    elif args.mode == "h100-preflight-sweep":
        run_h100_preflight_sweep(args)
    elif args.mode == "single":
        run_single(args)
    else:
        run_hlvid(args)


if __name__ == "__main__":
    main()
