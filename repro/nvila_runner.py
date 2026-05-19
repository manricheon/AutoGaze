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
from repro.hlvid import load_hlvid_manifest, parse_choice, read_manifest_file, score_predictions

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
REPEAT_SUMMARY_FIELDS = (
    "total_ms",
    "video_preprocess_ms",
    "video_decode_ms",
    "video_tiling_ms",
    "autogaze_ms",
    "autogaze_forward_ms",
    "vision_encoder_ms",
    "siglip_vision_ms",
    "mm_projector_ms",
    "llm_forward_ms",
    "ttft_ms",
    "decode_estimated_ms",
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


def patches_per_frame(scales: list[int] | None = None, patch_size: int = NVILA_TARGET_PATCH_SIZE) -> int:
    active_scales = scales or NVILA_TARGET_SCALES
    return sum((scale // patch_size) ** 2 for scale in active_scales)


def parse_int_sequence(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = [int(part) for part in re.findall(r"\d+", value)]
    return parsed or None


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
            "For example, 128 frames * 8 tiles/frame * 1060 multiscale patches = 1085440."
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
    question = payload.get("question")
    if question is None and isinstance(result, dict):
        question = result.get("question")
    return {
        "model_path": payload.get("model_path"),
        "gazing_mode": payload.get("gazing_mode"),
        "video": payload.get("video"),
        "prompt": payload.get("prompt"),
        "question": question,
        "video_input_summary": result.get("video_input_summary") if isinstance(result, dict) else None,
        "autogaze_token_summary": build_autogaze_token_summary(token_metrics),
        "key_autogaze_effect": {
            "gazing_mode": payload.get("gazing_mode"),
            "total_ms_median": summary_metric(payload, "total_ms"),
            "ttft_ms_median": summary_metric(payload, "ttft_ms"),
            "autogaze_total_ms_median": summary_metric(payload, "autogaze_ms"),
            "autogaze_forward_ms_median": summary_metric(payload, "autogaze_forward_ms"),
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
        "latency_ms": {
            "total_median": summary_metric(payload, "total_ms"),
            "ttft_median": summary_metric(payload, "ttft_ms"),
            "decode_estimated_median": summary_metric(payload, "decode_estimated_ms"),
            "video_decode_median": summary_metric(payload, "video_decode_ms"),
            "video_tiling_median": summary_metric(payload, "video_tiling_ms"),
            "autogaze_total_median": summary_metric(payload, "autogaze_ms"),
            "autogaze_forward_median": summary_metric(payload, "autogaze_forward_ms"),
            "siglip_vision_median": summary_metric(payload, "siglip_vision_ms"),
            "mm_projector_median": summary_metric(payload, "mm_projector_ms"),
            "llm_forward_median": summary_metric(payload, "llm_forward_ms"),
        },
        "memory_bytes": {
            "processor_peak_median": summary_metric(payload, "processor_peak_memory_bytes"),
            "ttft_peak_median": summary_metric(payload, "ttft_peak_memory_bytes"),
            "llm_peak_median": summary_metric(payload, "llm_peak_memory_bytes"),
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


def load_sampled_video_frames(video: str, sample_count: int, resize: dict[str, int | str]) -> list[Image.Image]:
    metadata = read_video_metadata(video)
    total_frames = metadata.get("frames")
    if total_frames is None:
        raise ValueError("Video frame count is required for runner-side resize sampling.")
    indices = uniform_sample_indices(int(total_frames), sample_count)
    target_counts: dict[int, int] = {}
    for index in indices:
        target_counts[index] = target_counts.get(index, 0) + 1
    max_index = max(target_counts)

    frames: list[Image.Image] = []
    container = av.open(video)
    try:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index > max_index:
                break
            count = target_counts.get(frame_index, 0)
            if count == 0:
                continue
            image = resize_frame(frame.to_image().convert("RGB"), resize)
            frames.extend(image.copy() for _ in range(count))
            if len(frames) >= sample_count:
                break
    finally:
        container.close()

    if not frames:
        raise ValueError(f"Could not extract any frames from video: {video}")
    while len(frames) < sample_count:
        frames.append(frames[-1].copy())
    return frames


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

    kwargs = {
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
    target_scales = parse_int_sequence(getattr(args, "autogaze_target_scales", None))
    if target_scales is not None:
        kwargs["target_scales"] = target_scales
    target_patch_size = getattr(args, "autogaze_target_patch_size", None)
    if target_patch_size is not None:
        kwargs["target_patch_size"] = int(target_patch_size)
    return kwargs


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
    return {str(scale): (scale // int(patch_size)) ** 2 for scale in parsed_scales}


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
    frames = load_sampled_video_frames(video, args.num_video_frames, resize)
    return frames, {
        "mode": "preloaded_resized_frames",
        "source_metadata": metadata,
        "resize": video_resize_config(args, metadata),
        "frames_loaded": len(frames),
    }


def processor_videos_argument(video_payload: Any, video_input_info: dict[str, Any]) -> Any:
    if video_input_info.get("mode") == "preloaded_resized_frames":
        return [video_payload]
    return video_payload


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
        "size": {"shortest_edge": largest_scale},
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
        add_external_autogaze(args.autogaze_repo)
    if args.gazing_mode == "autogaze":
        from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

        transform = AutoGazeImageProcessor.from_pretrained(
            args.autogaze_model,
            **autogaze_processor_size_kwargs(target_scales),
        )
        model = AutoGaze.from_pretrained(args.autogaze_model).to(device)
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
                gazing_ratio=[0.2] + [0.06] * 15,
                task_loss_requirement=args.task_loss_requirement_tile,
                target_scales=target_scales,
                target_patch_size=target_patch_size,
                patches_per_frame_value=patches_per_frame_value,
                profiler=profiler,
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


def generate_one(model, processor, video: str, prompt: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
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
        token_metrics = compute_visual_token_metrics(
            inputs,
            video_token_id=resolve_video_token_id(model, processor),
            patches_per_frame_value=model_patches_per_frame(model),
            patches_per_frame_by_scale=model_patches_per_frame_by_scale(model),
            token_shuffle=NVILA_TOKEN_SHUFFLE,
        )
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

    preprocess_ms = stage_total(processor_timings, "processor_total") or 0.0
    if video_input_info["mode"] == "preloaded_resized_frames":
        preprocess_ms += stage_total(processor_timings, "video_decode_sampling") or 0.0
    decode_estimated_ms = max(result["generate_ms"] - ttft_ms, 0.0) if ttft_ms is not None else None

    return {
        **result,
        **gaze_metrics,
        "video_input": video,
        "video_resolved": resolved_video,
        "video_input_info": video_input_info,
        "video_input_summary": video_input_summary,
        "gazing_mode": args.gazing_mode,
        "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
        "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
        "input_token_count": input_token_count,
        "input_shapes": tensor_shapes(inputs),
        "token_metrics": token_metrics,
        "autogaze_token_summary": build_autogaze_token_summary(token_metrics),
        "compute_metrics": compute_metrics,
        "video_preprocess_ms": preprocess_ms,
        "video_decode_ms": stage_total(processor_timings, "video_decode_sampling"),
        "video_tiling_ms": stage_total(processor_timings, "video_tiling_and_tensorize"),
        "autogaze_ms": stage_total(processor_timings, "autogaze_total"),
        "autogaze_forward_ms": stage_total(processor_timings, "autogaze_forward_batched"),
        "processor_peak_memory_bytes": processor_peak_memory_bytes,
        "ttft_ms": ttft_ms,
        "ttft_peak_memory_bytes": ttft_peak_memory_bytes,
        "llm_peak_memory_bytes": result["peak_memory_bytes"],
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
    warmup_runs = int(args.warmup_runs)
    repeat_runs = int(args.repeat_runs)
    for index in range(warmup_runs):
        print(f"Warmup run {index + 1}/{warmup_runs}", file=sys.stderr)
        generate_one(model, processor, args.video, args.prompt, device, args)
    repeat_results: list[dict[str, Any]] = []
    for index in range(repeat_runs):
        print(f"Measured run {index + 1}/{repeat_runs}", file=sys.stderr)
        run_result = generate_one(model, processor, args.video, args.prompt, device, args)
        run_result["repeat_index"] = index
        repeat_results.append(run_result)
    result = repeat_results[-1]
    payload = {
        "metadata": environment_metadata(device),
        "model_path": args.model_path,
        "autogaze_model": args.autogaze_model,
        "gazing_mode": args.gazing_mode,
        "video_resize": video_resize_config(args),
        "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
        "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
        "video": args.video,
        "video_input_summary": result.get("video_input_summary"),
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
            generate_one(model, processor, warmup_row["video_path"], warmup_row["question"], device, args)

    for row in rows:
        if row["question_id"] in completed_ids:
            continue
        try:
            result = generate_one(model, processor, row["video_path"], row["question"], device, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            prediction = {
                **row,
                "status": "failed",
                "error": repr(exc),
                "model_path": args.model_path,
                "autogaze_model": args.autogaze_model,
                "num_video_frames": args.num_video_frames,
                "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
                "max_tiles_video": args.max_tiles_video,
                "gazing_mode": args.gazing_mode,
                "video_resize": video_resize_config(args),
                "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
                "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
                "task_loss_requirement_tile": args.task_loss_requirement_tile,
            }
            append_jsonl(output_path, [prediction])
            continue
        prediction = {
            **row,
            **result,
            "status": "ok",
            "model_path": args.model_path,
            "autogaze_model": args.autogaze_model,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "gazing_mode": args.gazing_mode,
            "video_resize": video_resize_config(args),
            "autogaze_target_scales": parse_int_sequence(getattr(args, "autogaze_target_scales", None)),
            "autogaze_target_patch_size": getattr(args, "autogaze_target_patch_size", None),
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
    parser.add_argument("--mode", choices=["single", "hlvid", "preflight", "stream-profile"], default="single")
    parser.add_argument("--model-path", "--nvila-model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--video", default=DEFAULT_EXAMPLE_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
    parser.add_argument("--gazing-mode", choices=["autogaze", "keep-all"], default="autogaze")
    parser.add_argument("--autogaze-target-scales", "--autogaze-resize-scales", dest="autogaze_target_scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.6)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--warmup-runs", type=non_negative_int, default=0)
    parser.add_argument("--repeat-runs", type=positive_int, default=1)
    parser.add_argument("--stream-chunk-frames", type=int, default=AUTOGAZE_CHUNK_FRAMES)
    parser.add_argument("--stream-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--stream-decode-strategy", choices=["scan", "seek"], default="scan")
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
    return build_parser(defaults).parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "stream-profile":
        run_stream_profile(args)
    elif args.mode == "single":
        run_single(args)
    else:
        run_hlvid(args)


if __name__ == "__main__":
    main()
