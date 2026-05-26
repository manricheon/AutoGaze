from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from repro.common import write_csv, write_json
from repro.markdown_report import (
    as_mapping,
    detect_report_kind,
    enriched_key_metrics,
    first_present,
    get_budget_value,
    get_path,
    key_metrics,
    processing_budget_summary,
)
from repro.report_charts import ChartBar, ChartSegment, nonnegative_difference, numeric_or_none, shorten_label, write_bar_chart

HLVID_MODE_ORDER = ("keep_all", "single_scale_dense", "autogaze")


ROW_FIELDS = [
    "source_path",
    "report_kind",
    "mode",
    "model_path",
    "status",
    "oom",
    "oom_stage",
    "failure_kind",
    "failure_message",
    "total_ms",
    "preprocess_ms",
    "video_decode_read_ms",
    "video_prepare_total_ms",
    "video_frame_resize_ms",
    "video_tiling_ms",
    "selector_input_build_ms",
    "preprocess_rest_ms",
    "autogaze_ms",
    "vision_encoder_ms",
    "mm_projector_ms",
    "generate_ms",
    "llm_generation_ms",
    "llm_forward_ms",
    "generation_rest_ms",
    "llm_ms",
    "single_scale_dense_patch_tokens",
    "full_or_raw_patch_tokens",
    "autogaze_selected_patch_tokens",
    "llm_visual_tokens",
    "token_reduction_ratio",
    "peak_memory_bytes",
    "accuracy_total",
    "accuracy_scored",
    "failed",
    "parse_failed",
    "frames",
    "thumbnail_frames",
    "source_resolution",
    "processor_input_resolution",
    "max_tiles_video",
    "gazing_mode",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize_report_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    payload = load_json(source)
    if "modes" in payload and isinstance(payload.get("modes"), dict):
        return [_normalize_plugin_mode(source, mode, summary) for mode, summary in payload["modes"].items()]
    if "readable_summary" in payload and any(mode in payload for mode in HLVID_MODE_ORDER):
        return [_normalize_hlvid_mode(source, payload, mode) for mode in HLVID_MODE_ORDER if mode in payload]
    if isinstance(payload.get("rows"), list):
        return [_normalize_legacy_summary_row(source, row) for row in payload["rows"] if isinstance(row, dict)]
    return [_normalize_single(source, payload)]


def _blank_row(path: Path, *, report_kind: str, mode: str | None, model_path: str | None) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "report_kind": report_kind,
        "mode": mode,
        "model_path": model_path,
        "status": None,
        "oom": False,
        "oom_stage": None,
        "failure_kind": None,
        "failure_message": None,
        "total_ms": None,
        "preprocess_ms": None,
        "video_decode_read_ms": None,
        "video_prepare_total_ms": None,
        "video_frame_resize_ms": None,
        "video_tiling_ms": None,
        "selector_input_build_ms": None,
        "preprocess_rest_ms": None,
        "autogaze_ms": None,
        "vision_encoder_ms": None,
        "mm_projector_ms": None,
        "generate_ms": None,
        "llm_generation_ms": None,
        "llm_forward_ms": None,
        "generation_rest_ms": None,
        "llm_ms": None,
        "single_scale_dense_patch_tokens": None,
        "full_or_raw_patch_tokens": None,
        "autogaze_selected_patch_tokens": None,
        "llm_visual_tokens": None,
        "token_reduction_ratio": None,
        "peak_memory_bytes": None,
        "accuracy_total": None,
        "accuracy_scored": None,
        "failed": None,
        "parse_failed": None,
        "frames": None,
        "thumbnail_frames": None,
        "source_resolution": None,
        "processor_input_resolution": None,
        "max_tiles_video": None,
        "gazing_mode": None,
    }


def _sum_present(*values: Any) -> float | None:
    numbers = [numeric_or_none(value) for value in values]
    present = [value for value in numbers if value is not None]
    return sum(present) if present else None


def _normalize_single(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = enriched_key_metrics(payload, key_metrics(payload))
    report_kind = detect_report_kind(payload)
    if report_kind == "generic" and payload.get("mode") == "single":
        report_kind = "single_inference"
    row = _blank_row(
        path,
        report_kind=report_kind,
        mode=str(
            first_present(
                payload.get("gazing_mode"),
                get_path(payload, "generation.metrics.qwen_vit.mode"),
                get_path(payload, "experiment_spec.qwen_vit_mode"),
                payload.get("mode"),
                "single",
            )
        ),
        model_path=first_present(
            payload.get("model_path"),
            get_path(payload, "result.model_path"),
            get_path(payload, "experiment_spec.model_path"),
        ),
    )
    failure = as_mapping(payload.get("failure") or get_path(payload, "generation.failure"))
    status = first_present(
        failure.get("kind"),
        payload.get("status"),
        get_path(payload, "generation.status"),
        payload.get("implementation_status"),
        "ok",
    )
    _apply_failure(row, failure, status)
    _apply_metrics(row, metrics)
    _apply_video_budget(row, processing_budget_summary(payload))
    _apply_legacy_payload(row, payload)
    row["gazing_mode"] = payload.get("gazing_mode")
    return row


def _normalize_hlvid_mode(path: Path, payload: dict[str, Any], mode: str) -> dict[str, Any]:
    row = _blank_row(path, report_kind="hlvid_benchmark", mode=mode, model_path=payload.get("model_path"))
    mode_status = get_path(payload, f"readable_summary.mode_status.{mode}")
    row["status"] = "skipped" if mode_status == "skipped_or_missing" else "ok"
    accuracy = as_mapping(get_path(payload, f"{mode}.accuracy", {}))
    row["accuracy_total"] = numeric_or_none(accuracy.get("accuracy_total"))
    row["accuracy_scored"] = numeric_or_none(accuracy.get("accuracy_scored"))
    row["failed"] = numeric_or_none(accuracy.get("failed"))
    row["parse_failed"] = numeric_or_none(accuracy.get("parse_failed"))
    metrics = enriched_key_metrics(payload, key_metrics(payload))
    _apply_metrics(row, metrics, mode=mode)
    _apply_flat_budget(row, _mode_budget(payload, mode))
    return row


def _normalize_plugin_mode(path: Path, mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = _blank_row(path, report_kind="plugin_hlvid_summary", mode=mode, model_path=None)
    status_counts = as_mapping(summary.get("status_counts"))
    row["status"] = "oom" if status_counts.get("oom") else "ok"
    row["oom"] = bool(status_counts.get("oom"))
    row["accuracy_total"] = numeric_or_none(summary.get("accuracy_total"))
    row["accuracy_scored"] = numeric_or_none(summary.get("accuracy_scored"))
    row["failed"] = numeric_or_none(summary.get("failed"))
    row["parse_failed"] = numeric_or_none(summary.get("parse_failed"))
    budget = as_mapping(get_path(summary, "processing_budget_summary.mode_median"))
    _apply_flat_budget(row, budget)
    _zero_missing_accuracy(row)
    return row


def _normalize_legacy_summary_row(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    row = _blank_row(
        path,
        report_kind="legacy_summary_rows",
        mode=str(summary.get("mode") or summary.get("gazing_mode") or summary.get("name") or "summary_row"),
        model_path=summary.get("model_path"),
    )
    row["status"] = str(summary.get("status") or "ok")
    row["frames"] = numeric_or_none(summary.get("frames"))
    row["thumbnail_frames"] = numeric_or_none(summary.get("thumbnail_frames"))
    row["processor_input_resolution"] = _resolution(summary.get("effective_width"), summary.get("effective_height"))
    row["max_tiles_video"] = numeric_or_none(first_present(summary.get("tile_sequences"), summary.get("max_tiles_video")))
    row["video_decode_read_ms"] = numeric_or_none(summary.get("video_decode_read_ms"))
    row["selector_input_build_ms"] = numeric_or_none(summary.get("autogaze_tensorize_ms"))
    row["autogaze_ms"] = numeric_or_none(summary.get("autogaze_forward_ms"))
    row["vision_encoder_ms"] = numeric_or_none(
        first_present(summary.get("siglip_gazed_forward_ms"), summary.get("siglip_keep_all_forward_ms"))
    )
    row["total_ms"] = numeric_or_none(
        first_present(summary.get("estimated_autogaze_stream_ms"), summary.get("estimated_keep_all_stream_ms"))
    )
    row["token_reduction_ratio"] = numeric_or_none(
        first_present(
            summary.get("encoder_total_token_reduction_ratio"),
            summary.get("encoder_tile_token_reduction_ratio"),
            summary.get("llm_visual_token_lower_bound_reduction_ratio"),
        )
    )
    row["llm_visual_tokens"] = numeric_or_none(
        first_present(
            summary.get("llm_autogaze_visual_tokens_lower_bound_estimated"),
            summary.get("llm_keep_all_visual_tokens_estimated"),
        )
    )
    row["peak_memory_bytes"] = numeric_or_none(
        first_present(
            summary.get("raw_frame_buffer_peak_bytes"),
            summary.get("siglip_keep_all_hidden_peak_bytes"),
            summary.get("siglip_gazed_hidden_peak_bytes"),
        )
    )
    return row


def _apply_failure(row: dict[str, Any], failure: dict[str, Any], status: Any) -> None:
    row["status"] = status
    if failure:
        row["failure_kind"] = failure.get("kind")
        row["failure_message"] = failure.get("message")
        row["oom_stage"] = failure.get("stage") if failure.get("kind") == "oom" else None
        row["oom"] = failure.get("kind") == "oom"
    elif status == "oom":
        row["oom"] = True


def _apply_metrics(row: dict[str, Any], metrics: dict[str, Any], *, mode: str | None = None) -> None:
    latency = as_mapping(metrics.get("latency_ms"))
    tokens = as_mapping(metrics.get("tokens"))
    memory = as_mapping(metrics.get("memory_bytes"))
    row["total_ms"] = _metric(latency, ("total_ms", "total", "total_median"), mode)
    row["preprocess_ms"] = _metric(
        latency,
        ("preprocess_without_autogaze_ms", "preprocess_without_autogaze_median"),
        mode,
    )
    row["video_decode_read_ms"] = _metric(
        latency,
        ("video_decode_read_ms", "video_decode_read_median", "video_decode_ms", "video_decode_median"),
        mode,
    )
    row["video_prepare_total_ms"] = _metric(latency, ("video_prepare_total_ms",), mode)
    row["video_frame_resize_ms"] = _metric(latency, ("video_frame_resize_ms",), mode)
    row["video_tiling_ms"] = _metric(latency, ("video_tiling_ms", "tile_tensor_prep_ms"), mode)
    row["selector_input_build_ms"] = _metric(
        latency,
        ("selector_input_build_ms", "selector_input_ms", "autogaze_tensorize_ms", "tile_autogaze_tensorize_ms"),
        mode,
    )
    row["preprocess_rest_ms"] = _metric(
        latency,
        ("preprocess_rest_without_decode_autogaze_ms", "preprocess_rest_without_decode_autogaze_median"),
        mode,
    )
    if row["preprocess_rest_ms"] is None:
        row["preprocess_rest_ms"] = nonnegative_difference(row["preprocess_ms"], row["video_decode_read_ms"])
    row["autogaze_ms"] = _metric(
        latency,
        ("autogaze_total_ms", "autogaze_ms", "autogaze_total_median", "autogaze_median", "gazing_info_total_ms"),
        mode,
    )
    row["vision_encoder_ms"] = _metric(
        latency,
        (
            "vit_encoder_ms",
            "vision_encoder_ms",
            "siglip_vision_ms",
            "vit_encoder_median",
            "qwen_vit_prepare",
            "qwen_vit_prepare_ms",
        ),
        mode,
    )
    row["mm_projector_ms"] = _metric(latency, ("mm_projector_ms", "projector_ms"), mode)
    row["generate_ms"] = _metric(latency, ("generate_ms", "generate_median", "generate"), mode)
    row["llm_forward_ms"] = _metric(latency, ("llm_ms", "llm_median", "llm_forward_ms"), mode)
    row["generation_rest_ms"] = _metric(latency, ("generation_rest_ms", "generate_rest_ms"), mode)
    if row["generation_rest_ms"] is None:
        vision_parent = _metric(latency, ("vision_encoder_ms", "vision_encoder_median"), mode)
        vision_for_generate = vision_parent
        if vision_for_generate is None:
            vision_for_generate = _sum_present(
                _metric(latency, ("vision_input_build_ms", "vision_input_build_median"), mode),
                row["vision_encoder_ms"],
                row["mm_projector_ms"],
            )
        child_total = _sum_present(vision_for_generate, row["llm_forward_ms"])
        row["generation_rest_ms"] = nonnegative_difference(row["generate_ms"], child_total)
    row["llm_generation_ms"] = _metric(latency, ("llm_generation_ms", "llm_generation_median"), mode)
    if row["llm_generation_ms"] is None:
        row["llm_generation_ms"] = _sum_present(row["llm_forward_ms"], row["generation_rest_ms"])
    row["llm_ms"] = row["llm_forward_ms"]
    row["single_scale_dense_patch_tokens"] = _metric(
        tokens,
        ("single_scale_dense_siglip_reference_patch_tokens",),
        mode,
    )
    row["full_or_raw_patch_tokens"] = _metric(
        tokens,
        (
            "hd_multiscale_keep_all_patch_tokens",
            "vit_encoder_input_patch_tokens_before_autogaze",
            "single_scale_dense_siglip_reference_patch_tokens",
            "raw_vit_patch_tokens_before_selector",
            "encoder_patch_tokens_before_keep_all_or_raw",
            "visual_tokens_before_prune",
        ),
        mode,
    )
    row["autogaze_selected_patch_tokens"] = _metric(
        tokens,
        (
            "autogaze_selected_total_patch_tokens",
            "vit_encoder_input_patch_tokens_after_autogaze",
            "encoder_input_patch_tokens_after_autogaze",
            "encoder_patch_tokens_after_autogaze",
            "visual_tokens_after_prune",
        ),
        mode,
    )
    row["llm_visual_tokens"] = _metric(
        tokens,
        ("llm_visual_tokens_actual_from_budget", "llm_visual_tokens_after_actual", "llm_context_tokens"),
        mode,
    )
    row["token_reduction_ratio"] = _metric(
        tokens,
        (
            "patch_reduction_ratio_full_or_raw_over_autogaze",
            "encoder_token_reduction_ratio",
            "visual_token_reduction_ratio",
            "llm_visual_token_reduction_ratio_from_budget",
        ),
        mode,
    )
    row["peak_memory_bytes"] = _metric(
        memory,
        ("overall_peak", "overall_peak_median", "peak_cuda_allocated", "peak_cuda_reserved", "llm_peak", "llm_peak_median"),
        mode,
    )


def _metric(group: dict[str, Any], keys: tuple[str, ...], mode: str | None = None) -> float | None:
    for key in keys:
        value = group.get(key)
        if mode is not None and isinstance(value, dict):
            direct = numeric_or_none(value.get(mode))
            if direct is not None:
                return direct
        number = _number_from_value(value, mode=mode)
        if number is not None:
            return number
    return None


def _number_from_value(value: Any, *, mode: str | None = None) -> float | None:
    if isinstance(value, dict):
        ordered_keys = []
        if mode is not None:
            ordered_keys.append(mode)
        ordered_keys.extend(
            [
                "median",
                "value",
                "after_autogaze_actual",
                "after_autogaze",
                "autogaze",
                "before_keep_all_estimated",
                "before_keep_all_or_raw",
                "keep_all",
            ]
        )
        for key in ordered_keys:
            number = numeric_or_none(value.get(key))
            if number is not None:
                return number
        return None
    return numeric_or_none(value)


def _apply_video_budget(row: dict[str, Any], budget: dict[str, Any]) -> None:
    if not budget:
        return
    video = as_mapping(budget.get("video"))
    tiling = as_mapping(budget.get("tiling"))
    thumbnail = as_mapping(budget.get("thumbnail"))
    _set_if_missing(
        row,
        "frames",
        numeric_or_none(first_present(video.get("actual_video_frames"), video.get("requested_video_frames"))),
    )
    _set_if_missing(
        row,
        "thumbnail_frames",
        numeric_or_none(
            first_present(
                thumbnail.get("actual_frames"),
                thumbnail.get("effective_frames"),
                thumbnail.get("requested_frames"),
            )
        ),
    )
    _set_if_missing(row, "source_resolution", video.get("source_resolution"))
    _set_if_missing(row, "processor_input_resolution", video.get("processor_input_resolution"))
    _set_if_missing(
        row,
        "max_tiles_video",
        numeric_or_none(first_present(tiling.get("spatial_tiles_per_frame"), tiling.get("max_tiles_video"))),
    )


def _apply_flat_budget(row: dict[str, Any], budget: dict[str, Any]) -> None:
    if not budget:
        return
    _set_if_missing(
        row,
        "frames",
        numeric_or_none(first_present(budget.get("video.actual_video_frames"), budget.get("video.requested_video_frames"))),
    )
    _set_if_missing(
        row,
        "thumbnail_frames",
        numeric_or_none(
            first_present(
                budget.get("thumbnail.actual_frames"),
                budget.get("thumbnail.effective_frames"),
                budget.get("thumbnail.requested_frames"),
            )
        ),
    )
    _set_if_missing(row, "source_resolution", budget.get("video.source_resolution"))
    _set_if_missing(row, "processor_input_resolution", budget.get("video.processor_input_resolution"))
    _set_if_missing(
        row,
        "max_tiles_video",
        numeric_or_none(first_present(budget.get("tiling.spatial_tiles_per_frame"), budget.get("tiling.max_tiles_video"))),
    )
    _set_if_missing(
        row,
        "single_scale_dense_patch_tokens",
        numeric_or_none(
        first_present(
            budget.get("single_scale_dense_vision_budget.total_patch_tokens"),
            budget.get("single_scale_dense_vision_budget.estimated_total_patch_tokens"),
        )
        ),
    )
    _set_if_missing(
        row,
        "full_or_raw_patch_tokens",
        numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.keep_all_total_patch_tokens"),
            budget.get("patch_budget_before_siglip.keep_all_tile_patch_tokens"),
            budget.get("patch_budget_before_vit.actual_raw_patch_tokens_before_vit"),
            budget.get("patch_budget_before_vit.estimated_visual_tokens_before_prune"),
        )
        ),
    )
    _set_if_missing(
        row,
        "autogaze_selected_patch_tokens",
        numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.autogaze_selected_total_patch_tokens"),
            budget.get("patch_budget_before_siglip.autogaze_selected_tile_patch_tokens"),
            budget.get("patch_budget_before_vit.estimated_visual_tokens_after_prune"),
        )
        ),
    )
    _set_if_missing(
        row,
        "token_reduction_ratio",
        numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.total_patch_reduction_ratio"),
            budget.get("patch_budget_before_vit.estimated_visual_token_reduction_ratio"),
        )
        ),
    )
    _set_if_missing(row, "llm_visual_tokens", numeric_or_none(budget.get("llm_visual_budget.actual_visual_tokens")))


def _set_if_missing(row: dict[str, Any], field: str, value: Any) -> None:
    if row.get(field) in {None, ""} and value not in {None, ""}:
        row[field] = value


def _mode_budget(payload: dict[str, Any], mode: str | None) -> dict[str, Any]:
    if mode:
        for path in (
            f"readable_summary.processing_budget_summary.{mode}_median",
            f"readable_summary.processing_budget_summary.mode_median.{mode}",
            f"readable_summary.processing_budget_summary.comparison.{mode}",
        ):
            budget = as_mapping(get_path(payload, path))
            if budget:
                return budget
    return _flatten_budget(as_mapping(processing_budget_summary(payload)))


def _flatten_budget(budget: dict[str, Any]) -> dict[str, Any]:
    if not budget:
        return {}
    if any("." in key for key in budget):
        return budget
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        else:
            flat[prefix] = value

    walk("", budget)
    return flat


def _apply_legacy_payload(row: dict[str, Any], payload: dict[str, Any]) -> None:
    _apply_video_budget(row, as_mapping(get_path(payload, "generation.metrics.processing_budget_summary")))
    _apply_video_budget(row, as_mapping(payload.get("processing_budget_summary")))
    _apply_stream_profile_legacy(row, payload)
    _apply_generation_metrics_legacy(row, as_mapping(get_path(payload, "generation.metrics")))


def _apply_generation_metrics_legacy(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    _apply_metrics(row, metrics)
    _apply_video_budget(row, as_mapping(metrics.get("processing_budget_summary")))


def _apply_stream_profile_legacy(row: dict[str, Any], payload: dict[str, Any]) -> None:
    source = as_mapping(payload.get("source_metadata"))
    sampling = as_mapping(payload.get("sampling"))
    effective = as_mapping(payload.get("effective_video"))
    timings = as_mapping(payload.get("timing_ms"))
    tokens = as_mapping(payload.get("token_metrics"))
    memory = as_mapping(payload.get("memory_bytes"))
    if not any((source, sampling, effective, timings, tokens, memory)):
        return

    _set_if_missing(row, "source_resolution", _resolution(source.get("width"), source.get("height")))
    _set_if_missing(row, "processor_input_resolution", _resolution(effective.get("width"), effective.get("height")))
    _set_if_missing(row, "frames", numeric_or_none(first_present(sampling.get("num_video_frames"), sampling.get("decoded_selected_frames"))))
    _set_if_missing(row, "thumbnail_frames", numeric_or_none(sampling.get("num_video_frames_thumbnail")))
    _set_if_missing(row, "video_decode_read_ms", _sum_present(timings.get("video_decode_seek"), timings.get("video_decode_scan")))
    _set_if_missing(row, "video_frame_resize_ms", numeric_or_none(timings.get("video_frame_resize")))
    _set_if_missing(row, "video_tiling_ms", numeric_or_none(timings.get("spatial_tile_build")))
    _set_if_missing(row, "selector_input_build_ms", numeric_or_none(timings.get("tile_autogaze_tensorize")))
    _set_if_missing(row, "autogaze_ms", numeric_or_none(timings.get("tile_autogaze_forward")))
    _set_if_missing(row, "total_ms", numeric_or_none(timings.get("pre_llm_stream_total_measured")))
    _set_if_missing(row, "vision_encoder_ms", _stream_vision_ms(row, timings))
    _set_if_missing(
        row,
        "full_or_raw_patch_tokens",
        numeric_or_none(
            first_present(
                tokens.get("encoder_raw_patch_tokens"),
                tokens.get("encoder_raw_tile_patch_tokens"),
                tokens.get("encoder_input_patch_tokens_before_autogaze"),
            )
        ),
    )
    _set_if_missing(
        row,
        "autogaze_selected_patch_tokens",
        numeric_or_none(
            first_present(
                tokens.get("encoder_autogaze_selected_total_patch_tokens"),
                tokens.get("encoder_autogaze_selected_patch_tokens"),
                tokens.get("encoder_autogaze_selected_tile_patch_tokens"),
            )
        ),
    )
    if row.get("token_reduction_ratio") is None:
        before = numeric_or_none(row.get("full_or_raw_patch_tokens"))
        after = numeric_or_none(row.get("autogaze_selected_patch_tokens"))
        if before is not None and after not in {None, 0.0}:
            row["token_reduction_ratio"] = before / after
    _set_if_missing(row, "llm_visual_tokens", numeric_or_none(tokens.get("llm_actual_visual_tokens")))
    _set_if_missing(row, "peak_memory_bytes", _max_numeric(memory.values()))
    _set_if_missing(row, "max_tiles_video", _legacy_tiles(tokens))


def _stream_vision_ms(row: dict[str, Any], timings: dict[str, Any]) -> float | None:
    mode = str(row.get("mode") or row.get("gazing_mode") or "").lower()
    if "autogaze" in mode or "gazed" in mode:
        return numeric_or_none(timings.get("siglip_gazed_forward"))
    if "keep" in mode:
        return numeric_or_none(timings.get("siglip_keep_all_forward"))
    return numeric_or_none(first_present(timings.get("siglip_gazed_forward"), timings.get("siglip_keep_all_forward")))


def _legacy_tiles(tokens: dict[str, Any]) -> float | None:
    tiles = tokens.get("spatial_tiles_per_video")
    if isinstance(tiles, list) and tiles:
        return numeric_or_none(tiles[0])
    return numeric_or_none(first_present(tiles, tokens.get("spatial_tiles_per_frame")))


def _resolution(width: Any, height: Any) -> str | None:
    w = numeric_or_none(width)
    h = numeric_or_none(height)
    if w is None or h is None:
        return None
    return f"{int(w)}x{int(h)}"


def _max_numeric(values: Any) -> float | None:
    numbers = [numeric_or_none(value) for value in values]
    present = [value for value in numbers if value is not None]
    return max(present) if present else None


def _zero_missing_accuracy(row: dict[str, Any]) -> None:
    if row.get("accuracy_total") is None:
        row["accuracy_total"] = 0.0
    if row.get("accuracy_scored") is None:
        row["accuracy_scored"] = numeric_or_none(row.get("accuracy_total")) or 0.0


def aggregate_report_roots(input_roots: list[str | Path], output_dir: str | Path, *, sort: str = "comparison") -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for root in input_roots:
        for path in sorted(Path(root).rglob("*.json")):
            if output in path.parents:
                continue
            try:
                rows.extend(normalize_report_file(path))
            except (json.JSONDecodeError, OSError):
                continue
    output_rows = []
    for row in rows:
        compact = {field: row.get(field) for field in ROW_FIELDS}
        _zero_missing_accuracy(compact)
        _apply_derived_token_budgets(compact)
        output_rows.append(compact)
    rows = sort_rows(output_rows, sort=sort)
    csv_path = output / "aggregate_rows.csv"
    json_path = output / "aggregate_summary.json"
    md_path = output / "aggregate_report.md"
    assets = output / "assets"
    charts = _write_trend_charts(rows, assets)
    write_csv(csv_path, rows)
    write_json(json_path, {"row_count": len(rows), "rows": rows, "charts": {name: str(path) for name, path in charts.items()}})
    md_path.write_text(_render_markdown(rows, charts, output), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path, **charts}


def sort_rows(rows: list[dict[str, Any]], *, sort: str = "comparison") -> list[dict[str, Any]]:
    def number(row: dict[str, Any], field: str, default: float = float("inf")) -> float:
        value = numeric_or_none(row.get(field))
        return default if value is None else value

    if sort == "latency":
        return sorted(rows, key=lambda row: (_status_rank(row), number(row, "total_ms"), _comparison_key(row)))
    if sort == "token-reduction":
        return sorted(rows, key=lambda row: (_status_rank(row), -number(row, "token_reduction_ratio", 0.0), _comparison_key(row)))
    if sort == "memory":
        return sorted(rows, key=lambda row: (_status_rank(row), number(row, "peak_memory_bytes"), _comparison_key(row)))
    if sort == "accuracy":
        return sorted(rows, key=lambda row: (_status_rank(row), -number(row, "accuracy_scored", 0.0), _comparison_key(row)))
    if sort == "base-tokens":
        return sorted(rows, key=lambda row: (_status_rank(row), number(row, "base_token_budget"), _selector_rank(row), _comparison_key(row)))
    if sort == "actual-tokens":
        return sorted(rows, key=lambda row: (_status_rank(row), number(row, "actual_processed_tokens"), _selector_rank(row), _comparison_key(row)))
    if sort == "frames":
        return sorted(rows, key=lambda row: (_status_rank(row), number(row, "frames"), _selector_rank(row), _comparison_key(row)))
    if sort == "resolution":
        return sorted(rows, key=lambda row: (_status_rank(row), _resolution_pixels(row), number(row, "frames"), _selector_rank(row), _comparison_key(row)))
    if sort == "config":
        return sorted(rows, key=lambda row: (_status_rank(row), _config_group_key(row), _selector_rank(row), _comparison_key(row)))
    if sort == "status":
        return sorted(rows, key=lambda row: (_status_rank(row), str(row.get("status") or ""), _comparison_key(row)))
    return sorted(rows, key=lambda row: (_status_rank(row), _config_group_key(row), _selector_rank(row), _comparison_key(row)))


def _status_rank(row: dict[str, Any]) -> int:
    status = str(row.get("status") or "").lower()
    if row.get("oom") or status == "oom":
        return 9
    if status.startswith("failed"):
        return 8
    if "probe" in status or "sidecar" in status:
        return 6
    return 0


def _selector_rank(row: dict[str, Any]) -> int:
    mode = str(row.get("mode") or "").lower().replace("_", "-")
    gazing_mode = str(row.get("gazing_mode") or "").lower()
    if mode in {"keep-all", "keepall", "off", "baseline"} or "keep-all" in mode or gazing_mode == "keep-all":
        return 0
    if mode == "autogaze" or "autogaze" in mode or "sparse" in mode or "gazed" in mode:
        return 2
    if _is_dense_or_off_baseline(mode):
        return 1
    if "probe" in mode or "sidecar" in mode:
        return 4
    return 3


def _config_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model_path") or ""),
        numeric_or_none(row.get("frames")) if numeric_or_none(row.get("frames")) is not None else float("inf"),
        numeric_or_none(row.get("thumbnail_frames"))
        if numeric_or_none(row.get("thumbnail_frames")) is not None
        else float("inf"),
        _resolution_pixels(row),
        str(row.get("processor_input_resolution") or ""),
        numeric_or_none(row.get("max_tiles_video"))
        if numeric_or_none(row.get("max_tiles_video")) is not None
        else float("inf"),
    )


def _comparison_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("mode") or ""), str(row.get("source_path") or ""))


def _apply_derived_token_budgets(row: dict[str, Any]) -> None:
    row["base_token_budget"] = _base_token_budget(row)
    row["actual_processed_tokens"] = _actual_processed_tokens(row)


def _base_token_budget(row: dict[str, Any]) -> float | None:
    return numeric_or_none(
        first_present(
            row.get("full_or_raw_patch_tokens"),
            row.get("single_scale_dense_patch_tokens"),
            row.get("llm_visual_tokens"),
            row.get("autogaze_selected_patch_tokens"),
        )
    )


def _actual_processed_tokens(row: dict[str, Any]) -> float | None:
    mode_family = _selector_family(row)
    if mode_family == "autogaze":
        return numeric_or_none(
            first_present(
                row.get("autogaze_selected_patch_tokens"),
                row.get("llm_visual_tokens"),
                row.get("full_or_raw_patch_tokens"),
            )
        )
    if mode_family == "single":
        return numeric_or_none(
            first_present(
                row.get("single_scale_dense_patch_tokens"),
                row.get("full_or_raw_patch_tokens"),
                row.get("llm_visual_tokens"),
            )
        )
    return numeric_or_none(
        first_present(
            row.get("full_or_raw_patch_tokens"),
            row.get("single_scale_dense_patch_tokens"),
            row.get("llm_visual_tokens"),
            row.get("autogaze_selected_patch_tokens"),
        )
    )


def _selector_family(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "").lower().replace("_", "-")
    gazing_mode = str(row.get("gazing_mode") or "").lower().replace("_", "-")
    text = f"{mode} {gazing_mode}"
    if "autogaze" in text or "gazed" in text or "sparse" in text:
        return "autogaze"
    if _is_dense_or_off_baseline(mode):
        return "single"
    if "keep-all" in text or "keepall" in text or mode in {"off", "baseline"}:
        return "keep_all"
    return "other"


def _is_dense_or_off_baseline(mode: str) -> bool:
    return (
        mode in {"single-scale-dense", "keep-all-single", "single-scale", "off", "baseline"}
        or "single-scale" in mode
        or "dense" in mode
        or mode.endswith("-off")
        or mode.endswith("-full-vit")
        or mode.endswith("-chunked-vit")
        or "full-vit" in mode
        or "chunked-vit" in mode
    )


def _resolution_pixels(row: dict[str, Any]) -> float:
    width, height = _resolution_parts(row.get("processor_input_resolution"))
    if width is None or height is None:
        return float("inf")
    return float(width * height)


def _write_trend_charts(rows: list[dict[str, Any]], assets: Path) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    labels = [_row_label(row, index) for index, row in enumerate(rows)]
    latency_bars: list[ChartBar] = []
    attribution_bars: list[ChartBar] = []
    for label, row in zip(labels, rows):
        segments = [
            ChartSegment("Decode/read", value)
            for value in [numeric_or_none(row.get("video_decode_read_ms"))]
            if value is not None and value > 0
        ]
        frame_resize = numeric_or_none(row.get("video_frame_resize_ms"))
        if frame_resize is not None and frame_resize > 0:
            segments.append(ChartSegment("Frame resize", frame_resize))
        tile_tensor = numeric_or_none(row.get("video_tiling_ms"))
        if tile_tensor is not None and tile_tensor > 0:
            segments.append(ChartSegment("Tile/tensor prep", tile_tensor))
        prep_rest = numeric_or_none(row.get("preprocess_rest_ms"))
        if prep_rest is not None and prep_rest > 0 and frame_resize is None and tile_tensor is None:
            segments.append(ChartSegment("Prep rest", prep_rest))
        selector_input = numeric_or_none(row.get("selector_input_build_ms"))
        if selector_input is not None and selector_input > 0:
            segments.append(ChartSegment("Selector input", selector_input))
        autogaze = numeric_or_none(row.get("autogaze_ms"))
        if autogaze is not None and autogaze > 0:
            segments.append(ChartSegment("AutoGaze", autogaze))
        vision = numeric_or_none(row.get("vision_encoder_ms"))
        if vision is not None and vision > 0:
            segments.append(ChartSegment("ViT", vision))
        projector = numeric_or_none(row.get("mm_projector_ms"))
        if projector is not None and projector > 0:
            segments.append(ChartSegment("Projector", projector))
        llm = numeric_or_none(row.get("llm_forward_ms") or row.get("llm_ms"))
        if llm is not None and llm > 0:
            segments.append(ChartSegment("LLM forward", llm))
        generate_rest = numeric_or_none(row.get("generation_rest_ms"))
        if generate_rest is not None and generate_rest > 0:
            segments.append(ChartSegment("Generate rest", generate_rest))
        total = numeric_or_none(row.get("total_ms"))
        known = sum(segment.value for segment in segments)
        if total is not None and total > known:
            segments.append(ChartSegment("Other", total - known))
        if not segments and total is not None:
            segments.append(ChartSegment("total_ms", total))
        if segments:
            latency_bars.append(ChartBar(label, segments))

        attribution_segments: list[ChartSegment] = []
        decode = numeric_or_none(row.get("video_decode_read_ms"))
        if decode is not None and decode > 0:
            attribution_segments.append(ChartSegment("Video I/O", decode))
        pre_model_parts = [value for value in (frame_resize, tile_tensor) if value is not None and value > 0]
        if not pre_model_parts and prep_rest is not None and prep_rest > 0:
            pre_model_parts = [prep_rest]
        if pre_model_parts:
            attribution_segments.append(ChartSegment("Pre-model prep", sum(pre_model_parts)))
        autogaze_parts = [value for value in (selector_input, autogaze) if value is not None and value > 0]
        if autogaze_parts:
            attribution_segments.append(ChartSegment("AutoGaze pipeline", sum(autogaze_parts)))
        vision_parts = [value for value in (vision, projector) if value is not None and value > 0]
        if vision_parts:
            attribution_segments.append(ChartSegment("Vision pipeline", sum(vision_parts)))
        llm_generation = numeric_or_none(row.get("llm_generation_ms"))
        if llm_generation is None:
            llm_generation = _sum_present(llm, generate_rest)
        if llm_generation is not None and llm_generation > 0:
            attribution_segments.append(ChartSegment("LLM generation", llm_generation))
        known_attribution = sum(segment.value for segment in attribution_segments)
        if total is not None and total > known_attribution:
            attribution_segments.append(ChartSegment("Other", total - known_attribution))
        if not attribution_segments and total is not None:
            attribution_segments.append(ChartSegment("total_ms", total))
        if attribution_segments:
            attribution_bars.append(ChartBar(label, attribution_segments))
    charts["latency"] = write_bar_chart(
        assets / "latency_by_config.svg",
        title="Latency By Config",
        bars=latency_bars,
        unit="ms",
    ).path
    charts["latency_attribution"] = write_bar_chart(
        assets / "latency_attribution_by_config.svg",
        title="Latency Attribution By Config",
        bars=attribution_bars,
        unit="ms",
    ).path
    accuracy_bars = [
        ChartBar(label, [ChartSegment("accuracy", _accuracy_value(row))])
        for label, row in zip(labels, rows)
    ]
    charts["accuracy"] = write_bar_chart(
        assets / "accuracy_by_config.svg",
        title="Accuracy By Config",
        bars=accuracy_bars,
        unit="acc",
    ).path
    if len({row.get("frames") for row in rows if row.get("frames") is not None}) >= 2:
        frame_rows = sorted(rows, key=lambda row: (numeric_or_none(row.get("frames")) or float("inf"), _selector_rank(row)))
        charts["accuracy_vs_frames"] = write_bar_chart(
            assets / "accuracy_vs_frames.svg",
            title="Accuracy Vs Frames",
            bars=[ChartBar(_row_label(row, index), [ChartSegment("accuracy", _accuracy_value(row))]) for index, row in enumerate(frame_rows)],
            unit="acc",
        ).path
    if len({row.get("processor_input_resolution") for row in rows if row.get("processor_input_resolution")}) >= 2:
        resolution_rows = sorted(rows, key=lambda row: (_resolution_pixels(row), numeric_or_none(row.get("frames")) or float("inf"), _selector_rank(row)))
        charts["accuracy_vs_input_resolution"] = write_bar_chart(
            assets / "accuracy_vs_input_resolution.svg",
            title="Accuracy Vs Input Resolution",
            bars=[ChartBar(_row_label(row, index), [ChartSegment("accuracy", _accuracy_value(row))]) for index, row in enumerate(resolution_rows)],
            unit="acc",
        ).path
    charts["accuracy_vs_base_tokens"] = _write_token_accuracy_scatter(
        assets / "accuracy_vs_base_tokens.svg",
        title="Accuracy Vs Base Token Budget",
        rows=rows,
        token_field="base_token_budget",
        scored_only=True,
    )
    charts["accuracy_vs_actual_tokens"] = _write_token_accuracy_scatter(
        assets / "accuracy_vs_actual_processed_tokens.svg",
        title="Accuracy Vs Actual Processed Tokens",
        rows=rows,
        token_field="actual_processed_tokens",
        scored_only=True,
    )
    charts.update(_write_metric_scatter_suite(rows, assets))
    reduction_bars = [
        ChartBar(label, [ChartSegment("token_reduction_ratio", value)])
        for label, row in zip(labels, rows)
        if (value := numeric_or_none(row.get("token_reduction_ratio"))) is not None
    ]
    charts["token_reduction"] = write_bar_chart(
        assets / "token_reduction_by_config.svg",
        title="Token Reduction By Config",
        bars=reduction_bars,
        unit="x",
    ).path
    memory_bars = [
        ChartBar(label, [ChartSegment("peak_memory_bytes", value)])
        for label, row in zip(labels, rows)
        if (value := numeric_or_none(row.get("peak_memory_bytes"))) is not None
    ]
    charts["memory"] = write_bar_chart(
        assets / "memory_peak_by_config.svg",
        title="Memory Peak By Config",
        bars=memory_bars,
        unit="bytes",
    ).path
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    charts["status"] = write_bar_chart(
        assets / "status_by_config.svg",
        title="Status Counts",
        bars=[ChartBar(status, [ChartSegment(status, count)]) for status, count in sorted(status_counts.items())],
        unit="runs",
    ).path
    charts["runnability_vs_base_tokens"] = _write_status_scatter(
        assets / "runnability_vs_base_tokens.svg",
        title="Runnability Vs Base Tokens",
        rows=rows,
        x_field="base_token_budget",
    )
    return charts


def _write_token_accuracy_scatter(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    token_field: str,
    scored_only: bool,
) -> Path:
    return _write_metric_scatter(
        path,
        title=title,
        rows=rows,
        x_field=token_field,
        y_field="accuracy",
        y_label="Accuracy",
        note="Scored runnable rows only. x-axis uses log scale; OOM/unscored runs are shown in runnability charts.",
        scored_only=scored_only,
        runnable_only=True,
        require_resolution=True,
    )


def _write_metric_scatter_suite(rows: list[dict[str, Any]], assets: Path) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    specs = [
        (
            "total_latency_vs_base_tokens",
            assets / "total_latency_vs_base_tokens.svg",
            "Total Latency Vs Base Tokens",
            "base_token_budget",
            "total_ms",
            "Total latency (ms)",
        ),
        (
            "total_latency_vs_actual_tokens",
            assets / "total_latency_vs_actual_processed_tokens.svg",
            "Total Latency Vs Actual Processed Tokens",
            "actual_processed_tokens",
            "total_ms",
            "Total latency (ms)",
        ),
        (
            "autogaze_latency_vs_base_tokens",
            assets / "by_module" / "autogaze_latency_vs_base_tokens.svg",
            "AutoGaze Latency Vs Base Tokens",
            "base_token_budget",
            "autogaze_ms",
            "AutoGaze latency (ms)",
        ),
        (
            "encoder_latency_vs_actual_tokens",
            assets / "by_module" / "encoder_latency_vs_actual_tokens.svg",
            "ViT Encoder Latency Vs Actual Tokens",
            "actual_processed_tokens",
            "vision_encoder_ms",
            "ViT/encoder latency (ms)",
        ),
        (
            "llm_latency_vs_actual_tokens",
            assets / "by_module" / "llm_latency_vs_actual_tokens.svg",
            "MLLM Latency Vs Actual Tokens",
            "actual_processed_tokens",
            "llm_generation_ms",
            "MLLM generation latency (ms)",
        ),
        (
            "memory_vs_actual_tokens",
            assets / "memory_vs_actual_processed_tokens.svg",
            "Peak Memory Vs Actual Processed Tokens",
            "actual_processed_tokens",
            "peak_memory_bytes",
            "Peak memory (bytes)",
        ),
        (
            "token_reduction_vs_base_tokens",
            assets / "token_reduction_vs_base_tokens.svg",
            "Token Reduction Vs Base Tokens",
            "base_token_budget",
            "token_reduction_ratio",
            "Token reduction ratio (x)",
        ),
    ]
    for key, path, title, x_field, y_field, y_label in specs:
        charts[key] = _write_metric_scatter(
            path,
            title=title,
            rows=rows,
            x_field=x_field,
            y_field=y_field,
            y_label=y_label,
            note="Color shows selector mode, shape shows input resolution, marker size shows frame count.",
            scored_only=False,
            runnable_only=True,
            require_resolution=True,
        )

    for family in ("single", "autogaze", "keep_all"):
        family_rows = [row for row in rows if _selector_family(row) == family]
        if not family_rows:
            continue
        family_slug = family.replace("_", "-")
        family_title = _selector_label(family)
        charts[f"{family}_accuracy_vs_base_tokens"] = _write_metric_scatter(
            assets / "by_mode" / f"{family_slug}_accuracy_vs_base_tokens.svg",
            title=f"{family_title} Accuracy Vs Base Tokens",
            rows=family_rows,
            x_field="base_token_budget",
            y_field="accuracy",
            y_label="Accuracy",
            note="Mode-filtered view. Shape shows resolution and marker size shows frame count.",
            scored_only=True,
            runnable_only=True,
            require_resolution=True,
        )
        charts[f"{family}_accuracy_vs_actual_tokens"] = _write_metric_scatter(
            assets / "by_mode" / f"{family_slug}_accuracy_vs_actual_processed_tokens.svg",
            title=f"{family_title} Accuracy Vs Actual Processed Tokens",
            rows=family_rows,
            x_field="actual_processed_tokens",
            y_field="accuracy",
            y_label="Accuracy",
            note="Mode-filtered view. Shape shows resolution and marker size shows frame count.",
            scored_only=True,
            runnable_only=True,
            require_resolution=True,
        )
        charts[f"{family}_latency_vs_actual_tokens"] = _write_metric_scatter(
            assets / "by_mode" / f"{family_slug}_latency_vs_actual_processed_tokens.svg",
            title=f"{family_title} Total Latency Vs Actual Processed Tokens",
            rows=family_rows,
            x_field="actual_processed_tokens",
            y_field="total_ms",
            y_label="Total latency (ms)",
            note="Mode-filtered view. Shape shows resolution and marker size shows frame count.",
            scored_only=False,
            runnable_only=True,
            require_resolution=True,
        )
    return charts


def _write_metric_scatter(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    y_label: str,
    note: str,
    scored_only: bool,
    runnable_only: bool,
    require_resolution: bool,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [
        (row, x_value, y_value)
        for row in rows
        if _include_metric_point(row, y_field=y_field, scored_only=scored_only, runnable_only=runnable_only, require_resolution=require_resolution)
        and (x_value := _scatter_value(row, x_field)) is not None
        and x_value > 0
        and (y_value := _scatter_value(row, y_field)) is not None
        and y_value >= 0
    ]
    width = 980
    height = 560
    margin_left = 88
    margin_right = 190
    margin_top = 56
    margin_bottom = 82
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if not points:
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="180" viewBox="0 0 {width} 180">',
                    '<rect width="100%" height="100%" fill="#ffffff"/>',
                    f'<text x="24" y="48" font-size="20" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
                    '<text x="24" y="88" font-size="14" font-family="Arial, sans-serif" fill="#6b7280">No chartable rows after filtering OOM/unscored/unknown metadata.</text>',
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )
        return path

    x_values = [x_value for _, x_value, _ in points]
    y_values = [y_value for _, _, y_value in points]
    min_x = min(x_values)
    max_x = max(x_values)
    log_min = math.log10(min_x)
    log_max = math.log10(max_x)
    if math.isclose(log_min, log_max):
        log_min -= 0.5
        log_max += 0.5
    y_min, y_max = _scatter_y_range(y_field, y_values)

    def x_pos(value: float) -> float:
        return margin_left + ((math.log10(value) - log_min) / (log_max - log_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + ((y_max - value) / (y_max - y_min)) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-size="21" font-weight="700" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
        f'<text x="{margin_left}" y="50" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">{escape(note)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for tick in _y_ticks(y_field, y_min, y_max):
        y = y_pos(tick)
        lines.extend(
            [
                f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{escape(_format_axis_tick(tick, y_field))}</text>',
            ]
        )
    for tick in _log_token_ticks(min_x, max_x):
        x = x_pos(tick)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{escape(_format_axis_tick(tick, x_field))}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape(_axis_label(x_field))}</text>',
            f'<text x="22" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 22 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape(y_label)}</text>',
        ]
    )

    for index, (row, x_value, y_value) in enumerate(points):
        family = _selector_family(row)
        color = _selector_color(family)
        shape = _resolution_shape(row)
        x = x_pos(x_value)
        y = y_pos(y_value)
        size = _marker_size(row)
        tooltip = _point_tooltip(row, index, x_field, y_field, x_value, y_value)
        lines.append(f'<g opacity="0.92"><title>{escape(tooltip)}</title>{_svg_marker(shape, x, y, size, color)}</g>')

    legend_x = margin_left + plot_width + 28
    legend_y = margin_top + 16
    lines.append(f'<text x="{legend_x}" y="{legend_y}" font-size="13" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Mode</text>')
    for offset, family in enumerate(("single", "autogaze", "keep_all", "other"), start=1):
        y = legend_y + offset * 26
        lines.append(_svg_marker("circle", legend_x + 8, y - 4, 7, _selector_color(family)))
        lines.append(f'<text x="{legend_x + 26}" y="{y}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{escape(_selector_label(family))}</text>')

    shape_y = legend_y + 140
    lines.append(f'<text x="{legend_x}" y="{shape_y}" font-size="13" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Resolution</text>')
    used_shapes = {
        _resolution_shape(row)
        for row, _, _ in points
    }
    shape_order = [shape for shape in ("circle", "diamond", "square", "triangle", "cross") if shape in used_shapes]
    for offset, shape in enumerate(shape_order, start=1):
        y = shape_y + offset * 24
        lines.append(_svg_marker(shape, legend_x + 8, y - 4, 7, "#9ca3af"))
        lines.append(f'<text x="{legend_x + 26}" y="{y}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{escape(_shape_label(shape))}</text>')

    read_y = shape_y + 30 + len(shape_order) * 24
    lines.extend(
        [
            f'<text x="{legend_x}" y="{read_y}" font-size="12" font-weight="700" font-family="Arial, sans-serif" fill="#111827">Read</text>',
            f'<text x="{legend_x}" y="{read_y + 20}" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">Log x-axis separates token</text>',
            f'<text x="{legend_x}" y="{read_y + 36}" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">budgets across scale.</text>',
            f'<text x="{legend_x}" y="{read_y + 56}" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">Marker size grows with</text>',
            f'<text x="{legend_x}" y="{read_y + 72}" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">sampled frame count.</text>',
        ]
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_status_scatter(path: Path, *, title: str, rows: list[dict[str, Any]], x_field: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [
        (row, x_value, _status_y(row))
        for row in rows
        if (x_value := _scatter_value(row, x_field)) is not None and x_value > 0
    ]
    width = 980
    height = 360
    margin_left = 98
    margin_right = 190
    margin_top = 56
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if not points:
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="180" viewBox="0 0 {width} 180">',
                    '<rect width="100%" height="100%" fill="#ffffff"/>',
                    f'<text x="24" y="48" font-size="20" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
                    '<text x="24" y="88" font-size="14" font-family="Arial, sans-serif" fill="#6b7280">No runnability data available.</text>',
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )
        return path

    x_values = [x_value for _, x_value, _ in points]
    min_x = min(x_values)
    max_x = max(x_values)
    log_min = math.log10(min_x)
    log_max = math.log10(max_x)
    if math.isclose(log_min, log_max):
        log_min -= 0.5
        log_max += 0.5

    def x_pos(value: float) -> float:
        return margin_left + ((math.log10(value) - log_min) / (log_max - log_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + ((2.0 - value) / 2.0) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-size="21" font-weight="700" font-family="Arial, sans-serif" fill="#111827">{escape(title)}</text>',
        f'<text x="{margin_left}" y="50" font-size="12" font-family="Arial, sans-serif" fill="#6b7280">All rows are shown here, including OOM/failed/unscored runs.</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]
    for label, value in (("failed/OOM", 0.0), ("unscored", 1.0), ("scored", 2.0)):
        y = y_pos(value)
        lines.extend(
            [
                f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{escape(label)}</text>',
            ]
        )
    for tick in _log_token_ticks(min_x, max_x):
        x = x_pos(tick)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="#e5e7eb" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#4b5563">{escape(_format_axis_tick(tick, x_field))}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">{escape(_axis_label(x_field))}</text>',
            f'<text x="24" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#111827">Run status</text>',
        ]
    )
    for index, (row, x_value, status_value) in enumerate(points):
        color = _status_color(row)
        shape = "triangle" if _is_failure_row(row) else ("square" if not _is_scored_row(row) else "circle")
        x = x_pos(x_value)
        y = y_pos(status_value)
        tooltip = "\n".join(
            [
                _row_label(row, index),
                f"mode={row.get('mode')}",
                f"status={row.get('status')}",
                f"failure_kind={row.get('failure_kind')}",
                f"oom_stage={row.get('oom_stage')}",
                f"{x_field}={x_value}",
                f"accuracy={_accuracy_value(row)}",
            ]
        )
        lines.append(f'<g opacity="0.92"><title>{escape(tooltip)}</title>{_svg_marker(shape, x, y, _marker_size(row), color)}</g>')
    legend_x = margin_left + plot_width + 30
    legend_y = margin_top + 18
    for offset, (label, color, shape) in enumerate(
        (
            ("scored", "#16a34a", "circle"),
            ("unscored", "#f59e0b", "square"),
            ("OOM/failed", "#dc2626", "triangle"),
        )
    ):
        y = legend_y + offset * 28
        lines.append(_svg_marker(shape, legend_x + 8, y - 4, 7, color))
        lines.append(f'<text x="{legend_x + 26}" y="{y}" font-size="12" font-family="Arial, sans-serif" fill="#374151">{escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _scatter_value(row: dict[str, Any], field: str) -> float | None:
    if field == "accuracy":
        return _accuracy_value(row)
    if field == "llm_generation_ms":
        return numeric_or_none(first_present(row.get("llm_generation_ms"), row.get("generate_ms"), row.get("llm_forward_ms")))
    return numeric_or_none(row.get(field))


def _include_metric_point(
    row: dict[str, Any],
    *,
    y_field: str,
    scored_only: bool,
    runnable_only: bool,
    require_resolution: bool,
) -> bool:
    if runnable_only and _is_failure_row(row):
        return False
    if scored_only and not _is_scored_row(row):
        return False
    if require_resolution and not _has_known_resolution(row):
        return False
    if y_field != "accuracy" and not _has_metric_value(row, y_field):
        return False
    return True


def _is_failure_row(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    failure_kind = str(row.get("failure_kind") or "").lower()
    return bool(
        row.get("oom")
        or status == "oom"
        or status.startswith("failed")
        or failure_kind in {"oom", "failed", "error"}
        or row.get("oom_stage")
    )


def _is_scored_row(row: dict[str, Any]) -> bool:
    if _is_failure_row(row):
        return False
    if numeric_or_none(row.get("accuracy_scored")) is not None:
        return True
    if numeric_or_none(row.get("accuracy_total")) is not None:
        failed = numeric_or_none(row.get("failed"))
        parse_failed = numeric_or_none(row.get("parse_failed"))
        return failed is not None or parse_failed is not None
    return False


def _has_known_resolution(row: dict[str, Any]) -> bool:
    return _resolution_pixels(row) != float("inf")


def _has_metric_value(row: dict[str, Any], field: str) -> bool:
    return _scatter_value(row, field) is not None


def _status_y(row: dict[str, Any]) -> float:
    if _is_failure_row(row):
        return 0.0
    if _is_scored_row(row):
        return 2.0
    return 1.0


def _status_color(row: dict[str, Any]) -> str:
    if _is_failure_row(row):
        return "#dc2626"
    if _is_scored_row(row):
        return "#16a34a"
    return "#f59e0b"


def _scatter_y_range(y_field: str, y_values: list[float]) -> tuple[float, float]:
    if y_field == "accuracy":
        data_min = min(y_values)
        data_max = max(y_values)
        lower = min(0.25, data_min - 0.03)
        upper = max(0.66, data_max + 0.03)
        return max(0.0, lower), min(1.0, upper)
    return 0.0, max(1.0, max(y_values) * 1.08)


def _y_ticks(y_field: str, y_min: float, y_max: float) -> list[float]:
    if y_field == "accuracy":
        if y_min <= 0.25 and y_max >= 0.66:
            base_ticks = [0.25, 0.35, 0.45, 0.55, 0.66]
        else:
            step = (y_max - y_min) / 4
            base_ticks = [round(y_min + step * index, 2) for index in range(5)]
        return [tick for tick in base_ticks if y_min <= tick <= y_max]
    step = y_max / 4
    return [round(step * index, 2) for index in range(5)]


def _log_token_ticks(min_token: float, max_token: float) -> list[float]:
    if math.isclose(min_token, max_token):
        return [min_token]
    log_min = math.log10(min_token)
    log_max = math.log10(max_token)
    return [10 ** (log_min + (log_max - log_min) * index / 4) for index in range(5)]


def _format_axis_tick(value: float, field: str) -> str:
    if field == "accuracy":
        return f"{value:.2f}"
    if "memory" in field or field.endswith("_bytes"):
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}G"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
    if "ms" in field or "latency" in field:
        if value >= 1_000:
            return f"{value / 1_000:.1f}s"
        return f"{value:.0f}ms"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _axis_label(field: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "total_ms": "Total latency (ms)",
        "autogaze_ms": "AutoGaze latency (ms)",
        "vision_encoder_ms": "ViT/encoder latency (ms)",
        "llm_generation_ms": "MLLM generation latency (ms)",
        "peak_memory_bytes": "Peak memory (bytes)",
        "token_reduction_ratio": "Token reduction ratio (x)",
    }
    if field == "actual_processed_tokens":
        return "Actual processed visual tokens / patches"
    if field == "base_token_budget":
        return "Base visual token / patch budget before selector"
    return labels.get(field, field)


def _selector_color(family: str) -> str:
    return {
        "single": "#2563eb",
        "autogaze": "#f97316",
        "keep_all": "#6b7280",
        "other": "#7c3aed",
    }.get(family, "#7c3aed")


def _selector_label(family: str) -> str:
    return {
        "single": "single-scale/off",
        "autogaze": "AutoGaze/sparse",
        "keep_all": "keep-all",
        "other": "other",
    }.get(family, family)


def _marker_size(row: dict[str, Any]) -> float:
    frames = numeric_or_none(row.get("frames"))
    if frames is None or frames <= 0:
        return 6.5
    return min(12.0, max(5.5, 4.5 + math.log2(frames + 1) * 0.65))


def _resolution_shape(row: dict[str, Any]) -> str:
    pixels = _resolution_pixels(row)
    if pixels == float("inf"):
        return "cross"
    if pixels <= 448 * 448:
        return "circle"
    if pixels <= 720 * 1280:
        return "diamond"
    if pixels <= 1080 * 1920:
        return "square"
    return "triangle"


def _shape_label(shape: str) -> str:
    return {
        "circle": "<=448p-ish",
        "diamond": "<=720p-ish",
        "square": "<=1080p-ish",
        "triangle": ">1080p",
        "cross": "unknown",
    }.get(shape, shape)


def _svg_marker(shape: str, x: float, y: float, size: float, color: str) -> str:
    stroke = "#111827"
    if shape == "diamond":
        points = [
            (x, y - size),
            (x + size, y),
            (x, y + size),
            (x - size, y),
        ]
        point_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        return f'<polygon points="{point_text}" fill="{color}" stroke="{stroke}" stroke-width="0.7"/>'
    if shape == "square":
        return f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{size * 2:.2f}" height="{size * 2:.2f}" fill="{color}" stroke="{stroke}" stroke-width="0.7"/>'
    if shape == "triangle":
        points = [
            (x, y - size),
            (x + size * 0.92, y + size * 0.72),
            (x - size * 0.92, y + size * 0.72),
        ]
        point_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        return f'<polygon points="{point_text}" fill="{color}" stroke="{stroke}" stroke-width="0.7"/>'
    if shape == "cross":
        return (
            f'<line x1="{x - size:.2f}" y1="{y - size:.2f}" x2="{x + size:.2f}" y2="{y + size:.2f}" '
            f'stroke="{color}" stroke-width="2"/>'
            f'<line x1="{x + size:.2f}" y1="{y - size:.2f}" x2="{x - size:.2f}" y2="{y + size:.2f}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="0.7"/>'


def _point_tooltip(
    row: dict[str, Any],
    index: int,
    x_field: str,
    y_field: str,
    x_value: float,
    y_value: float,
) -> str:
    lines = [
        _row_label(row, index),
        f"mode={row.get('mode')}",
        f"status={row.get('status')}",
        f"frames={row.get('frames')}",
        f"thumbnail_frames={row.get('thumbnail_frames')}",
        f"input_resolution={row.get('processor_input_resolution')}",
        f"{x_field}={x_value}",
        f"{y_field}={y_value}",
    ]
    if x_field != "base_token_budget":
        lines.append(f"base_token_budget={row.get('base_token_budget')}")
    if x_field != "actual_processed_tokens":
        lines.append(f"actual_processed_tokens={row.get('actual_processed_tokens')}")
    if y_field != "accuracy":
        lines.append(f"accuracy={_accuracy_value(row)}")
    return "\n".join(lines)


def _row_label(row: dict[str, Any], index: int) -> str:
    mode = row.get("mode") or "run"
    mode_label = {"single_scale_dense": "single-scale"}.get(str(mode), str(mode))
    frames = row.get("frames")
    thumbnail = row.get("thumbnail_frames")
    resolution = row.get("processor_input_resolution")
    tiles = row.get("max_tiles_video")
    model = _model_label(row.get("model_path"))
    parts = []
    if model:
        parts.append(model)
    if frames is not None:
        frame_label = f"{int(float(frames))}f"
        if thumbnail not in {None, "", 0, 0.0}:
            frame_label += f"+{int(float(thumbnail))}t"
        parts.append(frame_label)
    if resolution:
        parts.append(str(resolution))
    if tiles not in {None, ""}:
        parts.append(f"tiles{int(float(tiles))}")
    parts.append(mode_label)
    if len(parts) == 1:
        parts.insert(0, Path(str(row.get("source_path") or f"run_{index}")).stem)
    return shorten_label("/".join(parts), max_chars=46)


def _model_label(model_path: Any) -> str | None:
    if not model_path:
        return None
    text = str(model_path).rstrip("/")
    return text.rsplit("/", 1)[-1] if "/" in text else text


def _resolution_parts(value: Any) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        return None, None
    left, right = text.split("x", 1)
    try:
        return int(float(left)), int(float(right))
    except ValueError:
        return None, None


def _accuracy_value(row: dict[str, Any]) -> float:
    return numeric_or_none(first_present(row.get("accuracy_scored"), row.get("accuracy_total"))) or 0.0


def _render_markdown(rows: list[dict[str, Any]], charts: dict[str, Path], output_dir: Path) -> str:
    lines = ["# AutoGaze Experiment Trend Report", ""]
    chart_sections = [
        (
            "Overview Charts",
            (
                ("Latency By Config", "latency"),
                ("Latency Attribution By Config", "latency_attribution"),
                ("Accuracy By Config", "accuracy"),
                ("Accuracy Vs Frames", "accuracy_vs_frames"),
                ("Accuracy Vs Input Resolution", "accuracy_vs_input_resolution"),
                ("Token Reduction By Config", "token_reduction"),
                ("Memory Peak By Config", "memory"),
                ("Status Counts", "status"),
                ("Runnability Vs Base Tokens", "runnability_vs_base_tokens"),
            ),
        ),
        (
            "Scale Efficiency Charts",
            (
                ("Accuracy Vs Base Tokens", "accuracy_vs_base_tokens"),
                ("Accuracy Vs Actual Processed Tokens", "accuracy_vs_actual_tokens"),
                ("Total Latency Vs Base Tokens", "total_latency_vs_base_tokens"),
                ("Total Latency Vs Actual Processed Tokens", "total_latency_vs_actual_tokens"),
                ("Peak Memory Vs Actual Processed Tokens", "memory_vs_actual_tokens"),
                ("Token Reduction Vs Base Tokens", "token_reduction_vs_base_tokens"),
            ),
        ),
        (
            "Module Latency Charts",
            (
                ("AutoGaze Latency Vs Base Tokens", "autogaze_latency_vs_base_tokens"),
                ("ViT Encoder Latency Vs Actual Tokens", "encoder_latency_vs_actual_tokens"),
                ("MLLM Latency Vs Actual Tokens", "llm_latency_vs_actual_tokens"),
            ),
        ),
        (
            "Mode-Specific Charts",
            (
                ("Single/Off Accuracy Vs Base Tokens", "single_accuracy_vs_base_tokens"),
                ("Single/Off Accuracy Vs Actual Tokens", "single_accuracy_vs_actual_tokens"),
                ("Single/Off Latency Vs Actual Tokens", "single_latency_vs_actual_tokens"),
                ("AutoGaze Accuracy Vs Base Tokens", "autogaze_accuracy_vs_base_tokens"),
                ("AutoGaze Accuracy Vs Actual Tokens", "autogaze_accuracy_vs_actual_tokens"),
                ("AutoGaze Latency Vs Actual Tokens", "autogaze_latency_vs_actual_tokens"),
                ("Keep-All Accuracy Vs Base Tokens", "keep_all_accuracy_vs_base_tokens"),
                ("Keep-All Accuracy Vs Actual Tokens", "keep_all_accuracy_vs_actual_tokens"),
                ("Keep-All Latency Vs Actual Tokens", "keep_all_latency_vs_actual_tokens"),
            ),
        ),
    ]
    lines.extend(["## Charts", ""])
    lines.extend(
        [
            "> Accuracy/latency scatter charts exclude OOM, failed, unscored, and unknown-resolution rows so that missing results are not plotted as real zero-accuracy model behavior.",
            "> Those excluded runs remain in the CSV/JSON and are summarized in the runnability/status charts.",
            "",
        ]
    )
    for section_title, entries in chart_sections:
        visible_entries = [(title, key) for title, key in entries if key in charts]
        if not visible_entries:
            continue
        lines.extend([f"### {section_title}", ""])
        for title, key in visible_entries:
            path = charts[key].relative_to(output_dir)
            lines.extend([f"#### {title}", "", f"![{title}]({path})", ""])
    lines.extend(["## Summary Rows", ""])
    columns = [
        ("mode", "Mode"),
        ("status", "Status"),
        ("oom_stage", "OOM stage"),
        ("frames", "Frames"),
        ("thumbnail_frames", "Thumb"),
        ("processor_input_resolution", "Input res"),
        ("source_resolution", "Source res"),
        ("max_tiles_video", "Tiles"),
        ("accuracy_scored", "Accuracy scored"),
        ("total_ms", "Total ms"),
        ("video_decode_read_ms", "Decode/read ms"),
        ("video_frame_resize_ms", "Frame resize ms"),
        ("video_tiling_ms", "Tile/tensor ms"),
        ("selector_input_build_ms", "Selector input ms"),
        ("preprocess_rest_ms", "Prep rest ms"),
        ("autogaze_ms", "AutoGaze ms"),
        ("vision_encoder_ms", "ViT ms"),
        ("mm_projector_ms", "Projector ms"),
        ("generate_ms", "Generate total ms"),
        ("llm_generation_ms", "LLM generation ms"),
        ("llm_forward_ms", "LLM forward ms"),
        ("generation_rest_ms", "Generate rest ms"),
        ("single_scale_dense_patch_tokens", "Single patch"),
        ("full_or_raw_patch_tokens", "Full patch"),
        ("autogaze_selected_patch_tokens", "Selected patch"),
        ("llm_visual_tokens", "LLM visual"),
        ("base_token_budget", "Base token budget"),
        ("actual_processed_tokens", "Actual processed"),
        ("token_reduction_ratio", "Patch x"),
        ("peak_memory_bytes", "Peak bytes"),
        ("accuracy_total", "Accuracy total"),
    ]
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows[:200]:
        lines.append("| " + " | ".join(_cell(row.get(field)) for field, _ in columns) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate AutoGaze/NVILA experiment JSON reports into trends.")
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sort",
        choices=(
            "comparison",
            "config",
            "frames",
            "resolution",
            "base-tokens",
            "actual-tokens",
            "latency",
            "token-reduction",
            "memory",
            "accuracy",
            "status",
        ),
        default="comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = aggregate_report_roots(args.input_root, args.output_dir, sort=args.sort)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
