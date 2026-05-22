from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    if payload.get("task_type") in {"captioning", "action_classification"} and isinstance(payload.get("modes"), dict):
        return [_normalize_video_task_mode(source, mode, summary) for mode, summary in payload["modes"].items()]
    if "modes" in payload and isinstance(payload.get("modes"), dict):
        return [_normalize_plugin_mode(source, mode, summary) for mode, summary in payload["modes"].items()]
    if "readable_summary" in payload and any(mode in payload for mode in HLVID_MODE_ORDER):
        return [_normalize_hlvid_mode(source, payload, mode) for mode in HLVID_MODE_ORDER if mode in payload]
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
        mode=str(payload.get("mode") or payload.get("gazing_mode") or "single"),
        model_path=payload.get("model_path") or get_path(payload, "result.model_path"),
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
    return row


def _normalize_video_task_mode(path: Path, mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = _blank_row(path, report_kind="video_task_summary", mode=mode, model_path=None)
    status_counts = as_mapping(summary.get("status_counts"))
    row["status"] = "oom" if status_counts.get("oom") else "ok"
    row["oom"] = bool(status_counts.get("oom"))
    row["accuracy_total"] = numeric_or_none(summary.get("accuracy_total"))
    row["accuracy_scored"] = numeric_or_none(summary.get("accuracy_scored"))
    row["failed"] = numeric_or_none(summary.get("failed"))
    row["parse_failed"] = numeric_or_none(summary.get("parse_failed"))
    row["total_ms"] = numeric_or_none(get_path(summary, "latency_ms.median"))
    row["generate_ms"] = numeric_or_none(get_path(summary, "generate_ms.median"))
    row["peak_memory_bytes"] = numeric_or_none(get_path(summary, "peak_memory_bytes.median"))
    before = numeric_or_none(get_path(summary, "visual_tokens_before_prune.median"))
    after = numeric_or_none(get_path(summary, "visual_tokens_after_prune.median"))
    row["full_or_raw_patch_tokens"] = before
    row["autogaze_selected_patch_tokens"] = after
    row["llm_visual_tokens"] = after
    row["token_reduction_ratio"] = before / after if before is not None and after not in {None, 0} else None
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
    row["frames"] = numeric_or_none(first_present(video.get("actual_video_frames"), video.get("requested_video_frames")))
    row["thumbnail_frames"] = numeric_or_none(first_present(thumbnail.get("actual_frames"), thumbnail.get("effective_frames")))
    row["source_resolution"] = video.get("source_resolution")
    row["processor_input_resolution"] = video.get("processor_input_resolution")
    row["max_tiles_video"] = numeric_or_none(tiling.get("spatial_tiles_per_frame"))


def _apply_flat_budget(row: dict[str, Any], budget: dict[str, Any]) -> None:
    if not budget:
        return
    row["frames"] = numeric_or_none(budget.get("video.requested_video_frames"))
    row["thumbnail_frames"] = numeric_or_none(
        first_present(budget.get("thumbnail.actual_frames"), budget.get("thumbnail.effective_frames"))
    )
    row["source_resolution"] = budget.get("video.source_resolution")
    row["processor_input_resolution"] = budget.get("video.processor_input_resolution")
    row["max_tiles_video"] = numeric_or_none(budget.get("tiling.spatial_tiles_per_frame"))
    row["single_scale_dense_patch_tokens"] = numeric_or_none(
        first_present(
            budget.get("single_scale_dense_vision_budget.total_patch_tokens"),
            budget.get("single_scale_dense_vision_budget.estimated_total_patch_tokens"),
        )
    )
    row["full_or_raw_patch_tokens"] = numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.keep_all_total_patch_tokens"),
            budget.get("patch_budget_before_vit.actual_raw_patch_tokens_before_vit"),
            budget.get("patch_budget_before_vit.estimated_visual_tokens_before_prune"),
        )
    )
    row["autogaze_selected_patch_tokens"] = numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.autogaze_selected_total_patch_tokens"),
            budget.get("patch_budget_before_vit.estimated_visual_tokens_after_prune"),
        )
    )
    row["token_reduction_ratio"] = numeric_or_none(
        first_present(
            budget.get("patch_budget_before_siglip.total_patch_reduction_ratio"),
            budget.get("patch_budget_before_vit.estimated_visual_token_reduction_ratio"),
        )
    )
    row["llm_visual_tokens"] = numeric_or_none(budget.get("llm_visual_budget.actual_visual_tokens"))


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
    rows = sort_rows([{field: row.get(field) for field in ROW_FIELDS} for row in rows], sort=sort)
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
    if mode in {"single-scale-dense", "keep-all-single", "single-scale"} or "single-scale" in mode:
        return 1
    if mode == "autogaze" or "autogaze" in mode:
        return 2
    if "probe" in mode or "sidecar" in mode:
        return 4
    return 3


def _config_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_path") or "",
        row.get("frames") or "",
        row.get("thumbnail_frames") or "",
        row.get("processor_input_resolution") or "",
        row.get("max_tiles_video") or "",
    )


def _comparison_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("mode") or ""), str(row.get("source_path") or ""))


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
    return charts


def _row_label(row: dict[str, Any], index: int) -> str:
    mode = row.get("mode") or "run"
    mode_label = {"single_scale_dense": "single-scale"}.get(str(mode), str(mode))
    frames = row.get("frames")
    resolution = row.get("processor_input_resolution")
    parts = [mode_label]
    if frames is not None:
        parts.append(f"{int(float(frames))}f")
    if resolution:
        parts.append(str(resolution))
    return shorten_label("/".join(parts) if parts else f"run_{index}", max_chars=34)


def _render_markdown(rows: list[dict[str, Any]], charts: dict[str, Path], output_dir: Path) -> str:
    lines = ["# AutoGaze Experiment Trend Report", ""]
    lines.extend(["## Charts", ""])
    for title, key in (
        ("Latency By Config", "latency"),
        ("Latency Attribution By Config", "latency_attribution"),
        ("Token Reduction By Config", "token_reduction"),
        ("Memory Peak By Config", "memory"),
        ("Status Counts", "status"),
    ):
        path = charts[key].relative_to(output_dir)
        lines.extend([f"### {title}", "", f"![{title}]({path})", ""])
    lines.extend(["## Summary Rows", ""])
    columns = [
        ("mode", "Mode"),
        ("status", "Status"),
        ("oom_stage", "OOM stage"),
        ("frames", "Frames"),
        ("processor_input_resolution", "Input res"),
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
        ("full_or_raw_patch_tokens", "Full patch"),
        ("autogaze_selected_patch_tokens", "Selected patch"),
        ("token_reduction_ratio", "Patch x"),
        ("peak_memory_bytes", "Peak bytes"),
        ("accuracy_total", "Accuracy"),
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
        choices=("comparison", "latency", "token-reduction", "memory", "accuracy", "status"),
        default="comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = aggregate_report_roots(args.input_root, args.output_dir, sort=args.sort)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
