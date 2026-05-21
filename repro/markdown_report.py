from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from repro.report_charts import ChartArtifact, build_standard_report_charts


PIPELINE_ASCII = """Video file(s)
  -> Video decode/sample
  -> Resize + spatial tiling + thumbnail build
  -> AutoGaze ON/OFF
  -> SigLIP / ViT Encoder
  -> TokenShuffle + MM projector
  -> LLM prefill/generation
  -> Answer or benchmark score"""


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def get_path(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
    return default if value is None else value


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return compact_json(value)
    return str(value)


def format_bytes(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return format_value(value)
    gib = number / (1024**3)
    return f"{int(number):,} B ({gib:.3f} GiB)"


def escape_cell(value: Any) -> str:
    return format_value(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data available._"
    escaped_headers = [escape_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(escape_cell(item) for item in padded[: len(headers)]) + " |")
    return "\n".join(lines)


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio_before_over_after(before: Any, after: Any) -> float | None:
    before_value = numeric_or_none(before)
    after_value = numeric_or_none(after)
    if before_value is None or after_value in {None, 0.0}:
        return None
    return before_value / after_value


def reduction_percent(before: Any, after: Any) -> float | None:
    before_value = numeric_or_none(before)
    after_value = numeric_or_none(after)
    if before_value in {None, 0.0} or after_value is None:
        return None
    return (1.0 - after_value / before_value) * 100.0


def get_budget_value(summary: dict[str, Any], field: str, default: Any = None) -> Any:
    if field in summary:
        value = summary.get(field)
        return default if value is None else value
    return get_path(summary, field, default)


def detect_report_kind(payload: dict[str, Any]) -> str:
    if "readable_summary" in payload and "keep_all" in payload and "autogaze" in payload:
        return "hlvid_benchmark"
    if payload.get("mode") == "stream-profile":
        return "stream_profile"
    if "estimate" in payload and "effective_video" in payload:
        return "preflight"
    if "key_metrics_summary" in payload or "result" in payload:
        return "single_inference"
    if "readable_performance_summary" in payload:
        return "hlvid_mode_summary"
    return "generic"


def result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    return result if isinstance(result, dict) else payload


def key_metrics_from_stream_profile(payload: dict[str, Any]) -> dict[str, Any]:
    stage_timings = as_mapping(payload.get("stage_timings_ms"))

    def stage_total(name: str) -> float | None:
        stage = as_mapping(stage_timings.get(name))
        value = stage.get("total_ms")
        return float(value) if value is not None else None

    preprocess_parts = [
        stage_total("video_keyframe_index_scan"),
        stage_total("video_decode_seek"),
        stage_total("video_decode_scan"),
        stage_total("video_frame_to_pil"),
        stage_total("video_frame_resize"),
        stage_total("spatial_tile_build"),
        stage_total("thumbnail_resize"),
        stage_total("thumbnail_tensorize"),
    ]
    preprocess_total = sum(part for part in preprocess_parts if part is not None)
    autogaze_total = sum(
        part
        for part in [stage_total("tile_autogaze_tensorize"), stage_total("tile_autogaze_forward")]
        if part is not None
    )
    vit_total = stage_total("siglip_gazed_forward") or stage_total("siglip_keep_all_forward")
    tokens = as_mapping(get_path(payload, "stream_plan.tokens", {}))
    gaze = as_mapping(payload.get("gaze"))
    memory = as_mapping(payload.get("memory_bytes"))
    return {
        "latency_ms": {
            "total_ms": preprocess_total + autogaze_total + (vit_total or 0.0),
            "preprocess_without_autogaze_ms": preprocess_total,
            "preprocess_total_ms": preprocess_total,
            "autogaze_ms": autogaze_total,
            "autogaze_total_ms": autogaze_total,
            "vit_encoder_ms": vit_total,
            "llm_ms": None,
        },
        "tokens": {
            "video_sampled_frames": get_path(payload, "sampling.num_video_frames"),
            "thumbnail_sampled_frames": get_path(payload, "sampling.num_video_frames_thumbnail"),
            "encoder_patch_tokens_before_keep_all_or_raw": tokens.get("encoder_raw_patch_tokens"),
            "encoder_patch_tokens_after_autogaze": gaze.get("selected_non_padded_patches"),
            "encoder_token_reduction_ratio": gaze.get("token_reduction_ratio"),
            "autogaze_input_tile_patch_tokens": gaze.get("raw_patch_budget"),
            "autogaze_selected_tile_patch_tokens": gaze.get("selected_non_padded_patches"),
            "autogaze_patch_reduction_ratio": gaze.get("token_reduction_ratio"),
            "llm_visual_tokens_before_keep_all_estimated": tokens.get("llm_keep_all_visual_tokens_estimated"),
            "llm_visual_tokens_after_actual": get_path(
                payload,
                "compute_metrics.mllm.autogaze_visual_tokens_lower_bound_estimated",
            ),
            "llm_visual_token_reduction_ratio": None,
        },
        "memory_bytes": {
            "raw_frame_buffer_peak": memory.get("raw_frame_buffer_peak"),
            "autogaze_tile_tensor_peak_per_temporal_chunk": memory.get(
                "autogaze_tile_tensor_peak_per_temporal_chunk"
            ),
            "siglip_gazed_hidden_peak": memory.get("siglip_gazed_hidden_peak"),
            "siglip_keep_all_hidden_peak": memory.get("siglip_keep_all_hidden_peak"),
            "cuda_peak_memory_bytes": memory.get("cuda_peak_memory_bytes"),
        },
    }


def key_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("key_metrics_summary"), dict):
        return payload["key_metrics_summary"]
    if isinstance(get_path(payload, "generation.metrics"), dict):
        return get_path(payload, "generation.metrics")
    if isinstance(get_path(payload, "readable_performance_summary.key_metrics_median"), dict):
        return get_path(payload, "readable_performance_summary.key_metrics_median")
    if isinstance(get_path(payload, "readable_summary.key_metrics_median"), dict):
        return get_path(payload, "readable_summary.key_metrics_median")
    if payload.get("mode") == "stream-profile":
        return key_metrics_from_stream_profile(payload)

    result = result_payload(payload)
    token_metrics = as_mapping(result.get("token_metrics"))
    return {
        "latency_ms": {
            "total_ms": result.get("total_ms"),
            "preprocess_without_autogaze_ms": result.get("video_preprocess_without_autogaze_ms"),
            "preprocess_total_ms": result.get("video_preprocess_ms"),
            "autogaze_ms": result.get("autogaze_ms"),
            "autogaze_total_ms": result.get("autogaze_total_ms"),
            "vit_encoder_ms": result.get("siglip_vision_ms"),
            "llm_ms": result.get("llm_forward_ms"),
        },
        "tokens": {
            "video_sampled_frames": token_metrics.get("video_sampled_frames"),
            "thumbnail_sampled_frames": token_metrics.get("thumbnail_sampled_frames"),
            "encoder_patch_tokens_before_keep_all_or_raw": token_metrics.get("encoder_raw_patch_tokens"),
            "encoder_patch_tokens_after_autogaze": token_metrics.get("encoder_autogaze_selected_patch_tokens"),
            "encoder_token_reduction_ratio": token_metrics.get("encoder_token_reduction_ratio"),
            "autogaze_input_tile_patch_tokens": token_metrics.get("autogaze_input_patch_tokens"),
            "autogaze_selected_tile_patch_tokens": token_metrics.get("autogaze_selected_patch_tokens"),
            "autogaze_patch_reduction_ratio": token_metrics.get("autogaze_patch_reduction_ratio"),
            "llm_visual_tokens_before_keep_all_estimated": token_metrics.get(
                "llm_keep_all_visual_tokens_estimated"
            ),
            "llm_visual_tokens_after_actual": token_metrics.get("llm_actual_visual_tokens"),
            "llm_visual_token_reduction_ratio": token_metrics.get("llm_visual_token_reduction_ratio"),
        },
        "memory_bytes": {
            "processor_peak": result.get("processor_peak_memory_bytes"),
            "ttft_peak": result.get("ttft_peak_memory_bytes"),
            "llm_peak": result.get("llm_peak_memory_bytes"),
            "overall_peak": result.get("peak_memory_bytes"),
        },
    }


def processing_budget_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = result_payload(payload)
    return as_mapping(
        result.get("processing_budget_summary")
        or payload.get("processing_budget_summary")
        or get_path(payload, "generation.metrics.processing_budget_summary")
    )


def readable_processing_budget_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return as_mapping(
        get_path(payload, "readable_summary.processing_budget_summary")
        or get_path(payload, "readable_performance_summary.processing_budget_summary")
    )


def enriched_key_metrics(payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = {key: dict(value) if isinstance(value, dict) else value for key, value in metrics.items()}
    tokens = dict(as_mapping(enriched.get("tokens")))

    readable_budget = readable_processing_budget_summary(payload)
    if readable_budget:
        add_readable_budget_token_metrics(tokens, readable_budget)
    else:
        add_single_budget_token_metrics(tokens, processing_budget_summary(payload))

    if tokens:
        enriched["tokens"] = tokens
    return enriched


def add_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def add_single_budget_token_metrics(tokens: dict[str, Any], summary: dict[str, Any]) -> None:
    if not summary:
        return
    single_scale = as_mapping(summary.get("single_scale_dense_vision_budget"))
    patch_budget_siglip = as_mapping(summary.get("patch_budget_before_siglip"))
    patch_budget_vit = as_mapping(summary.get("patch_budget_before_vit"))
    llm_budget = as_mapping(summary.get("llm_visual_budget"))

    single_scale_patches = first_present(
        single_scale.get("total_patch_tokens"),
        single_scale.get("estimated_total_patch_tokens"),
    )
    single_scale_llm = first_present(
        single_scale.get("llm_visual_tokens_estimated"),
        single_scale.get("estimated_llm_visual_tokens_after_token_shuffle"),
    )
    hd_multiscale_patches = patch_budget_siglip.get("keep_all_total_patch_tokens")
    raw_vit_patches = first_present(
        patch_budget_vit.get("actual_raw_patch_tokens_before_vit"),
        patch_budget_vit.get("estimated_visual_tokens_before_prune"),
    )
    selected_patches = first_present(
        patch_budget_siglip.get("autogaze_selected_total_patch_tokens"),
        patch_budget_vit.get("estimated_visual_tokens_after_prune"),
        tokens.get("visual_tokens_after_prune"),
    )
    raw_or_multiscale_patches = first_present(hd_multiscale_patches, raw_vit_patches)
    llm_keep_all = first_present(
        llm_budget.get("keep_all_visual_tokens_estimated"),
        tokens.get("llm_visual_tokens_before_keep_all_estimated"),
    )
    llm_actual = first_present(
        llm_budget.get("actual_visual_tokens"),
        tokens.get("llm_visual_tokens_after_actual"),
    )

    add_if_present(tokens, "single_scale_dense_siglip_reference_patch_tokens", single_scale_patches)
    add_if_present(tokens, "single_scale_dense_siglip_reference_llm_visual_tokens_estimated", single_scale_llm)
    add_if_present(tokens, "hd_multiscale_keep_all_patch_tokens", hd_multiscale_patches)
    add_if_present(tokens, "raw_vit_patch_tokens_before_selector", raw_vit_patches)
    add_if_present(tokens, "autogaze_selected_total_patch_tokens", selected_patches)
    add_if_present(tokens, "encoder_input_patch_tokens_after_autogaze", selected_patches)
    add_if_present(
        tokens,
        "patch_reduction_ratio_single_scale_over_autogaze",
        first_present(
            single_scale.get("ratio_over_autogaze_selected_total_patch_tokens"),
            single_scale.get("ratio_over_estimated_visual_tokens_after_prune"),
            ratio_before_over_after(single_scale_patches, selected_patches),
        ),
    )
    add_if_present(
        tokens,
        "patch_reduction_ratio_full_or_raw_over_autogaze",
        ratio_before_over_after(raw_or_multiscale_patches, selected_patches),
    )
    add_if_present(tokens, "llm_visual_tokens_keep_all_estimated_from_budget", llm_keep_all)
    add_if_present(tokens, "llm_visual_tokens_actual_from_budget", llm_actual)
    add_if_present(
        tokens,
        "llm_visual_token_reduction_ratio_from_budget",
        ratio_before_over_after(llm_keep_all, llm_actual),
    )


def add_readable_budget_token_metrics(tokens: dict[str, Any], readable_budget: dict[str, Any]) -> None:
    keep_all = as_mapping(readable_budget.get("keep_all_median") or readable_budget.get("mode_median"))
    autogaze = as_mapping(readable_budget.get("autogaze_median"))

    single_keep_all = first_present(
        get_budget_value(keep_all, "single_scale_dense_vision_budget.total_patch_tokens"),
        get_budget_value(keep_all, "single_scale_dense_vision_budget.estimated_total_patch_tokens"),
    )
    single_autogaze_basis = first_present(
        get_budget_value(autogaze, "single_scale_dense_vision_budget.total_patch_tokens"),
        get_budget_value(autogaze, "single_scale_dense_vision_budget.estimated_total_patch_tokens"),
        single_keep_all,
    )
    hd_keep_all = first_present(
        get_budget_value(autogaze, "patch_budget_before_siglip.keep_all_total_patch_tokens"),
        get_budget_value(keep_all, "patch_budget_before_siglip.keep_all_total_patch_tokens"),
        get_budget_value(autogaze, "patch_budget_before_vit.actual_raw_patch_tokens_before_vit"),
        get_budget_value(autogaze, "patch_budget_before_vit.estimated_visual_tokens_before_prune"),
    )
    selected = first_present(
        get_budget_value(autogaze, "patch_budget_before_siglip.autogaze_selected_total_patch_tokens"),
        get_budget_value(autogaze, "patch_budget_before_vit.estimated_visual_tokens_after_prune"),
    )
    llm_keep_all = first_present(
        get_budget_value(autogaze, "llm_visual_budget.keep_all_visual_tokens_estimated"),
        get_budget_value(keep_all, "llm_visual_budget.keep_all_visual_tokens_estimated"),
    )
    llm_actual = get_budget_value(autogaze, "llm_visual_budget.actual_visual_tokens")

    if single_keep_all is not None or single_autogaze_basis is not None:
        tokens["single_scale_dense_siglip_reference_patch_tokens"] = {
            "keep_all": single_keep_all,
            "autogaze": single_autogaze_basis,
        }
    if single_autogaze_basis is not None or selected is not None:
        tokens["single_scale_dense_reference_vs_autogaze_selected_patches"] = before_after_metric(
            single_autogaze_basis,
            selected,
        )
    if hd_keep_all is not None or selected is not None:
        tokens["hd_multiscale_keep_all_vs_autogaze_selected_patches"] = before_after_metric(
            hd_keep_all,
            selected,
        )
    if llm_keep_all is not None or llm_actual is not None:
        tokens["llm_visual_budget_keep_all_vs_actual"] = before_after_metric(llm_keep_all, llm_actual)


def before_after_metric(before: Any, after: Any) -> dict[str, Any]:
    return {
        "before_keep_all_estimated": before,
        "after_autogaze_actual": after,
        "reduction_ratio_before_over_after": ratio_before_over_after(before, after),
        "reduction_percent_of_before": reduction_percent(before, after),
    }


def render_simple_metric_table(metrics: dict[str, Any], *, memory: bool = False) -> str:
    rows: list[list[Any]] = []
    for name, value in metrics.items():
        rows.append([name, format_bytes(value) if memory else format_value(value)])
    return markdown_table(["Metric", "Value"], rows)


def render_comparison_metric_table(metrics: dict[str, Any], *, memory: bool = False) -> str:
    rows: list[list[Any]] = []
    for name, value in metrics.items():
        if isinstance(value, dict) and ("keep_all" in value or "autogaze" in value):
            keep_all = value.get("keep_all")
            autogaze = value.get("autogaze")
            rows.append(
                [
                    name,
                    format_bytes(keep_all) if memory else format_value(keep_all),
                    format_bytes(autogaze) if memory else format_value(autogaze),
                    value.get("speedup_ratio_keep_all_over_autogaze")
                    or value.get("reduction_ratio_keep_all_over_autogaze"),
                    value.get("reduction_percent_of_keep_all"),
                ]
            )
        elif isinstance(value, dict):
            rows.append(
                [
                    name,
                    first_present(
                        value.get("before_keep_all_or_raw"),
                        value.get("before_autogaze_selection"),
                        value.get("before_keep_all_estimated"),
                    ),
                    first_present(
                        value.get("after_autogaze"),
                        value.get("after_autogaze_selection"),
                        value.get("after_autogaze_actual"),
                    ),
                    value.get("reduction_ratio_before_over_after"),
                    value.get("reduction_percent_of_before"),
                ]
            )
        else:
            rows.append([name, value, "-", "-", "-"])
    return markdown_table(["Metric", "Before / Keep-all", "After / AutoGaze", "Ratio", "Reduction %"], rows)


def render_key_metrics_section(metrics: dict[str, Any]) -> str:
    sections = ["## Key Metrics"]
    for group_name in ("latency_ms", "tokens", "memory_bytes"):
        group = as_mapping(metrics.get(group_name))
        if not group:
            continue
        sections.append(f"### {group_name}")
        is_memory = group_name == "memory_bytes"
        if any(isinstance(value, dict) for value in group.values()):
            sections.append(render_comparison_metric_table(group, memory=is_memory))
        else:
            sections.append(render_simple_metric_table(group, memory=is_memory))
    return "\n\n".join(sections)


def render_charts_section(artifacts: list[ChartArtifact], *, markdown_dir: Path | None = None) -> str:
    if not artifacts:
        return ""
    rows: list[list[Any]] = []
    lines = ["## Charts"]
    for artifact in artifacts:
        path = artifact.path
        if markdown_dir is not None:
            try:
                display_path = os.path.relpath(path, markdown_dir)
            except ValueError:
                display_path = str(path)
        else:
            display_path = str(path)
        rows.append([artifact.title, display_path])
        lines.append(f"![{artifact.title}]({display_path})")
    lines.append(markdown_table(["Chart", "File"], rows))
    return "\n\n".join(lines)


def _latency_accounting(payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    for value in (
        payload.get("latency_accounting"),
        metrics.get("latency_accounting"),
        get_path(payload, "readable_summary.latency_accounting"),
        get_path(payload, "readable_performance_summary.latency_accounting"),
    ):
        if isinstance(value, dict) and value:
            return value
    result = result_payload(payload)
    value = result.get("latency_accounting") if isinstance(result, dict) else None
    return value if isinstance(value, dict) else {}


def render_latency_accounting_section(payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    accounting = _latency_accounting(payload, metrics)
    if not accounting:
        return ""

    sections = ["## Latency Accounting"]
    hierarchy = as_mapping(accounting.get("hierarchy"))
    if hierarchy:
        ascii_tree = hierarchy.get("ascii_tree")
        if ascii_tree:
            sections.append(f"### Time Hierarchy\n\n```text\n{ascii_tree}\n```")
        quick_answers = as_mapping(hierarchy.get("quick_answers"))
        if quick_answers:
            sections.append(
                "### Quick Answers\n\n"
                + markdown_table(
                    ["Question", "Answer"],
                    [[key, value] for key, value in quick_answers.items()],
                )
            )

    rows: list[list[Any]] = []
    additive = as_mapping(accounting.get("additive_total_ms"))
    if additive:
        rows.extend(
            [
                ["additive formula", additive.get("formula"), "Only these top-level fields recompute total latency."],
                ["total_ms", additive.get("total_ms"), "End-to-end measured latency."],
                [
                    "video_preprocess_without_autogaze_ms",
                    additive.get("video_preprocess_without_autogaze_ms"),
                    "Decode/tile/tokenization processor phase excluding AutoGaze.",
                ],
                ["autogaze_total_ms", additive.get("autogaze_total_ms"), "AutoGaze stage total."],
                ["generate_ms", additive.get("generate_ms"), "Full NVILA generation phase."],
                [
                    "recomputed_total_ms",
                    additive.get("recomputed_total_ms"),
                    "video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms.",
                ],
                ["delta_ms", additive.get("delta_ms"), "Difference between recorded total and recomputed total."],
                ["ttft_ms", additive.get("ttft_ms_excluded_from_total"), "Separate 1-token pass; do not add to total_ms."],
            ]
        )
        legacy = as_mapping(accounting.get("legacy_inclusive_total_ms"))
        if legacy:
            rows.extend(
                [
                    ["legacy formula", legacy.get("formula"), "Backward-compatible inclusive preprocess view."],
                    ["legacy video_preprocess_ms", legacy.get("video_preprocess_ms"), "Includes AutoGaze."],
                    ["legacy recomputed_total_ms", legacy.get("recomputed_total_ms"), "video_preprocess_ms + generate_ms."],
                ]
            )
    elif accounting.get("additive_formula"):
        rows.append(
            [
                "additive formula",
                accounting.get("additive_formula"),
                "Only this formula should be used to recompute total latency.",
            ]
        )
        rows.append(["additive fields", accounting.get("additive_fields"), "Fields that sum to total_ms."])

    for group_name, label in (
        ("nested_preprocess_breakdown_ms", "preprocess child"),
        ("nested_generate_breakdown_ms", "generate child"),
    ):
        group = as_mapping(accounting.get(group_name))
        for name, detail in group.items():
            detail_map = as_mapping(detail)
            rows.append(
                [
                    f"{label}: {name}",
                    detail_map.get("value"),
                    f"included_in={detail_map.get('included_in')}; add_to_total={detail_map.get('add_to_total_ms')}",
                ]
            )

    for field_name, label in (
        ("nested_preprocess_fields", "nested preprocess fields"),
        ("nested_generate_fields", "nested generate fields"),
        ("do_not_sum_with_total_ms", "do not sum with total_ms"),
    ):
        value = accounting.get(field_name)
        if value:
            rows.append([label, value, "Breakdown fields for diagnosis, not extra additive terms."])

    for note_name in ("decode_alias_note", "note"):
        if accounting.get(note_name):
            rows.append([note_name, accounting.get(note_name), ""])

    sections.append("### Accounting Fields\n\n" + markdown_table(["Field", "Value", "Meaning"], rows))
    return "\n\n".join(sections)


def video_info(payload: dict[str, Any]) -> dict[str, Any]:
    result = result_payload(payload)
    summary = as_mapping(result.get("video_input_summary"))
    source = as_mapping(payload.get("source_metadata"))
    effective = as_mapping(payload.get("effective_video"))
    estimate_video = as_mapping(get_path(payload, "estimate.video", {}))
    info: dict[str, Any] = dict(summary)
    source_width = summary.get("source_width") or source.get("width")
    source_height = summary.get("source_height") or source.get("height")
    processor_width = summary.get("processor_input_width") or effective.get("width") or estimate_video.get("width")
    processor_height = summary.get("processor_input_height") or effective.get("height") or estimate_video.get("height")
    info.update(
        {
            "source_frames": summary.get("source_frames") or source.get("frames") or estimate_video.get("source_frames"),
            "source_resolution": summary.get("source_resolution") or resolution(source_width, source_height),
            "source_fps": summary.get("source_fps") or source.get("fps"),
            "source_duration_seconds": summary.get("source_duration_seconds") or source.get("duration_seconds"),
            "processor_input_resolution": summary.get("processor_input_resolution")
            or resolution(processor_width, processor_height),
            "processor_input_width": processor_width,
            "processor_input_height": processor_height,
            "video_decode_strategy": summary.get("video_decode_strategy") or get_path(payload, "sampling.decode_strategy"),
            "video_decode_frames_read": summary.get("video_decode_frames_read")
            or get_path(payload, "sampling.decode_frames_read"),
        }
    )
    return info


def render_video_and_experiment_info(payload: dict[str, Any], source_path: str | None) -> str:
    info = video_info(payload)
    dataset = as_mapping(payload.get("dataset"))
    metadata = as_mapping(payload.get("metadata"))
    rows = [
        ["source_report", source_path],
        ["report_kind", detect_report_kind(payload)],
        ["model_path", payload.get("model_path") or get_path(payload, "result.model_path")],
        ["autogaze_model", payload.get("autogaze_model") or get_path(payload, "result.autogaze_model")],
        ["gazing_mode", payload.get("gazing_mode") or get_path(payload, "result.gazing_mode")],
        ["video", payload.get("video") or info.get("resolved_video")],
        ["video_root", dataset.get("video_root")],
        ["dataset_dir", dataset.get("dataset_dir")],
        ["manifest", dataset.get("manifest")],
        ["source_frames", info.get("source_frames")],
        ["source_resolution", info.get("source_resolution")],
        ["processor_input_resolution", info.get("processor_input_resolution")],
        ["fps", info.get("source_fps")],
        ["duration_seconds", info.get("source_duration_seconds")],
        ["decode_strategy", info.get("video_decode_strategy") or get_path(payload, "sampling.decode_strategy")],
        ["decode_frames_read", info.get("video_decode_frames_read") or get_path(payload, "sampling.decode_frames_read")],
        ["device", metadata.get("device")],
        ["torch", metadata.get("torch")],
    ]
    return "## Video And Experiment Info\n\n" + markdown_table(
        ["Field", "Value"],
        [row for row in rows if row[1] is not None],
    )


def resolution(width: Any, height: Any) -> str | None:
    if width is None or height is None:
        return None
    return f"{width}x{height}"


def render_pipeline_section(payload: dict[str, Any]) -> str:
    note = ""
    if payload.get("mode") == "stream-profile":
        note = "\n\nNote: this stream-profile report measures pre-LLM stages. Public NVILA generation still consumes the collected visual tokens."
    return f"## Model Pipeline\n\n```text\n{PIPELINE_ASCII}\n```{note}"


def render_step_pipeline_metrics(payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    info = video_info(payload)
    tokens = as_mapping(metrics.get("tokens"))
    latency = as_mapping(metrics.get("latency_ms"))
    memory = as_mapping(metrics.get("memory_bytes"))
    rows = [
        [
            "1",
            "Video decode/sample",
            f"{format_value(first_present(info.get('source_frames'), get_path(payload, 'sampling.num_video_frames')))} frames; "
            f"{first_present(info.get('source_resolution'), resolution(info.get('width'), info.get('height')), '-')}",
            f"sampled={format_value(tokens.get('video_sampled_frames'))}; thumbnail={format_value(tokens.get('thumbnail_sampled_frames'))}",
            first_present(latency.get("preprocess_total_ms"), latency.get("preprocess_total_median")),
            first_present(memory.get("raw_frame_buffer_peak"), memory.get("processor_peak"), memory.get("processor_peak_median")),
        ],
        [
            "2",
            "Resize / tile / thumbnail",
            info.get("processor_input_resolution") or resolution(info.get("width"), info.get("height")),
            f"spatial_tiles={format_value(info.get('spatial_tiles_per_video'))}; chunks={format_value(info.get('temporal_chunks_per_video'))}",
            latency.get("preprocess_total_ms") or latency.get("preprocess_total_median"),
            memory.get("processor_peak") or memory.get("processor_peak_median"),
        ],
        [
            "3",
            "AutoGaze ON/OFF",
            f"patch input={format_value(tokens.get('autogaze_input_tile_patch_tokens'))}",
            f"selected={format_value(tokens.get('autogaze_selected_tile_patch_tokens'))}; ratio={format_value(tokens.get('autogaze_patch_reduction_ratio'))}",
            latency.get("autogaze_ms") or latency.get("autogaze_median"),
            memory.get("autogaze_tile_tensor_peak_per_temporal_chunk"),
        ],
        [
            "4",
            "SigLIP / ViT Encoder",
            f"encoder before={format_value(tokens.get('encoder_patch_tokens_before_keep_all_or_raw'))}",
            f"encoder after={format_value(tokens.get('encoder_patch_tokens_after_autogaze'))}; ratio={format_value(tokens.get('encoder_token_reduction_ratio'))}",
            latency.get("vit_encoder_ms") or latency.get("vit_encoder_median"),
            memory.get("siglip_gazed_hidden_peak"),
        ],
        [
            "5",
            "TokenShuffle + MM projector",
            f"visual before={format_value(tokens.get('llm_visual_tokens_before_keep_all_estimated'))}",
            f"visual after={format_value(tokens.get('llm_visual_tokens_after_actual'))}; ratio={format_value(tokens.get('llm_visual_token_reduction_ratio'))}",
            first_present(get_path(payload, "latency_ms.mm_projector_median"), get_path(payload, "result.mm_projector_ms")),
            "-",
        ],
        [
            "6",
            "LLM prefill/generation",
            "prompt + visual tokens",
            first_present(payload.get("answer"), get_path(payload, "result.raw_output"), "benchmark scoring"),
            first_present(latency.get("llm_ms"), latency.get("llm_median")),
            first_present(memory.get("llm_peak"), memory.get("llm_peak_median")),
        ],
    ]
    return "## Step-by-step Pipeline Metrics\n\n" + markdown_table(
        ["Step", "Module", "Input / Output", "Counts / Tokens", "Latency ms", "Memory"],
        rows,
    )


def render_benchmark_score(payload: dict[str, Any]) -> str:
    if "keep_all" not in payload and "autogaze" not in payload:
        return ""
    rows = []
    for mode in ("keep_all", "autogaze"):
        accuracy = as_mapping(get_path(payload, f"{mode}.accuracy", {}))
        if accuracy:
            rows.append(
                [
                    mode,
                    accuracy.get("accuracy_scored"),
                    accuracy.get("accuracy_total"),
                    accuracy.get("correct"),
                    accuracy.get("scored"),
                    accuracy.get("failed"),
                    accuracy.get("parse_failed"),
                ]
            )
    if not rows:
        return ""
    return "## Benchmark Score\n\n" + markdown_table(
        ["Mode", "accuracy_scored", "accuracy_total", "correct", "scored", "failed", "parse_failed"],
        rows,
    )


def render_benchmark_samples(payload: dict[str, Any]) -> str:
    samples = get_path(payload, "benchmark_samples.autogaze", [])
    if not isinstance(samples, list) or not samples:
        return ""
    rows = []
    for sample in samples[:5]:
        if not isinstance(sample, dict):
            continue
        rows.append(
            [
                sample.get("target_video"),
                sample.get("question"),
                sample.get("model_answer"),
                sample.get("correct_answer"),
                sample.get("correct"),
                sample.get("status"),
            ]
        )
    if not rows:
        return ""
    return "## Benchmark Samples\n\n" + markdown_table(
        ["Video", "Question", "Model answer", "Correct answer", "Correct", "Status"],
        rows,
    )


def render_correctness_comparison(payload: dict[str, Any]) -> str:
    comparison = as_mapping(payload.get("correctness_comparison"))
    if not comparison:
        return ""
    counts = as_mapping(comparison.get("counts"))
    paired_rates = as_mapping(comparison.get("paired_rates"))
    count_rows = []
    for bucket in (
        "total_unique",
        "paired",
        "both_correct",
        "keep_all_only_correct",
        "autogaze_only_correct",
        "both_wrong",
        "keep_all_missing",
        "autogaze_missing",
    ):
        if bucket not in counts:
            continue
        count_rows.append([bucket, counts.get(bucket), paired_rates.get(bucket)])

    sample_rows = []
    samples = comparison.get("samples")
    if isinstance(samples, list):
        for sample in samples[:10]:
            if not isinstance(sample, dict):
                continue
            sample_rows.append(
                [
                    sample.get("target_video"),
                    sample.get("question"),
                    sample.get("correct_answer"),
                    sample.get("keep_all_answer"),
                    sample.get("keep_all_correct"),
                    sample.get("autogaze_answer"),
                    sample.get("autogaze_correct"),
                    sample.get("bucket"),
                ]
            )

    sections = ["## Benchmark Correctness Comparison"]
    if count_rows:
        sections.append(markdown_table(["Bucket", "Count", "Paired rate"], count_rows))
    if sample_rows:
        sections.append(
            markdown_table(
                [
                    "Video",
                    "Question",
                    "Correct answer",
                    "Keep-all answer",
                    "Keep-all correct",
                    "AutoGaze answer",
                    "AutoGaze correct",
                    "Bucket",
                ],
                sample_rows,
            )
        )
    return "\n\n".join(sections)


def is_mode_comparison_metric(value: Any) -> bool:
    return isinstance(value, dict) and ("keep_all" in value or "autogaze" in value)


def render_module_details(payload: dict[str, Any]) -> str:
    detail = get_path(payload, "readable_performance_summary.latency_ms_detail_median")
    if detail is None:
        detail = get_path(payload, "readable_summary.latency_ms_detail_median")
    if detail is None:
        detail = payload.get("stage_timings_ms")
    if not isinstance(detail, dict):
        return ""

    if any(is_mode_comparison_metric(value) for value in detail.values()):
        rows = []
        for name, value in detail.items():
            if is_mode_comparison_metric(value):
                rows.append(
                    [
                        name,
                        value.get("keep_all"),
                        value.get("autogaze"),
                        first_present(
                            value.get("speedup_ratio_keep_all_over_autogaze"),
                            value.get("reduction_ratio_keep_all_over_autogaze"),
                        ),
                        value.get("reduction_percent_of_keep_all"),
                    ]
                )
            else:
                rows.append([name, value, None, None, None])
        return "## Module Detail Metrics\n\n" + markdown_table(
            ["Metric", "Keep-all", "AutoGaze", "Speedup", "Reduction %"],
            rows,
        )

    rows = []
    for name, value in detail.items():
        if isinstance(value, dict):
            rows.append([name, value.get("total_ms") or value.get("median") or value.get("autogaze"), value.get("count")])
        else:
            rows.append([name, value, "-"])
    return "## Module Detail Metrics\n\n" + markdown_table(["Metric", "Value", "Count"], rows)


def render_input_tokenization(payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    tokens = as_mapping(metrics.get("tokens"))
    stream_tokens = as_mapping(get_path(payload, "stream_plan.tokens", {}))
    rows = [
        ["video_sampled_frames", tokens.get("video_sampled_frames")],
        ["thumbnail_sampled_frames", tokens.get("thumbnail_sampled_frames")],
        ["encoder_patches_per_frame_multiscale", stream_tokens.get("encoder_patches_per_frame_multiscale")],
        ["patches_per_frame_by_scale", stream_tokens.get("encoder_patches_per_frame_by_scale")],
        ["encoder_patch_tokens_before_keep_all_or_raw", tokens.get("encoder_patch_tokens_before_keep_all_or_raw")],
        ["encoder_patch_tokens_after_autogaze", tokens.get("encoder_patch_tokens_after_autogaze")],
        ["llm_visual_tokens_before_keep_all_estimated", tokens.get("llm_visual_tokens_before_keep_all_estimated")],
        ["llm_visual_tokens_after_actual", tokens.get("llm_visual_tokens_after_actual")],
    ]
    return "## Frame, Patch, And Tokenization Info\n\n" + markdown_table(
        ["Field", "Value"],
        [row for row in rows if row[1] is not None],
    )


def render_processing_budget_summary(payload: dict[str, Any]) -> str:
    summary = processing_budget_summary(payload)
    readable_budget = readable_processing_budget_summary(payload)
    if readable_budget:
        keep_all = as_mapping(readable_budget.get("keep_all_median") or readable_budget.get("mode_median"))
        autogaze = as_mapping(readable_budget.get("autogaze_median"))
        rows = []
        fields = readable_budget.get("fields") or sorted(set(keep_all) | set(autogaze))
        for field in fields:
            rows.append([field, keep_all.get(field), autogaze.get(field)])
        return "## Processing Budget Summary\n\n" + markdown_table(
            ["Field", "Keep-all / Mode", "AutoGaze"],
            [row for row in rows if row[1] is not None or row[2] is not None],
        )
    if not summary:
        return ""
    video = as_mapping(summary.get("video"))
    model_unit = as_mapping(summary.get("model_processing_unit"))
    tiling = as_mapping(summary.get("tiling"))
    thumbnail = as_mapping(summary.get("thumbnail"))
    single_scale = as_mapping(summary.get("single_scale_dense_vision_budget"))
    multiscale = as_mapping(summary.get("multiscale_patch_space"))
    patch_budget = as_mapping(summary.get("patch_budget_before_siglip") or summary.get("patch_budget_before_vit"))
    llm_budget = as_mapping(summary.get("llm_visual_budget"))
    rows = [
        ["source_resolution", video.get("source_resolution")],
        ["processor_input_resolution", video.get("processor_input_resolution")],
        ["requested_video_frames", video.get("requested_video_frames")],
        ["actual_video_frames", video.get("actual_video_frames")],
        ["runner_resize", video.get("resize_request") or video.get("runner_resize")],
        ["model_processing_unit", model_unit.get("name")],
        ["tile_size_px", model_unit.get("tile_size_px")],
        ["spatial_tiles_per_frame", tiling.get("spatial_tiles_per_frame") or tiling.get("spatial_chunks_per_frame_limit")],
        ["tile_frame_instances", tiling.get("tile_frame_instances")],
        ["thumbnail_enabled", thumbnail.get("enabled")],
        ["thumbnail_actual_frames", thumbnail.get("actual_frames") or thumbnail.get("effective_frames")],
        ["thumbnail_policy", thumbnail.get("policy") or thumbnail.get("pruning_policy")],
        ["single_scale_dense_scope", single_scale.get("comparison_scope")],
        [
            "single_scale_dense_patch_positions_per_tile_frame",
            single_scale.get("patch_positions_per_tile_frame")
            or single_scale.get("patch_positions_per_reference_tile_frame"),
        ],
        [
            "single_scale_dense_total_patch_tokens",
            single_scale.get("total_patch_tokens") or single_scale.get("estimated_total_patch_tokens"),
        ],
        [
            "single_scale_dense_llm_visual_tokens_estimated",
            single_scale.get("llm_visual_tokens_estimated")
            or single_scale.get("estimated_llm_visual_tokens_after_token_shuffle"),
        ],
        ["multiscale_patch_positions_per_tile_frame", multiscale.get("patch_positions_per_tile_frame")],
        ["patch_positions_by_scale", multiscale.get("patch_positions_by_scale")],
        ["keep_all_total_patch_tokens", patch_budget.get("keep_all_total_patch_tokens") or patch_budget.get("estimated_visual_tokens_before_prune")],
        ["autogaze_selected_total_patch_tokens", patch_budget.get("autogaze_selected_total_patch_tokens") or patch_budget.get("estimated_visual_tokens_after_prune")],
        ["total_patch_reduction_ratio", patch_budget.get("total_patch_reduction_ratio") or patch_budget.get("estimated_visual_token_reduction_ratio")],
        ["llm_keep_all_visual_tokens_estimated", llm_budget.get("keep_all_visual_tokens_estimated")],
        ["llm_actual_visual_tokens", llm_budget.get("actual_visual_tokens")],
        ["llm_visual_token_reduction_ratio", llm_budget.get("visual_token_reduction_ratio")],
    ]
    return "## Processing Budget Summary\n\n" + markdown_table(
        ["Field", "Value"],
        [row for row in rows if row[1] is not None],
    )


def render_autogaze_token_patch_flow(payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    readable_budget = readable_processing_budget_summary(payload)
    if readable_budget:
        return render_readable_budget_token_flow(readable_budget)

    summary = processing_budget_summary(payload)
    if not summary:
        return ""
    return render_single_budget_token_flow(summary, metrics)


def render_readable_budget_token_flow(readable_budget: dict[str, Any]) -> str:
    keep_all = as_mapping(readable_budget.get("keep_all_median") or readable_budget.get("mode_median"))
    autogaze = as_mapping(readable_budget.get("autogaze_median"))
    rows: list[list[Any]] = []

    def add_row(label: str, off_field: str, auto_baseline_field: str, auto_actual_field: str) -> None:
        off_baseline = get_budget_value(keep_all, off_field)
        autogaze_baseline = get_budget_value(autogaze, auto_baseline_field)
        autogaze_actual = get_budget_value(autogaze, auto_actual_field)
        rows.append(
            [
                label,
                off_baseline,
                autogaze_baseline,
                autogaze_actual,
                ratio_before_over_after(autogaze_baseline, autogaze_actual),
                reduction_percent(autogaze_baseline, autogaze_actual),
            ]
        )

    add_row(
        "Full multiscale patch budget before AutoGaze",
        "patch_budget_before_siglip.keep_all_total_patch_tokens",
        "patch_budget_before_siglip.keep_all_total_patch_tokens",
        "patch_budget_before_siglip.autogaze_selected_total_patch_tokens",
    )
    add_row(
        "Single-scale dense SigLIP reference",
        "single_scale_dense_vision_budget.total_patch_tokens",
        "single_scale_dense_vision_budget.total_patch_tokens",
        "patch_budget_before_siglip.autogaze_selected_total_patch_tokens",
    )
    add_row(
        "Tile patch budget before SigLIP",
        "patch_budget_before_siglip.keep_all_tile_patch_tokens",
        "patch_budget_before_siglip.keep_all_tile_patch_tokens",
        "patch_budget_before_siglip.autogaze_selected_tile_patch_tokens",
    )
    add_row(
        "Thumbnail patch budget before SigLIP",
        "patch_budget_before_siglip.keep_all_thumbnail_patch_tokens",
        "patch_budget_before_siglip.keep_all_thumbnail_patch_tokens",
        "patch_budget_before_siglip.autogaze_selected_thumbnail_patch_tokens",
    )
    add_row(
        "Encoder input patch tokens after AutoGaze",
        "patch_budget_before_siglip.keep_all_total_patch_tokens",
        "patch_budget_before_siglip.keep_all_total_patch_tokens",
        "patch_budget_before_siglip.autogaze_selected_total_patch_tokens",
    )
    add_row(
        "LLM visual tokens after TokenShuffle/projector",
        "llm_visual_budget.keep_all_visual_tokens_estimated",
        "llm_visual_budget.keep_all_visual_tokens_estimated",
        "llm_visual_budget.actual_visual_tokens",
    )

    rows = [row for row in rows if any(value is not None for value in row[1:4])]
    if not rows:
        return ""
    note = (
        "When an AutoGaze-off run is not present, the AutoGaze-off baseline column is still estimated "
        "from the AutoGaze-on input shape. Latency speedups still require both modes to be measured."
    )
    return (
        "## AutoGaze Token And Patch Flow\n\n"
        + note
        + "\n\n"
        + markdown_table(
            [
                "Stage",
                "Measured off / keep-all",
                "Off estimate for AutoGaze input",
                "AutoGaze-on actual",
                "Reduction ratio",
                "Reduction %",
            ],
            rows,
        )
    )


def render_single_budget_token_flow(summary: dict[str, Any], metrics: dict[str, Any]) -> str:
    tokens = as_mapping(metrics.get("tokens"))
    single_scale = as_mapping(summary.get("single_scale_dense_vision_budget"))
    single_scale_patch_tokens = first_present(
        single_scale.get("total_patch_tokens"),
        single_scale.get("estimated_total_patch_tokens"),
    )
    single_scale_llm_tokens = first_present(
        single_scale.get("llm_visual_tokens_estimated"),
        single_scale.get("estimated_llm_visual_tokens_after_token_shuffle"),
    )
    patch_budget = as_mapping(summary.get("patch_budget_before_siglip"))
    if patch_budget:
        full_patch = first_present(
            patch_budget.get("keep_all_total_patch_tokens"),
            tokens.get("encoder_patch_tokens_before_keep_all_or_raw"),
        )
        selected_patch = first_present(
            patch_budget.get("autogaze_selected_total_patch_tokens"),
            tokens.get("encoder_patch_tokens_after_autogaze"),
        )
        tile_full = patch_budget.get("keep_all_tile_patch_tokens")
        tile_selected = patch_budget.get("autogaze_selected_tile_patch_tokens")
        thumbnail_full = patch_budget.get("keep_all_thumbnail_patch_tokens")
        thumbnail_selected = patch_budget.get("autogaze_selected_thumbnail_patch_tokens")
        llm_budget = as_mapping(summary.get("llm_visual_budget"))
        llm_before = first_present(
            llm_budget.get("keep_all_visual_tokens_estimated"),
            tokens.get("llm_visual_tokens_before_keep_all_estimated"),
        )
        llm_after = first_present(
            llm_budget.get("actual_visual_tokens"),
            tokens.get("llm_visual_tokens_after_actual"),
        )
        rows = [
            [
                "full_multiscale_patch_budget_before_autogaze",
                full_patch,
                "All multiscale tile plus thumbnail patch positions that keep-all/off would send toward SigLIP.",
            ],
            [
                "single_scale_dense_siglip_reference_patch_tokens",
                single_scale_patch_tokens,
                "Reference dense SigLIP budget using only 392px scale with patch size 14, so 784 positions per tile-frame.",
            ],
            [
                "single_scale_dense_siglip_reference_llm_visual_tokens_estimated",
                single_scale_llm_tokens,
                "Reference visual-token estimate after TokenShuffle for the single-scale dense SigLIP budget.",
            ],
            [
                "autogaze_selected_patch_tokens",
                selected_patch,
                "Non-padded AutoGaze-selected patch positions, including keep-all thumbnail positions when enabled.",
            ],
            [
                "encoder_input_patch_tokens_after_autogaze",
                selected_patch,
                "The patch-token budget expected at the sparse SigLIP/ViT input boundary for AutoGaze-on.",
            ],
            [
                "tile_patch_tokens_before_to_after",
                f"{format_value(tile_full)} -> {format_value(tile_selected)}",
                "Main video tile patch budget before/after AutoGaze selection.",
            ],
            [
                "thumbnail_patch_tokens_before_to_after",
                f"{format_value(thumbnail_full)} -> {format_value(thumbnail_selected)}",
                "Thumbnail patch budget; NVILA runner keeps thumbnails all-on unless disabled.",
            ],
            [
                "llm_input_visual_tokens_after_token_shuffle_projector",
                llm_after,
                "Visual tokens that the LLM context sees after TokenShuffle/projector packing.",
            ],
            [
                "llm_visual_tokens_keep_all_estimated",
                llm_before,
                "Estimated visual-token baseline for AutoGaze-off/keep-all at the same input shape.",
            ],
            [
                "patch_reduction_ratio_before_over_after",
                ratio_before_over_after(full_patch, selected_patch),
                "Full patch budget divided by AutoGaze-selected patch budget.",
            ],
            [
                "single_scale_dense_reference_reduction_ratio_before_over_after",
                ratio_before_over_after(single_scale_patch_tokens, selected_patch),
                "Single-scale dense SigLIP reference patch budget divided by AutoGaze-selected patch budget.",
            ],
            [
                "llm_visual_token_reduction_ratio_before_over_after",
                ratio_before_over_after(llm_before, llm_after),
                "Estimated keep-all LLM visual tokens divided by actual AutoGaze LLM visual tokens.",
            ],
        ]
    else:
        patch_budget = as_mapping(summary.get("patch_budget_before_vit"))
        full_patch = first_present(
            patch_budget.get("actual_raw_patch_tokens_before_vit"),
            patch_budget.get("estimated_visual_tokens_before_prune"),
            tokens.get("visual_tokens_before_prune"),
        )
        selected_patch = first_present(
            patch_budget.get("estimated_visual_tokens_after_prune"),
            tokens.get("visual_tokens_after_prune"),
        )
        rows = [
            [
                "single_scale_dense_siglip_reference_patch_tokens",
                single_scale_patch_tokens,
                "Reference-only dense SigLIP budget; exact for 392px SigLIP tile pipelines, approximate for other adapters.",
            ],
            [
                "full_patch_budget_before_selector",
                full_patch,
                "Raw ViT/Qwen grid token budget before pre-encoder pruning or selector masking.",
            ],
            [
                "autogaze_selected_patch_tokens",
                selected_patch,
                "Estimated/actual selected visual patch tokens after the selector.",
            ],
            [
                "encoder_input_patch_tokens_after_autogaze",
                selected_patch,
                "The sparse ViT input token budget when the pre-ViT selector path is enabled.",
            ],
            [
                "llm_input_context_tokens",
                tokens.get("llm_context_tokens"),
                "Total MLLM context length measured after processor packing when available.",
            ],
            [
                "patch_reduction_ratio_before_over_after",
                ratio_before_over_after(full_patch, selected_patch),
                "Raw visual patch budget divided by selected visual patch budget.",
            ],
            [
                "single_scale_dense_reference_reduction_ratio_before_over_after",
                ratio_before_over_after(single_scale_patch_tokens, selected_patch),
                "Reference dense SigLIP patch budget divided by selected visual patch budget.",
            ],
        ]

    rows = [row for row in rows if row[1] is not None and row[1] != "- -> -"]
    if not rows:
        return ""
    return "## AutoGaze Token And Patch Flow\n\n" + markdown_table(["Field", "Value", "Meaning"], rows)


def render_markdown_report(
    payload: dict[str, Any],
    *,
    source_path: str | None = None,
    title: str = "AutoGaze Reproduction Report",
    chart_artifacts: list[ChartArtifact] | None = None,
    markdown_dir: str | Path | None = None,
) -> str:
    metrics = enriched_key_metrics(payload, key_metrics(payload))
    sections = [
        f"# {title}",
        render_video_and_experiment_info(payload, source_path),
        render_pipeline_section(payload),
        render_input_tokenization(payload, metrics),
        render_processing_budget_summary(payload),
        render_autogaze_token_patch_flow(payload, metrics),
        render_step_pipeline_metrics(payload, metrics),
        render_charts_section(chart_artifacts or [], markdown_dir=Path(markdown_dir) if markdown_dir else None),
        render_key_metrics_section(metrics),
        render_latency_accounting_section(payload, metrics),
        render_benchmark_score(payload),
        render_correctness_comparison(payload),
        render_benchmark_samples(payload),
        render_module_details(payload),
    ]
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def write_markdown_report(
    input_json: str | Path,
    output_md: str | Path,
    *,
    title: str = "AutoGaze Reproduction Report",
    include_charts: bool = True,
    charts_dir: str | Path | None = None,
) -> None:
    payload = load_json(input_json)
    target = Path(output_md)
    target.parent.mkdir(parents=True, exist_ok=True)
    chart_artifacts: list[ChartArtifact] = []
    if include_charts:
        metrics = enriched_key_metrics(payload, key_metrics(payload))
        chart_output_dir = Path(charts_dir) if charts_dir is not None else target.parent / f"{target.stem}_assets"
        chart_artifacts = build_standard_report_charts(metrics=metrics, output_dir=chart_output_dir)
    markdown = render_markdown_report(
        payload,
        source_path=str(input_json),
        title=title,
        chart_artifacts=chart_artifacts,
        markdown_dir=target.parent,
    )
    target.write_text(markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render AutoGaze/NVILA JSON outputs as Markdown reports.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--title", default="AutoGaze Reproduction Report")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--charts-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    write_markdown_report(
        args.input_json,
        args.output_md,
        title=args.title,
        include_charts=not args.no_charts,
        charts_dir=args.charts_dir,
    )
    print(str(args.output_md))


if __name__ == "__main__":
    main()
