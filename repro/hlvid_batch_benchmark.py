from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from repro.common import compute_stats, write_csv, write_json
from repro.hlvid import read_jsonl, read_manifest_file, score_predictions


MANIFEST_PATTERNS = (
    "manifest*.json",
    "manifest*.jsonl",
    "*test*.json",
    "*test*.jsonl",
    "metadata.jsonl",
    "*.parquet",
    "*.csv",
    "data/test-*.parquet",
    "data/*.parquet",
    "data/*.json",
    "data/*.jsonl",
    "data/*.csv",
    "default/test/*.parquet",
)
VIDEO_ROOT_CANDIDATES = (
    "videos",
    "video",
    "videos_extracted",
    "extracted",
    "example",
    ".",
)
VIDEO_ARCHIVE_PATTERN = "videos_part_*.tar"
CORRECTNESS_COMPARISON_SAMPLE_LIMIT = 20

LATENCY_FIELDS = (
    "total_ms",
    "generate_ms",
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
    "generation_decode_after_ttft_estimated_ms",
)
MODULE_LATENCY_FIELDS = (
    ("total_ms", "total_ms"),
    ("preprocess_total_ms", "video_preprocess_ms"),
    ("autogaze_ms", "autogaze_ms"),
    ("vit_encoder_ms", "siglip_vision_ms"),
    ("llm_ms", "llm_forward_ms"),
)
KEY_MEMORY_FIELDS = (
    ("processor_peak", "processor_peak_memory_bytes"),
    ("ttft_peak", "ttft_peak_memory_bytes"),
    ("llm_peak", "llm_peak_memory_bytes"),
    ("overall_peak", "peak_memory_bytes"),
)
MEMORY_FIELDS = (
    "processor_peak_memory_bytes",
    "ttft_peak_memory_bytes",
    "llm_peak_memory_bytes",
    "peak_memory_bytes",
)
AUTOGAZE_TOKEN_FIELDS = (
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
    "token_metrics.encoder_tile_token_reduction_ratio",
    "token_metrics.llm_visual_token_reduction_ratio",
    "token_metrics.llm_actual_visual_tokens",
    "token_metrics.llm_keep_all_visual_tokens_estimated",
)
COMPUTE_FIELDS = (
    "compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio",
    "compute_metrics.mllm.kv_cache_reduction_ratio",
    "compute_metrics.mllm.prefill_attention_pair_reduction_ratio",
    "compute_metrics.mllm.prefill_total_macs_reduction_ratio",
)


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _metric(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _metric(row, field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _stats_by_field(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {field: compute_stats(_numeric_values(rows, field)) for field in fields}


def _median_ratio(
    numerator_rows: list[dict[str, Any]],
    denominator_rows: list[dict[str, Any]],
    field: str,
) -> float | None:
    numerator_values = _numeric_values(numerator_rows, field)
    denominator_values = _numeric_values(denominator_rows, field)
    if not numerator_values or not denominator_values:
        return None
    numerator = compute_stats(numerator_values)["median"]
    denominator = compute_stats(denominator_values)["median"]
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _median_value(rows: list[dict[str, Any]], field: str) -> float | None:
    return compute_stats(_numeric_values(rows, field))["median"]


def _percent_reduction(before: float | None, after: float | None) -> float | None:
    if before in {None, 0} or after is None:
        return None
    return 100.0 * (float(before) - float(after)) / float(before)


def _comparison_summary(
    keep_all_rows: list[dict[str, Any]],
    autogaze_rows: list[dict[str, Any]],
    field: str,
    *,
    ratio_key: str,
) -> dict[str, float | None]:
    keep_all_value = _median_value(keep_all_rows, field)
    autogaze_value = _median_value(autogaze_rows, field)
    return {
        "keep_all": keep_all_value,
        "autogaze": autogaze_value,
        ratio_key: _median_ratio(keep_all_rows, autogaze_rows, field),
        "reduction_percent_of_keep_all": _percent_reduction(keep_all_value, autogaze_value),
    }


def _autogaze_before_after_summary(
    rows: list[dict[str, Any]],
    before_field: str,
    after_field: str,
    *,
    before_key: str,
    after_key: str,
    extra_fields: dict[str, str] | None = None,
) -> dict[str, float | None]:
    before = _median_value(rows, before_field)
    after = _median_value(rows, after_field)
    summary: dict[str, float | None] = {
        before_key: before,
        after_key: after,
        "reduction_ratio_before_over_after": _median_ratio(rows, rows, before_field)
        if before_field == after_field
        else None,
        "reduction_percent_of_before": _percent_reduction(before, after),
    }
    if after not in {None, 0} and before is not None:
        summary["reduction_ratio_before_over_after"] = float(before) / float(after)
    for key, field in (extra_fields or {}).items():
        summary[key] = _median_value(rows, field)
    return summary


def build_readable_summary(
    *,
    keep_all_rows: list[dict[str, Any]],
    autogaze_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    latency_detail = {
        field: _comparison_summary(
            keep_all_rows,
            autogaze_rows,
            field,
            ratio_key="speedup_ratio_keep_all_over_autogaze",
        )
        for field in LATENCY_FIELDS
    }
    latency = {
        label: _comparison_summary(
            keep_all_rows,
            autogaze_rows,
            field,
            ratio_key="speedup_ratio_keep_all_over_autogaze",
        )
        for label, field in MODULE_LATENCY_FIELDS
    }
    memory = {
        field: _comparison_summary(
            keep_all_rows,
            autogaze_rows,
            field,
            ratio_key="reduction_ratio_keep_all_over_autogaze",
        )
        for field in MEMORY_FIELDS
    }
    key_memory = {
        label: _comparison_summary(
            keep_all_rows,
            autogaze_rows,
            field,
            ratio_key="reduction_ratio_keep_all_over_autogaze",
        )
        for label, field in KEY_MEMORY_FIELDS
    }
    tokens = {
        "encoder_patches": _autogaze_before_after_summary(
            autogaze_rows,
            "token_metrics.encoder_raw_patch_tokens",
            "token_metrics.encoder_autogaze_selected_patch_tokens",
            before_key="before_keep_all_or_raw",
            after_key="after_autogaze",
        ),
        "autogaze_input_tile_patches": _autogaze_before_after_summary(
            autogaze_rows,
            "token_metrics.autogaze_input_patch_tokens",
            "token_metrics.autogaze_selected_patch_tokens",
            before_key="before_autogaze_selection",
            after_key="after_autogaze_selection",
            extra_fields={
                "tile_frame_instances": "token_metrics.autogaze_input_tile_frame_instances",
            },
        ),
        "llm_visual_tokens": _autogaze_before_after_summary(
            autogaze_rows,
            "token_metrics.llm_keep_all_visual_tokens_estimated",
            "token_metrics.llm_actual_visual_tokens",
            before_key="before_keep_all_estimated",
            after_key="after_autogaze_actual",
        ),
    }
    return {
        "mode_status": {
            "keep_all": "available" if keep_all_rows else "skipped_or_missing",
            "autogaze": "available" if autogaze_rows else "skipped_or_missing",
            "note": (
                "A skipped/missing mode is still shown, but cross-mode ratios are null because no baseline rows exist."
            ),
        },
        "run_counts": {
            "keep_all_rows": len(keep_all_rows),
            "autogaze_rows": len(autogaze_rows),
            "count_note": (
                "Counts are prediction rows per mode. With --limit 3 and both modes enabled, "
                "expect keep_all_rows=3 and autogaze_rows=3; warmup runs are not counted."
            ),
        },
        "latency_ms_median": latency,
        "latency_ms_detail_median": latency_detail,
        "key_metrics_median": {
            "latency_ms": latency,
            "tokens": tokens,
            "memory_bytes": key_memory,
        },
        "latency_accounting": {
            "additive_formula": "total_ms = video_preprocess_ms + generate_ms",
            "additive_fields": ["video_preprocess_ms", "generate_ms"],
            "nested_preprocess_fields": [
                "video_decode_ms",
                "video_tiling_ms",
                "autogaze_ms",
                "gazing_info_total_ms",
                "autogaze_forward_ms",
                "autogaze_model_forward_ms",
            ],
            "nested_generate_fields": [
                "vision_encoder_ms",
                "siglip_vision_ms",
                "mm_projector_ms",
                "llm_forward_ms",
                "ttft_ms",
                "generation_decode_after_ttft_estimated_ms",
            ],
            "decode_alias_note": (
                "generation_decode_after_ttft_estimated_ms is generate_ms - ttft_ms. "
                "It is generation decode time, not video decode time."
            ),
            "do_not_sum_with_total_ms": [
                field
                for field in LATENCY_FIELDS
                if field not in {"total_ms", "video_preprocess_ms", "generate_ms"}
            ],
        },
        "latency_field_note": (
            "Summary-level latency is intentionally coarse: "
            "preprocess_total=video_preprocess_ms, autogaze=autogaze_ms, "
            "vit_encoder=siglip_vision_ms, llm=llm_forward_ms. "
            "These fields are not additive because preprocess_total includes processor work. "
            "Use latency_accounting.additive_formula for the only additive total formula, "
            "and use latency_ms_detail_median or per-mode latency_ms for finer breakdowns."
        ),
        "memory_bytes_median": memory,
        "tokens_median": tokens,
        "ratio_note": (
            "Ratio fields use before_or_keep_all / after_or_autogaze. "
            "Percent fields use (before - after) / before * 100, so their denominator is the original value."
        ),
    }


def _reduction_percent_summary(readable_summary: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    return {
        "latency_ms": {
            field: summary.get("reduction_percent_of_keep_all")
            for field, summary in readable_summary.get("latency_ms_median", {}).items()
        },
        "memory_bytes": {
            field: summary.get("reduction_percent_of_keep_all")
            for field, summary in readable_summary.get("memory_bytes_median", {}).items()
        },
        "tokens": {
            field: summary.get("reduction_percent_of_before")
            for field, summary in readable_summary.get("tokens_median", {}).items()
        },
    }


def _candidate_manifest_paths(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in MANIFEST_PATTERNS:
        candidates.extend(sorted(root.glob(pattern)))
    # Preserve pattern priority while removing duplicates.
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _resolve_manifest_path(root: Path, manifest: str | Path | None = None) -> Path:
    if manifest is not None:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute() and not manifest_path.exists():
            manifest_path = root / manifest_path
        return manifest_path

    candidates = _candidate_manifest_paths(root)
    if not candidates:
        raise FileNotFoundError(
            f"No HLVid manifest found in {root}. Expected root manifest files or HF layout data/test-*.parquet."
        )
    return candidates[0]


def _video_candidates(video_root: Path, video_path: str) -> list[Path]:
    rel = Path(video_path)
    candidates = [video_root / rel]
    if rel.name != str(rel):
        candidates.append(video_root / rel.name)
    return candidates


def find_video_file(video_root: str | Path, video_path: str, *, recursive: bool = False) -> Path | None:
    root = Path(video_root)
    for candidate in _video_candidates(root, video_path):
        if candidate.exists():
            return candidate
    if recursive:
        matches = sorted(root.rglob(Path(video_path).name))
        return matches[0] if matches else None
    return None


def _count_available_videos(rows: list[dict[str, Any]], video_root: Path) -> int:
    unique = sorted({str(row["video_path"]) for row in rows})
    return sum(1 for video in unique if find_video_file(video_root, video) is not None)


def _discover_video_root(root: Path, rows: list[dict[str, Any]], video_root: str | Path | None = None) -> Path:
    if video_root is not None:
        candidate = Path(video_root)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = root / candidate
        return candidate

    scored: list[tuple[int, Path]] = []
    for name in VIDEO_ROOT_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            scored.append((_count_available_videos(rows, candidate), candidate))
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]
    return root


def discover_dataset_layout(
    dataset_dir: str | Path,
    manifest: str | Path | None = None,
    video_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = _resolve_manifest_path(root, manifest)
    rows = read_manifest_file(manifest_path)
    resolved_video_root = _discover_video_root(root, rows, video_root)
    archives = sorted(root.glob(VIDEO_ARCHIVE_PATTERN))
    return {
        "dataset_dir": root,
        "manifest": manifest_path,
        "video_root": resolved_video_root,
        "video_archives": archives,
    }


def build_prepare_report(
    dataset_dir: str | Path,
    manifest: str | Path | None = None,
    video_root: str | Path | None = None,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    layout = discover_dataset_layout(dataset_dir, manifest, video_root)
    rows = read_manifest_file(layout["manifest"])
    if limit is not None:
        rows = rows[:limit]
    unique_videos = sorted({str(row["video_path"]) for row in rows})
    found: list[str] = []
    missing: list[str] = []
    resolved_samples: dict[str, str] = {}
    for video in unique_videos:
        resolved = find_video_file(layout["video_root"], video, recursive=True)
        if resolved is None:
            missing.append(video)
        else:
            found.append(video)
            if len(resolved_samples) < 10:
                resolved_samples[video] = str(resolved)

    archives: list[Path] = layout.get("video_archives", [])
    return {
        "dataset_dir": str(layout["dataset_dir"]),
        "manifest": str(layout["manifest"]),
        "video_root": str(layout["video_root"]),
        "rows": len(rows),
        "unique_videos": len(unique_videos),
        "available_videos": len(found),
        "missing_videos": len(missing),
        "missing_video_samples": missing[:20],
        "resolved_video_samples": resolved_samples,
        "video_archives": [str(path) for path in archives],
        "video_archive_count": len(archives),
        "video_archive_total_bytes": sum(path.stat().st_size for path in archives if path.exists()),
        "hf_layout_detected": (Path(layout["dataset_dir"]) / "data").exists() or bool(archives),
        "ready_for_full_benchmark": len(missing) == 0,
        "note": (
            "HLVid HF snapshot stores metadata under data/test-*.parquet and videos as videos_part_*.tar. "
            "For full NVILA benchmark, extract the tar parts or pass --video-root to a directory containing mp4 files."
        ),
    }


def build_runner_command(
    args: argparse.Namespace,
    *,
    gazing_mode: str,
    manifest: str | Path,
    video_root: str | Path,
    predictions: str | Path,
    summary: str | Path,
    scored_predictions: str | Path,
) -> list[str]:
    command = [
        getattr(args, "python_executable", "python"),
        "-m",
        "repro.nvila_runner",
        "--mode",
        "hlvid",
        "--manifest",
        str(manifest),
        "--hlvid-video-root",
        str(video_root),
        "--model-path",
        str(args.model_path),
        "--autogaze-model",
        str(args.autogaze_model),
        "--device",
        str(args.device),
        "--device-map",
        str(args.device_map),
        "--gazing-mode",
        gazing_mode,
        "--num-video-frames",
        str(args.num_video_frames),
        "--num-video-frames-thumbnail",
        str(args.num_video_frames_thumbnail),
        "--max-tiles-video",
        str(args.max_tiles_video),
        "--max-batch-size-autogaze",
        str(args.max_batch_size_autogaze),
        "--max-batch-size-siglip",
        str(args.max_batch_size_siglip),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--task-loss-requirement-tile",
        str(args.task_loss_requirement_tile),
        "--predictions",
        str(predictions),
        "--summary",
        str(summary),
        "--scored-predictions",
        str(scored_predictions),
    ]
    if getattr(args, "limit", None) is not None:
        command.extend(["--limit", str(args.limit)])
    if getattr(args, "measure_ttft", False):
        command.append("--measure-ttft")
    if getattr(args, "continue_on_error", False):
        command.append("--continue-on-error")
    if getattr(args, "video_resize_shortest_edge", None) is not None:
        command.extend(["--video-resize-shortest-edge", str(args.video_resize_shortest_edge)])
    if getattr(args, "video_resize_longest_edge", None) is not None:
        command.extend(["--video-resize-longest-edge", str(args.video_resize_longest_edge)])
    if getattr(args, "video_resize_width", None) is not None:
        command.extend(["--video-resize-width", str(args.video_resize_width)])
    if getattr(args, "video_resize_height", None) is not None:
        command.extend(["--video-resize-height", str(args.video_resize_height)])
    if getattr(args, "video_decode_strategy", None) is not None:
        command.extend(["--video-decode-strategy", str(args.video_decode_strategy)])
    if getattr(args, "autogaze_target_scales", None) is not None:
        command.extend(["--autogaze-target-scales", str(args.autogaze_target_scales)])
    if getattr(args, "autogaze_target_patch_size", None) is not None:
        command.extend(["--autogaze-target-patch-size", str(args.autogaze_target_patch_size)])
    if getattr(args, "visualization_output_dir", None) is not None:
        command.extend(["--visualization-output-dir", str(args.visualization_output_dir)])
    if getattr(args, "visualization_fps", None) is not None:
        command.extend(["--visualization-fps", str(args.visualization_fps)])
    if getattr(args, "visualization_alpha", None) is not None:
        command.extend(["--visualization-alpha", str(args.visualization_alpha)])
    if getattr(args, "visualization_selected_max_long_side", None) is not None:
        command.extend(
            [
                "--visualization-selected-max-long-side",
                str(args.visualization_selected_max_long_side),
            ]
        )
    command.extend(getattr(args, "extra_runner_args", []) or [])
    return command


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy, _ = score_predictions(rows)
    return {
        "accuracy": accuracy,
        "latency_ms": _stats_by_field(rows, LATENCY_FIELDS),
        "memory_bytes": _stats_by_field(rows, MEMORY_FIELDS),
        "tokens": _stats_by_field(rows, AUTOGAZE_TOKEN_FIELDS),
        "compute": _stats_by_field(rows, COMPUTE_FIELDS),
    }


def _comparison_pair_key(row: dict[str, Any]) -> tuple[str, Any]:
    if row.get("question_id") is not None:
        return ("question_id", row.get("question_id"))
    if row.get("video_path") is not None or row.get("question") is not None:
        return ("video_question", f"{row.get('video_path')}::{row.get('question')}")
    return ("row", row.get("raw_output"))


def _sort_pair_key(key: tuple[str, Any]) -> tuple[str, int, Any]:
    key_type, value = key
    if key_type == "question_id":
        try:
            return (key_type, 0, int(value))
        except (TypeError, ValueError):
            return (key_type, 1, str(value))
    return (key_type, 1, str(value))


def _index_scored_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, Any], dict[str, Any]]:
    return {_comparison_pair_key(row): row for row in rows}


def _first_present_row(*rows: dict[str, Any] | None) -> dict[str, Any]:
    for row in rows:
        if row is not None:
            return row
    return {}


def _correctness_bucket(
    keep_all_row: dict[str, Any] | None,
    autogaze_row: dict[str, Any] | None,
) -> str:
    if keep_all_row is None:
        return "keep_all_missing"
    if autogaze_row is None:
        return "autogaze_missing"
    keep_all_correct = bool(keep_all_row.get("correct"))
    autogaze_correct = bool(autogaze_row.get("correct"))
    if keep_all_correct and autogaze_correct:
        return "both_correct"
    if keep_all_correct and not autogaze_correct:
        return "keep_all_only_correct"
    if autogaze_correct and not keep_all_correct:
        return "autogaze_only_correct"
    return "both_wrong"


def _paired_rate(count: int, paired: int) -> float | None:
    if paired == 0:
        return None
    return count / paired


def build_correctness_comparison(
    keep_all_rows: list[dict[str, Any]],
    autogaze_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _, keep_all_scored_rows = score_predictions(keep_all_rows)
    _, autogaze_scored_rows = score_predictions(autogaze_rows)
    keep_all_by_key = _index_scored_rows(keep_all_scored_rows)
    autogaze_by_key = _index_scored_rows(autogaze_scored_rows)
    keys = sorted(set(keep_all_by_key) | set(autogaze_by_key), key=_sort_pair_key)
    counts = {
        "total_unique": len(keys),
        "paired": 0,
        "both_correct": 0,
        "keep_all_only_correct": 0,
        "autogaze_only_correct": 0,
        "both_wrong": 0,
        "keep_all_missing": 0,
        "autogaze_missing": 0,
    }
    samples: list[dict[str, Any]] = []
    for key in keys:
        keep_all_row = keep_all_by_key.get(key)
        autogaze_row = autogaze_by_key.get(key)
        bucket = _correctness_bucket(keep_all_row, autogaze_row)
        counts[bucket] += 1
        if keep_all_row is not None and autogaze_row is not None:
            counts["paired"] += 1
        source_row = _first_present_row(keep_all_row, autogaze_row)
        if len(samples) >= CORRECTNESS_COMPARISON_SAMPLE_LIMIT:
            continue
        samples.append(
            {
                "bucket": bucket,
                "pair_key_type": key[0],
                "pair_key": key[1],
                "question_id": source_row.get("question_id"),
                "target_video": source_row.get("video_path", source_row.get("video")),
                "question": source_row.get("question", source_row.get("prompt")),
                "correct_answer": source_row.get("expected_answer"),
                "ground_truth_answer": source_row.get("answer"),
                "keep_all_answer": keep_all_row.get("raw_output") if keep_all_row else None,
                "keep_all_parsed_answer": keep_all_row.get("parsed_answer") if keep_all_row else None,
                "keep_all_correct": keep_all_row.get("correct") if keep_all_row else None,
                "keep_all_status": keep_all_row.get("status", "ok") if keep_all_row else "missing",
                "autogaze_answer": autogaze_row.get("raw_output") if autogaze_row else None,
                "autogaze_parsed_answer": autogaze_row.get("parsed_answer") if autogaze_row else None,
                "autogaze_correct": autogaze_row.get("correct") if autogaze_row else None,
                "autogaze_status": autogaze_row.get("status", "ok") if autogaze_row else "missing",
            }
        )
    paired = counts["paired"]
    return {
        "counts": counts,
        "paired_rates": {
            "both_correct": _paired_rate(counts["both_correct"], paired),
            "keep_all_only_correct": _paired_rate(counts["keep_all_only_correct"], paired),
            "autogaze_only_correct": _paired_rate(counts["autogaze_only_correct"], paired),
            "both_wrong": _paired_rate(counts["both_wrong"], paired),
        },
        "samples": samples,
        "sample_limit": CORRECTNESS_COMPARISON_SAMPLE_LIMIT,
        "note": (
            "Rows are paired by question_id when available, otherwise by video_path and question. "
            "Paired rates use only rows where both keep-all and AutoGaze outputs exist."
        ),
    }


def build_gain_report(
    *,
    keep_all_rows: list[dict[str, Any]],
    autogaze_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keep_all = summarize_run(keep_all_rows)
    autogaze = summarize_run(autogaze_rows)
    readable_summary = build_readable_summary(
        keep_all_rows=keep_all_rows,
        autogaze_rows=autogaze_rows,
    )
    correctness_comparison = build_correctness_comparison(
        keep_all_rows=keep_all_rows,
        autogaze_rows=autogaze_rows,
    )
    latency_speedups = {
        field: _median_ratio(keep_all_rows, autogaze_rows, field)
        for field in LATENCY_FIELDS
    }
    memory_reductions = {
        field: _median_ratio(keep_all_rows, autogaze_rows, field)
        for field in MEMORY_FIELDS
    }
    token_reductions = {
        field.split(".")[-1]: compute_stats(_numeric_values(autogaze_rows, field))["median"]
        for field in AUTOGAZE_TOKEN_FIELDS
    }
    compute_reductions = {
        "siglip_attention_macs": compute_stats(
            _numeric_values(
                autogaze_rows,
                "compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio",
            )
        )["median"],
        "siglip_mlp_macs": compute_stats(
            _numeric_values(
                autogaze_rows,
                "compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio",
            )
        )["median"],
        "siglip_total_macs": compute_stats(
            _numeric_values(
                autogaze_rows,
                "compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio",
            )
        )["median"],
        "mllm_kv_cache": compute_stats(
            _numeric_values(autogaze_rows, "compute_metrics.mllm.kv_cache_reduction_ratio")
        )["median"],
        "mllm_prefill_attention_pairs": compute_stats(
            _numeric_values(
                autogaze_rows,
                "compute_metrics.mllm.prefill_attention_pair_reduction_ratio",
            )
        )["median"],
    }
    accuracy_delta = (
        autogaze["accuracy"]["accuracy_scored"] - keep_all["accuracy"]["accuracy_scored"]
    )
    return {
        "keep_all": keep_all,
        "autogaze": autogaze,
        "readable_summary": readable_summary,
        "correctness_comparison": correctness_comparison,
        "benchmark_samples": {
            "keep_all": keep_all["accuracy"].get("benchmark_samples", []),
            "autogaze": autogaze["accuracy"].get("benchmark_samples", []),
            "correctness_comparison": correctness_comparison["samples"],
            "note": (
                "Readable per-sample benchmark context copied from the scoring summaries. "
                "Full per-row outputs are in hlvid_*_predictions.jsonl and hlvid_*_scored.jsonl."
            ),
        },
        "gains": {
            "accuracy_scored_delta": accuracy_delta,
            "latency_speedup_median": latency_speedups,
            "memory_reduction_ratio_median": memory_reductions,
            "autogaze_token_reduction_median": token_reductions,
            "compute_reduction_median": compute_reductions,
            "reduction_percent_median": _reduction_percent_summary(readable_summary),
        },
        "metric_note": (
            "Speedups and memory reductions are keep-all median divided by AutoGaze median. "
            "Token and compute reductions are read from AutoGaze rows because those rows contain both "
            "keep-all estimates and actual AutoGaze values."
        ),
    }


def flatten_metric_row(report: dict[str, Any]) -> dict[str, Any]:
    row = {
        "keep_all_accuracy_scored": report["keep_all"]["accuracy"].get("accuracy_scored"),
        "autogaze_accuracy_scored": report["autogaze"]["accuracy"].get("accuracy_scored"),
        "gain_accuracy_scored_delta": report["gains"].get("accuracy_scored_delta"),
    }
    for field, value in report["gains"].get("latency_speedup_median", {}).items():
        row[f"gain_latency_{field}_speedup_median"] = value
    for field, value in report["gains"].get("memory_reduction_ratio_median", {}).items():
        row[f"gain_memory_{field}_reduction_median"] = value
    for field, value in report["gains"].get("autogaze_token_reduction_median", {}).items():
        row[f"gain_token_{field}_median"] = value
    for field, value in report["gains"].get("compute_reduction_median", {}).items():
        row[f"gain_compute_{field}_median"] = value
    for field, value in report.get("correctness_comparison", {}).get("counts", {}).items():
        row[f"correctness_{field}"] = value
    return row


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "keep_all_predictions": root / "hlvid_keep_all_predictions.jsonl",
        "keep_all_summary": root / "hlvid_keep_all_summary.json",
        "keep_all_scored": root / "hlvid_keep_all_scored.jsonl",
        "autogaze_predictions": root / "hlvid_autogaze_predictions.jsonl",
        "autogaze_summary": root / "hlvid_autogaze_summary.json",
        "autogaze_scored": root / "hlvid_autogaze_scored.jsonl",
        "report_json": root / "hlvid_autogaze_gain_report.json",
        "report_csv": root / "hlvid_autogaze_gain_report.csv",
    }


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    layout = discover_dataset_layout(args.dataset_dir, args.manifest, args.video_root)
    paths = output_paths(args.output_dir)
    if args.prepare_only:
        report = build_prepare_report(
            args.dataset_dir,
            args.manifest,
            args.video_root,
            limit=args.limit,
        )
        output = Path(args.layout_report or (Path(args.output_dir) / "hlvid_dataset_layout_report.json"))
        write_json(output, report)
        return report

    if not args.report_only:
        prepare_report = build_prepare_report(
            args.dataset_dir,
            args.manifest,
            args.video_root,
            limit=args.limit,
        )
        if args.layout_report:
            write_json(args.layout_report, prepare_report)
        if prepare_report["missing_videos"] and not args.allow_missing_videos:
            raise FileNotFoundError(
                f"HLVid local video files are incomplete: {prepare_report['missing_videos']} missing "
                f"under {prepare_report['video_root']}. Run with --prepare-only to inspect the layout, "
                "extract videos_part_*.tar, pass --video-root, or use --allow-missing-videos intentionally."
            )

    if not args.report_only and not args.skip_keep_all:
        run_command(
            build_runner_command(
                args,
                gazing_mode="keep-all",
                manifest=layout["manifest"],
                video_root=layout["video_root"],
                predictions=paths["keep_all_predictions"],
                summary=paths["keep_all_summary"],
                scored_predictions=paths["keep_all_scored"],
            )
        )
    if not args.report_only and not args.skip_autogaze:
        run_command(
            build_runner_command(
                args,
                gazing_mode="autogaze",
                manifest=layout["manifest"],
                video_root=layout["video_root"],
                predictions=paths["autogaze_predictions"],
                summary=paths["autogaze_summary"],
                scored_predictions=paths["autogaze_scored"],
            )
        )

    report = build_gain_report(
        keep_all_rows=read_prediction_rows(paths["keep_all_predictions"]),
        autogaze_rows=read_prediction_rows(paths["autogaze_predictions"]),
    )
    report["dataset"] = {
        key: [str(item) for item in value] if isinstance(value, list) else str(value)
        for key, value in layout.items()
    }
    report["outputs"] = {key: str(value) for key, value in paths.items()}
    write_json(paths["report_json"], report)
    write_csv(paths["report_csv"], [flatten_metric_row(report)])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run keep-all and AutoGaze HLVid benchmarks from a local folder.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--video-root")
    parser.add_argument("--output-dir", default="outputs/autogaze_repro/hlvid_batch")
    parser.add_argument("--model-path", default="nvidia/NVILA-8B-HD-Video")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-video-frames", type=int, default=1024)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=128)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="default")
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--layout-report")
    parser.add_argument("--allow-missing-videos", action="store_true")
    parser.add_argument("--skip-keep-all", action="store_true")
    parser.add_argument("--skip-autogaze", action="store_true")
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
    parser.add_argument("--video-decode-strategy", choices=["auto", "seek", "scan"], default="auto")
    parser.add_argument("--autogaze-target-scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--visualization-output-dir")
    parser.add_argument("--visualization-fps", type=float)
    parser.add_argument("--visualization-alpha", type=float)
    parser.add_argument("--visualization-selected-max-long-side", type=int)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--extra-runner-args", nargs="*", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
