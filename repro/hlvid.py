from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

from repro.common import compute_stats, write_json, write_jsonl

REQUIRED_COLUMNS = ("question_id", "category", "video_path", "question", "answer")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
QUESTION_SUMMARY_SAMPLE_LIMIT = 5
BENCHMARK_SAMPLE_LIMIT = 5
LATENCY_FIELDS = (
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
    "generation_decode_after_ttft_estimated_ms",
)
STAGE_TIMING_FIELDS = (
    (
        "processor.autogaze_forward_batched.total_ms",
        "stage_timings_ms.processor.autogaze_forward_batched.total_ms",
    ),
    (
        "processor.autogaze_forward_batched.count",
        "stage_timings_ms.processor.autogaze_forward_batched.count",
    ),
    (
        "processor.autogaze_forward_batched.mean_ms",
        "stage_timings_ms.processor.autogaze_forward_batched.mean_ms",
    ),
    (
        "processor.autogaze_total.total_ms",
        "stage_timings_ms.processor.autogaze_total.total_ms",
    ),
    (
        "processor.autogaze_total.count",
        "stage_timings_ms.processor.autogaze_total.count",
    ),
)
READABLE_STAGE_TIMING_FIELDS = (
    (
        "processor_autogaze_forward_batched_total_ms",
        "stage_timings_ms.processor.autogaze_forward_batched.total_ms",
    ),
    (
        "processor_autogaze_forward_batched_count",
        "stage_timings_ms.processor.autogaze_forward_batched.count",
    ),
    (
        "processor_autogaze_forward_batched_mean_ms",
        "stage_timings_ms.processor.autogaze_forward_batched.mean_ms",
    ),
    (
        "processor_autogaze_total_total_ms",
        "stage_timings_ms.processor.autogaze_total.total_ms",
    ),
    (
        "processor_autogaze_total_count",
        "stage_timings_ms.processor.autogaze_total.count",
    ),
)
MODULE_LATENCY_FIELDS = (
    ("total_ms", "total_ms"),
    ("preprocess_without_autogaze_ms", "video_preprocess_without_autogaze_ms"),
    ("preprocess_total_ms", "video_preprocess_ms"),
    ("autogaze_ms", "autogaze_ms"),
    ("autogaze_total_ms", "autogaze_total_ms"),
    ("vit_encoder_ms", "siglip_vision_ms"),
    ("llm_ms", "llm_forward_ms"),
)
KEY_TOKEN_FIELDS = (
    ("video_sampled_frames", "token_metrics.video_sampled_frames"),
    ("thumbnail_sampled_frames", "token_metrics.thumbnail_sampled_frames"),
    ("encoder_patch_tokens_before_keep_all_or_raw", "token_metrics.encoder_raw_patch_tokens"),
    ("encoder_patch_tokens_after_autogaze", "token_metrics.encoder_autogaze_selected_patch_tokens"),
    ("encoder_token_reduction_ratio", "token_metrics.encoder_token_reduction_ratio"),
    ("autogaze_input_tile_patch_tokens", "token_metrics.autogaze_input_patch_tokens"),
    ("autogaze_selected_tile_patch_tokens", "token_metrics.autogaze_selected_patch_tokens"),
    ("autogaze_patch_reduction_ratio", "token_metrics.autogaze_patch_reduction_ratio"),
    ("llm_visual_tokens_before_keep_all_estimated", "token_metrics.llm_keep_all_visual_tokens_estimated"),
    ("llm_visual_tokens_after_actual", "token_metrics.llm_actual_visual_tokens"),
    ("llm_visual_token_reduction_ratio", "token_metrics.llm_visual_token_reduction_ratio"),
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
TOKEN_FIELDS = (
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
    "token_metrics.llm_keep_all_visual_tokens_estimated",
    "token_metrics.llm_actual_visual_tokens",
    "token_metrics.llm_visual_token_reduction_ratio",
)
COMPUTE_FIELDS = (
    "compute_metrics.siglip_encoder.keep_all_to_actual_attention_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_mlp_macs_ratio",
    "compute_metrics.siglip_encoder.keep_all_to_actual_total_macs_ratio",
    "compute_metrics.mllm.kv_cache_reduction_ratio",
    "compute_metrics.mllm.prefill_attention_pair_reduction_ratio",
    "compute_metrics.mllm.prefill_total_macs_reduction_ratio",
)


def parse_choice(text: str | None) -> str | None:
    if text is None:
        return None
    matches = [match.group(1).upper() for match in CHOICE_RE.finditer(text)]
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    missing_by_row = []
    for idx, row in enumerate(rows):
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            missing_by_row.append((idx, missing))
    if missing_by_row:
        raise ValueError(f"HLVid manifest rows missing required columns: {missing_by_row[:3]}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in REQUIRED_COLUMNS}


def viewer_row_to_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload.get("row", payload)
    return normalize_row(row)


def fetch_hlvid_manifest(
    dataset: str = "bfshi/HLVid",
    config: str = "default",
    split: str = "test",
    limit: int | None = None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        length = page_size if limit is None else min(page_size, limit - len(rows))
        if length <= 0:
            break
        response = requests.get(
            DATASET_VIEWER_ROWS_URL,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        batch = [viewer_row_to_manifest(item) for item in payload.get("rows", [])]
        rows.extend(batch)

        total = payload.get("num_rows_total")
        if not batch:
            break
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break
        offset += len(batch)
        if total is not None and offset >= int(total):
            break

    validate_manifest_rows(rows)
    return rows


def load_hlvid_manifest(split: str = "test", limit: int | None = None, config: str = "default") -> list[dict[str, Any]]:
    return fetch_hlvid_manifest(split=split, limit=limit, config=config)


def summarize_prompt_or_question_samples(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    samples: list[dict[str, Any]] = []
    for row in rows:
        if row.get("question") is None and row.get("prompt") is None:
            continue
        count += 1
        if len(samples) >= QUESTION_SUMMARY_SAMPLE_LIMIT:
            continue
        sample = {
            key: row[key]
            for key in ("question_id", "video_path", "question", "prompt", "answer")
            if row.get(key) is not None
        }
        samples.append(sample)
    return count, samples


def build_benchmark_samples(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in scored_rows:
        if len(samples) >= BENCHMARK_SAMPLE_LIMIT:
            break
        target_video = row.get("video_path", row.get("video"))
        question = row.get("question", row.get("prompt"))
        if target_video is None and question is None and row.get("raw_output") is None:
            continue
        sample = {
            "question_id": row.get("question_id"),
            "target_video": target_video,
            "question": question,
            "model_answer": row.get("raw_output"),
            "parsed_model_answer": row.get("parsed_answer"),
            "correct_answer": row.get("expected_answer"),
            "ground_truth_answer": row.get("answer"),
            "correct": row.get("correct"),
            "status": row.get("status", "ok"),
            "parse_status": row.get("parse_status"),
        }
        samples.append(sample)
    return samples


def metric_value(row: dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = metric_value(row, field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def stats_by_field(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    return {field: compute_stats(numeric_values(rows, field)) for field in fields}


def stats_by_labeled_field(
    rows: list[dict[str, Any]],
    fields: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, float | int]]:
    return {label: compute_stats(numeric_values(rows, field)) for label, field in fields}


def median_from_stats(stats: dict[str, dict[str, float | int]], field: str) -> float | int | None:
    field_stats = stats.get(field)
    if not field_stats:
        return None
    return field_stats.get("median")


def key_medians(
    stats: dict[str, dict[str, float | int]],
    fields: tuple[tuple[str, str], ...],
) -> dict[str, float | int | None]:
    return {label: median_from_stats(stats, field) for label, field in fields}


LATENCY_HIERARCHY_ASCII = """total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms
|-- video_preprocess_without_autogaze_ms
|   |-- video_decode_ms (included; not an extra total term)
|   |-- video_tiling_ms (included; not an extra total term)
|   `-- other processor/tokenization overhead
|-- autogaze_total_ms
|   `-- autogaze_model_forward_ms / autogaze_forward_ms (model forward only)
`-- generate_ms
    |-- vision_encoder_ms
    |   |-- siglip_vision_ms
    |   `-- mm_projector_ms
    |-- llm_forward_ms
    `-- generation_decode_after_ttft_estimated_ms (generation-side estimate, not video decode)
ttft_ms: separate 1-token measurement pass, excluded from total_ms"""


def first_metric_value(metrics: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = metrics.get(field)
        if value is not None:
            return value
    return None


def latency_hierarchy_summary(metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or {}
    autogaze_total_ms = first_metric_value(metrics, "autogaze_total_ms", "gazing_info_total_ms", "autogaze_ms")
    preprocess_without_autogaze_ms = first_metric_value(
        metrics,
        "video_preprocess_without_autogaze_ms",
    )
    return {
        "total_formula": "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms",
        "legacy_inclusive_formula": "total_ms = video_preprocess_ms + generate_ms",
        "ascii_tree": LATENCY_HIERARCHY_ASCII,
        "quick_answers": {
            "what_total_ms_is": (
                "End-to-end latency: video_preprocess_without_autogaze_ms plus "
                "autogaze_total_ms plus generate_ms."
            ),
            "what_generate_ms_includes": (
                "NVILA model.generate after preprocessing: vision_encoder_ms plus llm_forward_ms."
            ),
            "is_autogaze_in_generate_ms": False,
            "where_is_autogaze_ms_included": "autogaze_total_ms",
            "legacy_inclusive_preprocess_field": "video_preprocess_ms",
            "is_video_decode_in_preprocess_ms": True,
            "where_is_video_decode_ms_included": "video_preprocess_without_autogaze_ms",
            "is_ttft_in_total_ms": False,
        },
        "nodes": {
            "total_ms": {
                "value_ms": metrics.get("total_ms"),
                "adds": ["video_preprocess_without_autogaze_ms", "autogaze_total_ms", "generate_ms"],
                "description": "End-to-end latency for one measured run.",
            },
            "video_preprocess_without_autogaze_ms": {
                "value_ms": preprocess_without_autogaze_ms,
                "included_in": "total_ms",
                "includes": ["video_decode_ms", "video_tiling_ms"],
                "description": "Preprocess work excluding the AutoGaze stage.",
            },
            "video_preprocess_ms": {
                "value_ms": metrics.get("video_preprocess_ms"),
                "legacy": True,
                "includes": ["video_preprocess_without_autogaze_ms", "autogaze_total_ms"],
                "description": "Legacy inclusive preprocess field kept for backward compatibility.",
            },
            "video_decode_ms": {
                "value_ms": metrics.get("video_decode_ms"),
                "included_in": "video_preprocess_without_autogaze_ms",
                "add_to_total_ms": False,
                "description": "Video frame decode/sample/resize timing when measured.",
            },
            "video_tiling_ms": {
                "value_ms": metrics.get("video_tiling_ms"),
                "included_in": "video_preprocess_without_autogaze_ms",
                "add_to_total_ms": False,
                "description": "Dynamic tiling, thumbnail processing, and tensorization timing.",
            },
            "autogaze_total_ms": {
                "value_ms": autogaze_total_ms,
                "included_in": "total_ms",
                "aliases": ["gazing_info_total_ms", "autogaze_ms"],
                "includes": ["autogaze_model_forward_ms"],
                "description": "AutoGaze stage total: model forward plus gaze-info bookkeeping.",
            },
            "gazing_info_total_ms": {
                "value_ms": first_metric_value(metrics, "gazing_info_total_ms", "autogaze_ms"),
                "aliases": ["autogaze_ms"],
                "included_in": "autogaze_total_ms",
                "add_to_total_ms": False,
                "description": "AutoGaze stage total: model forward plus gaze-info bookkeeping.",
            },
            "autogaze_model_forward_ms": {
                "value_ms": first_metric_value(metrics, "autogaze_model_forward_ms", "autogaze_forward_ms"),
                "aliases": ["autogaze_forward_ms"],
                "included_in": "autogaze_total_ms",
                "add_to_total_ms": False,
                "description": "AutoGaze model forward-only timing over batched tile tensors.",
            },
            "generate_ms": {
                "value_ms": metrics.get("generate_ms"),
                "includes": [
                    "vision_encoder_ms",
                    "llm_forward_ms",
                    "generation_decode_after_ttft_estimated_ms",
                ],
                "description": "NVILA model.generate after preprocessing; AutoGaze is not included.",
            },
            "vision_encoder_ms": {
                "value_ms": metrics.get("vision_encoder_ms"),
                "included_in": "generate_ms",
                "includes": ["siglip_vision_ms", "mm_projector_ms"],
                "add_to_total_ms": False,
                "description": "Vision path inside generate, including SigLIP and projector hooks.",
            },
            "siglip_vision_ms": {
                "value_ms": metrics.get("siglip_vision_ms"),
                "included_in": "vision_encoder_ms",
                "add_to_total_ms": False,
                "description": "Pure SigLIP/ViT vision tower forward timing.",
            },
            "mm_projector_ms": {
                "value_ms": metrics.get("mm_projector_ms"),
                "included_in": "vision_encoder_ms",
                "add_to_total_ms": False,
                "description": "Projection from visual features to LLM hidden dimension.",
            },
            "llm_forward_ms": {
                "value_ms": metrics.get("llm_forward_ms"),
                "included_in": "generate_ms",
                "add_to_total_ms": False,
                "description": "Accumulated LLM forward calls during full generation.",
            },
            "generation_decode_after_ttft_estimated_ms": {
                "value_ms": first_metric_value(
                    metrics,
                    "generation_decode_after_ttft_estimated_ms",
                    "decode_estimated_ms",
                ),
                "aliases": ["decode_estimated_ms"],
                "included_in": "generate_ms",
                "add_to_total_ms": False,
                "description": "Estimated generation-side decode time; not video decode.",
            },
            "ttft_ms": {
                "value_ms": metrics.get("ttft_ms"),
                "included_in": None,
                "relationship_to_total_ms": "separate_measurement",
                "add_to_total_ms": False,
                "description": "Separate 1-token pass for first-token latency; excluded from total_ms.",
            },
        },
    }


def latency_accounting_summary() -> dict[str, Any]:
    return {
        "additive_formula": "total_ms = video_preprocess_without_autogaze_ms + autogaze_total_ms + generate_ms",
        "additive_fields": ["video_preprocess_without_autogaze_ms", "autogaze_total_ms", "generate_ms"],
        "legacy_inclusive_formula": "total_ms = video_preprocess_ms + generate_ms",
        "legacy_inclusive_fields": ["video_preprocess_ms", "generate_ms"],
        "hierarchy": latency_hierarchy_summary(),
        "nested_preprocess_fields": [
            "video_preprocess_ms",
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
            if field not in {"total_ms", "video_preprocess_without_autogaze_ms", "autogaze_total_ms", "generate_ms"}
        ],
    }


def summarize_prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latency = stats_by_field(rows, LATENCY_FIELDS)
    stage_timings = stats_by_labeled_field(rows, STAGE_TIMING_FIELDS)
    readable_stage_timings = stats_by_labeled_field(rows, READABLE_STAGE_TIMING_FIELDS)
    memory = stats_by_field(rows, MEMORY_FIELDS)
    tokens = stats_by_field(rows, TOKEN_FIELDS)
    compute = stats_by_field(rows, COMPUTE_FIELDS)
    latency_summary = key_medians(latency, MODULE_LATENCY_FIELDS)
    token_summary = key_medians(tokens, KEY_TOKEN_FIELDS)
    memory_summary = key_medians(memory, KEY_MEMORY_FIELDS)
    return {
        "latency_ms": latency,
        "stage_timings_ms": stage_timings,
        "memory_bytes": memory,
        "tokens": tokens,
        "compute": compute,
        "readable_performance_summary": {
            "key_metrics_median": {
                "latency_ms": latency_summary,
                "tokens": token_summary,
                "memory_bytes": memory_summary,
            },
            "latency_ms_median": latency_summary,
            "latency_ms_detail_median": {field: latency[field]["median"] for field in LATENCY_FIELDS},
            "stage_timings_ms_median": {
                field: readable_stage_timings[field]["median"] for field, _ in READABLE_STAGE_TIMING_FIELDS
            },
            "stage_timings_note": (
                "Nested stage_timings_ms are aggregated from per-row prediction JSONL. "
                "processor_autogaze_forward_batched_total_ms is the summed _run_autogaze_batched time "
                "within each row; processor_autogaze_forward_batched_count is the number of wrapped calls "
                "observed for that row."
            ),
            "latency_accounting": latency_accounting_summary(),
            "latency_field_note": (
                "Summary-level latency is intentionally coarse: "
                "preprocess_without_autogaze=video_preprocess_without_autogaze_ms, "
                "preprocess_total=legacy inclusive video_preprocess_ms, autogaze=autogaze_total_ms, "
                "vit_encoder=siglip_vision_ms, llm=llm_forward_ms. "
                "The primary additive formula separates preprocess_without_autogaze, autogaze_total, and generate. "
                "Use latency_accounting.additive_formula for the only additive total formula, "
                "and use latency_ms_detail_median or top-level latency_ms for finer breakdowns."
            ),
            "memory_bytes_median": {field: memory[field]["median"] for field in MEMORY_FIELDS},
            "tokens_median": token_summary,
            "compute_median": {
                field: compute[field]["median"] for field in COMPUTE_FIELDS
            },
            "metric_note": (
                "Per-mode HLVid summary medians are computed from prediction rows only. "
                "Warmup runs are excluded, and failed rows only contribute metrics that were recorded."
            ),
        },
    }


def score_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    parse_failed = 0
    failed = 0
    skipped = 0
    question_count, question_samples = summarize_prompt_or_question_samples(rows)

    for row in rows:
        status = row.get("status")
        if status in {"failed", "skipped"}:
            out = dict(row)
            out["parsed_answer"] = None
            out["expected_answer"] = parse_choice(str(row.get("answer", "")))
            out["correct"] = False
            out["parse_status"] = f"{status}_run"
            failed += int(status == "failed")
            skipped += int(status == "skipped")
            scored_rows.append(out)
            continue

        parsed = parse_choice(row.get("raw_output"))
        expected = parse_choice(str(row.get("answer", "")))
        out = dict(row)
        out["parsed_answer"] = parsed
        out["expected_answer"] = expected
        if parsed is None or expected is None:
            parse_failed += 1
            out["correct"] = False
            out["parse_status"] = "failed"
        else:
            scored += 1
            out["correct"] = parsed == expected
            out["parse_status"] = "parsed"
            correct += int(out["correct"])
        scored_rows.append(out)

    summary = {
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "parse_failed": parse_failed,
        "failed": failed,
        "skipped": skipped,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
        "question_count": question_count,
        "question_samples": question_samples,
        "question_note": (
            "Full per-row prompts/questions are stored in predictions and scored_predictions JSONL."
        ),
        "benchmark_samples": build_benchmark_samples(scored_rows),
        "benchmark_sample_note": (
            "Readable benchmark samples: target_video, question, model_answer, "
            "parsed_model_answer, correct_answer, and correctness. Full rows remain in JSONL."
        ),
    }
    summary.update(summarize_prediction_metrics(scored_rows))
    return summary, scored_rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def read_manifest_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(source)
    elif suffix == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"HLVid JSON manifest must be a list or contain rows/data: {source}")
        rows = payload
    elif suffix == ".csv":
        with source.open(newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional local dataset format
            raise RuntimeError("Reading parquet manifests requires pandas/pyarrow in the environment.") from exc
        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported HLVid manifest extension: {source.suffix}")

    validate_manifest_rows(rows)
    return [normalize_row(row) for row in rows]


def build_manifest(args: argparse.Namespace) -> None:
    rows = load_hlvid_manifest(split=args.split, limit=args.limit, config=args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} HLVid rows to {output}")


def score_file(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    summary, scored_rows = score_predictions(rows)
    write_json(args.summary, summary)
    write_jsonl(args.scored, scored_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="HLVid manifest and scoring helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--config", default="default")
    manifest.add_argument("--split", default="test")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--output", default="data/hlvid/manifest_test.json")
    manifest.set_defaults(func=build_manifest)

    score = sub.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--summary", default="outputs/autogaze_repro/hlvid_score_summary.json")
    score.add_argument("--scored", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    score.set_defaults(func=score_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
