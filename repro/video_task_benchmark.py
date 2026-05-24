from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import compute_stats, write_json, write_jsonl
from repro.failure_logging import classify_exception, failure_generation_payload
from repro.flexible_runner import parse_args as parse_flexible_args
from repro.flexible_runner import run_single
from repro.plugin_hlvid_benchmark import (
    DEFAULT_MODELS,
    build_mode_runner_args,
    first_present,
    parse_model_overrides,
)
from repro.video_task_schema import (
    TASK_TYPE_ACTION_CLASSIFICATION,
    TASK_TYPE_CAPTIONING,
    TASK_TYPE_VIDEOQA,
    TASK_TYPES,
    read_video_task_manifest,
    score_action_predictions,
    score_caption_predictions,
    score_videoqa_predictions,
)

DEFAULT_VIDEO_TASK_MODES = ["qwen3_full_vit", "qwen3_chunked_vit", "qwen3_chunked_vit_autogaze_sparse"]


def run_video_task_benchmark(
    *,
    manifest: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    task_type: str,
    modes: list[str],
    models: dict[str, str] | None = None,
    external_mllm_command: str = "vila-infer",
    limit: int | None = None,
    num_video_frames: int = 32,
    num_video_frames_thumbnail: int = 0,
    max_tiles_video: int = 4,
    max_new_tokens: int = 32,
    qwen_video_nframes: int | None = None,
    qwen_video_fps: float | None = None,
    qwen_video_max_pixels: int | None = None,
    qwen_video_min_pixels: int | None = None,
    qwen_vit_chunk_frames: int = 16,
    qwen_vit_max_spatial_chunks: int | None = None,
    qwen_thumbnail_mode: str = "none",
    video_resize_shortest_edge: int | None = None,
    video_resize_longest_edge: int | None = None,
    video_resize_width: int | None = None,
    video_resize_height: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_video_task_manifest(manifest, task_type=task_type)
    if limit is not None:
        rows = rows[:limit]
    model_paths = {**DEFAULT_MODELS, **(models or {})}
    predictions: list[dict[str, Any]] = []
    for mode in modes:
        for row_index, row in enumerate(rows):
            video_path = resolve_video_path(video_root, row["video_path"])
            run_json = output / "runs" / mode / f"{row_index:05d}.json"
            runner_row = _runner_row(row)
            runner_args = build_mode_runner_args(
                mode=mode,
                row=runner_row,
                video_path=video_path,
                output_json=run_json,
                models=model_paths,
                external_mllm_command=external_mllm_command,
                num_video_frames=num_video_frames,
                num_video_frames_thumbnail=num_video_frames_thumbnail,
                max_tiles_video=max_tiles_video,
                max_new_tokens=max_new_tokens,
                qwen_video_nframes=qwen_video_nframes,
                qwen_video_fps=qwen_video_fps,
                qwen_video_max_pixels=qwen_video_max_pixels,
                qwen_video_min_pixels=qwen_video_min_pixels,
                qwen_vit_chunk_frames=qwen_vit_chunk_frames,
                qwen_vit_max_spatial_chunks=qwen_vit_max_spatial_chunks,
                qwen_thumbnail_mode=qwen_thumbnail_mode,
                video_resize_shortest_edge=video_resize_shortest_edge,
                video_resize_longest_edge=video_resize_longest_edge,
                video_resize_width=video_resize_width,
                video_resize_height=video_resize_height,
            )
            parsed_args = parse_flexible_args(runner_args)
            try:
                payload = run_single(parsed_args)
            except Exception as exc:
                failure = classify_exception(exc, stage="video_task_row")
                payload = {
                    "runner": "flexible_runner",
                    "mode": "single",
                    "implementation_status": failure["kind"],
                    "failure": failure,
                    "generation": failure_generation_payload(parsed_args, failure),
                }
                write_json(run_json, payload)
                if failure["kind"] != "oom":
                    raise
            predictions.append(_prediction_from_payload(mode, row, video_path, payload))

    summary, scored = summarize_video_task_predictions(predictions, task_type=task_type)
    predictions_path = output / f"{task_type}_predictions.jsonl"
    scored_path = output / f"{task_type}_scored.jsonl"
    summary_path = output / f"{task_type}_summary.json"
    report_path = output / f"{task_type}_report.md"
    write_jsonl(predictions_path, predictions)
    write_jsonl(scored_path, scored)
    write_json(summary_path, summary)
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    return {
        "predictions": predictions,
        "scored": scored,
        "summary": summary,
        "artifacts": {
            "predictions": str(predictions_path),
            "scored": str(scored_path),
            "summary": str(summary_path),
            "markdown": str(report_path),
        },
    }


def summarize_video_task_predictions(predictions: list[dict[str, Any]], *, task_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modes = sorted({row["mode"] for row in predictions})
    summaries: dict[str, Any] = {}
    all_scored: list[dict[str, Any]] = []
    for mode in modes:
        rows = [row for row in predictions if row["mode"] == mode]
        if task_type == TASK_TYPE_ACTION_CLASSIFICATION:
            mode_summary, scored = score_action_predictions(rows)
        elif task_type == TASK_TYPE_CAPTIONING:
            mode_summary, scored = score_caption_predictions(rows)
        elif task_type == TASK_TYPE_VIDEOQA:
            mode_summary, scored = score_videoqa_predictions(rows)
        else:
            raise ValueError(f"unsupported video task_type: {task_type}")
        mode_summary["status_counts"] = _status_counts(rows)
        mode_summary["latency_ms"] = _stats_for(rows, "total_ms")
        mode_summary["generate_ms"] = _stats_for(rows, "generate_ms")
        mode_summary["peak_memory_bytes"] = _stats_for(rows, "peak_memory_bytes")
        mode_summary["visual_tokens_before_prune"] = _stats_for(rows, "visual_tokens_before_prune")
        mode_summary["visual_tokens_after_prune"] = _stats_for(rows, "visual_tokens_after_prune")
        summaries[mode] = mode_summary
        all_scored.extend(scored)
    return {
        "task_type": task_type,
        "modes": summaries,
        "mode_order": modes,
        "total_predictions": len(predictions),
    }, all_scored


def build_markdown_report(summary: dict[str, Any]) -> str:
    task_type = summary.get("task_type")
    lines = [
        "# Video Task Benchmark",
        "",
        f"- task_type: `{task_type}`",
        "",
        "| mode | total | correct | failed | parse_failed | accuracy_total | accuracy_scored | scoring_status | status_counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for mode, mode_summary in (summary.get("modes") or {}).items():
        lines.append(
            "| {mode} | {total} | {correct} | {failed} | {parse_failed} | {accuracy_total} | {accuracy_scored} | {scoring_status} | {status_counts} |".format(
                mode=mode,
                total=mode_summary.get("total", 0),
                correct=mode_summary.get("correct", 0),
                failed=mode_summary.get("failed", 0),
                parse_failed=mode_summary.get("parse_failed", 0),
                accuracy_total=_format_metric(mode_summary.get("accuracy_total")),
                accuracy_scored=_format_metric(mode_summary.get("accuracy_scored")),
                scoring_status=mode_summary.get("scoring_status", "scored"),
                status_counts=json.dumps(mode_summary.get("status_counts", {}), sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Action classification uses exact label or multiple-choice letter parsing.",
            "- VideoQA uses multiple-choice parsing when the answer is a letter, and text containment for open answers.",
            "- Captioning keeps references and overlap hints, but defaults to `not_scored` unless an external judge is added.",
            "- Per-row latency, token, memory, and failure details are stored in the predictions JSONL.",
        ]
    )
    return "\n".join(lines) + "\n"


def resolve_video_path(video_root: str | Path, video_path: str) -> Path:
    root = Path(video_root)
    direct = root / video_path
    if direct.exists():
        return direct
    flat = root / Path(video_path).name
    if flat.exists():
        return flat
    return direct


def _runner_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["sample_id"],
        "category": row.get("category"),
        "video_path": row["video_path"],
        "question": row["prompt"],
        "answer": row.get("answer") or row.get("label") or (row.get("references") or [None])[0],
    }


def _prediction_from_payload(mode: str, row: dict[str, Any], video_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation", {})
    metrics = generation.get("metrics", {})
    prediction = {
        "mode": mode,
        "sample_id": row.get("sample_id"),
        "task_type": row.get("task_type"),
        "category": row.get("category"),
        "source": row.get("source"),
        "duration": row.get("duration"),
        "video_path": row.get("video_path"),
        "resolved_video_path": str(video_path),
        "prompt": row.get("prompt"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "label": row.get("label"),
        "choices": row.get("choices"),
        "references": row.get("references"),
        "raw_output": generation.get("text"),
        "status": _prediction_status(generation.get("status")),
        "runner_status": payload.get("implementation_status"),
        "failure": payload.get("failure") or generation.get("failure"),
        "metric_status": metrics.get("metric_status"),
        "metrics": metrics,
    }
    prediction.update(_flatten_metrics(metrics))
    if prediction.get("failure") and not prediction.get("failure_stage"):
        prediction["failure_stage"] = prediction["failure"].get("stage")
        prediction["failure_kind"] = prediction["failure"].get("kind")
    return prediction


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    latency = metrics.get("latency_ms") or {}
    memory = metrics.get("memory_bytes") or {}
    tokens = metrics.get("tokens") or {}
    return {
        "total_ms": latency.get("total"),
        "generate_ms": latency.get("generate"),
        "vision_encoder_ms": first_present(latency.get("vision_encoder"), latency.get("qwen_vit_prepare")),
        "peak_memory_bytes": first_present(memory.get("peak_cuda_reserved"), memory.get("peak_cuda_allocated")),
        "visual_tokens_before_prune": tokens.get("visual_tokens_before_prune"),
        "visual_tokens_after_prune": tokens.get("visual_tokens_after_prune"),
        "llm_context_tokens": tokens.get("llm_context_tokens"),
    }


def _prediction_status(generation_status: str | None) -> str:
    if generation_status == "oom":
        return "oom"
    if generation_status in {"executed", "executed_dense_with_autogaze_sidecar", "probe_required", "probe_collected"}:
        return "ok"
    return "failed"


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("runner_status") or row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _stats_for(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or isinstance(value, bool):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return compute_stats(values)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VideoQA/caption/action video task benchmarks through flexible_runner")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", default="outputs/autogaze_repro/video_task_benchmark")
    parser.add_argument("--task-type", choices=TASK_TYPES, required=True)
    parser.add_argument("--modes", default=",".join(DEFAULT_VIDEO_TASK_MODES))
    parser.add_argument("--model", action="append", help="Override model path as adapter=path, e.g. qwen3-vl=weight/Qwen3")
    parser.add_argument("--external-mllm-command", default="vila-infer")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-video-frames", type=int, default=32)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=0)
    parser.add_argument("--max-tiles-video", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--qwen-video-nframes", type=int)
    parser.add_argument("--qwen-video-fps", type=float)
    parser.add_argument("--qwen-video-max-pixels", type=int)
    parser.add_argument("--qwen-video-min-pixels", type=int)
    parser.add_argument("--qwen-vit-chunk-frames", type=int, default=16)
    parser.add_argument("--qwen-vit-max-spatial-chunks", type=int)
    parser.add_argument("--qwen-thumbnail-mode", choices=["none", "append-video"], default="none")
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_video_task_benchmark(
        manifest=args.manifest,
        video_root=args.video_root,
        output_dir=args.output_dir,
        task_type=args.task_type,
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        models=parse_model_overrides(args.model),
        external_mllm_command=args.external_mllm_command,
        limit=args.limit,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        max_new_tokens=args.max_new_tokens,
        qwen_video_nframes=args.qwen_video_nframes,
        qwen_video_fps=args.qwen_video_fps,
        qwen_video_max_pixels=args.qwen_video_max_pixels,
        qwen_video_min_pixels=args.qwen_video_min_pixels,
        qwen_vit_chunk_frames=args.qwen_vit_chunk_frames,
        qwen_vit_max_spatial_chunks=args.qwen_vit_max_spatial_chunks,
        qwen_thumbnail_mode=args.qwen_thumbnail_mode,
        video_resize_shortest_edge=args.video_resize_shortest_edge,
        video_resize_longest_edge=args.video_resize_longest_edge,
        video_resize_width=args.video_resize_width,
        video_resize_height=args.video_resize_height,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
