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

LATENCY_FIELDS = (
    "total_ms",
    "video_preprocess_ms",
    "video_decode_ms",
    "video_tiling_ms",
    "autogaze_forward_ms",
    "vision_encoder_ms",
    "siglip_vision_ms",
    "mm_projector_ms",
    "llm_forward_ms",
    "ttft_ms",
)
MEMORY_FIELDS = (
    "processor_peak_memory_bytes",
    "ttft_peak_memory_bytes",
    "llm_peak_memory_bytes",
    "peak_memory_bytes",
)
AUTOGAZE_TOKEN_FIELDS = (
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
    numerator = compute_stats(_numeric_values(numerator_rows, field))["median"]
    denominator = compute_stats(_numeric_values(denominator_rows, field))["median"]
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


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
    if getattr(args, "autogaze_target_scales", None) is not None:
        command.extend(["--autogaze-target-scales", str(args.autogaze_target_scales)])
    if getattr(args, "autogaze_target_patch_size", None) is not None:
        command.extend(["--autogaze-target-patch-size", str(args.autogaze_target_patch_size)])
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


def build_gain_report(
    *,
    keep_all_rows: list[dict[str, Any]],
    autogaze_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keep_all = summarize_run(keep_all_rows)
    autogaze = summarize_run(autogaze_rows)
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
        "gains": {
            "accuracy_scored_delta": accuracy_delta,
            "latency_speedup_median": latency_speedups,
            "memory_reduction_ratio_median": memory_reductions,
            "autogaze_token_reduction_median": token_reductions,
            "compute_reduction_median": compute_reductions,
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
    parser.add_argument("--autogaze-target-scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--extra-runner-args", nargs="*", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
