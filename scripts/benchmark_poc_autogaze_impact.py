#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
POC_SCRIPT = ROOT / "scripts" / "poc_nvila_hd_video.py"
A1_CONFIG = "configs/experiment/A1_real.yaml"
A2_CONFIG = "configs/experiment/A2_real.yaml"


@dataclass(frozen=True)
class PocBenchmarkCommand:
    experiment_id: str
    config: str
    output_dir: str
    command: list[str]


@dataclass(frozen=True)
class PocBenchmarkPlan:
    status: str
    script: str
    mode: str
    output_dir: str
    repetitions: int
    axes: dict[str, Any]
    commands: list[PocBenchmarkCommand]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["commands"] = [asdict(item) for item in self.commands]
        return data


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None or value in {"N/A", ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_numeric(rows: list[Mapping[str, Any]], key: str) -> float | str:
    values = [_as_number(row.get(key)) for row in rows]
    numeric = [value for value in values if value is not None]
    if not numeric:
        return "N/A"
    return sum(numeric) / len(numeric)


def _first_present(rows: list[Mapping[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return "N/A"


def _metrics_paths(root: Path, experiment_id: str) -> list[Path]:
    experiment_root = root / experiment_id
    if not experiment_root.exists():
        return []
    direct = experiment_root / "logs" / "metrics.json"
    if direct.exists():
        return [direct]
    return sorted(experiment_root.glob("run_*/logs/metrics.json"))


def _load_experiment_metrics(root: Path, experiment_id: str) -> list[dict[str, Any]]:
    paths = _metrics_paths(root, experiment_id)
    if not paths:
        raise FileNotFoundError(f"No PoC metrics found for {experiment_id} under {root}")
    return [_read_json(path) for path in paths]


def _aggregate_experiment(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": _first_present(rows, "experiment_id"),
        "mode": _first_present(rows, "mode"),
        "status": _first_present(rows, "status"),
        "frame_selection_mode": _first_present(rows, "frame_selection_mode"),
        "effective_frame_selection_mode": _first_present(rows, "effective_frame_selection_mode"),
        "number_of_frames": _first_present(rows, "number_of_frames"),
        "number_of_windows": _first_present(rows, "number_of_windows"),
        "scaling_mode": _first_present(rows, "scaling_mode"),
        "original_resolution": _first_present(rows, "original_resolution"),
        "processed_resolution": _first_present(rows, "processed_resolution"),
        "original_visual_token_count": _first_present(rows, "original_visual_token_count"),
        "selected_visual_token_count": _first_present(rows, "selected_visual_token_count"),
        "token_reduction_ratio": _mean_numeric(rows, "token_reduction_ratio"),
        "autogaze_latency_ms": _mean_numeric(rows, "autogaze_latency_ms"),
        "preprocessing_latency_ms": _mean_numeric(rows, "preprocessing_latency_ms"),
        "scaling_chop_latency_ms": _mean_numeric(rows, "scaling_chop_latency_ms"),
        "vision_encoder_latency_ms": _mean_numeric(rows, "vision_encoder_latency_ms"),
        "mllm_decode_latency_ms": _mean_numeric(rows, "mllm_decode_latency_ms"),
        "end_to_end_latency_ms": _mean_numeric(rows, "end_to_end_latency_ms"),
        "peak_vram_mb": _mean_numeric(rows, "peak_vram_mb"),
        "memory_metric_unavailable": _first_present(rows, "memory_metric_unavailable"),
        "skipped_stages": _first_present(rows, "skipped_stages"),
        "result_label": _first_present(rows, "result_label"),
        "runs": len(rows),
    }


def _same_axis(a1: Mapping[str, Any], a2: Mapping[str, Any], key: str) -> bool:
    return a1.get(key) == a2.get(key)


def summarize_existing_outputs(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser()
    a1_rows = _load_experiment_metrics(root, "A1_real")
    a2_rows = _load_experiment_metrics(root, "A2_real")
    a1 = _aggregate_experiment(a1_rows)
    a2 = _aggregate_experiment(a2_rows)

    axis_keys = [
        "mode",
        "frame_selection_mode",
        "effective_frame_selection_mode",
        "number_of_frames",
        "scaling_mode",
        "processed_resolution",
    ]
    axis_match = {key: _same_axis(a1, a2, key) for key in axis_keys}
    before = _as_number(a2.get("original_visual_token_count"))
    after = _as_number(a2.get("selected_visual_token_count"))
    token_reduction_observed = bool(before is not None and after is not None and after < before)
    skipped = {
        "A1_real": a1.get("skipped_stages", []),
        "A2_real": a2.get("skipped_stages", []),
    }
    generation_skipped = any(
        item.get("stage") == "nvila_generation"
        for items in skipped.values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, Mapping)
    )
    summary = {
        "result_type": "poc_autogaze_impact_summary",
        "source": "scripts/poc_nvila_hd_video.py metrics",
        "comparison": "A1_real AutoGaze OFF vs A2_real AutoGaze ON",
        "a1": a1,
        "a2": a2,
        "axis_match": axis_match,
        "valid_internal_comparison": all(axis_match.values()),
        "token_reduction_observed": token_reduction_observed,
        "generation_skipped": generation_skipped,
        "encoder_side_acceleration_claim_allowed": False,
        "acceleration_note": (
            "Do not claim encoder-side acceleration from this summary alone. "
            "Only claim it after verifying A2 reduces tokens before the intended encoder compute stage."
        ),
    }
    _write_json(root / "autogaze_impact_summary.json", summary)
    _write_summary_csv(root / "autogaze_impact_summary.csv", summary)
    return summary


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    rows = []
    for experiment_key in ("a1", "a2"):
        item = summary[experiment_key]
        rows.append(
            {
                "experiment": item.get("experiment_id"),
                "mode": item.get("mode"),
                "frames": item.get("number_of_frames"),
                "scaling_mode": item.get("scaling_mode"),
                "processed_resolution": json.dumps(item.get("processed_resolution")),
                "original_tokens": item.get("original_visual_token_count"),
                "selected_tokens": item.get("selected_visual_token_count"),
                "token_reduction_ratio": item.get("token_reduction_ratio"),
                "autogaze_latency_ms": item.get("autogaze_latency_ms"),
                "vision_encoder_latency_ms": item.get("vision_encoder_latency_ms"),
                "mllm_decode_latency_ms": item.get("mllm_decode_latency_ms"),
                "end_to_end_latency_ms": item.get("end_to_end_latency_ms"),
                "peak_vram_mb": item.get("peak_vram_mb"),
                "result_label": item.get("result_label"),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_commands(args: argparse.Namespace, *, python_executable: str = sys.executable) -> list[PocBenchmarkCommand]:
    root = Path(args.output_dir).expanduser()
    commands: list[PocBenchmarkCommand] = []
    for experiment_id, config in (("A1_real", A1_CONFIG), ("A2_real", A2_CONFIG)):
        for run_index in range(args.repetitions):
            run_dir = root / experiment_id / f"run_{run_index:03d}"
            command = [
                python_executable,
                str(POC_SCRIPT),
                "--mode",
                args.mode,
                "--video",
                args.video,
                "--query-text",
                args.query_text,
                "--frame-selection-mode",
                args.frame_selection_mode,
                "--num-frames",
                str(args.num_frames),
                "--scaling-mode",
                args.scaling_mode,
                "--resolution",
                str(args.resolution),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--config",
                config,
                "--output-dir",
                str(run_dir),
            ]
            if args.video_path:
                command.extend(["--video-path", args.video_path])
            if args.allow_checkpoint_load:
                command.append("--allow-checkpoint-load")
            if args.checkpoint_metadata_only:
                command.append("--checkpoint-metadata-only")
            if args.include_visualization:
                command.extend(["--save-overlay-video", "--save-side-by-side-video"])
            if experiment_id == "A2_real":
                command.extend(
                    [
                        "--gaze-ratio",
                        str(args.gaze_ratio),
                        "--task-loss-requirement",
                        str(args.task_loss_requirement),
                    ]
                )
            commands.append(
                PocBenchmarkCommand(
                    experiment_id=experiment_id,
                    config=config,
                    output_dir=str(run_dir),
                    command=command,
                )
            )
    return commands


def build_plan(args: argparse.Namespace) -> PocBenchmarkPlan:
    axes = {
        "mode": args.mode,
        "video": args.video,
        "video_path": args.video_path,
        "query_text": args.query_text,
        "frame_selection_mode": args.frame_selection_mode,
        "num_frames": args.num_frames,
        "scaling_mode": args.scaling_mode,
        "resolution": args.resolution,
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "include_visualization": args.include_visualization,
    }
    notes = [
        "A1_real is AutoGaze OFF with modified SigLIP + NVILA.",
        "A2_real is AutoGaze ON with modified SigLIP + NVILA.",
        "Visualization is disabled by default so latency is not dominated by video export.",
        "MPS/CPU memory metrics may be N/A.",
        "Do not claim encoder-side acceleration unless A2 token reduction happens before the intended encoder compute stage.",
    ]
    if not args.allow_checkpoint_load:
        notes.append("Checkpoint loading is disabled; generated outputs may be stub/skipped.")
    return PocBenchmarkPlan(
        status="dry_run_plan" if not args.execute else "execute_plan",
        script=str(POC_SCRIPT),
        mode=args.mode,
        output_dir=str(Path(args.output_dir).expanduser()),
        repetitions=args.repetitions,
        axes=axes,
        commands=build_commands(args),
        notes=notes,
    )


def write_plan(plan: PocBenchmarkPlan) -> Path:
    root = Path(plan.output_dir)
    path = root / "benchmark_plan.json"
    _write_json(path, plan.to_dict())
    commands_path = root / "commands.sh"
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in plan.commands:
        lines.append(" ".join(_shell_quote(part) for part in item.command))
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _shell_quote(value: str) -> str:
    if not value or any(char.isspace() or char in "'\"$`\\!" for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def execute_plan(plan: PocBenchmarkPlan) -> None:
    for item in plan.commands:
        subprocess.run(item.command, check=True, cwd=ROOT)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare AutoGaze impact by running or summarizing PoC A1_real/A2_real full-pipeline metrics"
    )
    parser.add_argument("--output-dir", default="outputs/poc_autogaze_impact")
    parser.add_argument("--mode", choices=["full_pipeline", "autogaze_only"], default="full_pipeline")
    parser.add_argument("--video", choices=["dummy"], default="dummy")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default="Question: What is happening in this video? Please answer directly.")
    parser.add_argument("--frame-selection-mode", choices=["sample", "chunk", "interval", "all"], default="sample")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--scaling-mode", choices=["none", "resize", "fit_short_side", "fit_long_side", "quickstart", "chop"], default="resize")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--gaze-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--allow-checkpoint-load", action="store_true")
    parser.add_argument("--checkpoint-metadata-only", action="store_true")
    parser.add_argument("--include-visualization", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run the PoC commands. Default writes a dry-run plan only.")
    parser.add_argument("--summarize-existing", action="store_true", help="Read existing A1/A2 PoC metrics and write comparison summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        parser.error("--repetitions must be > 0")
    if args.num_frames <= 0 or args.resolution <= 0 or args.max_new_tokens <= 0:
        parser.error("--num-frames, --resolution, and --max-new-tokens must be > 0")

    plan = build_plan(args)
    plan_path = write_plan(plan)
    print(f"benchmark plan: {plan_path}")
    print(f"commands: {Path(plan.output_dir) / 'commands.sh'}")

    if args.execute:
        execute_plan(plan)
        summary = summarize_existing_outputs(args.output_dir)
        print(f"summary: {Path(args.output_dir) / 'autogaze_impact_summary.json'}")
        print(f"valid_internal_comparison: {summary['valid_internal_comparison']}")
        print(f"token_reduction_observed: {summary['token_reduction_observed']}")
    elif args.summarize_existing:
        summary = summarize_existing_outputs(args.output_dir)
        print(f"summary: {Path(args.output_dir) / 'autogaze_impact_summary.json'}")
        print(f"valid_internal_comparison: {summary['valid_internal_comparison']}")
        print(f"token_reduction_observed: {summary['token_reduction_observed']}")
    else:
        print("dry run only; pass --execute to run PoC A1/A2 commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
