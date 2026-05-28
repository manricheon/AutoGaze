from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from repro.common import write_json, write_jsonl
from repro.hlvid import read_manifest_file, score_predictions
from repro.plugin_hlvid_benchmark import resolve_hlvid_video_path
from repro.vjepa_qwen_runner import (
    VjepaQwenStageError,
    build_parser as build_runner_parser,
    run_actual_pipeline,
    write_markdown_summary,
)

DEFAULT_OUTPUT_DIR = "outputs/autogaze_vjepa/hlvid"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HLVid wrapper for AutoGaze -> V-JEPA sparse -> Qwen actual runner.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the per-row runner plan without loading V-JEPA, Qwen, or AutoGaze weights.",
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vjepa-qwen-modes",
        default="",
        help=(
            "Comma-separated run modes: dense_off, autogaze_single_grid, autogaze_scale_aware. "
            "Use --vjepa-selection-policies for the legacy AutoGaze-only policy list."
        ),
    )
    parser.add_argument("--vjepa-selection-policies", default="single_scale_union")

    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--autogaze-device", default="auto")
    parser.add_argument("--autogaze-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--autogaze-chunk-frames", type=int, default=16)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--autogaze-tile-size", type=int, default=224)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--gazing-ratio", type=float)
    parser.add_argument("--task-loss-requirement", type=float)
    parser.add_argument("--autogaze-target-scales", default="32+64+112+224")
    parser.add_argument("--autogaze-target-patch-size", type=int, default=16)
    parser.add_argument("--autogaze-encoder-patch-size", type=int)
    parser.add_argument("--autogaze-generate-only", action="store_true")

    parser.add_argument("--video-decode-strategy", default="auto", choices=["auto", "seek", "scan"])
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)

    parser.add_argument("--vjepa-model", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--vjepa-overlap-threshold", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--clear-cuda-cache-between-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualization-output-dir")
    parser.add_argument("--visualization-max-frames", type=int, default=16)
    return parser


def build_runner_args_for_row(
    *,
    row: dict[str, Any],
    video_path: Path,
    output_json: Path,
    mode: str,
    benchmark_args: argparse.Namespace,
) -> list[str]:
    autogaze_mode, selection_policy = mode_to_runner_settings(mode)
    argv = [
        "--video",
        str(video_path),
        "--prompt",
        str(row.get("question") or row.get("prompt") or ""),
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_json.with_suffix(".md")),
        "--autogaze-mode",
        autogaze_mode,
        "--autogaze-repo",
        str(benchmark_args.autogaze_repo),
        "--autogaze-model",
        str(benchmark_args.autogaze_model),
        "--autogaze-device",
        str(benchmark_args.autogaze_device),
        "--autogaze-dtype",
        str(benchmark_args.autogaze_dtype),
        "--num-video-frames",
        str(benchmark_args.num_video_frames),
        "--autogaze-chunk-frames",
        str(benchmark_args.autogaze_chunk_frames),
        "--max-tiles-video",
        str(benchmark_args.max_tiles_video),
        "--autogaze-tile-size",
        str(benchmark_args.autogaze_tile_size),
        "--max-batch-size-autogaze",
        str(benchmark_args.max_batch_size_autogaze),
        "--autogaze-target-scales",
        str(benchmark_args.autogaze_target_scales),
        "--autogaze-target-patch-size",
        str(benchmark_args.autogaze_target_patch_size),
        "--video-decode-strategy",
        str(benchmark_args.video_decode_strategy),
        "--vjepa-model",
        str(benchmark_args.vjepa_model),
        "--qwen-model",
        str(benchmark_args.qwen_model),
        "--device",
        str(benchmark_args.device),
        "--dtype",
        str(benchmark_args.dtype),
        "--frames-per-clip",
        str(benchmark_args.frames_per_clip),
        "--tubelet-size",
        str(benchmark_args.tubelet_size),
        "--crop-size",
        str(benchmark_args.crop_size),
        "--patch-size",
        str(benchmark_args.patch_size),
        "--vjepa-overlap-threshold",
        str(benchmark_args.vjepa_overlap_threshold),
        "--vjepa-selection-policy",
        selection_policy,
        "--max-new-tokens",
        str(benchmark_args.max_new_tokens),
        "--attn-implementation",
        str(benchmark_args.attn_implementation),
        "--visualization-output-dir",
        str(
            Path(benchmark_args.visualization_output_dir)
            if benchmark_args.visualization_output_dir
            else output_json.parent / "visualizations"
        ),
        "--visualization-max-frames",
        str(benchmark_args.visualization_max_frames),
    ]
    _append_optional(argv, "--gazing-ratio", benchmark_args.gazing_ratio)
    _append_optional(argv, "--task-loss-requirement", benchmark_args.task_loss_requirement)
    _append_optional(argv, "--autogaze-encoder-patch-size", benchmark_args.autogaze_encoder_patch_size)
    _append_optional(argv, "--video-resize-shortest-edge", benchmark_args.video_resize_shortest_edge)
    _append_optional(argv, "--video-resize-longest-edge", benchmark_args.video_resize_longest_edge)
    _append_optional(argv, "--video-resize-width", benchmark_args.video_resize_width)
    _append_optional(argv, "--video-resize-height", benchmark_args.video_resize_height)
    if benchmark_args.autogaze_generate_only:
        argv.append("--autogaze-generate-only")
    argv.append("--require-cuda" if benchmark_args.require_cuda else "--no-require-cuda")
    argv.append(
        "--clear-cuda-cache-between-stages"
        if benchmark_args.clear_cuda_cache_between_stages
        else "--no-clear-cuda-cache-between-stages"
    )
    return argv


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest_file(args.manifest)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    modes = resolve_modes(args)
    if getattr(args, "dry_run", False):
        return build_dry_run_plan(args=args, rows=rows, modes=modes, output_dir=output_dir, runs_dir=runs_dir)
    predictions: list[dict[str, Any]] = []
    runner_parser = build_runner_parser()

    for mode in modes:
        for row_index, row in enumerate(rows):
            video_path = resolve_hlvid_video_path(args.video_root, str(row["video_path"]))
            run_json = runs_dir / mode / f"{row_index:05d}.json"
            runner_args = runner_parser.parse_args(
                build_runner_args_for_row(
                    row=row,
                    video_path=video_path,
                    output_json=run_json,
                    mode=mode,
                    benchmark_args=args,
                )
            )
            try:
                payload = run_actual_pipeline(runner_args)
            except VjepaQwenStageError as exc:
                payload = _failure_payload(args, row, mode, exc.cause, stage=exc.stage)
                if not args.continue_on_error:
                    write_json(run_json, payload)
                    raise
            except Exception as exc:
                payload = _failure_payload(args, row, mode, exc, stage="unknown")
                if not args.continue_on_error:
                    write_json(run_json, payload)
                    raise
            write_json(run_json, payload)
            write_markdown_summary(run_json.with_suffix(".md"), payload)
            predictions.append(_prediction_from_payload(row, video_path, mode, payload))

    summary, scored = score_predictions(predictions)
    predictions_path = output_dir / "vjepa_qwen_hlvid_predictions.jsonl"
    scored_path = output_dir / "vjepa_qwen_hlvid_scored.jsonl"
    summary_path = output_dir / "vjepa_qwen_hlvid_summary.json"
    report_path = output_dir / "vjepa_qwen_hlvid_report.md"
    write_jsonl(predictions_path, predictions)
    write_jsonl(scored_path, scored)
    write_json(summary_path, summary)
    report_path.write_text(build_markdown_report(summary, predictions), encoding="utf-8")
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


def build_dry_run_plan(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    modes: list[str],
    output_dir: Path,
    runs_dir: Path,
) -> dict[str, Any]:
    plan = []
    runner_parser = build_runner_parser()
    for mode in modes:
        for row_index, row in enumerate(rows):
            video_path = resolve_hlvid_video_path(args.video_root, str(row["video_path"]))
            run_json = runs_dir / mode / f"{row_index:05d}.json"
            runner_argv = build_runner_args_for_row(
                row=row,
                video_path=video_path,
                output_json=run_json,
                mode=mode,
                benchmark_args=args,
            )
            parsed = runner_parser.parse_args(runner_argv)
            plan.append(
                {
                    "mode": mode,
                    "row_index": row_index,
                    "question_id": row.get("question_id"),
                    "video_path": str(video_path),
                    "output_json": str(run_json),
                    "autogaze_mode": parsed.autogaze_mode,
                    "vjepa_selection_policy": parsed.vjepa_selection_policy,
                    "requires_cuda": bool(parsed.require_cuda),
                    "runner_argv": runner_argv,
                }
            )
    summary = {
        "dry_run": True,
        "row_count": len(rows),
        "mode_count": len(modes),
        "planned_run_count": len(plan),
        "modes": modes,
    }
    plan_path = output_dir / "vjepa_qwen_hlvid_dry_run_plan.json"
    payload = {
        "runner": "repro.vjepa_qwen_hlvid_benchmark",
        "summary": summary,
        "artifacts": {"dry_run_plan": str(plan_path)},
        "config": {
            "manifest": str(args.manifest),
            "video_root": str(args.video_root),
            "output_dir": str(output_dir),
            "autogaze_model": str(args.autogaze_model),
            "vjepa_model": str(args.vjepa_model),
            "qwen_model": str(args.qwen_model),
            "num_video_frames": int(args.num_video_frames),
            "frames_per_clip": int(args.frames_per_clip),
            "autogaze_target_scales": str(args.autogaze_target_scales),
        },
        "plan": plan,
    }
    write_json(plan_path, payload)
    return payload


def resolve_modes(args: argparse.Namespace) -> list[str]:
    modes = [item.strip() for item in str(args.vjepa_qwen_modes or "").split(",") if item.strip()]
    if modes:
        return modes
    policies = [item.strip() for item in str(args.vjepa_selection_policies).split(",") if item.strip()]
    return [policy_to_mode(policy) for policy in policies]


def policy_to_mode(policy: str) -> str:
    if policy == "single_scale_union":
        return "autogaze_single_grid"
    if policy == "scale_aware_multi_pass":
        return "autogaze_scale_aware"
    raise ValueError(f"Unsupported V-JEPA selection policy: {policy}")


def mode_to_runner_settings(mode: str) -> tuple[str, str]:
    if mode == "dense_off":
        return "off", "single_scale_union"
    if mode == "autogaze_single_grid":
        return "on", "single_scale_union"
    if mode == "autogaze_scale_aware":
        return "on", "scale_aware_multi_pass"
    raise ValueError(f"Unsupported V-JEPA+Qwen mode: {mode}")


def build_markdown_report(summary: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    policy_rows = []
    for policy in sorted({row["mode"] for row in predictions}):
        rows = [row for row in predictions if row["mode"] == policy]
        policy_summary, _ = score_predictions(rows)
        policy_rows.append(
            "| {policy} | {total} | {correct} | {failed} | {parse_failed} | {accuracy_total:.4f} | {ag_tokens} | {vjepa_tokens} |".format(
                policy=policy,
                total=policy_summary["total"],
                correct=policy_summary["correct"],
                failed=policy_summary["failed"],
                parse_failed=policy_summary["parse_failed"],
                accuracy_total=policy_summary["accuracy_total"],
                ag_tokens=_median_present(rows, "autogaze_selected_patch_tokens"),
                vjepa_tokens=_median_present(rows, "vjepa_selected_tokens"),
            )
        )
    lines = [
        "# V-JEPA + Qwen HLVid Benchmark",
        "",
        "이 벤치마크는 AutoGaze 실제 selector output을 V-JEPA sparse encoder와 Qwen bridge로 연결한 zero-shot wiring probe입니다. 정확도는 projector 학습 전까지 참고용입니다.",
        "",
        "| policy | total | correct | failed | parse_failed | accuracy_total | median AG selected patches | median V-JEPA selected tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *policy_rows,
        "",
        "## Overall Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(args)
    print(json.dumps(payload["artifacts"], indent=2, sort_keys=True))
    return 0


def _prediction_from_payload(row: dict[str, Any], video_path: Path, policy: str, payload: dict[str, Any]) -> dict[str, Any]:
    tokens = payload.get("tokens") or {}
    latency = payload.get("latency_ms") or {}
    failure = payload.get("failure")
    return {
        "mode": policy,
        "question_id": row.get("question_id"),
        "category": row.get("category"),
        "video_path": row.get("video_path"),
        "resolved_video_path": str(video_path),
        "question": row.get("question") or row.get("prompt"),
        "answer": row.get("answer"),
        "raw_output": payload.get("generated_text"),
        "status": "ok" if payload.get("status") == "passed" else "failed",
        "runner_status": payload.get("status"),
        "accuracy_status": payload.get("accuracy_status"),
        "integration_level": payload.get("integration_level"),
        "failure": failure,
        "failure_stage": (failure or {}).get("stage") if isinstance(failure, dict) else None,
        "autogaze_raw_patch_tokens": tokens.get("autogaze_raw_patch_tokens"),
        "autogaze_selected_patch_tokens": tokens.get("autogaze_selected_patch_tokens"),
        "autogaze_reduction_ratio": tokens.get("autogaze_reduction_ratio"),
        "vjepa_raw_tokens": tokens.get("vjepa_raw_tokens"),
        "vjepa_selected_tokens": tokens.get("vjepa_selected_tokens"),
        "vjepa_reduction_ratio": tokens.get("vjepa_reduction_ratio"),
        "qwen_visual_tokens_inserted": tokens.get("qwen_visual_tokens_inserted"),
        "qwen_context_tokens": tokens.get("qwen_context_tokens"),
        "total_ms": latency.get("total"),
        "autogaze_selector_total_ms": latency.get("autogaze_selector_total"),
        "vjepa_video_decode_resize_ms": latency.get("vjepa_video_decode_resize"),
        "vjepa_sparse_encode_ms": latency.get("vjepa_sparse_encode"),
        "qwen_generate_ms": latency.get("qwen_generate"),
    }


def _failure_payload(args: argparse.Namespace, row: dict[str, Any], policy: str, exc: BaseException, *, stage: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "runner": "repro.vjepa_qwen_hlvid_benchmark",
        "accuracy_status": "not_claimed",
        "integration_level": "autogaze_actual_to_vjepa_sparse_encoder_to_qwen_inputs_embeds_zero_shot_probe",
        "mode": policy,
        "question_id": row.get("question_id"),
        "models": {
            "autogaze": str(args.autogaze_model),
            "vjepa": str(args.vjepa_model),
            "qwen": str(args.qwen_model),
        },
        "failure": {
            "stage": stage,
            "kind": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def _append_optional(argv: list[str], flag: str, value: Any) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def _median_present(rows: list[dict[str, Any]], key: str) -> Any:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


if __name__ == "__main__":
    raise SystemExit(main())
