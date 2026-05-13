#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from poc_infer_utils import (
    ProgressReporter,
    build_metrics,
    cli_or_config,
    load_config,
    nested_get,
    normalize_device,
    prepare_video,
    resolve_frame_selection_max_windows,
    run_autogaze_stage,
    write_autogaze_artifacts,
    write_json,
    write_summary_and_metrics,
    write_visualizations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Priority 1 AutoGaze-only PoC inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)

    parser.add_argument("--frame-selection-mode", choices=["sample", "chunk", "interval", "all"], default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None, help="maximum frame windows to process; 0 means unlimited")

    parser.add_argument("--scaling-mode", choices=["resize", "fit_short_side", "fit_long_side", "chop", "resize_then_chop", "quickstart", "none"], default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--chop-size", type=int, default=None)
    parser.add_argument("--chop-overlap", type=int, default=None)
    parser.add_argument("--max-chops", type=int, default=None)
    parser.add_argument("--chop-merge-mode", choices=["none", "metadata_only", "overlay_union"], default=None)
    parser.add_argument("--resize-before-chop-threshold", type=int, default=None)
    parser.add_argument("--resize-before-chop-factor", type=float, default=None)

    parser.add_argument("--gaze-ratio", type=float, default=None)
    parser.add_argument("--task-loss-requirement", type=float, default=None)
    parser.add_argument("--strict-autogaze-params", action="store_true")
    parser.add_argument("--allow-real-model-loading", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--overlay-style", choices=["mask", "box", "both"], default=None)
    parser.add_argument("--overlay-alpha", type=float, default=None)
    parser.add_argument("--multi-scale-overlay", action="store_true", default=None)
    parser.add_argument("--show-patch-index", action="store_true")
    parser.add_argument("--show-scale-label", action="store_true")
    parser.add_argument("--metadata-placement", choices=["outside", "inside", "none"], default=None)
    parser.add_argument("--info-panel-position", choices=["bottom", "right"], default=None)
    parser.add_argument("--save-frame-images", action="store_true")
    parser.add_argument("--save-overlay-video", action="store_true")
    parser.add_argument("--save-side-by-side-video", action="store_true")
    parser.add_argument("--save-scale-panel-video", action="store_true")
    parser.add_argument("--video-export-mode", choices=["sampled_only", "full_length"], default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.perf_counter()
    cfg = load_config(args.config)
    device = normalize_device(str(cli_or_config(args.device, cfg, "runtime.device", "cpu")))
    dtype = str(cli_or_config(args.dtype, cfg, "runtime.dtype", "float32"))
    warmup_runs = max(0, int(cli_or_config(args.warmup_runs, cfg, "runtime.warmup_runs", 1)))
    progress = ProgressReporter(enabled=not args.no_progress and bool(nested_get(cfg, "runtime.progress", True)))
    output_dir = Path(str(cli_or_config(args.output_dir, cfg, "output.output_dir", "outputs/poc_autogaze"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    preprocessing_start = time.perf_counter()
    frame_selection_mode = str(cli_or_config(args.frame_selection_mode, cfg, "frame_selection.mode", "sample"))
    prepared = prepare_video(
        cfg,
        video_path=args.video_path,
        frame_selection_mode=frame_selection_mode,
        num_frames=int(cli_or_config(args.num_frames, cfg, "frame_selection.num_frames", 16)),
        frame_interval=int(cli_or_config(args.frame_interval, cfg, "frame_selection.frame_interval", 1)),
        max_windows=resolve_frame_selection_max_windows(
            cli_max_windows=args.max_windows,
            cfg=cfg,
            frame_selection_mode=frame_selection_mode,
            cli_frame_selection_mode=args.frame_selection_mode,
        ),
        scaling_mode=str(cli_or_config(args.scaling_mode, cfg, "scaling.mode", "resize")),
        resolution=int(cli_or_config(args.resolution, cfg, "scaling.resolution", 224)),
        chop_size=int(cli_or_config(args.chop_size, cfg, "scaling.chop_size", 224)),
        chop_overlap=int(cli_or_config(args.chop_overlap, cfg, "scaling.chop_overlap", 0)),
        max_chops=cli_or_config(args.max_chops, cfg, "scaling.max_chops", None),
        chop_merge_mode=str(cli_or_config(args.chop_merge_mode, cfg, "scaling.chop_merge_mode", "metadata_only")),
        resize_before_chop_threshold=int(cli_or_config(args.resize_before_chop_threshold, cfg, "scaling.resize_before_chop_threshold", 1024)),
        resize_before_chop_factor=float(cli_or_config(args.resize_before_chop_factor, cfg, "scaling.resize_before_chop_factor", 0.5)),
    )
    preprocessing_latency_ms = (time.perf_counter() - preprocessing_start) * 1000
    allow_real = bool(args.allow_real_model_loading or nested_get(cfg, "runtime.allow_real_model_loading", False))
    gaze = run_autogaze_stage(
        cfg,
        prepared,
        device=device,
        dtype=dtype,
        gaze_ratio=float(cli_or_config(args.gaze_ratio, cfg, "autogaze.gaze_ratio", 0.75)),
        task_loss_requirement=cli_or_config(args.task_loss_requirement, cfg, "autogaze.task_loss_requirement", 0.7),
        strict_autogaze_params=bool(args.strict_autogaze_params or nested_get(cfg, "autogaze.strict_params", False)),
        allow_real_model_loading=allow_real,
        warmup_runs=warmup_runs if allow_real else 0,
        progress=progress,
    )

    artifacts = write_autogaze_artifacts(output_dir, prepared, gaze)
    visualization_start = time.perf_counter()
    artifacts.update(
        write_visualizations(
            output_dir,
            prepared,
            gaze,
            overlay_style=str(cli_or_config(args.overlay_style, cfg, "visualization.overlay_style", "mask")),
            overlay_alpha=float(cli_or_config(args.overlay_alpha, cfg, "visualization.overlay_alpha", 0.35)),
            multi_scale_overlay=bool(args.multi_scale_overlay or nested_get(cfg, "visualization.multi_scale_overlay", True)),
            show_patch_index=bool(args.show_patch_index or nested_get(cfg, "visualization.show_patch_index", False)),
            show_scale_label=bool(args.show_scale_label or nested_get(cfg, "visualization.show_scale_label", False)),
            metadata_placement=str(cli_or_config(args.metadata_placement, cfg, "visualization.metadata_placement", "outside")),
            info_panel_position=str(cli_or_config(args.info_panel_position, cfg, "visualization.info_panel_position", "bottom")),
            save_frame_images=bool(args.save_frame_images or nested_get(cfg, "visualization.save_frame_images", False)),
            save_overlay_video=bool(args.save_overlay_video or nested_get(cfg, "visualization.save_overlay_video", False)),
            save_side_by_side_video=bool(args.save_side_by_side_video or nested_get(cfg, "visualization.save_side_by_side_video", False)),
            save_scale_panel_video=bool(args.save_scale_panel_video or nested_get(cfg, "visualization.save_scale_panel_video", False)),
            video_fps=float(cli_or_config(args.video_fps, cfg, "visualization.video_fps", 4.0)),
            video_export_mode=str(cli_or_config(args.video_export_mode, cfg, "visualization.video_export_mode", "sampled_only")),
        )
    )
    visualization_latency_ms = (time.perf_counter() - visualization_start) * 1000

    skipped = []
    if gaze.status == "blocked":
        skipped.append({"stage": "autogaze", "reason": gaze.reason or "AutoGaze blocked"})
    elif gaze.status.startswith("stub"):
        skipped.append({"stage": "autogaze", "reason": gaze.reason or "stub AutoGaze output"})
    module_processing_latency_ms = gaze.latency_ms
    wall_clock_latency_ms = (time.perf_counter() - run_start) * 1000
    metrics = build_metrics(
        mode="autogaze_only",
        cfg=cfg,
        video_path=args.video_path,
        query_text=None,
        prepared=prepared,
        gaze=gaze,
        requested_vision_encoder=nested_get(cfg, "vision_encoder.name"),
        actual_vision_encoder=None,
        requested_mllm=None,
        actual_mllm=None,
        generation_status="not_applicable",
        output_text=None,
        skipped_stages=skipped,
        total_latency_ms=module_processing_latency_ms,
        preprocessing_latency_ms=preprocessing_latency_ms,
        visualization_latency_ms=visualization_latency_ms,
        wall_clock_latency_ms=wall_clock_latency_ms,
        warmup_runs=warmup_runs,
    )
    summary = {
        "mode": "autogaze_only",
        "status": "blocked" if gaze.status == "blocked" else "partial" if gaze.status.startswith("stub") else "completed",
        "experiment_id": nested_get(cfg, "experiment.id", "unknown"),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "metrics": metrics,
    }
    artifacts.update(write_summary_and_metrics(output_dir, summary=summary, metrics=metrics))
    if nested_get(cfg, "output.write_answer_not_applicable", False):
        write_json(output_dir / "predictions" / "answer.json", {"status": "not_applicable", "reason": "AutoGaze-only mode"})
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
