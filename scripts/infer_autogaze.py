#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from poc_infer_utils import (
    ProgressReporter,
    StreamingVisualizationSink,
    add_nvila_hd_cli_args,
    add_streaming_window_result,
    apply_nvila_hd_overrides,
    build_metrics,
    build_streaming_metrics,
    cli_or_config,
    iter_stream_windows,
    load_config,
    nested_get,
    new_streaming_aggregate,
    nvila_hd_gaze_ratio,
    nvila_hd_task_loss_requirement,
    normalize_device,
    prepare_stream_window,
    prepare_video,
    resolve_frame_selection_max_windows,
    run_autogaze_stage,
    validate_prepared_video_memory,
    validate_stream_window_memory,
    write_autogaze_artifacts,
    write_json,
    write_summary_and_metrics,
    write_streaming_autogaze_artifacts,
    write_visualizations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Priority 1 AutoGaze-only PoC inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    add_nvila_hd_cli_args(parser)

    parser.add_argument("--frame-selection-mode", choices=["sample", "chunk", "interval", "all"], default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None, help="maximum frame windows to process; 0 means unlimited")
    parser.add_argument("--video-read-mode", choices=["streaming", "full"], default=None)
    parser.add_argument("--stream-window-size", type=int, default=None)
    parser.add_argument("--stream-overlap", type=int, default=None)
    parser.add_argument("--max-stream-windows", type=int, default=None)
    parser.add_argument("--max-decode-frames", type=int, default=None)
    parser.add_argument("--decode-backend", choices=["auto", "opencv", "decord", "torchvision"], default=None)
    parser.add_argument("--decode-fps", type=float, default=None)
    parser.add_argument("--resize-before-buffer", action="store_true", default=None)
    parser.add_argument("--streaming-output", action="store_true", default=None)
    parser.add_argument("--flush-every-window", action="store_true", default=None)
    parser.add_argument("--cpu-offload-between-windows", action="store_true", default=None)
    parser.add_argument("--empty-cache-between-windows", action="store_true", default=None)
    parser.add_argument("--max-pixels-per-window", type=int, default=None)
    parser.add_argument("--max-frames-in-memory", type=int, default=None)
    parser.add_argument("--max-processed-frames-per-window", type=int, default=None)
    parser.add_argument("--max-processed-pixels-per-window", type=int, default=None)
    parser.add_argument("--fail-on-full-video-load", action="store_true", default=None)

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
    cfg = apply_nvila_hd_overrides(load_config(args.config), args)
    device = normalize_device(str(cli_or_config(args.device, cfg, "runtime.device", "cpu")))
    requested_dtype = str(cli_or_config(args.dtype, cfg, "runtime.dtype", "float32"))
    autogaze_dtype = "float32"
    warmup_runs = max(0, int(cli_or_config(args.warmup_runs, cfg, "runtime.warmup_runs", 1)))
    progress = ProgressReporter(enabled=not args.no_progress and bool(nested_get(cfg, "runtime.progress", True)))
    output_dir = Path(str(cli_or_config(args.output_dir, cfg, "output.output_dir", "outputs/poc_autogaze"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    video_read_mode = str(cli_or_config(args.video_read_mode, cfg, "video_input.read_mode", "full"))
    if video_read_mode == "streaming":
        return _run_streaming(args, cfg=cfg, output_dir=output_dir, run_start=run_start)
    fail_on_full = bool(cli_or_config(args.fail_on_full_video_load, cfg, "memory.fail_on_full_video_load", True))
    if fail_on_full and args.video_path != "dummy":
        raise RuntimeError(
            "full video loading is disabled by memory.fail_on_full_video_load; use --video-read-mode streaming "
            "or set memory.fail_on_full_video_load=false in config for an explicit full-load run"
        )

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
        max_decode_frames=cli_or_config(args.max_decode_frames, cfg, "video_input.max_decode_frames", None),
        max_frames_in_memory=cli_or_config(args.max_frames_in_memory, cfg, "memory.max_video_frames_in_memory", None),
        max_pixels_per_window=cli_or_config(args.max_pixels_per_window, cfg, "memory.max_pixels_per_window", None),
    )
    validate_prepared_video_memory(
        prepared,
        max_processed_frames_per_window=cli_or_config(
            args.max_processed_frames_per_window,
            cfg,
            "memory.max_processed_frames_per_window",
            None,
        ),
        max_processed_pixels_per_window=cli_or_config(
            args.max_processed_pixels_per_window,
            cfg,
            "memory.max_processed_pixels_per_window",
            None,
        ),
    )
    preprocessing_latency_ms = (time.perf_counter() - preprocessing_start) * 1000
    allow_real = bool(args.allow_real_model_loading or nested_get(cfg, "runtime.allow_real_model_loading", False))
    gaze = run_autogaze_stage(
        cfg,
        prepared,
        device=device,
        dtype=autogaze_dtype,
        requested_dtype=requested_dtype,
        gaze_ratio=nvila_hd_gaze_ratio(cfg, args.gaze_ratio),
        task_loss_requirement=nvila_hd_task_loss_requirement(cfg, args.task_loss_requirement),
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


def _run_streaming(args: argparse.Namespace, *, cfg: dict[str, Any], output_dir: Path, run_start: float) -> dict[str, Any]:
    device = normalize_device(str(cli_or_config(args.device, cfg, "runtime.device", "cpu")))
    requested_dtype = str(cli_or_config(args.dtype, cfg, "runtime.dtype", "float32"))
    autogaze_dtype = "float32"
    warmup_runs = max(0, int(cli_or_config(args.warmup_runs, cfg, "runtime.warmup_runs", 1)))
    progress = ProgressReporter(enabled=not args.no_progress and bool(nested_get(cfg, "runtime.progress", True)))
    frame_selection_mode = str(cli_or_config(args.frame_selection_mode, cfg, "frame_selection.mode", "chunk"))
    num_frames = int(cli_or_config(args.num_frames, cfg, "frame_selection.num_frames", 16))
    frame_interval = int(cli_or_config(args.frame_interval, cfg, "frame_selection.frame_interval", 1))
    frame_max_windows = resolve_frame_selection_max_windows(
        cli_max_windows=args.max_windows,
        cfg=cfg,
        frame_selection_mode=frame_selection_mode,
        cli_frame_selection_mode=args.frame_selection_mode,
    )
    stream_window_size_value = (
        args.stream_window_size
        if args.stream_window_size is not None
        else num_frames
        if args.num_frames is not None
        else nested_get(cfg, "streaming.window_size", num_frames)
    )
    stream_window_size = int(stream_window_size_value)
    stream_overlap = int(cli_or_config(args.stream_overlap, cfg, "streaming.overlap", 0))
    max_stream_windows = cli_or_config(args.max_stream_windows, cfg, "streaming.max_windows", frame_max_windows)
    max_decode_frames = cli_or_config(args.max_decode_frames, cfg, "video_input.max_decode_frames", None)
    decode_backend = str(cli_or_config(args.decode_backend, cfg, "video_input.decode_backend", "auto"))
    decode_fps = cli_or_config(args.decode_fps, cfg, "video_input.decode_fps", None)
    max_pixels_per_window = cli_or_config(args.max_pixels_per_window, cfg, "memory.max_pixels_per_window", None)
    max_frames_in_memory = cli_or_config(args.max_frames_in_memory, cfg, "memory.max_video_frames_in_memory", None)
    max_processed_frames_per_window = cli_or_config(
        args.max_processed_frames_per_window,
        cfg,
        "memory.max_processed_frames_per_window",
        None,
    )
    max_processed_pixels_per_window = cli_or_config(
        args.max_processed_pixels_per_window,
        cfg,
        "memory.max_processed_pixels_per_window",
        None,
    )
    scaling_mode = str(cli_or_config(args.scaling_mode, cfg, "scaling.mode", "resize"))
    resolution = int(cli_or_config(args.resolution, cfg, "scaling.resolution", 224))
    dummy_frames = int(nested_get(cfg, "input.dummy_frames", max(num_frames, 8)))
    dummy_resolution = int(nested_get(cfg, "input.dummy_resolution", max(resolution, 64)))
    allow_real = bool(args.allow_real_model_loading or nested_get(cfg, "runtime.allow_real_model_loading", False))
    aggregate = new_streaming_aggregate(
        video_read_mode="streaming",
        decode_backend=decode_backend,
        frame_selection_mode=frame_selection_mode,
        stream_window_size=stream_window_size,
        stream_overlap=stream_overlap,
        max_stream_windows=max_stream_windows,
        max_decode_frames=max_decode_frames,
        decode_fps=decode_fps,
    )
    viz_sink = StreamingVisualizationSink(
        output_dir,
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
    skipped: list[dict[str, str]] = []
    status = "completed"
    reason = None
    total_module_latency = 0.0
    visualization_latency_total = 0.0
    for stream_window in iter_stream_windows(
        args.video_path,
        frame_selection_mode=frame_selection_mode,
        num_frames=num_frames,
        frame_interval=frame_interval,
        max_windows=max_stream_windows,
        stream_window_size=stream_window_size,
        stream_overlap=stream_overlap,
        max_decode_frames=max_decode_frames,
        decode_backend=decode_backend,
        decode_fps=decode_fps,
        dummy_frames=dummy_frames,
        dummy_resolution=dummy_resolution,
    ):
        validate_stream_window_memory(
            stream_window,
            max_frames_in_memory=max_frames_in_memory,
            max_pixels_per_window=max_pixels_per_window,
        )
        preprocessing_start = time.perf_counter()
        prepared = prepare_stream_window(
            cfg,
            stream_window=stream_window,
            frame_selection_mode=frame_selection_mode,
            num_frames=num_frames,
            frame_interval=frame_interval,
            max_windows=max_stream_windows,
            scaling_mode=scaling_mode,
            resolution=resolution,
            chop_size=int(cli_or_config(args.chop_size, cfg, "scaling.chop_size", 224)),
            chop_overlap=int(cli_or_config(args.chop_overlap, cfg, "scaling.chop_overlap", 0)),
            max_chops=cli_or_config(args.max_chops, cfg, "scaling.max_chops", None),
            chop_merge_mode=str(cli_or_config(args.chop_merge_mode, cfg, "scaling.chop_merge_mode", "metadata_only")),
            resize_before_chop_threshold=int(cli_or_config(args.resize_before_chop_threshold, cfg, "scaling.resize_before_chop_threshold", 1024)),
            resize_before_chop_factor=float(cli_or_config(args.resize_before_chop_factor, cfg, "scaling.resize_before_chop_factor", 0.5)),
        )
        validate_prepared_video_memory(
            prepared,
            max_processed_frames_per_window=max_processed_frames_per_window,
            max_processed_pixels_per_window=max_processed_pixels_per_window,
        )
        preprocessing_latency_ms = (time.perf_counter() - preprocessing_start) * 1000
        gaze = run_autogaze_stage(
            cfg,
            prepared,
            device=device,
            dtype=autogaze_dtype,
            requested_dtype=requested_dtype,
            gaze_ratio=nvila_hd_gaze_ratio(cfg, args.gaze_ratio),
            task_loss_requirement=nvila_hd_task_loss_requirement(cfg, args.task_loss_requirement),
            strict_autogaze_params=bool(args.strict_autogaze_params or nested_get(cfg, "autogaze.strict_params", False)),
            allow_real_model_loading=allow_real,
            warmup_runs=warmup_runs if allow_real and stream_window.window_id == 0 else 0,
            progress=progress,
        )
        window_skipped: list[dict[str, str]] = []
        if gaze.status == "blocked":
            status = "blocked"
            reason = gaze.reason or "AutoGaze blocked"
            window_skipped.append({"stage": "autogaze", "reason": reason})
            skipped.append({"stage": f"window_{stream_window.window_id}_autogaze", "reason": reason})
        elif gaze.status.startswith("stub") and status != "blocked":
            status = "partial"
            window_skipped.append({"stage": "autogaze", "reason": gaze.reason or "stub AutoGaze output"})
        processed_offset = int(aggregate["processed_frame_count"])
        visualization_offset = int(aggregate["visualization_frame_count"])
        visualization_start = time.perf_counter()
        viz_sink.write_window(
            prepared,
            gaze,
            processed_frame_offset=processed_offset,
            visualization_frame_offset=visualization_offset,
        )
        visualization_latency_ms = (time.perf_counter() - visualization_start) * 1000
        visualization_latency_total += visualization_latency_ms
        add_streaming_window_result(
            aggregate,
            stream_window=stream_window,
            prepared=prepared,
            gaze=gaze,
            preprocessing_latency_ms=preprocessing_latency_ms,
            autogaze_latency_ms=gaze.latency_ms,
            visualization_latency_ms=visualization_latency_ms,
            generation_status="not_applicable",
            skipped_stages=window_skipped,
        )
        total_module_latency += gaze.latency_ms
        if bool(cli_or_config(args.empty_cache_between_windows, cfg, "streaming.empty_cache_between_windows", False)) and device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    artifacts = write_streaming_autogaze_artifacts(output_dir, aggregate, status=status, reason=reason)
    artifacts.update(viz_sink.close())
    wall_clock_latency_ms = (time.perf_counter() - run_start) * 1000
    metrics = build_streaming_metrics(
        mode="autogaze_only",
        cfg=cfg,
        video_path=args.video_path,
        query_text=None,
        aggregate=aggregate,
        requested_vision_encoder=nested_get(cfg, "vision_encoder.name"),
        actual_vision_encoder=None,
        requested_mllm=None,
        actual_mllm=None,
        generation_status="not_applicable",
        output_text=None,
        skipped_stages=skipped,
        total_latency_ms=total_module_latency,
        wall_clock_latency_ms=wall_clock_latency_ms,
        visualization_latency_ms=visualization_latency_total,
        warmup_runs=warmup_runs,
    )
    summary = {
        "mode": "autogaze_only",
        "status": status,
        "experiment_id": nested_get(cfg, "experiment.id", "unknown"),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "metrics": metrics,
    }
    artifacts.update(write_summary_and_metrics(output_dir, summary=summary, metrics=metrics))
    write_json(output_dir / "logs" / "streaming_metrics.json", metrics)
    if nested_get(cfg, "output.write_answer_not_applicable", False):
        write_json(output_dir / "predictions" / "answer.json", {"status": "not_applicable", "reason": "AutoGaze-only streaming mode"})
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
