#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
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
from poc_model_adapters import AdapterStatus
from poc_model_registry import build_mllm, build_vision_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Priority 1 full PoC inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--query-text", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--mllm-dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)

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

    parser.add_argument("--vision-encoder", default=None)
    parser.add_argument("--vision-encoder-ckpt", default=None)
    parser.add_argument("--vision-encoder-config", default=None)
    parser.add_argument("--vision-encoder-module", default=None)
    parser.add_argument("--vision-encoder-class", default=None)
    parser.add_argument("--mllm", default=None)
    parser.add_argument("--mllm-ckpt", default=None)
    parser.add_argument("--mllm-config", default=None)
    parser.add_argument("--mllm-module", default=None)
    parser.add_argument("--mllm-class", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.perf_counter()
    cfg = _with_model_overrides(load_config(args.config), args)
    device = normalize_device(str(cli_or_config(args.device, cfg, "runtime.device", "cpu")))
    requested_dtype = str(cli_or_config(args.dtype, cfg, "runtime.dtype", "float32"))
    autogaze_dtype = "float32"
    vision_dtype = str(nested_get(cfg, "runtime.vision_dtype", requested_dtype))
    mllm_dtype = str(cli_or_config(args.mllm_dtype, cfg, "runtime.mllm_dtype", requested_dtype))
    warmup_runs = max(0, int(cli_or_config(args.warmup_runs, cfg, "runtime.warmup_runs", 1)))
    progress = ProgressReporter(enabled=not args.no_progress and bool(nested_get(cfg, "runtime.progress", True)))
    output_dir = Path(str(cli_or_config(args.output_dir, cfg, "output.output_dir", "outputs/poc_full"))).expanduser()
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
        dtype=autogaze_dtype,
        requested_dtype=requested_dtype,
        gaze_ratio=float(cli_or_config(args.gaze_ratio, cfg, "autogaze.gaze_ratio", 0.75)),
        task_loss_requirement=cli_or_config(args.task_loss_requirement, cfg, "autogaze.task_loss_requirement", 0.7),
        strict_autogaze_params=bool(args.strict_autogaze_params or nested_get(cfg, "autogaze.strict_params", False)),
        allow_real_model_loading=allow_real,
        warmup_runs=warmup_runs if allow_real else 0,
        progress=progress,
    )
    artifacts = write_autogaze_artifacts(output_dir, prepared, gaze)

    skipped: list[dict[str, str]] = []
    if gaze.status == "blocked":
        skipped.append({"stage": "autogaze", "reason": gaze.reason or "AutoGaze blocked"})
    elif gaze.status.startswith("stub"):
        skipped.append({"stage": "autogaze", "reason": gaze.reason or "stub AutoGaze output"})

    vision_name = str(nested_get(cfg, "vision_encoder.name", "generic_vit"))
    mllm_name = str(nested_get(cfg, "mllm.name", "generic_mllm"))
    _sync_mllm_with_poc_autogaze_controls(cfg, gaze)
    mllm_owns_vision = bool(nested_get(cfg, "mllm.official_processor_owns_vision", mllm_name == "qwen"))
    generation_input_mode = str(nested_get(cfg, "mllm.generation_input_mode", "official_processor"))
    direct_visual_token_mode = generation_input_mode == "direct_visual_tokens"
    vision_required = bool(nested_get(cfg, "vision_encoder.required_for_full_pipeline", direct_visual_token_mode))
    vision = build_vision_encoder(vision_name, nested_get(cfg, "vision_encoder", {}))
    visual_tokens = None
    actual_vision_encoder = vision.name
    blocked_stages: list[str] = []
    vision_latency_ms: float | None = None
    if not vision_required:
        vision_status = AdapterStatus(
            vision.name,
            "skipped",
            f"{vision.name} skipped because {mllm_name} uses the official processor vision path",
        )
        vision.status = vision_status
        skipped.append(
            {
                "stage": "vision_encoder",
                "reason": vision_status.reason or "vision encoder skipped",
            }
        )
    else:
        vision_status = vision.load(allow_real_model_loading=allow_real, device=device, dtype=vision_dtype)
        if allow_real and vision_status.status != "real":
            skipped.append({"stage": "vision_encoder", "reason": vision_status.reason or f"{vision.name} real loading unavailable"})
            blocked_stages.append("vision_encoder")
        else:
            autogaze_payload = None if not gaze.autogaze_enabled or gaze.status == "blocked" else gaze.gazing_info_for_vit
            try:
                def vision_forward_once() -> dict[str, Any]:
                    return vision.forward(prepared.processed_video, autogaze=autogaze_payload)

                if allow_real and vision_status.status == "real":
                    progress.warmup("ViT encoder", vision_forward_once, runs=warmup_runs, device=device)
                vision_output, vision_latency_ms = progress.timed("ViT encoder", vision_forward_once, device=device)
                visual_tokens = vision_output.get("visual_tokens")
                if vision_status.status != "real":
                    skipped.append({"stage": "vision_encoder", "reason": vision_status.reason or f"{vision.name} used stub output"})
            except Exception as exc:
                skipped.append({"stage": "vision_encoder", "reason": f"vision encoder failed: {exc}"})
                if allow_real and vision_required:
                    blocked_stages.append("vision_encoder")

    mllm = build_mllm(mllm_name, nested_get(cfg, "mllm", {}))
    mllm_status = mllm.load(allow_real_model_loading=allow_real, device=device, dtype=mllm_dtype)
    max_new_tokens = int(cli_or_config(args.max_new_tokens, cfg, "generation.max_new_tokens", 32))
    mllm_autogaze_payload = _mllm_autogaze_payload(cfg, gaze)
    mllm_generation_latency_ms: float | None = None
    if allow_real and mllm_status.status != "real":
        blocked_stages.append("mllm_generation")
        generation = {
            "status": "blocked",
            "answer": None,
            "reason": mllm_status.reason or f"{mllm.name} real loading unavailable",
            "query_text_used": True,
        }
    elif direct_visual_token_mode and not mllm.supports_direct_visual_tokens():
        blocked_stages.append("mllm_generation")
        generation = {
            "status": "blocked",
            "answer": None,
            "reason": f"{mllm.name} does not support verified direct visual token injection; use official_processor mode",
            "query_text_used": True,
        }
    elif allow_real and vision_required and "vision_encoder" in blocked_stages and mllm.supports_direct_visual_tokens():
        generation = {
            "status": "blocked",
            "answer": None,
            "reason": "MLLM generation requires real visual tokens, but the requested vision encoder is blocked",
            "query_text_used": True,
        }
    else:
        mllm_video_input_source = str(nested_get(cfg, "mllm.video_input_source", "processed_tensor"))
        if mllm_video_input_source not in {"processed_tensor", "source_video"}:
            raise ValueError("mllm.video_input_source must be one of processed_tensor or source_video")
        mllm_video_path = args.video_path if mllm_video_input_source == "source_video" else None

        def mllm_generate_once() -> dict[str, Any]:
            return _generate_mllm_with_video_policy(
                mllm,
                query_text=args.query_text,
                video=prepared.processed_video,
                visual_tokens=visual_tokens if mllm.supports_direct_visual_tokens() else None,
                max_new_tokens=max_new_tokens,
                video_path=mllm_video_path,
                has_chop_metadata=prepared.chop_metadata is not None,
                autogaze=mllm_autogaze_payload,
            )

        if allow_real and mllm_status.status == "real":
            progress.warmup("MLLM", mllm_generate_once, runs=warmup_runs, device=device)
        generation, mllm_generation_latency_ms = progress.timed("MLLM", mllm_generate_once, device=device)
    generation_status = str(generation.get("status", "unknown"))
    output_text = generation.get("answer")
    if generation_status != "real":
        skipped.append({"stage": "mllm_generation", "reason": str(generation.get("reason") or mllm_status.reason or "generation unavailable")})
    answer_path = output_dir / "predictions" / "answer.json"
    write_json(
        answer_path,
        {
            "status": generation_status,
            "answer": output_text,
            "query_text": args.query_text,
            "query_text_used": bool(generation.get("query_text_used", True)),
            "skipped_reason": generation.get("reason"),
            "mllm": mllm.name,
            "max_new_tokens": max_new_tokens,
            "adapter_statuses": {
                "vision_encoder": vision_status.to_dict(),
                "mllm": mllm_status.to_dict(),
            },
            "generation_metadata": generation.get("metadata"),
        },
    )
    artifacts["answer"] = str(answer_path)
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
            query_text=args.query_text,
            generation_status=generation_status,
        )
    )
    visualization_latency_ms = (time.perf_counter() - visualization_start) * 1000

    module_processing_latency_ms = gaze.latency_ms + float(vision_latency_ms or 0.0) + float(mllm_generation_latency_ms or 0.0)
    wall_clock_latency_ms = (time.perf_counter() - run_start) * 1000
    metrics = build_metrics(
        mode="full_pipeline",
        cfg=cfg,
        video_path=args.video_path,
        query_text=args.query_text,
        prepared=prepared,
        gaze=gaze,
        requested_vision_encoder=vision_name,
        actual_vision_encoder=actual_vision_encoder,
        requested_mllm=mllm_name,
        actual_mllm=mllm.name,
        generation_status=generation_status,
        output_text=output_text,
        skipped_stages=skipped,
        total_latency_ms=module_processing_latency_ms,
        preprocessing_latency_ms=preprocessing_latency_ms,
        vision_encoder_latency_ms=vision_latency_ms,
        mllm_generation_latency_ms=mllm_generation_latency_ms,
        visualization_latency_ms=visualization_latency_ms,
        wall_clock_latency_ms=wall_clock_latency_ms,
        warmup_runs=warmup_runs,
    )
    metrics["adapter_statuses"] = {
        "vision_encoder": vision_status.to_dict(),
        "mllm": mllm_status.to_dict(),
    }
    metrics["requested_runtime_dtype"] = requested_dtype
    metrics["autogaze_dtype"] = autogaze_dtype
    metrics["vision_encoder_dtype"] = vision_dtype
    metrics["mllm_dtype"] = mllm_dtype
    metrics["vision_encoder_required_for_full_pipeline"] = vision_required
    metrics["generation_input_mode"] = "direct_visual_tokens" if direct_visual_token_mode else "official_processor"
    generation_metadata = generation.get("metadata") if isinstance(generation.get("metadata"), dict) else {}
    metrics["mllm_video_input_source"] = generation_metadata.get(
        "actual_video_input_source",
        "processed_chop_tensor" if prepared.chop_metadata is not None else "processed_tensor",
    )
    metrics["mllm_chop_tensor_attempted"] = bool(generation_metadata.get("chop_tensor_attempted", prepared.chop_metadata is not None))
    metrics["mllm_chop_source_fallback_used"] = bool(generation_metadata.get("chop_source_fallback_used", False))
    metrics["gazing_info_passed_to_vision_encoder"] = bool(gaze.gazing_info_for_vit is not None and vision_required)
    metrics["qwen_autogaze_integration"] = generation_metadata.get(
        "qwen_autogaze_integration",
        nested_get(cfg, "mllm.autogaze_integration", "none") if mllm.name == "qwen" else "not_applicable",
    )
    metrics["qwen_visual_mask_applied"] = bool(generation_metadata.get("qwen_visual_mask_applied", False))
    metrics["qwen_visual_tokens_shortened"] = bool(generation_metadata.get("qwen_visual_tokens_shortened", False))
    metrics["qwen_encoder_side_acceleration_claimed"] = bool(
        generation_metadata.get("qwen_encoder_side_acceleration_claimed", False)
    )
    for key in (
        "qwen_visual_tokens_before",
        "qwen_visual_tokens_kept_by_mask",
        "qwen_visual_mask_keep_ratio",
        "qwen_visual_mask_grid_thw",
        "qwen_autogaze_empty_temporal_chunks",
        "qwen_autogaze_empty_chunk_policy",
    ):
        if key in generation_metadata:
            metrics[key] = generation_metadata[key]
    metrics["mllm_visual_token_saving_claimed"] = bool(
        (
            direct_visual_token_mode
            and mllm.supports_direct_visual_tokens()
            and visual_tokens is not None
        )
        or (
            mllm.name == "nvila"
            and mllm_status.status == "real"
            and bool(nested_get(cfg, "autogaze.enabled", False))
            and bool(nested_get(cfg, "mllm.official_processor_path", False))
            and bool(nested_get(cfg, "mllm.sync_autogaze_controls_from_config", False))
        )
    )
    if gaze.status == "blocked":
        blocked_stages.append("autogaze")
    if blocked_stages:
        metrics["failure_reason"] = _failure_reason_for_blocked_stages(skipped, blocked_stages)
    else:
        metrics["failure_reason"] = None
    status = "blocked" if blocked_stages else "partial"
    if generation_status == "real" and not blocked_stages:
        status = "completed"
    summary = {
        "mode": "full_pipeline",
        "status": status,
        "experiment_id": nested_get(cfg, "experiment.id", "unknown"),
        "output_dir": str(output_dir),
        "query_text": args.query_text,
        "adapter_statuses": {
            "vision_encoder": vision_status.to_dict(),
            "mllm": mllm_status.to_dict(),
        },
        "blocked_stages": sorted(set(blocked_stages)),
        "artifacts": artifacts,
        "metrics": metrics,
    }
    artifacts.update(write_summary_and_metrics(output_dir, summary=summary, metrics=metrics))
    return summary


def _with_model_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    vision = updated.setdefault("vision_encoder", {})
    mllm = updated.setdefault("mllm", {})
    if args.vision_encoder:
        vision["name"] = args.vision_encoder
    if args.vision_encoder_ckpt:
        vision["checkpoint_path"] = args.vision_encoder_ckpt
    if args.vision_encoder_config:
        vision["config_path"] = args.vision_encoder_config
    if args.vision_encoder_module:
        vision["module_path"] = args.vision_encoder_module
    if args.vision_encoder_class:
        vision["class_name"] = args.vision_encoder_class
    if args.mllm:
        mllm["name"] = args.mllm
    if args.mllm_ckpt:
        mllm["checkpoint_path"] = args.mllm_ckpt
    if args.mllm_config:
        mllm["config_path"] = args.mllm_config
    if args.mllm_module:
        mllm["module_path"] = args.mllm_module
    if args.mllm_class:
        mllm["class_name"] = args.mllm_class
    if args.model_id:
        mllm["model_id"] = args.model_id
        if not args.mllm_ckpt:
            mllm["checkpoint_path"] = None
        if not args.processor_path:
            mllm["processor_path"] = args.model_id
        if not args.tokenizer_path:
            mllm["tokenizer_path"] = args.model_id
    if args.processor_path:
        mllm["processor_path"] = args.processor_path
    if args.tokenizer_path:
        mllm["tokenizer_path"] = args.tokenizer_path
    if args.trust_remote_code:
        mllm["trust_remote_code"] = True
        vision["trust_remote_code"] = True
    if args.local_files_only:
        mllm["local_files_only"] = True
        vision["local_files_only"] = True
    return updated


def _generate_mllm_with_video_policy(
    mllm: Any,
    *,
    query_text: str,
    video: Any,
    visual_tokens: Any,
    max_new_tokens: int,
    video_path: str | None,
    has_chop_metadata: bool,
    autogaze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = _call_mllm_generate(
        mllm,
        query_text=query_text,
        video=video,
        visual_tokens=visual_tokens,
        max_new_tokens=max_new_tokens,
        video_path=video_path,
        autogaze=autogaze,
    )
    if video_path:
        actual_source = "source_video_path"
    elif has_chop_metadata:
        actual_source = "processed_chop_tensor"
    else:
        actual_source = "processed_tensor"
    return _with_generation_metadata(
        result,
        {
            "actual_video_input_source": actual_source,
            "chop_tensor_attempted": bool(has_chop_metadata and not video_path),
            "chop_source_fallback_used": False,
        },
    )


def _call_mllm_generate(
    mllm: Any,
    *,
    query_text: str,
    video: Any,
    visual_tokens: Any,
    max_new_tokens: int,
    video_path: str | None,
    autogaze: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs = {
        "query_text": query_text,
        "video": video,
        "visual_tokens": visual_tokens,
        "max_new_tokens": max_new_tokens,
        "video_path": video_path,
    }
    if autogaze is not None and _generate_accepts_kwarg(mllm, "autogaze"):
        kwargs["autogaze"] = autogaze
    return mllm.generate(**kwargs)


def _generate_accepts_kwarg(mllm: Any, name: str) -> bool:
    try:
        signature = inspect.signature(mllm.generate)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _mllm_autogaze_payload(cfg: dict[str, Any], gaze: Any) -> dict[str, Any] | None:
    if str(nested_get(cfg, "mllm.autogaze_integration", "none")) != "qwen_vision_mask":
        return None
    return {
        "autogaze_enabled": bool(getattr(gaze, "autogaze_enabled", False)),
        "status": getattr(gaze, "status", None),
        "reason": getattr(gaze, "reason", None),
        "real_model_used": bool(getattr(gaze, "real_model_used", False)),
        "original_token_count": getattr(gaze, "original_token_count", None),
        "selected_token_count": getattr(gaze, "selected_token_count", None),
        "token_reduction_ratio": getattr(gaze, "token_reduction_ratio", None),
        "patch_grid": list(getattr(gaze, "patch_grid", []) or []),
        "patch_size": getattr(gaze, "patch_size", None),
        "per_frame": list(getattr(gaze, "per_frame", []) or []),
        "runtime_metadata": dict(getattr(gaze, "runtime_metadata", {}) or {}),
    }


def _with_generation_metadata(generation: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    updated = dict(generation)
    existing = updated.get("metadata")
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(metadata)
    updated["metadata"] = merged
    return updated


def _sync_mllm_with_poc_autogaze_controls(cfg: dict[str, Any], gaze: Any) -> None:
    mllm = cfg.setdefault("mllm", {})
    if not isinstance(mllm, dict) or not bool(mllm.get("sync_autogaze_controls_from_config", False)):
        return
    mllm["poc_autogaze_enabled"] = bool(nested_get(cfg, "autogaze.enabled", False))
    mllm["poc_autogaze_status"] = getattr(gaze, "status", None)
    mllm["poc_gaze_ratio"] = nested_get(cfg, "autogaze.gaze_ratio")
    mllm["poc_task_loss_requirement"] = nested_get(cfg, "autogaze.task_loss_requirement")
    mllm["poc_autogaze_checkpoint_path"] = nested_get(cfg, "autogaze.checkpoint_path")
    mllm["poc_autogaze_processor_path"] = nested_get(cfg, "autogaze.processor_path")


def _failure_reason_for_blocked_stages(skipped: list[dict[str, str]], blocked_stages: list[str]) -> str | None:
    blocked = set(blocked_stages)
    for item in skipped:
        stage = item.get("stage")
        if stage in blocked or (stage == "mllm_generation" and "mllm_generation" in blocked):
            return item.get("reason")
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if summary["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
