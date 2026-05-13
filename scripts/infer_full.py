#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from poc_infer_utils import (
    build_metrics,
    cli_or_config,
    load_config,
    nested_get,
    normalize_device,
    prepare_video,
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
    parser.add_argument("--max-new-tokens", type=int, default=None)

    parser.add_argument("--frame-selection-mode", choices=["sample", "chunk", "interval", "all"], default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument("--scaling-mode", choices=["resize", "fit_short_side", "fit_long_side", "chop", "quickstart", "none"], default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--chop-size", type=int, default=None)
    parser.add_argument("--chop-overlap", type=int, default=None)
    parser.add_argument("--max-chops", type=int, default=None)
    parser.add_argument("--chop-merge-mode", choices=["none", "metadata_only", "overlay_union"], default=None)

    parser.add_argument("--gaze-ratio", type=float, default=None)
    parser.add_argument("--task-loss-requirement", type=float, default=None)
    parser.add_argument("--strict-autogaze-params", action="store_true")
    parser.add_argument("--allow-real-model-loading", action="store_true")

    parser.add_argument("--overlay-style", choices=["mask", "box", "both"], default=None)
    parser.add_argument("--overlay-alpha", type=float, default=None)
    parser.add_argument("--multi-scale-overlay", action="store_true", default=None)
    parser.add_argument("--show-patch-index", action="store_true")
    parser.add_argument("--show-scale-label", action="store_true")
    parser.add_argument("--metadata-placement", choices=["outside", "inside", "none"], default=None)
    parser.add_argument("--info-panel-position", choices=["bottom", "right"], default=None)
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
    start = time.perf_counter()
    cfg = _with_model_overrides(load_config(args.config), args)
    device = normalize_device(str(cli_or_config(args.device, cfg, "runtime.device", "cpu")))
    dtype = str(cli_or_config(args.dtype, cfg, "runtime.dtype", "float32"))
    output_dir = Path(str(cli_or_config(args.output_dir, cfg, "output.output_dir", "outputs/poc_full"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    prepared = prepare_video(
        cfg,
        video_path=args.video_path,
        frame_selection_mode=str(cli_or_config(args.frame_selection_mode, cfg, "frame_selection.mode", "sample")),
        num_frames=int(cli_or_config(args.num_frames, cfg, "frame_selection.num_frames", 16)),
        frame_interval=int(cli_or_config(args.frame_interval, cfg, "frame_selection.frame_interval", 1)),
        max_windows=cli_or_config(args.max_windows, cfg, "frame_selection.max_windows", None),
        scaling_mode=str(cli_or_config(args.scaling_mode, cfg, "scaling.mode", "resize")),
        resolution=int(cli_or_config(args.resolution, cfg, "scaling.resolution", 224)),
        chop_size=int(cli_or_config(args.chop_size, cfg, "scaling.chop_size", 224)),
        chop_overlap=int(cli_or_config(args.chop_overlap, cfg, "scaling.chop_overlap", 0)),
        max_chops=cli_or_config(args.max_chops, cfg, "scaling.max_chops", None),
        chop_merge_mode=str(cli_or_config(args.chop_merge_mode, cfg, "scaling.chop_merge_mode", "metadata_only")),
    )
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
        vision_status = vision.load(allow_real_model_loading=allow_real, device=device, dtype=dtype)
        if allow_real and vision_status.status != "real":
            skipped.append({"stage": "vision_encoder", "reason": vision_status.reason or f"{vision.name} real loading unavailable"})
            blocked_stages.append("vision_encoder")
        else:
            autogaze_payload = None if not gaze.autogaze_enabled or gaze.status == "blocked" else gaze.gazing_info_for_vit
            try:
                vision_output = vision.forward(prepared.processed_video, autogaze=autogaze_payload)
                visual_tokens = vision_output.get("visual_tokens")
                if vision_status.status != "real":
                    skipped.append({"stage": "vision_encoder", "reason": vision_status.reason or f"{vision.name} used stub output"})
            except Exception as exc:
                skipped.append({"stage": "vision_encoder", "reason": f"vision encoder failed: {exc}"})
                if allow_real and vision_required:
                    blocked_stages.append("vision_encoder")

    mllm = build_mllm(mllm_name, nested_get(cfg, "mllm", {}))
    mllm_status = mllm.load(allow_real_model_loading=allow_real, device=device, dtype=dtype)
    max_new_tokens = int(cli_or_config(args.max_new_tokens, cfg, "generation.max_new_tokens", 32))
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
        generation = mllm.generate(
            query_text=args.query_text,
            video=prepared.processed_video,
            visual_tokens=visual_tokens if mllm.supports_direct_visual_tokens() else None,
            max_new_tokens=max_new_tokens,
            video_path=args.video_path,
        )
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
            save_overlay_video=bool(args.save_overlay_video or nested_get(cfg, "visualization.save_overlay_video", False)),
            save_side_by_side_video=bool(args.save_side_by_side_video or nested_get(cfg, "visualization.save_side_by_side_video", False)),
            save_scale_panel_video=bool(args.save_scale_panel_video or nested_get(cfg, "visualization.save_scale_panel_video", False)),
            video_fps=float(cli_or_config(args.video_fps, cfg, "visualization.video_fps", 4.0)),
            video_export_mode=str(cli_or_config(args.video_export_mode, cfg, "visualization.video_export_mode", "sampled_only")),
            query_text=args.query_text,
            generation_status=generation_status,
        )
    )

    total_latency = (time.perf_counter() - start) * 1000
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
        total_latency_ms=total_latency,
    )
    metrics["adapter_statuses"] = {
        "vision_encoder": vision_status.to_dict(),
        "mllm": mllm_status.to_dict(),
    }
    metrics["vision_encoder_required_for_full_pipeline"] = vision_required
    metrics["generation_input_mode"] = "direct_visual_tokens" if direct_visual_token_mode else "official_processor"
    metrics["gazing_info_passed_to_vision_encoder"] = bool(gaze.gazing_info_for_vit is not None and vision_required)
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


def _sync_mllm_with_poc_autogaze_controls(cfg: dict[str, Any], gaze: Any) -> None:
    mllm = cfg.setdefault("mllm", {})
    if not isinstance(mllm, dict) or not bool(mllm.get("sync_autogaze_controls_from_config", False)):
        return
    mllm["poc_autogaze_enabled"] = bool(nested_get(cfg, "autogaze.enabled", False))
    mllm["poc_autogaze_status"] = getattr(gaze, "status", None)
    mllm["poc_gaze_ratio"] = nested_get(cfg, "autogaze.gaze_ratio")
    mllm["poc_task_loss_requirement"] = nested_get(cfg, "autogaze.task_loss_requirement")


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
