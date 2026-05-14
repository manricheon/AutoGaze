#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from external_model_asset_utils import (
    load_asset_manifest,
    model_entry,
    validate_local_assets,
)
from poc_infer_utils import load_config, nested_get, write_json
from poc_model_registry import build_mllm, build_vision_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal external-model smoke scaffold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-real-model-loading", action="store_true")
    parser.add_argument("--allow-dummy-weights", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=None)
    parser.add_argument("--max-batch-size-siglip", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--manifest", default="configs/poc_inference/model_asset_manifest.yaml")
    parser.add_argument("--zero-mask-stage", choices=["pixel", "patch_embedding", "post_encoder"], default=None)
    parser.add_argument("--zero-mask-value", choices=["zero", "mean"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir or nested_get(cfg, "output.output_dir", "outputs/external_model_smoke")).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    model_name = _smoke_model_name(cfg)
    manifest = load_asset_manifest(args.manifest)
    entry = model_entry(manifest, model_name)
    local_status = validate_local_assets(entry)
    adapter_status = _adapter_route_status(model_name, cfg)
    allow_real = bool(args.allow_real_model_loading)
    allow_dummy = bool(args.allow_dummy_weights or nested_get(cfg, "runtime.allow_dummy_weights", False))
    local_files_only = bool(args.local_files_only or nested_get(cfg, "runtime.local_files_only", False))
    dry_run = bool(args.dry_run or (not allow_real and not allow_dummy))

    if dry_run:
        summary = _dry_run_summary(
            args=args,
            cfg=cfg,
            model_name=model_name,
            local_status=local_status,
            adapter_status=adapter_status,
            local_files_only=local_files_only,
            allow_dummy=allow_dummy,
        )
    else:
        summary = _real_smoke(
            args=args,
            cfg=cfg,
            model_name=model_name,
            local_status=local_status,
            local_files_only=local_files_only,
            allow_dummy=allow_dummy,
        )
    _write_smoke_outputs(output_dir, model_name=model_name, cfg=cfg, summary=summary)
    print(f"Wrote smoke report: {output_dir / 'logs' / 'poc_summary.json'}")
    return 2 if summary.get("status") == "blocked" and not args.dry_run else 0


def _dry_run_summary(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model_name: str,
    local_status: dict[str, Any],
    adapter_status: dict[str, Any],
    local_files_only: bool,
    allow_dummy: bool,
) -> dict[str, Any]:
    assets_ready = local_status.get("download_status") == "local_exists"
    return {
        "status": "dry_run_ok" if assets_ready and adapter_status.get("ok") else "blocked",
        "run_type": "dry_run",
        "model": model_name,
        "config": args.config,
        "integration_mode": _integration_mode(cfg),
        "allow_real_model_loading": False,
        "allow_dummy_weights": allow_dummy,
        "local_files_only": local_files_only,
        "assets": local_status,
        "adapter_route": adapter_status,
        "video_path": args.video_path,
        "query_text_present": bool(args.query_text or nested_get(cfg, "mllm.query_text", None)),
        "zero_mask": _zero_mask_metadata(args=args, cfg=cfg),
        "reason": (
            "assets and adapter route are available; no weights loaded"
            if assets_ready and adapter_status.get("ok")
            else "local assets or adapter route are not ready; no fallback was used"
        ),
    }


def _real_smoke(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    model_name: str,
    local_status: dict[str, Any],
    local_files_only: bool,
    allow_dummy: bool,
) -> dict[str, Any]:
    if local_status.get("download_status") != "local_exists" and not allow_dummy:
        return {
            "status": "blocked",
            "run_type": "real_smoke",
            "model": model_name,
            "reason": "local assets are not verified; run verifier or prepare assets first",
            "assets": local_status,
            "local_files_only": local_files_only,
        }
    try:
        from infer_full import parse_args as parse_infer_args
        from infer_full import run as run_infer_full

        infer_argv = [
            "--config",
            args.config,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if allow_dummy and not args.allow_real_model_loading:
            infer_argv.append("--allow-dummy-weights")
        else:
            infer_argv.append("--allow-real-model-loading")
        if args.video_path:
            infer_argv.extend(["--video-path", args.video_path])
        query_text = args.query_text or nested_get(cfg, "generation.query_text", None)
        if query_text:
            infer_argv.extend(["--query-text", str(query_text)])
        if args.num_frames is not None:
            infer_argv.extend(["--num-frames", str(args.num_frames)])
        if args.max_batch_size_autogaze is not None:
            infer_argv.extend(["--max-batch-size-autogaze", str(args.max_batch_size_autogaze)])
        if args.max_batch_size_siglip is not None:
            infer_argv.extend(["--max-batch-size-siglip", str(args.max_batch_size_siglip)])
        if args.warmup_runs is not None:
            infer_argv.extend(["--warmup-runs", str(args.warmup_runs)])
        if args.no_progress:
            infer_argv.append("--no-progress")
        if args.output_dir:
            infer_argv.extend(["--output-dir", args.output_dir])
        if local_files_only:
            infer_argv.append("--local-files-only")
        return run_infer_full(parse_infer_args(infer_argv))
    except Exception as exc:
        return {
            "status": "blocked",
            "run_type": "real_smoke",
            "model": model_name,
            "reason": f"real smoke failed: {exc}",
            "local_files_only": local_files_only,
            "allow_dummy_weights": allow_dummy,
        }


def _write_smoke_outputs(output_dir: Path, *, model_name: str, cfg: dict[str, Any], summary: dict[str, Any]) -> None:
    write_json(output_dir / "logs" / "poc_summary.json", summary)
    summary_metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else None
    write_json(
        output_dir / "logs" / "metrics.json",
        summary_metrics or {
            "status": summary.get("status"),
            "model": model_name,
            "run_type": summary.get("run_type"),
            "integration_mode": summary.get("integration_mode") or _integration_mode(cfg),
            "real_model_loaded": bool(summary.get("adapter", {}).get("real_checkpoint_loaded", False)),
            "dummy_weights": _summary_uses_dummy_weights(summary),
            **_zero_mask_metrics(summary.get("zero_mask") if isinstance(summary.get("zero_mask"), dict) else _zero_mask_metadata(args=None, cfg=cfg)),
        },
    )
    if model_name == "vjepa2":
        metrics = summary_metrics or {}
        write_json(
            output_dir / "features" / "vjepa2_feature_summary.json",
            {
                "status": summary.get("status"),
                "feature_extraction": "not_run_in_dry_run" if summary.get("run_type") == "dry_run" else "see_logs",
                "input_tensor_shape": metrics.get("processed_video_shape"),
                "feature_shape": None if summary.get("run_type") == "dry_run" else metrics.get("feature_shape"),
                "pooled_feature_shape": None if summary.get("run_type") == "dry_run" else metrics.get("pooled_feature_shape"),
                "pooling_method": metrics.get("pooling_method"),
                "latency_ms": metrics.get("vision_encoder_latency_ms"),
                "memory": {
                    "peak_vram": metrics.get("peak_vram"),
                    "memory_unavailable": metrics.get("memory_unavailable"),
                },
                "mllm_projection": "blocked_without_verified_frozen_projector",
                "decoder_type": nested_get(cfg, "vision_encoder.decoder_type", nested_get(cfg, "decoder.type", None)),
                "autogaze_used": bool(nested_get(cfg, "autogaze.enabled", False)),
                "zero_mask_used": bool(nested_get(cfg, "zero_mask.enabled", False)),
            },
        )
    if bool(nested_get(cfg, "autogaze.enabled", False)):
        write_json(
            output_dir / "autogaze" / "frame_selection_metadata.json",
            {
                "status": "not_run_in_smoke_dry_run" if summary.get("run_type") == "dry_run" else "see_full_pipeline_outputs",
                "frame_selection_mode": nested_get(cfg, "frame_selection.mode", None),
                "num_frames": nested_get(cfg, "frame_selection.num_frames", None),
            },
        )
        write_json(
            output_dir / "autogaze" / "token_counts_summary.json",
            {
                "status": "not_run_in_smoke_dry_run" if summary.get("run_type") == "dry_run" else "see_full_pipeline_outputs",
                "direct_visual_token_injection": False,
            },
        )


def _adapter_route_status(model_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        if model_name == "vjepa2":
            adapter = build_vision_encoder("vjepa2", nested_get(cfg, "vision_encoder", {}))
            return {"ok": adapter.name == "vjepa2", "adapter": adapter.name, "type": "vision_encoder"}
        adapter = build_mllm(model_name, nested_get(cfg, "mllm", {}))
        return {"ok": adapter.name == model_name, "adapter": adapter.name, "type": "mllm"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "adapter": None}


def _smoke_model_name(cfg: dict[str, Any]) -> str:
    mllm_name = str(nested_get(cfg, "mllm.name", "") or "")
    vision_name = str(nested_get(cfg, "vision_encoder.name", "") or "")
    if vision_name == "vjepa2":
        return "vjepa2"
    if mllm_name:
        return mllm_name
    raise ValueError("smoke config must define mllm.name or vision_encoder.name")


def _integration_mode(cfg: dict[str, Any]) -> str:
    if str(nested_get(cfg, "vision_encoder.name", "") or "") == "vjepa2":
        return str(nested_get(cfg, "vision_encoder.integration_mode", "vjepa2_official_dense"))
    return str(
        nested_get(cfg, "mllm.generation_input_mode", None)
        or nested_get(cfg, "vision_encoder.integration_mode", None)
        or nested_get(cfg, "integration_mode", "unknown")
    )


def _zero_mask_metadata(*, args: argparse.Namespace | None, cfg: dict[str, Any]) -> dict[str, Any] | None:
    integration_mode = _integration_mode(cfg)
    enabled = integration_mode == "autogaze_zero_mask" or bool(nested_get(cfg, "zero_mask.enabled", False))
    if not enabled:
        return None
    stage = (args.zero_mask_stage if args is not None else None) or nested_get(cfg, "zero_mask.stage", "pixel")
    value = (args.zero_mask_value if args is not None else None) or nested_get(cfg, "zero_mask.value", "zero")
    return {
        "integration_mode": "autogaze_zero_mask",
        "zero_mask_stage": stage,
        "zero_mask_value": value,
        "zero_mask_encoder_compute_reduction": False,
        "zero_mask_expected_speedup": "none",
        "zero_mask_support_status": "dry_run_metadata_only",
    }


def _zero_mask_metrics(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "zero_mask_stage": metadata.get("zero_mask_stage"),
        "zero_mask_value": metadata.get("zero_mask_value"),
        "zero_mask_encoder_compute_reduction": False,
        "zero_mask_expected_speedup": "none",
    }


def _summary_uses_dummy_weights(summary: dict[str, Any]) -> bool:
    if bool(summary.get("allow_dummy_weights")):
        return True
    statuses = summary.get("adapter_statuses")
    if isinstance(statuses, dict):
        for status in statuses.values():
            if isinstance(status, dict) and status.get("status") == "dummy":
                return True
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        return bool(metrics.get("dummy_weights_enabled", False))
    return False


if __name__ == "__main__":
    sys.exit(main())
