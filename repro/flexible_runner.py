from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from repro.common import write_json
from repro.failure_logging import classify_exception, failure_generation_payload
from repro.plugin_api import ExperimentSpec, MetricStatus
from repro.plugins.autogaze_sparse_selector import runtime_config_from_args, run_direct_autogaze_selector
from repro.plugins.mllm_adapters import MllmRunRequest, resolve_mllm_adapter
from repro.plugins.pixelprune_adapter import PixelPruneConfig, apply_pixelprune_if_available, pixelprune_model_key


MODEL_FAMILY_CHOICES = (
    "nvila-hd-video-autogaze",
    "nvila-video-baseline",
    "nvila-video-plugin",
    "longvila",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3-vl-moe",
    "llava-onevision",
    "internvl3",
)
TOKEN_SELECTOR_ADAPTER_CHOICES = ("none", "keep-all", "autogaze", "external-mask")
VISION_ENCODER_ADAPTER_CHOICES = (
    "auto",
    "nvila-hd-siglip",
    "nvila-video-vision",
    "longvila-siglip",
    "qwen2-vl-vision",
    "qwen2.5-vl-vision",
    "qwen3-vl-vision",
    "qwen3-vl-moe-vision",
    "llava-onevision-siglip",
    "internvl-dynamic-vision",
)
MLLM_ADAPTER_CHOICES = (
    "auto",
    "nvila-hd",
    "nvila-video",
    "longvila",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen3-vl",
    "qwen3-vl-moe",
    "llava-onevision",
    "internvl3",
)
AUTOGAZE_INTEGRATION_LEVEL_CHOICES = (
    "none",
    "native_processor",
    "pre_encoder_sparse",
    "post_encoder_token_prune",
    "planned_plugin",
)
PRE_ENCODER_PRUNE_ADAPTER_CHOICES = ("none", "autogaze-sparse", "pixelprune")
QWEN_VIT_MODE_CHOICES = (
    "qwen_full_vit",
    "qwen_chunked_vit",
    "qwen_chunked_vit_autogaze_sparse",
)
QWEN_THUMBNAIL_MODE_CHOICES = ("none", "append-video")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and run extensible AutoGaze plugin pipelines")
    parser.add_argument("--mode", choices=["inspect", "single"], default="inspect")
    parser.add_argument("--model-family", choices=MODEL_FAMILY_CHOICES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--token-selector-adapter", choices=TOKEN_SELECTOR_ADAPTER_CHOICES, required=True)
    parser.add_argument("--token-selector-path")
    parser.add_argument("--vision-encoder-adapter", choices=VISION_ENCODER_ADAPTER_CHOICES, required=True)
    parser.add_argument("--vision-encoder-path")
    parser.add_argument("--mllm-adapter", choices=MLLM_ADAPTER_CHOICES, required=True)
    parser.add_argument("--mllm-path")
    parser.add_argument("--autogaze-integration-level", choices=AUTOGAZE_INTEGRATION_LEVEL_CHOICES, required=True)
    parser.add_argument("--pre-encoder-prune-adapter", choices=PRE_ENCODER_PRUNE_ADAPTER_CHOICES, default="none")
    parser.add_argument("--gazing-ratio", type=float)
    parser.add_argument("--pixelprune-threshold", type=float, default=0.0)
    parser.add_argument("--pixelprune-verbose", action="store_true")
    parser.add_argument("--enable-qwen-prune-generate", action="store_true")
    parser.add_argument("--sparse-selection-plan-json")
    parser.add_argument("--run-autogaze-selector", action="store_true")
    parser.add_argument("--autogaze-selector-output-json")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model")
    parser.add_argument("--autogaze-device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--autogaze-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--autogaze-target-scales", default="32+64+112+224")
    parser.add_argument("--autogaze-target-patch-size", type=int, default=16)
    parser.add_argument("--autogaze-encoder-patch-size", type=int)
    parser.add_argument("--autogaze-tile-size", type=int)
    parser.add_argument("--autogaze-chunk-frames", type=int, default=16)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--task-loss-requirement", type=float)
    parser.add_argument("--autogaze-generate-only", action="store_true")
    parser.add_argument("--qwen-video-nframes", type=int)
    parser.add_argument("--qwen-video-fps", type=float)
    parser.add_argument("--qwen-video-max-pixels", type=int)
    parser.add_argument("--qwen-video-min-pixels", type=int)
    parser.add_argument("--qwen-vit-mode", choices=QWEN_VIT_MODE_CHOICES, default="qwen_full_vit")
    parser.add_argument("--qwen-vit-chunk-frames", type=int, default=16)
    parser.add_argument("--qwen-vit-max-spatial-chunks", type=int)
    parser.add_argument("--qwen-thumbnail-mode", choices=QWEN_THUMBNAIL_MODE_CHOICES, default="none")
    parser.add_argument("--video")
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="Describe the video.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--external-mllm-command", default="vila-infer")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=0)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)
    parser.add_argument("--output-json", default="outputs/autogaze_repro/flexible_runner_inspect.json")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if getattr(args, "mllm_path", None) is None:
        args.mllm_path = args.model_path
    if (
        getattr(args, "qwen_video_nframes", None) is None
        and getattr(args, "video", None)
        and str(getattr(args, "model_family", "")).startswith("qwen")
    ):
        args.qwen_video_nframes = int(getattr(args, "num_video_frames", 0) or 0)
    if (
        getattr(args, "qwen_vit_max_spatial_chunks", None) is None
        and str(getattr(args, "model_family", "")).startswith("qwen")
    ):
        args.qwen_vit_max_spatial_chunks = max(1, int(getattr(args, "max_tiles_video", 1) or 1))
    return args


def build_experiment_spec(args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec.from_args(args)


def build_inspect_payload(args: argparse.Namespace) -> dict[str, Any]:
    spec = build_experiment_spec(args)
    return {
        "runner": "flexible_runner",
        "mode": args.mode,
        "implementation_status": "inspect_only",
        "output_json": str(Path(args.output_json)),
        "paper_baseline_semantics": (
            "paper_baseline_candidate" if spec.is_paper_baseline_candidate else "not_a_paper_baseline"
        ),
        "experiment_spec": spec.to_dict(),
        "model_capabilities": build_model_capabilities(spec),
        "mllm_status_matrix": build_mllm_status_matrix(),
        "four_step_execution_plan": build_four_step_execution_plan(spec),
        "adapter_plan": build_adapter_plan(spec),
        "note": (
            "This runner is the extension surface for NVILA-Video, LongVILA, Qwen-style models, "
            "and future token selector / vision encoder / MLLM plugins. It does not mutate "
            "the stable NVILA-HD reproduction runner."
        ),
    }


def build_model_capabilities(spec: ExperimentSpec) -> dict[str, Any]:
    if spec.model_family in {"qwen3-vl", "qwen3-vl-moe"}:
        return {
            "family_group": "qwen3_vl",
            "source_notes": [
                "Transformers Qwen3-VL forward accepts pixel_values_videos and video_grid_thw.",
                "Qwen3-VL exposes get_video_features(pixel_values_videos, video_grid_thw).",
                "QwenLM recommends flash_attention_2 for multi-image and video memory/speed.",
            ],
            "video_forward_fields": ["pixel_values_videos", "video_grid_thw", "mm_token_type_ids"],
            "post_encoder_prune_hook": "after get_video_features output and before visual token insertion into MLLM context",
            "pre_encoder_sparse_hook": "before get_video_features; requires preserving video_grid_thw and positional semantics",
        }
    if spec.model_family in {"qwen2-vl", "qwen2.5-vl"}:
        return {
            "family_group": "qwen_grid_vl",
            "video_forward_fields": ["pixel_values_videos", "video_grid_thw"],
            "post_encoder_prune_hook": "after vision feature extraction and before MLLM context packing",
            "pre_encoder_sparse_hook": "requires model-specific grid and position probe",
        }
    if spec.model_family == "llava-onevision":
        return {
            "family_group": "llava_onevision",
            "video_forward_fields": ["videos", "input_ids", "attention_mask"],
            "video_token_policy": "pooled_196_tokens_per_frame",
            "post_encoder_prune_hook": "after SigLIP/video pooling and before Qwen2 language backbone context packing",
            "pre_encoder_sparse_hook": "before SigLIP/anyres video pooling; hard because video tokens are pooled to 196 per frame",
        }
    if spec.model_family == "internvl3":
        return {
            "family_group": "internvl",
            "video_forward_fields": ["pixel_values", "num_patches_list"],
            "post_encoder_prune_hook": "after dynamic visual feature extraction and before language model packing",
            "pre_encoder_sparse_hook": "requires dynamic tiling and num_patches_list probe",
        }
    if spec.model_family == "longvila":
        return {
            "family_group": "longvila",
            "video_forward_fields": [],
            "post_encoder_prune_hook": "after visual feature extraction and before MLLM packing",
            "pre_encoder_sparse_hook": "requires LongVILA-specific processor and vision tower probe",
        }
    if spec.model_family == "nvila-video-plugin":
        return {
            "family_group": "nvila_video_plugin",
            "video_forward_fields": [],
            "post_encoder_prune_hook": "after NVILA-Video vision output and before MLLM visual token packing",
            "pre_encoder_sparse_hook": "requires NVILA-Video patch/position alignment probe",
        }
    return {
        "family_group": spec.model_family,
        "video_forward_fields": [],
        "post_encoder_prune_hook": "native runner path",
        "pre_encoder_sparse_hook": "native runner path",
    }


def build_mllm_status_matrix() -> dict[str, dict[str, str]]:
    return {
        "nvila-video-plugin": {
            "native_off_status": "external_cli_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "probe_required",
        },
        "longvila": {
            "native_off_status": "external_cli_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "probe_required",
        },
        "qwen2-vl": {
            "native_off_status": "single_runtime_adapter_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "probe_required",
        },
        "qwen2.5-vl": {
            "native_off_status": "single_runtime_adapter_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "probe_required",
        },
        "qwen3-vl": {
            "native_off_status": "single_dry_run_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "pixelprune_reference_available",
        },
        "qwen3-vl-moe": {
            "native_off_status": "single_dry_run_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "pixelprune_reference_available",
        },
        "llava-onevision": {
            "native_off_status": "single_runtime_adapter_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "hard",
        },
        "internvl3": {
            "native_off_status": "external_cli_ready",
            "recommended_first_on_mode": "post_encoder_token_prune",
            "pre_encoder_sparse_status": "probe_required",
        },
    }


def build_four_step_execution_plan(spec: ExperimentSpec) -> dict[str, dict[str, Any]]:
    capabilities = build_model_capabilities(spec)
    post_status = "candidate_next" if spec.integration_level == "post_encoder_token_prune" else "planned"
    if spec.pre_encoder_prune_adapter == "pixelprune" and spec.model_family in {"qwen3-vl", "qwen3-vl-moe"}:
        pre_status = "pixelprune_reference_available"
    elif spec.integration_level == "pre_encoder_sparse":
        pre_status = "candidate_next"
    else:
        pre_status = "requires_model_specific_probe"
    return {
        "native_off_baseline": {
            "goal": "run the original vision encoder + MLLM without AutoGaze on the same sampled frames",
            "integration_level": "none",
            "status": "ready_for_native_off_adapter" if spec.token_selector_kind == "keep-all" else "comparison_required",
            "measures": ["baseline_latency", "baseline_memory", "baseline_visual_tokens", "baseline_accuracy"],
        },
        "autogaze_standalone_selector": {
            "goal": "run AutoGaze as an external token selector on the same sampled frames",
            "integration_level": "planned_plugin",
            "status": "ready_for_selector_adapter" if spec.token_selector_kind == "autogaze" else "planned",
            "measures": ["autogaze_latency", "selected_patch_tokens", "patch_reduction_ratio"],
        },
        "post_encoder_token_prune": {
            "goal": "keep the original vision encoder, then prune visual tokens before MLLM prefill",
            "integration_level": "post_encoder_token_prune",
            "status": post_status,
            "hook": capabilities["post_encoder_prune_hook"],
            "expected_gain": "MLLM context, KV cache, prefill, and TTFT reduction; no vision encoder latency reduction",
        },
        "pre_encoder_sparse": {
            "goal": "feed only AutoGaze-selected patches/tokens into the vision encoder",
            "integration_level": "pre_encoder_sparse",
            "status": pre_status,
            "hook": capabilities["pre_encoder_sparse_hook"],
            "expected_gain": "vision encoder compute plus MLLM context reduction when model-specific position/grid semantics are preserved",
        },
    }


def build_adapter_plan(spec: ExperimentSpec) -> dict[str, Any]:
    mllm_adapter = resolve_mllm_adapter(spec.mllm_kind)
    return {
        "pre_encoder_prune": {
            "adapter": spec.pre_encoder_prune_adapter,
            "status": _pre_encoder_prune_status(spec).to_dict(),
        },
        "token_selector": {
            "adapter": spec.token_selector_kind,
            "path": spec.token_selector_path,
            "status": _token_selector_status(spec).to_dict(),
        },
        "vision_encoder": {
            "adapter": spec.vision_encoder_kind,
            "path": spec.vision_encoder_path,
            "status": MetricStatus(value="planned", reason="adapter contract only").to_dict(),
        },
        "mllm": {
            "adapter": spec.mllm_kind,
            "path": spec.mllm_path,
            "status": MetricStatus(value=mllm_adapter.runtime_status, reason=_mllm_status_reason(mllm_adapter)).to_dict(),
        },
    }


def run_inspect(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_inspect_payload(args)
    write_json(args.output_json, payload)
    return payload


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_inspect_payload(args)
    payload["mode"] = "single"
    request = build_mllm_run_request(args)
    adapter = resolve_mllm_adapter(args.mllm_adapter)
    payload["mllm_runtime"] = adapter.describe_runtime(request)
    payload["measurement_plan"] = payload["mllm_runtime"]["metric_schema"]
    payload["post_encoder_prune_runtime"] = build_post_encoder_prune_runtime_plan(
        payload["experiment_spec"]["integration_level"],
        payload["four_step_execution_plan"]["post_encoder_token_prune"]["hook"],
        enabled=bool(getattr(args, "enable_qwen_prune_generate", False)),
    )
    if getattr(args, "dry_run", False):
        payload["implementation_status"] = "dry_run"
        write_json(args.output_json, payload)
        return payload

    stage = "pre_encoder_prune"
    pixelprune_status: dict[str, Any] = {"status": "not_started"}
    direct_selector_status: dict[str, Any] = {"status": "not_started"}
    try:
        pixelprune_status = maybe_apply_pre_encoder_prune(args, payload["experiment_spec"]["model_family"])
        if _pre_encoder_prune_failed(args, pixelprune_status):
            payload["implementation_status"] = "failed_missing_dependency"
            payload["pre_encoder_prune_runtime"] = pixelprune_status
            payload["generation"] = _pre_encoder_prune_failure_result(args, pixelprune_status)
            payload["measurement_plan"] = payload["generation"]["metrics"]
            write_json(args.output_json, payload)
            return payload

        stage = "autogaze"
        direct_selector_status = maybe_run_direct_autogaze_selector(args)
        if direct_selector_status.get("status") == "failed":
            payload["implementation_status"] = "failed_autogaze_selector"
            payload["pre_encoder_prune_runtime"] = pixelprune_status
            payload["direct_autogaze_selector"] = direct_selector_status
            write_json(args.output_json, payload)
            return payload

        stage = "mllm_generate"
        request = build_mllm_run_request(args)
        payload["mllm_runtime"] = adapter.describe_runtime(request)
        output = adapter.run(request).to_dict()
        payload["implementation_status"] = output.get("status", "executed")
        payload["pre_encoder_prune_runtime"] = pixelprune_status
        payload["direct_autogaze_selector"] = direct_selector_status
        payload["generation"] = output
        payload["measurement_plan"] = output.get("metrics", payload["measurement_plan"])
        write_json(args.output_json, payload)
        return payload
    except Exception as exc:
        failure = classify_exception(exc, stage=stage)
        payload["implementation_status"] = failure["kind"]
        payload["pre_encoder_prune_runtime"] = pixelprune_status
        payload["direct_autogaze_selector"] = direct_selector_status
        payload["failure"] = failure
        payload["generation"] = failure_generation_payload(args, failure)
        payload["measurement_plan"] = payload["generation"]["metrics"]
        write_json(args.output_json, payload)
        if failure["kind"] != "oom":
            raise
        return payload


def maybe_run_direct_autogaze_selector(args: argparse.Namespace) -> dict[str, Any]:
    if not getattr(args, "run_autogaze_selector", False):
        return {"status": "not_requested"}
    if not getattr(args, "video", None):
        return {"status": "failed", "reason": "--run-autogaze-selector requires --video"}
    if getattr(args, "qwen_video_nframes", None) is None:
        args.qwen_video_nframes = int(getattr(args, "num_video_frames", 0) or 0)
    try:
        status = run_direct_autogaze_selector(runtime_config_from_args(args))
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "exception_type": type(exc).__name__}
    args.sparse_selection_plan_json = status.get("sparse_selection_plan_json")
    return status


def maybe_apply_pre_encoder_prune(args: argparse.Namespace, model_family: str) -> dict[str, Any]:
    if getattr(args, "pre_encoder_prune_adapter", "none") != "pixelprune":
        return {"adapter": getattr(args, "pre_encoder_prune_adapter", "none"), "applied": False, "reason": "not requested"}
    config = PixelPruneConfig(
        model_key=pixelprune_model_key(model_family),
        threshold=float(getattr(args, "pixelprune_threshold", 0.0)),
        verbose=bool(getattr(args, "pixelprune_verbose", False)),
    )
    result = apply_pixelprune_if_available(config)
    return {"adapter": "pixelprune", **result}


def build_post_encoder_prune_runtime_plan(integration_level: str, hook: str, *, enabled: bool = False) -> dict[str, Any]:
    if integration_level != "post_encoder_token_prune":
        return {"status": "not_requested"}
    if enabled:
        return {
            "status": "experimental_prune_generate_enabled",
            "hook": hook,
            "runtime_behavior": (
                "Qwen-family adapters attempt get_video_features, reduce visual placeholders, "
                "and call generate with pruned inputs_embeds."
            ),
        }
    return {
        "status": "shape_probe_required",
        "hook": hook,
        "runtime_behavior": "no pruning applied until visual feature/token shape probe is implemented",
    }


def build_mllm_run_request(args: argparse.Namespace) -> MllmRunRequest:
    return MllmRunRequest(
        model_family=args.model_family,
        model_path=args.model_path,
        mllm_adapter=args.mllm_adapter,
        prompt=args.prompt,
        video=args.video,
        image=args.image,
        device_map=args.device_map,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=not args.no_trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        token_selector_kind=args.token_selector_adapter,
        integration_level=args.autogaze_integration_level,
        pre_encoder_prune_adapter=args.pre_encoder_prune_adapter,
        gazing_ratio=args.gazing_ratio,
        num_video_frames=args.num_video_frames,
        num_video_frames_thumbnail=args.num_video_frames_thumbnail,
        max_tiles_video=args.max_tiles_video,
        external_mllm_command=args.external_mllm_command,
        enable_qwen_prune_generate=args.enable_qwen_prune_generate,
        sparse_selection_plan_path=args.sparse_selection_plan_json,
        qwen_video_nframes=args.qwen_video_nframes,
        qwen_video_fps=args.qwen_video_fps,
        qwen_video_max_pixels=args.qwen_video_max_pixels,
        qwen_video_min_pixels=args.qwen_video_min_pixels,
        qwen_vit_mode=args.qwen_vit_mode,
        qwen_vit_chunk_frames=args.qwen_vit_chunk_frames,
        qwen_vit_max_spatial_chunks=args.qwen_vit_max_spatial_chunks,
        qwen_thumbnail_mode=args.qwen_thumbnail_mode,
        video_resize_shortest_edge=args.video_resize_shortest_edge,
        video_resize_longest_edge=args.video_resize_longest_edge,
        video_resize_width=args.video_resize_width,
        video_resize_height=args.video_resize_height,
    )


def _pre_encoder_prune_failed(args: argparse.Namespace, status: dict[str, Any]) -> bool:
    if getattr(args, "pre_encoder_prune_adapter", "none") != "pixelprune":
        return False
    return status.get("applied") is False


def _pre_encoder_prune_failure_result(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    reason = str(status.get("reason") or "pre-encoder prune adapter was not applied")
    return {
        "text": None,
        "prompt": args.prompt,
        "video": args.video,
        "image": args.image,
        "adapter": args.mllm_adapter,
        "status": "failed_missing_dependency",
        "metrics": {
            "latency_ms": {
                "model_load": None,
                "processor_load": None,
                "input_build": None,
                "generate": None,
                "total": None,
            },
            "tokens": {
                "prompt_tokens_estimated": len(args.prompt.split()),
                "input_ids_tokens": None,
                "visual_tokens_before_prune": None,
                "visual_tokens_after_prune": None,
                "llm_context_tokens": None,
            },
            "memory_bytes": {
                "peak_cuda_allocated": None,
                "peak_cuda_reserved": None,
            },
            "pre_encoder_prune": status,
            "metric_status": {
                "value": "failed_missing_dependency",
                "reason": reason,
            },
        },
    }


def _token_selector_status(spec: ExperimentSpec) -> MetricStatus:
    if spec.token_selector_kind == "none":
        return MetricStatus(value="not_applicable" if spec.is_paper_baseline_candidate else "disabled")
    if spec.token_selector_kind == "keep-all":
        return MetricStatus(value="native_off", reason="keeps all visual tokens for on/off comparison")
    if spec.token_selector_kind == "autogaze":
        return MetricStatus(value="planned", reason=f"integration_level={spec.integration_level}")
    return MetricStatus(value="planned", reason="external mask adapter contract only")


def _pre_encoder_prune_status(spec: ExperimentSpec) -> MetricStatus:
    if spec.pre_encoder_prune_adapter == "pixelprune":
        if spec.model_family in {"qwen3-vl", "qwen3-vl-moe"}:
            return MetricStatus(value="candidate", reason="PixelPrune upstream supports Qwen3-VL pre-ViT pruning.")
        return MetricStatus(value="unsupported", reason="PixelPrune support is only declared for Qwen3-style families here.")
    if spec.pre_encoder_prune_adapter == "autogaze-sparse":
        return MetricStatus(value="planned", reason="Requires model-specific position/grid sparse-input probe.")
    return MetricStatus(value="disabled", reason=None)


def _mllm_status_reason(mllm_adapter: Any) -> str:
    if mllm_adapter.runtime_status == "implemented":
        return "single runtime adapter"
    if mllm_adapter.runtime_status == "external_cli_ready":
        return "official VILA CLI adapter"
    return "adapter contract only"


def main() -> None:
    args = parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_inspect(args)


if __name__ == "__main__":
    main()
