from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore[assignment]


OOM_MARKERS = (
    "out of memory",
    "cuda oom",
    "mps backend out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "memory allocation",
)

DEPENDENCY_MARKERS = (
    "modulenotfounderror",
    "importerror",
    "no module named",
    "pip install",
    "missing dependency",
    "is required for",
)


def classify_exception(exc: BaseException, *, stage: str = "unknown") -> dict[str, Any]:
    message = str(exc)
    lower = f"{type(exc).__name__} {message}".lower()
    if any(marker in lower for marker in OOM_MARKERS):
        kind = "oom"
    elif any(marker in lower for marker in DEPENDENCY_MARKERS):
        kind = "failed_missing_dependency"
    else:
        kind = "exception"
    return {
        "kind": kind,
        "stage": infer_failure_stage(lower, fallback=stage),
        "exception_type": type(exc).__name__,
        "message": message,
        "traceback_tail": traceback.format_exception_only(type(exc), exc)[-1].strip(),
        "device_memory": device_memory_snapshot(),
    }


def infer_failure_stage(text: str, *, fallback: str = "unknown") -> str:
    lower = text.lower()
    if "qwen_vl_utils" in lower or "qwen video" in lower:
        return "qwen_video_input_build"
    if "qwen_vit" in lower or "qwen vit" in lower:
        return "qwen_vit_prepare"
    if "autogaze" in lower or "gaze" in lower:
        return "autogaze"
    if "siglip" in lower or "vision_encoder" in lower or "vision encoder" in lower:
        return "vision_encoder"
    if "sdpa" in lower or "attention" in lower or "prefill" in lower or "generate" in lower or "llm" in lower:
        return "llm_prefill_or_generate"
    if "decode" in lower or "video" in lower:
        return "video_decode_or_preprocess"
    return fallback


def device_memory_snapshot() -> dict[str, Any]:
    if torch is None:
        return {"cuda_available": False}
    snapshot: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    return snapshot


def failure_generation_payload(args: Any, failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": None,
        "prompt": getattr(args, "prompt", None),
        "video": getattr(args, "video", None),
        "image": getattr(args, "image", None),
        "adapter": getattr(args, "mllm_adapter", None),
        "status": failure["kind"],
        "failure": failure,
        "metrics": {
            "latency_ms": {
                "model_load": None,
                "processor_load": None,
                "input_build": None,
                "generate": None,
                "total": None,
            },
            "tokens": {
                "prompt_tokens_estimated": len(str(getattr(args, "prompt", "")).split()),
                "input_ids_tokens": None,
                "visual_tokens_before_prune": None,
                "visual_tokens_after_prune": None,
                "llm_context_tokens": None,
            },
            "memory_bytes": failure.get("device_memory", {}),
            "failure": failure,
            "metric_status": {
                "value": failure["kind"],
                "reason": f"{failure['exception_type']}: {failure['message']}",
            },
        },
    }


def minimal_runner_failure_payload(args: Any, failure: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": failure["kind"],
        "failure": failure,
        "model_path": getattr(args, "model_path", None),
        "video": getattr(args, "video", None),
        "prompt": getattr(args, "prompt", None),
        "gazing_mode": getattr(args, "gazing_mode", None),
        "output_json": str(Path(getattr(args, "output_json", ""))) if getattr(args, "output_json", None) else None,
    }
