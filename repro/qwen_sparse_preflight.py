from __future__ import annotations

import argparse
import json
import math
from typing import Any

GIB = 1024**3


def estimate_qwen_sparse_tokens(
    *,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    autogaze_reduction_ratio: float = 10.0,
    chunk_frames: int = 16,
    max_spatial_chunks: int | None = None,
) -> dict[str, Any]:
    frames = max(1, int(num_frames))
    patch = max(1, int(patch_size))
    merge = max(1, int(spatial_merge_size))
    reduction = max(1.0, float(autogaze_reduction_ratio))
    grid_h = math.ceil(int(height) / patch)
    grid_w = math.ceil(int(width) / patch)
    merged_h = math.ceil(grid_h / merge)
    merged_w = math.ceil(grid_w / merge)
    raw_patch_tokens = frames * grid_h * grid_w
    visual_tokens_before = frames * merged_h * merged_w
    selected_patch_tokens = max(1, math.ceil(raw_patch_tokens / reduction))
    visual_tokens_after = max(1, math.ceil(visual_tokens_before / reduction))
    chunk_count = math.ceil(frames / max(1, int(chunk_frames)))
    spatial_chunks = max(1, int(max_spatial_chunks or 1))
    return {
        "num_frames": frames,
        "input_resolution": [int(height), int(width)],
        "patch_size": patch,
        "spatial_merge_size": merge,
        "video_grid_thw": [frames, grid_h, grid_w],
        "merged_grid_thw": [frames, merged_h, merged_w],
        "raw_patch_tokens_before_vit": raw_patch_tokens,
        "visual_tokens_before_prune": visual_tokens_before,
        "selected_patch_tokens_after_autogaze": selected_patch_tokens,
        "visual_tokens_after_prune": visual_tokens_after,
        "vit_token_reduction_ratio": raw_patch_tokens / selected_patch_tokens,
        "llm_visual_token_reduction_ratio": visual_tokens_before / visual_tokens_after,
        "chunk_frames": max(1, int(chunk_frames)),
        "chunk_count": chunk_count,
        "max_spatial_chunks": spatial_chunks,
        "estimated_vit_work_units_full": raw_patch_tokens,
        "estimated_vit_work_units_sparse": selected_patch_tokens,
    }


def estimate_qwen_plugin_preflight(
    *,
    model_family: str,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    autogaze_reduction_ratio: float = 10.0,
    chunk_frames: int = 16,
    max_spatial_chunks: int | None = None,
    prompt_tokens: int = 512,
    max_new_tokens: int = 128,
    context_limit: int = 32768,
    h100_budget_gib: float = 70.0,
    resident_model_gib: float = 18.0,
    bytes_per_token_kv: int = 2 * 32 * 4096 * 2,
    vision_activation_bytes_per_patch: int = 1280 * 2 * 6,
    runtime_safety_gib: float = 4.0,
) -> dict[str, Any]:
    token_estimate = estimate_qwen_sparse_tokens(
        num_frames=num_frames,
        height=height,
        width=width,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        autogaze_reduction_ratio=autogaze_reduction_ratio,
        chunk_frames=chunk_frames,
        max_spatial_chunks=max_spatial_chunks,
    )
    llm_context_tokens = (
        int(prompt_tokens)
        + int(max_new_tokens)
        + int(token_estimate["visual_tokens_after_prune"])
    )
    kv_cache_gib = llm_context_tokens * int(bytes_per_token_kv) / GIB
    sparse_vit_gib = int(token_estimate["selected_patch_tokens_after_autogaze"]) * int(vision_activation_bytes_per_patch) / GIB
    estimated_peak_gib = float(resident_model_gib) + kv_cache_gib + sparse_vit_gib + float(runtime_safety_gib)
    return {
        "model_family": model_family,
        "token_estimate": token_estimate,
        "prompt_tokens": int(prompt_tokens),
        "max_new_tokens": int(max_new_tokens),
        "llm_context_tokens_estimated": llm_context_tokens,
        "context_limit": int(context_limit),
        "memory_estimate_gib": {
            "resident_model": float(resident_model_gib),
            "kv_cache": kv_cache_gib,
            "sparse_vit_activation_proxy": sparse_vit_gib,
            "runtime_safety": float(runtime_safety_gib),
            "estimated_peak": estimated_peak_gib,
            "budget": float(h100_budget_gib),
        },
        "risk": {
            "context": "red" if llm_context_tokens > int(context_limit) else "green",
            "memory": _memory_risk(estimated_peak_gib, float(h100_budget_gib)),
        },
        "notes": [
            "This is a static scheduler preflight, not a CUDA allocator measurement.",
            "Qwen ViT raw patch tokens use ceil(H/patch_size) * ceil(W/patch_size) * frames.",
            "Qwen LLM visual tokens are approximated after spatial_merge_size and AutoGaze reduction.",
        ],
    }


def _memory_risk(estimated_peak_gib: float, budget_gib: float) -> str:
    if estimated_peak_gib >= budget_gib:
        return "red"
    if estimated_peak_gib >= budget_gib * 0.78:
        return "yellow"
    return "green"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Qwen plugin AutoGaze sparse token/context/H100 risk without CUDA.")
    parser.add_argument("--model-family", default="qwen3-vl")
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument("--autogaze-reduction-ratio", type=float, default=10.0)
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument("--max-spatial-chunks", type=int)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--h100-budget-gib", type=float, default=70.0)
    parser.add_argument("--resident-model-gib", type=float, default=18.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            estimate_qwen_plugin_preflight(
                model_family=args.model_family,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                patch_size=args.patch_size,
                spatial_merge_size=args.spatial_merge_size,
                autogaze_reduction_ratio=args.autogaze_reduction_ratio,
                chunk_frames=args.chunk_frames,
                max_spatial_chunks=args.max_spatial_chunks,
                prompt_tokens=args.prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                context_limit=args.context_limit,
                h100_budget_gib=args.h100_budget_gib,
                resident_model_gib=args.resident_model_gib,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
