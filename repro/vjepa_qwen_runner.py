from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from repro.common import resolve_device, synchronize, write_json
from repro.hlvid_example_autogaze import read_video_metadata
from repro.nvila_runner import apply_resize_to_dimensions, load_sampled_video_frames
from repro.plugins.autogaze_sparse_selector import (
    AutogazeSelectorRuntimeConfig,
    run_direct_autogaze_selector,
    runtime_config_from_args,
)
from repro.plugins.gaze_plan import SparseSelectionPlan, sparse_selection_plan_from_dict
from repro.plugins.vjepa_mapping import (
    VjepaGridConfig,
    VjepaTokenSelection,
    dense_vjepa_token_selection,
    scale_aware_vjepa_selection_from_sparse_plan,
    vjepa_token_selection_from_sparse_plan,
)
from repro.plugins.vjepa_qwen_bridge import (
    build_qwen_bridge_inputs_from_vjepa_features,
    project_vjepa_features_to_qwen_dim,
)
from repro.plugins.vjepa_sparse_runtime import run_vjepa_encoder_on_selected_embeddings
from repro.vjepa_qwen_colab_smoke import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_VJEPA_MODEL,
    _device_metadata,
    _qwen_embedding_hidden_size,
    _resolve_qwen_model_class,
    _vjepa_encoder,
    _vjepa_patch_embeddings,
)

DEFAULT_AUTOGAZE_MODEL = "nvidia/AutoGaze"
DEFAULT_OUTPUT_JSON = "outputs/autogaze_vjepa/vjepa_qwen_actual.json"
DEFAULT_OUTPUT_MD = "outputs/autogaze_vjepa/vjepa_qwen_actual.md"


class VjepaQwenStageError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Actual video runner for AutoGaze selector -> sparse V-JEPA encoder -> Qwen generate.",
    )
    parser.add_argument("--video", required=True, help="Input mp4/video path.")
    parser.add_argument("--prompt", default="Describe the video in one short sentence.")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)

    parser.add_argument(
        "--autogaze-mode",
        choices=["on", "off"],
        default="on",
        help="Use AutoGaze sparse selection, or keep all V-JEPA tokens as the dense off baseline.",
    )
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default=DEFAULT_AUTOGAZE_MODEL)
    parser.add_argument("--autogaze-device", default="auto")
    parser.add_argument("--autogaze-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--autogaze-chunk-frames", type=int, default=16)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--autogaze-tile-size", type=int, default=392)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--gazing-ratio", type=float)
    parser.add_argument("--task-loss-requirement", type=float)
    parser.add_argument("--autogaze-target-scales", default="56+112+196+392")
    parser.add_argument("--autogaze-target-patch-size", type=int, default=16)
    parser.add_argument("--autogaze-encoder-patch-size", type=int)
    parser.add_argument("--autogaze-generate-only", action="store_true")
    parser.add_argument("--autogaze-selector-output-json")

    parser.add_argument("--video-decode-strategy", default="auto", choices=["auto", "seek", "scan"])
    parser.add_argument("--video-resize-shortest-edge", type=int)
    parser.add_argument("--video-resize-longest-edge", type=int)
    parser.add_argument("--video-resize-width", type=int)
    parser.add_argument("--video-resize-height", type=int)

    parser.add_argument("--vjepa-model", default=DEFAULT_VJEPA_MODEL, help="HF repo id or local V-JEPA checkpoint path.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="HF repo id or local Qwen VL checkpoint path.")
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--vjepa-overlap-threshold", type=float, default=0.0)
    parser.add_argument(
        "--vjepa-selection-policy",
        choices=["single_scale_union", "scale_aware_multi_pass"],
        default="single_scale_union",
        help=(
            "single_scale_union maps all AutoGaze multiscale boxes to one V-JEPA crop grid. "
            "scale_aware_multi_pass runs one sparse V-JEPA pass per AutoGaze scale and concatenates selected features."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--clear-cuda-cache-between-stages", action=argparse.BooleanOptionalAction, default=True)
    return parser


def build_selector_config_from_args(args: argparse.Namespace | Any) -> AutogazeSelectorRuntimeConfig:
    return runtime_config_from_args(args)


def vjepa_resize_plan_from_args(args: argparse.Namespace | Any) -> dict[str, int | str]:
    crop_size = int(getattr(args, "crop_size", 224))
    return {"width": crop_size, "height": crop_size, "mode": "exact"}


def pil_frames_to_vjepa_pixel_values(
    frames: list[Image.Image],
    *,
    crop_size: int,
    dtype: Any,
    device: Any,
) -> Any:
    if not frames:
        raise ValueError("frames must not be empty")
    import numpy as np
    import torch

    arrays = []
    for frame in frames:
        image = frame.convert("RGB").resize((int(crop_size), int(crop_size)))
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    values = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=values.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=values.dtype).view(1, 3, 1, 1)
    values = (values - mean) / std
    return values.unsqueeze(0).to(device=device, dtype=dtype)


def run_actual_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab, set Runtime > Change runtime type > GPU and rerun.")

    device = resolve_device(args.device)
    dtype = _torch_dtype(args.dtype, device, torch)
    total_start = time.perf_counter()
    latency_ms: dict[str, float] = {}
    memory_bytes: dict[str, int | None] = {}

    def run_stage(stage: str, fn: Callable[[], Any]) -> Any:
        try:
            synchronize(device)
            start = time.perf_counter()
            result = fn()
            synchronize(device)
            latency_ms[stage] = (time.perf_counter() - start) * 1000.0
            memory_bytes[f"{stage}_cuda_peak"] = _cuda_peak_memory(torch, device)
            return result
        except Exception as exc:
            raise VjepaQwenStageError(stage, exc) from exc

    if getattr(device, "type", "") == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    video_metadata = run_stage("video_metadata_read", lambda: read_video_metadata(args.video))
    frames, vjepa_decode_stats = run_stage(
        "vjepa_video_decode_resize",
        lambda: load_sampled_video_frames(
            args.video,
            int(args.frames_per_clip),
            vjepa_resize_plan_from_args(args),
            decode_strategy=str(args.video_decode_strategy),
        ),
    )

    if args.autogaze_mode == "on":
        selector_config = build_selector_config_from_args(args)
        selector_payload = run_stage("autogaze_selector_total", lambda: run_direct_autogaze_selector(selector_config))
        sparse_plan = load_sparse_selection_plan(selector_payload["sparse_selection_plan_json"])
        if args.clear_cuda_cache_between_stages:
            clear_accelerator_cache(torch, device)
    else:
        selector_payload = _autogaze_off_payload(args, video_metadata)
        sparse_plan = None

    vjepa = run_stage(
        "vjepa_model_load",
        lambda: AutoModel.from_pretrained(args.vjepa_model, trust_remote_code=True, torch_dtype=dtype),
    )
    vjepa.eval().to(device)

    selection, vjepa_result = run_stage(
        "vjepa_sparse_encode",
        lambda: run_vjepa_from_selector_mode(
            vjepa=vjepa,
            sparse_plan=sparse_plan,
            frames=frames,
            autogaze_mode=str(args.autogaze_mode),
            device=device,
            dtype=dtype,
            frames_per_clip=int(args.frames_per_clip),
            tubelet_size=int(args.tubelet_size),
            crop_size=int(args.crop_size),
            patch_size=int(args.patch_size),
            overlap_threshold=float(args.vjepa_overlap_threshold),
            selection_policy=str(args.vjepa_selection_policy),
        ),
    )
    vjepa_features_cpu = vjepa_result["last_hidden_state"].detach().cpu()
    vjepa_runtime_metrics = vjepa_result["metrics"]
    del vjepa, vjepa_result
    if args.clear_cuda_cache_between_stages:
        clear_accelerator_cache(torch, device)

    qwen_model_class_name, qwen_model_cls = _resolve_qwen_model_class()
    qwen = run_stage(
        "qwen_model_load",
        lambda: qwen_model_cls.from_pretrained(
            args.qwen_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        ),
    )
    qwen.eval().to(device)
    tokenizer = run_stage("qwen_tokenizer_load", lambda: AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True))

    bridge_inputs, bridge_metadata = run_stage(
        "qwen_bridge_pack",
        lambda: build_qwen_inputs_from_vjepa_cpu_features(
            qwen,
            tokenizer,
            prompt=args.prompt,
            vjepa_features_cpu=vjepa_features_cpu,
            device=device,
            qwen_hidden_size=int(_qwen_embedding_hidden_size(qwen)),
        ),
    )
    generate_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in bridge_inputs.items()}

    generated_ids = run_stage(
        "qwen_generate",
        lambda: qwen.generate(**generate_inputs, max_new_tokens=int(args.max_new_tokens)),
    )
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    latency_ms["total"] = (time.perf_counter() - total_start) * 1000.0
    memory_bytes["cuda_peak_total"] = _cuda_peak_memory(torch, device)

    payload = {
        "status": "passed",
        "runner": "repro.vjepa_qwen_runner",
        "accuracy_status": "not_claimed",
        "integration_level": "autogaze_actual_to_vjepa_sparse_encoder_to_qwen_inputs_embeds_zero_shot_probe",
        "autogaze_mode": str(args.autogaze_mode),
        "models": {
            "autogaze": str(args.autogaze_model),
            "vjepa": str(args.vjepa_model),
            "qwen": str(args.qwen_model),
            "qwen_model_class": qwen_model_class_name,
        },
        "device": _device_metadata(device, torch),
        "source_video": {
            "path": str(args.video),
            "metadata": video_metadata,
        },
        "autogaze_selector": {
            "status": selector_payload.get("status"),
            "sparse_selection_plan_json": selector_payload.get("sparse_selection_plan_json"),
            "runtime_config": selector_payload.get("runtime_config"),
            "tokens": selector_payload.get("tokens"),
            "latency_ms": selector_payload.get("latency_ms"),
            "preprocess_video_plan": selector_payload.get("preprocess_video_plan"),
        },
        "sparse_selection_plan_summary": sparse_selection_plan_summary(sparse_plan) if sparse_plan is not None else None,
        "vjepa_mapping": selection.to_dict()["vjepa"],
        "vjepa_runtime": {
            "decode_stats": vjepa_decode_stats,
            "metrics": vjepa_runtime_metrics,
        },
        "qwen_bridge": bridge_metadata,
        "tokens": {
            "autogaze_raw_patch_tokens": sparse_plan.raw_patch_tokens if sparse_plan is not None else None,
            "autogaze_selected_patch_tokens": sparse_plan.selected_patch_tokens if sparse_plan is not None else None,
            "autogaze_reduction_ratio": (
                _safe_ratio(sparse_plan.raw_patch_tokens, sparse_plan.selected_patch_tokens)
                if sparse_plan is not None
                else None
            ),
            "vjepa_raw_tokens": selection.raw_token_count,
            "vjepa_selected_tokens": selection.selected_token_count,
            "vjepa_reduction_ratio": selection.reduction_ratio,
            "qwen_visual_tokens_inserted": bridge_metadata["visual_tokens_inserted"],
            "qwen_context_tokens": int(generate_inputs["inputs_embeds"].shape[1]),
        },
        "latency_ms": latency_ms,
        "memory_bytes": memory_bytes,
        "generated_text": generated_text,
    }
    return payload


def run_vjepa_from_selector_mode(
    *,
    vjepa: Any,
    sparse_plan: SparseSelectionPlan | None,
    frames: list[Image.Image],
    autogaze_mode: str,
    device: Any,
    dtype: Any,
    frames_per_clip: int,
    tubelet_size: int,
    crop_size: int,
    patch_size: int,
    overlap_threshold: float,
    selection_policy: str,
) -> tuple[VjepaTokenSelection, dict[str, Any]]:
    if autogaze_mode == "off":
        return run_dense_vjepa(
            vjepa,
            frames,
            device=device,
            dtype=dtype,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            crop_size=crop_size,
            patch_size=patch_size,
        )
    if sparse_plan is None:
        raise ValueError("sparse_plan is required when autogaze_mode=on")
    return run_sparse_vjepa_from_plan(
        vjepa,
        sparse_plan,
        frames,
        device=device,
        dtype=dtype,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        crop_size=crop_size,
        patch_size=patch_size,
        overlap_threshold=overlap_threshold,
        selection_policy=selection_policy,
    )


def run_dense_vjepa(
    vjepa: Any,
    frames: list[Image.Image],
    *,
    device: Any,
    dtype: Any,
    frames_per_clip: int,
    tubelet_size: int,
    crop_size: int,
    patch_size: int,
) -> tuple[VjepaTokenSelection, dict[str, Any]]:
    grid_config = VjepaGridConfig(
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        crop_size=crop_size,
        patch_size=patch_size,
    )
    selection = dense_vjepa_token_selection(grid_config)
    pixel_values = pil_frames_to_vjepa_pixel_values(frames, crop_size=crop_size, dtype=dtype, device=device)
    with _torch_inference():
        patch_embeddings = _vjepa_patch_embeddings(vjepa, pixel_values)
        result = run_vjepa_encoder_on_selected_embeddings(
            _vjepa_encoder(vjepa),
            patch_embeddings,
            selected_token_indices=selection.selected_token_indices,
        )
    result["metrics"] = {
        **result["metrics"],
        "selection_policy": "dense_off",
        "passes": 1,
    }
    return selection, result


def run_sparse_vjepa_from_plan(
    vjepa: Any,
    sparse_plan: SparseSelectionPlan,
    frames: list[Image.Image],
    *,
    device: Any,
    dtype: Any,
    frames_per_clip: int,
    tubelet_size: int,
    crop_size: int,
    patch_size: int,
    overlap_threshold: float,
    selection_policy: str,
) -> tuple[VjepaTokenSelection, dict[str, Any]]:
    if selection_policy == "scale_aware_multi_pass":
        return run_scale_aware_sparse_vjepa_from_plan(
            vjepa,
            sparse_plan,
            frames,
            device=device,
            dtype=dtype,
            frames_per_clip=frames_per_clip,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            overlap_threshold=overlap_threshold,
        )
    grid_config = VjepaGridConfig(
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        crop_size=crop_size,
        patch_size=patch_size,
    )
    selection = vjepa_token_selection_from_sparse_plan(
        sparse_plan,
        grid_config,
        overlap_threshold=overlap_threshold,
    )
    if not selection.selected_token_indices:
        raise ValueError("AutoGaze selected no V-JEPA tokens for single_scale_union policy")
    pixel_values = pil_frames_to_vjepa_pixel_values(frames, crop_size=crop_size, dtype=dtype, device=device)
    with _torch_inference():
        patch_embeddings = _vjepa_patch_embeddings(vjepa, pixel_values)
        result = run_vjepa_encoder_on_selected_embeddings(
            _vjepa_encoder(vjepa),
            patch_embeddings,
            selected_token_indices=selection.selected_token_indices,
        )
    result["metrics"] = {
        **result["metrics"],
        "selection_policy": selection_policy,
        "passes": 1,
    }
    return selection, result


def run_scale_aware_sparse_vjepa_from_plan(
    vjepa: Any,
    sparse_plan: SparseSelectionPlan,
    frames: list[Image.Image],
    *,
    device: Any,
    dtype: Any,
    frames_per_clip: int,
    tubelet_size: int,
    patch_size: int,
    overlap_threshold: float,
) -> tuple[VjepaTokenSelection, dict[str, Any]]:
    import torch

    selection = scale_aware_vjepa_selection_from_sparse_plan(
        sparse_plan,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        overlap_threshold=overlap_threshold,
    )
    if not selection.selected_token_indices:
        raise ValueError("AutoGaze selected no V-JEPA tokens for scale_aware_multi_pass policy")

    pass_outputs = []
    pass_metrics: dict[str, Any] = {}
    for scale_key, pass_info in sorted(selection.scale_passes.items(), key=lambda item: int(item[0])):
        local_indices = [int(index) for index in pass_info.get("local_selected_token_indices") or []]
        if not local_indices:
            continue
        scale_size = int(pass_info["scale_size"])
        pixel_values = pil_frames_to_vjepa_pixel_values(frames, crop_size=scale_size, dtype=dtype, device=device)
        with _torch_inference():
            patch_embeddings = _vjepa_patch_embeddings(vjepa, pixel_values)
            result = run_vjepa_encoder_on_selected_embeddings(
                _vjepa_encoder(vjepa),
                patch_embeddings,
                selected_token_indices=local_indices,
            )
        pass_outputs.append(result["last_hidden_state"])
        pass_metrics[str(scale_key)] = {
            **result["metrics"],
            "scale_size": scale_size,
            "local_selected_token_count": len(local_indices),
        }

    if not pass_outputs:
        raise ValueError("scale_aware_multi_pass produced no V-JEPA pass outputs")

    hidden = torch.cat(pass_outputs, dim=1)
    selected_count = int(hidden.shape[1])
    result = {
        "last_hidden_state": hidden,
        "position_mask": None,
        "attentions": None,
        "metrics": {
            "raw_token_count": int(selection.raw_token_count),
            "selected_token_count": selected_count,
            "encoder_token_reduction_ratio": float(selection.raw_token_count) / float(selected_count) if selected_count else None,
            "selection_policy": "scale_aware_multi_pass",
            "passes": len(pass_outputs),
            "pass_metrics": pass_metrics,
        },
    }
    return selection, result


def build_qwen_inputs_from_vjepa_cpu_features(
    qwen: Any,
    tokenizer: Any,
    *,
    prompt: str,
    vjepa_features_cpu: Any,
    device: Any,
    qwen_hidden_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vjepa_features = vjepa_features_cpu.to(device=device)
    projected = project_vjepa_features_to_qwen_dim(vjepa_features, qwen_hidden_size=qwen_hidden_size)
    bridge_inputs = build_qwen_bridge_inputs_from_vjepa_features(
        qwen,
        tokenizer,
        prompt=prompt,
        projected_vjepa_features=projected,
    )
    bridge_metadata = bridge_inputs.pop("vjepa_qwen_bridge_metadata")
    return bridge_inputs, bridge_metadata


def load_sparse_selection_plan(path: str | Path) -> SparseSelectionPlan:
    return sparse_selection_plan_from_dict(json.loads(Path(path).read_text()))


def sparse_selection_plan_summary(plan: SparseSelectionPlan) -> dict[str, Any]:
    by_scale: dict[str, int] = {}
    by_tile: dict[str, int] = {}
    for patch in plan.selected_patches:
        by_scale[str(int(patch.scale_id))] = by_scale.get(str(int(patch.scale_id)), 0) + 1
        by_tile[str(int(patch.tile_id))] = by_tile.get(str(int(patch.tile_id)), 0) + 1
    return {
        "selector_name": plan.selector_name,
        "source_video": plan.source_video.to_dict(),
        "preprocess_space": plan.preprocess_space.to_dict(),
        "patch_space": plan.patch_space.to_dict(),
        "selected_patch_count": len(plan.selected_patches),
        "selected_by_scale": by_scale,
        "selected_by_tile": by_tile,
        "token_accounting": {
            "raw_patch_tokens": plan.raw_patch_tokens,
            "selected_patch_tokens": plan.selected_patch_tokens,
            "reduction_ratio": _safe_ratio(plan.raw_patch_tokens, plan.selected_patch_tokens),
        },
    }


def write_markdown_summary(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tokens = payload.get("tokens") or {}
    latency = payload.get("latency_ms") or {}
    lines = [
        "# AutoGaze + V-JEPA + Qwen Actual Runner",
        "",
        f"- status: `{payload.get('status')}`",
        f"- accuracy_status: `{payload.get('accuracy_status')}`",
        f"- integration_level: `{payload.get('integration_level')}`",
        f"- autogaze_mode: `{payload.get('autogaze_mode')}`",
        f"- video: `{(payload.get('source_video') or {}).get('path')}`",
        "",
        "## Token Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "autogaze_raw_patch_tokens",
        "autogaze_selected_patch_tokens",
        "autogaze_reduction_ratio",
        "vjepa_raw_tokens",
        "vjepa_selected_tokens",
        "vjepa_reduction_ratio",
        "qwen_visual_tokens_inserted",
        "qwen_context_tokens",
    ):
        lines.append(f"| {key} | {tokens.get(key)} |")
    lines.extend(["", "## Latency", "", "| stage | ms |", "|---|---:|"])
    for key, value in sorted(latency.items()):
        lines.append(f"| {key} | {value:.2f} |" if isinstance(value, (int, float)) else f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Output",
            "",
            str(payload.get("generated_text", "")),
            "",
        ]
    )
    target.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_actual_pipeline(args)
        exit_code = 0
    except VjepaQwenStageError as exc:  # pragma: no cover - exercised in CUDA/Colab failures
        payload = _failure_payload(args, exc.cause, stage=exc.stage)
        exit_code = 1
    except Exception as exc:  # pragma: no cover - exercised in CUDA/Colab failures
        payload = _failure_payload(args, exc, stage="unknown")
        exit_code = 1
    write_json(args.output_json, payload)
    if args.output_md:
        write_markdown_summary(args.output_md, payload)
    print(json.dumps(_summary_for_stdout(payload), indent=2, sort_keys=True))
    return exit_code


def _failure_payload(args: argparse.Namespace, exc: BaseException, *, stage: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "runner": "repro.vjepa_qwen_runner",
        "accuracy_status": "not_claimed",
        "integration_level": "autogaze_actual_to_vjepa_sparse_encoder_to_qwen_inputs_embeds_zero_shot_probe",
        "autogaze_mode": str(getattr(args, "autogaze_mode", "unknown")),
        "models": {
            "autogaze": str(getattr(args, "autogaze_model", "")),
            "vjepa": str(getattr(args, "vjepa_model", "")),
            "qwen": str(getattr(args, "qwen_model", "")),
        },
        "failure": {
            "stage": stage,
            "kind": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def _autogaze_off_payload(args: argparse.Namespace, video_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "sparse_selection_plan_json": None,
        "runtime_config": {
            "autogaze_mode": "off",
            "num_video_frames": int(getattr(args, "num_video_frames", 0) or 0),
            "frames_per_clip": int(getattr(args, "frames_per_clip", 0) or 0),
        },
        "tokens": {
            "raw_patch_tokens": None,
            "selected_patch_tokens": None,
            "reduction_ratio": None,
        },
        "latency_ms": {},
        "preprocess_video_plan": {
            "status": "skipped_autogaze_off",
            "source_video_metadata": video_metadata,
        },
    }


def _summary_for_stdout(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "passed":
        return payload
    return {
        "status": payload["status"],
        "accuracy_status": payload["accuracy_status"],
        "integration_level": payload["integration_level"],
        "autogaze_mode": payload.get("autogaze_mode"),
        "models": payload["models"],
        "device": payload["device"],
        "tokens": payload["tokens"],
        "latency_ms": payload["latency_ms"],
        "generated_text": payload["generated_text"],
    }


def _torch_dtype(dtype_name: str, device: Any, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return torch_module.float16 if getattr(device, "type", "") == "cuda" else torch_module.float32
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _torch_inference():
    import torch

    return torch.inference_mode()


def _cuda_peak_memory(torch_module: Any, device: Any) -> int | None:
    if getattr(device, "type", "") != "cuda" or not torch_module.cuda.is_available():
        return None
    return int(torch_module.cuda.max_memory_allocated(device))


def clear_accelerator_cache(torch_module: Any, device: Any) -> None:
    gc.collect()
    if getattr(device, "type", "") == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats(device)
    elif getattr(device, "type", "") == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.empty_cache()


def _safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())
