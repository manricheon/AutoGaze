from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from repro.common import write_json
from repro.plugins.vjepa_mapping import VjepaGridConfig, vjepa_token_selection_from_sparse_plan
from repro.plugins.vjepa_qwen_bridge import (
    build_qwen_bridge_inputs_from_vjepa_features,
    project_vjepa_features_to_qwen_dim,
)
from repro.plugins.vjepa_sparse_runtime import run_vjepa_encoder_on_selected_embeddings
from repro.vjepa_poc import build_synthetic_sparse_selection_plan


DEFAULT_VJEPA_MODEL = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_JSON = "outputs/autogaze_vjepa/colab_vjepa_qwen_smoke.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab CUDA smoke for AutoGaze + V-JEPA sparse + Qwen bridge.")
    parser.add_argument("--vjepa-model", default=DEFAULT_VJEPA_MODEL, help="HF repo id or local V-JEPA checkpoint path.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="HF repo id or local Qwen VL checkpoint path.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Describe the video in one short sentence.")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    return parser


def qwen_model_class_candidates() -> list[tuple[str, str]]:
    return [
        ("AutoModelForImageTextToText", "image-text-to-text"),
        ("AutoModelForVision2Seq", "vision2seq"),
        ("AutoModelForCausalLM", "causal-lm"),
    ]


def run_colab_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab, set Runtime > Change runtime type > GPU and rerun.")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = _torch_dtype(args.dtype, torch)

    grid_config = VjepaGridConfig(
        frames_per_clip=int(args.frames_per_clip),
        tubelet_size=int(args.tubelet_size),
        crop_size=int(args.crop_size),
        patch_size=int(args.patch_size),
    )
    sparse_plan = build_synthetic_sparse_selection_plan()
    selection = vjepa_token_selection_from_sparse_plan(sparse_plan, grid_config)

    vjepa = AutoModel.from_pretrained(
        args.vjepa_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    vjepa.eval().to(device)

    pixel_values_videos = torch.zeros(
        (1, 3, grid_config.frames_per_clip, grid_config.crop_size, grid_config.crop_size),
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        patch_embeddings = _vjepa_patch_embeddings(vjepa, pixel_values_videos)
        sparse_vjepa = run_vjepa_encoder_on_selected_embeddings(
            _vjepa_encoder(vjepa),
            patch_embeddings,
            selected_token_indices=selection.selected_token_indices,
        )
        vjepa_features = sparse_vjepa["last_hidden_state"]

    qwen_model_class_name, qwen_model_cls = _resolve_qwen_model_class()
    qwen = qwen_model_cls.from_pretrained(
        args.qwen_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    qwen.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)

    qwen_hidden_size = int(_qwen_embedding_hidden_size(qwen))
    projected = project_vjepa_features_to_qwen_dim(vjepa_features, qwen_hidden_size=qwen_hidden_size)
    bridge_inputs = build_qwen_bridge_inputs_from_vjepa_features(
        qwen,
        tokenizer,
        prompt=args.prompt,
        projected_vjepa_features=projected,
    )
    bridge_metadata = bridge_inputs.pop("vjepa_qwen_bridge_metadata")
    generate_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in bridge_inputs.items()
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    with torch.no_grad():
        generated_ids = qwen.generate(**generate_inputs, max_new_tokens=int(args.max_new_tokens))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "status": "passed",
        "runner": "repro.vjepa_qwen_colab_smoke",
        "accuracy_status": "not_claimed",
        "integration_level": "vjepa_sparse_encoder_to_qwen_inputs_embeds_zero_shot_probe",
        "models": {
            "vjepa": str(args.vjepa_model),
            "qwen": str(args.qwen_model),
            "qwen_model_class": qwen_model_class_name,
        },
        "device": _device_metadata(device, torch),
        "grid_config": grid_config.to_dict(),
        "tokens": {
            "vjepa_raw_tokens": selection.raw_token_count,
            "vjepa_selected_tokens": selection.selected_token_count,
            "vjepa_reduction_ratio": selection.reduction_ratio,
            "qwen_visual_tokens_inserted": bridge_metadata["visual_tokens_inserted"],
            "qwen_context_tokens": int(generate_inputs["inputs_embeds"].shape[1]),
        },
        "bridge_metadata": bridge_metadata,
        "vjepa_sparse_encoder_metrics": sparse_vjepa["metrics"],
        "generated_text": generated_text,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_colab_smoke(args)
        exit_code = 0
    except Exception as exc:  # pragma: no cover - exercised in CUDA notebook failure cases
        payload = {
            "status": "failed",
            "runner": "repro.vjepa_qwen_colab_smoke",
            "failure": {
                "kind": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = 1
    write_json(args.output_json, payload)
    print(json.dumps(_summary_for_stdout(payload), indent=2, sort_keys=True))
    return exit_code


def _summary_for_stdout(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "passed":
        return payload
    return {
        "status": payload["status"],
        "accuracy_status": payload["accuracy_status"],
        "integration_level": payload["integration_level"],
        "models": payload["models"],
        "device": payload["device"],
        "tokens": payload["tokens"],
        "generated_text": payload["generated_text"],
    }


def _resolve_qwen_model_class() -> tuple[str, Any]:
    import transformers

    for class_name, _ in qwen_model_class_candidates():
        candidate = getattr(transformers, class_name, None)
        if candidate is not None:
            return class_name, candidate
    raise RuntimeError("No supported Qwen model auto class is available in transformers.")


def _torch_dtype(name: str, torch_module: Any) -> Any:
    if name == "float32":
        return torch_module.float32
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _vjepa_encoder(model: Any) -> Any:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("V-JEPA model does not expose encoder")
    return encoder


def _vjepa_patch_embeddings(model: Any, pixel_values_videos: Any) -> Any:
    encoder = _vjepa_encoder(model)
    embeddings = getattr(encoder, "embeddings", None)
    if embeddings is None:
        raise ValueError("V-JEPA encoder does not expose embeddings")
    return embeddings(pixel_values_videos)


def _qwen_embedding_hidden_size(model: Any) -> int:
    embeddings = None
    for target in (model, getattr(model, "model", None)):
        getter = getattr(target, "get_input_embeddings", None)
        if getter is not None:
            embeddings = getter()
            if embeddings is not None:
                break
    if embeddings is None:
        raise ValueError("Qwen model does not expose input embeddings")
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise ValueError("Qwen input embeddings do not expose weight")
    return int(weight.shape[-1])


def _device_metadata(device: Any, torch_module: Any) -> dict[str, Any]:
    metadata = {
        "type": str(getattr(device, "type", device)),
        "cuda_available": bool(torch_module.cuda.is_available()),
    }
    if getattr(device, "type", None) == "cuda":
        metadata["cuda_device_name"] = torch_module.cuda.get_device_name(device)
        metadata["cuda_peak_memory_bytes"] = int(torch_module.cuda.max_memory_allocated(device))
    return metadata


if __name__ == "__main__":
    raise SystemExit(main())
