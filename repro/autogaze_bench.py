from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from repro.common import BenchmarkTimer, compute_stats, environment_metadata, resolve_device, write_csv, write_json


def add_external_autogaze(path: str) -> None:
    repo = Path(path).resolve()
    if not (repo / "autogaze").exists():
        raise FileNotFoundError(f"AutoGaze repo not found at {repo}")
    sys.path.insert(0, str(repo))


def _flatten_bools(value: Any) -> list[bool]:
    if isinstance(value, (list, tuple)):
        flattened: list[bool] = []
        for item in value:
            flattened.extend(_flatten_bools(item))
        return flattened
    return [bool(value)]


def _mask_counts(mask: Any) -> tuple[int, int, int]:
    if hasattr(mask, "bool") and hasattr(mask, "sum") and hasattr(mask, "numel"):
        bool_mask = mask.bool()
        padded = int(bool_mask.sum().item())
        total = int(bool_mask.numel())
        return total - padded, padded, total
    values = _flatten_bools(mask)
    padded = sum(values)
    total = len(values)
    return total - padded, padded, total


def _to_int_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_to_int_list(item))
        return flattened
    return [int(value)]


def parse_int_sequence(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = [int(part) for part in re.findall(r"\d+", value)]
    return parsed or None


def load_video_frames(video_path: str, frame_count: int) -> Any:
    import av
    from autogaze.datasets.video_utils import read_video_pyav

    container = av.open(video_path)
    try:
        return read_video_pyav(container=container, indices=list(range(frame_count)))
    finally:
        container.close()


def tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def repeat_video_batch(video_tensor: Any, batch_size: int) -> Any:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    repeats = [1] * len(video_tensor.shape)
    repeats[0] = batch_size
    return video_tensor.repeat(*repeats)


def flatten_video_batch_for_siglip_baseline(video_tensor: Any) -> Any:
    if len(video_tensor.shape) != 5:
        raise ValueError(f"Expected video tensor with shape (B, T, C, H, W), got {tuple(video_tensor.shape)}")
    batch, frames, channels, height, width = video_tensor.shape
    return video_tensor.reshape(batch * frames, channels, height, width)


def select_siglip_vision_model_class(model_type: str, siglip_cls: Any, siglip2_cls: Any) -> Any:
    if model_type == "siglip2_vision_model":
        return siglip2_cls
    return siglip_cls


def summarize_gaze(gaze_outputs: dict[str, Any], raw_patch_budget: int) -> dict[str, Any]:
    selected, padded_count, total_gaze_slots = _mask_counts(gaze_outputs["if_padded_gazing"])
    return {
        "raw_patch_budget": raw_patch_budget,
        "selected_non_padded_patches": selected,
        "padded_gazing_positions": padded_count,
        "total_gaze_slots": total_gaze_slots,
        "token_reduction_ratio": raw_patch_budget / selected if selected else 0.0,
        "num_gazing_each_frame": _to_int_list(gaze_outputs["num_gazing_each_frame"]),
    }


def target_patches_per_frame(target_scales: list[int] | None, target_patch_size: int | None, fallback: int) -> int:
    if not target_scales or not target_patch_size:
        return int(fallback)
    return sum((int(scale) // int(target_patch_size)) ** 2 for scale in target_scales)


def run(args: argparse.Namespace) -> None:
    import torch

    add_external_autogaze(args.autogaze_repo)

    from autogaze.datasets.video_utils import transform_video_for_pytorch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    batch_size = int(getattr(args, "batch_size", 1))
    target_scales = parse_int_sequence(getattr(args, "target_scales", None))
    target_patch_size = getattr(args, "target_patch_size", None)
    target_patch_size = int(target_patch_size) if target_patch_size is not None else None

    autogaze_transform = AutoGazeImageProcessor.from_pretrained(args.autogaze_model)
    autogaze_model = AutoGaze.from_pretrained(args.autogaze_model).to(device)
    autogaze_model.eval()

    frame_count = args.frames or int(autogaze_model.config.max_num_frames)
    raw_video = load_video_frames(args.video, frame_count)
    video_input_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device=device, dtype=dtype)
    video_input_autogaze = repeat_video_batch(video_input_autogaze, batch_size)

    baseline_siglip_model = None
    gazed_siglip_model = None
    video_input_siglip = None
    video_input_siglip_baseline = None
    if not args.skip_siglip:
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            Siglip2VisionModel as HFSiglip2VisionModel,
            SiglipVisionModel as HFSiglipVisionModel,
        )
        from autogaze.vision_encoders.siglip import SiglipVisionModel as AutoGazeSiglipVisionModel

        siglip_transform = AutoImageProcessor.from_pretrained(args.siglip_model)
        siglip_config = AutoConfig.from_pretrained(args.siglip_model)
        baseline_siglip_cls = select_siglip_vision_model_class(
            siglip_config.model_type,
            HFSiglipVisionModel,
            HFSiglip2VisionModel,
        )
        baseline_siglip_model = baseline_siglip_cls.from_pretrained(args.siglip_model).to(device=device, dtype=dtype)
        baseline_siglip_model.eval()
        gazed_siglip_model = AutoGazeSiglipVisionModel.from_pretrained(
            args.siglip_model,
            scales=target_scales or autogaze_model.config.scales,
            attn_implementation=args.attn_implementation,
        ).to(device=device, dtype=dtype)
        gazed_siglip_model.eval()
        video_input_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device=device, dtype=dtype)
        video_input_siglip = repeat_video_batch(video_input_siglip, batch_size)
        video_input_siglip_baseline = flatten_video_batch_for_siglip_baseline(video_input_siglip)

    patches_per_frame = target_patches_per_frame(
        target_scales,
        target_patch_size,
        int(autogaze_model.num_vision_tokens_each_frame),
    )
    raw_patch_budget = int(batch_size * frame_count * patches_per_frame)
    forward_kwargs: dict[str, Any] = {
        "gazing_ratio": args.gazing_ratio,
        "task_loss_requirement": args.task_loss_requirement,
    }
    if target_scales is not None:
        forward_kwargs["target_scales"] = target_scales
    if target_patch_size is not None:
        forward_kwargs["target_patch_size"] = target_patch_size

    with torch.inference_mode():
        for _ in range(args.warmup):
            gaze_outputs = autogaze_model({"video": video_input_autogaze}, **forward_kwargs)
            if not args.skip_siglip:
                _ = baseline_siglip_model(video_input_siglip_baseline)
                _ = gazed_siglip_model(video_input_siglip, gazing_info=gaze_outputs)

        autogaze_timer = BenchmarkTimer(device)
        siglip_full_timer = BenchmarkTimer(device)
        siglip_gazed_timer = BenchmarkTimer(device)
        final_gaze_outputs = None
        final_siglip_full = None
        final_siglip_gazed = None

        for _ in range(args.repeat):
            with autogaze_timer.measure():
                final_gaze_outputs = autogaze_model({"video": video_input_autogaze}, **forward_kwargs)
            if not args.skip_siglip:
                with siglip_full_timer.measure():
                    final_siglip_full = baseline_siglip_model(video_input_siglip_baseline)
                with siglip_gazed_timer.measure():
                    final_siglip_gazed = gazed_siglip_model(video_input_siglip, gazing_info=final_gaze_outputs)

    assert final_gaze_outputs is not None
    if not args.skip_siglip:
        assert final_siglip_full is not None
        assert final_siglip_gazed is not None

    gaze_summary = summarize_gaze(final_gaze_outputs, raw_patch_budget)
    latency_ms = {"autogaze": compute_stats(autogaze_timer.elapsed_ms)}
    shapes = {"video_input_autogaze": list(video_input_autogaze.shape)}
    input_tensor_bytes = {"autogaze": tensor_bytes(video_input_autogaze)}
    if not args.skip_siglip:
        latency_ms["siglip_full"] = compute_stats(siglip_full_timer.elapsed_ms)
        latency_ms["siglip_gazed"] = compute_stats(siglip_gazed_timer.elapsed_ms)
        shapes.update(
            {
                "video_input_siglip": list(video_input_siglip.shape),
                "video_input_siglip_baseline": list(video_input_siglip_baseline.shape),
                "siglip_full_hidden": list(final_siglip_full.last_hidden_state.shape),
                "siglip_gazed_hidden": list(final_siglip_gazed.last_hidden_state.shape),
            }
        )
        input_tensor_bytes["siglip"] = tensor_bytes(video_input_siglip)
    result = {
        "metadata": environment_metadata(device),
        "input": {
            "video": args.video,
            "frames": frame_count,
            "batch_size": batch_size,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "target_scales": target_scales,
            "target_patch_size": target_patch_size,
            "skip_siglip": args.skip_siglip,
            "dtype": args.dtype,
        },
        "autogaze_latency_options": {
            "batch_size": batch_size,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "target_scales": target_scales,
            "target_patch_size": target_patch_size,
            "patches_per_frame": patches_per_frame,
            "frames": frame_count,
            "dtype": args.dtype,
            "device": device.type,
            "siglip_enabled": not args.skip_siglip,
            "note": (
                "Quick Start default batch_size is 1. Larger batches usually improve throughput for multiple "
                "clips/tile sequences but can increase single-call latency and memory."
            ),
        },
        "models": {
            "autogaze": args.autogaze_model,
            "siglip": args.siglip_model,
        },
        "gaze": gaze_summary,
        "latency_ms": latency_ms,
        "shapes": shapes,
        "input_tensor_bytes": input_tensor_bytes,
    }

    write_json(args.output_json, result)
    write_csv(
        args.output_csv,
        [
            {
                "stage": stage,
                **stats,
                "token_reduction_ratio": gaze_summary["token_reduction_ratio"],
                "selected_non_padded_patches": gaze_summary["selected_non_padded_patches"],
                "raw_patch_budget": gaze_summary["raw_patch_budget"],
            }
            for stage, stats in result["latency_ms"].items()
        ],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AutoGaze and AutoGaze-compatible SigLIP")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gazing-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--target-scales")
    parser.add_argument("--target-patch-size", type=int)
    parser.add_argument("--skip-siglip", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/autogaze_siglip_bench.json")
    parser.add_argument("--output-csv", default="outputs/autogaze_repro/autogaze_siglip_bench.csv")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
