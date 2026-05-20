from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from repro.common import BenchmarkTimer, compute_stats, environment_metadata, resolve_device, write_json


def add_external_autogaze(path: str) -> None:
    repo = Path(path).resolve()
    if not (repo / "autogaze").exists():
        raise FileNotFoundError(f"AutoGaze repo not found at {repo}")
    sys.path.insert(0, str(repo))


def parse_int_sequence(value: str | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = [int(part) for part in re.findall(r"\d+", value)]
    return parsed or None


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


def autogaze_transform_kwargs(target_scales: list[int] | None) -> dict[str, dict[str, int]]:
    if not target_scales:
        return {}
    largest_scale = int(target_scales[-1])
    return {
        "size": {"height": largest_scale, "width": largest_scale},
        "crop_size": {"height": largest_scale, "width": largest_scale},
    }


def wall_time_ms(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def run(args: argparse.Namespace) -> None:
    import av
    import torch

    add_external_autogaze(args.autogaze_repo)

    from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
    from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    target_scales = parse_int_sequence(getattr(args, "target_scales", None))
    target_patch_size = getattr(args, "target_patch_size", None)
    target_patch_size = int(target_patch_size) if target_patch_size is not None else None

    autogaze_transform, transform_load_ms = wall_time_ms(
        lambda: AutoGazeImageProcessor.from_pretrained(
            args.autogaze_model,
            **autogaze_transform_kwargs(target_scales),
        )
    )
    autogaze_model, model_load_ms = wall_time_ms(lambda: AutoGaze.from_pretrained(args.autogaze_model))
    autogaze_model = autogaze_model.to(device=device, dtype=dtype)
    autogaze_model.eval()

    frame_count = args.frames or int(autogaze_model.config.max_num_frames)

    container = av.open(args.video)
    try:
        sample_indices = list(range(frame_count))
        raw_video, decode_ms = wall_time_ms(
            lambda: read_video_pyav(container=container, indices=sample_indices)
        )
    finally:
        container.close()

    video_input_autogaze, preprocess_ms = wall_time_ms(
        lambda: transform_video_for_pytorch(raw_video, autogaze_transform)
    )
    video_input_autogaze = video_input_autogaze[None]
    video_input_autogaze, input_move_ms = wall_time_ms(
        lambda: video_input_autogaze.to(device=device, dtype=dtype)
    )

    patches_per_frame = target_patches_per_frame(
        target_scales,
        target_patch_size,
        int(autogaze_model.num_vision_tokens_each_frame),
    )
    raw_patch_budget = int(frame_count * patches_per_frame)
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
            _ = autogaze_model({"video": video_input_autogaze}, **forward_kwargs)

        autogaze_timer = BenchmarkTimer(device)
        final_gaze_outputs = None
        for _ in range(args.repeat):
            with autogaze_timer.measure():
                final_gaze_outputs = autogaze_model({"video": video_input_autogaze}, **forward_kwargs)

    assert final_gaze_outputs is not None
    gaze_summary = summarize_gaze(final_gaze_outputs, raw_patch_budget)
    result = {
        "metadata": environment_metadata(device, autogaze_root=args.autogaze_repo),
        "mode": "quickstart-native",
        "source": {
            "reference": "external/AutoGaze/QUICK_START.md",
            "section": "Running AutoGaze on a Video",
            "note": (
                "The measured forward call follows the Quick Start code path. "
                "Video decode and preprocessing are timed separately and are not included in latency_ms.autogaze_forward."
            ),
        },
        "input": {
            "video": args.video,
            "frames": frame_count,
            "batch_size": 1,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "target_scales": target_scales,
            "target_patch_size": target_patch_size,
            "dtype": args.dtype,
        },
        "models": {
            "autogaze": args.autogaze_model,
        },
        "gaze": gaze_summary,
        "latency_ms": {
            "autogaze_forward": compute_stats(autogaze_timer.elapsed_ms),
            "video_decode": {"count": 1, "mean": decode_ms, "median": decode_ms, "min": decode_ms, "max": decode_ms},
            "video_preprocess": {
                "count": 1,
                "mean": preprocess_ms,
                "median": preprocess_ms,
                "min": preprocess_ms,
                "max": preprocess_ms,
            },
            "input_to_device": {
                "count": 1,
                "mean": input_move_ms,
                "median": input_move_ms,
                "min": input_move_ms,
                "max": input_move_ms,
            },
            "model_load": {
                "count": 1,
                "mean": model_load_ms,
                "median": model_load_ms,
                "min": model_load_ms,
                "max": model_load_ms,
            },
            "processor_load": {
                "count": 1,
                "mean": transform_load_ms,
                "median": transform_load_ms,
                "min": transform_load_ms,
                "max": transform_load_ms,
            },
        },
        "shapes": {
            "video_input_autogaze": list(video_input_autogaze.shape),
        },
        "input_tensor_bytes": {
            "autogaze": int(video_input_autogaze.numel() * video_input_autogaze.element_size()),
        },
        "autogaze_latency_options": {
            "batch_size": 1,
            "gazing_ratio": args.gazing_ratio,
            "task_loss_requirement": args.task_loss_requirement,
            "target_scales": target_scales,
            "target_patch_size": target_patch_size,
            "patches_per_frame": patches_per_frame,
            "frames": frame_count,
            "dtype": args.dtype,
            "device": device.type,
            "decode_preprocess_excluded_from_forward": True,
        },
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoGaze Quick Start video snippet with focused timing")
    parser.add_argument("--autogaze-repo", default=".")
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--video", default="assets/example_input.mp4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--frames", type=int)
    parser.add_argument("--gazing-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--target-scales")
    parser.add_argument("--target-patch-size", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-json", default="outputs/autogaze_repro/quickstart_native_autogaze.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
