from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_url
from transformers import AutoModel, AutoProcessor

from repro.common import append_jsonl, environment_metadata, resolve_device, synchronize, write_json, write_jsonl
from repro.hlvid import load_hlvid_manifest, parse_choice, score_predictions

DEFAULT_MODEL = "nvidia/NVILA-8B-HD-Video"
DEFAULT_EXAMPLE_VIDEO = "https://huggingface.co/datasets/bfshi/HLVid/resolve/main/example/clip_av_video_5_001.mp4"
DEFAULT_PROMPT = (
    "Question: What does the white text on the green road sign say?\n"
    "A. Hampden St\n"
    "B. Hampden Ave\n"
    "C. HampdenBlvd\n"
    "D. Hampden Rd\n"
    "Please answer directly with the letter of the correct answer."
)


def processor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_video_frames": args.num_video_frames,
        "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
        "max_tiles_video": args.max_tiles_video,
        "autogaze_model_id": args.autogaze_model,
        "gazing_ratio_tile": [0.2] + [0.06] * 15,
        "gazing_ratio_thumbnail": 1,
        "task_loss_requirement_tile": args.task_loss_requirement_tile,
        "task_loss_requirement_thumbnail": None,
        "max_batch_size_autogaze": args.max_batch_size_autogaze,
        "trust_remote_code": True,
    }


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs(args))
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map=args.device_map,
        max_batch_size_siglip=args.max_batch_size_siglip,
    )
    model.eval()
    return model, processor


def input_device(model, fallback: torch.device) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def resolve_video(video: str, args: argparse.Namespace) -> str:
    if video.startswith("http://") or video.startswith("https://"):
        return video
    local = Path(video)
    if local.exists():
        return str(local)
    cached = Path(args.hlvid_video_root) / video
    if cached.exists():
        return str(cached)
    return hf_hub_url(repo_id=args.hlvid_repo, filename=video, repo_type="dataset")


def tensor_shapes(payload: dict[str, Any]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in payload.items() if isinstance(value, torch.Tensor)}


def extract_gaze_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "autogaze_selected_patches": None,
        "autogaze_padded_patches": None,
        "autogaze_total_gaze_slots": None,
        "autogaze_token_reduction_ratio": None,
        "available_input_keys": sorted(payload.keys()),
    }
    for value in payload.values():
        if not isinstance(value, dict) or "if_padded_gazing" not in value:
            continue
        padded = value["if_padded_gazing"]
        if isinstance(padded, torch.Tensor):
            padded_bool = padded.bool()
            selected = int((~padded_bool).sum().item())
            padded_count = int(padded_bool.sum().item())
            total = int(padded_bool.numel())
            metrics.update(
                {
                    "autogaze_selected_patches": selected,
                    "autogaze_padded_patches": padded_count,
                    "autogaze_total_gaze_slots": total,
                    "autogaze_token_reduction_ratio": None,
                }
            )
            return metrics
    return metrics


def move_tensors(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}


def timed_generate(model, inputs: dict[str, Any], processor, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    synchronize(device)
    generate_ms = (time.perf_counter() - start) * 1000.0
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "raw_output": response,
        "parsed_answer": parse_choice(response),
        "generate_ms": generate_ms,
        "generated_tokens": int(generated.shape[1]),
        "peak_memory_bytes": peak_memory_bytes,
    }


def generate_one(model, processor, video: str, prompt: str, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    video_token = processor.tokenizer.video_token
    resolved_video = resolve_video(video, args)

    start = time.perf_counter()
    inputs = processor(text=f"{video_token}\n\n{prompt}", videos=resolved_video, return_tensors="pt")
    preprocess_ms = (time.perf_counter() - start) * 1000.0

    target_device = input_device(model, device)
    inputs = move_tensors(dict(inputs), target_device)

    ttft_ms = None
    if args.measure_ttft:
        one_token = timed_generate(model, inputs, processor, device, max_new_tokens=1)
        ttft_ms = one_token["generate_ms"]

    result = timed_generate(model, inputs, processor, device, max_new_tokens=args.max_new_tokens)
    decode_estimated_ms = max(result["generate_ms"] - ttft_ms, 0.0) if ttft_ms is not None else None
    gaze_metrics = extract_gaze_metrics(inputs)

    return {
        **result,
        **gaze_metrics,
        "video_input": video,
        "video_resolved": resolved_video,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "input_shapes": tensor_shapes(inputs),
        "video_preprocess_ms": preprocess_ms,
        "ttft_ms": ttft_ms,
        "decode_estimated_ms": decode_estimated_ms,
        "total_ms": preprocess_ms + result["generate_ms"],
        "vision_encoder_ms": None,
        "mllm_prefill_ms": ttft_ms,
        "metric_note": "Remote NVILA code may not expose separate vision encoder and AutoGaze internals; null fields mean unavailable from public generate outputs.",
    }


def run_single(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, processor = load_model_and_processor(args)
    result = generate_one(model, processor, args.video, args.prompt, device, args)
    payload = {
        "metadata": environment_metadata(device),
        "model_path": args.model_path,
        "autogaze_model": args.autogaze_model,
        "video": args.video,
        "prompt": args.prompt,
        "result": result,
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def completed_question_ids(path: Path) -> set[Any]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text().splitlines() if line.strip()}


def run_hlvid(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    rows = load_hlvid_manifest(split=args.split, limit=args.limit, config=args.config)
    model, processor = load_model_and_processor(args)
    output_path = Path(args.predictions)
    completed_ids = completed_question_ids(output_path) if args.resume else set()

    for row in rows:
        if row["question_id"] in completed_ids:
            continue
        result = generate_one(model, processor, row["video_path"], row["question"], device, args)
        prediction = {
            **row,
            **result,
            "model_path": args.model_path,
            "autogaze_model": args.autogaze_model,
            "num_video_frames": args.num_video_frames,
            "num_video_frames_thumbnail": args.num_video_frames_thumbnail,
            "max_tiles_video": args.max_tiles_video,
            "task_loss_requirement_tile": args.task_loss_requirement_tile,
        }
        append_jsonl(output_path, [prediction])

    all_rows = []
    if output_path.exists():
        all_rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    summary, scored = score_predictions(all_rows)
    write_json(args.summary, summary)
    write_jsonl(args.scored_predictions, scored)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NVILA-HD-Video quickstart and HLVid benchmark")
    parser.add_argument("--mode", choices=["single", "hlvid"], default="single")
    parser.add_argument("--model-path", "--nvila-model", dest="model_path", default=DEFAULT_MODEL)
    parser.add_argument("--autogaze-model", default="nvidia/AutoGaze")
    parser.add_argument("--device", default="cuda", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--video", default=DEFAULT_EXAMPLE_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=64)
    parser.add_argument("--max-tiles-video", type=int, default=48)
    parser.add_argument("--task-loss-requirement-tile", type=float, default=0.6)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--max-batch-size-siglip", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-ttft", action="store_true")
    parser.add_argument("--hlvid-repo", default="bfshi/HLVid")
    parser.add_argument("--hlvid-video-root", default="data/hlvid/videos")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default="outputs/autogaze_repro/nvila_single.json")
    parser.add_argument("--predictions", default="outputs/autogaze_repro/hlvid_predictions.jsonl")
    parser.add_argument("--summary", default="outputs/autogaze_repro/hlvid_summary.json")
    parser.add_argument("--scored-predictions", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "single":
        run_single(args)
    else:
        run_hlvid(args)


if __name__ == "__main__":
    main()
