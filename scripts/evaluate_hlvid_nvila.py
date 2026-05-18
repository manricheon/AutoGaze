#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from poc_infer_utils import capture_memory_snapshot, load_config, nested_get, resolve_path, write_json


CHOICES = ("A", "B", "C", "D")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate NVILA-HD-Video on HLVid-style multiple-choice video QA."
    )
    parser.add_argument("--config", default="configs/poc_inference/hlvid_nvila_hd_eval.yaml")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-path", default=None, help="Local JSON/JSONL/CSV file. Preferred for offline tests.")
    parser.add_argument("--video-root", default=None, help="Optional root for relative video paths in local records.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--allow-real-model-loading", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    parser.add_argument("--num-video-frames", type=int, default=None)
    parser.add_argument("--num-video-frames-thumbnail", type=int, default=None)
    parser.add_argument("--max-tiles-video", type=int, default=None)
    parser.add_argument("--gazing-ratio-tile", default=None)
    parser.add_argument("--task-loss-requirement-tile", type=float, default=None)
    parser.add_argument("--gazing-ratio-thumbnail", default=None)
    parser.add_argument("--task-loss-requirement-thumbnail", default=None)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=None)
    parser.add_argument("--max-batch-size-siglip", type=int, default=None)
    parser.add_argument("--use-fast", choices=["true", "false"], default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    cfg = load_config(args.config)
    output_dir = Path(str(_cli_or_config(args.output_dir, cfg, "output.output_dir", "outputs/hlvid_nvila_hd_eval"))).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    records_info = load_records(
        dataset_path=_cli_or_config(args.dataset_path, cfg, "dataset.path", None),
        dataset_name=_cli_or_config(args.dataset_name, cfg, "dataset.name", "bfshi/HLVid"),
        dataset_split=_cli_or_config(args.dataset_split, cfg, "dataset.split", None),
        start_index=int(_cli_or_config(args.start_index, cfg, "dataset.start_index", 0)),
        max_samples=_optional_int(_cli_or_config(args.max_samples, cfg, "dataset.max_samples", None)),
    )

    eval_cfg = build_eval_config(args, cfg)
    allow_real = bool(args.allow_real_model_loading or nested_get(cfg, "runtime.allow_real_model_loading", False))
    dry_run = bool(args.dry_run or not allow_real)
    processor = None
    model = None
    load_status: dict[str, Any] = {
        "status": "not_loaded",
        "reason": "dry-run; pass --allow-real-model-loading to load NVILA-HD-Video",
    }
    if not dry_run:
        processor, model, load_status = load_processor_and_model(eval_cfg)
        if load_status["status"] != "real":
            summary = build_summary(
                cfg=cfg,
                eval_cfg=eval_cfg,
                records_info=records_info,
                predictions=[],
                load_status=load_status,
                started_at=start,
                dry_run=False,
                output_dir=output_dir,
            )
            write_outputs(output_dir, predictions=[], summary=summary)
            raise RuntimeError(load_status["reason"])

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records_info["records"]):
        prepared = prepare_record(
            record,
            index=index + int(records_info["start_index"]),
            video_root=_cli_or_config(args.video_root, cfg, "dataset.video_root", None),
            cfg=cfg,
        )
        if dry_run:
            predictions.append(
                {
                    **prepared,
                    "status": "dry_run",
                    "prediction_text": None,
                    "prediction_choice": None,
                    "correct": None,
                    "reason": "model was not loaded",
                    "latency_ms": None,
                    "latency_breakdown_ms": {},
                    "token_counts": {},
                    "memory": {},
                }
            )
            continue
        try:
            sample_start = time.perf_counter()
            generation = generate_one(
                processor=processor,
                model=model,
                video_path=str(prepared["video_path"]),
                prompt=str(prepared["prompt"]),
                max_new_tokens=int(eval_cfg["max_new_tokens"]),
            )
            sample_latency_ms = (time.perf_counter() - sample_start) * 1000
            prediction_text = str(generation["prediction_text"])
            prediction_choice = extract_choice(prediction_text, prepared.get("options"))
            target_choice = prepared.get("target_choice")
            predictions.append(
                {
                    **prepared,
                    "status": "real",
                    "prediction_text": prediction_text,
                    "prediction_choice": prediction_choice,
                    "correct": bool(prediction_choice == target_choice) if target_choice else None,
                    "reason": None,
                    "latency_ms": sample_latency_ms,
                    "latency_breakdown_ms": generation["latency_breakdown_ms"],
                    "token_counts": generation["token_counts"],
                    "memory": generation["memory"],
                }
            )
        except Exception as exc:  # pragma: no cover - real checkpoint behavior is environment-dependent.
            predictions.append(
                {
                    **prepared,
                    "status": "failed",
                    "prediction_text": None,
                    "prediction_choice": None,
                    "correct": None,
                    "reason": str(exc),
                    "latency_ms": None,
                    "latency_breakdown_ms": {},
                    "token_counts": {},
                    "memory": {},
                }
            )
            if args.fail_fast:
                break

    summary = build_summary(
        cfg=cfg,
        eval_cfg=eval_cfg,
        records_info=records_info,
        predictions=predictions,
        load_status=load_status,
        started_at=start,
        dry_run=dry_run,
        output_dir=output_dir,
    )
    write_outputs(output_dir, predictions=predictions, summary=summary)
    return summary


def build_eval_config(args: argparse.Namespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    processor_kwargs = dict(nested_get(cfg, "model.processor_from_pretrained_kwargs", {}) or {})
    processor_kwargs["num_video_frames"] = int(
        _cli_or_config(args.num_video_frames, cfg, "model.processor_from_pretrained_kwargs.num_video_frames", 128)
    )
    processor_kwargs["num_video_frames_thumbnail"] = int(
        _cli_or_config(
            args.num_video_frames_thumbnail,
            cfg,
            "model.processor_from_pretrained_kwargs.num_video_frames_thumbnail",
            64,
        )
    )
    processor_kwargs["max_tiles_video"] = int(
        _cli_or_config(args.max_tiles_video, cfg, "model.processor_from_pretrained_kwargs.max_tiles_video", 48)
    )
    processor_kwargs["gazing_ratio_tile"] = parse_optional_value(
        _cli_or_config(args.gazing_ratio_tile, cfg, "model.processor_from_pretrained_kwargs.gazing_ratio_tile", [0.2] + [0.06] * 15)
    )
    processor_kwargs["task_loss_requirement_tile"] = parse_optional_value(
        _cli_or_config(
            args.task_loss_requirement_tile,
            cfg,
            "model.processor_from_pretrained_kwargs.task_loss_requirement_tile",
            0.6,
        )
    )
    processor_kwargs["gazing_ratio_thumbnail"] = parse_optional_value(
        _cli_or_config(args.gazing_ratio_thumbnail, cfg, "model.processor_from_pretrained_kwargs.gazing_ratio_thumbnail", 1)
    )
    processor_kwargs["task_loss_requirement_thumbnail"] = parse_optional_value(
        _cli_or_config(
            args.task_loss_requirement_thumbnail,
            cfg,
            "model.processor_from_pretrained_kwargs.task_loss_requirement_thumbnail",
            None,
        )
    )
    processor_kwargs["max_batch_size_autogaze"] = int(
        _cli_or_config(
            args.max_batch_size_autogaze,
            cfg,
            "model.processor_from_pretrained_kwargs.max_batch_size_autogaze",
            16,
        )
    )
    use_fast = _cli_or_config(args.use_fast, cfg, "model.processor_from_pretrained_kwargs.use_fast", False)
    processor_kwargs["use_fast"] = _parse_bool(use_fast)

    local_files_only = _cli_or_config(args.local_files_only, cfg, "model.local_files_only", True)
    trust_remote_code = _cli_or_config(args.trust_remote_code, cfg, "model.trust_remote_code", True)
    model_path = str(_cli_or_config(args.model_path, cfg, "model.model_path", "weights/NVILA-8B-HD-Video"))
    processor_path = str(_cli_or_config(args.processor_path, cfg, "model.processor_path", model_path))
    max_batch_size_siglip = int(_cli_or_config(args.max_batch_size_siglip, cfg, "model.from_pretrained_kwargs.max_batch_size_siglip", 32))
    max_new_tokens = int(_cli_or_config(args.max_new_tokens, cfg, "generation.max_new_tokens", 4))
    dtype = _cli_or_config(args.dtype, cfg, "runtime.dtype", None)
    device_map = _cli_or_config(args.device_map, cfg, "runtime.device_map", "auto")
    return {
        "model_path": _model_reference(model_path),
        "processor_path": _model_reference(processor_path),
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "processor_kwargs": processor_kwargs,
        "model_kwargs": {
            "trust_remote_code": bool(trust_remote_code),
            "device_map": device_map,
            "local_files_only": bool(local_files_only),
            "max_batch_size_siglip": max_batch_size_siglip,
            **_dtype_kwargs(dtype),
        },
        "max_new_tokens": max_new_tokens,
        "prompt_format": str(nested_get(cfg, "dataset.prompt_format", "hlvid_mcq")),
    }


def load_processor_and_model(eval_cfg: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            eval_cfg["processor_path"],
            trust_remote_code=bool(eval_cfg["trust_remote_code"]),
            local_files_only=bool(eval_cfg["local_files_only"]),
            **dict(eval_cfg["processor_kwargs"]),
        )
        model = AutoModel.from_pretrained(eval_cfg["model_path"], **dict(eval_cfg["model_kwargs"]))
        model.eval()
        return (
            processor,
            model,
            {
                "status": "real",
                "reason": None,
                "processor_path": eval_cfg["processor_path"],
                "model_path": eval_cfg["model_path"],
                "processor_kwargs": eval_cfg["processor_kwargs"],
                "model_kwargs": _json_safe_model_kwargs(eval_cfg["model_kwargs"]),
            },
        )
    except Exception as exc:
        return None, None, {"status": "blocked", "reason": f"NVILA-HD processor/model loading failed: {exc}"}


def generate_one(*, processor: Any, model: Any, video_path: str, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    video_token = getattr(getattr(processor, "tokenizer", None), "video_token", "<video>")
    full_prompt = f"{video_token}\n\n{prompt}"
    device = _model_device(model)
    _reset_cuda_peak_memory(device)
    memory_snapshots = [capture_memory_snapshot("before_processor", device=str(device))]

    processor_start = time.perf_counter()
    inputs = processor(text=full_prompt, videos=video_path, return_tensors="pt")
    processor_latency_ms = (time.perf_counter() - processor_start) * 1000
    memory_snapshots.append(capture_memory_snapshot("after_processor", device=str(device)))

    input_token_count = _input_token_count(inputs)
    video_placeholder_count = _video_placeholder_count(inputs, processor)
    input_tensor_bytes = _mapping_tensor_bytes(inputs) if isinstance(inputs, Mapping) else None

    move_start = time.perf_counter()
    if isinstance(inputs, Mapping):
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    input_move_latency_ms = (time.perf_counter() - move_start) * 1000
    memory_snapshots.append(capture_memory_snapshot("after_input_move", device=str(device)))

    import torch

    generate_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    model_generate_latency_ms = (time.perf_counter() - generate_start) * 1000
    memory_snapshots.append(capture_memory_snapshot("after_generate", device=str(device)))

    decode_start = time.perf_counter()
    input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
    decode_input = outputs[:, input_ids.shape[1] :] if hasattr(outputs, "shape") and hasattr(input_ids, "shape") else outputs
    decoded = processor.batch_decode(decode_input, skip_special_tokens=True)
    decode_latency_ms = (time.perf_counter() - decode_start) * 1000
    text = str(decoded[0]).strip() if decoded else ""
    output_new_token_count = _tensor_sequence_length(decode_input)
    output_total_token_count = _tensor_sequence_length(outputs)
    return {
        "prediction_text": text,
        "latency_breakdown_ms": {
            "processor_latency_ms": processor_latency_ms,
            "input_move_latency_ms": input_move_latency_ms,
            "model_generate_latency_ms": model_generate_latency_ms,
            "decode_latency_ms": decode_latency_ms,
            "total_generation_latency_ms": processor_latency_ms
            + input_move_latency_ms
            + model_generate_latency_ms
            + decode_latency_ms,
        },
        "token_counts": {
            "input_token_count": input_token_count,
            "video_placeholder_token_count": video_placeholder_count,
            "output_new_token_count": output_new_token_count,
            "output_total_token_count": output_total_token_count,
            "max_new_tokens": int(max_new_tokens),
        },
        "memory": {
            "snapshots": memory_snapshots,
            "input_tensor_bytes": input_tensor_bytes,
            "input_tensor_mib": _bytes_to_mib(input_tensor_bytes),
            "process_peak_rss_mib": _last_snapshot_value(memory_snapshots, "process_peak_rss_mib"),
            "cuda_max_memory_allocated_mib": _max_snapshot_value(memory_snapshots, "cuda_max_memory_allocated_mib"),
            "cuda_max_memory_reserved_mib": _max_snapshot_value(memory_snapshots, "cuda_max_memory_reserved_mib"),
        },
    }


def load_records(
    *,
    dataset_path: str | None,
    dataset_name: str | None,
    dataset_split: str | None,
    start_index: int,
    max_samples: int | None,
) -> dict[str, Any]:
    if dataset_path:
        records = _load_local_records(resolve_path(dataset_path))
        source = str(dataset_path)
        split = None
    else:
        if not dataset_name:
            raise ValueError("Either --dataset-path or --dataset-name is required")
        records, split = _load_hf_records(dataset_name, dataset_split)
        source = dataset_name
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    sliced = records[start_index:]
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("--max-samples must be >= 0")
        sliced = sliced[:max_samples]
    return {
        "source": source,
        "split": split,
        "start_index": start_index,
        "max_samples": max_samples,
        "total_available_records": len(records),
        "records": sliced,
    }


def prepare_record(record: Mapping[str, Any], *, index: int, video_root: str | None, cfg: Mapping[str, Any]) -> dict[str, Any]:
    question_id_field = str(nested_get(cfg, "dataset.fields.question_id", "question_id"))
    video_field = str(nested_get(cfg, "dataset.fields.video_path", "video_path"))
    question_field = str(nested_get(cfg, "dataset.fields.question", "question"))
    answer_field = str(nested_get(cfg, "dataset.fields.answer", "answer"))
    question = str(record.get(question_field) or record.get("query") or "")
    options = extract_options(record)
    prompt = build_hlvid_prompt(question, options)
    answer_raw = record.get(answer_field)
    video_value = record.get(video_field)
    if video_value is None:
        video_value = record.get("video") or record.get("video_url") or record.get("url")
    return {
        "index": index,
        "question_id": record.get(question_id_field, index),
        "video_path": resolve_video_path(video_value, video_root=video_root),
        "question": question,
        "options": options,
        "target_raw": answer_raw,
        "target_choice": normalize_choice(answer_raw, options),
        "prompt": prompt,
    }


def build_hlvid_prompt(question: str, options: Mapping[str, str] | None = None) -> str:
    lines = [f"Question: {question.strip()}"]
    if options:
        for key in CHOICES:
            if key in options:
                lines.append(f"{key}. {options[key]}")
    lines.append("Please answer directly with the letter of the correct answer.")
    return "\n".join(lines)


def extract_options(record: Mapping[str, Any]) -> dict[str, str] | None:
    direct = record.get("options") or record.get("choices")
    options: dict[str, str] = {}
    if isinstance(direct, Mapping):
        for key, value in direct.items():
            letter = str(key).strip().upper()[:1]
            if letter in CHOICES:
                options[letter] = str(value)
    elif isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        for letter, value in zip(CHOICES, direct):
            options[letter] = str(value)
    for letter in CHOICES:
        for key in (letter, letter.lower(), f"option_{letter.lower()}", f"option_{letter}"):
            if key in record:
                options[letter] = str(record[key])
                break
    return options or None


def normalize_choice(value: Any, options: Mapping[str, str] | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"\b([ABCD])\b", text.upper())
    if match:
        return match.group(1)
    if options:
        normalized = _normalize_text(text)
        for letter, option_text in options.items():
            if normalized and normalized == _normalize_text(option_text):
                return letter
    return None


def extract_choice(response: str | None, options: Mapping[str, str] | None = None) -> str | None:
    if not response:
        return None
    text = response.strip()
    patterns = [
        r"^\s*([ABCD])(?:[\.\):\s]|$)",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if options:
        normalized = _normalize_text(text)
        for letter, option_text in options.items():
            if _normalize_text(option_text) and _normalize_text(option_text) in normalized:
                return letter
    return None


def build_summary(
    *,
    cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    records_info: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    load_status: Mapping[str, Any],
    started_at: float,
    dry_run: bool,
    output_dir: Path,
) -> dict[str, Any]:
    evaluated = [item for item in predictions if item.get("target_choice") and item.get("prediction_choice")]
    correct = [item for item in evaluated if item.get("correct") is True]
    failed = [item for item in predictions if item.get("status") == "failed"]
    invalid_target = [item for item in predictions if item.get("target_choice") is None]
    status = "dry_run" if dry_run else ("partial" if failed else "pass")
    accuracy = len(correct) / len(evaluated) if evaluated else None
    real_latencies = [
        float(item["latency_ms"])
        for item in predictions
        if item.get("latency_ms") is not None and item.get("status") == "real"
    ]
    latency_summary = {
        "mean_ms": sum(real_latencies) / len(real_latencies) if real_latencies else None,
        "min_ms": min(real_latencies) if real_latencies else None,
        "max_ms": max(real_latencies) if real_latencies else None,
    }
    token_summary = _summarize_nested_numeric(predictions, "token_counts")
    latency_breakdown_summary = _summarize_nested_numeric(predictions, "latency_breakdown_ms")
    memory_summary = _summarize_memory(predictions)
    return {
        "status": status,
        "task": "hlvid_multiple_choice_video_qa",
        "dataset": {
            "source": records_info["source"],
            "split": records_info.get("split"),
            "total_available_records": records_info["total_available_records"],
            "selected_records": len(records_info["records"]),
            "start_index": records_info["start_index"],
            "max_samples": records_info["max_samples"],
        },
        "model": {
            "name": "NVILA-HD-Video",
            "model_path": eval_cfg["model_path"],
            "processor_path": eval_cfg["processor_path"],
            "load_status": load_status,
            "processor_setup_source": "docs/nvila-hd-video-readme.md",
        },
        "processor_kwargs": eval_cfg["processor_kwargs"],
        "official_high_resolution_processing": {
            "source": "weights/NVILA-8B-HD-Video/processing_nvila.py",
            "video_input": "source_video_path",
            "frame_sampling": "uniform num_video_frames sampled by NVILA processor",
            "spatial_processing": "dynamic aspect-ratio tile grid bounded by max_tiles_video; one tile is processor image_size",
            "thumbnail_processing": "whole-frame thumbnails bounded by num_video_frames_thumbnail",
            "autogaze_scope": "tiles and thumbnails through official processor-owned gazing_info",
            "not_replaced_by_poc_scaling": True,
        },
        "generation": {"max_new_tokens": eval_cfg["max_new_tokens"]},
        "metrics": {
            "num_predictions": len(predictions),
            "num_evaluated": len(evaluated),
            "num_correct": len(correct),
            "accuracy": accuracy,
            "num_failed": len(failed),
            "num_invalid_ground_truth": len(invalid_target),
            "mean_sample_latency_ms": latency_summary["mean_ms"],
            "min_sample_latency_ms": latency_summary["min_ms"],
            "max_sample_latency_ms": latency_summary["max_ms"],
            "mean_input_token_count": token_summary.get("input_token_count_mean"),
            "mean_video_placeholder_token_count": token_summary.get("video_placeholder_token_count_mean"),
            "mean_output_new_token_count": token_summary.get("output_new_token_count_mean"),
            "mean_processor_latency_ms": latency_breakdown_summary.get("processor_latency_ms_mean"),
            "mean_model_generate_latency_ms": latency_breakdown_summary.get("model_generate_latency_ms_mean"),
            "peak_process_rss_mib": memory_summary.get("peak_process_rss_mib"),
            "peak_cuda_memory_allocated_mib": memory_summary.get("peak_cuda_memory_allocated_mib"),
            "peak_cuda_memory_reserved_mib": memory_summary.get("peak_cuda_memory_reserved_mib"),
        },
        "latency_report": {
            "sample_latency_ms": latency_summary,
            "breakdown_ms": latency_breakdown_summary,
        },
        "token_consumption_report": token_summary,
        "memory_report": memory_summary,
        "notes": [
            "HLVid is treated as multiple-choice video QA.",
            "The prompt asks for a single answer letter and scoring uses exact option-letter match.",
            "No direct visual-token injection is used; NVILA owns video processing through its official processor.",
        ],
        "artifacts": {
            "predictions_json": str(output_dir / "predictions" / "hlvid_predictions.json"),
            "predictions_jsonl": str(output_dir / "predictions" / "hlvid_predictions.jsonl"),
            "predictions_csv": str(output_dir / "predictions" / "hlvid_predictions.csv"),
            "metrics": str(output_dir / "logs" / "metrics.json"),
            "metrics_csv": str(output_dir / "logs" / "metrics.csv"),
            "summary": str(output_dir / "logs" / "poc_summary.json"),
            "report": str(output_dir / "logs" / "hlvid_report.md"),
            "run_report_json": str(output_dir / "logs" / "run_report.json"),
        },
        "config_path": cfg.get("_config_path"),
        "wall_clock_latency_ms": (time.perf_counter() - started_at) * 1000,
    }


def write_outputs(output_dir: Path, *, predictions: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    predictions_dir = output_dir / "predictions"
    logs_dir = output_dir / "logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(predictions_dir / "hlvid_predictions.json", {"predictions": predictions})
    with (predictions_dir / "hlvid_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    _write_predictions_csv(predictions_dir / "hlvid_predictions.csv", predictions)
    write_json(logs_dir / "poc_summary.json", summary)
    write_json(logs_dir / "metrics.json", summary["metrics"])
    write_json(logs_dir / "run_report.json", _run_report(summary))
    _write_metrics_csv(logs_dir / "metrics.csv", summary["metrics"])
    _write_markdown_report(logs_dir / "hlvid_report.md", summary)


def _write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "index",
        "question_id",
        "video_path",
        "target_choice",
        "prediction_choice",
        "correct",
        "status",
        "latency_ms",
        "processor_latency_ms",
        "model_generate_latency_ms",
        "input_token_count",
        "video_placeholder_token_count",
        "output_new_token_count",
        "input_tensor_mib",
        "process_peak_rss_mib",
        "cuda_max_memory_allocated_mib",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            row = {field: _csv_value(item.get(field)) for field in fields}
            latency = item.get("latency_breakdown_ms") if isinstance(item.get("latency_breakdown_ms"), Mapping) else {}
            tokens = item.get("token_counts") if isinstance(item.get("token_counts"), Mapping) else {}
            memory = item.get("memory") if isinstance(item.get("memory"), Mapping) else {}
            row["processor_latency_ms"] = _csv_value(latency.get("processor_latency_ms"))
            row["model_generate_latency_ms"] = _csv_value(latency.get("model_generate_latency_ms"))
            row["input_token_count"] = _csv_value(tokens.get("input_token_count"))
            row["video_placeholder_token_count"] = _csv_value(tokens.get("video_placeholder_token_count"))
            row["output_new_token_count"] = _csv_value(tokens.get("output_new_token_count"))
            row["input_tensor_mib"] = _csv_value(memory.get("input_tensor_mib"))
            row["process_peak_rss_mib"] = _csv_value(memory.get("process_peak_rss_mib"))
            row["cuda_max_memory_allocated_mib"] = _csv_value(memory.get("cuda_max_memory_allocated_mib"))
            writer.writerow(row)


def _write_metrics_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": _csv_value(value)})


def _write_markdown_report(path: Path, summary: Mapping[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    dataset = summary.get("dataset", {})
    model = summary.get("model", {})
    high_res = summary.get("official_high_resolution_processing", {})
    lines = [
        "# HLVid NVILA-HD Evaluation Report",
        "",
        f"- status: `{summary.get('status')}`",
        f"- dataset_source: `{dataset.get('source')}`",
        f"- selected_records: `{dataset.get('selected_records')}`",
        f"- model_path: `{model.get('model_path')}`",
        f"- processor_path: `{model.get('processor_path')}`",
        f"- accuracy: `{metrics.get('accuracy')}`",
        f"- num_evaluated: `{metrics.get('num_evaluated')}`",
        f"- num_failed: `{metrics.get('num_failed')}`",
        "",
        "## Official High-Resolution Path",
        "",
        f"- video_input: `{high_res.get('video_input')}`",
        f"- frame_sampling: {high_res.get('frame_sampling')}",
        f"- spatial_processing: {high_res.get('spatial_processing')}",
        f"- thumbnail_processing: {high_res.get('thumbnail_processing')}",
        f"- autogaze_scope: {high_res.get('autogaze_scope')}",
        f"- not_replaced_by_poc_scaling: `{high_res.get('not_replaced_by_poc_scaling')}`",
        "",
        "## Processor Settings",
        "",
        "```json",
        json.dumps(summary.get("processor_kwargs", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Latency / Memory / Token Report",
        "",
        "```json",
        json.dumps(
            {
                "latency_report": summary.get("latency_report", {}),
                "token_consumption_report": summary.get("token_consumption_report", {}),
                "memory_report": summary.get("memory_report", {}),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        for key, value in artifacts.items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": summary.get("status"),
        "task": summary.get("task"),
        "dataset": summary.get("dataset"),
        "model": summary.get("model"),
        "processor_kwargs": summary.get("processor_kwargs"),
        "official_high_resolution_processing": summary.get("official_high_resolution_processing"),
        "metrics": summary.get("metrics"),
        "latency_report": summary.get("latency_report"),
        "memory_report": summary.get("memory_report"),
        "token_consumption_report": summary.get("token_consumption_report"),
        "artifacts": summary.get("artifacts"),
    }


def _load_local_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"local dataset file does not exist: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data]
        if isinstance(data, Mapping):
            for key in ("records", "data", "examples"):
                if isinstance(data.get(key), list):
                    return [dict(item) for item in data[key]]
        raise ValueError(f"JSON dataset must be a list or contain records/data/examples: {path}")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported local dataset format: {path.suffix}")


def _load_hf_records(dataset_name: str, dataset_split: str | None) -> tuple[list[dict[str, Any]], str | None]:
    try:
        from datasets import DatasetDict, load_dataset
    except Exception as exc:
        raise RuntimeError("datasets is required for --dataset-name; use --dataset-path for offline/local evaluation") from exc
    loaded = load_dataset(dataset_name, split=dataset_split) if dataset_split else load_dataset(dataset_name)
    if dataset_split:
        return [dict(item) for item in loaded], dataset_split
    if isinstance(loaded, DatasetDict):
        for preferred in ("validation", "val", "test", "eval", "train"):
            if preferred in loaded:
                return [dict(item) for item in loaded[preferred]], preferred
        first_split = next(iter(loaded.keys()))
        return [dict(item) for item in loaded[first_split]], first_split
    return [dict(item) for item in loaded], None


def resolve_video_path(value: Any, *, video_root: str | None) -> str:
    if value is None:
        raise ValueError("record is missing video_path")
    if isinstance(value, Mapping):
        value = value.get("path") or value.get("url") or value.get("filename")
        if value is None:
            raise ValueError("video mapping must contain path, url, or filename")
    text = str(value)
    if _is_url(text) or Path(text).is_absolute() or not video_root:
        return text
    if _is_url(str(video_root)):
        return f"{str(video_root).rstrip('/')}/{text.lstrip('/')}"
    return str(resolve_path(Path(str(video_root)) / text))


def parse_optional_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"none", "null"}:
            return None
        if text.startswith("["):
            return json.loads(text)
        if "," in text:
            return [float(item.strip()) for item in text.split(",") if item.strip()]
        try:
            return float(text)
        except ValueError:
            return value
    return value


def _cli_or_config(cli_value: Any, cfg: Mapping[str, Any], path: str, default: Any) -> Any:
    return cli_value if cli_value is not None else nested_get(cfg, path, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _model_reference(value: str) -> str:
    if _is_url(value) or "/" in value and not value.startswith(("weights/", "./", "../", "/", "~")):
        return value
    return str(resolve_path(value))


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    if dtype in {None, "", "auto"}:
        return {}
    import torch

    return {"dtype": getattr(torch, str(dtype))}


def _json_safe_model_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in kwargs.items():
        safe[key] = str(value) if key == "dtype" else value
    return safe


def _reset_cuda_peak_memory(device: Any) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _input_token_count(inputs: Any) -> int | None:
    if not isinstance(inputs, Mapping):
        return None
    input_ids = inputs.get("input_ids")
    return _tensor_sequence_length(input_ids)


def _video_placeholder_count(inputs: Any, processor: Any) -> int | None:
    if not isinstance(inputs, Mapping):
        return None
    input_ids = inputs.get("input_ids")
    if input_ids is None or not hasattr(input_ids, "__eq__"):
        return None
    token_id = _video_token_id(processor)
    if token_id is None:
        return None
    try:
        return int((input_ids == int(token_id)).sum().item())
    except Exception:
        return None


def _video_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    direct = getattr(tokenizer, "video_token_id", None)
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            pass
    token = getattr(tokenizer, "video_token", None)
    if token is None:
        return None
    try:
        converted = tokenizer.convert_tokens_to_ids(token)
        return int(converted)
    except Exception:
        return None


def _tensor_sequence_length(value: Any) -> int | None:
    if value is None or not hasattr(value, "shape"):
        return None
    try:
        shape = list(value.shape)
    except Exception:
        return None
    if not shape:
        return None
    return int(shape[-1])


def _mapping_tensor_bytes(inputs: Mapping[str, Any]) -> int:
    total = 0
    for value in inputs.values():
        total += _tensor_bytes(value)
    return total


def _tensor_bytes(value: Any) -> int:
    if not hasattr(value, "numel") or not hasattr(value, "element_size"):
        return 0
    try:
        return int(value.numel()) * int(value.element_size())
    except Exception:
        return 0


def _bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _last_snapshot_value(snapshots: Sequence[Mapping[str, Any]], key: str) -> Any:
    for snapshot in reversed(list(snapshots)):
        if snapshot.get(key) is not None:
            return snapshot.get(key)
    return None


def _max_snapshot_value(snapshots: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(snapshot[key]) for snapshot in snapshots if snapshot.get(key) is not None]
    return max(values) if values else None


def _summarize_nested_numeric(predictions: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = {}
    for item in predictions:
        nested = item.get(field)
        if not isinstance(nested, Mapping):
            continue
        for key, value in nested.items():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(str(key), []).append(float(value))
    summary: dict[str, Any] = {}
    for key, values in values_by_key.items():
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    return summary


def _summarize_memory(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    process_peak = []
    cuda_allocated = []
    cuda_reserved = []
    input_tensor_mib = []
    for item in predictions:
        memory = item.get("memory")
        if not isinstance(memory, Mapping):
            continue
        _append_numeric(process_peak, memory.get("process_peak_rss_mib"))
        _append_numeric(cuda_allocated, memory.get("cuda_max_memory_allocated_mib"))
        _append_numeric(cuda_reserved, memory.get("cuda_max_memory_reserved_mib"))
        _append_numeric(input_tensor_mib, memory.get("input_tensor_mib"))
    return {
        "peak_process_rss_mib": max(process_peak) if process_peak else None,
        "peak_cuda_memory_allocated_mib": max(cuda_allocated) if cuda_allocated else None,
        "peak_cuda_memory_reserved_mib": max(cuda_reserved) if cuda_reserved else None,
        "mean_input_tensor_mib": sum(input_tensor_mib) / len(input_tensor_mib) if input_tensor_mib else None,
        "max_input_tensor_mib": max(input_tensor_mib) if input_tensor_mib else None,
    }


def _append_numeric(target: list[float], value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        target.append(float(value))


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "s3://", "gs://"))


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        print(f"HLVid evaluation failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        metrics = summary["metrics"]
        print(f"Wrote HLVid evaluation outputs: {summary['artifacts']['summary']}")
        if metrics["accuracy"] is None:
            print(f"Status: {summary['status']} | evaluated={metrics['num_evaluated']} | accuracy=N/A")
        else:
            print(f"Status: {summary['status']} | evaluated={metrics['num_evaluated']} | accuracy={metrics['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
