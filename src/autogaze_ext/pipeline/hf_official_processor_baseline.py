from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from autogaze_ext.models.huggingface import HFModelLoader, HFProcessorLoader
from autogaze_ext.pipeline.hf_benchmark import _merge_non_null, _node_to_dict
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.profiling import MemoryTracker, synchronize_if_cuda
from autogaze_ext.utils import HFLoadConfig, hf_offline_mode, redacted_hf_config


SUPPORTED_VIDEO_INPUTS = {"dummy", "path"}
SUPPORTED_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class HFOfficialProcessorBaselineReport:
    experiment_id: str
    integration_mode: str
    autogaze_token_injection: bool
    model_id: str | None
    revision: str | None
    processor_id: str | None
    dataset_input_source: str
    video_path: str | None
    input_frame_count: int
    resolution: int
    query_text: str
    generated_answer: str | None
    generation_status: str
    skipped_reason: str | None
    latency_ms: float | None
    peak_vram_mb: float | str
    device: str
    effective_device: str
    dtype: str
    local_files_only: bool
    offline: bool
    trust_remote_code: bool
    processor_loaded: bool
    model_loaded: bool
    output_dir: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_hf_config(cfg: DictConfig) -> HFLoadConfig:
    runtime = _node_to_dict(cfg.get("runtime", {}).get("huggingface", {}))
    model = _node_to_dict(cfg.get("model", {}).get("huggingface", {}))
    merged = _merge_non_null(runtime, model)
    return HFLoadConfig.from_mapping(merged)


def _benchmark_cfg(cfg: DictConfig) -> dict[str, Any]:
    return _node_to_dict(cfg.get("benchmark", {}).get("huggingface", {}))


def _select_device(requested: str) -> str:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device.type == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def _make_dummy_frames(num_frames: int, resolution: int) -> list[Image.Image]:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if resolution <= 0:
        raise ValueError("resolution must be > 0")
    frames: list[Image.Image] = []
    for index in range(num_frames):
        color = (
            int((index * 53) % 255),
            int((80 + index * 37) % 255),
            int((160 + index * 19) % 255),
        )
        frames.append(Image.new("RGB", (resolution, resolution), color=color))
    return frames


def _resolve_video_input(video: str, video_path: str | None, num_frames: int, resolution: int) -> tuple[str, str | list[Image.Image] | None]:
    if video_path:
        path = Path(video_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Video path does not exist: {path}")
        return "local_video", str(path)

    if video not in SUPPORTED_VIDEO_INPUTS:
        raise ValueError(f"Unsupported video input source: {video}")
    if video == "dummy":
        return "local_dummy_video", _make_dummy_frames(num_frames, resolution)
    raise ValueError("--video path requires --video-path")


def _format_prompt(query_text: str, prompt_template: str | None, processor: Any | None = None) -> str:
    template = prompt_template or "{prompt}"
    video_token = ""
    tokenizer = getattr(processor, "tokenizer", None)
    for attr in ("video_token", "video_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str):
            video_token = value
            break
    return template.format(prompt=query_text, query=query_text, video_token=video_token).strip()


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return inputs


def _prepare_with_official_processor(processor: Any, prompt: str, video_input: str | list[Image.Image] | None) -> tuple[Any, dict[str, Any]]:
    attempts: list[tuple[str, dict[str, Any]]] = []
    if isinstance(video_input, str):
        attempts.append(("videos_path", {"text": prompt, "videos": video_input, "return_tensors": "pt"}))
    elif isinstance(video_input, list):
        attempts.append(("images_frames", {"text": prompt, "images": video_input, "return_tensors": "pt"}))
        attempts.append(("videos_frames", {"text": prompt, "videos": video_input, "return_tensors": "pt"}))

    errors: list[str] = []
    for attempt_name, kwargs in attempts:
        try:
            return processor(**kwargs), {"processor_attempt": attempt_name}
        except TypeError as exc:
            errors.append(f"{attempt_name}: {exc}")
        except ValueError as exc:
            errors.append(f"{attempt_name}: {exc}")

    raise RuntimeError(
        "Official processor did not accept the provided video input. "
        "No text-only fallback was used, so query/video input is not silently ignored. "
        f"Attempts: {errors}"
    )


def _decode_generated_answer(processor: Any, outputs: Any, inputs: Any) -> str:
    generated = outputs
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
    if isinstance(outputs, torch.Tensor) and isinstance(input_ids, torch.Tensor):
        if outputs.ndim == 2 and input_ids.ndim == 2 and outputs.shape[-1] > input_ids.shape[-1]:
            generated = outputs[:, input_ids.shape[-1] :]

    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0] if decoded else "")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return str(decoded[0] if decoded else "")

    if isinstance(generated, torch.Tensor):
        return str(generated.detach().cpu().tolist())
    return str(generated)


def _write_report(report: HFOfficialProcessorBaselineReport, output_dir: Path) -> Path:
    logs_dir = output_dir / "logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    report_path = logs_dir / "hf_official_processor_baseline.json"
    data = report.to_dict()
    report_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if report.generated_answer is not None:
        prediction_path = predictions_dir / "answer.json"
        prediction_path.write_text(
            json.dumps(
                {
                    "generated_answer": report.generated_answer,
                    "query_text": report.query_text,
                    "generation_status": report.generation_status,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return report_path


def run_hf_official_processor_baseline(
    cfg: DictConfig,
    *,
    output_dir: str | Path | None = None,
    dry_run: bool | None = None,
    video: str | None = None,
    video_path: str | None = None,
    query_text: str | None = None,
    num_frames: int | None = None,
    resolution: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    max_new_tokens: int | None = None,
    allow_download: bool = False,
    model_loader: HFModelLoader | None = None,
    processor_loader: HFProcessorLoader | None = None,
) -> Path:
    bench_cfg = _benchmark_cfg(cfg)
    model_config = _model_hf_config(cfg)

    dry = bool(bench_cfg.get("dry_run", True) if dry_run is None else dry_run)
    model_id = model_config.model_id
    processor_id = str(bench_cfg.get("processor_id") or model_id) if model_id or bench_cfg.get("processor_id") else None
    revision = model_config.revision
    requested_device = str(device or cfg.get("runtime", {}).get("device", {}).get("type", "cpu"))
    effective_device = _select_device(requested_device)
    dtype_name = str(dtype or cfg.get("runtime", {}).get("precision", {}).get("dtype", "float32"))
    if dtype_name not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    resolved_num_frames = int(num_frames or bench_cfg.get("num_frames", 2))
    resolved_resolution = int(resolution or bench_cfg.get("resolution", 224))
    resolved_query = str(query_text or bench_cfg.get("query_text", "Describe the video."))
    resolved_video = str(video or bench_cfg.get("input_source", "dummy"))
    resolved_max_new_tokens = int(max_new_tokens or bench_cfg.get("max_new_tokens", 1))
    prompt_template = bench_cfg.get("prompt_template")
    resolved_output_dir = Path(output_dir or bench_cfg.get("output_dir", "outputs/hf_official_processor_baseline"))

    if not allow_download:
        # Keep the default path network-silent. Some processor loaders still try
        # HEAD requests unless Hugging Face offline mode is set explicitly.
        model_config = HFLoadConfig.from_mapping({**model_config.__dict__, "local_files_only": True, "offline": True})

    input_source, resolved_video_input = _resolve_video_input(
        resolved_video,
        video_path or bench_cfg.get("video_path"),
        resolved_num_frames,
        resolved_resolution,
    )
    memory = MemoryTracker(effective_device)
    memory.reset_peak()

    processor = None
    model = None
    generated_answer: str | None = None
    generation_status = "dry_run"
    skipped_reason: str | None = "dry_run_no_model_or_processor_loaded"
    latency_ms: float | None = None
    processor_metadata: dict[str, Any] = {}

    if not dry:
        if not model_id:
            skipped_reason = "model_id_is_required_for_non_dry_run"
            generation_status = "skipped"
        elif not processor_id:
            skipped_reason = "processor_id_is_required_for_non_dry_run"
            generation_status = "skipped"
        else:
            try:
                processor_loader = processor_loader or HFProcessorLoader(model_config)
                model_loader = model_loader or HFModelLoader(model_config)
                with hf_offline_mode(model_config.offline):
                    processor = processor_loader.load_processor(processor_id, revision=revision)
                    model = model_loader.load_model(
                        model_id,
                        revision=revision,
                        device=effective_device,
                        dtype=SUPPORTED_DTYPES[dtype_name],
                    )
                if hasattr(model, "eval"):
                    model.eval()

                prompt = _format_prompt(resolved_query, prompt_template, processor)
                synchronize_if_cuda(effective_device)
                start = time.perf_counter()
                inputs, processor_metadata = _prepare_with_official_processor(processor, prompt, resolved_video_input)
                inputs = _move_inputs_to_device(inputs, effective_device)

                if hasattr(model, "generate"):
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, max_new_tokens=resolved_max_new_tokens)
                    generated_answer = _decode_generated_answer(processor, outputs, inputs)
                    generation_status = "generated"
                    skipped_reason = None
                else:
                    generation_status = "skipped"
                    skipped_reason = "model_has_no_generate_method"
                synchronize_if_cuda(effective_device)
                latency_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:  # pragma: no cover - exact HF errors vary by installed versions.
                generation_status = "failed"
                skipped_reason = f"{exc.__class__.__name__}: {exc}"

    snapshot = memory.snapshot()
    experiment_id = str(cfg.get("experiment", {}).get("id", "hf_official_processor_baseline"))
    report = HFOfficialProcessorBaselineReport(
        experiment_id=experiment_id,
        integration_mode="official_processor",
        autogaze_token_injection=False,
        model_id=model_id,
        revision=revision,
        processor_id=processor_id,
        dataset_input_source=input_source,
        video_path=str(resolved_video_input) if isinstance(resolved_video_input, str) else None,
        input_frame_count=resolved_num_frames,
        resolution=resolved_resolution,
        query_text=resolved_query,
        generated_answer=generated_answer,
        generation_status=generation_status,
        skipped_reason=skipped_reason,
        latency_ms=latency_ms,
        peak_vram_mb=snapshot.peak_vram_mb,
        device=requested_device,
        effective_device=effective_device,
        dtype=dtype_name,
        local_files_only=model_config.local_files_only or model_config.offline,
        offline=model_config.offline,
        trust_remote_code=model_config.trust_remote_code,
        processor_loaded=processor is not None,
        model_loaded=model is not None,
        output_dir=str(resolved_output_dir),
        metadata={
            "warning": "Official-processor HF baseline only; no AutoGaze token injection or compatibility claim.",
            "max_new_tokens": resolved_max_new_tokens,
            "processor_metadata": processor_metadata,
            "redacted_hf_config": redacted_hf_config(model_config),
            "allow_download": allow_download,
        },
    )
    return _write_report(report, resolved_output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hugging Face MLLM official-processor baseline")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="hf_benchmark/hf_official_processor_baseline")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--video", default=None, choices=sorted(SUPPORTED_VIDEO_INPUTS))
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--query-text", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default=None, choices=sorted(SUPPORTED_DTYPES))
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run", action="store_true", help="Load cached/local HF assets and run generation")
    parser.add_argument("--allow-download", action="store_true", help="Allow Hugging Face downloads")
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.config_name)
    cli_overrides: dict[str, Any] = {"benchmark": {"huggingface": {}}}
    model_overrides: dict[str, Any] = {"model": {"huggingface": {}}}
    if args.model_id is not None:
        model_overrides["model"]["huggingface"]["model_id"] = args.model_id
    if args.revision is not None:
        model_overrides["model"]["huggingface"]["revision"] = args.revision
    if args.allow_download:
        model_overrides["model"]["huggingface"]["local_files_only"] = False
        model_overrides["model"]["huggingface"]["offline"] = False
    if args.processor_id is not None:
        cli_overrides["benchmark"]["huggingface"]["processor_id"] = args.processor_id
    cfg = OmegaConf.merge(cfg, model_overrides, cli_overrides)

    path = run_hf_official_processor_baseline(
        cfg,
        output_dir=args.output_dir,
        dry_run=not args.run,
        video=args.video,
        video_path=args.video_path,
        query_text=args.query_text,
        num_frames=args.num_frames,
        resolution=args.resolution,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        allow_download=args.allow_download,
    )
    print(f"HF official-processor baseline report: {path}")


if __name__ == "__main__":
    main()
