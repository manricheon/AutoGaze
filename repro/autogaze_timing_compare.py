from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from repro.common import write_json


DEFAULT_AUTOGAZE_REPO = Path(os.environ.get("AUTOGAZE_REPO", "external/AutoGaze"))
DEFAULT_WEIGHTS_ROOT = Path(os.environ.get("AUTOGAZE_WEIGHTS_ROOT", "weights"))
DEFAULT_TARGET_PYTHON = Path(os.environ.get("AUTOGAZE_TIMING_PYTHON", sys.executable))
DEFAULT_VIDEO = DEFAULT_AUTOGAZE_REPO / "assets" / "example_input.mp4"
DEFAULT_OUTPUT_DIR = Path("outputs/autogaze_repro/timing_compare")


@dataclass(frozen=True)
class CompareConfig:
    python: Path = DEFAULT_TARGET_PYTHON
    workspace_root: Path = Path.cwd()
    autogaze_repo: Path = DEFAULT_AUTOGAZE_REPO
    weights_root: Path = DEFAULT_WEIGHTS_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    video: Path = DEFAULT_VIDEO
    device: str = "mps"
    dtype: str = "float32"
    frames: int = 16
    quickstart_batch_size: int = 1
    quickstart_run_siglip: bool = False
    thumbnail_frames: int = 1
    stream_chunk_frames: int = 16
    max_tiles_video: int = 1
    max_batch_size_autogaze: int = 16
    gazing_ratio: float = 0.75
    task_loss_requirement: float = 0.7
    warmup: int = 1
    repeat: int = 3
    stream_decode_strategy: str = "scan"
    stream_run_siglip: bool = False
    stream_siglip_mode: str = "gazed"
    autogaze_target_scales: str | None = None
    autogaze_target_patch_size: int | None = None
    require_mps: bool = True
    run_stream_profile: bool = True
    run_single: bool = True
    max_new_tokens: int = 1
    prompt: str = "Question: What is visible in the video? A. road B. kitchen C. beach D. office Please answer with the letter."

    @property
    def autogaze_model(self) -> Path:
        return self.weights_root / "AutoGaze"

    @property
    def siglip_model(self) -> Path:
        return self.weights_root / "siglip2-base-patch16-224"

    @property
    def nvila_model(self) -> Path:
        return self.weights_root / "NVILA-8B-HD-Video"

    @property
    def quickstart_json(self) -> Path:
        return self.output_dir / "quickstart_direct_autogaze.json"

    @property
    def quickstart_csv(self) -> Path:
        return self.output_dir / "quickstart_direct_autogaze.csv"

    @property
    def stream_json(self) -> Path:
        return self.output_dir / "current_stream_profile_autogaze.json"

    @property
    def single_json(self) -> Path:
        return self.output_dir / "current_single_autogaze.json"

    @property
    def single_summary_json(self) -> Path:
        return self.output_dir / "current_single_autogaze_summary.json"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "quickstart_vs_current_summary.json"

    @property
    def markdown_report(self) -> Path:
        return self.output_dir / "quickstart_vs_current_report.md"

    @property
    def sweep_summary_json(self) -> Path:
        return self.output_dir / "autogaze_policy_sweep_summary.json"

    @property
    def sweep_markdown_report(self) -> Path:
        return self.output_dir / "autogaze_policy_sweep_report.md"


def build_quickstart_command(config: CompareConfig) -> list[str]:
    command = [
        str(config.python),
        "-m",
        "repro.autogaze_bench",
        "--autogaze-repo",
        str(config.autogaze_repo),
        "--autogaze-model",
        str(config.autogaze_model),
        "--siglip-model",
        str(config.siglip_model),
        "--video",
        str(config.video),
        "--device",
        config.device,
        "--dtype",
        config.dtype,
        "--frames",
        str(config.frames),
        "--batch-size",
        str(config.quickstart_batch_size),
        "--gazing-ratio",
        str(config.gazing_ratio),
        "--task-loss-requirement",
        str(config.task_loss_requirement),
        "--warmup",
        str(config.warmup),
        "--repeat",
        str(config.repeat),
        "--output-json",
        str(config.quickstart_json),
        "--output-csv",
        str(config.quickstart_csv),
    ]
    if config.autogaze_target_scales:
        command.extend(["--target-scales", config.autogaze_target_scales])
    if config.autogaze_target_patch_size is not None:
        command.extend(["--target-patch-size", str(config.autogaze_target_patch_size)])
    if not config.quickstart_run_siglip:
        command.append("--skip-siglip")
    return command


def build_stream_profile_command(config: CompareConfig) -> list[str]:
    command = [
        str(config.python),
        "-m",
        "repro.nvila_runner",
        "--mode",
        "stream-profile",
        "--model-path",
        str(config.nvila_model),
        "--autogaze-repo",
        str(config.autogaze_repo),
        "--autogaze-model",
        str(config.autogaze_model),
        "--video",
        str(config.video),
        "--device",
        config.device,
        "--stream-dtype",
        config.dtype,
        "--gazing-mode",
        "autogaze",
        "--num-video-frames",
        str(config.frames),
        "--num-video-frames-thumbnail",
        str(config.thumbnail_frames),
        "--max-tiles-video",
        str(config.max_tiles_video),
        "--stream-chunk-frames",
        str(config.stream_chunk_frames),
        "--max-batch-size-autogaze",
        str(config.max_batch_size_autogaze),
        "--stream-decode-strategy",
        config.stream_decode_strategy,
        "--stream-gazing-ratio",
        str(config.gazing_ratio),
        "--task-loss-requirement-tile",
        str(config.task_loss_requirement),
        "--stream-profile-json",
        str(config.stream_json),
    ]
    if config.autogaze_target_scales:
        command.extend(["--autogaze-target-scales", config.autogaze_target_scales])
    if config.autogaze_target_patch_size is not None:
        command.extend(["--autogaze-target-patch-size", str(config.autogaze_target_patch_size)])
    if config.stream_run_siglip:
        command.extend(
            [
                "--stream-run-siglip",
                "--stream-siglip-model",
                str(config.siglip_model),
                "--stream-siglip-mode",
                config.stream_siglip_mode,
            ]
        )
    return command


def build_single_command(config: CompareConfig) -> list[str]:
    command = [
        str(config.python),
        "-m",
        "repro.nvila_runner",
        "--mode",
        "single",
        "--model-path",
        str(config.nvila_model),
        "--autogaze-repo",
        str(config.autogaze_repo),
        "--autogaze-model",
        str(config.autogaze_model),
        "--video",
        str(config.video),
        "--prompt",
        config.prompt,
        "--device",
        config.device,
        "--dtype",
        config.dtype,
        "--gazing-mode",
        "autogaze",
        "--gazing-ratio-tile",
        str(config.gazing_ratio),
        "--task-loss-requirement-tile",
        str(config.task_loss_requirement),
        "--num-video-frames",
        str(config.frames),
        "--num-video-frames-thumbnail",
        str(config.thumbnail_frames),
        "--max-tiles-video",
        str(config.max_tiles_video),
        "--max-batch-size-autogaze",
        str(config.max_batch_size_autogaze),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--measure-ttft",
        "--warmup-runs",
        str(config.warmup),
        "--repeat-runs",
        str(config.repeat),
        "--output-json",
        str(config.single_json),
        "--summary-json",
        str(config.single_summary_json),
        "--print-summary",
    ]
    if config.autogaze_target_scales:
        command.extend(["--autogaze-target-scales", config.autogaze_target_scales])
    if config.autogaze_target_patch_size is not None:
        command.extend(["--autogaze-target-patch-size", str(config.autogaze_target_patch_size)])
    return command


def build_subprocess_env(config: CompareConfig) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    pythonpath = [str(config.workspace_root), str(config.autogaze_repo)]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def format_sweep_value(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def parse_float_sweep(value: str | None, *, default: list[float]) -> list[float]:
    if not value:
        return list(default)
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    return parsed or list(default)


def build_sweep_configs(
    config: CompareConfig,
    *,
    gazing_ratios: list[float],
    task_loss_requirements: list[float],
) -> list[CompareConfig]:
    configs: list[CompareConfig] = []
    for gazing_ratio in gazing_ratios:
        for task_loss_requirement in task_loss_requirements:
            output_dir = config.output_dir / (
                f"gazing_{format_sweep_value(gazing_ratio)}__loss_{format_sweep_value(task_loss_requirement)}"
            )
            configs.append(
                replace(
                    config,
                    output_dir=output_dir,
                    gazing_ratio=float(gazing_ratio),
                    task_loss_requirement=float(task_loss_requirement),
                )
            )
    return configs


def _single_result(single_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not single_payload:
        return {}
    result = single_payload.get("result")
    return result if isinstance(result, dict) else single_payload


def build_autogaze_latency_options_summary(
    quickstart_payload: dict[str, Any],
    stream_payload: dict[str, Any] | None,
    single_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    quick_options = quickstart_payload.get("autogaze_latency_options") or {}
    stream_config = stream_payload.get("autogaze_runtime_config") if stream_payload else {}
    stream_config = stream_config or {}
    stream_sampling = stream_payload.get("sampling") if stream_payload else {}
    stream_sampling = stream_sampling or {}
    single_result = _single_result(single_payload)
    single_config = single_result.get("autogaze_runtime_config") or (single_payload or {}).get("autogaze_runtime_config") or {}
    return {
        "quickstart_direct": {
            "batch_size": quick_options.get("batch_size") or _metric(quickstart_payload, "input", "batch_size"),
            "gazing_ratio": quick_options.get("gazing_ratio") or _metric(quickstart_payload, "input", "gazing_ratio"),
            "task_loss_requirement": quick_options.get("task_loss_requirement")
            or _metric(quickstart_payload, "input", "task_loss_requirement"),
            "target_scales": quick_options.get("target_scales") or _metric(quickstart_payload, "input", "target_scales"),
            "target_patch_size": quick_options.get("target_patch_size")
            or _metric(quickstart_payload, "input", "target_patch_size"),
            "frames": quick_options.get("frames") or _metric(quickstart_payload, "input", "frames"),
            "dtype": quick_options.get("dtype") or _metric(quickstart_payload, "input", "dtype"),
            "siglip_enabled": quick_options.get("siglip_enabled"),
        },
        "current_implementation_stream_profile": (
            {
                "gazing_ratio_tile": stream_config.get("stream_gazing_ratio") or stream_config.get("gazing_ratio_tile"),
                "task_loss_requirement_tile": stream_config.get("task_loss_requirement_tile"),
                "target_scales": stream_config.get("target_scales") or stream_payload.get("autogaze_target_scales"),
                "target_patch_size": stream_config.get("target_patch_size") or stream_payload.get("autogaze_target_patch_size"),
                "max_batch_size_autogaze": stream_config.get("max_batch_size_autogaze"),
                "frames": stream_sampling.get("num_video_frames"),
                "thumbnail_frames": stream_sampling.get("num_video_frames_thumbnail"),
                "stream_chunk_frames": stream_sampling.get("stream_chunk_frames"),
                "max_tiles_video": _metric(stream_payload, "stream_plan", "tiling", "spatial_tiles"),
                "decode_strategy": stream_sampling.get("decode_strategy"),
            }
            if stream_payload
            else None
        ),
        "current_implementation_single": {
            "gazing_ratio_tile": single_config.get("gazing_ratio_tile"),
            "task_loss_requirement_tile": single_config.get("task_loss_requirement_tile"),
            "target_scales": single_config.get("target_scales") or (single_payload or {}).get("autogaze_target_scales"),
            "target_patch_size": single_config.get("target_patch_size") or (single_payload or {}).get("autogaze_target_patch_size"),
            "max_batch_size_autogaze": single_config.get("max_batch_size_autogaze"),
            "frames": _metric(single_result, "token_metrics", "video_sampled_frames"),
            "thumbnail_frames": _metric(single_result, "token_metrics", "thumbnail_sampled_frames"),
            "max_tiles_video": _metric(single_result, "video_input_summary", "spatial_tiles_per_video"),
        },
        "latency_note": (
            "Quick Start batch_size defaults to 1. It affects direct AutoGaze latency and throughput. "
            "NVILA max_batch_size_autogaze batches tile sequences, not user videos; larger values can improve "
            "throughput when many tile sequences exist, but may increase peak memory and may not reduce single-video latency."
        ),
    }


def summarize_comparison(
    quickstart_payload: dict[str, Any],
    stream_payload: dict[str, Any] | None,
    single_payload: dict[str, Any] | None = None,
    *,
    quickstart_json: Path,
    stream_json: Path | None,
    single_json: Path | None = None,
) -> dict[str, Any]:
    quick_autogaze_ms = _metric(quickstart_payload, "latency_ms", "autogaze", "median")
    quick_raw = _metric(quickstart_payload, "gaze", "raw_patch_budget")
    quick_selected = _metric(quickstart_payload, "gaze", "selected_non_padded_patches")
    stream_autogaze_ms = _metric(stream_payload, "timing_ms", "tile_autogaze_forward") if stream_payload else None
    stream_total_ms = _metric(stream_payload, "timing_ms", "pre_llm_stream_total_measured") if stream_payload else None
    stream_raw = _metric(stream_payload, "token_metrics", "autogaze_input_patch_tokens") if stream_payload else None
    if stream_payload and stream_raw is None:
        stream_raw = _metric(stream_payload, "gaze", "raw_patch_budget")
    stream_selected = _metric(stream_payload, "token_metrics", "autogaze_selected_patch_tokens") if stream_payload else None
    if stream_payload and stream_selected is None:
        stream_selected = _metric(stream_payload, "gaze", "selected_non_padded_patches")
    single_result = _single_result(single_payload)
    single_token_metrics = single_result.get("token_metrics", {}) if isinstance(single_result, dict) else {}
    single_autogaze_forward_ms = single_result.get("autogaze_model_forward_ms")
    if single_autogaze_forward_ms is None:
        single_autogaze_forward_ms = _metric(
            single_result,
            "stage_timings_ms",
            "processor",
            "autogaze_forward_batched",
            "total_ms",
        )
    single_autogaze_total_ms = single_result.get("autogaze_total_ms") or single_result.get("gazing_info_total_ms")
    single_raw = single_token_metrics.get("autogaze_input_patch_tokens")
    if single_raw is None:
        single_raw = single_token_metrics.get("encoder_raw_patch_tokens")
    single_selected = single_token_metrics.get("autogaze_selected_patch_tokens")
    if single_selected is None:
        single_selected = single_token_metrics.get("encoder_autogaze_selected_patch_tokens")

    interpretation = [
        "Quick Start direct time measures only the direct AutoGaze model call in repro.autogaze_bench.",
    ]
    if stream_payload:
        interpretation.append(
            "Current stream-profile time measures the repo NVILA-style pre-LLM path: decode, resize if configured, tiling, tensorization, AutoGaze, and optional SigLIP."
        )
    if single_payload:
        interpretation.append(
            "Current single mode measures the full NVILA processor plus model.generate path, including vision encoder, projector, and LLM."
        )
    interpretation.extend(
        [
            "Use raw_patch_budget_ratio and tile_sequences before judging speed. If they are not 1, the workloads are not apples-to-apples.",
            "For the 300ms vs 3s question, compare Quick Start direct against current_implementation_single.autogaze_model_forward_ms first, then autogaze_total_ms.",
        ]
    )

    summary = {
        "inputs": {
            "quickstart_json": str(quickstart_json),
            "stream_profile_json": str(stream_json) if stream_json else None,
            "single_json": str(single_json) if single_json else None,
            "quickstart_video": _metric(quickstart_payload, "input", "video"),
            "stream_video": stream_payload.get("video") if stream_payload else None,
            "single_video": single_payload.get("video") if single_payload else None,
        },
        "autogaze_latency_options": build_autogaze_latency_options_summary(
            quickstart_payload,
            stream_payload,
            single_payload,
        ),
        "quickstart_direct": {
            "device": _metric(quickstart_payload, "metadata", "device"),
            "frames": _metric(quickstart_payload, "input", "frames"),
            "batch_size": _metric(quickstart_payload, "input", "batch_size")
            or _metric(quickstart_payload, "autogaze_latency_options", "batch_size"),
            "dtype": _metric(quickstart_payload, "input", "dtype"),
            "autogaze_ms": quick_autogaze_ms,
            "siglip_full_ms": _metric(quickstart_payload, "latency_ms", "siglip_full", "median"),
            "siglip_gazed_ms": _metric(quickstart_payload, "latency_ms", "siglip_gazed", "median"),
            "raw_patch_budget": quick_raw,
            "selected_patches": quick_selected,
            "total_gaze_slots": _metric(quickstart_payload, "gaze", "total_gaze_slots"),
            "token_reduction_ratio": _metric(quickstart_payload, "gaze", "token_reduction_ratio"),
            "metric_source": "latency_ms.autogaze.median",
        },
        "current_implementation_stream_profile": (
            {
                "device": _metric(stream_payload, "metadata", "device"),
                "frames": _metric(stream_payload, "sampling", "num_video_frames"),
                "thumbnail_frames": _metric(stream_payload, "sampling", "thumbnail_frames_processed"),
                "stream_chunk_frames": _metric(stream_payload, "sampling", "stream_chunk_frames"),
                "autogaze_runtime_config": stream_payload.get("autogaze_runtime_config"),
                "tile_sequences": _metric(stream_payload, "gaze", "tile_sequences")
                or _metric(stream_payload, "token_metrics", "tile_sequences"),
                "video_decode_ms": _metric(stream_payload, "timing_ms", "video_decode_scan")
                or _metric(stream_payload, "timing_ms", "video_decode_seek"),
                "spatial_tile_build_ms": _metric(stream_payload, "timing_ms", "spatial_tile_build"),
                "autogaze_tensorize_ms": _metric(stream_payload, "timing_ms", "tile_autogaze_tensorize"),
                "autogaze_model_forward_ms": stream_autogaze_ms,
                "siglip_gazed_ms": _metric(stream_payload, "timing_ms", "siglip_gazed_forward"),
                "siglip_keep_all_ms": _metric(stream_payload, "timing_ms", "siglip_keep_all_forward"),
                "pre_llm_stream_total_ms": stream_total_ms,
                "raw_patch_budget": stream_raw,
                "selected_patches": stream_selected,
                "total_gaze_slots": _metric(stream_payload, "gaze", "total_gaze_slots"),
                "token_reduction_ratio": _metric(stream_payload, "token_metrics", "autogaze_patch_reduction_ratio")
                or _metric(stream_payload, "gaze", "token_reduction_ratio"),
                "metric_source": "timing_ms.tile_autogaze_forward",
            }
            if stream_payload
            else None
        ),
        "current_implementation_single": None,
        "comparison": {
            "stream_autogaze_forward_vs_quickstart_ratio": _ratio(stream_autogaze_ms, quick_autogaze_ms),
            "stream_total_vs_quickstart_autogaze_ratio": _ratio(stream_total_ms, quick_autogaze_ms),
            "raw_patch_budget_ratio": _ratio(stream_raw, quick_raw),
            "selected_patch_ratio": _ratio(stream_selected, quick_selected),
            "single_autogaze_forward_vs_quickstart_ratio": None,
            "single_autogaze_total_vs_quickstart_ratio": None,
            "single_total_vs_quickstart_autogaze_ratio": None,
            "single_raw_patch_budget_ratio": None,
            "single_selected_patch_ratio": None,
        },
        "interpretation": interpretation,
    }
    if single_payload:
        summary["current_implementation_single"] = {
            "device": _metric(single_payload, "metadata", "device"),
            "frames": single_token_metrics.get("video_sampled_frames"),
            "thumbnail_frames": single_token_metrics.get("thumbnail_sampled_frames"),
            "autogaze_runtime_config": single_result.get("autogaze_runtime_config")
            or single_payload.get("autogaze_runtime_config"),
            "video_preprocess_without_autogaze_ms": single_result.get("video_preprocess_without_autogaze_ms"),
            "video_preprocess_ms": single_result.get("video_preprocess_ms"),
            "video_decode_ms": single_result.get("video_decode_ms"),
            "video_tiling_ms": single_result.get("video_tiling_ms"),
            "autogaze_model_forward_ms": single_autogaze_forward_ms,
            "autogaze_total_ms": single_autogaze_total_ms,
            "generate_ms": single_result.get("generate_ms"),
            "total_ms": single_result.get("total_ms"),
            "ttft_ms": single_result.get("ttft_ms"),
            "siglip_vision_ms": single_result.get("siglip_vision_ms"),
            "vision_encoder_ms": single_result.get("vision_encoder_ms"),
            "llm_forward_ms": single_result.get("llm_forward_ms"),
            "raw_patch_budget": single_raw,
            "selected_patches": single_selected,
            "token_reduction_ratio": single_token_metrics.get("autogaze_patch_reduction_ratio")
            or single_token_metrics.get("encoder_token_reduction_ratio"),
            "metric_source": "result.autogaze_model_forward_ms / result.autogaze_total_ms",
        }
        summary["comparison"].update(
            {
                "single_autogaze_forward_vs_quickstart_ratio": _ratio(single_autogaze_forward_ms, quick_autogaze_ms),
                "single_autogaze_total_vs_quickstart_ratio": _ratio(single_autogaze_total_ms, quick_autogaze_ms),
                "single_total_vs_quickstart_autogaze_ratio": _ratio(single_result.get("total_ms"), quick_autogaze_ms),
                "single_raw_patch_budget_ratio": _ratio(single_raw, quick_raw),
                "single_selected_patch_ratio": _ratio(single_selected, quick_selected),
            }
        )
    return summary


def write_markdown_report(summary: dict[str, Any], path: Path, commands: dict[str, list[str]]) -> None:
    quick = summary["quickstart_direct"]
    stream = summary.get("current_implementation_stream_profile")
    single = summary.get("current_implementation_single")
    comparison = summary["comparison"]
    options = summary.get("autogaze_latency_options", {})
    lines = [
        "# AutoGaze Quick Start vs Current Implementation Timing",
        "",
        "## Scope",
        "",
        "- Quick Start direct: original AutoGaze-style 16-frame call on `example_input.mp4`.",
    ]
    if stream:
        lines.append(
            "- Current implementation: `repro.nvila_runner --mode stream-profile` with the same video and frame count."
        )
    if single:
        lines.append("- Current full path: `repro.nvila_runner --mode single` with the same video and frame count.")
    if stream and single:
        lines.append(
            "- The stream-profile lane validates pre-LLM boundaries; the single lane validates the real NVILA processor plus generate path."
        )
    elif single:
        lines.append("- This run compares Quick Start direct against the non-streaming NVILA `single` generate path.")
    lines.extend(
        [
            "",
            "## Result Summary",
            "",
            "| Path | Device | Frames | Raw patches | Selected patches | AutoGaze forward ms | AutoGaze total ms | Total measured ms | Reduction |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| Quick Start direct | {quick.get('device')} | {quick.get('frames')} | "
                f"{quick.get('raw_patch_budget')} | {quick.get('selected_patches')} | "
                f"{quick.get('autogaze_ms')} |  |  | {quick.get('token_reduction_ratio')} |"
            ),
        ]
    )
    if stream:
        lines.append(
            f"| Current stream-profile | {stream.get('device')} | {stream.get('frames')} | "
            f"{stream.get('raw_patch_budget')} | {stream.get('selected_patches')} | "
            f"{stream.get('autogaze_model_forward_ms')} |  | {stream.get('pre_llm_stream_total_ms')} | "
            f"{stream.get('token_reduction_ratio')} |"
        )
    if single:
        lines.append(
            f"| Current single | {single.get('device')} | {single.get('frames')} | "
            f"{single.get('raw_patch_budget')} | {single.get('selected_patches')} | "
            f"{single.get('autogaze_model_forward_ms')} | {single.get('autogaze_total_ms')} | "
            f"{single.get('total_ms')} | {single.get('token_reduction_ratio')} |"
        )
    lines.extend(
        [
            "",
            "## Ratios",
            "",
            "| Metric | Ratio | Meaning |",
            "| --- | ---: | --- |",
        ]
    )
    if stream:
        lines.extend(
            [
                (
                    "| Stream AutoGaze forward / Quick Start AutoGaze | "
                    f"{comparison.get('stream_autogaze_forward_vs_quickstart_ratio')} | "
                    "Model-forward timing gap before decode/tile overhead |"
                ),
                (
                    "| Stream total / Quick Start AutoGaze | "
                    f"{comparison.get('stream_total_vs_quickstart_autogaze_ratio')} | "
                    "Pre-LLM pipeline overhead gap |"
                ),
                (
                    "| Stream raw patches / Quick Start raw patches | "
                    f"{comparison.get('raw_patch_budget_ratio')} | Workload size check |"
                ),
            ]
        )
    lines.extend(
        [
            (
                "| Single AutoGaze forward / Quick Start AutoGaze | "
                f"{comparison.get('single_autogaze_forward_vs_quickstart_ratio')} | "
                "Full NVILA processor hook model-forward timing gap |"
            ),
            (
                "| Single AutoGaze total / Quick Start AutoGaze | "
                f"{comparison.get('single_autogaze_total_vs_quickstart_ratio')} | "
                "Full NVILA processor AutoGaze overhead gap |"
            ),
            (
                "| Single raw patches / Quick Start raw patches | "
                f"{comparison.get('single_raw_patch_budget_ratio')} | Full path workload size check |"
            ),
            "",
        ]
    )
    if stream:
        lines.extend(
            [
                "## Current Stream-Profile Breakdown",
                "",
                "| Stage | ms |",
                "| --- | ---: |",
                f"| video decode | {stream.get('video_decode_ms')} |",
                f"| spatial tile build | {stream.get('spatial_tile_build_ms')} |",
                f"| AutoGaze tensorize | {stream.get('autogaze_tensorize_ms')} |",
                f"| AutoGaze model forward | {stream.get('autogaze_model_forward_ms')} |",
                f"| SigLIP gazed forward | {stream.get('siglip_gazed_ms')} |",
                f"| SigLIP keep-all forward | {stream.get('siglip_keep_all_ms')} |",
                f"| pre-LLM stream total | {stream.get('pre_llm_stream_total_ms')} |",
                "",
            ]
        )
    if single:
        lines.extend(
            [
                "## Current Single Breakdown",
                "",
                "| Stage | ms |",
                "| --- | ---: |",
                f"| video preprocess without AutoGaze | {single.get('video_preprocess_without_autogaze_ms')} |",
                f"| AutoGaze model forward | {single.get('autogaze_model_forward_ms')} |",
                f"| AutoGaze total | {single.get('autogaze_total_ms')} |",
                f"| generate | {single.get('generate_ms')} |",
                f"| TTFT | {single.get('ttft_ms')} |",
                f"| SigLIP vision | {single.get('siglip_vision_ms')} |",
                f"| LLM forward | {single.get('llm_forward_ms')} |",
                f"| total | {single.get('total_ms')} |",
                "",
            ]
        )
    lines.extend(
        [
            "## AutoGaze Runtime Config",
            "",
        ]
    )
    if stream:
        lines.extend(
            [
                "Stream-profile:",
                "",
                "```json",
                json.dumps(stream.get("autogaze_runtime_config"), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    if single:
        lines.extend(
            [
                "Single:",
                "",
                "```json",
                json.dumps(single.get("autogaze_runtime_config"), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## AutoGaze Latency Options",
            "",
            "| Lane | Batch field | Gazing ratio | Task loss | Target scales | Patch size | Frames | Tiles/chunk | Note |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    quick_options = options.get("quickstart_direct", {})
    stream_options = options.get("current_implementation_stream_profile") or {}
    single_options = options.get("current_implementation_single", {})
    lines.extend(
        [
            (
                f"| Quick Start direct | batch_size={quick_options.get('batch_size')} | "
                f"{quick_options.get('gazing_ratio')} | {quick_options.get('task_loss_requirement')} | "
                f"{quick_options.get('target_scales')} | {quick_options.get('target_patch_size')} | "
                f"{quick_options.get('frames')} |  | Lowest-latency single clip default is batch 1 |"
            ),
        ]
    )
    if stream:
        lines.append(
            (
                f"| Stream-profile | max_batch_size_autogaze={stream_options.get('max_batch_size_autogaze')} | "
                f"{stream_options.get('gazing_ratio_tile')} | {stream_options.get('task_loss_requirement_tile')} | "
                f"{stream_options.get('target_scales')} | {stream_options.get('target_patch_size')} | "
                f"{stream_options.get('frames')} | chunk={stream_options.get('stream_chunk_frames')}, tiles={stream_options.get('max_tiles_video')} | "
                "Batches tile sequences before LLM |"
            )
        )
    lines.extend(
        [
            (
                f"| Single | max_batch_size_autogaze={single_options.get('max_batch_size_autogaze')} | "
                f"{single_options.get('gazing_ratio_tile')} | {single_options.get('task_loss_requirement_tile')} | "
                f"{single_options.get('target_scales')} | {single_options.get('target_patch_size')} | "
                f"{single_options.get('frames')} | tiles={single_options.get('max_tiles_video')} | "
                "Full NVILA processor/generate path |"
            ),
            "",
            str(options.get("latency_note", "")),
            "",
        ]
    )
    lines.extend(
        [
            "## Commands",
            "",
            "Quick Start direct:",
            "",
            "```bash",
            " ".join(commands["quickstart"]),
            "```",
            "",
        ]
    )
    if "stream_profile" in commands:
        lines.extend(
            [
                "Current stream-profile:",
                "",
                "```bash",
                " ".join(commands["stream_profile"]),
                "```",
                "",
            ]
        )
    if "single" in commands:
        lines.extend(
            [
                "Current single:",
                "",
                "```bash",
                " ".join(commands["single"]),
                "```",
                "",
            ]
        )
    lines.extend(["## Notes", ""])
    lines.extend(f"- {item}" for item in summary["interpretation"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def check_target_runtime(config: CompareConfig) -> dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'mps_available': torch.backends.mps.is_available(), "
        "'mps_built': torch.backends.mps.is_built()"
        "}))"
    )
    try:
        result = subprocess.run(
            [str(config.python), "-c", code],
            cwd=str(config.workspace_root),
            env=build_subprocess_env(config),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Python executable not found: {config.python}. "
            "Run this script with the desired venv python, or pass --python /path/to/.venv/bin/python. "
            "You can also set AUTOGAZE_TIMING_PYTHON."
        ) from exc
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if config.device == "mps" and config.require_mps and not info.get("mps_available"):
        raise RuntimeError(f"MPS requested but target runtime reports mps_available=False: {info}")
    return info


def run_command(command: list[str], config: CompareConfig) -> None:
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(
            command,
            cwd=str(config.workspace_root),
            env=build_subprocess_env(config),
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Subprocess executable not found: {command[0]}. "
            "Use --python /path/to/.venv/bin/python if the target venv differs from the current interpreter."
        ) from exc


def validate_compare_config(config: CompareConfig) -> None:
    if config.run_single and int(config.thumbnail_frames) <= 0:
        raise ValueError(
            "NVILA-HD single mode uses the public processor/generate path, which requires "
            "--thumbnail-frames >= 1. Use --thumbnail-frames 1 when the single lane is enabled, "
            "or add --skip-single for AutoGaze-only/stream-profile policy sweeps with "
            "--thumbnail-frames 0."
        )


def run_comparison(config: CompareConfig, *, dry_run: bool = False) -> dict[str, Any]:
    validate_compare_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    quickstart_command = build_quickstart_command(config)
    stream_command = build_stream_profile_command(config) if config.run_stream_profile else None
    single_command = build_single_command(config) if config.run_single else None
    commands = {
        "quickstart": quickstart_command,
    }
    if stream_command is not None:
        commands["stream_profile"] = stream_command
    if single_command is not None:
        commands["single"] = single_command
    if dry_run:
        payload = {"commands": commands}
        write_json(config.summary_json, payload)
        return payload

    runtime = check_target_runtime(config)
    run_command(quickstart_command, config)
    if stream_command is not None:
        run_command(stream_command, config)
    if single_command is not None:
        run_command(single_command, config)
    summary = summarize_comparison(
        _read_json(config.quickstart_json),
        _read_json(config.stream_json) if stream_command is not None else None,
        _read_json(config.single_json) if single_command is not None else None,
        quickstart_json=config.quickstart_json,
        stream_json=config.stream_json if stream_command is not None else None,
        single_json=config.single_json if single_command is not None else None,
    )
    summary["target_runtime"] = runtime
    summary["commands"] = commands
    write_json(config.summary_json, summary)
    write_markdown_report(summary, config.markdown_report, commands)
    return summary


def compact_sweep_run(config: CompareConfig, summary: dict[str, Any]) -> dict[str, Any]:
    quick = summary.get("quickstart_direct") or {}
    stream = summary.get("current_implementation_stream_profile") or {}
    single = summary.get("current_implementation_single") or {}
    comparison = summary.get("comparison") or {}
    run = {
        "gazing_ratio": config.gazing_ratio,
        "task_loss_requirement": config.task_loss_requirement,
        "output_dir": str(config.output_dir),
        "summary_json": str(config.summary_json),
        "markdown_report": str(config.markdown_report),
        "commands": summary.get("commands"),
        "metrics": {
            "quickstart_autogaze_ms": quick.get("autogaze_ms"),
            "stream_autogaze_forward_ms": stream.get("autogaze_model_forward_ms"),
            "stream_pre_llm_total_ms": stream.get("pre_llm_stream_total_ms"),
            "single_autogaze_forward_ms": single.get("autogaze_model_forward_ms"),
            "single_autogaze_total_ms": single.get("autogaze_total_ms"),
            "single_total_ms": single.get("total_ms"),
            "quickstart_raw_patch_budget": quick.get("raw_patch_budget"),
            "stream_raw_patch_budget": stream.get("raw_patch_budget"),
            "single_raw_patch_budget": single.get("raw_patch_budget"),
            "quickstart_selected_patches": quick.get("selected_patches"),
            "stream_selected_patches": stream.get("selected_patches"),
            "single_selected_patches": single.get("selected_patches"),
            "quickstart_token_reduction_ratio": quick.get("token_reduction_ratio"),
            "stream_token_reduction_ratio": stream.get("token_reduction_ratio"),
            "single_token_reduction_ratio": single.get("token_reduction_ratio"),
            "stream_autogaze_forward_vs_quickstart_ratio": comparison.get(
                "stream_autogaze_forward_vs_quickstart_ratio"
            ),
            "single_autogaze_forward_vs_quickstart_ratio": comparison.get(
                "single_autogaze_forward_vs_quickstart_ratio"
            ),
        },
    }
    if run["commands"] is None:
        run.pop("commands")
    return run


def write_sweep_markdown_report(payload: dict[str, Any], path: Path) -> None:
    stream_enabled = bool(payload.get("sweep", {}).get("stream_profile_enabled", True))
    single_enabled = bool(payload.get("sweep", {}).get("single_enabled", True))
    lanes = ["Quick Start direct"]
    if stream_enabled:
        lanes.append("stream-profile")
    if single_enabled:
        lanes.append("single")
    lines = [
        "# AutoGaze Policy Sweep",
        "",
        "## Scope",
        "",
        "- Sweeps `gazing_ratio` and `task_loss_requirement` with the same video/frame/tile settings.",
        f"- Each row runs: {', '.join(lanes)}.",
        "- Compare latency only after checking raw/selected patch counts. Different patch budgets are not apples-to-apples.",
        "",
        "## Runs",
        "",
        (
            "| Gazing ratio | Task loss | QuickStart ms | Stream AutoGaze ms | Stream total ms | "
            "Single AutoGaze ms | Single total ms | QuickStart reduction | Stream reduction | Output |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in payload["runs"]:
        metrics = run.get("metrics") or {}
        lines.append(
            f"| {run.get('gazing_ratio')} | {run.get('task_loss_requirement')} | "
            f"{metrics.get('quickstart_autogaze_ms')} | "
            f"{metrics.get('stream_autogaze_forward_ms')} | "
            f"{metrics.get('stream_pre_llm_total_ms')} | "
            f"{metrics.get('single_autogaze_forward_ms')} | "
            f"{metrics.get('single_total_ms')} | "
            f"{metrics.get('quickstart_token_reduction_ratio')} | "
            f"{metrics.get('stream_token_reduction_ratio')} | "
            f"`{run.get('output_dir')}` |"
        )
    lines.extend(
        [
            "",
            "## Reading Order",
            "",
            "1. Check `quickstart_raw_patch_budget`, `stream_raw_patch_budget`, and `single_raw_patch_budget` first.",
            "2. Then compare `selected_patches` and token reduction ratio for the same policy.",
            "3. Use `quickstart_autogaze_ms` vs `single_autogaze_forward_ms` for the 3s vs 300ms sanity check.",
            "4. Use stream total only when you want decode/tile/tensorize overhead included.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_sweep(
    config: CompareConfig,
    *,
    gazing_ratios: list[float],
    task_loss_requirements: list[float],
    dry_run: bool = False,
) -> dict[str, Any]:
    sweep_configs = build_sweep_configs(
        config,
        gazing_ratios=gazing_ratios,
        task_loss_requirements=task_loss_requirements,
    )
    runs = []
    for sweep_config in sweep_configs:
        summary = run_comparison(sweep_config, dry_run=dry_run)
        runs.append(compact_sweep_run(sweep_config, summary))
    payload = {
        "sweep": {
            "dry_run": dry_run,
            "root_output_dir": str(config.output_dir),
            "num_runs": len(runs),
            "stream_profile_enabled": config.run_stream_profile,
            "single_enabled": config.run_single,
            "gazing_ratios": [float(item) for item in gazing_ratios],
            "task_loss_requirements": [float(item) for item in task_loss_requirements],
            "note": (
                "Use this sweep to find whether AutoGaze latency is dominated by the gaze policy. "
                "The key comparison is quickstart_autogaze_ms vs single_autogaze_forward_ms with matching "
                "raw patch budget, selected patch count, target scales, and batch settings."
            ),
        },
        "runs": runs,
    }
    write_json(config.sweep_summary_json, payload)
    write_sweep_markdown_report(payload, config.sweep_markdown_report)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original AutoGaze Quick Start timing with current runner timing")
    parser.add_argument("--python", type=Path, default=DEFAULT_TARGET_PYTHON)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--autogaze-repo", type=Path, default=DEFAULT_AUTOGAZE_REPO)
    parser.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--quickstart-batch-size", type=int, default=1)
    parser.add_argument("--quickstart-run-siglip", action="store_true")
    parser.add_argument("--thumbnail-frames", type=int, default=1)
    parser.add_argument("--stream-chunk-frames", type=int, default=16)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--max-batch-size-autogaze", type=int, default=16)
    parser.add_argument("--gazing-ratio", type=float, default=0.75)
    parser.add_argument("--task-loss-requirement", type=float, default=0.7)
    parser.add_argument("--gazing-ratio-sweep")
    parser.add_argument("--task-loss-sweep")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--stream-decode-strategy", choices=["scan", "seek"], default="scan")
    parser.add_argument("--stream-run-siglip", action="store_true")
    parser.add_argument("--stream-siglip-mode", choices=["gazed", "keep-all", "both"], default="gazed")
    parser.add_argument("--autogaze-target-scales")
    parser.add_argument("--autogaze-target-patch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--prompt", default=CompareConfig.prompt)
    parser.add_argument("--skip-stream-profile", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--no-require-mps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> CompareConfig:
    return CompareConfig(
        python=args.python,
        workspace_root=args.workspace_root,
        autogaze_repo=args.autogaze_repo,
        weights_root=args.weights_root,
        output_dir=args.output_dir,
        video=args.video,
        device=args.device,
        dtype=args.dtype,
        frames=args.frames,
        quickstart_batch_size=args.quickstart_batch_size,
        quickstart_run_siglip=args.quickstart_run_siglip,
        thumbnail_frames=args.thumbnail_frames,
        stream_chunk_frames=args.stream_chunk_frames,
        max_tiles_video=args.max_tiles_video,
        max_batch_size_autogaze=args.max_batch_size_autogaze,
        gazing_ratio=args.gazing_ratio,
        task_loss_requirement=args.task_loss_requirement,
        warmup=args.warmup,
        repeat=args.repeat,
        stream_decode_strategy=args.stream_decode_strategy,
        stream_run_siglip=args.stream_run_siglip,
        stream_siglip_mode=args.stream_siglip_mode,
        autogaze_target_scales=args.autogaze_target_scales,
        autogaze_target_patch_size=args.autogaze_target_patch_size,
        require_mps=not args.no_require_mps,
        run_stream_profile=not args.skip_stream_profile,
        run_single=not args.skip_single,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    try:
        if args.gazing_ratio_sweep or args.task_loss_sweep:
            summary = run_sweep(
                config,
                gazing_ratios=parse_float_sweep(args.gazing_ratio_sweep, default=[config.gazing_ratio]),
                task_loss_requirements=parse_float_sweep(args.task_loss_sweep, default=[config.task_loss_requirement]),
                dry_run=args.dry_run,
            )
        else:
            summary = run_comparison(config, dry_run=args.dry_run)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
