from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf

from autogaze_ext.investigation.canonical_smoke_inference import SmokeInferenceReport, run_canonical_smoke_inference
from autogaze_ext.metrics import BenchmarkResult, compute_fps, compute_throughput, write_csv_results, write_json_result
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.profiling import MemoryTracker, count_selected_patches_per_scale, token_reduction_ratio
from autogaze_ext.utils.reproducibility import save_reproducibility_manifest


InferenceFn = Callable[..., SmokeInferenceReport]


@dataclass(frozen=True)
class TinyRealBenchmarkArtifacts:
    result: BenchmarkResult
    json_path: Path
    csv_path: Path
    manifest_path: Path
    iteration_summary_paths: list[Path]


def _to_plain_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        data = OmegaConf.to_container(config, resolve=True)
    else:
        data = dict(config)
    if not isinstance(data, dict):
        raise TypeError("Config must resolve to a mapping")
    return data


def _load_benchmark_preset(config_dir: str | Path, config_name: str) -> DictConfig:
    config_dir = Path(config_dir)
    name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    path = Path(config_name)
    if path.exists():
        return OmegaConf.load(path)
    if name.startswith("benchmark/"):
        path = config_dir / f"{name}.yaml"
    else:
        path = config_dir / "benchmark" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Tiny benchmark preset not found: {path}")
    return OmegaConf.load(path)


def _benchmark_node(preset: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    plain = _to_plain_config(preset)
    node = plain.get("benchmark", plain)
    if not isinstance(node, Mapping):
        raise TypeError("Benchmark preset must contain a benchmark mapping")
    return dict(node)


def _get_nested(config: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _model_paths(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "autogaze": {
            "checkpoint": _get_nested(experiment_cfg, "model", "autogaze", "checkpoint"),
            "model_config_path": _get_nested(experiment_cfg, "model", "autogaze", "model_config_path"),
            "processor_path": _get_nested(experiment_cfg, "model", "autogaze", "processor_path"),
        },
        "vision_encoder": {
            "checkpoint": _get_nested(experiment_cfg, "model", "vision_encoder", "checkpoint"),
            "model_config_path": _get_nested(experiment_cfg, "model", "vision_encoder", "model_config_path"),
            "processor_path": _get_nested(experiment_cfg, "model", "vision_encoder", "processor_path"),
        },
        "mllm": {
            "checkpoint": _get_nested(experiment_cfg, "model", "mllm", "checkpoint"),
            "model_config_path": _get_nested(experiment_cfg, "model", "mllm", "model_config_path"),
            "processor_path": _get_nested(experiment_cfg, "model", "mllm", "processor_path"),
        },
    }


def _numeric_values(values: Iterable[float | int | str | None]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
    return numeric


def _mean_or_na(values: Iterable[float | int | str | None]) -> float | str:
    numeric = _numeric_values(values)
    return mean(numeric) if numeric else "N/A"


def _stage(report: SmokeInferenceReport, name: str):
    for stage in report.stages:
        if stage.name == name:
            return stage
    return None


def _stage_latency(report: SmokeInferenceReport, name: str) -> float | str:
    stage = _stage(report, name)
    if stage is None:
        return "N/A"
    if stage.status == "disabled_full_token_path":
        return 0.0
    return stage.latency_ms if stage.latency_ms is not None else "N/A"


def _executed_stages(report: SmokeInferenceReport) -> list[str]:
    return [
        stage.name
        for stage in report.stages
        if stage.status in {"passed", "disabled_full_token_path"}
    ]


def _token_reduction(before: int, after: int) -> float:
    if before <= 0:
        return 0.0
    return token_reduction_ratio(before, after)


def _selected_scales_from_artifact(report: SmokeInferenceReport) -> list[int] | None:
    path = report.artifacts.get("selected_scales")
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("selected_scales")
    return value if isinstance(value, list) else None


def _peak_vram(reports: Iterable[SmokeInferenceReport]) -> float | str:
    values = _numeric_values(report.peak_vram_mb for report in reports)
    return max(values) if values else "N/A"


def _run_iterations(
    *,
    benchmark_cfg: Mapping[str, Any],
    experiment_cfg: DictConfig,
    mode: str,
    output_dir: Path,
    config_dir: str | Path,
    inference_fn: InferenceFn,
) -> tuple[list[SmokeInferenceReport], list[Path]]:
    warmup_iterations = int(benchmark_cfg.get("warmup_iterations", 1))
    benchmark_iterations = int(benchmark_cfg.get("benchmark_iterations", 3))
    experiment = str(benchmark_cfg["experiment"])
    num_frames = int(benchmark_cfg.get("num_frames", 2))
    resolution = int(benchmark_cfg.get("resolution", 224))
    query_text = str(benchmark_cfg.get("query_text") or "") or None
    if mode == "autogaze_only":
        query_text = None

    common = {
        "experiment": experiment,
        "mode": mode,
        "video": "dummy",
        "query_text": query_text,
        "num_frames": num_frames,
        "resolution": resolution,
        "scale_resolution": benchmark_cfg.get("scale_resolution"),
        "device": str(benchmark_cfg.get("device", "cpu")),
        "dtype": str(benchmark_cfg.get("dtype", "float32")),
        "max_new_tokens": int(benchmark_cfg.get("max_new_tokens", 1)),
        "config_dir": config_dir,
        "cfg": experiment_cfg,
        "allow_mllm_load": bool(benchmark_cfg.get("allow_mllm_load", False)),
    }

    for index in range(warmup_iterations):
        inference_fn(
            **common,
            output_dir=output_dir / "warmup" / f"{mode}_{index:03d}",
        )

    reports: list[SmokeInferenceReport] = []
    summary_paths: list[Path] = []
    for index in range(benchmark_iterations):
        iteration_output_dir = output_dir / "iterations" / f"{mode}_{index:03d}"
        report = inference_fn(**common, output_dir=iteration_output_dir)
        reports.append(report)
        if "inference_summary" in report.artifacts:
            summary_paths.append(Path(report.artifacts["inference_summary"]))
    return reports, summary_paths


def _build_result(
    *,
    benchmark_cfg: Mapping[str, Any],
    experiment_cfg: DictConfig,
    mode: str,
    reports: list[SmokeInferenceReport],
) -> BenchmarkResult:
    if not reports:
        raise ValueError("reports must not be empty")
    plain_experiment_cfg = _to_plain_config(experiment_cfg)
    last = reports[-1]
    batch_size = int(benchmark_cfg.get("batch_size", 1))
    frame_count = int(last.frame_count)
    before = int(last.original_visual_token_count or 0)
    after = int(last.selected_token_count if last.selected_token_count is not None else before)
    end_to_end_latency = float(mean(report.latency_ms for report in reports))
    autogaze_latency = _mean_or_na(_stage_latency(report, "autogaze") for report in reports)
    vision_latency = _mean_or_na(_stage_latency(report, "vision_encoder") for report in reports)
    mllm_latency = _mean_or_na(_stage_latency(report, "mllm") for report in reports)
    selected_scales = _selected_scales_from_artifact(last)

    return BenchmarkResult(
        experiment_id=str(benchmark_cfg["experiment"]),
        task_type=str(_get_nested(plain_experiment_cfg, "task", "type") or "video_vqa"),
        device=str(benchmark_cfg.get("device", "cpu")),
        precision=str(benchmark_cfg.get("dtype", "float32")),
        peak_vram_mb=_peak_vram(reports),
        inference_latency_ms=end_to_end_latency,
        throughput_videos_per_sec=compute_throughput(batch_size=batch_size, latency_ms=end_to_end_latency),
        fps=compute_fps(batch_size=batch_size, frames_per_video=frame_count, latency_ms=end_to_end_latency),
        visual_token_count_before_autogaze=before,
        visual_token_count_after_autogaze=after,
        token_reduction_ratio=_token_reduction(before, after),
        selected_patches_per_frame=(after / float(frame_count)) if frame_count else 0.0,
        selected_patches_per_scale=count_selected_patches_per_scale(selected_scales),
        autogaze_latency_ms=autogaze_latency,
        vit_latency_ms=vision_latency,
        mllm_prefill_latency_ms=mllm_latency if mllm_latency != "N/A" else "N/A",
        mllm_decode_latency_ms="N/A",
        end_to_end_latency_ms=end_to_end_latency,
        autogaze="ON" if bool(_get_nested(plain_experiment_cfg, "model", "autogaze", "enabled")) else "OFF",
        vision_encoder_type=str(_get_nested(plain_experiment_cfg, "model", "vision_encoder", "type") or "modified_siglip"),
        mllm_type=str(_get_nested(plain_experiment_cfg, "model", "mllm", "type") or "nvila"),
        integration_mode=str(_get_nested(plain_experiment_cfg, "experiment", "integration_mode") or "unknown"),
        input_frame_count=frame_count,
        input_resolution=f"{last.processed_resolution[0]}x{last.processed_resolution[1]}",
        dummy_task_metric=None,
        acceleration_type_note=(
            "tiny real smoke benchmark; encoder-side reduction is measured only when AutoGaze and modified SigLIP stages execute"
        ),
        stub_status="tiny_real_partial" if last.skipped_stages else "tiny_real_complete",
        metadata={
            "warning": benchmark_cfg.get("warning", "tiny real benchmark only; not a paper reproduction result"),
            "benchmark_mode": mode,
            "query_text": last.query_text,
            "original_resolution": last.original_resolution,
            "processed_resolution": last.processed_resolution,
            "frame_count": frame_count,
            "batch_size": batch_size,
            "warmup_iterations": int(benchmark_cfg.get("warmup_iterations", 1)),
            "benchmark_iterations": int(benchmark_cfg.get("benchmark_iterations", 3)),
            "construction_level": int(benchmark_cfg.get("construction_level", 3)),
            "model_paths": _model_paths(plain_experiment_cfg),
            "executed_stages": _executed_stages(last),
            "skipped_stages": last.skipped_stages,
            "quick_start_scaling": last.scaling,
            "mllm_decode_latency_note": "not available unless MLLM generation exposes separate decode timing",
            "iteration_latencies_ms": [report.latency_ms for report in reports],
        },
    )


def run_tiny_real_benchmark(
    *,
    config_name: str,
    mode: str,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path | None = None,
    device: str | None = None,
    dtype: str | None = None,
    warmup_iterations: int | None = None,
    benchmark_iterations: int | None = None,
    allow_mllm_load: bool | None = None,
    inference_fn: InferenceFn = run_canonical_smoke_inference,
) -> TinyRealBenchmarkArtifacts:
    if mode not in {"autogaze_only", "full_pipeline"}:
        raise ValueError("mode must be autogaze_only or full_pipeline")
    preset = _load_benchmark_preset(config_dir, config_name)
    benchmark_cfg = _benchmark_node(preset)
    if mode not in set(benchmark_cfg.get("modes", [])):
        raise ValueError(f"Benchmark preset {config_name} does not support mode {mode}")

    if device is not None:
        benchmark_cfg["device"] = device
    if dtype is not None:
        benchmark_cfg["dtype"] = dtype
    if warmup_iterations is not None:
        benchmark_cfg["warmup_iterations"] = warmup_iterations
    if benchmark_iterations is not None:
        benchmark_cfg["benchmark_iterations"] = benchmark_iterations
    if allow_mllm_load is not None:
        benchmark_cfg["allow_mllm_load"] = allow_mllm_load

    experiment_cfg = load_config(config_dir, f"experiment/{benchmark_cfg['experiment']}")
    root_output_dir = Path(output_dir or benchmark_cfg.get("output_dir") or "outputs/tiny_real_benchmarks")
    mode_output_dir = root_output_dir / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    memory = MemoryTracker(str(benchmark_cfg.get("device", "cpu")))
    memory.reset_peak()
    reports, summary_paths = _run_iterations(
        benchmark_cfg=benchmark_cfg,
        experiment_cfg=experiment_cfg,
        mode=mode,
        output_dir=mode_output_dir,
        config_dir=config_dir,
        inference_fn=inference_fn,
    )
    result = _build_result(
        benchmark_cfg=benchmark_cfg,
        experiment_cfg=experiment_cfg,
        mode=mode,
        reports=reports,
    )

    json_path = write_json_result(result, mode_output_dir / "benchmark_result.json")
    csv_path = write_csv_results([result], mode_output_dir / "benchmark_result.csv")
    manifest_path = save_reproducibility_manifest(
        {
            "benchmark": benchmark_cfg,
            "experiment_config": _to_plain_config(experiment_cfg),
        },
        mode_output_dir / "reproducibility_manifest.json",
        repo_root=Path(config_dir).parent,
        metadata={
            "benchmark_result_json": str(json_path),
            "benchmark_result_csv": str(csv_path),
            "iteration_summary_paths": [str(path) for path in summary_paths],
        },
    )
    return TinyRealBenchmarkArtifacts(
        result=result,
        json_path=json_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        iteration_summary_paths=summary_paths,
    )


def _print_artifact(artifact: TinyRealBenchmarkArtifacts) -> None:
    result = artifact.result
    print("WARNING: tiny real benchmark only; not a paper reproduction result")
    print(f"experiment ID: {result.experiment_id}")
    print(f"benchmark mode: {result.metadata.get('benchmark_mode') if result.metadata else 'N/A'}")
    print(f"AutoGaze: {result.autogaze}")
    print(f"vision encoder type: {result.vision_encoder_type}")
    print(f"MLLM type: {result.mllm_type}")
    print(f"end-to-end latency ms: {result.end_to_end_latency_ms}")
    print(f"AutoGaze latency ms: {result.autogaze_latency_ms}")
    print(f"vision encoder latency ms: {result.vit_latency_ms}")
    print(f"MLLM prefill latency ms: {result.mllm_prefill_latency_ms}")
    print(f"token count before AutoGaze: {result.visual_token_count_before_autogaze}")
    print(f"token count after AutoGaze: {result.visual_token_count_after_autogaze}")
    print(f"token reduction ratio: {result.token_reduction_ratio}")
    print(f"peak VRAM MB: {result.peak_vram_mb}")
    print(f"JSON result: {artifact.json_path}")
    print(f"CSV result: {artifact.csv_path}")
    print(f"reproducibility manifest: {artifact.manifest_path}")
    if result.metadata:
        print(f"executed stages: {', '.join(result.metadata.get('executed_stages', [])) or 'none'}")
        skipped = result.metadata.get("skipped_stages", [])
        if skipped:
            print(f"skipped stages: {json.dumps(skipped, sort_keys=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded real A1/A2 benchmark presets")
    parser.add_argument(
        "--config-name",
        required=True,
        help=(
            "Benchmark preset name under configs/benchmark, for example tiny_a1_real, "
            "canonical_a1_small, or canonical_a2_medium."
        ),
    )
    parser.add_argument("--mode", choices=["autogaze_only", "full_pipeline"], required=True)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--warmup-iterations", type=int, default=None)
    parser.add_argument("--benchmark-iterations", type=int, default=None)
    parser.add_argument("--allow-mllm-load", action="store_true")
    args = parser.parse_args()

    artifact = run_tiny_real_benchmark(
        config_name=args.config_name,
        mode=args.mode,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        allow_mllm_load=True if args.allow_mllm_load else None,
    )
    _print_artifact(artifact)


if __name__ == "__main__":
    main()
