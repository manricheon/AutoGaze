from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from autogaze_ext.metrics import BenchmarkResult, compute_fps, compute_throughput, write_csv_results, write_json_result
from autogaze_ext.pipeline.experiment_registry import EXPERIMENT_REGISTRY, validate_experiment_config
from autogaze_ext.pipeline.inference import run_inference
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.profiling import MemoryTracker, measure_latency_ms, token_reduction_ratio


@dataclass(frozen=True)
class DummyBenchmarkArtifacts:
    result: BenchmarkResult
    json_path: Path
    csv_path: Path


def _dummy_execution_config(cfg: DictConfig) -> DictConfig:
    """Create a checkpoint-free execution config while preserving benchmark metadata elsewhere."""
    return OmegaConf.merge(
        cfg,
        {
            "model": {
                "autogaze": {"enabled": False, "mode": "full"},
                "vision_encoder": {"type": "generic_vit", "mode": "dummy"},
                "mllm": {"type": "generic_mllm", "mode": "dummy"},
            }
        },
    )


def _autogaze_note(cfg: DictConfig) -> tuple[str, str]:
    if bool(cfg.model.autogaze.enabled):
        return (
            "downstream token reduction only: stubbed AutoGaze ON path uses dummy full-token execution",
            "stubbed_autogaze_on_no_real_selector_no_token_reduction",
        )
    return (
        "compatibility-only adapter path: AutoGaze OFF full-token dummy execution",
        "dummy_full_token_baseline",
    )


def run_dummy_benchmark(
    experiment_id: str,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path = "outputs/dummy_benchmarks",
    batch_size: int = 2,
    warmup_iters: int = 0,
) -> DummyBenchmarkArtifacts:
    experiment_id = experiment_id.upper()
    if experiment_id not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown canonical experiment ID: {experiment_id}")

    cfg = load_config(Path(config_dir), f"experiment/{experiment_id}")
    validate_experiment_config(cfg)

    execution_cfg = _dummy_execution_config(cfg)
    memory = MemoryTracker(str(cfg.runtime.device.type))
    memory.reset_peak()
    result, latency_ms = measure_latency_ms(
        lambda: run_inference(execution_cfg, batch_size=batch_size),
        device=str(cfg.runtime.device.type),
        warmup_iters=warmup_iters,
        repeat=1,
    )
    snapshot = memory.snapshot()

    before = int(result.logs["visual token count before AutoGaze"])
    # A2/A3 are stubbed and intentionally do not perform real token selection.
    after = int(result.logs["visual token count after AutoGaze"])
    frame_count = int(result.logs["input video shape"][1])
    height = int(result.logs["input video shape"][3])
    width = int(result.logs["input video shape"][4])
    note, stub_status = _autogaze_note(cfg)

    if result.task_type == "video_vqa":
        predictions = result.outputs["generated_text"]
        dummy_metric = sum(pred == "dummy" for pred in predictions) / len(predictions)
    else:
        labels = result.outputs["predicted_labels"]
        dummy_metric = float((labels == 0).to("cpu").float().mean().item())

    benchmark_result = BenchmarkResult(
        experiment_id=experiment_id,
        task_type=str(cfg.task.type),
        device=str(cfg.runtime.device.type),
        precision=str(cfg.runtime.precision.dtype),
        peak_vram_mb=snapshot.peak_vram_mb,
        inference_latency_ms=latency_ms,
        throughput_videos_per_sec=compute_throughput(batch_size=batch_size, latency_ms=latency_ms),
        fps=compute_fps(batch_size=batch_size, frames_per_video=frame_count, latency_ms=latency_ms),
        visual_token_count_before_autogaze=before,
        visual_token_count_after_autogaze=after,
        token_reduction_ratio=token_reduction_ratio(before, after),
        selected_patches_per_frame=after / float(frame_count),
        selected_patches_per_scale="N/A",
        autogaze_latency_ms=0.0,
        vit_latency_ms="N/A",
        mllm_prefill_latency_ms="N/A",
        mllm_decode_latency_ms="N/A",
        end_to_end_latency_ms=latency_ms,
        autogaze="ON" if bool(cfg.model.autogaze.enabled) else "OFF",
        vision_encoder_type=str(cfg.model.vision_encoder.type),
        mllm_type=str(cfg.model.mllm.type),
        integration_mode=str(cfg.experiment.integration_mode),
        input_frame_count=frame_count,
        input_resolution=f"{height}x{width}",
        dummy_task_metric=dummy_metric,
        acceleration_type_note=note,
        stub_status=stub_status,
        metadata={
            "warning": "dummy/stub benchmark only; not a reproduction result",
            "canonical_config": {
                "autogaze_enabled": bool(cfg.model.autogaze.enabled),
                "vision_encoder_type": str(cfg.model.vision_encoder.type),
                "mllm_type": str(cfg.model.mllm.type),
            },
            "dummy_execution": {
                "vision_encoder_type": "generic_vit",
                "mllm_type": "generic_mllm",
                "real_checkpoints_loaded": False,
            },
            "sampled_frame_indices": result.logs["sampled frame indices"],
        },
    )

    output_dir = Path(output_dir)
    json_path = write_json_result(benchmark_result, output_dir / f"{experiment_id}.json")
    csv_path = write_csv_results([benchmark_result], output_dir / f"{experiment_id}.csv")
    return DummyBenchmarkArtifacts(result=benchmark_result, json_path=json_path, csv_path=csv_path)


def run_all_dummy_benchmarks(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path = "outputs/dummy_benchmarks",
    batch_size: int = 2,
    warmup_iters: int = 0,
) -> list[DummyBenchmarkArtifacts]:
    return [
        run_dummy_benchmark(
            experiment_id,
            config_dir=config_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            warmup_iters=warmup_iters,
        )
        for experiment_id in sorted(EXPERIMENT_REGISTRY)
    ]


def _print_artifact(artifact: DummyBenchmarkArtifacts) -> None:
    result = artifact.result
    print("WARNING: dummy/stub benchmark only; not a real reproduction result")
    print(f"experiment ID: {result.experiment_id}")
    print(f"AutoGaze: {result.autogaze}")
    print(f"vision encoder type: {result.vision_encoder_type}")
    print(f"MLLM type: {result.mllm_type}")
    print(f"integration mode: {result.integration_mode}")
    print(f"dummy task metric: {result.dummy_task_metric}")
    print(f"acceleration type note: {result.acceleration_type_note}")
    print(f"JSON result: {artifact.json_path}")
    print(f"CSV result: {artifact.csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dummy/stub canonical A0-A3 benchmark wiring")
    parser.add_argument("--experiment-id", choices=[*sorted(EXPERIMENT_REGISTRY), "all"], default="all")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default="outputs/dummy_benchmarks")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=0)
    args = parser.parse_args()

    if args.experiment_id == "all":
        artifacts = run_all_dummy_benchmarks(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            warmup_iters=args.warmup_iters,
        )
        for artifact in artifacts:
            _print_artifact(artifact)
    else:
        _print_artifact(
            run_dummy_benchmark(
                args.experiment_id,
                config_dir=args.config_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                warmup_iters=args.warmup_iters,
            )
        )


if __name__ == "__main__":
    main()
