from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autogaze_ext.pipeline.benchmark import run_all_dummy_benchmarks
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config, summarize_config
from autogaze_ext.pipeline.inference import run_inference
from autogaze_ext.utils import save_reproducibility_manifest
from autogaze_ext.visualization import FullPipelineVisualizer


@dataclass(frozen=True)
class IntegrationSmokeArtifacts:
    output_root: Path
    summary_path: Path
    benchmark_json_paths: list[Path]
    benchmark_csv_paths: list[Path]
    visualization_paths: list[Path]
    manifest_path: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _assert_no_checkpoint_required(cfg: Any) -> None:
    model_cfg = cfg.get("model", {})
    for group in ("autogaze", "vision_encoder", "mllm", "task_decoder"):
        checkpoint = model_cfg.get(group, {}).get("checkpoint")
        if checkpoint:
            raise ValueError(f"Integration smoke test must not require a real checkpoint: model.{group}.checkpoint")


def _dummy_video() -> torch.Tensor:
    return torch.linspace(0, 1, steps=2 * 3 * 32 * 32).reshape(2, 3, 32, 32)


def run_integration_smoke(
    *,
    output_root: str | Path = "outputs/integration_smoke",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    batch_size: int = 1,
) -> IntegrationSmokeArtifacts:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    vqa_cfg = load_config(config_dir, "dummy_video_vqa")
    action_cfg = load_config(config_dir, "dummy_action_recognition")
    _assert_no_checkpoint_required(vqa_cfg)
    _assert_no_checkpoint_required(action_cfg)

    vqa_result = run_inference(vqa_cfg, batch_size=batch_size)
    action_result = run_inference(action_cfg, batch_size=batch_size)

    experiment_summaries: dict[str, dict[str, Any]] = {}
    for experiment_id in ("A0", "A1", "A2", "A3"):
        cfg = load_config(config_dir, f"experiment/{experiment_id}")
        _assert_no_checkpoint_required(cfg)
        experiment_summaries[experiment_id] = summarize_config(cfg)

    benchmark_dir = output_root / "benchmarks"
    benchmark_artifacts = run_all_dummy_benchmarks(
        config_dir=config_dir,
        output_dir=benchmark_dir,
        batch_size=batch_size,
        warmup_iters=0,
    )

    visualizer = FullPipelineVisualizer(output_root=output_root, exp_name="integration_smoke")
    visualizer.required_dirs()
    visualization_paths = visualizer.visualize_full_pipeline(
        _dummy_video(),
        selected_patch_indices=[0, 1, 5],
        patch_grid=(4, 4),
        scales=[224, 224, 448],
        answer="dummy",
        action_labels=["dummy_action", "background"],
    )
    visualization_paths.append(visualizer.visualize_scale_indicators([224, 224, 448]))

    manifest_path = save_reproducibility_manifest(
        vqa_cfg,
        output_root / "reproducibility_manifest.json",
        repo_root=Path(__file__).resolve().parents[3],
        metadata={
            "smoke_test": "integration_dummy_only",
            "real_checkpoints_loaded": False,
            "external_downloads": False,
        },
    )

    summary = {
        "warning": "dummy/stub integration smoke only; not a real benchmark result",
        "no_real_model_checkpoint_required": True,
        "external_downloads": False,
        "dummy_video_vqa": {
            "task_type": vqa_result.task_type,
            "logs": vqa_result.logs,
            "generated_text": vqa_result.outputs.get("generated_text"),
        },
        "dummy_action_recognition": {
            "task_type": action_result.task_type,
            "logs": action_result.logs,
            "logits_shape": tuple(action_result.outputs["logits"].shape),
        },
        "canonical_experiments": experiment_summaries,
        "benchmark_json_paths": [artifact.json_path for artifact in benchmark_artifacts],
        "benchmark_csv_paths": [artifact.csv_path for artifact in benchmark_artifacts],
        "visualization_paths": visualization_paths,
        "reproducibility_manifest_path": manifest_path,
    }
    summary_path = output_root / "integration_smoke_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")

    return IntegrationSmokeArtifacts(
        output_root=output_root,
        summary_path=summary_path,
        benchmark_json_paths=[artifact.json_path for artifact in benchmark_artifacts],
        benchmark_csv_paths=[artifact.csv_path for artifact in benchmark_artifacts],
        visualization_paths=visualization_paths,
        manifest_path=manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint-free integration smoke test across dummy modules")
    parser.add_argument("--output-root", default="outputs/integration_smoke")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    artifacts = run_integration_smoke(
        output_root=args.output_root,
        config_dir=args.config_dir,
        batch_size=args.batch_size,
    )
    print("WARNING: dummy/stub integration smoke only; not a real benchmark result")
    print(f"summary: {artifacts.summary_path}")
    print(f"benchmark JSON files: {len(artifacts.benchmark_json_paths)}")
    print(f"benchmark CSV files: {len(artifacts.benchmark_csv_paths)}")
    print(f"visualization files: {len(artifacts.visualization_paths)}")
    print(f"reproducibility manifest: {artifacts.manifest_path}")


if __name__ == "__main__":
    main()

