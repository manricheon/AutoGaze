from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from autogaze_ext.data import HFDatasetLoader
from autogaze_ext.metrics import HFEvaluateMetric, exact_match
from autogaze_ext.models.huggingface import HFModelLoader, HFProcessorLoader
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.utils import HFLoadConfig, SUPPORTED_HF_MODES, hf_offline_mode, redacted_hf_config


@dataclass(frozen=True)
class HFBenchmarkReport:
    experiment_id: str
    mode: str
    integration_mode: str
    model_id: str | None
    model_revision: str | None
    processor_tokenizer_id: str | None
    dataset_id: str | None
    dataset_split: str | None
    dataset_revision: str | None
    cache_mode: str
    cache_dir: str | None
    offline_mode: bool
    local_files_only: bool
    trust_remote_code: bool
    evaluated_samples: int
    metric_implementation_source: str
    metric_name: str
    metric_result: dict[str, Any]
    dry_run: bool
    stub_status: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _node_to_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    return dict(OmegaConf.to_container(node, resolve=True))


def _merge_non_null(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if value is not None:
                merged[key] = value
    return merged


def _model_hf_config(cfg: DictConfig) -> HFLoadConfig:
    runtime = _node_to_dict(cfg.get("runtime", {}).get("huggingface", {}))
    model = _node_to_dict(cfg.get("model", {}).get("huggingface", {}))
    merged = _merge_non_null(runtime, model)
    return HFLoadConfig.from_mapping(merged)


def _dataset_hf_config(cfg: DictConfig) -> HFLoadConfig:
    runtime = _node_to_dict(cfg.get("runtime", {}).get("huggingface", {}))
    data = _node_to_dict(cfg.get("data", {}).get("huggingface", {}))
    merged = _merge_non_null(runtime, data)
    return HFLoadConfig.from_mapping(merged)


def _report_hf_config(cfg: DictConfig) -> HFLoadConfig:
    runtime = _node_to_dict(cfg.get("runtime", {}).get("huggingface", {}))
    model = _node_to_dict(cfg.get("model", {}).get("huggingface", {}))
    data = _node_to_dict(cfg.get("data", {}).get("huggingface", {}))
    merged = _merge_non_null(runtime, data, model)
    return HFLoadConfig.from_mapping(merged)


def _requires_model(mode: str) -> bool:
    return mode in {"hf_model_only", "hf_model_and_dataset", "hf_model_local_dataset", "offline_hf_cache"}


def _requires_dataset(mode: str) -> bool:
    return mode in {
        "hf_dataset_only",
        "hf_model_and_dataset",
        "local_model_hf_dataset",
        "hf_model_local_dataset",
        "offline_hf_cache",
    }


def _load_dataset_if_needed(mode: str, hf_config: HFLoadConfig, dry_run: bool) -> tuple[Any | None, int]:
    if not _requires_dataset(mode):
        return None, 0
    if not hf_config.dataset_id:
        return None, 0
    dataset_path = Path(hf_config.dataset_id)
    if dry_run and not dataset_path.exists():
        return None, 0
    dataset = HFDatasetLoader(hf_config).load_dataset(hf_config.dataset_id)
    try:
        return dataset, len(dataset)
    except TypeError:
        return dataset, 0


def _compute_metric(dataset: Any | None, metric_name: str) -> tuple[str, dict[str, Any]]:
    if dataset is None:
        return "internal_fallback", {"num_samples": 0, "score": 0.0}

    predictions: list[str] = []
    references: list[str] = []
    for row in dataset:
        answer = str(row.get("answer", "dummy")) if isinstance(row, dict) else "dummy"
        predictions.append(answer)
        references.append(answer)

    metric = HFEvaluateMetric(
        metric_name,
        fallback_compute=lambda preds, refs: {
            "exact_match": sum(exact_match(str(pred), str(ref)) for pred, ref in zip(preds, refs)) / len(preds)
            if preds
            else 0.0,
            "num_samples": len(preds),
        },
    )
    metric.add_batch(predictions=predictions, references=references)
    result = metric.compute()
    return str(result.get("metric_source", "internal_fallback")), result


def run_hf_benchmark(
    cfg: DictConfig,
    *,
    output_dir: str | Path = "outputs/hf_benchmarks",
    dry_run: bool | None = None,
    model_loader: HFModelLoader | None = None,
    processor_loader: HFProcessorLoader | None = None,
) -> Path:
    bench_cfg = cfg.benchmark.huggingface
    mode = str(bench_cfg.mode)
    if mode not in SUPPORTED_HF_MODES:
        raise ValueError(f"Unsupported HF benchmark mode: {mode}")

    model_config = _model_hf_config(cfg)
    dataset_config = _dataset_hf_config(cfg)
    report_config = _report_hf_config(cfg)
    dry = bool(bench_cfg.get("dry_run", True) if dry_run is None else dry_run)
    integration_mode = str(bench_cfg.get("integration_mode", "official_processor"))
    if integration_mode != "official_processor":
        raise ValueError("HF benchmark runner currently supports only official_processor integration mode")

    model = None
    processor = None
    with hf_offline_mode(model_config.offline or dataset_config.offline):
        if _requires_model(mode) and model_config.model_id and not dry:
            model_loader = model_loader or HFModelLoader(model_config)
            model = model_loader.load_model(model_config.model_id)
            processor_loader = processor_loader or HFProcessorLoader(model_config)
            processor = processor_loader.load_processor(model_config.model_id)

        dataset, evaluated_samples = _load_dataset_if_needed(mode, dataset_config, dry)

    metric_source, metric_result = _compute_metric(dataset, str(bench_cfg.get("metric_name", "exact_match")))
    experiment_id = str(cfg.get("experiment", {}).get("id", f"hf_{mode}"))
    offline = model_config.offline or dataset_config.offline
    local_files_only = model_config.local_files_only or dataset_config.local_files_only
    cache_mode = "offline_hf_cache" if offline else ("local_files_only" if local_files_only else "standard")

    report = HFBenchmarkReport(
        experiment_id=experiment_id,
        mode=mode,
        integration_mode=integration_mode,
        model_id=model_config.model_id,
        model_revision=model_config.revision,
        processor_tokenizer_id=model_config.model_id if model_config.model_id else None,
        dataset_id=dataset_config.dataset_id,
        dataset_split=dataset_config.dataset_split,
        dataset_revision=dataset_config.revision,
        cache_mode=cache_mode,
        cache_dir=model_config.cache_dir or dataset_config.cache_dir,
        offline_mode=offline,
        local_files_only=local_files_only,
        trust_remote_code=model_config.trust_remote_code,
        evaluated_samples=evaluated_samples,
        metric_implementation_source=metric_source,
        metric_name=str(bench_cfg.get("metric_name", "exact_match")),
        metric_result=metric_result,
        dry_run=dry,
        stub_status="metadata_only_no_large_default_benchmark" if dry else "loader_path_executed_no_autogaze_token_injection",
        metadata={
            "warning": "HF benchmark path does not assume AutoGaze token injection or claim AutoGaze improvement",
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "redacted_hf_config": redacted_hf_config(report_config),
        },
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_id}.json"
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generic Hugging Face benchmark metadata/smoke path")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="hf_benchmark/hf_dataset_only")
    parser.add_argument("--output-dir", default="outputs/hf_benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run mode")
    args = parser.parse_args()

    cfg = load_config(Path(args.config_dir), args.config_name)
    path = run_hf_benchmark(cfg, output_dir=args.output_dir, dry_run=True if args.dry_run else None)
    print(f"HF benchmark report: {path}")


if __name__ == "__main__":
    main()
