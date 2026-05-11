from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf

from autogaze_ext.pipeline.experiment_registry import (
    EXPERIMENT_REGISTRY,
    ExperimentSpec,
    validate_experiment_config,
)


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _merge_group(config_dir: Path, merged: DictConfig, group: str, name: str) -> DictConfig:
    group_path = config_dir / group / f"{name}.yaml"
    if not group_path.exists():
        raise FileNotFoundError(f"Config group not found: {group_path}")

    group_cfg = OmegaConf.load(group_path)
    nested_cfg = OmegaConf.create()
    cursor = nested_cfg
    parts = group.split("/")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = group_cfg
    return OmegaConf.merge(merged, nested_cfg)


def _iter_defaults(defaults: Iterable[object]) -> Iterable[tuple[str, str] | str]:
    for item in defaults:
        if isinstance(item, str):
            yield item
        elif isinstance(item, (dict, DictConfig)):
            for group, name in item.items():
                yield str(group), str(name)
        else:
            raise TypeError(f"Unsupported defaults entry: {item!r}")


def load_config(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    config_name: str = "config",
) -> DictConfig:
    """Load the Hydra-style smoke-test config without initializing Hydra."""
    config_dir = Path(config_dir)
    root_path = config_dir / f"{config_name}.yaml"
    if not root_path.exists():
        raise FileNotFoundError(f"Root config not found: {root_path}")

    root_cfg = OmegaConf.load(root_path)
    merged = OmegaConf.create()
    defaults = root_cfg.get("defaults", [])

    if not isinstance(defaults, (list, ListConfig)):
        raise TypeError("Root config 'defaults' must be a list")

    for entry in _iter_defaults(defaults):
        if entry == "_self_":
            continue
        group, name = entry
        merged = _merge_group(config_dir, merged, group, name)

    root_without_defaults = OmegaConf.masked_copy(
        root_cfg,
        [key for key in root_cfg.keys() if key != "defaults"],
    )
    return OmegaConf.merge(merged, root_without_defaults)


def summarize_config(cfg: DictConfig) -> dict[str, object]:
    hf_enabled = bool(cfg.runtime.huggingface.enabled or cfg.benchmark.huggingface.enabled)
    hf_mode = cfg.runtime.huggingface.mode if hf_enabled else None
    spec: ExperimentSpec | None = None
    if str(cfg.experiment.id).upper() in EXPERIMENT_REGISTRY:
        spec = validate_experiment_config(cfg)

    summary = {
        "experiment ID": cfg.experiment.id,
        "AutoGaze": "ON" if cfg.model.autogaze.enabled else "OFF",
        "vision encoder type": cfg.model.vision_encoder.type,
        "MLLM type": cfg.model.mllm.type,
        "task type": cfg.task.type,
        "device": cfg.runtime.device.type,
        "precision": cfg.runtime.precision.dtype,
        "Hugging Face mode": hf_mode,
    }
    if spec is not None:
        summary.update(
            {
                "AutoGaze config": spec.autogaze_config,
                "vision encoder config": spec.vision_encoder_config,
                "MLLM config": spec.mllm_config,
                "task config": spec.task_config,
                "integration mode": cfg.experiment.integration_mode,
                "compatibility status": cfg.experiment.compatibility_status,
            }
        )
    return summary


def print_summary(cfg: DictConfig) -> None:
    for key, value in summarize_config(cfg).items():
        if value is not None:
            print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGaze extension config smoke runner")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--experiment-id", choices=sorted(EXPERIMENT_REGISTRY), default=None)
    args = parser.parse_args()

    config_name = f"experiment/{args.experiment_id}" if args.experiment_id else args.config_name
    cfg = load_config(args.config_dir, config_name)
    print_summary(cfg)


if __name__ == "__main__":
    main()
