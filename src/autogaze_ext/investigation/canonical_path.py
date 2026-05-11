from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf

from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import


ImportProbe = Callable[[str], bool]


DEFAULT_IMPORT_CANDIDATES = {
    "original_autogaze": (
        "autogaze.models.autogaze",
        "autogaze",
    ),
    "modified_siglip": (
        "autogaze.models.modified_siglip",
        "autogaze.models.siglip",
        "models.modified_siglip",
        "siglip",
    ),
    "nvila": (
        "nvila",
        "llava",
        "vila",
    ),
}

CHECKPOINT_ENV_VARS = {
    "original_autogaze": "AUTOGAZE_CHECKPOINT",
    "modified_siglip": "MODIFIED_SIGLIP_CHECKPOINT",
    "nvila": "NVILA_CHECKPOINT",
}


@dataclass(frozen=True)
class ComponentCheck:
    name: str
    import_candidates: list[str]
    configured_module_path: str | None
    configured_class_or_factory: str | None
    module_available: bool
    detected_module: str | None
    class_or_factory_exists: bool
    checkpoint_path: str | None
    checkpoint_exists: bool
    config_path: str | None
    config_path_exists: bool | None
    tokenizer_or_processor_path: str | None
    tokenizer_or_processor_path_exists: bool | None
    device: str | None
    dtype: str | None
    local_files_only: bool | None
    trust_remote_code: bool | None
    strict_checkpoint_loading: bool | None
    extra_kwargs: dict[str, Any]
    ready_for_model_construction: bool
    expected_paths: list[str]
    missing: list[str]

    @property
    def real_ready(self) -> bool:
        return self.ready_for_model_construction


@dataclass(frozen=True)
class ExperimentReadiness:
    experiment_id: str
    config_path: str
    config_exists: bool
    required_components: list[str]
    can_run_real: bool
    can_run_stub: bool
    missing: list[str]


@dataclass(frozen=True)
class CanonicalPathReport:
    config_dir: str
    components: dict[str, ComponentCheck]
    experiments: dict[str, ExperimentReadiness]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_import_probe(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _first_available_module(candidates: Iterable[str], import_probe: ImportProbe) -> tuple[bool, str | None]:
    for candidate in candidates:
        if import_probe(candidate):
            return True, candidate
    return False, None


def _module_probe_import(module_name: str, import_module_fn: ImportModuleFn) -> bool:
    try:
        import_module_fn(module_name)
    except Exception:
        return False
    return True


def _get_nested(config: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _plain_config(config: DictConfig) -> dict[str, Any]:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved config must be a mapping")
    return resolved


def _checkpoint_from_config(component_name: str, cfg: Mapping[str, Any]) -> str | None:
    if component_name == "original_autogaze":
        value = _get_nested(cfg, "model", "autogaze", "checkpoint")
    elif component_name == "modified_siglip":
        value = _get_nested(cfg, "model", "vision_encoder", "checkpoint")
    elif component_name == "nvila":
        value = _get_nested(cfg, "model", "mllm", "checkpoint")
    else:
        value = None
    return str(value) if value else None


def _component_node(component_name: str, cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if component_name == "original_autogaze":
        value = _get_nested(cfg, "model", "autogaze")
    elif component_name == "modified_siglip":
        value = _get_nested(cfg, "model", "vision_encoder")
    elif component_name == "nvila":
        value = _get_nested(cfg, "model", "mllm")
    else:
        value = None
    return value if isinstance(value, Mapping) else {}


def _optional_path_exists(path: str | None) -> bool | None:
    if not path:
        return None
    return Path(path).expanduser().exists()


def _resolve_checkpoint(
    component_name: str,
    cfg: Mapping[str, Any],
    checkpoint_overrides: Mapping[str, str | Path] | None,
) -> tuple[str | None, list[str]]:
    env_var = CHECKPOINT_ENV_VARS[component_name]
    config_path = _checkpoint_from_config(component_name, cfg)
    override_path = checkpoint_overrides.get(component_name) if checkpoint_overrides else None
    env_path = os.environ.get(env_var)

    expected = [
        f"config:{_component_config_checkpoint_key(component_name)}",
        f"env:{env_var}",
    ]
    if override_path:
        return str(override_path), [str(override_path), *expected]
    if config_path:
        return config_path, [config_path, *expected]
    if env_path:
        return env_path, [env_path, *expected]
    return None, expected


def _component_config_checkpoint_key(component_name: str) -> str:
    if component_name == "original_autogaze":
        return "model.autogaze.checkpoint"
    if component_name == "modified_siglip":
        return "model.vision_encoder.checkpoint"
    if component_name == "nvila":
        return "model.mllm.checkpoint"
    return "unknown"


def _check_component(
    name: str,
    cfg: Mapping[str, Any],
    *,
    import_candidates: Iterable[str],
    import_probe: ImportProbe | None,
    import_module_fn: ImportModuleFn,
    checkpoint_overrides: Mapping[str, str | Path] | None,
) -> ComponentCheck:
    node = _component_node(name, cfg)
    candidates = list(import_candidates)
    configured_module_path = node.get("module_path")
    configured_class_or_factory = node.get("class_or_factory")
    configured_module_path = str(configured_module_path) if configured_module_path else None
    configured_class_or_factory = str(configured_class_or_factory) if configured_class_or_factory else None

    if configured_module_path:
        resolution = resolve_import(
            configured_module_path,
            configured_class_or_factory,
            import_module_fn=import_module_fn,
        )
        module_available = resolution.module_available
        detected_module = configured_module_path if resolution.module_available else None
        class_or_factory_exists = resolution.object_available
        import_error = resolution.error
        if configured_module_path not in candidates:
            candidates = [configured_module_path, *candidates]
    else:
        probe = import_probe or _default_import_probe
        module_available, detected_module = _first_available_module(candidates, probe)
        class_or_factory_exists = True if configured_class_or_factory is None else False
        import_error = None

    checkpoint_path, expected_paths = _resolve_checkpoint(name, cfg, checkpoint_overrides)
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).expanduser().exists())
    config_path = node.get("config_path")
    tokenizer_or_processor_path = node.get("tokenizer_or_processor_path")
    config_path = str(config_path) if config_path else None
    tokenizer_or_processor_path = str(tokenizer_or_processor_path) if tokenizer_or_processor_path else None
    config_path_exists = _optional_path_exists(config_path)
    tokenizer_or_processor_path_exists = _optional_path_exists(tokenizer_or_processor_path)
    extra_kwargs = node.get("extra_kwargs") if isinstance(node.get("extra_kwargs"), Mapping) else {}

    missing: list[str] = []
    if not module_available:
        if configured_module_path:
            missing.append(import_error or f"configured module import not available: {configured_module_path}")
        else:
            missing.append(f"module import not available; tried {', '.join(candidates)}")
    if module_available and configured_class_or_factory and not class_or_factory_exists:
        missing.append(import_error or f"class/factory not available: {configured_class_or_factory}")
    if not checkpoint_path:
        missing.append(f"checkpoint path not configured; expected {_component_config_checkpoint_key(name)} or {CHECKPOINT_ENV_VARS[name]}")
    elif not checkpoint_exists:
        missing.append(f"checkpoint path does not exist: {checkpoint_path}")
    if config_path and not config_path_exists:
        missing.append(f"config path does not exist: {config_path}")
    if tokenizer_or_processor_path and not tokenizer_or_processor_path_exists:
        missing.append(f"tokenizer/processor path does not exist: {tokenizer_or_processor_path}")

    ready_for_model_construction = (
        module_available
        and class_or_factory_exists
        and checkpoint_exists
        and config_path_exists is not False
        and tokenizer_or_processor_path_exists is not False
    )

    return ComponentCheck(
        name=name,
        import_candidates=candidates,
        configured_module_path=configured_module_path,
        configured_class_or_factory=configured_class_or_factory,
        module_available=module_available,
        detected_module=detected_module,
        class_or_factory_exists=class_or_factory_exists,
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_exists,
        config_path=config_path,
        config_path_exists=config_path_exists,
        tokenizer_or_processor_path=tokenizer_or_processor_path,
        tokenizer_or_processor_path_exists=tokenizer_or_processor_path_exists,
        device=str(node.get("device")) if node.get("device") else None,
        dtype=str(node.get("dtype")) if node.get("dtype") else None,
        local_files_only=bool(node.get("local_files_only")) if "local_files_only" in node else None,
        trust_remote_code=bool(node.get("trust_remote_code")) if "trust_remote_code" in node else None,
        strict_checkpoint_loading=bool(node.get("strict_checkpoint_loading"))
        if "strict_checkpoint_loading" in node
        else None,
        extra_kwargs=dict(extra_kwargs),
        ready_for_model_construction=ready_for_model_construction,
        expected_paths=expected_paths,
        missing=missing,
    )


def _experiment_readiness(
    experiment_id: str,
    config_dir: Path,
    components: Mapping[str, ComponentCheck],
) -> ExperimentReadiness:
    config_path = config_dir / "experiment" / f"{experiment_id}.yaml"
    config_exists = config_path.exists()
    required = ["modified_siglip", "nvila"]
    if experiment_id in {"A2", "A2_real"}:
        required = ["original_autogaze", *required]

    missing: list[str] = []
    if not config_exists:
        missing.append(f"required config missing: {config_path}")
    for component_name in required:
        component = components[component_name]
        if not component.module_available:
            missing.append(f"{component_name}: module import unavailable")
        if not component.class_or_factory_exists:
            missing.append(f"{component_name}: class/factory unavailable")
        if not component.checkpoint_exists:
            missing.append(f"{component_name}: checkpoint unavailable")
        if component.config_path_exists is False:
            missing.append(f"{component_name}: config path unavailable")
        if component.tokenizer_or_processor_path_exists is False:
            missing.append(f"{component_name}: tokenizer/processor path unavailable")

    return ExperimentReadiness(
        experiment_id=experiment_id,
        config_path=str(config_path),
        config_exists=config_exists,
        required_components=required,
        can_run_real=config_exists and not missing,
        can_run_stub=config_exists,
        missing=missing,
    )


def check_canonical_path(
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    experiment_ids: Iterable[str] = ("A1", "A2"),
    source_config_name: str | None = None,
    import_probe: ImportProbe | None = None,
    import_module_fn: ImportModuleFn = importlib.import_module,
    import_candidates: Mapping[str, Iterable[str]] | None = None,
    checkpoint_overrides: Mapping[str, str | Path] | None = None,
) -> CanonicalPathReport:
    config_dir = Path(config_dir)
    candidates = {**DEFAULT_IMPORT_CANDIDATES, **dict(import_candidates or {})}
    experiment_ids = tuple(experiment_ids)
    if not experiment_ids:
        raise ValueError("experiment_ids must not be empty")

    if source_config_name is None:
        if "A2_real" in experiment_ids:
            source_config_name = "experiment/A2_real"
        elif "A2" in experiment_ids:
            source_config_name = "experiment/A2"
        else:
            source_config_name = f"experiment/{experiment_ids[-1]}"
    elif not source_config_name.startswith("experiment/"):
        source_config_name = f"experiment/{source_config_name}"

    cfg = _plain_config(load_config(config_dir, source_config_name))

    components = {
        name: _check_component(
            name,
            cfg,
            import_candidates=candidates[name],
            import_probe=import_probe,
            import_module_fn=import_module_fn,
            checkpoint_overrides=checkpoint_overrides,
        )
        for name in ("original_autogaze", "modified_siglip", "nvila")
    }
    experiments = {
        experiment_id: _experiment_readiness(experiment_id, config_dir, components)
        for experiment_id in experiment_ids
    }
    return CanonicalPathReport(
        config_dir=str(config_dir),
        components=components,
        experiments=experiments,
    )


def _print_report(report: CanonicalPathReport) -> None:
    print("Canonical AutoGaze/SigLIP/NVILA path check")
    print(f"config_dir: {report.config_dir}")
    print()
    print("Available components:")
    for component in report.components.values():
        status = "available" if component.real_ready else "stub-only/missing"
        print(f"- {component.name}: {status}")
        print(f"  configured_module_path: {component.configured_module_path or 'N/A'}")
        print(f"  configured_class_or_factory: {component.configured_class_or_factory or 'N/A'}")
        print(f"  module_available: {component.module_available}")
        print(f"  detected_module: {component.detected_module or 'N/A'}")
        print(f"  class_or_factory_exists: {component.class_or_factory_exists}")
        print(f"  checkpoint_path: {component.checkpoint_path or 'N/A'}")
        print(f"  checkpoint_exists: {component.checkpoint_exists}")
        print(f"  config_path: {component.config_path or 'N/A'}")
        print(f"  config_path_exists: {component.config_path_exists if component.config_path_exists is not None else 'N/A'}")
        print(f"  tokenizer_or_processor_path: {component.tokenizer_or_processor_path or 'N/A'}")
        print(
            "  tokenizer_or_processor_path_exists: "
            f"{component.tokenizer_or_processor_path_exists if component.tokenizer_or_processor_path_exists is not None else 'N/A'}"
        )
        print(f"  device: {component.device or 'N/A'}")
        print(f"  dtype: {component.dtype or 'N/A'}")
        print(f"  local_files_only: {component.local_files_only if component.local_files_only is not None else 'N/A'}")
        print(f"  trust_remote_code: {component.trust_remote_code if component.trust_remote_code is not None else 'N/A'}")
        print(
            "  strict_checkpoint_loading: "
            f"{component.strict_checkpoint_loading if component.strict_checkpoint_loading is not None else 'N/A'}"
        )
        print(f"  ready_for_model_construction: {component.ready_for_model_construction}")
        print(f"  expected_paths: {', '.join(component.expected_paths)}")
        if component.missing:
            print(f"  missing: {'; '.join(component.missing)}")
    print()
    print("Experiment readiness:")
    for readiness in report.experiments.values():
        mode = "real models" if readiness.can_run_real else "stubs only"
        print(f"- {readiness.experiment_id}: {mode}")
        print(f"  config_exists: {readiness.config_exists}")
        print(f"  required_components: {', '.join(readiness.required_components)}")
        if readiness.missing:
            print(f"  blockers: {'; '.join(readiness.missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real canonical AutoGaze/SigLIP/NVILA path readiness")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument(
        "--experiment-id",
        action="append",
        dest="experiment_ids",
        help="Experiment ID to validate. Repeatable. Defaults to A1 and A2.",
    )
    parser.add_argument(
        "--source-config-name",
        default=None,
        help="Experiment config used to read component module/path settings. Defaults to A2 or A2_real when selected.",
    )
    parser.add_argument("--autogaze-checkpoint", default=None)
    parser.add_argument("--modified-siglip-checkpoint", default=None)
    parser.add_argument("--nvila-checkpoint", default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    overrides = {
        key: value
        for key, value in {
            "original_autogaze": args.autogaze_checkpoint,
            "modified_siglip": args.modified_siglip_checkpoint,
            "nvila": args.nvila_checkpoint,
        }.items()
        if value
    }
    report = check_canonical_path(
        config_dir=args.config_dir,
        experiment_ids=args.experiment_ids or ("A1", "A2"),
        source_config_name=args.source_config_name,
        checkpoint_overrides=overrides,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
