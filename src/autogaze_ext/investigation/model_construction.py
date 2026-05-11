from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from autogaze_ext.investigation.quick_start_reference import QuickStartLocation, locate_quick_start
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import


COMPONENT_CONFIG_PATHS = {
    "autogaze": ("model", "autogaze"),
    "vision_encoder": ("model", "vision_encoder"),
    "mllm": ("model", "mllm"),
}


@dataclass(frozen=True)
class ComponentConstructionResult:
    component: str
    status: str
    construction_level: int
    module_path: str | None
    class_or_factory: str | None
    module_available: bool
    class_or_factory_exists: bool
    config_module_path: str | None
    config_class_or_factory: str | None
    config_constructed: bool
    model_constructed: bool
    checkpoint_path: str | None
    checkpoint_exists: bool
    checkpoint_load_attempted: bool
    checkpoint_metadata_checked: bool
    device: str
    dtype: str | None
    quick_start_fields_used: dict[str, Any]
    stayed_stub: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class ModelConstructionReport:
    experiment_id: str
    construction_level: int
    checkpoint_policy: str
    device: str
    dtype: str | None
    quick_start_found: bool
    quick_start_path: str | None
    quick_start_reference_found: bool
    quick_start_reference_path: str | None
    memory_safety: dict[str, Any]
    components: list[ComponentConstructionResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_plain_config(config: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Resolved config must be a mapping")
    return data


def _get_nested(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _component_node(cfg: Mapping[str, Any], component: str) -> dict[str, Any]:
    value = _get_nested(cfg, COMPONENT_CONFIG_PATHS[component])
    return dict(value) if isinstance(value, Mapping) else {}


def _required_components(experiment_id: str) -> list[str]:
    if experiment_id == "A1_real":
        return ["vision_encoder", "mllm"]
    if experiment_id == "A2_real":
        return ["autogaze", "vision_encoder", "mllm"]
    raise ValueError(f"Unsupported real canonical experiment: {experiment_id}")


def _selected_components(experiment_id: str, component: str) -> list[str]:
    if component == "all":
        return _required_components(experiment_id)
    if component not in COMPONENT_CONFIG_PATHS:
        raise ValueError(f"Unsupported component: {component}")
    return [component]


def _checkpoint_exists(path: str | None) -> bool:
    return bool(path and Path(path).expanduser().exists())


def _quick_start_fields(node: Mapping[str, Any], experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    inference = experiment_cfg.get("inference") if isinstance(experiment_cfg.get("inference"), Mapping) else {}
    fields = {
        "input_resolution": node.get("input_resolution", inference.get("input_resolution")),
        "max_resolution": node.get("max_resolution", inference.get("max_resolution")),
        "scale_resolution": node.get("scale_resolution", inference.get("scale_resolution")),
        "frame_count": node.get("frame_count", inference.get("frame_count")),
        "model_config_path": node.get("model_config_path", inference.get("model_config_path")),
        "processor_path": node.get("processor_path"),
        "query_text": node.get("query_text", inference.get("query_text")),
        "output_dir": node.get("output_dir", inference.get("output_dir")),
        "visualization_dir": node.get("visualization_dir", inference.get("visualization_dir")),
        "original_cli_args": dict(node.get("original_cli_args", {})) if isinstance(node.get("original_cli_args"), Mapping) else {},
    }
    return fields


def _instantiate_config(
    node: Mapping[str, Any],
    *,
    import_module_fn: ImportModuleFn,
) -> tuple[bool, Any | None, str | None]:
    config_module_path = node.get("config_module_path")
    config_class_or_factory = node.get("config_class_or_factory")
    if not config_module_path or not config_class_or_factory:
        return False, None, "config class/factory is not configured"

    resolution = resolve_import(str(config_module_path), str(config_class_or_factory), import_module_fn=import_module_fn)
    if not resolution.ready:
        return False, None, resolution.error

    module = import_module_fn(str(config_module_path))
    factory = module
    for part in str(config_class_or_factory).split("."):
        factory = getattr(factory, part)
    kwargs = dict(node.get("config_kwargs", {})) if isinstance(node.get("config_kwargs"), Mapping) else {}
    try:
        return True, factory(**kwargs), None
    except Exception as exc:
        return False, None, f"config construction failed: {exc}"


def _construct_model(
    node: Mapping[str, Any],
    config_obj: Any | None,
    *,
    import_module_fn: ImportModuleFn,
    load_checkpoint: bool,
) -> tuple[bool, Any | None, str | None, bool]:
    module_path = node.get("module_path")
    class_or_factory = node.get("class_or_factory")
    if not module_path or not class_or_factory:
        return False, None, "model class/factory is not configured", False

    resolution = resolve_import(str(module_path), str(class_or_factory), import_module_fn=import_module_fn)
    if not resolution.ready:
        return False, None, resolution.error, False

    module = import_module_fn(str(module_path))
    factory = module
    for part in str(class_or_factory).split("."):
        factory = getattr(factory, part)

    kwargs = dict(node.get("construction_kwargs", {})) if isinstance(node.get("construction_kwargs"), Mapping) else {}
    checkpoint_path = node.get("checkpoint")
    checkpoint_load_attempted = False

    try:
        if load_checkpoint and checkpoint_path and hasattr(factory, "from_pretrained"):
            checkpoint_load_attempted = True
            model = factory.from_pretrained(
                str(checkpoint_path),
                local_files_only=bool(node.get("local_files_only", True)),
                trust_remote_code=bool(node.get("trust_remote_code", False)),
                **kwargs,
            )
        elif config_obj is not None:
            model = factory(config_obj, **kwargs)
        else:
            model = factory(**kwargs)
    except Exception as exc:
        return False, None, f"model construction failed: {exc}", checkpoint_load_attempted

    return True, model, None, checkpoint_load_attempted


def _metadata_check(path: str | None) -> tuple[bool, str | None]:
    if not path:
        return False, "checkpoint path is not configured"
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return False, f"checkpoint path does not exist: {path}"
    # Metadata only: intentionally no torch.load or tensor deserialization.
    return True, None


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def _check_component(
    component: str,
    cfg: Mapping[str, Any],
    *,
    construction_level: int,
    device: str,
    no_checkpoint_load: bool,
    checkpoint_metadata_only: bool,
    load_checkpoint: bool,
    import_module_fn: ImportModuleFn,
) -> ComponentConstructionResult:
    node = _component_node(cfg, component)
    quick_start_fields = _quick_start_fields(node, cfg)
    module_path = str(node.get("module_path")) if node.get("module_path") else None
    class_or_factory = str(node.get("class_or_factory")) if node.get("class_or_factory") else None
    config_module_path = str(node.get("config_module_path")) if node.get("config_module_path") else None
    config_class_or_factory = str(node.get("config_class_or_factory")) if node.get("config_class_or_factory") else None
    checkpoint_path = str(node.get("checkpoint")) if node.get("checkpoint") else None
    checkpoint_exists = _checkpoint_exists(checkpoint_path)
    dtype = str(node.get("dtype")) if node.get("dtype") else None

    module_resolution = resolve_import(module_path, None, import_module_fn=import_module_fn)
    class_resolution = resolve_import(module_path, class_or_factory, import_module_fn=import_module_fn)
    config_constructed = False
    model_constructed = False
    checkpoint_load_attempted = False
    checkpoint_metadata_checked = False
    failure_reason: str | None = None
    config_obj: Any | None = None

    if construction_level >= 0 and not module_resolution.module_available:
        failure_reason = module_resolution.error
    elif construction_level >= 1 and not class_resolution.ready:
        failure_reason = class_resolution.error

    if failure_reason is None and construction_level >= 2:
        config_constructed, config_obj, failure_reason = _instantiate_config(node, import_module_fn=import_module_fn)
        if failure_reason and "not configured" in failure_reason:
            # Some components, notably external MLLMs, may not expose a config-only constructor yet.
            failure_reason = None

    if failure_reason is None and (construction_level == 3 or checkpoint_metadata_only):
        checkpoint_metadata_checked, metadata_error = _metadata_check(checkpoint_path)
        if metadata_error:
            failure_reason = metadata_error

    if failure_reason is None and construction_level >= 4:
        if load_checkpoint and no_checkpoint_load:
            failure_reason = "--load-checkpoint cannot be combined with --no-checkpoint-load"
        elif load_checkpoint and not checkpoint_exists:
            failure_reason = f"full checkpoint loading requested but checkpoint path is unavailable: {checkpoint_path}"
        else:
            model_constructed, _, failure_reason, checkpoint_load_attempted = _construct_model(
                node,
                config_obj,
                import_module_fn=import_module_fn,
                load_checkpoint=load_checkpoint and not no_checkpoint_load,
            )

    status = "passed" if failure_reason is None else "failed"
    stayed_stub = status != "passed" or (construction_level >= 4 and not model_constructed)
    return ComponentConstructionResult(
        component=component,
        status=status,
        construction_level=construction_level,
        module_path=module_path,
        class_or_factory=class_or_factory,
        module_available=module_resolution.module_available,
        class_or_factory_exists=class_resolution.object_available,
        config_module_path=config_module_path,
        config_class_or_factory=config_class_or_factory,
        config_constructed=config_constructed,
        model_constructed=model_constructed,
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_exists,
        checkpoint_load_attempted=checkpoint_load_attempted,
        checkpoint_metadata_checked=checkpoint_metadata_checked,
        device=device,
        dtype=dtype,
        quick_start_fields_used=quick_start_fields,
        stayed_stub=stayed_stub,
        failure_reason=failure_reason,
    )


def run_model_construction_check(
    *,
    experiment: str,
    component: str = "all",
    construction_level: int = 1,
    no_checkpoint_load: bool = True,
    checkpoint_metadata_only: bool = False,
    load_checkpoint: bool = False,
    device: str = "cpu",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    quick_start_path: str | Path | None = None,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> ModelConstructionReport:
    if construction_level < 0 or construction_level > 4:
        raise ValueError("construction_level must be between 0 and 4")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps")

    cfg = _to_plain_config(load_config(config_dir, f"experiment/{experiment}"))
    selected = _selected_components(experiment, component)

    quick_start_location: QuickStartLocation | None = None
    quick_start_error: str | None = None
    try:
        quick_start_location = locate_quick_start(quick_start_path, repo_root=Path(config_dir).parent)
    except FileNotFoundError as exc:
        quick_start_error = str(exc)

    quick_start_reference = Path(config_dir).parent / "docs" / "QUICK_START_reference.md"
    memory_safety = {
        "default_device": "cpu",
        "requested_device": device,
        "device_available": _device_available(device),
        "no_checkpoint_load": no_checkpoint_load,
        "checkpoint_metadata_only": checkpoint_metadata_only,
        "load_checkpoint": load_checkpoint,
        "full_construction_risk": "high" if construction_level >= 4 else "low",
        "risk_note": "Level 4 may allocate large model objects; checkpoint tensors load only with --load-checkpoint.",
        "quick_start_error": quick_start_error,
    }

    components = [
        _check_component(
            item,
            cfg,
            construction_level=construction_level,
            device=device,
            no_checkpoint_load=no_checkpoint_load,
            checkpoint_metadata_only=checkpoint_metadata_only,
            load_checkpoint=load_checkpoint,
            import_module_fn=import_module_fn,
        )
        for item in selected
    ]

    return ModelConstructionReport(
        experiment_id=experiment,
        construction_level=construction_level,
        checkpoint_policy="load_checkpoint" if load_checkpoint and not no_checkpoint_load else "no_checkpoint_load",
        device=device,
        dtype=str(_get_nested(cfg, ("runtime", "precision", "dtype")) or "unknown"),
        quick_start_found=quick_start_location is not None,
        quick_start_path=str(quick_start_location.path) if quick_start_location else None,
        quick_start_reference_found=quick_start_reference.exists(),
        quick_start_reference_path=str(quick_start_reference) if quick_start_reference.exists() else None,
        memory_safety=memory_safety,
        components=components,
    )


def _print_report(report: ModelConstructionReport) -> None:
    print("Canonical real-path model construction check")
    print(f"experiment: {report.experiment_id}")
    print(f"construction_level: {report.construction_level}")
    print(f"checkpoint_policy: {report.checkpoint_policy}")
    print(f"device: {report.device}")
    print(f"dtype: {report.dtype}")
    print(f"QUICK_START.md found: {report.quick_start_found} ({report.quick_start_path or 'N/A'})")
    print(
        "QUICK_START_reference.md found: "
        f"{report.quick_start_reference_found} ({report.quick_start_reference_path or 'N/A'})"
    )
    print(f"memory_safety: {json.dumps(report.memory_safety, sort_keys=True)}")
    print()
    for component in report.components:
        print(f"- {component.component}: {component.status}")
        print(f"  module: {component.module_path or 'N/A'}")
        print(f"  class_or_factory: {component.class_or_factory or 'N/A'}")
        print(f"  module_available: {component.module_available}")
        print(f"  class_or_factory_exists: {component.class_or_factory_exists}")
        print(f"  config_constructed: {component.config_constructed}")
        print(f"  model_constructed: {component.model_constructed}")
        print(f"  checkpoint_path: {component.checkpoint_path or 'N/A'}")
        print(f"  checkpoint_exists: {component.checkpoint_exists}")
        print(f"  checkpoint_load_attempted: {component.checkpoint_load_attempted}")
        print(f"  checkpoint_metadata_checked: {component.checkpoint_metadata_checked}")
        print(f"  device: {component.device}")
        print(f"  dtype: {component.dtype or 'N/A'}")
        print(f"  quick_start_fields_used: {json.dumps(component.quick_start_fields_used, sort_keys=True)}")
        print(f"  stayed_stub: {component.stayed_stub}")
        if component.failure_reason:
            print(f"  failure_reason: {component.failure_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-inference model-construction smoke check for A1_real/A2_real")
    parser.add_argument("--experiment", choices=["A1_real", "A2_real"], required=True)
    parser.add_argument("--component", choices=["autogaze", "vision_encoder", "mllm", "all"], default="all")
    parser.add_argument("--construction-level", type=int, choices=[0, 1, 2, 3, 4], default=1)
    parser.add_argument("--no-checkpoint-load", action="store_true", default=True)
    parser.add_argument("--checkpoint-metadata-only", action="store_true")
    parser.add_argument("--load-checkpoint", action="store_true", help="Explicitly allow checkpoint tensor loading")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--quick-start-path", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.load_checkpoint:
        args.no_checkpoint_load = False

    report = run_model_construction_check(
        experiment=args.experiment,
        component=args.component,
        construction_level=args.construction_level,
        no_checkpoint_load=args.no_checkpoint_load,
        checkpoint_metadata_only=args.checkpoint_metadata_only,
        load_checkpoint=args.load_checkpoint,
        device=args.device,
        config_dir=args.config_dir,
        quick_start_path=args.quick_start_path,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
