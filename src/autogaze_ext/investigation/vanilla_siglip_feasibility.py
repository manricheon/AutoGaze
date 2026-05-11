from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from autogaze_ext.models.vision.base_vision_encoder import patch_grid_from_resolution
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.utils.imports import ImportModuleFn, resolve_import


@dataclass(frozen=True)
class ImportCheck:
    module_path: str | None
    class_or_factory: str | None
    module_available: bool
    class_or_factory_exists: bool
    error: str | None

    @property
    def ready(self) -> bool:
        return self.module_available and (self.class_or_factory is None or self.class_or_factory_exists)


@dataclass(frozen=True)
class PatchGridCheck:
    input_resolution: tuple[int, int] | None
    patch_size: int | None
    patch_grid: tuple[int, int] | None
    visual_tokens_per_frame: int | None
    error: str | None

    @property
    def ready(self) -> bool:
        return self.error is None and self.patch_grid is not None


@dataclass(frozen=True)
class NVILAVisualInputCheck:
    expected_visual_dim: int | None
    expected_tokens_per_frame: int | None
    expected_image_size: int | None
    expected_patch_size: int | None
    vanilla_output_dim: int | None
    vanilla_tokens_per_frame: int | None
    feature_dim_compatible: bool
    token_count_compatible: bool | None
    compatible: bool
    issues: list[str]


@dataclass(frozen=True)
class AutoGazePatchIndexCheck:
    autogaze_enabled: bool
    direct_gazing_info_supported: bool
    can_consume_selected_patch_indices_without_adapter: bool
    requires_adapter: bool
    notes: list[str]


@dataclass(frozen=True)
class ModeFeasibility:
    mode: str
    status: str
    true_encoder_side_acceleration: bool
    post_patch_embedding_masking: bool
    post_encoder_pruning: bool
    downstream_token_reduction_only: bool
    compatibility_only_path: bool
    nvila_visual_input_compatible: bool
    autogaze_patch_indices_compatible: bool
    blockers: list[str]
    notes: list[str]


@dataclass(frozen=True)
class VanillaSigLIPFeasibilityReport:
    experiment_id: str
    config_dir: str
    vision_encoder_type: str
    autogaze_enabled: bool
    integration_mode: str
    module_import: ImportCheck
    processor_import: ImportCheck
    hf_loading_option: dict[str, Any]
    checkpoint_path: str | None
    checkpoint_exists: bool
    config_path: str | None
    config_path_exists: bool
    processor_path: str | None
    processor_path_exists: bool
    patch_grid: PatchGridCheck
    output_dim: int | None
    output_dim_source: str
    nvila_visual_input: NVILAVisualInputCheck
    autogaze_patch_indices: AutoGazePatchIndexCheck
    modes: list[ModeFeasibility]
    ready_for_vision_construction_smoke: bool
    ready_for_full_pipeline_construction_smoke: bool
    ready_for_a3_experimental_construction_smoke: bool
    blockers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain_config(config: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Resolved config must be a mapping")
    return data


def _get_nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path_exists(path: str | None) -> bool:
    return bool(path and Path(path).expanduser().exists())


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        return {}
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_resolution(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _int_from_mapping(mapping: Mapping[str, Any], *keys: str) -> int | None:
    value = _get_nested(mapping, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _import_check(
    module_path: str | None,
    class_or_factory: str | None,
    *,
    import_module_fn: ImportModuleFn,
) -> ImportCheck:
    resolution = resolve_import(module_path, class_or_factory, import_module_fn=import_module_fn)
    return ImportCheck(
        module_path=module_path,
        class_or_factory=class_or_factory,
        module_available=resolution.module_available,
        class_or_factory_exists=resolution.object_available,
        error=resolution.error,
    )


def _patch_grid_check(vision_cfg: Mapping[str, Any], experiment_cfg: Mapping[str, Any]) -> PatchGridCheck:
    input_resolution = _as_resolution(
        vision_cfg.get("input_resolution") or _get_nested(experiment_cfg, "inference", "input_resolution")
    )
    patch_size_value = vision_cfg.get("patch_size") or vision_cfg.get("target_patch_size")
    try:
        patch_size = int(patch_size_value) if patch_size_value is not None else None
    except (TypeError, ValueError):
        patch_size = None

    if input_resolution is None:
        return PatchGridCheck(None, patch_size, None, None, "input_resolution is not configured")
    if patch_size is None:
        return PatchGridCheck(input_resolution, None, None, None, "patch_size is not configured")

    try:
        patch_grid = patch_grid_from_resolution(input_resolution, patch_size)
    except ValueError as exc:
        return PatchGridCheck(input_resolution, patch_size, None, None, str(exc))

    return PatchGridCheck(
        input_resolution=input_resolution,
        patch_size=patch_size,
        patch_grid=patch_grid,
        visual_tokens_per_frame=patch_grid[0] * patch_grid[1],
        error=None,
    )


def _infer_output_dim(vision_cfg: Mapping[str, Any], vision_config_json: Mapping[str, Any]) -> tuple[int | None, str]:
    configured = vision_cfg.get("output_dim")
    if configured is not None:
        try:
            return int(configured), str(vision_cfg.get("output_dim_source") or "configured")
        except (TypeError, ValueError):
            return None, "configured_output_dim_invalid"

    for path, source in [
        (("vision_config", "hidden_size"), "vision_config.hidden_size"),
        (("hidden_size",), "hidden_size"),
        (("vision_config", "projection_dim"), "vision_config.projection_dim"),
    ]:
        value = _get_nested(vision_config_json, *path)
        if value is not None:
            try:
                return int(value), source
            except (TypeError, ValueError):
                return None, f"{source}_invalid"

    return None, "unknown"


def _nvila_visual_input_check(
    nvila_cfg: Mapping[str, Any],
    *,
    vanilla_output_dim: int | None,
    vanilla_tokens_per_frame: int | None,
) -> NVILAVisualInputCheck:
    nvila_json = _read_json(str(nvila_cfg.get("config_path")) if nvila_cfg.get("config_path") else None)
    expected_visual_dim = _int_from_mapping(nvila_json, "vision_config", "hidden_size")
    expected_tokens_per_frame = _int_from_mapping(nvila_json, "vision_config", "num_image_tokens")
    expected_image_size = _int_from_mapping(nvila_json, "vision_config", "image_size")
    expected_patch_size = _int_from_mapping(nvila_json, "vision_config", "patch_size")

    issues: list[str] = []
    feature_dim_compatible = (
        expected_visual_dim is not None
        and vanilla_output_dim is not None
        and expected_visual_dim == vanilla_output_dim
    )
    if expected_visual_dim is None:
        issues.append("NVILA expected visual hidden size is unavailable")
    elif vanilla_output_dim is None:
        issues.append("vanilla SigLIP output dimension is unavailable")
    elif expected_visual_dim != vanilla_output_dim:
        issues.append(f"feature dimension mismatch: vanilla SigLIP {vanilla_output_dim} != NVILA {expected_visual_dim}")

    token_count_compatible: bool | None
    if expected_tokens_per_frame is None or vanilla_tokens_per_frame is None:
        token_count_compatible = None
        if expected_tokens_per_frame is None:
            issues.append("NVILA expected visual token count is unavailable")
        if vanilla_tokens_per_frame is None:
            issues.append("vanilla SigLIP token count is unavailable")
    else:
        token_count_compatible = expected_tokens_per_frame == vanilla_tokens_per_frame
        if not token_count_compatible:
            issues.append(
                "token count mismatch: "
                f"vanilla SigLIP {vanilla_tokens_per_frame} tokens/frame != NVILA {expected_tokens_per_frame}"
            )

    compatible = feature_dim_compatible and token_count_compatible is True
    return NVILAVisualInputCheck(
        expected_visual_dim=expected_visual_dim,
        expected_tokens_per_frame=expected_tokens_per_frame,
        expected_image_size=expected_image_size,
        expected_patch_size=expected_patch_size,
        vanilla_output_dim=vanilla_output_dim,
        vanilla_tokens_per_frame=vanilla_tokens_per_frame,
        feature_dim_compatible=feature_dim_compatible,
        token_count_compatible=token_count_compatible,
        compatible=compatible,
        issues=issues,
    )


def _autogaze_patch_index_check(
    *,
    autogaze_enabled: bool,
    vision_cfg: Mapping[str, Any],
) -> AutoGazePatchIndexCheck:
    direct_gazing_info_supported = bool(vision_cfg.get("supports_gazing_info", False))
    direct_patch_selection_supported = bool(vision_cfg.get("supports_direct_patch_selection", False))
    requires_adapter = autogaze_enabled and not (direct_gazing_info_supported or direct_patch_selection_supported)
    notes: list[str] = []
    if not autogaze_enabled:
        notes.append("AutoGaze is OFF; selected patch indices are not consumed in A0.")
    elif requires_adapter:
        notes.append("Vanilla SigLIP does not expose the modified SigLIP gazing_info path.")
        notes.append("Selected patch indices require an explicit adapter and cannot be assumed compatible.")
    else:
        notes.append("Config claims direct AutoGaze patch-index support; this still requires a smoke check.")

    return AutoGazePatchIndexCheck(
        autogaze_enabled=autogaze_enabled,
        direct_gazing_info_supported=direct_gazing_info_supported,
        can_consume_selected_patch_indices_without_adapter=direct_gazing_info_supported
        or direct_patch_selection_supported,
        requires_adapter=requires_adapter,
        notes=notes,
    )


def _a0_modes(nvila_check: NVILAVisualInputCheck) -> list[ModeFeasibility]:
    blockers = list(nvila_check.issues)
    return [
        ModeFeasibility(
            mode="full_token_vanilla_siglip_baseline",
            status="vision_encoder_metadata_feasible" if not blockers else "blocked_for_full_nvila_pipeline",
            true_encoder_side_acceleration=False,
            post_patch_embedding_masking=False,
            post_encoder_pruning=False,
            downstream_token_reduction_only=False,
            compatibility_only_path=True,
            nvila_visual_input_compatible=nvila_check.compatible,
            autogaze_patch_indices_compatible=True,
            blockers=blockers,
            notes=[
                "A0 is AutoGaze OFF, so it is a full-token vanilla SigLIP baseline.",
                "This path does not test AutoGaze-selected patch indices.",
            ],
        )
    ]


def _a3_modes(nvila_check: NVILAVisualInputCheck, patch_index_check: AutoGazePatchIndexCheck) -> list[ModeFeasibility]:
    shared_blockers = list(nvila_check.issues)
    if patch_index_check.requires_adapter:
        shared_blockers.append("vanilla SigLIP cannot consume AutoGaze selected patch indices without an adapter")

    return [
        ModeFeasibility(
            mode="input_level_crop_region_reconstruction",
            status="experimental_stub_only",
            true_encoder_side_acceleration=False,
            post_patch_embedding_masking=False,
            post_encoder_pruning=False,
            downstream_token_reduction_only=False,
            compatibility_only_path=True,
            nvila_visual_input_compatible=nvila_check.compatible,
            autogaze_patch_indices_compatible=False,
            blockers=[
                "crop/region reconstruction adapter is not implemented",
                *shared_blockers,
            ],
            notes=[
                "Could feed reconstructed crops through vanilla SigLIP, but this changes the visual input path.",
                "No encoder-side acceleration is claimed until fewer encoder pixels/tokens are actually computed.",
            ],
        ),
        ModeFeasibility(
            mode="post_patch_embedding_token_masking",
            status="incompatible_without_model_hooks",
            true_encoder_side_acceleration=False,
            post_patch_embedding_masking=True,
            post_encoder_pruning=False,
            downstream_token_reduction_only=False,
            compatibility_only_path=False,
            nvila_visual_input_compatible=nvila_check.compatible,
            autogaze_patch_indices_compatible=False,
            blockers=[
                "vanilla Hugging Face SigLIP does not expose the modified SigLIP gazing_info/mask path",
                "adding patch-embedding hooks would be a model modification, not vanilla SigLIP",
                *shared_blockers,
            ],
            notes=[
                "This is the closest conceptual analogue to modified SigLIP masking, but it is unsupported here.",
            ],
        ),
        ModeFeasibility(
            mode="compact_token_gathering",
            status="possible_post_encoder_only_with_adapter" if nvila_check.compatible else "blocked_for_full_nvila_pipeline",
            true_encoder_side_acceleration=False,
            post_patch_embedding_masking=False,
            post_encoder_pruning=True,
            downstream_token_reduction_only=True,
            compatibility_only_path=False,
            nvila_visual_input_compatible=nvila_check.compatible,
            autogaze_patch_indices_compatible=not patch_index_check.requires_adapter,
            blockers=[
                "full vanilla SigLIP encoder computation is still paid before token gathering",
                "positional metadata adapter is required",
                *shared_blockers,
            ],
            notes=[
                "This can only reduce downstream visual tokens unless a real encoder-side skip is added.",
            ],
        ),
    ]


def check_vanilla_siglip_feasibility(
    *,
    experiment: str,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    import_module_fn: ImportModuleFn = importlib.import_module,
) -> VanillaSigLIPFeasibilityReport:
    if experiment not in {"A0_real", "A3_real"}:
        raise ValueError("experiment must be one of: A0_real, A3_real")

    cfg = _plain_config(load_config(config_dir, f"experiment/{experiment}"))
    vision_cfg = _mapping(_get_nested(cfg, "model", "vision_encoder"))
    nvila_cfg = _mapping(_get_nested(cfg, "model", "mllm"))
    autogaze_cfg = _mapping(_get_nested(cfg, "model", "autogaze"))
    experiment_cfg = _mapping(cfg.get("experiment"))

    module_path = str(vision_cfg.get("module_path")) if vision_cfg.get("module_path") else None
    class_or_factory = str(vision_cfg.get("class_or_factory")) if vision_cfg.get("class_or_factory") else None
    processor_module_path = (
        str(vision_cfg.get("processor_module_path")) if vision_cfg.get("processor_module_path") else None
    )
    processor_class_or_factory = (
        str(vision_cfg.get("processor_class_or_factory")) if vision_cfg.get("processor_class_or_factory") else None
    )

    module_import = _import_check(module_path, class_or_factory, import_module_fn=import_module_fn)
    processor_import = _import_check(
        processor_module_path,
        processor_class_or_factory,
        import_module_fn=import_module_fn,
    )

    checkpoint_path = str(vision_cfg.get("checkpoint")) if vision_cfg.get("checkpoint") else None
    config_path = str(vision_cfg.get("config_path")) if vision_cfg.get("config_path") else None
    processor_path = str(vision_cfg.get("processor_path")) if vision_cfg.get("processor_path") else None
    checkpoint_exists = _path_exists(checkpoint_path)
    config_path_exists = _path_exists(config_path)
    processor_path_exists = _path_exists(processor_path)

    patch_grid = _patch_grid_check(vision_cfg, cfg)
    vision_config_json = _read_json(config_path)
    output_dim, output_dim_source = _infer_output_dim(vision_cfg, vision_config_json)
    nvila_check = _nvila_visual_input_check(
        nvila_cfg,
        vanilla_output_dim=output_dim,
        vanilla_tokens_per_frame=patch_grid.visual_tokens_per_frame,
    )
    autogaze_enabled = bool(autogaze_cfg.get("enabled", False))
    patch_index_check = _autogaze_patch_index_check(
        autogaze_enabled=autogaze_enabled,
        vision_cfg=vision_cfg,
    )

    hf_loading_option = {
        "configured": module_path == "transformers" and bool(class_or_factory),
        "model_loader": class_or_factory,
        "processor_loader": processor_class_or_factory,
        "local_files_only": bool(vision_cfg.get("local_files_only", True)),
        "trust_remote_code": bool(vision_cfg.get("trust_remote_code", False)),
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_exists,
        "metadata_only": True,
        "note": "No model weights are loaded by this feasibility check.",
    }

    modes = _a3_modes(nvila_check, patch_index_check) if autogaze_enabled else _a0_modes(nvila_check)

    blockers: list[str] = []
    if not module_import.ready:
        blockers.append(module_import.error or "vanilla SigLIP model import is unavailable")
    if not processor_import.ready:
        blockers.append(processor_import.error or "vanilla SigLIP processor import is unavailable")
    if not checkpoint_exists:
        blockers.append(f"vanilla SigLIP checkpoint path does not exist: {checkpoint_path}")
    if not config_path_exists:
        blockers.append(f"vanilla SigLIP config path does not exist: {config_path}")
    if not processor_path_exists:
        blockers.append(f"vanilla SigLIP processor path does not exist: {processor_path}")
    if not patch_grid.ready:
        blockers.append(patch_grid.error or "patch grid extraction failed")
    blockers.extend(nvila_check.issues)
    if autogaze_enabled and patch_index_check.requires_adapter:
        blockers.append("A3 requires an explicit AutoGaze patch-index adapter for vanilla SigLIP")

    ready_for_vision_construction_smoke = (
        module_import.ready
        and processor_import.ready
        and checkpoint_exists
        and config_path_exists
        and processor_path_exists
        and patch_grid.ready
        and output_dim is not None
    )
    ready_for_full_pipeline_construction_smoke = ready_for_vision_construction_smoke and nvila_check.compatible
    ready_for_a3_experimental_construction_smoke = (
        autogaze_enabled
        and ready_for_vision_construction_smoke
        and nvila_check.compatible
        and not patch_index_check.requires_adapter
    )

    return VanillaSigLIPFeasibilityReport(
        experiment_id=str(experiment_cfg.get("id") or experiment),
        config_dir=str(config_dir),
        vision_encoder_type=str(vision_cfg.get("type") or "unknown"),
        autogaze_enabled=autogaze_enabled,
        integration_mode=str(experiment_cfg.get("integration_mode") or "unknown"),
        module_import=module_import,
        processor_import=processor_import,
        hf_loading_option=hf_loading_option,
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_exists,
        config_path=config_path,
        config_path_exists=config_path_exists,
        processor_path=processor_path,
        processor_path_exists=processor_path_exists,
        patch_grid=patch_grid,
        output_dim=output_dim,
        output_dim_source=output_dim_source,
        nvila_visual_input=nvila_check,
        autogaze_patch_indices=patch_index_check,
        modes=modes,
        ready_for_vision_construction_smoke=ready_for_vision_construction_smoke,
        ready_for_full_pipeline_construction_smoke=ready_for_full_pipeline_construction_smoke,
        ready_for_a3_experimental_construction_smoke=ready_for_a3_experimental_construction_smoke,
        blockers=blockers,
    )


def _print_report(report: VanillaSigLIPFeasibilityReport) -> None:
    print("Vanilla SigLIP A0/A3 feasibility check")
    print(f"experiment: {report.experiment_id}")
    print(f"autogaze_enabled: {report.autogaze_enabled}")
    print(f"integration_mode: {report.integration_mode}")
    print(f"vision_encoder_type: {report.vision_encoder_type}")
    print()
    print("Vanilla SigLIP import/loading:")
    print(f"- module_path: {report.module_import.module_path or 'N/A'}")
    print(f"- class_or_factory: {report.module_import.class_or_factory or 'N/A'}")
    print(f"- module_available: {report.module_import.module_available}")
    print(f"- class_or_factory_exists: {report.module_import.class_or_factory_exists}")
    print(f"- processor: {report.processor_import.module_path or 'N/A'}.{report.processor_import.class_or_factory or 'N/A'}")
    print(f"- processor_available: {report.processor_import.ready}")
    print(f"- checkpoint_path: {report.checkpoint_path or 'N/A'}")
    print(f"- checkpoint_exists: {report.checkpoint_exists}")
    print(f"- config_path: {report.config_path or 'N/A'}")
    print(f"- config_path_exists: {report.config_path_exists}")
    print(f"- processor_path: {report.processor_path or 'N/A'}")
    print(f"- processor_path_exists: {report.processor_path_exists}")
    print(f"- hf_loading_option: {json.dumps(report.hf_loading_option, sort_keys=True)}")
    print()
    print("Shape checks:")
    print(f"- input_resolution: {report.patch_grid.input_resolution or 'N/A'}")
    print(f"- patch_size: {report.patch_grid.patch_size or 'N/A'}")
    print(f"- patch_grid: {report.patch_grid.patch_grid or 'N/A'}")
    print(f"- visual_tokens_per_frame: {report.patch_grid.visual_tokens_per_frame or 'N/A'}")
    print(f"- output_dim: {report.output_dim or 'N/A'} ({report.output_dim_source})")
    print(f"- NVILA expected visual dim: {report.nvila_visual_input.expected_visual_dim or 'N/A'}")
    print(f"- NVILA expected tokens/frame: {report.nvila_visual_input.expected_tokens_per_frame or 'N/A'}")
    print(f"- NVILA compatible: {report.nvila_visual_input.compatible}")
    if report.nvila_visual_input.issues:
        print(f"- NVILA issues: {'; '.join(report.nvila_visual_input.issues)}")
    print()
    print("AutoGaze patch-index compatibility:")
    print(f"- direct_gazing_info_supported: {report.autogaze_patch_indices.direct_gazing_info_supported}")
    print(
        "- can_consume_selected_patch_indices_without_adapter: "
        f"{report.autogaze_patch_indices.can_consume_selected_patch_indices_without_adapter}"
    )
    print(f"- requires_adapter: {report.autogaze_patch_indices.requires_adapter}")
    print()
    print("Integration modes:")
    for mode in report.modes:
        print(f"- {mode.mode}: {mode.status}")
        print(f"  true_encoder_side_acceleration: {mode.true_encoder_side_acceleration}")
        print(f"  post_patch_embedding_masking: {mode.post_patch_embedding_masking}")
        print(f"  post_encoder_pruning: {mode.post_encoder_pruning}")
        print(f"  downstream_token_reduction_only: {mode.downstream_token_reduction_only}")
        print(f"  compatibility_only_path: {mode.compatibility_only_path}")
        print(f"  nvila_visual_input_compatible: {mode.nvila_visual_input_compatible}")
        print(f"  autogaze_patch_indices_compatible: {mode.autogaze_patch_indices_compatible}")
        if mode.blockers:
            print(f"  blockers: {'; '.join(mode.blockers)}")
    print()
    print("Readiness:")
    print(f"- ready_for_vision_construction_smoke: {report.ready_for_vision_construction_smoke}")
    print(f"- ready_for_full_pipeline_construction_smoke: {report.ready_for_full_pipeline_construction_smoke}")
    print(f"- ready_for_a3_experimental_construction_smoke: {report.ready_for_a3_experimental_construction_smoke}")
    if report.blockers:
        print(f"- blockers: {'; '.join(report.blockers)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check A0/A3 vanilla SigLIP real-path feasibility")
    parser.add_argument("--experiment", choices=["A0_real", "A3_real"], required=True)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = check_vanilla_siglip_feasibility(
        experiment=args.experiment,
        config_dir=args.config_dir,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
