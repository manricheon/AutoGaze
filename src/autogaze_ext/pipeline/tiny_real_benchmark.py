from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from autogaze_ext.investigation.canonical_smoke_inference import SmokeInferenceReport, run_canonical_smoke_inference
from autogaze_ext.metrics import BenchmarkResult, compute_fps, compute_throughput, write_csv_results, write_json_result
from autogaze_ext.pipeline.runner import DEFAULT_CONFIG_DIR, load_config
from autogaze_ext.profiling import MemoryTracker, count_selected_patches_per_scale, token_reduction_ratio
from autogaze_ext.scaling import resolve_autogaze_scaling_policy
from autogaze_ext.utils.imports import resolve_import
from autogaze_ext.utils.reproducibility import save_reproducibility_manifest


InferenceFn = Callable[..., SmokeInferenceReport]
CANONICAL_BENCHMARK_CONFIGS = (
    "canonical_a1_small",
    "canonical_a2_small",
    "canonical_a1_medium",
    "canonical_a2_medium",
)


@dataclass(frozen=True)
class TinyRealBenchmarkArtifacts:
    result: BenchmarkResult
    json_path: Path
    csv_path: Path
    manifest_path: Path
    stage_latency_path: Path
    token_counts_path: Path
    skipped_stages_path: Path
    iteration_summary_paths: list[Path]


@dataclass(frozen=True)
class DryRunCheck:
    name: str
    status: str
    details: dict[str, Any]
    message: str | None = None


@dataclass(frozen=True)
class BenchmarkDryRunReport:
    config_name: str
    preset_id: str
    experiment_id: str
    mode: str | None
    safe_to_execute: bool
    report_path: Path
    checks: list[DryRunCheck]
    failures: list[str]
    warnings: list[str]
    dry_run_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "preset_id": self.preset_id,
            "experiment_id": self.experiment_id,
            "mode": self.mode,
            "safe_to_execute": self.safe_to_execute,
            "report_path": str(self.report_path),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "details": _json_ready(check.details),
                    "message": check.message,
                }
                for check in self.checks
            ],
            "failures": self.failures,
            "warnings": self.warnings,
            "dry_run_only": self.dry_run_only,
            "no_heavy_model_instantiation": True,
            "no_checkpoint_tensor_loading": True,
            "no_inference": True,
        }


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


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


def _required_components(experiment_id: str) -> list[str]:
    if experiment_id == "A1_real":
        return ["vision_encoder", "mllm"]
    if experiment_id == "A2_real":
        return ["autogaze", "vision_encoder", "mllm"]
    return []


def _component_node_key(component: str) -> str:
    if component == "autogaze":
        return "autogaze"
    if component == "vision_encoder":
        return "vision_encoder"
    if component == "mllm":
        return "mllm"
    raise ValueError(f"Unsupported component: {component}")


def _path_exists(value: Any) -> bool:
    return bool(value and Path(str(value)).expanduser().exists())


def _check_path(
    *,
    checks: list[DryRunCheck],
    failures: list[str],
    name: str,
    path: Any,
    required: bool,
) -> None:
    exists = _path_exists(path)
    status = "passed" if exists or not required else "failed"
    message = None if status == "passed" else f"{name} does not exist: {path}"
    checks.append(
        DryRunCheck(
            name=name,
            status=status,
            details={"path": path, "required": required, "exists": exists},
            message=message,
        )
    )
    if message:
        failures.append(message)


def _import_check_name(component: str) -> str:
    if component == "autogaze":
        return "autogaze_module_availability"
    if component == "vision_encoder":
        return "siglip_vit_module_availability"
    if component == "mllm":
        return "nvila_hd_video_module_availability"
    return f"{component}_module_availability"


def _resolve_component_object(node: Mapping[str, Any]) -> tuple[str | None, str | None]:
    return (
        node.get("module_path") or node.get("nvila_hd_video_module_path"),
        node.get("class_or_factory")
        or node.get("class_name")
        or node.get("factory_name")
        or node.get("nvila_hd_video_class_name"),
    )


def _check_module_availability(
    experiment_cfg: Mapping[str, Any],
    experiment_id: str,
    *,
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    for component in _required_components(experiment_id):
        key = _component_node_key(component)
        node = _get_nested(experiment_cfg, "model", key)
        if not isinstance(node, Mapping):
            continue

        module_path, object_name = _resolve_component_object(node)
        resolution = resolve_import(module_path, object_name)
        details: dict[str, Any] = {
            "module_path": resolution.module_path,
            "class_or_factory": resolution.object_name,
            "module_available": resolution.module_available,
            "class_or_factory_available": resolution.object_available,
        }

        config_module_path = node.get("config_module_path")
        config_object_name = node.get("config_class_or_factory")
        config_resolution = None
        if config_module_path or config_object_name:
            config_resolution = resolve_import(config_module_path, config_object_name)
            details["config_module_path"] = config_resolution.module_path
            details["config_class_or_factory"] = config_resolution.object_name
            details["config_module_available"] = config_resolution.module_available
            details["config_class_or_factory_available"] = config_resolution.object_available

        ready = resolution.ready and (config_resolution.ready if config_resolution else True)
        error = resolution.error or (config_resolution.error if config_resolution else None)
        message = None if ready else f"{_import_check_name(component)} failed: {error}"
        checks.append(
            DryRunCheck(
                name=_import_check_name(component),
                status="passed" if ready else "failed",
                details=details,
                message=message,
            )
        )
        if message:
            failures.append(message)

        if component == "mllm":
            processor_resolution = resolve_import(
                node.get("nvila_hd_video_processor_module_path"),
                node.get("nvila_hd_video_processor_class_name"),
            )
            processor_ready = processor_resolution.ready
            processor_message = (
                None
                if processor_ready
                else f"nvila_hd_video_processor_module_availability failed: {processor_resolution.error}"
            )
            checks.append(
                DryRunCheck(
                    name="nvila_hd_video_processor_module_availability",
                    status="passed" if processor_ready else "failed",
                    details={
                        "module_path": processor_resolution.module_path,
                        "class_or_factory": processor_resolution.object_name,
                        "module_available": processor_resolution.module_available,
                        "class_or_factory_available": processor_resolution.object_available,
                    },
                    message=processor_message,
                )
            )
            if processor_message:
                failures.append(processor_message)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value) if isinstance(value, tuple) else [value]


def _check_nvila_hd_video_guide_compatibility(
    *,
    config_dir: str | Path,
    benchmark_cfg: Mapping[str, Any],
    experiment_cfg: Mapping[str, Any],
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    repo_root = Path(config_dir).parent
    reference_path = repo_root / "docs" / "NVILA_HD_VIDEO_REFERENCE.md"
    source_path = repo_root / "docs" / "nvila-hd-video-readme.md"
    mllm = _get_nested(experiment_cfg, "model", "mllm")
    inference = _get_nested(experiment_cfg, "inference")
    experiment_id = str(_get_nested(experiment_cfg, "experiment", "id") or benchmark_cfg.get("experiment") or "")
    errors: list[str] = []
    notes: list[str] = []

    if not isinstance(mllm, Mapping):
        errors.append("model.mllm config is missing")
        mllm = {}
    if not isinstance(inference, Mapping):
        errors.append("inference config is missing")
        inference = {}

    expectations = {
        "nvila_hd_video_model_id": "nvidia/NVILA-8B-HD-Video",
        "module_path": "transformers",
        "class_or_factory": "AutoModel",
        "nvila_hd_video_processor_module_path": "transformers",
        "nvila_hd_video_processor_class_name": "AutoProcessor",
        "video_preprocess_mode": "official_processor",
        "prompt_template": "{video_token}\n\n{prompt}",
        "num_video_frames": 128,
        "num_video_frames_thumbnail": 64,
        "max_tiles_video": 48,
        "max_batch_size_autogaze": 16,
        "max_batch_size_siglip": 32,
        "task_loss_requirement_tile": 0.6,
        "gazing_ratio_thumbnail": 1,
        "trust_remote_code": True,
    }
    for key, expected in expectations.items():
        actual = mllm.get(key, inference.get(key))
        if actual != expected:
            errors.append(f"{key} must match NVILA-HD-Video guide value {expected!r}; got {actual!r}")

    expected_gazing_ratio_tile = [0.2] + [0.06] * 15
    if list(_as_list(mllm.get("gazing_ratio_tile"))) != expected_gazing_ratio_tile:
        errors.append("gazing_ratio_tile must match NVILA-HD-Video guide value [0.2] + [0.06] * 15")

    autogaze_enabled = mllm.get("autogaze_enabled")
    if experiment_id == "A2_real" and autogaze_enabled is not True:
        errors.append("A2_real must set model.mllm.autogaze_enabled=true for the canonical AutoGaze path")
    if experiment_id == "A1_real":
        if autogaze_enabled is not False:
            errors.append("A1_real must set model.mllm.autogaze_enabled=false for the baseline ablation")
        notes.append("A1_real is the AutoGaze-OFF ablation; the NVILA-HD-Video guide primarily documents the AutoGaze path")

    if not reference_path.exists():
        errors.append(f"NVILA-HD-Video extracted reference is missing: {reference_path}")
    if not source_path.exists():
        errors.append(f"NVILA-HD-Video source guide is missing: {source_path}")

    benchmark_num_frames = int(benchmark_cfg.get("num_frames", 0))
    benchmark_resolution = int(benchmark_cfg.get("resolution", 0))
    if benchmark_num_frames <= 0:
        errors.append("benchmark num_frames must be positive for NVILA-HD-Video compatibility")
    if benchmark_resolution <= 0:
        errors.append("benchmark resolution must be positive for NVILA-HD-Video compatibility")

    status = "failed" if errors else "passed"
    message = "; ".join(errors) if errors else None
    checks.append(
        DryRunCheck(
            name="nvila_hd_video_guide_compatibility",
            status=status,
            details={
                "reference_path": reference_path,
                "source_path": source_path,
                "experiment_id": experiment_id,
                "autogaze_enabled": autogaze_enabled,
                "benchmark_num_frames": benchmark_num_frames,
                "benchmark_resolution": benchmark_resolution,
                "video_preprocess_mode": mllm.get("video_preprocess_mode", inference.get("video_preprocess_mode")),
                "prompt_template": mllm.get("prompt_template", inference.get("prompt_template")),
                "model_id": mllm.get("nvila_hd_video_model_id", inference.get("nvila_hd_video_model_id")),
                "notes": notes,
            },
            message=message,
        )
    )
    failures.extend(errors)


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(torch.backends.mps.is_available())
    return False


def _validate_dtype(device: str, dtype: str) -> tuple[bool, str | None]:
    if dtype not in {"float32", "float16", "bfloat16"}:
        return False, f"unsupported dtype: {dtype}"
    if device == "mps" and dtype == "bfloat16":
        return False, "MPS bfloat16 is not treated as benchmark-safe in this dry run"
    return True, None


def _check_benchmark_axes(
    benchmark_cfg: Mapping[str, Any],
    *,
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    axis_errors: list[str] = []
    int_axes = {
        "num_frames": (1, None),
        "resolution": (1, None),
        "warmup_iterations": (0, None),
        "benchmark_iterations": (1, None),
        "max_new_tokens": (1, None),
        "batch_size": (1, None),
    }
    for key, (minimum, maximum) in int_axes.items():
        try:
            value = int(benchmark_cfg.get(key))
        except (TypeError, ValueError):
            axis_errors.append(f"{key} must be an integer")
            continue
        if value < minimum:
            axis_errors.append(f"{key} must be >= {minimum}")
        if maximum is not None and value > maximum:
            axis_errors.append(f"{key} must be <= {maximum}")

    token_budget = benchmark_cfg.get("token_budget")
    if token_budget is not None:
        try:
            if int(token_budget) <= 0:
                axis_errors.append("token_budget must be > 0 when configured")
        except (TypeError, ValueError):
            axis_errors.append("token_budget must be an integer or null")

    modes = benchmark_cfg.get("modes", [])
    if not isinstance(modes, list) or not set(modes).issubset({"autogaze_only", "full_pipeline"}):
        axis_errors.append("modes must contain only autogaze_only and/or full_pipeline")

    safety = benchmark_cfg.get("safety_limits") if isinstance(benchmark_cfg.get("safety_limits"), Mapping) else {}
    if safety:
        if int(benchmark_cfg.get("num_frames", 0)) > int(safety.get("max_default_frames", 0)):
            axis_errors.append("num_frames exceeds configured safety limit")
        if int(benchmark_cfg.get("resolution", 0)) > int(safety.get("max_default_resolution", 0)):
            axis_errors.append("resolution exceeds configured safety limit")
        if int(benchmark_cfg.get("batch_size", 0)) > int(safety.get("max_default_batch_size", 0)):
            axis_errors.append("batch_size exceeds configured safety limit")
        if int(benchmark_cfg.get("max_new_tokens", 0)) > int(safety.get("max_default_new_tokens", 0)):
            axis_errors.append("max_new_tokens exceeds configured safety limit")

    if bool(benchmark_cfg.get("auto_download_datasets", False)):
        axis_errors.append("auto_download_datasets must be false for canonical dry run")

    checks.append(
        DryRunCheck(
            name="benchmark_axis_values",
            status="failed" if axis_errors else "passed",
            details={
                "num_frames": benchmark_cfg.get("num_frames"),
                "resolution": benchmark_cfg.get("resolution"),
                "scale_resolution": benchmark_cfg.get("scale_resolution"),
                "scaling_mode": benchmark_cfg.get("scaling_mode"),
                "temporal_chunk_size": benchmark_cfg.get("temporal_chunk_size"),
                "spatial_tile_size": benchmark_cfg.get("spatial_tile_size"),
                "token_budget": benchmark_cfg.get("token_budget"),
                "warmup_iterations": benchmark_cfg.get("warmup_iterations"),
                "benchmark_iterations": benchmark_cfg.get("benchmark_iterations"),
                "max_new_tokens": benchmark_cfg.get("max_new_tokens"),
                "batch_size": benchmark_cfg.get("batch_size"),
                "modes": modes,
            },
            message="; ".join(axis_errors) if axis_errors else None,
        )
    )
    failures.extend(axis_errors)


def _check_scaling(
    benchmark_cfg: Mapping[str, Any],
    *,
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    errors: list[str] = []
    mode = benchmark_cfg.get("scale_resolution")
    resolution = int(benchmark_cfg.get("resolution", 0))
    target_scales = benchmark_cfg.get("target_scales")
    target_patch_size = benchmark_cfg.get("target_patch_size")
    alignment = benchmark_cfg.get("quick_start_alignment") if isinstance(benchmark_cfg.get("quick_start_alignment"), Mapping) else {}

    try:
        if mode in {"quick_start_spatio_temporal_224", "spatio_temporal_224"}:
            policy = resolve_autogaze_scaling_policy(
                mode="spatio_temporal",
                resolution=224,
                patch_size=16,
                temporal_chunk_size=int(benchmark_cfg.get("temporal_chunk_size", 16)),
                spatial_tile_size=224,
            )
        elif mode in {"quick_start_spatio_temporal_392", "spatio_temporal_392"}:
            policy = resolve_autogaze_scaling_policy(
                mode="spatio_temporal",
                resolution=392,
                patch_size=int(target_patch_size or alignment.get("medium_target_patch_size", 14)),
                target_scales=target_scales or alignment.get("medium_target_scales", [56, 112, 196, 392]),
                target_patch_size=int(target_patch_size or alignment.get("medium_target_patch_size", 14)),
                temporal_chunk_size=int(benchmark_cfg.get("temporal_chunk_size", 16)),
                spatial_tile_size=392,
            )
        elif mode == "quick_start_target_scales":
            policy = resolve_autogaze_scaling_policy(
                mode="resize",
                resolution=resolution,
                patch_size=int(target_patch_size or alignment.get("medium_target_patch_size", 14)),
                target_scales=target_scales,
                target_patch_size=target_patch_size,
            )
        elif mode is None:
            policy = resolve_autogaze_scaling_policy(mode="resize", resolution=resolution, patch_size=16)
        else:
            policy = None
    except ValueError as exc:
        policy = None
        errors.append(str(exc))

    if mode is None:
        if resolution != int(alignment.get("default_resolution", 224)):
            errors.append("unscaled benchmark should use QUICK_START default resolution")
        if target_scales is not None or target_patch_size is not None:
            errors.append("unscaled benchmark must not configure target_scales/target_patch_size")
    elif mode == "quick_start_target_scales":
        expected_scales = list(alignment.get("medium_target_scales", [56, 112, 196, 392]))
        if list(target_scales or []) != expected_scales:
            errors.append(f"target_scales must match QUICK_START high-res example: {expected_scales}")
        if expected_scales and resolution != int(expected_scales[-1]):
            errors.append(f"resolution must match largest QUICK_START target scale: {expected_scales[-1]}")
        if int(target_patch_size or 0) != int(alignment.get("medium_target_patch_size", 14)):
            errors.append("target_patch_size must match QUICK_START high-res example")
        if target_patch_size and resolution % int(target_patch_size) != 0:
            errors.append("resolution must be divisible by target_patch_size")
    elif mode in {"quick_start_spatio_temporal_224", "spatio_temporal_224"}:
        if int(benchmark_cfg.get("temporal_chunk_size", 16)) != int(alignment.get("default_frame_count", 16)):
            errors.append("spatio-temporal 224 mode should use QUICK_START 16-frame chunks")
    elif mode in {"quick_start_spatio_temporal_392", "spatio_temporal_392"}:
        expected_scales = list(alignment.get("medium_target_scales", [56, 112, 196, 392]))
        if list(target_scales or expected_scales) != expected_scales:
            errors.append(f"target_scales must match QUICK_START high-res example: {expected_scales}")
        if int(target_patch_size or 0) != int(alignment.get("medium_target_patch_size", 14)):
            errors.append("target_patch_size must match QUICK_START high-res example")
        if int(benchmark_cfg.get("temporal_chunk_size", 16)) != int(alignment.get("default_frame_count", 16)):
            errors.append("spatio-temporal 392 mode should use QUICK_START 16-frame chunks")
    else:
        errors.append(f"unsupported scale_resolution mode: {mode}")

    checks.append(
        DryRunCheck(
            name="quick_start_scaling_compatibility",
            status="failed" if errors else "passed",
            details={
                "scale_resolution": mode,
                "resolution": resolution,
                "target_scales": target_scales,
                "target_patch_size": target_patch_size,
                "quick_start_alignment": dict(alignment),
                "resolved_policy": policy.to_dict() if policy is not None else None,
            },
            message="; ".join(errors) if errors else None,
        )
    )
    failures.extend(errors)


def _check_video_input(
    benchmark_cfg: Mapping[str, Any],
    video_path: str | Path | None,
    *,
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    configured_path = video_path or benchmark_cfg.get("video_path")
    if configured_path:
        exists = Path(configured_path).expanduser().exists()
        status = "passed" if exists else "failed"
        message = None if exists else f"video input does not exist: {configured_path}"
        checks.append(
            DryRunCheck(
                name="video_input_availability",
                status=status,
                details={"source": "local_video", "path": configured_path, "exists": exists},
                message=message,
            )
        )
        if message:
            failures.append(message)
        return

    checks.append(
        DryRunCheck(
            name="video_input_availability",
            status="passed",
            details={"source": "dummy", "note": "canonical benchmark runner defaults to dummy video input"},
        )
    )


def _check_output_writable(
    output_dir: Path,
    *,
    checks: list[DryRunCheck],
    failures: list[str],
) -> None:
    probe = output_dir / ".dry_run_write_probe"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(
            DryRunCheck(
                name="output_directory_writability",
                status="passed",
                details={"output_dir": output_dir},
            )
        )
    except OSError as exc:
        message = f"output directory is not writable: {output_dir}: {exc}"
        checks.append(
            DryRunCheck(
                name="output_directory_writability",
                status="failed",
                details={"output_dir": output_dir},
                message=message,
            )
        )
        failures.append(message)


def _write_dry_run_report(report: BenchmarkDryRunReport) -> None:
    report.report_path.parent.mkdir(parents=True, exist_ok=True)
    report.report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def run_benchmark_dry_run(
    *,
    config_name: str,
    mode: str | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    output_dir: str | Path | None = None,
    device: str | None = None,
    dtype: str | None = None,
    video_path: str | Path | None = None,
    batch_size: int | None = None,
    num_frames: int | None = None,
    resolution: int | None = None,
    scale_resolution: str | None = None,
    max_new_tokens: int | None = None,
) -> BenchmarkDryRunReport:
    preset = _load_benchmark_preset(config_dir, config_name)
    benchmark_cfg = _benchmark_node(preset)
    if device is not None:
        benchmark_cfg["device"] = device
    if dtype is not None:
        benchmark_cfg["dtype"] = dtype
    if batch_size is not None:
        benchmark_cfg["batch_size"] = batch_size
    if num_frames is not None:
        benchmark_cfg["num_frames"] = num_frames
    if resolution is not None:
        benchmark_cfg["resolution"] = resolution
    if scale_resolution is not None:
        benchmark_cfg["scale_resolution"] = scale_resolution
    if max_new_tokens is not None:
        benchmark_cfg["max_new_tokens"] = max_new_tokens

    checks: list[DryRunCheck] = []
    failures: list[str] = []
    warnings: list[str] = []

    experiment_id = str(benchmark_cfg.get("experiment", ""))
    experiment_cfg = load_config(config_dir, f"experiment/{experiment_id}")
    plain_experiment_cfg = _to_plain_config(experiment_cfg)
    checks.append(
        DryRunCheck(
            name="config_loading",
            status="passed",
            details={
                "benchmark_config": config_name,
                "preset_id": benchmark_cfg.get("preset_id"),
                "experiment_config": f"experiment/{experiment_id}",
                "experiment_id": experiment_id,
            },
        )
    )

    if mode is not None and mode not in set(benchmark_cfg.get("modes", [])):
        message = f"mode {mode} is not supported by {config_name}"
        checks.append(
            DryRunCheck(
                name="mode_support",
                status="failed",
                details={"mode": mode, "supported_modes": benchmark_cfg.get("modes", [])},
                message=message,
            )
        )
        failures.append(message)
    else:
        checks.append(
            DryRunCheck(
                name="mode_support",
                status="passed",
                details={"mode": mode, "supported_modes": benchmark_cfg.get("modes", [])},
            )
        )

    required_components = _required_components(experiment_id)
    if not required_components:
        failures.append(f"unsupported canonical real experiment: {experiment_id}")
    for component in required_components:
        key = _component_node_key(component)
        node = _get_nested(plain_experiment_cfg, "model", key)
        if not isinstance(node, Mapping):
            failures.append(f"missing model.{key} config")
            continue
        _check_path(
            checks=checks,
            failures=failures,
            name=f"{component}_checkpoint_path",
            path=node.get("checkpoint"),
            required=True,
        )
        _check_path(
            checks=checks,
            failures=failures,
            name=f"{component}_model_config_path",
            path=node.get("model_config_path") or node.get("config_path"),
            required=True,
        )
        _check_path(
            checks=checks,
            failures=failures,
            name=f"{component}_config_path",
            path=node.get("config_path"),
            required=bool(node.get("config_path")),
        )
        processor_path = node.get("processor_path") or node.get("tokenizer_or_processor_path")
        _check_path(
            checks=checks,
            failures=failures,
            name=f"{component}_tokenizer_or_processor_path",
            path=processor_path,
            required=bool(processor_path),
        )

    _check_module_availability(
        plain_experiment_cfg,
        experiment_id,
        checks=checks,
        failures=failures,
    )
    _check_nvila_hd_video_guide_compatibility(
        config_dir=config_dir,
        benchmark_cfg=benchmark_cfg,
        experiment_cfg=plain_experiment_cfg,
        checks=checks,
        failures=failures,
    )

    requested_device = str(benchmark_cfg.get("device", "cpu"))
    available = _device_available(requested_device)
    device_message = None if available else f"device is not available: {requested_device}"
    checks.append(
        DryRunCheck(
            name="device_availability",
            status="passed" if available else "failed",
            details={
                "requested_device": requested_device,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": torch.backends.mps.is_available(),
            },
            message=device_message,
        )
    )
    if device_message:
        failures.append(device_message)

    requested_dtype = str(benchmark_cfg.get("dtype", "float32"))
    dtype_ok, dtype_error = _validate_dtype(requested_device, requested_dtype)
    checks.append(
        DryRunCheck(
            name="dtype_compatibility",
            status="passed" if dtype_ok else "failed",
            details={"device": requested_device, "dtype": requested_dtype},
            message=dtype_error,
        )
    )
    if dtype_error:
        failures.append(dtype_error)

    if bool(benchmark_cfg.get("allow_mllm_load", False)):
        warnings.append("allow_mllm_load is true; executing this config may instantiate a large NVILA model")

    _check_video_input(benchmark_cfg, video_path, checks=checks, failures=failures)

    root_output_dir = Path(output_dir or benchmark_cfg.get("output_dir") or "outputs/tiny_real_benchmarks")
    _check_output_writable(root_output_dir, checks=checks, failures=failures)
    _check_benchmark_axes(benchmark_cfg, checks=checks, failures=failures)
    _check_scaling(benchmark_cfg, checks=checks, failures=failures)

    quick_start = Path(config_dir).parent / "QUICK_START.md"
    quick_start_ref = Path(config_dir).parent / "docs" / "QUICK_START_reference.md"
    for label, path in {
        "quick_start_found": quick_start,
        "quick_start_reference_found": quick_start_ref,
    }.items():
        _check_path(checks=checks, failures=failures, name=label, path=path, required=True)

    report_path = root_output_dir / "dry_run_report.json"
    report = BenchmarkDryRunReport(
        config_name=config_name,
        preset_id=str(benchmark_cfg.get("preset_id", config_name)),
        experiment_id=experiment_id,
        mode=mode,
        safe_to_execute=not failures,
        report_path=report_path,
        checks=checks,
        failures=failures,
        warnings=warnings,
    )
    _write_dry_run_report(report)
    return report


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


def _write_auxiliary_reports(
    *,
    result: BenchmarkResult,
    reports: list[SmokeInferenceReport],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    stage_rows: list[dict[str, Any]] = []
    for iteration, report in enumerate(reports):
        for stage in report.stages:
            stage_rows.append(
                {
                    "iteration": iteration,
                    "stage": stage.name,
                    "status": stage.status,
                    "latency_ms": stage.latency_ms,
                    "output_shape": stage.output_shape,
                    "skipped_reason": stage.skipped_reason,
                    "details": stage.details,
                }
            )

    token_report = {
        "experiment_id": result.experiment_id,
        "autogaze": result.autogaze,
        "visual_token_count_before_autogaze": result.visual_token_count_before_autogaze,
        "visual_token_count_after_autogaze": result.visual_token_count_after_autogaze,
        "token_reduction_ratio": result.token_reduction_ratio,
        "selected_patches_per_frame": result.selected_patches_per_frame,
        "selected_patches_per_scale": result.selected_patches_per_scale,
        "input_frame_count": result.input_frame_count,
        "input_resolution": result.input_resolution,
        "note": result.acceleration_type_note,
    }

    skipped_report = {
        "experiment_id": result.experiment_id,
        "mode": result.metadata.get("benchmark_mode") if result.metadata else None,
        "skipped_stages": result.metadata.get("skipped_stages", []) if result.metadata else [],
        "warning": result.metadata.get("warning") if result.metadata else None,
        "stub_status": result.stub_status,
    }

    stage_latency_path = output_dir / "stage_latency_breakdown.json"
    token_counts_path = output_dir / "token_count_report.json"
    skipped_stages_path = output_dir / "skipped_stage_report.json"
    stage_latency_path.write_text(json.dumps(stage_rows, indent=2, sort_keys=True), encoding="utf-8")
    token_counts_path.write_text(json.dumps(token_report, indent=2, sort_keys=True), encoding="utf-8")
    skipped_stages_path.write_text(json.dumps(skipped_report, indent=2, sort_keys=True), encoding="utf-8")
    return stage_latency_path, token_counts_path, skipped_stages_path


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
    batch_size: int | None = None,
    num_frames: int | None = None,
    resolution: int | None = None,
    scale_resolution: str | None = None,
    max_new_tokens: int | None = None,
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
    if batch_size is not None:
        benchmark_cfg["batch_size"] = batch_size
    if num_frames is not None:
        benchmark_cfg["num_frames"] = num_frames
    if resolution is not None:
        benchmark_cfg["resolution"] = resolution
    if scale_resolution is not None:
        benchmark_cfg["scale_resolution"] = scale_resolution
    if max_new_tokens is not None:
        benchmark_cfg["max_new_tokens"] = max_new_tokens
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
    stage_latency_path, token_counts_path, skipped_stages_path = _write_auxiliary_reports(
        result=result,
        reports=reports,
        output_dir=mode_output_dir,
    )
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
        stage_latency_path=stage_latency_path,
        token_counts_path=token_counts_path,
        skipped_stages_path=skipped_stages_path,
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
    print(f"stage-level latency breakdown: {artifact.stage_latency_path}")
    print(f"token count report: {artifact.token_counts_path}")
    print(f"skipped-stage report: {artifact.skipped_stages_path}")
    if result.metadata:
        print(f"executed stages: {', '.join(result.metadata.get('executed_stages', [])) or 'none'}")
        skipped = result.metadata.get("skipped_stages", [])
        if skipped:
            print(f"skipped stages: {json.dumps(skipped, sort_keys=True)}")


def _print_dry_run_report(report: BenchmarkDryRunReport) -> None:
    print("DRY RUN: no models instantiated, no checkpoint tensors loaded, no inference executed")
    print(f"config: {report.config_name}")
    print(f"experiment ID: {report.experiment_id}")
    print(f"mode: {report.mode or 'all_configured_modes'}")
    print(f"safe_to_execute: {report.safe_to_execute}")
    print(f"report: {report.report_path}")
    if report.failures:
        print("failures:")
        for item in report.failures:
            print(f"- {item}")
    if report.warnings:
        print("warnings:")
        for item in report.warnings:
            print(f"- {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded real A1/A2 benchmark presets")
    parser.add_argument(
        "--config-name",
        required=True,
        help=(
            "Benchmark preset name under configs/benchmark, for example tiny_a1_real, "
            "canonical_a1_small, canonical_a2_medium, or canonical_all for dry-run validation."
        ),
    )
    parser.add_argument("--mode", choices=["autogaze_only", "full_pipeline"], default=None)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--warmup-iterations", type=int, default=None)
    parser.add_argument("--benchmark-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--scale-resolution", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--allow-mllm-load", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate benchmark readiness without running inference")
    parser.add_argument("--video-path", default=None, help="Optional local video path to validate for dry-run")
    args = parser.parse_args()

    if args.dry_run:
        config_names = CANONICAL_BENCHMARK_CONFIGS if args.config_name == "canonical_all" else (args.config_name,)
        reports = [
            run_benchmark_dry_run(
                config_name=config_name,
                mode=args.mode,
                config_dir=args.config_dir,
                output_dir=(Path(args.output_dir) / config_name) if args.output_dir and len(config_names) > 1 else args.output_dir,
                device=args.device,
                dtype=args.dtype,
                video_path=args.video_path,
                batch_size=args.batch_size,
                num_frames=args.num_frames,
                resolution=args.resolution,
                scale_resolution=args.scale_resolution,
                max_new_tokens=args.max_new_tokens,
            )
            for config_name in config_names
        ]
        for report in reports:
            _print_dry_run_report(report)
        if any(not report.safe_to_execute for report in reports):
            raise SystemExit(1)
        return

    if args.mode is None:
        parser.error("--mode is required unless --dry-run is set")

    artifact = run_tiny_real_benchmark(
        config_name=args.config_name,
        mode=args.mode,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        resolution=args.resolution,
        scale_resolution=args.scale_resolution,
        max_new_tokens=args.max_new_tokens,
        allow_mllm_load=True if args.allow_mllm_load else None,
    )
    _print_artifact(artifact)


if __name__ == "__main__":
    main()
