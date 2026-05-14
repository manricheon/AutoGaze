#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from poc_infer_utils import load_config, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODEL_ORDER = [
    "llava_ov",
    "longva",
    "longvila_r1",
    "apollo",
    "videollama3",
    "videochat_flash",
    "internvl3_5",
    "qwen2_5_vl",
    "vjepa2",
]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def resolve_repo_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or REPO_ROOT) / path


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_config(path)
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise TypeError("asset manifest must contain a mapping at key 'models'")
    return manifest


def select_model_names(manifest: Mapping[str, Any], requested: list[str] | None) -> list[str]:
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise TypeError("asset manifest must contain a mapping at key 'models'")
    if not requested or "all" in requested:
        return [name for name in TARGET_MODEL_ORDER if name in models]
    missing = [name for name in requested if name not in models]
    if missing:
        raise ValueError(f"Unknown model(s) in manifest: {missing}; valid names: {sorted(models)}")
    return requested


def select_models_by_tier(manifest: Mapping[str, Any], tiers: list[str] | None) -> list[str]:
    if not tiers:
        return []
    tier_map = manifest.get("tiers")
    if not isinstance(tier_map, Mapping):
        raise TypeError("asset manifest must contain a mapping at key 'tiers' when --tier is used")
    selected: list[str] = []
    for tier in tiers:
        if tier not in tier_map:
            raise ValueError(f"Unknown tier {tier!r}; valid tiers: {sorted(tier_map)}")
        tier_entry = tier_map[tier]
        models = tier_entry.get("models") if isinstance(tier_entry, Mapping) else None
        if not isinstance(models, list):
            raise TypeError(f"manifest tier {tier!r} must contain a list at key 'models'")
        for name in models:
            if str(name) not in selected:
                selected.append(str(name))
    return selected


def prioritized_model_names(manifest: Mapping[str, Any], names: list[str]) -> list[str]:
    return sorted(
        names,
        key=lambda item: (
            _tier_order(model_entry(manifest, item).get("tier")),
            int(model_entry(manifest, item).get("priority_rank", 999)),
            item,
        ),
    )


def model_entry(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    models = manifest.get("models")
    if not isinstance(models, Mapping) or name not in models:
        raise ValueError(f"No manifest entry for {name!r}")
    entry = dict(models[name])
    entry.setdefault("name", name)
    return entry


def local_dir_for_entry(entry: Mapping[str, Any], *, weights_root: str | Path | None = None) -> Path:
    raw = str(entry.get("local_target_directory") or f"weights/{entry.get('name')}")
    path = Path(raw)
    if weights_root is not None and not path.is_absolute():
        parts = path.parts
        if parts and parts[0] == "weights":
            return resolve_repo_path(Path(weights_root).joinpath(*parts[1:]))
    return resolve_repo_path(path)


def expected_size_gb(entry: Mapping[str, Any]) -> float | None:
    value = entry.get("expected_checkpoint_size_gb")
    if value in {None, "", "unknown"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_local_assets(
    entry: Mapping[str, Any],
    *,
    weights_root: str | Path | None = None,
    config_root: str | Path | None = None,
) -> dict[str, Any]:
    local_dir = local_dir_for_entry(entry, weights_root=weights_root)
    exists = local_dir.exists() and local_dir.is_dir()
    config_files = list(entry.get("expected_config_files") or ["config.json"])
    processor_files = list(entry.get("expected_processor_tokenizer_files") or [])
    config_status = _file_expectation_status(local_dir, config_files)
    processor_status = _file_expectation_status(local_dir, processor_files)
    weights_status = _weight_status(local_dir)
    name = str(entry.get("name"))
    config_base = resolve_repo_path(config_root or "configs/poc_inference/external")
    config_example = config_base / f"smoke_{name}.yaml"
    status = "local_exists" if exists and config_status["ok"] and processor_status["ok"] and weights_status["ok"] else "missing"
    if exists and weights_status.get("partial_download"):
        status = "blocked"
    return {
        "model": name,
        "local_dir": str(local_dir),
        "local_exists": exists,
        "download_status": status,
        "config_status": config_status,
        "processor_tokenizer_status": processor_status,
        "weights_status": weights_status,
        "config_example": str(config_example),
        "config_example_exists": config_example.exists(),
    }


def inspect_local_config(entry: Mapping[str, Any], *, weights_root: str | Path | None = None) -> dict[str, Any]:
    local_dir = local_dir_for_entry(entry, weights_root=weights_root)
    config_path = local_dir / "config.json"
    result: dict[str, Any] = {
        "model": entry.get("name"),
        "local_dir": str(local_dir),
        "config_path": str(config_path),
        "status": "missing",
        "reason": None,
    }
    if not config_path.exists():
        result["reason"] = "config.json not found"
        return result
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = f"failed to parse config.json: {exc}"
        return result
    key_map = _recursive_key_map(config)
    vision_config = config.get("vision_config") or config.get("vision_config_dict") or config.get("vision_tower_cfg")
    result.update(
        {
            "status": "inspected",
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "vision_config_visible": isinstance(vision_config, Mapping),
            "vision_config": _compact_mapping(vision_config) if isinstance(vision_config, Mapping) else None,
            "patch_size": _first_key_value(key_map, "patch_size"),
            "spatial_patch_size": _first_key_value(key_map, "spatial_patch_size"),
            "temporal_patch_size": _first_key_value(key_map, "temporal_patch_size"),
            "image_size": _first_key_value(key_map, "image_size"),
            "crop_size": _first_key_value(key_map, "crop_size"),
            "frames_per_clip": _first_key_value(key_map, "frames_per_clip"),
            "tubelet_size": _first_key_value(key_map, "tubelet_size"),
            "hidden_size": _first_key_value(key_map, "hidden_size"),
            "vision_hidden_size": _first_key_value(key_map, "vision_hidden_size"),
            "rope_indicators": _matching_key_values(key_map, ("rope", "mrope", "rotary")),
            "window_attention_indicators": _matching_key_values(key_map, ("window", "spatial_merge")),
            "projector_connector_indicators": _matching_key_values(
                key_map,
                ("projector", "connector", "resampler", "perceiver", "qformer", "merger", "pixel_shuffle"),
            ),
            "config_inspection_scope": "config_only_no_weight_load",
        }
    )
    return result


def write_markdown_report(path: str | Path, title: str, rows: list[Mapping[str, Any]], *, columns: list[str]) -> None:
    report_path = resolve_repo_path(path)
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_markdown_cell(row.get(col)) for col in columns) + " |")
    else:
        lines.append("No rows.")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(path: str | Path, data: Mapping[str, Any]) -> None:
    write_json(resolve_repo_path(path), data)


def _file_expectation_status(local_dir: Path, expected_files: list[str]) -> dict[str, Any]:
    if not expected_files:
        return {"ok": True, "expected": [], "present": [], "missing": []}
    present = [name for name in expected_files if (local_dir / name).exists()]
    missing = [name for name in expected_files if name not in present]
    return {"ok": not missing, "expected": expected_files, "present": present, "missing": missing}


def _weight_status(local_dir: Path) -> dict[str, Any]:
    if not local_dir.exists():
        return {"ok": False, "files": [], "index_files": [], "missing_shards": [], "partial_download": False}
    weight_files = sorted(
        str(path.relative_to(local_dir))
        for path in local_dir.iterdir()
        if path.is_file() and path.suffix in WEIGHT_SUFFIXES
    )
    index_files = [
        name
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (local_dir / name).exists()
    ]
    missing_shards: list[str] = []
    for index_name in index_files:
        try:
            index = json.loads((local_dir / index_name).read_text(encoding="utf-8"))
        except Exception:
            missing_shards.append(f"{index_name}:unreadable")
            continue
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if isinstance(weight_map, Mapping):
            for shard in sorted(set(str(value) for value in weight_map.values())):
                if not (local_dir / shard).exists():
                    missing_shards.append(shard)
    partial_markers = sorted(path.name for path in local_dir.glob("*.incomplete"))
    return {
        "ok": bool(weight_files) and not missing_shards and not partial_markers,
        "files": weight_files,
        "index_files": index_files,
        "missing_shards": missing_shards,
        "partial_markers": partial_markers,
        "partial_download": bool(missing_shards or partial_markers),
    }


def _recursive_key_map(value: Any) -> dict[str, list[Any]]:
    key_map: dict[str, list[Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                key_map.setdefault(str(key), []).append(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return key_map


def _first_key_value(key_map: Mapping[str, list[Any]], key: str) -> Any:
    values = key_map.get(key) or []
    for value in values:
        if not isinstance(value, (dict, list)):
            return value
    return None


def _matching_key_values(key_map: Mapping[str, list[Any]], needles: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, values in key_map.items():
        lower = key.lower()
        if any(needle in lower for needle in needles):
            simple_values = [value for value in values if not isinstance(value, (dict, list))]
            result[key] = simple_values[:4] if simple_values else "present_nested"
    return result


def _compact_mapping(value: Mapping[str, Any], *, limit: int = 24) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= limit:
            result["..."] = "truncated"
            break
        if isinstance(item, Mapping):
            result[str(key)] = _compact_mapping(item, limit=8)
        elif isinstance(item, list):
            result[str(key)] = item[:8]
        else:
            result[str(key)] = item
    return result


def _markdown_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _tier_order(tier: Any) -> int:
    order = {"tier1": 0, "tier1b": 1, "tier2": 2}
    return order.get(str(tier), 99)
