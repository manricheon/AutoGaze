from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_json


def run_internvl_dynamic_tile_probe(
    *,
    model_path: str,
    model_family: str,
    video: str | None,
    num_video_frames: int,
    max_tiles_video: int,
) -> dict[str, Any]:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    base: dict[str, Any] = {
        "runner": "internvl_dynamic_tile_probe",
        "model_family": model_family,
        "model_path": str(model_dir),
        "video": video,
        "num_video_frames": int(num_video_frames),
        "max_tiles_video": int(max_tiles_video),
        "runtime_probe_required": True,
        "next_action": "instrument_internvl_dynamic_tile_preprocess_and_num_patches_list",
    }
    if not config_path.is_file():
        return {
            **base,
            "status": "config_missing",
            "config_path": str(config_path),
            "config_summary": None,
            "dynamic_tile_probe": {
                "status": "config_missing",
                "reason": "model config.json was not found",
            },
        }

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        **base,
        "status": "static_probe_collected",
        "config_path": str(config_path),
        "config_summary": summarize_internvl_config(config),
        "dynamic_tile_probe": build_internvl_dynamic_tile_probe(config, max_tiles_video=max_tiles_video),
    }


def summarize_internvl_config(config: dict[str, Any]) -> dict[str, Any]:
    vision_config = config.get("vision_config") or {}
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "image_size": _first_not_none(vision_config.get("image_size"), config.get("image_size")),
        "patch_size": _first_not_none(vision_config.get("patch_size"), config.get("patch_size")),
        "downsample_ratio": config.get("downsample_ratio"),
        "dynamic_image_size": config.get("dynamic_image_size"),
        "use_thumbnail": config.get("use_thumbnail"),
        "min_dynamic_patch": config.get("min_dynamic_patch"),
        "max_dynamic_patch": config.get("max_dynamic_patch"),
    }


def build_internvl_dynamic_tile_probe(config: dict[str, Any], *, max_tiles_video: int) -> dict[str, Any]:
    summary = summarize_internvl_config(config)
    image_size = _as_int(summary.get("image_size"))
    patch_size = _as_int(summary.get("patch_size"))
    tokens_per_tile = None
    if image_size and patch_size:
        tokens_per_tile = (image_size // patch_size) ** 2
    return {
        "status": "dynamic_tile_probe_required",
        "integration_level": "pre_encoder_sparse",
        "model_grid_fields": ["pixel_values", "num_patches_list"],
        "tile_level_strategy": "select_dynamic_tiles_before_vit",
        "patch_level_strategy": "requires_patch_row_col_within_each_dynamic_tile",
        "position_alignment": {
            "status": "must_preserve_dynamic_tile_order",
            "fields": ["frame_order", "tile_order", "patch_row", "patch_col"],
        },
        "estimated_tokens_per_tile_before_downsample": tokens_per_tile,
        "max_tiles_video": int(max_tiles_video),
        "thumbnail_policy": (
            "keep_all_until_thumbnail_mapping_is_verified"
            if summary.get("use_thumbnail") is True
            else "not_configured"
        ),
        "next_actions": [
            "capture dynamic preprocess tile order",
            "capture num_patches_list per frame/video",
            "map SparseSelectionPlan patches to tile_id and patch row/col",
            "apply tile-level prune before patch-level prune",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a static InternVL dynamic tile probe")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-family", default="internvl3")
    parser.add_argument("--video")
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--output-json", default="outputs/autogaze_repro/internvl_dynamic_tile_probe.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_internvl_dynamic_tile_probe(
        model_path=args.model_path,
        model_family=args.model_family,
        video=args.video,
        num_video_frames=args.num_video_frames,
        max_tiles_video=args.max_tiles_video,
    )
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
