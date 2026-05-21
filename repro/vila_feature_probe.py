from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_json


FEATURE_PACKING_KEYWORDS = (
    "vision",
    "video",
    "image",
    "mm",
    "projector",
    "patch",
    "token",
    "tower",
    "select",
)

PROBE_TARGETS = [
    "processor video tensor/frame contract",
    "vision feature shape",
    "projector output shape",
    "LLM visual token insertion boundary",
]


def run_vila_feature_probe(
    *,
    model_path: str,
    model_family: str,
    video: str | None,
    prompt: str,
    num_video_frames: int,
    max_tiles_video: int,
) -> dict[str, Any]:
    model_dir = Path(model_path)
    config_path = model_dir / "config.json"
    payload: dict[str, Any] = {
        "runner": "vila_feature_probe",
        "model_family": model_family,
        "model_path": str(model_dir),
        "video": video,
        "prompt": prompt,
        "num_video_frames": int(num_video_frames),
        "max_tiles_video": int(max_tiles_video),
        "probe_targets": PROBE_TARGETS,
        "runtime_probe_required": True,
        "next_action": "instrument_vila_remote_code_feature_packing",
    }
    if not config_path.is_file():
        return {
            **payload,
            "status": "config_missing",
            "config_path": str(config_path),
            "config_summary": None,
            "reason": "model config.json was not found; runtime feature packing probe still required",
        }

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        **payload,
        "status": "static_probe_collected",
        "config_path": str(config_path),
        "config_summary": summarize_vila_config(config),
    }


def summarize_vila_config(config: dict[str, Any]) -> dict[str, Any]:
    related_keys = [
        key
        for key in sorted(config)
        if any(keyword in key.lower() for keyword in FEATURE_PACKING_KEYWORDS)
    ]
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "feature_packing_related_keys": related_keys,
        "feature_packing_related_values": {
            key: _json_scalar_or_shape(config.get(key)) for key in related_keys
        },
    }


def _json_scalar_or_shape(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if len(value) <= 12 and all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
            return value
        return {"type": "list", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value)[:24]}
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a static VILA-family feature packing probe")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--video")
    parser.add_argument("--prompt", default="Describe the video.")
    parser.add_argument("--num-video-frames", type=int, default=128)
    parser.add_argument("--max-tiles-video", type=int, default=1)
    parser.add_argument("--output-json", default="outputs/autogaze_repro/vila_feature_probe.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_vila_feature_probe(
        model_path=args.model_path,
        model_family=args.model_family,
        video=args.video,
        prompt=args.prompt,
        num_video_frames=args.num_video_frames,
        max_tiles_video=args.max_tiles_video,
    )
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
