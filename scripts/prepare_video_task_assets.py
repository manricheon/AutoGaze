from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_REVISION = "main"

MODEL_PRESETS: dict[str, list[dict[str, str]]] = {
    "none": [],
    "qwen-video-task": [
        {
            "name": "qwen3-vl-8b",
            "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
            "local_subdir": "Qwen3-VL-8B-Instruct",
        },
        {
            "name": "autogaze",
            "repo_id": "nvidia/AutoGaze",
            "local_subdir": "AutoGaze",
        },
    ],
    "qwen-compare": [
        {
            "name": "qwen2.5-vl-7b",
            "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "local_subdir": "Qwen2.5-VL-7B-Instruct",
        },
        {
            "name": "qwen3-vl-8b",
            "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
            "local_subdir": "Qwen3-VL-8B-Instruct",
        },
        {
            "name": "autogaze",
            "repo_id": "nvidia/AutoGaze",
            "local_subdir": "AutoGaze",
        },
    ],
    "expand-smoke": [
        {
            "name": "qwen2.5-vl-7b",
            "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "local_subdir": "Qwen2.5-VL-7B-Instruct",
        },
        {
            "name": "qwen3-vl-8b",
            "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
            "local_subdir": "Qwen3-VL-8B-Instruct",
        },
        {
            "name": "autogaze",
            "repo_id": "nvidia/AutoGaze",
            "local_subdir": "AutoGaze",
        },
        {
            "name": "nvila-hd-video",
            "repo_id": "nvidia/NVILA-8B-HD-Video",
            "local_subdir": "NVILA-8B-HD-Video",
        },
        {
            "name": "nvila-video",
            "repo_id": "Efficient-Large-Model/NVILA-8B-Video",
            "local_subdir": "NVILA-8B-Video",
        },
        {
            "name": "llava-onevision",
            "repo_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "local_subdir": "LLaVA-OneVision-Qwen2-7B-OV",
        },
        {
            "name": "internvl3-8b",
            "repo_id": "OpenGVLab/InternVL3-8B",
            "local_subdir": "InternVL3-8B",
        },
    ],
}

DATASET_PRESETS: dict[str, list[dict[str, str]]] = {
    "none": [],
    "custom": [],
    "caption-action-smoke": [
        {
            "name": "msrvtt-caption",
            "repo_id": "VLM2Vec/MSR-VTT",
            "local_subdir": "msrvtt",
        },
        {
            "name": "ucf101-action",
            "repo_id": "bitmind/UCF101-Videos",
            "local_subdir": "ucf101-videos",
        },
    ],
    "videoqa-smoke": [
        {
            "name": "egoschema-videoqa",
            "repo_id": "VLM2Vec/EgoSchema",
            "local_subdir": "egoschema",
        },
        {
            "name": "nextqa-videoqa",
            "repo_id": "VLM2Vec/nextqa",
            "local_subdir": "nextqa",
        },
        {
            "name": "videomme-videoqa",
            "repo_id": "vid-modeling/videomme",
            "local_subdir": "videomme",
        },
        {
            "name": "activitynet-videoqa",
            "repo_id": "VLM2Vec/ActivityNetQA",
            "local_subdir": "activitynetqa",
        },
    ],
}


def parse_asset_spec(value: str, *, repo_type: str) -> dict[str, str]:
    if "=" not in value:
        raise ValueError(f"asset spec must be name=repo_id[@revision], got: {value}")
    name, repo_and_revision = value.split("=", 1)
    if not name.strip() or not repo_and_revision.strip():
        raise ValueError(f"asset spec must be name=repo_id[@revision], got: {value}")
    if "@" in repo_and_revision:
        repo_id, revision = repo_and_revision.rsplit("@", 1)
    else:
        repo_id, revision = repo_and_revision, DEFAULT_REVISION
    if not repo_id.strip() or not revision.strip():
        raise ValueError(f"asset spec must be name=repo_id[@revision], got: {value}")
    return {
        "name": name.strip(),
        "repo_id": repo_id.strip(),
        "repo_type": repo_type,
        "revision": revision.strip(),
        "local_subdir": name.strip(),
    }


def _preset_assets(preset: str, *, repo_type: str) -> list[dict[str, str]]:
    preset_map = DATASET_PRESETS if repo_type == "dataset" else MODEL_PRESETS
    if preset not in preset_map:
        choices = ", ".join(sorted(preset_map))
        raise ValueError(f"unknown {repo_type} preset: {preset}. choices: {choices}")
    rows: list[dict[str, str]] = []
    for row in preset_map[preset]:
        rows.append(
            {
                "name": row["name"],
                "repo_id": row["repo_id"],
                "repo_type": repo_type,
                "revision": row.get("revision", DEFAULT_REVISION),
                "local_subdir": row["local_subdir"],
            }
        )
    return rows


def _asset_download_row(
    asset: dict[str, str],
    *,
    root: Path,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    max_workers: int,
) -> dict[str, Any]:
    return {
        "name": asset["name"],
        "repo_id": asset["repo_id"],
        "repo_type": asset["repo_type"],
        "revision": asset.get("revision", DEFAULT_REVISION),
        "local_dir": str(root / asset["local_subdir"]),
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "max_workers": int(max_workers),
    }


def build_asset_plan(args: argparse.Namespace) -> dict[str, Any]:
    dataset_assets = _preset_assets(args.dataset_preset, repo_type="dataset")
    dataset_assets.extend(parse_asset_spec(row, repo_type="dataset") for row in args.datasets)
    model_assets = _preset_assets(args.model_preset, repo_type="model")
    model_assets.extend(parse_asset_spec(row, repo_type="model") for row in args.models)
    local_root = Path(args.local_root)
    weight_root = Path(args.weight_root)
    return {
        "datasets": [
            _asset_download_row(
                row,
                root=local_root,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                max_workers=args.max_workers,
            )
            for row in dataset_assets
        ],
        "models": [
            _asset_download_row(
                row,
                root=weight_root,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                max_workers=args.max_workers,
            )
            for row in model_assets
        ],
        "notes": [
            "Use --dataset-preset caption-action-smoke for MSR-VTT/UCF101, --dataset-preset videoqa-smoke for EgoSchema/NextQA/VideoMME/ActivityNetQA, or pass --dataset name=org/repo for custom HF datasets.",
            "Use --dry-run on the CUDA machine first to check local_dir, repo_type, and revision before large downloads.",
        ],
    }


def download_asset_plan(plan: dict[str, Any]) -> dict[str, Any]:
    downloaded: list[dict[str, Any]] = []
    for row in [*(plan.get("datasets") or []), *(plan.get("models") or [])]:
        local_dir = Path(row["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(
            repo_id=row["repo_id"],
            repo_type=row["repo_type"],
            revision=row["revision"],
            local_dir=str(local_dir),
            allow_patterns=row.get("allow_patterns"),
            ignore_patterns=row.get("ignore_patterns"),
            max_workers=row["max_workers"],
        )
        downloaded.append({**row, "snapshot_path": str(snapshot_path)})
    return {**plan, "downloaded": downloaded}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Hugging Face datasets and model weights for AutoGaze video-task benchmarks."
    )
    parser.add_argument("--local-root", default="inputs/video_tasks")
    parser.add_argument("--weight-root", default="weight")
    parser.add_argument("--dataset-preset", default="custom", choices=sorted(DATASET_PRESETS))
    parser.add_argument("--model-preset", default="qwen-video-task", choices=sorted(MODEL_PRESETS))
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        dest="datasets",
        help="HF dataset spec as name=org/repo[@revision]. Local dir becomes LOCAL_ROOT/name.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        help="HF model spec as name=org/repo[@revision]. Local dir becomes WEIGHT_ROOT/name.",
    )
    parser.add_argument("--include", action="append", dest="allow_patterns")
    parser.add_argument("--exclude", action="append", dest="ignore_patterns")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plan = build_asset_plan(args)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return
    print(json.dumps(download_asset_plan(plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
