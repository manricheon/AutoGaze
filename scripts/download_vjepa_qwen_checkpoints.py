from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_VJEPA_MODEL = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_ROOT = "/content/autogaze_weights"
DEFAULT_REVISION = "main"


def build_download_plan(
    *,
    vjepa_model: str,
    qwen_model: str,
    output_root: Path,
    revision: str,
    max_workers: int,
) -> dict[str, Any]:
    return {
        "revision": revision,
        "repo_type": "model",
        "output_root": str(output_root),
        "max_workers": int(max_workers),
        "models": {
            "vjepa": {
                "repo_id": vjepa_model,
                "local_dir": str(output_root / _safe_local_name(vjepa_model)),
            },
            "qwen": {
                "repo_id": qwen_model,
                "local_dir": str(output_root / _safe_local_name(qwen_model)),
            },
        },
    }


def download_checkpoints(
    *,
    vjepa_model: str,
    qwen_model: str,
    output_root: Path,
    revision: str,
    max_workers: int,
) -> dict[str, Any]:
    plan = build_download_plan(
        vjepa_model=vjepa_model,
        qwen_model=qwen_model,
        output_root=output_root,
        revision=revision,
        max_workers=max_workers,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result = {**plan, "models": {key: dict(value) for key, value in plan["models"].items()}}
    for key, spec in result["models"].items():
        snapshot_path = snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="model",
            revision=revision,
            local_dir=spec["local_dir"],
            max_workers=max_workers,
        )
        result["models"][key]["snapshot_path"] = str(snapshot_path)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download V-JEPA and Qwen checkpoints for Colab smoke tests.")
    parser.add_argument("--vjepa-model", default=DEFAULT_VJEPA_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.dry_run:
        result = {"dry_run": True, **build_download_plan(
            vjepa_model=args.vjepa_model,
            qwen_model=args.qwen_model,
            output_root=output_root,
            revision=args.revision,
            max_workers=args.max_workers,
        )}
    else:
        result = download_checkpoints(
            vjepa_model=args.vjepa_model,
            qwen_model=args.qwen_model,
            output_root=output_root,
            revision=args.revision,
            max_workers=args.max_workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _safe_local_name(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "__")


if __name__ == "__main__":
    main()
