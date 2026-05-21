from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT_DIR = "weight/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "main"


def build_download_plan(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    max_workers: int,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "repo_type": "model",
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "max_workers": int(max_workers),
    }


def download_qwen_model(
    *,
    repo_id: str,
    output_dir: Path,
    revision: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    max_workers: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        max_workers=max_workers,
    )
    return {
        **build_download_plan(
            repo_id=repo_id,
            output_dir=output_dir,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
        ),
        "snapshot_path": str(snapshot_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Qwen3-VL model used by flexible_runner")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include", action="append", dest="allow_patterns")
    parser.add_argument("--exclude", action="append", dest="ignore_patterns")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plan = build_download_plan(
        repo_id=args.repo_id,
        output_dir=Path(args.output_dir),
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.max_workers,
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return

    result = download_qwen_model(
        repo_id=args.repo_id,
        output_dir=Path(args.output_dir),
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
        max_workers=args.max_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
