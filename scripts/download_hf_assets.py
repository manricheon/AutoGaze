#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROCESSOR_TOKENIZER_PATTERNS = [
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "*.txt",
]


@dataclass
class AssetRecord:
    asset_type: str
    repo_id: str
    revision: str | None
    cache_dir: str | None
    cache_path: str | None
    include_processor_tokenizer: bool = False
    dry_run: bool = False


@dataclass
class AssetManifest:
    timestamp: str
    dry_run: bool
    cache_dir: str | None
    token_env_var: str
    token_present: bool
    models: list[dict[str, Any]] = field(default_factory=list)
    datasets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _token_from_env(token_env_var: str) -> str | None:
    return os.environ.get(token_env_var) if token_env_var else None


def _snapshot_download(
    *,
    repo_id: str,
    repo_type: str | None,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
    include_processor_tokenizer: bool,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download Hugging Face assets") from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "repo_type": repo_type,
    }
    if include_processor_tokenizer and repo_type is None:
        kwargs["allow_patterns"] = PROCESSOR_TOKENIZER_PATTERNS
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return str(snapshot_download(**kwargs))


def build_manifest(
    *,
    model_id: str | None,
    dataset_id: str | None,
    revision: str | None,
    cache_dir: str | None,
    token_env_var: str,
    include_processor_tokenizer: bool,
    dry_run: bool,
) -> AssetManifest:
    token = _token_from_env(token_env_var)
    manifest = AssetManifest(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dry_run=dry_run,
        cache_dir=cache_dir,
        token_env_var=token_env_var,
        token_present=token is not None,
    )

    if model_id:
        cache_path = None
        if not dry_run:
            cache_path = _snapshot_download(
                repo_id=model_id,
                repo_type=None,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                include_processor_tokenizer=include_processor_tokenizer,
            )
        manifest.models.append(
            asdict(
                AssetRecord(
                    asset_type="model",
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    cache_path=cache_path,
                    include_processor_tokenizer=include_processor_tokenizer,
                    dry_run=dry_run,
                )
            )
        )

    if dataset_id:
        cache_path = None
        if not dry_run:
            cache_path = _snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                include_processor_tokenizer=False,
            )
        manifest.datasets.append(
            asdict(
                AssetRecord(
                    asset_type="dataset",
                    repo_id=dataset_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    cache_path=cache_path,
                    dry_run=dry_run,
                )
            )
        )

    return manifest


def write_manifest(manifest: AssetManifest, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face model/dataset assets and write a manifest")
    parser.add_argument("--model-id", default=None, help="Hugging Face model repository ID")
    parser.add_argument("--dataset-id", default=None, help="Hugging Face dataset repository ID")
    parser.add_argument("--revision", default=None, help="Revision, branch, tag, or commit hash")
    parser.add_argument("--cache-dir", default=None, help="Local Hugging Face cache directory")
    parser.add_argument(
        "--manifest-out",
        default="outputs/hf_assets/manifest.json",
        help="Path to write the asset manifest JSON",
    )
    parser.add_argument(
        "--include-processor-tokenizer",
        action="store_true",
        help="For model assets, download only common processor/tokenizer files via allow_patterns",
    )
    parser.add_argument("--token-env-var", default="HF_TOKEN", help="Environment variable containing an access token")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest without downloading assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_id and not args.dataset_id:
        raise SystemExit("At least one of --model-id or --dataset-id is required")

    manifest = build_manifest(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token_env_var=args.token_env_var,
        include_processor_tokenizer=args.include_processor_tokenizer,
        dry_run=args.dry_run,
    )
    path = write_manifest(manifest, args.manifest_out)
    print(f"manifest: {path}")


if __name__ == "__main__":
    main()
