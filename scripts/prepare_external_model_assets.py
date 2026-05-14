#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from external_model_asset_utils import (
    expected_size_gb,
    load_asset_manifest,
    local_dir_for_entry,
    model_entry,
    prioritized_model_names,
    select_models_by_tier,
    select_model_names,
    validate_local_assets,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or explicitly download external PoC model assets.")
    parser.add_argument("--manifest", default="configs/poc_inference/model_asset_manifest.yaml")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--tier", action="append", choices=["tier1", "tier1b", "tier2"], default=None)
    parser.add_argument("--select-top-k", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-selected", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-root", default="weights")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-gated", action="store_true")
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    parser.add_argument("--max-total-gb", type=float, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-report", default="docs/MODEL_ASSET_DOWNLOAD_REPORT.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_asset_manifest(args.manifest)
    requested = args.model or ["all"]
    tier_names = select_models_by_tier(manifest, args.tier)
    names = tier_names if tier_names else select_model_names(manifest, requested)
    names = prioritized_model_names(manifest, names)
    if args.select_top_k is not None:
        if args.select_top_k <= 0:
            raise SystemExit("--select-top-k must be > 0")
        names = names[: args.select_top_k]
    if args.download_selected:
        args.download = True
    total_known_gb = sum(size for name in names if (size := expected_size_gb(model_entry(manifest, name))) is not None)
    unknown_sizes = [name for name in names if expected_size_gb(model_entry(manifest, name)) is None]
    if ("all" in requested or args.tier) and args.max_total_gb is not None and total_known_gb > args.max_total_gb:
        raise SystemExit(
            f"Estimated known model size {total_known_gb:.2f} GB exceeds --max-total-gb {args.max_total_gb:.2f}. "
            "Run with a smaller --model selection or a larger limit."
        )

    dry_run = args.dry_run or not args.download
    rows: list[dict[str, Any]] = []
    failures = 0
    for name in names:
        entry = model_entry(manifest, name)
        row = _prepare_one(entry, args=args, dry_run=dry_run)
        rows.append(row)
        if row.get("result") in {"failed", "blocked"} and args.download:
            failures += 1

    columns = [
        "model",
        "requested_action",
        "result",
        "tier",
        "priority_rank",
        "local_status",
        "local_dir",
        "remote_model_id",
        "gated_private_status",
        "storage_gb",
        "reason",
    ]
    write_markdown_report(
        args.write_report,
        "External Model Asset Download Dry-Run Report" if dry_run else "External Model Asset Download Report",
        rows,
        columns=columns,
    )
    if unknown_sizes:
        print(f"Storage size unknown for: {', '.join(unknown_sizes)}")
    print(f"Wrote report: {args.write_report}")
    return 1 if failures else 0


def _prepare_one(entry: Mapping[str, Any], *, args: argparse.Namespace, dry_run: bool) -> dict[str, Any]:
    name = str(entry.get("name"))
    local_status = validate_local_assets(entry, weights_root=args.output_root)
    local_dir = local_dir_for_entry(entry, weights_root=args.output_root)
    model_id = str(entry.get("candidate_model_id") or "")
    gated_status = str(entry.get("gated_private_status") or "unknown")
    storage = expected_size_gb(entry)
    base_row = {
        "model": name,
        "requested_action": "dry-run" if dry_run else "download",
        "local_status": local_status.get("download_status"),
        "local_dir": str(local_dir),
        "remote_model_id": model_id,
        "gated_private_status": gated_status,
        "tier": entry.get("tier"),
        "priority_rank": entry.get("priority_rank"),
        "storage_gb": storage if storage is not None else "unknown",
    }
    if dry_run:
        result = "dry_run_ok" if local_status.get("download_status") == "local_exists" else "not_downloaded"
        return {
            **base_row,
            "result": result,
            "reason": "default dry-run; pass --download to fetch remote assets",
        }
    if entry.get("tier") == "tier2" and not _tier2_download_explicit(args):
        return {
            **base_row,
            "result": "blocked",
            "reason": "Tier 2 downloads are disabled by default; pass --tier tier2 or --model <tier2_model> with --download to request explicitly",
        }
    if gated_status == "gated" and not args.allow_gated:
        return {
            **base_row,
            "result": "blocked",
            "reason": "manifest marks model gated; pass --allow-gated and set token env var if access is approved",
        }
    if args.skip_existing and local_status.get("download_status") == "local_exists":
        return {**base_row, "result": "skipped", "reason": "local assets already verified"}
    token = None
    token_env_var = str(entry.get("required_token_env_var") or args.token_env_var)
    if args.allow_gated or os.environ.get(token_env_var):
        token = os.environ.get(token_env_var)
    if gated_status == "gated" and not token:
        return {**base_row, "result": "blocked", "reason": f"token env var {token_env_var} is required but not set"}
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return {**base_row, "result": "blocked", "reason": f"huggingface_hub is unavailable: {exc}"}
    if not model_id:
        return {**base_row, "result": "blocked", "reason": "candidate_model_id is empty"}
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            cache_dir=args.cache_dir,
            revision=args.revision,
            token=token,
            local_files_only=bool(args.local_files_only),
        )
    except Exception as exc:
        return {**base_row, "result": "failed", "reason": f"snapshot_download failed: {exc}"}
    refreshed = validate_local_assets(entry, weights_root=args.output_root)
    result = "downloaded" if refreshed.get("download_status") == "local_exists" else "downloaded_but_unverified"
    return {**base_row, "result": result, "local_status": refreshed.get("download_status"), "reason": "token value was not logged"}


def _tier2_download_explicit(args: argparse.Namespace) -> bool:
    if args.tier and "tier2" in args.tier:
        return True
    explicit_models = set(args.model or [])
    tier2_names = {"qwen2_5_vl", "internvl3_5", "videochat_flash"}
    return bool(explicit_models & tier2_names)


if __name__ == "__main__":
    sys.exit(main())
