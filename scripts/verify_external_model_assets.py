#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

from external_model_asset_utils import (
    load_asset_manifest,
    model_entry,
    select_model_names,
    validate_local_assets,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local external PoC model assets without loading weights.")
    parser.add_argument("--manifest", default="configs/poc_inference/model_asset_manifest.yaml")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--weights-root", default="weights")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--write-report", default="docs/MODEL_ASSET_VERIFY_REPORT.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_asset_manifest(args.manifest)
    names = select_model_names(manifest, args.model or ["all"])
    rows: list[dict[str, Any]] = []
    for name in names:
        entry = model_entry(manifest, name)
        validation = validate_local_assets(entry, weights_root=args.weights_root)
        adapter_status = _adapter_resolution_status(entry)
        rows.append(
            {
                "model": name,
                "download_status": validation["download_status"],
                "local_exists": validation["local_exists"],
                "config_ok": validation["config_status"]["ok"],
                "processor_tokenizer_ok": validation["processor_tokenizer_status"]["ok"],
                "weights_ok": validation["weights_status"]["ok"],
                "config_example_exists": validation["config_example_exists"],
                "adapter_resolves": adapter_status["ok"],
                "reason": _verification_reason(validation, adapter_status),
            }
        )
    write_markdown_report(
        args.write_report,
        "External Model Asset Verification Report",
        rows,
        columns=[
            "model",
            "download_status",
            "local_exists",
            "config_ok",
            "processor_tokenizer_ok",
            "weights_ok",
            "config_example_exists",
            "adapter_resolves",
            "reason",
        ],
    )
    print(f"Wrote report: {args.write_report}")
    return 0


def _adapter_resolution_status(entry: dict[str, Any]) -> dict[str, Any]:
    name = str(entry.get("expected_adapter_name") or entry.get("name"))
    try:
        from poc_model_registry import build_mllm, build_vision_encoder

        if name == "vjepa2":
            adapter = build_vision_encoder("vjepa2", {"model_id": entry.get("local_target_directory")})
        else:
            adapter = build_mllm(name, {"model_id": entry.get("local_target_directory")})
        return {"ok": adapter.name == name, "adapter": adapter.name}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _verification_reason(validation: dict[str, Any], adapter_status: dict[str, Any]) -> str:
    reasons: list[str] = []
    if not validation["local_exists"]:
        reasons.append("local directory missing")
    if not validation["config_status"]["ok"]:
        reasons.append(f"missing config files: {validation['config_status']['missing']}")
    if not validation["processor_tokenizer_status"]["ok"]:
        reasons.append(f"missing processor/tokenizer files: {validation['processor_tokenizer_status']['missing']}")
    if not validation["weights_status"]["ok"]:
        reasons.append("weights missing or incomplete")
    if not validation["config_example_exists"]:
        reasons.append("smoke config missing")
    if not adapter_status["ok"]:
        reasons.append(f"adapter resolution failed: {adapter_status.get('reason')}")
    return "; ".join(reasons) if reasons else "local files verified without loading model weights"


if __name__ == "__main__":
    sys.exit(main())
