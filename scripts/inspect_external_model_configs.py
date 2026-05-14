#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

from external_model_asset_utils import (
    inspect_local_config,
    load_asset_manifest,
    model_entry,
    select_model_names,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect external model config files without loading weights.")
    parser.add_argument("--manifest", default="configs/poc_inference/model_asset_manifest.yaml")
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--weights-root", default="weights")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--write-report", default="docs/MODEL_CONFIG_INSPECTION_REPORT.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_asset_manifest(args.manifest)
    names = select_model_names(manifest, args.model or ["all"])
    rows: list[dict[str, Any]] = []
    for name in names:
        entry = model_entry(manifest, name)
        inspected = inspect_local_config(entry, weights_root=args.weights_root)
        rows.append(
            {
                "model": name,
                "status": inspected["status"],
                "architecture": inspected.get("architectures"),
                "model_type": inspected.get("model_type"),
                "patch_size": inspected.get("patch_size"),
                "crop_or_image_size": inspected.get("crop_size") or inspected.get("image_size"),
                "frames_per_clip": inspected.get("frames_per_clip"),
                "tubelet_size": inspected.get("tubelet_size"),
                "hidden_size": inspected.get("hidden_size") or inspected.get("vision_hidden_size"),
                "rope_indicators": inspected.get("rope_indicators") or {},
                "projector_connector_indicators": inspected.get("projector_connector_indicators") or {},
                "reason": inspected.get("reason") or "config-only inspection; no weights loaded",
            }
        )
    write_markdown_report(
        args.write_report,
        "External Model Config Inspection Report",
        rows,
        columns=[
            "model",
            "status",
            "architecture",
            "model_type",
            "patch_size",
            "crop_or_image_size",
            "frames_per_clip",
            "tubelet_size",
            "hidden_size",
            "rope_indicators",
            "projector_connector_indicators",
            "reason",
        ],
    )
    print(f"Wrote report: {args.write_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
