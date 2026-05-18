from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.common import write_csv, write_json


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_autogaze_bench(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    autogaze_ms = float(payload["latency_ms"]["autogaze"]["mean"])
    siglip_full_ms = float(payload["latency_ms"]["siglip_full"]["mean"])
    siglip_gazed_ms = float(payload["latency_ms"]["siglip_gazed"]["mean"])
    return {
        "source": str(path),
        "token_reduction_ratio": payload["gaze"]["token_reduction_ratio"],
        "selected_non_padded_patches": payload["gaze"]["selected_non_padded_patches"],
        "raw_patch_budget": payload["gaze"]["raw_patch_budget"],
        "autogaze_ms_mean": autogaze_ms,
        "siglip_full_ms_mean": siglip_full_ms,
        "siglip_gazed_ms_mean": siglip_gazed_ms,
        "siglip_speedup_excluding_autogaze": safe_div(siglip_full_ms, siglip_gazed_ms),
        "siglip_speedup_including_autogaze": safe_div(siglip_full_ms, siglip_gazed_ms + autogaze_ms),
    }


def summarize_hlvid(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "source": str(path),
        "total": payload["total"],
        "scored": payload["scored"],
        "correct": payload["correct"],
        "parse_failed": payload["parse_failed"],
        "accuracy_scored": payload["accuracy_scored"],
        "paper_target_hlvid": 0.526,
        "paper_delta": payload["accuracy_scored"] - 0.526,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AutoGaze reproduction outputs")
    parser.add_argument("--autogaze-json", action="append", default=[])
    parser.add_argument("--hlvid-summary", action="append", default=[])
    parser.add_argument("--output-json", default="outputs/autogaze_repro/report_summary.json")
    parser.add_argument("--output-csv", default="outputs/autogaze_repro/report_summary.csv")
    args = parser.parse_args()

    rows = []
    for path in args.autogaze_json:
        rows.append({"kind": "autogaze_siglip", **summarize_autogaze_bench(path)})
    for path in args.hlvid_summary:
        rows.append({"kind": "hlvid", **summarize_hlvid(path)})
    write_json(args.output_json, {"rows": rows})
    write_csv(args.output_csv, rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
