from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import requests

from repro.common import write_json, write_jsonl

REQUIRED_COLUMNS = ("question_id", "category", "video_path", "question", "answer")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def parse_choice(text: str | None) -> str | None:
    if text is None:
        return None
    matches = [match.group(1).upper() for match in CHOICE_RE.finditer(text)]
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    missing_by_row = []
    for idx, row in enumerate(rows):
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            missing_by_row.append((idx, missing))
    if missing_by_row:
        raise ValueError(f"HLVid manifest rows missing required columns: {missing_by_row[:3]}")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in REQUIRED_COLUMNS}


def viewer_row_to_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    row = payload.get("row", payload)
    return normalize_row(row)


def fetch_hlvid_manifest(
    dataset: str = "bfshi/HLVid",
    config: str = "default",
    split: str = "test",
    limit: int | None = None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        length = page_size if limit is None else min(page_size, limit - len(rows))
        if length <= 0:
            break
        response = requests.get(
            DATASET_VIEWER_ROWS_URL,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        batch = [viewer_row_to_manifest(item) for item in payload.get("rows", [])]
        rows.extend(batch)

        total = payload.get("num_rows_total")
        if not batch:
            break
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break
        offset += len(batch)
        if total is not None and offset >= int(total):
            break

    validate_manifest_rows(rows)
    return rows


def load_hlvid_manifest(split: str = "test", limit: int | None = None, config: str = "default") -> list[dict[str, Any]]:
    return fetch_hlvid_manifest(split=split, limit=limit, config=config)


def score_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    parse_failed = 0
    failed = 0
    skipped = 0

    for row in rows:
        status = row.get("status")
        if status in {"failed", "skipped"}:
            out = dict(row)
            out["parsed_answer"] = None
            out["expected_answer"] = parse_choice(str(row.get("answer", "")))
            out["correct"] = False
            out["parse_status"] = f"{status}_run"
            failed += int(status == "failed")
            skipped += int(status == "skipped")
            scored_rows.append(out)
            continue

        parsed = parse_choice(row.get("raw_output"))
        expected = parse_choice(str(row.get("answer", "")))
        out = dict(row)
        out["parsed_answer"] = parsed
        out["expected_answer"] = expected
        if parsed is None or expected is None:
            parse_failed += 1
            out["correct"] = False
            out["parse_status"] = "failed"
        else:
            scored += 1
            out["correct"] = parsed == expected
            out["parse_status"] = "parsed"
            correct += int(out["correct"])
        scored_rows.append(out)

    summary = {
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "parse_failed": parse_failed,
        "failed": failed,
        "skipped": skipped,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
    return summary, scored_rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def read_manifest_file(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(source)
    elif suffix == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"HLVid JSON manifest must be a list or contain rows/data: {source}")
        rows = payload
    elif suffix == ".csv":
        with source.open(newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional local dataset format
            raise RuntimeError("Reading parquet manifests requires pandas/pyarrow in the environment.") from exc
        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported HLVid manifest extension: {source.suffix}")

    validate_manifest_rows(rows)
    return [normalize_row(row) for row in rows]


def build_manifest(args: argparse.Namespace) -> None:
    rows = load_hlvid_manifest(split=args.split, limit=args.limit, config=args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} HLVid rows to {output}")


def score_file(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    summary, scored_rows = score_predictions(rows)
    write_json(args.summary, summary)
    write_jsonl(args.scored, scored_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="HLVid manifest and scoring helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--config", default="default")
    manifest.add_argument("--split", default="test")
    manifest.add_argument("--limit", type=int)
    manifest.add_argument("--output", default="data/hlvid/manifest_test.json")
    manifest.set_defaults(func=build_manifest)

    score = sub.add_parser("score")
    score.add_argument("--predictions", required=True)
    score.add_argument("--summary", default="outputs/autogaze_repro/hlvid_score_summary.json")
    score.add_argument("--scored", default="outputs/autogaze_repro/hlvid_scored_predictions.jsonl")
    score.set_defaults(func=score_file)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
