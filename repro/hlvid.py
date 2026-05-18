from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from repro.common import append_jsonl, write_json

REQUIRED_COLUMNS = ("question_id", "category", "video_path", "question", "answer")
CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


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


def load_hlvid_manifest(split: str = "test", limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("bfshi/HLVid", split=split)
    rows = [normalize_row(dict(row)) for row in dataset]
    if limit is not None:
        rows = rows[:limit]
    validate_manifest_rows(rows)
    return rows


def score_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    parse_failed = 0

    for row in rows:
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
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
    return summary, scored_rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def build_manifest(args: argparse.Namespace) -> None:
    rows = load_hlvid_manifest(split=args.split, limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} HLVid rows to {output}")


def score_file(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    summary, scored_rows = score_predictions(rows)
    write_json(args.summary, summary)
    append_jsonl(args.scored, scored_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="HLVid manifest and scoring helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest")
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
