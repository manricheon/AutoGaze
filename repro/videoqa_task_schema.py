from __future__ import annotations

from typing import Any

from repro.hlvid import parse_choice

REQUIRED_VIDEOQA_FIELDS = ("video_path", "question", "answer")
OPTIONAL_VIDEOQA_FIELDS = ("question_id", "choices", "category", "duration", "source")


def normalize_videoqa_task(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_VIDEOQA_FIELDS if row.get(field) is None]
    if missing:
        raise ValueError(f"missing required VideoQA fields: {missing}")
    normalized = {field: row[field] for field in REQUIRED_VIDEOQA_FIELDS}
    for field in OPTIONAL_VIDEOQA_FIELDS:
        if row.get(field) is not None:
            normalized[field] = row[field]
    return normalized


def score_multiple_choice_videoqa_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    scored = 0
    parse_failed = 0
    for row in rows:
        parsed = parse_choice(row.get("raw_output"))
        expected = parse_choice(str(row.get("answer", "")))
        if parsed is None or expected is None:
            parse_failed += 1
            continue
        scored += 1
        correct += int(parsed == expected)
    return {
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "parse_failed": parse_failed,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
