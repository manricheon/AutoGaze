from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from repro.common import compute_stats
from repro.hlvid import parse_choice, read_jsonl

TASK_TYPE_CAPTIONING = "captioning"
TASK_TYPE_ACTION_CLASSIFICATION = "action_classification"
TASK_TYPE_VIDEOQA = "videoqa"
TASK_TYPES = (TASK_TYPE_CAPTIONING, TASK_TYPE_ACTION_CLASSIFICATION, TASK_TYPE_VIDEOQA)

DEFAULT_CAPTION_PROMPT = "Describe the video."
DEFAULT_ACTION_PROMPT = "What action is shown in the video?"


def read_video_task_manifest(path: str | Path, *, task_type: str) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = read_jsonl(source)
    elif suffix == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"Video task JSON manifest must be a list or contain rows/data: {source}")
        rows = payload
    elif suffix == ".csv":
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dataset format
            raise RuntimeError("Reading parquet manifests requires pandas/pyarrow in the environment.") from exc
        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported video task manifest extension: {source.suffix}")
    return [normalize_video_task(row, task_type=task_type, row_index=index) for index, row in enumerate(rows)]


def normalize_video_task(row: dict[str, Any], *, task_type: str, row_index: int) -> dict[str, Any]:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported video task_type: {task_type}")
    if not row.get("video_path"):
        raise ValueError(f"{task_type} row missing required fields: ['video_path']")
    if task_type == TASK_TYPE_CAPTIONING:
        return _normalize_caption_task(row, row_index=row_index)
    if task_type == TASK_TYPE_ACTION_CLASSIFICATION:
        return _normalize_action_task(row, row_index=row_index)
    return _normalize_videoqa_task(row, row_index=row_index)


def score_action_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    failed = 0
    parse_failed = 0
    skipped = 0
    for row in rows:
        output = row.get("raw_output")
        expected = str(row.get("label", row.get("answer", ""))).strip()
        status = str(row.get("status") or "")
        scored_row = dict(row)
        if status in {"failed", "oom"}:
            failed += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = False
            scored_rows.append(scored_row)
            continue
        if status == "skipped":
            skipped += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = False
            scored_rows.append(scored_row)
            continue
        parsed = _parse_action_answer(output, expected=expected)
        if parsed is None or not expected:
            parse_failed += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = True
            scored_rows.append(scored_row)
            continue
        is_correct = _normalize_label(parsed) == _normalize_label(expected)
        correct += int(is_correct)
        scored += 1
        scored_row["parsed_answer"] = parsed
        scored_row["correct"] = is_correct
        scored_row["parse_failed"] = False
        scored_rows.append(scored_row)
    summary = {
        "task_type": TASK_TYPE_ACTION_CLASSIFICATION,
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "failed": failed,
        "parse_failed": parse_failed,
        "skipped": skipped,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
    return summary, scored_rows


def score_caption_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    failed = 0
    skipped = 0
    overlaps: list[float] = []
    for row in rows:
        scored_row = dict(row)
        status = str(row.get("status") or "")
        if status in {"failed", "oom"}:
            failed += 1
        elif status == "skipped":
            skipped += 1
        else:
            overlap = _best_reference_overlap(row.get("raw_output"), row.get("references") or [])
            if overlap is not None:
                scored_row["caption_overlap_f1"] = overlap
                overlaps.append(overlap)
        scored_row["correct"] = None
        scored_row["parse_failed"] = False
        scored_rows.append(scored_row)
    summary = {
        "task_type": TASK_TYPE_CAPTIONING,
        "scoring_status": "not_scored",
        "total": len(rows),
        "scored": 0,
        "correct": 0,
        "failed": failed,
        "parse_failed": 0,
        "skipped": skipped,
        "accuracy_scored": None,
        "accuracy_total": None,
        "caption_overlap": compute_stats(overlaps),
    }
    return summary, scored_rows


def score_videoqa_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    correct = 0
    scored = 0
    failed = 0
    parse_failed = 0
    skipped = 0
    for row in rows:
        output = row.get("raw_output")
        expected = str(row.get("answer", row.get("label", ""))).strip()
        status = str(row.get("status") or "")
        scored_row = dict(row)
        if status in {"failed", "oom"}:
            failed += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = False
            scored_rows.append(scored_row)
            continue
        if status == "skipped":
            skipped += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = False
            scored_rows.append(scored_row)
            continue
        parsed = _parse_videoqa_answer(output, expected=expected)
        if parsed is None or not expected:
            parse_failed += 1
            scored_row["correct"] = False
            scored_row["parse_failed"] = True
            scored_rows.append(scored_row)
            continue
        is_correct = _videoqa_answers_match(parsed, expected)
        correct += int(is_correct)
        scored += 1
        scored_row["parsed_answer"] = parsed
        scored_row["correct"] = is_correct
        scored_row["parse_failed"] = False
        scored_rows.append(scored_row)
    summary = {
        "task_type": TASK_TYPE_VIDEOQA,
        "total": len(rows),
        "scored": scored,
        "correct": correct,
        "failed": failed,
        "parse_failed": parse_failed,
        "skipped": skipped,
        "accuracy_scored": correct / scored if scored else 0.0,
        "accuracy_total": correct / len(rows) if rows else 0.0,
    }
    return summary, scored_rows


def _normalize_caption_task(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    references = _caption_references(row)
    if not references:
        raise ValueError("captioning row missing required fields: ['caption' or 'references']")
    normalized = _common_task(row, task_type=TASK_TYPE_CAPTIONING, row_index=row_index)
    normalized["prompt"] = str(row.get("prompt") or row.get("question") or DEFAULT_CAPTION_PROMPT)
    normalized["references"] = references
    return normalized


def _normalize_action_task(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    label = row.get("label", row.get("answer"))
    if label is None or str(label).strip() == "":
        raise ValueError("action_classification row missing required fields: ['label' or 'answer']")
    normalized = _common_task(row, task_type=TASK_TYPE_ACTION_CLASSIFICATION, row_index=row_index)
    normalized["prompt"] = str(row.get("prompt") or row.get("question") or DEFAULT_ACTION_PROMPT)
    normalized["label"] = str(label).strip()
    choices = _coerce_choices(row.get("choices"))
    if choices:
        normalized["choices"] = choices
    return normalized


def _normalize_videoqa_task(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    question = row.get("question", row.get("prompt"))
    answer = row.get("answer", row.get("label"))
    missing = []
    if question is None or str(question).strip() == "":
        missing.append("question")
    if answer is None or str(answer).strip() == "":
        missing.append("answer")
    if missing:
        raise ValueError(f"videoqa row missing required fields: {missing}")
    normalized = _common_task(row, task_type=TASK_TYPE_VIDEOQA, row_index=row_index)
    question_text = str(question).strip()
    normalized["prompt"] = question_text
    normalized["question"] = question_text
    normalized["answer"] = str(answer).strip()
    choices = _coerce_choices(row.get("choices"))
    if choices:
        normalized["choices"] = choices
    return normalized


def _common_task(row: dict[str, Any], *, task_type: str, row_index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "sample_id": str(row.get("sample_id") or row.get("question_id") or row.get("qid") or row.get("id") or row_index),
        "task_type": task_type,
        "video_path": str(row["video_path"]),
    }
    for field in ("category", "source", "duration"):
        if row.get(field) is not None:
            normalized[field] = row[field]
    return normalized


def _caption_references(row: dict[str, Any]) -> list[str]:
    value = row.get("references", row.get("captions", row.get("caption")))
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _coerce_choices(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        if "|" in stripped:
            return [part.strip() for part in stripped.split("|") if part.strip()]
        return [stripped]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _parse_action_answer(output: Any, *, expected: str) -> str | None:
    if output is None:
        return None
    parsed_choice = parse_choice(str(output))
    if re.fullmatch(r"[A-D]", expected.strip(), re.IGNORECASE):
        return parsed_choice
    text = str(output).strip()
    return text if text else None


def _parse_videoqa_answer(output: Any, *, expected: str) -> str | None:
    if output is None:
        return None
    if re.fullmatch(r"[A-E]", expected.strip(), re.IGNORECASE):
        return parse_choice(str(output))
    text = str(output).strip()
    if not text:
        return None
    normalized_text = _normalize_label(text)
    normalized_expected = _normalize_label(expected)
    if normalized_expected and normalized_expected in normalized_text:
        return expected
    return text


def _videoqa_answers_match(parsed: str, expected: str) -> bool:
    if re.fullmatch(r"[A-E]", expected.strip(), re.IGNORECASE):
        return parsed.strip().upper() == expected.strip().upper()
    return _normalize_label(expected) in _normalize_label(parsed)


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _best_reference_overlap(output: Any, references: list[str]) -> float | None:
    if output is None or not references:
        return None
    output_tokens = _token_set(str(output))
    if not output_tokens:
        return None
    best = 0.0
    for reference in references:
        reference_tokens = _token_set(reference)
        if not reference_tokens:
            continue
        overlap = len(output_tokens & reference_tokens)
        if overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(output_tokens)
            recall = overlap / len(reference_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))
