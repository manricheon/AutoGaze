import pytest

from repro.videoqa_task_schema import (
    OPTIONAL_VIDEOQA_FIELDS,
    REQUIRED_VIDEOQA_FIELDS,
    normalize_videoqa_task,
    score_multiple_choice_videoqa_rows,
)


def test_normalize_videoqa_task_requires_video_question_and_answer():
    row = {
        "video_path": "clip.mp4",
        "question": "Q? A. a B. b C. c D. d",
        "answer": "A",
        "question_id": "q1",
        "choices": ["a", "b", "c", "d"],
        "category": "high-res",
        "duration": 300.0,
        "source": "hlvid",
        "unused": "ignored",
    }

    normalized = normalize_videoqa_task(row)

    assert REQUIRED_VIDEOQA_FIELDS == ("video_path", "question", "answer")
    assert "choices" in OPTIONAL_VIDEOQA_FIELDS
    assert normalized == {
        "video_path": "clip.mp4",
        "question": "Q? A. a B. b C. c D. d",
        "answer": "A",
        "question_id": "q1",
        "choices": ["a", "b", "c", "d"],
        "category": "high-res",
        "duration": 300.0,
        "source": "hlvid",
    }


def test_normalize_videoqa_task_reports_missing_required_fields():
    with pytest.raises(ValueError, match="missing required VideoQA fields"):
        normalize_videoqa_task({"video_path": "clip.mp4", "question": "Q?"})


def test_score_multiple_choice_videoqa_rows_counts_parse_failures_separately():
    rows = [
        {"answer": "A", "raw_output": "A"},
        {"answer": "B", "raw_output": "A"},
        {"answer": "C", "raw_output": "I cannot tell"},
    ]

    summary = score_multiple_choice_videoqa_rows(rows)

    assert summary == {
        "total": 3,
        "scored": 2,
        "correct": 1,
        "parse_failed": 1,
        "accuracy_scored": 0.5,
        "accuracy_total": 1 / 3,
    }
