from repro.hlvid import (
    REQUIRED_COLUMNS,
    parse_choice,
    score_predictions,
    validate_manifest_rows,
)


def test_parse_choice_accepts_direct_letters_and_prefixed_text():
    assert parse_choice("A") == "A"
    assert parse_choice("Answer: c.") == "C"
    assert parse_choice("The correct answer is D because the sign says Duke.") == "D"


def test_parse_choice_returns_none_for_ambiguous_output():
    assert parse_choice("A or B") is None
    assert parse_choice("No idea") is None


def test_validate_manifest_rows_requires_official_columns():
    row = {
        "question_id": 1,
        "category": "av",
        "video_path": "clip_av_video_5_001.mp4",
        "question": "Question? A. One B. Two C. Three D. Four",
        "answer": "A",
    }
    validate_manifest_rows([row])
    assert set(REQUIRED_COLUMNS).issubset(row)


def test_score_predictions_tracks_parse_failures_separately():
    rows = [
        {"answer": "A", "raw_output": "A"},
        {"answer": "B", "raw_output": "Answer: C"},
        {"answer": "D", "raw_output": "A or D"},
    ]
    summary, scored = score_predictions(rows)
    assert summary["total"] == 3
    assert summary["scored"] == 2
    assert summary["correct"] == 1
    assert summary["parse_failed"] == 1
    assert summary["accuracy_scored"] == 0.5
    assert scored[2]["parse_status"] == "failed"
