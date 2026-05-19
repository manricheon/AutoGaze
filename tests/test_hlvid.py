from repro.hlvid import (
    REQUIRED_COLUMNS,
    parse_choice,
    read_manifest_file,
    score_predictions,
    validate_manifest_rows,
    viewer_row_to_manifest,
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


def test_score_predictions_counts_model_failures_without_parse_failures():
    rows = [
        {"answer": "A", "raw_output": "A", "status": "ok"},
        {"answer": "B", "raw_output": None, "status": "failed", "error": "OOM"},
    ]

    summary, scored = score_predictions(rows)

    assert summary["total"] == 2
    assert summary["failed"] == 1
    assert summary["parse_failed"] == 0
    assert summary["scored"] == 1
    assert scored[1]["parse_status"] == "failed_run"


def test_score_predictions_summarizes_questions_without_dropping_row_details():
    rows = [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
            "raw_output": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
            "raw_output": "B",
        },
    ]

    summary, scored = score_predictions(rows)

    assert summary["question_count"] == 2
    assert summary["question_samples"] == [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
        },
        {
            "question_id": 2,
            "video_path": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
        },
    ]
    assert "predictions" in summary["question_note"]
    assert scored[0]["question"] == rows[0]["question"]


def test_score_predictions_adds_readable_benchmark_samples():
    rows = [
        {
            "question_id": 1,
            "video_path": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "answer": "A",
            "raw_output": "Answer: A",
            "status": "ok",
        },
        {
            "question_id": 2,
            "video": "clip_2.mp4",
            "prompt": "What text appears? A. Left B. Right C. Up D. Down",
            "answer": "B",
            "raw_output": None,
            "status": "failed",
            "error": "OOM",
        },
    ]

    summary, _ = score_predictions(rows)

    assert summary["benchmark_samples"] == [
        {
            "question_id": 1,
            "target_video": "clip_1.mp4",
            "question": "What is shown? A. One B. Two C. Three D. Four",
            "model_answer": "Answer: A",
            "parsed_model_answer": "A",
            "correct_answer": "A",
            "ground_truth_answer": "A",
            "correct": True,
            "status": "ok",
            "parse_status": "parsed",
        },
        {
            "question_id": 2,
            "target_video": "clip_2.mp4",
            "question": "What text appears? A. Left B. Right C. Up D. Down",
            "model_answer": None,
            "parsed_model_answer": None,
            "correct_answer": "B",
            "ground_truth_answer": "B",
            "correct": False,
            "status": "failed",
            "parse_status": "failed_run",
        },
    ]
    assert "model_answer" in summary["benchmark_sample_note"]


def test_read_manifest_file_supports_jsonl_and_validates_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"question_id": 1, "category": "av", "video_path": "clip.mp4", '
        '"question": "Q? A. a B. b C. c D. d", "answer": "A"}\n'
    )

    rows = read_manifest_file(manifest)

    assert rows == [
        {
            "question_id": 1,
            "category": "av",
            "video_path": "clip.mp4",
            "question": "Q? A. a B. b C. c D. d",
            "answer": "A",
        }
    ]


def test_viewer_row_to_manifest_uses_dataset_viewer_row_payload():
    payload = {
        "row_idx": 0,
        "row": {
            "question_id": 7,
            "category": "av",
            "video_path": "clip_av_video_5_001.mp4",
            "question": "What text is visible? A. A B. B C. C D. D",
            "answer": "C",
            "extra": "ignored",
        },
    }

    assert viewer_row_to_manifest(payload) == {
        "question_id": 7,
        "category": "av",
        "video_path": "clip_av_video_5_001.mp4",
        "question": "What text is visible? A. A B. B C. C D. D",
        "answer": "C",
    }
