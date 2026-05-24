import json

import pytest

from repro.video_task_schema import (
    normalize_video_task,
    read_video_task_manifest,
    score_action_predictions,
    score_caption_predictions,
    score_videoqa_predictions,
)


def test_normalize_caption_task_preserves_references_and_default_prompt():
    task = normalize_video_task(
        {
            "id": "cap-1",
            "video_path": "clips/a.mp4",
            "caption": "A person opens a door.",
            "category": "daily",
        },
        task_type="captioning",
        row_index=0,
    )

    assert task == {
        "sample_id": "cap-1",
        "task_type": "captioning",
        "video_path": "clips/a.mp4",
        "prompt": "Describe the video.",
        "references": ["A person opens a door."],
        "category": "daily",
    }


def test_normalize_action_task_accepts_answer_choices_and_question_prompt():
    task = normalize_video_task(
        {
            "video_path": "clips/b.mp4",
            "question": "Which action is shown? A. running B. cooking",
            "answer": "A",
            "choices": ["running", "cooking"],
            "source": "toy-action",
        },
        task_type="action_classification",
        row_index=3,
    )

    assert task["sample_id"] == "3"
    assert task["prompt"] == "Which action is shown? A. running B. cooking"
    assert task["label"] == "A"
    assert task["choices"] == ["running", "cooking"]
    assert task["source"] == "toy-action"


def test_normalize_videoqa_task_preserves_question_answer_and_choices():
    task = normalize_video_task(
        {
            "question_id": "vq-1",
            "video_path": "clips/c.mp4",
            "question": "What does the person do? A. run B. cook",
            "answer": "B",
            "choices": "run|cook",
            "duration": 12.5,
        },
        task_type="videoqa",
        row_index=4,
    )

    assert task == {
        "sample_id": "vq-1",
        "task_type": "videoqa",
        "video_path": "clips/c.mp4",
        "prompt": "What does the person do? A. run B. cook",
        "question": "What does the person do? A. run B. cook",
        "answer": "B",
        "choices": ["run", "cook"],
        "duration": 12.5,
    }


def test_normalize_video_task_reports_missing_required_fields():
    with pytest.raises(ValueError, match="captioning row missing required fields"):
        normalize_video_task({"caption": "no video"}, task_type="captioning", row_index=0)

    with pytest.raises(ValueError, match="action_classification row missing required fields"):
        normalize_video_task({"video_path": "x.mp4"}, task_type="action_classification", row_index=0)


def test_score_action_predictions_exact_and_parse_failed():
    summary, scored = score_action_predictions(
        [
            {"sample_id": "1", "label": "A", "raw_output": "A", "status": "ok"},
            {"sample_id": "2", "label": "B", "raw_output": "A", "status": "ok"},
            {"sample_id": "3", "label": "C", "raw_output": "I cannot tell", "status": "ok"},
            {"sample_id": "4", "label": "D", "raw_output": None, "status": "failed"},
        ]
    )

    assert summary == {
        "task_type": "action_classification",
        "total": 4,
        "scored": 2,
        "correct": 1,
        "failed": 1,
        "parse_failed": 1,
        "skipped": 0,
        "accuracy_scored": 0.5,
        "accuracy_total": 0.25,
    }
    assert scored[0]["correct"] is True
    assert scored[1]["correct"] is False
    assert scored[2]["parse_failed"] is True


def test_score_caption_predictions_is_not_scored_but_keeps_overlap_hint():
    summary, scored = score_caption_predictions(
        [
            {
                "sample_id": "1",
                "references": ["a person opens a door"],
                "raw_output": "A person opens the door.",
                "status": "ok",
            },
            {"sample_id": "2", "references": ["a dog runs"], "raw_output": None, "status": "oom"},
        ]
    )

    assert summary["task_type"] == "captioning"
    assert summary["scoring_status"] == "not_scored"
    assert summary["total"] == 2
    assert summary["failed"] == 1
    assert summary["caption_overlap"]["count"] == 1
    assert scored[0]["caption_overlap_f1"] > 0


def test_score_videoqa_predictions_handles_multiple_choice_and_text_answers():
    summary, scored = score_videoqa_predictions(
        [
            {"sample_id": "1", "answer": "A", "raw_output": "A", "status": "ok"},
            {"sample_id": "2", "answer": "open door", "raw_output": "The answer is open door.", "status": "ok"},
            {"sample_id": "3", "answer": "C", "raw_output": "No clue", "status": "ok"},
            {"sample_id": "4", "answer": "D", "raw_output": None, "status": "oom"},
        ]
    )

    assert summary["task_type"] == "videoqa"
    assert summary["total"] == 4
    assert summary["correct"] == 2
    assert summary["failed"] == 1
    assert summary["parse_failed"] == 1
    assert summary["accuracy_scored"] == 1.0
    assert summary["accuracy_total"] == 0.5
    assert scored[1]["parsed_answer"] == "open door"


def test_read_video_task_manifest_supports_jsonl_json_and_csv(tmp_path):
    jsonl_path = tmp_path / "tasks.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"video_path": "a.mp4", "caption": "a"}),
                json.dumps({"video_path": "b.mp4", "caption": "b"}),
            ]
        )
        + "\n"
    )
    json_path = tmp_path / "tasks.json"
    json_path.write_text(json.dumps({"rows": [{"video_path": "c.mp4", "caption": "c"}]}))
    csv_path = tmp_path / "tasks.csv"
    csv_path.write_text("video_path,caption\nx.mp4,x\n")

    assert len(read_video_task_manifest(jsonl_path, task_type="captioning")) == 2
    assert read_video_task_manifest(json_path, task_type="captioning")[0]["sample_id"] == "0"
    assert read_video_task_manifest(csv_path, task_type="captioning")[0]["video_path"] == "x.mp4"
