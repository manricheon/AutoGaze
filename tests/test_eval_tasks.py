import pytest

from autogaze.eval.tasks import TASKS, TaskConfig


def test_task_registry_contains_expected_branch_tasks():
    expected = {
        "videomme",
        "videomme_w_sub",
        "mvbench",
        "nextqa",
        "egoschema",
        "mlvu",
        "longvideobench",
        "hlvid",
        "actionatlas",
    }

    assert expected.issubset(TASKS)


def test_task_config_builds_plain_and_subtitle_prompts():
    task = TaskConfig(
        name="toy",
        hf_repo="local/toy",
        hf_split="test",
        video_col="video",
        question_col="question",
        options_col="options",
        answer_col="answer",
        subtitle_col="subtitle",
    )
    sample = {
        "video": {"path": "clip.mp4"},
        "question": "What happens?",
        "options": ["A. Opens the door", "B. Closes the window"],
        "answer": "A",
        "subtitle": "A person reaches for the handle.",
    }

    prompt = task.build_prompt(sample)
    prompt_with_subtitle = task.build_prompt(sample, use_subtitle=True)

    assert "Question: What happens?" in prompt
    assert "A. Opens the door" in prompt
    assert "B. Closes the window" in prompt
    assert "Subtitles:" not in prompt
    assert "Subtitles:" in prompt_with_subtitle
    assert "A person reaches for the handle." in prompt_with_subtitle


@pytest.mark.parametrize(
    ("answer_type", "answer", "expected"),
    [
        ("letter", " b ", "B"),
        ("index", 2, "C"),
        ("text", "green", "B"),
    ],
)
def test_task_config_normalizes_ground_truth(answer_type, answer, expected):
    task = TaskConfig(
        name="toy",
        hf_repo="local/toy",
        hf_split="test",
        video_col="video",
        question_col="question",
        options_col="options",
        answer_col="answer",
        answer_type=answer_type,
    )
    sample = {
        "video": "clip",
        "question": "Pick one",
        "options": ["red", "green", "blue"],
        "answer": answer,
    }

    assert task.get_ground_truth(sample) == expected


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("A.", "A"),
        ("Answer: c", "C"),
        ("The answer is (D).", "D"),
        ("no usable option", "?"),
    ],
)
def test_task_config_parse_prediction(generated, expected):
    task = TASKS["videomme"]

    assert task.parse_prediction(generated) == expected


def test_task_config_video_id_prefers_hf_path_when_available():
    task = TASKS["videomme"]

    assert task.get_video_id({"videoID": {"path": "nested/clip.mp4"}}) == "nested/clip.mp4"
    assert task.get_video_id({"videoID": "clip_001"}) == "clip_001"
