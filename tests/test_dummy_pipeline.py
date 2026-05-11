from __future__ import annotations

from autogaze_ext.pipeline.inference import run_inference
from autogaze_ext.pipeline.runner import load_config


def test_dummy_video_vqa_inference() -> None:
    cfg = load_config(config_name="dummy_video_vqa")

    result = run_inference(cfg, batch_size=2)

    assert result.task_type == "video_vqa"
    assert result.outputs["generated_text"] == ["dummy", "dummy"]
    assert result.logs["generated dummy answer"] == ["dummy", "dummy"]


def test_dummy_action_recognition_inference() -> None:
    cfg = load_config(config_name="dummy_action_recognition")

    result = run_inference(cfg, batch_size=2)

    assert result.task_type == "action_recognition"
    assert tuple(result.outputs["logits"].shape) == (2, 4)
    assert result.outputs["predicted_labels"].tolist() == [0, 0]


def test_autogaze_off_path_reports_equal_token_counts() -> None:
    cfg = load_config(config_name="dummy_video_vqa")

    result = run_inference(cfg, batch_size=2)

    assert result.logs["AutoGaze"] == "OFF"
    assert result.logs["visual token count before AutoGaze"] == result.logs["visual token count after AutoGaze"]


def test_batch_shape_propagation() -> None:
    cfg = load_config(config_name="dummy_video_vqa")

    result = run_inference(cfg, batch_size=2)

    assert result.logs["input video shape"] == (2, 4, 3, 224, 224)
    assert result.logs["sampled frame indices"] == [0, 2, 5, 7]
