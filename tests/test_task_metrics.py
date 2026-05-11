from __future__ import annotations

import torch

from autogaze_ext.metrics import (
    ActionRecognitionMetricAggregator,
    HFEvaluateMetric,
    VideoVQAMetricAggregator,
    exact_match,
    relaxed_accuracy_placeholder,
    top1_accuracy,
    top5_accuracy,
)


def test_video_vqa_exact_match() -> None:
    assert exact_match(" A Cat! ", "a cat") == 1.0
    assert exact_match("dog", "cat") == 0.0


def test_relaxed_accuracy_placeholder_behavior() -> None:
    assert relaxed_accuracy_placeholder("Answer.", "answer") == 1.0
    assert relaxed_accuracy_placeholder("answer with detail", "answer") == 0.0


def test_action_top1_accuracy() -> None:
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.1, 0.9]])
    labels = torch.tensor([1, 1, 1])

    assert top1_accuracy(logits, labels) == 2 / 3


def test_action_top5_accuracy() -> None:
    logits = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ]
    )
    labels = torch.tensor([4, 0])

    assert top5_accuracy(logits, labels) == 0.5


def test_video_vqa_metric_aggregation_and_logging() -> None:
    metric = VideoVQAMetricAggregator(relaxed=True)
    metric.add_batch(
        predictions=["cat", "dog"],
        references=["Cat!", "bird"],
        questions=["q1", "q2"],
    )

    result = metric.compute()

    assert result["exact_match"] == 0.5
    assert result["relaxed_accuracy"] == 0.5
    assert result["num_samples"] == 2
    assert result["generated_answers"][0]["question"] == "q1"
    assert result["relaxed_accuracy_note"] == "placeholder_exact_match_fallback"


def test_action_metric_aggregation() -> None:
    metric = ActionRecognitionMetricAggregator()
    metric.add_batch(
        logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        labels=torch.tensor([1, 1]),
    )
    metric.add_batch(
        logits=torch.tensor([[0.1, 0.9]]),
        labels=torch.tensor([0]),
    )

    result = metric.compute()

    assert result["top1_accuracy"] == 1 / 3
    assert result["top5_accuracy"] == 1.0
    assert result["num_samples"] == 3
    assert result["metric_source"] == "internal"


def test_hf_evaluate_fallback_behavior() -> None:
    metric = HFEvaluateMetric("__definitely_missing_metric_for_autogaze_tests__")
    metric.add_batch(predictions=["a"], references=["a"])
    result = metric.compute()

    if result["metric_source"] == "huggingface_evaluate":
        assert result["hf_evaluate_available"] is True
    else:
        assert result["metric_source"] == "internal_fallback"
        assert result["hf_evaluate_available"] is False
