"""Metric and benchmark result utilities."""

from autogaze_ext.metrics.benchmark import (
    BenchmarkResult,
    compute_fps,
    compute_throughput,
    write_csv_results,
    write_json_result,
)
from autogaze_ext.metrics.action_recognition_metrics import (
    ActionRecognitionMetricAggregator,
    top1_accuracy,
    top5_accuracy,
    topk_accuracy,
)
from autogaze_ext.metrics.hf_evaluate_metric import HFEvaluateMetric
from autogaze_ext.metrics.video_vqa_metrics import (
    GeneratedAnswerLog,
    VideoVQAMetricAggregator,
    exact_match,
    normalize_answer,
    relaxed_accuracy_placeholder,
)

__all__ = [
    "ActionRecognitionMetricAggregator",
    "BenchmarkResult",
    "GeneratedAnswerLog",
    "HFEvaluateMetric",
    "VideoVQAMetricAggregator",
    "compute_fps",
    "compute_throughput",
    "exact_match",
    "normalize_answer",
    "relaxed_accuracy_placeholder",
    "top1_accuracy",
    "top5_accuracy",
    "topk_accuracy",
    "write_csv_results",
    "write_json_result",
]
