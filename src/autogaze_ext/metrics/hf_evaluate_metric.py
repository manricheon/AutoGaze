from __future__ import annotations

from typing import Any, Callable


class HFEvaluateMetric:
    """Optional Hugging Face Evaluate wrapper with internal fallback behavior."""

    def __init__(
        self,
        metric_name: str,
        config_name: str | None = None,
        fallback_compute: Callable[[list[Any], list[Any]], dict[str, Any]] | None = None,
        **load_kwargs: Any,
    ) -> None:
        self.metric_name = metric_name
        self.config_name = config_name
        self.metric = None
        self.available = False
        self.unavailable_reason: str | None = None
        self.fallback_compute = fallback_compute
        self._predictions: list[Any] = []
        self._references: list[Any] = []
        try:
            import evaluate  # type: ignore

            self.metric = evaluate.load(metric_name, config_name, **load_kwargs)
            self.available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self.unavailable_reason = str(exc)

    def add_batch(self, predictions: list[Any], references: list[Any]) -> None:
        self._predictions.extend(predictions)
        self._references.extend(references)
        if self.metric is None:
            return
        self.metric.add_batch(predictions=predictions, references=references)

    def compute(self) -> dict[str, Any]:
        if self.metric is None:
            if self.fallback_compute is not None:
                result = self.fallback_compute(self._predictions, self._references)
                result["metric_source"] = "internal_fallback"
                result["hf_evaluate_available"] = False
                result["hf_evaluate_metric"] = self.metric_name
                return result
            return {
                "metric_source": "internal_fallback",
                "hf_evaluate_available": False,
                "hf_evaluate_metric": self.metric_name,
                "unavailable_reason": self.unavailable_reason,
            }
        result = self.metric.compute()
        result["metric_source"] = "huggingface_evaluate"
        result["hf_evaluate_available"] = True
        result["hf_evaluate_metric"] = self.metric_name
        return result
