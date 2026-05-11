from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9가-힣 ]+", "", text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def relaxed_accuracy_placeholder(prediction: str, reference: str) -> float:
    """Placeholder relaxed VQA accuracy.

    This intentionally falls back to normalized exact match until a benchmark-
    specific relaxed scoring protocol is implemented.
    """

    return exact_match(prediction, reference)


@dataclass
class GeneratedAnswerLog:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, prediction: str, reference: str | None = None, question: str | None = None, **metadata: Any) -> None:
        self.rows.append(
            {
                "question": question,
                "prediction": prediction,
                "reference": reference,
                "metadata": metadata,
            }
        )

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.rows)


class VideoVQAMetricAggregator:
    def __init__(self, relaxed: bool = False) -> None:
        self.relaxed = relaxed
        self.predictions: list[str] = []
        self.references: list[str] = []
        self.answer_log = GeneratedAnswerLog()

    def add_batch(
        self,
        predictions: list[str],
        references: list[str],
        questions: list[str] | None = None,
    ) -> None:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        questions = questions or [None] * len(predictions)
        if len(questions) != len(predictions):
            raise ValueError("questions must match predictions length")

        self.predictions.extend(predictions)
        self.references.extend(references)
        for prediction, reference, question in zip(predictions, references, questions):
            self.answer_log.add(prediction=prediction, reference=reference, question=question)

    def compute(self) -> dict[str, Any]:
        if not self.predictions:
            return {
                "exact_match": 0.0,
                "relaxed_accuracy": 0.0,
                "num_samples": 0,
                "generated_answers": [],
                "metric_source": "internal",
                "relaxed_accuracy_note": "placeholder_exact_match_fallback",
            }

        exact_scores = [exact_match(pred, ref) for pred, ref in zip(self.predictions, self.references)]
        relaxed_scores = [
            relaxed_accuracy_placeholder(pred, ref) for pred, ref in zip(self.predictions, self.references)
        ]
        return {
            "exact_match": sum(exact_scores) / len(exact_scores),
            "relaxed_accuracy": sum(relaxed_scores) / len(relaxed_scores),
            "num_samples": len(self.predictions),
            "generated_answers": self.answer_log.to_list(),
            "metric_source": "internal",
            "relaxed_accuracy_note": "placeholder_exact_match_fallback",
        }
