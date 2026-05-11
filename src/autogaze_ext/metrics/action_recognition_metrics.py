from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _as_tensor(values: torch.Tensor | list[int] | list[list[float]]) -> torch.Tensor:
    return values if isinstance(values, torch.Tensor) else torch.as_tensor(values)


def topk_accuracy(logits: torch.Tensor | list[list[float]], labels: torch.Tensor | list[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    logits_tensor = _as_tensor(logits)
    labels_tensor = _as_tensor(labels).to(torch.long)
    if logits_tensor.ndim != 2:
        raise ValueError("logits must have shape [B, num_classes]")
    if labels_tensor.ndim != 1 or labels_tensor.shape[0] != logits_tensor.shape[0]:
        raise ValueError("labels must have shape [B]")

    effective_k = min(k, logits_tensor.shape[1])
    topk = logits_tensor.topk(effective_k, dim=1).indices
    correct = topk.eq(labels_tensor.unsqueeze(1)).any(dim=1)
    return int(correct.sum().item()) / int(correct.numel())


def top1_accuracy(logits: torch.Tensor | list[list[float]], labels: torch.Tensor | list[int]) -> float:
    return topk_accuracy(logits, labels, k=1)


def top5_accuracy(logits: torch.Tensor | list[list[float]], labels: torch.Tensor | list[int]) -> float:
    return topk_accuracy(logits, labels, k=5)


@dataclass
class ActionRecognitionMetricAggregator:
    logits: list[torch.Tensor] = field(default_factory=list)
    labels: list[torch.Tensor] = field(default_factory=list)

    def add_batch(self, logits: torch.Tensor | list[list[float]], labels: torch.Tensor | list[int]) -> None:
        logits_tensor = _as_tensor(logits).detach().cpu()
        labels_tensor = _as_tensor(labels).to(torch.long).detach().cpu()
        if logits_tensor.ndim != 2:
            raise ValueError("logits must have shape [B, num_classes]")
        if labels_tensor.ndim != 1 or labels_tensor.shape[0] != logits_tensor.shape[0]:
            raise ValueError("labels must have shape [B]")
        self.logits.append(logits_tensor)
        self.labels.append(labels_tensor)

    def compute(self) -> dict[str, float | int | str]:
        if not self.logits:
            return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "num_samples": 0, "metric_source": "internal"}

        logits = torch.cat(self.logits, dim=0)
        labels = torch.cat(self.labels, dim=0)
        return {
            "top1_accuracy": top1_accuracy(logits, labels),
            "top5_accuracy": top5_accuracy(logits, labels),
            "num_samples": int(labels.numel()),
            "metric_source": "internal",
        }
