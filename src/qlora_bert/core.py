"""Configuration and metrics with no GPU dependencies, so they can be tested anywhere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class Config:
    method: str  # "qlora" or "full"
    epochs: int = 3
    batch_size: int = 16
    max_length: int = 512
    seed: int = 42
    # Standard learning rates: adapters train from scratch and need a larger step than the
    # pretrained weights updated in full fine-tuning.
    learning_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in {"qlora", "full"}:
            raise ValueError(f"unknown method {self.method!r}")
        if not self.learning_rate:
            object.__setattr__(self, "learning_rate", 2e-4 if self.method == "qlora" else 2e-5)


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 and ROC AUC from two-class logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    predicted = logits.argmax(axis=1)
    return {
        "accuracy": round(float(accuracy_score(labels, predicted)), 4),
        "f1": round(float(f1_score(labels, predicted, average="macro")), 4),
        "auc": round(float(roc_auc_score(labels, probs[:, 1])), 4),
    }
