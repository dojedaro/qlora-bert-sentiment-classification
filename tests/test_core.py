import numpy as np
import pytest

from qlora_bert.core import Config, classification_metrics


def test_learning_rate_defaults_differ_by_method():
    assert Config("qlora").learning_rate == 2e-4
    assert Config("full").learning_rate == 2e-5
    assert Config("full", learning_rate=1e-5).learning_rate == 1e-5


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Config("lora")


def test_metrics_from_logits():
    logits = np.array([[2.0, -1.0], [-1.0, 3.0], [0.5, 0.2], [-2.0, 1.0]])
    labels = np.array([0, 1, 1, 1])
    result = classification_metrics(logits, labels)
    assert result["accuracy"] == 0.75
    assert result["auc"] == 1.0  # positives always score higher, even the misclassified one
    assert 0 < result["f1"] < 1


def test_metrics_are_stable_for_large_logits():
    logits = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    assert classification_metrics(logits, np.array([0, 1]))["accuracy"] == 1.0
