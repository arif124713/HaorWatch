import numpy as np
import pytest

from app.ml.evaluate import classification_metrics, confusion_counts, evaluate_mask, merge_confusion


def test_confusion_counts_and_metrics():
    pred = np.array([[1, 0], [1, 0]], dtype=bool)
    label = np.array([[1, 0], [0, 0]], dtype=bool)
    valid = np.ones((2, 2), dtype=bool)
    counts = confusion_counts(pred, label, valid)
    assert counts == {"true_positive": 1, "false_positive": 1, "true_negative": 2, "false_negative": 0}
    metrics = classification_metrics(counts)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["iou"] == pytest.approx(0.5)


def test_metrics_null_when_undefined():
    counts = {"true_positive": 0, "false_positive": 0, "true_negative": 5, "false_negative": 0}
    metrics = classification_metrics(counts)
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["iou"] is None


def test_evaluate_mask_reports_pixel_count():
    pred = np.array([[1, 1]], dtype=bool)
    label = np.array([[1, 0]], dtype=bool)
    valid = np.array([[True, True]])
    result = evaluate_mask(pred, label, valid)
    assert result["test_pixel_count"] == 2
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1


def test_merge_confusion_accumulates():
    total = {"true_positive": 1, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    merge_confusion(total, {"true_positive": 2, "false_positive": 3, "true_negative": 0, "false_negative": 0})
    assert total == {"true_positive": 3, "false_positive": 3, "true_negative": 0, "false_negative": 0}
