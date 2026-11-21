import numpy as np
import pytest

from app.ml.evaluate import (
    annotated_confusion_counts,
    classification_metrics,
    confusion_counts,
    evaluate_mask,
    is_full_binary_evaluation,
    merge_confusion,
)


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


# --- annotated (positive/negative/unannotated) evaluation -------------------

def test_annotated_counts_separate_classes():
    pred = np.array([[1, 1, 0, 0]])
    positive = np.array([[1, 0, 1, 0]])
    negative = np.array([[0, 1, 0, 1]])
    valid = np.ones((1, 4), dtype=bool)
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    assert counts["true_positive"] == 1
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1
    assert counts["true_negative"] == 1
    assert counts["annotated_pixel_count"] == 4
    assert counts["unannotated_valid_pixel_count"] == 0


def test_unannotated_pixels_are_excluded_not_background():
    pred = np.array([[1, 1, 1]])
    positive = np.array([[1, 0, 0]])
    negative = np.zeros((1, 3), dtype=bool)
    valid = np.ones((1, 3), dtype=bool)
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    # Pixel 1 and 2 are unannotated: predictions there must NOT count as FP or TN.
    assert counts["true_positive"] == 1
    assert counts["false_positive"] == 0
    assert counts["true_negative"] == 0
    assert counts["false_negative"] == 0
    assert counts["annotated_pixel_count"] == 1
    assert counts["unannotated_valid_pixel_count"] == 2


def test_nodata_pixels_are_excluded():
    pred = np.array([[1, 1]])
    positive = np.array([[1, 0]])
    negative = np.zeros((1, 2), dtype=bool)
    valid = np.array([[True, False]])
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    assert counts["annotated_pixel_count"] == 1
    assert counts["unannotated_valid_pixel_count"] == 0


def test_empty_annotations_produce_no_evaluated_pixels():
    pred = np.ones((2, 2), dtype=bool)
    empty = np.zeros((2, 2), dtype=bool)
    valid = np.ones((2, 2), dtype=bool)
    counts = annotated_confusion_counts(pred, empty, empty, valid)
    assert counts["annotated_pixel_count"] == 0
    assert counts["unannotated_valid_pixel_count"] == 4
    metrics = classification_metrics(counts)
    assert metrics["precision"] is None
    assert metrics["iou"] is None


def test_positive_only_annotations_are_not_full_binary():
    pred = np.array([[1, 1, 0]])
    positive = np.array([[1, 0, 1]])
    negative = np.zeros((1, 3), dtype=bool)
    valid = np.ones((1, 3), dtype=bool)
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    assert not is_full_binary_evaluation(counts)
    # Only TP and FN are representable; FP outside the polygon is invisible.
    assert counts["false_positive"] == 0
    assert counts["false_negative"] == 1


def test_positive_and_negative_annotations_are_full_binary():
    pred = np.array([[1, 0]])
    positive = np.array([[1, 0]])
    negative = np.array([[0, 1]])
    valid = np.ones((1, 2), dtype=bool)
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    assert is_full_binary_evaluation(counts)
    assert counts["true_positive"] == 1
    assert counts["true_negative"] == 1


def test_annotated_metrics_catch_false_positives_outside_positive_polygon():
    """Regression: masking predictions by the positive polygon hid FPs."""
    pred = np.array([[1, 1, 1, 1]])
    positive = np.array([[1, 0, 0, 0]])
    negative = np.array([[0, 1, 1, 1]])
    valid = np.ones((1, 4), dtype=bool)
    counts = annotated_confusion_counts(pred, positive, negative, valid)
    metrics = classification_metrics(counts)
    assert counts["false_positive"] == 3
    assert metrics["precision"] == pytest.approx(0.25)
