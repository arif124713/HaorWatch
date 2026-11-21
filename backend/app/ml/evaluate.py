import numpy as np

# Validation categories are deliberately distinct. Never promote a weaker
# category to a stronger one (e.g. a Sentinel-2 optical proxy is not an
# independent external reference).
VALIDATION_CATEGORIES = {
    "operator_annotated": "Operator-annotated reference validation",
    "operator_annotated_positive_only": "Operator-annotated reference (positive-only; partial)",
    "proxy_validated": "Optical satellite proxy comparison",
    "external_reference_checked": "Independent external reference validation",
    "internal_test_only": "Internal held-out test evaluation",
    "not_available": "No validation reference available",
    "externally_validated": "Independent external reference validation",
}


def confusion_counts(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict:
    """Confusion counts treating every valid pixel as an annotated binary label.

    ``label`` must be a complete binary reference (True = positive/water,
    False = negative/background). Callers whose annotations are positive-only
    must not use this function for full binary metrics; use
    :func:`annotated_confusion_counts` instead.
    """
    pred_flat = pred[valid].astype(bool)
    label_flat = label[valid].astype(bool)
    tp = int(np.logical_and(pred_flat, label_flat).sum())
    fp = int(np.logical_and(pred_flat, ~label_flat).sum())
    tn = int(np.logical_and(~pred_flat, ~label_flat).sum())
    fn = int(np.logical_and(~pred_flat, label_flat).sum())
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn}


def annotated_confusion_counts(
    pred: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    valid: np.ndarray,
) -> dict:
    """Confusion counts restricted to explicitly annotated pixels only.

    ``positive`` marks labelled pools/flood pixels and ``negative`` marks
    labelled non-flood (background) pixels. Any pixel that is neither positive
    nor negative is treated as *unannotated* and excluded from the confusion
    matrix entirely, so it can never be silently counted as true negative (which
    would deflate false positives). ``valid`` excludes nodata/invalid pixels.

    Returns TP/FP/TN/FN plus ``annotated_pixel_count`` and
    ``unannotated_valid_pixel_count`` for honest reporting of evaluated coverage.
    """
    in_valid = valid.astype(bool)
    pos = positive.astype(bool) & in_valid & ~negative.astype(bool)
    neg = negative.astype(bool) & in_valid & ~positive.astype(bool)
    pred_b = pred.astype(bool)

    tp = int(np.logical_and(pred_b, pos).sum())
    fn = int(np.logical_and(~pred_b, pos).sum())
    fp = int(np.logical_and(pred_b, neg).sum())
    tn = int(np.logical_and(~pred_b, neg).sum())
    unannotated = int((in_valid & ~positive.astype(bool) & ~negative.astype(bool)).sum())
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "annotated_pixel_count": tp + fp + tn + fn,
        "unannotated_valid_pixel_count": unannotated,
    }


def classification_metrics(counts: dict) -> dict:
    tp = counts.get("true_positive", 0)
    fp = counts.get("false_positive", 0)
    fn = counts.get("false_negative", 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator > 0 else None
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator > 0 else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def is_full_binary_evaluation(counts: dict) -> bool:
    """True only when both positive and negative classes were annotated.

    Precision/F1/IoU over positive-only annotations cannot expose false
    positives outside the drawn polygons, so results are not a full binary
    classification validation.
    """
    return counts.get("true_negative", 0) + counts.get("false_positive", 0) > 0 and (
        counts.get("true_positive", 0) + counts.get("false_negative", 0) > 0
    )


def evaluate_mask(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict:
    counts = confusion_counts(pred, label, valid)
    metrics = classification_metrics(counts)
    return {
        **metrics,
        **counts,
        "test_pixel_count": int(valid.sum()),
    }


def merge_confusion(total: dict, counts: dict) -> dict:
    for key in ("true_positive", "false_positive", "true_negative", "false_negative"):
        total[key] = total.get(key, 0) + counts.get(key, 0)
    return total
