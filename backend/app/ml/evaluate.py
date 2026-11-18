import numpy as np


def confusion_counts(pred: np.ndarray, label: np.ndarray, valid: np.ndarray) -> dict:
    pred_flat = pred[valid].astype(bool)
    label_flat = label[valid].astype(bool)
    tp = int(np.logical_and(pred_flat, label_flat).sum())
    fp = int(np.logical_and(pred_flat, ~label_flat).sum())
    tn = int(np.logical_and(~pred_flat, ~label_flat).sum())
    fn = int(np.logical_and(~pred_flat, label_flat).sum())
    return {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn}


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
