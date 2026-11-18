import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return float(np.median(finite))
    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_first = np.cumsum(hist)
    weight_second = weight_first[-1] - weight_first
    sum_all = np.cumsum(hist * centers)
    mean_first = sum_all / np.maximum(weight_first, 1e-12)
    mean_second = (sum_all[-1] - sum_all) / np.maximum(weight_second, 1e-12)
    between_variance = weight_first[:-1] * weight_second[:-1] * (mean_first[:-1] - mean_second[:-1]) ** 2
    best = int(np.argmax(between_variance))
    return float(centers[best])


def otsu_flood_mask(
    band_db: np.ndarray, valid: np.ndarray, water_is_low: bool = True
) -> tuple[np.ndarray, float | None]:
    threshold = otsu_threshold(band_db[valid])
    if threshold is None:
        return np.zeros_like(band_db, dtype=bool), None
    if water_is_low:
        flood = band_db < threshold
    else:
        flood = band_db > threshold
    return flood & valid, threshold
