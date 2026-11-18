import numpy as np


def to_db(linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(linear, 1e-6, None))


def looks_like_db(band: np.ndarray) -> bool:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return True
    return float(np.median(finite)) < 0.0


def ensure_db(band: np.ndarray) -> tuple[np.ndarray, bool]:
    if looks_like_db(band):
        return band.astype(np.float32), False
    return to_db(band.astype(np.float64)).astype(np.float32), True


def compute_norm_stats(samples: list[np.ndarray], valid_masks: list[np.ndarray]) -> dict:
    sums = None
    sq_sums = None
    count = 0
    for sample, valid in zip(samples, valid_masks):
        stacked = sample[:, valid]
        if stacked.size == 0:
            continue
        sums = stacked.sum(axis=1) if sums is None else sums + stacked.sum(axis=1)
        sq_sums = (stacked ** 2).sum(axis=1) if sq_sums is None else sq_sums + (stacked ** 2).sum(axis=1)
        count += stacked.shape[1]
    if count == 0:
        raise ValueError("No valid pixels found for normalization statistics.")
    mean = sums / count
    variance = np.maximum(sq_sums / count - mean ** 2, 1e-8)
    return {
        "mean": [float(value) for value in mean],
        "std": [float(np.sqrt(value)) for value in variance],
        "pixel_count": int(count),
    }


def normalize(band: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((band - mean) / std).astype(np.float32)
