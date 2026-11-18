import numpy as np


def apply_permanent_water(
    flood: np.ndarray, occurrence: np.ndarray | None, threshold: float
) -> np.ndarray:
    if occurrence is None:
        return flood.copy()
    permanent = (occurrence >= threshold) & np.isfinite(occurrence)
    return flood & ~permanent


def compute_valid(*arrays: np.ndarray, nodata_values: tuple | None = None) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    valid = np.isfinite(stacked).all(axis=0)
    if nodata_values is not None:
        for band_index, nodata_value in enumerate(nodata_values):
            if nodata_value is None:
                continue
            valid &= arrays[band_index] != nodata_value
    return valid


def counts(valid: np.ndarray, total: int | None = None) -> tuple[int, int]:
    valid_count = int(valid.sum())
    total = total if total is not None else int(valid.size)
    return valid_count, total - valid_count
