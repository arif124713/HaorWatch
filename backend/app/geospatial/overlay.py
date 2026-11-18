import numpy as np


def intersect_flood_agriculture(
    flood: np.ndarray, agricultural: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    exposure = flood.astype(bool) & agricultural.astype(bool)
    if valid is not None:
        exposure &= valid
    return exposure
