"""Pseudo-label validity helpers for the Hakaluki S1/S2 dataset build.

These are pure array functions (no Earth Engine / rasterio dependency) so they
can be unit-tested on synthetic data. They encode the rule that invalid
Sentinel-2 reference pixels (NaN/inf or numeric nodata sentinels such as -9999)
must never silently become negative (non-water) labels.
"""

import numpy as np

# Sentinel-2 export nodata sentinels that are finite but invalid.
S2_NODATA_SENTINELS = (-9999.0, -32768.0, -3.4028235e38)
# Value written for pixels excluded from supervision (ignored by the loader).
IGNORE_LABEL = -1.0


def build_validity_mask(
    sar_valid: np.ndarray,
    reference: np.ndarray | None,
    aoi_mask: np.ndarray,
    nodata_sentinels: tuple[float, ...] = S2_NODATA_SENTINELS,
) -> np.ndarray:
    """Intersection of the required valid-data masks for supervision.

    A supervised pixel is valid only when it is inside the AOI, has finite SAR,
    and (when a Sentinel-2 reference is present) has a finite, non-sentinel
    reference value.
    """
    aoi = aoi_mask.astype(bool)
    valid = aoi & sar_valid.astype(bool)
    if reference is not None:
        ref = reference.astype(np.float64)
        ref_valid = np.isfinite(ref)
        for sentinel in nodata_sentinels:
            ref_valid &= ~np.isclose(ref, sentinel)
        valid &= ref_valid
    return valid


def apply_validity(reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return the pseudo-label with invalid pixels set to the ignore sentinel.

    Preserves genuine non-water (0) and water (1) labels for valid pixels; only
    invalid pixels are marked :data:`IGNORE_LABEL` so the loader excludes them
    from loss and metrics.
    """
    label = reference.astype(np.float32).copy()
    label[~valid.astype(bool)] = IGNORE_LABEL
    return label
