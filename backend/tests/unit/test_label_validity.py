"""Sentinel-2 pseudo-label validity: invalid pixels must never become negatives."""

import numpy as np

from app.ml.labels import IGNORE_LABEL, apply_validity, build_validity_mask


def test_valid_water_and_nonwater_are_preserved():
    sar_valid = np.ones((2, 3), dtype=bool)
    reference = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    aoi = np.ones((2, 3), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi)
    assert valid.all()
    label = apply_validity(reference, valid)
    np.testing.assert_allclose(label, reference)


def test_nan_reference_is_not_a_negative_label():
    sar_valid = np.ones((1, 4), dtype=bool)
    reference = np.array([[1.0, np.nan, 0.0, np.inf]], dtype=np.float32)
    aoi = np.ones((1, 4), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi)
    assert valid.tolist() == [[True, False, True, False]]
    label = apply_validity(np.nan_to_num(reference, nan=0.0), valid)
    # The NaN pixel must NOT be a label 0 (non-water); it must be the ignore sentinel.
    assert label[0, 1] == IGNORE_LABEL
    # Genuine non-water (index 2) stays 0.
    assert label[0, 2] == 0.0


def test_finite_nodata_sentinel_is_invalid_not_nonwater():
    sar_valid = np.ones((1, 3), dtype=bool)
    reference = np.array([[1.0, -9999.0, 0.0]], dtype=np.float32)
    aoi = np.ones((1, 3), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi)
    assert valid.tolist() == [[True, False, True]]
    label = apply_validity(reference, valid)
    assert label[0, 1] == IGNORE_LABEL  # sentinel would otherwise be a fake negative
    assert label[0, 2] == 0.0  # real non-water preserved


def test_outside_aoi_is_invalid():
    sar_valid = np.ones((1, 3), dtype=bool)
    reference = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    aoi = np.array([[True, False, True]])
    valid = build_validity_mask(sar_valid, reference, aoi)
    assert valid.tolist() == [[True, False, True]]
    label = apply_validity(reference, valid)
    assert label[0, 1] == IGNORE_LABEL


def test_invalid_sar_pixel_is_invalid():
    sar_valid = np.array([[True, False, True]])
    reference = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    aoi = np.ones((1, 3), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi)
    assert valid.tolist() == [[True, False, True]]


def test_no_reference_uses_sar_and_aoi_only():
    sar_valid = np.array([[True, True, False]])
    aoi = np.array([[True, False, True]])
    valid = build_validity_mask(sar_valid, None, aoi)
    assert valid.tolist() == [[True, False, False]]


def test_custom_sentinel_is_respected():
    sar_valid = np.ones((1, 2), dtype=bool)
    reference = np.array([[0.5, -1234.0]], dtype=np.float32)
    aoi = np.ones((1, 2), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi, nodata_sentinels=(-1234.0,))
    assert valid.tolist() == [[True, False]]


def test_invalid_pixels_excluded_like_loader_loss_masking():
    """Mimic the loader: flood = (label > 0.5) & valid; invalid must not count."""
    sar_valid = np.ones((1, 5), dtype=bool)
    reference = np.array([[1.0, np.nan, 0.0, -9999.0, 1.0]], dtype=np.float32)
    aoi = np.ones((1, 5), dtype=bool)
    valid = build_validity_mask(sar_valid, reference, aoi)
    label = apply_validity(np.nan_to_num(reference, nan=0.0, neginf=0.0, posinf=0.0), valid)
    flood = (label > 0.5) & valid
    # Only the two genuine water pixels are supervised positives.
    assert int(flood.sum()) == 2
    # No invalid pixel contributes to the loss/metric mask.
    assert int(valid.sum()) == 3
