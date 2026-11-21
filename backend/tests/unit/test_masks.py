import numpy as np

from app.geospatial.masks import apply_permanent_water, compute_valid, counts
from app.geospatial.overlay import intersect_flood_agriculture


def test_apply_permanent_water_excludes_permanent_pixels():
    flood = np.array([[1, 1], [1, 0]], dtype=bool)
    occurrence = np.array([[98.0, 10.0], [np.nan, 50.0]])
    masked = apply_permanent_water(flood, occurrence, threshold=95.0)
    assert not masked[0, 0]
    assert masked[0, 1]
    assert masked[1, 0]
    assert not masked[1, 1]


def test_apply_permanent_water_without_occurrence_returns_copy():
    flood = np.array([[True, False]])
    masked = apply_permanent_water(flood, None, threshold=95.0)
    assert masked.sum() == 1
    assert masked is not flood


def test_compute_valid_and_counts():
    vv = np.array([[1.0, np.nan], [2.0, 3.0]])
    vh = np.array([[1.0, 1.0], [np.nan, 3.0]])
    valid = compute_valid(vv, vh)
    assert valid.tolist() == [[True, False], [False, True]]
    valid_count, nodata_count = counts(valid)
    assert valid_count == 2
    assert nodata_count == 2


def test_permanent_water_threshold_is_configurable():
    flood = np.array([[1, 1, 1]], dtype=bool)
    occurrence = np.array([[60.0, 80.0, 99.0]])
    masked_low = apply_permanent_water(flood, occurrence, threshold=70.0)
    assert masked_low.tolist() == [[True, False, False]]
    masked_high = apply_permanent_water(flood, occurrence, threshold=95.0)
    assert masked_high.tolist() == [[True, True, False]]


def test_overlay_excludes_nodata_pixels():
    flood = np.array([[1, 1, 0]])
    cropland = np.array([[1, 1, 1]])
    valid = np.array([[True, False, True]])
    exposure = intersect_flood_agriculture(flood, cropland, valid)
    # Pixel 1 is nodata: excluded from the overlap even though both masks are 1.
    assert exposure.tolist() == [[True, False, False]]
