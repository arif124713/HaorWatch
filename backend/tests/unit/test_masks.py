import numpy as np

from app.geospatial.masks import apply_permanent_water, compute_valid, counts


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
