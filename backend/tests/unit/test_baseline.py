import numpy as np
import pytest

from app.ml.baseline import otsu_flood_mask, otsu_threshold


def test_otsu_separates_bimodal_modes():
    rng = np.random.default_rng(7)
    image = np.concatenate([rng.normal(-22.0, 1.0, 4000), rng.normal(-10.0, 1.0, 2000)])
    threshold = otsu_threshold(image)
    assert -21.0 < threshold < -11.0


def test_otsu_returns_none_for_empty_input():
    assert otsu_threshold(np.array([])) is None
    assert otsu_threshold(np.array([np.nan, np.nan])) is None


def test_otsu_flood_mask_marks_low_backscatter_as_water():
    image = np.full((8, 8), -20.0)
    image[:, :4] = -8.0
    valid = np.ones((8, 8), dtype=bool)
    mask, threshold = otsu_flood_mask(image, valid, water_is_low=True)
    assert threshold is not None
    assert mask[:, 4:].all()
    assert not mask[:, :4].any()


def test_otsu_flood_mask_respects_valid_mask():
    image = np.full((4, 4), -20.0)
    image[:, :2] = -8.0
    valid = np.ones((4, 4), dtype=bool)
    valid[0, :] = False
    mask, threshold = otsu_flood_mask(image, valid, water_is_low=True)
    assert threshold is not None
    assert not mask[0, :].any()
    assert not mask[1:, :2].any()
    assert mask[1:, 2:].all()
