"""Reprojection must preserve nodata / invalid pixels and never interpolate
categorical labels.

These tests build small synthetic rasters (no network, no GEE) and assert that
invalid source pixels (NaN, finite sentinels, dataset masks) and outside-source
destination pixels remain invalid after reprojection, and that categorical
(binary water) data is reprojected with nearest-neighbour only.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin
from rasterio.warp import Resampling

from app.geospatial.reprojection import (
    reproject_with_validity,
    resolve_categorical_resampling,
)

SRC_TRANSFORM = from_origin(0.0, 100.0, 10.0, 10.0)
DST_TRANSFORM = from_origin(0.0, 100.0, 5.0, 5.0)  # different resolution
SRC_CRS = "EPSG:32646"
DST_CRS = "EPSG:32646"
DST_SHAPE = (40, 40)


def _reproject(values, *, src_nodata=None, src_valid_mask=None, categorical=False):
    return reproject_with_validity(
        values, SRC_TRANSFORM, SRC_CRS, DST_TRANSFORM, DST_CRS, DST_SHAPE,
        src_nodata=src_nodata, src_valid_mask=src_valid_mask,
        categorical=categorical, dst_nodata=np.nan,
    )


def test_nan_source_stays_invalid_after_reprojection():
    values = np.ones((10, 10), dtype=np.float32)
    values[3:6, 3:6] = np.nan
    out, valid = _reproject(values)
    # The NaN block must not become valid finite data.
    assert not valid[12:24, 12:24].all()
    assert np.isfinite(out[valid]).all()


def test_infinite_source_stays_invalid():
    values = np.ones((10, 10), dtype=np.float32)
    values[0, 0] = np.inf
    _out, valid = _reproject(values)
    assert not valid.all()


def test_finite_nodata_sentinel_excluded():
    values = np.ones((10, 10), dtype=np.float32)
    values[2:4, 2:4] = -9999.0
    out, valid = _reproject(values, src_nodata=-9999.0)
    assert not valid.all()
    # The sentinel must not appear as a valid value anywhere.
    assert not np.any(valid & np.isclose(out, -9999.0))


def test_dataset_mask_invalid_pixels_excluded():
    values = np.ones((10, 10), dtype=np.float32)
    mask = np.ones((10, 10), dtype=bool)
    mask[5:8, 5:8] = False  # dataset mask marks these invalid
    _out, valid = _reproject(values, src_valid_mask=mask)
    assert not valid.all()
    # Region corresponding to the masked-out block must be invalid.
    assert not valid[20:32, 20:32].any()


def test_outside_source_pixels_remain_invalid():
    # A tiny source in the corner of a large destination: far corner stays invalid.
    values = np.ones((4, 4), dtype=np.float32)
    src_transform = from_origin(0.0, 20.0, 10.0, 10.0)
    out, valid = reproject_with_validity(
        values, src_transform, SRC_CRS, DST_TRANSFORM, DST_CRS, (100, 100),
        dst_nodata=np.nan,
    )
    assert not valid[60:, 60:].any()
    assert np.isnan(out[90, 90])


def test_binary_labels_use_nearest_and_stay_binary():
    water = np.zeros((10, 10), dtype=np.float32)
    water[:, 5:] = 1.0
    out, valid = _reproject(water, categorical=True)
    # Nearest-neighbour must never produce fractional label values.
    unique = set(np.unique(out[valid]).tolist())
    assert unique.issubset({0.0, 1.0}), unique


def test_categorical_reprojection_rejects_bilinear():
    with pytest.raises(ValueError):
        resolve_categorical_resampling(Resampling.bilinear)
    assert resolve_categorical_resampling(None) == Resampling.nearest
    assert resolve_categorical_resampling(Resampling.mode) == Resampling.mode


def test_different_crs_reprojection_keeps_validity():
    values = np.ones((10, 10), dtype=np.float32)
    values[0:2, 0:2] = np.nan
    out, valid = reproject_with_validity(
        values, SRC_TRANSFORM, "EPSG:32646", DST_TRANSFORM, "EPSG:32647", DST_SHAPE,
        dst_nodata=np.nan,
    )
    assert not valid.all()
    assert np.isfinite(out[valid]).all()


def test_valid_pixels_survive_and_are_finite():
    values = np.full((10, 10), 7.0, dtype=np.float32)
    out, valid = _reproject(values)
    assert valid.any()
    np.testing.assert_allclose(out[valid], 7.0)
