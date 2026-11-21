import numpy as np
import pytest
from rasterio.transform import Affine

from app.geospatial.grid import (
    AnalysisGrid,
    GridMismatchError,
    align_to_grid,
    grid_from_profile,
    grids_match,
    validate_many,
)


def _profile(crs="EPSG:32646", transform=None, width=8, height=6, nodata=np.nan):
    return {
        "crs": crs,
        "transform": transform or Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2700000.0),
        "width": width,
        "height": height,
        "nodata": nodata,
    }


def test_grid_from_profile_requires_projected_metre_grid():
    with pytest.raises(GridMismatchError):
        grid_from_profile(_profile(transform=Affine(0.001, 0.0, 0.0, 0.0, -0.001, 0.0)))


def test_grid_from_profile_reads_crs_transform_dims():
    grid = grid_from_profile(_profile())
    assert grid.crs == "EPSG:32646"
    assert grid.width == 8 and grid.height == 6
    assert grid.resolution_m == 10.0


def test_grids_match_true_for_identical_grids():
    a = grid_from_profile(_profile())
    b = grid_from_profile(_profile())
    assert grids_match(a, b)


def test_grids_match_false_for_different_crs():
    a = grid_from_profile(_profile(crs="EPSG:32646"))
    b = grid_from_profile(_profile(crs="EPSG:32645"))
    assert not grids_match(a, b)


def test_grids_match_false_for_different_origin_same_shape():
    # Same dimensions/resolution but shifted origin: matching shapes alone is not enough.
    a = grid_from_profile(_profile(transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2700000.0)))
    b = grid_from_profile(_profile(transform=Affine(10.0, 0.0, 500100.0, 0.0, -10.0, 2700000.0)))
    assert (a.width, a.height) == (b.width, b.height)
    assert not grids_match(a, b)


def test_grids_match_false_for_different_resolution():
    a = grid_from_profile(_profile(transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)))
    b = grid_from_profile(_profile(transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)))
    assert not grids_match(a, b)


def test_validate_many_raises_on_misalignment():
    a = grid_from_profile(_profile())
    b = grid_from_profile(_profile(crs="EPSG:32645"))
    with pytest.raises(GridMismatchError):
        validate_many({"s1": a, "jrc": b})


def test_validate_many_returns_reference_when_all_match():
    a = grid_from_profile(_profile())
    b = grid_from_profile(_profile())
    assert validate_many({"s1": a, "wc": b}) is a


def test_align_to_grid_noop_when_grids_match():
    a = grid_from_profile(_profile())
    data = np.ones((6, 8), dtype=np.float32)
    out = align_to_grid(data, a, a, categorical=False)
    np.testing.assert_allclose(out, data)


def test_align_to_grid_nearest_for_categorical():
    # Overlapping source grid at a different resolution/origin, same UTM zone.
    src = grid_from_profile(_profile(crs="EPSG:32646", transform=Affine(20.0, 0.0, 500000.0, 0.0, -20.0, 2700000.0), width=6, height=6))
    dst = grid_from_profile(_profile())
    class_ids = np.array([[40, 40, 80, 80, 40, 40]] * 6, dtype=np.float32)
    out = align_to_grid(class_ids, src, dst, categorical=True)
    finite = out[np.isfinite(out)]
    # Nearest resampling must yield only values from the original class set,
    # never an interpolated blend of class IDs.
    assert set(np.unique(finite)).issubset({40.0, 80.0})


def test_align_to_grid_marks_out_of_bounds_as_nodata():
    src = grid_from_profile(_profile(crs="EPSG:32645", transform=Affine(20.0, 0.0, 300000.0, 0.0, -20.0, 2700000.0), width=4, height=4))
    dst = grid_from_profile(_profile())
    class_ids = np.full((4, 4), 40, dtype=np.float32)
    out = align_to_grid(class_ids, src, dst, categorical=True)
    # Non-overlapping area must be nodata, not a fake class 0.
    assert not np.any(out[np.isfinite(out)] == 0)
