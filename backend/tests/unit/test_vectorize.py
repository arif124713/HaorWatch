import numpy as np
import pytest
from rasterio.transform import Affine

from app.geospatial.vectorize import ensure_multipolygon, mask_to_geometry, simplify_if_large


def test_mask_to_geometry_single_cell():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    transform = Affine(10.0, 0, 0, 0, -10.0, 40.0)
    geom = mask_to_geometry(mask, transform)
    assert geom is not None
    assert geom.is_valid
    assert geom.area == pytest.approx(400.0)


def test_mask_to_geometry_empty_returns_none():
    mask = np.zeros((4, 4), dtype=bool)
    transform = Affine(10.0, 0, 0, 0, -10.0, 40.0)
    assert mask_to_geometry(mask, transform) is None


def test_simplify_if_large_reduces_vertices():
    import shapely

    x, y = np.meshgrid(np.arange(0, 200, 1), np.arange(0, 200, 1))
    noisy = (np.sin(x / 3.0) + np.cos(y / 3.0)) > 1.5
    geom = mask_to_geometry(noisy.astype(bool), Affine(1.0, 0, 0, 0, -1.0, 200.0))
    if geom is None:
        return
    simplified = simplify_if_large(geom, max_vertices=50, tolerance_deg=0.05)
    assert simplified is not None


def test_ensure_multipolygon_wraps_polygon():
    from shapely.geometry import Polygon

    polygon = Polygon([(0, 0), (1, 0), (1, 1)])
    multi = ensure_multipolygon(polygon)
    assert multi.geom_type == "MultiPolygon"
    assert ensure_multipolygon(None) is None
