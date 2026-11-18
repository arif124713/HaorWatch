import numpy as np
import pytest
from rasterio.transform import Affine

from app.geospatial.area import geodesic_area_km2, mask_area_km2, projected_area_km2
from app.geospatial.reprojection import utm_epsg_for


def test_utm_zone_for_bangladesh():
    assert utm_epsg_for(91.9, 24.6) == 32646
    assert utm_epsg_for(-60.0, -30.0) == 32721


def test_projected_area_matches_geodesic_for_small_square():
    from shapely.geometry import Polygon

    lon, lat = 91.9, 24.6
    d_lat = 0.01
    d_lon = 0.01 / np.cos(np.deg2rad(lat))
    square = Polygon(
        [(lon - d_lon, lat - d_lat), (lon + d_lon, lat - d_lat), (lon + d_lon, lat + d_lat), (lon - d_lon, lat + d_lat)]
    )
    geodesic = geodesic_area_km2(square)
    projected = projected_area_km2(square)
    assert geodesic == pytest.approx(4.93, rel=0.1)
    assert projected == pytest.approx(geodesic, rel=0.02)


def test_mask_area_uses_metre_transform():
    mask = np.ones((100, 100), dtype=bool)
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    assert mask_area_km2(mask, transform) == pytest.approx(1.0)


def test_mask_area_rejects_degree_transform():
    mask = np.ones((10, 10), dtype=bool)
    transform = Affine(0.001, 0, 0, 0, -0.001, 0)
    with pytest.raises(ValueError):
        mask_area_km2(mask, transform)
