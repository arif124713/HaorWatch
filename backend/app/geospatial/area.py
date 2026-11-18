import numpy as np
from pyproj import Geod
from shapely.geometry.base import BaseGeometry

from app.geospatial.reprojection import reproject_geometry, utm_epsg_for

GEOD = Geod(ellps="WGS84")


def geodesic_area_km2(geom: BaseGeometry) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    area_m2, _ = GEOD.geometry_area_perimeter(geom)
    return abs(area_m2) / 1e6


def projected_area_km2(geom: BaseGeometry, epsg: int | None = None) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    if epsg is None:
        epsg = utm_epsg_for(geom.centroid.x, geom.centroid.y)
    projected = reproject_geometry(geom, "EPSG:4326", f"EPSG:{epsg}")
    return abs(projected.area) / 1e6


def mask_area_km2(mask: np.ndarray, transform) -> float:
    if mask is None or not mask.any():
        return 0.0
    if abs(transform.a) <= 1 or abs(transform.e) <= 1:
        raise ValueError("Raster transform must be in projected metres, not degrees.")
    pixel_area_m2 = abs(transform.a * transform.e)
    return float(mask.sum()) * pixel_area_m2 / 1e6
