import numpy as np
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry

SIMPLIFY_VERTEX_LIMIT = 15000
SIMPLIFY_TOLERANCE_DEG = 0.0005


def mask_to_geometry(mask: np.ndarray, transform) -> BaseGeometry | None:
    if mask is None or mask.size == 0 or not mask.any():
        return None
    polygons = []
    for geom, value in features.shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        polygon = shape(geom)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        return None
    merged = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    return merged


def simplify_if_large(
    geom: BaseGeometry,
    max_vertices: int = SIMPLIFY_VERTEX_LIMIT,
    tolerance_deg: float = SIMPLIFY_TOLERANCE_DEG,
) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    vertex_count = 0
    for part in getattr(geom, "geoms", [geom]):
        vertex_count += len(part.exterior.coords)
        for ring in part.interiors:
            vertex_count += len(ring.coords)
    if vertex_count <= max_vertices:
        return geom
    simplified = geom.simplify(tolerance_deg, preserve_topology=True)
    return simplified if not simplified.is_empty else geom


def ensure_multipolygon(geom: BaseGeometry) -> MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    polygons = [part for part in getattr(geom, "geoms", []) if isinstance(part, Polygon)]
    if not polygons:
        return None
    return MultiPolygon(polygons)
