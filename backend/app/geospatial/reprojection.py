import math

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject as rio_reproject
from shapely import ops as shp_ops
from shapely.geometry.base import BaseGeometry


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    zone = min(max(zone, 1), 60)
    return (326 if lat >= 0 else 327) * 100 + zone


def reproject_geometry(geom: BaseGeometry, src_crs, dst_crs) -> BaseGeometry:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_ops.transform(transformer.transform, geom)


def reproject_array(
    src_array: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.nearest,
    dst_nodata=None,
) -> np.ndarray:
    dst = np.full(dst_shape, dst_nodata if dst_nodata is not None else np.nan, dtype=np.float32)
    rio_reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=dst_nodata,
    )
    return dst


def grid_for_bounds(
    bounds: tuple[float, float, float, float],
    epsg: int,
    resolution_m: float,
) -> tuple[Affine, tuple[int, int]]:
    west, south, east, north = bounds
    transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
    x0, y0 = transformer.transform(west, south)
    x1, y1 = transformer.transform(east, north)
    minx, maxx = min(x0, x1), max(x0, x1)
    miny, maxy = min(y0, y1), max(y0, y1)
    width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    transform = Affine(resolution_m, 0.0, minx, 0.0, -resolution_m, maxy)
    return transform, (height, width)
