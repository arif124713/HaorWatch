import math

import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject as rio_reproject
from shapely import ops as shp_ops
from shapely.geometry.base import BaseGeometry


# Resampling methods that must never be applied to categorical (class/label)
# rasters. Bilinear/cubic average neighbouring class codes into meaningless
# in-between values.
NON_CATEGORICAL_SAFE = (Resampling.nearest, Resampling.mode)
# Categorical-safe default when the caller does not choose explicitly.
CATEGORICAL_RESAMPLING = Resampling.nearest


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    zone = min(max(zone, 1), 60)
    return (326 if lat >= 0 else 327) * 100 + zone


def reproject_geometry(geom: BaseGeometry, src_crs, dst_crs) -> BaseGeometry:
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_ops.transform(transformer.transform, geom)


def resolve_categorical_resampling(resampling: Resampling | None) -> Resampling:
    """Return a resampling method safe for categorical rasters.

    Bilinear/cubic (and similar continuous operators) are refused for category
    data; the caller is forced to nearest/mode. ``None`` defaults to nearest.
    """
    if resampling is None:
        return CATEGORICAL_RESAMPLING
    if resampling not in NON_CATEGORICAL_SAFE:
        raise ValueError(
            f"resampling {resampling!r} is not valid for categorical rasters; "
            f"use one of {NON_CATEGORICAL_SAFE} (nearest or mode)"
        )
    return resampling


def reproject_array(
    src_array: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.nearest,
    dst_nodata=None,
    src_nodata=None,
) -> np.ndarray:
    """Reproject a 2-D array, preserving source nodata semantics.

    ``src_nodata`` is passed to the resampler so declared nodata is never
    averaged into a valid value. Destination pixels with no valid source are
    filled with ``dst_nodata`` (default NaN).
    """
    dst = np.full(dst_shape, dst_nodata if dst_nodata is not None else np.nan, dtype=np.float32)
    rio_reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata,
    )
    return dst


def reproject_with_validity(
    src_values: np.ndarray,
    src_transform,
    src_crs,
    dst_transform,
    dst_crs,
    dst_shape: tuple[int, int],
    *,
    src_nodata=None,
    src_valid_mask: np.ndarray | None = None,
    categorical: bool = False,
    resampling: Resampling | None = None,
    dst_nodata=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject values and a validity mask separately, returning both.

    Validity is derived from, in order of precedence:
    1. ``src_valid_mask`` when provided (dataset masks / prior invalid flags),
    2. finite values (NaN and +/-inf excluded),
    3. ``src_nodata`` (declared nodata sentinel) when provided.

    The validity mask is reprojected with nearest-neighbour so it stays binary
    and conservative: a destination pixel is valid only if a valid source pixel
    lands on it. Pixels outside the source footprint remain invalid. Categorical
    data uses nearest/mode resampling only (never bilinear interpolation).
    """
    values = np.asarray(src_values, dtype=np.float64)
    finite = np.isfinite(values)
    valid = finite.copy()
    if src_nodata is not None:
        # Compare with a tolerance so float32 sentinels survive the read.
        valid &= ~np.isclose(values, float(src_nodata))
    if src_valid_mask is not None:
        valid &= np.asarray(src_valid_mask).astype(bool)

    if categorical:
        chosen = resolve_categorical_resampling(resampling)
    else:
        chosen = Resampling.nearest if resampling is None else resampling

    # Replace invalid source values with the destination nodata so they never
    # interpolate into their neighbours.
    fill = dst_nodata if dst_nodata is not None else np.nan
    src_for_warp = np.where(valid, values, fill).astype(np.float32)

    out_values = reproject_array(
        src_for_warp,
        src_transform,
        src_crs,
        dst_transform,
        dst_crs,
        dst_shape,
        resampling=chosen,
        dst_nodata=fill,
        src_nodata=fill,
    )
    out_valid = reproject_array(
        valid.astype(np.uint8),
        src_transform,
        src_crs,
        dst_transform,
        dst_crs,
        dst_shape,
        resampling=Resampling.nearest,
        dst_nodata=0,
        src_nodata=None,
    ).astype(bool)

    # A destination pixel is only valid if its warped value is also finite and
    # not the fill sentinel (covers outside-source pixels).
    out_valid &= np.isfinite(out_values)
    if dst_nodata is not None and np.isfinite(dst_nodata):
        out_valid &= ~np.isclose(out_values, float(dst_nodata))
    return out_values.astype(np.float32), out_valid


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
