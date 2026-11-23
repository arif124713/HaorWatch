"""Explicit common analysis grid for all HaorWatch raster products.

Every raster that participates in a pixel-wise operation (preprocessed SAR,
U-Net output, Otsu baseline, JRC permanent-water, ESA WorldCover) must share one
authoritative grid: the same CRS, affine transform, pixel resolution, and
dimensions. Matching array shapes alone is **not** sufficient - two rasters can
have identical shapes while covering different geographic areas. This module
makes the grid explicit and validates alignment before any pixel-wise work.

Resampling policy
-----------------
* Categorical rasters (class IDs such as ESA WorldCover codes or JRC
  occurrence bands) always use nearest-neighbour resampling. Interpolating
  class IDs would invent non-existent classes.
* Continuous rasters (SAR backscatter in dB or linear) use an explicitly chosen
  continuous resampling method. The default is bilinear, which is the
  conventional choice for continuous radiometric surfaces; callers may override
  it. The choice is always explicit, never implicit.

Nodata policy
-------------
* Source NaN values and declared source nodata are excluded from resampling and
  never bleed into destination pixels as valid values.
* Destination pixels that receive no valid source data are set to the
  destination grid's declared nodata (never a fabricated ``0``).
* All products share the same destination nodata representation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject as rio_reproject

DEFAULT_NODATA = np.nan
CATEGORICAL_RESAMPLING = Resampling.nearest
CONTINUOUS_RESAMPLING = Resampling.bilinear

SUPPORTED_CONTINUOUS_RESAMPLING = {
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


class GridMismatchError(ValueError):
    """Raised when two rasters do not share the same analysis grid."""


@dataclass(frozen=True)
class AnalysisGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    resolution_m: float
    nodata: Any = DEFAULT_NODATA

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def to_raster_profile(self) -> dict:
        return {
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
            "nodata": self.nodata,
        }


def _normalize_crs(crs) -> str:
    if crs is None:
        raise GridMismatchError("Raster has no CRS; cannot establish an analysis grid.")
    return CRS.from_user_input(crs).to_string()


def _assert_unrotated(transform: Affine, label: str = "transform") -> None:
    if transform.b != 0.0 or transform.d != 0.0:
        raise GridMismatchError(
            f"{label} has rotation/shear terms (b={transform.b}, d={transform.d}); "
            "rotated or sheared grids are not supported. Reproject to a north-up grid first."
        )


def _transform_close(a: Affine, b: Affine, tolerance: float = 1e-6) -> bool:
    return all(abs(getattr(a, name) - getattr(b, name)) <= tolerance for name in ("a", "b", "c", "d", "e", "f"))


def grid_from_profile(
    profile: dict,
    resolution_m: float | None = None,
    require_projected: bool = True,
) -> AnalysisGrid:
    """Build an :class:`AnalysisGrid` description from a raster profile.

    ``require_projected`` (default True) enforces that the grid is a metred
    projected grid, which is required for the authoritative *analysis* grid.
    Source rasters that are still in a geographic CRS (e.g. a raw EPSG:4326
    chip) may be described with ``require_projected=False``; they will be
    reprojected onto the analysis grid by :func:`align_to_grid`.
    """
    transform = profile["transform"]
    crs = _normalize_crs(profile.get("crs"))
    width = int(profile["width"])
    height = int(profile["height"])
    _assert_unrotated(transform, label="raster transform")
    if require_projected and (abs(transform.a) <= 1.0 or abs(transform.e) <= 1.0):
        raise GridMismatchError(
            "Analysis grid must be in a projected CRS with metre pixel size, "
            f"got transform pixel size ({transform.a}, {transform.e})."
        )
    if resolution_m is None:
        resolution_m = float(abs(transform.a))
    return AnalysisGrid(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        resolution_m=resolution_m,
        nodata=profile.get("nodata", DEFAULT_NODATA),
    )


def grids_match(a: AnalysisGrid, b: AnalysisGrid, tolerance: float = 1e-6) -> bool:
    return (
        a.crs == b.crs
        and (a.width, a.height) == (b.width, b.height)
        and _transform_close(a.transform, b.transform, tolerance)
    )


def assert_grids_match(a: AnalysisGrid, b: AnalysisGrid, label: str = "raster") -> None:
    problems = []
    if a.crs != b.crs:
        problems.append(f"CRS differs ({a.crs} vs {b.crs})")
    if (a.width, a.height) != (b.width, b.height):
        problems.append(f"dimensions differ ({a.width}x{a.height} vs {b.width}x{b.height})")
    if not _transform_close(a.transform, b.transform):
        problems.append(
            f"affine transform differs ({tuple(a.transform)[:6]} vs {tuple(b.transform)[:6]})"
        )
    if problems:
        raise GridMismatchError(f"{label} is not aligned to the analysis grid: " + "; ".join(problems))


def validate_array_shape(data: np.ndarray, grid: AnalysisGrid, label: str = "array") -> None:
    """Ensure an array's spatial dimensions match its declared grid."""
    if data.ndim == 3:
        _, height, width = data.shape
    elif data.ndim == 2:
        height, width = data.shape
    else:
        raise GridMismatchError(f"{label} has unsupported number of dimensions: {data.ndim}")
    if (height, width) != grid.shape:
        raise GridMismatchError(
            f"{label} array shape {(height, width)} does not match its grid {grid.shape}."
        )


def _source_nodata(src_grid: AnalysisGrid, explicit_nodata: Any) -> Any:
    """Determine the source nodata value passed to rasterio.

    ``explicit_nodata`` wins when provided. Otherwise the source grid's declared
    nodata is used. NaN is always preserved by rasterio even when nodata is None,
    but declaring it makes the exclusion explicit.
    """
    if explicit_nodata is not None:
        return explicit_nodata
    return src_grid.nodata


def align_to_grid(
    data: np.ndarray,
    src_grid: AnalysisGrid,
    dst_grid: AnalysisGrid,
    categorical: bool = False,
    resampling: str | None = None,
    src_nodata: Any = None,
) -> np.ndarray:
    """Reproject ``data`` onto ``dst_grid`` with explicit resampling and nodata.

    Categorical data always uses nearest-neighbour. Continuous data uses
    ``resampling`` (a key of :data:`SUPPORTED_CONTINUOUS_RESAMPLING`, default
    ``"bilinear"``). Source nodata (declared value or NaN) is excluded so that
    nodata pixels can never become valid zero-valued observations. Destination
    pixels with no valid source are set to the destination grid's nodata.
    """
    validate_array_shape(data, src_grid, label="source")

    if src_grid.crs != dst_grid.crs or not _transform_close(src_grid.transform, dst_grid.transform) or src_grid.shape != dst_grid.shape:
        # Reprojection is required; fall through to rio_reproject.
        pass
    else:
        return data.astype(np.float32)

    if categorical:
        chosen = CATEGORICAL_RESAMPLING
    else:
        method = resampling or CONTINUOUS_RESAMPLING.name
        if method not in SUPPORTED_CONTINUOUS_RESAMPLING:
            raise GridMismatchError(
                f"Unsupported continuous resampling {method!r}; "
                f"choose one of {sorted(SUPPORTED_CONTINUOUS_RESAMPLING)}."
            )
        chosen = SUPPORTED_CONTINUOUS_RESAMPLING[method]

    dst_nodata = dst_grid.nodata if dst_grid.nodata is not None else DEFAULT_NODATA
    src_nodata_value = _source_nodata(src_grid, src_nodata)

    dst = np.full(dst_grid.shape, dst_nodata, dtype=np.float32)
    rio_reproject(
        source=data.astype(np.float32),
        destination=dst,
        src_transform=src_grid.transform,
        src_crs=src_grid.crs,
        dst_transform=dst_grid.transform,
        dst_crs=dst_grid.crs,
        resampling=chosen,
        src_nodata=src_nodata_value,
        dst_nodata=dst_nodata,
    )
    # Defensive: any destination pixel that received no valid source keeps nodata.
    # rasterio may write 0 for some edge cases; convert non-finite handling so
    # nodata is never a fabricated zero.
    if dst_nodata is not None and not (isinstance(dst_nodata, float) and np.isnan(dst_nodata)):
        dst[dst == dst_nodata] = dst_nodata
    return dst


def validate_many(grids: dict[str, AnalysisGrid]) -> AnalysisGrid:
    """Ensure every named raster shares one grid and return the authoritative grid."""
    if not grids:
        raise GridMismatchError("No rasters supplied to validate a common analysis grid.")
    reference_name = next(iter(grids))
    reference = grids[reference_name]
    for name, grid in grids.items():
        if name == reference_name:
            continue
        assert_grids_match(reference, grid, label=name)
    return reference
