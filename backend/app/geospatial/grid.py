"""Explicit common analysis grid for all HaorWatch raster products.

Every raster that participates in a pixel-wise operation (preprocessed SAR,
U-Net output, Otsu baseline, JRC permanent-water, ESA WorldCover) must share one
authoritative grid: the same CRS, affine transform, pixel resolution, and
dimensions. Matching array shapes alone is **not** sufficient - two rasters can
have identical shapes while covering different geographic areas. This module
makes the grid explicit and validates alignment before any pixel-wise work.
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


def _transform_close(a: Affine, b: Affine, tolerance: float = 1e-6) -> bool:
    return all(abs(getattr(a, name) - getattr(b, name)) <= tolerance for name in ("a", "b", "c", "d", "e", "f"))


def grid_from_profile(profile: dict, resolution_m: float | None = None) -> AnalysisGrid:
    transform = profile["transform"]
    crs = _normalize_crs(profile.get("crs"))
    width = int(profile["width"])
    height = int(profile["height"])
    if resolution_m is None:
        resolution_m = float(abs(transform.a))
    if abs(transform.a) <= 1.0 or abs(transform.e) <= 1.0:
        raise GridMismatchError(
            "Analysis grid must be in a projected CRS with metre pixel size, "
            f"got transform pixel size ({transform.a}, {transform.e})."
        )
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


def align_to_grid(
    data: np.ndarray,
    src_grid: AnalysisGrid,
    dst_grid: AnalysisGrid,
    categorical: bool = False,
) -> np.ndarray:
    """Reproject ``data`` onto ``dst_grid``.

    Categorical data (class IDs such as WorldCover codes) always uses
    nearest-neighbour resampling; continuous SAR bands may use bilinear. This
    function only checks CRS/transform/shape metadata, so it is safe to call
    even when the array shapes happen to match.
    """
    if grids_match(src_grid, dst_grid):
        return data.astype(np.float32)

    resampling = CATEGORICAL_RESAMPLING if categorical else Resampling.nearest
    nodata = dst_grid.nodata if dst_grid.nodata is not None else DEFAULT_NODATA
    dst = np.full(dst_grid.shape, nodata, dtype=np.float32)
    rio_reproject(
        source=data,
        destination=dst,
        src_transform=src_grid.transform,
        src_crs=src_grid.crs,
        dst_transform=dst_grid.transform,
        dst_crs=dst_grid.crs,
        resampling=resampling,
        src_nodata=None,
        dst_nodata=nodata,
    )
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
