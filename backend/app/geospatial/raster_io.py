from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


def read_bands(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def read_single(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def write_raster(
    path: str | Path,
    data: np.ndarray,
    transform: Affine,
    crs,
    dtype: str = "float32",
    nodata=None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        count, height, width = 1, *data.shape
    else:
        count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "count": count,
        "height": height,
        "width": width,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        data_3d = np.asarray(data, dtype=dtype)
        if data_3d.ndim == 2:
            data_3d = data_3d[np.newaxis, :, :]
        dst.write(data_3d)
    return path


def nodata_mask(data: np.ndarray, nodata_values: tuple | None = None) -> np.ndarray:
    valid = np.isfinite(data).all(axis=0) if data.ndim == 3 else np.isfinite(data)
    if nodata_values is not None:
        if data.ndim == 3:
            for band_index, nodata_value in enumerate(nodata_values):
                if nodata_value is not None:
                    valid &= data[band_index] != nodata_value
        elif nodata_values[0] is not None:
            valid &= data != nodata_values[0]
    return valid
