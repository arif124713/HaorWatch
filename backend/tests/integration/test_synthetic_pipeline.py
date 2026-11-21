"""Synthetic end-to-end pipeline test (no real satellite data, no DB).

Exercises: synthetic S1 VV/VH -> preprocess -> common-grid alignment -> mock
U-Net inference -> Otsu baseline -> permanent-water masking -> cropland overlay
-> area calculation -> output metadata/provenance. All data is synthetic and
deterministic.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.transform import Affine

from app.geospatial.area import mask_area_km2
from app.geospatial.grid import AnalysisGrid, align_to_grid, grid_from_profile
from app.geospatial.masks import apply_permanent_water, compute_valid
from app.geospatial.overlay import intersect_flood_agriculture
from app.geospatial.raster_io import read_bands, read_single, write_raster
from app.geospatial.reprojection import grid_for_bounds
from app.ml.baseline import otsu_flood_mask
from app.ml.features import band_to_db, normalize
from app.ml.inference import predict_tiles


class _ThresholdModel(torch.nn.Module):
    """Deterministic mock U-Net: high probability where VH (channel 1) is low."""

    def forward(self, x):  # (N, C, H, W)
        vh = x[:, 1:2, :, :]
        return (0.5 - vh) * 8.0  # logits: low VH -> positive logits


def _write_s1(path: Path, vv: np.ndarray, vh: np.ndarray, transform, crs):
    write_raster(path, np.stack([vv.astype(np.float32), vh.astype(np.float32)]),
                 transform, crs, dtype="float32", nodata=np.nan)


def _make_aoi_bounds():
    # Small AOI near Hakaluki in WGS84.
    return (91.60, 24.30, 91.62, 24.32)


def test_synthetic_pipeline_end_to_end(tmp_path):
    bounds = _make_aoi_bounds()
    transform, shape = grid_for_bounds(bounds, 32646, 10.0)
    target = AnalysisGrid("EPSG:32646", transform, int(shape[1]), int(shape[0]), 10.0, nodata=np.nan)
    h, w = target.height, target.width

    # Synthetic SAR in a *different* grid (coarser, same CRS) to force alignment.
    coarse_transform = Affine(20.0, 0.0, transform.c, 0.0, -20.0, transform.f)
    vv_coarse = np.full((h // 2, w // 2), -8.0, dtype=np.float32)   # mostly dry
    vh_coarse = np.full((h // 2, w // 2), -18.0, dtype=np.float32)
    vh_coarse[: h // 4, :] = -22.0  # water-like low backscatter in top half
    s1_path = tmp_path / "s1.tif"
    _write_s1(s1_path, vv_coarse, vh_coarse, coarse_transform, "EPSG:32646")

    # Prepare: read + align to the analysis grid.
    data, profile = read_bands(s1_path)
    src_grid = grid_from_profile(profile, resolution_m=20.0)
    vv = align_to_grid(data[0], src_grid, target, categorical=False)
    vh = align_to_grid(data[1], src_grid, target, categorical=False)
    assert vv.shape == (h, w)
    assert target.shape == (h, w)

    valid = compute_valid(vv, vh)
    assert valid.any()

    # Preprocess (dB -> normalize) and run mock U-Net inference on the aligned raster.
    unit = "dB"
    stats = {"mean": [float(np.nanmean(vv)), float(np.nanmean(vh))],
             "std": [float(np.nanstd(vv)) or 1.0, float(np.nanstd(vh)) or 1.0]}
    image = np.stack([
        normalize(band_to_db(vv, unit), stats["mean"][0], stats["std"][0]),
        normalize(band_to_db(vh, unit), stats["mean"][1], stats["std"][1]),
    ])
    image[:, ~valid] = 0.0
    prob = predict_tiles(_ThresholdModel(), image, valid, tile=64, stride=32, device="cpu")
    ml_mask = (prob >= 0.5) & valid

    # Otsu on the same VH band/observation.
    baseline_mask, threshold = otsu_flood_mask(band_to_db(vh, unit), valid, water_is_low=True)
    assert threshold is not None

    # Permanent-water masking (JRC-like occurrence, aligned to grid).
    occurrence = np.full((h, w), 10.0, dtype=np.float32)
    occurrence[: h // 8, :] = 99.0  # permanent water in a corner band
    masked_ml = apply_permanent_water(ml_mask, occurrence, threshold=95.0)
    assert not masked_ml[: h // 8, :].any()

    # Cropland overlay (WorldCover class 40), aligned and valid-restricted.
    worldcover = np.full((h, w), 40, dtype=np.float32)
    worldcover[:, w // 2:] = 80  # right half not cropland
    agri_mask = (np.isin(worldcover.astype(np.int32), [40])) & valid
    exposure = intersect_flood_agriculture(masked_ml, agri_mask, valid)
    assert exposure.shape == (h, w)
    assert not exposure[:, w // 2:].any()

    # Area calculations with consistent pixel area.
    pixel_area_km2 = mask_area_km2(np.ones((1, 1), dtype=bool), transform)
    detected_area = mask_area_km2(masked_ml, transform)
    agri_area = mask_area_km2(agri_mask, transform)
    exposure_area = mask_area_km2(exposure, transform)
    assert detected_area >= 0 and agri_area > 0
    assert exposure_area <= detected_area + 1e-9
    assert exposure_area <= agri_area + 1e-9
    assert pixel_area_km2 == pytest.approx(0.0001, rel=1e-6)  # 10 m x 10 m


def test_empty_flood_mask_and_empty_cropland(tmp_path):
    bounds = _make_aoi_bounds()
    transform, shape = grid_for_bounds(bounds, 32646, 10.0)
    h, w = int(shape[0]), int(shape[1])
    valid = np.ones((h, w), dtype=bool)

    empty_flood = np.zeros((h, w), dtype=bool)
    empty_crop = np.zeros((h, w), dtype=bool)
    assert mask_area_km2(empty_flood, transform) == 0.0
    assert mask_area_km2(empty_crop, transform) == 0.0
    exposure = intersect_flood_agriculture(empty_flood, empty_crop, valid)
    assert not exposure.any()


def test_partial_overlap_area_is_consistent(tmp_path):
    transform, shape = grid_for_bounds(_make_aoi_bounds(), 32646, 10.0)
    h, w = int(shape[0]), int(shape[1])
    valid = np.ones((h, w), dtype=bool)
    flood = np.zeros((h, w), dtype=bool)
    flood[: h // 2, :] = True
    crop = np.zeros((h, w), dtype=bool)
    crop[:, : w // 2] = True
    exposure = intersect_flood_agriculture(flood, crop, valid)
    flood_area = mask_area_km2(flood, transform)
    crop_area = mask_area_km2(crop, transform)
    exposure_area = mask_area_km2(exposure, transform)
    # Exposure covers one quadrant: half of flood area and half of crop area.
    assert exposure_area == pytest.approx(flood_area / 2, rel=1e-6)
    assert exposure_area == pytest.approx(crop_area / 2, rel=1e-6)
