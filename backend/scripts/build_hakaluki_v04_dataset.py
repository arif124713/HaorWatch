import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape as shp_shape

from app.core.config import get_settings
from app.core.exceptions import ProviderUnavailableError
from app.ml.dataset_cache import (
    BUILDER_VERSION,
    build_pair_record,
    evaluate_pair_cache,
    load_dataset_manifest,
    pair_record_for,
)
from app.geospatial.reprojection import (
    reproject_array,
    reproject_geometry,
    reproject_with_validity,
    utm_epsg_for,
)
from app.geospatial.raster_io import write_raster
from app.integrations.gee_client import GEEClient
from app.ml.features import ensure_db
from app.ml.labels import IGNORE_LABEL, S2_NODATA_SENTINELS, apply_validity, build_validity_mask

AOI_PATH = Path(r"data\hakaluki_aoi.geojson")
OUT = Path(r"data\hakaluki_v04_dataset")
S2_CLOUD_MAX = 12

# Reusable pair requires the full validity contract; a stale pair is rebuilt
# only when --rebuild-stale is passed (we never silently overwrite). By default
# stale pairs are skipped with a clear reason so no data is destroyed.
REBUILD_STALE = False


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the Hakaluki v0.4 S1/S2 pseudo-label dataset.")
    parser.add_argument(
        "--out-dir", default=str(OUT),
        help="Output dataset directory (default: the tracked v0.4 dataset dir).",
    )
    parser.add_argument(
        "--rebuild-stale", action="store_true",
        help="Rebuild cached pairs that fail the cache-validity contract. Without this, "
             "stale pairs are reported and skipped (no data is destroyed).",
    )
    parser.add_argument(
        "--version", default=None,
        help="Dataset version label to record in the manifest.",
    )
    return parser.parse_args(argv)
WINDOW_DAYS = 1
# GEE COPERNICUS/S1_GRD VV/VH are dB backscatter. Declared explicitly; never guessed.
SAR_UNIT = "dB"
# Sentinel-2 export nodata sentinels that are finite but invalid. Any pixel equal
# to one of these must NOT be treated as a valid non-water (0) label.
S2_NODATA_SENTINELS = (-9999.0, -32768.0, -3.4028235e38)
# Value written for pixels excluded from supervision (ignored by the loader).
IGNORE_LABEL = -1.0

_args = _parse_args()
OUT = Path(_args.out_dir)
REBUILD_STALE = bool(_args.rebuild_stale)

settings = get_settings()
client = GEEClient(settings.google_earth_engine_project)
ee = client.ee

aoi_geojson = json.load(open(AOI_PATH, encoding="utf-8"))["features"][0]["geometry"]
aoi_geom = shp_shape(aoi_geojson)
lon_min, lat_min, lon_max, lat_max = aoi_geom.bounds
epsg = utm_epsg_for(aoi_geom.centroid.x, aoi_geom.centroid.y)
utm_crs = f"EPSG:{epsg}"

from pyproj import CRS, Transformer

transformer = Transformer.from_crs("EPSG:4326", CRS.from_epsg(epsg), always_xy=True)
x0, y0 = transformer.transform(lon_min, lat_min)
x1, y1 = transformer.transform(lon_max, lat_max)
grid_west, grid_east = min(x0, x1), max(x0, x1)
grid_south, grid_north = min(y0, y1), max(y0, y1)
grid_w, grid_h = int(np.ceil((grid_east - grid_west) / 10.0)), int(np.ceil((grid_north - grid_south) / 10.0))
from rasterio.transform import from_origin

grid_transform = from_origin(grid_west, grid_north, 10.0, 10.0)
grid_shape = (grid_h, grid_w)
print(f"working grid: {grid_w}x{grid_h} @10m ({epsg})")

aoi_utm = reproject_geometry(aoi_geom, "EPSG:4326", utm_crs)
aoi_mask = rasterize(
    [(aoi_utm.__geo_interface__, 1)],
    out_shape=grid_shape,
    transform=grid_transform,
    fill=0,
    dtype="uint8",
).astype(bool)
print(f"AOI pixels: {int(aoi_mask.sum())}")

s1_collection = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(ee.Geometry(aoi_geojson))
    .filterDate("2026-06-01", "2026-11-01")
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
)
s1_days = sorted({
    datetime.fromtimestamp(d / 1000).date().isoformat()
    for d in s1_collection.aggregate_array("system:time_start").getInfo()
})

s2_collection = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(ee.Geometry(aoi_geojson))
    .filterDate("2026-06-01", "2026-11-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_MAX))
)
s2_days = sorted({
    datetime.fromtimestamp(d / 1000).date().isoformat()
    for d in s2_collection.aggregate_array("system:time_start").getInfo()
})

pairs = []
for s2_day in s2_days:
    s2_d = date.fromisoformat(s2_day)
    near = sorted(
        [d for d in s1_days if abs((date.fromisoformat(d) - s2_d).days) <= WINDOW_DAYS],
        key=lambda d: abs((date.fromisoformat(d) - s2_d).days),
    )
    if near:
        pairs.append((near[0], s2_day))
print(f"training pairs: {len(pairs)} -> {pairs}")

OUT.mkdir(parents=True, exist_ok=True)

prior_manifest = load_dataset_manifest(OUT / "manifest.json")


def _code_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CODE_COMMIT = _code_commit()
AOI_ID = hashlib.sha256(json.dumps(aoi_geojson, sort_keys=True).encode("utf-8")).hexdigest()[:16]
expected_grid = {"crs": utm_crs, "transform": list(grid_transform)[:6], "shape": list(grid_shape)}

manifest = {
    "dataset_version": _args.version or "0.4.0",
    "builder_version": BUILDER_VERSION,
    "code_commit": CODE_COMMIT,
    "sar_unit": SAR_UNIT,
    "aoi_id": AOI_ID,
    "pairs": [],
    "grid": {"epsg": epsg, "transform": list(grid_transform)[:6], "shape": list(grid_shape)},
    "aoi_geojson": aoi_geojson,
}

for index, (s1_day, s2_day) in enumerate(pairs):
    pair_dir = OUT / f"pair_{index:02d}_{s1_day}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Cache-validity contract: reuse only when every required artifact AND the
    # per-pair metadata record agree with the expected grid/unit/ignore value.
    record = pair_record_for(prior_manifest, pair_dir)
    decision = evaluate_pair_cache(
        pair_dir,
        expected_grid=expected_grid,
        expected_sar_unit=SAR_UNIT,
        expected_ignore_value=IGNORE_LABEL,
        expected_s1_day=s1_day,
        expected_s2_day=s2_day,
        record=record,
    )
    if decision.reusable:
        with rasterio.open(pair_dir / "pseudo_label.tif") as src:
            water_px = int((src.read(1) > 0.5).sum())
        reused = dict(record)
        reused["cached"] = True
        manifest["pairs"].append(reused)
        print(
            f"pair {index:02d}: cached & valid ({s1_day} <-> {s2_day}), "
            f"water {water_px / max(record.get('valid_px', 1), 1) * 100:.1f}%"
        )
        continue
    if pair_dir.exists() and any(pair_dir.iterdir()) and not REBUILD_STALE:
        print(
            f"pair {index:02d}: STALE, skipped (pass --rebuild-stale to rebuild): "
            f"{'; '.join(decision.stale_reasons)}"
        )
        stale_record = dict(record or {"pair_dir": str(pair_dir)})
        stale_record.update({"stale": True, "stale_reasons": decision.stale_reasons})
        manifest["pairs"].append(stale_record)
        continue
    print(f"pair {index:02d}: rebuilding ({s1_day} <-> {s2_day}): {'; '.join(decision.stale_reasons) or 'no cache'}")

    try:
        s1_image, s1_meta = client.s1_mosaic(aoi_geojson, f"{s1_day}T00:00:00Z", f"{s1_day}T23:59:59Z")
        s1_path = client.export_image(s1_image, aoi_geojson, utm_crs, 10.0, pair_dir / "s1.tif", band_names=["VV", "VH"])
    except ProviderUnavailableError as error:
        print(f"skip pair {index:02d} ({s1_day} <-> {s2_day}): {error.message}")
        continue
    with rasterio.open(s1_path) as src:
        s1_data = src.read().astype(np.float32)
        s1_transform = src.transform
        s1_crs = src.crs
        s1_nodata = src.nodata
        # Honour any per-dataset/band invalid mask the exporter or source
        # provides; a validity helper cannot recover pixels already warped to a
        # finite value, so mask them before resampling.
        s1_src_valid = None
        mask_flags = src.mask_flag_enums if hasattr(src, "mask_flag_enums") else None
        if mask_flags and any("nodata" in flags for flags in mask_flags):
            s1_src_valid = src.read_masks().astype(bool).all(axis=0)
    s1_valid = np.ones(grid_shape, dtype=bool)
    if s1_data.shape[1:] != grid_shape or s1_crs.to_string() != utm_crs:
        reprojected = []
        for band_index in range(s1_data.shape[0]):
            values, valid_band = reproject_with_validity(
                s1_data[band_index], s1_transform, s1_crs, grid_transform, utm_crs, grid_shape,
                src_nodata=s1_nodata if s1_nodata is not None else -9999.0,
                src_valid_mask=s1_src_valid,
                categorical=False,
                dst_nodata=np.nan,
            )
            reprojected.append(values)
            s1_valid &= valid_band
        s1_data = np.stack(reprojected)
    elif s1_src_valid is not None:
        s1_valid &= s1_src_valid
    if s1_nodata is not None:
        s1_valid &= ~np.isclose(s1_data, float(s1_nodata))
    bands = []
    converted_any = False
    for band_index in range(2):
        band, converted = ensure_db(s1_data[band_index], SAR_UNIT)
        converted_any = converted_any or converted
        bands.append(band)
    s1_data = np.stack(bands)
    valid = np.isfinite(s1_data).all(axis=0) & aoi_mask
    valid &= s1_valid
    # Preserve invalidity explicitly in the stored grid: invalid pixels are set
    # to a non-finite sentinel (NaN) and the nodata metadata is declared, so
    # downstream steps never mistake a filled value for a real observation.
    s1_data = np.where(valid[None, :, :], s1_data, np.nan).astype(np.float32)
    write_raster(pair_dir / "s1_grid.tif", s1_data, grid_transform, utm_crs, dtype="float32", nodata=np.nan)
    write_raster(
        pair_dir / "s1_valid_mask.tif", valid.astype(np.float32),
        grid_transform, utm_crs, dtype="float32", nodata=0.0,
    )

    s2_collection_dated = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(ee.Geometry(aoi_geojson))
        .filterDate(s2_day, (date.fromisoformat(s2_day) + timedelta(days=1)).isoformat())
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_MAX))
    )
    mosaic = s2_collection_dated.mosaic()
    water = mosaic.normalizedDifference(["B3", "B11"]).gt(0.0).rename("water")
    try:
        ref_path = client.export_image(water, aoi_geojson, utm_crs, 10.0, pair_dir / "s2_water.tif", band_names=["water"])
    except ProviderUnavailableError as error:
        print(f"skip pair {index:02d} ({s1_day} <-> {s2_day}): S2 export failed: {error.message}")
        continue
    with rasterio.open(ref_path) as src:
        reference = src.read(1).astype(np.float32)
        ref_transform = src.transform
        ref_crs = src.crs
        ref_nodata = src.nodata
        ref_src_valid = None
        ref_mask_flags = src.mask_flag_enums if hasattr(src, "mask_flag_enums") else None
        if ref_mask_flags and any("nodata" in flags for flags in ref_mask_flags):
            ref_src_valid = src.read_masks(1).astype(bool)
    ref_valid = np.ones(grid_shape, dtype=bool)
    if reference.shape != grid_shape or ref_crs.to_string() != utm_crs:
        # The S2 water raster is a categorical classification (0/1). Reproject
        # with nearest-neighbour and carry the validity mask separately so
        # bilinear interpolation can never fabricate fractional water labels or
        # promote an outside-source / nodata pixel to a valid class.
        reference, ref_valid = reproject_with_validity(
            reference, ref_transform, ref_crs, grid_transform, utm_crs, grid_shape,
            src_nodata=ref_nodata,
            src_valid_mask=ref_src_valid,
            categorical=True,
            dst_nodata=np.nan,
        )
    elif ref_src_valid is not None:
        ref_valid &= ref_src_valid
    if ref_nodata is not None:
        ref_valid &= ~np.isclose(reference, float(ref_nodata))
    # Explicit validity mask: AOI ∧ valid SAR ∧ valid (finite, non-sentinel) S2.
    # Invalid S2 pixels must never silently become non-water (0) labels.
    pair_valid = build_validity_mask(valid, reference, aoi_mask) & ref_valid
    reference = np.where(np.isfinite(reference), reference, np.nan).astype(np.float32)
    label = apply_validity(reference, pair_valid)
    write_raster(pair_dir / "pseudo_label.tif", label, grid_transform, utm_crs, dtype="float32", nodata=IGNORE_LABEL)
    write_raster(
        pair_dir / "valid_mask.tif", pair_valid.astype(np.float32),
        grid_transform, utm_crs, dtype="float32", nodata=0.0,
    )

    water_px = int(((label > 0.5) & pair_valid).sum())
    invalid_px = int((~pair_valid).sum())
    manifest["pairs"].append(build_pair_record(
        pair_dir=pair_dir,
        sar_unit=SAR_UNIT,
        s1_day=s1_day,
        s2_day=s2_day,
        s1_meta=s1_meta,
        aoi_id=AOI_ID,
        grid=expected_grid,
        label_ignore_value=IGNORE_LABEL,
        nodata={"s1_nodata": None if s1_nodata is None else float(s1_nodata),
                "s2_nodata": None if ref_nodata is None else float(ref_nodata),
                "label_ignore_value": IGNORE_LABEL},
        valid_px=int(pair_valid.sum()),
        invalid_px=invalid_px,
        water_px=water_px,
        code_commit=CODE_COMMIT,
        source_hashes={
            "s1_grid": _sha256_file(pair_dir / "s1_grid.tif"),
            "pseudo_label": _sha256_file(pair_dir / "pseudo_label.tif"),
            "valid_mask": _sha256_file(pair_dir / "valid_mask.tif"),
        },
    ))
    print(f"pair {index:02d}: S1 {s1_day} ({s1_meta['image_count']} scenes) <-> S2 {s2_day}, water {water_px / max(int(pair_valid.sum()), 1) * 100:.1f}% (invalid {invalid_px})")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

operator_check = OUT / "operator_check"
operator_check.mkdir(parents=True, exist_ok=True)
import shutil

labels_dir = Path(r"data\hakaluki_labels")
label_arrays = sorted(labels_dir.glob("*_label.npy"))
for label_path in label_arrays:
    tile_id = label_path.name.replace("_label.npy", "")
    s1_src = Path(r"data\hakaluki_labelkit") / tile_id / "s1_vv_vh.tif"
    if not s1_src.exists():
        continue
    label_mask = np.load(label_path)
    with rasterio.open(s1_src) as src:
        image = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs
    write_raster(operator_check / f"{tile_id}_s1.tif", image, transform, crs)
    write_raster(operator_check / f"{tile_id}_label.tif", label_mask.astype(np.float32), transform, crs)
    manifest.setdefault("operator_check", []).append({"tile_id": tile_id, "labeled_px": int(label_mask.sum())})

base_checkpoint = Path(r"models\unet\0.1.0") / "checkpoint.pt"
if base_checkpoint.exists():
    shutil.copy2(base_checkpoint, OUT / "base_checkpoint.pt")
    manifest["base_checkpoint"] = "base_checkpoint.pt (v0.1.0 weights, threshold 0.40)"

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"operator check tiles: {len(label_arrays)} | base checkpoint included: {base_checkpoint.exists()}")
print("dataset ready:", OUT)
