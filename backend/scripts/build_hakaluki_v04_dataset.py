import json
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
from app.geospatial.reprojection import reproject_array, reproject_geometry, utm_epsg_for
from app.geospatial.raster_io import write_raster
from app.integrations.gee_client import GEEClient
from app.ml.features import ensure_db

AOI_PATH = Path(r"data\hakaluki_aoi.geojson")
OUT = Path(r"data\hakaluki_v04_dataset")
S2_CLOUD_MAX = 12
WINDOW_DAYS = 1

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
manifest = {"pairs": [], "grid": {"epsg": epsg, "transform": list(grid_transform)[:6], "shape": grid_shape}, "aoi_geojson": aoi_geojson}

for index, (s1_day, s2_day) in enumerate(pairs):
    pair_dir = OUT / f"pair_{index:02d}_{s1_day}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    if (pair_dir / "s1_grid.tif").exists() and (pair_dir / "pseudo_label.tif").exists():
        with rasterio.open(pair_dir / "pseudo_label.tif") as src:
            water_px = int((src.read(1) > 0.5).sum())
        manifest["pairs"].append({
            "pair_dir": str(pair_dir), "s1_day": s1_day, "s2_day": s2_day,
            "water_px": water_px, "valid_px": int(aoi_mask.sum()), "cached": True,
        })
        print(f"pair {index:02d}: cached ({s1_day} <-> {s2_day}), water {water_px / max(int(aoi_mask.sum()), 1) * 100:.1f}%")
        continue

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
    if s1_data.shape[1:] != grid_shape or s1_crs.to_string() != utm_crs:
        s1_data = np.stack([
            reproject_array(s1_data[band_index], s1_transform, s1_crs, grid_transform, utm_crs, grid_shape)
            for band_index in range(s1_data.shape[0])
        ])
    bands = []
    converted_any = False
    for band_index in range(2):
        band, converted = ensure_db(s1_data[band_index])
        converted_any = converted_any or converted
        bands.append(band)
    s1_data = np.stack(bands)
    valid = np.isfinite(s1_data).all(axis=0) & aoi_mask
    s1_data[:, ~valid] = -25.0
    write_raster(pair_dir / "s1_grid.tif", s1_data, grid_transform, utm_crs)

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
    if reference.shape != grid_shape or ref_crs.to_string() != utm_crs:
        reference = reproject_array(reference, ref_transform, ref_crs, grid_transform, utm_crs, grid_shape)
    reference = np.where(np.isfinite(reference), reference, 0.0).astype(np.float32)
    write_raster(pair_dir / "pseudo_label.tif", reference, grid_transform, utm_crs)

    water_px = int((reference > 0.5).sum())
    manifest["pairs"].append({
        "pair_dir": str(pair_dir),
        "s1_day": s1_day,
        "s2_day": s2_day,
        "s1_meta": s1_meta,
        "water_px": water_px,
        "valid_px": int(valid.sum()),
    })
    print(f"pair {index:02d}: S1 {s1_day} ({s1_meta['image_count']} scenes) <-> S2 {s2_day}, water {water_px / max(int(valid.sum()), 1) * 100:.1f}%")

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
