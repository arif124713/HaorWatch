import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import box, shape as shp_shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.geospatial.reprojection import reproject_geometry

TILE = 512
MIN_AOI_COVERAGE = 0.05

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"data\runs\87078641-c08d-4d56-8762-00ad64349668")
aoi_geojson = json.load(open(r"data\hakaluki_aoi.geojson", encoding="utf-8"))
aoi_geom = shp_shape(aoi_geojson["features"][0]["geometry"])
out = Path(r"data\hakaluki_labelkit")
out.mkdir(parents=True, exist_ok=True)

with rasterio.open(run_dir / "s1.tif") as s1_src:
    s1 = s1_src.read().astype(np.float32)
    grid_transform = s1_src.transform
    grid_crs = s1_src.crs
    grid_shape = (s1_src.height, s1_src.width)
with rasterio.open(run_dir / "flood_prob.tif") as src:
    prob = src.read(1)
with rasterio.open(run_dir / "flood_mask.tif") as src:
    flood = src.read(1) > 0.5
s2_ref_path = next(iter(run_dir.glob("s2_mndwi_water_*.tif")), None)
reference = None
if s2_ref_path:
    with rasterio.open(s2_ref_path) as src:
        reference = src.read(1).astype(np.float32)
        if reference.shape != grid_shape or src.crs != grid_crs:
            from app.geospatial.reprojection import reproject_array
            from pyproj import CRS as _CRS

            reference = reproject_array(reference, src.transform, _CRS.from_user_input(src.crs), grid_transform, grid_crs, grid_shape)

utm_geom = reproject_geometry(aoi_geom, "EPSG:4326", grid_crs)
aoi_mask = rasterize(
    [(utm_geom.__geo_interface__, 1)],
    out_shape=grid_shape,
    transform=grid_transform,
    fill=0,
    dtype="uint8",
).astype(bool)

vh = s1[1]
vh_lo, vh_hi = np.percentile(vh[aoi_mask & np.isfinite(vh)], [2, 98]) if aoi_mask.any() else (-25, -5)


def to_png(band_or_rgb: np.ndarray, path: Path):
    array = band_or_rgb
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = np.clip(array, 0, 1)
    Image.fromarray((array * 255).astype(np.uint8)).save(path)


tiles_geojson = {"type": "FeatureCollection", "features": []}
manifest = {"tiles": [], "run_dir": str(run_dir), "s2_reference": str(s2_ref_path) if s2_ref_path else None}

rows = range(0, grid_shape[0], TILE)
cols = range(0, grid_shape[1], TILE)
for row in rows:
    for col in cols:
        bottom = min(row + TILE, grid_shape[0])
        right = min(col + TILE, grid_shape[1])
        if bottom - row < 128 or right - col < 128:
            continue
        aoi_fraction = float(aoi_mask[row:bottom, col:right].mean())
        if aoi_fraction < MIN_AOI_COVERAGE:
            continue
        tile_id = f"hakaluki_r{row:05d}_c{col:05d}"
        tile_dir = out / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)

        transform = rasterio.transform.from_origin(
            grid_transform.c + col * grid_transform.a,
            grid_transform.f + row * grid_transform.e,
            grid_transform.a,
            abs(grid_transform.e),
        )
        profile = {
            "driver": "GTiff", "height": bottom - row, "width": right - col,
            "count": 2, "dtype": "float32", "crs": grid_crs, "transform": transform,
        }
        with rasterio.open(tile_dir / "s1_vv_vh.tif", "w", **profile) as dst:
            dst.write(s1[:, row:bottom, col:right])

        vh_tile = np.clip((vh[row:bottom, col:right] - vh_lo) / max(vh_hi - vh_lo, 1e-6), 0, 1)
        to_png(vh_tile, tile_dir / "1_vh_gray.png")

        water_like = (vh_tile < 0.35).astype(np.float32)
        overlay = np.stack([
            np.clip(vh_tile * 0.4 + 0.1, 0, 1),
            np.clip(vh_tile * 0.4 + 0.1, 0, 1),
            np.clip(water_like * 0.9 + vh_tile * 0.1, 0, 1),
        ], axis=-1)
        to_png(overlay, tile_dir / "2_low_backscatter_hint.png")

        if reference is not None:
            ref_tile = np.nan_to_num(reference[row:bottom, col:right], nan=0.0)
            overlay_s2 = np.stack([
                np.clip(vh_tile * 0.3 + 0.1, 0, 1),
                np.clip(ref_tile * 0.8 + vh_tile * 0.1, 0, 1),
                np.clip(vh_tile * 0.3 + 0.1, 0, 1),
            ], axis=-1)
            to_png(overlay_s2, tile_dir / "3_s2_water_hint_green.png")

        prob_tile = np.nan_to_num(prob[row:bottom, col:right], nan=0.0)
        overlay_model = np.stack([
            np.clip(prob_tile * 0.9 + vh_tile * 0.1, 0, 1),
            np.clip(vh_tile * 0.4 + 0.1, 0, 1),
            np.clip(vh_tile * 0.4 + 0.1, 0, 1),
        ], axis=-1)
        to_png(overlay_model, tile_dir / "4_model_prediction_overlay.png")

        tile_geom_utm = box(
            grid_transform.c + col * grid_transform.a,
            grid_transform.f + (bottom - 1) * grid_transform.e,
            grid_transform.c + (right - 1) * grid_transform.a,
            grid_transform.f + row * grid_transform.e,
        )
        tile_geom = reproject_geometry(tile_geom_utm, grid_crs, "EPSG:4326")
        tiles_geojson["features"].append({
            "type": "Feature",
            "id": tile_id,
            "properties": {"tile_id": tile_id, "aoi_fraction": round(aoi_fraction, 3)},
            "geometry": json.loads(json.dumps(tile_geom.__geo_interface__)),
        })
        manifest["tiles"].append({"tile_id": tile_id, "dir": str(tile_dir), "aoi_fraction": round(aoi_fraction, 3)})

(out / "tiles.geojson").write_text(json.dumps(tiles_geojson, indent=1), encoding="utf-8")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"label kit ready: {len(manifest['tiles'])} tiles -> {out}")
