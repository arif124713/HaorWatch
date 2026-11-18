import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.geospatial.reprojection import reproject_array
from app.integrations.gee_client import GEEClient
from app.ml.baseline import otsu_flood_mask
from app.ml.evaluate import classification_metrics, confusion_counts
from app.models import AnalysisMetric
from app.repositories.analysis import AnalysisRepository

RUN_ID = "87078641-c08d-4d56-8762-00ad64349668"
S2_DATE = "2026-09-19"
S2_WINDOW_DAYS = 1

run_dir = Path("data") / "runs" / RUN_ID
flood_mask_path = run_dir / "flood_mask.tif"

with rasterio.open(flood_mask_path) as src:
    ml_mask = src.read(1) > 0.5
    grid_transform = src.transform
    grid_crs = src.crs
    grid_shape = (src.height, src.width)
    grid_bounds = src.bounds

aoi_geojson = json.load(open(r"data\hakaluki_aoi.geojson", encoding="utf-8"))["features"][0]["geometry"]

settings = get_settings()
client = GEEClient(settings.google_earth_engine_project)
ee = client.ee

start = (datetime.fromisoformat(S2_DATE) - timedelta(days=S2_WINDOW_DAYS)).isoformat()
end = (datetime.fromisoformat(S2_DATE) + timedelta(days=S2_WINDOW_DAYS + 1)).isoformat()
s2_collection = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterBounds(ee.Geometry(aoi_geojson))
    .filterDate(start, end)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 12))
)
count = int(s2_collection.size().getInfo())
if count == 0:
    raise SystemExit("No low-cloud S2 scene near the target date.")
mosaic = s2_collection.mosaic()
mndwi = mosaic.normalizedDifference(["B3", "B11"]).rename("mndwi")
water_ref = mndwi.gt(0.0).rename("water")
print(f"S2 scenes used: {count}")

ref_path = run_dir / f"s2_mndwi_water_{S2_DATE}.tif"
client.export_image(
    water_ref,
    aoi_geojson,
    grid_crs.to_string(),
    10.0,
    ref_path,
    band_names=["water"],
)

with rasterio.open(ref_path) as src:
    reference = src.read(1).astype(np.float32)
    ref_transform = src.transform
    ref_crs = src.crs
if reference.shape != ml_mask.shape or ref_crs != grid_crs:
    reference = reproject_array(
        reference, ref_transform, ref_crs, grid_transform, grid_crs, grid_shape
    )
reference_valid = np.isfinite(reference)
reference_water = (reference > 0.5) & reference_valid

with rasterio.open(run_dir / "s1.tif") as src:
    s1 = src.read().astype(np.float32)
    s1_transform = src.transform
if s1.shape[1:] != ml_mask.shape:
    from app.geospatial.reprojection import reproject_array as reproject
    from pyproj import CRS as _CRS

    s1 = reproject_array(s1, s1_transform, _CRS.from_user_input("EPSG:4326"), grid_transform, grid_crs, ml_mask.shape)
    s1_transform = grid_transform

jrc_path = run_dir / "jrc_occurrence.tif"
jrc = None
if jrc_path.exists():
    with rasterio.open(jrc_path) as src:
        jrc = src.read(1).astype(np.float32)
        jrc_transform = src.transform
        jrc_crs = src.crs
    if jrc.shape != ml_mask.shape or jrc_crs != grid_crs:
        from app.geospatial.reprojection import reproject_array as reproject
        from pyproj import CRS as _CRS

        jrc = reproject_array(jrc, jrc_transform, _CRS.from_user_input(jrc_crs), grid_transform, grid_crs, ml_mask.shape)

permanent = (jrc >= 95.0) & np.isfinite(jrc) if jrc is not None else None
vh = s1[1]
otsu_raw, otsu_threshold = otsu_flood_mask(vh, np.isfinite(vh), water_is_low=True)
otsu_mask = otsu_raw & ~permanent if permanent is not None else otsu_raw

from shapely.geometry import shape as shp_shape

aoi_geom = shp_shape(aoi_geojson)
from pyproj import CRS as _CRS
from app.geospatial.reprojection import reproject_geometry

aoi_utm = reproject_geometry(aoi_geom, "EPSG:4326", grid_crs)
from rasterio.features import rasterize

aoi_mask = rasterize(
    [(aoi_utm.__geo_interface__, 1)],
    out_shape=ml_mask.shape,
    transform=grid_transform,
    fill=0,
    dtype="uint8",
).astype(bool)

overlap = reference_valid & np.isfinite(vh) & aoi_mask


def score(name, pred):
    counts = confusion_counts(pred & overlap, reference_water & overlap, overlap)
    metrics = classification_metrics(counts)
    print(
        name,
        {k: (round(v, 4) if v is not None else None) for k, v in metrics.items()},
        "valid_px:", int(overlap.sum()),
        "ref_water_px:", int((reference_water & overlap).sum()),
    )
    return metrics, counts


print(f"reference water: {float((reference_water & overlap).sum() / overlap.sum() * 100):.1f}% of overlap")
ml_metrics, ml_counts = score("U-Net@0.40 ", ml_mask)
otsu_metrics, otsu_counts = score("Otsu      ", otsu_mask)

factory = get_session_factory()
note = (
    f"Proxy reference: Sentinel-2 MNDWI>0 water on {S2_DATE} (<=1 day from S1 acquisition). "
    "Optical water misses flooded vegetation and differs from SAR flood semantics; treat as independent proxy, not ground truth."
)
with factory() as db:
    repo = AnalysisRepository(db)
    run = repo.get(RUN_ID)
    if run is None:
        raise SystemExit("run not found")
    now = datetime.now(timezone.utc)
    rows = []
    for method_key, method_name, metrics, counts in [
        ("s2_proxy_unet", "U-Net vs Sentinel-2 MNDWI water", ml_metrics, ml_counts),
        ("s2_proxy_otsu", "Otsu baseline vs Sentinel-2 MNDWI water", otsu_metrics, otsu_counts),
    ]:
        rows.append(AnalysisMetric(
            analysis_run_id=run.id,
            method_key=method_key,
            method_name=method_name,
            precision_score=metrics["precision"],
            recall_score=metrics["recall"],
            f1_score=metrics["f1"],
            iou_score=metrics["iou"],
            true_positive=counts["true_positive"],
            false_positive=counts["false_positive"],
            true_negative=counts["true_negative"],
            false_negative=counts["false_negative"],
            test_pixel_count=int(overlap.sum()),
            split_strategy="single-date Hakaluki AOI vs S2 MNDWI proxy",
            evaluated_at=now,
            notes=note,
        ))
    repo.replace_metrics(run.id, rows)
    repo.update_run(run, validation_status="proxy_validated", validation_note=note)
    print("persisted metric rows + validation_status=proxy_validated for run", run.id)
