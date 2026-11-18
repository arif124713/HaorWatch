import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import get_session_factory
from app.ml.baseline import otsu_flood_mask
from app.ml.evaluate import classification_metrics, confusion_counts
from app.ml.features import ensure_db, normalize
from app.ml.inference import load_bundle, predict_tiles
from app.models import AnalysisMetric
from app.repositories.analysis import AnalysisRepository

MIN_LABELED_TILES = 3

parser = __import__("argparse").ArgumentParser(description="Import hand labels and run external validation.")
parser.add_argument("--labels", required=True, help="labels.geojson exported from the labeler")
parser.add_argument("--kit", default=r"data\hakaluki_labelkit")
parser.add_argument("--bundle", default=r"models\unet\0.1.0")
parser.add_argument("--run-id", default=None, help="Run to attach validation to (default: latest hakaluki run)")
parser.add_argument("--min-tiles", type=int, default=MIN_LABELED_TILES)
parser.add_argument("--dry-run", action="store_true", help="Score labels without writing to the database")
args = parser.parse_args()

labels_fc = json.loads(Path(args.labels).read_text(encoding="utf-8"))
polygons = [f for f in labels_fc.get("features", []) if f.get("geometry")]
if not polygons:
    raise SystemExit("No labeled polygons found in the file.")

tiles_fc = json.loads((Path(args.kit) / "tiles.geojson").read_text(encoding="utf-8"))
from shapely.geometry import shape as shp_shape

tile_geoms = {
    f["properties"]["tile_id"]: shp_shape(f["geometry"]) for f in tiles_fc["features"]
}

by_tile = {}
unassigned = 0
for feature in polygons:
    geom = shp_shape(feature["geometry"])
    centroid = geom.centroid
    tile_id = (feature.get("properties") or {}).get("tile_id")
    if tile_id not in tile_geoms or not tile_geoms[tile_id].contains(centroid):
        containing = [tid for tid, tgeom in tile_geoms.items() if tgeom.contains(centroid)]
        tile_id = containing[0] if containing else None
    if tile_id is None:
        unassigned += 1
        continue
    stamped = (feature.get("properties") or {}).get("tile_id")
    by_tile.setdefault(tile_id, []).append({**feature, "stamped_tile_id": stamped})
if unassigned:
    print(f"WARNING: {unassigned} polygons outside all kit tiles were skipped.")

kit = Path(args.kit)
bundle = load_bundle(args.bundle)
model = bundle["model"]
threshold = float(bundle["config"].get("probability_threshold", 0.5))
stats = bundle["norm_stats"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

labeled_date = "2026-09-18"
rows = []
totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
labeled_tile_ids = []

for tile_id, features in sorted(by_tile.items()):
    s1_path = kit / tile_id / "s1_vv_vh.tif"
    if not s1_path.exists():
        print(f"skip {tile_id}: kit tile missing")
        continue
    label_dir = Path(r"data\hakaluki_labels")
    label_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s1_path) as src:
        image = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs
    from shapely.geometry import shape as shp_shape
    from app.geospatial.reprojection import reproject_geometry

    shapes = [
        (reproject_geometry(shp_shape(f["geometry"]), "EPSG:4326", crs).__geo_interface__, 1)
        for f in features
        if f.get("geometry")
    ]
    label_mask = rasterize(
        shapes, out_shape=image.shape[1:], transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    if label_mask.sum() < 32:
        print(f"skip {tile_id}: labeled pixels < 32")
        continue
    np.save(label_dir / f"{tile_id}_label.npy", label_mask)
    labeled_tile_ids.append(tile_id)

    valid = np.isfinite(image).all(axis=0)
    bands = []
    for band_index in range(image.shape[0]):
        band, _ = ensure_db(image[band_index])
        bands.append(normalize(band, stats["mean"][band_index], stats["std"][band_index]))
    image = np.stack(bands)
    image[:, ~valid] = 0.0
    prob = predict_tiles(model, image, valid, device=device)
    pred = (prob >= threshold) & valid & label_mask

    flood_label = (label_mask & valid).astype(bool)
    counts = confusion_counts(pred, flood_label, label_mask)
    metrics = classification_metrics(counts)
    for key, value in counts.items():
        totals[key] += value
    rows.append({
        "tile": tile_id,
        "labeled_px": int(label_mask.sum()),
        "label_water_pct": round(float(flood_label.sum() / max(label_mask.sum(), 1) * 100), 1),
        **{k: (round(v, 4) if v is not None else None) for k, v in metrics.items()},
    })

if len(labeled_tile_ids) < args.min_tiles:
    print(f"only {len(labeled_tile_ids)} tiles labeled; need >= {args.min_tiles} for external validation")
    for row in rows:
        print(row)
    sys.exit(0)

aggregate = classification_metrics(totals)
print("=== hand-label external validation (U-Net v0.1.0 @", threshold, ") ===")
for row in rows:
    print(row)
print("aggregate:", {k: (round(v, 4) if v is not None else None) for k, v in aggregate.items()})

factory = get_session_factory()
note = (
    f"Operator hand labels on {len(labeled_tile_ids)} Hakaluki tiles (label date {labeled_date}); "
    "pixel metrics computed over drawn polygons only. True external validation for the pilot domain."
)
if args.dry_run:
    print("DRY RUN: metrics above NOT persisted to the database.")
    sys.exit(0)
with factory() as db:
    repo = AnalysisRepository(db)
    run = None
    if args.run_id:
        run = repo.get(uuid.UUID(args.run_id))
    if run is None:
        from sqlalchemy import select
        from app.models import AnalysisRun

        run = db.scalars(
            select(AnalysisRun).order_by(AnalysisRun.created_at.desc())
        ).first()
    if run is None:
        raise SystemExit("no analysis run found")
    now = datetime.now(timezone.utc)
    metric_rows = [
        AnalysisMetric(
            analysis_run_id=run.id,
            method_key="hand_validation_unet",
            method_name=f"U-Net vs operator hand labels ({len(labeled_tile_ids)} tiles)",
            precision_score=aggregate["precision"],
            recall_score=aggregate["recall"],
            f1_score=aggregate["f1"],
            iou_score=aggregate["iou"],
            true_positive=totals["true_positive"],
            false_positive=totals["false_positive"],
            true_negative=totals["true_negative"],
            false_negative=totals["false_negative"],
            test_pixel_count=sum(totals.values()),
            split_strategy="operator hand labels, Hakaluki AOI 2026-09-18",
            evaluated_at=now,
            notes=note,
        )
    ]
    otsu_totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for tile_id in labeled_tile_ids:
        s1_path = kit / tile_id / "s1_vv_vh.tif"
        label_mask = np.load(Path(r"data\hakaluki_labels") / f"{tile_id}_label.npy")
        with rasterio.open(s1_path) as src:
            image = src.read().astype(np.float32)
        valid = np.isfinite(image).all(axis=0)
        otsu_mask, _ = otsu_flood_mask(image[1], valid & label_mask, water_is_low=True)
        counts = confusion_counts(otsu_mask & label_mask, (label_mask & valid).astype(bool), label_mask & valid)
        for key, value in counts.items():
            otsu_totals[key] += value
    otsu_metrics = classification_metrics(otsu_totals)
    metric_rows.append(AnalysisMetric(
        analysis_run_id=run.id,
        method_key="hand_validation_otsu",
        method_name="Otsu baseline vs operator hand labels",
        precision_score=otsu_metrics["precision"],
        recall_score=otsu_metrics["recall"],
        f1_score=otsu_metrics["f1"],
        iou_score=otsu_metrics["iou"],
        true_positive=otsu_totals["true_positive"],
        false_positive=otsu_totals["false_positive"],
        true_negative=otsu_totals["true_negative"],
        false_negative=otsu_totals["false_negative"],
        test_pixel_count=sum(otsu_totals.values()),
        split_strategy="operator hand labels, Hakaluki AOI 2026-09-18",
        evaluated_at=now,
        notes=note,
    ))
    repo.replace_metrics(run.id, metric_rows)
    repo.update_run(run, validation_status="externally_validated", validation_note=note)
    print(f"persisted external validation for run {run.id} (validation_status=externally_validated)")
