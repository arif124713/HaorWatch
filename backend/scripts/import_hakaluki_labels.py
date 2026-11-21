"""Import operator hand labels and evaluate the U-Net / Otsu outputs against them.

Annotation semantics
--------------------
Each polygon carries a ``label`` property:

* ``water`` / ``1`` / ``positive`` / ``flood`` -> positive (water) annotation
* ``land`` / ``0`` / ``negative`` / ``not_water`` -> negative (non-water) annotation

Pixels inside **positive** polygons are the positive class. Pixels inside
**negative** polygons are the negative class. Every other valid pixel is
*unannotated* and is excluded from the confusion matrix entirely: unannotated
pixels are never treated as background. When the input file contains only
positive polygons the result is a positive-class agreement check, not a full
binary classification validation, and it is recorded as such.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import get_session_factory
from app.ml.baseline import otsu_flood_mask
from app.ml.evaluate import (
    annotated_confusion_counts,
    classification_metrics,
    is_full_binary_evaluation,
)
from app.ml.features import DEFAULT_SAR_UNIT, normalize_unit
from app.ml.inference import load_bundle, predict_tiles
from app.models import AnalysisMetric
from app.repositories.analysis import AnalysisRepository

POSITIVE_LABELS = {"water", "1", "positive", "flood", "wet"}
NEGATIVE_LABELS = {"land", "0", "negative", "not_water", "dry", "background"}

parser = argparse.ArgumentParser(description="Import hand labels and evaluate model outputs.")
parser.add_argument("--labels", required=True, help="GeoJSON exported from the labeler")
parser.add_argument("--kit", default=r"data\hakaluki_labelkit")
parser.add_argument("--bundle", default=r"models\unet\0.1.0")
parser.add_argument("--run-id", default=None, help="Run to attach evaluation to (default: latest hakaluki run)")
parser.add_argument("--min-tiles", type=int, default=1)
parser.add_argument("--observation-date", default=None,
                    help="ISO-8601 observation date these labels refer to (stored in provenance)")
parser.add_argument("--s1-unit", default=DEFAULT_SAR_UNIT, choices=["dB", "linear"],
                    help="Declared SAR unit of the kit rasters (default: dB)")
parser.add_argument("--dry-run", action="store_true", help="Score labels without writing to the database")
args = parser.parse_args()


def _classify(label_value) -> str | None:
    if label_value is None:
        # No explicit label property: treat legacy drawings as positive water.
        return "positive"
    key = str(label_value).strip().lower()
    if key in POSITIVE_LABELS:
        return "positive"
    if key in NEGATIVE_LABELS:
        return "negative"
    return None


labels_fc = json.loads(Path(args.labels).read_text(encoding="utf-8"))
polygons = [f for f in labels_fc.get("features", []) if f.get("geometry")]
if not polygons:
    raise SystemExit("No labeled polygons found in the file.")

tiles_fc = json.loads((Path(args.kit) / "tiles.geojson").read_text(encoding="utf-8"))
from shapely.geometry import shape as shp_shape

tile_geoms = {
    f["properties"]["tile_id"]: shp_shape(f["geometry"]) for f in tiles_fc["features"]
}

by_tile: dict[str, list[dict]] = {}
unassigned = 0
unknown_label = 0
for feature in polygons:
    geometry = shp_shape(feature["geometry"])
    centroid = geometry.centroid
    tile_id = (feature.get("properties") or {}).get("tile_id")
    if tile_id not in tile_geoms or not tile_geoms[tile_id].contains(centroid):
        containing = [tid for tid, tgeom in tile_geoms.items() if tgeom.contains(centroid)]
        tile_id = containing[0] if containing else None
    if tile_id is None:
        unassigned += 1
        continue
    polarity = _classify((feature.get("properties") or {}).get("label"))
    if polarity is None:
        unknown_label += 1
        continue
    by_tile.setdefault(tile_id, []).append({**feature, "polarity": polarity})
if unassigned:
    print(f"WARNING: {unassigned} polygons outside all kit tiles were skipped.")
if unknown_label:
    print(f"WARNING: {unknown_label} polygons with unrecognised label values were skipped.")

s1_unit = normalize_unit(args.s1_unit)
kit = Path(args.kit)
bundle = load_bundle(args.bundle)
model = bundle["model"]
threshold = float(bundle["config"].get("probability_threshold", 0.5))
stats = bundle["norm_stats"]
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

rows = []
totals = {
    "true_positive": 0,
    "false_positive": 0,
    "true_negative": 0,
    "false_negative": 0,
    "annotated_pixel_count": 0,
    "unannotated_valid_pixel_count": 0,
}
evaluated_tile_ids = []
any_negative_annotation = False

for tile_id, features in sorted(by_tile.items()):
    s1_path = kit / tile_id / "s1_vv_vh.tif"
    if not s1_path.exists():
        print(f"skip {tile_id}: kit tile missing")
        continue
    with rasterio.open(s1_path) as src:
        image = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs

    from app.geospatial.reprojection import reproject_geometry

    def _shapes(polarity):
        return [
            (reproject_geometry(shp_shape(f["geometry"]), "EPSG:4326", crs).__geo_interface__, 1)
            for f in features
            if f.get("geometry") and f["polarity"] == polarity
        ]

    positive_mask = rasterize(
        _shapes("positive"), out_shape=image.shape[1:], transform=transform, fill=0, dtype="uint8"
    ).astype(bool)
    negative_shapes = _shapes("negative")
    negative_mask = (
        rasterize(negative_shapes, out_shape=image.shape[1:], transform=transform, fill=0, dtype="uint8").astype(bool)
        if negative_shapes
        else np.zeros_like(positive_mask)
    )
    annotated = positive_mask | negative_mask
    if annotated.sum() < 32:
        print(f"skip {tile_id}: annotated pixels < 32")
        continue
    if negative_mask.any():
        any_negative_annotation = True
    evaluated_tile_ids.append(tile_id)

    valid = np.isfinite(image).all(axis=0)
    bands = []
    for band_index in range(image.shape[0]):
        from app.ml.features import band_to_db

        band = band_to_db(image[band_index], s1_unit)
        bands.append(normalize(band, stats["mean"][band_index], stats["std"][band_index]))
    image = np.stack(bands)
    image[:, ~valid] = 0.0
    prob = predict_tiles(model, image, valid, device=device)
    # Predictions are computed over the full tile; the confusion matrix simply
    # restricts which pixels are scored. Predictions outside the annotations are
    # NOT discarded as a mask (that would hide false positives).
    pred = (prob >= threshold) & valid

    counts = annotated_confusion_counts(pred, positive_mask, negative_mask, valid)
    metrics = classification_metrics(counts)
    for key in totals:
        totals[key] += counts[key]
    rows.append({
        "tile": tile_id,
        "annotated_px": counts["annotated_pixel_count"],
        "positive_px": int(positive_mask.sum()),
        "negative_px": int(negative_mask.sum()),
        "unannotated_valid_px": counts["unannotated_valid_pixel_count"],
        **{k: (round(v, 4) if v is not None else None) for k, v in metrics.items()},
    })

if not evaluated_tile_ids:
    print("No tiles had enough annotated pixels to evaluate.")
    sys.exit(0)

aggregate = classification_metrics(totals)
full_binary = is_full_binary_evaluation(totals)
print("=== hand-label evaluation (U-Net @", threshold, ",", s1_unit, ") ===")
for row in rows:
    print(row)
print("aggregate:", {k: (round(v, 4) if v is not None else None) for k, v in aggregate.items()})
print(
    f"evaluated tiles: {len(evaluated_tile_ids)}; annotated pixels: {totals['annotated_pixel_count']}; "
    f"unannotated valid pixels excluded: {totals['unannotated_valid_pixel_count']}"
)
print(
    "FULL BINARY EVALUATION: "
    + ("yes (both positive and negative annotations present)" if full_binary
       else "no (positive-only annotations; precision/F1/IoU cannot expose false positives outside drawn polygons)")
)

scope = "full binary" if full_binary else "positive-only (partial)"
note = (
    f"Operator hand labels on {len(evaluated_tile_ids)} Hakaluki tiles ({scope} evaluation); "
    "metrics computed over explicitly annotated valid pixels only; unannotated pixels excluded "
    "and never treated as background. This is operator-annotated reference validation, not an "
    "independent external reference."
)
if args.observation_date:
    note = f"Observation date {args.observation_date}. " + note
observation_tag = args.observation_date or "unspecified observation date"

if args.dry_run:
    print("DRY RUN: metrics above NOT persisted to the database.")
    sys.exit(0)

factory = get_session_factory()
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
    split_strategy = f"operator hand labels, Hakaluki AOI ({observation_tag}), {scope} annotations"
    metric_rows = [
        AnalysisMetric(
            analysis_run_id=run.id,
            method_key="hand_validation_unet",
            method_name=f"U-Net vs operator hand labels ({len(evaluated_tile_ids)} tiles)",
            precision_score=aggregate["precision"],
            recall_score=aggregate["recall"],
            f1_score=aggregate["f1"],
            iou_score=aggregate["iou"],
            true_positive=totals["true_positive"],
            false_positive=totals["false_positive"],
            true_negative=totals["true_negative"],
            false_negative=totals["false_negative"],
            test_pixel_count=totals["annotated_pixel_count"],
            split_strategy=split_strategy,
            evaluated_at=now,
            notes=note,
        )
    ]
    otsu_totals = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "annotated_pixel_count": 0,
        "unannotated_valid_pixel_count": 0,
    }
    for tile_id, features in sorted(by_tile.items()):
        if tile_id not in evaluated_tile_ids:
            continue
        s1_path = kit / tile_id / "s1_vv_vh.tif"
        with rasterio.open(s1_path) as src:
            image = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs

        def _shapes2(polarity):
            return [
                (reproject_geometry(shp_shape(f["geometry"]), "EPSG:4326", crs).__geo_interface__, 1)
                for f in features
                if f.get("geometry") and f["polarity"] == polarity
            ]

        positive_mask = rasterize(
            _shapes2("positive"), out_shape=image.shape[1:], transform=transform, fill=0, dtype="uint8"
        ).astype(bool)
        negative_shapes = _shapes2("negative")
        negative_mask = (
            rasterize(negative_shapes, out_shape=image.shape[1:], transform=transform, fill=0, dtype="uint8").astype(bool)
            if negative_shapes
            else np.zeros_like(positive_mask)
        )
        valid = np.isfinite(image).all(axis=0)
        from app.ml.features import band_to_db

        vh_db = band_to_db(image[1], s1_unit)
        otsu_mask, _ = otsu_flood_mask(vh_db, valid, water_is_low=True)
        counts = annotated_confusion_counts(otsu_mask, positive_mask, negative_mask, valid)
        for key in otsu_totals:
            otsu_totals[key] += counts[key]
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
        test_pixel_count=otsu_totals["annotated_pixel_count"],
        split_strategy=split_strategy,
        evaluated_at=now,
        notes=note,
    ))
    repo.replace_metrics(run.id, metric_rows)
    if full_binary:
        repo.update_run(run, validation_status="operator_annotated", validation_note=note)
        status_label = "operator_annotated"
    else:
        repo.update_run(run, validation_status="operator_annotated_positive_only", validation_note=note)
        status_label = "operator_annotated_positive_only"
    print(f"persisted hand-label metrics for run {run.id} (validation_status={status_label})")
