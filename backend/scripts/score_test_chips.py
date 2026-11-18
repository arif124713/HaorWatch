import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.evaluate import classification_metrics, confusion_counts, merge_confusion
from app.ml.features import ensure_db, normalize
from app.ml.inference import load_bundle, predict_tiles

CHIPS_DIR = Path(r"data\prepared\test-chips")
OUT = Path(r"models\unet\benchmark_20chips.json")


def score(bundle_dir: str) -> dict:
    bundle = load_bundle(bundle_dir)
    model = bundle["model"]
    threshold = bundle["threshold"]
    stats = bundle["norm_stats"]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model.to(device)
    rows = []
    totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for s1_path in sorted(CHIPS_DIR.glob("*_S1Hand.tif")):
        chip_id = s1_path.name.replace("_S1Hand.tif", "")
        with rasterio.open(s1_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(CHIPS_DIR / f"{chip_id}_LabelHand.tif") as src:
            label = src.read(1)
        valid = np.isfinite(image).all(axis=0) & (label != -1)
        if valid.sum() == 0:
            continue
        bands = []
        for band_index in range(image.shape[0]):
            band, _ = ensure_db(image[band_index])
            bands.append(normalize(band, stats["mean"][band_index], stats["std"][band_index]))
        image = np.stack(bands)
        image[:, ~valid] = 0.0
        prob = predict_tiles(model, image, valid, device=device)
        counts = confusion_counts((prob >= threshold) & valid, label == 1, valid)
        merge_confusion(totals, counts)
        metrics = classification_metrics(counts)
        rows.append({
            "chip": chip_id,
            "water_pct": round(float((label == 1)[valid].mean() * 100), 2),
            "iou": metrics["iou"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
    aggregate = classification_metrics(totals)
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    buckets = {}
    for low, high in [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 1000.0)]:
        selected = [r for r in rows if r["water_pct"] is not None and low <= r["water_pct"] < high]
        b_ious = [r["iou"] for r in selected if r["iou"] is not None]
        b_recalls = [r["recall"] for r in selected if r["recall"] is not None]
        buckets[f"{low:g}-{high:g}%"] = {
            "chips": len(selected),
            "mean_iou": round(float(np.mean(b_ious)), 4) if b_ious else None,
            "mean_recall": round(float(np.mean(b_recalls)), 4) if b_recalls else None,
        }
    return {
        "bundle": bundle_dir,
        "threshold": threshold,
        "aggregate": {k: round(v, 4) if v is not None else None for k, v in aggregate.items()},
        "mean_per_chip_iou": round(float(np.mean(ious)), 4) if ious else None,
        "by_water_coverage": buckets,
        "per_chip": rows,
    }


def main():
    results = [score(path) for path in sys.argv[1:]] or []
    OUT.write_text(json.dumps({"benchmark": "20 held-out test chips, local inference path", "results": results}, indent=2), encoding="utf-8")
    for result in results:
        print(result["bundle"], result["aggregate"], "mean_chip_iou:", result["mean_per_chip_iou"])
        for bucket, stats in result["by_water_coverage"].items():
            print("  ", bucket, stats)


if __name__ == "__main__":
    main()
