import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.evaluate import classification_metrics, confusion_counts
from app.ml.features import ensure_db, normalize
from app.ml.inference import load_bundle, predict_tiles

chips_dir = Path(r"data\prepared\test-chips")
# Sen1Floods11 hand-labelled S1 chips store VV/VH in dB. Declared explicitly.
SAR_UNIT = "dB"
bundle = load_bundle(r"models\unet\0.1.0")
model = bundle["model"]
stats = bundle["norm_stats"]
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
model.to(device)

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.55]
BUCKETS = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 1000.0)]

per_chip = []
for s1_path in sorted(chips_dir.glob("*_S1Hand.tif")):
    chip_id = s1_path.name.replace("_S1Hand.tif", "")
    with rasterio.open(s1_path) as src:
        image = src.read().astype(np.float32)
    with rasterio.open(chips_dir / f"{chip_id}_LabelHand.tif") as src:
        label = src.read(1)
    valid = np.isfinite(image).all(axis=0) & (label != -1)
    if valid.sum() == 0:
        continue
    bands = []
    for band_index in range(image.shape[0]):
        band, _ = ensure_db(image[band_index], SAR_UNIT)
        bands.append(normalize(band, stats["mean"][band_index], stats["std"][band_index]))
    image = np.stack(bands)
    image[:, ~valid] = 0.0
    prob = predict_tiles(model, image, valid, device=device)
    water_pct = float((label == 1)[valid].mean() * 100)
    row = {"chip": chip_id, "water_pct": round(water_pct, 2), "probs": {}}
    for threshold in THRESHOLDS:
        counts = confusion_counts((prob >= threshold) & valid, label == 1, valid)
        metrics = classification_metrics(counts)
        row["probs"][threshold] = metrics["iou"]
    per_chip.append(row)

print("=== mean IoU by threshold and water coverage (TEST chips (verify rule chosen on val)) ===")
header = "bucket          chips  " + "  ".join(f"t={t:.2f}" for t in THRESHOLDS)
print(header)
for low, high in BUCKETS:
    selected = [r for r in per_chip if low <= r["water_pct"] < high]
    cells = []
    for threshold in THRESHOLDS:
        ious = [r["probs"][threshold] for r in selected if r["probs"][threshold] is not None]
        cells.append(f"{np.mean(ious):.3f}" if ious else "  n/a")
    print(f"{f'{low:g}-{high:g}%':<14}  {len(selected):>4}   " + "  ".join(cells))
overall = []
for threshold in THRESHOLDS:
    ious = [r["probs"][threshold] for r in per_chip if r["probs"][threshold] is not None]
    overall.append(f"{np.mean(ious):.3f}")
print(f"{'ALL':<14}  {len(per_chip):>4}   " + "  ".join(overall))

