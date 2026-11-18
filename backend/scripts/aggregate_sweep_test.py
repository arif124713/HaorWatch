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
bundle = load_bundle(r"models\unet\0.1.0")
model = bundle["model"]
stats = bundle["norm_stats"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.55]
totals = {t: {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0} for t in THRESHOLDS}

for s1_path in sorted(chips_dir.glob("*_S1Hand.tif")):
    with rasterio.open(s1_path) as src:
        image = src.read().astype(np.float32)
    chip_id = s1_path.name.replace("_S1Hand.tif", "")
    with rasterio.open(chips_dir / f"{chip_id}_LabelHand.tif") as src:
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
    for threshold in THRESHOLDS:
        counts = confusion_counts((prob >= threshold) & valid, label == 1, valid)
        merge = totals[threshold]
        for key, value in counts.items():
            merge[key] += value

print("threshold  precision recall  f1     iou")
for threshold in THRESHOLDS:
    metrics = classification_metrics(totals[threshold])
    cells = [f"{metrics[k]:.3f}" if metrics[k] is not None else " n/a" for k in ("precision", "recall", "f1", "iou")]
    print(f"{threshold:.2f}      " + "   ".join(cells))
