"""Hakaluki operator-tile diagnostics for the U-Net and Otsu baseline.

Produces a probability histogram, water fraction over VALID pixels only, an
operator positive-label overlay, U-Net and Otsu prediction overlays, a
difference map, the validity mask, and per-tile metrics.

The operator labels are **positive-only**: they mark water pixels but do not
enumerate non-water pixels. Recall / precision over them are positive-class
agreement, not full binary accuracy. No threshold is tuned on these tiles, and
no accuracy claim is made from them.

Run from the ``backend`` directory:

    python scripts/hakaluki_diagnostics.py --bundle models/unet/0.4.0 --out data/hakaluki_diagnostics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio

from app.ml.baseline import otsu_flood_mask
from app.ml.evaluate import annotation_positive_agreement, classification_metrics
from app.ml.features import band_to_db, normalize_unit
from app.ml.inference import load_bundle, predict_tiles

DATA_ROOT = Path("data")
LABELS_DIR = DATA_ROOT / "hakaluki_labels"
LABELKIT_DIR = DATA_ROOT / "hakaluki_labelkit"


def _load_tile(tile_id: str):
    s1_path = LABELKIT_DIR / tile_id / "s1_vv_vh.tif"
    label_path = LABELS_DIR / f"{tile_id}_label.npy"
    if not s1_path.exists() or not label_path.exists():
        return None
    with rasterio.open(s1_path) as src:
        image = src.read().astype(np.float32)
        profile = src.profile.copy()
        bounds = src.bounds
    positive = np.load(label_path).astype(bool)
    return image, positive, profile, bounds


def _prepare_input(image: np.ndarray, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SAR exactly as inference does; return (image_norm, valid)."""
    unit = bundle["s1_unit"]
    data = np.stack([band_to_db(image[i], unit) for i in range(image.shape[0])])
    valid = np.isfinite(data).all(axis=0)
    stats = bundle["norm_stats"]
    for i in range(data.shape[0]):
        data[i] = ((data[i] - stats["mean"][i]) / stats["std"][i]).astype(np.float32)
    data[:, ~valid] = 0.0
    return data, valid


def _fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    denominator = int(valid.sum())
    return float((mask & valid).sum()) / denominator if denominator else 0.0


def run(bundle_dir: Path, out_dir: Path, tiles: list[str]) -> dict:
    bundle = load_bundle(bundle_dir)
    threshold = float(bundle["threshold"])
    unit = normalize_unit(bundle["s1_unit"])
    model = bundle["model"]

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "bundle": str(bundle_dir),
        "threshold": threshold,
        "s1_unit": unit,
        "annotation_semantics": "positive_only",
        "note": (
            "Operator labels are positive-only; precision/recall over them are "
            "positive-class agreement, not full binary accuracy. No threshold was "
            "tuned on these tiles."
        ),
        "tiles": {},
    }

    for tile_id in tiles:
        loaded = _load_tile(tile_id)
        if loaded is None:
            report["tiles"][tile_id] = {"skipped": "missing s1 tile or label"}
            continue
        image, positive, profile, bounds = loaded
        data, valid = _prepare_input(image, bundle)

        probability = predict_tiles(model, data, valid)
        unet_mask = (probability >= threshold) & valid

        vh_db = band_to_db(image[1], unit)
        otsu_mask, otsu_threshold = otsu_flood_mask(vh_db, valid, water_is_low=True)

        # Metrics over positive-only annotations (unannotated pixels excluded).
        unet_agreement = annotation_positive_agreement(unet_mask, positive, valid)
        otsu_agreement = annotation_positive_agreement(otsu_mask, positive, valid)

        # Probability histogram over valid pixels only.
        valid_probs = probability[valid]
        histogram, edges = np.histogram(valid_probs, bins=20, range=(0.0, 1.0))

        # Difference map: where U-Net and Otsu disagree (over valid pixels only).
        difference_map = (unet_mask.astype(np.int8) - otsu_mask.astype(np.int8))
        difference_map[~valid] = 0
        difference = difference_map[valid]

        tile_report = {
            "valid_pixels": int(valid.sum()),
            "annotated_positive_pixels": int((positive & valid).sum()),
            "operator_positive_only": True,
            "unet_water_fraction_valid_only": _fraction(unet_mask, valid),
            "otsu_water_fraction_valid_only": _fraction(otsu_mask, valid),
            "unet_positive_agreement": {
                k: unet_agreement[k] for k in ("true_positive", "false_negative", "precision", "recall", "f1", "iou")
            },
            "otsu_positive_agreement": {
                k: otsu_agreement[k] for k in ("true_positive", "false_negative", "precision", "recall", "f1", "iou")
            },
            "otsu_threshold_db": float(otsu_threshold) if otsu_threshold is not None else None,
            "probability_histogram": {
                "bins": [float(round(e, 3)) for e in edges.tolist()],
                "counts": [int(c) for c in histogram.tolist()],
            },
            "unet_vs_otsu_agreement": {
                "both_water": int((difference == 0).sum() - ((~unet_mask & ~otsu_mask) & valid).sum()),
                "unet_only": int((difference == 1).sum()),
                "otsu_only": int((difference == -1).sum()),
            },
            "bounds": list(bounds),
        }
        report["tiles"][tile_id] = tile_report

        # Persist diagnostic rasters for manual inspection.
        _write(profile, out_dir / f"{tile_id}_probability.tif", probability)
        _write(profile, out_dir / f"{tile_id}_valid_mask.tif", valid.astype(np.float32))
        _write(profile, out_dir / f"{tile_id}_unet_mask.tif", unet_mask.astype(np.uint8))
        _write(profile, out_dir / f"{tile_id}_otsu_mask.tif", otsu_mask.astype(np.uint8))
        _write(profile, out_dir / f"{tile_id}_operator_positive.tif", positive.astype(np.uint8))
        _write(profile, out_dir / f"{tile_id}_difference.tif", difference_map.astype(np.int8))
        print(f"{tile_id}: valid={int(valid.sum())}, "
              f"U-Net water frac={tile_report['unet_water_fraction_valid_only']:.3f}, "
              f"Otsu water frac={tile_report['otsu_water_fraction_valid_only']:.3f}")

    # Overlap check between tiles (spatial).
    report["tiles_overlap"] = _tiles_overlap(tiles)
    (out_dir / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _tiles_overlap(tiles: list[str]) -> dict:
    boxes = {}
    for tile_id in tiles:
        s1_path = LABELKIT_DIR / tile_id / "s1_vv_vh.tif"
        if s1_path.exists():
            with rasterio.open(s1_path) as src:
                boxes[tile_id] = tuple(src.bounds)
    ids = list(boxes)
    overlaps = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = boxes[ids[i]], boxes[ids[j]]
            intersects = not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
            overlaps[f"{ids[i]}__{ids[j]}"] = intersects
    return overlaps


def _write(profile: dict, path: Path, array: np.ndarray) -> None:
    profile = dict(profile)
    profile.update(driver="GTiff", count=1, height=array.shape[0], width=array.shape[1], dtype=str(array.dtype))
    profile.pop("nodata", None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hakaluki operator-tile diagnostics.")
    parser.add_argument("--bundle", default="models/unet/0.4.0")
    parser.add_argument("--out", default="data/hakaluki_diagnostics")
    parser.add_argument(
        "--tiles", nargs="*",
        default=["hakaluki_r00000_c00512", "hakaluki_r00000_c01024"],
    )
    args = parser.parse_args(argv)
    report = run(Path(args.bundle), Path(args.out), args.tiles)
    print(json.dumps(report["tiles_overlap"], indent=2))


if __name__ == "__main__":
    main()
