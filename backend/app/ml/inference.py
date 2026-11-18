import json
from pathlib import Path

import numpy as np
import torch

import rasterio
from app.geospatial.raster_io import write_raster
from app.ml.features import ensure_db, normalize
from app.ml.unet import UNet

DEFAULT_THRESHOLD = 0.5
TILE_SIZE = 512
TILE_STRIDE = 256


def load_bundle(model_dir: str | Path) -> dict:
    directory = Path(model_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    norm_stats = json.loads((directory / "norm_stats.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    arch = config.get("architecture", {})
    model = UNet(
        in_channels=int(arch.get("in_channels", config.get("in_channels", 2))),
        out_channels=int(arch.get("out_channels", 1)),
        base_channels=int(arch.get("base_channels", config.get("base_channels", 32))),
    )
    model.load_state_dict(state_dict)
    model.eval()
    threshold = float(config.get("probability_threshold", DEFAULT_THRESHOLD))
    return {"model": model, "config": config, "norm_stats": norm_stats, "threshold": threshold}


def predict_tiles(model, image: np.ndarray, valid: np.ndarray, tile: int = TILE_SIZE,
                  stride: int = TILE_STRIDE, device: str = "cpu") -> np.ndarray:
    _, height, width = image.shape
    prob_sum = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)
    tensor = torch.from_numpy(image[None].copy())
    with torch.no_grad():
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                row_end = min(row + tile, height)
                col_end = min(col + tile, width)
                row_start = max(0, row_end - tile)
                col_start = max(0, col_end - tile)
                patch = tensor[:, :, row_start:row_end, col_start:col_end]
                logits = model(patch.to(device))
                probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prob_sum[row_start:row_end, col_start:col_end] += probability
                weight[row_start:row_end, col_start:col_end] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        probability_map = np.where(weight > 0, prob_sum / np.maximum(weight, 1e-12), 0.0)
    probability_map[~valid] = 0.0
    return probability_map.astype(np.float32)


def predict_raster(
    model_dir: str | Path,
    s1_tif: str | Path,
    out_dir: str | Path,
    threshold: float | None = None,
    device: str | None = None,
) -> dict:
    bundle = load_bundle(model_dir)
    model = bundle["model"]
    threshold = threshold if threshold is not None else bundle["threshold"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with rasterio.open(s1_tif) as src:
        data = src.read().astype(np.float32)
        profile = src.profile

    converted = []
    was_linear = False
    for band_index in range(data.shape[0]):
        band, converted_flag = ensure_db(data[band_index])
        was_linear = was_linear or converted_flag
        converted.append(band)
    data = np.stack(converted)

    valid = np.isfinite(data).all(axis=0)
    for band_index in range(data.shape[0]):
        data[band_index] = normalize(
            data[band_index],
            bundle["norm_stats"]["mean"][band_index],
            bundle["norm_stats"]["std"][band_index],
        )
    data[:, ~valid] = 0.0

    probability = predict_tiles(model, data, valid, device=device)
    flood_mask = (probability >= threshold) & valid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prob_path = write_raster(out_dir / "flood_prob.tif", probability, profile["transform"], profile["crs"])
    mask_path = write_raster(
        out_dir / "flood_mask.tif", flood_mask.astype(np.uint8), profile["transform"], profile["crs"], dtype="uint8", nodata=None
    )
    return {
        "probability_path": str(prob_path),
        "mask_path": str(mask_path),
        "threshold": threshold,
        "unit_converted_to_db": was_linear,
        "valid_pixel_count": int(valid.sum()),
    }
