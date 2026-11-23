import json
import os
from pathlib import Path

import numpy as np
import torch

import rasterio
from app.geospatial.raster_io import write_raster
from app.ml.features import SarUnitError, band_to_db, normalize, normalize_unit
from app.ml.model_registry import load_bundle_manifest, verify_or_raise
from app.ml.unet import UNet

DEFAULT_THRESHOLD = 0.5
TILE_SIZE = 512
TILE_STRIDE = 256

# A fallback to another bundle is refused unless explicitly opted in. Set
# HAORWATCH_ALLOW_MODEL_FALLBACK=1 to permit selecting a verified alternative
# when the configured default bundle fails verification (documented, deliberate).
_ALLOW_FALLBACK_ENV = "HAORWATCH_ALLOW_MODEL_FALLBACK"


def _bundle_version_of(directory: Path, config: dict) -> str | None:
    """Determine the registered version of a bundle directory.

    Prefers the directory name (which is how bundles are stored under
    ``models/unet/<version>``); falls back to the config's ``model_version``.
    """
    return directory.name or config.get("model_version")


def _resolve_bundle_unit(config: dict, norm_stats: dict, model_dir: Path) -> str:
    """Resolve the declared SAR unit from a model bundle, rejecting conflicts.

    A production model bundle must declare its SAR input unit. The unit is read
    from ``config.json`` (``s1_unit``) and/or ``norm_stats.json`` (``s1_unit``).
    If both are present they must agree; if neither is present the bundle is
    rejected rather than silently assuming dB, because that could apply the
    wrong conversion and corrupt predictions.
    """
    config_unit = config.get("s1_unit")
    stats_unit = norm_stats.get("s1_unit")
    if config_unit is None and stats_unit is None:
        raise SarUnitError(
            f"Model bundle {model_dir} does not declare a SAR input unit. "
            "Add 's1_unit' ('dB' or 'linear') to config.json (and norm_stats.json)."
        )
    if config_unit is not None and stats_unit is not None:
        if normalize_unit(config_unit) != normalize_unit(stats_unit):
            raise SarUnitError(
                f"Model bundle {model_dir} declares conflicting SAR units: "
                f"config.json s1_unit={config_unit!r} vs norm_stats.json s1_unit={stats_unit!r}."
            )
    return normalize_unit(config_unit if config_unit is not None else stats_unit)


def load_bundle(model_dir: str | Path, verify: bool = True) -> dict:
    directory = Path(model_dir)

    # Fail closed: verify all required files + manifest metadata before loading.
    if verify:
        version = directory.name
        manifest = load_bundle_manifest()
        known_versions = set(manifest.get("bundles", {}))
        if version in known_versions:
            root = directory.parent.parent
            verify_or_raise(version, root)
        # A bundle not present in the tracked manifest (e.g. an untracked
        # experimental directory) is allowed only when it is complete on disk;
        # required files must still exist.
        for name in ("config.json", "norm_stats.json", "checkpoint.pt"):
            if not (directory / name).exists():
                raise FileNotFoundError(
                    f"model bundle {directory} is missing required file {name!r}; "
                    "a valid bundle requires checkpoint.pt, config.json and norm_stats.json."
                )

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
    s1_unit = _resolve_bundle_unit(config, norm_stats, directory)
    return {"model": model, "config": config, "norm_stats": norm_stats,
            "threshold": threshold, "s1_unit": s1_unit}


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
    s1_unit: str | None = None,
) -> dict:
    bundle = load_bundle(model_dir)
    model = bundle["model"]
    threshold = threshold if threshold is not None else bundle["threshold"]
    unit = normalize_unit(s1_unit) if s1_unit is not None else bundle["s1_unit"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with rasterio.open(s1_tif) as src:
        data = src.read().astype(np.float32)
        profile = src.profile

    converted = []
    unit_converted = False
    for band_index in range(data.shape[0]):
        band = band_to_db(data[band_index], unit)
        if unit == "linear":
            unit_converted = True
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
        "s1_unit": unit,
        "unit_converted_to_db": unit_converted,
        "valid_pixel_count": int(valid.sum()),
    }
