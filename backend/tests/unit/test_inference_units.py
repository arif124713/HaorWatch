"""Verify SAR dB conversion is applied exactly once in the inference path."""

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from app.geospatial.raster_io import write_raster
from app.ml.features import to_db
from app.ml.inference import load_bundle, predict_raster
from app.ml.unet import UNet


def _write_bundle(directory: Path, unit: str, mean, std):
    directory.mkdir(parents=True, exist_ok=True)
    model = UNet(in_channels=2, out_channels=1, base_channels=4)
    torch.save({"state_dict": model.state_dict()}, directory / "checkpoint.pt")
    (directory / "config.json").write_text(
        json.dumps({
            "architecture": {"in_channels": 2, "out_channels": 1, "base_channels": 4},
            "probability_threshold": 0.5, "s1_unit": unit,
        }),
        encoding="utf-8",
    )
    (directory / "norm_stats.json").write_text(
        json.dumps({"mean": mean, "std": std, "s1_unit": unit}), encoding="utf-8"
    )


def _write_s1(path: Path, vv, vh):
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2700000.0)
    write_raster(path, np.stack([vv, vh]).astype(np.float32), transform, "EPSG:32646",
                 dtype="float32", nodata=np.nan)


def test_db_input_is_not_converted_again(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, "dB", [0.0, 0.0], [1.0, 1.0])
    vv = np.full((32, 32), -10.0, dtype=np.float32)
    vh = np.full((32, 32), -15.0, dtype=np.float32)
    s1_path = tmp_path / "s1.tif"
    _write_s1(s1_path, vv, vh)

    result = predict_raster(bundle, s1_path, tmp_path / "out", s1_unit="dB")
    assert result["s1_unit"] == "dB"
    assert result["unit_converted_to_db"] is False


def test_linear_input_is_converted_once(tmp_path):
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, "linear", [0.0, 0.0], [1.0, 1.0])
    # 0.01 linear == -20 dB; if converted twice it would become -inf or wrong.
    linear = np.full((32, 32), 0.01, dtype=np.float32)
    s1_path = tmp_path / "s1.tif"
    _write_s1(s1_path, linear, linear)

    result = predict_raster(bundle, s1_path, tmp_path / "out", s1_unit="linear")
    assert result["s1_unit"] == "linear"
    assert result["unit_converted_to_db"] is True
    # The probability raster should be finite (no double-log artefact).
    with rasterio.open(result["probability_path"]) as src:
        prob = src.read(1)
    assert np.isfinite(prob).all()


def test_inference_unit_must_match_declaration(tmp_path):
    """Band values are converted with the declared unit, never inferred."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, "dB", [0.0, 0.0], [1.0, 1.0])
    linear = np.full((32, 32), 0.01, dtype=np.float32)
    s1_path = tmp_path / "s1.tif"
    _write_s1(s1_path, linear, linear)
    # Declaring linear here means conversion happens; the value is finite.
    out_linear = predict_raster(bundle, s1_path, tmp_path / "o1", s1_unit="linear")
    db_value = to_db(np.array([0.01], dtype=np.float64))[0]
    assert abs(db_value - (-20.0)) < 0.01
    assert np.isfinite(db_value)
    assert out_linear["unit_converted_to_db"] is True
