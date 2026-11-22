import json
from pathlib import Path

import pytest
import torch

from app.ml.features import SarUnitError
from app.ml.inference import load_bundle
from app.ml.unet import UNet


def _write_bundle(directory: Path, *, config_unit, stats_unit, with_stats=True):
    directory.mkdir(parents=True, exist_ok=True)
    model = UNet(in_channels=2, out_channels=1, base_channels=4)
    torch.save({"state_dict": model.state_dict()}, directory / "checkpoint.pt")
    config = {
        "architecture": {"in_channels": 2, "out_channels": 1, "base_channels": 4},
        "probability_threshold": 0.4,
    }
    if config_unit is not None:
        config["s1_unit"] = config_unit
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if with_stats:
        stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
        if stats_unit is not None:
            stats["s1_unit"] = stats_unit
        (directory / "norm_stats.json").write_text(json.dumps(stats), encoding="utf-8")


def test_bundle_unit_from_config(tmp_path):
    _write_bundle(tmp_path, config_unit="dB", stats_unit="dB")
    bundle = load_bundle(tmp_path)
    assert bundle["s1_unit"] == "dB"


def test_bundle_unit_from_stats_only(tmp_path):
    _write_bundle(tmp_path, config_unit=None, stats_unit="linear")
    bundle = load_bundle(tmp_path)
    assert bundle["s1_unit"] == "linear"


def test_bundle_conflicting_units_rejected(tmp_path):
    _write_bundle(tmp_path, config_unit="dB", stats_unit="linear")
    with pytest.raises(SarUnitError):
        load_bundle(tmp_path)


def test_bundle_missing_unit_rejected(tmp_path):
    _write_bundle(tmp_path, config_unit=None, stats_unit=None)
    with pytest.raises(SarUnitError):
        load_bundle(tmp_path)


def test_bundle_invalid_unit_rejected(tmp_path):
    _write_bundle(tmp_path, config_unit="bananas", stats_unit="bananas")
    with pytest.raises(SarUnitError):
        load_bundle(tmp_path)
