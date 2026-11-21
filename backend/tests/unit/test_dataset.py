from pathlib import Path

import numpy as np
import rasterio
import pytest
from rasterio.transform import Affine

from app.ml.dataset import Sen1Floods11ChipDataset, scan_declared_unit
from app.ml.features import SarUnitError


def _write_chip(directory: Path, name: str, vv, vh):
    directory.mkdir(parents=True, exist_ok=True)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    with rasterio.open(
        directory / name, "w", driver="GTiff", count=2, height=vv.shape[0], width=vv.shape[1],
        dtype="float32", crs="EPSG:32646", transform=transform,
    ) as dst:
        dst.write(vv.astype("float32"), 1)
        dst.write(vh.astype("float32"), 2)


def _write_label(directory: Path, name: str, label):
    directory.mkdir(parents=True, exist_ok=True)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    with rasterio.open(
        directory / name, "w", driver="GTiff", count=1, height=label.shape[0], width=label.shape[1],
        dtype="float32", crs="EPSG:32646", transform=transform,
    ) as dst:
        dst.write(label.astype("float32"), 1)


def test_scan_declared_unit_matches_s1hand_suffix(tmp_path):
    image_dir = tmp_path / "S1Hand"
    vv = np.full((4, 4), -12.0, dtype="float32")
    vh = np.full((4, 4), -18.0, dtype="float32")
    _write_chip(image_dir, "Ghana_123_S1Hand.tif", vv, vh)
    assert scan_declared_unit(image_dir, ["Ghana_123"]) == "dB"


def test_scan_declared_unit_missing_files_returns_unknown(tmp_path):
    assert scan_declared_unit(tmp_path, ["nope"]) == "unknown"


def test_dataset_declared_db_unit_is_deterministic(tmp_path):
    image_dir = tmp_path / "S1Hand"
    label_dir = tmp_path / "LabelHand"
    vv = np.array([[-12.0, -12.0], [-12.0, -12.0]], dtype="float32")
    vh = np.array([[-18.0, -18.0], [-18.0, -18.0]], dtype="float32")
    label = np.array([[1.0, 0.0], [-1.0, 1.0]], dtype="float32")
    _write_chip(image_dir, "Ghana_1_S1Hand.tif", vv, vh)
    _write_label(label_dir, "Ghana_1_LabelHand.tif", label)

    stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
    dataset = Sen1Floods11ChipDataset(image_dir, label_dir, ["Ghana_1"], norm_stats=stats, s1_unit="dB")
    image, flood, valid, chip_id = dataset[0]
    # dB input is preserved (VV stays -12, VH stays -18) before normalization;
    # invalid pixels (label == -1) are zeroed after normalization.
    np.testing.assert_allclose(image[0].numpy()[valid[0].numpy()], vv[valid.numpy()[0]])
    np.testing.assert_allclose(image[1].numpy()[valid[0].numpy()], vh[valid.numpy()[0]])
    assert chip_id == "Ghana_1"
    assert valid[0].numpy().tolist() == [[True, True], [False, True]]
    assert flood[0].numpy().tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_dataset_declared_linear_unit_converts_to_db(tmp_path):
    image_dir = tmp_path / "S1Hand"
    label_dir = tmp_path / "LabelHand"
    vv = np.full((2, 2), 0.01, dtype="float32")
    vh = np.full((2, 2), 0.01, dtype="float32")
    label = np.zeros((2, 2), dtype="float32")
    _write_chip(image_dir, "India_9_S1Hand.tif", vv, vh)
    _write_label(label_dir, "India_9_LabelHand.tif", label)

    dataset = Sen1Floods11ChipDataset(
        image_dir, label_dir, ["India_9"], norm_stats=None, s1_unit="linear"
    )
    image, _, _, _ = dataset[0]
    assert image[0].numpy()[0, 0] == pytest.approx(-20.0, abs=0.01)


def test_dataset_requires_declared_unit(tmp_path):
    with pytest.raises(SarUnitError):
        Sen1Floods11ChipDataset(tmp_path, tmp_path, ["x"], s1_unit=None)
