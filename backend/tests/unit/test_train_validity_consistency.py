"""Training/evaluation validity consistency for the runtime loss and loader.

Confirms that:
* the ignore label -1 is always excluded from supervision,
* SAR invalid (non-finite) pixels are excluded,
* invalid pixels never become true negatives / false negatives,
* empty-valid batches do not produce NaN loss.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.transform import Affine

from app.ml.dataset import Sen1Floods11ChipDataset
from app.ml.train import BCEDiceLoss, BCETverskyLoss, TverskyLoss


def _write_image(directory: Path, name: str, vv, vh):
    import rasterio

    directory.mkdir(parents=True, exist_ok=True)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    with rasterio.open(
        directory / f"{name}_S1Hand.tif", "w", driver="GTiff", count=2,
        height=vv.shape[0], width=vv.shape[1], dtype="float32",
        crs="EPSG:32646", transform=transform,
    ) as dst:
        dst.write(vv.astype("float32"), 1)
        dst.write(vh.astype("float32"), 2)


def _write_label(directory: Path, name: str, label):
    import rasterio

    directory.mkdir(parents=True, exist_ok=True)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    with rasterio.open(
        directory / f"{name}_LabelHand.tif", "w", driver="GTiff", count=1,
        height=label.shape[0], width=label.shape[1], dtype="float32",
        crs="EPSG:32646", transform=transform,
    ) as dst:
        dst.write(label.astype("float32"), 1)


def test_ignore_label_and_nan_are_excluded(tmp_path):
    image_dir = tmp_path / "S1Hand"
    label_dir = tmp_path / "LabelHand"
    vv = np.array([[-12.0, np.nan], [-12.0, -12.0]], dtype="float32")
    vh = np.array([[-18.0, -18.0], [-18.0, -18.0]], dtype="float32")
    label = np.array([[1.0, 1.0], [0.0, -1.0]], dtype="float32")
    _write_image(image_dir, "G_1", vv, vh)
    _write_label(label_dir, "G_1", label)

    ds = Sen1Floods11ChipDataset(image_dir, label_dir, ["G_1"], norm_stats=None, s1_unit="dB")
    _img, flood, valid, _id = ds[0]
    valid_np = valid[0].numpy()
    # NaN SAR pixel and the -1 ignore pixel are both invalid.
    assert valid_np.tolist() == [[True, False], [True, False]]
    # The ignored pixel (label == -1) must NOT appear as a supervised negative:
    # it is excluded by the validity mask, which the loss/metrics honour.
    assert not valid_np[1, 1]
    # Every pixel the loss trains on is a genuine label; invalid pixels are
    # removed from supervision (not turned into negatives).
    supervised = flood[0].numpy()[valid_np]
    assert supervised.tolist() == [1.0, 0.0]


def test_genuine_non_water_is_supervised_negative(tmp_path):
    image_dir = tmp_path / "S1Hand"
    label_dir = tmp_path / "LabelHand"
    vv = np.full((2, 2), -12.0, dtype="float32")
    vh = np.full((2, 2), -18.0, dtype="float32")
    label = np.array([[0.0, 0.0], [0.0, 0.0]], dtype="float32")
    _write_image(image_dir, "G_2", vv, vh)
    _write_label(label_dir, "G_2", label)
    ds = Sen1Floods11ChipDataset(image_dir, label_dir, ["G_2"], norm_stats=None, s1_unit="dB")
    _img, flood, valid, _id = ds[0]
    assert valid[0].numpy().all()
    assert int(flood.sum()) == 0  # all valid, all genuine negatives


@pytest.mark.parametrize("loss_cls", [BCEDiceLoss, lambda: BCETverskyLoss(0.3, 0.7), TverskyLoss])
def test_empty_valid_batch_does_not_produce_nan(loss_cls):
    loss_fn = loss_cls()
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros((1, 1, 4, 4))
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.bool)  # no valid pixels
    result = loss_fn(logits, target, valid)
    total = result[0] if isinstance(result, tuple) else result
    assert torch.isfinite(total).all(), total


@pytest.mark.parametrize("loss_cls", [BCEDiceLoss, lambda: BCETverskyLoss(0.3, 0.7), TverskyLoss])
def test_invalid_pixels_do_not_affect_loss(loss_cls):
    loss_fn = loss_cls()
    torch.manual_seed(0)
    logits = torch.randn((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros((1, 1, 4, 4))
    valid = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    valid[:, :, :2, :2] = False
    base = loss_fn(logits, target, valid)
    base_total = base[0] if isinstance(base, tuple) else base
    # Corrupt the target in the invalid region: loss must be unchanged.
    target2 = target.clone()
    target2[:, :, :2, :2] = 1.0
    other = loss_fn(logits, target2, valid)
    other_total = other[0] if isinstance(other, tuple) else other
    assert torch.allclose(base_total, other_total, atol=1e-6)
