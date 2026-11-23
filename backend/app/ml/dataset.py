from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import rasterio

from app.ml.features import DEFAULT_SAR_UNIT, normalize_unit


class Sen1Floods11ChipDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_dir: str | Path,
        chip_ids: list[str],
        norm_stats: dict | None = None,
        s1_unit: str = DEFAULT_SAR_UNIT,
        min_valid_pixels: int = 1,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.chip_ids = list(chip_ids)
        self.norm_stats = norm_stats
        self.s1_unit = normalize_unit(s1_unit)
        self.min_valid_pixels = min_valid_pixels

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, index: int):
        chip_id = self.chip_ids[index]
        with rasterio.open(self.image_dir / f"{chip_id}_S1Hand.tif") as src:
            image = src.read().astype(np.float32)
        with rasterio.open(self.label_dir / f"{chip_id}_LabelHand.tif") as src:
            label = src.read(1).astype(np.float32)

        valid = np.isfinite(image).all(axis=0) & (label != -1)
        flood = (label == 1).astype(np.float32)

        # Deterministic unit handling: the declared unit (default dB for
        # Sen1Floods11 hand chips) governs conversion; pixel values never
        # decide the unit.
        if self.s1_unit == "linear":
            from app.ml.features import to_db

            for band_index in range(image.shape[0]):
                image[band_index] = to_db(np.asarray(image[band_index], dtype=np.float64)).astype(np.float32)
        for band_index in range(image.shape[0]):
            band = image[band_index]
            if self.norm_stats is not None:
                mean = self.norm_stats["mean"][band_index]
                std = self.norm_stats["std"][band_index]
                image[band_index] = ((band - mean) / std).astype(np.float32)
        image[:, ~valid] = 0.0

        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(flood[None].copy()),
            torch.from_numpy(valid[None].copy()),
            chip_id,
        )


def scan_declared_unit(image_dir: str | Path, chip_ids: list[str], sample_size: int = 8) -> str:
    """Report the *observed* unit of the training chips for dataset auditing.

    This inspects pixel medians only to *describe* the data; it never drives a
    silent conversion. Training and inference must use an explicitly declared
    unit (see ``s1_unit``). Matches the actual chip filenames, which include the
    ``_S1Hand.tif`` suffix.
    """
    medians = []
    for chip_id in chip_ids[:sample_size]:
        path = Path(image_dir) / f"{chip_id}_S1Hand.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        finite = data[np.isfinite(data)]
        if finite.size:
            medians.append(float(np.median(finite)))
    if not medians:
        return "unknown"
    return "dB" if float(np.median(medians)) < 0.0 else "linear"


def scan_assume_db(image_dir: str | Path, chip_ids: list[str], sample_size: int = 8) -> bool:
    """Deprecated compatibility shim; prefer :func:`scan_declared_unit`."""
    return scan_declared_unit(image_dir, chip_ids, sample_size) == "dB"


def compute_dataset_stats(dataset: Sen1Floods11ChipDataset, max_chips: int = 400) -> dict:
    samples = []
    valid_masks = []
    for index in range(0, min(len(dataset), max_chips)):
        image, _, valid, _ = dataset[index]
        samples.append(image.numpy())
        valid_masks.append(valid[0].numpy())
    from app.ml.features import compute_norm_stats

    return compute_norm_stats(samples, valid_masks)
