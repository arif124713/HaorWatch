# %% [markdown]
# # HaorWatch AI — U-Net Flood Segmentation Training (Kaggle)
#
# **Run this on Kaggle with a GPU accelerator (T4 x2 or P100).** Nothing here runs locally.
#
# 1. Kaggle: `Create` -> `New Notebook` -> `File` -> `Import Notebook` -> upload `train_haorwatch_unet.ipynb`
# 2. Session options: Accelerator = **GPU T4 x2** (or P100), Internet = **On**
# 3. `Run All`. It downloads Sen1Floods11 v1.1, trains a 2D U-Net, evaluates it
#    **and** an Otsu baseline on the same official held-out hand-labelled test
#    split, then packages an artifact bundle.
# 4. Download `haorwatch_unet_v0.1.0.zip` from the Output panel and unzip it into
#    `backend/models/` (see final cell for the exact command).

# %%
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "seed": 42,
    "dataset": "sen1floods11",
    "dataset_version": "v1.1",
    "model_family": "unet",
    "model_name": "HaorWatch Flood Classifier",
    "model_version": "0.3.0",
    "in_channels": 2,
    "out_channels": 1,
    "band_order": ["VV_dB", "VH_dB"],
    "base_channels": 32,
    "tile_size": 512,
    "train_crop": 256,
    "crop_water_bias": 0.5,
    "batch_size": 16,
    "epochs": 60,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "num_workers": 2,
    "loss": "bce_tversky",
    "tversky_alpha_fp": 0.3,
    "tversky_beta_fn": 0.7,
    "optimizer": "adamw",
    "min_valid_pixels_per_chip": 32,
    "weak_pretrain": {
        "enabled": True,
        "epochs": 8,
        "max_chips": 4000,
        "label_semantics": "S1OtsuLabelWeak: Otsu-derived weak water labels (noisier than hand labels)",
        "leakage_rule": "weak chips from validation/test EVENTS are excluded from pretraining",
    },
    "threshold_selection": "max mean per-chip IoU on validation split",
}

SEED = CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

WORK = Path("/kaggle/working")
ARTIFACTS = WORK / f"haorwatch_unet_v{CONFIG['model_version']}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("/kaggle/temp/sen1floods11")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: no GPU attached. Enable a GPU in Notebook options before running training.")

# %% [markdown]
# ## 1. Download Sen1Floods11 v1.1 (hand-labelled subsets + official splits)

# %%
BUCKET = "gs://sen1floods11"
SUBSETS = [
    "v1.1/splits",
    "v1.1/data/flood_events/HandLabeled/S1Hand",
    "v1.1/data/flood_events/HandLabeled/LabelHand",
    "v1.1/data/flood_events/WeaklyLabeled/S1Weak",
    "v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak",
]
LICENSE_URLS = [
    "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/main/LICENSE",
    "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/LICENSE",
]


def run_shell(command: str) -> None:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr[-2000:]}")


if not (DATA_ROOT / "v1.1" / "splits").exists():
    run_shell("pip install -q gsutil")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        target = DATA_ROOT / subset
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {subset} ...")
        run_shell(f'gsutil -m cp -r "{BUCKET}/{subset}" "{target.parent}"')

print("Download verification:")
print(f"  split files: {len(list((DATA_ROOT / 'v1.1' / 'splits' / 'flood_handlabeled').glob('*.csv')))}")
print(f"  S1 chips: {len(list((DATA_ROOT / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'S1Hand').glob('*.tif')))}")
print(f"  label chips: {len(list((DATA_ROOT / 'v1.1' / 'data' / 'flood_events' / 'HandLabeled' / 'LabelHand').glob('*.tif')))}")

import httpx

license_text = None
for url in LICENSE_URLS:
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        if response.status_code == 200:
            license_text = response.text
            break
    except Exception as error:
        print(f"License fetch failed from {url}: {error}")
if license_text:
    (ARTIFACTS / "LICENSE_dataset.txt").write_text(license_text, encoding="utf-8")

IMAGE_DIR = DATA_ROOT / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
LABEL_DIR = DATA_ROOT / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
SPLITS_DIR = DATA_ROOT / "v1.1" / "splits" / "flood_handlabeled"

SPLIT_IDS = {}
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
}
for name, split_file in SPLIT_FILES.items():
    ids = []
    missing = []
    for line in (SPLITS_DIR / split_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        chip_id = line.split(",")[0].strip().replace("_S1Hand.tif", "")
        if (IMAGE_DIR / f"{chip_id}_S1Hand.tif").exists() and (LABEL_DIR / f"{chip_id}_LabelHand.tif").exists():
            ids.append(chip_id)
        else:
            missing.append(chip_id)
    SPLIT_IDS[name] = ids
    if missing:
        print(f"WARNING: {name}: {len(missing)} split chips missing on disk (e.g. {missing[:3]})")
print({name: len(ids) for name, ids in SPLIT_IDS.items()})

VAL_TEST_EVENTS = {chip.rsplit("_", 1)[0] for name in ("val", "test") for chip in SPLIT_IDS[name]}
WEAK_IMAGE_DIR = DATA_ROOT / "v1.1" / "data" / "flood_events" / "WeaklyLabeled" / "S1Weak"
WEAK_LABEL_DIR = DATA_ROOT / "v1.1" / "data" / "flood_events" / "WeaklyLabeled" / "S1OtsuLabelWeak"
WEAK_IDS = []
if CONFIG["weak_pretrain"]["enabled"] and WEAK_IMAGE_DIR.exists():
    weak_all = sorted(p.name.replace("_S1Weak.tif", "") for p in WEAK_IMAGE_DIR.glob("*_S1Weak.tif"))
    weak_allowed = [c for c in weak_all if c.rsplit("_", 1)[0] not in VAL_TEST_EVENTS]
    random.Random(SEED).shuffle(weak_allowed)
    WEAK_IDS = weak_allowed[: CONFIG["weak_pretrain"]["max_chips"]]
    excluded = len(weak_all) - len(weak_allowed)
    print(f"Weak pretrain pool: {len(WEAK_IDS)} chips kept, {excluded} excluded (val/test events)")
else:
    print("Weak pretrain disabled or weak data missing; training from scratch on hand labels.")

# %% [markdown]
# ## 2. Dataset audit (recorded into the artifact bundle)

# %%


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_value_sample(chip_ids: list[str], limit: int = 40) -> dict:
    counts = {"flood": 0, "non_flood": 0, "nodata": 0}
    for chip_id in chip_ids[:limit]:
        with rasterio.open(LABEL_DIR / f"{chip_id}_LabelHand.tif") as src:
            values = src.read(1)
        counts["flood"] += int((values == 1).sum())
        counts["non_flood"] += int((values == 0).sum())
        counts["nodata"] += int((values == -1).sum())
    return counts


audit = {
    "dataset": "sen1floods11",
    "version": CONFIG["dataset_version"],
    "source_bucket": BUCKET,
    "audit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "subset": "HandLabeled (S1Hand + LabelHand) with official flood_handlabeled splits",
    "split_chip_counts": {name: len(ids) for name, ids in SPLIT_IDS.items()},
    "s1_chip_count": len(list(IMAGE_DIR.glob("*.tif"))),
    "label_chip_count": len(list(LABEL_DIR.glob("*.tif"))),
    "label_schema": "0=non-flood, 1=flood, -1=nodata/invalid (hand-labelled)",
    "s1_bands": ["VV", "VH"],
    "weak_pretrain_pool": {
        "enabled": CONFIG["weak_pretrain"]["enabled"],
        "chips_kept": len(WEAK_IDS),
        "label_semantics": CONFIG["weak_pretrain"]["label_semantics"],
        "leakage_rule": CONFIG["weak_pretrain"]["leakage_rule"],
    },
    "label_value_sample_counts": {
        name: label_value_sample(ids) for name, ids in SPLIT_IDS.items()
    },
    "license_note": (
        "Sen1Floods11 is distributed for research use; verify the exact license in the "
        "repository LICENSE file copied into this bundle. Record it before any redistribution."
    ),
    "split_strategy": "sen1floods11_official_event_split (event/geographic separation, no random chip mixing)",
    "label_type": "hand_labelled",
}
(ARTIFACTS / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))

# %% [markdown]
# ## 3. Dataset, normalization statistics, U-Net, loss

# %%


def to_db(linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(linear, 1e-6, None))


def looks_like_db(band: np.ndarray) -> bool:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return True
    return float(np.median(finite)) < 0.0


class FloodChipDataset(Dataset):
    def __init__(self, image_dir, label_dir, chip_ids, norm_stats=None, assume_db=True,
                 image_suffix="_S1Hand.tif", label_suffix="_LabelHand.tif",
                 train_crop=None, crop_water_bias=0.0):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.chip_ids = list(chip_ids)
        self.norm_stats = norm_stats
        self.assume_db = assume_db
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.train_crop = train_crop
        self.crop_water_bias = crop_water_bias

    def __len__(self):
        return len(self.chip_ids)

    def _crop_window(self, flood, valid, height, width):
        crop = self.train_crop
        if crop is None or crop >= height:
            return 0, 0
        has_water = bool((flood == 1)[valid].any())
        if has_water and torch.rand(1).item() < self.crop_water_bias:
            water_rows, water_cols = torch.nonzero(
                torch.from_numpy(((flood == 1) & valid).astype(np.uint8)), as_tuple=True
            )
            index = int(torch.randint(0, water_rows.shape[0], (1,)).item())
            center_row = int(water_rows[index].item())
            center_col = int(water_cols[index].item())
        else:
            center_row = int(torch.randint(0, height, (1,)).item())
            center_col = int(torch.randint(0, width, (1,)).item())
        top = max(0, min(center_row - crop // 2, height - crop))
        left = max(0, min(center_col - crop // 2, width - crop))
        return top, left

    def __getitem__(self, index):
        chip_id = self.chip_ids[index]
        with rasterio.open(self.image_dir / f"{chip_id}{self.image_suffix}") as src:
            image = src.read().astype(np.float32)
        with rasterio.open(self.label_dir / f"{chip_id}{self.label_suffix}") as src:
            label = src.read(1).astype(np.float32)

        valid = np.isfinite(image).all(axis=0) & (label != -1)
        flood = (label == 1).astype(np.float32)

        height, width = image.shape[1:]
        top, left = self._crop_window(flood, valid, height, width)
        if self.train_crop is not None and self.train_crop < height:
            bottom = min(top + self.train_crop, height)
            right = min(left + self.train_crop, width)
            image = image[:, top:bottom, left:right]
            valid = valid[top:bottom, left:right]
            flood = flood[top:bottom, left:right]

        for band_index in range(image.shape[0]):
            band = image[band_index]
            if not self.assume_db and valid.any() and float(np.median(band[valid])) > 0:
                band = to_db(band.astype(np.float64)).astype(np.float32)
            if self.norm_stats is not None:
                band = (band - self.norm_stats["mean"][band_index]) / self.norm_stats["std"][band_index]
            image[band_index] = np.where(valid, band, 0.0).astype(np.float32)

        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(flood[None].copy()),
            torch.from_numpy(valid[None].copy()),
            chip_id,
        )


def scan_assume_db(chip_ids, sample_size=8):
    medians = []
    for chip_id in chip_ids[:sample_size]:
        path = IMAGE_DIR / f"{chip_id}_S1Hand.tif"
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        finite = data[np.isfinite(data)]
        if finite.size:
            medians.append(float(np.median(finite)))
    return bool(not medians or np.median(medians) < 0.0)


ASSUME_DB = scan_assume_db(SPLIT_IDS["train"])
print(f"S1 unit detection: assume_db={ASSUME_DB}")


def compute_norm_stats(chip_ids, max_chips=400):
    sums = None
    sq_sums = None
    count = 0
    stats_dataset = FloodChipDataset(IMAGE_DIR, LABEL_DIR, chip_ids, assume_db=ASSUME_DB)
    for index in range(0, min(len(stats_dataset), max_chips)):
        image, _, valid, _ = stats_dataset[index]
        image = image.numpy()
        valid_np = valid[0].numpy()
        if valid_np.sum() == 0:
            continue
        stacked = image[:, valid_np]
        sums = stacked.sum(axis=1) if sums is None else sums + stacked.sum(axis=1)
        sq_sums = (stacked ** 2).sum(axis=1) if sq_sums is None else sq_sums + (stacked ** 2).sum(axis=1)
        count += stacked.shape[1]
    if count == 0:
        raise RuntimeError("No valid pixels found; audit the dataset download.")
    mean = sums / count
    variance = np.maximum(sq_sums / count - mean ** 2, 1e-8)
    return {
        "mean": [float(v) for v in mean],
        "std": [float(np.sqrt(v)) for v in variance],
        "pixel_count": int(count),
        "computed_on": "train_split_only",
        "s1_unit": "dB" if ASSUME_DB else "linear_converted_to_dB",
    }


NORM_STATS = compute_norm_stats(SPLIT_IDS["train"])
print("Norm stats:", json.dumps(NORM_STATS, indent=2))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]))
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.d1 = Down(b, 2 * b)
        self.d2 = Down(2 * b, 4 * b)
        self.d3 = Down(4 * b, 8 * b)
        self.d4 = Down(8 * b, 16 * b)
        self.u1 = Up(16 * b, 8 * b)
        self.u2 = Up(8 * b, 4 * b)
        self.u3 = Up(4 * b, 2 * b)
        self.u4 = Up(2 * b, b)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, valid):
        valid_float = valid.float()
        raw = self.bce(logits, target)
        bce = (raw * valid_float).sum() / valid_float.sum().clamp(min=1.0)
        probs = torch.sigmoid(logits) * valid_float
        targets = target * valid_float
        intersection = (probs * targets).sum()
        denominator = probs.sum() + targets.sum()
        if denominator == 0:
            dice = logits.sum() * 0.0
        else:
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        return bce + dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha_fp=0.3, beta_fn=0.7, smooth=1.0):
        super().__init__()
        self.alpha_fp = alpha_fp
        self.beta_fn = beta_fn
        self.smooth = smooth

    def forward(self, logits, target, valid):
        valid_float = valid.float()
        probs = torch.sigmoid(logits) * valid_float
        targets = target * valid_float
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets) * valid_float).sum()
        fn = ((1.0 - probs) * targets * valid_float).sum()
        denominator = tp + self.alpha_fp * fp + self.beta_fn * fn
        if denominator == 0:
            return logits.sum() * 0.0
        return 1.0 - (tp + self.smooth) / (denominator + self.smooth)


class BCETverskyLoss(nn.Module):
    def __init__(self, alpha_fp, beta_fn):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.tversky = TverskyLoss(alpha_fp, beta_fn)

    def forward(self, logits, target, valid):
        valid_float = valid.float()
        raw = self.bce(logits, target)
        bce = (raw * valid_float).sum() / valid_float.sum().clamp(min=1.0)
        return bce + self.tversky(logits, target, valid)


def confusion_counts(pred, label, valid):
    p = pred[valid].astype(bool)
    y = label[valid].astype(bool)
    return {
        "true_positive": int(np.logical_and(p, y).sum()),
        "false_positive": int(np.logical_and(p, ~y).sum()),
        "true_negative": int(np.logical_and(~p, ~y).sum()),
        "false_negative": int(np.logical_and(~p, y).sum()),
    }


def classification_metrics(counts):
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1_den = 2 * tp + fp + fn
    f1 = 2 * tp / f1_den if f1_den > 0 else None
    iou_den = tp + fp + fn
    iou = tp / iou_den if iou_den > 0 else None
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def merge_confusion(total, counts):
    for key in ("true_positive", "false_positive", "true_negative", "false_negative"):
        total[key] = total.get(key, 0) + counts.get(key, 0)
    return total

# %% [markdown]
# ## 4. Training with checkpointing, early stopping on validation IoU

# %%
def make_loader(split_name, shuffle, train_crop=None):
    dataset = FloodChipDataset(
        IMAGE_DIR, LABEL_DIR, SPLIT_IDS[split_name], norm_stats=NORM_STATS, assume_db=ASSUME_DB,
        train_crop=train_crop, crop_water_bias=CONFIG["crop_water_bias"] if train_crop else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
        drop_last=shuffle,
    )


train_loader = make_loader("train", shuffle=True, train_crop=CONFIG["train_crop"])
val_loader = make_loader("val", shuffle=False)

model = UNet(
    in_channels=CONFIG["in_channels"],
    out_channels=CONFIG["out_channels"],
    base_channels=CONFIG["base_channels"],
).to(DEVICE)
if CONFIG["loss"] == "bce_tversky":
    criterion = BCETverskyLoss(CONFIG["tversky_alpha_fp"], CONFIG["tversky_beta_fn"])
else:
    criterion = BCEDiceLoss()

if WEAK_IDS:
    print(f"Phase 1: pretraining on {len(WEAK_IDS)} weak chips for {CONFIG['weak_pretrain']['epochs']} epochs")
    weak_loader = DataLoader(
        FloodChipDataset(
            WEAK_IMAGE_DIR, WEAK_LABEL_DIR, WEAK_IDS, norm_stats=NORM_STATS, assume_db=ASSUME_DB,
            image_suffix="_S1Weak.tif", label_suffix="_S1OtsuLabelWeak.tif",
            train_crop=CONFIG["train_crop"], crop_water_bias=CONFIG["crop_water_bias"],
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )
    weak_optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    model.train()
    for weak_epoch in range(1, CONFIG["weak_pretrain"]["epochs"] + 1):
        running = 0.0
        steps = 0
        for images, floods, valids, _ in weak_loader:
            images = images.to(DEVICE, non_blocking=True)
            floods = floods.to(DEVICE, non_blocking=True)
            valids = valids.to(DEVICE, non_blocking=True)
            weak_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(images)
            loss = criterion(logits, floods, valids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            weak_optimizer.step()
            running += float(loss.item())
            steps += 1
        print({"phase": "weak_pretrain", "epoch": weak_epoch, "loss": round(running / max(steps, 1), 5)})

torch.manual_seed(SEED)
np.random.seed(SEED)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")


@torch.no_grad()
def collect_probs(loader):
    probs, floods, valids, chip_ids = [], [], [], []
    model.eval()
    for images, floods_batch, valid_batch, ids in loader:
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(images)
        probability = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for index in range(probability.shape[0]):
            label = floods_batch[index, 0].numpy()
            valid = valid_batch[index, 0].numpy().astype(bool) & (label != -1)
            probs.append(probability[index])
            floods.append(label.astype(bool))
            valids.append(valid)
            chip_ids.extend(list(ids))
    return probs, floods, valids, chip_ids


def iou_at_threshold(probs, floods, valids, threshold):
    total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for prob, flood, valid in zip(probs, floods, valids):
        if valid.sum() == 0:
            continue
        pred = (prob >= threshold) & valid
        merge_confusion(total, confusion_counts(pred, flood, valid))
    return classification_metrics(total)["iou"], total


def iou_stats_at_threshold(probs, floods, valids, threshold):
    total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    chip_ious = []
    for prob, flood, valid in zip(probs, floods, valids):
        if valid.sum() == 0:
            continue
        counts = confusion_counts((prob >= threshold) & valid, flood, valid)
        merge_confusion(total, counts)
        chip_iou = classification_metrics(counts)["iou"]
        if chip_iou is not None:
            chip_ious.append(chip_iou)
    aggregate = classification_metrics(total)["iou"]
    mean_chip_iou = float(np.mean(chip_ious)) if chip_ious else None
    return aggregate, mean_chip_iou, total


log_rows = []
best_iou = -1.0
best_epoch = -1
patience_left = CONFIG["patience"]

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    train_loss = 0.0
    batches = 0
    for images, floods, valids, _ in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        floods = floods.to(DEVICE, non_blocking=True)
        valids = valids.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(images)
        loss = criterion(logits, floods, valids)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += float(loss.item())
        batches += 1

    probs, floods_np, valids_np, _ = collect_probs(val_loader)
    val_iou_agg, val_iou_mean, _ = iou_stats_at_threshold(probs, floods_np, valids_np, 0.5)
    scheduler.step(val_iou_mean or 0.0)

    row = {
        "epoch": epoch,
        "train_loss": round(train_loss / max(batches, 1), 5),
        "val_iou_aggregate": round(val_iou_agg, 5) if val_iou_agg is not None else None,
        "val_iou_mean_per_chip": round(val_iou_mean, 5) if val_iou_mean is not None else None,
        "lr": optimizer.param_groups[0]["lr"],
    }
    log_rows.append(row)
    print(row)

    if (val_iou_mean or 0.0) > best_iou:
        best_iou = val_iou_mean
        best_epoch = epoch
        patience_left = CONFIG["patience"]
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch, "val_iou_mean_per_chip": val_iou_mean},
            ARTIFACTS / "checkpoint.pt",
        )
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} (val IoU {best_iou:.4f})")
            break

with open(ARTIFACTS / "train_log.csv", "w", encoding="utf-8") as handle:
    handle.write("epoch,train_loss,val_iou_aggregate,val_iou_mean_per_chip,lr\n")
    for row in log_rows:
        handle.write(
            f"{row['epoch']},{row['train_loss']},{row['val_iou_aggregate']},{row['val_iou_mean_per_chip']},{row['lr']}\n"
        )

print(f"Best mean per-chip validation IoU {best_iou:.4f} at epoch {best_epoch}")

# %% [markdown]
# ## 5. Threshold selection on validation, evaluation on held-out test
#
# The probability threshold is selected **only on the validation split**, then
# applied unchanged to the untouched official test split. The Otsu baseline is
# evaluated on exactly the same test chips and valid-pixel mask.

# %%
checkpoint = torch.load(ARTIFACTS / "checkpoint.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])

val_probs, val_floods, val_valids, _ = collect_probs(val_loader)
best_threshold = 0.5
best_threshold_iou = -1.0
for threshold in np.arange(0.10, 0.91, 0.05):
    _, mean_iou, _ = iou_stats_at_threshold(val_probs, val_floods, val_valids, float(threshold))
    if (mean_iou or 0.0) > best_threshold_iou:
        best_threshold_iou = mean_iou
        best_threshold = float(threshold)
print(f"Selected probability threshold {best_threshold:.2f} (mean per-chip val IoU {best_threshold_iou:.4f})")

test_loader = make_loader("test", shuffle=False)
test_probs, test_floods, test_valids, test_ids = collect_probs(test_loader)

unet_total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
otsu_total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
per_event = {}
per_chip_rows = []
otsu_thresholds = []

for prob, flood, valid, chip_id in zip(test_probs, test_floods, test_valids, test_ids):
    if valid.sum() == 0:
        continue
    event = chip_id.rsplit("_", 1)[0] if "_" in chip_id else chip_id
    event_stats = per_event.setdefault(
        event,
        {
            "pixels": 0,
            "unet": {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0},
            "otsu": {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0},
        },
    )
    event_stats["pixels"] += int(valid.sum())

    pred = (prob >= best_threshold) & valid
    counts = confusion_counts(pred, flood, valid)
    merge_confusion(unet_total, counts)
    merge_confusion(event_stats["unet"], counts)
    chip_metrics = classification_metrics(counts)
    per_chip_rows.append({
        "chip": chip_id,
        "event": event,
        "water_pct": round(float((flood & valid).sum() / valid.sum() * 100), 2),
        "pixels": int(valid.sum()),
        "iou": round(chip_metrics["iou"], 4) if chip_metrics["iou"] is not None else None,
        "f1": round(chip_metrics["f1"], 4) if chip_metrics["f1"] is not None else None,
        "precision": round(chip_metrics["precision"], 4) if chip_metrics["precision"] is not None else None,
        "recall": round(chip_metrics["recall"], 4) if chip_metrics["recall"] is not None else None,
    })

    with rasterio.open(IMAGE_DIR / f"{chip_id}_S1Hand.tif") as src:
        vh = src.read(2).astype(np.float32)
    finite_vh = vh[np.isfinite(vh) & valid]
    if finite_vh.size > 1:
        lo, hi = np.percentile(finite_vh, [1, 99])
        if hi > lo:
            hist, edges = np.histogram(finite_vh, bins=256, range=(lo, hi))
            centers = (edges[:-1] + edges[1:]) / 2.0
            w1 = np.cumsum(hist)
            w2 = w1[-1] - w1
            s = np.cumsum(hist * centers)
            m1 = s / np.maximum(w1, 1e-12)
            m2 = (s[-1] - s) / np.maximum(w2, 1e-12)
            between = w1[:-1] * w2[:-1] * (m1[:-1] - m2[:-1]) ** 2
            otsu_thr = float(centers[int(np.argmax(between))])
            otsu_pred = (vh < otsu_thr) & valid
            otsu_thresholds.append(otsu_thr)
            otsu_counts = confusion_counts(otsu_pred, flood, valid)
            merge_confusion(otsu_total, otsu_counts)
            merge_confusion(event_stats["otsu"], otsu_counts)


def method_summary(counts):
    return {
        **classification_metrics(counts),
        **counts,
        "test_pixel_count": sum(counts.values()),
    }


BUCKETS = [(0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 1000.0)]
by_water = {}
for low, high in BUCKETS:
    rows = [r for r in per_chip_rows if r["water_pct"] is not None and low <= r["water_pct"] < high]
    if not rows:
        continue
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    by_water[f"{low:g}-{high:g}%"] = {
        "chips": len(rows),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou": round(float(np.median(ious)), 4) if ious else None,
        "min_iou": round(float(min(ious)), 4) if ious else None,
        "mean_recall": round(float(np.mean(recalls)), 4) if recalls else None,
    }


evaluation = {
    "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "test_dataset": "sen1floods11_v1.1_handlabeled",
    "split_strategy": audit["split_strategy"],
    "test_chip_count": len(SPLIT_IDS["test"]),
    "positive_class": "flood",
    "probability_threshold": best_threshold,
    "threshold_selected_on": "validation_split_only, by mean per-chip IoU",
    "unet": method_summary(unet_total),
    "otsu_baseline": {
        **method_summary(otsu_total),
        "band": "VH_dB",
        "rule": "flood when VH dB < per-chip Otsu threshold",
        "mean_threshold_dB": float(np.mean(otsu_thresholds)) if otsu_thresholds else None,
    },
    "mean_per_chip_iou": round(float(np.mean([r["iou"] for r in per_chip_rows if r["iou"] is not None])), 4)
    if any(r["iou"] is not None for r in per_chip_rows)
    else None,
    "by_water_coverage": by_water,
    "per_chip": per_chip_rows,
    "per_event": {
        event: {
            "pixels": stats["pixels"],
            "unet": method_summary(stats["unet"]),
            "otsu": method_summary(stats["otsu"]),
        }
        for event, stats in per_event.items()
    },
    "limitations": [
        "Public flood labels may not fully represent haor conditions.",
        "Internal held-out testing only; no independent external validation.",
        "Otsu threshold is unsupervised and per-chip; U-Net threshold fixed from validation.",
    ],
}
(ARTIFACTS / "metrics_eval.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
print(json.dumps({k: evaluation[k] for k in ("unet", "otsu_baseline")}, indent=2))

# %% [markdown]
# ## 6. Package the reproducible artifact bundle

# %%
final_config = {
    **CONFIG,
    "architecture": {
        "family": "unet",
        "in_channels": CONFIG["in_channels"],
        "out_channels": CONFIG["out_channels"],
        "base_channels": CONFIG["base_channels"],
    },
    "probability_threshold": best_threshold,
    "s1_unit": "dB" if ASSUME_DB else "linear_converted_to_dB",
    "norm_stats_file": "norm_stats.json",
    "band_order_note": "channel 0 = VV, channel 1 = VH (dB)",
    "dataset_reference": {
        "name": "Sen1Floods11",
        "version": "v1.1",
        "source": BUCKET,
        "subset": "HandLabeled",
        "label_type": "hand_labelled",
        "license": "See LICENSE_dataset.txt; record exact terms before redistribution.",
    },
    "training": {
        "optimizer": CONFIG["optimizer"],
        "lr": CONFIG["lr"],
        "weight_decay": CONFIG["weight_decay"],
        "loss": CONFIG["loss"],
        "tversky_alpha_fp": CONFIG["tversky_alpha_fp"],
        "tversky_beta_fn": CONFIG["tversky_beta_fn"],
        "batch_size": CONFIG["batch_size"],
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_iou_mean_per_chip": best_iou,
        "early_stopping": "patience=8 on mean per-chip validation IoU",
        "threshold_selection": "max mean per-chip val IoU over 0.10-0.90, applied unchanged to test",
        "weak_pretrain": {
            **CONFIG["weak_pretrain"],
            "chips_used": len(WEAK_IDS),
            "note": "Weak (Otsu-derived) labels used for phase-1 pretraining only; "
            "chips from validation/test events excluded; fine-tuning and all "
            "evaluation use hand labels only.",
        },
    },
    "code_version": f"kaggle-notebook-{time.strftime('%Y%m%d')}",
}
(ARTIFACTS / "config.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")
(ARTIFACTS / "norm_stats.json").write_text(json.dumps(NORM_STATS, indent=2), encoding="utf-8")
(ARTIFACTS / "split_manifest.json").write_text(json.dumps(SPLIT_IDS, indent=2), encoding="utf-8")

model_card = {
    "model_name": CONFIG["model_name"],
    "model_family": "unet",
    "model_version": CONFIG["model_version"],
    "task": "pixel-wise flood / non-flood segmentation from Sentinel-1 VV+VH",
    "intended_use": "HaorWatch AI MVP flood extent mapping; potential agricultural exposure estimation.",
    "out_of_scope": [
        "flood forecasting or real-time warning",
        "verified crop-loss estimation",
        "nationwide coverage claims",
    ],
    "training_data": final_config["dataset_reference"],
    "weak_pretrain_disclosure": {
        "data": "S1OtsuLabelWeak (Otsu-derived weak labels)",
        "usage": "phase-1 pretraining only; validation and test use hand labels only",
        "leakage_control": "weak chips from validation/test events excluded",
    },
    "evaluation": {
        "test_split": "official Sen1Floods11 hand-labelled test (event-separated)",
        "unet": evaluation["unet"],
        "otsu_baseline": evaluation["otsu_baseline"],
        "mean_per_chip_iou": evaluation.get("mean_per_chip_iou"),
        "by_water_coverage": evaluation.get("by_water_coverage"),
        "validation_status": "internal_test_only",
    },
    "inputs": {
        "bands": CONFIG["band_order"],
        "normalization": NORM_STATS,
        "probability_threshold": best_threshold,
    },
    "artifacts": ["checkpoint.pt", "config.json", "norm_stats.json", "split_manifest.json", "metrics_eval.json"],
}
(ARTIFACTS / "model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

zip_base = WORK / f"haorwatch_unet_v{CONFIG['model_version']}"
shutil.make_archive(str(zip_base), "zip", root_dir=WORK, base_dir=ARTIFACTS.name)
zip_path = WORK / f"haorwatch_unet_v{CONFIG['model_version']}.zip"
print("Bundle contents:")
for path in sorted(ARTIFACTS.iterdir()):
    print(f"  {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
print(f"Checkpoint sha256: {sha256_of(ARTIFACTS / 'checkpoint.pt')}")
print(f"\nDONE. Download {zip_path.name} from the Output panel, then on your PC:")
print(f"  Expand-Archive haorwatch_unet_v{CONFIG['model_version']}.zip -DestinationPath backend\\models\\unet\\")
print(f"  (creates backend\\models\\unet\\haorwatch_unet_v{CONFIG['model_version']}\\)")
print("  Then run: cd backend; ..\\venv\\Scripts\\python.exe -m scripts.train_model --register models\\unet\\haorwatch_unet_v" + CONFIG["model_version"])
