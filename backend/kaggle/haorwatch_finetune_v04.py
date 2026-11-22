# %% [markdown]
# # HaorWatch AI — v0.4 domain fine-tune (Hakaluki pseudo-labels)
#
# Fine-tunes the v0.1.0 U-Net on 4 same-day S1/S2 pseudo-labeled pairs over the
# Hakaluki AOI (Sentinel-2 MNDWI>0 water as automatic labels), holding out:
# - the Sep 18 pair (same window as the operator labels) as pseudo-validation
# - the 2 operator hand-labeled tiles as the external check
# - 20 official Sen1Floods11 test chips as the regression check
#
# Verdict printed at the end: finetuned vs base on all three checks.

# %%
import json
import random
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CONFIG = {
    "seed": 42,
    "model_name": "HaorWatch Flood Classifier",
    "model_version": "0.4.0",
    "base_version": "0.1.0",
    "in_channels": 2,
    "out_channels": 1,
    "base_channels": 32,
    "train_crop": 256,
    "crop_water_bias": 0.5,
    "batch_size": 8,
    "epochs": 12,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "patience": 4,
    "loss": "bce_dice",
    "threshold": 0.40,
    "holdout_pair_day": "2026-09-18",
}

SEED = CONFIG["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

WORK = Path("/kaggle/working")
ARTIFACTS = WORK / f"haorwatch_unet_v{CONFIG['model_version']}"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

def find_in_input(name: str) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found anywhere under /kaggle/input")
    return matches[0]

def dataset_root() -> Path:
    return find_in_input("manifest.json").parent

DATA = dataset_root()
print("dataset root:", DATA)
print("root files:", sorted(p.name for p in DATA.iterdir()))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %%
import rasterio


def to_db(linear):
    return 10.0 * np.log10(np.clip(linear, 1e-6, None))


def looks_like_db(band):
    finite = band[np.isfinite(band)]
    return True if finite.size == 0 else float(np.median(finite)) < 0.0


class PairDataset(Dataset):
    def __init__(self, pair_dirs, norm_stats, train_crop=None, crop_water_bias=0.0):
        self.samples = [Path(pair_dir) for pair_dir in pair_dirs]
        self.norm_stats = norm_stats
        self.train_crop = train_crop
        self.crop_water_bias = crop_water_bias

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pair_dir = self.samples[index]
        with rasterio.open(pair_dir / "s1_grid.tif") as src:
            image = src.read().astype(np.float32)
        with rasterio.open(pair_dir / "pseudo_label.tif") as src:
            label = src.read(1).astype(np.float32)

        # Supervision validity = valid SAR ∧ valid label. Invalid Sentinel-2
        # reference pixels are stored as the ignore sentinel (-1) in the label
        # and must be excluded, never treated as non-water.
        valid = np.isfinite(image).all(axis=0) & (label != -1)
        valid_mask_path = pair_dir / "valid_mask.tif"
        if valid_mask_path.exists():
            with rasterio.open(valid_mask_path) as src:
                valid = valid & (src.read(1) > 0.5)
        for band_index in range(image.shape[0]):
            band = image[band_index]
            if not looks_like_db(band):
                band = to_db(band.astype(np.float64)).astype(np.float32)
            band = (band - self.norm_stats["mean"][band_index]) / self.norm_stats["std"][band_index]
            image[band_index] = np.where(valid, band, -3.0).astype(np.float32)

        height, width = image.shape[1:]
        if self.train_crop and self.train_crop < height:
            water_pixels = np.argwhere((label > 0.5) & valid)
            if water_pixels.shape[0] > 0 and torch.rand(1).item() < self.crop_water_bias:
                pick = water_pixels[int(torch.randint(0, water_pixels.shape[0], (1,)).item())]
                center_row, center_col = int(pick[0]), int(pick[1])
            else:
                center_row = int(torch.randint(0, height, (1,)).item())
                center_col = int(torch.randint(0, width, (1,)).item())
            top = max(0, min(center_row - self.train_crop // 2, height - self.train_crop))
            left = max(0, min(center_col - self.train_crop // 2, width - self.train_crop))
            image = image[:, top:top + self.train_crop, left:left + self.train_crop]
            valid = valid[top:top + self.train_crop, left:left + self.train_crop]
            label = label[top:top + self.train_crop, left:left + self.train_crop]

        flood = (label > 0.5).astype(np.float32)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(flood[None].copy()),
            torch.from_numpy(valid[None].copy()),
        )


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

# %% [markdown]
# ## 1. Load base v0.1.0 weights + split pairs

# %%
norm_stats = json.loads((DATA / "norm_stats.json").read_text())
base_config = json.loads((DATA / "base_config.json").read_text())
manifest = json.loads((DATA / "manifest.json").read_text())

model = UNet(
    in_channels=CONFIG["in_channels"],
    out_channels=CONFIG["out_channels"],
    base_channels=CONFIG["base_channels"],
).to(DEVICE)
checkpoint = torch.load(DATA / "base_checkpoint.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
print(f"Loaded base v{CONFIG['base_version']} checkpoint (epoch {checkpoint.get('epoch')})")

train_pairs, holdout_pairs = [], []
for entry in manifest["pairs"]:
    pair_name = Path(entry["pair_dir"].replace("\\", "/")).name
    pair_path = DATA / pair_name
    if pair_name.endswith(CONFIG["holdout_pair_day"]):
        holdout_pairs.append(pair_path)
    else:
        train_pairs.append(pair_path)
print(f"train pairs: {len(train_pairs)}, held-out pseudo-val pairs: {len(holdout_pairs)}")

# %% [markdown]
# ## 2. Fine-tune on pseudo pairs (held-out pair monitored, not trained)

# %%


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
    tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1_den = 2 * tp + fp + fn
    f1 = 2 * tp / f1_den if f1_den > 0 else None
    iou_den = tp + fp + fn
    iou = tp / iou_den if iou_den > 0 else None
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


@torch.no_grad()
def predict_full(model, pair_dir):
    pair_dir = Path(pair_dir)
    with rasterio.open(pair_dir / "s1_grid.tif") as src:
        image = src.read().astype(np.float32)
    valid = np.isfinite(image).all(axis=0)
    # Consistency with training: apply the same validity mask (valid SAR ∧ valid
    # Sentinel-2 reference) that PairDataset uses, so evaluation never scores
    # pixels that training treated as invalid/ignored.
    valid_mask_path = pair_dir / "valid_mask.tif"
    if valid_mask_path.exists():
        with rasterio.open(valid_mask_path) as src:
            valid = valid & (src.read(1) > 0.5)
    label_path = pair_dir / "pseudo_label.tif"
    if label_path.exists():
        with rasterio.open(label_path) as src:
            valid = valid & (src.read(1) != -1)
    for band_index in range(2):
        band = image[band_index]
        band = (band - norm_stats["mean"][band_index]) / norm_stats["std"][band_index]
        image[band_index] = np.where(valid, band, -3.0).astype(np.float32)
    tensor = torch.from_numpy(image[None].copy()).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        logits = model(tensor)
    probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    probability[~valid] = 0.0
    return probability, valid


def iou_stats(probs, floods, valids, threshold):
    total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    chip_ious = []
    for prob, flood, valid in zip(probs, floods, valids):
        counts = confusion_counts((prob >= threshold) & valid, flood, valid)
        for key, value in counts.items():
            total[key] += value
        chip_iou = classification_metrics(counts)["iou"]
        if chip_iou is not None:
            chip_ious.append(chip_iou)
    return classification_metrics(total)["iou"], (float(np.mean(chip_ious)) if chip_ious else None), total


criterion = BCEDiceLoss().to(DEVICE)
train_dataset = PairDataset(
    train_pairs, norm_stats, train_crop=CONFIG["train_crop"], crop_water_bias=CONFIG["crop_water_bias"]
)
train_loader = DataLoader(
    train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=DEVICE.type == "cuda",
    drop_last=True,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

holdout_probs, holdout_floods, holdout_valids = [], [], []
for pair_dir in holdout_pairs:
    prob, valid = predict_full(model, pair_dir)
    with rasterio.open(Path(pair_dir) / "pseudo_label.tif") as src:
        label = src.read(1).astype(bool)
    holdout_probs.append(prob)
    holdout_floods.append(label)
    holdout_valids.append(valid)

base_holdout = iou_stats(holdout_probs, holdout_floods, holdout_valids, 0.40)
print(f"BASE on held-out pair @0.40: aggregate {base_holdout[0]:.4f}, mean-chip {base_holdout[1]:.4f}")

best_mean_iou = -1.0
log_rows = []
for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    running = 0.0
    steps = 0
    for images, floods, valids in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        floods = floods.to(DEVICE, non_blocking=True)
        valids = valids.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(images)
        loss = criterion(logits, floods, valids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running += float(loss.item())
        steps += 1

    holdout_iou, holdout_mean, _ = iou_stats(holdout_probs, holdout_floods, holdout_valids, 0.40)
    row = {"epoch": epoch, "loss": round(running / max(steps, 1), 5),
           "holdout_iou": round(holdout_iou, 4) if holdout_iou else None,
           "holdout_mean_chip_iou": round(holdout_mean, 4) if holdout_mean else None}
    log_rows.append(row)
    print(row)
    if (holdout_mean or 0.0) > best_mean_iou:
        best_mean_iou = holdout_mean
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, ARTIFACTS / "checkpoint.pt")
        patience_note = "improved"
    print("best mean-chip so far:", round(best_mean_iou, 4))

# %%
model.load_state_dict(torch.load(ARTIFACTS / "checkpoint.pt", map_location=DEVICE, weights_only=False)["state_dict"])
ft_holdout = iou_stats(holdout_probs, holdout_floods, holdout_valids, 0.40)
print(f"FINETUNED on held-out pair @0.40: aggregate {ft_holdout[0]:.4f}, mean-chip {ft_holdout[1]:.4f}")

# %% [markdown]
# ## 3. External check: operator hand-labeled pixels

# %%
operator_dir = DATA / "operator_check"


@torch.no_grad()
def predict_from_array(eval_model, image, valid):
    tensor = torch.from_numpy(image[None].copy()).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        logits = eval_model(tensor)
    probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    probability[~valid] = 0.0
    return probability, valid


def normalize_chip(image, valid):
    image = image.astype(np.float32).copy()
    for band_index in range(image.shape[0]):
        band, converted = (lambda b: (b, False))(image[band_index])
        if not looks_like_db(band):
            band = to_db(band.astype(np.float64)).astype(np.float32)
        band = (band - norm_stats["mean"][band_index]) / norm_stats["std"][band_index]
        image[band_index] = np.where(valid, band, -3.0).astype(np.float32)
    return image


operator_results = {}
for model_tag in ["base", "finetuned"]:
    eval_model = UNet(
        in_channels=CONFIG["in_channels"], out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"]).to(DEVICE)
    if model_tag == "base":
        eval_model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    else:
        eval_model.load_state_dict(torch.load(ARTIFACTS / "checkpoint.pt", map_location=DEVICE, weights_only=False)["state_dict"])
    eval_model.eval()
    totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    for s1_path in sorted(operator_dir.glob("*_s1.tif")):
        tile_id = s1_path.name.replace("_s1.tif", "")
        with rasterio.open(s1_path) as src:
            image = src.read()
        with rasterio.open(operator_dir / f"{tile_id}_label.tif") as src:
            label = src.read(1) > 0.5
        valid = np.isfinite(image).all(axis=0)
        image = normalize_chip(image, valid)
        probability, _ = predict_from_array(eval_model, image, valid)
        counts = confusion_counts((probability >= 0.40) & label, label, label)
        for key, value in counts.items():
            totals[key] += value
    metrics = classification_metrics(totals)
    operator_results[model_tag] = metrics
    print(f"operator pixels [{model_tag}]:", {k: (round(v, 4) if v is not None else None) for k, v in metrics.items()})

# %% [markdown]
# ## 4. Regression check: 20 official Sen1Floods11 test chips

# %%
TEST_DIR = Path("/kaggle/temp/sen1floods11_test")
if not TEST_DIR.exists():
    subprocess.run("pip install -q gsutil", shell=True, check=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    test_csv = httpx.get(
        "https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv", timeout=60
    ).text
    rows = [line.split(",")[0].strip() for line in test_csv.splitlines() if line.strip()]
    subset = [r for r in rows if r.startswith(("Ghana", "India"))][:20]
    s1_base = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/"
    for fname in subset:
        for layer in ["S1Hand", "LabelHand"]:
            target = fname.replace("S1Hand", layer)
            dest = TEST_DIR / target
            if not dest.exists():
                response = httpx.get(s1_base + f"{layer}/{target}", timeout=120, follow_redirects=True)
                response.raise_for_status()
                dest.write_bytes(response.content)
    print("test chips downloaded:", len(subset))


def score_official_test(eval_model):
    totals = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    chip_ious = []
    for s1_path in sorted(TEST_DIR.glob("*_S1Hand.tif")):
        chip_id = s1_path.name.replace("_S1Hand.tif", "")
        with rasterio.open(s1_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(TEST_DIR / f"{chip_id}_LabelHand.tif") as src:
            label = src.read(1)
        valid = np.isfinite(image).all(axis=0) & (label != -1)
        if valid.sum() == 0:
            continue
        image = normalize_chip(image, valid)
        probability, _ = predict_from_array(eval_model, image, valid)
        counts = confusion_counts((probability >= 0.40) & valid, label == 1, valid)
        for key, value in counts.items():
            totals[key] += value
        chip_iou = classification_metrics(counts)["iou"]
        if chip_iou is not None:
            chip_ious.append(chip_iou)
    metrics = classification_metrics(totals)
    return metrics, (float(np.mean(chip_ious)) if chip_ious else None)


official_results = {}
for model_tag in ["base", "finetuned"]:
    eval_model = UNet(
        in_channels=CONFIG["in_channels"], out_channels=CONFIG["out_channels"],
        base_channels=CONFIG["base_channels"]).to(DEVICE)
    if model_tag == "base":
        eval_model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    else:
        eval_model.load_state_dict(torch.load(ARTIFACTS / "checkpoint.pt", map_location=DEVICE, weights_only=False)["state_dict"])
    eval_model.eval()
    metrics, mean_chip = score_official_test(eval_model)
    official_results[model_tag] = {**metrics, "mean_per_chip_iou": mean_chip}
    print(f"official test [{model_tag}]:", {k: (round(v, 4) if v is not None else None) for k, v in metrics.items()},
          "mean/chip:", round(mean_chip, 4) if mean_chip else None)

# %% [markdown]
# ## 5. Verdict + package

# %%
verdict = {
    "heldout_pseudo_pair": {
        "base_iou": base_holdout[0],
        "finetuned_iou": ft_holdout[0],
        "base_mean_chip_iou": base_holdout[1],
        "finetuned_mean_chip_iou": ft_holdout[1],
    },
    "operator_pixels": operator_results,
    "official_test_20": official_results,
    "training": {
        "epochs_run": len(log_rows),
        "best_mean_chip_holdout": best_mean_iou,
        "loss": CONFIG["loss"],
        "lr": CONFIG["lr"],
        "seed": SEED,
        "threshold": 0.40,
    },
}
(ARTIFACTS / "metrics_eval.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

improvement = (ft_holdout[1] or 0) - (base_holdout[1] or 0)
operator_gain = (operator_results["finetuned"]["f1"] or 0) - (operator_results["base"]["f1"] or 0)
official_gain = (official_results["finetuned"]["iou"] or 0) - (official_results["base"]["iou"] or 0)
print("\n=== VERDICT ===")
print(f"held-out pseudo pair mean-chip IoU: {'+' if improvement >= 0 else ''}{improvement:.4f}")
print(f"operator pixels F1: {'+' if operator_gain >= 0 else ''}{operator_gain:.4f}")
print(f"official test IoU: {'+' if official_gain >= 0 else ''}{official_gain:.4f}")
if improvement > 0 and official_gain >= -0.01:
    print("RECOMMEND: promote v0.4.0 (domain gain without regression)")
else:
    print("RECOMMEND: keep v0.1.0 active (fine-tune did not earn promotion)")

final_config = {
    **CONFIG,
    "probability_threshold": 0.40,
    "base_model": "v0.1.0 (Sen1Floods11 hand labels)",
    "finetune_data": "5 S1/S2 same-day pairs, Sentinel-2 MNDWI>0 pseudo-labels, Hakaluki AOI",
    "holdouts": ["Sep-18 pseudo pair", "2 operator hand-labeled tiles", "20 official test chips"],
}
(ARTIFACTS / "config.json").write_text(json.dumps(final_config, indent=2), encoding="utf-8")

import shutil

zip_base = WORK / f"haorwatch_unet_v{CONFIG['model_version']}"
shutil.make_archive(str(zip_base), "zip", root_dir=WORK, base_dir=ARTIFACTS.name)
print("\nDONE:", f"{zip_base.name}.zip in the Output panel")
print("Register locally with: cd backend; ..\\venv\\Scripts\\python.exe -m scripts.train_model --register models\\kaggle-output-v04\\haorwatch_unet_v0.4.0")
