import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.dataset import Sen1Floods11ChipDataset, compute_dataset_stats, scan_declared_unit
from app.ml.features import DEFAULT_SAR_UNIT
from app.ml.evaluate import classification_metrics, confusion_counts, evaluate_mask, merge_confusion
from app.ml.baseline import otsu_flood_mask
from app.ml.unet import UNet

TRAIN_CONFIG_DEFAULTS = {
    "seed": 42,
    "batch_size": 16,
    "epochs": 60,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 8,
    "num_workers": 4,
    "base_channels": 32,
    "loss": "bce_tversky",
    "tversky_alpha_fp": 0.3,
    "tversky_beta_fn": 0.7,
}


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.sum() == 0 and logits.numel() == 0:
            return logits.sum() * 0.0
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        denominator = probs.sum() + target.sum()
        if denominator == 0:
            return logits.sum() * 0.0
        return 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)

    def forward(self, logits, target, valid):
        raw = self.bce(logits, target)
        valid_float = valid.float()
        bce_count = valid_float.sum().clamp(min=1.0)
        bce = (raw * valid_float).sum() / bce_count
        flat_valid = valid.float().sum()
        if flat_valid > 0:
            probs = torch.sigmoid(logits) * valid_float
            targets = target * valid_float
            intersection = (probs * targets).sum()
            denominator = probs.sum() + targets.sum()
            dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        else:
            dice = logits.sum() * 0.0
        return bce + dice, float(bce.item()), float(dice.item())


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


@torch.no_grad()
def predict_probs(model, loader, device):
    probs, floods, valids, chip_ids = [], [], [], []
    model.eval()
    for images, floods_batch, valid_batch, ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probability = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        # One chip id per sample, paired with its own probability/label/valid.
        # (Extending the whole batch inside the loop duplicated ids and broke the
        # downstream zip() alignment between predictions and chip identities.)
        batch_ids = list(ids)
        for index in range(probability.shape[0]):
            probs.append(probability[index])
            label = floods_batch[index, 0].numpy()
            valids_np = valid_batch[index, 0].numpy().astype(bool) & (label != -1)
            valids.append(valids_np)
            floods.append(label.astype(bool))
            chip_ids.append(batch_ids[index])
    return probs, floods, valids, chip_ids


def iou_from_probs(probs, floods, valids, threshold):
    total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    chip_ious = []
    for prob, flood, valid in zip(probs, floods, valids):
        if valid.sum() == 0:
            continue
        counts = confusion_counts((prob >= threshold) & valid, flood, valid)
        merge_confusion(total, counts)
        chip_metrics = classification_metrics(counts)
        if chip_metrics["iou"] is not None:
            chip_ious.append(chip_metrics["iou"])
    metrics = classification_metrics(total)
    mean_chip_iou = float(np.mean(chip_ious)) if chip_ious else None
    return metrics.get("iou"), mean_chip_iou, total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the HaorWatch U-Net locally (Kaggle GPU recommended).")
    parser.add_argument("--data-dir", required=True, help="Path to the Sen1Floods11 v1.1 directory (contains splits/ and data/)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--s1-unit", default=None, choices=["dB", "linear"],
                        help="Declared SAR input unit for the training chips (default: dB)")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    label_dir = data_dir / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    splits_dir = data_dir / "splits" / "flood_handlabeled"

    split_ids = {}
    split_files = {
        "train": "flood_train_data.csv",
        "val": "flood_valid_data.csv",
        "test": "flood_test_data.csv",
    }
    for name, split_file in split_files.items():
        ids = []
        for line in (splits_dir / split_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            chip_id = line.split(",")[0].strip().replace("_S1Hand.tif", "")
            if (image_dir / f"{chip_id}_S1Hand.tif").exists() and (label_dir / f"{chip_id}_LabelHand.tif").exists():
                ids.append(chip_id)
        split_ids[name] = ids

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device found. Training is intended for Kaggle GPU; CPU will be very slow.")

    s1_unit = args.s1_unit or DEFAULT_SAR_UNIT
    observed_unit = scan_declared_unit(image_dir, split_ids["train"])
    if observed_unit != "unknown" and observed_unit != s1_unit:
        raise SystemExit(
            f"Declared --s1-unit={s1_unit!r} disagrees with observed training chips "
            f"({observed_unit!r}). Refusing to train with a mismatched SAR unit; "
            "resolve the dataset unit explicitly."
        )
    stats_dataset = Sen1Floods11ChipDataset(image_dir, label_dir, split_ids["train"], s1_unit=s1_unit)
    norm_stats = compute_dataset_stats(stats_dataset)
    norm_stats["computed_on"] = "train_split_only"
    norm_stats["s1_unit"] = s1_unit

    config = {**TRAIN_CONFIG_DEFAULTS, "s1_unit": s1_unit, "norm_stats": norm_stats,
              "band_order": ["VV_dB", "VH_dB"]}
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    loaders = {}
    for name in ("train", "val"):
        dataset = Sen1Floods11ChipDataset(
            image_dir, label_dir, split_ids[name], norm_stats=norm_stats, s1_unit=s1_unit
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=name == "train",
            num_workers=config["num_workers"],
            pin_memory=device.type == "cuda",
            drop_last=name == "train",
        )

    model = UNet(in_channels=2, out_channels=1, base_channels=config["base_channels"]).to(device)
    if config["loss"] == "bce_tversky":
        criterion = BCETverskyLoss(config["tversky_alpha_fp"], config["tversky_beta_fn"])
    else:
        criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    log_path = out_dir / "train_log.csv"
    log_rows = []
    best_iou = -1.0
    best_epoch = -1
    patience_left = config["patience"]
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for images, floods, valids, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            floods = floods.to(device, non_blocking=True)
            valids = valids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            loss, _, _ = criterion(logits, floods, valids)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            batches += 1

        probs, floods_np, valids_np, _ = predict_probs(model, loaders["val"], device)
        val_iou, val_iou_mean, _ = iou_from_probs(probs, floods_np, valids_np, 0.5)
        scheduler.step(val_iou_mean or 0.0)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_iou": val_iou,
            "val_iou_mean_per_chip": val_iou_mean,
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)
        print(row)

        if (val_iou_mean or 0.0) > best_iou:
            best_iou = val_iou_mean
            best_epoch = epoch
            patience_left = config["patience"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_iou_mean_per_chip": val_iou_mean}, out_dir / "checkpoint.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} mean per-chip IoU {best_iou:.4f}")
                break

    with open(log_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_iou", "val_iou_mean_per_chip", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)

    checkpoint = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    val_loader = DataLoader(
        Sen1Floods11ChipDataset(image_dir, label_dir, split_ids["val"], norm_stats=norm_stats, s1_unit=s1_unit),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    probs, floods_np, valids_np, _ = predict_probs(model, val_loader, device)
    best_threshold = 0.5
    best_threshold_iou = -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        _, mean_iou, _ = iou_from_probs(probs, floods_np, valids_np, float(threshold))
        if (mean_iou or 0.0) > best_threshold_iou:
            best_threshold_iou = mean_iou
            best_threshold = float(threshold)

    test_loader = DataLoader(
        Sen1Floods11ChipDataset(image_dir, label_dir, split_ids["test"], norm_stats=norm_stats, s1_unit=s1_unit),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    probs, floods_np, valids_np, chip_ids = predict_probs(model, test_loader, device)

    unet_total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    otsu_total = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    per_event = {}
    for prob, flood, valid, chip_id in zip(probs, floods_np, valids_np, chip_ids):
        if valid.sum() == 0:
            continue
        event = chip_id.rsplit("_", 1)[0] if "_" in chip_id else chip_id
        event_stats = per_event.setdefault(event, {"unet": dict(unet_total), "otsu": dict(otsu_total), "pixels": 0})

        pred = (prob >= best_threshold) & valid
        counts = confusion_counts(pred, flood, valid)
        merge_confusion(unet_total, counts)
        merge_confusion(event_stats["unet"], counts)

        with rasterio.open(image_dir / f"{chip_id}_S1Hand.tif") as src:
            vh = src.read(2).astype(np.float32)
        otsu_pred, otsu_threshold = otsu_flood_mask(vh, valid, water_is_low=True)
        otsu_counts = confusion_counts(otsu_pred, flood, valid)
        merge_confusion(otsu_total, otsu_counts)
        merge_confusion(event_stats["otsu"], otsu_counts)
        event_stats["pixels"] += int(valid.sum())

    evaluation = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_split_size": len(split_ids["test"]),
        "probability_threshold": best_threshold,
        "threshold_selected_on": "validation_split, by mean per-chip IoU",
        "split_strategy": "sen1floods11_official_event_split",
        "unet": {**classification_metrics(unet_total), **unet_total},
        "otsu_baseline": {**classification_metrics(otsu_total), **otsu_total},
        "per_event": {
            event: {
                "pixels": stats["pixels"],
                "unet": {**classification_metrics(stats["unet"]), **stats["unet"]},
                "otsu": {**classification_metrics(stats["otsu"]), **stats["otsu"]},
            }
            for event, stats in per_event.items()
        },
        "best_val_iou_mean_per_chip": best_iou,
        "best_epoch": best_epoch,
    }

    (out_dir / "metrics_eval.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in config.items()},
                "model_name": "HaorWatch Flood Classifier",
                "model_family": "unet",
                "model_version": "0.1.0-local",
                "in_channels": 2,
                "out_channels": 1,
                "architecture": {"family": "unet", "in_channels": 2, "out_channels": 1, "base_channels": config["base_channels"]},
                "probability_threshold": best_threshold,
                "dataset": "sen1floods11",
                "dataset_version": "v1.1",
                "loss": config["loss"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "norm_stats.json").write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    print(f"Training complete. Best mean per-chip val IoU {best_iou:.4f} at epoch {best_epoch}; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
