import argparse
import json
import subprocess
from pathlib import Path

from app.core.config import get_settings

BUCKET = "gs://sen1floods11"
SUBSETS = [
    "v1.1/splits",
    "v1.1/data/flood_events/HandLabeled/S1Hand",
    "v1.1/data/flood_events/HandLabeled/LabelHand",
]
LICENSE_URL = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/main/LICENSE"


def audit(data_dir: Path) -> dict:
    splits = {}
    splits_dir = data_dir / "v1.1" / "splits" / "flood_handlabeled"
    if splits_dir.exists():
        for name in ("train", "val", "test"):
            path = splits_dir / f"flood_{name}_data.csv"
            splits[name] = len(path.read_text().splitlines()) if path.exists() else 0
    s1_dir = data_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    label_dir = data_dir / "v1.1" / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    audit_result = {
        "dataset": "sen1floods11",
        "version": "v1.1",
        "split_counts": splits,
        "s1_chip_count": len(list(s1_dir.glob("*.tif"))) if s1_dir.exists() else 0,
        "label_chip_count": len(list(label_dir.glob("*.tif"))) if label_dir.exists() else 0,
        "label_schema": "0=non-flood, 1=flood, -1=nodata/invalid",
        "s1_bands": ["VV", "VH"],
        "license_check": "Verify LICENSE terms in the Sen1Floods11 repository before training.",
        "note": "Hand-labelled chips only for training and held-out evaluation; event-level official splits.",
    }
    return audit_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare Sen1Floods11 training data or AOI inputs.")
    parser.add_argument("--download", action="store_true", help="Download Sen1Floods11 subsets with gsutil")
    parser.add_argument("--out", default=None, help="Target directory (defaults to DATA_ROOT/sen1floods11)")
    parser.add_argument("--audit", action="store_true", help="Audit a local download and write an audit file")
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = Path(args.out) if args.out else Path(settings.data_root) / "sen1floods11"

    if args.download:
        data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pip", "install", "-q", "gsutil"], check=True)
        for subset in SUBSETS:
            target = data_dir / subset
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {subset} ...")
            subprocess.run(["gsutil", "-m", "cp", "-r", f"{BUCKET}/{subset}", str(target.parent)], check=True)
        try:
            import httpx

            response = httpx.get(LICENSE_URL, timeout=30)
            if response.status_code == 200:
                (data_dir / "LICENSE.txt").write_text(response.text, encoding="utf-8")
        except Exception as error:
            print(f"License fetch failed ({error}); verify terms manually.", file=__import__("sys").stderr)

    if args.audit or not args.download:
        result = audit(data_dir)
        out_path = data_dir / "dataset_audit.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
