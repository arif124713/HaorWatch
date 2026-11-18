import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

BUNDLE_FILES = ["checkpoint.pt", "config.json", "norm_stats.json"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_bundle(src_dir: str | Path, model_root: str | Path, version: str | None = None) -> Path:
    src = Path(src_dir)
    config_path = src / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing in {src}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    version = version or config.get("model_version") or "0.0.0"
    dest = Path(model_root) / "unet" / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        source_file = src / name
        if source_file.exists():
            shutil.copy2(source_file, dest / name)
    for extra in ("metrics_eval.json", "dataset_audit.json", "split_manifest.json"):
        extra_file = src / extra
        if extra_file.exists():
            shutil.copy2(extra_file, dest / extra)
    manifest = {
        "model_name": config.get("model_name", "HaorWatch Flood Classifier"),
        "model_family": config.get("model_family", "unet"),
        "model_version": version,
        "checkpoint_sha256": _sha256(dest / "checkpoint.pt")
        if (dest / "checkpoint.pt").exists()
        else None,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "config": config,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def resolve_model_dir(model_root: str | Path) -> Path | None:
    root = Path(model_root) / "unet"
    if not root.exists():
        return None
    default_file = root / "default.txt"
    if default_file.exists():
        default_version = default_file.read_text(encoding="utf-8").strip()
        default_dir = root / default_version
        if default_dir.is_dir() and (default_dir / "checkpoint.pt").exists():
            return default_dir
    candidates = []
    for version_dir in root.iterdir():
        manifest_path = version_dir / "manifest.json"
        if not (version_dir / "checkpoint.pt").exists():
            continue
        registered_at = ""
        if manifest_path.exists():
            try:
                registered_at = json.loads(manifest_path.read_text(encoding="utf-8")).get("registered_at", "")
            except json.JSONDecodeError:
                registered_at = ""
        candidates.append((registered_at, str(version_dir.name), version_dir))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]
