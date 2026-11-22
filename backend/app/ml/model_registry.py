import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

BUNDLE_FILES = ["checkpoint.pt", "config.json", "norm_stats.json"]

MANIFEST_PATH = Path(__file__).resolve().parent / "bundles_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle_manifest() -> dict:
    """Load the tracked reproducibility manifest of model bundles."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verify_bundle(version: str, model_root: str | Path) -> dict:
    """Verify an on-disk bundle against the tracked manifest checksums.

    Returns a report dict with ``ok`` plus per-file checks. A missing artifact or
    a checksum mismatch is reported, never silently ignored.
    """
    manifest = load_bundle_manifest()
    expected = manifest.get("bundles", {}).get(version)
    if expected is None:
        return {"ok": False, "reason": f"version {version!r} not in bundles_manifest.json"}
    bundle_dir = Path(model_root) / "unet" / version
    report = {"ok": True, "version": version, "checks": {}}
    for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("norm_stats.json", "norm_stats_sha256")):
        path = bundle_dir / name
        if not path.exists():
            report["checks"][name] = "missing"
            report["ok"] = False
            continue
        actual = _sha256(path)
        expected_hash = expected.get(key)
        match = expected_hash is not None and actual == expected_hash
        report["checks"][name] = {"actual": actual, "expected": expected_hash, "match": match}
        if not match:
            report["ok"] = False
    return report


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
