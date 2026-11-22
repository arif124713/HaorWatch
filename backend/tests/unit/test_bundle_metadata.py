"""Model bundle integrity: manifest contract, checksums, and fail-closed loading.

The tracked manifest is the source of truth for what a valid bundle contains.
These tests exercise the verification contract on synthetic bundles (no real
weights required) and, when real bundles are present, confirm they match disk.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from app.ml.features import normalize_unit
from app.ml.inference import load_bundle
from app.ml.model_registry import (
    BundleVerificationError,
    load_bundle_manifest,
    register_bundle,
    resolve_model_dir,
    verify_bundle,
    verify_or_raise,
)
from app.ml.unet import UNet

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "unet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_dirs():
    if not MODEL_ROOT.is_dir():
        return []
    return [d for d in MODEL_ROOT.iterdir() if d.is_dir() and (d / "checkpoint.pt").exists()]


# --------------------------------------------------------------------------- #
# Tracked manifest structure
# --------------------------------------------------------------------------- #
def test_tracked_manifest_is_wellformed():
    manifest = load_bundle_manifest()
    assert manifest["schema_version"] >= 2
    assert manifest["default_version"] in manifest["bundles"]
    for version, entry in manifest["bundles"].items():
        assert entry["s1_unit"] in ("dB", "linear"), version
        assert entry["checkpoint_sha256"], version
        assert entry["config_sha256"], version
        assert entry["norm_stats_sha256"], version
        assert entry["model_version"] == version
        assert entry["band_order"]
        assert entry["probability_threshold"] is not None


@pytest.mark.skipif(not _bundle_dirs(), reason="No model bundles present.")
def test_tracked_manifest_checksums_match_disk():
    manifest = load_bundle_manifest()
    for bundle_dir in _bundle_dirs():
        version = bundle_dir.name
        assert version in manifest["bundles"], f"{version} missing from bundles_manifest.json"
        report = verify_bundle(version, MODEL_ROOT.parent)
        assert report["ok"], f"{version} bundle verification failed: {report}"


@pytest.mark.skipif(not _bundle_dirs(), reason="No model bundles present.")
def test_every_bundle_declares_consistent_sar_unit():
    for bundle_dir in _bundle_dirs():
        config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        stats_path = bundle_dir / "norm_stats.json"
        assert stats_path.exists(), f"{bundle_dir.name} is missing norm_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert "s1_unit" in config, f"{bundle_dir.name} config.json lacks s1_unit"
        assert "s1_unit" in stats, f"{bundle_dir.name} norm_stats.json lacks s1_unit"
        assert normalize_unit(config["s1_unit"]) == normalize_unit(stats["s1_unit"])


@pytest.mark.skipif(not (MODEL_ROOT / "0.4.0" / "checkpoint.pt").exists(), reason="0.4.0 absent")
def test_v04_bundle_loads_with_declared_unit():
    bundle = load_bundle(MODEL_ROOT / "0.4.0")
    assert bundle["s1_unit"] == "dB"
    assert "s1_unit_provenance" in bundle["config"]


# --------------------------------------------------------------------------- #
# Synthetic bundles: build/verify contract without real artifacts
# --------------------------------------------------------------------------- #
def _write_synthetic_bundle(
    directory: Path,
    *,
    version="9.9.9",
    unit="dB",
    bands=("VV_dB", "VH_dB"),
    threshold=0.4,
    include_checkpoint=True,
    include_config=True,
    include_stats=True,
    stats_unit=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    if include_checkpoint:
        model = UNet(in_channels=2, out_channels=1, base_channels=4)
        torch.save({"state_dict": model.state_dict()}, directory / "checkpoint.pt")
    if include_config:
        config = {
            "model_version": version,
            "architecture": {"in_channels": 2, "out_channels": 1, "base_channels": 4},
            "band_order": list(bands),
            "probability_threshold": threshold,
        }
        if unit is not None:
            config["s1_unit"] = unit
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if include_stats:
        stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0], "s1_unit": stats_unit if stats_unit is not None else unit}
        (directory / "norm_stats.json").write_text(json.dumps(stats), encoding="utf-8")


def _maybe_sha256(path: Path) -> str:
    return _sha256(path) if path.exists() else "0" * 64


def _manifest_for(directory: Path, version: str) -> dict:
    return {
        "schema_version": 2,
        "default_version": version,
        "bundles": {
            version: {
                "base_model": None,
                "model_version": version,
                "dataset": "synthetic",
                "s1_unit": "dB",
                "band_order": ["VV_dB", "VH_dB"],
                "probability_threshold": 0.4,
                "checkpoint_sha256": _maybe_sha256(directory / "checkpoint.pt"),
                "config_sha256": _maybe_sha256(directory / "config.json"),
                "norm_stats_sha256": _maybe_sha256(directory / "norm_stats.json"),
            }
        },
    }


@pytest.fixture()
def patched_manifest(tmp_path, monkeypatch):
    """Patch the registry manifest path so synthetic bundles drive verification."""
    manifest_path = tmp_path / "bundles_manifest.json"
    import app.ml.model_registry as registry

    monkeypatch.setattr(registry, "MANIFEST_PATH", manifest_path)
    return manifest_path


def _make_root(tmp_path, version):
    return tmp_path / "models"


def test_valid_bundle_verifies_and_loads(tmp_path, patched_manifest):
    version = "9.9.9"
    bundle = _make_root(tmp_path, version) / "unet" / version
    _write_synthetic_bundle(bundle, version=version)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")

    report = verify_bundle(version, _make_root(tmp_path, version))
    assert report["ok"], report
    loaded = load_bundle(bundle)
    assert loaded["s1_unit"] == "dB"


def test_missing_checkpoint_reported(tmp_path, patched_manifest):
    version = "9.9.9"
    bundle = _make_root(tmp_path, version) / "unet" / version
    _write_synthetic_bundle(bundle, version=version, include_checkpoint=False)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    report = verify_bundle(version, _make_root(tmp_path, version))
    assert not report["ok"]
    assert report["checks"]["checkpoint.pt"] == "missing"


def test_missing_config_prevents_load(tmp_path, patched_manifest):
    version = "9.9.9"
    bundle = _make_root(tmp_path, version) / "unet" / version
    _write_synthetic_bundle(bundle, version=version, include_config=False)
    with pytest.raises(FileNotFoundError):
        load_bundle(bundle)


def test_missing_norm_stats_prevents_load(tmp_path, patched_manifest):
    version = "9.9.9"
    bundle = _make_root(tmp_path, version) / "unet" / version
    _write_synthetic_bundle(bundle, version=version, include_stats=False)
    with pytest.raises(FileNotFoundError):
        load_bundle(bundle)


def test_corrupted_checkpoint_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    (bundle / "checkpoint.pt").write_bytes(b"corrupted")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["checks"]["checkpoint.pt"]["match"] is False


def test_corrupted_config_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    (bundle / "config.json").write_text("{not json", encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]


def test_corrupted_norm_stats_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    (bundle / "norm_stats.json").write_text('{"bad": 1}', encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["checks"]["norm_stats.json"]["match"] is False


def test_wrong_model_version_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version="1.2.3")  # config version mismatch
    manifest = _manifest_for(bundle, version)
    patched_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["metadata"]["model_version"]["match"] is False


def test_wrong_sar_unit_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version, unit="linear", stats_unit="linear")
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["metadata"]["s1_unit"]["match"] is False


def test_wrong_band_order_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version, bands=("VH_dB", "VV_dB"))
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["metadata"]["band_order"]["match"] is False


def test_wrong_probability_threshold_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version, threshold=0.9)
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["metadata"]["probability_threshold"]["match"] is False


def test_norm_stats_unit_disagreement_detected(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version, unit="dB", stats_unit="linear")
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    report = verify_bundle(version, root)
    assert not report["ok"]
    assert report["metadata"]["norm_stats_s1_unit"]["match"] is False


def test_missing_default_bundle_raises(tmp_path, patched_manifest):
    root = tmp_path / "models"
    (root / "unet").mkdir(parents=True)
    (root / "unet" / "default.txt").write_text("9.9.9", encoding="utf-8")
    patched_manifest.write_text(json.dumps({"schema_version": 2, "default_version": "9.9.9", "bundles": {}}), encoding="utf-8")
    with pytest.raises(BundleVerificationError):
        resolve_model_dir(root)


def test_corrupt_default_bundle_does_not_silently_fallback(tmp_path, patched_manifest):
    version = "9.9.9"
    root = _make_root(tmp_path, version)
    bundle = root / "unet" / version
    _write_synthetic_bundle(bundle, version=version)
    (root / "unet" / "default.txt").write_text(version, encoding="utf-8")
    patched_manifest.write_text(json.dumps(_manifest_for(bundle, version)), encoding="utf-8")
    # Corrupt the default bundle after the manifest was recorded.
    (bundle / "checkpoint.pt").write_bytes(b"corrupt")
    with pytest.raises(BundleVerificationError):
        resolve_model_dir(root)


def test_register_bundle_refuses_incomplete(tmp_path, patched_manifest):
    src = tmp_path / "src"
    _write_synthetic_bundle(src, version="1.0.0", include_stats=False)
    with pytest.raises(FileNotFoundError):
        register_bundle(src, tmp_path / "models", version="1.0.0")


def test_register_bundle_updates_tracked_manifest(tmp_path, patched_manifest):
    src = tmp_path / "src"
    _write_synthetic_bundle(src, version="1.0.0")
    patched_manifest.write_text(
        json.dumps({"schema_version": 2, "default_version": "1.0.0", "bundles": {}}), encoding="utf-8"
    )
    dest = register_bundle(src, tmp_path / "models", version="1.0.0")
    assert (dest / "manifest.json").exists()
    tracked = json.loads(patched_manifest.read_text(encoding="utf-8"))
    entry = tracked["bundles"]["1.0.0"]
    assert entry["model_version"] == "1.0.0"
    assert entry["config_sha256"] == _sha256(dest / "config.json")
