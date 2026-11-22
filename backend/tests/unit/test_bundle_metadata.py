"""On-disk model bundles must declare a SAR unit in config and norm_stats.

Guards against a bundle being registered without the mandatory unit metadata
(which would make inference fail, or worse, silently mis-convert).
"""

from pathlib import Path

import pytest

from app.ml.features import normalize_unit
from app.ml.inference import load_bundle
from app.ml.model_registry import load_bundle_manifest, verify_bundle

import json

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models" / "unet"


def _bundle_dirs():
    if not MODEL_ROOT.is_dir():
        return []
    return [d for d in MODEL_ROOT.iterdir() if d.is_dir() and (d / "checkpoint.pt").exists()]


def test_tracked_manifest_is_wellformed():
    manifest = load_bundle_manifest()
    assert manifest["schema_version"] >= 1
    assert manifest["default_version"] in manifest["bundles"]
    for version, entry in manifest["bundles"].items():
        assert entry["s1_unit"] in ("dB", "linear"), version
        assert entry["checkpoint_sha256"]


@pytest.mark.skipif(not _bundle_dirs(), reason="No model bundles present.")
def test_tracked_manifest_checksums_match_disk():
    manifest = load_bundle_manifest()
    for bundle_dir in _bundle_dirs():
        version = bundle_dir.name
        assert version in manifest["bundles"], f"{version} missing from bundles_manifest.json"
        report = verify_bundle(version, MODEL_ROOT.parent)
        assert report["ok"], f"{version} bundle checksum verification failed: {report}"


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
