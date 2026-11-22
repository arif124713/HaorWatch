"""Model bundle registry: checksums, metadata contract, and fail-closed resolution.

The tracked manifest (``bundles_manifest.json``) is the single source of truth
for what a valid production bundle must contain. ``verify_bundle`` checks every
required artifact and every recorded metadata field against the manifest before
a bundle is selected or loaded. Verification is fail-closed: a corrupt or
incomplete configured default bundle is an error, never a silent fallback.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

# Files every loadable bundle must contain.
BUNDLE_FILES = ["checkpoint.pt", "config.json", "norm_stats.json"]

MANIFEST_PATH = Path(__file__).resolve().parent / "bundles_manifest.json"

# Metadata fields a manifest entry must declare (in addition to checksums).
REQUIRED_MANIFEST_FIELDS = (
    "model_version",
    "s1_unit",
    "band_order",
    "probability_threshold",
    "checkpoint_sha256",
    "config_sha256",
    "norm_stats_sha256",
)

# Metadata fields the on-disk artifacts must agree with the manifest on.
_ARTIFACT_CHECKSUM_FIELDS = (
    ("checkpoint.pt", "checkpoint_sha256"),
    ("config.json", "config_sha256"),
    ("norm_stats.json", "norm_stats_sha256"),
)

# Valid Sentinel-1 band identifiers for band_order validation.
_VALID_BAND_IDENTIFIERS = ("VV", "VH", "VVP", "VHP", "VV_DB", "VH_DB")
_HEX_DIGITS = set("0123456789abcdef")


def _is_sha256_hex(value) -> bool:
    """True only for a 64-character lowercase hex SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in _HEX_DIGITS for ch in value.lower())


def _finite_number(value, *, allow_string: bool = False) -> float | None:
    """Return the value as a finite float, or None when it is not numeric/finite.

    When ``allow_string`` is false (the default) a numeric string such as
    ``"0.4"`` is rejected: typed metadata must be declared with the correct JSON
    type, not a string that happens to parse.
    """
    import math

    if isinstance(value, bool):
        return None
    if isinstance(value, str) and not allow_string:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _valid_band_order(order) -> list[str] | None:
    """Return a normalised band_order list when it is valid, else None.

    A valid band order is a non-empty list (or comma string) of recognised SAR
    band identifiers. A dict, a number, an empty list, or an unknown identifier
    is invalid.
    """
    if isinstance(order, dict) or isinstance(order, (int, float, bool)):
        return None
    if isinstance(order, str):
        order = [order]
    if not isinstance(order, (list, tuple)) or len(order) == 0:
        return None
    normalized = []
    for item in order:
        if not isinstance(item, str) or not item.strip():
            return None
        normalized.append(item.strip())
    # Every identifier must be a recognised SAR band (case-insensitive).
    for item in normalized:
        if item.upper().replace("-", "_") not in _VALID_BAND_IDENTIFIERS:
            return None
    return normalized


def _validate_norm_stats(stats) -> tuple[bool, str]:
    """Validate norm-stats structure: object with finite numeric mean/std lists."""
    if not isinstance(stats, dict):
        return False, "norm_stats is not a JSON object"
    mean = stats.get("mean")
    std = stats.get("std")
    if not isinstance(mean, list) or not isinstance(std, list) or not mean or not std:
        return False, "norm_stats must have non-empty 'mean' and 'std' lists"
    if len(mean) != len(std):
        return False, "norm_stats 'mean' and 'std' must have equal length"
    for value in list(mean) + list(std):
        if _finite_number(value) is None:
            return False, f"norm_stats contains a non-finite or non-numeric value {value!r}"
    for value in std:
        if float(value) <= 0.0:
            return False, f"norm_stats 'std' must be positive, found {value!r}"
    return True, ""


class BundleVerificationError(RuntimeError):
    """Raised when a bundle fails manifest verification (fail-closed)."""

    def __init__(self, version: str, report: dict):
        self.version = version
        self.report = report
        super().__init__(f"model bundle {version!r} failed verification: {report.get('reason') or report}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle_manifest() -> dict:
    """Load the tracked reproducibility manifest of model bundles."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _normalize_band_order(order) -> list[str] | None:
    if order is None:
        return None
    if isinstance(order, str):
        return [order]
    return [str(item) for item in order]


def _units_agree(left, right) -> bool:
    if left is None or right is None:
        return False
    from app.ml.features import normalize_unit

    try:
        return normalize_unit(left) == normalize_unit(right)
    except Exception:  # noqa: BLE001 - unknown unit is a mismatch
        return False


def _validate_manifest_entry(expected: dict, version: str) -> list[str]:
    """Validate the manifest entry's metadata types; return issue descriptions."""
    issues: list[str] = []

    if not isinstance(expected.get("model_version"), str) or not expected["model_version"].strip():
        issues.append("model_version must be a non-empty string")
    elif expected["model_version"] != version:
        issues.append(f"model_version {expected['model_version']!r} != requested {version!r}")

    try:
        from app.ml.features import normalize_unit

        normalize_unit(expected.get("s1_unit"))
    except Exception:  # noqa: BLE001 - unknown/missing unit is a manifest issue
        issues.append(f"s1_unit {expected.get('s1_unit')!r} is not a supported unit")

    if _valid_band_order(expected.get("band_order")) is None:
        issues.append(f"band_order {expected.get('band_order')!r} is not a non-empty list of valid bands")

    threshold = _finite_number(expected.get("probability_threshold"))
    if threshold is None:
        issues.append(f"probability_threshold {expected.get('probability_threshold')!r} is not numeric/finite")
    elif not (0.0 <= threshold <= 1.0):
        issues.append(f"probability_threshold {threshold!r} is outside [0, 1]")

    for key in ("checkpoint_sha256", "config_sha256", "norm_stats_sha256"):
        if not _is_sha256_hex(expected.get(key)):
            issues.append(f"{key} is not a valid SHA-256 hex string")

    return issues


def verify_bundle(version: str, model_root: str | Path) -> dict:
    """Verify an on-disk bundle against the tracked manifest checksums and metadata.

    Returns a report dict with ``ok``, per-file checks, and per-metadata checks.
    A missing artifact, a checksum mismatch, or metadata that disagrees with the
    manifest is reported and flips ``ok`` to ``False``; nothing is silently
    ignored.
    """
    try:
        manifest = load_bundle_manifest()
    except json.JSONDecodeError as exc:
        return {"ok": False, "version": version, "reason": f"bundles_manifest.json is not valid JSON: {exc}"}
    if not isinstance(manifest, dict):
        return {"ok": False, "version": version, "reason": "bundles_manifest.json is not a JSON object"}
    bundles = manifest.get("bundles")
    if not isinstance(bundles, dict):
        return {"ok": False, "version": version, "reason": "bundles_manifest.json has no 'bundles' object"}
    expected = bundles.get(version)
    if not isinstance(expected, dict):
        return {"ok": False, "version": version, "reason": f"version {version!r} not in bundles_manifest.json"}

    missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if field not in expected]
    if missing_fields:
        return {
            "ok": False,
            "version": version,
            "reason": f"manifest entry for {version!r} is missing required fields: {missing_fields}",
        }

    # Validate the manifest's own metadata types before comparing anything, so a
    # malformed manifest is a structured failure rather than a raw exception.
    manifest_issues = _validate_manifest_entry(expected, version)
    if manifest_issues:
        return {
            "ok": False,
            "version": version,
            "reason": f"manifest entry for {version!r} is malformed: {manifest_issues}",
            "metadata": {"manifest_issues": manifest_issues},
        }

    bundle_dir = Path(model_root) / "unet" / version
    report: dict = {"ok": True, "version": version, "checks": {}, "metadata": {}}

    # 1. Required files present + checksums match.
    for name, key in _ARTIFACT_CHECKSUM_FIELDS:
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

    # 2. Cross-check on-disk artifact metadata with the manifest contract.
    # JSON parse errors are expected validation failures; unexpected exceptions
    # (e.g. an out-of-memory or programming error) are not swallowed.
    config_path = bundle_dir / "config.json"
    stats_path = bundle_dir / "norm_stats.json"
    config = stats = None
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["metadata"]["config_json"] = "unparseable"
            report["ok"] = False
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["metadata"]["norm_stats_json"] = "unparseable"
            report["ok"] = False

    if config is not None and not isinstance(config, dict):
        report["metadata"]["config_json"] = "not a JSON object"
        report["ok"] = False
        config = None

    if config is not None:
        # Model version.
        cfg_version = config.get("model_version")
        version_ok = isinstance(cfg_version, str) and cfg_version == expected["model_version"] == version
        report["metadata"]["model_version"] = {
            "config": cfg_version,
            "manifest": expected["model_version"],
            "requested": version,
            "match": version_ok,
        }
        if not version_ok:
            report["ok"] = False

        # SAR unit (config vs manifest).
        unit_ok = _units_agree(config.get("s1_unit"), expected["s1_unit"])
        report["metadata"]["s1_unit"] = {
            "config": config.get("s1_unit"),
            "manifest": expected["s1_unit"],
            "match": unit_ok,
        }
        if not unit_ok:
            report["ok"] = False

        # Band order (config vs manifest), order-sensitive and type-validated.
        cfg_bands = _valid_band_order(config.get("band_order"))
        band_ok = cfg_bands is not None and cfg_bands == _normalize_band_order(expected["band_order"])
        report["metadata"]["band_order"] = {
            "config": cfg_bands if cfg_bands is not None else config.get("band_order"),
            "manifest": expected["band_order"],
            "match": band_ok,
        }
        if not band_ok:
            report["ok"] = False

        # Probability threshold (config vs manifest): must be numeric + in range.
        cfg_threshold_number = _finite_number(config.get("probability_threshold"))
        threshold_ok = (
            cfg_threshold_number is not None
            and 0.0 <= cfg_threshold_number <= 1.0
            and abs(cfg_threshold_number - float(expected["probability_threshold"])) < 1e-9
        )
        report["metadata"]["probability_threshold"] = {
            "config": config.get("probability_threshold"),
            "manifest": expected["probability_threshold"],
            "match": threshold_ok,
        }
        if not threshold_ok:
            report["ok"] = False

    if stats is not None and not isinstance(stats, dict):
        report["metadata"]["norm_stats_json"] = "not a JSON object"
        report["ok"] = False
        stats = None

    if stats is not None:
        # Norm-stats structure must be valid (finite mean/std, positive std).
        stats_ok, stats_issue = _validate_norm_stats(stats)
        report["metadata"]["norm_stats_structure"] = {"match": stats_ok, "issue": stats_issue}
        if not stats_ok:
            report["ok"] = False

        # Norm stats unit vs manifest unit.
        stats_unit_ok = _units_agree(stats.get("s1_unit"), expected["s1_unit"])
        report["metadata"]["norm_stats_s1_unit"] = {
            "norm_stats": stats.get("s1_unit"),
            "manifest": expected["s1_unit"],
            "match": stats_unit_ok,
        }
        if not stats_unit_ok:
            report["ok"] = False

        # Norm stats unit must also agree with the config unit when both exist.
        if config is not None:
            config_stats_ok = _units_agree(stats.get("s1_unit"), config.get("s1_unit"))
            report["metadata"]["config_vs_stats_unit"] = {
                "config": config.get("s1_unit"),
                "norm_stats": stats.get("s1_unit"),
                "match": config_stats_ok,
            }
            if not config_stats_ok:
                report["ok"] = False

    return report


def verify_or_raise(version: str, model_root: str | Path) -> dict:
    """Verify a bundle and raise :class:`BundleVerificationError` on failure."""
    report = verify_bundle(version, model_root)
    if not report["ok"]:
        raise BundleVerificationError(version, report)
    return report


def register_bundle(src_dir: str | Path, model_root: str | Path, version: str | None = None) -> Path:
    """Register a complete bundle and update the tracked manifest.

    Refuses to register an incomplete bundle: all of :data:`BUNDLE_FILES` must
    exist in ``src_dir``. The generated ``manifest.json`` is written next to the
    copied artifacts and records checksums plus the metadata contract. Existing
    manifest entries for other versions are preserved.
    """
    src = Path(src_dir)
    missing = [name for name in BUNDLE_FILES if not (src / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot register an incomplete bundle from {src}: missing {missing}. "
            f"A valid bundle requires {BUNDLE_FILES}."
        )

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    norm_stats = json.loads((src / "norm_stats.json").read_text(encoding="utf-8"))

    version = version or config.get("model_version")
    if not version:
        raise ValueError(
            "cannot register bundle: no version given and config.json has no 'model_version'."
        )

    # Contract checks before writing anything.
    if config.get("model_version") is not None and config["model_version"] != version:
        raise ValueError(
            f"config.json model_version {config['model_version']!r} does not match "
            f"the registered version {version!r}."
        )
    if not _units_agree(config.get("s1_unit"), norm_stats.get("s1_unit")):
        raise ValueError(
            f"config.json s1_unit={config.get('s1_unit')!r} and norm_stats.json "
            f"s1_unit={norm_stats.get('s1_unit')!r} must both be present and agree."
        )
    if not config.get("band_order"):
        raise ValueError("config.json must declare 'band_order' to register a bundle.")
    if config.get("probability_threshold") is None:
        raise ValueError("config.json must declare 'probability_threshold' to register a bundle.")

    dest = Path(model_root) / "unet" / version
    dest.mkdir(parents=True, exist_ok=True)
    for name in BUNDLE_FILES:
        shutil.copy2(src / name, dest / name)
    for extra in ("metrics_eval.json", "dataset_audit.json", "split_manifest.json"):
        extra_file = src / extra
        if extra_file.exists():
            shutil.copy2(extra_file, dest / extra)

    bundle_manifest = {
        "model_name": config.get("model_name", "HaorWatch Flood Classifier"),
        "model_family": config.get("model_family", "unet"),
        "model_version": version,
        "s1_unit": config.get("s1_unit"),
        "band_order": config.get("band_order"),
        "probability_threshold": config.get("probability_threshold"),
        "checkpoint_sha256": _sha256(dest / "checkpoint.pt"),
        "config_sha256": _sha256(dest / "config.json"),
        "norm_stats_sha256": _sha256(dest / "norm_stats.json"),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(src),
        "config": config,
    }
    (dest / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")

    # Update the tracked top-level manifest (preserving other versions).
    _update_tracked_manifest(version, bundle_manifest, config)
    return dest


def _update_tracked_manifest(version: str, bundle_manifest: dict, config: dict) -> None:
    tracked = load_bundle_manifest()
    bundles = tracked.setdefault("bundles", {})
    bundles[version] = {
        "base_model": config.get("base_model", bundles.get(version, {}).get("base_model")),
        "model_version": version,
        "dataset": config.get("dataset", "unspecified"),
        "s1_unit": bundle_manifest["s1_unit"],
        "band_order": bundle_manifest["band_order"],
        "probability_threshold": bundle_manifest["probability_threshold"],
        "checkpoint_sha256": bundle_manifest["checkpoint_sha256"],
        "config_sha256": bundle_manifest["config_sha256"],
        "norm_stats_sha256": bundle_manifest["norm_stats_sha256"],
    }
    MANIFEST_PATH.write_text(json.dumps(tracked, indent=2) + "\n", encoding="utf-8")


def resolve_model_dir(model_root: str | Path, *, verify: bool = True) -> Path | None:
    """Resolve the active bundle directory.

    When ``default.txt`` names a version, that bundle is authoritative: if it is
    missing or (with ``verify=True``) fails manifest verification, this raises
    instead of silently falling back to another checkpoint. Fallback to the
    newest registered bundle happens only when no default pointer exists.
    """
    root = Path(model_root) / "unet"
    if not root.exists():
        return None

    default_file = root / "default.txt"
    if default_file.exists():
        default_version = default_file.read_text(encoding="utf-8").strip()
        if not default_version:
            raise BundleVerificationError("(default)", {"reason": "default.txt is empty"})
        default_dir = root / default_version
        if not (default_dir / "checkpoint.pt").exists():
            raise BundleVerificationError(
                default_version,
                {"reason": f"configured default bundle {default_version!r} has no checkpoint.pt at {default_dir}"},
            )
        if verify:
            verify_or_raise(default_version, model_root)
        return default_dir

    # No explicit default: choose the newest registered, complete, verified bundle.
    candidates = []
    for version_dir in root.iterdir():
        if not version_dir.is_dir() or not (version_dir / "checkpoint.pt").exists():
            continue
        if verify and not verify_bundle(version_dir.name, model_root)["ok"]:
            continue
        manifest_path = version_dir / "manifest.json"
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
