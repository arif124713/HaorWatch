"""Cache-validity contract for built S1/S2 pseudo-label dataset pairs.

A cached pair may be reused only when it satisfies the full validity contract,
not merely when the two "obvious" rasters exist. This guards against reusing a
pair generated before the validity-mask fix: such a stale pair may lack
``valid_mask.tif`` or carry inconsistent grid/nodata metadata, and must be
rebuilt rather than silently trusted.

The functions here are dependency-light (JSON + a small amount of rasterio) so
they can be unit-tested with synthetic fixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Builder code version recorded in the per-pair metadata. Bump when the pair
# generation semantics change so stale pairs are detected and rebuilt.
BUILDER_VERSION = "0.4.1-validity"

# Artifacts that a reusable pair must contain.
REQUIRED_PAIR_ARTIFACTS = ("s1_grid.tif", "pseudo_label.tif", "valid_mask.tif", "s1_valid_mask.tif")
# Core artifacts whose absence makes the pair definitely stale.
CRITICAL_PAIR_ARTIFACTS = ("s1_grid.tif", "pseudo_label.tif", "valid_mask.tif")


@dataclass
class CacheDecision:
    reusable: bool
    stale_reasons: list[str] = field(default_factory=list)


def _read_raster_meta(path: Path) -> dict | None:
    try:
        import rasterio
    except Exception:  # noqa: BLE001
        return None
    try:
        with rasterio.open(path) as src:
            return {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": list(src.transform)[:6],
                "shape": [int(src.height), int(src.width)],
                "res": [abs(float(src.transform.a)), abs(float(src.transform.e))],
                "nodata": src.nodata,
                "count": int(src.count),
            }
    except Exception:  # noqa: BLE001
        return None


def _close(a, b, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def evaluate_pair_cache(
    pair_dir: Path,
    *,
    expected_grid: dict,
    expected_sar_unit: str,
    expected_ignore_value: float,
    expected_s1_day: str | None = None,
    expected_s2_day: str | None = None,
    record: dict | None = None,
    builder_version: str = BUILDER_VERSION,
) -> CacheDecision:
    """Decide whether a cached pair may be reused.

    ``expected_grid`` must contain ``crs``, ``transform`` (6-tuple), ``shape``
    (h, w). ``record`` is the per-pair metadata entry previously written to the
    dataset manifest (or ``None`` if absent).
    """
    reasons: list[str] = []

    missing = [name for name in CRITICAL_PAIR_ARTIFACTS if not (pair_dir / name).exists()]
    if missing:
        reasons.append(f"missing required artifacts: {missing}")
    # valid_mask.tif is central to the validity fix: absent => definitely stale.
    if not (pair_dir / "valid_mask.tif").exists():
        reasons.append("valid_mask.tif absent (pre-validity-fix pair)")

    if record is None:
        reasons.append("no per-pair manifest record")
        return CacheDecision(reusable=False, stale_reasons=reasons)

    if record.get("builder_version") != builder_version:
        reasons.append(
            f"builder_version {record.get('builder_version')!r} != {builder_version!r}"
        )

    if expected_sar_unit and record.get("sar_unit") != expected_sar_unit:
        reasons.append(f"sar_unit {record.get('sar_unit')!r} != {expected_sar_unit!r}")

    if record.get("label_ignore_value") is None or not _close(
        record.get("label_ignore_value"), expected_ignore_value
    ):
        reasons.append(
            f"label_ignore_value {record.get('label_ignore_value')!r} != {expected_ignore_value!r}"
        )

    if expected_s1_day and record.get("s1_day") != expected_s1_day:
        reasons.append(f"s1_day {record.get('s1_day')!r} != {expected_s1_day!r}")
    if expected_s2_day and record.get("s2_day") != expected_s2_day:
        reasons.append(f"s2_day {record.get('s2_day')!r} != {expected_s2_day!r}")

    grid = record.get("grid") or {}
    if expected_grid.get("crs") is not None and grid.get("crs") != expected_grid.get("crs"):
        reasons.append(f"grid crs {grid.get('crs')!r} != {expected_grid.get('crs')!r}")
    if expected_grid.get("shape") is not None and list(grid.get("shape") or []) != list(
        expected_grid.get("shape")
    ):
        reasons.append(f"grid shape {grid.get('shape')!r} != {expected_grid.get('shape')!r}")
    exp_transform = expected_grid.get("transform")
    got_transform = grid.get("transform")
    if exp_transform is not None:
        if got_transform is None or len(got_transform) != len(exp_transform) or not all(
            _close(a, b, tol=1e-4) for a, b in zip(got_transform, exp_transform)
        ):
            reasons.append(f"grid transform {got_transform!r} != {exp_transform!r}")

    # valid pixel count must be present and consistent with a real mask.
    if record.get("valid_px") is None:
        reasons.append("valid_px not recorded")

    # Verify on-disk raster metadata agrees with the recorded grid where possible.
    for name in CRITICAL_PAIR_ARTIFACTS:
        path = pair_dir / name
        if not path.exists():
            continue
        meta = _read_raster_meta(path)
        if meta is None:
            reasons.append(f"{name} unreadable")
            continue
        if meta["crs"] and expected_grid.get("crs") and meta["crs"] != expected_grid["crs"]:
            reasons.append(f"{name} crs {meta['crs']!r} != {expected_grid['crs']!r}")
        if expected_grid.get("shape") and meta["shape"] != list(expected_grid["shape"]):
            reasons.append(f"{name} shape {meta['shape']!r} != {expected_grid['shape']!r}")

    return CacheDecision(reusable=not reasons, stale_reasons=reasons)


def build_pair_record(
    *,
    pair_dir: Path,
    builder_version: str = BUILDER_VERSION,
    sar_unit: str,
    s1_day: str,
    s2_day: str,
    s1_meta: dict | None,
    aoi_id: str | None,
    grid: dict,
    label_ignore_value: float,
    nodata: dict | None,
    valid_px: int,
    invalid_px: int,
    water_px: int,
    code_commit: str | None = None,
    source_hashes: dict | None = None,
) -> dict:
    """Construct the per-pair metadata record written to the dataset manifest."""
    return {
        "pair_dir": str(pair_dir),
        "builder_version": builder_version,
        "code_commit": code_commit,
        "sar_unit": sar_unit,
        "s1_day": s1_day,
        "s2_day": s2_day,
        "s1_meta": s1_meta,
        "aoi_id": aoi_id,
        "grid": grid,
        "label_ignore_value": label_ignore_value,
        "nodata": nodata or {},
        "valid_px": int(valid_px),
        "invalid_px": int(invalid_px),
        "water_px": int(water_px),
        "source_hashes": source_hashes or {},
    }


def load_dataset_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def pair_record_for(manifest: dict, pair_dir: Path) -> dict | None:
    """Find a per-pair record by pair directory (string or resolved match)."""
    target = str(pair_dir)
    for entry in manifest.get("pairs", []):
        if entry.get("pair_dir") == target:
            return entry
        try:
            if Path(entry.get("pair_dir", "")).resolve() == pair_dir.resolve():
                return entry
        except OSError:
            continue
    return None
