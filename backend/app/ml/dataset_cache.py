"""Cache-validity contract for built S1/S2 pseudo-label dataset pairs.

A cached pair may be reused only when it satisfies the full validity contract,
not merely when the "obvious" rasters exist. This guards against reusing a pair
generated before the validity-mask fix: such a stale pair may lack
``valid_mask.tif`` or ``s1_valid_mask.tif``, or carry inconsistent
grid/nodata/self-consistency metadata, and must be rebuilt rather than silently
trusted.

The functions here are dependency-light (JSON + a small amount of rasterio) so
they can be unit-tested with synthetic fixtures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Builder code version recorded in the per-pair metadata. Bump when the pair
# generation semantics change so stale pairs are detected and rebuilt.
BUILDER_VERSION = "0.4.1-validity"

# Artifacts a reusable pair MUST contain. All four are mandatory: the reuse
# contract checks the full required set (there is no "critical subset" that can
# be satisfied while a required artifact is absent).
REQUIRED_PAIR_ARTIFACTS = (
    "s1_grid.tif",
    "pseudo_label.tif",
    "valid_mask.tif",
    "s1_valid_mask.tif",
)
# Backwards-compatible alias. The earlier design distinguished a "critical"
# subset from the full set, which allowed a pair missing ``s1_valid_mask.tif`` to
# be accepted. The alias now equals the full required set so no caller can
# accidentally validate only a subset.
CRITICAL_PAIR_ARTIFACTS = REQUIRED_PAIR_ARTIFACTS

# Binary masks whose on-disk values must be exactly {0, 1}.
_BINARY_MASK_ARTIFACTS = ("valid_mask.tif", "s1_valid_mask.tif")
# Rasters that must preserve the configured ignore sentinel for invalid pixels.
_LABEL_ARTIFACT = "pseudo_label.tif"
# SAR grid whose invalid pixels must stay non-finite / sentinel.
_SAR_GRID_ARTIFACT = "s1_grid.tif"

_BINARY_TOL = 1e-6
_RES_TOL = 1e-3


@dataclass
class CacheDecision:
    reusable: bool
    stale_reasons: list[str] = field(default_factory=list)


def _read_raster_meta(path: Path) -> dict | None:
    """Read CRS/transform/shape/resolution/nodata. Return None if unreadable."""
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(path) as src:
            return {
                "crs": src.crs.to_string() if src.crs else None,
                "transform": [float(v) for v in list(src.transform)[:6]],
                "shape": [int(src.height), int(src.width)],
                "res": [abs(float(src.transform.a)), abs(float(src.transform.e))],
                "nodata": src.nodata,
                "count": int(src.count),
            }
    except (rasterio.errors.RasterioIOError, OSError, ValueError):
        # A corrupted/unreadable TIFF is a validation failure, not a crash.
        return None


def _read_raster_values(path: Path, max_pixels: int = 4_000_000):
    """Read band 1 as float64 for value-level checks. None if unreadable/too big."""
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(path) as src:
            if src.width * src.height > max_pixels:
                # Too large to fully read: sample a decimated window instead.
                indices = src.read(1, out_shape=(min(src.height, 512), min(src.width, 512)))
                return indices.astype("float64")
            return src.read(1).astype("float64")
    except (rasterio.errors.RasterioIOError, OSError, ValueError):
        return None


def _close(a, b, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _transform_close(got, expected, tol: float = 1e-4) -> bool:
    if got is None or expected is None:
        return False
    return len(got) == len(expected) and all(_close(a, b, tol=tol) for a, b in zip(got, expected))


def _validate_raster_against_grid(name: str, meta: dict, expected_grid: dict, reasons: list[str]) -> None:
    """Check a raster's CRS/transform/shape/resolution against the expected grid."""
    exp_crs = expected_grid.get("crs")
    exp_transform = expected_grid.get("transform")
    exp_shape = expected_grid.get("shape")
    if exp_crs is not None and meta.get("crs") and meta["crs"] != exp_crs:
        reasons.append(f"{name} crs {meta['crs']!r} != {exp_crs!r}")
    if exp_transform is not None and not _transform_close(meta.get("transform"), exp_transform):
        reasons.append(f"{name} transform {meta.get('transform')!r} != {exp_transform!r}")
    if exp_shape is not None and meta.get("shape") != list(exp_shape):
        reasons.append(f"{name} shape {meta['shape']!r} != {list(exp_shape)!r}")
    # Resolution is derived from the transform; validate it when both sides have it.
    exp_res = expected_grid.get("res")
    if exp_res is not None and meta.get("res") is not None:
        if not all(_close(a, b, tol=_RES_TOL) for a, b in zip(meta["res"], exp_res)):
            reasons.append(f"{name} resolution {meta['res']!r} != {list(exp_res)!r}")


def _validate_nodata_semantics(
    pair_dir: Path,
    *,
    expected_ignore_value: float,
    expected_grid: dict,
    reasons: list[str],
) -> None:
    """Validate nodata/binary semantics of the four artifacts.

    * Binary masks must contain only {0, 1} (plus their declared nodata).
    * The pseudo-label must use the configured ignore value for invalid pixels
      and must not turn them into silent non-water zeros.
    * The SAR grid must keep invalid pixels non-finite or at its declared
      nodata sentinel.
    """
    # Binary validity masks: values must be binary.
    for name in _BINARY_MASK_ARTIFACTS:
        path = pair_dir / name
        values = _read_raster_values(path)
        meta = _read_raster_meta(path)
        if values is None or meta is None:
            continue  # unreadable already reported elsewhere
        nodata = meta.get("nodata")
        finite = values[~_is_nodata(values, nodata)]
        unique = set()
        for value in finite.ravel():
            unique.add(round(float(value), 6))
            if len(unique) > 8:
                break
        bad = [v for v in unique if not (_close(v, 0.0, _BINARY_TOL) or _close(v, 1.0, _BINARY_TOL))]
        if bad:
            reasons.append(f"{name} has non-binary mask values: {sorted(bad)[:5]}")

    # Pseudo-label: invalid pixels must be the ignore value, not a valid class.
    label_meta = _read_raster_meta(pair_dir / _LABEL_ARTIFACT)
    label_values = _read_raster_values(pair_dir / _LABEL_ARTIFACT)
    if label_meta is not None and label_values is not None:
        declared = label_meta.get("nodata")
        if declared is None or not _close(declared, expected_ignore_value):
            reasons.append(
                f"{_LABEL_ARTIFACT} nodata {declared!r} != expected ignore value {expected_ignore_value!r}"
            )
        # Values must lie in {ignore, 0, 1}; anything else is an uninterpretable label.
        finite = label_values[np.isfinite(label_values)]
        allowed = {round(float(expected_ignore_value), 6), 0.0, 1.0}
        for value in finite.ravel():
            if round(float(value), 6) not in allowed:
                reasons.append(f"{_LABEL_ARTIFACT} contains an unexpected label value {value!r}")
                break

    # SAR grid: invalid pixels must be preserved (non-finite) rather than a
    # finite value that could later be mistaken for a real observation.
    sar_meta = _read_raster_meta(pair_dir / _SAR_GRID_ARTIFACT)
    sar_values = _read_raster_values(pair_dir / _SAR_GRID_ARTIFACT)
    if sar_meta is not None and sar_values is not None:
        nodata = sar_meta.get("nodata")
        if nodata is None:
            reasons.append(f"{_SAR_GRID_ARTIFACT} does not declare nodata; invalid SAR cannot be preserved")
        elif np.isfinite(nodata):
            # When nodata is finite, declared nodata pixels must actually take
            # that value (not a plausible backscatter magnitude).
            declared_count = int((np.abs(sar_values - float(nodata)) <= _BINARY_TOL).sum())
            if declared_count == 0 and (sar_values.size and not bool(np.isnan(sar_values).any())):
                reasons.append(
                    f"{_SAR_GRID_ARTIFACT} declares a finite nodata {nodata!r} but carries none"
                )


def _is_nodata(values, nodata):
    if nodata is None:
        return ~np.isfinite(values)
    return (~np.isfinite(values)) | (np.abs(values - float(nodata)) <= _BINARY_TOL)


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
    (h, w) and may contain ``res`` (x_res, y_res). ``record`` is the per-pair
    metadata entry previously written to the dataset manifest (or ``None``).

    A pair is reusable only when every required artifact is present and readable,
    each raster matches the expected grid (CRS/transform/shape/resolution), the
    nodata/binary semantics hold, and the manifest record is present and
    consistent with the expected SAR unit, ignore value, builder version, source
    dates, grid, and valid-pixel count.
    """
    pair_dir = Path(pair_dir)
    reasons: list[str] = []

    # 1. Every required artifact must be present.
    missing = [name for name in REQUIRED_PAIR_ARTIFACTS if not (pair_dir / name).exists()]
    if missing:
        reasons.append(f"missing required artifacts: {missing}")

    # 2. Manifest record must exist; without it the pair cannot be trusted.
    if record is None:
        reasons.append("no per-pair manifest record")
        return CacheDecision(reusable=False, stale_reasons=reasons)

    # 3. Metadata contract: version, unit, ignore value, dates.
    if record.get("builder_version") != builder_version:
        reasons.append(f"builder_version {record.get('builder_version')!r} != {builder_version!r}")

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

    # 4. Recorded grid must match the expected grid.
    grid = record.get("grid") or {}
    if expected_grid.get("crs") is not None and grid.get("crs") != expected_grid.get("crs"):
        reasons.append(f"recorded grid crs {grid.get('crs')!r} != {expected_grid.get('crs')!r}")
    if expected_grid.get("shape") is not None and list(grid.get("shape") or []) != list(
        expected_grid.get("shape")
    ):
        reasons.append(f"recorded grid shape {grid.get('shape')!r} != {expected_grid.get('shape')!r}")
    exp_transform = expected_grid.get("transform")
    if exp_transform is not None and not _transform_close(grid.get("transform"), exp_transform):
        reasons.append(f"recorded grid transform {grid.get('transform')!r} != {exp_transform!r}")

    if record.get("valid_px") is None:
        reasons.append("valid_px not recorded")

    # 5. Every required raster must be readable and match the expected grid.
    raster_meta: dict[str, dict] = {}
    for name in REQUIRED_PAIR_ARTIFACTS:
        path = pair_dir / name
        if not path.exists():
            continue
        meta = _read_raster_meta(path)
        if meta is None:
            reasons.append(f"{name} unreadable or corrupted")
            continue
        raster_meta[name] = meta
        _validate_raster_against_grid(name, meta, expected_grid, reasons)

    # 6. Grid consistency between the four rasters.
    shapes = {name: tuple(meta["shape"]) for name, meta in raster_meta.items()}
    if len(set(shapes.values())) > 1:
        reasons.append(f"rasters disagree on shape: {shapes}")

    # 7. Nodata / binary semantics.
    if not missing:
        _validate_nodata_semantics(
            pair_dir,
            expected_ignore_value=expected_ignore_value,
            expected_grid=expected_grid,
            reasons=reasons,
        )

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
    effective_config: dict | None = None,
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
        "effective_config": effective_config or {},
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
