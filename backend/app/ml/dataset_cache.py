"""Cache-validity contract for built S1/S2 pseudo-label dataset pairs.

A cached pair may be reused only when it satisfies the full validity contract,
not merely when the "obvious" rasters exist. This guards against reusing a pair
generated before the validity-mask fix: such a stale pair may lack
``valid_mask.tif`` or ``s1_valid_mask.tif``, or carry inconsistent
grid/nodata/self-consistency metadata, and must be rebuilt rather than silently
trusted.

Builder semantics encoded here (do not invent new meanings):

* ``s1_valid_mask.tif`` / ``valid_mask.tif`` — single-band float32 masks written
  with ``nodata=0.0``; their only permitted pixel values are the binary classes
  ``{0, 1}``. ``0`` is the ``nodata`` metadata value *and* a legitimate mask
  value, so it is never treated as an invalid pixel; nodata metadata never
  authorizes any non-binary pixel (NaN/Inf or a finite sentinel such as
  ``-9999``).
* ``s1_valid_mask.tif`` also records valid SAR observations: AOI ∧ finite/
  sentinel-free SAR over **all** bands (the builder's ``valid`` mask).
  ``s1_grid.tif`` is a two-band (VV, VH) float32 SAR grid written with a single
  shared ``nodata=NaN``; the builder sets **every** band to NaN wherever SAR is
  invalid. The validator checks each SAR band **independently** against
  ``s1_valid_mask.tif``: a clean VV band must not excuse invalid VH data. At a
  pixel marked SAR-valid, a value is rejected if it is non-finite **or** equal to
  the declared finite nodata sentinel (a sentinel is a validity contradiction,
  not a measurement). At a pixel marked SAR-invalid, the value must be
  non-finite or equal to the declared nodata sentinel; a plausible finite
  backscatter value is rejected.
* ``valid_mask.tif`` — valid supervision region: AOI ∧ valid SAR ∧ valid
  Sentinel-2 reference (the builder's ``pair_valid``). This is a subset of
  ``s1_valid_mask.tif``. The manifest ``valid_px`` field is defined as the count
  of pixels where ``valid_mask.tif`` is 1.
* ``pseudo_label.tif`` — the label with the configured ignore value wherever
  ``valid_mask.tif`` is 0, and only ``{0, 1}`` where ``valid_mask.tif`` is 1.

The functions here are dependency-light (JSON + rasterio) so they can be
unit-tested with synthetic fixtures. Large rasters are validated with bounded
windowed reads; no lossy decimation is used as proof of validity.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Builder code version recorded in the per-pair metadata. Bump when the pair
# generation semantics change so stale pairs are detected and rebuilt.
BUILDER_VERSION = "0.4.1-validity"

# Artifacts a reusable pair MUST contain. All four are mandatory.
REQUIRED_PAIR_ARTIFACTS = (
    "s1_grid.tif",
    "pseudo_label.tif",
    "valid_mask.tif",
    "s1_valid_mask.tif",
)
# Backwards-compatible alias; equals the full required set by design.
CRITICAL_PAIR_ARTIFACTS = REQUIRED_PAIR_ARTIFACTS

# Artifact roles.
_SAR_GRID_ARTIFACT = "s1_grid.tif"
_LABEL_ARTIFACT = "pseudo_label.tif"
_VALID_MASK_ARTIFACT = "valid_mask.tif"       # supervision validity (defines valid_px)
_S1_VALID_MASK_ARTIFACT = "s1_valid_mask.tif"  # valid SAR observations
_BINARY_MASK_ARTIFACTS = (_VALID_MASK_ARTIFACT, _S1_VALID_MASK_ARTIFACT)

# Expected single-band count for the artefacts above (s1_grid has 2 bands).
_EXPECTED_BAND_COUNT = {
    _SAR_GRID_ARTIFACT: 2,
    _LABEL_ARTIFACT: 1,
    _VALID_MASK_ARTIFACT: 1,
    _S1_VALID_MASK_ARTIFACT: 1,
}

_BINARY_TOL = 1e-6
_RES_TOL = 1e-3
# Bounded windowed-read size (pixels per block). Keeps memory constant
# regardless of raster size while still validating every pixel.
_WINDOW_PIXELS = 1_048_576


@dataclass
class CacheDecision:
    reusable: bool
    stale_reasons: list[str] = field(default_factory=list)


class _RasterUnreadable(Exception):
    """Internal marker: a raster could not be opened/read (a stale reason)."""


def _open_raster(path: Path):
    """Open a raster, raising ``_RasterUnreadable`` on any read/open failure."""
    import rasterio

    try:
        return rasterio.open(path)
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise _RasterUnreadable(f"{path.name}: {type(exc).__name__}") from exc


def _read_raster_meta(path: Path) -> dict | None:
    """Read CRS/transform/shape/resolution/band-count/nodata. None if unreadable."""
    try:
        src = _open_raster(path)
    except _RasterUnreadable:
        return None
    try:
        with src:
            crs = src.crs
            transform = src.transform
            # Resolution from the transform's pixel size; always two elements.
            res = (abs(float(transform.a)), abs(float(transform.e)))
            return {
                "crs": crs.to_string() if crs else None,
                "transform": [float(v) for v in list(transform)[:6]],
                "shape": [int(src.height), int(src.width)],
                "res": [float(res[0]), float(res[1])],
                "nodata": src.nodata,
                "count": int(src.count),
            }
    except (rasterio.errors.RasterioIOError, OSError, ValueError):
        return None


def _iter_windows(path: Path):
    """Yield (window, np.ndarray band-1 float64) blocks covering the whole raster.

    Reads every pixel exactly once with bounded memory. Raises
    ``_RasterUnreadable`` if the raster cannot be opened or any block read fails.
    """
    import rasterio
    from rasterio.windows import Window

    src = _open_raster(path)
    with src:
        width, height = src.width, src.height
        if width <= 0 or height <= 0:
            raise _RasterUnreadable(f"{path.name}: non-positive dimensions")
        block = max(1, int(math.isqrt(_WINDOW_PIXELS)))
        try:
            # Prefer the raster's own block layout when available.
            for row_off in range(0, height, block):
                rows = min(block, height - row_off)
                for col_off in range(0, width, block):
                    cols = min(block, width - col_off)
                    window = Window(col_off, row_off, cols, rows)
                    values = src.read(1, window=window).astype("float64")
                    yield window, values
        except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
            raise _RasterUnreadable(f"{path.name}: {type(exc).__name__}") from exc


def _close(a, b, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _crs_equal(got, expected, tol: float = 1e-9) -> bool:
    """Semantic CRS comparison using pyproj; falls back to string equality."""
    if got is None or expected is None:
        return False
    try:
        from pyproj import CRS

        return CRS.from_user_input(got).equals(CRS.from_user_input(expected), ignore_axis_order=True)
    except Exception:  # noqa: BLE001 - fall back to normalized string compare
        return str(got).strip().upper() == str(expected).strip().upper()


def _transform_close(got, expected, tol: float = 1e-4) -> bool:
    if got is None or expected is None or len(got) != 6 or len(expected) != 6:
        return False
    return all(_close(a, b, tol=tol) for a, b in zip(got, expected))


def _res_close(got, expected, tol: float = _RES_TOL) -> bool:
    if not isinstance(got, (list, tuple)) or not isinstance(expected, (list, tuple)):
        return False
    if len(got) != 2 or len(expected) != 2:
        return False
    return all(_close(a, b, tol=tol) for a, b in zip(got, expected))


def _is_positive_int(value) -> bool:
    """True only for a genuine positive integer (not a bool, not fractional)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer() and value > 0
    return False


def _finite_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_expected_grid(expected_grid: dict, reasons: list[str]) -> bool:
    """Validate that the expected grid carries usable, well-formed metadata.

    Returns True when the expected grid is usable. Missing/malformed expected
    fields are a cache failure (we cannot prove safety), not a silent skip.

    Checks: shape is exactly two positive integers (bools and fractional values
    are rejected, not truncated); transform has six finite coefficients; the
    resolution (when present) has two finite, strictly positive values; and the
    CRS is present and parseable.
    """
    if not isinstance(expected_grid, dict):
        reasons.append(f"expected grid is not an object: {type(expected_grid).__name__}")
        return False

    ok = True

    # CRS: present and parseable.
    crs = expected_grid.get("crs")
    if not crs or not isinstance(crs, str):
        reasons.append("expected grid is missing a usable 'crs'")
        ok = False
    else:
        try:
            from pyproj import CRS

            CRS.from_user_input(crs)
        except Exception:  # noqa: BLE001 - any parse failure is a grid failure
            reasons.append(f"expected grid 'crs' is not parseable: {crs!r}")
            ok = False

    # Shape: two positive integers.
    shape = expected_grid.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        reasons.append("expected grid shape must be a 2-element sequence")
        ok = False
    elif not all(_is_positive_int(dim) for dim in shape):
        reasons.append(f"expected grid shape dimensions must be positive integers: {list(shape)!r}")
        ok = False

    # Transform: six finite coefficients.
    transform = expected_grid.get("transform")
    if not isinstance(transform, (list, tuple)) or len(transform) != 6:
        reasons.append("expected grid transform must be a 6-element sequence")
        ok = False
    elif not all(_finite_float(coef) is not None for coef in transform):
        reasons.append(f"expected grid transform coefficients must be finite: {list(transform)!r}")
        ok = False

    # Resolution (optional): two finite, strictly positive values.
    res = expected_grid.get("res")
    if res is not None:
        if not isinstance(res, (list, tuple)) or len(res) != 2:
            reasons.append("expected grid resolution must be a 2-element sequence")
            ok = False
        else:
            res_values = [_finite_float(v) for v in res]
            if any(v is None or v <= 0.0 for v in res_values):
                reasons.append(
                    f"expected grid resolution must be finite and strictly positive: {list(res)!r}"
                )
                ok = False

    return ok


def _validate_raster_against_grid(name: str, meta: dict, expected_grid: dict, reasons: list[str]) -> None:
    """Check a raster's CRS/transform/shape/resolution/band-count vs the grid."""
    exp_crs = expected_grid.get("crs")
    if exp_crs:
        if not meta.get("crs"):
            reasons.append(f"{name} has no CRS but the expected grid is {exp_crs!r}")
        elif not _crs_equal(meta["crs"], exp_crs):
            reasons.append(f"{name} crs {meta['crs']!r} != {exp_crs!r}")

    exp_transform = expected_grid.get("transform")
    if exp_transform is not None and not _transform_close(meta.get("transform"), exp_transform):
        reasons.append(f"{name} transform {meta.get('transform')!r} != {list(exp_transform)!r}")

    exp_shape = expected_grid.get("shape")
    if exp_shape is not None and meta.get("shape") != list(exp_shape):
        reasons.append(f"{name} shape {meta['shape']!r} != {list(exp_shape)!r}")

    exp_res = expected_grid.get("res")
    if exp_res is not None:
        if not _res_close(meta.get("res"), exp_res):
            reasons.append(f"{name} resolution {meta.get('res')!r} != {list(exp_res)!r}")

    expected_count = _EXPECTED_BAND_COUNT.get(name)
    if expected_count is not None and meta.get("count") != expected_count:
        reasons.append(f"{name} has {meta.get('count')} bands; expected {expected_count}")


def _mask_is_binary(path: Path, *, reasons: list[str]) -> bool:
    """Windowed scan: return True iff every pixel is a binary ``{0, 1}`` value.

    A binary mask accepts **only** ``0`` and ``1``:

    * NaN and +/-Inf are **rejected** explicitly, not filtered away — a
      non-finite mask value is invalid and must not be silently treated as zero
      (or as a valid pixel).
    * Any other finite value (e.g. a finite nodata sentinel such as ``-9999``)
      is **rejected**. Mask nodata metadata does **not** authorize non-binary
      pixels: the builder declares ``nodata=0.0`` for masks, and ``0`` is a
      legitimate *valid* mask value, so it is never excused as an invalid pixel.
      A non-zero, non-one nodata sentinel is likewise not a permitted pixel.
    """
    try:
        for _window, values in _iter_windows(path):
            if values.size == 0:
                continue
            # 1. Reject NaN / +Inf / -Inf outright.
            if not np.isfinite(values).all():
                reasons.append(f"{path.name} contains a non-finite (NaN/Inf) mask value")
                return False
            # 2. Every finite value must be exactly 0 or 1. Nodata metadata is
            #    deliberately NOT consulted here: only {0, 1} are permitted.
            binary = (np.abs(values) <= _BINARY_TOL) | (np.abs(values - 1.0) <= _BINARY_TOL)
            if not binary.all():
                example = float(values[~binary][0])
                reasons.append(f"{path.name} has non-binary mask values (e.g. {example!r})")
                return False
    except _RasterUnreadable as exc:
        reasons.append(f"{path.name} unreadable: {exc}")
        return False
    return True


def _iter_window_arrays(paths: list[Path]):
    """Yield aligned (window, {name: (bands, rows, cols) float64}) blocks.

    Every band of every raster is read for each window (bounded memory), so a
    multi-band SAR grid is validated band-by-band rather than only band 1.
    """
    import rasterio

    handles = {}
    try:
        for path in paths:
            handles[path.name] = _open_raster(path)
        ref = handles[paths[0].name]
        width, height = ref.width, ref.height
        block = max(1, int(math.isqrt(_WINDOW_PIXELS)))
        from rasterio.windows import Window

        for row_off in range(0, height, block):
            rows = min(block, height - row_off)
            for col_off in range(0, width, block):
                cols = min(block, width - col_off)
                window = Window(col_off, row_off, cols, rows)
                block_arrays = {}
                for path in paths:
                    block_arrays[path.name] = handles[path.name].read(window=window).astype("float64")
                yield window, block_arrays
    except _RasterUnreadable:
        raise
    except (rasterio.errors.RasterioIOError, OSError, ValueError) as exc:
        raise _RasterUnreadable(str(exc)) from exc
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass


def _validate_pixel_semantics(
    pair_dir: Path,
    *,
    expected_ignore_value: float,
    reasons: list[str],
) -> tuple[int, int]:
    """Windowed pixel-level consistency checks across label/SAR/masks.

    Returns ``(valid_px_count, invalid_px_count)`` derived from ``valid_mask.tif``.
    All checks scan every pixel with bounded memory.
    """
    s1_grid = pair_dir / _SAR_GRID_ARTIFACT
    label = pair_dir / _LABEL_ARTIFACT
    valid_mask = pair_dir / _VALID_MASK_ARTIFACT
    s1_valid_mask = pair_dir / _S1_VALID_MASK_ARTIFACT

    # Binary mask value checks first (cheap, and required for the cross-checks).
    bin_ok = True
    for mask_name in _BINARY_MASK_ARTIFACTS:
        bin_ok &= _mask_is_binary(pair_dir / mask_name, reasons=reasons)
    if not bin_ok:
        return 0, 0

    # Manifests for nodata conventions.
    s1_nodata = (_read_raster_meta(s1_grid) or {}).get("nodata")
    label_nodata = (_read_raster_meta(label) or {}).get("nodata")

    if label_nodata is None or not _close(label_nodata, expected_ignore_value):
        reasons.append(
            f"{_LABEL_ARTIFACT} nodata {label_nodata!r} != expected ignore value {expected_ignore_value!r}"
        )
    if s1_nodata is None:
        reasons.append(f"{_SAR_GRID_ARTIFACT} does not declare nodata; invalid SAR cannot be preserved")

    valid_px = 0
    invalid_px = 0
    supervision_subset_ok = True
    label_ignore_ok = True
    label_class_ok = True
    # Per-band SAR flags: a defect in band 2 must not be masked by a clean band 1.
    sar_invalid_ok: dict[int, bool] = {}
    sar_valid_finite_ok: dict[int, bool] = {}

    try:
        for _window, arrays in _iter_window_arrays([valid_mask, s1_valid_mask, label, s1_grid]):
            # Masks and label are single-band; SAR is multi-band.
            vm = arrays[valid_mask.name][0] > 0.5
            s1m = arrays[s1_valid_mask.name][0] > 0.5
            lab = arrays[label.name][0]
            sar_stack = arrays[s1_grid.name]  # (bands, rows, cols)

            valid_px += int(vm.sum())
            invalid_px += int((~vm).sum())

            # valid_mask (supervision) must be a subset of s1_valid_mask (SAR).
            if supervision_subset_ok and np.any(vm & ~s1m):
                reasons.append(
                    f"{_VALID_MASK_ARTIFACT} marks pixels valid that {_S1_VALID_MASK_ARTIFACT} marks invalid"
                )
                supervision_subset_ok = False

            # Label ignore consistency: invalid supervision => ignore value.
            if label_ignore_ok:
                invalid_supervision = ~vm
                bad_ignore = invalid_supervision & ~np.isclose(lab, float(expected_ignore_value), atol=_BINARY_TOL)
                if np.any(bad_ignore):
                    reasons.append(
                        f"{_LABEL_ARTIFACT} does not carry the ignore value where {_VALID_MASK_ARTIFACT} is 0"
                    )
                    label_ignore_ok = False

            # Label class consistency: valid supervision => only {0, 1}.
            if label_class_ok:
                lv = lab[vm]
                if lv.size:
                    bad = lv[~((np.abs(lv) <= _BINARY_TOL) | (np.abs(lv - 1.0) <= _BINARY_TOL))]
                    if bad.size:
                        reasons.append(
                            f"{_LABEL_ARTIFACT} contains a class other than 0/1 where supervision is valid: {float(bad[0])!r}"
                        )
                        label_class_ok = False

            # Validate EVERY SAR band against the SAR validity mask.
            for band_index in range(sar_stack.shape[0]):
                band = sar_stack[band_index]
                band_no = band_index + 1  # 1-based for messages

                # Invalid SAR pixels must be represented either non-finite (NaN)
                # or at the declared nodata sentinel — never as a plausibly valid
                # backscatter value. The builder's documented convention is
                # `nodata=NaN` shared across both bands, but a finite declared
                # sentinel is also accepted so the validator matches the actual
                # builder contract rather than inventing one.
                invalid_mask = ~s1m
                if sar_invalid_ok.get(band_index, True) and np.any(invalid_mask):
                    invalid_values = band[invalid_mask]
                    represented = ~np.isfinite(invalid_values)
                    if s1_nodata is not None:
                        represented |= np.isclose(
                            invalid_values, float(s1_nodata), atol=_BINARY_TOL, equal_nan=True
                        )
                    if not represented.all():
                        reasons.append(
                            f"{_SAR_GRID_ARTIFACT} band {band_no} has a plausible value where "
                            f"{_S1_VALID_MASK_ARTIFACT} is 0 (invalid SAR must be nodata/non-finite)"
                        )
                        sar_invalid_ok[band_index] = False

                # Valid SAR pixels must be finite AND must not equal the declared
                # nodata sentinel. A finite sentinel (e.g. -9999) at a pixel the
                # mask marks valid is a nodata/validity contradiction, not a
                # measurement — reject it. NaN nodata is handled via `isfinite`
                # (never by ordinary equality).
                if sar_valid_finite_ok.get(band_index, True) and np.any(s1m):
                    valid_values = band[s1m]
                    ok = np.isfinite(valid_values)
                    if s1_nodata is not None and math.isfinite(float(s1_nodata)):
                        ok &= ~np.isclose(valid_values, float(s1_nodata), atol=_BINARY_TOL)
                    if not ok.all():
                        reasons.append(
                            f"{_SAR_GRID_ARTIFACT} band {band_no} has a non-finite or nodata "
                            f"sentinel value where {_S1_VALID_MASK_ARTIFACT} is 1"
                        )
                        sar_valid_finite_ok[band_index] = False
    except _RasterUnreadable as exc:
        reasons.append(f"pixel-semantics scan failed: {exc}")
        return 0, 0

    return valid_px, invalid_px


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

    A pair is reusable only when every required artifact is present and readable,
    each raster matches the expected grid (CRS/transform/shape/resolution/band
    count), the pixel-level label/SAR/mask semantics hold, and the manifest
    record is present and consistent (including ``valid_px`` matching
    ``valid_mask.tif``).
    """
    pair_dir = Path(pair_dir)
    reasons: list[str] = []

    # 0. The expected grid must be usable for safe reuse.
    grid_ok = _validate_expected_grid(expected_grid, reasons)

    # 1. Every required artifact must be present.
    missing = [name for name in REQUIRED_PAIR_ARTIFACTS if not (pair_dir / name).exists()]
    if missing:
        reasons.append(f"missing required artifacts: {missing}")

    # 2. Manifest record must exist and be a well-formed object.
    if record is None:
        reasons.append("no per-pair manifest record")
        return CacheDecision(reusable=False, stale_reasons=reasons)
    if not isinstance(record, dict):
        reasons.append(f"manifest record is not an object: {type(record).__name__}")
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
    recorded_grid = record.get("grid")
    if not isinstance(recorded_grid, dict):
        reasons.append(f"recorded grid is not an object: {recorded_grid!r}")
    elif grid_ok:
        if expected_grid.get("crs") and not _crs_equal(recorded_grid.get("crs"), expected_grid.get("crs")):
            reasons.append(
                f"recorded grid crs {recorded_grid.get('crs')!r} != {expected_grid.get('crs')!r}"
            )
        if expected_grid.get("shape") is not None and list(recorded_grid.get("shape") or []) != list(
            expected_grid.get("shape")
        ):
            reasons.append(
                f"recorded grid shape {recorded_grid.get('shape')!r} != {expected_grid.get('shape')!r}"
            )
        exp_transform = expected_grid.get("transform")
        if exp_transform is not None and not _transform_close(
            recorded_grid.get("transform"), exp_transform
        ):
            reasons.append(
                f"recorded grid transform {recorded_grid.get('transform')!r} != {list(exp_transform)!r}"
            )
        exp_res = expected_grid.get("res")
        if exp_res is not None and not _res_close(recorded_grid.get("res"), exp_res):
            reasons.append(
                f"recorded grid resolution {recorded_grid.get('res')!r} != {list(exp_res)!r}"
            )

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
        if grid_ok:
            _validate_raster_against_grid(name, meta, expected_grid, reasons)

    # 6. Inter-raster grid consistency (CRS/transform/shape/resolution, not shape only).
    if len(raster_meta) > 1:
        ref_name, ref_meta = next(iter(raster_meta.items()))
        for name, meta in raster_meta.items():
            if name == ref_name:
                continue
            if meta["shape"] != ref_meta["shape"]:
                reasons.append(f"{name} shape {meta['shape']!r} != {ref_name} shape {ref_meta['shape']!r}")
            if not _transform_close(meta.get("transform"), ref_meta.get("transform")):
                reasons.append(f"{name} transform differs from {ref_name}")
            if not _crs_equal(meta.get("crs"), ref_meta.get("crs")):
                reasons.append(f"{name} CRS differs from {ref_name}")

    # 7. Pixel-level semantics. Only meaningful when all four artifacts are
    # present AND share an identical grid; otherwise the grid checks already
    # marked the pair stale and a cross-raster pixel scan cannot be aligned.
    all_same_grid = (
        len(raster_meta) == len(REQUIRED_PAIR_ARTIFACTS)
        and len({tuple(m["shape"]) for m in raster_meta.values()}) == 1
        and all(_transform_close(m.get("transform"), next(iter(raster_meta.values())).get("transform"))
                for m in raster_meta.values())
    )
    valid_px = None
    if not missing and all_same_grid:
        valid_px, _invalid_px = _validate_pixel_semantics(
            pair_dir,
            expected_ignore_value=expected_ignore_value,
            reasons=reasons,
        )

    # 8. Recorded valid_px must agree with the actual valid_mask count.
    recorded_valid_px = record.get("valid_px")
    if valid_px is not None and recorded_valid_px is not None:
        if not _close(recorded_valid_px, valid_px, tol=0.5):
            reasons.append(
                f"recorded valid_px {recorded_valid_px!r} != actual valid_mask count {valid_px}"
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
    """Construct the per-pair metadata record written to the dataset manifest.

    ``valid_px`` must be the count of pixels where ``valid_mask.tif`` is 1
    (the supervision-valid region).
    """
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def pair_record_for(manifest: dict, pair_dir: Path) -> dict | None:
    """Find a per-pair record by pair directory (string or resolved match)."""
    if not isinstance(manifest, dict):
        return None
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        return None
    target = str(pair_dir)
    for entry in pairs:
        if not isinstance(entry, dict):
            continue
        if entry.get("pair_dir") == target:
            return entry
        try:
            if Path(entry.get("pair_dir", "")).resolve() == pair_dir.resolve():
                return entry
        except OSError:
            continue
    return None
