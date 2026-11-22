"""Cache-validity contract for built S1/S2 pairs.

A cached pair may be reused only when it contains **all four** required
artifacts (``s1_grid.tif``, ``pseudo_label.tif``, ``valid_mask.tif``,
``s1_valid_mask.tif``), every raster matches the expected grid
(CRS/transform/shape/resolution), the nodata/binary semantics hold, and the
per-pair manifest record agrees with the expected unit, ignore value, builder
version, source dates, grid, and valid-pixel count. Anything else is stale.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from app.ml.dataset_cache import (
    BUILDER_VERSION,
    CRITICAL_PAIR_ARTIFACTS,
    REQUIRED_PAIR_ARTIFACTS,
    build_pair_record,
    evaluate_pair_cache,
    load_dataset_manifest,
    pair_record_for,
)

GRID = {
    "crs": "EPSG:32646",
    "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 0.0],
    "shape": [8, 8],
    "res": [10.0, 10.0],
}
TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
IGNORE = -1.0
# A row with invalid pixels so the label/SAR semantics are exercised.
VALID = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ],
    dtype="float32",
)


def _write(path: Path, data, *, crs="EPSG:32646", transform=TRANSFORM, nodata=None, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", count=1, height=data.shape[0], width=data.shape[1],
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(np.asarray(data, dtype=dtype), 1)


def _make_pair(
    pair_dir: Path,
    *,
    with_valid_mask=True,
    with_s1_valid_mask=True,
    missing=None,
    crs="EPSG:32646",
    transform=TRANSFORM,
    label_nodata=IGNORE,
    label_values=None,
    mask_values=None,
    s1_nodata=np.nan,
    s1_values=None,
):
    """Write a complete, valid pair; individual defects can be injected."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    s1 = VALID.copy()
    if s1_values is not None:
        s1 = s1_values
    else:
        # Invalid pixels become NaN (matches the builder's preserved invalidity).
        s1 = np.where(VALID > 0.5, -12.0, np.nan).astype("float32")
    label = np.where(VALID > 0.5, 1.0, IGNORE).astype("float32")
    if label_values is not None:
        label = label_values
    mask = VALID.copy() if mask_values is None else mask_values

    specs = {
        "s1_grid.tif": dict(data=s1, nodata=s1_nodata),
        "pseudo_label.tif": dict(data=label, nodata=label_nodata),
        "valid_mask.tif": dict(data=mask, nodata=0.0),
        "s1_valid_mask.tif": dict(data=mask, nodata=0.0),
    }
    for name, spec in specs.items():
        if name == missing:
            continue
        if name == "valid_mask.tif" and not with_valid_mask:
            continue
        if name == "s1_valid_mask.tif" and not with_s1_valid_mask:
            continue
        _write(pair_dir / name, spec["data"], crs=crs, transform=transform, nodata=spec["nodata"])


def _good_record(pair_dir: Path, **overrides):
    record = build_pair_record(
        pair_dir=pair_dir, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=GRID, label_ignore_value=IGNORE,
        nodata={"s1_nodata": None, "s2_nodata": None, "label_ignore_value": IGNORE},
        valid_px=60, invalid_px=4, water_px=40,
    )
    record.update(overrides)
    return record


def _evaluate(pair_dir, record, **kwargs):
    return evaluate_pair_cache(
        pair_dir, expected_grid=GRID, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record, **kwargs,
    )


def test_required_artifacts_equals_critical_no_contradiction():
    # The full required set must be validated; the legacy "critical" alias must
    # not allow a smaller subset.
    assert set(CRITICAL_PAIR_ARTIFACTS) == set(REQUIRED_PAIR_ARTIFACTS)
    assert "s1_valid_mask.tif" in REQUIRED_PAIR_ARTIFACTS


def test_complete_valid_pair_is_reusable(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


@pytest.mark.parametrize(
    "missing",
    ["s1_grid.tif", "pseudo_label.tif", "valid_mask.tif", "s1_valid_mask.tif"],
)
def test_missing_any_required_artifact_is_stale(tmp_path, missing):
    pair = tmp_path / "pair_00"
    _make_pair(pair, missing=missing)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable, missing
    assert any("missing required" in r for r in decision.stale_reasons)


def test_missing_s1_valid_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, with_s1_valid_mask=False)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_valid_mask.tif" in r for r in decision.stale_reasons)


def test_missing_valid_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, with_valid_mask=False)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask.tif" in r for r in decision.stale_reasons)


def test_no_manifest_record_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, None)
    assert not decision.reusable
    assert any("manifest record" in r for r in decision.stale_reasons)


def test_wrong_builder_version_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, builder_version="0.0.1-old"))
    assert not decision.reusable
    assert any("builder_version" in r for r in decision.stale_reasons)


def test_wrong_sar_unit_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, sar_unit="linear"))
    assert not decision.reusable


def test_wrong_ignore_value_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, label_ignore_value=0.0))
    assert not decision.reusable


def test_wrong_grid_shape_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, grid=dict(GRID, shape=[4, 4])))
    assert not decision.reusable


def test_wrong_grid_transform_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, grid=dict(GRID, transform=[20.0, 0.0, 0.0, 0.0, -20.0, 0.0])))
    assert not decision.reusable


def test_wrong_source_dates_are_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair), expected_s1_day="2026-01-01")
    assert not decision.reusable


def test_on_disk_crs_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, crs="EPSG:32647")
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("crs" in r for r in decision.stale_reasons)


def test_on_disk_transform_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, transform=Affine(10.0, 0.0, 5.0, 0.0, -10.0, 5.0))
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("transform" in r for r in decision.stale_reasons)


def test_on_disk_shape_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    small = np.ones((4, 4), dtype="float32")
    _make_pair(pair)
    _write(pair / "s1_grid.tif", small, crs="EPSG:32646", nodata=np.nan)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_grid.tif shape" in r for r in decision.stale_reasons)


def test_on_disk_resolution_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    # 8x8 at 20 m instead of 10 m: shape still 8x8, resolution differs.
    _make_pair(pair, transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0))
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("resolution" in r or "transform" in r for r in decision.stale_reasons)


def test_corrupted_tiff_is_stale_not_exception(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    (pair / "valid_mask.tif").write_bytes(b"not a tiff at all")
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask.tif unreadable" in r for r in decision.stale_reasons)


def test_non_binary_validity_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    non_binary = np.full((8, 8), 0.5, dtype="float32")
    _make_pair(pair, mask_values=non_binary)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("non-binary" in r for r in decision.stale_reasons)


def test_label_without_ignore_sentinel_nodata_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, label_nodata=None)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("nodata" in r for r in decision.stale_reasons)


def test_label_with_unexpected_class_value_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    bad = np.full((8, 8), 0.5, dtype="float32")  # neither 0, 1, nor ignore
    _make_pair(pair, label_values=bad, label_nodata=None)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable


def test_sar_grid_without_nodata_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, s1_nodata=None)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_grid.tif does not declare nodata" in r for r in decision.stale_reasons)


def test_sar_grid_finite_nodata_but_no_such_pixels_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    finite_vals = np.full((8, 8), -12.0, dtype="float32")  # all valid, no -9999 pixels
    _make_pair(pair, s1_values=finite_vals, s1_nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("carries none" in r for r in decision.stale_reasons)


def test_existing_v04_cache_without_validity_artifacts_is_stale(tmp_path):
    """A pre-validity-fix pair (only s1_grid + pseudo_label) must be stale."""
    pair = tmp_path / "pair_00"
    old = np.full((8, 8), 1.0, dtype="float32")
    _write(pair / "s1_grid.tif", old, nodata=np.nan)
    _write(pair / "pseudo_label.tif", old, nodata=IGNORE)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("missing required" in r for r in decision.stale_reasons)


def test_manifest_roundtrip_and_lookup(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    manifest = {"pairs": [_good_record(pair)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_dataset_manifest(tmp_path / "manifest.json")
    found = pair_record_for(loaded, pair)
    assert found is not None
    assert found["builder_version"] == BUILDER_VERSION


def test_load_manifest_missing_returns_empty(tmp_path):
    assert load_dataset_manifest(tmp_path / "nope.json") == {}
