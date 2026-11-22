"""Cache-validity contract for built S1/S2 pairs.

A cached pair may be reused only when it contains every required artifact,
including ``valid_mask.tif``, and its per-pair metadata agrees with the expected
grid, SAR unit, ignore value, and source dates. Anything else is stale.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from app.ml.dataset_cache import (
    BUILDER_VERSION,
    build_pair_record,
    evaluate_pair_cache,
    load_dataset_manifest,
    pair_record_for,
)

GRID = {"crs": "EPSG:32646", "transform": [10.0, 0.0, 0.0, 0.0, -10.0, 0.0], "shape": [8, 8]}
TRANSFORM = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)


def _write(path: Path, data, crs="EPSG:32646"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", count=1, height=data.shape[0], width=data.shape[1],
        dtype="float32", crs=crs, transform=TRANSFORM,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def _make_pair(pair_dir: Path, *, with_valid_mask=True, with_s1_valid=False, crs="EPSG:32646"):
    arr = np.ones((8, 8), dtype=np.float32)
    _write(pair_dir / "s1_grid.tif", arr, crs=crs)
    _write(pair_dir / "pseudo_label.tif", arr, crs=crs)
    if with_valid_mask:
        _write(pair_dir / "valid_mask.tif", arr, crs=crs)
    if with_s1_valid:
        _write(pair_dir / "s1_valid_mask.tif", arr, crs=crs)


def _good_record(pair_dir: Path, **overrides):
    record = build_pair_record(
        pair_dir=pair_dir, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=GRID, label_ignore_value=-1.0,
        nodata={"s1_nodata": None, "s2_nodata": None, "label_ignore_value": -1.0},
        valid_px=64, invalid_px=0, water_px=10,
    )
    record.update(overrides)
    return record


def _evaluate(pair_dir, record, **kwargs):
    return evaluate_pair_cache(
        pair_dir, expected_grid=GRID, expected_sar_unit="dB",
        expected_ignore_value=-1.0, record=record, **kwargs,
    )


def test_complete_valid_pair_is_reusable(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_missing_valid_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, with_valid_mask=False)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask" in r for r in decision.stale_reasons)


def test_missing_pseudo_label_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    (pair / "pseudo_label.tif").unlink()
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable


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
    bad_grid = dict(GRID, shape=[4, 4])
    decision = _evaluate(pair, _good_record(pair, grid=bad_grid))
    assert not decision.reusable


def test_wrong_grid_transform_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad_grid = dict(GRID, transform=[20.0, 0.0, 0.0, 0.0, -20.0, 0.0])
    decision = _evaluate(pair, _good_record(pair, grid=bad_grid))
    assert not decision.reusable


def test_wrong_source_dates_are_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair), expected_s1_day="2026-01-01")
    assert not decision.reusable


def test_on_disk_crs_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, crs="EPSG:32647")  # on-disk CRS differs from expected grid
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable


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
