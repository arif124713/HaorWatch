"""Cache-validity contract for built S1/S2 pairs.

A cached pair may be reused only when it contains **all four** required
artifacts (``s1_grid.tif``, ``pseudo_label.tif``, ``valid_mask.tif``,
``s1_valid_mask.tif``), every raster matches the expected grid
(CRS/transform/shape/resolution/band count), the pixel-level semantics hold, and
the per-pair manifest record agrees with the expected unit, ignore value,
builder version, source dates, grid, and valid-pixel count.

Builder semantics encoded by these fixtures (do not invent new meanings):
``valid_mask`` = supervision validity (AOI ∧ valid SAR ∧ valid S2);
``s1_valid_mask`` = valid SAR observations; ``valid_mask ⊆ s1_valid_mask``;
``pseudo_label`` = ignore value where ``valid_mask`` is 0; ``s1_grid`` = NaN
where ``s1_valid_mask`` is 0.
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

# Base masks over an 8x8 grid.
# s1_valid_mask: 1 = valid SAR. Bottom row invalid.
S1_VALID = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype="float32",
)
# valid_mask (supervision): subset of S1_VALID (last two columns of row 6 also invalid S2).
VALID = S1_VALID.copy()
VALID[6, 6:] = 0.0


def _write(path: Path, data, *, crs="EPSG:32646", transform=TRANSFORM, nodata=None,
           count=1, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    with rasterio.open(
        path, "w", driver="GTiff", count=count, height=data.shape[1], width=data.shape[2],
        dtype=dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)


def _make_pair(
    pair_dir: Path,
    *,
    missing=None,
    crs="EPSG:32646",
    transform=TRANSFORM,
    label_nodata=IGNORE,
    label_values=None,
    s1_valid_values=None,
    valid_values=None,
    s1_nodata=np.nan,
    s1_values=None,
    with_valid_mask=True,
    with_s1_valid_mask=True,
):
    """Write a complete, valid pair; individual defects can be injected."""
    pair_dir.mkdir(parents=True, exist_ok=True)
    s1m = S1_VALID if s1_valid_values is None else s1_valid_values
    vm = VALID if valid_values is None else valid_values

    # s1_grid: 2 bands (VV, VH); finite where SAR valid, NaN where invalid.
    if s1_values is None:
        band = np.where(s1m > 0.5, -12.0, np.nan).astype("float32")
        s1 = np.stack([band, band])
    else:
        arr = np.asarray(s1_values, dtype="float32")
        s1 = arr if arr.ndim == 3 else np.stack([arr, arr])
    # label: ignore where supervision invalid, 1 where valid (water).
    if label_values is None:
        label = np.where(vm > 0.5, 1.0, IGNORE).astype("float32")
    else:
        label = np.asarray(label_values, dtype="float32")

    specs = {
        "s1_grid.tif": dict(data=s1, nodata=s1_nodata, count=2),
        "pseudo_label.tif": dict(data=label, nodata=label_nodata, count=1),
        "valid_mask.tif": dict(data=vm, nodata=0.0, count=1),
        "s1_valid_mask.tif": dict(data=s1m, nodata=0.0, count=1),
    }
    for name, spec in specs.items():
        if name == missing:
            continue
        if name == "valid_mask.tif" and not with_valid_mask:
            continue
        if name == "s1_valid_mask.tif" and not with_s1_valid_mask:
            continue
        _write(pair_dir / name, spec["data"], crs=crs, transform=transform,
               nodata=spec["nodata"], count=spec["count"])


def _good_record(pair_dir: Path, **overrides):
    record = build_pair_record(
        pair_dir=pair_dir, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=GRID, label_ignore_value=IGNORE,
        nodata={"s1_nodata": None, "s2_nodata": None, "label_ignore_value": IGNORE},
        valid_px=int(VALID.sum()), invalid_px=int((VALID == 0).sum()), water_px=int(VALID.sum()),
    )
    record.update(overrides)
    return record


def _evaluate(pair_dir, record, **kwargs):
    return evaluate_pair_cache(
        pair_dir, expected_grid=GRID, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record, **kwargs,
    )


def test_required_artifacts_equals_critical_no_contradiction():
    assert set(CRITICAL_PAIR_ARTIFACTS) == set(REQUIRED_PAIR_ARTIFACTS)
    assert "s1_valid_mask.tif" in REQUIRED_PAIR_ARTIFACTS


def test_complete_valid_pair_is_reusable(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


# --- missing artifacts ------------------------------------------------------ #

@pytest.mark.parametrize(
    "missing", ["s1_grid.tif", "pseudo_label.tif", "valid_mask.tif", "s1_valid_mask.tif"]
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


# --- CRS / grid ------------------------------------------------------------- #

def test_on_disk_crs_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, crs="EPSG:32647")
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("crs" in r for r in decision.stale_reasons)


def test_missing_crs_on_raster_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    # Rewrite one raster with no CRS at all.
    _write(pair / "valid_mask.tif", VALID, crs=None, nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("no CRS" in r for r in decision.stale_reasons)


def test_transform_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, transform=Affine(10.0, 0.0, 5.0, 0.0, -10.0, 5.0))
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("transform" in r for r in decision.stale_reasons)


def test_shape_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    _write(pair / "s1_grid.tif", np.full((2, 4, 4), -12.0), nodata=np.nan, count=2)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_grid.tif shape" in r for r in decision.stale_reasons)


def test_resolution_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0))
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("resolution" in r or "transform" in r for r in decision.stale_reasons)


def test_expected_grid_resolution_length_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad_grid = dict(GRID, res=[10.0, 10.0, 10.0])
    decision = _evaluate(pair, _good_record(pair, grid=bad_grid))
    assert not decision.reusable


def test_missing_expected_crs_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad_grid = dict(GRID)
    bad_grid.pop("crs")
    decision = evaluate_pair_cache(
        pair, expected_grid=bad_grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=_good_record(pair),
    )
    assert not decision.reusable
    assert any("crs" in r for r in decision.stale_reasons)


def test_inter_raster_grid_mismatch_is_stale(tmp_path):
    """One raster has a different transform than the others (same shape)."""
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    _write(pair / "s1_valid_mask.tif", S1_VALID, transform=Affine(10.0, 0.0, 3.0, 0.0, -10.0, 0.0),
           nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable


def test_band_count_mismatch_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    # s1_grid should have 2 bands; write 1.
    _write(pair / "s1_grid.tif", np.where(S1_VALID > 0.5, -12.0, np.nan), nodata=np.nan, count=1)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("bands" in r for r in decision.stale_reasons)


# --- pixel-level semantics -------------------------------------------------- #

def test_label_ignore_inconsistent_with_valid_mask_is_stale(tmp_path):
    """valid_mask says invalid, but the label carries a valid class there."""
    pair = tmp_path / "pair_00"
    bad_label = np.where(VALID > 0.5, 1.0, 0.0)  # invalid pixels labelled 0, not ignore
    _make_pair(pair, label_values=bad_label)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("ignore value" in r for r in decision.stale_reasons)


def test_label_class_outside_binary_where_valid_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    bad_label = np.where(VALID > 0.5, 2.0, IGNORE)  # class 2 is unsupported
    _make_pair(pair, label_values=bad_label)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("class other than 0/1" in r for r in decision.stale_reasons)


def test_sar_invalid_inconsistent_with_s1_valid_mask_is_stale(tmp_path):
    """s1_valid_mask says invalid, but s1_grid has a finite value there."""
    pair = tmp_path / "pair_00"
    bad_sar = np.full((8, 8), -12.0, dtype="float32")  # all finite even where invalid
    _make_pair(pair, s1_values=bad_sar)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("plausible value where" in r for r in decision.stale_reasons)


def test_non_finite_valid_sar_pixels_is_stale(tmp_path):
    """s1_valid_mask says valid, but s1_grid is NaN there."""
    pair = tmp_path / "pair_00"
    bad_sar = np.where(S1_VALID > 0.5, np.nan, np.nan)  # all NaN
    _make_pair(pair, s1_values=bad_sar)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("where" in r and "is 1" in r for r in decision.stale_reasons)


def test_supervision_mask_not_subset_of_sar_mask_is_stale(tmp_path):
    """valid_mask must be a subset of s1_valid_mask."""
    pair = tmp_path / "pair_00"
    # Both masks valid at the pixel where SAR was made finite; but supervision
    # valid where SAR invalid.
    bad_valid = S1_VALID.copy()
    bad_valid[7, :] = 1.0  # supervision valid where SAR (S1_VALID row 7) is invalid
    bad_sar = np.where(S1_VALID > 0.5, -12.0, np.nan)
    bad_label = np.where(bad_valid > 0.5, 1.0, IGNORE)
    _make_pair(pair, valid_values=bad_valid, label_values=bad_label, s1_values=bad_sar)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("subset" in r or "marks pixels valid" in r for r in decision.stale_reasons)


def test_non_binary_validity_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    non_binary = np.full((8, 8), 0.5, dtype="float32")
    _make_pair(pair, valid_values=non_binary, label_values=np.where(non_binary > 0.5, 1.0, IGNORE))
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("non-binary" in r for r in decision.stale_reasons)


def test_sar_grid_without_nodata_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, s1_nodata=None)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("does not declare nodata" in r for r in decision.stale_reasons)


def test_label_without_ignore_sentinel_nodata_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair, label_nodata=None)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("nodata" in r for r in decision.stale_reasons)


# --- invalid pixels outside the first 512x512 sample ------------------------ #

def test_invalid_pixel_beyond_first_512_block_is_detected(tmp_path):
    """Regression: a lossy 512x512 sample would miss an invalid pixel at (600, 600)."""
    pair = tmp_path / "pair_00"
    size = 700
    grid = dict(GRID, shape=[size, size], res=[10.0, 10.0])
    s1m = np.ones((size, size), dtype="float32")
    valid = np.ones((size, size), dtype="float32")
    s1 = np.full((size, size), -12.0, dtype="float32")
    label = np.ones((size, size), dtype="float32")

    # Inject a single label inconsistency far beyond a 512x512 decimation:
    # valid_mask says valid, but label carries ignore there.
    label[600, 600] = IGNORE

    _write(pair / "s1_grid.tif", np.stack([s1, s1]), nodata=np.nan, count=2)
    _write(pair / "pseudo_label.tif", label, nodata=IGNORE)
    _write(pair / "valid_mask.tif", valid, nodata=0.0)
    _write(pair / "s1_valid_mask.tif", s1m, nodata=0.0)

    record = build_pair_record(
        pair_dir=pair, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=grid, label_ignore_value=IGNORE,
        nodata={}, valid_px=int(valid.sum()), invalid_px=0, water_px=int(valid.sum()),
    )
    decision = evaluate_pair_cache(
        pair, expected_grid=grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record,
    )
    assert not decision.reusable
    assert any("class other than 0/1" in r for r in decision.stale_reasons)


def test_large_raster_validates_with_bounded_reads(tmp_path, monkeypatch):
    """A large, fully valid raster must validate without loading it all at once."""
    import app.ml.dataset_cache as dc

    pair = tmp_path / "pair_00"
    size = 1500  # 2.25M pixels per band; > the old 4M-sample threshold combined
    grid = dict(GRID, shape=[size, size], res=[10.0, 10.0])
    ones = np.ones((size, size), dtype="float32")
    _write(pair / "s1_grid.tif", np.stack([ones * -12.0, ones * -18.0]), nodata=np.nan, count=2)
    _write(pair / "pseudo_label.tif", ones, nodata=IGNORE)
    _write(pair / "valid_mask.tif", ones, nodata=0.0)
    _write(pair / "s1_valid_mask.tif", ones, nodata=0.0)

    # Track the largest single window read to prove bounded memory.
    max_window_pixels = 0
    orig = dc._iter_windows

    record = build_pair_record(
        pair_dir=pair, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=grid, label_ignore_value=IGNORE,
        nodata={}, valid_px=size * size, invalid_px=0, water_px=size * size,
    )
    decision = evaluate_pair_cache(
        pair, expected_grid=grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record,
    )
    assert decision.reusable, decision.stale_reasons

    # Confirm the windowed reader is used and bounded.
    import rasterio
    from app.ml.dataset_cache import _WINDOW_PIXELS

    with rasterio.open(pair / "valid_mask.tif") as src:
        assert src.width * src.height > _WINDOW_PIXELS  # genuinely large
    # The maximum block area must be <= the bounded window size.
    for _window, values in orig(pair / "valid_mask.tif"):
        max_window_pixels = max(max_window_pixels, values.size)
    assert max_window_pixels <= _WINDOW_PIXELS


# --- corruption / manifest -------------------------------------------------- #

def test_corrupted_tiff_is_stale_not_exception(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    (pair / "valid_mask.tif").write_bytes(b"not a tiff at all")
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask.tif" in r and ("unreadable" in r or "corrupted" in r)
               for r in decision.stale_reasons)


def test_no_manifest_record_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, None)
    assert not decision.reusable
    assert any("manifest record" in r for r in decision.stale_reasons)


def test_malformed_manifest_record_is_stale_not_crash(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    # Not a dict at all.
    decision = evaluate_pair_cache(
        pair, expected_grid=GRID, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=["not", "a", "dict"],
    )
    assert not decision.reusable

    # Dict but missing grid.
    decision2 = evaluate_pair_cache(
        pair, expected_grid=GRID, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record={"sar_unit": "dB"},
    )
    assert not decision2.reusable
    assert any("grid is not an object" in r or "builder_version" in r for r in decision2.stale_reasons)


def test_wrong_valid_px_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair, valid_px=123456))
    assert not decision.reusable
    assert any("valid_px" in r for r in decision.stale_reasons)


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


def test_wrong_source_dates_are_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair), expected_s1_day="2026-01-01")
    assert not decision.reusable


def test_existing_v04_cache_without_validity_artifacts_is_stale(tmp_path):
    """A pre-validity-fix pair (only s1_grid + pseudo_label) must be stale."""
    pair = tmp_path / "pair_00"
    old = np.full((8, 8), 1.0, dtype="float32")
    _write(pair / "s1_grid.tif", np.stack([old, old]), nodata=np.nan, count=2)
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


def test_load_manifest_non_object_returns_empty(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_dataset_manifest(p) == {}


def test_load_manifest_corrupt_json_returns_empty(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_dataset_manifest(p) == {}


def test_pair_record_for_malformed_manifest_is_none(tmp_path):
    assert pair_record_for({"pairs": "notalist"}, tmp_path) is None
    assert pair_record_for([], tmp_path) is None


# =========================================================================== #
# §1 Per-SAR-band validation (VV and VH must each be checked)
# =========================================================================== #

def _write_pair_two_band(pair, vv, vh, *, valid_mask=None, s1_valid_mask=None,
                         label=None, nodata=np.nan):
    """Write a pair with explicit VV/VH bands and masks for per-band tests."""
    pair.mkdir(parents=True, exist_ok=True)
    s1m = S1_VALID if s1_valid_mask is None else s1_valid_mask
    vm = VALID if valid_mask is None else valid_mask
    if label is None:
        label = np.where(vm > 0.5, 1.0, IGNORE).astype("float32")
    _write(pair / "s1_grid.tif", np.stack([vv.astype("float32"), vh.astype("float32")]),
           nodata=nodata, count=2)
    _write(pair / "pseudo_label.tif", label, nodata=IGNORE, count=1)
    _write(pair / "valid_mask.tif", vm, nodata=0.0, count=1)
    _write(pair / "s1_valid_mask.tif", s1m, nodata=0.0, count=1)


def test_valid_two_band_sar_passes(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)  # both bands valid
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_invalid_value_in_sar_band1_is_rejected(tmp_path):
    """Band 1 (VV) has a plausible value where SAR is invalid; band 2 is clean."""
    pair = tmp_path / "pair_00"
    vv = np.where(S1_VALID > 0.5, -12.0, np.nan).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, np.nan).astype("float32")
    # Corrupt band 1 only: finite value on an invalid pixel.
    vv[7, 3] = -12.0
    _write_pair_two_band(pair, vv, vh)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 1" in r for r in decision.stale_reasons)


def test_invalid_value_in_sar_band2_is_rejected(tmp_path):
    """Band 1 (VV) is clean but band 2 (VH) has an invalid pixel — must be caught."""
    pair = tmp_path / "pair_00"
    vv = np.where(S1_VALID > 0.5, -12.0, np.nan).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, np.nan).astype("float32")
    # Corrupt band 2 only.
    vh[7, 5] = -18.0
    _write_pair_two_band(pair, vv, vh)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 2" in r for r in decision.stale_reasons)


def test_non_finite_valid_pixel_in_band2_is_rejected(tmp_path):
    """Band 2 has NaN where SAR is valid (band 1 is clean)."""
    pair = tmp_path / "pair_00"
    vv = np.where(S1_VALID > 0.5, -12.0, np.nan).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, np.nan).astype("float32")
    vh[2, 2] = np.nan  # valid pixel, but band 2 missing
    _write_pair_two_band(pair, vv, vh)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 2" in r for r in decision.stale_reasons)


def test_corrupt_pixel_in_later_window_is_detected(tmp_path):
    """A band-2 defect far beyond the first processing window must be detected."""
    pair = tmp_path / "pair_00"
    size = 1400  # forces >1 window at the bounded block size
    grid = dict(GRID, shape=[size, size], res=[10.0, 10.0])
    vv = np.full((size, size), -12.0, dtype="float32")
    vh = np.full((size, size), -18.0, dtype="float32")
    s1m = np.ones((size, size), dtype="float32")
    valid = np.ones((size, size), dtype="float32")
    label = np.ones((size, size), dtype="float32")
    # Inject a single wrong-band-2 value at a valid pixel at (1300, 1300).
    vh[1300, 1300] = np.nan
    _write(pair / "s1_grid.tif", np.stack([vv, vh]), nodata=np.nan, count=2)
    _write(pair / "pseudo_label.tif", label, nodata=IGNORE)
    _write(pair / "valid_mask.tif", valid, nodata=0.0)
    _write(pair / "s1_valid_mask.tif", s1m, nodata=0.0)
    record = build_pair_record(
        pair_dir=pair, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=grid, label_ignore_value=IGNORE,
        nodata={}, valid_px=size * size, invalid_px=0, water_px=size * size,
    )
    decision = evaluate_pair_cache(
        pair, expected_grid=grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record,
    )
    assert not decision.reusable
    assert any("band 2" in r for r in decision.stale_reasons)


# =========================================================================== #
# §2 NaN / Inf in binary masks must be rejected, not silently ignored
# =========================================================================== #

@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_in_valid_mask_is_stale(tmp_path, bad_value):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad = VALID.copy()
    bad[3, 3] = bad_value
    _write(pair / "valid_mask.tif", bad, nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask.tif" in r and "non-finite" in r for r in decision.stale_reasons)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nan_or_inf_in_s1_valid_mask_is_stale(tmp_path, bad_value):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad = S1_VALID.copy()
    bad[4, 4] = bad_value
    _write(pair / "s1_valid_mask.tif", bad, nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_valid_mask.tif" in r and "non-finite" in r for r in decision.stale_reasons)


def test_nan_mask_value_beyond_first_window_is_detected(tmp_path):
    pair = tmp_path / "pair_00"
    size = 1400
    grid = dict(GRID, shape=[size, size], res=[10.0, 10.0])
    vv = np.full((size, size), -12.0, dtype="float32")
    vh = np.full((size, size), -18.0, dtype="float32")
    mask = np.ones((size, size), dtype="float32")
    mask[1300, 1300] = np.nan  # far beyond the first window
    label = np.ones((size, size), dtype="float32")
    _write(pair / "s1_grid.tif", np.stack([vv, vh]), nodata=np.nan, count=2)
    _write(pair / "pseudo_label.tif", label, nodata=IGNORE)
    _write(pair / "valid_mask.tif", mask, nodata=0.0)
    _write(pair / "s1_valid_mask.tif", mask, nodata=0.0)
    record = build_pair_record(
        pair_dir=pair, sar_unit="dB", s1_day="2026-06-07", s2_day="2026-06-07",
        s1_meta={"image_count": 1}, aoi_id="abc", grid=grid, label_ignore_value=IGNORE,
        nodata={}, valid_px=size * size, invalid_px=0, water_px=size * size,
    )
    decision = evaluate_pair_cache(
        pair, expected_grid=grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=record,
    )
    assert not decision.reusable
    assert any("non-finite" in r for r in decision.stale_reasons)


def test_valid_binary_masks_still_pass(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


# =========================================================================== #
# §3 Expected-grid metadata validation
# =========================================================================== #

def _eval_with_grid(pair, grid):
    return evaluate_pair_cache(
        pair, expected_grid=grid, expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=_good_record(pair),
    )


def test_malformed_shape_wrong_length_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, shape=[8]))
    assert not decision.reusable
    assert any("shape" in r for r in decision.stale_reasons)


@pytest.mark.parametrize("bad_shape", [[8.5, 8], [8, 8.0001], [0, 8], [-1, 8], [8, 0]])
def test_fractional_or_non_positive_shape_is_stale(tmp_path, bad_shape):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, shape=bad_shape))
    assert not decision.reusable
    assert any("positive integers" in r for r in decision.stale_reasons)


@pytest.mark.parametrize("bad_shape", [[True, 8], [8, False]])
def test_boolean_shape_dimension_is_stale(tmp_path, bad_shape):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, shape=bad_shape))
    assert not decision.reusable
    assert any("positive integers" in r for r in decision.stale_reasons)


@pytest.mark.parametrize("bad_transform", [
    [10.0, 0.0, 0.0, 0.0, -10.0],              # too few
    [10.0, 0.0, 0.0, 0.0, -10.0, 0.0, 1.0],    # too many
    [np.nan, 0.0, 0.0, 0.0, -10.0, 0.0],       # NaN coefficient
    [np.inf, 0.0, 0.0, 0.0, -10.0, 0.0],       # Inf coefficient
])
def test_malformed_transform_is_stale(tmp_path, bad_transform):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, transform=bad_transform))
    assert not decision.reusable
    assert any("transform" in r for r in decision.stale_reasons)


@pytest.mark.parametrize("bad_res", [[10.0], [10.0, 10.0, 10.0], [0.0, 10.0], [-10.0, 10.0],
                                     [np.nan, 10.0], [np.inf, 10.0]])
def test_invalid_resolution_is_stale(tmp_path, bad_res):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, res=bad_res))
    assert not decision.reusable
    assert any("resolution" in r for r in decision.stale_reasons)


def test_malformed_crs_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, crs="not-a-crs"))
    assert not decision.reusable
    assert any("crs" in r for r in decision.stale_reasons)


def test_non_string_crs_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _eval_with_grid(pair, dict(GRID, crs=32646))
    assert not decision.reusable
    assert any("crs" in r for r in decision.stale_reasons)


def test_expected_grid_not_object_is_structured_failure(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = evaluate_pair_cache(
        pair, expected_grid=["not", "a", "dict"], expected_sar_unit="dB",
        expected_ignore_value=IGNORE, record=_good_record(pair),
    )
    assert not decision.reusable
    assert any("not an object" in r for r in decision.stale_reasons)


# =========================================================================== #
# §4 Nodata / metadata consistency between builder and validator
# =========================================================================== #

def test_sar_grid_nan_nodata_convention_passes(tmp_path):
    """Builder convention: nodata=NaN shared across bands; invalid pixels NaN."""
    pair = tmp_path / "pair_00"
    _make_pair(pair)  # nodata=np.nan
    with rasterio.open(pair / "s1_grid.tif") as src:
        assert np.isnan(src.nodata)
        assert src.count == 2
        assert src.dtypes[0] == "float32"
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_finite_nodata_sentinel_with_matching_pixels_passes(tmp_path):
    """A finite declared nodata sentinel is accepted when invalid pixels use it."""
    pair = tmp_path / "pair_00"
    vv = np.where(S1_VALID > 0.5, -12.0, -9999.0).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, -9999.0).astype("float32")
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_finite_nodata_sentinel_but_plausible_invalid_value_is_stale(tmp_path):
    """Declared finite sentinel, but an invalid pixel holds a plausible value."""
    pair = tmp_path / "pair_00"
    vv = np.where(S1_VALID > 0.5, -12.0, -9999.0).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, -9999.0).astype("float32")
    vh[7, 1] = -18.0  # plausible value on an invalid pixel
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 2" in r for r in decision.stale_reasons)


# =========================================================================== #
# §1 (review) Finite SAR nodata sentinel must never appear at a SAR-valid pixel
# =========================================================================== #

def _sentinel_pair(nodata=-9999.0):
    """A pair with a finite declared sentinel used at invalid pixels in both bands."""
    vv = np.where(S1_VALID > 0.5, -12.0, nodata).astype("float32")
    vh = np.where(S1_VALID > 0.5, -18.0, nodata).astype("float32")
    return vv, vh


def test_finite_sentinel_at_valid_pixel_band1_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    vv, vh = _sentinel_pair()
    vv[2, 2] = -9999.0  # sentinel where s1_valid_mask == 1
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 1" in r for r in decision.stale_reasons)


def test_finite_sentinel_at_valid_pixel_band2_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    vv, vh = _sentinel_pair()
    vh[3, 5] = -9999.0  # sentinel where s1_valid_mask == 1, band 2 only
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("band 2" in r for r in decision.stale_reasons)


def test_finite_sentinel_at_invalid_pixels_both_bands_is_reusable(tmp_path):
    """Sentinel used consistently at invalid pixels (both bands) is the contract."""
    pair = tmp_path / "pair_00"
    vv, vh = _sentinel_pair()
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_nan_nodata_convention_still_passes(tmp_path):
    """The default NaN-nodata builder convention continues to be accepted."""
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons


def test_each_sar_band_validated_independently_for_sentinel(tmp_path):
    """Band 1 clean at valid pixels; band 2 carries a sentinel there -> stale."""
    pair = tmp_path / "pair_00"
    vv, vh = _sentinel_pair()
    # Band 1 has no sentinel on valid pixels; band 2 does.
    vh[1, 1] = -9999.0
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    # Only the offending band is reported.
    assert any("band 2" in r for r in decision.stale_reasons)
    assert not any("band 1" in r for r in decision.stale_reasons)


def test_plausible_finite_value_at_invalid_pixel_still_fails(tmp_path):
    """Regression: the existing invalid-pixel check is preserved."""
    pair = tmp_path / "pair_00"
    vv, vh = _sentinel_pair()
    vh[7, 0] = -18.0  # plausible value where s1_valid_mask == 0
    _write_pair_two_band(pair, vv, vh, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("plausible value where" in r for r in decision.stale_reasons)


# =========================================================================== #
# §2 (review) Binary masks accept only {0, 1}; nodata cannot excuse -9999
# =========================================================================== #

def test_finite_sentinel_in_valid_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad = VALID.copy()
    bad[3, 3] = -9999.0  # non-binary finite value in valid_mask
    _write(pair / "valid_mask.tif", bad, nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("valid_mask.tif" in r and "non-binary" in r for r in decision.stale_reasons)


def test_finite_sentinel_in_s1_valid_mask_is_stale(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad = S1_VALID.copy()
    bad[4, 4] = -9999.0
    _write(pair / "s1_valid_mask.tif", bad, nodata=0.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable
    assert any("s1_valid_mask.tif" in r and "non-binary" in r for r in decision.stale_reasons)


def test_sentinel_as_declared_mask_nodata_still_non_binary_is_stale(tmp_path):
    """Even if a mask DECLARES -9999 as nodata, a -9999 pixel is still rejected."""
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    bad = VALID.copy()
    bad[2, 5] = -9999.0
    # Declare the sentinel as the mask nodata; it must still be rejected because
    # masks only permit {0, 1}.
    _write(pair / "valid_mask.tif", bad, nodata=-9999.0)
    decision = _evaluate(pair, _good_record(pair))
    assert not decision.reusable


def test_valid_binary_masks_only_zero_and_one_pass(tmp_path):
    pair = tmp_path / "pair_00"
    _make_pair(pair)
    decision = _evaluate(pair, _good_record(pair))
    assert decision.reusable, decision.stale_reasons
