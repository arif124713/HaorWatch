import numpy as np
import pytest

from app.ml.features import (
    DEFAULT_SAR_UNIT,
    SarUnitError,
    band_to_db,
    compute_norm_stats,
    ensure_db,
    normalize_unit,
    to_db,
)


def test_to_db_converts_linear_power():
    linear = np.array([100.0, 10.0, 1.0])
    db = to_db(linear)
    assert db[0] == pytest.approx(20.0)
    assert db[1] == pytest.approx(10.0)
    assert db[2] == pytest.approx(0.0)


def test_band_to_db_linear_input_is_converted():
    linear = np.array([0.031, 0.01, 0.063])
    out = band_to_db(linear, "linear")
    assert out[0] == pytest.approx(-15.0, abs=0.2)


def test_band_to_db_db_input_is_preserved():
    db_values = np.array([-15.0, -20.0, -12.0])
    out = band_to_db(db_values, "dB")
    np.testing.assert_allclose(out, db_values)


def test_unknown_unit_raises():
    with pytest.raises(SarUnitError):
        band_to_db(np.array([1.0, 2.0]), "bananas")


def test_missing_unit_raises():
    with pytest.raises(SarUnitError):
        band_to_db(np.array([1.0, 2.0]), None)


def test_auto_detection_is_refused():
    with pytest.raises(SarUnitError):
        ensure_db(np.array([-15.0, -20.0]), "auto")


def test_nan_and_nodata_values_are_preserved_by_conversion():
    linear = np.array([0.01, np.nan])
    out = band_to_db(linear, "linear")
    assert np.isfinite(out[0])
    # NaN is propagated deterministically so downstream valid masks can exclude it.
    assert np.isnan(out[1])


def test_negative_linear_values_are_clipped_not_nan():
    out = band_to_db(np.array([-1.0, 0.0]), "linear")
    assert np.isfinite(out).all()
    assert out[0] == out[1] == pytest.approx(10.0 * np.log10(1e-6))


def test_default_unit_is_db():
    assert normalize_unit(DEFAULT_SAR_UNIT) == "dB"
    assert normalize_unit("linear") == "linear"


def test_norm_stats_mean_std():
    samples = [
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ]
        )
    ]
    valid = [np.array([[True, True], [True, False]])]
    stats = compute_norm_stats(samples, valid)
    assert stats["mean"][0] == pytest.approx(2.0)
    assert stats["mean"][1] == pytest.approx(20.0)
    assert stats["std"][0] == pytest.approx(np.std([1.0, 2.0, 3.0]), rel=1e-5)
    assert stats["pixel_count"] == 3
