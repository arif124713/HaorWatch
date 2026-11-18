import numpy as np
import pytest

from app.ml.features import compute_norm_stats, ensure_db, to_db


def test_to_db_converts_linear_power():
    linear = np.array([100.0, 10.0, 1.0])
    db = to_db(linear)
    assert db[0] == pytest.approx(20.0)
    assert db[1] == pytest.approx(10.0)
    assert db[2] == pytest.approx(0.0)


def test_ensure_db_detects_units():
    db_values = np.array([-15.0, -20.0, -12.0])
    out, converted = ensure_db(db_values)
    assert converted is False
    assert out[0] == pytest.approx(-15.0)
    linear_values = np.array([0.031, 0.01, 0.063])
    out, converted = ensure_db(linear_values)
    assert converted is True
    assert out[0] == pytest.approx(-15.0, abs=0.2)


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
