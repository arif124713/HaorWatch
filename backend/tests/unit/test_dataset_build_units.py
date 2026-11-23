"""Dataset-preparation unit handling: dB inputs must be declared and never double-converted."""

import numpy as np
import pytest

from app.ml.features import SarUnitError, ensure_db


def test_declared_db_is_not_converted():
    # GEE COPERNICUS/S1_GRD and Sen1Floods11 _S1Hand chips are dB.
    db_values = np.array([-12.0, -18.0, -25.0], dtype=np.float32)
    out, converted = ensure_db(db_values, "dB")
    assert converted is False
    np.testing.assert_allclose(out, db_values)


def test_missing_unit_raises_in_dataset_build_path():
    # The dataset builder must not silently guess; missing unit must fail clearly.
    with pytest.raises(SarUnitError):
        ensure_db(np.array([0.01, 0.02], dtype=np.float32))


def test_declared_linear_is_converted_once():
    linear = np.array([0.01], dtype=np.float32)  # -20 dB
    out, converted = ensure_db(linear, "linear")
    assert converted is True
    assert out[0] == pytest.approx(-20.0, abs=0.01)


def test_already_db_values_are_not_converted_to_linear_garbage():
    # Regression: passing dB values through a linear conversion would corrupt them.
    db_values = np.array([-15.0], dtype=np.float32)
    out, converted = ensure_db(db_values, "dB")
    assert converted is False
    assert out[0] == pytest.approx(-15.0)
