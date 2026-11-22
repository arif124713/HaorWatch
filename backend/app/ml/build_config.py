"""Runtime configuration for the Hakaluki v0.4 dataset builder.

This module is the single, explicit, validated configuration path for the
dataset builder. It connects the parameters declared in ``params.yaml`` (and the
DVC stage) to the actual runtime behaviour, without importing ``params.yaml``
through fragile working-directory assumptions: DVC passes the values explicitly
as CLI arguments.

Standalone CLI defaults preserve the historical constants so the script still
runs without DVC. Invalid values are rejected before any network or filesystem
side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

from app.ml.features import SAR_UNITS, normalize_unit, SarUnitError

# Defaults mirroring the historical builder constants.
DEFAULT_SAR_UNIT = "dB"
DEFAULT_S2_CLOUD_MAX = 12
DEFAULT_PAIR_WINDOW_DAYS = 1
DEFAULT_LABEL_IGNORE_VALUE = -1.0
DEFAULT_REBUILD_STALE = False

# Sane bounds for validation.
_MAX_CLOUD_PERCENT = 100

# Boolean tokens accepted for parameters that may be supplied as strings (e.g.
# from a config file). Anything else is rejected, so "false"/"0"/"no" can never
# silently become True.
_TRUE_TOKENS = {"true", "1", "yes", "on"}
_FALSE_TOKENS = {"false", "0", "no", "off"}


class BuildConfigError(ValueError):
    """Raised when a builder runtime parameter is invalid."""


@dataclass(frozen=True)
class BuildRuntimeConfig:
    sar_unit: str
    s2_cloud_max: int
    pair_window_days: int
    label_ignore_value: float
    rebuild_stale: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _validate_sar_unit(value: str) -> str:
    try:
        return normalize_unit(value)
    except SarUnitError as exc:
        raise BuildConfigError(str(exc)) from exc


def _coerce_int(name: str, value) -> int:
    """Coerce to int, rejecting bools, fractional values, and non-numeric input.

    A bool is rejected (``True``/``False`` are not valid integer counts). A
    fractional value such as ``12.9`` or ``1.9`` is rejected rather than
    silently truncated. Numeric strings that represent a whole number are
    accepted for config-file friendliness.
    """
    if isinstance(value, bool):
        raise BuildConfigError(f"{name} must be an integer, got a boolean {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BuildConfigError(f"{name} must be a finite integer, got {value!r}")
        if not float(value).is_integer():
            raise BuildConfigError(f"{name} must be a whole number, got {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            raise BuildConfigError(f"{name} must be an integer, got {value!r}") from None
        if not math.isfinite(number) or not number.is_integer():
            raise BuildConfigError(f"{name} must be a whole number, got {value!r}")
        return int(number)
    raise BuildConfigError(f"{name} must be an integer, got {value!r}")


def _validate_cloud_max(value) -> int:
    number = _coerce_int("s2_cloud_max", value)
    if not (0 <= number <= _MAX_CLOUD_PERCENT):
        raise BuildConfigError(
            f"s2_cloud_max must be between 0 and {_MAX_CLOUD_PERCENT}, got {value!r}"
        )
    return number


def _validate_pair_window_days(value) -> int:
    number = _coerce_int("pair_window_days", value)
    if number < 0:
        raise BuildConfigError(f"pair_window_days must be non-negative, got {value!r}")
    return number


def _validate_label_ignore_value(value) -> float:
    if isinstance(value, bool):
        raise BuildConfigError(f"label_ignore_value must be numeric, got a boolean {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BuildConfigError(f"label_ignore_value must be numeric, got {value!r}") from None
    if not math.isfinite(number):
        raise BuildConfigError(f"label_ignore_value must be finite, got {value!r}")
    # The ignore sentinel must not collide with a real class value (0 or 1).
    if number in (0.0, 1.0):
        raise BuildConfigError(
            f"label_ignore_value {value!r} collides with a real class value (0 or 1)"
        )
    return number


def _validate_bool(name: str, value) -> bool:
    """Coerce to bool, accepting only explicit boolean inputs.

    ``bool("false")`` would otherwise be ``True``; string tokens are parsed
    explicitly and anything unrecognised is rejected.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        raise BuildConfigError(f"{name} must be a boolean, got {value!r}")
    raise BuildConfigError(f"{name} must be a boolean, got {value!r}")


def build_runtime_config(
    *,
    sar_unit: str = DEFAULT_SAR_UNIT,
    s2_cloud_max=DEFAULT_S2_CLOUD_MAX,
    pair_window_days=DEFAULT_PAIR_WINDOW_DAYS,
    label_ignore_value=DEFAULT_LABEL_IGNORE_VALUE,
    rebuild_stale=DEFAULT_REBUILD_STALE,
) -> BuildRuntimeConfig:
    """Validate and construct the builder runtime configuration.

    Every declared DVC parameter flows through here so that a parameter change
    actually changes runtime behaviour, and an invalid value fails loudly before
    any GEE request or output file creation.
    """
    return BuildRuntimeConfig(
        sar_unit=_validate_sar_unit(sar_unit),
        s2_cloud_max=_validate_cloud_max(s2_cloud_max),
        pair_window_days=_validate_pair_window_days(pair_window_days),
        label_ignore_value=_validate_label_ignore_value(label_ignore_value),
        rebuild_stale=_validate_bool("rebuild_stale", rebuild_stale),
    )
