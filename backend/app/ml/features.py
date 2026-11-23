import numpy as np

SAR_UNITS = ("dB", "linear")
SAR_UNIT_ALIASES = {
    "db": "dB",
    "decibel": "dB",
    "decibels": "dB",
    "linear": "linear",
    "power": "linear",
    "amplitude": "linear",
}

# Sen1Floods11 hand-labelled S1 chips and all HaorWatch training artifacts store
# VV/VH in decibels. This default is only used when a caller does not declare a
# unit explicitly; callers that handle external data must pass a unit.
DEFAULT_SAR_UNIT = "dB"


class SarUnitError(ValueError):
    """Raised when a SAR band unit is missing, unknown, or inconsistent."""


def normalize_unit(unit: str | None) -> str:
    if unit is None:
        raise SarUnitError(
            "Sentinel-1 band unit is not declared. Pass an explicit unit "
            "('dB' or 'linear'); units are never inferred from pixel values."
        )
    canonical = SAR_UNIT_ALIASES.get(str(unit).strip().lower())
    if canonical is None:
        raise SarUnitError(
            f"Unknown Sentinel-1 band unit {unit!r}. Supported units: {', '.join(SAR_UNITS)}."
        )
    return canonical


def to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear power to dB. Negative linear values become the floor."""
    return 10.0 * np.log10(np.clip(linear, 1e-6, None))


def band_to_db(band: np.ndarray, unit: str | None = None) -> np.ndarray:
    """Deterministically convert a SAR band to dB based on its declared unit.

    Never guesses the unit from pixel values. Raises :class:`SarUnitError` when
    the unit is missing or unknown.
    """
    canonical = normalize_unit(unit)
    if canonical == "dB":
        return band.astype(np.float32)
    return to_db(np.asarray(band, dtype=np.float64)).astype(np.float32)


def looks_like_db(band: np.ndarray) -> bool:
    """Heuristic used for dataset auditing only; never for silent conversion."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return True
    return float(np.median(finite)) < 0.0


def ensure_db(band: np.ndarray, unit: str | None = None) -> tuple[np.ndarray, bool]:
    """Return ``(band_db, converted)``.

    When ``unit`` is given the conversion is deterministic. When ``unit`` is
    ``None`` the legacy explicit opt-in ``'auto'`` behaviour is refused; callers
    that need the historical median heuristic must pass ``unit='auto'`` knowingly
    via :func:`band_to_db` on linear data. This function therefore requires a
    declared unit to avoid silent unit changes.
    """
    if unit is not None and str(unit).strip().lower() == "auto":
        raise SarUnitError(
            "Automatic SAR unit detection is disabled. Declare the band unit explicitly."
        )
    canonical = normalize_unit(unit)
    if canonical == "dB":
        return band.astype(np.float32), False
    return to_db(np.asarray(band, dtype=np.float64)).astype(np.float32), True


def compute_norm_stats(samples: list[np.ndarray], valid_masks: list[np.ndarray]) -> dict:
    sums = None
    sq_sums = None
    count = 0
    for sample, valid in zip(samples, valid_masks):
        stacked = sample[:, valid]
        if stacked.size == 0:
            continue
        sums = stacked.sum(axis=1) if sums is None else sums + stacked.sum(axis=1)
        sq_sums = (stacked ** 2).sum(axis=1) if sq_sums is None else sq_sums + (stacked ** 2).sum(axis=1)
        count += stacked.shape[1]
    if count == 0:
        raise ValueError("No valid pixels found for normalization statistics.")
    mean = sums / count
    variance = np.maximum(sq_sums / count - mean ** 2, 1e-8)
    return {
        "mean": [float(value) for value in mean],
        "std": [float(np.sqrt(value)) for value in variance],
        "pixel_count": int(count),
    }


def normalize(band: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((band - mean) / std).astype(np.float32)
