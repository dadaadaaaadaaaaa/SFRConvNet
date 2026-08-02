"""Shared, invertible transforms for physical SFR targets and baselines."""

from __future__ import annotations

import numpy as np


SUPPORTED_TARGET_TRANSFORMS = ("log10_1p", "log10_floor")


def transform_sfr(values, *, transform: str = "log10_1p", scale: float = 1.0, floor: float = 1e-9):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("SFR values contain NaN or Inf.")
    if np.any(values < 0):
        raise ValueError("SFR values must be non-negative.")

    if transform == "log10_1p":
        if scale <= 0:
            raise ValueError("target scale must be positive for log10_1p.")
        return np.log10(1.0 + values / scale)
    if transform == "log10_floor":
        if floor <= 0:
            raise ValueError("target floor must be positive for log10_floor.")
        return np.log10(values + floor)
    raise ValueError(f"Unsupported target transform {transform!r}; expected {SUPPORTED_TARGET_TRANSFORMS}.")


def inverse_transform_sfr(
    values,
    *,
    transform: str = "log10_1p",
    scale: float = 1.0,
    floor: float = 1e-9,
    clip_nonnegative: bool = True,
):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Transformed SFR values contain NaN or Inf.")
    if transform == "log10_1p":
        if scale <= 0:
            raise ValueError("target scale must be positive for log10_1p.")
        with np.errstate(over="ignore", invalid="ignore"):
            physical = scale * np.expm1(np.log(10.0) * values)
    elif transform == "log10_floor":
        if floor <= 0:
            raise ValueError("target floor must be positive for log10_floor.")
        with np.errstate(over="ignore", invalid="ignore"):
            physical = np.power(10.0, values) - floor
    else:
        raise ValueError(f"Unsupported target transform {transform!r}; expected {SUPPORTED_TARGET_TRANSFORMS}.")
    if not np.all(np.isfinite(physical)):
        raise ValueError("Inverse SFR transform overflowed or produced non-finite values.")
    return np.maximum(physical, 0.0) if clip_nonnegative else physical
