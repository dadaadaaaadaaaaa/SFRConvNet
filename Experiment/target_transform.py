"""Shared, invertible transforms for physical SFR targets and baselines."""

from __future__ import annotations

import numpy as np


SUPPORTED_TARGET_TRANSFORMS = ("log10_1p", "log10_floor")
SUPPORTED_CLOSURE_TRANSFORMS = ("signed_asinh", "identity")


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


def transform_signed_closure(
    values,
    *,
    transform: str = "signed_asinh",
    scale: float = 1.0,
):
    """对可正可负的闭合项执行可逆变换。

    ``signed_asinh`` 在零点附近近似线性，在长尾区近似带符号对数，因而不会像
    SFR 的对数变换那样丢失负残差信息。``scale`` 必须由训练配置明确给出，并在
    训练、验证、测试缓存间保持一致。
    """

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Closure values contain NaN or Inf.")
    if scale <= 0:
        raise ValueError("closure scale must be positive.")
    if transform == "signed_asinh":
        return np.arcsinh(values / scale)
    if transform == "identity":
        return values.copy()
    raise ValueError(
        f"Unsupported closure transform {transform!r}; "
        f"expected {SUPPORTED_CLOSURE_TRANSFORMS}."
    )


def inverse_transform_signed_closure(
    values,
    *,
    transform: str = "signed_asinh",
    scale: float = 1.0,
):
    """反变换闭合项；该函数不会裁剪负值。"""

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Transformed closure values contain NaN or Inf.")
    if scale <= 0:
        raise ValueError("closure scale must be positive.")
    if transform == "signed_asinh":
        with np.errstate(over="ignore", invalid="ignore"):
            physical = scale * np.sinh(values)
    elif transform == "identity":
        physical = values.copy()
    else:
        raise ValueError(
            f"Unsupported closure transform {transform!r}; "
            f"expected {SUPPORTED_CLOSURE_TRANSFORMS}."
        )
    if not np.all(np.isfinite(physical)):
        raise ValueError("Inverse closure transform overflowed or produced non-finite values.")
    return physical
