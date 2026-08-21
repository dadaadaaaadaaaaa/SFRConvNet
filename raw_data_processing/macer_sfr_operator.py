"""MACER2D 恒星形成算子的纯 NumPy 实现。

本模块刻意不依赖 ``GalData`` 或任何粗粒化实现，便于分别在 HR 场、
``R_h`` 输出和 ``R_m`` 输出上调用。输入和输出均保持 MACER 代码单位：

* 长度：kpc；
* 时间：Gyr；
* 质量：``2.5e7 Msun``；
* ``cell_sfr``：代码质量 / Gyr，与 HDF Data-Set-11 的数值语义一致；
* ``sfr_density``：代码质量 / (代码体积 * Gyr)。

Data-Set-11 在 ``hdfall.F`` 中写作 ``volume * rhodot_save``。它的 HDF
标签虽写有 ``M/yr``，源码并未在写出时执行物理单位换算。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np


SFRMode = Literal["data_compatible", "paper_physical"]

DATA_COMPATIBLE_MODE: SFRMode = "data_compatible"
PAPER_PHYSICAL_MODE: SFRMode = "paper_physical"

# 这些参数来自当前数据反演结论；取得生产运行 zmp_inp 后应再次核对。
MACER_G_CODE = 112.3
DATA_COMPATIBLE_CSTAR = 0.01
PAPER_PHYSICAL_CSTAR = 0.1

TEMPERATURE_FACTOR = 93.0
TEMPERATURE_CEILING_K = 4.0e4
NUMBER_DENSITY_FACTOR = 1.02
NUMBER_DENSITY_FLOOR = 1.0

LIMITER_INELIGIBLE = 0
LIMITER_COOLING = 1
LIMITER_FREE_FALL = 2
LIMITER_ROTATION = 3


@dataclass(frozen=True)
class MacerSFRResult:
    """一次恒星形成算子求值的结果和逐格诊断。

    ``limiter_code`` 的含义为：0=未通过恒星形成阈值，1=冷却限制，
    2=自由落体限制，3=旋转限制。边界相等时采用确定性的优先级：
    冷却优先于自由落体，自由落体优先于旋转。
    """

    sfr_density: np.ndarray
    cell_sfr: np.ndarray
    total_sfr: float
    temperature: np.ndarray
    number_density_proxy: np.ndarray
    eligible_mask: np.ndarray
    tau_cool: np.ndarray
    tau_free_fall: np.ndarray
    tau_rotation: np.ndarray
    tau_dynamical_min: np.ndarray
    tau_star_formation: np.ndarray
    finite_step_factor: np.ndarray
    limiter_code: np.ndarray
    mode: SFRMode
    parameters: Mapping[str, object]


def exp_factor_sf(time_step: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray:
    """复现 ``star.F::exp_factor_sf`` 的有限时间步转换率。

    ``time_step == 0`` 被定义为连续极限 ``1/tau``，用于没有内部流体
    时间步的离线诊断。若要逐值复现 Data-Set-11，应传入生成该输出时的
    内部流体时间步，而不是相邻 HDF snapshot 的时间间隔。
    """

    dt_values, tau_values = np.broadcast_arrays(
        np.asarray(time_step, dtype=np.float64),
        np.asarray(tau, dtype=np.float64),
    )
    if not np.all(np.isfinite(dt_values)) or np.any(dt_values < 0):
        raise ValueError("time_step must be finite and non-negative.")
    if not np.all(np.isfinite(tau_values)) or np.any(tau_values <= 0):
        raise ValueError("tau must be finite and strictly positive.")

    result = np.empty(dt_values.shape, dtype=np.float64)
    continuous = dt_values == 0
    taylor = (~continuous) & (dt_values < 0.01 * tau_values)
    exponential = ~(continuous | taylor)

    result[continuous] = 1.0 / tau_values[continuous]
    result[taylor] = (
        1.0 - 0.5 * dt_values[taylor] / tau_values[taylor]
    ) / tau_values[taylor]
    # expm1 在 dt/tau 很小时比直接计算 1-exp(-dt/tau) 更稳定；
    # 该分支的代数含义与 Fortran 源码一致。
    result[exponential] = -np.expm1(
        -dt_values[exponential] / tau_values[exponential]
    ) / dt_values[exponential]
    return result


def calculate_macer_sfr(
    *,
    density: np.ndarray,
    internal_energy_density: np.ndarray,
    cooling_time: np.ndarray,
    cell_volume: np.ndarray,
    radius: np.ndarray,
    time_step: np.ndarray | float = 0.0,
    mode: SFRMode = DATA_COMPATIBLE_MODE,
    black_hole_mass: np.ndarray | float | None = None,
    radial_acceleration: np.ndarray | None = None,
) -> MacerSFRResult:
    """计算 MACER 的 ``isf=2`` 型局部 SFR 和逐格 SFR。

    参数
    ----
    density
        Data-Set-5，即源码变量 ``d``，不是需要再除以 1.02 的数密度。
    internal_energy_density
        Data-Set-10，即源码变量 ``e``（气体内能密度）。
    cooling_time
        Data-Set-17，单位为 Gyr。
    cell_volume
        Data-Set-15。逐格 SFR 通过 ``sfr_density * cell_volume`` 得到。
    radius
        球极坐标半径，单位 kpc。可为完整二维场或可广播的一维/二维数组。
    time_step
        MACER 内部流体时间步，单位 Gyr；默认 0 表示连续极限。
    mode
        ``data_compatible``：``G=112.3``、``cstar=0.01``，并以
        ``M_code(<r) ~= M_BH`` 计算旋转周期。
        ``paper_physical``：``G=112.3``、``cstar=0.1``，并由局部径向
        加速度计算旋转周期。其余阈值与有限时间步处理保持一致，以便对照。
    black_hole_mass
        数据兼容模式必需，单位为代码质量，可为正标量或可广播数组。
    radial_acceleration
        论文物理模式必需，对应 Data-Set-8；公式使用其绝对值。
    """

    if mode not in (DATA_COMPATIBLE_MODE, PAPER_PHYSICAL_MODE):
        raise ValueError(
            f"Unsupported mode {mode!r}; expected 'data_compatible' or 'paper_physical'."
        )

    rho = _finite_array("density", density, non_negative=True)
    energy_density = _finite_array(
        "internal_energy_density", internal_energy_density, non_negative=True
    )
    cool = _finite_array("cooling_time", cooling_time, non_negative=True)
    volume = _finite_array("cell_volume", cell_volume, strictly_positive=True)

    shape = rho.shape
    if rho.ndim == 0:
        raise ValueError("density must be an array with at least one dimension.")
    for name, values in (
        ("internal_energy_density", energy_density),
        ("cooling_time", cool),
        ("cell_volume", volume),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {shape}.")

    radius_field = _broadcast_field(
        "radius", radius, shape, strictly_positive=True
    )
    time_step_field = _broadcast_field(
        "time_step", time_step, shape, non_negative=True
    )

    # rho=0 的格点不能形成恒星；其自由落体时标在诊断中记为无穷大，
    # 而不是用任意密度占位值制造一个看似有限的物理时标。
    positive_density = rho > 0
    temperature = np.full(shape, np.inf, dtype=np.float64)
    temperature[positive_density] = (
        TEMPERATURE_FACTOR
        * energy_density[positive_density]
        / rho[positive_density]
    )
    number_density_proxy = NUMBER_DENSITY_FACTOR * rho

    tau_free_fall = np.full(shape, np.inf, dtype=np.float64)
    tau_free_fall[positive_density] = np.sqrt(
        3.0 * np.pi
        / (32.0 * MACER_G_CODE * rho[positive_density])
    )

    if mode == DATA_COMPATIBLE_MODE:
        if black_hole_mass is None:
            raise ValueError("black_hole_mass is required in data_compatible mode.")
        enclosed_mass = _broadcast_field(
            "black_hole_mass", black_hole_mass, shape, strictly_positive=True
        )
        tau_rotation = 2.0 * np.pi * np.sqrt(
            radius_field**3 / (MACER_G_CODE * enclosed_mass)
        )
        cstar = DATA_COMPATIBLE_CSTAR
        rotation_definition = "2pi*sqrt(r^3/(G*M_BH))"
    else:
        if radial_acceleration is None:
            raise ValueError("radial_acceleration is required in paper_physical mode.")
        acceleration = _broadcast_field(
            "radial_acceleration", radial_acceleration, shape
        )
        acceleration_magnitude = np.abs(acceleration)
        tau_rotation = np.full(shape, np.inf, dtype=np.float64)
        nonzero_acceleration = acceleration_magnitude > 0
        tau_rotation[nonzero_acceleration] = 2.0 * np.pi * np.sqrt(
            radius_field[nonzero_acceleration]
            / acceleration_magnitude[nonzero_acceleration]
        )
        cstar = PAPER_PHYSICAL_CSTAR
        rotation_definition = "2pi*sqrt(r/abs(a_r))"

    free_fall_selected = tau_free_fall <= tau_rotation
    tau_dynamical_min = np.minimum(tau_free_fall, tau_rotation)
    cooling_selected = cool >= tau_dynamical_min
    tau_star = np.maximum(cool, tau_dynamical_min)

    eligible = (
        positive_density
        & (temperature < TEMPERATURE_CEILING_K)
        & (number_density_proxy > NUMBER_DENSITY_FLOOR)
    )
    if np.any(~np.isfinite(tau_star[eligible])) or np.any(tau_star[eligible] <= 0):
        raise ValueError(
            "The star-formation timescale in every eligible cell must be finite and positive. "
            "Check cooling_time, radius, and the selected rotation input."
        )
    finite_step_factor = np.zeros(shape, dtype=np.float64)
    if np.any(eligible):
        finite_step_factor[eligible] = exp_factor_sf(
            time_step_field[eligible], tau_star[eligible]
        )
    sfr_density = np.where(
        eligible,
        cstar * rho * finite_step_factor,
        0.0,
    )
    cell_sfr = sfr_density * volume

    limiter_code = np.full(shape, LIMITER_INELIGIBLE, dtype=np.int8)
    dynamical_code = np.where(
        free_fall_selected, LIMITER_FREE_FALL, LIMITER_ROTATION
    )
    limiter_code[eligible] = np.where(
        cooling_selected[eligible],
        LIMITER_COOLING,
        dynamical_code[eligible],
    )

    parameters = MappingProxyType(
        {
            "operator_id": f"macer_sfr_{mode}",
            "operator_version": 1,
            "G_code": MACER_G_CODE,
            "cstar": cstar,
            "isf_scheme": 2,
            "length_unit": "kpc",
            "time_unit": "Gyr",
            "mass_unit": "2.5e7_Msun",
            "sfr_stored_unit": "MACER_mass_code/Gyr",
            "sfr_to_msun_per_yr": 0.025,
            "temperature_ceiling_K": TEMPERATURE_CEILING_K,
            "number_density_definition": "n=1.02*density",
            "number_density_floor": NUMBER_DENSITY_FLOOR,
            "rotation_timescale_definition": rotation_definition,
            "cooling_time_source": "Data-Set-17",
            "cell_volume_source": "Data-Set-15",
            "target_semantics": "Data-Set-11=cell_volume*sfr_density",
            "time_step_zero_semantics": "continuous_limit",
            "finite_step_correction": "star.F::exp_factor_sf",
            "finite_step_input_mode": (
                "continuous_limit" if np.all(time_step_field == 0) else "provided_time_step"
            ),
            "mass_enclosed_approximation": (
                "M_code(<r)~=M_BH"
                if mode == DATA_COMPATIBLE_MODE
                else "not_used"
            ),
            "parameter_provenance": (
                "data_inferred_pending_runtime_configuration"
                if mode == DATA_COMPATIBLE_MODE
                else "published_formula"
            ),
        }
    )
    return MacerSFRResult(
        sfr_density=sfr_density,
        cell_sfr=cell_sfr,
        total_sfr=float(cell_sfr.sum(dtype=np.float64)),
        temperature=temperature,
        number_density_proxy=number_density_proxy,
        eligible_mask=eligible,
        tau_cool=cool.copy(),
        tau_free_fall=tau_free_fall,
        tau_rotation=tau_rotation,
        tau_dynamical_min=tau_dynamical_min,
        tau_star_formation=tau_star,
        finite_step_factor=finite_step_factor,
        limiter_code=limiter_code,
        mode=mode,
        parameters=parameters,
    )


def _finite_array(
    name: str,
    value: np.ndarray,
    *,
    non_negative: bool = False,
    strictly_positive: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if strictly_positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    if non_negative and np.any(values < 0):
        raise ValueError(f"{name} must be non-negative.")
    return values


def _broadcast_field(
    name: str,
    value: np.ndarray | float,
    shape: tuple[int, ...],
    *,
    non_negative: bool = False,
    strictly_positive: bool = False,
) -> np.ndarray:
    values = _finite_array(
        name,
        value,
        non_negative=non_negative,
        strictly_positive=strictly_positive,
    )
    try:
        return np.broadcast_to(values, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {values.shape} cannot broadcast to {shape}."
        ) from exc
