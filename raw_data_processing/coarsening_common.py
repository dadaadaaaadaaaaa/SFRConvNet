"""共享的验证、几何、简化及守恒工具"""

from __future__ import annotations

from collections.abc import Collection
from typing import Tuple

import numpy as np

try:
    from raw_data_processing.coarsening_types import CoarseningInput, CoarseningResult
except ModuleNotFoundError:  # Support ``python raw_data_processing/galaxy_data.py``.
    from coarsening_types import CoarseningInput, CoarseningResult


SUPPORTED_COARSENING_FIELDS = frozenset(
    {
        "density",
        "gas_energy",
        "temperature",
        "v1",
        "v2",
        "v3",
        "initial_stellar_density",
        "radial_acceleration",
        "theta_acceleration",
        "cooling_time",
        "cooling_rate_density",
    }
)


# Data-Set-10 保存的是气体内能密度。字段名 ``gas_energy`` 仅为兼容旧缓存；
# 任何守恒沉积都必须先乘单元体积，得到单元气体内能。
GAS_INTERNAL_ENERGY_DENSITY_FIELD = "gas_energy"
GAS_INTERNAL_ENERGY_DENSITY_DATASET = "Data-Set-10"
TEMPERATURE_FROM_INTERNAL_ENERGY_DENSITY_FACTOR = 93.0
COOLING_TIME_FIELD = "cooling_time"
COOLING_RATE_DENSITY_FIELD = "cooling_rate_density"


def cooling_rate_density_from_time(
    internal_energy_density: np.ndarray,
    cooling_time: np.ndarray,
) -> np.ndarray:
    """由 ``C=e/t_cool`` 构造冷却率密度。

    原始盘星系的初始帧可能把尚未定义的冷却时标写为 0；这里将其解释为
    零冷却功率，而不是执行除零。正常演化帧的正时标严格按公式换算。
    """

    energy_density = np.asarray(internal_energy_density, dtype=np.float64)
    timescale = np.asarray(cooling_time, dtype=np.float64)
    if energy_density.shape != timescale.shape:
        raise ValueError("Internal-energy density and cooling time must have identical shapes.")
    if not np.all(np.isfinite(timescale)) or np.any(timescale < 0):
        raise ValueError("Cooling time must contain only finite, non-negative values.")
    return np.divide(
        energy_density,
        timescale,
        out=np.zeros_like(energy_density),
        where=timescale > 0,
    )


def cooling_time_from_rate_density(
    internal_energy_density: np.ndarray,
    cooling_rate_density: np.ndarray,
) -> np.ndarray:
    """由粗粒化后的 ``e_C/C_C`` 重建冷却时标。"""

    energy_density = np.asarray(internal_energy_density, dtype=np.float64)
    rate_density = np.asarray(cooling_rate_density, dtype=np.float64)
    if energy_density.shape != rate_density.shape:
        raise ValueError(
            "Internal-energy density and cooling-rate density must have identical shapes."
        )
    if not np.all(np.isfinite(rate_density)) or np.any(rate_density < 0):
        raise ValueError("Cooling-rate density must contain only finite, non-negative values.")
    return np.divide(
        energy_density,
        rate_density,
        out=np.zeros_like(energy_density),
        where=rate_density > 0,
    )


def spherical_meridional_to_cylindrical(
    radial: np.ndarray,
    theta_component: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把球坐标 ``(r,theta)`` 分量转到固定 ``(R,z)`` 基。"""

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return (
        radial * sin_theta + theta_component * cos_theta,
        radial * cos_theta - theta_component * sin_theta,
    )


def cylindrical_meridional_to_spherical(
    cylindrical_r: np.ndarray,
    cylindrical_z: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把固定 ``(R,z)`` 分量转回粗网格中心的球坐标基。"""

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return (
        cylindrical_r * sin_theta + cylindrical_z * cos_theta,
        cylindrical_r * cos_theta - cylindrical_z * sin_theta,
    )


def temperature_from_internal_energy_density(
    internal_energy_density: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """按 MACER 恒等式 ``T = 93 e / rho`` 由内能密度重建温度。"""

    energy_density = np.asarray(internal_energy_density, dtype=np.float64)
    density_values = np.asarray(density, dtype=np.float64)
    if energy_density.shape != density_values.shape:
        raise ValueError(
            "Internal-energy-density and density arrays must have identical shapes."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return TEMPERATURE_FROM_INTERNAL_ENERGY_DENSITY_FACTOR * energy_density / density_values


def thermodynamic_coarsening_metadata() -> dict[str, object]:
    """返回可序列化的 Data-Set-10 字段语义，供粗粒化结果写入 HDF5。"""

    return {
        "gas_energy_semantics": "gas_internal_energy_density",
        "gas_energy_source_dataset": GAS_INTERNAL_ENERGY_DENSITY_DATASET,
        "conserved_internal_energy_definition": "sum(gas_energy*cell_volume)",
        "temperature_relation": "temperature=93*gas_energy/density",
        "cooling_coarsening": "sum((gas_energy/cooling_time)*cell_volume)/sum(cell_volume)",
        "cooling_time_reconstruction": "cooling_time=gas_energy/cooling_rate_density",
    }


def block_reduce_sum(data: np.ndarray, new_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum non-overlapping index blocks in float64.

    No normalization is applied: this operation is intended for volumes and
    other extensive quantities.
    """

    values = np.asarray(data, dtype=np.float64)
    old_shape = np.asarray(values.shape, dtype=int)
    new_shape_array = np.asarray(new_shape, dtype=int)
    if len(old_shape) != len(new_shape_array):
        raise ValueError("Old and new shapes must have the same number of dimensions.")
    if np.any(new_shape_array <= 0) or not np.all(old_shape % new_shape_array == 0):
        raise ValueError(
            f"New shape {tuple(new_shape)} must divide source shape {tuple(old_shape)} exactly."
        )
    factors = (old_shape // new_shape_array).astype(int)
    reshaped = values.reshape(
        *[size for pair in zip(new_shape_array, factors) for size in pair]
    )
    reduction_axes = tuple(range(1, 2 * len(new_shape_array), 2))
    return reshaped.sum(axis=reduction_axes, dtype=np.float64)


def validate_input_fields(
    data: CoarseningInput,
    *,
    supported_fields: Collection[str] = SUPPORTED_COARSENING_FIELDS,
) -> tuple[int, ...]:
    """Validate the common field and geometry contract and return source shape."""

    if isinstance(data.ndim, bool) or int(data.ndim) not in (2, 3):
        raise ValueError("ndim must be either 2 or 3.")
    ndim = int(data.ndim)
    if data.coordinate_mode not in {"polar", "cartesian"}:
        raise ValueError(
            "coordinate_mode must be either 'polar' or 'cartesian', "
            f"got {data.coordinate_mode!r}."
        )
    if "density" not in data.snapshot or "gas_energy" not in data.snapshot:
        raise KeyError("Coarsening input requires both 'density' and 'gas_energy'.")

    unsupported = sorted(set(data.snapshot).difference(supported_fields))
    if unsupported:
        raise ValueError(
            "No conservative coarsening rule is defined for snapshot field(s): "
            f"{unsupported}. Define their physical semantics before preprocessing."
        )

    density = np.asarray(data.snapshot["density"], dtype=np.float64)
    fine_shape = tuple(int(value) for value in density.shape)
    if density.ndim != ndim:
        raise ValueError(
            f"Density has {density.ndim} dimensions, but input ndim is {ndim}."
        )
    if not np.all(np.isfinite(density)) or np.any(density < 0):
        raise ValueError("Density must contain only finite, non-negative values.")

    internal_energy_density = np.asarray(
        data.snapshot["gas_energy"], dtype=np.float64
    )
    if internal_energy_density.shape != fine_shape:
        raise ValueError(
            "Internal-energy-density shape does not match the density shape."
        )
    if not np.all(np.isfinite(internal_energy_density)) or np.any(
        internal_energy_density < 0
    ):
        raise ValueError(
            "Gas internal-energy density must contain only finite, non-negative values."
        )

    if "temperature" in data.snapshot:
        stored_temperature = np.asarray(data.snapshot["temperature"], dtype=np.float64)
        if stored_temperature.shape != fine_shape:
            raise ValueError("Temperature shape does not match the density shape.")
        expected_temperature = temperature_from_internal_energy_density(
            internal_energy_density, density
        )
        positive_density = density > 0
        if np.any(positive_density) and not np.allclose(
            stored_temperature[positive_density],
            expected_temperature[positive_density],
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                "Stored temperature violates the MACER relation "
                "T = 93 * gas_energy / density."
            )

    if COOLING_TIME_FIELD in data.snapshot:
        cooling_time = np.asarray(data.snapshot[COOLING_TIME_FIELD], dtype=np.float64)
        if cooling_time.shape != fine_shape:
            raise ValueError("Cooling-time shape does not match the density shape.")
        derived_rate = cooling_rate_density_from_time(
            internal_energy_density, cooling_time
        )
        if COOLING_RATE_DENSITY_FIELD in data.snapshot and not np.allclose(
            np.asarray(data.snapshot[COOLING_RATE_DENSITY_FIELD], dtype=np.float64),
            derived_rate,
            rtol=1e-6,
            atol=1e-12,
        ):
            raise ValueError(
                "Stored cooling_rate_density violates C = gas_energy/cooling_time."
            )

    volume = np.asarray(data.grid_volume, dtype=np.float64)
    if volume.shape != fine_shape:
        raise ValueError(
            f"Grid-volume shape {volume.shape} does not match snapshot shape {fine_shape}."
        )
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("Grid volumes must be finite and strictly positive.")

    for name, field in data.snapshot.items():
        values = np.asarray(field)
        if values.shape != fine_shape:
            raise ValueError(
                f"Snapshot field '{name}' has shape {values.shape}, expected {fine_shape}."
            )
        # Fine temperature is a derived field and is recomputed after the
        # remap.  Preserve the legacy behavior that does not use its values.
        if name != "temperature" and not np.all(np.isfinite(values)):
            raise ValueError(f"Snapshot field '{name}' contains NaN or infinity.")

    if len(data.axes) != ndim or len(data.axis_edges) != ndim:
        raise ValueError(
            f"Expected {ndim} coordinate axes and edge arrays, got "
            f"{len(data.axes)} axes and {len(data.axis_edges)} edge arrays."
        )
    for axis_index, (axis, edges) in enumerate(zip(data.axes, data.axis_edges)):
        axis_values = np.asarray(axis, dtype=np.float64)
        edge_values = np.asarray(edges, dtype=np.float64)
        if axis_values.ndim != 1 or axis_values.size < 1:
            raise ValueError(f"axis_{axis_index} must be a non-empty 1D array.")
        if not np.all(np.isfinite(axis_values)) or np.any(np.diff(axis_values) <= 0):
            raise ValueError(f"axis_{axis_index} must be finite and strictly increasing.")
        if edge_values.shape != (axis_values.size + 1,):
            raise ValueError(
                f"axis_{axis_index}_edges must have length len(axis_{axis_index}) + 1."
            )
        if not np.all(np.isfinite(edge_values)) or np.any(np.diff(edge_values) <= 0):
            raise ValueError(
                f"axis_{axis_index}_edges must be finite and strictly increasing."
            )

    # np.meshgrid's default ``xy`` convention swaps the first two axes, which
    # is exactly the established MACER2D (theta, r) snapshot ordering.
    expected_axis_lengths = tuple(reversed(fine_shape)) if ndim == 2 else None
    if expected_axis_lengths is not None:
        actual_axis_lengths = tuple(len(np.asarray(axis)) for axis in data.axes)
        if actual_axis_lengths != expected_axis_lengths:
            raise ValueError(
                "Coordinate-axis lengths are inconsistent with MACER2D array order: "
                f"got {actual_axis_lengths}, expected {expected_axis_lengths}."
            )
    return fine_shape


def build_regular_coarse_geometry(
    data: CoarseningInput,
    new_shape: Tuple[int, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build the exact regular block geometry used by the historical R_h path."""

    fine_shape = validate_input_fields(data)
    new_shape = tuple(int(value) for value in new_shape)
    if len(new_shape) != data.ndim:
        raise ValueError(
            f"New shape must have {data.ndim} dimensions, got {len(new_shape)}."
        )
    fine_shape_array = np.asarray(fine_shape, dtype=int)
    new_shape_array = np.asarray(new_shape, dtype=int)
    if np.any(new_shape_array <= 0) or not np.all(fine_shape_array % new_shape_array == 0):
        raise ValueError(
            f"New shape {new_shape} must divide source shape {fine_shape} exactly."
        )
    if data.ndim != 2:
        raise NotImplementedError(
            "Conservative coordinate coarsening is currently defined for MACER2D only."
        )

    coarse_volume = block_reduce_sum(data.grid_volume, new_shape)
    # Snapshot arrays are (theta, r), while axes are (r, theta).
    axis_new_shape = tuple(reversed(new_shape))
    new_axes: list[np.ndarray] = []
    new_axis_edges: list[np.ndarray] = []
    for axis_index, (old_axis, old_edges, axis_size) in enumerate(
        zip(data.axes, data.axis_edges, axis_new_shape)
    ):
        old_axis = np.asarray(old_axis, dtype=np.float64)
        old_edges = np.asarray(old_edges, dtype=np.float64)
        factor = len(old_axis) // axis_size
        new_edges = old_edges[::factor]
        if len(new_edges) != axis_size + 1:
            new_edges = np.concatenate((new_edges, old_edges[-1:]))
        if data.coordinate_mode == "polar" and axis_index == 0:
            if np.any(new_edges <= 0):
                raise ValueError("Radial edges must be positive for geometric-mean centres.")
            new_axis = np.sqrt(new_edges[:-1] * new_edges[1:])
        else:
            new_axis = 0.5 * (new_edges[1:] + new_edges[:-1])
        new_axes.append(new_axis)
        new_axis_edges.append(new_edges)
    return coarse_volume, tuple(new_axes), tuple(new_axis_edges)


def validate_coarsening_conservation(
    fine: CoarseningInput,
    coarse: CoarseningResult,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    """验证广延量守恒，并返回可持久化的逐量诊断报告。"""

    fine_volume = np.asarray(fine.grid_volume, dtype=np.float64)
    coarse_volume = np.asarray(coarse.grid_volume, dtype=np.float64)
    fine_mass = np.asarray(fine.snapshot["density"], dtype=np.float64) * fine_volume
    coarse_mass = np.asarray(coarse.snapshot["density"], dtype=np.float64) * coarse_volume

    fine_internal_energy = (
        np.asarray(fine.snapshot["gas_energy"], dtype=np.float64) * fine_volume
    )
    coarse_internal_energy = (
        np.asarray(coarse.snapshot["gas_energy"], dtype=np.float64) * coarse_volume
    )
    quantities = {
        "cell_volume": (
            fine_volume.sum(dtype=np.float64),
            coarse_volume.sum(dtype=np.float64),
            np.abs(fine_volume).sum(dtype=np.float64),
        ),
        "gas_mass": (
            fine_mass.sum(dtype=np.float64),
            coarse_mass.sum(dtype=np.float64),
            np.abs(fine_mass).sum(dtype=np.float64),
        ),
        "gas_internal_energy": (
            fine_internal_energy.sum(dtype=np.float64),
            coarse_internal_energy.sum(dtype=np.float64),
            np.abs(fine_internal_energy).sum(dtype=np.float64),
        ),
    }
    for name in ("v1", "v2", "v3"):
        if name in fine.snapshot:
            fine_component = np.asarray(fine.snapshot[name], dtype=np.float64) * fine_mass
            coarse_component = np.asarray(coarse.snapshot[name], dtype=np.float64) * coarse_mass
            quantities[f"{name}_momentum"] = (
                fine_component.sum(dtype=np.float64),
                coarse_component.sum(dtype=np.float64),
                np.abs(fine_component).sum(dtype=np.float64),
            )
    acceleration_fields = {"radial_acceleration", "theta_acceleration"}
    if acceleration_fields.issubset(fine.snapshot):
        fine_shape = fine_mass.shape
        coarse_shape = coarse_mass.shape
        fine_theta = np.broadcast_to(
            np.asarray(fine.axes[1], dtype=np.float64)[:, None], fine_shape
        )
        coarse_theta = np.broadcast_to(
            np.asarray(coarse.axes[1], dtype=np.float64)[:, None], coarse_shape
        )
        fine_a_r, fine_a_z = spherical_meridional_to_cylindrical(
            np.asarray(fine.snapshot["radial_acceleration"], dtype=np.float64),
            np.asarray(fine.snapshot["theta_acceleration"], dtype=np.float64),
            fine_theta,
        )
        coarse_a_r, coarse_a_z = spherical_meridional_to_cylindrical(
            np.asarray(coarse.snapshot["radial_acceleration"], dtype=np.float64),
            np.asarray(coarse.snapshot["theta_acceleration"], dtype=np.float64),
            coarse_theta,
        )
        for component, fine_values, coarse_values in (
            ("R", fine_a_r, coarse_a_r),
            ("z", fine_a_z, coarse_a_z),
        ):
            fine_force = fine_values * fine_mass
            coarse_force = coarse_values * coarse_mass
            quantities[f"gravitational_force_{component}"] = (
                fine_force.sum(dtype=np.float64),
                coarse_force.sum(dtype=np.float64),
                np.abs(fine_force).sum(dtype=np.float64),
            )
    if "initial_stellar_density" in fine.snapshot:
        fine_stellar_mass = (
            np.asarray(fine.snapshot["initial_stellar_density"], dtype=np.float64)
            * fine_volume
        )
        coarse_stellar_mass = (
            np.asarray(coarse.snapshot["initial_stellar_density"], dtype=np.float64)
            * coarse_volume
        )
        quantities["initial_stellar_mass"] = (
            fine_stellar_mass.sum(dtype=np.float64),
            coarse_stellar_mass.sum(dtype=np.float64),
            np.abs(fine_stellar_mass).sum(dtype=np.float64),
        )
    if COOLING_RATE_DENSITY_FIELD in fine.snapshot:
        fine_cooling_power = (
            np.asarray(fine.snapshot[COOLING_RATE_DENSITY_FIELD], dtype=np.float64)
            * fine_volume
        )
        coarse_cooling_power = (
            np.asarray(coarse.snapshot[COOLING_RATE_DENSITY_FIELD], dtype=np.float64)
            * coarse_volume
        )
        quantities["cooling_power"] = (
            fine_cooling_power.sum(dtype=np.float64),
            coarse_cooling_power.sum(dtype=np.float64),
            np.abs(fine_cooling_power).sum(dtype=np.float64),
        )

    failures = []
    report: dict[str, float] = {}
    maximum_scaled_relative_error = 0.0
    for name, (fine_total, coarse_total, reference_scale) in quantities.items():
        error = abs(coarse_total - fine_total)
        tolerance = atol + rtol * reference_scale
        scaled_relative_error = error / max(reference_scale, atol)
        maximum_scaled_relative_error = max(
            maximum_scaled_relative_error, scaled_relative_error
        )
        prefix = f"conservation_{name}"
        report[f"{prefix}_fine"] = float(fine_total)
        report[f"{prefix}_coarse"] = float(coarse_total)
        report[f"{prefix}_absolute_error"] = float(error)
        report[f"{prefix}_scaled_relative_error"] = float(scaled_relative_error)
        report[f"{prefix}_tolerance"] = float(tolerance)
        if error > tolerance:
            failures.append(
                f"{name}: fine={fine_total:.16g}, coarse={coarse_total:.16g}, "
                f"scaled_rel_error={scaled_relative_error:.3e}, tolerance={tolerance:.3e}"
            )
    if failures:
        raise ValueError("Conservative coarsening validation failed: " + "; ".join(failures))
    report["epsilon_conservation"] = float(maximum_scaled_relative_error)
    report["conservation_passed"] = 1.0
    return report
