"""保守的常规空间块粗粒化，空间降采样"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from raw_data_processing.coarsening_common import (
        block_reduce_sum,
        build_regular_coarse_geometry,
        cooling_rate_density_from_time,
        cooling_time_from_rate_density,
        cylindrical_meridional_to_spherical,
        spherical_meridional_to_cylindrical,
        temperature_from_internal_energy_density,
        thermodynamic_coarsening_metadata,
        validate_coarsening_conservation,
        validate_input_fields,
    )
    from raw_data_processing.coarsening_types import CoarseningInput, CoarseningResult
except ModuleNotFoundError:  # Support ``python raw_data_processing/galaxy_data.py``.
    from coarsening_common import (
        block_reduce_sum,
        build_regular_coarse_geometry,
        cooling_rate_density_from_time,
        cooling_time_from_rate_density,
        cylindrical_meridional_to_spherical,
        spherical_meridional_to_cylindrical,
        temperature_from_internal_energy_density,
        thermodynamic_coarsening_metadata,
        validate_coarsening_conservation,
        validate_input_fields,
    )
    from coarsening_types import CoarseningInput, CoarseningResult


def coarsen_spatial_block(
    data: CoarseningInput,
    new_shape: Tuple[int, ...],
    *,
    conservation_rtol: float = 1e-6,
    conservation_atol: float = 1e-10,
) -> CoarseningResult:
    """在纯数组上应用基于 Data-Set-15 的既有 R_h 语义。

    密度场先乘体积再沉积，速度分量采用气体质量加权；温度由粗网格内能密度
    和气体密度重新计算。
    """

    fine_shape = validate_input_fields(data)
    new_shape = tuple(int(value) for value in new_shape)
    coarse_volume, coarse_axes, coarse_axis_edges = build_regular_coarse_geometry(
        data, new_shape
    )

    fine_volume = np.asarray(data.grid_volume, dtype=np.float64)
    fine_density = np.asarray(data.snapshot["density"], dtype=np.float64)
    fine_mass = fine_density * fine_volume
    coarse_mass = block_reduce_sum(fine_mass, new_shape)
    if np.any(coarse_mass <= 0):
        raise ValueError("Every coarse cell must contain strictly positive gas mass.")

    fine_internal_energy = (
        np.asarray(data.snapshot["gas_energy"], dtype=np.float64) * fine_volume
    )
    coarse_snapshot: dict[str, np.ndarray] = {
        "density": coarse_mass / coarse_volume,
        "gas_energy": (
            block_reduce_sum(fine_internal_energy, new_shape) / coarse_volume
        ),
    }
    for name in ("v1", "v2", "v3"):
        if name in data.snapshot:
            fine_component = np.asarray(data.snapshot[name], dtype=np.float64) * fine_mass
            coarse_component = block_reduce_sum(fine_component, new_shape)
            coarse_snapshot[name] = np.divide(
                coarse_component,
                coarse_mass,
                out=np.zeros_like(coarse_component),
                where=coarse_mass > 0,
            )
    if "initial_stellar_density" in data.snapshot:
        fine_stellar_mass = (
            np.asarray(data.snapshot["initial_stellar_density"], dtype=np.float64)
            * fine_volume
        )
        coarse_snapshot["initial_stellar_density"] = (
            block_reduce_sum(fine_stellar_mass, new_shape) / coarse_volume
        )

    if "cooling_time" in data.snapshot:
        fine_cooling_rate_density = np.asarray(
            data.snapshot.get(
                "cooling_rate_density",
                cooling_rate_density_from_time(
                    data.snapshot["gas_energy"], data.snapshot["cooling_time"]
                ),
            ),
            dtype=np.float64,
        )
        coarse_snapshot["cooling_rate_density"] = (
            block_reduce_sum(fine_cooling_rate_density * fine_volume, new_shape)
            / coarse_volume
        )
        coarse_snapshot["cooling_time"] = cooling_time_from_rate_density(
            coarse_snapshot["gas_energy"], coarse_snapshot["cooling_rate_density"]
        )

    acceleration_fields = {"radial_acceleration", "theta_acceleration"}
    if acceleration_fields.intersection(data.snapshot):
        if not acceleration_fields.issubset(data.snapshot):
            raise ValueError(
                "radial_acceleration and theta_acceleration must be coarsened together."
            )
        fine_theta = np.broadcast_to(
            np.asarray(data.axes[1], dtype=np.float64)[:, None], fine_shape
        )
        acceleration_r, acceleration_z = spherical_meridional_to_cylindrical(
            np.asarray(data.snapshot["radial_acceleration"], dtype=np.float64),
            np.asarray(data.snapshot["theta_acceleration"], dtype=np.float64),
            fine_theta,
        )
        coarse_acceleration_r = (
            block_reduce_sum(acceleration_r * fine_mass, new_shape) / coarse_mass
        )
        coarse_acceleration_z = (
            block_reduce_sum(acceleration_z * fine_mass, new_shape) / coarse_mass
        )
        coarse_theta = np.broadcast_to(
            np.asarray(coarse_axes[1], dtype=np.float64)[:, None], new_shape
        )
        (
            coarse_snapshot["radial_acceleration"],
            coarse_snapshot["theta_acceleration"],
        ) = cylindrical_meridional_to_spherical(
            coarse_acceleration_r, coarse_acceleration_z, coarse_theta
        )

    coarse_snapshot["temperature"] = temperature_from_internal_energy_density(
        coarse_snapshot["gas_energy"], coarse_snapshot["density"]
    )
    for name, values in coarse_snapshot.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Coarsened field '{name}' contains NaN or infinity.")

    result = CoarseningResult(
        snapshot=coarse_snapshot,
        grid_volume=coarse_volume,
        axes=coarse_axes,
        axis_edges=coarse_axis_edges,
        volume_source="Data-Set-15:block_sum",
        metadata={
            "operator": "spatial_block",
            "operator_version": 2,
            "source_shape": fine_shape,
            "output_shape": new_shape,
            "operator_parameters": {
                "source_shape": fine_shape,
                "output_shape": new_shape,
                "aggregation": "non_overlapping_index_blocks",
                "volume_source": "Data-Set-15:block_sum",
                "cooling_aggregation": "conserve_cooling_power",
                "acceleration_aggregation": "mass_weighted_in_fixed_Rz_basis",
            },
            **thermodynamic_coarsening_metadata(),
        },
        diagnostics={},
    )
    validate_coarsening_conservation(
        data,
        result,
        rtol=conservation_rtol,
        atol=conservation_atol,
    )
    return result
