"""Conservative regular spatial-block coarse graining (R_h)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from raw_data_processing.coarsening_common import (
        block_reduce_sum,
        build_regular_coarse_geometry,
        validate_coarsening_conservation,
        validate_input_fields,
    )
    from raw_data_processing.coarsening_types import CoarseningInput, CoarseningResult
except ModuleNotFoundError:  # Support ``python raw_data_processing/galaxy_data.py``.
    from coarsening_common import (
        block_reduce_sum,
        build_regular_coarse_geometry,
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
    """Apply the existing Data-Set-15-based R_h semantics to pure arrays.

    Volume densities are volume-integrated, velocity components are gas-mass
    weighted, and temperature is recomputed from coarse gas energy and density.
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

    fine_energy = np.asarray(data.snapshot["gas_energy"], dtype=np.float64) * fine_volume
    coarse_snapshot: dict[str, np.ndarray] = {
        "density": coarse_mass / coarse_volume,
        "gas_energy": block_reduce_sum(fine_energy, new_shape) / coarse_volume,
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

    coarse_snapshot["temperature"] = (
        93.0 * coarse_snapshot["gas_energy"] / coarse_snapshot["density"]
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
            "source_shape": fine_shape,
            "output_shape": new_shape,
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
