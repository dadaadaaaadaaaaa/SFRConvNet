"""Shared validation, geometry, reduction, and conservation utilities."""

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
    }
)


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
) -> None:
    """Validate the same stored extensive quantities as the legacy GalData path."""

    fine_volume = np.asarray(fine.grid_volume, dtype=np.float64)
    coarse_volume = np.asarray(coarse.grid_volume, dtype=np.float64)
    fine_mass = np.asarray(fine.snapshot["density"], dtype=np.float64) * fine_volume
    coarse_mass = np.asarray(coarse.snapshot["density"], dtype=np.float64) * coarse_volume

    fine_energy = np.asarray(fine.snapshot["gas_energy"], dtype=np.float64) * fine_volume
    coarse_energy = np.asarray(coarse.snapshot["gas_energy"], dtype=np.float64) * coarse_volume
    quantities = {
        "cell volume": (
            fine_volume.sum(dtype=np.float64),
            coarse_volume.sum(dtype=np.float64),
            np.abs(fine_volume).sum(dtype=np.float64),
        ),
        "gas mass": (
            fine_mass.sum(dtype=np.float64),
            coarse_mass.sum(dtype=np.float64),
            np.abs(fine_mass).sum(dtype=np.float64),
        ),
        "gas energy": (
            fine_energy.sum(dtype=np.float64),
            coarse_energy.sum(dtype=np.float64),
            np.abs(fine_energy).sum(dtype=np.float64),
        ),
    }
    for name in ("v1", "v2", "v3"):
        if name in fine.snapshot:
            fine_component = np.asarray(fine.snapshot[name], dtype=np.float64) * fine_mass
            coarse_component = np.asarray(coarse.snapshot[name], dtype=np.float64) * coarse_mass
            quantities[f"{name} momentum"] = (
                fine_component.sum(dtype=np.float64),
                coarse_component.sum(dtype=np.float64),
                np.abs(fine_component).sum(dtype=np.float64),
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
        quantities["initial stellar mass"] = (
            fine_stellar_mass.sum(dtype=np.float64),
            coarse_stellar_mass.sum(dtype=np.float64),
            np.abs(fine_stellar_mass).sum(dtype=np.float64),
        )

    failures = []
    for name, (fine_total, coarse_total, reference_scale) in quantities.items():
        error = abs(coarse_total - fine_total)
        tolerance = atol + rtol * reference_scale
        if error > tolerance:
            relative_error = error / max(reference_scale, atol)
            failures.append(
                f"{name}: fine={fine_total:.16g}, coarse={coarse_total:.16g}, "
                f"scaled_rel_error={relative_error:.3e}, tolerance={tolerance:.3e}"
            )
    if failures:
        raise ValueError("Conservative coarsening validation failed: " + "; ".join(failures))
