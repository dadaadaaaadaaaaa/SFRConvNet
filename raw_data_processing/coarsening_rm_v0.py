"""Axisymmetric fixed-mass-support coarse graining, version 0.

This module deliberately contains only the numerical operator.  HDF I/O,
target handling, and construction of ``GalData`` objects remain in
``galaxy_data.py``.  The method is a fixed-mass *support filter* followed by a
balanced conservative projection to a regular output grid; it is not an
exclusive equal-mass tessellation.
"""

from __future__ import annotations

from time import perf_counter
import warnings

import numpy as np

try:
    from raw_data_processing.coarsening_common import build_regular_coarse_geometry
    from raw_data_processing.coarsening_types import CoarseningInput, CoarseningResult
except ModuleNotFoundError:  # Support direct execution of galaxy_data.py.
    from coarsening_common import build_regular_coarse_geometry
    from coarsening_types import CoarseningInput, CoarseningResult


SUPPORTED_FIELDS = {
    "density",
    "gas_energy",
    "temperature",
    "v1",
    "v2",
    "v3",
    "initial_stellar_density",
}


def mass_adaptive_kernel_profile(q: np.ndarray, *, kernel: str) -> np.ndarray:
    """Return the v0 compact-support radial profile for ``0 <= q < 1``."""
    q = np.asarray(q, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("Kernel radii must be finite and non-negative.")
    if kernel == "top_hat":
        return (q < 1.0).astype(np.float64)
    if kernel != "cubic_spline":
        raise ValueError(f"Unsupported mass-adaptive kernel {kernel!r}.")
    values = np.zeros_like(q)
    inner = q < 0.5
    outer = (q >= 0.5) & (q < 1.0)
    values[inner] = 1.0 - 6.0 * q[inner] ** 2 + 6.0 * q[inner] ** 3
    values[outer] = 2.0 * (1.0 - q[outer]) ** 3
    return values


def balance_sparse_transport(
    kernel_matrix,
    *,
    row_marginal: np.ndarray,
    column_marginal: np.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_row_scale: np.ndarray | None = None,
    initial_column_scale: np.ndarray | None = None,
):
    """Balance a positive sparse kernel to prescribed volume marginals.

    This is the legacy IPFP/Sinkhorn implementation used by Rm-v0.  Optional
    initial scales are accepted so the same routine can be regression-tested
    against accelerated variants without changing the v0 default.
    """
    matrix = kernel_matrix.tocsr(copy=True)
    matrix.eliminate_zeros()
    if matrix.nnz == 0 or np.any(matrix.data <= 0) or np.any(~np.isfinite(matrix.data)):
        raise ValueError("The adaptive transport kernel must have finite positive entries.")
    row_marginal = np.asarray(row_marginal, dtype=np.float64).reshape(-1)
    column_marginal = np.asarray(column_marginal, dtype=np.float64).reshape(-1)
    if matrix.shape != (row_marginal.size, column_marginal.size):
        raise ValueError("Transport matrix shape does not match its requested marginals.")
    if (
        np.any(row_marginal <= 0)
        or np.any(column_marginal <= 0)
        or np.any(~np.isfinite(row_marginal))
        or np.any(~np.isfinite(column_marginal))
    ):
        raise ValueError("Transport marginals must be finite and strictly positive.")
    total_scale = max(row_marginal.sum(dtype=np.float64), column_marginal.sum(dtype=np.float64))
    if abs(row_marginal.sum() - column_marginal.sum()) > 1e-12 * total_scale:
        raise ValueError("Transport row and column marginals have different total volume.")
    if np.any(np.asarray(matrix.sum(axis=1)).reshape(-1) <= 0):
        raise ValueError("At least one output cell has no adaptive-kernel support.")
    if np.any(np.asarray(matrix.sum(axis=0)).reshape(-1) <= 0):
        raise ValueError("At least one source cell has no adaptive-kernel destination.")

    row_scale = (
        np.ones(matrix.shape[0], dtype=np.float64)
        if initial_row_scale is None
        else np.asarray(initial_row_scale, dtype=np.float64).reshape(-1).copy()
    )
    column_scale = (
        np.ones(matrix.shape[1], dtype=np.float64)
        if initial_column_scale is None
        else np.asarray(initial_column_scale, dtype=np.float64).reshape(-1).copy()
    )
    if row_scale.shape != (matrix.shape[0],) or column_scale.shape != (matrix.shape[1],):
        raise ValueError("Initial transport scales have incompatible shapes.")
    if (
        np.any(row_scale <= 0)
        or np.any(column_scale <= 0)
        or np.any(~np.isfinite(row_scale))
        or np.any(~np.isfinite(column_scale))
    ):
        raise ValueError("Initial transport scales must be finite and positive.")

    balance_error = np.inf
    for iteration in range(1, max_iterations + 1):
        row_denominator = np.asarray(matrix @ column_scale).reshape(-1)
        if np.any(row_denominator <= 0) or np.any(~np.isfinite(row_denominator)):
            raise RuntimeError("Sparse transport row scaling became singular.")
        row_scale = row_marginal / row_denominator

        column_denominator = np.asarray(matrix.T @ row_scale).reshape(-1)
        if np.any(column_denominator <= 0) or np.any(~np.isfinite(column_denominator)):
            raise RuntimeError("Sparse transport column scaling became singular.")
        column_scale = column_marginal / column_denominator

        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            current_rows = row_scale * np.asarray(matrix @ column_scale).reshape(-1)
            current_columns = column_scale * np.asarray(matrix.T @ row_scale).reshape(-1)
            row_error = np.max(np.abs(current_rows - row_marginal) / row_marginal)
            column_error = np.max(
                np.abs(current_columns - column_marginal) / column_marginal
            )
            balance_error = float(max(row_error, column_error))
            if balance_error <= tolerance:
                break
    else:
        iteration = max_iterations

    if not np.isfinite(balance_error) or balance_error > tolerance:
        raise RuntimeError(
            "Adaptive transport balancing did not converge: "
            f"error={balance_error:.3e}, tolerance={tolerance:.3e}, "
            f"iterations={iteration}. Increase kernel support or max_iterations."
        )
    transport = matrix.multiply(row_scale[:, None]).multiply(column_scale[None, :]).tocsr()
    transport.eliminate_zeros()
    return transport, balance_error, iteration, row_scale, column_scale


def _transport_diagnostics(
    transport,
    fractions,
    *,
    fine_to_coarse: np.ndarray,
    fine_mass: np.ndarray,
    coarse_mass: np.ndarray,
    support_scale: np.ndarray,
    fine_shape: tuple[int, int],
    new_shape: tuple[int, int],
):
    """Return compact diagnostics without persisting the full transport plan."""
    from scipy.sparse import coo_matrix

    n_fine = fine_mass.size
    n_coarse = int(np.prod(new_shape))
    dominant_destination = np.asarray(fractions.argmax(axis=0)).reshape(-1).astype(np.int32)
    dominant_fraction = np.asarray(fractions.max(axis=0).toarray()).reshape(-1)
    entropy_terms = fractions.copy()
    entropy_terms.data = -entropy_terms.data * np.log(
        np.clip(entropy_terms.data, np.finfo(np.float64).tiny, 1.0)
    )
    transport_entropy = np.asarray(entropy_terms.sum(axis=0)).reshape(-1)

    native_assignment = coo_matrix(
        (
            np.ones(n_fine, dtype=np.float64),
            (np.arange(n_fine, dtype=np.int64), fine_to_coarse),
        ),
        shape=(n_fine, n_coarse),
    ).tocsr()
    volume_flow = np.asarray((transport @ native_assignment).toarray())
    mass_contribution = fractions.multiply(fine_mass[None, :]).tocsr()
    mass_flow = np.asarray((mass_contribution @ native_assignment).toarray())
    effective_support = np.divide(
        np.asarray(mass_contribution @ support_scale).reshape(new_shape),
        coarse_mass,
        out=np.zeros(new_shape, dtype=np.float64),
        where=coarse_mass > 0,
    )
    squared_contribution_sum = np.asarray(
        fractions.power(2) @ np.square(fine_mass)
    ).reshape(new_shape)
    effective_source_count = np.divide(
        np.square(coarse_mass),
        squared_contribution_sum,
        out=np.zeros(new_shape, dtype=np.float64),
        where=squared_contribution_sum > 0,
    )
    return {
        "dominant_destination": dominant_destination.reshape(fine_shape),
        "dominant_fraction": dominant_fraction.reshape(fine_shape),
        "source_transport_entropy": transport_entropy.reshape(fine_shape),
        "transport_volume_by_destination_native": volume_flow,
        "transport_mass_by_destination_native": mass_flow,
        "effective_support_scale": effective_support,
        "effective_source_count": effective_source_count,
        "output_gas_mass": coarse_mass,
    }


def coarsen_mass_v0(
    source: CoarseningInput,
    new_shape: tuple[int, int],
    *,
    mass_filter: float,
    kernel: str = "top_hat",
    initial_neighbors: int = 32,
    balance_tolerance: float = 1e-6,
    balance_max_iterations: int = 5000,
) -> CoarseningResult:
    """Apply the legacy axisymmetric fixed-mass-support operator."""
    started = perf_counter()
    if source.ndim != 2 or source.coordinate_mode != "polar":
        raise NotImplementedError("Rm-v0 is defined only for axisymmetric MACER2D polar data.")
    if len(new_shape) != 2:
        raise ValueError(f"new_shape must have two dimensions, got {new_shape!r}.")
    new_shape = tuple(int(value) for value in new_shape)
    mass_filter = float(mass_filter)
    if not np.isfinite(mass_filter) or mass_filter <= 0:
        raise ValueError("mass_filter must be positive and finite.")
    if isinstance(initial_neighbors, bool) or int(initial_neighbors) < 2:
        raise ValueError("initial_neighbors must be an integer >= 2.")
    initial_neighbors = int(initial_neighbors)
    if balance_tolerance <= 0 or not np.isfinite(balance_tolerance):
        raise ValueError("balance_tolerance must be positive and finite.")
    if isinstance(balance_max_iterations, bool) or int(balance_max_iterations) <= 0:
        raise ValueError("balance_max_iterations must be a positive integer.")
    balance_max_iterations = int(balance_max_iterations)
    kernel = str(kernel).lower().replace("-", "_")
    if kernel not in {"top_hat", "cubic_spline"}:
        raise ValueError("kernel must be either 'top_hat' or 'cubic_spline'.")

    snapshot = source.snapshot
    fine_shape = tuple(int(value) for value in np.asarray(snapshot["density"]).shape)
    fine_shape_array = np.asarray(fine_shape, dtype=int)
    new_shape_array = np.asarray(new_shape, dtype=int)
    if np.any(new_shape_array <= 0) or not np.all(fine_shape_array % new_shape_array == 0):
        raise ValueError(f"new_shape {new_shape} must divide source shape {fine_shape} exactly.")
    unsupported = sorted(set(snapshot).difference(SUPPORTED_FIELDS))
    if unsupported:
        raise ValueError(
            "No conservative Rm-v0 rule is defined for snapshot field(s): "
            f"{unsupported}. Define their extensive semantics before preprocessing."
        )
    fine_volume = np.asarray(source.grid_volume, dtype=np.float64)
    fine_density = np.asarray(snapshot["density"], dtype=np.float64)
    if fine_volume.shape != fine_shape or np.any(fine_volume <= 0) or not np.all(np.isfinite(fine_volume)):
        raise ValueError("Fine-cell volumes must match density and be finite and positive.")
    if np.any(fine_density < 0) or not np.all(np.isfinite(fine_density)):
        raise ValueError("Density must contain only finite, non-negative values.")
    fine_mass = fine_density * fine_volume
    total_mass = fine_mass.sum(dtype=np.float64)
    if total_mass <= 0:
        raise ValueError("The source snapshot has no positive gas mass.")
    if mass_filter > total_mass * (1.0 + 1e-12):
        raise ValueError(
            f"mass_filter={mass_filter:.16g} exceeds total gas mass={total_mass:.16g}."
        )
    mass_filter = min(mass_filter, total_mass)

    coarse_volume, new_axes, new_axis_edges = build_regular_coarse_geometry(source, new_shape)
    r_grid, theta_grid = np.meshgrid(*source.axes)
    points = np.column_stack(
        ((r_grid * np.sin(theta_grid)).reshape(-1), (r_grid * np.cos(theta_grid)).reshape(-1))
    )
    fine_volume_flat = fine_volume.reshape(-1)
    fine_mass_flat = fine_mass.reshape(-1)
    n_fine = fine_mass_flat.size
    n_coarse = int(np.prod(new_shape))
    theta_factor, radial_factor = (fine_shape_array // new_shape_array).astype(int)
    theta_index, radial_index = np.indices(fine_shape)
    fine_to_coarse = (
        (theta_index // theta_factor) * new_shape[1] + radial_index // radial_factor
    ).reshape(-1)

    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError("Rm-v0 requires scipy.sparse.") from exc
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
    except (ImportError, AttributeError):
        tree = None
        warnings.warn(
            "scipy.spatial.cKDTree is unavailable; Rm-v0 is using a slow NumPy all-pairs "
            "neighbor search.",
            RuntimeWarning,
        )

    neighbor_started = perf_counter()
    support_scale = np.empty(n_fine, dtype=np.float64)
    enclosed_mass = np.empty(n_fine, dtype=np.float64)
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_values: list[float] = []
    characteristic_scale = float(np.cbrt(fine_volume_flat[0])) if n_fine == 1 else None

    for source_index in range(n_fine):
        if n_fine == 1:
            distances = np.asarray([0.0])
            neighbors = np.asarray([0], dtype=np.int64)
            selected_count = 1
            support = characteristic_scale
        elif tree is None:
            distances = np.linalg.norm(points - points[source_index], axis=1)
            neighbors = np.argsort(distances, kind="stable")
            distances = distances[neighbors]
            cumulative_mass = np.cumsum(fine_mass_flat[neighbors], dtype=np.float64)
            reached = np.flatnonzero(cumulative_mass >= mass_filter)
            if not reached.size:
                raise RuntimeError("Could not enclose mass_filter in the full domain.")
            threshold_index = int(reached[0])
            threshold_distance = distances[threshold_index]
            next_distinct = np.flatnonzero(distances > threshold_distance)
            next_distinct = next_distinct[next_distinct > threshold_index]
            if next_distinct.size:
                selected_count = int(next_distinct[0])
                support = 0.5 * (threshold_distance + distances[selected_count])
            else:
                selected_count = n_fine
                support = np.nextafter(float(distances[-1]), np.inf)
        else:
            query_count = min(max(initial_neighbors, 2), n_fine)
            while True:
                distances, neighbors = tree.query(points[source_index], k=query_count)
                distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
                neighbors = np.atleast_1d(np.asarray(neighbors, dtype=np.int64))
                cumulative_mass = np.cumsum(fine_mass_flat[neighbors], dtype=np.float64)
                reached = np.flatnonzero(cumulative_mass >= mass_filter)
                if reached.size:
                    threshold_index = int(reached[0])
                    threshold_distance = distances[threshold_index]
                    next_distinct = np.flatnonzero(distances > threshold_distance)
                    next_distinct = next_distinct[next_distinct > threshold_index]
                    if next_distinct.size:
                        selected_count = int(next_distinct[0])
                        support = 0.5 * (threshold_distance + distances[selected_count])
                        break
                    if query_count == n_fine:
                        selected_count = n_fine
                        support = np.nextafter(float(distances[-1]), np.inf)
                        break
                if query_count == n_fine:
                    raise RuntimeError("Could not enclose mass_filter in the full domain.")
                query_count = min(2 * query_count, n_fine)

        selected_neighbors = neighbors[:selected_count]
        selected_distances = distances[:selected_count]
        if not np.isfinite(support) or support <= 0:
            positive_distances = distances[distances > 0]
            support = (
                0.5 * float(positive_distances[0])
                if positive_distances.size
                else float(np.cbrt(fine_volume_flat[source_index]))
            )
        support_scale[source_index] = support
        enclosed_mass[source_index] = fine_mass_flat[selected_neighbors].sum(dtype=np.float64)
        profile = mass_adaptive_kernel_profile(selected_distances / support, kernel=kernel)
        quadrature_weights = profile * fine_volume_flat[selected_neighbors]
        if not np.any(quadrature_weights > 0):
            raise RuntimeError(f"Adaptive kernel for source {source_index} has zero weight.")
        destination_ids = fine_to_coarse[selected_neighbors]
        unique_destinations, inverse = np.unique(destination_ids, return_inverse=True)
        destination_weights = np.bincount(
            inverse, weights=quadrature_weights, minlength=len(unique_destinations)
        )
        positive = destination_weights > 0
        matrix_rows.extend(unique_destinations[positive].tolist())
        matrix_cols.extend(np.full(np.count_nonzero(positive), source_index).tolist())
        matrix_values.extend(destination_weights[positive].tolist())
    neighbor_seconds = perf_counter() - neighbor_started

    raw_kernel = coo_matrix(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(n_coarse, n_fine),
        dtype=np.float64,
    ).tocsr()
    balance_started = perf_counter()
    transport, balance_error, balance_iterations, _, _ = balance_sparse_transport(
        raw_kernel,
        row_marginal=coarse_volume.reshape(-1),
        column_marginal=fine_volume_flat,
        tolerance=balance_tolerance,
        max_iterations=balance_max_iterations,
    )
    balance_seconds = perf_counter() - balance_started
    fractions = transport.multiply((1.0 / fine_volume_flat)[None, :]).tocsr()

    def deposit_extensive(values: np.ndarray) -> np.ndarray:
        return np.asarray(fractions @ np.asarray(values, dtype=np.float64).reshape(-1)).reshape(new_shape)

    coarse_mass = deposit_extensive(fine_mass)
    if np.any(coarse_mass <= 0):
        raise ValueError("Every Rm-v0 output cell must contain positive gas mass.")
    coarse_energy = deposit_extensive(np.asarray(snapshot["gas_energy"]) * fine_volume)
    new_snapshot = {
        "density": coarse_mass / coarse_volume,
        "gas_energy": coarse_energy / coarse_volume,
    }
    for name in ("v1", "v2", "v3"):
        if name in snapshot:
            new_snapshot[name] = deposit_extensive(np.asarray(snapshot[name]) * fine_mass) / coarse_mass
    if "initial_stellar_density" in snapshot:
        new_snapshot["initial_stellar_density"] = deposit_extensive(
            np.asarray(snapshot["initial_stellar_density"]) * fine_volume
        ) / coarse_volume
    new_snapshot["temperature"] = 93 * new_snapshot["gas_energy"] / new_snapshot["density"]
    for name, values in new_snapshot.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Rm-v0 field {name!r} contains NaN or infinity.")

    diagnostics = {
        "source_support_scale": support_scale.reshape(fine_shape),
        "source_enclosed_mass": enclosed_mass.reshape(fine_shape),
    }
    diagnostics.update(
        _transport_diagnostics(
            transport,
            fractions,
            fine_to_coarse=fine_to_coarse,
            fine_mass=fine_mass_flat,
            coarse_mass=coarse_mass,
            support_scale=support_scale,
            fine_shape=fine_shape,
            new_shape=new_shape,
        )
    )
    total_seconds = perf_counter() - started
    metadata = {
        "operator": "adaptive_mass_axisymmetric_v0",
        "operator_version": 0,
        "source_shape": fine_shape,
        "output_shape": new_shape,
        "mass_filter": mass_filter,
        "kernel": kernel,
        "distance_metric": "meridional_euclidean_Rz",
        "mass_geometry": "full_axisymmetric_ring",
        "transport_balance_tolerance": float(balance_tolerance),
        "transport_balance_error": float(balance_error),
        "transport_balance_iterations": int(balance_iterations),
        "kernel_nnz": int(raw_kernel.nnz),
        "runtime_neighbor_kernel_seconds": float(neighbor_seconds),
        "runtime_balance_seconds": float(balance_seconds),
        "runtime_total_seconds": float(total_seconds),
        "partition_semantics": "overlapping_support_plus_soft_transport",
    }
    return CoarseningResult(
        snapshot=new_snapshot,
        grid_volume=coarse_volume,
        axes=new_axes,
        axis_edges=new_axis_edges,
        volume_source="Data-Set-15:block_sum",
        metadata=metadata,
        diagnostics=diagnostics,
        runtime={
            "neighbor_kernel_seconds": float(neighbor_seconds),
            "balance_seconds": float(balance_seconds),
            "total_seconds": float(total_seconds),
        },
    )
