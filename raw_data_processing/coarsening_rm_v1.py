"""Accelerated axisymmetric fixed-mass-support coarse graining, version 1.

The operator remains an overlapping adaptive support followed by a soft,
conservative transport to a regular output grid.  It is *not* an exclusive
equal-mass tessellation.  Compared with :mod:`coarsening_rm_v0`, this module
adds a reusable geometry-neighbour cache, a smooth Wendland C2 kernel,
structural transport-graph auditing, and a low-dimensional dual solver.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
import numpy as np

try:
    from raw_data_processing.coarsening_common import build_regular_coarse_geometry
    from raw_data_processing.coarsening_rm_v0 import (
        _transport_diagnostics,
    )
    from raw_data_processing.coarsening_types import CoarseningInput, CoarseningResult
except ModuleNotFoundError:  # Support direct execution from raw_data_processing.
    from coarsening_common import build_regular_coarse_geometry
    from coarsening_rm_v0 import _transport_diagnostics
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


class CoarseningTimeout(TimeoutError):
    """Raised when an Rm-v1 safety wall-clock limit is exceeded."""


@dataclass
class _GeometryKNN:
    points: np.ndarray
    tree: object | None
    distances: np.ndarray | None = None
    indices: np.ndarray | None = None
    k: int = 0
    backend: str = ""


_KNN_CACHE: "OrderedDict[str, _GeometryKNN]" = OrderedDict()
_MAX_CACHED_GEOMETRIES = 4


def clear_geometry_neighbor_cache() -> None:
    """Clear the process-local geometry KNN cache (primarily for tests)."""

    _KNN_CACHE.clear()


def wendland_c2_profile(q: np.ndarray) -> np.ndarray:
    """Unnormalised compact Wendland C2 profile with support ``q < 1``."""

    q = np.asarray(q, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("Kernel radii must be finite and non-negative.")
    values = np.zeros_like(q)
    inside = q < 1.0
    one_minus_q = 1.0 - q[inside]
    values[inside] = one_minus_q**4 * (1.0 + 4.0 * q[inside])
    return values


def _geometry_key(points: np.ndarray) -> str:
    values = np.ascontiguousarray(points, dtype=np.float64)
    digest = sha256()
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _check_deadline(deadline: float, stage: str) -> None:
    if perf_counter() > deadline:
        raise CoarseningTimeout(
            f"axisymmetric_rm_v1 exceeded max_runtime_seconds during {stage}."
        )


def _get_cache_entry(points: np.ndarray) -> tuple[_GeometryKNN, bool]:
    key = _geometry_key(points)
    if key in _KNN_CACHE:
        entry = _KNN_CACHE.pop(key)
        _KNN_CACHE[key] = entry
        return entry, True
    points = np.ascontiguousarray(points, dtype=np.float64)
    tree = None
    backend = "numpy_all_pairs"
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        backend = "scipy_ckdtree_vectorized"
    except (ImportError, AttributeError):
        if len(points) > 512:
            raise ImportError(
                "Rm-v1 requires a working scipy.spatial.cKDTree for production-sized "
                "snapshots. The NumPy fallback is intentionally limited to 512 cells."
            )
    entry = _GeometryKNN(points=points, tree=tree, backend=backend)
    _KNN_CACHE[key] = entry
    while len(_KNN_CACHE) > _MAX_CACHED_GEOMETRIES:
        _KNN_CACHE.popitem(last=False)
    return entry, False


def _ensure_knn(
    entry: _GeometryKNN,
    k: int,
    *,
    workers: int,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted KNN arrays, enlarging the process-local cache if needed."""

    n_points = len(entry.points)
    k = min(max(int(k), 1), n_points)
    if entry.k >= k and entry.distances is not None and entry.indices is not None:
        return entry.distances[:, :k], entry.indices[:, :k]
    _check_deadline(deadline, "KNN cache expansion")
    if entry.tree is not None:
        distances, indices = entry.tree.query(entry.points, k=k, workers=int(workers))
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
    else:
        delta = entry.points[:, None, :] - entry.points[None, :, :]
        full_distances = np.linalg.norm(delta, axis=2)
        indices = np.argsort(full_distances, axis=1, kind="stable")[:, :k]
        distances = np.take_along_axis(full_distances, indices, axis=1)
    entry.distances = np.ascontiguousarray(distances, dtype=np.float64)
    entry.indices = np.ascontiguousarray(indices, dtype=np.int64)
    entry.k = k
    _check_deadline(deadline, "KNN cache expansion")
    return entry.distances, entry.indices


def _find_mass_supports(
    entry: _GeometryKNN,
    fine_mass: np.ndarray,
    mass_filter: float,
    *,
    initial_neighbors: int,
    max_neighbors: int,
    max_expansions: int,
    workers: int,
    deadline: float,
    single_cell_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], int]:
    """Find Hm for every source using vectorized cached KNN queries."""

    n_fine = fine_mass.size
    if n_fine == 1:
        support = np.asarray([float(single_cell_scale)])
        return support, fine_mass.copy(), [np.zeros(1)], [np.zeros(1, dtype=np.int64)], 0

    query_count = min(max(initial_neighbors, 2), max_neighbors, n_fine)
    initial_distances, initial_neighbors_array = _ensure_knn(
        entry, query_count, workers=workers, deadline=deadline
    )
    distance_rows = [initial_distances[index].copy() for index in range(n_fine)]
    neighbor_rows = [initial_neighbors_array[index].copy() for index in range(n_fine)]
    expansions = 0
    while True:
        unresolved_indices: list[int] = []
        first = np.full(n_fine, -1, dtype=np.int64)
        next_indices = np.full(n_fine, -1, dtype=np.int64)
        for source_index, (distances, neighbors) in enumerate(
            zip(distance_rows, neighbor_rows)
        ):
            cumulative = np.cumsum(fine_mass[neighbors], dtype=np.float64)
            reached = np.flatnonzero(cumulative >= mass_filter)
            if not reached.size:
                unresolved_indices.append(source_index)
                continue
            threshold_index = int(reached[0])
            first[source_index] = threshold_index
            threshold_distance = distances[threshold_index]
            candidates = np.flatnonzero(
                distances[threshold_index + 1 :] > threshold_distance
            )
            if candidates.size:
                next_indices[source_index] = threshold_index + 1 + int(candidates[0])
            elif len(distances) < n_fine:
                unresolved_indices.append(source_index)

        if not unresolved_indices:
            break
        current_max = max(len(distance_rows[index]) for index in unresolved_indices)
        if current_max >= min(max_neighbors, n_fine):
            raise RuntimeError(
                "KNN support search reached max_neighbors before every source enclosed "
                "mass_filter with a complete distance shell. Increase max_neighbors or "
                "reduce mass_filter."
            )
        if expansions >= max_expansions:
            raise RuntimeError(
                "KNN support search reached max_neighbor_expansions before convergence."
            )
        query_count = min(2 * current_max, max_neighbors, n_fine)
        unresolved_array = np.asarray(unresolved_indices, dtype=np.int64)
        _check_deadline(deadline, "mass-support KNN tail query")
        if entry.tree is not None:
            queried_distances, queried_neighbors = entry.tree.query(
                entry.points[unresolved_array], k=query_count, workers=int(workers)
            )
            queried_distances = np.atleast_2d(np.asarray(queried_distances, dtype=np.float64))
            queried_neighbors = np.atleast_2d(np.asarray(queried_neighbors, dtype=np.int64))
        else:
            full_distances, full_neighbors = _ensure_knn(
                entry, n_fine, workers=workers, deadline=deadline
            )
            queried_distances = full_distances[unresolved_array, :query_count]
            queried_neighbors = full_neighbors[unresolved_array, :query_count]
        for row_index, source_index in enumerate(unresolved_indices):
            distance_rows[source_index] = queried_distances[row_index].copy()
            neighbor_rows[source_index] = queried_neighbors[row_index].copy()
        expansions += 1
        _check_deadline(deadline, "mass-support search")

    support = np.empty(n_fine, dtype=np.float64)
    enclosed_mass = np.empty(n_fine, dtype=np.float64)
    for source_index in range(n_fine):
        distances = distance_rows[source_index]
        neighbors = neighbor_rows[source_index]
        threshold_distance = distances[first[source_index]]
        next_index = int(next_indices[source_index])
        if next_index >= 0:
            support[source_index] = 0.5 * (
                threshold_distance + distances[next_index]
            )
        else:
            support[source_index] = np.nextafter(
                float(distances[-1]), np.inf
            )
        if support[source_index] <= 0 or not np.isfinite(support[source_index]):
            positive = distances[distances > 0]
            support[source_index] = (
                0.5 * float(positive[0]) if positive.size else np.finfo(float).eps
            )
        selected = distances < support[source_index]
        enclosed_mass[source_index] = fine_mass[neighbors[selected]].sum(
            dtype=np.float64
        )
    return support, enclosed_mass, distance_rows, neighbor_rows, expansions


def _ensure_graph_neighbors(
    entry: _GeometryKNN,
    graph_support: np.ndarray,
    *,
    distance_rows: list[np.ndarray],
    neighbor_rows: list[np.ndarray],
    max_neighbors: int,
    max_expansions: int,
    workers: int,
    deadline: float,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Ensure every cached row extends beyond its graph support radius."""

    n_fine = len(graph_support)
    expansions = 0
    while True:
        unresolved = [
            index
            for index, distances in enumerate(distance_rows)
            if len(distances) < n_fine and distances[-1] <= graph_support[index]
        ]
        if not unresolved:
            return distance_rows, neighbor_rows, expansions
        current_max = max(len(distance_rows[index]) for index in unresolved)
        if current_max >= min(max_neighbors, n_fine):
            raise RuntimeError(
                "Graph support extends beyond max_neighbors for at least one source. "
                "Increase max_neighbors or reduce graph_eta/graph_eta_max."
            )
        if expansions >= max_expansions:
            raise RuntimeError(
                "Graph-neighbour search reached max_neighbor_expansions."
            )
        query_count = min(2 * current_max, max_neighbors, n_fine)
        unresolved_array = np.asarray(unresolved, dtype=np.int64)
        if entry.tree is not None:
            queried_distances, queried_neighbors = entry.tree.query(
                entry.points[unresolved_array], k=query_count, workers=int(workers)
            )
            queried_distances = np.atleast_2d(np.asarray(queried_distances, dtype=np.float64))
            queried_neighbors = np.atleast_2d(np.asarray(queried_neighbors, dtype=np.int64))
        else:
            full_distances, full_neighbors = _ensure_knn(
                entry, n_fine, workers=workers, deadline=deadline
            )
            queried_distances = full_distances[unresolved_array, :query_count]
            queried_neighbors = full_neighbors[unresolved_array, :query_count]
        for row_index, source_index in enumerate(unresolved):
            distance_rows[source_index] = queried_distances[row_index].copy()
            neighbor_rows[source_index] = queried_neighbors[row_index].copy()
        expansions += 1
        _check_deadline(deadline, "transport-graph neighbour expansion")


def _build_smooth_kernel(
    distance_rows: list[np.ndarray],
    neighbor_rows: list[np.ndarray],
    graph_support: np.ndarray,
    *,
    fine_volume: np.ndarray,
    fine_mass: np.ndarray,
    fine_to_coarse: np.ndarray,
    n_coarse: int,
    deadline: float,
):
    """Build the source-to-output Wendland kernel and graph-enclosed masses."""

    from scipy.sparse import coo_matrix

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    graph_enclosed_mass = np.empty(len(graph_support), dtype=np.float64)
    for source_index in range(len(graph_support)):
        if source_index % 128 == 0:
            _check_deadline(deadline, "smooth-kernel construction")
        distances = distance_rows[source_index]
        neighbors = neighbor_rows[source_index]
        selected = distances < graph_support[source_index]
        selected_neighbors = neighbors[selected]
        selected_distances = distances[selected]
        if selected_neighbors.size == 0:
            raise RuntimeError(f"Source {source_index} has an empty graph support.")
        graph_enclosed_mass[source_index] = fine_mass[selected_neighbors].sum(
            dtype=np.float64
        )
        profile = wendland_c2_profile(
            selected_distances / graph_support[source_index]
        )
        weights = profile * fine_volume[selected_neighbors]
        destinations = fine_to_coarse[selected_neighbors]
        destination_weights = np.bincount(
            destinations, weights=weights, minlength=n_coarse
        )
        positive = np.flatnonzero(destination_weights > 0)
        if positive.size == 0:
            raise RuntimeError(f"Source {source_index} has zero Wendland kernel weight.")
        rows.extend(positive.tolist())
        cols.extend(np.full(positive.size, source_index, dtype=np.int64).tolist())
        values.extend(destination_weights[positive].tolist())
    kernel = coo_matrix(
        (values, (rows, cols)),
        shape=(n_coarse, len(graph_support)),
        dtype=np.float64,
    ).tocsr()
    kernel.eliminate_zeros()
    return kernel, graph_enclosed_mass


def _coarse_graph_audit(kernel, fine_to_coarse: np.ndarray, n_coarse: int) -> dict:
    """Audit whether positive coarse transport edges belong to circulation cycles."""

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    coo = kernel.tocoo()
    native = fine_to_coarse[coo.col]
    adjacency = coo_matrix(
        (
            np.ones(coo.nnz, dtype=np.int8),
            (native, coo.row),
        ),
        shape=(n_coarse, n_coarse),
    ).tocsr()
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()
    n_components, labels = connected_components(
        adjacency, directed=True, connection="strong"
    )
    cross_scc = labels[native] != labels[coo.row]
    return {
        "coo": coo,
        "adjacency": adjacency,
        "labels": labels.astype(np.int32),
        "n_components": int(n_components),
        "cross_scc": cross_scc,
        "cross_count": int(np.count_nonzero(cross_scc)),
        "cross_weight": float(coo.data[cross_scc].sum(dtype=np.float64)),
        "native_blocks": native,
    }


def audit_and_prune_transport_graph(
    kernel,
    fine_to_coarse: np.ndarray,
    n_coarse: int,
):
    """Remove edges that cannot carry positive volume in a circulation.

    Because every source has a native destination and each output marginal is
    the sum of its native source volumes, cross-SCC edges in the induced coarse
    graph must have zero flow in every feasible circulation.  Keeping them
    causes ordinary IPFP to drive scales toward a boundary solution slowly.
    """

    from scipy.sparse import coo_matrix

    audit = _coarse_graph_audit(kernel, fine_to_coarse, n_coarse)
    coo = audit["coo"]
    keep = ~audit["cross_scc"]
    pruned = coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=coo.shape,
    ).tocsr()
    pruned.eliminate_zeros()
    return pruned, audit


def _validate_native_feasibility(
    kernel,
    *,
    fine_to_coarse: np.ndarray,
    fine_volume: np.ndarray,
    coarse_volume: np.ndarray,
) -> None:
    """Validate the aligned-grid native-edge feasible transport witness."""

    matrix = kernel.tocsc()
    for source_index, native_row in enumerate(fine_to_coarse):
        start, end = matrix.indptr[source_index : source_index + 2]
        rows = matrix.indices[start:end]
        if not np.any(rows == native_row):
            raise RuntimeError(
                f"Source {source_index} lost its native transport edge; feasibility is not guaranteed."
            )
    witness_rows = np.bincount(
        fine_to_coarse, weights=fine_volume, minlength=coarse_volume.size
    )
    if not np.allclose(witness_rows, coarse_volume, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "Native transport witness does not reproduce the requested output volumes."
        )


def _marginal_error(transport, row_marginal: np.ndarray, column_marginal: np.ndarray) -> float:
    rows = np.asarray(transport.sum(axis=1)).reshape(-1)
    columns = np.asarray(transport.sum(axis=0)).reshape(-1)
    row_error = np.max(np.abs(rows - row_marginal) / row_marginal)
    column_error = np.max(np.abs(columns - column_marginal) / column_marginal)
    return float(max(row_error, column_error))


def _polish_ipfp_with_deadline(
    matrix,
    *,
    row_marginal: np.ndarray,
    column_marginal: np.ndarray,
    row_scale: np.ndarray,
    column_scale: np.ndarray,
    tolerance: float,
    max_iterations: int,
    deadline: float,
):
    """Deadline-aware IPFP polish starting from accelerated dual scales."""

    row_scale = np.asarray(row_scale, dtype=np.float64).copy()
    column_scale = np.asarray(column_scale, dtype=np.float64).copy()
    error = np.inf
    for iteration in range(1, int(max_iterations) + 1):
        row_denominator = np.asarray(matrix @ column_scale).reshape(-1)
        if np.any(row_denominator <= 0) or np.any(~np.isfinite(row_denominator)):
            raise RuntimeError("IPFP polish row scaling became singular.")
        row_scale = row_marginal / row_denominator
        column_denominator = np.asarray(matrix.T @ row_scale).reshape(-1)
        if np.any(column_denominator <= 0) or np.any(~np.isfinite(column_denominator)):
            raise RuntimeError("IPFP polish column scaling became singular.")
        column_scale = column_marginal / column_denominator
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            _check_deadline(deadline, "IPFP balance polish")
            current_rows = row_scale * np.asarray(matrix @ column_scale).reshape(-1)
            current_columns = column_scale * np.asarray(matrix.T @ row_scale).reshape(-1)
            error = float(
                max(
                    np.max(np.abs(current_rows - row_marginal) / row_marginal),
                    np.max(
                        np.abs(current_columns - column_marginal) / column_marginal
                    ),
                )
            )
            if error <= tolerance:
                break
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            "Accelerated transport polish did not converge: "
            f"error={error:.3e}, tolerance={tolerance:.3e}, iterations={iteration}."
        )
    transport = matrix.multiply(row_scale[:, None]).multiply(
        column_scale[None, :]
    ).tocsr()
    transport.eliminate_zeros()
    return transport, error, iteration, row_scale


def _balance_dual_lbfgs(
    kernel,
    *,
    row_marginal: np.ndarray,
    column_marginal: np.ndarray,
    tolerance: float,
    solver_max_iterations: int,
    polish_max_iterations: int,
    warm_start_row_log_scale: np.ndarray | None,
    deadline: float,
):
    """Balance K through an (n_rows-1)-dimensional convex dual plus IPFP polish."""

    matrix = kernel.tocsr(copy=True)
    column_sum = np.asarray(matrix.sum(axis=0)).reshape(-1)
    if np.any(column_sum <= 0):
        raise RuntimeError("Transport kernel contains an empty source column.")
    matrix = matrix.multiply((1.0 / column_sum)[None, :]).tocsr()
    row_marginal = np.asarray(row_marginal, dtype=np.float64).reshape(-1)
    column_marginal = np.asarray(column_marginal, dtype=np.float64).reshape(-1)
    total = float(row_marginal.sum(dtype=np.float64))
    r = row_marginal / total
    c = column_marginal / total
    n_rows = len(r)

    if warm_start_row_log_scale is None:
        x0_full = np.zeros(n_rows, dtype=np.float64)
    else:
        x0_full = np.asarray(warm_start_row_log_scale, dtype=np.float64).reshape(-1)
        if x0_full.shape != (n_rows,) or np.any(~np.isfinite(x0_full)):
            raise ValueError("warm_start_row_log_scale has incompatible or non-finite values.")
        x0_full = x0_full - x0_full[-1]
    x0 = x0_full[:-1]

    evaluations = 0

    def objective(x_reduced: np.ndarray):
        nonlocal evaluations
        evaluations += 1
        if evaluations % 8 == 0:
            _check_deadline(deadline, "L-BFGS transport balance")
        x = np.concatenate((np.asarray(x_reduced, dtype=np.float64), [0.0]))
        a = np.exp(x)
        denominator = np.asarray(matrix.T @ a).reshape(-1)
        if np.any(denominator <= 0) or np.any(~np.isfinite(denominator)):
            return np.inf, np.full_like(x_reduced, np.inf)
        # Elementwise reductions avoid a Windows BLAS ABI failure observed for
        # np.dot on the 20k-element dual term in the isolated preprocessing env.
        value = float(np.sum(c * np.log(denominator)) - np.sum(r * x))
        row_sum = a * np.asarray(matrix @ (c / denominator)).reshape(-1)
        gradient = row_sum - r
        return value, gradient[:-1]

    # A compact NumPy L-BFGS implementation is used instead of SciPy's
    # compiled L-BFGS-B wrapper.  The latter is not ABI-stable in one of the
    # project's Windows preprocessing environments and can terminate the
    # interpreter rather than raise a catchable exception.
    x_current = x0.copy()
    value, gradient = objective(x_current)
    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    rho_history: list[float] = []
    history_size = 10
    success = False
    message = "maximum iterations reached"
    iterations = 0
    for iterations in range(1, int(solver_max_iterations) + 1):
        _check_deadline(deadline, "L-BFGS transport balance")
        x_full = np.concatenate((x_current, [0.0]))
        a_current = np.exp(x_full)
        denominator_current = np.asarray(matrix.T @ a_current).reshape(-1)
        row_current = a_current * np.asarray(
            matrix @ (c / denominator_current)
        ).reshape(-1)
        relative_error = float(np.max(np.abs(row_current - r) / r))
        if relative_error <= tolerance:
            success = True
            message = "requested marginal tolerance reached"
            break

        q = gradient.copy()
        alphas: list[float] = []
        for s_vector, y_vector, rho in reversed(
            list(zip(s_history, y_history, rho_history))
        ):
            alpha = rho * float(np.dot(s_vector, q))
            alphas.append(alpha)
            q = q - alpha * y_vector
        if s_history:
            sy = float(np.dot(s_history[-1], y_history[-1]))
            yy = float(np.dot(y_history[-1], y_history[-1]))
            gamma = sy / yy if sy > 0 and yy > 0 else 1.0
            direction = gamma * q
        else:
            # The dual Hessian has a row-marginal scale.  Dividing by r is a
            # useful diagonal inverse-Hessian approximation for the first
            # step and after any safeguarded history reset.
            direction = q / np.maximum(r[:-1], np.finfo(np.float64).tiny)
        for (s_vector, y_vector, rho), alpha in zip(
            zip(s_history, y_history, rho_history), reversed(alphas)
        ):
            beta = rho * float(np.dot(y_vector, direction))
            direction = direction + s_vector * (alpha - beta)
        direction = -direction
        directional_derivative = float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0:
            direction = -gradient / np.maximum(
                r[:-1], np.finfo(np.float64).tiny
            )
            directional_derivative = -float(np.dot(gradient, gradient))
            s_history.clear()
            y_history.clear()
            rho_history.clear()

        step = 1.0
        accepted = False
        for _ in range(30):
            trial = np.clip(x_current + step * direction, -60.0, 60.0)
            trial_value, trial_gradient = objective(trial)
            if np.isfinite(trial_value) and trial_value <= (
                value + 1e-4 * step * directional_derivative
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "Armijo line search failed"
            break

        s_vector = trial - x_current
        y_vector = trial_gradient - gradient
        curvature = float(np.dot(s_vector, y_vector))
        if curvature > 1e-14 * max(
            1.0, np.linalg.norm(s_vector) * np.linalg.norm(y_vector)
        ):
            if len(s_history) == history_size:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            s_history.append(s_vector)
            y_history.append(y_vector)
            rho_history.append(1.0 / curvature)
        x_current = trial
        value = trial_value
        gradient = trial_gradient

    _check_deadline(deadline, "L-BFGS transport balance")
    x = np.concatenate((np.asarray(x_current, dtype=np.float64), [0.0]))
    row_scale = np.exp(x)
    denominator = np.asarray(matrix.T @ row_scale).reshape(-1)
    column_scale = column_marginal / denominator
    candidate = matrix.multiply(row_scale[:, None]).multiply(
        column_scale[None, :]
    ).tocsr()
    pre_polish_error = _marginal_error(candidate, row_marginal, column_marginal)

    if pre_polish_error <= tolerance:
        return (
            candidate,
            pre_polish_error,
            int(iterations),
            0,
            x,
            bool(success),
            message,
            pre_polish_error,
        )
    if not np.isfinite(pre_polish_error):
        raise RuntimeError(
            "L-BFGS transport balance failed to approach the requested marginals: "
            f"error={pre_polish_error:.3e}, success={success}, message={message}."
        )
    transport, error, polish_iterations, final_row = _polish_ipfp_with_deadline(
        matrix,
        row_marginal=row_marginal,
        column_marginal=column_marginal,
        tolerance=tolerance,
        max_iterations=polish_max_iterations,
        row_scale=row_scale,
        column_scale=column_scale,
        deadline=deadline,
    )
    _check_deadline(deadline, "IPFP balance polish")
    final_log = np.log(final_row)
    final_log -= final_log[-1]
    return (
        transport,
        error,
        int(iterations),
        int(polish_iterations),
        final_log,
        bool(success),
        message,
        pre_polish_error,
    )


def coarsen_mass_v1(
    source: CoarseningInput,
    new_shape: tuple[int, int],
    *,
    mass_filter: float,
    graph_eta: float = 1.0,
    graph_eta_growth: float = 1.35,
    graph_eta_max: float = 2.5,
    initial_neighbors: int = 64,
    max_neighbors: int = 4096,
    max_neighbor_expansions: int = 8,
    max_graph_repairs: int = 2,
    solver_max_iterations: int = 300,
    polish_max_iterations: int = 500,
    balance_tolerance: float = 1e-6,
    max_runtime_seconds: float = 60.0,
    knn_workers: int = 1,
    warm_start_row_log_scale: np.ndarray | None = None,
) -> CoarseningResult:
    """Apply accelerated axisymmetric Rm-v1 to a MACER2D snapshot."""

    started = perf_counter()
    max_runtime_seconds = float(max_runtime_seconds)
    if not np.isfinite(max_runtime_seconds) or max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive and finite.")
    deadline = started + max_runtime_seconds
    _check_deadline(deadline, "input validation")
    if source.ndim != 2 or source.coordinate_mode != "polar":
        raise NotImplementedError("Rm-v1 is defined only for axisymmetric MACER2D polar data.")
    if len(new_shape) != 2:
        raise ValueError("new_shape must contain (N_theta, N_r).")
    new_shape = tuple(int(value) for value in new_shape)
    mass_filter = float(mass_filter)
    if not np.isfinite(mass_filter) or mass_filter <= 0:
        raise ValueError("mass_filter must be positive and finite.")
    for name, value, minimum in (
        ("initial_neighbors", initial_neighbors, 2),
        ("max_neighbors", max_neighbors, 2),
        ("max_neighbor_expansions", max_neighbor_expansions, 0),
        ("max_graph_repairs", max_graph_repairs, 0),
        ("solver_max_iterations", solver_max_iterations, 1),
        ("polish_max_iterations", polish_max_iterations, 1),
    ):
        if isinstance(value, bool) or int(value) < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if int(initial_neighbors) > int(max_neighbors):
        raise ValueError("initial_neighbors cannot exceed max_neighbors.")
    for name, value in (
        ("graph_eta", graph_eta),
        ("graph_eta_growth", graph_eta_growth),
        ("graph_eta_max", graph_eta_max),
        ("balance_tolerance", balance_tolerance),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if graph_eta < 1 or graph_eta_growth <= 1 or graph_eta_max < graph_eta:
        raise ValueError(
            "Require graph_eta >= 1, graph_eta_growth > 1, and graph_eta_max >= graph_eta."
        )

    snapshot = source.snapshot
    unsupported = sorted(set(snapshot).difference(SUPPORTED_FIELDS))
    if unsupported:
        raise ValueError(f"No conservative Rm-v1 rule exists for fields: {unsupported}.")
    fine_density = np.asarray(snapshot["density"], dtype=np.float64)
    fine_shape = tuple(int(value) for value in fine_density.shape)
    fine_volume = np.asarray(source.grid_volume, dtype=np.float64)
    if fine_volume.shape != fine_shape or np.any(fine_volume <= 0) or not np.all(
        np.isfinite(fine_volume)
    ):
        raise ValueError("Fine-cell volumes must match density and be finite and positive.")
    if np.any(fine_density < 0) or not np.all(np.isfinite(fine_density)):
        raise ValueError("Density must be finite and non-negative.")
    fine_mass = fine_density * fine_volume
    fine_mass_flat = fine_mass.reshape(-1)
    fine_volume_flat = fine_volume.reshape(-1)
    total_mass = float(fine_mass_flat.sum(dtype=np.float64))
    if total_mass <= 0:
        raise ValueError("The source snapshot has no positive gas mass.")
    if mass_filter > total_mass * (1 + 1e-12):
        raise ValueError("mass_filter exceeds the snapshot total gas mass.")
    mass_filter = min(mass_filter, total_mass)

    coarse_volume, new_axes, new_axis_edges = build_regular_coarse_geometry(
        source, new_shape
    )
    fine_shape_array = np.asarray(fine_shape, dtype=int)
    new_shape_array = np.asarray(new_shape, dtype=int)
    theta_factor, radial_factor = (fine_shape_array // new_shape_array).astype(int)
    theta_index, radial_index = np.indices(fine_shape)
    fine_to_coarse = (
        (theta_index // theta_factor) * new_shape[1]
        + radial_index // radial_factor
    ).reshape(-1)
    n_coarse = int(np.prod(new_shape))

    r_grid, theta_grid = np.meshgrid(*source.axes)
    points = np.column_stack(
        (
            (r_grid * np.sin(theta_grid)).reshape(-1),
            (r_grid * np.cos(theta_grid)).reshape(-1),
        )
    )
    neighbor_started = perf_counter()
    cache_entry, cache_hit = _get_cache_entry(points)
    mass_support, enclosed_mass, distance_rows, neighbor_rows, mass_expansions = (
        _find_mass_supports(
            cache_entry,
            fine_mass_flat,
            mass_filter,
            initial_neighbors=int(initial_neighbors),
            max_neighbors=min(int(max_neighbors), len(points)),
            max_expansions=int(max_neighbor_expansions),
            workers=int(knn_workers),
            deadline=deadline,
            single_cell_scale=float(np.cbrt(fine_volume_flat[0])),
        )
    )
    graph_support = float(graph_eta) * mass_support
    distance_rows, neighbor_rows, graph_expansions = _ensure_graph_neighbors(
        cache_entry,
        graph_support,
        distance_rows=distance_rows,
        neighbor_rows=neighbor_rows,
        max_neighbors=min(int(max_neighbors), len(points)),
        max_expansions=int(max_neighbor_expansions),
        workers=int(knn_workers),
        deadline=deadline,
    )
    kernel, graph_enclosed_mass = _build_smooth_kernel(
        distance_rows,
        neighbor_rows,
        graph_support,
        fine_volume=fine_volume_flat,
        fine_mass=fine_mass_flat,
        fine_to_coarse=fine_to_coarse,
        n_coarse=n_coarse,
        deadline=deadline,
    )
    initial_audit = _coarse_graph_audit(kernel, fine_to_coarse, n_coarse)
    initial_cross_count = initial_audit["cross_count"]
    repair_rounds = 0
    repair_limited_by_neighbor_cap = False
    for _ in range(int(max_graph_repairs)):
        audit = _coarse_graph_audit(kernel, fine_to_coarse, n_coarse)
        if audit["cross_count"] == 0:
            break
        cross_rows = audit["coo"].row[audit["cross_scc"]]
        cross_native = audit["native_blocks"][audit["cross_scc"]]
        affected_blocks = np.unique(np.concatenate((cross_rows, cross_native)))
        affected_sources = np.isin(fine_to_coarse, affected_blocks)
        expanded = np.minimum(
            graph_support[affected_sources] * float(graph_eta_growth),
            mass_support[affected_sources] * float(graph_eta_max),
        )
        if np.allclose(expanded, graph_support[affected_sources], rtol=0, atol=0):
            break
        previous_support = graph_support.copy()
        graph_support[affected_sources] = expanded
        try:
            distance_rows, neighbor_rows, extra_expansions = _ensure_graph_neighbors(
                cache_entry,
                graph_support,
                distance_rows=distance_rows,
                neighbor_rows=neighbor_rows,
                max_neighbors=min(int(max_neighbors), len(points)),
                max_expansions=int(max_neighbor_expansions),
                workers=int(knn_workers),
                deadline=deadline,
            )
        except RuntimeError as exc:
            if "max_neighbors" not in str(exc) and "max_neighbor_expansions" not in str(exc):
                raise
            graph_support = previous_support
            repair_limited_by_neighbor_cap = True
            break
        graph_expansions += extra_expansions
        kernel, graph_enclosed_mass = _build_smooth_kernel(
            distance_rows,
            neighbor_rows,
            graph_support,
            fine_volume=fine_volume_flat,
            fine_mass=fine_mass_flat,
            fine_to_coarse=fine_to_coarse,
            n_coarse=n_coarse,
            deadline=deadline,
        )
        repair_rounds += 1
    kernel_before_prune_nnz = int(kernel.nnz)
    kernel, final_preprune_audit = audit_and_prune_transport_graph(
        kernel, fine_to_coarse, n_coarse
    )
    pruned_count = final_preprune_audit["cross_count"]
    pruned_weight = final_preprune_audit["cross_weight"]
    final_audit = _coarse_graph_audit(kernel, fine_to_coarse, n_coarse)
    _validate_native_feasibility(
        kernel,
        fine_to_coarse=fine_to_coarse,
        fine_volume=fine_volume_flat,
        coarse_volume=coarse_volume.reshape(-1),
    )
    neighbor_graph_seconds = perf_counter() - neighbor_started

    balance_started = perf_counter()
    (
        transport,
        balance_error,
        lbfgs_iterations,
        polish_iterations,
        final_log_scale,
        lbfgs_success,
        lbfgs_message,
        pre_polish_error,
    ) = _balance_dual_lbfgs(
        kernel,
        row_marginal=coarse_volume.reshape(-1),
        column_marginal=fine_volume_flat,
        tolerance=float(balance_tolerance),
        solver_max_iterations=int(solver_max_iterations),
        polish_max_iterations=int(polish_max_iterations),
        warm_start_row_log_scale=warm_start_row_log_scale,
        deadline=deadline,
    )
    balance_seconds = perf_counter() - balance_started
    fractions = transport.multiply((1.0 / fine_volume_flat)[None, :]).tocsr()

    def deposit_extensive(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            fractions @ np.asarray(values, dtype=np.float64).reshape(-1)
        ).reshape(new_shape)

    coarse_mass = deposit_extensive(fine_mass)
    if np.any(coarse_mass <= 0):
        raise ValueError("Every Rm-v1 output cell must contain positive gas mass.")
    coarse_energy = deposit_extensive(
        np.asarray(snapshot["gas_energy"], dtype=np.float64) * fine_volume
    )
    new_snapshot: dict[str, np.ndarray] = {
        "density": coarse_mass / coarse_volume,
        "gas_energy": coarse_energy / coarse_volume,
    }
    for name in ("v1", "v2", "v3"):
        if name in snapshot:
            new_snapshot[name] = deposit_extensive(
                np.asarray(snapshot[name], dtype=np.float64) * fine_mass
            ) / coarse_mass
    if "initial_stellar_density" in snapshot:
        new_snapshot["initial_stellar_density"] = deposit_extensive(
            np.asarray(snapshot["initial_stellar_density"], dtype=np.float64)
            * fine_volume
        ) / coarse_volume
    new_snapshot["temperature"] = (
        93.0 * new_snapshot["gas_energy"] / new_snapshot["density"]
    )
    if any(not np.all(np.isfinite(values)) for values in new_snapshot.values()):
        raise ValueError("Rm-v1 generated a non-finite coarse field.")

    diagnostics: dict[str, np.ndarray] = {
        "source_support_scale": mass_support.reshape(fine_shape),
        "source_mass_support_scale": mass_support.reshape(fine_shape),
        "source_graph_support_scale": graph_support.reshape(fine_shape),
        "source_enclosed_mass": enclosed_mass.reshape(fine_shape),
        "source_enclosed_mass_relative_excess": (
            enclosed_mass / mass_filter - 1.0
        ).reshape(fine_shape),
        "source_graph_enclosed_mass": graph_enclosed_mass.reshape(fine_shape),
        "coarse_transport_adjacency": final_audit["adjacency"].toarray().astype(np.int8),
        "coarse_transport_scc_label": final_audit["labels"],
        "solver_row_log_scale": final_log_scale,
    }
    diagnostics.update(
        _transport_diagnostics(
            transport,
            fractions,
            fine_to_coarse=fine_to_coarse,
            fine_mass=fine_mass_flat,
            coarse_mass=coarse_mass,
            support_scale=mass_support,
            fine_shape=fine_shape,
            new_shape=new_shape,
        )
    )
    native_fraction = np.asarray(
        fractions[fine_to_coarse, np.arange(len(fine_to_coarse))]
    ).reshape(-1)
    diagnostics["source_native_transport_fraction"] = native_fraction.reshape(fine_shape)

    total_seconds = perf_counter() - started
    _check_deadline(deadline, "result assembly")
    metadata = {
        "operator": "adaptive_mass_axisymmetric_v1",
        "operator_version": 1,
        "source_shape": fine_shape,
        "output_shape": new_shape,
        "mass_filter": mass_filter,
        "kernel": "wendland_c2",
        "distance_metric": "meridional_euclidean_Rz",
        "mass_geometry": "full_axisymmetric_ring",
        "mass_support_semantics": "fixed_enclosed_ring_mass",
        "graph_support_semantics": "eta_times_mass_support_with_local_repair",
        "partition_semantics": "overlapping_support_plus_soft_transport",
        "graph_eta_initial": float(graph_eta),
        "graph_eta_max": float(graph_eta_max),
        "knn_backend": cache_entry.backend,
        "knn_cache_hit": bool(cache_hit),
        "knn_cached_k": int(cache_entry.k),
        "knn_max_row_k": int(max(len(row) for row in distance_rows)),
        "neighbor_expansions": int(mass_expansions + graph_expansions),
        "graph_repair_rounds": int(repair_rounds),
        "graph_repair_limited_by_neighbor_cap": bool(repair_limited_by_neighbor_cap),
        "graph_initial_cross_scc_edges": int(initial_cross_count),
        "graph_pruned_cross_scc_edges": int(pruned_count),
        "graph_pruned_kernel_weight": float(pruned_weight),
        "graph_scc_count": int(final_audit["n_components"]),
        "kernel_nnz_before_prune": int(kernel_before_prune_nnz),
        "kernel_nnz": int(kernel.nnz),
        "transport_balance_tolerance": float(balance_tolerance),
        "transport_balance_error": float(balance_error),
        "transport_balance_iterations": int(lbfgs_iterations + polish_iterations),
        "lbfgs_iterations": int(lbfgs_iterations),
        "ipfp_polish_iterations": int(polish_iterations),
        "lbfgs_success": bool(lbfgs_success),
        "lbfgs_message": lbfgs_message,
        "lbfgs_pre_polish_balance_error": float(pre_polish_error),
        "max_runtime_seconds": float(max_runtime_seconds),
        "runtime_neighbor_graph_seconds": float(neighbor_graph_seconds),
        "runtime_balance_seconds": float(balance_seconds),
        "runtime_total_seconds": float(total_seconds),
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
            "neighbor_graph_seconds": float(neighbor_graph_seconds),
            "balance_seconds": float(balance_seconds),
            "total_seconds": float(total_seconds),
        },
    )
