import unittest

import numpy as np

from raw_data_processing.coarsening_common import validate_coarsening_conservation
from raw_data_processing.coarsening_rm_v0 import coarsen_mass_v0
from raw_data_processing.coarsening_rm_v1 import (
    CoarseningTimeout,
    audit_and_prune_transport_graph,
    clear_geometry_neighbor_cache,
    coarsen_mass_v1,
    wendland_c2_profile,
)
from raw_data_processing.coarsening_spatial import coarsen_spatial_block
from raw_data_processing.coarsening_types import CoarseningInput


def _source(*, constant=False, void=False, giant=False) -> CoarseningInput:
    radius = np.asarray([1.0, 2.0, 3.0, 4.0])
    theta = np.asarray([0.3, 0.9, 1.5, 2.1])
    radial_edges = np.asarray([0.7, 1.4, 2.45, 3.46, 4.62])
    theta_edges = np.asarray([0.0, 0.6, 1.2, 1.8, 2.4])
    volume = (
        (2.0 * np.pi / 3.0)
        * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)[None, :]
        * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))[:, None]
    )
    if constant:
        density = np.full((4, 4), 3.0)
        energy = np.full((4, 4), 7.0)
    else:
        density = np.ones((4, 4))
        density[1:3, 1:3] = 5.0
        energy = 2.0 * density
    if void:
        density[1:3, 1:3] = 0.0
        energy[1:3, 1:3] = 0.0
    if giant:
        density[1, 1] = 1.0e6
        energy[1, 1] = 2.0e6
    velocity = np.arange(16, dtype=np.float64).reshape(4, 4)
    snapshot = {
        "density": density,
        "gas_energy": energy,
        "temperature": np.divide(
            93.0 * energy,
            density,
            out=np.zeros_like(energy),
            where=density > 0,
        ),
        "v1": velocity,
        "v2": -velocity,
        "v3": np.zeros_like(velocity),
        "initial_stellar_density": 0.1 * density,
    }
    return CoarseningInput(
        snapshot=snapshot,
        grid_volume=volume,
        axes=(radius, theta),
        axis_edges=(radial_edges, theta_edges),
        coordinate_mode="polar",
        ndim=2,
    )


def _run(source, mass_filter, **kwargs):
    return coarsen_mass_v1(
        source,
        (2, 2),
        mass_filter=mass_filter,
        graph_eta=1.0,
        graph_eta_max=2.0,
        initial_neighbors=4,
        max_neighbors=16,
        max_runtime_seconds=10.0,
        **kwargs,
    )


class MassAdaptiveCoarseningV1Tests(unittest.TestCase):
    def setUp(self):
        clear_geometry_neighbor_cache()

    def test_wendland_profile_is_smooth_compact_and_nonnegative(self):
        values = wendland_c2_profile(np.asarray([0.0, 0.5, 0.999999, 1.0, 2.0]))
        self.assertAlmostEqual(values[0], 1.0)
        self.assertTrue(np.all(values >= 0))
        self.assertGreater(values[1], values[2])
        self.assertEqual(values[3], 0.0)
        self.assertEqual(values[4], 0.0)

    def test_constant_fields_conservation_and_cache_reuse(self):
        source = _source(constant=True)
        mass_filter = 3.0 * float(np.median(source.snapshot["density"] * source.grid_volume))
        first = _run(source, mass_filter)
        second = _run(
            source,
            mass_filter,
            warm_start_row_log_scale=first.diagnostics["solver_row_log_scale"],
        )
        validate_coarsening_conservation(source, first, rtol=2e-6, atol=1e-9)
        np.testing.assert_allclose(first.snapshot["density"], 3.0, rtol=2e-6)
        np.testing.assert_allclose(first.snapshot["gas_energy"], 7.0, rtol=2e-6)
        self.assertFalse(first.metadata["knn_cache_hit"])
        self.assertTrue(second.metadata["knn_cache_hit"])
        self.assertEqual(first.metadata["partition_semantics"], "overlapping_support_plus_soft_transport")

    def test_tiny_mass_filter_reduces_to_spatial_block_and_v0(self):
        source = _source()
        positive_mass = (source.snapshot["density"] * source.grid_volume)
        mass_filter = 0.1 * float(np.min(positive_mass[positive_mass > 0]))
        v1 = _run(source, mass_filter)
        rh = coarsen_spatial_block(source, (2, 2))
        v0 = coarsen_mass_v0(
            source,
            (2, 2),
            mass_filter=mass_filter,
            kernel="top_hat",
            initial_neighbors=4,
            balance_max_iterations=100,
        )
        for name in v1.snapshot:
            np.testing.assert_allclose(v1.snapshot[name], rh.snapshot[name], rtol=2e-6, atol=1e-9)
            np.testing.assert_allclose(v1.snapshot[name], v0.snapshot[name], rtol=2e-6, atol=1e-9)

    def test_total_mass_filter_and_void_remain_conservative(self):
        source = _source(void=True)
        total_mass = float(np.sum(source.snapshot["density"] * source.grid_volume))
        result = _run(source, total_mass)
        validate_coarsening_conservation(source, result, rtol=2e-6, atol=1e-9)
        self.assertTrue(np.all(np.isfinite(result.snapshot["density"])))
        self.assertTrue(np.all(result.snapshot["density"] > 0))
        self.assertTrue(np.all(result.diagnostics["source_enclosed_mass"] >= total_mass * (1 - 1e-12)))

    def test_single_giant_cell_records_mass_overshoot(self):
        source = _source(giant=True)
        ordinary = source.snapshot["density"] * source.grid_volume
        mass_filter = float(np.median(ordinary))
        result = _run(source, mass_filter)
        excess = result.diagnostics["source_enclosed_mass_relative_excess"]
        self.assertGreater(float(np.max(excess)), 100.0)
        validate_coarsening_conservation(source, result, rtol=2e-6, atol=1e-9)

    def test_one_way_cross_scc_edge_is_pruned(self):
        from scipy.sparse import csr_matrix

        # Columns are sources with native rows [0, 1].  K[1, 0] creates only
        # the coarse directed edge 0 -> 1, which cannot carry circulation.
        kernel = csr_matrix(np.asarray([[1.0, 0.0], [0.25, 1.0]]))
        pruned, audit = audit_and_prune_transport_graph(
            kernel, np.asarray([0, 1], dtype=np.int64), 2
        )
        self.assertEqual(audit["cross_count"], 1)
        np.testing.assert_allclose(pruned.toarray(), np.eye(2))

    def test_timeout_fails_fast_without_partial_result(self):
        source = _source()
        with self.assertRaises(CoarseningTimeout):
            coarsen_mass_v1(
                source,
                (2, 2),
                mass_filter=1.0,
                initial_neighbors=4,
                max_neighbors=16,
                max_runtime_seconds=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
