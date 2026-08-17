import unittest

import numpy as np

from raw_data_processing.coarsening_common import block_reduce_sum
from raw_data_processing.coarsening_spatial import coarsen_spatial_block
from raw_data_processing.coarsening_types import CoarseningInput


def _input() -> CoarseningInput:
    radius_edges = np.asarray([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    theta_edges = np.asarray([0.1, 0.6, 1.1, 1.6, 2.1], dtype=np.float64)
    radius = np.sqrt(radius_edges[:-1] * radius_edges[1:])
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    volume = (
        (2.0 * np.pi / 3.0)
        * (radius_edges[1:] ** 3 - radius_edges[:-1] ** 3)[None, :]
        * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))[:, None]
    )
    density = 1.0 + np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0
    velocity = np.arange(16, dtype=np.float64).reshape(4, 4) - 7.5
    energy_density = 2.0 + 0.5 * density
    return CoarseningInput(
        snapshot={
            "density": density,
            "gas_energy": energy_density,
            "temperature": 93.0 * energy_density / density,
            "v1": velocity,
            "v2": -2.0 * velocity,
            "v3": np.full_like(velocity, 3.0),
            "initial_stellar_density": 0.25 * density,
        },
        grid_volume=volume,
        axes=(radius, theta),
        axis_edges=(radius_edges, theta_edges),
        coordinate_mode="polar",
        ndim=2,
    )


class SpatialCoarseningModuleTests(unittest.TestCase):
    def test_preserves_legacy_spatial_field_semantics(self):
        fine = _input()
        coarse = coarsen_spatial_block(fine, (2, 2))

        expected_volume = block_reduce_sum(fine.grid_volume, (2, 2))
        fine_mass = fine.snapshot["density"] * fine.grid_volume
        expected_mass = block_reduce_sum(fine_mass, (2, 2))
        np.testing.assert_allclose(coarse.grid_volume, expected_volume)
        np.testing.assert_allclose(coarse.snapshot["density"], expected_mass / expected_volume)
        np.testing.assert_allclose(
            coarse.snapshot["gas_energy"],
            block_reduce_sum(fine.snapshot["gas_energy"] * fine.grid_volume, (2, 2))
            / expected_volume,
        )
        for name in ("v1", "v2", "v3"):
            np.testing.assert_allclose(
                coarse.snapshot[name],
                block_reduce_sum(fine.snapshot[name] * fine_mass, (2, 2)) / expected_mass,
            )
        np.testing.assert_allclose(
            coarse.snapshot["initial_stellar_density"],
            block_reduce_sum(
                fine.snapshot["initial_stellar_density"] * fine.grid_volume, (2, 2)
            )
            / expected_volume,
        )
        np.testing.assert_allclose(
            coarse.snapshot["temperature"],
            93.0 * coarse.snapshot["gas_energy"] / coarse.snapshot["density"],
        )
        self.assertEqual(coarse.metadata["operator"], "spatial_block")
        self.assertEqual(coarse.metadata["source_shape"], (4, 4))
        self.assertEqual(coarse.metadata["output_shape"], (2, 2))
        self.assertEqual(coarse.volume_source, "Data-Set-15:block_sum")
        self.assertEqual(coarse.runtime, {})

    def test_conserves_stored_extensive_quantities(self):
        fine = _input()
        coarse = coarsen_spatial_block(fine, (2, 2))
        fine_mass = fine.snapshot["density"] * fine.grid_volume
        coarse_mass = coarse.snapshot["density"] * coarse.grid_volume
        self.assertAlmostEqual(float(fine.grid_volume.sum()), float(coarse.grid_volume.sum()))
        self.assertAlmostEqual(float(fine_mass.sum()), float(coarse_mass.sum()))
        self.assertAlmostEqual(
            float(np.sum(fine.snapshot["gas_energy"] * fine.grid_volume)),
            float(np.sum(coarse.snapshot["gas_energy"] * coarse.grid_volume)),
        )
        for name in ("v1", "v2", "v3"):
            self.assertAlmostEqual(
                float(np.sum(fine.snapshot[name] * fine_mass)),
                float(np.sum(coarse.snapshot[name] * coarse_mass)),
            )

    def test_rejects_unknown_field_semantics(self):
        fine = _input()
        invalid = CoarseningInput(
            snapshot={**fine.snapshot, "unknown_tracer": np.ones((4, 4))},
            grid_volume=fine.grid_volume,
            axes=fine.axes,
            axis_edges=fine.axis_edges,
            coordinate_mode=fine.coordinate_mode,
            ndim=fine.ndim,
        )
        with self.assertRaisesRegex(ValueError, "No conservative coarsening rule"):
            coarsen_spatial_block(invalid, (2, 2))


if __name__ == "__main__":
    unittest.main()
