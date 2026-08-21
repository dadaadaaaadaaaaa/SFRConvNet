import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from raw_data_processing.galaxy_data import GalData


def _synthetic_galaxy(*, constant=False):
    galaxy = GalData(ndim=2, coordinate_mode="polar")
    radius = np.asarray([1.0, 2.0, 3.0, 4.0])
    theta = np.asarray([0.3, 0.9, 1.5, 2.1])
    galaxy._axes = [radius, theta]
    galaxy._axis_edges = [
        galaxy._infer_axis_edges(radius, is_radial=True),
        galaxy._infer_axis_edges(theta, is_radial=False),
    ]
    galaxy._coord = tuple(np.meshgrid(*galaxy._axes))

    radial_edges, theta_edges = galaxy._axis_edges
    galaxy._grid_volume = (
        (2.0 * np.pi / 3.0)
        * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)[None, :]
        * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))[:, None]
    )
    galaxy._volume_source = "synthetic_axisymmetric_volume"

    if constant:
        density = np.full((4, 4), 3.0)
        energy_density = np.full((4, 4), 7.0)
    else:
        density = np.ones((4, 4))
        density[1:3, 1:3] = 5.0
        energy_density = 2.0 * density
    velocity = np.arange(16, dtype=np.float64).reshape(4, 4)
    galaxy._data_hdfra = {
        "density": density,
        "gas_energy": energy_density,
        "temperature": 93.0 * energy_density / density,
        "v1": velocity,
        "v2": -velocity,
        "v3": np.zeros_like(velocity),
        "initial_stellar_density": 0.1 * density,
    }
    galaxy.time = 0.0
    galaxy._corr = {}
    galaxy._data_log = {}
    galaxy._target_metadata = {}
    return galaxy


class MassAdaptiveCoarseningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Some development Python environments can import scipy.sparse but
        # not scipy.spatial.  The production path should use cKDTree; this
        # small synthetic test deliberately permits the NumPy fallback.
        warnings.filterwarnings(
            "ignore",
            message="scipy.spatial.cKDTree is unavailable",
            category=RuntimeWarning,
        )

    def test_conserves_extensive_quantities(self):
        fine = _synthetic_galaxy()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        coarse = fine.rescale_mass_adaptive(
            (2, 2), mass_filter=mass_filter, kernel="top_hat"
        )

        fine_volume = fine.grid_volume
        coarse_volume = coarse.grid_volume
        fine_mass = fine.snapshot["density"] * fine_volume
        coarse_mass = coarse.snapshot["density"] * coarse_volume
        self.assertAlmostEqual(fine_volume.sum(), coarse_volume.sum(), places=10)
        self.assertAlmostEqual(fine_mass.sum(), coarse_mass.sum(), places=10)
        self.assertAlmostEqual(
            np.sum(fine.snapshot["gas_energy"] * fine_volume),
            np.sum(coarse.snapshot["gas_energy"] * coarse_volume),
            places=9,
        )
        for name in ("v1", "v2", "v3"):
            self.assertAlmostEqual(
                np.sum(fine.snapshot[name] * fine_mass),
                np.sum(coarse.snapshot[name] * coarse_mass),
                places=9,
            )

    def test_preserves_constant_volume_densities(self):
        fine = _synthetic_galaxy(constant=True)
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        coarse = fine.rescale_mass_adaptive(
            (2, 2), mass_filter=mass_filter, kernel="top_hat"
        )
        np.testing.assert_allclose(coarse.snapshot["density"], 3.0, rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(
            coarse.snapshot["gas_energy"], 7.0, rtol=2e-6, atol=1e-8
        )

    def test_larger_mass_filter_does_not_shrink_support(self):
        fine = _synthetic_galaxy()
        base_mass = float(np.median(fine.gas_mass))
        smaller = fine.rescale_mass_adaptive(
            (2, 2), mass_filter=2.0 * base_mass, kernel="top_hat"
        )
        larger = fine.rescale_mass_adaptive(
            (2, 2), mass_filter=5.0 * base_mass, kernel="top_hat"
        )
        np.testing.assert_array_less(
            smaller.coarsening_diagnostics["source_support_scale"] - 1e-12,
            larger.coarsening_diagnostics["source_support_scale"],
        )

    def test_hdf5_round_trip_preserves_operator_metadata(self):
        fine = _synthetic_galaxy()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        coarse = fine.rescale_mass_adaptive(
            (2, 2), mass_filter=mass_filter, kernel="top_hat"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rm_v0.h5"
            coarse.save_h5(path)
            restored = GalData().load_h5(path)
        self.assertEqual(
            restored.coarsening_metadata["operator"],
            "adaptive_mass_axisymmetric_v0",
        )
        self.assertEqual(
            restored.coarsening_metadata["gas_energy_semantics"],
            "gas_internal_energy_density",
        )
        np.testing.assert_allclose(
            restored.coarsening_diagnostics["effective_support_scale"],
            coarse.coarsening_diagnostics["effective_support_scale"],
        )


if __name__ == "__main__":
    unittest.main()
