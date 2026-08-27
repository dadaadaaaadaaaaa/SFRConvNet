import tempfile
import unittest
from pathlib import Path

import numpy as np

from raw_data_processing.galaxy_data import GalData, GalDataSet


def _galaxy() -> GalData:
    galaxy = GalData(ndim=2, coordinate_mode="polar")
    radius = np.asarray([0.8, 1.2, 1.8, 2.7], dtype=np.float64)
    theta = np.asarray([0.3, 0.8, 1.3, 1.8], dtype=np.float64)
    galaxy._axes = [radius, theta]
    galaxy._axis_edges = [
        galaxy._infer_axis_edges(radius, is_radial=True),
        galaxy._infer_axis_edges(theta, is_radial=False),
    ]
    galaxy._coord = tuple(np.meshgrid(*galaxy._axes))
    galaxy._grid_volume = np.ones((4, 4), dtype=np.float64)
    galaxy._volume_source = "synthetic"
    density = np.full((4, 4), 2.0, dtype=np.float64)
    energy = density * 100.0 / 93.0
    zeros = np.zeros((4, 4), dtype=np.float64)
    galaxy._data_hdfra = {
        "density": density,
        "gas_energy": energy,
        "temperature": 93.0 * energy / density,
        "v1": zeros,
        "v2": zeros,
        "v3": zeros,
        "initial_stellar_density": zeros,
        "cooling_time": np.full((4, 4), 0.05, dtype=np.float64),
        "radial_acceleration": np.full((4, 4), 100.0, dtype=np.float64),
        "theta_acceleration": zeros,
    }
    galaxy._corr = {}
    galaxy._data_log = {"mbh": 2.0, "dt": 1.0e-6, "sfr": 3.0, "sfr_target": 3.0}
    galaxy._target_metadata = {
        "dataset": "Data-Set-11",
        "name": "sfr",
        "reduction": "sum",
        "quantity": "instantaneous_sfr",
        "units": "MACER_mass_code/Gyr",
        "to_msun_per_yr": 0.025,
    }
    galaxy.time = 1.0
    galaxy.attach_fine_source_metadata(
        {
            "simulation_code": "MACER2D",
            "galaxy_id": "synthetic",
            "run_id": "synthetic",
            "source_series_id": "synthetic",
            "raw_snapshot_index": 1,
        }
    )
    return galaxy.compute_macer_sfr_products()


class SFRSupervisionProductTests(unittest.TestCase):
    def test_interpolated_log_context_is_recorded_in_supervision_metadata(self):
        galaxy = _galaxy()
        galaxy._data_log.update(
            {
                "log_context_selection_code": 1,
                "log_context_window_count": 0,
                "log_context_left_time": 0.9,
                "log_context_right_time": 1.2,
                "log_context_bracket_width": 0.3,
                "log_context_interpolation_weight": 1.0 / 3.0,
                "log_context_scope": 0.05,
            }
        )
        galaxy.compute_macer_sfr_products()
        context = galaxy.sfr_products["data_compatible"]["context"]

        self.assertEqual(
            context["log_context_selection"],
            "linear_interpolation_empty_window",
        )
        self.assertTrue(context["log_context_fallback_used"])
        self.assertEqual(context["log_context_window_count"], 0)
        self.assertAlmostEqual(context["log_context_left_time_gyr"], 0.9)
        self.assertAlmostEqual(context["log_context_right_time_gyr"], 1.2)
        self.assertAlmostEqual(context["log_context_bracket_width_gyr"], 0.3)

    def test_hdf_round_trip_and_lr_closure_identity(self):
        fine = _galaxy()
        coarse = fine.coarsen("spatial_block", (2, 2))
        self.assertAlmostEqual(
            coarse.sfr_target,
            coarse.sfr_macer_baseline + coarse.sfr_closure_true,
            places=12,
        )
        self.assertNotEqual(
            fine.sfr_products["data_compatible"]["physics_product_uid"],
            coarse.sfr_products["data_compatible"]["physics_product_uid"],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "00001.h5"
            coarse.save_h5(path)
            restored = GalData().load_h5(path)
            self.assertAlmostEqual(restored.sfr_macer_baseline, coarse.sfr_macer_baseline)
            self.assertAlmostEqual(restored.sfr_closure_true, coarse.sfr_closure_true)
            dataset = GalDataSet().load_data(folder)
            self.assertNotIn("sfr_target", dataset.input_channels)
            self.assertNotIn("sfr_macer_baseline", dataset.input_channels)
            self.assertNotIn("sfr_closure_true", dataset.input_channels)
            np.testing.assert_allclose(dataset.sfr_macer_baseline, [coarse.sfr_macer_baseline])
            np.testing.assert_allclose(dataset.sfr_closure_true, [coarse.sfr_closure_true])

    def test_product_uid_depends_on_operator_mode(self):
        galaxy = _galaxy()
        products = galaxy.sfr_products
        self.assertNotEqual(
            products["data_compatible"]["physics_product_uid"],
            products["paper_physical"]["physics_product_uid"],
        )

    def test_rm_v0_and_v1_propagate_sfr_fields(self):
        fine = _galaxy()
        mass_filter = 0.1 * float(np.min(fine.gas_mass))
        coarse_samples = (
            fine.coarsen(
                "axisymmetric_rm_v0",
                (2, 2),
                mass_filter=mass_filter,
                balance_max_iterations=30000,
            ),
            fine.coarsen(
                "axisymmetric_rm_v1",
                (2, 2),
                mass_filter=mass_filter,
                initial_neighbors=8,
                max_neighbors=16,
                max_runtime_seconds=10.0,
            ),
        )
        fine_power = np.sum(
            fine.snapshot["cooling_rate_density"] * fine.grid_volume
        )
        for coarse in coarse_samples:
            coarse_power = np.sum(
                coarse.snapshot["cooling_rate_density"] * coarse.grid_volume
            )
            self.assertAlmostEqual(float(fine_power), float(coarse_power), places=8)
            self.assertAlmostEqual(
                coarse.sfr_target,
                coarse.sfr_macer_baseline + coarse.sfr_closure_true,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
