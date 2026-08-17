import tempfile
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np

from raw_data_processing.audit_mass_coarsening import (
    _gini,
    audit_galdata,
    characterize_snapshot,
    select_diverse_snapshots,
    write_rebuild_artifacts,
)
from raw_data_processing.galaxy_data import GalData


def _synthetic_galaxy() -> GalData:
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
    density = np.ones((4, 4), dtype=np.float64)
    density[1:3, 1:3] = 5.0
    velocity = np.arange(16, dtype=np.float64).reshape(4, 4) / 10.0
    galaxy._data_hdfra = {
        "density": density,
        "gas_energy": 2.0 * density,
        "temperature": np.full_like(density, 186.0),
        "v1": velocity,
        "v2": -0.25 * velocity,
        "v3": 0.5 + 0.1 * velocity,
        "initial_stellar_density": 0.1 * density,
    }
    galaxy.time = 0.25
    galaxy._corr = {}
    galaxy._data_log = {"sfr_target": 1.5, "sfr": 1.5}
    galaxy._target_metadata = {
        "dataset": "Data-Set-11",
        "name": "sfr",
        "reduction": "sum",
        "quantity": "instantaneous_sfr",
        "units": "Msun/yr",
    }
    return galaxy


class CoarseningAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore",
            message="scipy.spatial.cKDTree is unavailable",
            category=RuntimeWarning,
        )

    def test_v0_audit_separates_support_and_output_mass(self):
        fine = _synthetic_galaxy()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        summary, cells = audit_galdata(
            fine,
            operator="axisymmetric_rm_v0",
            new_shape=(2, 2),
            mass_filter=mass_filter,
            operator_kwargs={
                "kernel": "top_hat",
                "balance_tolerance": 1e-6,
                "balance_max_iterations": 10000,
            },
        )
        self.assertEqual(summary["status"], "ok")
        self.assertTrue(summary["stored_extensive_conservation_pass"])
        self.assertLess(summary["gas_mass_scaled_relative_error"], 1e-10)
        self.assertLess(summary["stored_internal_energy_scaled_relative_error"], 1e-10)
        self.assertEqual(summary["source_enclosed_mass_below_filter_count"], 0)
        self.assertIn("overlapping adaptive support", summary["source_enclosed_mass_semantics"])
        self.assertIn("not constrained", summary["output_cell_mass_semantics"])
        self.assertEqual(len(cells), 4)
        self.assertAlmostEqual(
            sum(cell["output_cell_mass"] for cell in cells),
            float(np.sum(fine.gas_mass)),
            places=8,
        )
        self.assertIn("pz_scaled_relative_error", summary)
        self.assertIn("lz_scaled_relative_error", summary)
        self.assertIn("resolved_kinetic_energy_loss", summary)
        self.assertNotIn("total_energy_conservation_pass", summary)

    def test_v1_audit_uses_accelerated_solver_metadata(self):
        fine = _synthetic_galaxy()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        summary, cells = audit_galdata(
            fine,
            operator="axisymmetric_rm_v1",
            new_shape=(2, 2),
            mass_filter=mass_filter,
            operator_kwargs={
                "initial_neighbors": 8,
                "max_neighbors": 16,
                "max_runtime_seconds": 10.0,
                "balance_tolerance": 1e-6,
            },
        )
        self.assertTrue(summary["stored_extensive_conservation_pass"])
        self.assertEqual(summary["operator_version"], 1)
        self.assertIn("lbfgs_iterations", summary)
        self.assertIn("graph_scc_count", summary)
        self.assertEqual(len(cells), 4)

    def test_diverse_selection_is_reproducible_and_covers_runs(self):
        records = []
        for run_index, run in enumerate(("run_a", "run_b", "run_c")):
            for index in range(8):
                record = {
                    "path": f"/{run}/{index:05d}.h5",
                    "run": run,
                    "sample_uid": f"{run}:{index}",
                    "selection_status": "ok",
                    "time": float(index),
                    "log1p_sfr": float((index + run_index) % 5),
                }
                for feature_index, name in enumerate(
                    key for key in (
                        "log10_total_gas_mass",
                        "log10_density_p50_mass_weighted",
                        "log10_density_p90_mass_weighted",
                        "log10_density_p99_mass_weighted",
                        "log10_density_max",
                        "cold_gas_mass_fraction",
                        "central_gas_mass_fraction",
                        "mass_weighted_speed_rms",
                        "inflowing_gas_mass_fraction",
                        "mass_weighted_cylindrical_radius",
                        "mass_weighted_abs_z",
                        "meridional_axis_ratio",
                    )
                ):
                    record[name] = index + 0.1 * feature_index + run_index
                records.append(record)
        first = select_diverse_snapshots(records, 9, seed=7)
        second = select_diverse_snapshots(records, 9, seed=7)
        self.assertEqual(
            [record["sample_uid"] for record in first],
            [record["sample_uid"] for record in second],
        )
        self.assertEqual({record["run"] for record in first}, {"run_a", "run_b", "run_c"})

    def test_legacy_h5_can_be_selected_but_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_dir = root / "disk_galaxy_test" / "fine"
            legacy_dir.mkdir(parents=True)
            path = legacy_dir / "00003.h5"
            radius = np.asarray([1.0, 2.0, 3.0, 4.0])
            theta = np.asarray([0.3, 0.9, 1.5, 2.1])
            density = np.ones((4, 4))
            with h5py.File(path, "w") as handle:
                handle.attrs["time"] = 0.5
                coordinates = handle.create_group("coordinates")
                coordinates.create_dataset("axis_0", data=radius)
                coordinates.create_dataset("axis_1", data=theta)
                snapshot = handle.create_group("snapshot")
                snapshot.create_dataset("density", data=density)
                snapshot.create_dataset("gas_energy", data=2.0 * density)
                snapshot.create_dataset("temperature", data=186.0 * np.ones_like(density))
                for name in ("v1", "v2", "v3"):
                    snapshot.create_dataset(name, data=np.zeros_like(density))
                log_data = handle.create_group("log_data")
                log_data.create_dataset("sfr_target", data=2.5)

            descriptor = {
                "path": str(path),
                "run": "disk_galaxy_test",
                "sample_uid": "disk_galaxy_test:00003",
            }
            record = characterize_snapshot(
                descriptor,
                cold_temperature_threshold=2.0e4,
                central_radius=2.0,
            )
            self.assertEqual(record["selection_status"], "ok")
            self.assertFalse(record["audit_ready"])
            self.assertTrue(record["rebuild_required"])
            self.assertIn("selection_only", record["selection_volume_source"])

            rebuilt_root = root / "rebuilt"
            manifest = write_rebuild_artifacts(
                root,
                [record],
                rebuilt_root=rebuilt_root,
                preprocess_python="python",
            )
            self.assertEqual(
                Path(manifest[0]["path"]),
                (rebuilt_root / "disk_galaxy_test" / "00003.h5").resolve(),
            )
            script = (root / "rebuild_selected_snapshots.ps1").read_text(encoding="utf-8")
            self.assertIn("--snapshot-index 3", script)
            self.assertIn("--galaxy 'disk_galaxy_test'", script)

    def test_gini_known_values(self):
        self.assertAlmostEqual(_gini(np.ones(4)), 0.0)
        self.assertAlmostEqual(_gini(np.asarray([0.0, 0.0, 0.0, 4.0])), 0.75)


if __name__ == "__main__":
    unittest.main()
