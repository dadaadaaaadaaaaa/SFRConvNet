import tempfile
import unittest
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from Experiment.prepare_data import CachePt
from Experiment.models import SFRConvNet
from VisualScripts.ResultAnalysis.analysis_utils import CachedPTDataset
from raw_data_processing.galaxy_data import GalData, GalDataSet
from raw_data_processing.metadata_schema import (
    build_fine_metadata,
    validate_complete_metadata,
)


def _fine_with_metadata(raw_snapshot_index: int = 7) -> GalData:
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
    galaxy._volume_source = "Data-Set-15"
    density = 1.0 + np.arange(16, dtype=np.float64).reshape(4, 4) / 8.0
    energy_density = 2.0 * density
    velocity = np.arange(16, dtype=np.float64).reshape(4, 4) - 7.5
    galaxy._data_hdfra = {
        "density": density,
        "gas_energy": energy_density,
        "temperature": 93.0 * energy_density / density,
        "v1": velocity,
        "v2": -velocity,
        "v3": np.zeros_like(velocity),
        "initial_stellar_density": 0.1 * density,
    }
    galaxy.time = 0.125
    galaxy._corr = {}
    galaxy._data_log = {"sfr_target": 1.5}
    galaxy._target_metadata = {
        "dataset": "Data-Set-11",
        "name": "sfr",
        "reduction": "sum",
        "quantity": "instantaneous_sfr",
        "units": "Msun/yr",
    }
    galaxy._metadata = build_fine_metadata(
        source={
            "simulation_code": "MACER2D",
            "galaxy_id": "synthetic_galaxy",
            "run_id": "synthetic_run",
            "source_series_id": "synthetic_series",
            "raw_snapshot_index": raw_snapshot_index,
            "source_file_id": f"hdfra.{raw_snapshot_index:05d}",
            "time": galaxy.time,
            "time_units": "Gyr",
        },
        snapshot_fields=tuple(galaxy.snapshot),
        axes=galaxy._axes,
        axis_edges=galaxy._axis_edges,
        coordinate_mode="polar",
        ndim=2,
        target_metadata=galaxy._target_metadata,
        volume_source=galaxy._volume_source,
    )
    validate_complete_metadata(galaxy.metadata)
    return galaxy


class MetadataContractTests(unittest.TestCase):
    def test_tampered_uid_is_rejected(self):
        fine = _fine_with_metadata()
        tampered = deepcopy(fine.metadata)
        tampered["identity"]["sample_uid"] = "macnetsample:tampered"
        with self.assertRaisesRegex(ValueError, "sample_uid"):
            validate_complete_metadata(tampered)

    def test_convnet_uses_log_mass_filter_condition(self):
        torch.manual_seed(5)
        model = SFRConvNet(
            c_in=2,
            base_ch=8,
            stage_depths=(1, 1),
            pos_num_bands=2,
            use_film=True,
        ).eval()
        x = torch.ones(2, 2, 4, 4)
        radius = torch.linspace(1.0, 4.0, 4).view(1, 1, 1, 4).expand(2, 1, 4, 4)
        theta = torch.linspace(0.2, 2.8, 4).view(1, 1, 4, 1).expand(2, 1, 4, 4)
        with torch.no_grad():
            prediction = model(
                x,
                radius,
                theta,
                torch.tensor([4.0, 6.0]),
            )
        self.assertEqual(tuple(prediction.shape), (2,))
        self.assertFalse(torch.allclose(prediction[0], prediction[1]))

    def test_uid_parentage_and_parameter_sensitivity(self):
        fine = _fine_with_metadata()
        base_mass = float(np.median(fine.gas_mass))
        first = fine.rescale_mass_adaptive((2, 2), mass_filter=3.0 * base_mass)
        second = fine.rescale_mass_adaptive((2, 2), mass_filter=4.0 * base_mass)

        self.assertEqual(first.parent_hr_uid, fine.parent_hr_uid)
        self.assertEqual(second.parent_hr_uid, fine.parent_hr_uid)
        self.assertNotEqual(first.sample_uid, fine.sample_uid)
        self.assertNotEqual(first.sample_uid, second.sample_uid)
        self.assertEqual(first.metadata["metadata_completeness"], "complete")

    def test_hdf_round_trip_preserves_contract_and_conservation(self):
        fine = _fine_with_metadata()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        coarse = fine.rescale_mass_adaptive((2, 2), mass_filter=mass_filter)
        self.assertLessEqual(
            float(coarse.coarsening_diagnostics["epsilon_conservation"]), 1e-6
        )

        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left" / "sample.h5"
            right = Path(directory) / "moved" / "renamed.h5"
            left.parent.mkdir()
            right.parent.mkdir()
            coarse.save_h5(left)
            coarse.save_h5(right)
            restored_left = GalData().load_h5(left)
            restored_right = GalData().load_h5(right)

        self.assertEqual(restored_left.sample_uid, coarse.sample_uid)
        self.assertEqual(restored_right.sample_uid, coarse.sample_uid)
        self.assertEqual(restored_left.parent_hr_uid, fine.parent_hr_uid)
        self.assertEqual(
            restored_left.coarsening_metadata["operator_parameters"]["mass_filter"],
            mass_filter,
        )
        np.testing.assert_allclose(
            restored_left.coarsening_diagnostics["epsilon_conservation"],
            coarse.coarsening_diagnostics["epsilon_conservation"],
        )

    def test_cache_contract_uses_only_log_mass_filter_as_condition(self):
        fine = _fine_with_metadata()
        mass_filter = 3.0 * float(np.median(fine.gas_mass))
        coarse = fine.rescale_mass_adaptive((2, 2), mass_filter=mass_filter)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "coarse"
            folder.mkdir()
            coarse.save_h5(folder / "00007.h5")
            dataset = GalDataSet().load_data(
                [[str(folder), "synthetic"]], require_complete_metadata=True
            )
            cache_path = Path(directory) / "ExpMetadata_train.pt"
            CachePt(dataset).to_pt(str(cache_path))
            payload = torch.load(cache_path, weights_only=False)
            analysis_dataset = CachedPTDataset(cache_path)

        self.assertEqual(payload["schema"]["version"], 3)
        self.assertEqual(
            payload["model_input_contract"]["resolution_features"],
            ["log_mass_filter"],
        )
        self.assertEqual(
            payload["model_input_contract"]["reserved_resolution_features"],
            ["h_eff", "n_eff"],
        )
        self.assertAlmostEqual(
            float(payload["tensors"]["log_mass_filter"][0]),
            np.log10(mass_filter),
            places=5,
        )
        self.assertIn("sfr_ks_baseline", payload["tensors"])
        self.assertEqual(analysis_dataset.conditioning_variable, "log_mass_filter")
        self.assertAlmostEqual(
            float(analysis_dataset.conditioning_scalar[0]),
            np.log10(mass_filter),
            places=5,
        )
        self.assertFalse(
            torch.equal(
                analysis_dataset.conditioning_scalar,
                analysis_dataset.y_baseline,
            )
        )

    def test_parent_uid_grouped_split_keeps_resolution_variants_together(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "mixed"
            folder.mkdir()
            file_index = 0
            for snapshot_index in range(4):
                fine = _fine_with_metadata(snapshot_index)
                base_mass = float(np.median(fine.gas_mass))
                variants = (
                    fine.rescale((2, 2)),
                    fine.rescale_mass_adaptive(
                        (2, 2), mass_filter=3.0 * base_mass
                    ),
                )
                for coarse in variants:
                    coarse.save_h5(folder / f"{file_index:05d}.h5")
                    file_index += 1
            dataset = GalDataSet().load_data(
                [[str(folder), "synthetic"]],
                allow_mixed_coarsening=True,
                require_complete_metadata=True,
            )
            train, validation, test = dataset.split(
                0.5,
                0.25,
                random_state=3,
                group_by_parent_uid=True,
            )

        train_parent = set(train.parent_hr_uid.tolist())
        validation_parent = set(validation.parent_hr_uid.tolist())
        test_parent = set(test.parent_hr_uid.tolist())
        self.assertFalse(train_parent & validation_parent)
        self.assertFalse(train_parent & test_parent)
        self.assertFalse(validation_parent & test_parent)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()
