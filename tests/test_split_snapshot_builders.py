"""验证拆分后的 raw→fine 与 fine→coarse 入口及训练 metadata 契约。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raw_data_processing.build_coarse_from_fine import build_coarse_from_fine
from raw_data_processing.build_fine_from_raw import (
    _select_snapshot_rows,
    build_fine_from_raw,
)
from raw_data_processing.galaxy_data import (
    GalData,
    GalDataSet,
    _build_snapshot_schedule,
    _validate_macer_snapshot_time,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_minimal_config(root: Path, *, raw_dir: Path, data_dir: Path) -> Path:
    config_dir = root / ".config"
    config_dir.mkdir()
    _write_json(
        config_dir / "BaseConfig.json",
        {"raw_data_dir": str(raw_dir), "data_dir": str(data_dir)},
    )
    _write_json(
        config_dir / "RawDataConfig.json",
        {
            "disk_galaxy": {
                "fiducial": {
                    "folder_name": "DiskGalaxy/Fiducial_flat",
                    "mode": "filename_index",
                    "snapshot_dt": 0.01,
                    "range": [0, 10],
                    "offset": 0,
                }
            }
        },
    )
    _write_json(
        config_dir / "HdfraConfig.json",
        {
            "disk_galaxy": {
                "correspondence": {
                    "Data-Set-2": "v1",
                    "Data-Set-3": "v2",
                    "Data-Set-4": "v3",
                    "Data-Set-5": "density",
                    "Data-Set-8": "radial_acceleration",
                    "Data-Set-9": "theta_acceleration",
                    "Data-Set-10": "gas_energy",
                    "Data-Set-13": "initial_stellar_density",
                    "Data-Set-17": "cooling_time",
                },
                "target": {
                    "dataset": "Data-Set-11",
                    "name": "sfr",
                    "reduction": "sum",
                    "quantity": "instantaneous_sfr",
                    "units": "MACER_mass_code/Gyr",
                },
            }
        },
    )
    return config_dir


def _synthetic_fine(snapshot_index: int) -> GalData:
    galaxy = GalData(ndim=2, coordinate_mode="polar")
    radius = np.geomspace(0.5, 4.0, 6)
    theta = np.linspace(0.2, 2.9, 4)
    galaxy._axes = [radius, theta]
    galaxy._axis_edges = [
        galaxy._infer_axis_edges(radius, is_radial=True),
        galaxy._infer_axis_edges(theta, is_radial=False),
    ]
    galaxy._coord = tuple(np.meshgrid(*galaxy._axes))
    galaxy._grid_volume = np.ones((4, 6), dtype=np.float64)
    galaxy._volume_source = "Data-Set-15"
    density = 1.0 + 0.02 * np.arange(24, dtype=np.float64).reshape(4, 6)
    energy = density * 100.0 / 93.0
    zeros = np.zeros_like(density)
    galaxy._data_hdfra = {
        "v1": zeros,
        "v2": zeros,
        "v3": zeros,
        "density": density,
        "radial_acceleration": np.full_like(density, -50.0),
        "theta_acceleration": zeros,
        "gas_energy": energy,
        "initial_stellar_density": 0.1 * density,
        "cooling_time": np.full_like(density, 0.05),
    }
    galaxy._corr = {}
    galaxy._data_log = {
        "mbh": 2.0,
        "dt": 1.0e-5,
        "sfr": 3.0 + snapshot_index,
        "sfr_target": 3.0 + snapshot_index,
    }
    galaxy._target_metadata = {
        "dataset": "Data-Set-11",
        "name": "sfr",
        "reduction": "sum",
        "quantity": "instantaneous_sfr",
        "units": "MACER_mass_code/Gyr",
    }
    galaxy.time = 0.01 * snapshot_index
    galaxy.attach_fine_source_metadata(
        {
            "simulation_code": "MACER2D",
            "galaxy_type": "disk_galaxy",
            "galaxy_id": "disk_galaxy_fiducial",
            "run_id": "disk_galaxy_fiducial",
            "source_series_id": "DiskGalaxy/Fiducial_flat",
            "raw_snapshot_index": snapshot_index,
            "source_file_id": f"hdfra.{snapshot_index:05d}",
            "time": galaxy.time,
            "time_units": "Gyr",
        }
    )
    return galaxy.compute_macer_sfr_products()


class CoarseBuilderContractTests(unittest.TestCase):
    def test_rh_output_loads_in_direct_and_signed_closure_training_interfaces(self):
        try:
            from Experiment.prepare_data import CachePt
            from Experiment.models import SFRConvNet
            from Experiment.train import GalPTDataset, predict_batch
        except ModuleNotFoundError as exc:
            self.skipTest(f"当前环境缺少训练依赖：{exc}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            fine_dir = data_dir / "disk_galaxy_fiducial" / "fine"
            fine_dir.mkdir(parents=True)
            config_dir = _write_minimal_config(
                root, raw_dir=root / "raw", data_dir=data_dir
            )
            for index in (0, 1):
                _synthetic_fine(index).save_h5(fine_dir / f"{index:05d}.h5")

            outputs = build_coarse_from_fine(
                config_dir=config_dir,
                galaxies=("disk_galaxy_fiducial",),
                coarsener="spatial_block",
                coarse_shape=(2, 3),
            )
            self.assertEqual(len(outputs), 2)
            restored = GalData().load_h5(outputs[0])
            self.assertEqual(restored.metadata["derivation"]["operator_id"], "spatial_block")
            self.assertAlmostEqual(
                restored.sfr_target,
                restored.sfr_macer_baseline + restored.sfr_closure_true,
                places=11,
            )

            dataset = GalDataSet().load_data(
                str(outputs[0].parent),
                require_complete_metadata=True,
                show_progress=True,
            )
            direct_path = root / "direct.pt"
            closure_path = root / "closure.pt"
            CachePt(dataset).to_pt(direct_path, objective_mode="direct_sfr")
            CachePt(dataset).to_pt(closure_path, objective_mode="signed_closure")
            direct = GalPTDataset(direct_path, objective_mode="direct_sfr")
            closure = GalPTDataset(closure_path, objective_mode="signed_closure")
            self.assertEqual(len(direct), 2)
            self.assertEqual(len(closure), 2)
            np.testing.assert_allclose(
                closure.target_sfr_physical.numpy(),
                (closure.macer_sfr_baseline + closure.closure_true_physical).numpy(),
                rtol=1e-6,
            )
            model = SFRConvNet(
                c_in=closure.x.shape[1],
                base_ch=8,
                stage_depths=(1, 1),
                pos_num_bands=2,
                use_film=False,
                prediction_mode="signed_closure",
            ).eval()
            prediction = predict_batch(
                model,
                closure.x,
                closure.coord,
                closure.conditioning_scalar,
            )
            self.assertEqual(tuple(prediction.shape), (2,))


class RawBuilderIntegrationTests(unittest.TestCase):
    def test_hdf_time_guard_uses_printed_precision_and_rejects_mismatch(self):
        class FakeDataset:
            @staticmethod
            def attributes():
                return {"long_name": "SFR (M/yr) AT TIME=6.20E-01"}

        class FakeHandle:
            @staticmethod
            def select(name):
                self = name
                return FakeDataset()

        _validate_macer_snapshot_time(FakeHandle(), 0.6204)
        with self.assertRaisesRegex(ValueError, "日志"):
            _validate_macer_snapshot_time(FakeHandle(), 0.0575)

    def test_segmented_restart_absolute_time_mapping(self):
        names = [
            "hdfra.00000",
            "hdfra.00001",
            "hdfra.00999",
            "hdfra.01000",
            "hdfra.01998",
            "hdfra.01999",
        ]
        times, indices = _build_snapshot_schedule(
            names,
            {
                "mode": "filename_index",
                "snapshot_dt": 0.00125,
                "range": [0, 9541],
                "offset": 0,
                "time_mapping": {
                    "mode": "segmented_restart_absolute",
                    "index_zero_time": 0.0,
                    "positive_index_offset": 0.5625,
                    "segment_cycle_size": 999,
                    "extra_time_steps_per_segment": 1,
                },
            },
        )
        np.testing.assert_array_equal(indices, [0, 1, 999, 1000, 1998, 1999])
        np.testing.assert_allclose(
            times,
            [0.0, 0.56375, 1.81125, 1.81375, 3.06125, 3.06375],
            rtol=0.0,
            atol=1e-14,
        )

    def test_explicit_exclusions_replace_default_stability_selection(self):
        rows = [
            (index, f"hdfra.{index:03d}", index * 1.0e-7, index)
            for index in range(50)
        ]
        selected = _select_snapshot_rows(
            rows,
            requested_indices=set(),
            excluded_indices={2, 7, 49},
            max_snapshots=None,
        )
        self.assertEqual([row[3] for row in selected[:3]], [0, 1, 3])
        self.assertNotIn(2, {row[3] for row in selected})
        self.assertNotIn(7, {row[3] for row in selected})
        self.assertNotIn(49, {row[3] for row in selected})

    def test_requested_and_excluded_snapshot_cannot_overlap(self):
        rows = [(0, "hdfra.000", 0.0, 0)]
        with self.assertRaisesRegex(ValueError, "同时"):
            _select_snapshot_rows(
                rows,
                requested_indices={0},
                excluded_indices={0},
                max_snapshots=None,
            )

    def test_raw_hdf4_to_fine_for_two_snapshots(self):
        try:
            import pandas as pd
            from pyhdf.SD import SD, SDC
        except ModuleNotFoundError as exc:
            self.skipTest(f"当前环境缺少 HDF4 集成测试依赖：{exc}")

        labels = {
            "Data-Set-5": "DENSITY",
            "Data-Set-8": "A1",
            "Data-Set-9": "A2",
            "Data-Set-10": "GAS ENERGY",
            "Data-Set-11": "SFR",
            "Data-Set-15": "VOL",
            "Data-Set-17": "COOLT",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 同时覆盖 Windows 跨盘符和 pyhdf 无法处理中文绝对路径的回归场景。
            raw_dir = root / "含中文的原始数据"
            data_dir = root / "data"
            source_dir = raw_dir / "DiskGalaxy" / "Fiducial_flat"
            source_dir.mkdir(parents=True)
            data_dir.mkdir()
            config_dir = _write_minimal_config(root, raw_dir=raw_dir, data_dir=data_dir)
            shape = (1, 4, 6)
            density = np.full(shape, 2.0, dtype=np.float64)
            energy = density * 100.0 / 93.0
            field_values = {
                "Data-Set-2": np.zeros(shape),
                "Data-Set-3": np.zeros(shape),
                "Data-Set-4": np.zeros(shape),
                "Data-Set-5": density,
                "Data-Set-8": np.full(shape, -50.0),
                "Data-Set-9": np.zeros(shape),
                "Data-Set-10": energy,
                "Data-Set-11": np.full(shape, 0.125),
                "Data-Set-13": np.full(shape, 0.2),
                "Data-Set-15": np.ones(shape),
                "Data-Set-17": np.full(shape, 0.05),
            }
            for index in (0, 1):
                previous_directory = os.getcwd()
                try:
                    os.chdir(source_dir)
                    handle = SD(
                        f"hdfra.{index:05d}",
                        SDC.WRITE | SDC.CREATE | SDC.TRUNC,
                    )
                    for name, values in field_values.items():
                        dataset = handle.create(name, SDC.FLOAT64, shape)
                        dataset[:] = np.asarray(values, dtype=np.float64)
                        if name in labels:
                            dataset.long_name = (
                                f"{labels[name]} AT TIME={index * 0.01:.2E}"
                            )
                        dataset.endaccess()
                    radial = handle.create("fakeDim2", SDC.FLOAT64, (6,))
                    radial[:] = np.geomspace(0.5, 4.0, 6)
                    radial.endaccess()
                    angular = handle.create("fakeDim1", SDC.FLOAT64, (4,))
                    angular[:] = np.linspace(0.2, 2.9, 4)
                    angular.endaccess()
                    handle.end()
                finally:
                    os.chdir(previous_directory)
            pd.DataFrame(
                {
                    "time": [0.0, 0.01],
                    "dt": [1.0e-5, 1.0e-5],
                    "NEWstar": [0.0, 0.1],
                    "mbh": [2.0, 2.0],
                }
            ).to_parquet(data_dir / "disk_galaxy_fiducial.parquet")

            outputs = build_fine_from_raw(
                config_dir=config_dir,
                galaxies=("disk_galaxy_fiducial",),
                exclude_snapshot_indices=(999,),
                max_snapshots=2,
            )
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                restored = GalData().load_h5(output)
                self.assertEqual(restored.metadata["derivation"]["operator_id"], "identity_fine")
                self.assertEqual(restored.snapshot["density"].shape, (4, 6))
                self.assertAlmostEqual(restored.sfr_target, 3.0)
                self.assertAlmostEqual(
                    restored.sfr_target,
                    restored.sfr_macer_baseline + restored.sfr_closure_true,
                    places=11,
                )
                self.assertTrue(restored.has_cell_sfr_supervision)
                np.testing.assert_array_equal(
                    restored.cell_sfr_supervision()["Y"], np.full((4, 6), 0.125))

            # 新入口自动保留逐格标签，并且旧 fine/coarse 可从原始文件补齐。
            from raw_data_processing.build_cell_sfr_supervision import augment_product, attach_from_raw
            from raw_data_processing.metadata_schema import sha256_file
            coarse_paths = build_coarse_from_fine(
                config_dir=config_dir, galaxies=("disk_galaxy_fiducial",),
                coarse_shape=(2, 3), coarse_dirname="coarse_cell", require_cell_supervision=True)
            for output in [outputs[0], coarse_paths[0]]:
                expected = GalData().load_h5(output)
                expected_maps = expected.cell_sfr_supervision()
                legacy = root / ("legacy_" + output.parent.name + ".h5")
                expected._sfr_target_map = None
                expected.save_h5(legacy)
                before = sha256_file(legacy)
                migrated = root / ("migrated_" + output.parent.name + ".h5")
                augment_product(legacy, migrated, raw_root=raw_dir)
                actual = GalData().load_h5(migrated)
                self.assertEqual(sha256_file(legacy), before)
                self.assertEqual(actual.sample_uid, expected.sample_uid)
                for name in ("Y", "B", "C_true"):
                    np.testing.assert_array_equal(actual.cell_sfr_supervision()[name], expected_maps[name])
                for name, values in expected.snapshot.items():
                    np.testing.assert_array_equal(actual.snapshot[name], values)
                with self.assertRaises(FileExistsError):
                    augment_product(legacy, migrated, raw_root=raw_dir)
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    attach_from_raw(expected, source_dir / "hdfra.00001")


if __name__ == "__main__":
    unittest.main()
