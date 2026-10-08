"""验证迁移的时间/日志等价性、物理内容保留和拒绝覆盖契约。"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from raw_data_processing.galaxy_data import GalData
from raw_data_processing.metadata_schema import sha256_file
from raw_data_processing.migrate_fine_time_context import (
    CachedDiskLog, _assert_preserved, migrate_fine_time_context,
)
from raw_data_processing.parse_log_file import ll_dg_from_frame
from tests.test_split_snapshot_builders import _synthetic_fine, _write_minimal_config


class CachedLogTests(unittest.TestCase):
    def test_matches_full_frame_for_windows_interpolation_duplicates_endpoints(self):
        frame = pd.DataFrame({
            "time": [2., 0., 0., 1., np.nan], "NEWstar": [2., 0., 0., 1., 9.],
            "mbh": [12., 6., 10., 10., 99.], "dt": [0.03, 0.01, 0.02, 0.02, 1.],
        })
        reader = CachedDiskLog(frame)
        for time, scope in [(0., 0.), (1., 0.2), (0.5, 0.1), (-1., 0.1), (4., 0.1), (1., 2.)]:
            with self.subTest(time=time, scope=scope):
                expected = ll_dg_from_frame(frame, time, scope)
                actual = reader.select(time, scope)
                self.assertEqual(set(expected.index), set(actual))
                np.testing.assert_allclose([actual[k] for k in expected.index], expected.to_numpy(), rtol=1e-14)

    def test_rejects_invalid_log_or_query(self):
        with self.assertRaises(ValueError):
            CachedDiskLog(pd.DataFrame({"time": [0.]}))
        reader = CachedDiskLog(pd.DataFrame({"time": [0.], "mbh": [2.], "NEWstar": [0.]}))
        with self.assertRaises(ValueError):
            reader.select(0., -1.)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.data = self.root / "data"
        self.run = "disk_galaxy_fiducial"
        self.input = self.data / self.run / "fine_sfr_v6"
        self.input.mkdir(parents=True)
        self.raw_folder = self.raw / "DiskGalaxy/Fiducial_flat"
        self.raw_folder.mkdir(parents=True)
        self.config_dir = _write_minimal_config(self.root, raw_dir=self.raw, data_dir=self.data)
        raw_config = json.loads((self.config_dir / "RawDataConfig.json").read_text())
        raw_config["disk_galaxy"]["fiducial"].update(snapshot_dt=0.00125, range=[0, 9541], time_mapping={
            "mode": "segmented_restart_absolute", "index_zero_time": 0.,
            "positive_index_offset": 0.5625, "segment_cycle_size": 999, "extra_time_steps_per_segment": 1,
        })
        (self.config_dir / "RawDataConfig.json").write_text(json.dumps(raw_config), encoding="utf-8")
        self.fields = json.loads((self.config_dir / "HdfraConfig.json").read_text())["disk_galaxy"]
        # 原始目录仅用来枚举/核验大小；测试保证不会尝试读取这些空的 HDF4 文件。
        for i in list(range(50)) + [999, 1000, 1999, 9531]:
            (self.raw_folder / f"hdfra.{i:05d}").touch()
        for i in [0, 46, 47, 48, 49, 999, 1000, 1999, 9531]:
            fine = _synthetic_fine(i)
            fine._corr = self.fields["correspondence"]
            fine.time = i * 0.00125
            source = deepcopy(fine.metadata["source"])
            source.update(time=fine.time, source_file_sha256=sha256_file(self.raw_folder / source["source_file_id"]), source_file_size_bytes=0)
            fine.attach_fine_source_metadata(source).compute_macer_sfr_products()
            fine.save_h5(self.input / f"{i:05d}.h5")
        self.log = pd.DataFrame({"time": [0., 0.6, 2., 13.], "mbh": [2., 3., 4., 10.],
                                 "dt": [0.001, 0.002, 0.003, 0.004], "NEWstar": [0., 1., 2., 3.]})
        self.log_path = self.data / f"{self.run}.parquet"
        self.log.to_parquet(self.log_path)

    def migrate(self, **kwargs):
        return migrate_fine_time_context(config_dir=self.config_dir, galaxies=[self.run], **kwargs)

    def test_boundary_times_round_trip_preserves_fields_target_units_and_identity(self):
        indices = [46, 999, 1000, 1999, 9531]
        before_hash = {i: sha256_file(self.input / f"{i:05d}.h5") for i in indices}
        with patch.object(GalData, "_load_snapshot_hdfra", side_effect=AssertionError("不能读取 raw")):
            outputs = self.migrate(snapshot_indices=indices)
        expected = [0.62, 1.81125, 1.81375, 3.06375, 12.4875]
        for path, time in zip(outputs, expected):
            old = GalData().load_h5(self.input / path.name)
            new = GalData().load_h5(path)
            _assert_preserved(old, new)
            self.assertAlmostEqual(new.time, time, places=12)
            self.assertEqual(new.metadata["source"]["time"], new.time)
            self.assertEqual(new.metadata["source"]["time_mapping"]["mode"], "segmented_restart_absolute")
            expected_log = ll_dg_from_frame(self.log, time, 0.000625)
            self.assertAlmostEqual(new._data_log["mbh"], expected_log["mbh"])
            self.assertAlmostEqual(new._data_log["dt"], expected_log["dt"])
            for mode, product in new._sfr_products.items():
                self.assertNotEqual(old._sfr_products[mode]["physics_product_uid"], product["physics_product_uid"])
                self.assertAlmostEqual(product["baseline_total"] + product["closure_true"], new.sfr_target, places=10)
                self.assertEqual(product["context"]["snapshot_time_gyr"], new.time)
            self.assertEqual(sha256_file(self.input / path.name), before_hash[int(path.stem)])

    def test_reads_parquet_once_and_default_skip_matches_raw_builder(self):
        with patch("raw_data_processing.migrate_fine_time_context.pd.read_parquet", wraps=pd.read_parquet) as read:
            outputs = self.migrate(max_snapshots=2)
            self.assertEqual(read.call_count, 1)
        self.assertEqual([p.stem for p in outputs], ["00046", "00047"])

    def test_explicit_exclusion_replaces_default_and_preserves_zero_frame(self):
        outputs = self.migrate(snapshot_indices=[0, 46], exclude_snapshot_indices=[47])
        self.assertEqual([p.stem for p in outputs], ["00000", "00046"])
        zero = GalData().load_h5(outputs[0])
        self.assertEqual(zero.time, 0.)

    def test_zero_target_and_negative_closure_are_preserved(self):
        path = self.input / "00046.h5"
        fine = GalData().load_h5(path)
        fine._data_log.update(sfr=0., sfr_target=0.)
        fine.compute_macer_sfr_products().save_h5(path)
        restored = GalData().load_h5(self.migrate(snapshot_indices=[46])[0])
        self.assertEqual(restored.sfr_target, 0.)
        self.assertGreater(restored.sfr_macer_baseline, 0.)
        self.assertLess(restored.sfr_closure_true, 0.)

    def test_failed_write_leaves_no_partial_final_file(self):
        with patch.object(GalData, "save_h5", side_effect=RuntimeError("模拟写入失败")):
            with self.assertRaises(RuntimeError):
                self.migrate(snapshot_indices=[46])
        self.assertEqual(list((self.data / self.run / "fine_sfr_v7_timefix").glob("*")), [])

    def test_same_contract_resumes_without_touching_output(self):
        outputs = self.migrate(snapshot_indices=[46])
        before = sha256_file(outputs[0])
        self.assertEqual(self.migrate(snapshot_indices=[46]), [])
        self.assertEqual(sha256_file(outputs[0]), before)
        with self.assertRaises(FileExistsError):
            self.migrate(snapshot_indices=[46], resume=False)

    def test_changed_log_refuses_existing_output(self):
        outputs = self.migrate(snapshot_indices=[46])
        before = sha256_file(outputs[0])
        self.log.loc[0, "mbh"] = 999.
        self.log.to_parquet(self.log_path)
        with self.assertRaisesRegex(ValueError, "契约不匹配"):
            self.migrate(snapshot_indices=[46])
        self.assertEqual(sha256_file(outputs[0]), before)

    def test_different_selection_cannot_mix_even_for_nonoverlapping_subset(self):
        self.migrate(snapshot_indices=[46])
        with self.assertRaisesRegex(ValueError, "契约不匹配"):
            self.migrate(snapshot_indices=[47], exclude_snapshot_indices=[0])
        self.assertFalse((self.data / self.run / "fine_sfr_v7_timefix/00047.h5").exists())

    def test_refuses_inplace_invalid_dir_overlap_and_unavailable_snapshot(self):
        for kwargs in [dict(fine_dirname="fine_sfr_v6"), dict(fine_dirname="../escape"), dict(fine_dirname="C:escape"),
                       dict(snapshot_indices=[46], exclude_snapshot_indices=[46]), dict(snapshot_indices=[1])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.migrate(**kwargs)

    def test_missing_fine_fails_without_creating_output(self):
        with self.assertRaises(FileNotFoundError):
            self.migrate(snapshot_indices=[2], exclude_snapshot_indices=[])
        self.assertFalse((self.data / self.run / "fine_sfr_v7_timefix").exists())

    def test_source_size_change_is_not_silently_accepted(self):
        (self.raw_folder / "hdfra.00046").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source_file_size_bytes"):
            self.migrate(snapshot_indices=[46])

    def test_format_and_coarse_inputs_are_rejected(self):
        with h5py.File(self.input / "00046.h5", "r+") as handle:
            handle.attrs["processed_format_version"] = 4
        with self.assertRaisesRegex(ValueError, "format-version 6"):
            self.migrate(snapshot_indices=[46])
        fine = GalData().load_h5(self.input / "00047.h5")
        fine.coarsen("spatial_block", (2, 3)).save_h5(self.input / "00047.h5")
        with self.assertRaisesRegex(ValueError, "不是 fine"):
            self.migrate(snapshot_indices=[47])


if __name__ == "__main__":
    unittest.main()
