"""逐格标签保守归约、HDF5 契约、旧标量接口及 cache UID 对齐回归。"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from tests.test_sfr_supervision_products import _galaxy
from raw_data_processing.coarsening_common import block_reduce_sum
from raw_data_processing.galaxy_data import GalData, GalDataSet
from raw_data_processing.metadata_schema import sha256_file
from Experiment.prepare_data import CachePt
from Experiment.train import GalPTDataset
from Experiment.build_cell_supervision_cache import export_cell_cache


def fine_with_labels(zero=False):
    """非均匀体积刻意与标签无关，能检出重复乘体积的错误。"""
    galaxy = _galaxy()
    galaxy._grid_volume = np.geomspace(0.1, 100, 16).reshape(4, 4)
    y = np.arange(16, dtype=np.float64).reshape(4, 4)
    y = y * (0.0 if zero else 3.0 / y.sum())
    galaxy._data_log.update(sfr_target=float(y.sum()), sfr=float(y.sum()))
    galaxy.compute_macer_sfr_products()
    return galaxy.attach_cell_sfr_target(y)


class CellSFRTests(unittest.TestCase):
    def test_nonuniform_volume_rh_reduction_roundtrip_and_both_modes(self):
        fine = fine_with_labels()
        coarse = fine.coarsen("spatial_block", (2, 2))
        expected = block_reduce_sum(fine.cell_sfr_supervision()["Y"], (2, 2))
        for mode in ("data_compatible", "paper_physical"):
            maps = coarse.cell_sfr_supervision(mode)
            np.testing.assert_array_equal(maps["Y"], expected)
            np.testing.assert_array_equal(maps["C_true"], maps["Y"] - maps["B"])
            self.assertAlmostEqual(maps["B"].sum(), coarse.sfr_products[mode]["baseline_total"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fine.h5"
            fine.save_h5(path)
            restored = GalData().load_h5(path)
            np.testing.assert_array_equal(restored.cell_sfr_supervision()["Y"], fine.cell_sfr_supervision()["Y"])
            path = Path(directory) / "coarse.h5"
            coarse.save_h5(path)
            restored = GalData().load_h5(path)
            self.assertEqual(restored.metadata, coarse.metadata)
            self.assertEqual(restored.sample_uid, coarse.sample_uid)
            np.testing.assert_array_equal(restored.cell_sfr_supervision()["Y"], expected)
            self.assertNotIn("Y_alpha_cell", restored.snapshot)

    def test_zero_labels_and_negative_closure_are_supported(self):
        galaxy = fine_with_labels(zero=True)
        maps = galaxy.cell_sfr_supervision()
        self.assertEqual(maps["Y"].sum(), 0)
        self.assertTrue(np.any(maps["C_true"] < 0))
        self.assertAlmostEqual(maps["C_true"].sum(), galaxy.sfr_closure_true)

    def test_bad_shape_negative_nonfinite_and_wrong_sum_are_rejected(self):
        galaxy = fine_with_labels()
        original = galaxy.cell_sfr_supervision()["Y"]
        for bad in (original.T + 1, original[:2], original - 2, original * np.nan):
            with self.assertRaises(ValueError):
                galaxy.attach_cell_sfr_target(bad)
        np.testing.assert_array_equal(galaxy.cell_sfr_supervision()["Y"], original)

    def test_load_rejects_permutation_even_if_global_sum_is_preserved(self):
        galaxy = fine_with_labels()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            galaxy.save_h5(path)
            with h5py.File(path, "a") as handle:
                dataset = handle["sfr_supervision/canonical/Y_alpha_cell"]
                dataset[:] = dataset[()][::-1]
            with self.assertRaisesRegex(ValueError, "摘要"):
                GalData().load_h5(path)

    def test_load_rejects_missing_or_tampered_maps(self):
        galaxy = fine_with_labels()
        for name in ("canonical/B_alpha_cell", "data_compatible/cell_sfr_closure_true"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sample.h5"
                galaxy.save_h5(path)
                with h5py.File(path, "a") as handle:
                    handle[f"sfr_supervision/{name}"][0, 0] += 1
                with self.assertRaises(ValueError):
                    GalData().load_h5(path)

    def test_old_scalar_product_and_rm_do_not_fabricate_local_labels(self):
        old = _galaxy()
        self.assertFalse(old.has_cell_sfr_supervision)
        with self.assertRaisesRegex(ValueError, "Data-Set-11"):
            old.cell_sfr_supervision()
        fine = fine_with_labels()
        coarse = fine.coarsen("axisymmetric_rm_v0", (2, 2),
                              mass_filter=0.1 * float(np.min(fine.gas_mass)),
                              balance_max_iterations=30000)
        self.assertFalse(coarse.has_cell_sfr_supervision)
        self.assertNotIn("cell_supervision", coarse.metadata)
        self.assertAlmostEqual(coarse.sfr_target, coarse.sfr_macer_baseline + coarse.sfr_closure_true)

    def test_scalar_cache_unchanged_and_sidecar_preserves_oversampled_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = root / "products"
            products.mkdir()
            galaxy = fine_with_labels().coarsen("spatial_block", (2, 2))
            galaxy.save_h5(products / "sample.h5")
            dataset = GalDataSet().load_data(str(products), require_complete_metadata=True)
            cache_path = root / "scalar.pt"
            CachePt(dataset).to_pt(cache_path, objective_mode="signed_closure")
            for mode in ("direct_sfr", "signed_closure"):
                loaded = GalPTDataset(cache_path, objective_mode=mode)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded.x.shape[-2:], (2, 2))
            self.assertFalse(any("cell" in name or "closure" in name or "baseline" in name
                                 for name in dataset.input_channels))
            payload = torch.load(cache_path, weights_only=False)
            for name, values in payload["tensors"].items():
                payload["tensors"][name] = np.concatenate([values, values], axis=0)
            for name, values in payload["sample_identity"].items():
                payload["sample_identity"][name] = values + values
            torch.save(payload, cache_path)
            before = sha256_file(cache_path)
            output = root / "cell.pt"
            export_cell_cache(cache_path, [products], output)
            self.assertEqual(sha256_file(cache_path), before)
            cell = torch.load(output, weights_only=False)
            self.assertEqual(cell["sample_identity"], payload["sample_identity"])
            self.assertEqual(cell["source_cache"]["sha256"], before)
            for name, scalar in (("target_sfr_map_physical", "target_sfr_physical"),
                                 ("macer_sfr_baseline_map", "macer_sfr_baseline"),
                                 ("closure_true_map_physical", "closure_true_physical")):
                np.testing.assert_allclose(cell["tensors"][name].sum(dim=(1, 2)), payload["tensors"][scalar])
            with self.assertRaises(FileExistsError):
                export_cell_cache(cache_path, [products], output)
            payload["tensors"]["macer_sfr_baseline"] += 1
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "总量"):
                export_cell_cache(cache_path, [products], root / "bad.pt")


if __name__ == "__main__":
    unittest.main()
