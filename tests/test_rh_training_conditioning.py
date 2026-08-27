import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Experiment.prepare_data import CachePt
from Experiment.train import GalPTDataset, validate_convnet_conditioning


class _SyntheticConditioningDataset:
    def __init__(self, operator_ids, *, mass_filters=None):
        self.coordinate = np.ones((2, 2, 4, 4), dtype=np.float32)
        self.sfr_target = np.asarray([2.0, 3.0], dtype=np.float64)
        self.sfr_ks = np.asarray([0.5, 0.6], dtype=np.float64)
        self.h_eff = np.full((2, 4, 4), np.nan, dtype=np.float32)
        self.n_eff = np.full((2, 4, 4), np.nan, dtype=np.float32)
        self.groups = np.asarray(["a", "b"])
        self.time = np.asarray([0.1, 0.2], dtype=np.float64)
        self.sample_uid = np.asarray(["sample:0", "sample:1"])
        self.parent_hr_uid = np.asarray(["parent:0", "parent:1"])
        self.raw_data = {
            "operator_id": np.asarray(operator_ids),
            "sfr_baseline": np.asarray([3.0, 1.0], dtype=np.float64),
            "sfr_closure_true": np.asarray([-1.0, 2.0], dtype=np.float64),
        }
        self._mass_filters = (
            None
            if mass_filters is None
            else np.asarray(mass_filters, dtype=np.float64)
        )

    @property
    def log_mass_filter(self):
        if self._mass_filters is None:
            raise ValueError("R_h 没有 mass_filter。")
        if np.any(~np.isfinite(self._mass_filters)) or np.any(self._mass_filters <= 0):
            raise ValueError("mass_filter 必须正且有限。")
        return np.log10(self._mass_filters)

    def standardize(self, mean=None, std=None):
        x = np.ones((2, 2, 4, 4), dtype=np.float32)
        if mean is None:
            return x, np.zeros(2), np.ones(2)
        return x


class RhTrainingConditioningTests(unittest.TestCase):
    def test_fixed_rh_writes_v5_inactive_conditioning_and_supports_both_objectives(self):
        dataset = _SyntheticConditioningDataset(["spatial_block", "spatial_block"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rh.pt"
            cache = CachePt(dataset)
            cache.to_pt(
                path,
                objective_mode="signed_closure",
                closure_scale=1.0,
            )
            payload = torch.load(path, weights_only=False)
            direct = GalPTDataset(path, objective_mode="direct_sfr")
            closure = GalPTDataset(path, objective_mode="signed_closure")

        self.assertEqual(payload["schema"]["version"], 5)
        self.assertIn("conditioning_scalar", payload["tensors"])
        self.assertNotIn("log_mass_filter", payload["tensors"])
        np.testing.assert_array_equal(payload["tensors"]["conditioning_scalar"], 0.0)
        self.assertEqual(
            payload["model_input_contract"]["conditioning"]["name"], "none"
        )
        self.assertFalse(
            payload["model_input_contract"]["conditioning"]["active"]
        )
        self.assertEqual(payload["model_input_contract"]["resolution_features"], [])
        self.assertEqual(direct.conditioning_variable, "none")
        self.assertFalse(direct.conditioning_active)
        self.assertEqual(len(direct[0]), 6)
        self.assertEqual(len(closure[0]), 8)

    def test_rm_writer_keeps_v4_log_mass_filter_contract(self):
        mass_filters = np.asarray([1.0e4, 1.0e5])
        dataset = _SyntheticConditioningDataset(
            ["adaptive_mass_axisymmetric_v0"] * 2,
            mass_filters=mass_filters,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rm.pt"
            CachePt(dataset).to_pt(
                path,
                objective_mode="signed_closure",
                closure_scale=1.0,
            )
            payload = torch.load(path, weights_only=False)
            loaded = GalPTDataset(path, objective_mode="signed_closure")

        self.assertEqual(payload["schema"]["version"], 4)
        self.assertIn("log_mass_filter", payload["tensors"])
        self.assertNotIn("conditioning_scalar", payload["tensors"])
        np.testing.assert_allclose(
            payload["tensors"]["log_mass_filter"], np.log10(mass_filters)
        )
        self.assertEqual(
            payload["model_input_contract"]["resolution_features"],
            ["log_mass_filter"],
        )
        self.assertEqual(loaded.conditioning_variable, "log_mass_filter")
        self.assertTrue(loaded.conditioning_active)

    def test_mixed_rh_rm_cache_is_rejected(self):
        dataset = _SyntheticConditioningDataset(
            ["spatial_block", "adaptive_mass_axisymmetric_v0"],
            mass_filters=[1.0e4, 1.0e5],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pt"
            with self.assertRaisesRegex(ValueError, "不允许混合"):
                CachePt(dataset).to_pt(path)

    def test_film_switch_matches_conditioning_semantics(self):
        validate_convnet_conditioning("none", False, use_film=False)
        validate_convnet_conditioning("log_mass_filter", True, use_film=True)
        with self.assertRaisesRegex(RuntimeError, "没有活动条件变量"):
            validate_convnet_conditioning("none", False, use_film=True)
        with self.assertRaisesRegex(RuntimeError, "require --use_film"):
            validate_convnet_conditioning(
                "log_mass_filter", True, use_film=False
            )


if __name__ == "__main__":
    unittest.main()
