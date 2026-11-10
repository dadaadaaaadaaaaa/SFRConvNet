"""Fiducial 分层的训练阈值、身份配对及时间块误差检查。"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from VisualScripts.ResultAnalysis.compare_fiducial_domains import (
    sfr_partition, validate_alignment, block_delta_ci, audit_historical_cnn, load_cache,
)


class DomainTests(unittest.TestCase):
    def test_thresholds_use_train_and_include_endpoints(self):
        part, low, high = sfr_partition(np.array([0, 1, 2, 3, 100]), np.array([1, 2, 3]))
        self.assertEqual((low, high), (1, 3))
        self.assertEqual(part.tolist(), ["low", "inside", "inside", "inside", "high"])
        part, _, _ = sfr_partition(np.array([34.]), np.array([.24, 84.]))
        self.assertEqual(part.tolist(), ["inside"])

    def test_alignment_uses_identity_not_row_order(self):
        ref = pd.DataFrame({"uid": ["a", "b"], "parent": ["pa", "pb"],
                            "time": [1., 2.], "Y": [1., 3.], "B": [.5, .7]})
        self.assertEqual(validate_alignment(ref.iloc[::-1], ref).uid.tolist(), ["a", "b"])
        bad = ref.copy()
        bad.loc[0, "parent"] = "wrong"
        with self.assertRaises(ValueError):
            validate_alignment(bad, ref)
        with self.assertRaises(ValueError):
            validate_alignment(pd.concat([ref, ref.iloc[:1]]), ref)

    def test_paired_block_ci_preserves_difference(self):
        time = np.arange(100)*.1
        true = np.ones(100)
        low, high, count = block_delta_ci(time, true, true+1, true+3,
                                          np.arange(100) % 2 == 0, .5, 500)
        self.assertAlmostEqual(low, -2.)
        self.assertAlmostEqual(high, -2.)
        self.assertGreater(count, 5)

    def test_insufficient_time_blocks_are_not_confidence(self):
        t = np.arange(10)*.01
        low, high, count = block_delta_ci(t, t, t, t, np.ones(10, bool), .5, 100)
        self.assertTrue(np.isnan(low) and np.isnan(high))
        self.assertEqual(count, 1)
        low, high, count = block_delta_ci(t, t, t, t, np.zeros(10, bool), .5, 100)
        self.assertTrue(np.isnan(low) and np.isnan(high))
        self.assertEqual(count, 0)

    def test_historical_comparison_audits_inputs_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/".cache"
            cache.mkdir()
            tensors = {"x": np.zeros((2, 1, 1, 1)), "coord": np.zeros((2, 2, 1, 1)),
                "groups": np.array(["held", "held"]), "time": np.array([1., 2.]),
                "target_sfr_physical": np.array([40., 80.]),
                "macer_sfr_baseline": np.array([20., 40.]), "closure_true_physical": np.array([20., 40.])}
            torch.save({"tensors": tensors, "sample_identity": {"sample_uid": ["oa", "ob"],
                "parent_hr_uid": ["pa", "pb"]}}, cache/"old_out.pt")
            torch.save({"sample_identity": {"sample_uid": ["ta", "tb"],
                "parent_hr_uid": ["pta", "ptb"]}}, cache/"old_train.pt")
            for name in ["old_", "new_"]:
                np.savez(cache/f"{name}norm.npz", mean=np.zeros((1, 1, 1, 1)), std=np.ones((1, 1, 1, 1)))
            hyper = {"data_exp": "old_", "data_split_metadata": {"train_groups": ["train"]},
                     "input_channels": ["density"]}
            metadata = {"cache_prefix": "new_", "train_groups": ["train"], "input_channels": ["density"]}
            train = pd.DataFrame({"uid": ["ta"], "parent": ["pta"]})
            with patch("VisualScripts.ResultAnalysis.compare_fiducial_domains.ROOT", root):
                out, x, coord = load_cache("out", "old_")
                result = audit_historical_cnn(hyper, metadata, out, train, lambda _: None, x, coord)
                self.assertEqual(result["train_overlap_N"], 1)
                self.assertEqual(result["historical_only_train_N"], 1)
                with self.assertRaises(ValueError):
                    audit_historical_cnn(hyper, metadata, out, train, lambda _: None, x+1, coord)
                wrong = out.copy()
                wrong.loc[0, "Y"] += 1
                with self.assertRaises(ValueError):
                    audit_historical_cnn(hyper, metadata, wrong, train, lambda _: None, x, coord)
                with self.assertRaises(ValueError):
                    audit_historical_cnn(hyper, metadata, out, pd.DataFrame({"uid": ["oa"], "parent": ["pa"]}), lambda _: None, x, coord)


if __name__ == "__main__":
    unittest.main()
