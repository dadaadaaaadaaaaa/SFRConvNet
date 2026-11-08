"""Fiducial 分层的训练阈值、身份配对及时间块误差检查。"""
import unittest

import numpy as np
import pandas as pd

from VisualScripts.ResultAnalysis.compare_fiducial_domains import (
    sfr_partition, validate_alignment, block_delta_ci,
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


if __name__ == "__main__":
    unittest.main()
