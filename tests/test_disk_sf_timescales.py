"""时标统计的权重、缺帧和真实标签区间测试。"""
import unittest

import numpy as np

from VisualScripts.DataAnalysis.analyze_disk_sf_timescales import (
    branch_bounds, time_weights, weighted_quantiles,
)


class TimescaleStatisticsTests(unittest.TestCase):
    def test_massive_weight_determines_quantile_not_cell_count(self):
        actual = weighted_quantiles([1, 2, 100], [1, 1, 98])
        np.testing.assert_array_equal(actual, [100, 100, 100])

    def test_zero_sfr_has_no_artificial_timescale(self):
        self.assertTrue(np.isnan(weighted_quantiles([np.inf], [0])).all())

    def test_zero_acceleration_can_have_infinite_rotation_time(self):
        self.assertTrue(np.isinf(weighted_quantiles([np.inf], [1])).all())

    def test_missing_snapshots_do_not_bridge_gap(self):
        weights = time_weights([0, 1, 2, 10, 11], 1)
        np.testing.assert_allclose(weights, [.5, 1, .5, .5, .5])
        self.assertEqual(weights.sum(), 3)

    def test_true_branch_bounds_contain_arbitrary_subcell_labels(self):
        codes = np.array([[2, 2, 2, 3], [2, 2, 0, 1]])
        truth = np.array([[10., 20.]])
        self.assertEqual(branch_bounds(codes, truth, 2), (10., 30.))
        self.assertEqual(branch_bounds(codes, truth, 3), (0., 20.))
        self.assertEqual(branch_bounds(codes, truth, 1), (0., 20.))

    def test_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):
            time_weights([1, 1], 1)


if __name__ == "__main__":
    unittest.main()
