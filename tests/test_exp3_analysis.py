"""验证 Exp3 结果分析的目标模式、物理指标和时间积分口径。"""

import unittest

import numpy as np
import pandas as pd

from VisualScripts.ResultAnalysis.analyze_exp3 import (
    _physical_metrics,
    _total_period_sfr_underestimate_percent,
    _validate_exp3_contract,
    _validate_objective_mode,
)


class _DatasetStub:
    group_names = np.asarray(["disk_galaxy_torus_100yr"] * 3)


class Exp3AnalysisTests(unittest.TestCase):
    def test_direct_and_closure_objectives_are_supported(self):
        self.assertEqual(
            _validate_objective_mode({"objective_mode": "direct_sfr"}),
            "direct_sfr",
        )
        self.assertEqual(
            _validate_objective_mode({"objective_mode": "signed_closure"}),
            "signed_closure",
        )

    def test_unknown_objective_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "只支持"):
            _validate_objective_mode({"objective_mode": "unknown"})

    def test_physical_metrics(self):
        frame = pd.DataFrame(
            {"sfr_true": [1.0, 2.0, 3.0], "sfr_pred": [1.0, 2.0, 4.0]}
        )
        metrics = _physical_metrics(frame)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["MAE"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["R2"], 0.5)
        self.assertEqual(set(metrics), {"RMSE", "MAE", "R2"})

    def test_total_period_error_uses_physical_time_intervals(self):
        frame = pd.DataFrame(
            {
                "time": [0.0, 1.0, 3.0],
                "sfr_true": [2.0, 2.0, 2.0],
                "sfr_pred": [1.0, 1.0, 1.0],
            }
        )
        self.assertAlmostEqual(_total_period_sfr_underestimate_percent(frame), 50.0)

    def test_exp3_contract_requires_torus_only_test(self):
        metadata = {
            "split_strategy": "random_snapshot_75_25_torus_test",
            "train_groups": ["zou_a", "zou_b"],
            "validation_groups": ["zou_a", "zou_b"],
            "test_groups": ["disk_galaxy_torus_100yr"],
        }
        self.assertEqual(
            _validate_exp3_contract(metadata, _DatasetStub()),
            "disk_galaxy_torus_100yr",
        )


if __name__ == "__main__":
    unittest.main()
