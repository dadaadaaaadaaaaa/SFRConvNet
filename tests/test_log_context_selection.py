import unittest

import numpy as np
import pandas as pd

from raw_data_processing.parse_log_file import (
    LOG_CONTEXT_LINEAR_INTERPOLATION,
    LOG_CONTEXT_NEAREST_ENDPOINT,
    LOG_CONTEXT_WINDOW_MEAN,
    mean_around_time_with_linear_fallback,
)


class LogContextSelectionTests(unittest.TestCase):
    def test_nonempty_window_preserves_window_mean(self):
        frame = pd.DataFrame(
            {
                "time": [0.8, 0.95, 1.05, 1.3],
                "mbh": [8.0, 9.0, 11.0, 20.0],
                "dt": [1.0, 2.0, 4.0, 8.0],
            }
        )
        result = mean_around_time_with_linear_fallback(
            frame, time=1.0, scope=0.1
        )
        self.assertAlmostEqual(float(result["mbh"]), 10.0)
        self.assertAlmostEqual(float(result["dt"]), 3.0)
        self.assertEqual(
            int(result["log_context_selection_code"]),
            LOG_CONTEXT_WINDOW_MEAN,
        )
        self.assertEqual(int(result["log_context_window_count"]), 2)

    def test_empty_window_uses_two_sided_linear_interpolation(self):
        frame = pd.DataFrame(
            {
                "time": [0.0, 2.0],
                "mbh": [8.0, 12.0],
                "dt": [1.0, 5.0],
            }
        )
        result = mean_around_time_with_linear_fallback(
            frame, time=0.5, scope=0.1
        )
        self.assertAlmostEqual(float(result["mbh"]), 9.0)
        self.assertAlmostEqual(float(result["dt"]), 2.0)
        self.assertEqual(
            int(result["log_context_selection_code"]),
            LOG_CONTEXT_LINEAR_INTERPOLATION,
        )
        self.assertEqual(int(result["log_context_window_count"]), 0)
        self.assertAlmostEqual(float(result["log_context_left_time"]), 0.0)
        self.assertAlmostEqual(float(result["log_context_right_time"]), 2.0)
        self.assertAlmostEqual(
            float(result["log_context_interpolation_weight"]), 0.25
        )

    def test_duplicate_times_are_averaged_before_interpolation(self):
        frame = pd.DataFrame(
            {
                "time": [0.0, 0.0, 2.0],
                "mbh": [6.0, 10.0, 12.0],
                "dt": [1.0, 3.0, 6.0],
            }
        )
        result = mean_around_time_with_linear_fallback(
            frame, time=1.0, scope=0.1
        )
        self.assertAlmostEqual(float(result["mbh"]), 10.0)
        self.assertAlmostEqual(float(result["dt"]), 4.0)

    def test_outside_log_range_uses_nearest_endpoint(self):
        frame = pd.DataFrame(
            {"time": [1.0, 2.0], "mbh": [8.0, 9.0], "dt": [2.0, 3.0]}
        )
        result = mean_around_time_with_linear_fallback(
            frame, time=0.0, scope=0.1
        )
        self.assertAlmostEqual(float(result["mbh"]), 8.0)
        self.assertEqual(
            int(result["log_context_selection_code"]),
            LOG_CONTEXT_NEAREST_ENDPOINT,
        )
        self.assertTrue(np.isfinite(result.to_numpy(dtype=np.float64)).all())


if __name__ == "__main__":
    unittest.main()
