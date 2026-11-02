"""sSFR 质量来源选择与缺帧绘图的回归检查。"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from VisualScripts.DataAnalysis.plot_disk_ssfr_evolution import choose_mass_source, plot_arrays, read_field


class DiskSSFRAnalysisTests(unittest.TestCase):
    def test_raw_new_stellar_mass_keeps_signed_numerical_residuals(self):
        dataset = MagicMock()
        dataset.attributes.return_value = {'long_name': 'MNEWSTAR AT TIME=1.00E+00'}
        dataset.__getitem__.return_value = np.array([1.0, -1e-8])
        handle = MagicMock()
        handle.select.return_value = dataset
        diagnostics = {}
        total = read_field(handle, 12, 'MNEWSTAR', diagnostics)
        self.assertEqual(total, 1.0 - 1e-8)
        self.assertEqual(diagnostics['new_star_negative_cell_count'], 1)
        self.assertEqual(diagnostics['new_star_negative_mass_code'], 1e-8)

    def test_mass_selection_uses_worst_total_mass_error(self):
        raw = np.array([1e11, 2e11, 3e11])
        cached = raw * np.array([1.0, 1.00001, 1.0001])
        source, errors = choose_mass_source(cached, raw)
        self.assertEqual(source, 'fine_plus_log')
        self.assertAlmostEqual(float(errors.max()), 1e-4)
        cached[-1] = raw[-1] * 1.002
        self.assertEqual(choose_mass_source(cached, raw)[0], 'raw_hdf4')


    def test_mass_selection_rejects_zero_denominator(self):
        with self.assertRaises(ValueError):
            choose_mass_source(np.array([1.0]), np.array([0.0]))


    def test_missing_frames_break_lines_without_dropping_zero(self):
        frame = pd.DataFrame({'time_gyr': [0.0, 0.01, 0.04], 'ssfr_per_yr': [0.0, 1e-11, 2e-11]})
        x, y = plot_arrays(frame, cadence=0.01)
        self.assertEqual(x[:2], [0.0, 0.01])
        self.assertEqual(y[0], 0.0)
        self.assertTrue(np.isnan(x[2]) and np.isnan(y[2]))
        self.assertEqual(x[-1], 0.04)


    def test_production_time_excludes_relaxation(self):
        frame = pd.DataFrame({'time_gyr': [0.0, 0.5625, 0.575], 'ssfr_per_yr': [0.0, 1e-11, 2e-11]})
        x, y = plot_arrays(frame, cadence=0.0125, production=True)
        np.testing.assert_allclose(x, [0.0, 0.0125], rtol=0, atol=1e-14)
        self.assertEqual(y, [1e-11, 2e-11])
