import unittest

from VisualScripts.ResultAnalysis.analysis_utils import _resolve_mirror_mode


class AnalysisMirrorModeTests(unittest.TestCase):
    def test_native_shape_disables_mirror(self):
        self.assertFalse(_resolve_mirror_mode("match_reference", (8, 8), (8, 8)))

    def test_legacy_double_width_enables_mirror(self):
        self.assertTrue(_resolve_mirror_mode("match_reference", (8, 8), (8, 16)))

    def test_incompatible_shape_is_rejected(self):
        with self.assertRaises(RuntimeError):
            _resolve_mirror_mode("match_reference", (8, 8), (16, 8))


if __name__ == "__main__":
    unittest.main()
