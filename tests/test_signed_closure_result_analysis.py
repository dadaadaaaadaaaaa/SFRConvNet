import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Experiment.target_transform import transform_sfr
from VisualScripts.ResultAnalysis import analysis_utils


class _SyntheticDataset(SimpleNamespace):
    def __len__(self):
        return len(self.y)


class _ClosureEchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning_variable = "none"
        self.objective_mode = "signed_closure"
        self.closure_transform = "signed_asinh"
        self.closure_scale = 10.0
        self.target_metadata = {
            "quantity": "instantaneous_sfr",
            "dataset": "Data-Set-11",
            "reduction": "sum",
            "units": "MACER_mass_code/Gyr",
            "transform": "log10_1p",
            "scale": 1.0,
            "floor": 1e-9,
        }

    def forward(self, x, r, theta, conditioning_scalar):
        return x[:, 0, 0, 0]


class SignedClosureResultAnalysisTests(unittest.TestCase):
    def _dataset(self, closure_prediction):
        baseline = np.asarray([10.0, 20.0])
        closure_true = np.asarray([2.0, -3.0])
        target = baseline + closure_true
        transformed_prediction = np.arcsinh(
            np.asarray(closure_prediction, dtype=np.float64) / 10.0
        )
        x = np.zeros((2, 1, 2, 2), dtype=np.float32)
        x[:, 0, 0, 0] = transformed_prediction
        metadata = {
            "quantity": "instantaneous_sfr",
            "dataset": "Data-Set-11",
            "reduction": "sum",
            "units": "MACER_mass_code/Gyr",
            "transform": "log10_1p",
            "scale": 1.0,
            "floor": 1e-9,
        }
        return _SyntheticDataset(
            x=x,
            coord=np.zeros((2, 2, 2, 2), dtype=np.float32),
            y=transform_sfr(target, transform="log10_1p", scale=1.0),
            y_baseline=transform_sfr(
                np.asarray([1.0, 2.0]), transform="log10_1p", scale=1.0
            ),
            conditioning_scalar=np.zeros(2, dtype=np.float32),
            conditioning_variable="none",
            target_metadata=metadata,
            target_sfr_physical=target,
            macer_sfr_baseline=baseline,
            closure_true_physical=closure_true,
            time=np.asarray([0.0, 1.0]),
            group_names=np.asarray(["a", "a"]),
            sample_uid=np.asarray(["sample:0", "sample:1"]),
            parent_hr_uid=np.asarray(["hr:0", "hr:1"]),
        )

    def test_prediction_explicitly_composes_baseline_and_closure(self):
        dataset = self._dataset([2.0, -3.0])
        frame = analysis_utils.predict_dataset(
            dataset,
            _ClosureEchoModel(),
            torch.device("cpu"),
            "convnet",
            batch_size=2,
        )
        np.testing.assert_allclose(frame["closure_pred_physical"], [2.0, -3.0])
        np.testing.assert_allclose(frame["sfr_pred"], [12.0, 17.0])
        np.testing.assert_allclose(frame["sfr_pred"], frame["sfr_true"])
        np.testing.assert_allclose(
            frame["sfr_pred"],
            frame["sfr_macer_baseline"] + frame["closure_pred_physical"],
        )

    def test_negative_final_sfr_is_reported_without_clipping(self):
        dataset = self._dataset([2.0, -25.0])
        frame = analysis_utils.predict_dataset(
            dataset,
            _ClosureEchoModel(),
            torch.device("cpu"),
            "convnet",
            batch_size=2,
        )
        self.assertAlmostEqual(frame.loc[1, "sfr_pred"], -5.0, places=5)
        self.assertTrue(np.isnan(frame.loc[1, "y_pred"]))
        metrics = analysis_utils.get_physical_sfr_metrics(frame)
        self.assertEqual(metrics["NegativePredFrac"], 0.5)

    def test_saved_h5_keeps_both_baselines_and_closure(self):
        dataset = self._dataset([2.0, -3.0])
        frame = analysis_utils.predict_dataset(
            dataset,
            _ClosureEchoModel(),
            torch.device("cpu"),
            "convnet",
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "prediction.h5"
            analysis_utils.save_to_h5(
                path, frame, feats=None, x_data=dataset.x, ds_type="synthetic"
            )
            import h5py

            with h5py.File(path, "r") as handle:
                self.assertEqual(handle.attrs["objective_mode"], "signed_closure")
                self.assertIn("sfr_macer_baseline", handle)
                self.assertIn("sfr_ks_baseline", handle)
                self.assertIn("closure_pred_physical", handle)

    def test_prediction_propagates_identity_and_compact_h5_omits_inputs(self):
        dataset = self._dataset([2.0, -3.0])
        frame = analysis_utils.predict_dataset(
            dataset,
            _ClosureEchoModel(),
            torch.device("cpu"),
            "convnet",
            batch_size=2,
        )
        self.assertEqual(frame["sample_uid"].tolist(), ["sample:0", "sample:1"])
        self.assertEqual(frame["parent_hr_uid"].tolist(), ["hr:0", "hr:1"])
        frame["split"] = "test"

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "compact_prediction.h5"
            analysis_utils.save_to_h5(
                path, frame, feats=None, x_data=None, ds_type="test_cached"
            )
            import h5py

            with h5py.File(path, "r") as handle:
                self.assertNotIn("x", handle)
                self.assertNotIn("latent", handle)
                self.assertIn("sample_uid", handle)
                self.assertIn("parent_hr_uid", handle)
                self.assertIn("split", handle)

    def test_macer_code_sfr_integration_factor(self):
        dataset = self._dataset([2.0, -3.0])
        frame = analysis_utils.predict_dataset(
            dataset,
            _ClosureEchoModel(),
            torch.device("cpu"),
            "convnet",
            batch_size=2,
        )
        self.assertEqual(analysis_utils._sfr_time_integration_factor(frame), 2.5e7)


if __name__ == "__main__":
    unittest.main()
