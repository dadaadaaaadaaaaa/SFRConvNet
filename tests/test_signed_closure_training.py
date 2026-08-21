import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Experiment.models import SFRConvNet
from Experiment.prepare_data import CachePt
from Experiment.target_transform import (
    inverse_transform_signed_closure,
    transform_signed_closure,
)
from Experiment.train import (
    GalPTDataset,
    closure_training_loss,
    compose_signed_closure_prediction,
    make_criterion,
)


def _versioned_payload(version: int):
    target_physical = torch.tensor([2.0, 3.0])
    baseline = torch.tensor([3.0, 1.0])
    closure = target_physical - baseline
    tensors = {
        "x": torch.ones(2, 2, 4, 4),
        "coord": torch.ones(2, 2, 4, 4),
        "target_sfr": torch.log10(1.0 + target_physical),
        "sfr_ks_baseline": torch.zeros(2),
        "log_mass_filter": torch.tensor([4.0, 5.0]),
        "groups": np.asarray(["a", "b"]),
        "time": torch.tensor([0.1, 0.2]),
    }
    if version >= 4:
        tensors.update(
            {
                "target_sfr_physical": target_physical,
                "macer_sfr_baseline": baseline,
                "closure_true_physical": closure,
                "closure_true": torch.asinh(closure),
            }
        )
    return {
        "schema": {"name": "macnet_sfr_cache", "version": version},
        "tensors": tensors,
        "sample_identity": {
            "sample_uid": ["sample:0", "sample:1"],
            "parent_hr_uid": ["parent:0", "parent:1"],
        },
        "model_input_contract": {
            "resolution_features": ["log_mass_filter"],
            "baseline_is_model_input": False,
            "baseline_is_output_composition": version >= 4,
        },
        "learning_contract": {
            "available_objectives": ["direct_sfr", "signed_closure"],
            "closure_transform": "signed_asinh",
            "closure_scale": 1.0,
        },
    }


class SignedClosureTransformTests(unittest.TestCase):
    def test_signed_asinh_round_trip_preserves_sign(self):
        physical = np.asarray([-1.0e4, -0.2, 0.0, 0.5, 2.0e3])
        transformed = transform_signed_closure(
            physical, transform="signed_asinh", scale=2.5
        )
        restored = inverse_transform_signed_closure(
            transformed, transform="signed_asinh", scale=2.5
        )
        np.testing.assert_allclose(restored, physical, rtol=1e-12, atol=1e-12)

    def test_final_prediction_is_explicit_baseline_plus_closure(self):
        raw = torch.asinh(torch.tensor([-1.5, 2.0]))
        baseline = torch.tensor([1.0, 3.0])
        output = compose_signed_closure_prediction(
            raw,
            baseline,
            transform="signed_asinh",
            scale=1.0,
        )
        torch.testing.assert_close(output["closure_physical"], torch.tensor([-1.5, 2.0]))
        torch.testing.assert_close(output["final_sfr_physical"], torch.tensor([-0.5, 5.0]))

    def test_negative_penalty_keeps_gradient_in_invalid_region(self):
        prediction = torch.asinh(torch.tensor([-2.0])).requires_grad_()
        criterion = make_criterion("mse", 1.0, weight_mode="none")
        loss, details = closure_training_loss(
            prediction,
            prediction.detach(),
            torch.tensor([0.25]),
            torch.tensor([0.0]),
            criterion=criterion,
            groups=torch.tensor([0]),
            transform="signed_asinh",
            scale=1.0,
            negative_weight=1.0,
        )
        self.assertGreater(float(details["negative_penalty"]), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertNotEqual(float(prediction.grad), 0.0)


class SignedClosureCacheTests(unittest.TestCase):
    def test_cache_writer_saves_y_b_and_signed_closure(self):
        class SyntheticDataset:
            coordinate = np.ones((2, 2, 4, 4), dtype=np.float32)
            sfr_target = np.asarray([2.0, 3.0])
            sfr_ks = np.asarray([0.5, 0.6])
            log_mass_filter = np.asarray([4.0, 5.0], dtype=np.float32)
            h_eff = np.ones((2, 4, 4), dtype=np.float32)
            n_eff = np.ones((2, 4, 4), dtype=np.float32)
            groups = np.asarray(["a", "b"])
            time = np.asarray([0.1, 0.2])
            sample_uid = np.asarray(["sample:0", "sample:1"])
            parent_hr_uid = np.asarray(["parent:0", "parent:1"])
            raw_data = {
                "sfr_baseline": np.asarray([3.0, 1.0]),
                "sfr_closure_true": np.asarray([-1.0, 2.0]),
            }

            def standardize(self, mean=None, std=None):
                x = np.ones((2, 2, 4, 4), dtype=np.float32)
                if mean is None:
                    return x, np.zeros(2), np.ones(2)
                return x

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            CachePt(SyntheticDataset()).to_pt(
                path,
                objective_mode="signed_closure",
                closure_transform="signed_asinh",
                closure_scale=1.0,
            )
            payload = torch.load(path, weights_only=False)

        self.assertEqual(payload["schema"]["version"], 4)
        self.assertEqual(
            payload["learning_contract"]["available_objectives"],
            ["direct_sfr", "signed_closure"],
        )
        np.testing.assert_allclose(payload["tensors"]["target_sfr_physical"], [2.0, 3.0])
        np.testing.assert_allclose(payload["tensors"]["macer_sfr_baseline"], [3.0, 1.0])
        np.testing.assert_allclose(payload["tensors"]["closure_true_physical"], [-1.0, 2.0])

    def test_version4_supports_both_objectives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(_versioned_payload(4), path)
            direct = GalPTDataset(path, objective_mode="direct_sfr")
            closure = GalPTDataset(path, objective_mode="signed_closure")

        self.assertEqual(len(direct[0]), 6)
        self.assertEqual(len(closure[0]), 8)
        self.assertAlmostEqual(float(closure[0][3]), float(torch.asinh(torch.tensor(-1.0))))
        self.assertAlmostEqual(float(closure[0][4]), 3.0)
        self.assertAlmostEqual(float(closure[0][5]), 2.0)

    def test_version3_is_direct_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(_versioned_payload(3), path)
            GalPTDataset(path, objective_mode="direct_sfr")
            with self.assertRaisesRegex(RuntimeError, "version-4 cache"):
                GalPTDataset(path, objective_mode="signed_closure")

    def test_inconsistent_physical_closure_is_rejected(self):
        payload = _versioned_payload(4)
        payload["tensors"]["closure_true_physical"][0] = 7.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Y_alpha"):
                GalPTDataset(path, objective_mode="signed_closure")

    def test_convnet_records_output_semantics(self):
        model = SFRConvNet(
            c_in=2,
            base_ch=8,
            stage_depths=(1, 1),
            pos_num_bands=2,
            prediction_mode="signed_closure",
        )
        self.assertEqual(model.prediction_mode, "signed_closure")
        with self.assertRaises(ValueError):
            SFRConvNet(c_in=2, prediction_mode="unknown")


if __name__ == "__main__":
    unittest.main()
