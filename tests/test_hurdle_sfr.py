"""双头物理组合、梯度、验证阈值与真实训练入口的回归。"""

from dataclasses import replace
from pathlib import Path
import json
import math
import tempfile
import unittest

import torch

from Experiment.hurdle_sfr import HurdleCellSFRLoss, calibrate_threshold
from Experiment.SFRUNet import SFRUNet
from Experiment.train_sfr_unet import train
from VisualScripts.ResultAnalysis.analyze_sfr_unet import analyze_result, load_predictions
from tests.test_sfr_unet_training import make_cache


class HurdleTests(unittest.TestCase):
    def loss(self, mode="hurdle_direct", total_weight=1):
        return HurdleCellSFRLoss(dict(source_split="train", closure_scale=1, total_scale=2, cell_scale=1),
                                 mode=mode, amount_scale=1, total_weight=total_weight)

    def test_hand_calculation_and_gradients(self):
        x = torch.zeros((1, 1, 1, 2), dtype=torch.float64, requires_grad=True)
        logits = torch.tensor([[[[-9., 0.]]]], requires_grad=True)
        y = torch.tensor([[[[2., 0.]]]])
        outputs = {"amount_raw": x, "occurrence_logits": logits}
        loss, detail = self.loss()(outputs, torch.zeros_like(y), y)
        amount = math.log(2)
        p = 1/(1+math.exp(9))
        expected = (math.log1p(math.exp(9))+amount)/2 + (amount-2)**2 + (((p+.5)*amount-2)/2)**2
        self.assertAlmostEqual(loss.item(), expected, places=10)
        self.assertEqual(detail["positive_count"], 1)
        self.loss(total_weight=0)(outputs, y, y)[0].backward()
        self.assertLess(x.grad.flatten()[0], 0)
        self.assertEqual(x.grad.flatten()[1], 0)
        self.assertLess(logits.grad.flatten()[0], 0)

    def test_all_zero_and_closed_gate(self):
        b = torch.ones(1, 1, 2, 2)
        outputs = {"amount_raw": torch.zeros_like(b, requires_grad=True),
                   "occurrence_logits": torch.zeros_like(b, requires_grad=True)}
        loss, d = self.loss("hurdle_closure")(outputs, b, torch.zeros_like(b))
        self.assertEqual(d["positive_loss_sum"], 0)
        self.assertTrue(torch.allclose(d["amount_map"], torch.full_like(d["amount_map"], math.log1p(math.e))))
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["amount_raw"].grad).all())
        with self.assertRaises(ValueError):
            self.loss()(outputs, b, b[:, :, :1])

    def test_threshold_ties_and_split(self):
        scores = torch.tensor([.1, .4, .4, .9, .3], dtype=torch.float64)
        y = torch.tensor([0, 0, 0, 0, 1])
        for fpr, expected in ((0, 0), (.5, .25), (.75, .75), (1, 1)):
            cal = calibrate_threshold(scores, y, fpr, split="val")
            self.assertEqual(cal["actual_fpr"], expected)
        with self.assertRaises(ValueError):
            calibrate_threshold(scores, y, .5, split="out")

    def test_contract_shapes_and_backbone_gradient(self):
        torch.manual_seed(11)
        model = SFRUNet(3, base_ch=4, hurdle=True)
        x = torch.randn(2, 3, 9, 13)
        r = torch.ones_like(x[:, :1])
        outputs = model(x, r, r)
        self.assertEqual(outputs["amount_raw"].shape, r.shape)
        for key in ("amount_raw", "occurrence_logits"):
            grad, = torch.autograd.grad(outputs[key].sum(), model.stem[0].weight, retain_graph=True)
            self.assertGreater(grad.abs().sum().item(), 0)
        restored = SFRUNet(**model.model_contract()["config"])
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_train_restore_export_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            config = replace(make_cache(folder, "exp2"), model_mode="hurdle_direct", stop_after_epochs=1)
            run = Path(train(config)["output_dir"])
            report = train(replace(config, stop_after_epochs=0, resume=str(run/"last.pt")))
            self.assertEqual(report["status"], "complete")
            analyze_result(run, plots=False)
            payload = torch.load(run/"out_predictions.pt", weights_only=False)
            hard = payload["hard_map"]
            closed = payload["probability_map"] < payload["calibration"]["threshold"]
            self.assertTrue(torch.equal(hard[closed], torch.zeros_like(hard[closed])))
            torch.testing.assert_close(payload["hard_closure_map"][closed], -payload["baseline_map"][closed])
            self.assertEqual(payload["calibration"]["source_split"], "val")
            payload["hard_map"] = hard + 1
            bad = Path(folder)/"corrupt.pt"
            torch.save(payload, bad)
            identity = json.loads((run/"split_manifest.json").read_text())["out"]
            with self.assertRaisesRegex(ValueError, "恒等式"):
                load_predictions(bad, identity=identity, expected_groups=set(payload["groups"]))
