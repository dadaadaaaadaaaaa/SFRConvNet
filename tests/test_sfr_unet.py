"""逐格 U-Net 的尺寸、梯度、物理损失、尺度隔离和 sidecar 接口验证。"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from Experiment.SFRUNet import SFRUNet
from Experiment.cell_sfr_objective import CellSFRLoss, compose_cell_sfr, fit_cell_loss_scales
from Experiment.cell_sfr_dataset import CellSFRDataset, validate_cell_splits
from Experiment.build_cell_supervision_cache import export_cell_cache
from Experiment.prepare_data import CachePt
from raw_data_processing.galaxy_data import GalDataSet
from tests.test_cell_sfr_supervision import fine_with_labels


def inputs(h=12, w=20, n=2, channels=12):
    x = torch.randn(n, channels, h, w)
    r = torch.logspace(-1, 2, w)[None, None, None, :].expand(n, 1, h, w)
    theta = torch.linspace(.1, 3., h)[None, None, :, None].expand_as(r)
    return x, r, theta


class SFRUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in SFRUNet(12).parameters()), 1156097)
        self.assertEqual(sum(p.numel() for p in SFRUNet(12, bottleneck_depth=1).parameters()), 852321)

    def test_shapes_gradients_and_serialization(self):
        for h, w in ((12, 20), (8, 8), (16, 16), (13, 21)):
            model = SFRUNet(12, base_ch=8)
            args = inputs(h, w)
            z = model(*args)
            self.assertEqual(z.shape, (2, 1, h, w))
            loss, _ = CellSFRLoss(closure_scale=1, cell_scale=1, total_scale=h*w)(
                z, torch.ones_like(z), torch.ones_like(z) * 2)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            # 两条浅层跳跃连接都保持在计算图内。
            self.assertGreater(model.encoder1[0].conv1.weight.grad.abs().sum().item(), 0)
            restored = SFRUNet(**model.model_contract()["config"])
            restored.load_state_dict(model.state_dict())
            model.eval(); restored.eval()
            torch.testing.assert_close(model(*args), restored(*args), rtol=0, atol=0)

    def test_signed_head_and_invalid_condition(self):
        model = SFRUNet(12, base_ch=8).eval()
        with torch.no_grad():
            model.head.weight.zero_(); model.head.bias.fill_(-1)
        self.assertTrue((model(*inputs()) == -1).all())
        with self.assertRaises(ValueError):
            model(*inputs(), conditioning_scalar=torch.ones(2))
        with self.assertRaises(ValueError):
            model.train()(*inputs(4, 4, n=1))

    def test_cell_loss_resolves_equal_total_ambiguity(self):
        y = torch.tensor([[[.5, .5]]], dtype=torch.float64)
        objective = CellSFRLoss(closure_scale=1, cell_scale=1, total_scale=1)
        exact, _ = objective(torch.zeros_like(y), y, y)
        wrong, details = objective(torch.asinh(torch.tensor([[[-.5, .5]]], dtype=torch.float64)), y, y)
        self.assertEqual(exact.item(), 0)
        self.assertAlmostEqual(wrong.item(), .25)
        self.assertEqual(details["total_sfr_loss"].item(), 0)

    def test_hand_computed_losses_negative_and_zero_target(self):
        z = torch.asinh(torch.tensor([[[-1., 2.]]], dtype=torch.float64)).requires_grad_()
        y = torch.zeros_like(z)
        loss, parts = CellSFRLoss(closure_scale=1, cell_scale=1, total_scale=1)(z, y, y)
        self.assertAlmostEqual(parts["cell_loss"].item(), 2.5)
        self.assertAlmostEqual(parts["total_sfr_loss"].item(), 1)
        self.assertAlmostEqual(parts["negative_penalty"].item(), .5)
        self.assertAlmostEqual(loss.item(), 4)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        result = compose_cell_sfr(z, y, closure_scale=1)
        torch.testing.assert_close(result["sfr_map_physical"], parts["sfr_map_physical"])

    def test_perfect_negative_closure(self):
        b = torch.ones(2, 1, 3, 4, dtype=torch.float64) * 3
        y = torch.zeros_like(b)
        z = torch.asinh((y-b)/2)
        loss, _ = CellSFRLoss(closure_scale=2, cell_scale=1, total_scale=1)(z, b, y)
        self.assertLess(loss.item(), 1e-25)

    def test_invalid_and_overflow_fail_explicitly(self):
        objective = CellSFRLoss(closure_scale=1, cell_scale=1, total_scale=1)
        y = torch.ones(1, 2, 2)
        with self.assertRaises(ValueError):
            objective(y, y.expand(2, 2, 2), y)
        with self.assertRaises(FloatingPointError):
            objective(y * 1000, y, y)
        with self.assertRaises(FloatingPointError):
            objective(y, -y, y)

    def test_train_only_scales_and_unit_invariance(self):
        y = torch.tensor([[[0., 2.]], [[3., 4.]]], dtype=torch.float64)
        b = y*.2
        scales = fit_cell_loss_scales(y, b, split="train")
        self.assertAlmostEqual(scales["cell_scale"], (29/4)**.5)
        self.assertAlmostEqual(scales["total_scale"], (53/2)**.5)
        self.assertAlmostEqual(scales["closure_scale"], 2.4)
        with self.assertRaises(ValueError):
            fit_cell_loss_scales(y, b, split="val")
        fallback = fit_cell_loss_scales(y*0, b*0, split="train")
        self.assertEqual(len(fallback["zero_fallbacks"]), 3)
        z = torch.zeros_like(y)
        first, _ = CellSFRLoss.from_scale_contract(scales)(z, b, y)
        other = fit_cell_loss_scales(y*.025, b*.025, split="train")
        second, _ = CellSFRLoss.from_scale_contract(other)(z, b*.025, y*.025)
        torch.testing.assert_close(first, second)

    def test_sidecar_adapter_training_step_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            products = root / "products"; products.mkdir()
            fine_with_labels().coarsen("spatial_block", (4, 4)).save_h5(products / "sample.h5")
            data = GalDataSet().load_data(str(products), require_complete_metadata=True)
            cache = root / "train.pt"
            CachePt(data).to_pt(cache, objective_mode="direct_sfr")
            sidecar = root / "cells.pt"
            export_cell_cache(cache, [products], sidecar)
            dataset = CellSFRDataset(cache, sidecar, split="train")
            scales = dataset.fit_scales()
            item = dataset[0]
            x = item["x"].unsqueeze(0).expand(2, -1, -1, -1)
            coord = item["coord"].unsqueeze(0).expand(2, -1, -1, -1)
            model = SFRUNet(x.shape[1], base_ch=8)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            z = model(x, coord[:, :1], coord[:, 1:2])
            loss, _ = CellSFRLoss.from_scale_contract(scales)(
                z, dataset.baseline.expand(2, -1, -1, -1), dataset.target.expand(2, -1, -1, -1))
            loss.backward(); optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            validate_cell_splits(train=dataset)
            other = copy.copy(dataset); other.split = "val"
            with self.assertRaises(ValueError):
                other.fit_scales()
            with self.assertRaisesRegex(ValueError, "交叉"):
                validate_cell_splits(train=dataset, val=other)
            payload = torch.load(sidecar, weights_only=False)
            payload["tensors"]["target_sfr_map_physical"][0, 0, 0] += 1
            torch.save(payload, sidecar)
            with self.assertRaisesRegex(ValueError, "摘要"):
                CellSFRDataset(cache, sidecar, split="train")


if __name__ == "__main__":
    unittest.main()
