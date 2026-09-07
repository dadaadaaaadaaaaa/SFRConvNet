"""检查性能优化不改变 batch 对齐、尾批、随机顺序及验证指标的物理含义。"""

import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader
from Experiment.training_performance import TensorBatchLoader
from Experiment import train


class SampleDataset(Dataset):
    def __init__(self, objective="signed_closure", count=19):
        self.objective_mode = objective
        self.augmentor = None
        self.x = torch.arange(count, dtype=torch.float32).view(-1, 1, 1, 1)
        self.coord = self.x.expand(-1, 2, 1, 1)
        self.conditioning_scalar = self.x.flatten().view(-1, 1)
        self.y = self.conditioning_scalar / 10
        self.macer_sfr_baseline = self.y + 2
        self.target_sfr_physical = self.y + 4
        self.groups = (torch.arange(count) % 3).view(-1, 1)
        self.n_groups = 3

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        values = [self.x[index], self.coord[index], self.conditioning_scalar[index], self.y[index]]
        if self.objective_mode == "signed_closure":
            values += [self.macer_sfr_baseline[index], self.target_sfr_physical[index]]
        return (*values, self.groups[index], self.n_groups)


class PerformanceTests(unittest.TestCase):
    def test_batch_order_rng_tail_and_field_alignment(self):
        for objective in ("direct_sfr", "signed_closure"):
            data = SampleDataset(objective)
            old = DataLoader(data, batch_size=8, shuffle=True)
            new = TensorBatchLoader(data, 8, shuffle=True)
            torch.manual_seed(27)
            before = [list(old) for _ in range(3)]
            before_rng = torch.get_rng_state()
            torch.manual_seed(27)
            after = [list(new) for _ in range(3)]
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            for a_epoch, b_epoch in zip(before, after):
                self.assertEqual(len(b_epoch[-1][0]), 3)
                for a_batch, b_batch in zip(a_epoch, b_epoch):
                    for a, b in zip(a_batch[:-1], b_batch[:-1]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertEqual(a_batch[-1][0], b_batch[-1])

    def test_batch_loader_rejects_augmentation(self):
        data = SampleDataset()
        data.augmentor = object()
        with self.assertRaises(ValueError):
            TensorBatchLoader(data, 8, shuffle=True)

    def test_vector_validation_keeps_equal_domain_weight_and_missing_domain(self):
        class Model(torch.nn.Module):
            def forward(self, x, r, theta, condition):
                return x.flatten() / 10
        data = SampleDataset(count=19)
        data.groups[:] = 0
        data.groups[-2:] = 2  # 星系样本数不均，且中间星系没有样本。
        writer = SimpleNamespace(add_scalar=lambda *args: None)
        loader = TensorBatchLoader(data, 8, shuffle=False)
        criterion = train.make_criterion("l1", 1, weight_mode="none")
        metric, groups = train.eval_epoch(Model(), loader, criterion, torch.device("cpu"),
                                         writer, 0, objective_mode="signed_closure",
                                         closure_transform="signed_asinh", closure_scale=2,
                                         negative_sfr_weight=1)
        pred = data.macer_sfr_baseline.flatten() + 2 * torch.sinh(data.x.flatten() / 10)
        errors = ((pred-data.target_sfr_physical.flatten())/2).abs()
        expected = (errors[:-2].mean() + errors[-2:].mean()) / 2
        self.assertAlmostEqual(metric, expected.item(), places=6)
        self.assertTrue(torch.isnan(torch.tensor(groups[1])))


if __name__ == "__main__":
    unittest.main()
