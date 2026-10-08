"""检查性能优化不改变 batch 对齐、尾批、随机顺序及验证指标的物理含义。"""

import unittest
import tempfile
import io
import contextlib
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader
from Experiment.training_performance import (
    TensorBatchLoader,
    DatasetCache,
    compute_run_time_sampling_weights,
    make_training_loader,
)
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
        self.time = torch.arange(count, dtype=torch.float64)
        self.n_groups = 3

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        values = [self.x[index], self.coord[index], self.conditioning_scalar[index], self.y[index]]
        if self.objective_mode == "signed_closure":
            values += [self.macer_sfr_baseline[index], self.target_sfr_physical[index]]
        return (*values, self.groups[index], self.n_groups)


class PerformanceTests(unittest.TestCase):
    def test_run_time_weights_equalize_runs_and_follow_time_intervals(self):
        time = torch.tensor([0.0, 1.0, 4.0, 10.0, 12.0])
        groups = torch.tensor([0, 0, 0, 1, 1])
        weights = compute_run_time_sampling_weights(time, groups)
        self.assertAlmostEqual(weights[groups == 0].sum().item(), 1.0)
        self.assertAlmostEqual(weights[groups == 1].sum().item(), 1.0)
        torch.testing.assert_close(
            weights[:3],
            torch.tensor([0.125, 0.5, 0.375], dtype=torch.float64),
        )
        torch.testing.assert_close(
            weights[3:],
            torch.tensor([0.5, 0.5], dtype=torch.float64),
        )

    def test_run_time_sampling_only_changes_training_loader(self):
        data = SampleDataset(count=12)
        args = train.parse_args(["--sampling_mode", "run_time_balanced"])
        training = make_training_loader(
            data, args, torch.device("cpu"), shuffle=True
        )
        validation = make_training_loader(
            data, args, torch.device("cpu"), shuffle=False
        )
        self.assertEqual(type(training.sampler).__name__, "WeightedRandomSampler")
        self.assertEqual(type(validation.sampler).__name__, "SequentialSampler")

    def test_worker_one_is_not_divided_to_zero(self):
        args = train.parse_args(["--num_workers", "1"])
        loader = make_training_loader(SampleDataset(), args, torch.device("cpu"), shuffle=False)
        self.assertEqual(loader.num_workers, 1)
        self.assertEqual(loader.prefetch_factor, 3)

    def test_tensor_path_rejects_workers(self):
        args = train.parse_args(["--num_workers", "1", "--batch_backend", "tensor_cpu"])
        with self.assertRaises(ValueError):
            make_training_loader(SampleDataset(), args, torch.device("cpu"), shuffle=False)

    def test_dataset_cache_reuses_and_invalidates_without_sharing_augmentor(self):
        calls = []
        def dataset_type(path, *, objective_mode):
            calls.append(str(path))
            return SampleDataset(objective_mode)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Exp_train.pt"
            metadata = Path(directory) / "Exp_metadata.json"
            path.write_bytes(b"fixture")
            metadata.write_text("{}")
            store = DatasetCache(max_entries=2)
            first = store.load(dataset_type, path, objective_mode="direct_sfr")
            augmented = store.load(dataset_type, path, objective_mode="direct_sfr", augmentor=object())
            self.assertIs(first.x, augmented.x)
            self.assertIsNone(first.augmentor)
            self.assertIsNotNone(augmented.augmentor)
            self.assertEqual(len(calls), 1)
            store.load(dataset_type, path, objective_mode="signed_closure")
            self.assertEqual(len(calls), 2)
            metadata.write_text('{"changed": true}')
            renewed = store.load(dataset_type, path, objective_mode="direct_sfr")
            self.assertIsNot(renewed.x, first.x)
            path.write_bytes(b"new fixture")
            store.load(dataset_type, path, objective_mode="direct_sfr")
            self.assertEqual(len(calls), 4)
            self.assertLessEqual(len(store.entries), 2)

    def test_main_direct_and_closure_skip_test_and_close_writer(self):
        from tests.test_signed_closure_training import _versioned_payload
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for split in ("train", "val", "test"):
                    torch.save(_versioned_payload(4), Path(directory) / f"Tiny_{split}.pt")
                # 正式路径仍读取 test 并测试最佳权重；HPO 路径不加载 test。
                for objective in ("direct_sfr", "signed_closure"):
                    for skip_test in (False, True):
                        args = train.parse_args([])
                        args.cache_dir, args.data_exp = directory, "Tiny_"
                        args.model_type, args.objective_mode = "convnet", objective
                        args.use_film, args.no_save, args.no_amp = True, True, True
                        args.skip_test, args.skip_model_graph = skip_test, True
                        args.base_ch, args.batch_size, args.epochs = 2, 2, 1
                        args.device, args.num_workers = "cpu", 0
                        args.batch_backend = "tensor_cpu"
                        writer = mock.Mock()
                        with contextlib.redirect_stdout(io.StringIO()), \
                             mock.patch.object(train, "SummaryWriter", return_value=writer), \
                             mock.patch.object(train, "GalPTDataset", wraps=train.GalPTDataset) as loader:
                            result = train.main(args)
                        self.assertTrue(torch.isfinite(torch.tensor(result)))
                        paths = [str(call.args[0]) for call in loader.call_args_list]
                        self.assertEqual(any(p.endswith("test.pt") for p in paths), not skip_test)
                        writer.close.assert_called_once()
                        writer.add_graph.assert_not_called()
                # 即使在 cache 读取时报错也必须释放 SummaryWriter。
                with contextlib.redirect_stdout(io.StringIO()), \
                     mock.patch.object(train, "SummaryWriter", return_value=mock.Mock()) as factory, \
                     mock.patch.object(train, "GalPTDataset", side_effect=RuntimeError("broken cache")):
                    with self.assertRaisesRegex(RuntimeError, "broken cache"):
                        train.main(args)
                    factory.return_value.close.assert_called_once()
        finally:
            torch.set_num_threads(previous_threads)

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

    def test_direct_validation_is_per_sample_for_all_supported_losses(self):
        class Model(torch.nn.Module):
            def forward(self, x, r, theta, condition):
                return x.flatten() * 0.05
        data = SampleDataset("direct_sfr")
        writer = SimpleNamespace(add_scalar=lambda *args: None)
        for loss_name in ("l1", "mse", "huber"):
            criterion = train.make_criterion(loss_name, 0.3, weight_mode="none")
            metric, groups = train.eval_epoch(Model(), TensorBatchLoader(data, 8, shuffle=False),
                criterion, torch.device("cpu"), writer, 0, objective_mode="direct_sfr")
            expected_errors = criterion.base_criterion(data.x.flatten()*0.05, data.y.flatten())
            expected = [expected_errors[data.groups.flatten() == g].mean().item() for g in range(3)]
            torch.testing.assert_close(torch.tensor(groups), torch.tensor(expected), rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(metric, sum(expected)/3, places=6)

    def test_empty_validation_returns_nan_without_crashing(self):
        data = SampleDataset(count=0)
        # eval_epoch 需要可切换 eval 状态的模型，但空数据不应该执行 forward。
        with contextlib.redirect_stdout(io.StringIO()):
            value, groups = train.eval_epoch(torch.nn.Identity(), TensorBatchLoader(data, 8, shuffle=False),
                None, torch.device("cpu"), None, 0)
        self.assertTrue(torch.isnan(torch.tensor(value)))
        self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
