"""Exp3.1/3.2 的 cache、训练与分析接口及无泄漏回归测试。"""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from Experiment.geometry_contract import validate_geometry_ablation
from Experiment.models import SFRConvNet
from Experiment.prepare_data import CachePt, _prepare_exp3_torus_ood, _ZOU_FIVE_GROUPS, _TORUS_TEST_GROUP
from Experiment.train import GalPTDataset, compose_signed_closure_prediction, predict_batch
from tests.test_geometry_input_features import _galaxy


class _NullWriter:
    """smoke test 不产生真实 TensorBoard 日志，但不跳过训练和评估。"""
    def __init__(self, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class GeometryAblationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train_entries = []
        self.test_entries = []
        for number, group in enumerate((*_ZOU_FIVE_GROUPS, _TORUS_TEST_GROUP)):
            folder = self.root / group
            folder.mkdir()
            for index in range(1, 5):
                galaxy = _galaxy(index=index + number * 10, target=3.0 + index)
                # 测试域改变几何，验证统计量仍只来自训练 split。
                if group == _TORUS_TEST_GROUP:
                    galaxy._axes[0] = galaxy._axes[0] * 0.001
                    galaxy._axis_edges[0] = galaxy._axis_edges[0] * 0.001
                    galaxy._coord = tuple(np.meshgrid(*galaxy._axes))
                galaxy.save_h5(folder / f"snapshot_{index:03d}.h5")
            entry = (str(folder), group)
            (self.test_entries if group == _TORUS_TEST_GROUP else self.train_entries).append(entry)
        (self.root / ".cache").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _prepare(self, mode):
        args = argparse.Namespace(seed=42, cache_tag=mode, exclude_channels=("initial_stellar_density",), drop_zero_targets=False)
        original = Path.cwd()
        try:
            os.chdir(self.root)
            prefix = _prepare_exp3_torus_ood(
                self.train_entries, self.test_entries, args=args,
                load_options={"exclude_channels": args.exclude_channels, "geometry_input_mode": mode},
                transform_options={"objective_mode": "signed_closure", "closure_transform": "signed_asinh", "closure_scale": "auto"},
            )
            return str(self.root / prefix)
        finally:
            os.chdir(original)

    def test_same_split_fields_and_labels_with_training_only_geometry_statistics(self):
        original_prefix = self._prepare("none")
        for mode, increment in (("absolute_radius", 1), ("absolute_radius_widths", 3)):
            prefix = self._prepare(mode)
            for split in ("train", "val", "test"):
                original = torch.load(original_prefix + split + ".pt", weights_only=False)
                payload = torch.load(prefix + split + ".pt", weights_only=False)
                self.assertEqual(payload["schema"]["version"], 6)
                self.assertEqual(payload["sample_identity"], original["sample_identity"])
                old_x, new_x = original["tensors"]["x"], payload["tensors"]["x"]
                self.assertEqual(new_x.shape[1], old_x.shape[1] + increment)
                np.testing.assert_array_equal(new_x[:, :old_x.shape[1]], old_x)
                for field in ("coord", "target_sfr", "target_sfr_physical", "macer_sfr_baseline", "closure_true_physical", "time"):
                    np.testing.assert_array_equal(payload["tensors"][field], original["tensors"][field])
                names = payload["model_input_contract"]["input_channels"]
                for name in payload["model_input_contract"]["geometry_inputs"]["channels"]:
                    self.assertFalse(payload["preprocessing_contract"]["log_transform_mask"][names.index(name)])
            metadata = json.loads(Path(prefix + "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["sample_counts"], {"train": 15, "val": 5, "test": 4, "out": 0})
            self.assertEqual(metadata["geometry_input_contract"]["mode"], mode)
            train = GalPTDataset(prefix + "train.pt")
            test = GalPTDataset(prefix + "test.pt")
            self.assertAlmostEqual(float(train.x[:, -increment].mean()), 0.0, places=5)
            self.assertGreater(abs(float(test.x[:, -increment].mean())), 1.0)
            norm = np.load(prefix + "norm.npz")
            self.assertEqual(json.loads(str(norm["geometry_input_contract_json"]))["mode"], mode)
            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                self._prepare(mode)

    def test_both_objectives_forward_backward_and_analysis_roundtrip(self):
        from VisualScripts.ResultAnalysis.analysis_utils import CachedPTDataset, HDF5Dataset
        for mode in ("absolute_radius", "absolute_radius_widths"):
            prefix = self._prepare(mode)
            analysis_dataset = CachedPTDataset(Path(prefix + "test.pt"))
            self.assertEqual(analysis_dataset.geometry_input_contract["mode"], mode)
            norm = np.load(prefix + "norm.npz")
            hdf5_dataset = HDF5Dataset(
                self.test_entries[0][0], reference_dataset=analysis_dataset,
                mean=norm["mean"], std=norm["std"],
            )
            np.testing.assert_allclose(hdf5_dataset.x, analysis_dataset.x, rtol=1e-6, atol=1e-6)
            self.assertEqual(hdf5_dataset.geometry_input_contract, analysis_dataset.geometry_input_contract)
            for objective in ("direct_sfr", "signed_closure"):
                dataset = GalPTDataset(prefix + "train.pt", objective_mode=objective)
                validate_geometry_ablation({"train": dataset}, mode)
                model = SFRConvNet(dataset.x.shape[1], base_ch=4, stage_depths=(1,), pos_num_bands=2, prediction_mode=objective)
                prediction = predict_batch(model, dataset.x[:2], dataset.coord[:2], dataset.conditioning_scalar[:2])
                if objective == "signed_closure":
                    prediction = compose_signed_closure_prediction(
                        prediction, dataset.macer_sfr_baseline[:2], transform=dataset.closure_transform, scale=dataset.closure_scale,
                    )["final_sfr_physical"]
                loss = prediction.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.stem[0].weight.grad)
                self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())
                with self.assertRaisesRegex(RuntimeError, "消融要求"):
                    validate_geometry_ablation({"train": dataset}, "none")

    def test_payload_sidecar_and_duplicate_log_transform_mismatch_rejected(self):
        prefix = self._prepare("absolute_radius")
        path = prefix + "train.pt"
        original = torch.load(path, weights_only=False)
        original["preprocessing_contract"]["log_transform_mask"][-1] = True
        torch.save(original, path)
        with self.assertRaisesRegex(RuntimeError, "不得再次"):
            GalPTDataset(path)
        original["preprocessing_contract"]["log_transform_mask"][-1] = False
        original["model_input_contract"]["geometry_inputs"]["reference_length_kpc"] = 1000.0
        torch.save(original, path)
        with self.assertRaisesRegex(RuntimeError, "几何输入契约不一致"):
            GalPTDataset(path)

    def test_missing_single_channel_mask_and_inactive_v6_analysis_are_rejected(self):
        from Experiment.geometry_contract import validate_geometry_contract
        from raw_data_processing.geometry_features import geometry_input_contract
        from VisualScripts.ResultAnalysis.analysis_utils import CachedPTDataset
        with self.assertRaisesRegex(RuntimeError, "缺少完整"):
            validate_geometry_contract(geometry_input_contract("absolute_radius"), ["log10_radius_kpc"], None)
        prefix = self._prepare("absolute_radius")
        path = Path(prefix + "test.pt")
        payload = torch.load(path, weights_only=False)
        payload["model_input_contract"]["geometry_inputs"] = geometry_input_contract("none")
        torch.save(payload, path)
        with self.assertRaisesRegex(RuntimeError, "必须声明活动"):
            CachedPTDataset(path)

    def test_one_epoch_cpu_checkpoint_and_analysis_for_all_four_variants(self):
        from Experiment import train
        from VisualScripts.ResultAnalysis.analysis_utils import CachedPTDataset, load_analysis_model
        for mode in ("absolute_radius", "absolute_radius_widths"):
            prefix = self._prepare(mode)
            for objective in ("direct_sfr", "signed_closure"):
                args = train.parse_args([
                    "--cache_dir", str(self.root / ".cache"),
                    "--data_exp", Path(prefix).name,
                    "--exp_name", f"smoke_{mode}_{objective}",
                    "--expected_geometry_input_mode", mode,
                    "--objective_mode", objective, "--model_type", "convnet",
                    "--base_ch", "4", "--stage_depths", "1", "--pos_num_bands", "2",
                    "--pos_max_freq", "4", "--batch_size", "5", "--epochs", "1",
                    "--device", "cpu", "--num_workers", "0", "--no_amp",
                    "--sampling_mode", "run_time_balanced", "--batch_backend", "tensor_cpu",
                    "--skip_model_graph", "--loss", "l1", "--seed", "42",
                ])
                original = Path.cwd()
                try:
                    os.chdir(self.root)
                    with patch.object(train, "SummaryWriter", _NullWriter):
                        metric = train.main(args)
                finally:
                    os.chdir(original)
                self.assertTrue(np.isfinite(metric))
                checkpoint = next((self.root / "Results").glob(f"{args.exp_name}_*/*_best_model.pth"))
                state = torch.load(checkpoint, weights_only=True)
                parameters = json.loads((checkpoint.parent / "hyperparameters.json").read_text(encoding="utf-8"))
                self.assertEqual(state["geometry_input_contract"]["mode"], mode)
                self.assertEqual(state["geometry_input_contract"], parameters["geometry_input_contract"])
                dataset = CachedPTDataset(Path(prefix + "test.pt"))
                model, kind = load_analysis_model(checkpoint, torch.device("cpu"), {"test": dataset})
                self.assertEqual(kind, "convnet")
                with torch.no_grad():
                    prediction = predict_batch(model, dataset.x, dataset.coord, dataset.conditioning_scalar)
                self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
