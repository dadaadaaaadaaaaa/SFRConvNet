"""逐格 Exp1/2 完整训练、固定评价指标、精确恢复与 Optuna 契约回归。"""

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from Experiment.prepare_data import CachePt
from Experiment.build_cell_supervision_cache import export_cell_cache
from Experiment.train_sfr_unet import TrainConfig, DataBundle, train, batch_indices, evaluate
from Experiment.run_train_sfr_unet import main as run_main, resolve_folds
from Experiment.run_optuna_sfr_unet import optimize, parse_search
from Experiment.cell_sfr_objective import CellSFRLoss
from Experiment.SFRUNet import SFRUNet
from raw_data_processing.galaxy_data import GalDataSet
from tests.test_cell_sfr_supervision import fine_with_labels


def make_cache(root, experiment="exp1"):
    """不同 split 使用独立原始帧，训练统计原样用于验证/测试，构造真实 sidecar。"""
    root = Path(root)
    prefix = "Exp1_Tiny_" if experiment == "exp1" else "Exp2_Tiny_LOO_run_c_"
    splits = ["train", "val", "test"] + (["out"] if experiment == "exp2" else [])
    mean = std = None
    metadata = {"experiment": experiment, "split_strategy": "random_snapshot_within_group" if experiment == "exp1"
                else "leave_one_galaxy_out", "train_groups": ["run_a", "run_b"],
                "held_out_groups": [] if experiment == "exp1" else ["run_c"], "sample_counts": {},
                "cache_prefix": prefix}
    index = 0
    for split in splits:
        directories = []
        groups = ["run_c"] if split == "out" else ["run_a", "run_b"]
        for group in groups:
            directory = root / "products" / split / group
            directory.mkdir(parents=True)
            directories.append([str(directory), group])
            for repeat in range(2 if split in {"train", "out"} else 1):
                index += 1
                galaxy = fine_with_labels(zero=index == 1)
                galaxy.time = float(index)
                galaxy.attach_fine_source_metadata({"simulation_code": "MACER2D", "galaxy_id": group,
                    "run_id": group, "source_series_id": group, "raw_snapshot_index": index})
                galaxy.compute_macer_sfr_products()
                galaxy.attach_cell_sfr_target(np.zeros((4, 4)) if index == 1 else np.arange(16).reshape(4,4)*.025)
                galaxy.coarsen("spatial_block", (4,4)).save_h5(directory / f"{index:05d}.h5")
        dataset = GalDataSet().load_data(directories, require_complete_metadata=True)
        cache = CachePt(dataset)
        path = root / f"{prefix}{split}.pt"
        resolved_mean, resolved_std = cache.to_pt(path, mean, std, objective_mode="signed_closure")
        if split == "train":
            mean, std = resolved_mean, resolved_std
            metadata.update(input_channels=list(dataset.input_channels), target_units="MACER_mass_code/Gyr",
                            conditioning_variable="none", conditioning_active=False)
            np.savez(root / f"{prefix}norm.npz", mean=mean, std=std, input_channels=dataset.input_channels)
        metadata["sample_counts"][split] = len(dataset)
        export_cell_cache(path, [Path(d[0]) for d in directories], root / f"{prefix}{split}_cells.pt")
    (root / f"{prefix}metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if experiment == "exp2":
        (root / "Exp2_Tiny_folds.json").write_text(json.dumps({"experiment": "exp2",
            "split_strategy": "leave_one_galaxy_out", "folds": [{"cache_prefix": prefix,
             "held_out_groups": ["run_c"], "train_groups": ["run_a", "run_b"]}]}), encoding="utf-8")
    return TrainConfig(data_exp=prefix, cache_dir=str(root), output_root=str(root/"results"),
                       epochs=2, patience=0, batch_size=2, base_ch=4, cpu_threads=1,
                       tensorboard=False, device="cpu", p_drop=.1)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tail_batches_and_invalid_config(self):
        for n in (2,3,5,7):
            chunks = batch_indices(n, 2, shuffle=True)
            self.assertEqual(sorted(torch.cat(chunks).tolist()), list(range(n)))
            self.assertTrue(all(len(c) >= 2 for c in chunks))
        with self.assertRaises(ValueError):
            TrainConfig(data_exp="test", lr=float("nan")).validate()
        with self.assertRaises(ValueError):
            parse_search(["seed"])

    def test_exp1_training_resume_exact_and_config_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_cache(directory)
            bundle = DataBundle(config)
            full = train(config, bundle=bundle)
            paused = train(replace(config, stop_after_epochs=1), bundle=bundle)
            self.assertEqual(paused["status"], "paused_at_epoch_boundary")
            self.assertFalse((Path(paused["output_dir"])/"test_predictions.pt").exists())
            resumed_config = replace(config, resume=str(Path(paused["output_dir"])/"last.pt"))
            result = train(resumed_config, bundle=bundle)
            a = torch.load(Path(full["output_dir"])/"last.pt", weights_only=False)
            b = torch.load(Path(result["output_dir"])/"last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            self.assertEqual(a["best_score"], b["best_score"])
            with self.assertRaisesRegex(ValueError, "恢复配置"):
                train(replace(resumed_config, lr=.01), bundle=bundle)
            predictions = torch.load(Path(full["output_dir"])/"test_predictions.pt", weights_only=False)
            torch.testing.assert_close(predictions["prediction_total"], predictions["prediction_map"].sum((1,2,3)))
            self.assertEqual(predictions["sample_uid"], bundle.datasets["test"].sample_uid)

    def test_exp2_launcher_manifest_and_out(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_cache(directory, "exp2")
            jobs = resolve_folds(replace(config, data_exp="Exp2_Tiny_"), ["run_c"])
            self.assertEqual(jobs[0].data_exp, config.data_exp)
            with self.assertRaises(ValueError):
                resolve_folds(replace(config, data_exp="Exp2_Tiny_"), ["unknown"])
            report = run_main(["--data-exp", "Exp2_Tiny_", "--cache-dir", directory,
                "--output-root", str(Path(directory)/"results"), "--held-out-galaxy", "run_c",
                "--epochs", "1", "--batch-size", "2", "--base-ch", "4", "--cpu-threads", "1",
                "--device", "cpu", "--no-tensorboard"])[0]
            self.assertIn("out", report["evaluation"])
            self.assertEqual(set(report["evaluation"]["out"]["model"]["per_run"]), {"run_c"})

    def test_fixed_metric_independent_of_train_weights_and_batch_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_cache(directory)
            bundle = DataBundle(config)
            model = SFRUNet(len(bundle.datasets["train"].input_channels), base_ch=4)
            first = CellSFRLoss.from_scale_contract(bundle.scales, total_weight=0, negative_weight=.3)
            second = CellSFRLoss.from_scale_contract(bundle.scales, total_weight=3, negative_weight=3)
            a,_ = evaluate(model, bundle.datasets["train"], first, bundle.scales, device=torch.device("cpu"), batch_size=3)
            b,_ = evaluate(model, bundle.datasets["train"], second, bundle.scales, device=torch.device("cpu"), batch_size=2)
            self.assertAlmostEqual(a["model"]["macro_joint_score"], b["model"]["macro_joint_score"], places=6)

    def test_optuna_resume_and_best_config_without_reading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(make_cache(directory, "exp2"), epochs=1)
            # HPO 不需要打开 test/out，即使它们暂未传到服务器也可调参。
            for split in ("test", "out"):
                (Path(directory)/f"{config.data_exp}{split}.pt").unlink()
                (Path(directory)/f"{config.data_exp}{split}_cells.pt").unlink()
            result = optimize(config, search=["bottleneck_depth"], n_trials=1, timeout=60, pruning=False)[0]
            second = optimize(config, search=["bottleneck_depth"], n_trials=1, timeout=60, pruning=False)[0]
            self.assertEqual(second["completed_trials"], 2)
            self.assertEqual(result["database"], second["database"])
            best = json.loads(Path(second["best_config"]).read_text())
            self.assertEqual(best["data_exp"], config.data_exp)
            self.assertEqual(best["base_ch"], 4)
            self.assertFalse(list((Path(directory)/"results").rglob("test_predictions.pt")))

    def test_wrong_experiment_groups_and_normalization_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_cache(directory, "exp2")
            path = Path(directory)/f"{config.data_exp}metadata.json"
            original = json.loads(path.read_text())
            bad = dict(original, experiment="exp3")
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "Exp1/Exp2"):
                DataBundle(config)
            path.write_text(json.dumps(dict(original, held_out_groups=["run_a"])))
            with self.assertRaises(ValueError):
                DataBundle(config, evaluation=False)
            path.write_text(json.dumps(original))
            np.savez(Path(directory)/f"{config.data_exp}norm.npz", mean=[float("nan")], std=[1])
            with self.assertRaisesRegex(ValueError, "mean/std"):
                DataBundle(config)

    def test_pruning_callback_preserves_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_cache(directory)
            import optuna
            def prune(epoch, score):
                raise optuna.TrialPruned("测试回调")
            with self.assertRaises(optuna.TrialPruned):
                train(config, hpo=True, epoch_callback=prune)
            last = list(Path(config.output_root).rglob("last.pt"))
            self.assertEqual(len(last), 1)
            self.assertEqual(torch.load(last[0], weights_only=False)["epoch"], 1)
            self.assertFalse(list(Path(config.output_root).rglob("test_predictions.pt")))

    def test_json_config_cli_override_and_multiple_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            from dataclasses import asdict
            config = replace(make_cache(directory), epochs=1)
            path = Path(directory)/"config.json"
            path.write_text(json.dumps(asdict(config)))
            reports = run_main(["--config", str(path), "--seeds", "11", "12", "--base-ch", "8"])
            self.assertEqual(len(reports), 2)
            for report, seed in zip(reports, (11,12)):
                checkpoint = torch.load(Path(report["output_dir"])/"best.pt", weights_only=False)
                self.assertEqual(checkpoint["config"]["seed"], seed)
                self.assertEqual(checkpoint["model_contract"]["config"]["base_ch"], 8)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 设备")
    def test_cuda_forward_training_and_bfloat16(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(make_cache(directory), epochs=1, device="cuda")
            bundle = DataBundle(config)
            modes = ["none"] + (["bfloat16"] if torch.cuda.is_bf16_supported() else [])
            for amp in modes:
                result = train(replace(config, amp=amp), bundle=bundle)
                self.assertTrue(np.isfinite(result["best_score"]))
                self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
