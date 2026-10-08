"""验证新 Exp3/迁移后 Exp8 的数据角色、划分和泄漏防线。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Experiment.prepare_data import _parse_experiment_tokens, get_folder_paths
from Experiment.run_train import _canonical_shorthand, load_config
from Experiment.train import _validate_cache_split_contract
from raw_data_processing.galaxy_data import GalDataSet


class _DatasetStub:
    def __init__(self, groups, sample_uid, parent_hr_uid):
        self.group_names = np.asarray(groups, dtype=str)
        self.sample_uid = np.asarray(sample_uid, dtype=str)
        self.parent_hr_uid = np.asarray(parent_hr_uid, dtype=str)

    def __len__(self):
        return len(self.group_names)


class Exp3TorusContractTests(unittest.TestCase):
    def test_role_filter_keeps_test_only_torus_out_of_default_disk_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / ".config"
            config.mkdir()
            (config / "BaseConfig.json").write_text(
                json.dumps({"data_dir": str(root / "data")}), encoding="utf-8"
            )
            (config / "RawDataConfig.json").write_text(
                json.dumps(
                    {
                        "disk_galaxy": {
                            "fiducial": {"folder_name": "unused"},
                            "torus_100yr": {
                                "folder_name": "unused",
                                "role": "test_only",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            training = get_folder_paths(config, "disk_galaxy", "coarse")
            test_only = get_folder_paths(
                config,
                "disk_galaxy",
                "coarse",
                include_roles=("test_only",),
            )
        self.assertEqual([entry[1] for entry in training], ["disk_galaxy_fiducial"])
        self.assertEqual(
            [entry[1] for entry in test_only], ["disk_galaxy_torus_100yr"]
        )

    def test_exp8_is_a_supported_prepare_data_token(self):
        self.assertEqual(
            _parse_experiment_tokens(["exp3", "8"]), ("exp3", "exp8")
        )

    def test_exp3_training_alias_uses_time_sampling_and_disables_film(self):
        config = load_config(_canonical_shorthand("exp3"))
        self.assertEqual(config["sampling_mode"], "run_time_balanced")
        self.assertFalse(config["use_film"])
        self.assertEqual(
            config["data_exp"],
            "Exp3_InstantSFR_NoInitialStellarDensity_",
        )

    def test_75_25_split_allocates_every_sample(self):
        dataset = GalDataSet()
        train, validation, test = dataset._split_group(
            np.arange(954),
            0.75,
            0.25,
            np.random.default_rng(42),
            True,
        )
        self.assertEqual(len(train), 715)
        self.assertEqual(len(validation), 239)
        self.assertEqual(len(test), 0)
        self.assertEqual(len(set(train) | set(validation)), 954)

    def test_training_contract_accepts_zou_train_val_and_torus_test(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "Exp3_Test_"
            metadata = {
                "split_strategy": "random_snapshot_75_25_torus_test",
                "train_groups": ["zou_a", "zou_b"],
                "validation_groups": ["zou_a", "zou_b"],
                "test_groups": ["torus"],
            }
            Path(f"{prefix}metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            datasets = {
                "train": _DatasetStub(
                    ["zou_a", "zou_b"], ["s1", "s2"], ["p1", "p2"]
                ),
                "val": _DatasetStub(
                    ["zou_a", "zou_b"], ["s3", "s4"], ["p3", "p4"]
                ),
                "test": _DatasetStub(["torus"], ["s5"], ["p5"]),
            }
            result = _validate_cache_split_contract(
                directory,
                "Exp3_Test_",
                datasets,
                validate_holdout=True,
            )
        self.assertEqual(result["test_groups"], ["torus"])

    def test_training_contract_rejects_torus_in_training(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "Exp3_Test_"
            Path(f"{prefix}metadata.json").write_text(
                json.dumps(
                    {
                        "split_strategy": "random_snapshot_75_25_torus_test",
                        "train_groups": ["zou"],
                        "validation_groups": ["zou"],
                        "test_groups": ["torus"],
                    }
                ),
                encoding="utf-8",
            )
            datasets = {
                "train": _DatasetStub(["zou", "torus"], ["s1", "s2"], ["p1", "p2"]),
                "val": _DatasetStub(["zou"], ["s3"], ["p3"]),
                "test": _DatasetStub(["torus"], ["s4"], ["p4"]),
            }
            with self.assertRaisesRegex(RuntimeError, "train.pt groups"):
                _validate_cache_split_contract(
                    directory,
                    "Exp3_Test_",
                    datasets,
                    validate_holdout=True,
                )


if __name__ == "__main__":
    unittest.main()
