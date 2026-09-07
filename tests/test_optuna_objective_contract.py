import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import optuna

from Experiment import run_optuna


class _FakeTrial:
    def __init__(self):
        self.number = 3
        self.params = {}

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, **kwargs):
        self.params[name] = low
        return low


def _write_cache_contract(directory, prefix, *, prepared="signed_closure"):
    root = Path(directory)
    for split in ("train", "val", "test"):
        (root / f"{prefix}{split}.pt").touch()
    metadata = {
        "prepared_objective": prepared,
        "available_objectives": ["direct_sfr", "signed_closure"],
        "closure_transform": "signed_asinh",
        "closure_scale": 250.0,
        "conditioning_variable": "none",
        "conditioning_active": False,
    }
    with (root / f"{prefix}metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file)


class OptunaObjectiveContractTests(unittest.TestCase):
    def test_auto_objective_reads_prepared_objective_from_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = "ExpAuto_"
            _write_cache_contract(directory, prefix)
            contract = run_optuna._load_cache_contract(directory, prefix)

        self.assertEqual(
            run_optuna._resolve_objective_mode("auto", contract),
            "signed_closure",
        )

    def test_direct_and_closure_use_distinct_study_names(self):
        direct = run_optuna._study_identity("Exp1_", "direct_sfr")
        closure = run_optuna._study_identity("Exp1_", "signed_closure")
        self.assertNotEqual(direct, closure)
        self.assertIn("direct_sfr", direct)
        self.assertIn("signed_closure", closure)

    def test_resume_rejects_changed_fixed_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = "ExpContract_"
            _write_cache_contract(directory, prefix)
            contract = run_optuna._load_cache_contract(directory, prefix)
            study = optuna.create_study(direction="minimize")
            run_optuna._bind_study_contract(
                study,
                data_exp=prefix,
                objective_mode="signed_closure",
                contract=contract,
                loss="l1",
                negative_sfr_weight=1.0,
            )
            with self.assertRaisesRegex(RuntimeError, "loss"):
                run_optuna._bind_study_contract(
                    study,
                    data_exp=prefix,
                    objective_mode="signed_closure",
                    contract=contract,
                    loss="mse",
                    negative_sfr_weight=1.0,
                )

    def test_trial_explicitly_propagates_closure_contract_and_skips_test(self):
        captured = {}

        def fake_train_main(args):
            captured.update(vars(args))
            return 0.125

        with mock.patch.object(run_optuna.train, "main", side_effect=fake_train_main):
            result = run_optuna.objective(
                _FakeTrial(),
                "ExpClosure_",
                objective_mode="signed_closure",
                negative_sfr_weight=2.5,
                loss="l1",
                use_film=False,
            )

        self.assertEqual(result, 0.125)
        self.assertEqual(captured["objective_mode"], "signed_closure")
        self.assertEqual(captured["negative_sfr_weight"], 2.5)
        self.assertEqual(captured["loss"], "l1")
        self.assertTrue(captured["skip_test"])
        self.assertTrue(captured["no_save"])

    def test_non_oom_runtime_error_is_not_hidden_as_infinite_loss(self):
        with mock.patch.object(
            run_optuna.train,
            "main",
            side_effect=RuntimeError("cache schema mismatch"),
        ):
            with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                run_optuna.objective(_FakeTrial(), "ExpBroken_")

    def test_cuda_oom_prunes_only_current_trial(self):
        with mock.patch.object(
            run_optuna.train,
            "main",
            side_effect=RuntimeError("CUDA out of memory"),
        ):
            with self.assertRaises(optuna.TrialPruned):
                run_optuna.objective(_FakeTrial(), "ExpLarge_")

    def test_train_skip_test_is_opt_in(self):
        self.assertFalse(run_optuna.train.parse_args([]).skip_test)


if __name__ == "__main__":
    unittest.main()
