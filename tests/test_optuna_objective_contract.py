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
        self.user_attrs = {}
        self.categorical_choices = {}

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def suggest_categorical(self, name, choices):
        self.categorical_choices[name] = tuple(choices)
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
        self.assertTrue(direct.endswith("_v2"))

    def test_search_params_default_to_the_complete_search_space(self):
        self.assertEqual(
            run_optuna._resolve_search_params(None),
            run_optuna.SEARCHABLE_PARAMS,
        )
        self.assertEqual(
            run_optuna._parse_args([]).search_params,
            run_optuna.SEARCHABLE_PARAMS,
        )
        self.assertEqual(run_optuna._parse_args([]).seeds, (42,))

    def test_search_params_accept_spaces_or_commas_and_use_canonical_order(self):
        expected = ("pos_num_bands", "pos_max_freq")
        self.assertEqual(
            run_optuna._resolve_search_params(
                ["pos_max_freq,pos_num_bands"]
            ),
            expected,
        )
        self.assertEqual(
            run_optuna._parse_args(
                ["--search-params", "pos_max_freq", "pos_num_bands"]
            ).search_params,
            expected,
        )

    def test_search_params_reject_unknown_or_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "未知"):
            run_optuna._resolve_search_params(["not_a_parameter"])
        with self.assertRaisesRegex(ValueError, "重复"):
            run_optuna._resolve_search_params(["lr", "lr"])

    def test_fixed_params_are_typed_and_fill_unset_values_from_defaults(self):
        args = run_optuna._parse_args(
            [
                "--search-params",
                "pos_num_bands,pos_max_freq",
                "--fixed-params",
                "base_ch=128",
                "lr=2e-4,weight_decay=1e-5",
            ]
        )
        self.assertEqual(
            args.fixed_params,
            {
                "base_ch": 128,
                "se_reduction": 4,
                "p_drop": 0.3,
                "lr": 2e-4,
                "weight_decay": 1e-5,
                "batch_size": 64,
            },
        )

    def test_fixed_params_reject_search_overlap_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "同时搜索并固定"):
            run_optuna._fixed_hyperparameters(
                ("pos_num_bands",), {"pos_num_bands": 8}
            )
        with self.assertRaisesRegex(ValueError, "正整数"):
            run_optuna._fixed_hyperparameters(
                ("pos_num_bands",), {"base_ch": 0}
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            run_optuna._fixed_hyperparameters(
                ("pos_num_bands",), {"p_drop": 1.1}
            )

    def test_grid_sampler_covers_all_fourier_combinations_once(self):
        search_params = ("pos_num_bands", "pos_max_freq")
        sampler = run_optuna._build_sampler("grid", search_params, (42,), 7)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def grid_objective(trial):
            bands = trial.suggest_categorical("pos_num_bands", [4, 8, 16])
            max_freq = trial.suggest_categorical(
                "pos_max_freq", [2, 4, 8, 16, 32, 101]
            )
            return float(bands + max_freq)

        study.optimize(grid_objective, n_trials=100)
        combinations = {
            (trial.params["pos_num_bands"], trial.params["pos_max_freq"])
            for trial in study.trials
        }
        self.assertEqual(len(study.trials), 18)
        self.assertEqual(len(combinations), 18)

    def test_grid_sampler_rejects_continuous_search_parameters(self):
        with self.assertRaisesRegex(ValueError, "lr"):
            run_optuna._build_sampler("grid", ("lr",), (42,))
        with self.assertRaisesRegex(ValueError, "多个 training seed"):
            run_optuna._build_sampler(
                "tpe", ("pos_num_bands",), (42, 43)
            )

    def test_search_subset_has_stable_distinct_study_identity(self):
        full = run_optuna._study_identity("Exp1_", "direct_sfr")
        subset_a = run_optuna._study_identity(
            "Exp1_", "direct_sfr", ["pos_max_freq", "pos_num_bands"]
        )
        subset_b = run_optuna._study_identity(
            "Exp1_", "direct_sfr", ["pos_num_bands,pos_max_freq"]
        )
        self.assertNotEqual(full, subset_a)
        self.assertEqual(subset_a, subset_b)
        self.assertTrue(subset_a.endswith("_only_pos_num_bands-pos_max_freq"))

    def test_sampler_and_training_seeds_have_distinct_study_identities(self):
        search_params = ("pos_num_bands", "pos_max_freq")
        tpe = run_optuna._study_identity(
            "Exp1_", "direct_sfr", search_params
        )
        grid = run_optuna._study_identity(
            "Exp1_", "direct_sfr", search_params, sampler_name="grid"
        )
        replicated_grid = run_optuna._study_identity(
            "Exp1_",
            "direct_sfr",
            search_params,
            sampler_name="grid",
            training_seeds=(42, 43, 44),
        )
        self.assertNotEqual(tpe, grid)
        self.assertTrue(grid.endswith("_grid"))
        self.assertRegex(replicated_grid, r"_grid_seeds_[0-9a-f]{10}$")

    def test_nondefault_fixed_values_have_distinct_study_identities(self):
        search_params = ("pos_num_bands", "pos_max_freq")
        default = run_optuna._study_identity(
            "Exp1_", "direct_sfr", search_params
        )
        fixed_64 = run_optuna._study_identity(
            "Exp1_", "direct_sfr", search_params, {"base_ch": 64}
        )
        fixed_128_a = run_optuna._study_identity(
            "Exp1_",
            "direct_sfr",
            search_params,
            {"base_ch": 128, "lr": 2e-4},
        )
        fixed_128_b = run_optuna._study_identity(
            "Exp1_",
            "direct_sfr",
            search_params,
            {"lr": 2e-4, "base_ch": 128},
        )
        self.assertEqual(default, fixed_64)
        self.assertNotEqual(default, fixed_128_a)
        self.assertEqual(fixed_128_a, fixed_128_b)
        self.assertRegex(fixed_128_a, r"_fixed_[0-9a-f]{10}$")

    def test_trial_searches_joint_fourier_configuration(self):
        trial = _FakeTrial()
        with mock.patch.object(run_optuna.train, "main", return_value=0.2) as run:
            run_optuna.objective(trial, "Example_")

        self.assertEqual(trial.categorical_choices["pos_num_bands"], (4, 8, 16))
        self.assertEqual(
            trial.categorical_choices["pos_max_freq"],
            (2, 4, 8, 16, 32, 101),
        )
        args = run.call_args.args[0]
        self.assertEqual(args.pos_num_bands, 4)
        self.assertEqual(args.pos_max_freq, 2)

    def test_trial_only_suggests_selected_parameters_and_fixes_the_rest(self):
        trial = _FakeTrial()
        with mock.patch.object(run_optuna.train, "main", return_value=0.2) as run:
            run_optuna.objective(
                trial,
                "Example_",
                search_params=("pos_num_bands", "pos_max_freq"),
            )

        self.assertEqual(
            tuple(trial.params),
            ("pos_num_bands", "pos_max_freq"),
        )
        args = run.call_args.args[0]
        self.assertEqual(args.base_ch, 64)
        self.assertEqual(args.se_reduction, 4)
        self.assertEqual(args.p_drop, 0.3)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.weight_decay, 0.0)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(
            trial.user_attrs["search_params"],
            ["pos_num_bands", "pos_max_freq"],
        )
        self.assertEqual(
            trial.user_attrs["fixed_hyperparameters"],
            {
                "base_ch": 64,
                "se_reduction": 4,
                "p_drop": 0.3,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 64,
            },
        )

    def test_trial_applies_explicit_fixed_parameter_values(self):
        trial = _FakeTrial()
        with mock.patch.object(run_optuna.train, "main", return_value=0.2) as run:
            run_optuna.objective(
                trial,
                "Example_",
                search_params=("pos_num_bands", "pos_max_freq"),
                fixed_params={"base_ch": 128, "p_drop": 0.1, "lr": 2e-4},
            )

        args = run.call_args.args[0]
        self.assertEqual(args.base_ch, 128)
        self.assertEqual(args.p_drop, 0.1)
        self.assertEqual(args.lr, 2e-4)
        self.assertEqual(args.se_reduction, 4)
        self.assertEqual(trial.user_attrs["fixed_hyperparameters"]["base_ch"], 128)

    def test_grid_trial_selects_training_seed_as_replication_dimension(self):
        trial = _FakeTrial()
        with mock.patch.object(run_optuna.train, "main", return_value=0.2) as run:
            run_optuna.objective(
                trial,
                "Example_",
                search_params=("pos_num_bands", "pos_max_freq"),
                sampler_name="grid",
                training_seeds=(42, 43, 44),
            )

        args = run.call_args.args[0]
        self.assertEqual(args.seed, 42)
        self.assertEqual(trial.params["seed"], 42)
        self.assertEqual(trial.user_attrs["training_seed"], 42)

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
        self.assertEqual(captured["batch_backend"], "tensor_cpu")
        self.assertTrue(captured["skip_model_graph"])

    def test_dataset_store_and_performance_overrides_reach_train(self):
        store = object()
        with mock.patch.object(run_optuna.train, "main", return_value=0.2) as run:
            run_optuna.objective(_FakeTrial(), "Example_", dataset_cache=store,
                                 batch_backend="standard", write_model_graph=True)
        args = run.call_args.args[0]
        self.assertEqual(args.batch_backend, "standard")
        self.assertFalse(args.skip_model_graph)
        self.assertIs(run.call_args.kwargs["dataset_cache"], store)

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
