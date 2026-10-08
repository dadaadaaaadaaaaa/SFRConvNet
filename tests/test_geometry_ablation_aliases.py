"""验证 Exp3.1/Exp3.2 的独立配方、简写与几何 CachePT 约束。"""

import unittest
from unittest import mock

from Experiment import run_train


class GeometryAblationAliasTests(unittest.TestCase):
    def _config(self, shorthand):
        return run_train.load_config(run_train._canonical_shorthand(shorthand))

    def test_closure_is_default_and_explicit_objectives_are_supported(self):
        for number, label in (("3.1", "AbsRadius"), ("3.2", "AbsRadiusWidths")):
            expected = f"Exp{number}_Rh12x20_FiveRuns_Closure_{label}"
            self.assertEqual(run_train._canonical_shorthand(f"exp{number}"), expected)
            self.assertEqual(
                run_train._canonical_shorthand(f"EXP{number}-CLOSURE"), expected
            )
            self.assertEqual(
                run_train._canonical_shorthand(f"exp{number}-direct"),
                f"Exp{number}_Rh12x20_FiveRuns_Direct_{label}",
            )

    def test_geometry_modes_cache_prefixes_and_short_result_names(self):
        for number, mode, label in (
            ("3.1", "absolute_radius", "AbsRadius"),
            ("3.2", "absolute_radius_widths", "AbsRadiusWidths"),
        ):
            for objective, trial in (("Closure", "Trial23"), ("Direct", "Trial61")):
                config = self._config(f"exp{number}-{objective.lower()}")
                self.assertEqual(config["expected_geometry_input_mode"], mode)
                self.assertEqual(
                    config["data_exp"], f"Exp3_Rh12x20_FiveRuns_{objective}_{label}_"
                )
                self.assertEqual(
                    config["exp_name"], f"Exp{number}_{objective}_{trial}_seed42"
                )

    def test_saved_closure_trial23_hyperparameters_are_preserved(self):
        expected = {
            "objective_mode": "signed_closure",
            "base_ch": 32,
            "batch_size": 64,
            "pos_num_bands": 8,
            "pos_max_freq": 101.0,
            "lr": 0.004388374713974493,
            "weight_decay": 6.833477992130433e-06,
            "loss": "l1",
            "negative_sfr_weight": 1.0,
        }
        for shorthand in ("exp3.1", "exp3.2"):
            config = self._config(shorthand)
            for key, value in expected.items():
                self.assertEqual(config[key], value, f"{shorthand}: {key}")

    def test_saved_direct_trial61_hyperparameters_are_preserved(self):
        expected = {
            "objective_mode": "direct_sfr",
            "base_ch": 128,
            "batch_size": 128,
            "pos_num_bands": 16,
            "pos_max_freq": 32.0,
            "lr": 0.00992825268659391,
            "weight_decay": 2.440185070480983e-06,
            "loss": "mse",
        }
        for shorthand in ("exp3.1-direct", "exp3.2-direct"):
            config = self._config(shorthand)
            for key, value in expected.items():
                self.assertEqual(config[key], value, f"{shorthand}: {key}")

    def test_common_training_recipe_and_no_film_or_augmentation(self):
        expected = {
            "model_type": "convnet",
            "stage_depths": "2,2,2,2",
            "se_reduction": 4,
            "p_drop": 0.0,
            "epochs": 90,
            "patience": 15,
            "seed": 42,
            "sampling_mode": "run_time_balanced",
            "batch_backend": "tensor_cpu",
            "num_workers": 0,
            "prefetch_factor": 3,
            "lr_min": 0.0,
            "lr_warmup_steps": 0,
            "lr_warmup_ratio": 0.2,
            "lr_warmup_start_factor": 0.0,
            "grad_clip": 0.3,
            "weight_mode": "none",
            "weight_alpha": 0.5,
            "weight_tau": 3.0,
            "weight_q_low": 0.1,
            "weight_q_high": 0.9,
            "weight_power": 1.0,
            "huber_delta": 1.0,
            "aug_prob": 0.0,
            "aug_min_crop_ratio": 0.6,
            "sfr_aug_prob": 0.0,
            "use_film": False,
            "persistent_workers": False,
            "cudnn_benchmark": False,
            "no_amp": False,
            "no_save": False,
            "skip_test": False,
            "skip_model_graph": True,
            "hpo_params": None,
        }
        for shorthand in ("exp3.1", "exp3.2", "exp3.1-direct", "exp3.2-direct"):
            config = self._config(shorthand)
            for key, value in expected.items():
                self.assertEqual(config[key], value, f"{shorthand}: {key}")

    def test_aliases_do_not_read_untracked_local_json(self):
        with mock.patch("builtins.open", side_effect=AssertionError("不得读取本机 JSON")):
            for shorthand in ("exp3.1", "exp3.2", "exp3.1-direct", "exp3.2-direct"):
                self.assertEqual(self._config(shorthand)["seed"], 42)

    def test_returned_configs_cannot_mutate_registered_recipe(self):
        config = self._config("exp3.1")
        config["base_ch"] = 999
        config["expected_geometry_input_mode"] = "none"
        fresh = self._config("exp3.1")
        self.assertEqual(fresh["base_ch"], 32)
        self.assertEqual(fresh["expected_geometry_input_mode"], "absolute_radius")

    def test_previous_shorthand_behavior_is_unchanged(self):
        for number in range(1, 9):
            self.assertEqual(
                run_train._canonical_shorthand(f"exp{number}"),
                f"Exp{number}_InstantSFR_NoInitialStellarDensity_SFRConvNet",
            )
        self.assertEqual(run_train._canonical_shorthand("custom_key"), "custom_key")

    def test_launcher_passes_geometry_expectation_and_preserves_boolean_flags(self):
        config = self._config("exp3.2-direct")
        with mock.patch.object(run_train.subprocess, "run") as process:
            self.assertTrue(run_train._run_config(config["exp_name"], config))
        command = process.call_args.args[0]
        mode_index = command.index("--expected_geometry_input_mode")
        self.assertEqual(command[mode_index + 1], "absolute_radius_widths")
        self.assertIn("--skip_model_graph", command)
        self.assertNotIn("--use_film", command)
        self.assertNotIn("--hpo_params", command)


if __name__ == "__main__":
    unittest.main()
