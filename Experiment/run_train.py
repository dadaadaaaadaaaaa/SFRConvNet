"""从实验 JSON 配方或代码别名展开训练任务，并调用 ``train.py`` 执行。

本模块支持单实验、批量实验和 Exp2 leave-one-galaxy-out 多折 manifest，
也允许为固定分辨率 R_h 显式关闭 FiLM。它只负责解析与调度，不生成训练缓存。
"""

import os
import argparse
import copy
import subprocess
import sys
import json

# =============================================================================
# Configuration Source
# =============================================================================
# Specify the experiment key from ExperimentSetting.json to run
TARGET_EXPERIMENT = "Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet"

EXPERIMENT_ALIASES = {
    # .config is ignored by git, so keep the ablation entry reproducible in
    # tracked code while still allowing a local config entry to override it.
    "Exp1_NoDnewstar_SFRConvNet": {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {"BaseInformation": {"data_exp": "Exp1_NoDnewstar_"}},
    },
    "Exp1_InstantSFR_SFRConvNet": {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {"BaseInformation": {"data_exp": "Exp1_InstantSFR_"}},
    },
    "Exp1_InstantSFR_NoDstar_SFRConvNet": {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {"BaseInformation": {"data_exp": "Exp1_InstantSFR_NoDstar_"}},
    },
    "Exp1_InstantSFR_NoInitialStellarDensity_SFRConvNet": {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {
            "BaseInformation": {"data_exp": "Exp1_InstantSFR_NoInitialStellarDensity_"}
        },
    },
    "Exp2_NoAug_SFRConvNet": {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {
            "BaseInformation": {
                "data_exp": "Exp2_InstantSFR_NoInitialStellarDensity_NoAug_"
            },
            "DataAugmentation": {"sfr_aug_prob": 0.0},
        },
    },
}

# Shorthand experiments use the current instantaneous-SFR, no-initial-stellar-
# density SFRConvNet recipe. Explicit ExperimentSetting.json keys remain fully
# supported for other architectures.
for _experiment_number in range(2, 9):
    EXPERIMENT_ALIASES[
        f"Exp{_experiment_number}_InstantSFR_NoInitialStellarDensity_SFRConvNet"
    ] = {
        "base_experiment": "Exp1_SFRConvNet",
        "overrides": {
            "BaseInformation": {
                "data_exp": f"Exp{_experiment_number}_InstantSFR_NoInitialStellarDensity_"
            }
        },
    }

# 新 Exp3 使用无重复 CachePT；训练期按运行等权、运行内物理时间间隔加权。
EXPERIMENT_ALIASES[
    "Exp3_InstantSFR_NoInitialStellarDensity_SFRConvNet"
]["overrides"].update(
    {
        "DataSampling": {"sampling_mode": "run_time_balanced"},
        # Exp3 使用固定 12×20 R_h，conditioning 不活动，必须关闭 FiLM。
        "ModelArchitecture": {"use_film": False},
    }
)


# 几何消融复用已完成 Exp3 的 Trial23/Trial61 配方，不继承本机 JSON 或 HPO。
# 只增加 CachePT 中的几何输入通道；原相对 Fourier 编码与输出结构保持不变。
_EXP3_GEOMETRY_COMMON_CONFIG = {
    "BaseInformation": {
        "cache_dir": ".cache",
        "log_dir": "runs",
        "seed": 42,
        "device": "auto",
    },
    "ModelArchitecture": {
        "model_type": "convnet",
        "stage_depths": "2,2,2,2",
        "se_reduction": 4,
        "p_drop": 0.0,
        "use_film": False,
    },
    "TrainingLoop": {
        "epochs": 90,
        "patience": 15,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": 3,
        "batch_backend": "tensor_cpu",
    },
    "Optimization": {
        "lr_min": 0.0,
        "lr_warmup_steps": 0,
        "lr_warmup_ratio": 0.2,
        "lr_warmup_start_factor": 0.0,
        "grad_clip": 0.3,
        "hpo_params": None,
    },
    "LossFunction": {
        "weight_mode": "none",
        "weight_alpha": 0.5,
        "weight_tau": 3.0,
        "weight_q_low": 0.1,
        "weight_q_high": 0.9,
        "weight_power": 1.0,
        "huber_delta": 1.0,
        "negative_sfr_weight": 1.0,
    },
    "DataSampling": {"sampling_mode": "run_time_balanced"},
    "DataAugmentation": {
        "aug_prob": 0.0,
        "aug_min_crop_ratio": 0.6,
        "sfr_aug_prob": 0.0,
    },
    "System": {
        "no_amp": False,
        "cudnn_benchmark": False,
        "skip_model_graph": True,
        "skip_test": False,
        "no_save": False,
    },
}

_EXP3_GEOMETRY_OBJECTIVE_CONFIGS = {
    "Closure": {
        "BaseInformation": {"objective_mode": "signed_closure"},
        "ModelArchitecture": {
            "base_ch": 32,
            "pos_num_bands": 8,
            "pos_max_freq": 101.0,
        },
        "TrainingLoop": {"batch_size": 64},
        "Optimization": {
            "lr": 0.004388374713974493,
            "weight_decay": 6.833477992130433e-06,
        },
        "LossFunction": {"loss": "l1"},
    },
    "Direct": {
        "BaseInformation": {"objective_mode": "direct_sfr"},
        "ModelArchitecture": {
            "base_ch": 128,
            "pos_num_bands": 16,
            "pos_max_freq": 32.0,
        },
        "TrainingLoop": {"batch_size": 128},
        "Optimization": {
            "lr": 0.00992825268659391,
            "weight_decay": 2.440185070480983e-06,
        },
        "LossFunction": {"loss": "mse"},
    },
}

_EXP3_GEOMETRY_SHORTHANDS = {}
for _ablation_number, _geometry_mode, _geometry_label in (
    ("3.1", "absolute_radius", "AbsRadius"),
    ("3.2", "absolute_radius_widths", "AbsRadiusWidths"),
):
    for _objective_label, _objective_recipe in _EXP3_GEOMETRY_OBJECTIVE_CONFIGS.items():
        _geometry_key = (
            f"Exp{_ablation_number}_Rh12x20_FiveRuns_"
            f"{_objective_label}_{_geometry_label}"
        )
        _geometry_config = copy.deepcopy(_EXP3_GEOMETRY_COMMON_CONFIG)
        for _section, _values in _objective_recipe.items():
            _geometry_config.setdefault(_section, {}).update(_values)
        _trial_label = "Trial23" if _objective_label == "Closure" else "Trial61"
        _geometry_config["BaseInformation"].update(
            {
                "data_exp": (
                    f"Exp3_Rh12x20_FiveRuns_{_objective_label}_{_geometry_label}_"
                ),
                # 缩短实际输出目录，完整 cache 和参数仍写入 hyperparameters.json。
                "exp_name": f"Exp{_ablation_number}_{_objective_label}_{_trial_label}_seed42",
                "expected_geometry_input_mode": _geometry_mode,
            }
        )
        EXPERIMENT_ALIASES[_geometry_key] = {"standalone_config": _geometry_config}
        _EXP3_GEOMETRY_SHORTHANDS[
            f"exp{_ablation_number}-{_objective_label.lower()}"
        ] = _geometry_key
        if _objective_label == "Closure":
            _EXP3_GEOMETRY_SHORTHANDS[f"exp{_ablation_number}"] = _geometry_key


def _canonical_shorthand(value):
    value = str(value).strip()
    if value.lower() in _EXP3_GEOMETRY_SHORTHANDS:
        return _EXP3_GEOMETRY_SHORTHANDS[value.lower()]
    if value.lower().startswith("exp") and value[3:].isdigit():
        number = int(value[3:])
        if 1 <= number <= 8:
            return f"Exp{number}_InstantSFR_NoInitialStellarDensity_SFRConvNet"
    return value


def _short_group_name(group_name):
    name = str(group_name).strip().lower().removeprefix("disk_galaxy_")
    return "".join(character for character in name if character.isalnum())


def _parse_bool(value):
    """Parse explicit CLI booleans such as --use-film=false."""
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def load_config(exp_key):
    """
    Load configuration from ExperimentSetting.json and flatten it.
    """
    # 几何消融必须独立于未跟踪的本机 JSON，避免服务器或旧配置改变对照配方。
    alias = EXPERIMENT_ALIASES.get(exp_key, {})
    if "standalone_config" in alias:
        return _flatten_config(exp_key, copy.deepcopy(alias["standalone_config"]))

    # Path to the JSON configuration file (Located in .config folder, sibling to Experiment folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(current_dir, "..", ".config", "ExperimentSetting.json"))
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON configuration file: {e}")
        sys.exit(1)
        
    if exp_key not in full_config and exp_key in EXPERIMENT_ALIASES:
        alias = EXPERIMENT_ALIASES[exp_key]
        base_key = alias["base_experiment"]
        if base_key not in full_config:
            print(f"Error: Base experiment key '{base_key}' not found in {config_path}")
            sys.exit(1)
        exp_config = copy.deepcopy(full_config[base_key])
        for section, values in alias["overrides"].items():
            exp_config.setdefault(section, {}).update(values)
    elif exp_key not in full_config:
        print(f"Error: Experiment key '{exp_key}' not found in {config_path}")
        available = list(full_config.keys()) + list(EXPERIMENT_ALIASES.keys())
        print(f"Available keys: {available}")
        sys.exit(1)
    else:
        exp_config = full_config[exp_key]
    
    return _flatten_config(exp_key, exp_config)


def _flatten_config(exp_key, exp_config):
    """展平配置段，保持旧配置字段映射及实验名称覆盖规则。"""

    flat_config = {"exp_name": exp_key} # Use the key as the experiment name
    
    for section, params in exp_config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                if key == "description":
                    continue
                # Handle specific key mappings if the script arguments differ from JSON keys
                if key == "cache_file":
                    flat_config["cache_dir"] = value
                else:
                    flat_config[key] = value
        else:
             # Description or other top level simple types
             if section != "description": 
                 flat_config[section] = params
                 
    return flat_config

# =============================================================================
# Execution Script
# 执行脚本
# =============================================================================

def _load_fold_manifest(cache_dir, data_exp):
    manifest_path = os.path.join(cache_dir, f"{data_exp}folds.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read fold manifest '{manifest_path}'.") from exc
    if manifest.get("split_strategy") != "leave_one_galaxy_out":
        raise RuntimeError(
            f"Unsupported split strategy in '{manifest_path}': {manifest.get('split_strategy')!r}"
        )
    if not manifest.get("folds"):
        raise RuntimeError(f"Fold manifest '{manifest_path}' contains no folds.")
    return manifest


def _held_out_matches(requested, held_out_groups):
    if not requested:
        return True
    actual = set(held_out_groups)
    for value in requested:
        value = str(value).strip()
        candidates = {value}
        if not value.startswith("disk_galaxy_"):
            candidates.add(f"disk_galaxy_{value}")
        if candidates.intersection(actual):
            return True
    return False


def _expand_training_jobs(exp_key, held_out_galaxies=None):
    requested_key = str(exp_key).strip()
    exp_key = _canonical_shorthand(exp_key)
    print(f"Loading configuration for: {exp_key}")
    config = load_config(exp_key)
    cache_dir = config.get("cache_dir", ".cache")
    data_exp = config.get("data_exp", exp_key)
    manifest = _load_fold_manifest(cache_dir, data_exp)
    if manifest is None:
        requires_logo_manifest = requested_key.lower() in {"2", "exp2"} or exp_key == (
            "Exp2_InstantSFR_NoInitialStellarDensity_SFRConvNet"
        )
        if requires_logo_manifest:
            raise RuntimeError(
                f"No leave-one-galaxy-out fold manifest found at "
                f"{os.path.join(cache_dir, data_exp + 'folds.json')}. "
                "Prepare Exp2 first with Experiment/prepare_data.py --experiments exp2."
            )
        return [(exp_key, config)]

    jobs = []
    for fold in manifest["folds"]:
        held_out_groups = fold.get("held_out_groups", ())
        if not _held_out_matches(held_out_galaxies, held_out_groups):
            continue
        fold_config = copy.deepcopy(config)
        fold_config["data_exp"] = fold["cache_prefix"]
        # Keep Windows paths and user-facing result names concise. The complete
        # architecture and cache identity remain recorded in hyperparameters.json.
        model_label = exp_key.rsplit("_", 1)[-1]
        short_group = _short_group_name(held_out_groups[0])
        if model_label == "SFRConvNet":
            suffix = "_NoAug" if "NoAug" in exp_key else ""
            fold_config["exp_name"] = f"Exp2_LOO{short_group}{suffix}"
        else:
            fold_config["exp_name"] = f"Exp2_{model_label}_LOO{short_group}"
        jobs.append((fold_config["exp_name"], fold_config))
    if not jobs:
        raise RuntimeError(
            f"No folds in {data_exp}folds.json matched --held-out-galaxy "
            f"{held_out_galaxies}."
        )
    return jobs


def _run_config(job_name, config):

    # Locate the training script in the same directory
    # 定位同一目录下的训练脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "train.py")

    if not os.path.exists(script_path):
        print(f"Error: Could not find {script_path}")
        return

    # Build the command
    cmd = [sys.executable, script_path]

    for key, value in config.items():
        # Handle boolean flags (store_true arguments)
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        # Handle None values (skip)
        elif value is None:
            continue
        # Handle list/tuple (e.g., stage_depths)
        elif isinstance(value, (list, tuple)):
            cmd.append(f"--{key}")
            cmd.append(",".join(str(item) for item in value))
        # Handle standard key-value arguments
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))

    # Print the command for verification
    print("=" * 60)
    print("Running Experiment with config:")
    print("=" * 60)
    for k, v in config.items():
        print(f"{k:<20}: {v}")
    print("-" * 60)
    
    # Run
    try:
        subprocess.run(cmd, check=True) # 执行train
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Process exited with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")
        return False


def main(
    experiments=None,
    *,
    held_out_galaxies=None,
    continue_on_error=False,
    use_film_override=None,
):
    if experiments is None:
        experiments = [TARGET_EXPERIMENT]
    elif isinstance(experiments, str):
        experiments = [experiments]

    jobs = []
    for experiment in experiments:
        jobs.extend(_expand_training_jobs(experiment, held_out_galaxies))

    if use_film_override is not None:
        overridden_jobs = []
        for _, config in jobs:
            config = copy.deepcopy(config)
            config["use_film"] = bool(use_film_override)
            if not use_film_override and not str(config["exp_name"]).endswith("_NoFiLM"):
                config["exp_name"] = f"{config['exp_name']}_NoFiLM"
            overridden_jobs.append((config["exp_name"], config))
        jobs = overridden_jobs

    print(f"Resolved {len(jobs)} training job(s).")
    for index, (job_name, config) in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {job_name}")
        succeeded = _run_config(job_name, config)
        if not succeeded and not continue_on_error:
            print("[Info] Stopping batch after the first failed training job.")
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a configured MACNet experiment.")
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Backward-compatible single experiment key.",
    )
    parser.add_argument(
        "--experiment",
        dest="experiment_flags",
        action="append",
        default=None,
        help="Repeatable experiment key or shorthand such as exp3, exp3.1, exp3.2-direct.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "One or more experiment keys/shorthands. For example, --experiments exp2 "
            "runs every leave-one-galaxy-out fold from the Exp2 manifest."
        ),
    )
    parser.add_argument(
        "--held-out-galaxy",
        action="append",
        default=None,
        help="For a multi-fold manifest, run only folds holding out this galaxy.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later jobs if one training process fails.",
    )
    parser.add_argument(
        "--use-film",
        type=_parse_bool,
        default=None,
        metavar="BOOL",
        help=(
            "Override the recipe for all selected jobs. Use --use-film=false "
            "for fixed-resolution R_h caches; R_m uses log10(M_filter) via FiLM."
        ),
    )
    cli_args = parser.parse_args()
    flag_experiments = (cli_args.experiments or []) + (cli_args.experiment_flags or [])
    if cli_args.experiment and flag_experiments:
        parser.error("Do not combine the positional experiment with --experiment/--experiments.")
    selected = flag_experiments or ([cli_args.experiment] if cli_args.experiment else [TARGET_EXPERIMENT])
    success = main(
        selected,
        held_out_galaxies=cli_args.held_out_galaxy,
        continue_on_error=cli_args.continue_on_error,
        use_film_override=cli_args.use_film,
    )
    raise SystemExit(0 if success else 1)
