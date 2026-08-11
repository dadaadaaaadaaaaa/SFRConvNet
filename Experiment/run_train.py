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
}

# Shorthand experiments use the current instantaneous-SFR, no-initial-stellar-
# density SFRConvNet recipe. Explicit ExperimentSetting.json keys remain fully
# supported for other architectures.
for _experiment_number in range(2, 8):
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


def _canonical_shorthand(value):
    value = str(value).strip()
    if value.lower().startswith("exp") and value[3:].isdigit():
        number = int(value[3:])
        if 1 <= number <= 7:
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
    
    # Flatten the configuration
    # 展平配置
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
            fold_config["exp_name"] = f"Exp2_LOO{short_group}"
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
        help="Repeatable experiment key or shorthand such as exp3.",
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
            "to prevent KS-law SFR from entering SFRConvNet via FiLM."
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
