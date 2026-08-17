"""Analyze one or more leave-one-galaxy-out Exp2 training results.

This script deliberately separates four evaluation domains:
  1. cached in-distribution test snapshots (test.pt),
  2. cached held-out galaxy snapshots (out.pt),
  3. complete raw trajectories for galaxies used by training,
  4. complete raw trajectory for the held-out galaxy.

The in-distribution and raw in-domain branches are optional. Disabling either
branch skips its dataset loading/inference as well as its figures and outputs.
The two held-out branches are always evaluated.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VisualScripts.ResultAnalysis import analysis_utils as utils


def _unique_groups(dataset) -> set[str]:
    names = getattr(dataset, "group_names", None)
    if names is None:
        return set()
    return {str(value) for value in np.asarray(names).reshape(-1).tolist()}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def _short_group_name(group_name: str) -> str:
    """Convert disk_galaxy_fiducial_4 to the concise label fiducial4."""
    name = str(group_name).strip().lower()
    name = name.removeprefix("disk_galaxy_")
    return re.sub(r"[^a-z0-9]+", "", name)


def _compact_group_title(domain_title: str, group_name: str) -> str:
    domain_title = {
        "Held-Out Galaxy Test": "Held-Out Test",
    }.get(domain_title, domain_title)
    return f"{domain_title} {_short_group_name(group_name)}"


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read JSON file: {path}") from exc


def _find_model(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("*best_model.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best model in {result_dir}, found {len(candidates)}: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def _validate_logo_contract(metadata: dict, test_ds, out_ds) -> None:
    if metadata.get("split_strategy") != "leave_one_galaxy_out":
        raise RuntimeError(
            "Exp2 analysis requires split_strategy='leave_one_galaxy_out'; "
            f"got {metadata.get('split_strategy')!r}."
        )
    train_groups = set(metadata.get("train_groups", ()))
    held_out_groups = set(metadata.get("held_out_groups", ()))
    if not train_groups or len(held_out_groups) != 1:
        raise RuntimeError(
            "Invalid Exp2 metadata: train_groups must be non-empty and exactly one "
            f"held_out_group is required. Got train={sorted(train_groups)}, "
            f"held_out={sorted(held_out_groups)}."
        )
    if train_groups.intersection(held_out_groups):
        raise RuntimeError("Exp2 metadata contains overlapping train and held-out groups.")

    actual_out = _unique_groups(out_ds)
    if actual_out != held_out_groups:
        raise RuntimeError(
            f"out.pt groups {sorted(actual_out)} do not match held_out_groups "
            f"{sorted(held_out_groups)}."
        )
    if test_ds is not None:
        actual_test = _unique_groups(test_ds)
        if actual_test != train_groups:
            raise RuntimeError(
                f"test.pt groups {sorted(actual_test)} do not match train_groups "
                f"{sorted(train_groups)}."
            )
        if actual_test.intersection(actual_out):
            raise RuntimeError("Actual cached test and out data share a galaxy group.")
        test_status = sorted(actual_test)
    else:
        # Metadata still enforces disjoint train/held-out galaxy identities.
        # The potentially large test.pt tensor is deliberately not loaded when
        # its analysis branch is disabled.
        test_status = "skipped (metadata-only validation)"
    print(
        "[PASS] Analysis split contract: "
        f"in-domain={test_status}, held-out={sorted(actual_out)}"
    )


def _concat_prediction_frames(frames, target_metadata):
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    result.attrs["target_metadata"] = dict(target_metadata or {})
    return result


def _predict_raw_domain(
    group_names,
    *,
    metadata,
    mean,
    std,
    reference_dataset,
    model,
    model_type,
    device,
    batch_size,
    domain_label,
):
    group_paths = metadata.get("group_paths", {})
    frames = []
    for group_name in group_names:
        if group_name not in group_paths:
            raise RuntimeError(f"No raw HDF5 path recorded for group {group_name!r}.")
        path = Path(group_paths[group_name])
        if not path.is_dir():
            raise FileNotFoundError(f"Recorded HDF5 directory does not exist: {path}")
        dataset = utils.HDF5Dataset(
            (str(path), group_name),
            mean=mean,
            std=std,
            mirror=True,
            reference_dataset=reference_dataset,
        )
        if dataset.type != group_name:
            raise RuntimeError(
                f"Raw path identity mismatch: metadata says {group_name!r}, "
                f"but path resolves to {dataset.type!r}."
            )
        print(f"  Predicting {domain_label}: {group_name} ({len(dataset)} snapshots)")
        frame = utils.predict_dataset(
            dataset, model, device, model_type, batch_size=batch_size
        )
        frame["evaluation_domain"] = domain_label
        frame["type"] = group_name
        frame.attrs["target_metadata"] = dataset.target_metadata
        frames.append(frame)
        del dataset
        gc.collect()
    return _concat_prediction_frames(frames, reference_dataset.target_metadata)


def _save_domain_outputs(result_dir, key, frame, title, fig_dir, metrics, *, make_plots=True):
    if frame is None or frame.empty:
        return
    frame.to_csv(result_dir / f"{key}_predictions.csv", index=False)
    metrics.append(utils.get_metrics_summary(frame["y_true"], frame["y_pred"], title))
    if not make_plots:
        return
    utils.plot_scatter(frame, title, fig_dir)
    utils.plot_error_dist(frame, title, fig_dir)
    for group_name, group_frame in frame.groupby("type", sort=True):
        group_frame = group_frame.sort_values("time") if "time" in group_frame else group_frame
        group_frame.attrs["target_metadata"] = frame.attrs.get("target_metadata", {})
        group_title = _compact_group_title(title, group_name)
        utils.plot_sfr_history(group_frame, group_title, fig_dir)
        utils.plot_cumulative_sfr(group_frame, group_title, fig_dir)


def _compare_held_out_predictions(cached_frame, raw_frame):
    """Verify that cached out.pt and raw-HDF5 inference describe the same trajectory."""
    sort_columns = [column for column in ("type", "time") if column in cached_frame.columns]
    cached = cached_frame.sort_values(sort_columns).reset_index(drop=True)
    raw = raw_frame.sort_values(sort_columns).reset_index(drop=True)
    if len(cached) != len(raw):
        raise RuntimeError(
            "Cached held-out and raw held-out predictions contain different sample counts: "
            f"{len(cached)} != {len(raw)}."
        )
    time_max_abs_diff = None
    if "time" in cached:
        time_diff = np.abs(cached["time"].to_numpy() - raw["time"].to_numpy())
        time_max_abs_diff = float(np.max(time_diff))
        # Cached PT time is commonly float32 while raw HDF5 time may be
        # float64. 1e-5 Gyr is only 0.8% of the current 0.00125-Gyr snapshot
        # cadence, so it accepts representation round-off but not a one-frame
        # temporal displacement.
        time_atol_gyr = 1e-5
        if time_max_abs_diff > time_atol_gyr:
            mismatch_index = int(np.argmax(time_diff))
            raise RuntimeError(
                "Cached held-out and raw held-out time grids do not match: "
                f"max |time diff|={time_max_abs_diff:.3e} Gyr at row {mismatch_index} "
                f"(cached={cached.loc[mismatch_index, 'time']:.12g}, "
                f"raw={raw.loc[mismatch_index, 'time']:.12g})."
            )
    true_max_abs_diff = float(np.max(np.abs(cached["y_true"] - raw["y_true"])))
    pred_max_abs_diff = float(np.max(np.abs(cached["y_pred"] - raw["y_pred"])))
    if true_max_abs_diff > 1e-6 or pred_max_abs_diff > 1e-5:
        raise RuntimeError(
            "Cached held-out and raw held-out inference are inconsistent: "
            f"max |true diff|={true_max_abs_diff:.3e}, "
            f"max |prediction diff|={pred_max_abs_diff:.3e}."
        )
    print(
        "[PASS] Cached held-out and raw held-out trajectories agree: "
        f"max |time diff|={time_max_abs_diff:.3e} Gyr, "
        f"max |prediction diff|={pred_max_abs_diff:.3e}"
    )
    return {
        "sample_count": len(cached),
        "max_abs_time_diff_gyr": time_max_abs_diff,
        "max_abs_y_true_diff": true_max_abs_diff,
        "max_abs_y_pred_diff": pred_max_abs_diff,
    }


def analyze_result(
    result_dir: Path,
    device: torch.device,
    *,
    plot_in_distribution: bool = True,
    plot_in_domain: bool = True,
    batch_size: int = 128,
) -> Path:
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    result_dir = result_dir.resolve()
    hyperparameters_path = result_dir / "hyperparameters.json"
    if not hyperparameters_path.exists():
        raise FileNotFoundError(f"Missing hyperparameters.json in {result_dir}")
    hyperparameters = _load_json(hyperparameters_path)
    data_exp = hyperparameters.get("data_exp")
    if not data_exp:
        raise RuntimeError(f"Training result {result_dir} does not record data_exp.")

    cache_dir = Path(hyperparameters.get("cache_dir", ".cache"))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_prefix = cache_dir / data_exp
    metadata = utils.load_cache_metadata(cache_prefix)
    test_path = Path(f"{cache_prefix}test.pt")
    out_path = Path(f"{cache_prefix}out.pt")
    norm_path = Path(f"{cache_prefix}norm.npz")
    required_paths = [out_path, norm_path]
    if plot_in_distribution:
        required_paths.append(test_path)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(f"Required Exp2 artifact is missing: {required}")

    print("\n" + "=" * 80)
    print(f"Analyzing: {result_dir.name}")
    print(f"Cache prefix: {data_exp}")
    print(f"Analysis inference batch size: {batch_size}")
    test_ds = utils.CachedPTDataset(test_path) if plot_in_distribution else None
    out_ds = utils.CachedPTDataset(out_path)
    _validate_logo_contract(metadata, test_ds, out_ds)

    model_path = _find_model(result_dir)
    model, model_type = utils.load_analysis_model(
        model_path,
        device,
        {"test": test_ds, "out": out_ds},
    )
    norm = utils.load_norm_stats(norm_path)
    mean, std = norm["mean"], norm["std"]
    reference_ds = test_ds if test_ds is not None else out_ds

    test_frame = None
    if plot_in_distribution:
        test_frame = utils.predict_dataset(
            test_ds, model, device, model_type, batch_size=batch_size
        )
        test_frame["evaluation_domain"] = "In-Distribution Test"
    else:
        print("  Skipping In-Distribution Test loading, inference, outputs, and figures.")

    out_frame = utils.predict_dataset(
        out_ds, model, device, model_type, batch_size=batch_size
    )
    out_frame["evaluation_domain"] = "Held-Out Galaxy Test"

    raw_in = None
    if plot_in_domain:
        raw_in = _predict_raw_domain(
            metadata["train_groups"],
            metadata=metadata,
            mean=mean,
            std=std,
            reference_dataset=reference_ds,
            model=model,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            domain_label="Raw In-Domain",
        )
    else:
        print("  Skipping Raw In-Domain loading, inference, outputs, and figures.")

    raw_out = _predict_raw_domain(
        metadata["held_out_groups"],
        metadata=metadata,
        mean=mean,
        std=std,
        reference_dataset=reference_ds,
        model=model,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        domain_label="Raw Held-Out",
    )
    held_out_consistency = _compare_held_out_predictions(out_frame, raw_out)

    if plot_in_distribution and plot_in_domain:
        figure_dir_name = "figures_all"
    elif plot_in_distribution:
        figure_dir_name = "figures_held_out_in_dist"
    elif plot_in_domain:
        figure_dir_name = "figures_held_out_in_domain"
    else:
        figure_dir_name = "figures_held_out"
    fig_dir = result_dir / figure_dir_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    _save_domain_outputs(
        result_dir, "in_dist_test", test_frame,
        "In-Distribution Test", fig_dir, metrics,
        make_plots=True,
    )
    _save_domain_outputs(
        result_dir, "held_out_test", out_frame,
        "Held-Out Test", fig_dir, metrics,
        make_plots=True,
    )
    _save_domain_outputs(
        result_dir, "raw_in_domain", raw_in,
        "Raw In-Domain", fig_dir, metrics,
        make_plots=True,
    )
    _save_domain_outputs(
        result_dir, "raw_held_out", raw_out,
        "Raw Held-Out", fig_dir, metrics,
        make_plots=True,
    )

    metrics_path = result_dir / "metrics_exp2.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    analysis_metadata = {
        "format_version": 2,
        "model": model_path.name,
        "data_exp": data_exp,
        "split_strategy": metadata["split_strategy"],
        "train_groups": metadata["train_groups"],
        "held_out_groups": metadata["held_out_groups"],
        "plot_options": {
            "plot_in_distribution": bool(plot_in_distribution),
            "plot_in_domain": bool(plot_in_domain),
            "plot_held_out": True,
        },
        "domain_execution": {
            "in_distribution_test": bool(plot_in_distribution),
            "held_out_galaxy_test": True,
            "raw_in_domain": bool(plot_in_domain),
            "raw_held_out": True,
        },
        "figure_directory": figure_dir_name,
        "inference_batch_size": batch_size,
        "held_out_cache_raw_consistency": held_out_consistency,
        "outputs": {
            "in_distribution_test": (
                "in_dist_test_predictions.csv" if plot_in_distribution else None
            ),
            "held_out_galaxy_test": "held_out_test_predictions.csv",
            "raw_in_domain": "raw_in_domain_predictions.csv" if plot_in_domain else None,
            "raw_held_out": "raw_held_out_predictions.csv",
            "metrics": metrics_path.name,
        },
    }
    with (result_dir / "exp2_analysis_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(analysis_metadata, file, indent=2)
    print(f"Saved domain-separated Exp2 analysis to: {result_dir}")
    return metrics_path


def _discover_latest_fold_results(results_root: Path):
    latest_by_cache = {}
    for hyperparameters_path in results_root.glob("*/hyperparameters.json"):
        try:
            params = _load_json(hyperparameters_path)
        except RuntimeError:
            continue
        split = params.get("data_split_metadata", {})
        if split.get("split_strategy") != "leave_one_galaxy_out":
            continue
        data_exp = params.get("data_exp")
        run_stamp = str(params.get("run_stamp", ""))
        previous = latest_by_cache.get(data_exp)
        if previous is None or run_stamp > previous[0]:
            latest_by_cache[data_exp] = (run_stamp, hyperparameters_path.parent)
    return [value[1] for _, value in sorted(latest_by_cache.items())]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze domain-separated Exp2 results.")
    parser.add_argument(
        "--result-dir",
        action="append",
        type=Path,
        help="Result directory to analyze; repeat for multiple folds.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Analyze the latest trained result for every discovered Exp2 fold.",
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "Results")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size for ConvNet and non-ViT models (default: 128).",
    )
    parser.add_argument(
        "--plot-in-distribution",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help=(
            "Run cached in-distribution inference and save its outputs/figures "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--plot-in-domain",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help=(
            "Run complete raw in-domain inference and save its outputs/figures "
            "(default: true)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.result_dir and args.all_folds:
        raise SystemExit("Do not combine --result-dir with --all-folds.")
    if args.all_folds:
        result_dirs = _discover_latest_fold_results(args.results_root)
    else:
        result_dirs = args.result_dir or []
    if not result_dirs:
        raise SystemExit("No result directories selected or discovered.")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    summary_frames = []
    for result_dir in result_dirs:
        metrics_path = analyze_result(
            Path(result_dir),
            device,
            plot_in_distribution=args.plot_in_distribution,
            plot_in_domain=args.plot_in_domain,
            batch_size=args.batch_size,
        )
        frame = pd.read_csv(metrics_path)
        analysis_metadata = _load_json(Path(result_dir) / "exp2_analysis_metadata.json")
        held_out = _short_group_name(analysis_metadata["held_out_groups"][0])
        frame.insert(0, "HeldOut", held_out)
        frame.insert(0, "ResultDir", Path(result_dir).name)
        summary_frames.append(frame)
    if len(summary_frames) > 1:
        all_metrics = pd.concat(summary_frames, ignore_index=True)
        details_path = args.results_root / "Exp2_LOO_metrics_all_folds.csv"
        all_metrics.to_csv(details_path, index=False)
        numeric_columns = ["RMSE", "MAE", "R2", "OutlierFrac (|target error|>1)"]
        summary = all_metrics.groupby("Dataset")[numeric_columns].agg(["mean", "std", "min", "max"])
        summary_path = args.results_root / "Exp2_LOO_cross_fold_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved all-fold metrics: {details_path}")
        print(f"Saved cross-fold summary: {summary_path}")


if __name__ == "__main__":
    main()
