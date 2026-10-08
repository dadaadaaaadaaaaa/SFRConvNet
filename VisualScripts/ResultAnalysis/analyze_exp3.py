"""分析 Exp3 的 torus 域外测试结果。

只报告用户指定的总体物理 SFR RMSE、MAE、R²，以及全时间段积分 SFR
低估百分比；自动适配 direct-SFR 与 signed-closure checkpoint，图表功能与
Exp2 的测试域保持一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VisualScripts.ResultAnalysis import analysis_utils as utils


SUPPORTED_OBJECTIVE_MODES = frozenset({"direct_sfr", "signed_closure"})


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 JSON：{path}") from exc


def _find_model(result_dir: Path) -> Path:
    candidates = sorted(result_dir.glob("*best_model.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{result_dir} 应恰好包含一个 best model，实际为："
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def _validate_objective_mode(hyperparameters: dict) -> str:
    """读取并验证分析所支持的训练目标，不根据结果目录名称猜测。"""

    objective_mode = hyperparameters.get("objective_mode")
    if objective_mode not in SUPPORTED_OBJECTIVE_MODES:
        raise RuntimeError(
            "Exp3 分析只支持 objective_mode='direct_sfr' 或 "
            f"'signed_closure'，实际为 {objective_mode!r}。"
        )
    return str(objective_mode)


def _unique_groups(dataset) -> set[str]:
    values = getattr(dataset, "group_names", None)
    if values is None:
        return set()
    return {str(value) for value in np.asarray(values).reshape(-1).tolist()}


def _validate_exp3_contract(metadata: dict, test_dataset) -> str:
    if metadata.get("split_strategy") != "random_snapshot_75_25_torus_test":
        raise RuntimeError(
            "Exp3 分析要求 split_strategy='random_snapshot_75_25_torus_test'，"
            f"实际为 {metadata.get('split_strategy')!r}。"
        )
    train_groups = set(metadata.get("train_groups", ()))
    validation_groups = set(metadata.get("validation_groups", ()))
    test_groups = set(metadata.get("test_groups", ()))
    if not train_groups or validation_groups != train_groups:
        raise RuntimeError("Exp3 metadata 中 train/validation groups 不完整或不一致。")
    if len(test_groups) != 1 or train_groups.intersection(test_groups):
        raise RuntimeError("Exp3 test 必须是与 Zou 训练域不相交的单一 group。")
    actual_test_groups = _unique_groups(test_dataset)
    if actual_test_groups != test_groups:
        raise RuntimeError(
            f"test.pt groups={sorted(actual_test_groups)}，"
            f"metadata 预期为 {sorted(test_groups)}。"
        )
    test_group = next(iter(test_groups))
    if test_group != "disk_galaxy_torus_100yr":
        raise RuntimeError(f"Exp3 test group 不是 torus：{test_group!r}。")
    return test_group


def _physical_metrics(frame: pd.DataFrame) -> dict[str, float]:
    true = frame["sfr_true"].to_numpy(dtype=np.float64)
    pred = frame["sfr_pred"].to_numpy(dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not np.any(valid):
        raise RuntimeError("torus test 没有有限的物理 SFR 真值/预测。")
    true = true[valid]
    pred = pred[valid]
    error = pred - true
    residual = float(np.sum(error**2))
    total = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float(residual == 0)
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(r2),
    }


def _total_period_sfr_underestimate_percent(frame: pd.DataFrame) -> float:
    required = {"time", "sfr_true", "sfr_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"全时间段 SFR 误差缺少字段：{sorted(missing)}")
    ordered = frame.sort_values("time").drop_duplicates("time")
    if len(ordered) < 2:
        raise RuntimeError("全时间段 SFR 积分至少需要两个不同时间点。")
    time = ordered["time"].to_numpy(dtype=np.float64)
    true = ordered["sfr_true"].to_numpy(dtype=np.float64)
    pred = ordered["sfr_pred"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise RuntimeError("torus 时间坐标必须有限且严格递增。")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise RuntimeError("全时间段 SFR 积分包含非有限值。")
    true_integral = float(np.trapezoid(true, time))
    pred_integral = float(np.trapezoid(pred, time))
    if true_integral <= 0:
        raise RuntimeError("真实全时间段积分 SFR 必须为正。")
    # 与 Exp2 notebook 的口径一致：正值表示模型低估，负值表示模型高估。
    return 100.0 * (true_integral - pred_integral) / true_integral


def analyze_result(
    result_dir: Path,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> Path:
    result_dir = result_dir.resolve()
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数。")
    hyperparameters_path = result_dir / "hyperparameters.json"
    if not hyperparameters_path.exists():
        raise FileNotFoundError(f"结果目录缺少 hyperparameters.json：{result_dir}")
    hyperparameters = _load_json(hyperparameters_path)
    objective_mode = _validate_objective_mode(hyperparameters)
    data_exp = hyperparameters.get("data_exp")
    if not data_exp:
        raise RuntimeError("hyperparameters.json 缺少 data_exp。")

    cache_dir = Path(hyperparameters.get("cache_dir", ".cache"))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    cache_prefix = cache_dir / data_exp
    test_path = Path(f"{cache_prefix}test.pt")
    metadata_path = Path(f"{cache_prefix}metadata.json")
    for required in (test_path, metadata_path):
        if not required.exists():
            raise FileNotFoundError(f"缺少 Exp3 分析输入：{required}")

    metadata = utils.load_cache_metadata(cache_prefix)
    test_dataset = utils.CachedPTDataset(test_path)
    test_group = _validate_exp3_contract(metadata, test_dataset)
    model_path = _find_model(result_dir)
    model, model_type = utils.load_analysis_model(
        model_path,
        device,
        {"test": test_dataset},
    )
    model_objective_mode = getattr(model, "objective_mode", None)
    if model_objective_mode != objective_mode:
        raise RuntimeError(
            "hyperparameters 与 checkpoint 的 objective_mode 不一致："
            f"{objective_mode!r} != {model_objective_mode!r}。"
        )
    frame = utils.predict_dataset(
        test_dataset,
        model,
        device,
        model_type,
        batch_size=batch_size,
    )
    frame["evaluation_domain"] = "Torus Test"
    frame.to_csv(result_dir / "torus_test_predictions.csv", index=False)

    metrics = _physical_metrics(frame)
    metrics["TotalPeriod_SFR_UnderestimatePct"] = (
        _total_period_sfr_underestimate_percent(frame)
    )
    metrics_path = result_dir / "metrics_exp3.csv"
    pd.DataFrame([{"Dataset": "Torus Test", **metrics}]).to_csv(
        metrics_path, index=False
    )

    figure_dir = result_dir / "figures_exp3"
    figure_dir.mkdir(parents=True, exist_ok=True)
    title = "Torus Test"
    utils.plot_scatter(frame, title, figure_dir)
    utils.plot_error_dist(frame, title, figure_dir)
    utils.plot_sfr_history(frame, title, figure_dir)
    utils.plot_cumulative_sfr(frame, title, figure_dir)

    analysis_metadata = {
        "format_version": 1,
        "model": model_path.name,
        "data_exp": data_exp,
        "objective_mode": objective_mode,
        "split_strategy": metadata["split_strategy"],
        "train_groups": metadata["train_groups"],
        "test_group": test_group,
        "inference_batch_size": batch_size,
        "metric_space": "physical_sfr",
        "total_period_error_definition": (
            "100 * (integral_true_sfr - integral_predicted_sfr) / "
            "integral_true_sfr; positive means underestimation"
        ),
        "outputs": {
            "predictions": "torus_test_predictions.csv",
            "metrics": metrics_path.name,
            "figures": figure_dir.name,
        },
    }
    with (result_dir / "exp3_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(analysis_metadata, file, indent=2, ensure_ascii=False)

    print(f"Exp3 torus predictions: {result_dir / 'torus_test_predictions.csv'}")
    print(f"Exp3 metrics: {metrics_path}")
    print(f"Exp3 objective mode: {objective_mode}")
    print(pd.DataFrame([{"Dataset": "Torus Test", **metrics}]).to_string(index=False))
    return metrics_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="分析 Exp3 torus test 结果。")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或合法 torch device。")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    analyze_result(
        args.result_dir,
        torch.device(device_name),
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
