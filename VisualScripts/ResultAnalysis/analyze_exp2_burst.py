"""评估 Exp2 留出 Model0.75 在论文 starburst 时段内的预测表现。

脚本直接读取 analyze_exp2.py 已生成的 raw_held_out_predictions.csv，不会重新
加载模型或执行推理。Zou et al. (2026) 指出 Model0.75 在生产时标约 6 Gyr
快速升高，并一直维持高 SFR 到模拟结束，因此该事件的结束时间是右删失的。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELAXATION_TIME_GYR = 0.5625
MACER_MASS_CODE_MSUN = 2.5e7
GYR_TO_YR = 1.0e9
PAPER_BURST_START_GYR = 6.0


def _json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _conversion_to_msun_per_year(target_units: str) -> float:
    normalized = target_units.replace(" ", "").lower()
    if normalized in {"macer_mass_code/gyr", "macer-mass-code/gyr"}:
        return MACER_MASS_CODE_MSUN / GYR_TO_YR
    if normalized in {"msun/yr", "m_sun/yr", "solar_mass/yr"}:
        return 1.0
    raise RuntimeError(
        f"尚未定义 target_units={target_units!r} 到 Msun/yr 的转换；"
        "请先核对 cache 的目标单位契约。"
    )


def _regression_metrics(true, pred) -> dict:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[valid], pred[valid]
    if true.size == 0:
        raise RuntimeError("没有可用于 burst 评估的有限数值。")
    error = pred - true
    absolute = np.abs(error)
    denominator = np.maximum(np.abs(true), np.finfo(np.float64).eps)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    pearson = float(np.corrcoef(true, pred)[0, 1]) if true.size > 1 else float("nan")
    true_rank = pd.Series(true).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if true.size > 1
        else float("nan")
    )
    return {
        "n_samples": int(true.size),
        "mae_msun_per_yr": float(np.mean(absolute)),
        "rmse_msun_per_yr": float(np.sqrt(np.mean(error**2))),
        "bias_msun_per_yr": float(np.mean(error)),
        "median_abs_error_msun_per_yr": float(np.median(absolute)),
        "p90_abs_error_msun_per_yr": float(np.quantile(absolute, 0.90)),
        "max_abs_error_msun_per_yr": float(np.max(absolute)),
        "median_abs_relative_error": float(np.median(absolute / denominator)),
        "mean_abs_relative_error": float(np.mean(absolute / denominator)),
        "r2": None if ss_tot == 0 else 1.0 - ss_res / ss_tot,
        "pearson_r": _json_float(pearson),
        "spearman_r": _json_float(spearman),
    }


def _integral_metrics(time_gyr, true, pred) -> dict:
    time_gyr = np.asarray(time_gyr, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    true_mass = float(np.trapezoid(true, time_gyr) * GYR_TO_YR)
    pred_mass = float(np.trapezoid(pred, time_gyr) * GYR_TO_YR)
    signed_error = pred_mass - true_mass
    absolute_curve_error = float(
        np.trapezoid(np.abs(pred - true), time_gyr) * GYR_TO_YR
    )
    return {
        "integrated_true_formed_mass_msun": true_mass,
        "integrated_pred_formed_mass_msun": pred_mass,
        "integrated_signed_error_msun": signed_error,
        "integrated_relative_error": signed_error / true_mass if true_mass else None,
        "integrated_absolute_curve_error_msun": absolute_curve_error,
        "normalized_integrated_absolute_error": (
            absolute_curve_error / abs(true_mass) if true_mass else None
        ),
    }


def _segments(time, above, cadence_gyr, minimum_duration_gyr) -> list[dict]:
    indices = np.flatnonzero(np.asarray(above, dtype=bool))
    if indices.size == 0:
        return []
    raw_segments = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index != previous + 1:
            raw_segments.append((start, previous))
            start = index
        previous = index
    raw_segments.append((start, previous))
    segments = []
    for start, end in raw_segments:
        duration = float(time[end] - time[start] + cadence_gyr)
        if duration + 1e-12 < minimum_duration_gyr:
            continue
        segments.append(
            {
                "start_gyr": float(time[start]),
                "end_gyr": float(time[end]),
                "duration_gyr": duration,
            }
        )
    return segments


def _timing_analysis(
    frame: pd.DataFrame,
    smooth_width_gyr: float,
    threshold_multiplier: float,
    minimum_duration_gyr: float,
) -> tuple[dict, pd.DataFrame]:
    time = frame["paper_time_gyr"].to_numpy(dtype=np.float64)
    cadence = float(np.median(np.diff(time)))
    window = max(3, int(round(smooth_width_gyr / cadence)))
    if window % 2 == 0:
        window += 1
    minimum_periods = window // 2 + 1
    true_smoothed = frame["sfr_true_msun_per_yr"].rolling(
        window, center=True, min_periods=minimum_periods
    ).median()
    pred_smoothed = frame["sfr_pred_msun_per_yr"].rolling(
        window, center=True, min_periods=minimum_periods
    ).median()

    # 论文没有定义 starburst 的 SFR 阈值。这里使用同一个、可复现的后处理阈值
    # 检测真值和预测：真实 pre-burst 平滑 SFR 中位数的固定倍数。
    pre_mask = (frame["paper_time_gyr"] >= 1.0) & (
        frame["paper_time_gyr"] < 5.5
    )
    pre_values = true_smoothed[pre_mask].dropna()
    if pre_values.empty:
        raise RuntimeError("无法从 1.0--5.5 Gyr 估计 pre-burst 背景 SFR。")
    preburst_median = float(pre_values.median())
    threshold = threshold_multiplier * preburst_median
    search_mask = time >= 4.0
    true_above = search_mask & (true_smoothed.to_numpy() >= threshold)
    pred_above = search_mask & (pred_smoothed.to_numpy() >= threshold)
    true_segments = _segments(
        time, true_above, cadence, minimum_duration_gyr
    )
    pred_segments = _segments(
        time, pred_above, cadence, minimum_duration_gyr
    )
    true_primary = true_segments[0] if true_segments else None
    pred_primary = pred_segments[0] if pred_segments else None
    observed_end = float(time[-1])
    true_censored = bool(
        true_primary and true_primary["end_gyr"] >= observed_end - cadence / 2
    )
    pred_censored = bool(
        pred_primary and pred_primary["end_gyr"] >= observed_end - cadence / 2
    )

    start_error = None
    if true_primary and pred_primary:
        start_error = pred_primary["start_gyr"] - true_primary["start_gyr"]
    end_error = None
    premature_end_lower_bound = None
    if true_primary and pred_primary:
        if true_censored and not pred_censored:
            premature_end_lower_bound = observed_end - pred_primary["end_gyr"]
        elif not true_censored:
            end_error = pred_primary["end_gyr"] - true_primary["end_gyr"]

    paper_mask = time >= PAPER_BURST_START_GYR
    true_state, pred_state = true_above[paper_mask], pred_above[paper_mask]
    tp = int(np.sum(true_state & pred_state))
    fp = int(np.sum(~true_state & pred_state))
    fn = int(np.sum(true_state & ~pred_state))
    tn = int(np.sum(~true_state & ~pred_state))
    timing = {
        "definition": {
            "smoothing": "centered rolling median",
            "smooth_width_gyr": smooth_width_gyr,
            "smooth_window_samples": window,
            "preburst_baseline_window_gyr": [1.0, 5.5],
            "preburst_true_median_sfr_msun_per_yr": preburst_median,
            "threshold_rule": "threshold_multiplier * true_preburst_median",
            "threshold_multiplier": threshold_multiplier,
            "threshold_sfr_msun_per_yr": threshold,
            "minimum_segment_duration_gyr": minimum_duration_gyr,
            "event_search_start_gyr": 4.0,
        },
        "true_segments": true_segments,
        "predicted_segments": pred_segments,
        "true_primary_start_gyr": (
            None if true_primary is None else true_primary["start_gyr"]
        ),
        "predicted_primary_start_gyr": (
            None if pred_primary is None else pred_primary["start_gyr"]
        ),
        "start_time_error_pred_minus_true_gyr": start_error,
        "paper_start_gyr_approximate": PAPER_BURST_START_GYR,
        "true_start_minus_paper_start_gyr": (
            None if true_primary is None
            else true_primary["start_gyr"] - PAPER_BURST_START_GYR
        ),
        "predicted_start_minus_paper_start_gyr": (
            None if pred_primary is None
            else pred_primary["start_gyr"] - PAPER_BURST_START_GYR
        ),
        "true_primary_end_gyr": (
            None if true_primary is None else true_primary["end_gyr"]
        ),
        "predicted_primary_end_gyr": (
            None if pred_primary is None else pred_primary["end_gyr"]
        ),
        "true_end_is_right_censored": true_censored,
        "predicted_end_is_right_censored": pred_censored,
        "end_time_error_pred_minus_true_gyr": end_error,
        "premature_end_lower_bound_gyr": premature_end_lower_bound,
        "observed_end_gyr": observed_end,
        "paper_window_state_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "paper_window_burst_recall": tp / (tp + fn) if tp + fn else None,
        "paper_window_burst_precision": tp / (tp + fp) if tp + fp else None,
        "paper_window_burst_f1": (
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        ),
        "paper_window_burst_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "paper_window_true_duty_fraction": float(np.mean(true_state)),
        "paper_window_predicted_duty_fraction": float(np.mean(pred_state)),
    }
    enriched = frame.copy()
    enriched["sfr_true_smoothed_msun_per_yr"] = true_smoothed
    enriched["sfr_pred_smoothed_msun_per_yr"] = pred_smoothed
    enriched["event_threshold_msun_per_yr"] = threshold
    enriched["true_burst_state"] = true_above.astype(np.int8)
    enriched["predicted_burst_state"] = pred_above.astype(np.int8)
    return timing, enriched


def _plot(frame: pd.DataFrame, output_path: Path):
    burst = frame[frame["in_paper_burst_window"]]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(
        burst["paper_time_gyr"], burst["sfr_true_msun_per_yr"],
        color="black", lw=1.8, label="MACER truth",
    )
    axes[0].plot(
        burst["paper_time_gyr"], burst["sfr_pred_msun_per_yr"],
        color="#d62728", lw=1.5, label="Closure CNN",
    )
    axes[0].plot(
        burst["paper_time_gyr"], burst["sfr_macer_baseline_msun_per_yr"],
        color="#1f77b4", lw=1.2, alpha=0.8, label="MACER-LR baseline",
    )
    axes[0].set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    axes[0].set_title("Model0.75 held-out burst evaluation")
    axes[0].legend(ncol=3, fontsize=9)

    axes[1].plot(
        burst["paper_time_gyr"], burst["signed_error_msun_per_yr"],
        color="#9467bd", lw=1.3,
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].fill_between(
        burst["paper_time_gyr"], 0, burst["signed_error_msun_per_yr"],
        where=burst["signed_error_msun_per_yr"] >= 0,
        color="#d62728", alpha=0.18, label="over-prediction",
    )
    axes[1].fill_between(
        burst["paper_time_gyr"], 0, burst["signed_error_msun_per_yr"],
        where=burst["signed_error_msun_per_yr"] < 0,
        color="#1f77b4", alpha=0.18, label="under-prediction",
    )
    axes[1].set_ylabel(r"$\widehat{\mathrm{SFR}}-\mathrm{SFR}$")
    axes[1].legend(fontsize=9)

    relative_axis = axes[2].twinx()
    absolute_line = axes[2].plot(
        burst["paper_time_gyr"], burst["absolute_error_msun_per_yr"],
        color="#ff7f0e", lw=1.2, label="absolute error",
    )
    relative_line = relative_axis.plot(
        burst["paper_time_gyr"], 100 * burst["symmetric_relative_error"],
        color="#2ca02c", lw=1.0, alpha=0.85,
        label="symmetric relative error [%]",
    )
    axes[2].set_ylabel(r"Absolute error [$M_\odot\,\mathrm{yr}^{-1}$]")
    relative_axis.set_ylabel("Symmetric relative error [%]", color="#2ca02c")
    relative_axis.tick_params(axis="y", colors="#2ca02c")
    axes[2].set_xlabel("Production-run time [Gyr]")
    axes[2].legend(
        absolute_line + relative_line,
        [line.get_label() for line in absolute_line + relative_line],
        fontsize=9,
    )
    for axis in axes:
        axis.axvline(PAPER_BURST_START_GYR, color="0.35", ls="--", lw=1)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_timing(frame: pd.DataFrame, timing: dict, output_path: Path):
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(
        frame["paper_time_gyr"], frame["sfr_true_smoothed_msun_per_yr"],
        color="black", lw=2.0, label="smoothed truth",
    )
    axis.plot(
        frame["paper_time_gyr"], frame["sfr_pred_smoothed_msun_per_yr"],
        color="#d62728", lw=1.7, label="smoothed prediction",
    )
    threshold = timing["definition"]["threshold_sfr_msun_per_yr"]
    axis.axhline(
        threshold, color="#2ca02c", ls="--", lw=1.2,
        label="event threshold",
    )
    axis.axvline(
        PAPER_BURST_START_GYR, color="0.35", ls=":", lw=1.2,
        label="paper start (approx.)",
    )
    for index, segment in enumerate(timing["true_segments"]):
        axis.axvspan(
            segment["start_gyr"], segment["end_gyr"],
            color="black", alpha=0.08,
            label="detected truth event" if index == 0 else None,
        )
    for index, segment in enumerate(timing["predicted_segments"]):
        axis.axvspan(
            segment["start_gyr"], segment["end_gyr"],
            color="#d62728", alpha=0.08,
            label="detected predicted event" if index == 0 else None,
        )
    axis.set_xlim(4.0, float(frame["paper_time_gyr"].max()))
    axis.set_xlabel("Production-run time [Gyr]")
    axis.set_ylabel(r"Smoothed SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    axis.set_title("Operational burst timing detection")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_burst(
    result_dir: Path,
    smooth_width_gyr: float = 0.5,
    threshold_multiplier: float = 3.0,
    minimum_duration_gyr: float = 0.25,
) -> Path:
    result_dir = result_dir.resolve()
    prediction_path = result_dir / "raw_held_out_predictions.csv"
    hyperparameters_path = result_dir / "hyperparameters.json"
    if not prediction_path.exists() or not hyperparameters_path.exists():
        raise FileNotFoundError(
            "需要 raw_held_out_predictions.csv 和 hyperparameters.json；"
            "请先运行 analyze_exp2.py。"
        )
    frame = pd.read_csv(prediction_path).sort_values("time").reset_index(drop=True)
    required = {
        "time", "type", "y_true", "y_pred", "sfr_true", "sfr_pred",
        "sfr_macer_baseline", "sfr_ks_baseline",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"预测 CSV 缺少字段：{sorted(missing)}")
    groups = frame["type"].dropna().unique().tolist()
    if groups != ["disk_galaxy_model_0_75"]:
        raise RuntimeError(
            "当前只定义了 Model0.75 的论文 burst 窗口；"
            f"实际 groups={groups!r}。"
        )
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    target_units = hyperparameters.get("target_units") or hyperparameters.get(
        "target_metadata", {}
    ).get("target_units")
    if not target_units:
        raise RuntimeError("hyperparameters.json 缺少 target_units。")
    conversion = _conversion_to_msun_per_year(target_units)

    frame["paper_time_gyr"] = frame["time"].astype(float) - RELAXATION_TIME_GYR
    for source in ("sfr_true", "sfr_pred", "sfr_macer_baseline", "sfr_ks_baseline"):
        frame[f"{source}_msun_per_yr"] = frame[source].astype(float) * conversion
    frame["signed_error_msun_per_yr"] = (
        frame["sfr_pred_msun_per_yr"] - frame["sfr_true_msun_per_yr"]
    )
    frame["absolute_error_msun_per_yr"] = np.abs(
        frame["signed_error_msun_per_yr"]
    )
    frame["absolute_relative_error"] = (
        frame["absolute_error_msun_per_yr"]
        / np.maximum(
            np.abs(frame["sfr_true_msun_per_yr"]),
            np.finfo(np.float64).eps,
        )
    )
    frame["symmetric_relative_error"] = (
        2 * frame["absolute_error_msun_per_yr"]
        / np.maximum(
            np.abs(frame["sfr_true_msun_per_yr"])
            + np.abs(frame["sfr_pred_msun_per_yr"]),
            np.finfo(np.float64).eps,
        )
    )
    frame["in_paper_burst_window"] = (
        frame["paper_time_gyr"] >= PAPER_BURST_START_GYR
    )
    burst = frame[frame["in_paper_burst_window"]].copy()
    if len(burst) < 2:
        raise RuntimeError("预测结果不覆盖论文 burst 窗口。")

    methods = {
        "Closure CNN": "sfr_pred_msun_per_yr",
        "MACER-LR baseline": "sfr_macer_baseline_msun_per_yr",
        "KS baseline": "sfr_ks_baseline_msun_per_yr",
    }
    true = burst["sfr_true_msun_per_yr"].to_numpy()
    time = burst["paper_time_gyr"].to_numpy()
    metric_rows = []
    for method, column in methods.items():
        pred = burst[column].to_numpy()
        row = {"method": method}
        row.update(_regression_metrics(true, pred))
        row.update(_integral_metrics(time, true, pred))
        metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)
    model_mse = metric_frame.loc[
        metric_frame["method"] == "Closure CNN", "rmse_msun_per_yr"
    ].iloc[0] ** 2
    model_mae = metric_frame.loc[
        metric_frame["method"] == "Closure CNN", "mae_msun_per_yr"
    ].iloc[0]
    for baseline in ("MACER-LR baseline", "KS baseline"):
        base_row = metric_frame[metric_frame["method"] == baseline].iloc[0]
        safe_baseline = baseline.replace(" ", "_").replace("-", "_").lower()
        metric_frame.loc[
            metric_frame["method"] == "Closure CNN",
            f"mse_skill_vs_{safe_baseline}",
        ] = 1 - model_mse / base_row["rmse_msun_per_yr"] ** 2
        metric_frame.loc[
            metric_frame["method"] == "Closure CNN",
            f"mae_skill_vs_{safe_baseline}",
        ] = 1 - model_mae / base_row["mae_msun_per_yr"]

    true_peak_index = burst["sfr_true_msun_per_yr"].idxmax()
    pred_peak_index = burst["sfr_pred_msun_per_yr"].idxmax()
    true_peak = float(frame.loc[true_peak_index, "sfr_true_msun_per_yr"])
    pred_peak = float(frame.loc[pred_peak_index, "sfr_pred_msun_per_yr"])
    peak_metrics = {
        "true_peak_sfr_msun_per_yr": true_peak,
        "true_peak_time_gyr": float(frame.loc[true_peak_index, "paper_time_gyr"]),
        "predicted_peak_sfr_msun_per_yr": pred_peak,
        "predicted_peak_time_gyr": float(
            frame.loc[pred_peak_index, "paper_time_gyr"]
        ),
        "peak_sfr_relative_error": (pred_peak - true_peak) / true_peak,
        "peak_time_error_pred_minus_true_gyr": float(
            frame.loc[pred_peak_index, "paper_time_gyr"]
            - frame.loc[true_peak_index, "paper_time_gyr"]
        ),
        "prediction_at_true_peak_msun_per_yr": float(
            frame.loc[true_peak_index, "sfr_pred_msun_per_yr"]
        ),
    }

    timing, frame = _timing_analysis(
        frame,
        smooth_width_gyr,
        threshold_multiplier,
        minimum_duration_gyr,
    )
    output_dir = result_dir / "burst_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame[frame["in_paper_burst_window"]].to_csv(
        output_dir / "burst_pointwise_errors.csv", index=False
    )
    metric_frame.to_csv(output_dir / "burst_metrics.csv", index=False)
    summary = {
        "format_version": 1,
        "held_out_group": groups[0],
        "paper_source": (
            "Zou et al. (2026), Section 3.3 and Figure 5."
        ),
        "paper_window": {
            "time_coordinate": "production-run time",
            "relaxation_time_subtracted_gyr": RELAXATION_TIME_GYR,
            "start_gyr_approximate": PAPER_BURST_START_GYR,
            "end_gyr": None,
            "end_definition": "through simulation end; right-censored",
            "actual_evaluated_start_gyr": float(burst["paper_time_gyr"].min()),
            "actual_evaluated_end_gyr": float(burst["paper_time_gyr"].max()),
            "n_samples": int(len(burst)),
        },
        "target_units_in_cache": target_units,
        "conversion_to_msun_per_yr": conversion,
        "peak_metrics": peak_metrics,
        "timing_metrics": timing,
    }
    (output_dir / "burst_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot(frame, output_dir / "burst_timeseries_errors.png")
    _plot_timing(frame, timing, output_dir / "burst_event_timing.png")
    print(f"已写入 burst 分析：{output_dir}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="评估 Exp2 Model0.75 burst 时段。")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--smooth-width-gyr", type=float, default=0.5)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--minimum-duration-gyr", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_burst(
        args.result_dir,
        smooth_width_gyr=args.smooth_width_gyr,
        threshold_multiplier=args.threshold_multiplier,
        minimum_duration_gyr=args.minimum_duration_gyr,
    )


if __name__ == "__main__":
    main()
