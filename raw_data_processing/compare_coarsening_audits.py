"""Rm-v0 与 Rm-v1 数值审计表的成对比较。

本模块仅比较粗粒化诊断结果。它不评估SFR 闭合算子，因此无法确定哪种方法对SFR 的预测效果更好。
同样，“dominant_destination”连通性是重叠软传输的argmax诊断指标，而非物理上
互斥的网格划分所具有的连通性。常规输出单元质量的 CV/基尼系数衡量的是CNN 网格的异质性，并非相对于等质量
目标的误差。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SFR_LIMITATION = (
    "No SFR operator/closure is evaluated here; numerical audit differences "
    "must not be interpreted as SFR accuracy or superiority."
)
PSEUDO_PARTITION_LIMITATION = (
    "Argmax connectivity describes a hard diagnostic derived from overlapping "
    "soft transport, not a mutually-exclusive physical partition."
)
OUTPUT_MASS_LIMITATION = (
    "Output-cell mass CV/Gini describe regular CNN-grid mass heterogeneity; "
    "output cells are not constrained to equal mass or to mass_filter."
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    aliases: tuple[str, ...]
    label: str
    interpretation: str


METRICS = (
    MetricSpec(
        "epsilon_conservation_max",
        ("epsilon_conservation_max",),
        "maximum stored-extensive conservation error",
        "lower is numerically smaller; stored quantities only",
    ),
    MetricSpec(
        "epsilon_balance",
        ("epsilon_balance", "balance_error"),
        "transport marginal balance error",
        "lower is numerically smaller",
    ),
    MetricSpec(
        "runtime_wall_seconds",
        ("runtime_wall_seconds", "runtime_operator_total_seconds"),
        "wall runtime [s]",
        "lower is faster on the audited hardware",
    ),
    MetricSpec(
        "balance_iterations",
        ("n_balance_iterations", "balance_iterations"),
        "balance iterations",
        "solver-specific work count; compare with solver definitions in mind",
    ),
    MetricSpec(
        "source_enclosed_mass_to_filter_median",
        ("source_enclosed_mass_to_filter_median",),
        "median enclosed-support mass / Mfilter",
        "support overshoot, not output partition mass",
    ),
    MetricSpec(
        "source_enclosed_mass_to_filter_p95",
        ("source_enclosed_mass_to_filter_p95",),
        "p95 enclosed-support mass / Mfilter",
        "support overshoot, not output partition mass",
    ),
    MetricSpec(
        "source_enclosed_mass_to_filter_max",
        ("source_enclosed_mass_to_filter_max",),
        "max enclosed-support mass / Mfilter",
        "support overshoot, not output partition mass",
    ),
    MetricSpec(
        "output_cell_mass_cv",
        ("output_cell_mass_cv",),
        "regular output-cell mass CV",
        OUTPUT_MASS_LIMITATION,
    ),
    MetricSpec(
        "output_cell_mass_gini",
        ("output_cell_mass_gini",),
        "regular output-cell mass Gini",
        OUTPUT_MASS_LIMITATION,
    ),
    MetricSpec(
        "h_eff_median",
        ("h_eff_median", "cells_h_eff_median"),
        "median effective support scale",
        "mass-weighted effective soft-support scale",
    ),
    MetricSpec(
        "h_eff_p95",
        ("h_eff_p95", "cells_h_eff_p95"),
        "p95 effective support scale",
        "mass-weighted effective soft-support scale",
    ),
    MetricSpec(
        "n_eff_median",
        ("n_eff_median", "cells_n_eff_median"),
        "median effective source count",
        "effective soft-transport source count",
    ),
    MetricSpec(
        "n_eff_p95",
        ("n_eff_p95", "cells_n_eff_p95"),
        "p95 effective source count",
        "effective soft-transport source count",
    ),
    MetricSpec(
        "dominant_partition_component_total",
        ("dominant_partition_component_total", "cells_dominant_partition_component_total"),
        "argmax component total",
        PSEUDO_PARTITION_LIMITATION,
    ),
    MetricSpec(
        "dominant_partition_disconnected_count",
        ("dominant_partition_disconnected_count", "cells_dominant_partition_disconnected_count"),
        "argmax disconnected destination count",
        PSEUDO_PARTITION_LIMITATION,
    ),
    MetricSpec(
        "dominant_partition_empty_count",
        ("dominant_partition_empty_count", "cells_dominant_partition_empty_count"),
        "argmax empty destination count",
        PSEUDO_PARTITION_LIMITATION,
    ),
    MetricSpec(
        "dominant_partition_largest_component_fraction_mean",
        (
            "dominant_partition_largest_component_fraction_mean",
            "cells_dominant_partition_largest_component_fraction_mean",
        ),
        "mean largest argmax-component fraction",
        PSEUDO_PARTITION_LIMITATION,
    ),
    MetricSpec(
        "resolved_kinetic_energy_loss_fraction",
        ("resolved_kinetic_energy_loss_fraction",),
        "resolved kinetic-energy loss fraction",
        "resolved kinetic loss under averaging; not a physical total-energy error",
    ),
)


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "sample_uid" not in frame.columns:
        raise ValueError(f"{path} has no sample_uid column.")
    frame["sample_uid"] = frame["sample_uid"].astype(str)
    if frame["sample_uid"].duplicated().any():
        duplicates = frame.loc[frame["sample_uid"].duplicated(False), "sample_uid"].unique()
        raise ValueError(
            f"{path} contains duplicate sample_uid values: {duplicates[:5].tolist()}. "
            "Compare one operator/mass-filter configuration at a time."
        )
    return frame


def _normalise_status(value: Any) -> str:
    value = str(value).strip().lower()
    if value in {"ok", "success", "succeeded", "pass", "passed"}:
        return "ok"
    if value in {"timeout", "timed_out", "timed-out"}:
        return "timeout"
    return "failed"


def _status_information(frame: pd.DataFrame) -> dict[str, Any]:
    raw = frame["status"].fillna("missing") if "status" in frame else pd.Series("missing", index=frame.index)
    normalised = raw.map(_normalise_status)
    details = {str(key): int(value) for key, value in raw.astype(str).value_counts().items()}
    return {
        "total": int(len(frame)),
        "ok": int((normalised == "ok").sum()),
        "timeout": int((normalised == "timeout").sum()),
        "failed": int((normalised == "failed").sum()),
        "raw_status_counts": details,
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0):
        return np.nan
    total = float(values.sum())
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(
        np.sum((2.0 * indices - ordered.size - 1.0) * ordered)
        / (ordered.size * total)
    )


def _aggregate_output_cells(path: str | Path) -> pd.DataFrame:
    cells = pd.read_csv(path, encoding="utf-8-sig")
    if "sample_uid" not in cells:
        raise ValueError(f"{path} has no sample_uid column.")
    rows = []
    for sample_uid, group in cells.groupby(cells["sample_uid"].astype(str), sort=False):
        row: dict[str, Any] = {"sample_uid": sample_uid}
        if "output_cell_mass" in group:
            values = pd.to_numeric(group["output_cell_mass"], errors="coerce").to_numpy()
            finite = values[np.isfinite(values)]
            if finite.size:
                mean = float(np.mean(finite))
                row["output_cell_mass_cv"] = (
                    float(np.std(finite) / abs(mean)) if mean != 0 else np.nan
                )
                row["output_cell_mass_gini"] = _gini(finite)
        for source, prefix in (
            ("effective_support_scale", "cells_h_eff"),
            ("effective_source_count", "cells_n_eff"),
        ):
            if source in group:
                values = pd.to_numeric(group[source], errors="coerce").to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    row[f"{prefix}_median"] = float(np.median(values))
                    row[f"{prefix}_p95"] = float(np.quantile(values, 0.95))
        if "dominant_component_count" in group:
            counts = pd.to_numeric(group["dominant_component_count"], errors="coerce").fillna(0)
            row["cells_dominant_partition_component_total"] = float(counts.sum())
            row["cells_dominant_partition_disconnected_count"] = int((counts > 1).sum())
            if "dominant_assigned_source_count" in group:
                assigned = pd.to_numeric(
                    group["dominant_assigned_source_count"], errors="coerce"
                ).fillna(0)
                row["cells_dominant_partition_empty_count"] = int((assigned <= 0).sum())
            if "dominant_largest_component_fraction" in group:
                fractions = pd.to_numeric(
                    group["dominant_largest_component_fraction"], errors="coerce"
                ).to_numpy()
                fractions = fractions[np.isfinite(fractions)]
                if fractions.size:
                    row["cells_dominant_partition_largest_component_fraction_mean"] = float(
                        np.mean(fractions)
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def _augment_with_cells(summary: pd.DataFrame, cells_path: str | Path | None) -> pd.DataFrame:
    if cells_path is None:
        return summary
    aggregated = _aggregate_output_cells(cells_path)
    merged = summary.merge(aggregated, on="sample_uid", how="left", validate="one_to_one")
    # Prefer audited summary values; use per-cell reconstruction only when a
    # summary column is absent or missing.
    reconstructed_columns: dict[str, Any] = {}
    drop_columns: list[str] = []
    for base in ("output_cell_mass_cv", "output_cell_mass_gini"):
        reconstructed = f"{base}_y" if f"{base}_y" in merged else None
        original = f"{base}_x" if f"{base}_x" in merged else None
        if original and reconstructed:
            reconstructed_columns[base] = pd.to_numeric(
                merged[original], errors="coerce"
            ).combine_first(
                pd.to_numeric(merged[reconstructed], errors="coerce")
            )
            drop_columns.extend([original, reconstructed])
    if reconstructed_columns:
        merged = pd.concat(
            [
                merged.drop(columns=drop_columns),
                pd.DataFrame(reconstructed_columns, index=merged.index),
            ],
            axis=1,
        )
    return merged


def _resolve_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for spec in METRICS:
        if spec.name in frame:
            frame[spec.name] = pd.to_numeric(frame[spec.name], errors="coerce")
            continue
        for alias in spec.aliases:
            if alias in frame:
                frame[spec.name] = pd.to_numeric(frame[alias], errors="coerce")
                break
        else:
            frame[spec.name] = np.nan
    return frame


def _finite_quantiles(values: Iterable[float]) -> dict[str, float | None]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: None for key in ("n", "mean", "q05", "q25", "median", "q75", "q95")}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _boolean(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def _paired_table(v0: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    common = v0.merge(v1, on="sample_uid", how="inner", suffixes=("_v0", "_v1"), validate="one_to_one")
    output = pd.DataFrame({"sample_uid": common["sample_uid"]})
    for identity in ("run", "path", "time", "sfr_target", "selection_order"):
        for side in ("v0", "v1"):
            column = f"{identity}_{side}"
            if column in common:
                output[column] = common[column]
    for side in ("v0", "v1"):
        status_column = f"status_{side}"
        raw = common[status_column] if status_column in common else pd.Series("missing", index=common.index)
        output[status_column] = raw.astype(str)
        output[f"status_normalized_{side}"] = raw.map(_normalise_status)
        for field in ("error_type", "error_message"):
            column = f"{field}_{side}"
            if column in common:
                output[column] = common[column]
    output["status_pair"] = (
        output["status_normalized_v0"] + "__" + output["status_normalized_v1"]
    )
    output["both_success"] = (
        (output["status_normalized_v0"] == "ok")
        & (output["status_normalized_v1"] == "ok")
    )

    success = output["both_success"].to_numpy(dtype=bool)
    for spec in METRICS:
        v0_column = f"{spec.name}_v0"
        v1_column = f"{spec.name}_v1"
        values_v0 = pd.to_numeric(common[v0_column], errors="coerce").to_numpy(dtype=float)
        values_v1 = pd.to_numeric(common[v1_column], errors="coerce").to_numpy(dtype=float)
        valid = success & np.isfinite(values_v0) & np.isfinite(values_v1)
        output[f"v0_{spec.name}"] = np.where(valid, values_v0, np.nan)
        output[f"v1_{spec.name}"] = np.where(valid, values_v1, np.nan)
        output[f"delta_v1_minus_v0_{spec.name}"] = np.where(
            valid, values_v1 - values_v0, np.nan
        )

    runtime_v0 = output["v0_runtime_wall_seconds"].to_numpy(dtype=float)
    runtime_v1 = output["v1_runtime_wall_seconds"].to_numpy(dtype=float)
    valid_runtime = success & np.isfinite(runtime_v0) & np.isfinite(runtime_v1) & (runtime_v0 > 0) & (runtime_v1 > 0)
    output["runtime_speedup_v0_over_v1"] = np.where(
        valid_runtime, runtime_v0 / runtime_v1, np.nan
    )
    for side in ("v0", "v1"):
        column = f"stored_extensive_conservation_pass_{side}"
        if column in common:
            output[f"conservation_pass_{side}"] = [
                _boolean(value) if ok else None
                for value, ok in zip(common[column], success)
            ]
    return output


def _comparison_summary(
    v0: pd.DataFrame,
    v1: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    v0_path: Path,
    v1_path: Path,
) -> dict[str, Any]:
    ids_v0 = set(v0["sample_uid"])
    ids_v1 = set(v1["sample_uid"])
    both_success = paired[paired["both_success"]]
    metrics: dict[str, Any] = {}
    for spec in METRICS:
        values_v0 = both_success[f"v0_{spec.name}"].to_numpy(dtype=float)
        values_v1 = both_success[f"v1_{spec.name}"].to_numpy(dtype=float)
        delta = both_success[f"delta_v1_minus_v0_{spec.name}"].to_numpy(dtype=float)
        finite = np.isfinite(values_v0) & np.isfinite(values_v1) & np.isfinite(delta)
        metrics[spec.name] = {
            "label": spec.label,
            "interpretation": spec.interpretation,
            "v0": _finite_quantiles(values_v0[finite]),
            "v1": _finite_quantiles(values_v1[finite]),
            "delta_v1_minus_v0": _finite_quantiles(delta[finite]),
        }

    status_pairs = {
        str(key): int(value) for key, value in paired["status_pair"].value_counts().items()
    }
    return {
        "schema_version": 1,
        "inputs": {"v0_summary": str(v0_path), "v1_summary": str(v1_path)},
        "limitations": [SFR_LIMITATION, PSEUDO_PARTITION_LIMITATION, OUTPUT_MASS_LIMITATION],
        "coverage": {
            "v0": _status_information(v0),
            "v1": _status_information(v1),
            "common_sample_uid_count": int(len(ids_v0 & ids_v1)),
            "v0_only_count": int(len(ids_v0 - ids_v1)),
            "v1_only_count": int(len(ids_v1 - ids_v0)),
            "v0_only_sample_uids": sorted(ids_v0 - ids_v1),
            "v1_only_sample_uids": sorted(ids_v1 - ids_v0),
            "status_pair_counts_on_common_uids": status_pairs,
            "both_success_count": int(paired["both_success"].sum()),
            "common_but_not_both_success_count": int((~paired["both_success"]).sum()),
        },
        "runtime_speedup_v0_over_v1": _finite_quantiles(
            paired["runtime_speedup_v0_over_v1"].to_numpy(dtype=float)
        ),
        "paired_metrics": metrics,
    }


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6g}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    coverage = summary["coverage"]
    lines = [
        "# Rm-v0 与 Rm-v1 数值审计配对比较",
        "",
        "> " + SFR_LIMITATION,
        "",
        "> " + PSEUDO_PARTITION_LIMITATION,
        "",
        "> " + OUTPUT_MASS_LIMITATION,
        "",
        "## 样本覆盖与运行状态",
        "",
        "| 数据 | 总数 | 成功 | 超时 | 失败 |",
        "|---|---:|---:|---:|---:|",
        f"| v0 | {coverage['v0']['total']} | {coverage['v0']['ok']} | {coverage['v0']['timeout']} | {coverage['v0']['failed']} |",
        f"| v1 | {coverage['v1']['total']} | {coverage['v1']['ok']} | {coverage['v1']['timeout']} | {coverage['v1']['failed']} |",
        "",
        f"共同 sample_uid：{coverage['common_sample_uid_count']}；双方成功并进入数值配对：{coverage['both_success_count']}；共同但至少一方未成功：{coverage['common_but_not_both_success_count']}。",
        "",
        f"仅v0存在：{coverage['v0_only_count']}；仅v1存在：{coverage['v1_only_count']}。这些样本未被静默当作成功配对。",
        "",
        "## 双方成功样本的配对统计",
        "",
        "所有差值定义为 v1 − v0。只有双方状态均为 `ok` 且两边指标均有限时才进入该指标统计。",
        "",
        "| 指标 | N | v0中位数 | v1中位数 | 差值中位数 | 差值q05 | 差值q95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = summary["paired_metrics"][metric.name]
        lines.append(
            f"| {metric.label} | {_fmt(item['delta_v1_minus_v0']['n'])} | "
            f"{_fmt(item['v0']['median'])} | {_fmt(item['v1']['median'])} | "
            f"{_fmt(item['delta_v1_minus_v0']['median'])} | "
            f"{_fmt(item['delta_v1_minus_v0']['q05'])} | "
            f"{_fmt(item['delta_v1_minus_v0']['q95'])} |"
        )
    speedup = summary["runtime_speedup_v0_over_v1"]
    lines.extend(
        [
            "",
            "## 运行时间",
            "",
            "speedup定义为 `v0 wall time / v1 wall time`，大于1表示v1在此次硬件与配置上更快。",
            "",
            f"有效配对 N={_fmt(speedup['n'])}，中位speedup={_fmt(speedup['median'])}，q05–q95={_fmt(speedup['q05'])}–{_fmt(speedup['q95'])}。",
            "",
            "## 解释边界",
            "",
            "- 守恒统计针对代码中明确审计的存储广延量；resolved kinetic loss被单独报告，不等同于物理总能量不守恒。",
            "- balance iterations可能对应不同求解器阶段，必须结合v0/v1求解器定义解释。",
            "- enclosed-support mass相对Mfilter的超调属于重叠支撑诊断，不是等质量分区误差。",
            "- 输出格质量CV/Gini不是等质量目标的偏差。",
            "- argmax连通性不是soft transport本身的物理区域连通性。",
            "- 在实现并评估SFR算子与闭合修正前，本报告不能给出SFR优劣结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_audits(
    v0_summary_path: str | Path,
    v1_summary_path: str | Path,
    output_dir: str | Path,
    *,
    v0_cells_path: str | Path | None = None,
    v1_cells_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two audit runs and write CSV, JSON, and Markdown artifacts."""

    v0_summary_path = Path(v0_summary_path)
    v1_summary_path = Path(v1_summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    v0 = _resolve_metrics(_augment_with_cells(_read_csv(v0_summary_path), v0_cells_path))
    v1 = _resolve_metrics(_augment_with_cells(_read_csv(v1_summary_path), v1_cells_path))
    paired = _paired_table(v0, v1)
    paired.sort_values("sample_uid", inplace=True, ignore_index=True)
    paired.to_csv(
        output_dir / "comparison_per_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = _comparison_summary(
        v0,
        v1,
        paired,
        v0_path=v0_summary_path,
        v1_path=v1_summary_path,
    )
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    _write_markdown(output_dir / "comparison_report.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair Rm-v0/Rm-v1 numerical audit rows by sample_uid. This does not compare SFR "
            "accuracy and argmax connectivity is not a physical partition."
        )
    )
    parser.add_argument("--v0-summary", required=True, type=Path, help="v0 audit_summary.csv")
    parser.add_argument("--v1-summary", required=True, type=Path, help="v1 audit_summary.csv")
    parser.add_argument("--v0-cells", type=Path, default=None, help="optional v0 audit_output_cells.csv")
    parser.add_argument("--v1-cells", type=Path, default=None, help="optional v1 audit_output_cells.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="comparison artifact directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compare_audits(
        args.v0_summary,
        args.v1_summary,
        args.output_dir,
        v0_cells_path=args.v0_cells,
        v1_cells_path=args.v1_cells,
    )
    coverage = summary["coverage"]
    print(f"common_sample_uids={coverage['common_sample_uid_count']}")
    print(f"both_success={coverage['both_success_count']}")
    print(f"v0_only={coverage['v0_only_count']}, v1_only={coverage['v1_only_count']}")
    print(f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
