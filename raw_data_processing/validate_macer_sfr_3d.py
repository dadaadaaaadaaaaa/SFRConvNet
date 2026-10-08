"""扫描全序列、按 SFR 分位数与演化阶段选样，验证未降采样的三维算子。"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raw_data_processing.athdf_reader import read_athdf
from raw_data_processing.macer_sfr_operator_3d import (
    SFR3DParameters, calculate_macer_sfr_3d, validate_gravity_geometry,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_parameters(path):
    """读取实际运行配置中的效率与阈值，不使用二维拟合参数。"""
    fields = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return SFR3DParameters(gamma=float(fields["gamma"]), efficiency=float(fields["etaSF"]),
                           temperature_ceiling=float(fields["temp_sf"]),
                           number_density_floor=float(fields["nrho_sf"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gravity", type=Path, default=Path("Fid_yuan18.out1.00000.athdf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gravity = read_athdf(args.gravity, fields=("grav",), dtype=np.float64)
    params = input_parameters(args.input)
    files = sorted(args.raw_dir.glob("*.athdf"))
    if not files:
        raise ValueError("没有找到 ATHDF 文件。")
    scan = []
    for i, path in enumerate(files):
        snap = read_athdf(path, fields=("rhodotsf",), dtype=np.float64)
        validate_gravity_geometry(snap, gravity)
        truth = snap.fields["rhodotsf"]
        if np.any(truth < 0):
            raise ValueError(f"负 rhodotsf: {path}")
        total = float(np.sum(truth * snap.geometry.cell_volume, dtype=np.float64))
        scan.append(dict(file=path.name, time=snap.metadata.time, total_code=total,
                         total_msun_per_yr=total*params.sfr_to_msun_per_yr,
                         sf_cells=int(np.count_nonzero(truth))))
        if i % 25 == 0 or i == len(files)-1:
            print(f"扫描 {i+1}/{len(files)}，SFR={total*.025:.6g} Msun/yr", flush=True)
    (args.output / "scan.json").write_text(json.dumps(scan, indent=2), encoding="utf-8")
    totals = np.array([row["total_code"] for row in scan])
    nonzero = np.flatnonzero(totals > 0)
    if not len(nonzero):
        raise ValueError("全部监督值为零，无法验证算子。")
    selected = {}

    def select(index, reason):
        selected.setdefault(int(index), []).append(reason)

    for q in (0, .1, .25, .5, .75, .9, 1):
        index = nonzero[np.argmin(abs(totals[nonzero]-np.quantile(totals[nonzero], q)))]
        select(index, f"positive_SFR_quantile_{q}")
    for index in np.linspace(0, len(files)-1, 7).round().astype(int):
        select(index, "time_coverage")
    # 正 SFR 区间内最大上升和下降，避免初始化零值支配变化率选样。
    valid = np.flatnonzero((totals[:-1] > 0) & (totals[1:] > 0))
    if len(valid):
        slope = (totals[1:]-totals[:-1]) / np.diff([r["time"] for r in scan])
        for index, label in ((valid[np.argmax(slope[valid])], "fastest_rise"),
                             (valid[np.argmin(slope[valid])], "fastest_fall")):
            select(index, label+"_before")
            select(index+1, label+"_after")
    variants = {
        "rotation_2pi_total_n": params,
        "rotation_1_total_n": replace(params, rotation_factor=1.),
        "rotation_2pi_hydrogen_n": replace(params, number_density_mode="hydrogen_nuclei"),
        "rotation_2pi_rho_over_mH": replace(params, number_density_mode="rho_over_mH"),
        "rotation_2pi_fixed_temperature": replace(params, temperature_mode="fixed_71.41"),
    }
    rows = []
    for index in sorted(selected):
        path = files[index]
        snap = read_athdf(path, fields=("rho", "press", "metallicity", "cool_rate", "rhodotsf"), dtype=np.float64)
        validate_gravity_geometry(snap, gravity)
        f = snap.fields
        truth = f["rhodotsf"]
        volume = snap.geometry.cell_volume
        actual = scan[index]["total_code"]
        row = {**scan[index], "selection": selected[index], "source_sha256": sha256(path),
               "cooling_min": float(f["cool_rate"].min()),
               "cooling_nonpositive_cells": int(np.count_nonzero(f["cool_rate"] <= 0)),
               "variants": {}}
        for name, configuration in variants.items():
            result = calculate_macer_sfr_3d(density=f["rho"], pressure=f["press"],
                metallicity=f["metallicity"], cooling_rate=f["cool_rate"], cell_volume=volume,
                radius=snap.geometry.centers[0][:, None, None],
                radial_acceleration=gravity.fields["grav"], parameters=configuration)
            predicted = result.sfr_density
            difference = predicted-truth
            positive = truth > 0
            false_positive = (predicted > 0) & ~positive
            false_negative = (predicted == 0) & positive
            metrics = dict(predicted_msun_per_yr=result.total_sfr*params.sfr_to_msun_per_yr,
                signed_relative_error=None if actual == 0 else (result.total_sfr-actual)/actual,
                volume_L1_relative=None if actual == 0 else float(np.sum(abs(difference)*volume)/actual),
                absolute_error_msun_per_yr=(result.total_sfr-actual)*params.sfr_to_msun_per_yr,
                false_positive_cells=int(false_positive.sum()), false_negative_cells=int(false_negative.sum()),
                parameters=result.parameters)
            if actual > 0:
                ratio = predicted[positive]/truth[positive]
                metrics.update(rate_ratio_quantiles=dict(zip(("min", "p01", "p10", "p50", "p90", "p99", "max"),
                    map(float, np.quantile(ratio, [0,.01,.1,.5,.9,.99,1])))),
                    false_negative_sfr_fraction=float(np.sum((truth*volume)[false_negative])/actual),
                    false_positive_sfr_fraction=float(np.sum(result.cell_sfr[false_positive])/actual),
                    true_sfr_above_temperature_threshold_fraction=float(np.sum((truth*volume)[positive & (result.temperature >= configuration.temperature_ceiling)])/actual),
                    true_sfr_below_density_threshold_fraction=float(np.sum((truth*volume)[positive & (result.number_density <= configuration.number_density_floor)])/actual),
                    limiter_true_sfr_fractions={str(k): float(np.sum((truth*volume)[result.limiter_code==k])/actual) for k in range(4)})
            row["variants"][name] = metrics
        rows.append(row)
        best = row["variants"]["rotation_2pi_total_n"]
        print(f"验证 {path.name}: true={actual*.025:.8g}, rel={best['signed_relative_error']}, L1={best['volume_L1_relative']}", flush=True)
        # 每次验证后保存，便于长时间读取被中断后检查已完成结果。
        payload = dict(raw_dir=str(args.raw_dir.resolve()), input_path=str(args.input.resolve()),
                       input_sha256=sha256(args.input), gravity_path=str(args.gravity.resolve()),
                       gravity_sha256=sha256(args.gravity), scan_count=len(scan),
                       selected_count=len(selected), completed_count=len(rows), snapshots=rows)
        (args.output / "validation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# MACER3D 原始分辨率 SFR 算子验证", "",
        f"扫描 {len(scan)} 个 snapshot；验证 {len(rows)} 个。原始网格 {snap.shape}。",
        "", "参考值是原文件 rhodotsf 的精确球坐标体积积分，不是另一个独立 hst 总 SFR。",
        "", "| snapshot | time (code) | True (Msun/yr) | 2π 相对误差 (%) | 不带2π 相对误差 (%) | 2π 场 L1 (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        a = row["variants"]["rotation_2pi_total_n"]
        b = row["variants"]["rotation_1_total_n"]
        def pct(x): return "未定义（True=0）" if x is None else f"{100*x:.6g}"
        lines.append(f"| {row['file']} | {row['time']:.6f} | {row['total_msun_per_yr']:.8g} | {pct(a['signed_relative_error'])} | {pct(b['signed_relative_error'])} | {pct(a['volume_L1_relative'])} |")
    lines += ["", "## 口径和限制", "",
        "- η、温度和数密度阈值读取 input.txt；γ 同样来自 input.txt。",
        "- 作者确认静态 grav 跨时间复用；每个文件均核对坐标中心与 faces。",
        "- t_cool=p/((γ−1)cool_rate)，正冷却率采用代码单位；零冷却率记为无限冷却时间。",
        "- n=rho/(mu*mH_code*L_unit_cm^3)，温度使用作者的 mu(Z)；metallicity 为绝对质量分数的解释由数据支持。",
        "- 固定 X=0.7 的公式在 Z>0.3 区域不对应合理成分；仍按作者公式计算，并记录这些 cell 的数量。",
        "- 默认连续时间率；未假定二维 star.F 有限步长函数也适用于三维。",
        "- 积分吻合不等于逐格吻合。局部旋转残差及 hst 中心质量数值对照可运行 diagnose_sfr3d_rotation.py，详见 3DSFRConvnet.md 第 18 节。",
        "- 动力学主导的 SFR 不能独立检验冷却时间公式；需要另有冷却主导样本或模拟源码。",
        "- 初始化全零输出单独列示，零参考值不计算相对误差。",
        "- 变体用于源公式辨识，不构成独立预测或机器学习泛化验证。",
        "- 完整参数、源文件 SHA256、局部阈值误差和所有变体见 validation.json；全序列扫描见 scan.json。"]
    (args.output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
