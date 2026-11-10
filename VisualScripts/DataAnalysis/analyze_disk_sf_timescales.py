"""诊断五组盘星系 HR 成星处方的时标分支，只读已有物理产品。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from raw_data_processing.macer_sfr_operator import calculate_macer_sfr
from raw_data_processing.metadata_schema import validate_complete_metadata

RUNS = ["fiducial", "fiducial_4", "fiducial_7", "model_0_75", "model_1_1"]
LABELS = dict(zip(RUNS, ["Fiducial", "Fiducial4", "Fiducial7", "Model0.75", "Model1.1"]))
BRANCHES = {1: "cooling", 2: "freefall", 3: "rotation"}
MODES = ["data_compatible", "paper_physical"]
FACTOR = 0.025


def weighted_quantiles(values, weights, probabilities=(.16, .5, .84)):
    """只对正权重求加权分位数；空成星区域返回 NaN。"""
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    if values.shape != weights.shape or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("权重必须匹配且为非负有限值。")
    keep = weights > 0
    if not keep.any():
        return np.full(len(probabilities), np.nan)
    values, weights = values[keep], weights[keep]
    if np.isnan(values).any() or np.any(values < 0):
        raise ValueError("正权重对应的时标不能为 NaN 或负数。")
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    indices = np.searchsorted(cumulative, np.asarray(probabilities)*cumulative[-1])
    return values[order][indices.clip(0, len(order)-1)]


def time_weights(times, nominal_dt):
    """仅对不跨缺帧的相邻输出做梯形积分，缺口保持未覆盖。"""
    times = np.asarray(times, dtype=float)
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("同一运行的时间必须严格递增。")
    valid = intervals <= 1.5*nominal_dt
    weights = np.zeros_like(times)
    weights[:-1] += np.where(valid, intervals/2, 0)
    weights[1:] += np.where(valid, intervals/2, 0)
    return weights


def block_view(field, shape):
    """将 HR 单元按现有 R_h 的非重叠索引块重排。"""
    a, b = field.shape
    c, d = shape
    if a % c or b % d:
        raise ValueError("HR shape 不能被真实标签块整除。")
    return field.reshape(c, a//c, d, b//d).transpose(0, 2, 1, 3)


def branch_bounds(codes, true_coarse, branch):
    """无需假设块内标签分布，给出该分支真实 SFR 的严格上下界。"""
    selected = block_view(codes == branch, true_coarse.shape)
    return (float(true_coarse[selected.all(axis=(2, 3))].sum()),
            float(true_coarse[selected.any(axis=(2, 3))].sum()))


def read_snapshot(path, coarse_path, nickname, nominal_dt):
    """复算时标并核验保存的 HR baseline、UID、标签与权威体积。"""
    with h5py.File(path, "r") as f, h5py.File(coarse_path, "r") as c:
        meta = json.loads(f["metadata/contract_json"][()])
        cm = json.loads(c["metadata/contract_json"][()])
        validate_complete_metadata(meta)
        if meta["derivation"]["operator_id"] != "identity_fine":
            raise ValueError("时标主分析必须使用原生 HR 气体场。")
        uid = meta["identity"]["parent_hr_uid"]
        if uid != cm["identity"]["parent_hr_uid"]:
            raise ValueError("HR 与真实块标签的来源 UID 不一致。")
        index = int(meta["source"]["raw_snapshot_index"])
        time = float(f.attrs["time"])
        expected = index*nominal_dt
        mapping = meta["source"].get("time_mapping", {})
        if mapping.get("mode") == "segmented_restart_absolute" and index > 0:
            expected = .5625 + (index + (index-1)//999)*nominal_dt
        if not np.isclose(time, expected, rtol=0, atol=1e-10):
            raise ValueError("fine 时间与已核验的原始编号映射不一致。")
        if not np.isclose(time, meta["source"]["time"], atol=1e-10, rtol=0):
            raise ValueError("文件时间与 metadata 不一致。")
        if not np.isclose(time, c.attrs["time"], atol=1e-10, rtol=0):
            raise ValueError("fine 与真实块标签的时间不一致。")
        fields = {key: np.asarray(f["snapshot/"+name], dtype=float) for key, name in [
            ("density", "density"), ("internal_energy_density", "gas_energy"),
            ("cooling_time", "cooling_time"), ("radial_acceleration", "radial_acceleration")]}
        volume = np.asarray(f["geometry/cell_volume"])
        radius = np.asarray(f["coordinates/axis_0"])
        reference = np.asarray(f["sfr_supervision/data_compatible/cell_sfr_baseline"])
        truth = np.asarray(c["sfr_supervision/canonical/Y_alpha_cell"])
        y = float(f["sfr_supervision/canonical/Y_alpha"][()])
        b = float(f["sfr_supervision/canonical/B_alpha"][()])
        closure = float(f["sfr_supervision/canonical/C_alpha_true"][()])
        if not np.isfinite(truth).all() or np.any(truth < 0):
            raise ValueError("真实块标签含负数或非有限值。")
        np.testing.assert_allclose(truth.sum(), y, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y, b+closure, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(block_view(volume, truth.shape).sum(axis=(2, 3)),
                                   c["geometry/cell_volume"], rtol=1e-12, atol=1e-14)
        context = json.loads(f["sfr_supervision/data_compatible/context_json"][()])
        fields.update(cell_volume=volume, radius=radius, time_step=context["time_step"],
                      black_hole_mass=context["black_hole_mass"])
        rows = []
        for mode in MODES:
            result = calculate_macer_sfr(**fields, mode=mode)
            stored = np.asarray(f[f"sfr_supervision/{mode}/cell_sfr_baseline"])
            np.testing.assert_allclose(result.cell_sfr, stored, rtol=1e-11, atol=1e-13)
            reconstructed_blocks = block_view(reference, truth.shape).sum(axis=(2, 3))
            row = dict(run=nickname, mode=mode, file=path.name, fine_dir=path.parent.name,
                       raw_snapshot_index=index, parent_hr_uid=uid, time_gyr=time,
                       production_time_gyr=time-.5625, true_sfr=y*FACTOR,
                       reference_sfr=reference.sum()*FACTOR, mode_sfr=result.total_sfr*FACTOR,
                       block_l1_error=np.abs(reconstructed_blocks-truth).sum()*FACTOR,
                       true_zero=bool(y == 0), eligible_cells=int(result.eligible_mask.sum()),
                       source_sha256=meta["source"]["source_file_sha256"],
                       source_series=meta["source"]["source_series_id"])
            mass = fields["density"]*volume
            for code, name in BRANCHES.items():
                mask = result.limiter_code == code
                row[f"ref_{name}"] = reference[mask].sum()*FACTOR
                row[f"own_{name}"] = result.cell_sfr[mask].sum()*FACTOR
                row[f"mass_{name}"] = mass[mask].sum()
                row[f"cells_{name}"] = int(mask.sum())
                # 0 类也保留在块混合判断中，不假设更新前后阈值完全相同。
                lo, hi = branch_bounds(result.limiter_code, truth, code)
                row[f"true_lower_{name}"] = lo*FACTOR
                row[f"true_upper_{name}"] = hi*FACTOR
            for name, values in [("cool_myr", result.tau_cool*1000),
                                 ("ff_myr", result.tau_free_fall*1000),
                                 ("rot_myr", result.tau_rotation*1000),
                                 ("cool_over_ff", result.tau_cool/result.tau_free_fall),
                                 ("rot_over_ff", result.tau_rotation/result.tau_free_fall)]:
                for q, value in zip([16, 50, 84], weighted_quantiles(values, reference)):
                    row[f"{name}_q{q}"] = value
            rows.append(row)
        return rows


def summarize(frame, nominal):
    """按每个运行独立积分，避免不同 snapshot cadence 改变比较权重。"""
    rows = []
    for (run, mode), data in frame.groupby(["run", "mode"], sort=False):
        data = data.sort_values("time_gyr")
        w = time_weights(data.time_gyr, nominal[run])
        integral = lambda key: float(np.dot(data[key], w))
        denom, own, truth = integral("reference_sfr"), integral("mode_sfr"), integral("true_sfr")
        if denom <= 0 or truth <= 0:
            raise ValueError("没有可积分的正 SFR。")
        row = dict(run=run, mode=mode, snapshots=len(data), zero_snapshots=int(data.true_zero.sum()),
                   first_time_gyr=data.time_gyr.min(), last_time_gyr=data.time_gyr.max(),
                   covered_gyr=w.sum(), coverage_fraction=w.sum()/np.ptp(data.time_gyr),
                   true_formed_mass_msun=truth*1e9,
                   relative_integrated_sfr_error=(denom-truth)/truth,
                   relative_integrated_block_l1=integral("block_l1_error")/truth)
        positive = data.true_sfr > 0
        relative = np.abs(data.loc[positive, "reference_sfr"]/data.loc[positive, "true_sfr"]-1)
        row.update(sfr_error_median=relative.median(), sfr_error_p99=relative.quantile(.99),
                   sfr_error_max=relative.max())
        for name in BRANCHES.values():
            row[f"ref_fraction_{name}"] = integral(f"ref_{name}")/denom
            row[f"own_fraction_{name}"] = integral(f"own_{name}")/own
            row[f"true_lower_{name}"] = integral(f"true_lower_{name}")/truth
            row[f"true_upper_{name}"] = integral(f"true_upper_{name}")/truth
            row[f"mass_fraction_{name}"] = integral(f"mass_{name}")/sum(integral(f"mass_{n}") for n in BRANCHES.values())
        row["rotation_dominated_snapshots"] = int((data.ref_rotation > .5*data.reference_sfr).sum())
        row["cooling_dominated_snapshots"] = int((data.ref_cooling > .5*data.reference_sfr).sum())
        row["rotation_dominated_true_mass_fraction"] = float(np.dot(
            data.true_sfr*(data.ref_rotation > .5*data.reference_sfr), w))/truth
        row["max_rotation_fraction"] = (data.ref_rotation/data.reference_sfr.replace(0, np.nan)).max()
        for key in ["ff_myr", "cool_myr", "rot_myr", "cool_over_ff", "rot_over_ff"]:
            row[f"frame_median_{key}"] = data[f"{key}_q50"].median()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(frame, summary, output):
    colors = ["#d98a2b", "#2978a0", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mode in zip(axes, MODES):
        table = summary[summary["mode"] == mode].set_index("run").loc[RUNS]
        bottom = np.zeros(5)
        for name, color in zip(BRANCHES.values(), colors):
            values = table[f"ref_fraction_{name}"].to_numpy()*100
            ax.bar(range(5), values, bottom=bottom, color=color, label=name)
            bottom += values
        for k, run in enumerate(RUNS):
            ff = table.loc[run, "ref_fraction_freefall"]*100
            rot = table.loc[run, "ref_fraction_rotation"]*100
            ax.text(k, 75, f"FF {ff:.4f}%\nrot {rot:.4f}%", ha="center", fontsize=8, color="white")
        ax.set_xticks(range(5), [LABELS[r] for r in RUNS], rotation=20)
        ax.set_title("BH-mass rotation proxy" if mode == MODES[0] else "Radial-acceleration rotation proxy")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Fraction of reconstructed HR SFR (%)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Local SF branch; same HR SFR weights in both panels")
    fig.tight_layout()
    fig.savefig(output/"timescale_fractions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(5, 3, figsize=(14, 13), sharex=True)
    for i, run in enumerate(RUNS):
        ref = frame[(frame.run == run) & (frame["mode"] == MODES[0])].sort_values("time_gyr")
        axes[i, 0].scatter(ref.production_time_gyr, ref.true_sfr, s=1, color="#333333")
        axes[i, 0].set_yscale("symlog", linthresh=1e-5)
        axes[i, 0].set_ylabel(LABELS[run])
        for j, mode in enumerate(MODES):
            data = frame[(frame.run == run) & (frame["mode"] == mode)].sort_values("time_gyr")
            # 用散点保留真实缺口，不通过连线插值缺失帧。
            for name, color in zip(BRANCHES.values(), colors):
                fraction = data[f"ref_{name}"]/data.reference_sfr.replace(0, np.nan)
                axes[i, j+1].scatter(data.production_time_gyr, fraction*100, s=1, color=color, label=name)
            axes[i, j+1].set_ylim(-1, 101)
        for ax in axes[i]:
            ax.grid(alpha=.2)
    axes[0, 0].set_title("True total SFR (Msun/yr)")
    axes[0, 1].set_title("BH proxy: HR SFR share (%)")
    axes[0, 2].set_title("Acceleration proxy: same weights (%)")
    axes[0, 2].legend(markerscale=4, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Production time (Gyr; absolute time - 0.5625)")
    fig.tight_layout()
    fig.savefig(output/"timescale_history.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(11, 12), sharex=True)
    for i, run in enumerate(RUNS):
        ref = frame[(frame.run == run) & (frame["mode"] == MODES[0])]
        axes[i, 0].scatter(ref.production_time_gyr, ref.cool_over_ff_q50, s=1, color="#d98a2b")
        for mode, color in zip(MODES, ["#9467bd", "#2978a0"]):
            data = frame[(frame.run == run) & (frame["mode"] == mode)]
            axes[i, 1].scatter(data.production_time_gyr, data.rot_over_ff_q50, s=1,
                               color=color, label="BH proxy" if mode == MODES[0] else "Acceleration proxy")
        axes[i, 0].set_ylabel(LABELS[run])
        for ax in axes[i]:
            ax.set_yscale("log")
            ax.axhline(1, color="#555555", ls="--", lw=.7)
            ax.grid(alpha=.2)
    axes[0, 0].set_title("Cooling / freefall: SFR-weighted cell median")
    axes[0, 1].set_title("Rotation / freefall: same HR SFR weights")
    axes[0, 1].legend(markerscale=4, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Production time (Gyr)")
    fig.tight_layout()
    fig.savefig(output/"timescale_ratios.png", dpi=160)
    plt.close(fig)


def export_results(frame, nominal, sources, output):
    """从逐帧统计导出结果；可独立重绘，不再次读取大型 fine 产品。"""
    if frame[frame["mode"] == MODES[0]].parent_hr_uid.duplicated().any():
        raise ValueError("HR UID 重复，不能重复计数。")
    summary = summarize(frame, nominal)
    summary.to_csv(output/"summary.csv", index=False)
    phases = []
    for start, stop in [(0, 4), (4, 8), (8, 12)]:
        subset = frame[(frame.production_time_gyr >= start) & (frame.production_time_gyr < stop)]
        table = summarize(subset, nominal)
        table["production_phase_gyr"] = f"{start}-{stop}"
        phases.append(table)
    pd.concat(phases, ignore_index=True).to_csv(output/"phase_summary.csv", index=False)
    events = frame.assign(rotation_fraction=frame.ref_rotation/frame.reference_sfr.replace(0, np.nan))
    events = events.sort_values("rotation_fraction", ascending=False).groupby(["run", "mode"], sort=False).head(10)
    events.to_csv(output/"rotation_examples.csv", index=False)
    plot_results(frame, summary, output)
    method = dict(sources=sources, modes=MODES, branches=BRANCHES,
                  weighting="saved data_compatible HR cell_sfr_baseline; NOT true HR cell labels",
                  integration="trapezoidal; intervals > 1.5 nominal cadence omitted",
                  true_bounds="true coarse labels; all/any HR limiter membership within each block",
                  scope="available post-initialization fine products; 2D Zou five runs only",
                  nominal_dt_gyr=nominal, true_sfr_to_msun_per_yr=FACTOR)
    (output/"method.json").write_text(json.dumps(method, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-snapshots", type=int, help="每组等间隔取样，仅用于独立 smoke 目录")
    parser.add_argument("--reuse-snapshots", action="store_true", help="复用指定输出目录的完整逐帧统计，更新摘要和图")
    args = parser.parse_args()
    if args.max_snapshots is not None and args.max_snapshots < 2:
        parser.error("--max-snapshots 至少为 2")
    if args.reuse_snapshots:
        if args.max_snapshots:
            parser.error("不能将 smoke 和全量统计复用混用")
        method = json.loads((args.output_dir/"method.json").read_text(encoding="utf-8"))
        nominal = method["nominal_dt_gyr"]
        frame = pd.concat([pd.read_parquet(args.output_dir/f"{name}_snapshots.parquet") for name in RUNS], ignore_index=True)
        for source in method["sources"]:
            if len(frame[(frame.run == source["run"]) & (frame["mode"] == MODES[0])]) != source["available"]:
                raise ValueError("逐帧统计不完整，拒绝作为全量产品复用。")
        export_results(frame, nominal, method["sources"], args.output_dir)
        return
    config = json.loads((ROOT/".config/RawDataConfig.json").read_text(encoding="utf-8"))["disk_galaxy"]
    nominal = {name: config[name]["snapshot_dt"] for name in RUNS}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames, sources = [], []
    for name in RUNS:
        directory = ROOT/"data"/f"disk_galaxy_{name}"
        fine = directory/("fine_sfr_v6" if name.startswith("model") else "fine_sfr_v7")
        paths = sorted(fine.glob("*.h5"))
        if not paths:
            raise FileNotFoundError(fine)
        sources.append(dict(run=name, fine_directory=str(fine.relative_to(ROOT)), available=len(paths)))
        if args.max_snapshots:
            paths = [paths[i] for i in np.unique(np.linspace(0, len(paths)-1, args.max_snapshots, dtype=int))]
        rows = []
        for i, path in enumerate(paths, 1):
            rows.extend(read_snapshot(path, directory/"coarse_rh_12x20_sfr_v7_cell_v1"/path.name,
                                      name, nominal[name]))
            if i % 500 == 0:
                print(f"{name}: {i}/{len(paths)}", flush=True)
        table = pd.DataFrame(rows)
        table.to_parquet(args.output_dir/f"{name}_snapshots.parquet", index=False)
        frames.append(table)
        print(f"完成 {name}: {len(paths)} 帧", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    # smoke 只验证 I/O 与数值，不对稀疏样本伪造时间积分。
    if args.max_snapshots:
        print("smoke 完成；不生成时间积分结论。", flush=True)
        return
    export_results(frame, nominal, sources, args.output_dir)


if __name__ == "__main__":
    main()
