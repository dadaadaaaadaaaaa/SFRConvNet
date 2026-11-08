"""比较已完成的 Fiducial Exp2：训练 SFR 范围与 LR 输入近邻覆盖分别分层。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from VisualScripts.ResultAnalysis.analyze_sfr_unet import file_hash, load_predictions, regression
from VisualScripts.DataAnalysis.check_disk_lr_generalization import anchors, global_distance, feature_sets

PREFIX = "Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_"
HELD = "disk_galaxy_fiducial"
CNN = {
    "CNN43": "Exp2_Rh12x20_Closure_v7_LOO_Fiducial_Trial43_seed42_20261025_224050",
    "CNN7": "Exp2_Rh12x20_Closure_v7_LOO_Fiducial_Trial7_seed42_20261025_180348",
}
UNET = {
    "UNet_original42": "Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_SFRUNet_seed42_20261104_131515_e66884f5",
    "UNet_ReLU42_initial": "Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed42_20261106_155134_361de833",
    "UNet_ReLU42": "Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed42_20261107_040054_2f216a19",
    "UNet_ReLU43": "Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed43_20261107_041950_b08b68f2",
    "UNet_ReLU44": "Exp2_LOO_Fiducial_SFRUNet_ReLU_Candidate1_seed44_20261107_043829_12d55372",
}


def sfr_partition(y, training_y, quantiles=(0., 1.)):
    """端点属于范围内；阈值只由去重训练标签确定，不用留出标签调参。"""
    low, high = np.quantile(training_y, quantiles)
    return np.where(y < low, "low", np.where(y > high, "high", "inside")), float(low), float(high)


def validate_alignment(frame, reference):
    """按 UID 重排，拒绝缺帧、重复、parent 不同或时间/标签不一致。"""
    if frame.uid.duplicated().any() or set(frame.uid) != set(reference.uid):
        raise ValueError("预测 UID 与 out cache 不一致。")
    frame = frame.set_index("uid").loc[reference.uid].reset_index()
    if not np.array_equal(frame.parent, reference.parent):
        raise ValueError("parent_hr_uid 不一致。")
    for col, atol, rtol in [("time", 1e-6, 0), ("Y", 1e-7, 2e-7), ("B", 1e-7, 2e-7)]:
        if not np.allclose(frame[col], reference[col], atol=atol, rtol=rtol):
            raise ValueError(f"{col} 与 out cache 不一致。")
    return frame


def block_delta_ci(time, truth, candidate, reference, mask, block_width, repeats, seed=42):
    """配对固定时间块 bootstrap；整块抽样，保留误差相关性及两个模型的配对。"""
    blocks = np.floor((time-time.min())/block_width).astype(int)
    unique = np.unique(blocks)
    count = np.array([np.sum(mask & (blocks == b)) for b in unique])
    delta = np.abs(candidate-truth)-np.abs(reference-truth)
    total = np.array([delta[mask & (blocks == b)].sum() for b in unique])
    if np.count_nonzero(count) < 5:
        return np.nan, np.nan, int(np.count_nonzero(count))
    draw = np.random.default_rng(seed).integers(len(unique), size=(repeats, len(unique)))
    denominator = count[draw].sum(1)
    values = total[draw].sum(1)[denominator > 0]/denominator[denominator > 0]
    return *np.quantile(values, [.025, .975]), int(np.count_nonzero(count))


def load_cache(split):
    path = ROOT/".cache"/f"{PREFIX}{split}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    t = payload["tensors"]
    frame = pd.DataFrame({"uid": payload["sample_identity"]["sample_uid"],
        "parent": payload["sample_identity"]["parent_hr_uid"], "group": np.asarray(t["groups"]),
        "time": np.asarray(t["time"]).reshape(-1),
        "Y": np.asarray(t["target_sfr_physical"]).reshape(-1)*.025,
        "B": np.asarray(t["macer_sfr_baseline"]).reshape(-1)*.025})
    if not np.allclose(np.asarray(t["target_sfr_physical"]),
                       np.asarray(t["macer_sfr_baseline"])+np.asarray(t["closure_true_physical"]),
                       rtol=1e-10, atol=1e-10):
        raise ValueError("cache 不满足 Y=B+C_true。")
    keep = ~frame.uid.duplicated()
    # 过采样只改变权重；重复身份的标签必须完全相同。
    if frame.groupby("uid")[["Y", "B", "time"]].nunique().to_numpy().max() != 1:
        raise ValueError("重复 UID 数据不一致。")
    return frame[keep].reset_index(drop=True), np.asarray(t["x"])[keep].copy(), np.asarray(t["coord"])[0].copy()


def run(output, repeats=2000):
    if output.exists():
        raise FileExistsError("输出目录已存在，拒绝覆盖。")
    torch.set_num_threads(4)
    hashes = {}
    def record(path):
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
        return hashes[str(path.relative_to(ROOT))]
    for path in [Path(__file__), ROOT/"VisualScripts/DataAnalysis/check_disk_lr_generalization.py",
                 ROOT/"VisualScripts/ResultAnalysis/analyze_sfr_unet.py"]:
        record(path)
    train, tx, coord = load_cache("train")
    out, ox, _ = load_cache("out")
    order = np.argsort(out.time.to_numpy())
    out, ox = out.iloc[order].reset_index(drop=True), ox[order]
    if set(out.group) != {HELD} or HELD in set(train.group) or out.time.duplicated().any():
        raise ValueError("不是唯一留出 Fiducial 的有效轨迹。")
    cache_hash = {s: record(ROOT/".cache"/f"{PREFIX}{s}.pt") for s in ["train", "out"]}
    meta_path = ROOT/".cache"/f"{PREFIX}metadata.json"
    norm_path = ROOT/".cache"/f"{PREFIX}norm.npz"
    meta_hash, norm_hash = record(meta_path), record(norm_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    predictions = {"Baseline": out.B.to_numpy()}
    contracts = {}
    for label, folder in CNN.items():
        root = ROOT/"Results"/folder
        hpath = root/"hyperparameters.json"
        hyper = json.loads(hpath.read_text(encoding="utf-8"))
        if (hyper["data_exp"] != PREFIX or hyper["objective_mode"] != "signed_closure"
                or hyper["data_split_metadata"] != metadata):
            raise ValueError("CNN 数据来源/目标契约不一致。")
        record(hpath)
        path = root/"held_out_test_predictions.csv"
        f = pd.read_csv(path)
        if set(f["type"]) != {HELD}:
            raise ValueError("CNN 预测不属于 Fiducial。")
        f = f.rename(columns={"sample_uid": "uid", "parent_hr_uid": "parent", "sfr_true": "Y",
                              "sfr_macer_baseline": "B", "sfr_pred": "prediction"})
        for col in ["Y", "B", "prediction"]:
            f[col] *= .025
        f = validate_alignment(f, out)
        # 旧 CSV 的 baseline 列仅保留 float32 精度，不能要求 float64 的逐位相等。
        if not np.allclose(f.prediction, f.B+.025*f.closure_pred_physical, rtol=2e-7, atol=1e-7):
            raise ValueError("CNN 不满足 B+C_hat。")
        predictions[label] = f.prediction.to_numpy()
        record(path)
        contracts[label] = {k: hyper.get(k) for k in ["seed", "loss", "base_ch", "batch_size", "lr", "pos_num_bands", "pos_max_freq"]}
    for label, folder in UNET.items():
        root = ROOT/"Results"/folder
        summary = json.loads((root/"summary.json").read_text(encoding="utf-8"))
        contract = json.loads((root/"contract.json").read_text(encoding="utf-8"))
        manifest = json.loads((root/"split_manifest.json").read_text(encoding="utf-8"))
        config = json.loads((root/"config.json").read_text(encoding="utf-8"))
        if (summary["status"] != "complete" or summary["hpo"] is not False
                or contract["cache_metadata"] != metadata or config["data_exp"] != PREFIX):
            raise ValueError("UNet 非正式完成训练或数据来源不匹配。")
        data = contract["data"]
        if (data["train_val"]["train"]["scalar_sha256"] != cache_hash["train"]
                or contract["evaluation_identities"]["out"]["scalar_sha256"] != cache_hash["out"]
                or data["metadata_sha256"] != meta_hash or data["norm_sha256"] != norm_hash):
            raise ValueError("UNet cache/归一化摘要不一致。")
        for key in ["sample_uid", "parent_hr_uid"]:
            for i, left in enumerate(manifest):
                for right in list(manifest)[i+1:]:
                    if set(manifest[left][key]) & set(manifest[right][key]):
                        raise ValueError("split 身份交叉。")
            actual = train.uid if key == "sample_uid" else train.parent
            if set(manifest["train"][key]) != set(actual):
                raise ValueError("实际训练身份与 manifest 不同。")
        path = root/"out_predictions.pt"
        d = load_predictions(path, identity=manifest["out"], expected_groups={HELD}, expected_loss=contract["loss"])
        f = pd.DataFrame({"uid": d["sample_uid"], "parent": d["parent_hr_uid"], "time": d["time"],
            "Y": d["target_total"]*.025, "B": d["baseline_total"]*.025, "prediction": d["prediction_total"]*.025})
        predictions[label] = validate_alignment(f, out).prediction.to_numpy()
        contracts[label] = {k: config.get(k) for k in ["seed", "epochs", "output_constraint", "candidate_weight"]}
        contracts[label].update(best_epoch=summary["best_epoch"], best_score=summary["best_score"])
        for name in ["summary.json", "contract.json", "split_manifest.json", "config.json", "out_predictions.pt"]:
            record(root/name)
        print(f"已核验 {label}", flush=True)
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise ValueError("预测包含非有限值。")

    # 重新计算输入覆盖；只用实际 train，不沿用早期全 split 的探索性参考。
    combined = pd.concat([train, out], ignore_index=True)
    x = np.concatenate([tx, ox])
    groups = sorted(train.group.unique())
    ref = np.concatenate([anchors(np.flatnonzero(train.group == g), combined, 192) for g in groups])
    cal = np.concatenate([anchors(ref[combined.loc[ref, "group"].to_numpy() == g], combined, 64) for g in groups])
    query = np.arange(len(train), len(combined))
    names = np.load(norm_path)["input_channels"].tolist()
    input_method = {}
    masks = {"all": np.ones(len(out), bool)}
    label_thresholds = {}
    for name, q in [("support", (0, 1)), ("p01_p99", (.01, .99)), ("p05_p95", (.05, .95))]:
        part, low, high = sfr_partition(out.Y.to_numpy(), train.Y.to_numpy(), q)
        out[f"Y_{name}"] = part
        label_thresholds[name] = {"low": low, "high": high, "quantiles": q}
        for group in ["inside", "low", "high"]:
            masks[f"Y_{name}_{group}"] = part == group
        masks[f"Y_{name}_outside"] = part != "inside"
    sets = feature_sets(names, coord)
    for mode in ["all_input", "all_input_disk10", "thermo_inner30"]:
        channels, spatial = sets[mode]
        cd, _ = global_distance(x, combined, ref, cal, channels, spatial, "cpu", True)
        distance, _ = global_distance(x, combined, ref, query, channels, spatial, "cpu")
        out[f"X_{mode}_distance"] = distance[:, 0]
        input_method[mode] = {}
        for quantile in [.90, .95, .99]:
            threshold = float(np.quantile(cd[:, 0], quantile))
            name = f"X_{mode}_p{round(quantile*100)}"
            covered = distance[:, 0] <= threshold
            masks[name+"_inside"], masks[name+"_outside"] = covered, ~covered
            input_method[mode][str(quantile)] = threshold
            if quantile == .95:
                out[f"X_{mode}_inside"] = covered
                out[f"X_{mode}_ratio"] = distance[:, 0]/threshold
        print(f"已计算输入覆盖 {mode}", flush=True)
    for yi in [True, False]:
        for xi in [True, False]:
            masks[f"joint_Y{'in' if yi else 'out'}_X{'in' if xi else 'out'}"] = (
                (out.Y_support.eq("inside").to_numpy() == yi) & (out.X_all_input_inside.to_numpy() == xi))
    # 时间窗仅用于描述，不把它们称为已验证的吸积事件。
    for label, mask in [("t_lt9", out.time < 9), ("t_9to10", (out.time >= 9) & (out.time < 10)), ("t_ge10", out.time >= 10)]:
        masks[label] = np.asarray(mask)
    rows, ci = [], []
    y, t = out.Y.to_numpy(), out.time.to_numpy()
    for regime, mask in masks.items():
        for model, pred in predictions.items():
            rows.append({"regime": regime, "model": model, "N": int(mask.sum()),
                **(regression(y[mask], pred[mask]) if mask.any() else {})})
        if regime in ["Y_support_inside", "Y_support_outside", "X_all_input_p95_inside", "X_all_input_p95_outside"]:
            for model in UNET:
                for width in [.25, .5]:
                    low, high, nblock = block_delta_ci(t, y, predictions[model], predictions["CNN43"], mask, width, repeats)
                    ci.append({"regime": regime, "model": model, "block_gyr": width,
                        "delta_MAE": float(np.mean(np.abs(predictions[model][mask]-y[mask])-np.abs(predictions["CNN43"][mask]-y[mask]))),
                        "CI025": low, "CI975": high, "nonempty_blocks": nblock})
    output.mkdir(parents=True)
    for model, pred in predictions.items():
        out[model] = pred
    out.to_csv(output/"predictions_and_domains.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output/"metrics.csv", index=False)
    pd.DataFrame(ci).to_csv(output/"paired_block_bootstrap.csv", index=False)
    train.groupby("group").Y.agg(["count", "min", "median", "max"]).to_csv(output/"train_sfr_by_run.csv")
    # 连续段只连接相同类别的相邻有效帧；明显数据缺口会切段。
    segments = []
    gap = 5*np.median(np.diff(t))
    for column in ["Y_support", "Y_p01_p99", "X_all_input_inside"]:
        values = out[column].to_numpy()
        starts = np.r_[0, np.flatnonzero((values[1:] != values[:-1]) | (np.diff(t) > gap))+1]
        stops = np.r_[starts[1:], len(t)]
        for start, stop in zip(starts, stops):
            segments.append({"definition": column, "category": str(values[start]), "start_gyr": t[start],
                             "end_gyr": t[stop-1], "N": stop-start})
    pd.DataFrame(segments).to_csv(output/"time_segments.csv", index=False)
    method = {"sources": {**CNN, **UNET}, "sha256": hashes, "contracts": contracts,
        "train_unique_N": len(train), "out_N": len(out), "label_thresholds": label_thresholds,
        "input_thresholds": input_method, "reference_N": len(ref), "calibration_N": len(cal),
        "input_method": "每训练运行最多192时间锚点；每运行最多64校准点；排除同运行±0.25Gyr近邻；参考通道标准化RMS；train-only阈值",
        "bootstrap": {"repeats": repeats, "seed": 42, "block_gyr": [.25, .5], "scope": "给定训练模型与单一留出运行的描述性配对区间，不包含训练/运行间不确定性"},
        "units": "Msun/yr", "time": "snapshot绝对时间/Gyr；未减0.5625Gyr", "sfr_factor": .025,
        "interpretation": "Y分层依赖事后标签，不能当成部署OOD检测器；X覆盖依赖度量及空间窗口，不是物理支持集证明。全部原始预测保留，无裁剪或排除。"}
    (output/"method.json").write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(out, metrics, output, label_thresholds)
    print(metrics[metrics.regime.isin(["all", "Y_support_inside", "Y_support_low", "X_all_input_p95_inside", "X_all_input_p95_outside"])].to_string(index=False), flush=True)
    return method


def plot(frame, metrics, output, thresholds):
    """保留离群值；symlog 同时显示低 SFR、负预测和大尖峰。"""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"CNN43": "#cf594b", "UNet_original42": "#2377a4", "UNet_ReLU42": "#269080",
              "UNet_ReLU43": "#b28a27", "UNet_ReLU44": "#8056a0"}
    t = frame.time.to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [3, 2, .65, 2]})
    gaps = np.flatnonzero(np.diff(t) > 5*np.median(np.diff(t)))+1
    def line(ax, values, **kwargs):
        # 数据缺口断线，避免图中连线被误读为缺失时段的预测。
        ax.plot(np.insert(t, gaps, np.nan), np.insert(np.asarray(values, float), gaps, np.nan), **kwargs)
    for ix in gaps:
        for ax in axes:
            ax.axvspan(t[ix-1], t[ix], color="#eeeeee", zorder=-1)
    line(axes[0], frame.Y, color="#222222", label="True", linewidth=1.3)
    line(axes[0], frame.Baseline, color="#999999", label="Baseline", linewidth=.8)
    for model, color in colors.items():
        line(axes[0], frame[model], color=color, label=model, linewidth=.7, alpha=.8)
    axes[0].set_yscale("symlog", linthresh=.1)
    axes[0].set_ylabel("SFR [Msun/yr]")
    axes[0].axhline(thresholds["support"]["low"], color="#666666", linestyle=":", linewidth=.8)
    axes[0].legend(ncol=4, fontsize=8)
    for model, color in colors.items():
        line(axes[1], np.abs(frame[model]-frame.Y), color=color, linewidth=.7, label=model)
    axes[1].set_yscale("symlog", linthresh=.05)
    axes[1].set_ylabel("Absolute error [Msun/yr]")
    for column, label, ypos in [("Y_support", "SFR outside min-max", 0), ("Y_p01_p99", "SFR outside p01-p99", 1)]:
        axes[2].scatter(t, np.full(len(t), ypos), c=np.where(frame[column].eq("inside"), "#d5e7e2", "#cb5a49"), s=3, marker="|")
    axes[2].set_yticks([0, 1], ["SFR support", "SFR p01-p99"])
    axes[2].set_title("Red = outside training SFR interval; green = inside", loc="left", fontsize=10)
    for column, label in [("all_input", "Whole field, all channels"), ("all_input_disk10", "Inner disk, all channels"), ("thermo_inner30", "r < 30 kpc, thermodynamics")]:
        line(axes[3], frame[f"X_{column}_ratio"], linewidth=.8, label=label)
    axes[3].axhline(1, color="#222222", linestyle="--", linewidth=1)
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Input distance / train p95")
    axes[3].set_xlabel("Snapshot absolute time [Gyr]")
    axes[3].legend(fontsize=8)
    fig.suptitle("Exp2 held-out Fiducial: SFR interval and input-state coverage")
    fig.savefig(output/"domain_history.png", dpi=165)
    plt.close(fig)
    regimes = ["Y_support_inside", "Y_support_low", "X_all_input_p95_inside", "X_all_input_p95_outside"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        for i, model in enumerate(["Baseline", *colors]):
            values = metrics[metrics.model.eq(model)].set_index("regime").loc[regimes, metric]
            ax.bar(np.arange(4)+(i-2.5)*.12, values, width=.115, label=model, color=colors.get(model, "#999999"))
        ax.set_xticks(range(4), ["SFR inside", "SFR low outside", "Input inside", "Input outside"], rotation=15)
        ax.set_ylabel(f"{metric} [Msun/yr]")
        ax.set_yscale("log")
    axes[1].legend(fontsize=8)
    fig.savefig(output/"domain_metrics.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    run(args.output_dir, args.bootstrap_repeats)
