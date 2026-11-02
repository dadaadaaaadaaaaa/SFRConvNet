"""比较五组盘星系的 LR 状态覆盖、局部组合与相似输入的 SFR 差异。

读取已有 v7 CachePT，去除 oversampling 的重复 UID；仅在独立目录保存研究统计和图。
五组轮流留出的比较是数据分析，不训练模型，也不据此重选已有实验的超参数。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_"
GROUPS = ["disk_galaxy_fiducial", "disk_galaxy_fiducial_4", "disk_galaxy_fiducial_7",
          "disk_galaxy_model_0_75", "disk_galaxy_model_1_1"]
SHORT = ["Fiducial", "Fiducial4", "Fiducial7", "Model0.75", "Model1.1"]
SHORT_MAP = dict(zip(GROUPS, SHORT))
PREDICTIONS = ROOT / "Results/Exp2_Rh12x20_Closure_v7_LOO_Fiducial_Trial7_seed42_20261025_180348/raw_held_out_predictions.csv"
FACTOR = 0.025


def anchors(indices, frame, count):
    """按物理时间等间隔取最近可用帧，避免高输出频率运行支配参考库。"""
    indices = np.asarray(indices)
    ordered = indices[np.argsort(frame.loc[indices, "time"].to_numpy())]
    t = frame.loc[ordered, "time"].to_numpy()
    wanted = np.linspace(t[0], t[-1], min(count, len(t)))
    right = np.searchsorted(t, wanted).clip(0, len(t)-1)
    left = (right-1).clip(0, len(t)-1)
    chosen = np.where(abs(t[left]-wanted) < abs(t[right]-wanted), left, right)
    return np.unique(ordered[chosen])


def load_data():
    """只读取一个统一 cache 家族，保留真实训练身份，不重新划分训练数据。"""
    arrays, frames = [], []
    for split in ["train", "val", "test", "out"]:
        path = ROOT / ".cache" / f"{PREFIX}{split}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        t = payload["tensors"]
        f = pd.DataFrame({"uid": payload["sample_identity"]["sample_uid"],
                          "group": np.asarray(t["groups"]).reshape(-1),
                          "time": np.asarray(t["time"]).reshape(-1),
                          "Y": np.asarray(t["target_sfr_physical"]).reshape(-1)*FACTOR,
                          "B": np.asarray(t["macer_sfr_baseline"]).reshape(-1)*FACTOR,
                          "split": split})
        keep = ~f.uid.duplicated()
        arrays.append(np.asarray(t["x"])[keep].astype(np.float32))
        frames.append(f.loc[keep].copy())
        if split == "train":
            coord = np.asarray(t["coord"])[0].copy()
        print(f"读取 {split}: {len(f)} -> {keep.sum()} 独立 snapshot", flush=True)
        del payload, t
        gc.collect()
    frame = pd.concat(frames, ignore_index=True)
    assert not frame.uid.duplicated().any()
    x = np.concatenate(arrays)
    del arrays
    norm = np.load(ROOT / ".cache" / f"{PREFIX}norm.npz")
    names = norm["input_channels"].tolist()
    mask = norm["log_transform_mask"]
    return x, frame, coord, names, dict(mean=norm["mean"], std=norm["std"], mask=mask)


def transform_physical(x, norm):
    """反变换只用于物理摘要；距离计算仍使用明确声明的输入表示。"""
    raw = x.astype(np.float64)*norm["std"]+norm["mean"]
    raw[:, norm["mask"]] = np.expm1(raw[:, norm["mask"]])
    return raw


def feature_sets(names, coord):
    thermo = [names.index(k) for k in ["density", "temperature", "cooling_time"]]
    # 相同位置比较时，三个热力学量能较直接地表征 SFR 阈值和冷却状态。
    disk = (coord[0] < 10.0) & (abs(np.cos(coord[1])) < .3)
    return {"all_input": (list(range(len(names))), np.ones(coord.shape[1:], bool)),
            "thermo_inner30": (thermo, coord[0] < 30.0),
            "all_input_disk10": (list(range(len(names))), disk),
            "thermo_disk10": (thermo, disk)}


def global_distance(x, frame, reference, query, channels, spatial, device, calibrate=False):
    """精确欧氏近邻；距离为参考通道标准化后的全特征 RMS。"""
    r = x[reference][:, channels]
    # 减去共同空间模板不改变欧氏距离，同时减少相近状态相减的浮点消减误差。
    mu = r.mean(axis=0, keepdims=True)
    sd = r.std(axis=(0, 2, 3), keepdims=True).clip(1e-5)
    r = ((r-mu)/sd)[:, :, spatial].reshape(len(reference), -1)
    rt = torch.as_tensor(r, device=device)
    rn = (rt*rt).sum(1)
    out_distance, out_ids = [], []
    for start in range(0, len(query), 64):
        ids = query[start:start+64]
        q = ((x[ids][:, channels]-mu)/sd)[:, :, spatial].reshape(len(ids), -1)
        qt = torch.as_tensor(q, device=device)
        d = ((qt*qt).sum(1)[:, None]+rn[None]-2*qt@rt.T).clamp_min(0)
        if calibrate:
            # 参考状态的正常变化尺度：排除同一运行 ±0.25 Gyr 的近邻帧。
            same = frame.group.to_numpy()[ids, None] == frame.group.to_numpy()[None, reference]
            close = abs(frame.time.to_numpy()[ids, None]-frame.time.to_numpy()[None, reference]) < .25
            d.masked_fill_(torch.as_tensor(same & close, device=device), float("inf"))
        value, ix = torch.topk(d, k=min(32, len(reference)), dim=1, largest=False)
        out_distance.append(np.sqrt(value.cpu().numpy()/r.shape[1]))
        out_ids.append(reference[ix.cpu().numpy()])
    return np.concatenate(out_distance), np.concatenate(out_ids)


def local_distance(x, frame, reference, query, channels, device, calibrate=False, return_indices=False):
    """逐空间位置查找局部近邻，保持 r/theta/体积匹配，不混合内外盘单元。"""
    r = x[reference][:, channels].reshape(len(reference), len(channels), -1).transpose(2, 0, 1)
    mu = r.mean(1, keepdims=True)
    global_sd = r.reshape(-1, len(channels)).std(0)
    sd = np.maximum(r.std(1, keepdims=True), .05*global_sd[None, None]).clip(1e-5)
    r = (r-mu)/sd
    q = x[query][:, channels].reshape(len(query), len(channels), -1).transpose(2, 0, 1)
    q = (q-mu)/sd
    if calibrate:
        same = frame.group.to_numpy()[query, None] == frame.group.to_numpy()[None, reference]
        close = abs(frame.time.to_numpy()[query, None]-frame.time.to_numpy()[None, reference]) < .25
        excluded = torch.as_tensor(same & close, device=device)
    distances, neighbors = [], []
    for start in range(0, r.shape[0], 12):
        rt = torch.as_tensor(r[start:start+12], device=device)
        qt = torch.as_tensor(q[start:start+12], device=device)
        d = ((qt*qt).sum(-1)[:, :, None]+(rt*rt).sum(-1)[:, None]-2*torch.bmm(qt, rt.transpose(1,2))).clamp_min(0)
        if calibrate:
            d.masked_fill_(excluded[None], float("inf"))
        nearest=d.min(-1)
        distances.append(np.sqrt(nearest.values.cpu().numpy()/len(channels)))
        neighbors.append(reference[nearest.indices.cpu().numpy()])
    result=np.concatenate(distances).T
    return (result,np.concatenate(neighbors).T) if return_indices else result


def raw_index(group, absolute_time):
    """反演已经核验的二维时间映射，用于定位抽样帧，随后再核对其真实总 SFR。"""
    if "model_" in group:
        return int(round(absolute_time/.0125))
    n=int(round((absolute_time-.5625)/.00125))
    return n-(n-1)//1000


def raw_local_labels(frame, count, out):
    """从 Data-Set-11 保守归约局部标签，HR 信息只用于结果解释。"""
    from pyhdf.SD import SD, SDC
    config=json.loads((ROOT/".config/BaseConfig.json").read_text(encoding="utf-8"))
    raw_config=json.loads((ROOT/".config/RawDataConfig.json").read_text(encoding="utf-8"))["disk_galaxy"]
    labels=np.full((len(frame),240),np.nan,dtype=np.float64)
    eligible_mass=np.full_like(labels,np.nan)
    clumping=np.full_like(labels,np.nan)
    old_cwd=os.getcwd()
    sources=[]
    for group in GROUPS:
        indices=np.flatnonzero(frame.group==group)
        selected=anchors(indices,frame,count)
        training=indices[frame.loc[indices,"split"].to_numpy()=="train"]
        if len(training):
            selected=np.union1d(selected,anchors(training,frame,count))
        folder=Path(config["raw_data_dir"])/raw_config[group.removeprefix("disk_galaxy_")]["folder_name"]
        if not folder.exists():
            raise FileNotFoundError(f"局部真实 SFR 检查需要原始目录：{folder}")
        files={int(p.name.split(".")[-1]):p for p in folder.glob("hdfra.*") if p.name.split(".")[-1].isdigit()}
        try:
            os.chdir(folder)
            for row_id in selected:
                path=files[raw_index(group,frame.time[row_id])]
                handle=SD(path.name,SDC.READ)
                try:
                    fields=[]
                    for name in ["Data-Set-11","Data-Set-5","Data-Set-10","Data-Set-15"]:
                        ds=handle.select(name)
                        fields.append(np.asarray(ds[:],dtype=np.float64).squeeze())
                        ds.endaccess()
                    sfr,rho,energy,volume=fields
                finally:
                    handle.end()
                assert sfr.shape==(72,280)
                def block(a):
                    return a.reshape(12,6,20,14).sum(axis=(1,3)).reshape(-1)
                labels[row_id]=block(sfr)*FACTOR
                assert np.isclose(labels[row_id].sum(),frame.Y[row_id],rtol=2e-6,atol=1e-7)
                assert (labels[row_id]>=0).all()
                cold=(rho*1.02>1)&(93*energy/rho<4e4)
                eligible_mass[row_id]=block(rho*volume*cold)*2.5e7
                cv=block(volume)
                clumping[row_id]=(block(rho*rho*volume)/cv)/(block(rho*volume)/cv)**2
                sources.append(dict(row_id=int(row_id),group=group,path=str(path),Y=frame.Y[row_id]))
        finally:
            os.chdir(old_cwd)
        print(f"原始局部标签 {SHORT_MAP[group]}: {len(selected)} 帧",flush=True)
    pd.DataFrame(sources).to_csv(out/"local_label_sources.csv",index=False)
    ids=np.flatnonzero(np.isfinite(labels).all(1))
    np.savez_compressed(out/"hr_labels_diagnostic_only.npz",row_ids=ids,cell_sfr=labels[ids],
                        eligible_mass_msun=eligible_mass[ids],density_clumping=clumping[ids])
    return labels,eligible_mass,clumping


def add_physical_summary(frame, x, coord, names, norm):
    """摘要全部来自 LR；冷致密质量不表示真实未解析的 HR 成星气体质量。"""
    records = []
    for start in range(0, len(frame), 256):
        a = transform_physical(x[start:start+256], norm)
        f = {name: a[:, i] for i, name in enumerate(names)}
        m = f["gas_mass"]
        eligible = (f["density"]*1.02 > 1) & (f["temperature"] < 4e4)
        inner = coord[0] < 30
        inner_mass = (m*inner).sum((1,2))
        records.append(pd.DataFrame({"gas_mass_msun": m.sum((1,2))*2.5e7,
            "inner30_mass_msun": inner_mass*2.5e7,
            "lr_eligible_mass_msun": (m*eligible).sum((1,2))*2.5e7,
            "lr_eligible_cell_count": eligible.sum((1,2)),
            "inner30_mass_temperature": (m*f["temperature"]*inner).sum((1,2))/inner_mass,
            "max_lr_density": f["density"].max((1,2)),
            "min_lr_temperature": f["temperature"].min((1,2))}))
    return pd.concat([frame.reset_index(drop=True), pd.concat(records, ignore_index=True)], axis=1)


def run_fold(x, frame, coord, names, norm, group, samples, device, actual_train=False, local_truth=None):
    candidates = np.flatnonzero((frame.group != group) & ((frame.split == "train") if actual_train else True))
    ref = np.concatenate([anchors(candidates[frame.loc[candidates, "group"].to_numpy() == g], frame, samples)
                          for g in GROUPS if g != group])
    # 校准查询包含每组相同数量的时间锚点，避免按输出频率给参考分布加权。
    cal = np.concatenate([anchors(ref[frame.loc[ref, "group"].to_numpy() == g], frame, 64)
                          for g in GROUPS if g != group])
    target = np.flatnonzero(frame.group == group)
    query = target if actual_train else anchors(target, frame, samples)
    local_query = anchors(target, frame, samples)
    rows, summary, examples = [], [], []
    for mode, (channels, spatial) in feature_sets(names, coord).items():
        cd, ci = global_distance(x, frame, ref, cal, channels, spatial, device, True)
        threshold = np.quantile(cd[:, 0], .95)
        d, ix = global_distance(x, frame, ref, query, channels, spatial, device)
        q = frame.loc[query].copy()
        q["row_id"] = query
        q["mode"] = mode
        q["fold"] = SHORT_MAP[group]+("_actual_train" if actual_train else "_exploratory")
        q["distance"] = d[:,0]
        q["threshold"] = threshold
        q["distance_ratio"] = d[:,0]/threshold
        q["covered"] = d[:,0] <= threshold
        nearest = frame.loc[ix[:,0]].reset_index(drop=True)
        q["nearest_uid"] = nearest.uid.to_numpy()
        q["nearest_group"] = nearest.group.to_numpy()
        q["nearest_time"] = nearest.time.to_numpy()
        q["nearest_Y"] = nearest.Y.to_numpy()
        q["nearest_B"] = nearest.B.to_numpy()
        q["pair_label_ratio"] = (np.maximum(q.Y, q.nearest_Y)+.01)/(np.minimum(q.Y, q.nearest_Y)+.01)
        q["pair_label_absolute_difference"] = abs(q.Y-q.nearest_Y)
        # 五近邻尽量来自相隔至少 0.25 Gyr 的片段，减少连续帧重复的假象。
        predictions, closure_predictions = [], []
        spread = []
        group_values=frame.group.to_numpy()
        time_values=frame.time.to_numpy()
        y_values=frame.Y.to_numpy()
        c_values=(frame.Y-frame.B).to_numpy()
        for ids in ix:
            selected = []
            for k in ids:
                if all(group_values[k] != group_values[j] or abs(time_values[k]-time_values[j]) >= .25 for j in selected):
                    selected.append(k)
                if len(selected) == 5:
                    break
            labels = y_values[selected]
            predictions.append(np.median(labels))
            closure_predictions.append(np.median(c_values[selected]))
            spread.append((labels.max()+.01)/(labels.min()+.01))
        q["neighbor5_median_Y"] = predictions
        q["neighbor5_median_C"] = closure_predictions
        q["neighbor5_label_range_ratio"] = spread
        rows.append(q)
        # 参考集上的比较只用于建立相似距离下的标签差异标尺。
        near_y = frame.loc[ci[:,0], "Y"].to_numpy()
        cal_y = frame.loc[cal, "Y"].to_numpy()
        summary.append(dict(fold=q.fold.iloc[0], group=group, mode=mode, n=len(q),
            reference_count=len(ref), threshold=threshold,
            coverage=float(q.covered.mean()), median_distance_ratio=float(np.median(q.distance_ratio)),
            median_label_ratio_covered=float(q.loc[q.covered, "pair_label_ratio"].median()),
            covered_pairs_factor3=float(((q.pair_label_ratio >= 3) & (q.pair_label_absolute_difference > .1))[q.covered].mean()),
            reference_calibration_median_label_ratio=float(np.median((np.maximum(cal_y,near_y)+.01)/(np.minimum(cal_y,near_y)+.01))),
            neighbor5_mae=float(np.mean(abs(q.Y-q.neighbor5_median_Y))),
            neighbor5_closure_mae=float(np.mean(abs(q.Y-q.B-q.neighbor5_median_C))),
            baseline_mae=float(np.mean(abs(q.Y-q.B)))))
        examples.append(q.sort_values(["covered", "pair_label_ratio"], ascending=[False,False]).head(15))
        print(f"{q.fold.iloc[0]} {mode}: 覆盖 {q.covered.mean():.3f}, 中位距离比 {q.distance_ratio.median():.3f}", flush=True)
    local_rows = []
    for mode, channels in {"thermo": [names.index(k) for k in ["density","temperature","cooling_time"]],
                           "all_input": list(range(len(names)))}.items():
        cd = local_distance(x, frame, ref, cal, channels, device, True)
        threshold = np.quantile(cd, .95, axis=0).clip(1e-6)
        d, nearest = local_distance(x, frame, ref, local_query, channels, device,return_indices=True)
        cover = d <= threshold[None]
        raw = transform_physical(x[local_query], norm)
        m = raw[:,names.index("gas_mass")].reshape(len(local_query), -1)
        rho = raw[:,names.index("density")].reshape(len(local_query), -1)
        temp = raw[:,names.index("temperature")].reshape(len(local_query), -1)
        inner = (coord[0].reshape(-1) < 30)[None]
        cold_mass = m*((rho*1.02 > 1) & (temp < 4e4))
        def weighted(w):
            return np.divide((w*cover).sum(1), w.sum(1), out=np.full(len(w), np.nan), where=w.sum(1)>0)
        l = frame.loc[local_query].copy()
        l["row_id"] = local_query
        l["fold"] = SHORT_MAP[group]+("_actual_train" if actual_train else "_exploratory")
        l["local_mode"] = mode
        l["cell_coverage"] = cover.mean(1)
        l["inner30_cell_coverage"] = cover[:,inner[0]].mean(1)
        l["inner30_mass_coverage"] = weighted(m*inner)
        l["lr_eligible_mass_coverage"] = weighted(cold_mass)
        l["inner30_mass_distance_ratio"] = ((d/threshold)*m*inner).sum(1)/(m*inner).sum(1)
        if local_truth is not None:
            labels,hr_cold,clumping=local_truth
            truth=labels[local_query]
            assert np.isfinite(truth).all() and np.isfinite(labels[ref]).all()
            near_truth=labels[nearest,np.arange(240)[None]]
            ratio=(np.maximum(truth,near_truth)+1e-5)/(np.minimum(truth,near_truth)+1e-5)
            conflict=(ratio>=3)&(abs(truth-near_truth)>1e-4)
            l["true_sfr_coverage"] = weighted(truth)
            l["hr_eligible_mass_coverage"] = weighted(hr_cold[local_query])
            l["covered_sfr_factor3_conflict"] = np.divide((truth*cover*conflict).sum(1),(truth*cover).sum(1),
                out=np.full(len(truth),np.nan),where=(truth*cover).sum(1)>0)
            l["strong_near_sfr_fraction"] = np.divide((truth*(d/threshold<=.5)).sum(1),truth.sum(1),
                out=np.full(len(truth),np.nan),where=truth.sum(1)>0)
            l["strong_near_conflict_sfr_fraction"] = np.divide((truth*(d/threshold<=.5)*conflict).sum(1),truth.sum(1),
                out=np.full(len(truth),np.nan),where=truth.sum(1)>0)
            l["local_neighbor_sum_Y"] = near_truth.sum(1)
            # 保存贡献较大的实际局部配对，便于核查温度、密度等连续差异。
            interesting=cover & (truth>truth.sum(1)[:,None]*.01) & conflict
            qi,cell=np.where(interesting)
            pair=pd.DataFrame(dict(fold=l.fold.iloc[0],local_mode=mode,
                query_row_id=local_query[qi],cell_index=cell,reference_row_id=nearest[qi,cell],
                distance_ratio=(d/threshold)[qi,cell],query_cell_sfr=truth[qi,cell],
                reference_cell_sfr=near_truth[qi,cell],label_ratio=ratio[qi,cell],
                query_clumping=clumping[local_query[qi],cell],reference_clumping=clumping[nearest[qi,cell],cell]))
            l.attrs["label_pairs"]=pair
        local_rows.append(l)
    pairs=[a.attrs.pop("label_pairs") for a in local_rows if "label_pairs" in a.attrs]
    local_result=pd.concat(local_rows)
    if pairs:
        local_result.attrs["label_pairs"]=pd.concat(pairs,ignore_index=True)
    return pd.concat(rows), local_result, summary, pd.concat(examples)


def fine_comparison(frame, selected, out):
    """补充查看 HR 未解析结构，仅作解释；不将 HR 特征提供给模型。"""
    import h5py
    records = []
    for group in GROUPS:
        available = []
        for dirname in ["fine_sfr_v7", "fine_sfr_v6"]:
            folder = ROOT / "data" / group / dirname
            if folder.exists():
                available = sorted(folder.glob("*.h5"))
                if available:
                    break
        by_index = {int(p.stem): p for p in available}
        for row_id in selected[group]:
            row = frame.loc[row_id]
            index=raw_index(group,row.time)
            path = by_index.get(index)
            if path is None:
                continue
            with h5py.File(path) as h:
                rho = h["snapshot/density"][:]
                e = h["snapshot/gas_energy"][:]
                v = h["geometry/cell_volume"][:]
                temperature = 93*e/rho
                mass = rho*v
                eligible = (rho*1.02 > 1) & (temperature < 4e4)
                # 只保存解释用的 HR 冷致密质量；其信息在粗粒化中可能被平均掉。
                hr_cold = float(mass[eligible].sum()*2.5e7)
                hr_y = float(h["sfr_supervision/canonical/Y_alpha"][()])*.025
                assert np.isclose(hr_y, row.Y, rtol=2e-6, atol=1e-7)
                records.append(dict(row_id=row_id, uid=row.uid, group=group, time=row.time,
                    Y=row.Y, hr_eligible_mass_msun=hr_cold,
                    lr_eligible_mass_msun=row.lr_eligible_mass_msun,
                    hr_to_lr_eligible_mass_ratio=(hr_cold+1)/(row.lr_eligible_mass_msun+1)))
        print(f"HR 结构摘要 {SHORT_MAP[group]}: {sum(r['group']==group for r in records)} 帧", flush=True)
    table = pd.DataFrame(records)
    table.to_csv(out/"hr_structure_context.csv", index=False)
    return table


def plots(frame, whole, local, out):
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(12,4), layout="constrained")
    for mode, color in [("all_input","#0072b2"),("thermo_inner30","#d55e00")]:
        values = [whole[(whole.fold==g+"_exploratory") & (whole["mode"]==mode)].covered.mean()*100 for g in SHORT]
        axes[0].plot(SHORT, values, "o-", label=mode, color=color)
    axes[0].set_ylabel("Whole-field coverage [%]");axes[0].legend()
    for mode, color in [("all_input","#0072b2"),("thermo","#d55e00")]:
        values = [local[(local.fold==g+"_exploratory") & (local.local_mode==mode)].inner30_mass_coverage.mean()*100 for g in SHORT]
        axes[1].plot(SHORT, values, "o-", label=mode, color=color)
    axes[1].set_ylabel("Local coverage, inner-30-kpc mass weighted [%]");axes[1].legend()
    for a in axes: a.tick_params(axis="x",rotation=25);a.set_ylim(-3,103)
    fig.savefig(out/"five_run_coverage.png", dpi=160);plt.close(fig)

    fid=whole[(whole.fold=="Fiducial_actual_train") & (whole["mode"]=="thermo_inner30")].sort_values("time")
    if len(fid):
        fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True,layout="constrained")
        ax[0].plot(fid.time-.5625,fid.Y,label="True SFR",color="black")
        ax[0].plot(fid.time-.5625,fid.B,label="Baseline",color="#0072b2")
        if "Yhat" in fid: ax[0].plot(fid.time-.5625,fid.Yhat,label="CNN",color="#d55e00",alpha=.8)
        ax[0].set_yscale("log");ax[0].set_ylabel(r"SFR [$M_\odot$/yr]");ax[0].legend()
        for mode,color in [("all_input","#0072b2"),("thermo_inner30","#d55e00")]:
            s=whole[(whole.fold=="Fiducial_actual_train") & (whole["mode"]==mode)].sort_values("time")
            ax[1].plot(s.time-.5625,s.distance_ratio,label=mode,color=color,lw=.8)
        ax[1].axhline(1,color="black",ls="--");ax[1].set_ylabel("Whole-field distance / reference p95");ax[1].legend()
        for mode,color in [("all_input","#0072b2"),("thermo","#d55e00")]:
            s=local[(local.fold=="Fiducial_actual_train") & (local.local_mode==mode)].sort_values("time")
            ax[2].plot(s.time-.5625,s.inner30_mass_coverage,label=mode,color=color)
        ax[2].set_ylabel("Local inner-30-kpc mass coverage");ax[2].set_xlabel("Production time [Gyr]");ax[2].legend()
        fig.savefig(out/"fiducial_state_and_prediction.png",dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,mode in zip(axes,["all_input","thermo_inner30"]):
        for g in SHORT:
            q=whole[(whole.fold==g+"_exploratory") & (whole["mode"]==mode) & whole.covered]
            ax.scatter(q.Y+.01,q.nearest_Y+.01,s=12,alpha=.55,label=g)
        ax.plot([.01,100],[.01,100],color="black",lw=1);ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlabel("Query true SFR + 0.01");ax.set_ylabel("Nearest reference true SFR + 0.01");ax.set_title(mode)
    axes[1].legend(fontsize=8)
    fig.savefig(out/"similar_inputs_different_labels.png",dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-per-run",type=int,default=192)
    p.add_argument("--output-dir",type=Path,default=ROOT/"VisualScripts/DataAnalysis/outputs/lr_state_generalization_20261025")
    p.add_argument("--refresh",action="store_true",help="重新生成本脚本已有的分析产物")
    args=p.parse_args()
    out=args.output_dir
    if out.exists():
        marker=out/"method.json"
        if not args.refresh or not marker.exists() or json.loads(marker.read_text(encoding="utf-8")).get("cache_prefix") != PREFIX:
            raise FileExistsError(f"已有目录不是本次可更新的分析产品：{out}")
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device="cuda" if torch.cuda.is_available() else "cpu"
    x,frame,coord,names,norm=load_data()
    frame=add_physical_summary(frame,x,coord,names,norm)
    print(f"总计 {len(frame)} 个独立 snapshot；设备 {device}",flush=True)
    local_truth=raw_local_labels(frame,args.samples_per_run,out)
    all_whole,all_local,summaries,examples,label_pairs=[],[],[],[],[]
    for group in GROUPS:
        w,l,s,e=run_fold(x,frame,coord,names,norm,group,args.samples_per_run,device,local_truth=local_truth)
        label_pairs.append(l.attrs.pop("label_pairs"))
        all_whole.append(w);all_local.append(l);summaries.extend(s);examples.append(e)
    w,l,s,e=run_fold(x,frame,coord,names,norm,GROUPS[0],args.samples_per_run,device,True,local_truth)
    label_pairs.append(l.attrs.pop("label_pairs"))
    all_whole.append(w);all_local.append(l);summaries.extend(s);examples.append(e)
    whole=pd.concat(all_whole,ignore_index=True)
    local=pd.concat(all_local,ignore_index=True)
    # 只关联已有正式模型的预测，不产生新模型，也不把探索性折当作已训练实验。
    pred=pd.read_csv(PREDICTIONS)
    pt=pred.time.to_numpy()
    fidrows=np.flatnonzero(frame.group==GROUPS[0])
    for row in fidrows:
        j=int(np.argmin(abs(pt-frame.time[row])))
        assert abs(pt[j]-frame.time[row])<1e-5
        assert np.isclose(pred.sfr_true.iloc[j]*FACTOR,frame.Y[row],rtol=2e-6,atol=1e-7)
        frame.loc[row,"Yhat"]=pred.sfr_pred.iloc[j]*FACTOR
    whole["Yhat"]=frame.loc[whole.row_id,"Yhat"].to_numpy()
    local["Yhat"]=frame.loc[local.row_id,"Yhat"].to_numpy()
    frame["production_time"]=frame.time-.5625
    whole["production_time"]=whole.time-.5625
    local["production_time"]=local.time-.5625
    whole.to_csv(out/"whole_field_neighbors.csv",index=False)
    local.to_csv(out/"local_state_coverage.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"coverage_summary.csv",index=False)
    pd.concat(examples).to_csv(out/"nearest_label_examples.csv",index=False)
    pd.concat(label_pairs,ignore_index=True).to_csv(out/"local_label_pairs.csv",index=False)
    frame.to_csv(out/"snapshot_physical_summary.csv",index=False)
    selected={g:anchors(np.flatnonzero(frame.group==g),frame,48) for g in GROUPS}
    fine=fine_comparison(frame,selected,out)
    phase_rows=[]
    for lo,hi in [(0,7.5),(7.5,9),(9,12)]:
        for mode in feature_sets(names,coord):
            f=whole[(whole.fold=="Fiducial_actual_train")&(whole["mode"]==mode)&(whole.production_time>=lo)&(whole.production_time<hi)]
            phase_rows.append(dict(stage=f"[{lo},{hi})",mode=mode,n=len(f),coverage=f.covered.mean(),
                distance_ratio_median=f.distance_ratio.median(),nearest_Y_median=f.nearest_Y.median(),
                Y_median=f.Y.median(),model_mae=abs(f.Yhat-f.Y).mean(),baseline_mae=abs(f.B-f.Y).mean(),
                neighbor_mae=abs(f.neighbor5_median_Y-f.Y).mean(),
                neighbor_closure_mae=abs(f.B+f.neighbor5_median_C-f.Y).mean()))
    pd.DataFrame(phase_rows).to_csv(out/"fiducial_phase_summary.csv",index=False)
    plots(frame,whole,local,out)
    metadata=dict(cache_prefix=PREFIX,groups=GROUPS,input_channels=names,unique_snapshots=len(frame),
        samples_per_run=args.samples_per_run,reference_time_gap_gyr=.25,
        distance_threshold="reference-query nearest-distance p95; same-run +/-0.25 Gyr excluded",
        local_distance="same (theta,r) cell; reference per-cell standardization, std floor 0.05 global std",
        exploratory_reference="other four runs, all splits; no model training or performance claim",
        actual_reference="existing fiducial fold train split only; unique UID; time-uniform subset per run",
        labels_used_only_after_neighbor_search=True,hr_context_is_deployment_input=False,
        raw_hdf4_available=True,local_true_sfr_available=True,
        elapsed_seconds=time.perf_counter()-started)
    (out/"method.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(pd.DataFrame(phase_rows).to_string(index=False),flush=True)
    print(f"完成，用时 {time.perf_counter()-started:.1f} s；产物 {out}",flush=True)


if __name__=="__main__":
    main()
