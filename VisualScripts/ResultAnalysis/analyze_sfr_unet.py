"""分析 SFRUNet 已导出的物理预测：Exp1/2 总量、逐格、积分历史及显式 burst 窗口。"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd
import torch

SFR_FACTOR = .025


def array(value):
    """预测文件为可信本项目产物；仅在 CPU 分析，不加载模型或原始数据。"""
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path, *, identity, expected_groups, expected_loss=None):
    """核对文件 schema、单位、逐格恒等式、总和、UID、运行及时间，不猜测旧标量格式。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") not in {"macnet_sfr_unet_predictions_v1", "macnet_sfr_unet_predictions_v2"}
            or payload.get("units") != "MACER_mass_code/Gyr" or payload.get("to_msun_per_yr") != SFR_FACTOR):
        raise ValueError("预测 schema/单位不匹配。")
    result = {k: array(payload[k]).astype(np.float64) for k in
              ("prediction_map", "target_map", "baseline_map", "closure_map")}
    shape = result["target_map"].shape
    if len(shape) != 4 or shape[1] != 1 or min(shape) < 1:
        raise ValueError("地图必须为非空 N1HW。")
    for key, value in result.items():
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key} 形状或有限性不合法。")
    if (result["target_map"] < 0).any() or (result["baseline_map"] < 0).any():
        raise ValueError("Y/B 必须非负；预测负值保留用于诊断。")
    if not np.allclose(result["prediction_map"], result["baseline_map"]+result["closure_map"], rtol=1e-10, atol=1e-10):
        raise ValueError("预测不满足 B+C_hat。")
    constraint = payload.get("output_constraint", "none")
    if constraint not in {"none", "relu"}:
        raise ValueError("未知输出约束。")
    if expected_loss is not None and constraint != expected_loss.get("output_constraint", "none"):
        raise ValueError("预测输出约束与训练 loss 契约不一致。")
    if payload["schema"] == "macnet_sfr_unet_predictions_v2":
        if "output_constraint" not in payload or payload.get("loss_contract", {}).get("output_constraint") != constraint:
            raise ValueError("v2 缺少一致的输出约束契约。")
        if expected_loss is not None and payload["loss_contract"] != expected_loss:
            raise ValueError("预测与训练 loss 契约不一致。")
        for key in ("candidate_map", "candidate_closure_map"):
            value = array(payload[key]).astype(np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("候选地图形状/有限性不合法。")
            result[key] = value
        if not np.allclose(result["candidate_map"], result["baseline_map"]+result["candidate_closure_map"], rtol=1e-10, atol=1e-10):
            raise ValueError("候选 SFR 与候选修正不一致。")
        expected = np.maximum(result["candidate_map"], 0) if constraint == "relu" else result["candidate_map"]
        if not np.allclose(result["prediction_map"], expected, rtol=1e-10, atol=1e-10):
            raise ValueError("最终预测与候选值的输出映射不一致。")
    elif constraint != "none":
        raise ValueError("v1 只支持旧无约束输出。")
    if constraint == "relu" and (result["prediction_map"] < 0).any():
        raise ValueError("relu 模式的最终 SFR 不得为负。")
    for name in ("prediction", "target", "baseline"):
        total = array(payload[name+"_total"]).reshape(-1)
        if total.shape != (shape[0],) or not np.isfinite(total).all() or not np.allclose(
                total, result[name+"_map"].sum((1,2,3)), rtol=1e-10, atol=1e-10):
            raise ValueError(f"{name} 地图求和与总量不一致。")
        result[name+"_total"] = total
    for key in ("sample_uid", "parent_hr_uid"):
        values = list(payload[key])
        if values != identity[key] or len(values) != shape[0] or not all(values) or len(set(values)) != len(values):
            raise ValueError(f"预测 {key} 与 split manifest 不匹配或重复。")
        result[key] = values
    result["groups"] = np.asarray(payload["groups"], dtype=str).reshape(-1)
    result["time"] = array(payload["time"]).astype(float).reshape(-1)
    if (len(result["groups"]) != shape[0] or set(result["groups"]) != set(expected_groups)
            or len(result["time"]) != shape[0] or not np.isfinite(result["time"]).all()):
        raise ValueError("运行集合或时间与实验契约不匹配。")
    return result


def regression(true, pred):
    """物理量统计；常数真值 R2、零分母相对误差不伪造数值。"""
    true, pred = np.asarray(true), np.asarray(pred)
    error = pred-true
    denominator = np.sum((true-true.mean())**2)
    nonzero = true != 0
    return {"MAE": float(np.mean(abs(error))), "RMSE": float(np.sqrt(np.mean(error**2))),
            "Bias": float(error.mean()), "R2": float(1-np.sum(error**2)/denominator) if denominator > 0 else None,
            "MedianAbsRelativeError_nonzero": float(np.median(abs(error[nonzero]/true[nonzero]))) if nonzero.any() else None,
            "NegativePredFrac": float(np.mean(pred < 0))}


def sample_frame(data, split):
    """每行一个 snapshot，物理列显式注明 Msun/yr，避免与旧变换域 y_pred 混淆。"""
    frame = pd.DataFrame({"sample_uid": data["sample_uid"], "parent_hr_uid": data["parent_hr_uid"],
        "type": data["groups"], "time_gyr": data["time"], "split": split,
        "true_sfr_msun_per_yr": data["target_total"]*SFR_FACTOR,
        "model_sfr_msun_per_yr": data["prediction_total"]*SFR_FACTOR,
        "baseline_sfr_msun_per_yr": data["baseline_total"]*SFR_FACTOR})
    for label, key in (("model", "prediction"), ("baseline", "baseline")):
        error = (data[key+"_map"]-data["target_map"])*SFR_FACTOR
        frame[label+"_error"] = error.sum((1,2,3))
        frame[label+"_cell_mae"] = abs(error).mean((1,2,3))
        frame[label+"_cell_rmse"] = np.sqrt((error**2).mean((1,2,3)))
        frame[label+"_spatial_abs_error"] = abs(error).sum((1,2,3))
        frame[label+"_negative_cell_fraction"] = (data[key+"_map"] < 0).mean((1,2,3))
        frame[label+"_negative_sfr_magnitude"] = np.maximum(-data[key+"_map"],0).sum((1,2,3))*SFR_FACTOR
        truth = data["target_map"]
        # 每帧虚假正SFR与正标签低估量，单位Msun/yr；不与零真值做除法。
        frame[label+"_zero_cell_false_sfr"] = (np.maximum(data[key+"_map"],0)*(truth==0)).sum((1,2,3))*SFR_FACTOR
        frame[label+"_positive_cell_under_sfr"] = (np.maximum(truth-data[key+"_map"],0)*(truth>0)).sum((1,2,3))*SFR_FACTOR
    return frame


def integrate_series(time, values, gap_factor=5.):
    """只积分已覆盖的相邻区间，明显大间隙不跨越；返回从首帧起的观测区间累计质量。"""
    time, values = np.asarray(time), np.asarray(values)
    dt = np.diff(time)
    if not np.isfinite(time).all() or np.any(dt <= 0):
        raise ValueError("同一运行时间必须严格递增；不能静默合并重复时刻。")
    threshold = gap_factor*np.median(dt) if len(dt) else np.inf
    valid = dt <= threshold
    increments = .5*(values[1:]+values[:-1])*dt*1e9
    increments[~valid] = 0
    cumulative = np.r_[0., np.cumsum(increments)]
    return cumulative, valid


def integral_row(frame, *, split, group, gap_factor):
    row = {"split": split, "run": group, "n": len(frame), "t_start_gyr": float(frame.time_gyr.iloc[0]),
           "t_end_gyr": float(frame.time_gyr.iloc[-1]), "scope": "cached_out_observed_intervals"}
    time = frame.time_gyr.to_numpy()
    trajectories = frame.copy()
    for label in ("true", "model", "baseline"):
        cumulative, valid = integrate_series(time, frame[label+"_sfr_msun_per_yr"].to_numpy(), gap_factor)
        trajectories[label+"_cumulative_mass_msun"] = cumulative
        row[label+"_formed_mass_msun"] = float(cumulative[-1]) if len(frame)>1 else None
    row["excluded_gap_count"] = int(np.sum(~valid))
    row["covered_duration_gyr"] = float(np.diff(time)[valid].sum())
    for label, display in (("model", "Model"), ("baseline", "Baseline")):
        truth, pred = row["true_formed_mass_msun"], row[label+"_formed_mass_msun"]
        row[display+" Error all time"] = ((pred-truth)/truth*100) if truth not in (None,0) else None
        row[label+"_mass_error_msun"] = pred-truth if truth is not None else None
    return row, trajectories


def threshold_event(time, values, threshold):
    """窗口内首个阈值事件；边界处已在事件中则标注删失，交叉时刻线性插值。"""
    above = np.asarray(values) >= threshold
    positions = np.flatnonzero(above)
    if not len(positions):
        return {"detected": False}
    start = int(positions[0]); end = start
    while end+1 < len(above) and above[end+1]:
        end += 1
    def cross(i,j):
        return float(time[i]+(threshold-values[i])/(values[j]-values[i])*(time[j]-time[i]))
    return {"detected": True, "start_gyr": float(time[0]) if start==0 else cross(start-1,start),
            "end_gyr": float(time[-1]) if end==len(time)-1 else cross(end,end+1),
            "left_censored": start==0, "right_censored": end==len(time)-1,
            "event_count": int(above[0])+int(np.sum(~above[:-1]&above[1:]))}


def burst_metrics(frame, window, *, offset=0., threshold=None, gap_factor=5.):
    """显式窗口属于 time_gyr+offset 坐标；不内置任何星系的论文窗口或背景阈值。"""
    f = frame.sort_values("time_gyr").copy()
    t = f.time_gyr.to_numpy()+offset
    start,end = map(float,window)
    if not np.isfinite([start,end,offset]).all() or start >= end or start < t[0] or end > t[-1]:
        raise ValueError("burst 窗口必须有限、递增且在数据覆盖范围内；不外推。")
    mask = (t>=start)&(t<=end)
    if not mask.any():
        raise ValueError("burst 窗口内没有真实采样点。")
    _, valid = integrate_series(t,np.zeros(len(t)),gap_factor)
    if np.any((t[:-1]<end)&(t[1:]>start)&~valid):
        raise ValueError("burst 窗口跨越明显数据缺口，拒绝自动插值积分或测量时刻。")
    times = np.unique(np.r_[start,t[mask],end])
    series = {label: np.interp(times,t,f[label+"_sfr_msun_per_yr"]) for label in ("true","model","baseline")}
    truth = f.true_sfr_msun_per_yr.to_numpy()[mask]
    result=[]
    for label in ("model","baseline"):
        row = {"run": str(f.type.iloc[0]),"method": label,"start_gyr":start,"end_gyr":end,
               "time_offset_gyr":offset,"n_samples":int(mask.sum()),
               **regression(truth,f[label+"_sfr_msun_per_yr"].to_numpy()[mask])}
        integral = lambda x: float(np.sum((x[1:]+x[:-1])*.5*np.diff(times))*1e9)
        mtrue,mpred=integral(series["true"]),integral(series[label])
        peak_true,peak_pred=float(series["true"].max()),float(series[label].max())
        row.update(true_formed_mass_msun=mtrue,predicted_formed_mass_msun=mpred,
                   mass_error_msun=mpred-mtrue, mass_relative_error=(mpred-mtrue)/mtrue if mtrue else None,
                   peak_relative_error=(peak_pred-peak_true)/peak_true if peak_true else None,
                   peak_time_error_gyr=float(times[np.argmax(series[label])]-times[np.argmax(series["true"])]))
        if threshold is not None:
            if not np.isfinite(threshold) or threshold<=0:
                raise ValueError("事件阈值必须是正的有限 SFR（Msun/yr）。")
            actual,pred=threshold_event(times,series["true"],threshold),threshold_event(times,series[label],threshold)
            row.update(threshold_msun_per_yr=threshold,true_event=actual,predicted_event=pred,
                       start_error_gyr=None,end_error_gyr=None)
            for edge,censored in (("start","left_censored"),("end","right_censored")):
                if actual["detected"] and pred["detected"] and not actual[censored] and not pred[censored]:
                    row[edge+"_error_gyr"]=pred[edge+"_gyr"]-actual[edge+"_gyr"]
        result.append(row)
    return result


def save_figure(fig,path):
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)


def plot_group(frame, directory, *, trajectory=None, burst_window=None, time_offset=0., gap_factor=5.):
    """总量曲线保留零和负预测；随机test标明稀疏采样，不绘制全时段累计质量。"""
    t=frame.time_gyr.to_numpy()
    fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True,constrained_layout=True)
    for label,color in (("true","black"),("model","tab:red"),("baseline","tab:green")):
        axes[0].plot(t,frame[label+"_sfr_msun_per_yr"],label=label,color=color,lw=1)
    axes[0].set_yscale("symlog",linthresh=1e-4)
    axes[0].set_ylabel("SFR [Msun/yr]"); axes[0].legend()
    for label in ("model","baseline"):
        axes[1].plot(t,frame[label+"_error"],label=label)
        axes[2].plot(t,abs(frame[label+"_error"]),label=label+" |sum error|")
        axes[2].plot(t,frame[label+"_spatial_abs_error"],ls="--",label=label+" sum |error|")
    axes[1].axhline(0,color="gray",lw=.6); axes[1].set_ylabel("Signed error [Msun/yr]")
    axes[1].legend()
    axes[2].set_yscale("symlog",linthresh=1e-4); axes[2].legend(fontsize=8)
    axes[2].set_ylabel("Error cancellation [Msun/yr]"); axes[2].set_xlabel("Snapshot time [Gyr]")
    if burst_window:
        for axis in axes:
            axis.axvspan(burst_window[0]-time_offset,burst_window[1]-time_offset,color="orange",alpha=.12)
    gaps=[]
    if trajectory is not None:
        _,valid=integrate_series(t,np.zeros(len(t)),gap_factor)
        gaps=[(t[i],t[i+1]) for i in np.flatnonzero(~valid)]
        for axis in axes:
            for a,b in gaps:
                axis.axvspan(a,b,color="gray",alpha=.2)
    fig.suptitle(f"{frame.type.iloc[0]} / {frame.split.iloc[0]}" + (" (sparse test samples)" if trajectory is None else ""))
    save_figure(fig,directory/"sfr_history.png")
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for label in ("model","baseline"):
        axes[0].scatter(frame.true_sfr_msun_per_yr,frame[label+"_sfr_msun_per_yr"],s=5,label=label,alpha=.5)
        axes[1].plot(t,frame[label+"_cell_rmse"],label=label)
    limits=[min(0.,float(frame.model_sfr_msun_per_yr.min())),float(frame[["true_sfr_msun_per_yr","model_sfr_msun_per_yr","baseline_sfr_msun_per_yr"]].max().max())]
    axes[0].plot(limits,limits,color="gray",ls="--"); axes[0].set_xlabel("True SFR [Msun/yr]")
    axes[0].set_ylabel("Predicted SFR [Msun/yr]"); axes[0].legend()
    axes[1].set_xlabel("Snapshot time [Gyr]"); axes[1].set_ylabel("Cell RMSE [Msun/yr]"); axes[1].legend()
    save_figure(fig,directory/"scatter_and_cell_error.png")
    if trajectory is not None:
        fig,axis=plt.subplots(figsize=(10,4))
        for label in ("true","model","baseline"):
            axis.plot(t,trajectory[label+"_cumulative_mass_msun"],label=label)
        axis.set_xlabel("Snapshot time [Gyr]"); axis.set_ylabel("Cumulative formed mass [Msun]")
        axis.set_title("Observed intervals, zero at first sample; excluded gaps not integrated")
        for a,b in gaps:
            axis.axvspan(a,b,color="gray",alpha=.2)
        axis.legend(); save_figure(fig,directory/"cumulative_sfr.png")


def plot_maps(data,index,path):
    """仅索引网格，不伪装为真实等面积空间；统一色标比较逐格积分SFR。"""
    y,b,p=[data[name+"_map"][index,0]*SFR_FACTOR for name in ("target","baseline","prediction")]
    high=max(float(np.max(abs(x))) for x in (y,b,p)) or 1e-9
    error_high=max(float(np.max(abs(x))) for x in (b-y,p-y,p-b)) or 1e-9
    norms=[SymLogNorm(max(high*1e-4,1e-12),vmin=-high if (p<0).any() else 0,vmax=high)]*3
    norms += [SymLogNorm(max(error_high*1e-4,1e-12),vmin=-error_high,vmax=error_high)]*3
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for axis,values,title,norm in zip(axes.flat,(y,b,p,b-y,p-y,p-b),
            ("True","Baseline","Model","Baseline - True","Model - True","Predicted closure"),norms):
        im=axis.imshow(values,origin="lower",aspect="auto",norm=norm,cmap="RdBu_r" if norm.vmin<0 else "viridis")
        axis.set_title(title); axis.set_xlabel("r cell index"); axis.set_ylabel("theta cell index")
        ticks=([-norm.vmax,-norm.linthresh,0,norm.linthresh,norm.vmax] if norm.vmin<0
               else [0,norm.linthresh,np.sqrt(norm.linthresh*norm.vmax),norm.vmax])
        fig.colorbar(im,ax=axis,label="Cell SFR [Msun/yr]",ticks=ticks,format="%.1e")
    fig.suptitle(f"{data['groups'][index]} t={data['time'][index]:.6g} Gyr / index grid")
    save_figure(fig,path)


def analyze_result(result_dir, *, output_dir=None, plots=True, gap_factor=5., burst_window=None,
                   time_offset=0., burst_threshold=None):
    """读取正式结果，生成独立分析目录；不覆盖已有分析，不重跑模型或访问训练数据。"""
    root=Path(result_dir).resolve()
    if not np.isfinite(gap_factor) or gap_factor<=1:
        raise ValueError("gap_factor 必须有限且>1。")
    if burst_threshold is not None and burst_window is None:
        raise ValueError("事件阈值需要同时指定 burst_window。")
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    contract=json.loads((root/"contract.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"split_manifest.json").read_text(encoding="utf-8"))
    meta=contract["cache_metadata"]
    if summary.get("status")!="complete" or summary.get("hpo") is not False or contract["model"].get("architecture")!="SFRUNet":
        raise ValueError("只分析已完成正式 SFRUNet 训练，不能把暂停或HPO当作最终结果。")
    exp=meta.get("experiment")
    if exp not in {"exp1","exp2"}:
        raise ValueError("仅支持 Exp1/2。")
    train_groups=set(meta["train_groups"]); held=set(meta.get("held_out_groups",[]))
    if train_groups&held or (exp=="exp2" and len(held)!=1):
        raise ValueError("训练/留出运行契约非法。")
    for key in ("sample_uid","parent_hr_uid"):
        for i,left in enumerate(manifest):
            for right in list(manifest)[i+1:]:
                if set(manifest[left][key])&set(manifest[right][key]):
                    raise ValueError(f"{left}/{right} {key} 交叉。")
    loaded={s:load_predictions(root/f"{s}_predictions.pt",identity=manifest[s],
                              expected_groups=held if s=="out" else train_groups, expected_loss=contract.get("loss"))
            for s in (["test","out"] if exp=="exp2" else ["test"])}
    if burst_window and exp!="exp2":
        raise ValueError("burst 分析仅对 Exp2 held-out 轨迹启用。")
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    output=Path(output_dir) if output_dir else root/f"analysis_sfr_unet_{stamp}_{uuid4().hex[:6]}"
    output.mkdir(parents=True,exist_ok=False)
    rows,integrals,bursts=[],[],[]
    for split,data in loaded.items():
        frame=sample_frame(data,split)
        frame.to_csv(output/f"{split}_predictions.csv",index=False)
        for group,part in frame.groupby("type",sort=True):
            part=part.sort_values("time_gyr")
            if np.any(np.diff(part.time_gyr)<=0):
                raise ValueError("同一运行存在重复/倒序时间。")
            indices=part.index.to_numpy()
            for label,key in (("Model","prediction"),("Baseline","baseline")):
                row={"split":split,"run":group,"method":label,"n":len(part),
                     **regression(data["target_total"][indices]*SFR_FACTOR,data[key+"_total"][indices]*SFR_FACTOR)}
                cells=regression(data["target_map"][indices].ravel()*SFR_FACTOR,data[key+"_map"][indices].ravel()*SFR_FACTOR)
                row.update({"Cell_"+k:v for k,v in cells.items()})
                prefix = "model" if label == "Model" else "baseline"
                row["ZeroCellFalseSFR_mean"] = float(part[prefix+"_zero_cell_false_sfr"].mean())
                row["PositiveCellUnderSFR_mean"] = float(part[prefix+"_positive_cell_under_sfr"].mean())
                rows.append(row)
            trajectory=None
            if split=="out":
                row,trajectory=integral_row(part,split=split,group=group,gap_factor=gap_factor)
                integrals.append(row); trajectory.to_csv(output/f"{split}_trajectory.csv",index=False)
                if burst_window:
                    bursts.extend(burst_metrics(part,burst_window,offset=time_offset,threshold=burst_threshold,gap_factor=gap_factor))
            if plots:
                folder=output/(split+"_"+re.sub(r"[^A-Za-z0-9_-]","_",group)); folder.mkdir()
                plot_group(part,folder,trajectory=trajectory,burst_window=burst_window if split=="out" else None,
                           time_offset=time_offset,gap_factor=gap_factor)
                selected={indices[0],indices[-1],indices[np.argmax(data["target_total"][indices])]}
                for index in sorted(selected):
                    plot_maps(data,index,folder/f"maps_{index:05d}.png")
    metrics=pd.DataFrame(rows); integrated=pd.DataFrame(integrals)
    metrics.to_csv(output/"metrics.csv",index=False)
    integrated.to_csv(output/"integrated_metrics.csv",index=False)
    audit={"schema":"macnet_sfr_unet_analysis_v1","result_dir":str(root),"best_epoch":summary["best_epoch"],
           "output_constraint":contract.get("loss",{}).get("output_constraint","none"),
           "loss_contract":contract.get("loss"),
           "units":{"SFR":"Msun/yr","time":"Gyr","mass":"Msun"},"integral_error_convention":"100*(prediction-true)/true",
           "gap_factor":gap_factor,"burst_window":burst_window,"time_offset_gyr":time_offset,
           "burst_threshold_msun_per_yr":burst_threshold,"burst_metrics":bursts,
           "sources":{p.name:file_hash(p) for p in [root/"contract.json",root/"split_manifest.json",root/"summary.json"]+
                      [root/f"{s}_predictions.pt" for s in loaded]},
           "notes":["test为域内稀疏样本，不计算完整历史积分", "累计质量从首帧置零，不等同日志NEWstar",
                    "地图使用索引坐标，每格量已积分；不代表等体积或SFR密度", "无自动论文窗口；阈值事件取窗口内首个事件，删失时不报告边界误差"]}
    (output/"analysis.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    if bursts:
        pd.json_normalize(bursts).to_csv(output/"burst_metrics.csv",index=False)
    if plots and (root/"history.json").is_file():
        history=json.loads((root/"history.json").read_text(encoding="utf-8"))
        fig,axis=plt.subplots(figsize=(8,4))
        axis.plot([h["epoch"] for h in history],[h["train_loss"][0] for h in history],label="Training weighted loss")
        axis.plot([h["epoch"] for h in history],[h["score"] for h in history],label="Validation fixed macro score")
        axis.set_yscale("symlog",linthresh=1e-5); axis.set_xlabel("Epoch"); axis.legend()
        save_figure(fig,output/"learning_curves.png")
    return {"output_dir":str(output),"metrics":metrics,"integrated_metrics":integrated,"burst_metrics":pd.json_normalize(bursts)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir",required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--plots",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--gap-factor",type=float,default=5.)
    parser.add_argument("--burst-window",type=float,nargs=2)
    parser.add_argument("--time-offset",type=float,default=0.)
    parser.add_argument("--burst-threshold",type=float)
    args=parser.parse_args(argv)
    result=analyze_result(**vars(args))
    print(result["metrics"].to_string(index=False)); print("Analysis:",result["output_dir"])


if __name__=="__main__":
    main()
