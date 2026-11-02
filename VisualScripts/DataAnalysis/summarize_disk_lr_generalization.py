"""将五组 LR 泛化检查的数值产物整理成中文报告和真实成星区域覆盖图。"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
GROUPS=["Fiducial","Fiducial4","Fiducial7","Model0.75","Model1.1"]


def table(frame):
    """用普通 Markdown 输出，避免额外依赖 tabulate。"""
    rows=["| "+" | ".join(frame.columns)+" |", "|"+"---|"*len(frame.columns)]
    for row in frame.itertuples(index=False,name=None):
        rows.append("| "+" | ".join(f"{x:.3f}" if isinstance(x,(float,np.floating)) else str(x) for x in row)+" |")
    return "\n".join(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"VisualScripts/DataAnalysis/outputs/lr_state_generalization_20261025")
    out=parser.parse_args().output_dir
    method=json.loads((out/"method.json").read_text(encoding="utf-8"))
    assert method["local_true_sfr_available"]
    whole=pd.read_csv(out/"whole_field_neighbors.csv")
    local=pd.read_csv(out/"local_state_coverage.csv")
    summary=pd.read_csv(out/"coverage_summary.csv")
    phases=pd.read_csv(out/"fiducial_phase_summary.csv")
    fine=pd.read_csv(out/"hr_structure_context.csv")
    pairs=pd.read_csv(out/"local_label_pairs.csv")
    snapshots=pd.read_csv(out/"snapshot_physical_summary.csv")
    rows=[]
    for g in GROUPS:
        w=summary[(summary.fold==g+"_exploratory")&(summary["mode"]=="all_input")].iloc[0]
        a=local[(local.fold==g+"_exploratory")&(local.local_mode=="all_input")]
        b=local[(local.fold==g+"_exploratory")&(local.local_mode=="thermo")]
        strong=whole[(whole.fold==g+"_exploratory")&(whole["mode"]=="all_input")&(whole.distance_ratio<=.5)]
        rows.append(dict(group=g,whole_field_coverage=w.coverage,
            local_true_sfr_coverage=a.true_sfr_coverage.mean(),local_thermo_true_sfr_coverage=b.true_sfr_coverage.mean(),
            inner30_mass_coverage=a.inner30_mass_coverage.mean(),
            covered_sfr_factor3_conflict=a.covered_sfr_factor3_conflict.mean(),
            strong_near_sfr_fraction=a.strong_near_sfr_fraction.mean(),
            strong_near_conflict_sfr_fraction=a.strong_near_conflict_sfr_fraction.mean(),
            global_strong_pairs=len(strong),global_strong_pair_median_ratio=strong.pair_label_ratio.median()))
    five=pd.DataFrame(rows)
    five.to_csv(out/"five_run_interpretation.csv",index=False)
    phases_local=[]
    for lo,hi in [(0,7.5),(7.5,9),(9,12)]:
        for mode in ["all_input","thermo"]:
            s=local[(local.fold=="Fiducial_actual_train")&(local.local_mode==mode)&(local.production_time>=lo)&(local.production_time<hi)]
            phases_local.append(dict(stage=f"[{lo},{hi})",mode=mode,n=len(s),
                true_sfr_coverage=s.true_sfr_coverage.mean(),inner30_mass_coverage=s.inner30_mass_coverage.mean(),
                covered_sfr_factor3_conflict=s.covered_sfr_factor3_conflict.mean(),
                strong_near_sfr_fraction=s.strong_near_sfr_fraction.mean(),
                local_neighbor_mae=abs(s.local_neighbor_sum_Y-s.Y).mean()))
    lp=pd.DataFrame(phases_local)
    lp.to_csv(out/"fiducial_local_phase_summary.csv",index=False)
    probe_rows=[]
    for lo,hi in [(0,7.5),(7.5,9),(9,12)]:
        q=local[(local.fold=="Fiducial_actual_train")&(local.local_mode=="all_input")&(local.production_time>=lo)&(local.production_time<hi)]
        for name,s in [("全部抽样帧",q),("真实SFR局部覆盖>=90%",q[q.true_sfr_coverage>=.9]),("真实SFR局部覆盖<=10%",q[q.true_sfr_coverage<=.1])]:
            if len(s):
                probe_rows.append(dict(stage=f"[{lo},{hi})",subset=name,n=len(s),model_mae=abs(s.Yhat-s.Y).mean(),
                    baseline_mae=abs(s.B-s.Y).mean(),local_neighbor_mae=abs(s.local_neighbor_sum_Y-s.Y).mean()))
    probe=pd.DataFrame(probe_rows)
    probe.to_csv(out/"local_probe_comparison.csv",index=False)
    late_good=probe[(probe.stage=="[9,12)")&(probe.subset=="真实SFR局部覆盖>=90%")]
    late_bad=probe[(probe.stage=="[9,12)")&(probe.subset=="真实SFR局部覆盖<=10%")]
    probe_text="本次没有同时形成足够的高覆盖与低覆盖后期子集，参见逐子集表。"
    if len(late_good) and len(late_bad):
        good=late_good.iloc[0];bad=late_bad.iloc[0]
        probe_text=(f"fiducial 后期有 {int(good.n)} 帧的真实 SFR 局部覆盖达到 90%。这些帧的 CNN MAE 为 {good.model_mae:.5f}，"
                    f"baseline 为 {good.baseline_mae:.5f}，局部近邻求和为 {good.local_neighbor_mae:.5f}。"
                    f"另有 {int(bad.n)} 帧覆盖不足 10%，对应三者 MAE 为 {bad.model_mae:.5f}、{bad.baseline_mae:.5f}、{bad.local_neighbor_mae:.5f}。"
                    "前一子集为尝试局部监督与求和提供了直接依据；后一子集说明局部结构不能保证解决未覆盖状态。")

    combo=[]
    for g in GROUPS:
        for mode in ["all_input","thermo"]:
            w=whole[(whole.fold==g+"_exploratory")&(whole["mode"]=="all_input")]
            l=local[(local.fold==g+"_exploratory")&(local.local_mode==mode)]
            a=w.merge(l[["uid","true_sfr_coverage"]],on="uid",validate="one_to_one")
            combo.append(dict(group=g,local_mode=mode,n=len(a),
                local90_global_uncovered=float(((a.true_sfr_coverage>=.9)&~a.covered).mean()),
                local90_global_covered=float(((a.true_sfr_coverage>=.9)&a.covered).mean())))
    pd.DataFrame(combo).to_csv(out/"local_global_combination.csv",index=False)

    # 用真实逐格 SFR 加权，避免背景单元数过多而掩盖成星区域的状态差异。
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    xx=np.arange(5)
    axes[0].bar(xx-.24,five.whole_field_coverage*100,width=.24,label="Whole field, all inputs")
    axes[0].bar(xx,five.local_true_sfr_coverage*100,width=.24,label="Local, all inputs, true-SFR weighted")
    axes[0].bar(xx+.24,five.local_thermo_true_sfr_coverage*100,width=.24,label="Local thermodynamics, true-SFR weighted")
    axes[0].set_xticks(xx,GROUPS);axes[0].set_ylim(0,105);axes[0].set_ylabel("Coverage [%]")
    axes[0].legend(fontsize=8,loc="lower left",bbox_to_anchor=(0,1.01),frameon=False)
    for mode,color in [("all_input","#0072b2"),("thermo","#d55e00")]:
        s=local[(local.fold=="Fiducial_actual_train")&(local.local_mode==mode)].sort_values("time")
        axes[1].plot(s.production_time,s.true_sfr_coverage*100,color=color,label=mode+", true-SFR weighted")
    axes[1].set_ylim(0,105);axes[1].set_ylabel("Fiducial local coverage [%]");axes[1].set_xlabel("Production time [Gyr]")
    axes[1].axvspan(7.5,9,color="grey",alpha=.12);axes[1].legend(fontsize=8)
    fig.savefig(out/"star_forming_state_coverage.png",dpi=160);plt.close(fig)

    five_report=five[["group","whole_field_coverage","local_true_sfr_coverage","local_thermo_true_sfr_coverage"]].copy()
    five_report.iloc[:,1:]=five_report.iloc[:,1:]*100
    five_report.columns=["留出运行","全场覆盖/%","局部全输入：真实SFR覆盖/%","局部热力学：真实SFR覆盖/%"]
    phase_report=phases[phases["mode"]=="all_input"][["stage","n","coverage","model_mae","baseline_mae","neighbor_closure_mae"]].copy()
    phase_report.coverage*=100
    phase_report.columns=["production时段/Gyr","帧数","全场覆盖/%","CNN MAE","baseline MAE","五近邻闭合 MAE"]
    local_report=lp[lp["mode"]=="all_input"][["stage","n","true_sfr_coverage","covered_sfr_factor3_conflict"]].copy()
    local_report.iloc[:,2:]*=100
    local_report.columns=["production时段/Gyr","抽样帧数","真实SFR局部覆盖/%","已覆盖SFR中三倍标签差异/%"]
    nearest_report=five[["group","global_strong_pairs","global_strong_pair_median_ratio","strong_near_sfr_fraction","strong_near_conflict_sfr_fraction"]].copy()
    nearest_report.iloc[:,3:]*=100
    nearest_report.columns=["运行","较近全场配对数","总SFR倍数差中位数","较近局部状态占SFR/%","较近且三倍差异占SFR/%"]
    probe_report=probe[probe.subset=="全部抽样帧"][["stage","n","model_mae","baseline_mae","local_neighbor_mae"]].copy()
    probe_report.columns=["时段/Gyr","相同抽样帧数","CNN MAE","baseline MAE","局部近邻求和 MAE"]
    # 低 SFR 晚期的 HR/LR 阈值质量比只作辅助，不等同于 SFR 修正倍数。
    h=fine[(fine.group=="disk_galaxy_fiducial")&(fine.time>=9.5625)]
    md=f"""# 五组盘星系 LR 状态、局部组合与 SFR 关系检查

本报告来自现有 v7、12×20 数据的实际计算。共 {method['unique_snapshots']} 个独立 snapshot；训练过采样重复 UID 已去除。正式 fiducial 的全场比较使用其实际 train split 中按时间抽取的参考库，覆盖全部 8429 个留出 snapshot。其余五折为数据分布的探索性比较，不是五个已训练模型的性能。

## 如何理解“覆盖”

距离在输入空间计算，不使用 SFR 选近邻。阈值来自参考运行自身变化的 95% 分位数，并排除同一运行 ±0.25 Gyr 的近邻帧。因此覆盖表示“距离不超过本次参考变化尺度”，不是同一物理状态的证明。局部比较固定 r、theta 和体积，避免拿不同空间位置作近邻。局部 SFR 权重来自原始 Data-Set-11 的保守块求和。

参考库按物理时间抽样，每组最多 192 帧；数据缺口产生重复时间锚点时会去重。全输入使用当前模型的 12 个通道；热力学对照只使用密度、温度和冷却时标。全输入与热力学距离的归一化、维数和阈值不同，覆盖率差不能解释成某个通道的因果贡献。

## 检查一：不同运行的 LR 状态覆盖

{table(five_report)}

fiducial 实际 train 参考库下的结果如下。MAE 单位为 Msun/yr；时段是描述性窗口，不是本次重新识别的 burst 起止时间。

{table(phase_report)}

全场平均会稀释少数内盘单元的差异。额外比较 r<10 kpc 且 abs(cos theta)<0.3 的内盘输入，详见 `coverage_summary.csv`、`fiducial_phase_summary.csv`。这些窗口只改变比较范围，没有改变原始粗粒化分辨率。

## 检查二：局部已见与全局组合

表中“局部真实SFR覆盖”先在每帧计算形成恒星贡献的覆盖比例，再对时间锚点平均；它不同于对所有帧 SFR 求和后计算的总权重比例。没有 SFR 的帧不进入有分母的比例平均。

{table(local_report)}

`local_global_combination.csv` 另统计了局部真实 SFR 覆盖超过 90%、而全场未覆盖的情形。如果某运行局部覆盖高而全局预测差，局部贡献的组合结构值得优先检验；如果成星区域的局部覆盖也低，局部求和模型本身不能保证解决外推。

进一步用每个粗单元的最近参考单元真实 SFR 作估计，再对 240 个单元求和。各单元可来自不同参考 snapshot；它是局部关系可组合性的数值探针，不是一套连贯气体场，也不是已训练的局部 CNN。它使用训练帧的逐格标签，监督信息比现有 CNN 的 snapshot 总量标签更丰富，因此不是控制全部条件后的模型排名。以下 MAE 在完全相同的抽样帧上计算，单位 Msun/yr：

{table(probe_report)}

{probe_text}

## 检查三：相似 LR 输入与不同 SFR

采用更严格的“距离不超过参考 p95 的一半”查看配对。全场标签倍数使用 0.01 Msun/yr 的稳定项；局部使用 1e-5 Msun/yr，并要求绝对差超过 1e-4 Msun/yr 才计三倍差异。

{table(nearest_report)}

这些配对提示当前平均量不能可靠保证成星率相近，但不能单凭它们证明 LR→SFR 数学上非唯一。有限距离、未完整覆盖的运行状态、距离度量对关键单元的敏感性，以及 HR 写出时气体和 SFR 的时间层差异，都应保留在解释中。具体局部配对保存在 `local_label_pairs.csv`，并保存 HR 密度 clumping 供解释。

例如一对相同空间位置、球半径约 22.42 kpc 的单元，Model0.75 的平均密度为 0.13585、温度约 10807 K；Fiducial4 对应近邻的密度为 0.13758、温度约 10891 K。两者真实逐粗格 SFR 分别为 2.35806 与 0 Msun/yr，HR 密度 clumping 分别为 6.32 与 2.43。但它们的冷却时标和速度仍不同，所以这个例子只提示未解析结构值得关注，不证明完整 LR 输入相同却产生不同标签。

对更接近的全场配对，Model0.75 的总 SFR 倍数差中位数约为 1.25，fiducial 约为 1.26。当前证据不足以将两者预测失败主要归因于无法恢复的多值映射；仍应优先检验模型是否有效利用了已有输入。

## 对改进泛化的实际帮助

1. 优先区分局部状态缺口与全局组合失败。前者需要独立运行补充相应状态，或加强可外推的物理关系；后者适合检验局部 SFR/闭合贡献预测后求和。当前结果不足以保证其中一种结构一定胜出。
2. 全局相似度依赖空间窗口。固定对全图做平均会让背景相近掩盖关键内盘变化；后续模型可以保留大尺度上下文，同时提高成星区域的局部监督和空间权重，而不是简单裁掉外围。
3. 可由 LR 计算的密度、温度阈值距离、冷却/自由落体时标比、流动压缩与局部梯度等，可作为物理特征消融。它们仍来自现有输入，不引入 HR 标签或未来量。
4. 对 fiducial，五近邻闭合参照使用训练标签、固定 k=5，不经过模型训练；它与 CNN 的差异说明当前网络和学习方式仍有改进空间。它并未解决全部低 SFR 外推，也不是经独立选择的新最优模型。
5. HR/LR 冷致密质量比在晚期抽查 {len(h)} 帧的中位数为 {h.hr_to_lr_eligible_mass_ratio.median():.3f}。这表明部分后期状态的阈值质量本身已能在 LR 上保留；不能默认继续施加与前期相同的大闭合修正。冷致密质量比并不等于 SFR 修正倍数，时标仍然重要。
6. 本轮不支持仅凭高/低 SFR 标签缺失就认定无法泛化；也没有证据保证单独调高 burst 的 loss 权重会同时修复后期高估。应根据覆盖和局部配对设计下一轮模型/特征消融。

更具体的下一轮顺序：保留当前全局 CNN 作对照，尝试输出每个粗单元的 signed closure 并保守求和，同时保留邻域或多尺度上下文。先只用 snapshot 总量损失检验结构，再加入原始 Data-Set-11 的块求和局部监督，以区分结构和监督信息的作用。然后在相同 split 下加入 LR 可计算的物理特征，最后才单独改变高/低 SFR 的损失权重。HR 冷致密比例或 clumping 可以作为辅助预测目标进行消融，不能直接作为部署输入。不要仅裁剪到 10 kpc：Fiducial4/7 和 Model0.75 的部分时期确实存在重要的更外侧成星贡献。

## 产物与可复现性

- [真实成星区域覆盖图](star_forming_state_coverage.png)
- [fiducial 气体状态与预测](fiducial_state_and_prediction.png)
- [五运行覆盖表](five_run_interpretation.csv)
- [正式 fiducial 分阶段表](fiducial_phase_summary.csv)
- [正式 fiducial 局部分阶段表](fiducial_local_phase_summary.csv)
- [逐格近邻标签差异](local_label_pairs.csv)
- [完整方法参数](method.json)

原始局部标签逐帧核对求和等于现有 cache 总 Y；距离实现已用独立 NumPy 枚举核对。未进行新训练、参数选择或训练代码修改。HR 结构字段仅作解释，不能进入部署输入。详细运行方法见 `VisualScripts/DataAnalysis/check_disk_lr_generalization.md`。
"""
    (out/"report.md").write_text(md,encoding="utf-8")
    print(table(five_report));print(table(phase_report));print(table(local_report));print(table(nearest_report))


if __name__=="__main__":
    main()
