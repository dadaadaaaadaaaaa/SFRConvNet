# 五组盘星系的局部成星时标诊断

入口：`analyze_disk_sf_timescales.py`。只读已有二维 fine 和逐块真实 SFR 产品，不修改训练数据、算子或实验参数。

## 判据与含义

复用 `raw_data_processing/macer_sfr_operator.py` 的两套算子：

\[
\tau_\star=\max[\tau_{\rm cool},\min(\tau_{\rm ff},\tau_{\rm rot})],\qquad
\tau_{\rm ff}=\sqrt{3\pi/(32G\rho)}.
\]

- 冷却：`tau_cool >= min(tau_ff,tau_rot)`；
- 自由落体：冷却更快且 `tau_ff <= tau_rot`；
- 旋转：冷却更快且 `tau_rot < tau_ff`。

阈值是 `T < 4e4 K`、`1.02*rho > 1`。阈值外记为 0，不能用全域热 CGM 的平均冷却时间推断实际成星分支。相等时的优先级沿用现有算子。

`data_compatible` 使用 `2*pi*sqrt(r**3/(G*M_BH))`；`paper_physical` 使用 `2*pi*sqrt(r/abs(a_r))`。前者是已记录的 enclosed-mass 近似，后者是已有物理对照；都不能直接当成由气体 `v_phi` 测量的实际轨道周期。两者 `cstar` 也不同，分别为 0.01 与 0.1。

这里的主导指局部成星处方当前采用哪个分支。`tau_star` 也不是气体耗尽时间；连续极限的局部耗尽时间约为 `tau_star/cstar`。本分析不能单独确定整星系淬火、反馈、供气或 SFR 振荡的因果机制。

## 权重与真实标签边界

本机配置的原始 E: 数据盘不可用，现有 fine 未保存 HR 逐格 Data-Set-11。因此主统计明确使用保存的 `data_compatible/cell_sfr_baseline` 作为**重建 HR SFR 权重**，不是直接测得的真实逐格 SFR。两种旋转定义使用同一权重，避免把权重变化误解为分类变化；另存各模式自己的 SFR 权重和冷致密气体质量权重结果。

每帧复算两套算子并与保存的 HR baseline 对照；用原 Data-Set-11 的标量和 `coarse_rh_12x20_sfr_v7_cell_v1` 的真实逐块标签核验总误差及逐块 L1 误差。总 SFR 一致不证明每个 HR 单元一致。

额外提供真实分支占比的严格上下界：若一个粗块的所有 HR 单元都属于分支 k，则其真实 SFR 全部进入下界；若任一 HR 单元属于 k，则其真实 SFR 全部进入上界。0 类也参与混合判断，不假设输出前后阈值完全相同。边界常较宽，但不需要块内真实标签分布或重建正确的假设。

## 来源与积分

- Fiducial/Fiducial4/Fiducial7：`fine_sfr_v7`，已有修正后的绝对时间。
- Model0.75/Model1.1：`fine_sfr_v6`，原时间映射未受上述 restart 修正影响。
- 只分析现有去初始化产品，torus 完全不参与。
- 时间、原始编号、来源 SHA256、parent HR UID 和 fine 目录保存在逐帧 Parquet。SHA256 来自产品已有 metadata；本次不能重新访问原始盘核对散列。
- 每个运行独立以真实时间梯形积分；超过 `1.5 * nominal cadence` 的缺口不插值，报告实际覆盖时长与覆盖率。积分形成质量只是这些已覆盖时间段内的 SFR 积分，不等于包含缺帧区间的完整累计恒星质量。
- 零 SFR 帧保留，零权重区域不伪造时标或分支比例。图中使用散点保留缺口。
- 分位数先逐帧对 HR 成星单元按重建 SFR 加权；时间图中的分位数不是全域气体平均，也不是全程合并样本的分位数。

## 运行与验证

已有输出目录会被拒绝覆盖。建议使用具备 NumPy、h5py、pandas、pyarrow、Matplotlib 的项目环境：

```powershell
python -m unittest discover -s tests -p test_disk_sf_timescales.py -v
python -m unittest discover -s tests -p test_macer_sfr_operator.py -v
python VisualScripts/DataAnalysis/analyze_disk_sf_timescales.py --output-dir tmp/sf_timescales_smoke --max-snapshots 3
python VisualScripts/DataAnalysis/analyze_disk_sf_timescales.py --output-dir VisualScripts/DataAnalysis/outputs/sf_timescales_20261111
# 完整统计已有时，仅更新派生摘要与图，逐帧统计不改写
python VisualScripts/DataAnalysis/analyze_disk_sf_timescales.py --output-dir VisualScripts/DataAnalysis/outputs/sf_timescales_20261111 --reuse-snapshots
```

smoke 仅验证少量真实数据 I/O，不从稀疏帧生成时间积分结论。完整运行产生逐运行 Parquet、`summary.csv`、`phase_summary.csv`、旋转分支示例、来源说明 `method.json` 及三张 PNG。`frame_median_*` 是非零 SFR 帧内加权中位数在各帧之间的中位数，不是全程合并分位数。它们是研究诊断产品，不是新增训练结果，不登记到 `Train_history.md`。
