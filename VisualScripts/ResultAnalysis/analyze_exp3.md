# `analyze_exp3.py` 使用说明

## 功能

该脚本只分析新 Exp3 的 torus test-only 数据，并验证 cache metadata 符合：

```text
split_strategy = random_snapshot_75_25_torus_test
test_groups = [disk_galaxy_torus_100yr]
objective_mode = direct_sfr 或 signed_closure
```

分析模式从结果目录的 `hyperparameters.json` 和 checkpoint 自动读取，不根据目录名
猜测，也不需要在 notebook 中手动切换。`direct_sfr` 直接反变换网络输出；
`signed_closure` 则按 checkpoint 契约计算 `macer_sfr_baseline + closure_hat_physical`。
两种模式最终都在同一物理 SFR 空间计算指标并绘图。

统计量严格限定为：

- 总体物理 SFR RMSE；
- 总体物理 SFR MAE；
- 总体物理 SFR R²；
- 全时间段积分 SFR 低估百分比。

全时间段指标定义为：

\[
100\frac{\int \mathrm{SFR}_{true}dt-\int \mathrm{SFR}_{pred}dt}
{\int \mathrm{SFR}_{true}dt}.
\]

正值表示模型低估，负值表示模型高估。积分按真实时间排序并用梯形公式计算；被排除坏帧形成的时间缺口不会被当成等间隔帧重新编号。

## 命令

```powershell
python VisualScripts/ResultAnalysis/analyze_exp3.py `
  --result-dir Results/<Exp3_Direct_or_Closure_result_directory> `
  --device auto `
  --batch-size 256
```

对应 notebook 为 `Results_stat_Exp3.ipynb`。

## 输出

结果保存在原训练结果目录：

- `torus_test_predictions.csv`；
- `metrics_exp3.csv`；
- `exp3_analysis_metadata.json`；
- `figures_exp3/scatter_torus_test.png`；
- `figures_exp3/error_dist_torus_test.png`；
- `figures_exp3/sfr_history_torus_test.png`；
- `figures_exp3/cum_sfr_torus_test.png`。

图表与 Exp2 测试域保持相同功能。CSV 不额外加入相对误差分位数、负预测率、baseline 指标或分段统计。
