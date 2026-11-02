# 闭合项幅值与泛化误差审计

`analyze_closure_regimes.py` 对已完成的二维 signed-closure 实验进行只读推理和统计，不启动训练，不修改 cache、checkpoint 或原结果。默认输出到 `outputs/closure_regimes_20261025/`。

## 运行

在项目根目录使用包含 PyTorch、NumPy、pandas、Matplotlib 的环境：

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\conda_envs\physics_ai\python.exe -B VisualScripts\ResultAnalysis\analyze_closure_regimes.py --output-dir VisualScripts\ResultAnalysis\outputs\closure_regimes_new
```

已完成推理后，可核对来源指纹并重新生成统计和图：

```powershell
& D:\conda_envs\physics_ai\python.exe -B VisualScripts\ResultAnalysis\analyze_closure_regimes.py --reuse-predictions
```

默认拒绝覆盖已有输出目录；`--reuse-predictions` 允许更新本脚本的分析产物。复用前检查 checkpoint、cache、metadata、hyperparameters 的路径、文件大小与修改时间；这不是内容哈希。实验与 cache 清单在脚本 `MODELS`、`CACHES` 中明确列出，不能把文件存在视为正式训练有效性的独立证据。

## 统计约定

- 全部物理 SFR、baseline 和 closure 乘以 `0.025`，单位为 `Msun/yr`。`C=Y-B`、`Chat=Yhat-B`，因此两种空间的物理误差相同。
- 去除训练 cache 内因 oversampling 重复的 sample UID，再统计原始样本分布和训练分位数。去重分布不是每个训练 epoch 的实际曝光分布。
- 核验有限值、闭合恒等式、重复 UID 的物理量一致性，以及 train/val/test/out 的 sample UID 和 parent HR UID 隔离。UID 隔离不等同于时间独立。
- 固定幅值箱为 `[0,1)、[1,3)、[3,10)、[10,20)、[20,∞)`；另按各模型去重训练集的 50%、90%、99% 分位数分箱。前者是事后描述阈值，不是预先指定的优化权重。
- closure 相对误差只在 `|C|>1e-3` 上计算，SFR 相对误差只在 `Y>1e-4` 上计算；输出保留有效分母样本数。不能用该过滤后的相对误差评价零 SFR 行为。
- baseline 对应 `Chat=0`；`MAE skill = 1 - sum(|Chat-C|)/sum(|C|)`，MSE skill 同理。R² 在各评价子集自身均值下计算，Y 与 C 的 R² 不同。
- 主实验连续 held-out 历史采用原始归档 CSV，按身份和物理量对齐并与当前 checkpoint 推理交叉核验。跨模型表使用当前推理结果；不宣称逐位复现。
- 主图横轴为 `t_production = t_abs - 0.5625 Gyr`；采样间隔 12.5 Myr。`t>=6 Gyr` 沿用已有 burst 分析窗口，不是新识别的独立 burst 事件。
- 积分使用梯形法；单条连续轨迹若有超过中位间隔 1.5 倍的缺口则拒绝积分。随机 test/val 不作连续时间积分。积分 SFR 表示该已给定轨迹上的形成质量，不是在线演化后的质量，也未扣除恒星质量返还。
- 时间块 bootstrap 使用 0.25、0.5、1 Gyr 块宽、600 次重采样；仅作单轨迹敏感性分析，不能充当多次独立 burst 或多模型随机种子的误差条。

## 产物

- `report.md`：中文研究解释和局限。
- 三组 PNG/SVG：held-out 时间演化、幅值分层、跨 cache/model 对照。
- `cache_distributions.csv`：全体和逐运行的训练分布。
- `closure_bins.csv`、`closure_bins_by_group.csv`、`train_quantile_bins.csv`：幅值条件误差。
- `overall_metrics.csv`、`heldout_phases.csv`：全体、逐运行、时间阶段结果。
- `burst_definition_sensitivity.csv`：旧 burst 检测定义的九组参数敏感性。
- `*_predictions.csv`、`*_source.json`、`audit.json`：推理数据、来源指纹和核验记录。

## 本次验证（2026-10-25）

完整推理已执行；随后运行上述 `--reuse-predictions` 命令成功，所有脚本断言通过，三张 PNG 已人工检查。原 burst MAE、积分偏差和关键事件时刻复现通过。当前推理相对归档预测的最大逐点差异为 0.02660 Msun/yr，最大相对差异约 0.199%；总体 MAE 相对差异约 0.000386%。历史运行环境不完整，不将差异归因于未经证实的单一设置。

```powershell
& D:\conda_envs\physics_ai\python.exe -B -m unittest tests.test_signed_closure_result_analysis tests.test_signed_closure_training -q
```

结果：13 项测试通过。未进行新训练、全套数据重建、跨种子统计或 GIZMO 在线耦合测试。
