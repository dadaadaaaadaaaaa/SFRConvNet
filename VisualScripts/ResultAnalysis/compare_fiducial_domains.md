# Exp2：CNN 与 UNet 的分布分层比较

入口 `compare_fiducial_domains.py` 读取已完成的 v7、12×20 Fiducial 留出折，不重新训练或推理。
固定结果目录在脚本顶部明确列出：CNN Trial43/Trial7、原始 UNet seed42、首次 ReLU Candidate1 seed42，
以及随后同权重的 seed42/43/44。全部原始输出进入统计，不裁剪负值或剔除尖峰。

2026-11-10新增 `--case model075` 和 `--case fiducial7`；默认仍为原来的 `fiducial`。
2026-11-11新增 `--case fiducial4`，读取用户指定的Final UNet。该折未配置CNN，配对bootstrap以MACER-LR Baseline为参照，`reference`列明确记录。
各 case 的来源目录在 `CASES` 中明确列出，训练阈值与输入参考按各折实际 train 独立建立，不能复用 Fiducial 阈值。

```powershell
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_fiducial_domains.py --output-dir VisualScripts/ResultAnalysis/outputs/新的独立目录
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_fiducial_domains.py --case model075 --output-dir VisualScripts/ResultAnalysis/outputs/新的Model075目录
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_fiducial_domains.py --case fiducial7 --output-dir VisualScripts/ResultAnalysis/outputs/新的Fiducial7目录
C:/Users/augus/miniconda3/python.exe -m unittest tests.test_fiducial_domains tests.test_analyze_sfr_unet -q
```

Python 需有 PyTorch、NumPy、pandas、Matplotlib。示例解释器是本机实际验证路径，不是可移植的环境约定。
输出目录必须不存在。仅加载本项目可信 PT；需要对应 scalar train/out、metadata、norm 和正式模型预测。

## 两种分布定义

- SFR 分层：按 `sample_uid` 去重实际训练集；主结果用训练 SFR 的 min–max，补充 p01–p99、p05–p95。
  端点算范围内，分别保留低端域外、高端域外与两者合并。阈值完全来自训练集。
  这是标签范围诊断，不能直接当成物理输入分布或部署时的 OOD 检测器。
- 输入分层：复用 `check_disk_lr_generalization.py` 的距离实现，重新从实际 train 建立参考。
  每个训练运行最多192个物理时间锚点；每组最多64个校准锚点。
  校准时排除同一运行相隔不足0.25 Gyr的近邻。以校准近邻距离 p95 分界，并补充 p90/p99。
  比较全场12通道、r<10 kpc 且 abs(cos(theta))<0.3 的内盘12通道、r<30 kpc 的热力学三通道。
  距离、标准化及阈值不读取留出标签；不使用 val/test 或其他运行的全 split 作参考。

输入分层是有限参考库、指定表示与欧氏距离下的覆盖代理；它不是高维物理支持集的严格证明。
脚本也输出标签分层与全场输入分层的交叉统计。没有预设“burst 一定是高 SFR 域外”。

## 契约及统计

检查训练/留出运行、完整 cache SHA256、归一化/metadata、UNet 正式完成状态、manifest 身份隔离、
逐格求和与 B+C_hat；各模型按 UID 对齐并检查 parent、时间、Y/B。
旧 CNN CSV 的 baseline 为 float32 精度，恒等式检查采用相应容差。
CNN 元数据以实际 hyperparameters.json 为准，不能用 Train_history 中相邻命令替代它。

Fiducial7 使用相同v7 metadata的CNN Trial68。Model0.75没有相同新训练缓存的已登记CNN：
仅该case显式允许已登记的Trial53作为历史对照，并核验out UID/parent、时间、Y/B、坐标、
撤销各自标准化后的输入均一致；同时核验旧train与out隔离，存档新旧训练UID交集及差集。
这不等于训练数据完全控制一致。其他case继续拒绝不相同的metadata，不提供通用静默跳过开关。

全部 SFR 乘0.025转 Msun/yr；统计针对 snapshot 总量，绝不再次乘体积。
输出 MAE/RMSE/Bias/R²、非零真值中位相对误差和负预测比例。
低 SFR 子集的 R² 对窄真值范围敏感，不能只凭这个指标比较实用误差。

MAE 差值另用0.25/0.5 Gyr固定时间块做2000次配对 bootstrap，两个模型共享整块抽样。
区间仅描述给定模型、单一留出轨迹的时间块不确定性，不涵盖训练种子或运行间不确定性。
不把8429帧视为8429个独立模拟。少于5个非空块不生成置信区间。

时间均为 snapshot 绝对 Gyr，未减0.5625 Gyr。t<9、9≤t<10、t≥10只作描述性窗口，
不是论文核验的吸积事件边界。分层可不连续，不能把“首次越界至末帧”全部算成同一类别。
图中缺口断线并着灰；不计算或声称缺失时段的积分。连续段清单按中位间隔5倍切断大缺口。

## 产物

- `metrics.csv`：全部模型、分布定义、阈值和交叉分层指标。
- `predictions_and_domains.csv`：逐帧预测与标签/输入分层，本地生成，不纳入版本控制。
- `paired_block_bootstrap.csv`：相对 CNN43 的 MAE 差及区间。
- 新case的bootstrap参照为各自CNN，`reference`列明确其身份；少于5个时间块或没有域外样本时留空。
- `integrated_metrics.csv`：在同一实际观测区间上，对所有模型与baseline做梯形积分，跨大缺口的区间剔除。
- `train_sfr_by_run.csv`、`time_segments.csv`：训练运行分布及连续类别段。
- `method.json`：来源目录、SHA256、实际模型参数、阈值、单位和解释边界。
- `domain_history.png`、`domain_metrics.png`：完整轨迹和主要误差对比。

2026-11-08实际数据的解释见 `outputs/fiducial_domains_20261108/report.md`。
当前比较同时改变结构、监督信息、损失、容量与优化设置，不能称为只改变 UNet 架构的消融。
后续两折报告在 `outputs/model075_domains_20261110_final/report.md` 与
`outputs/fiducial7_domains_20261110_final/report.md`，合并解读见 `outputs/exp2_domains_20261110_report.md`。
