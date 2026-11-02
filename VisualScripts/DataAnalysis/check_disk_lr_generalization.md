# 五组盘星系的 LR 状态与泛化检查

`check_disk_lr_generalization.py` 对统一 v7、12×20 数据进行三项研究检查：

1. 整幅 LR 输入与其他运行的状态覆盖关系；
2. 同一空间位置的局部状态覆盖，以及实际贡献 SFR 的区域是否被覆盖；
3. 输入近邻之间的真实总 SFR、逐粗单元 SFR 差异。

本脚本使用现有数据，不训练新模型。探索性五折比较不能冒充五个已经训练完成的 Exp2。

## 运行

在项目根目录、包含 PyTorch、NumPy、pandas、Matplotlib、h5py、pyhdf 的环境中运行：

```powershell
python VisualScripts/DataAnalysis/check_disk_lr_generalization.py
python VisualScripts/DataAnalysis/summarize_disk_lr_generalization.py
```

主要参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--samples-per-run` | `192` | 按物理时间等间隔寻找最近可用帧；缺口导致重复选择时去重，因此实际可能少于 192 |
| `--output-dir` | `VisualScripts/DataAnalysis/outputs/lr_state_generalization_20261025` | 独立分析目录 |
| `--refresh` | 关闭 | 允许更新本脚本自己已经生成、且 cache 前缀匹配的分析目录 |

Cache 前缀为 `Exp2_Rh12x20_FiveRuns_Closure_v7_LOO_disk_galaxy_fiducial_`。
路径与字段名单在脚本常量中明确给出；原始 HDF4 路径读取 `.config`。
更新数据来源时应使用新的输出目录，并核对版本、单位和字段。
原始局部标签检查需要挂载 RawData 所在磁盘。第二条命令只读取已经生成的分析表，汇总中文报告与真实成星区域覆盖图，不再次读取大型 cache 或原始 snapshot。

## 方法与解释边界

- 从 train/val/test/out 合并独立 UID，共同表示五组已纳入本次 cache 的数据；oversampling 重复帧只计一次。
- 五折探索性比较以另四组的所有 split 作为参考，用于描述运行之间的关系。它不用于训练或报告训练性能。
- fiducial 另做一次严格使用现有 `train` 身份的参考比较，并关联已保存正式模型的预测。
- 参考库按运行分组、按物理时间抽样，避免三个高输出频率运行完全支配近邻数量。
- 距离计算不读取 SFR。找到近邻后才比较 Data-Set-11 标签。
- 全场距离采用参考库的逐通道标准化 RMS。全输入使用 12 个模型通道；热力学对照只使用 density、temperature、cooling_time。
- 空间窗口包括全网格、半径小于 30 kpc，以及半径小于 10 kpc 且 `abs(cos(theta))<0.3` 的内盘。内盘窗口为物理尺度检查，不声称包含全部真实成星区域。
- 同位置局部比较采用逐单元参考均值与标准差；标准差下限为对应通道全局标准差的 5%，防止近乎静态的单元被噪声支配。
- 阈值是参考运行自身近邻距离的 95% 分位数。建立阈值时排除同一运行 ±0.25 Gyr 的相邻帧。阈值表示本次参考变化尺度，不是物理状态等价性的证明。
- 五近邻标签中位数仅是无需训练的参照；近邻优先来自相隔至少 0.25 Gyr 的片段，但候选池有限，不能保证总能获得五个独立片段。
- 全场标签倍数加 `0.01 Msun/yr` 稳定分母；局部标签倍数加 `1e-5 Msun/yr`。局部三倍差异还要求绝对差大于 `1e-4 Msun/yr`。
- 局部真值直接读取原始 Data-Set-11，以 6×14 个 fine cells 求和组成每个 12×20 单元，不再乘体积；并核对逐粗格求和等于 cache 的总 Y。
- 原始 HR 冷致密质量、密度 clumping 仅用于解释近邻差异，不是部署输入。成星阈值质量并不等于 SFR，还存在时标等因素。
- 局部检查使用抽样帧。全体 snapshot 的物理摘要来自 cache；fiducial 的全场检查覆盖全部 8429 个留出样本。

## 主要产物

- `coverage_summary.csv`：各运行、各输入表示的全场覆盖。
- `whole_field_neighbors.csv`：逐 snapshot 的最近参考状态、距离、标签差异与五近邻结果。
- `local_state_coverage.csv`：按单元数、气体质量、真实 SFR 贡献加权的局部覆盖率。
- `local_label_pairs.csv`：贡献超过当前 snapshot 总 SFR 的 1%、有参考近邻但标签相差至少三倍的局部配对。
- `snapshot_physical_summary.csv`：全部独立样本的 LR 物理摘要。
- `hr_labels_diagnostic_only.npz`：抽样帧的逐粗单元真值和 HR 结构摘要；禁止直接作为训练输入。
- `local_label_sources.csv`：局部标签对应的原始文件路径。
- `hr_structure_context.csv`：额外小样本的 HR/LR 冷致密质量比较。
- `fiducial_phase_summary.csv`：正式 fiducial 模型的分阶段误差与气体状态覆盖。
- PNG 图和 `method.json`：直观比较及方法参数。

数值检查包括：全场和局部距离对独立 NumPy 枚举的比较；抽样原始局部标签归约与 cache 总真值的一致性；原始标签非负。相似输入仍有不同标签，只能提示表示、状态覆盖或未解析结构值得研究，不能单凭近邻距离证明 LR→SFR 在数学上非唯一。
