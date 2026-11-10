# Exp2：C、C_ref 与预测闭合的分布和关系

`compare_closure_reference.py` 汇总留出 Fiducial、Model0.75、Fiducial7 的正式预测。
复用 `compare_fiducial_domains.py` 已核验的模型清单，Fiducial 的多个 UNet 版本全部保留，
CNN 同时列作对照；不按 out 误差选模型。

```powershell
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_closure_reference.py --output-dir VisualScripts/ResultAnalysis/outputs/新的闭合比较目录
C:/Users/augus/miniconda3/python.exe -m unittest tests.test_closure_reference tests.test_fiducial_domains tests.test_analyze_sfr_unet -q
```

输出目录必须不存在。脚本只读取本项目可信 cache/PT/CSV，不加载权重，不重新训练、推理或校准。

## 定义与单位

- `C=Y-B`：snapshot 的真实总闭合。
- `Chat=P-B`：最终预测的有效总闭合。ReLU 模型使用投影后的最终结果，不与投影前 candidate correction 混淆。
- `Cref`：在当前模型实际 train 中，满足匹配条件的输入最近帧的真实闭合。
- `Cref5`：最多五个最近候选的真实闭合中位数，作为单近邻敏感性对照。
- `Delta=C-Cref`，`R=Chat-Cref`，因此 `P-Y=Chat-C=R-Delta`。
- 全部数值在物理空间、单位 Msun/yr；signed-asinh 只用于原网络输出参数化。
- 原标签与逐格量已包含体积；统计总闭合不再乘体积。

由于误差和 Delta 共用 C，相关性不能单独证明模型受训练近邻牵引。
脚本同时报告参考预测 `B+Cref` 的误差、模型与真值及参考的距离，
以及 `alpha=(Chat-Cref)/(C-Cref)` 的中位数。
alpha=0表示等于参考，alpha=1表示等于真值；它只是描述量。
分母小于 `max(0.1,0.05*median(abs(C_train)))` 的样本不进入alpha统计，
但仍保留在所有预测误差、分布和逐帧表中。

## 训练与各星系的统计口径

每折分别读取 train/val/test/out；train 同时报告去重 UID 的分布和按 cache 重复次数加权的分布。
该加权描述 cache 的过采样构成，不是独立星系重复实验，也不是实际每个epoch的访问日志。
五组星系的总体真实闭合是三套当前 UNet cache 全 split 的 UID 并集；
合并前检查相同 UID 的 parent、运行、时间、Y/B/C 一致，拒绝同 parent 的不同契约静默混合。
这个并集不等于未缺帧的完整原始模拟。

正式 UNet 仅导出 test/out 预测。这里的训练预测性能用已知运行独立 UID 的 test 作对照，
不把这些 test 预测称为 train 预测；没有假造未导出的训练集 Chat。
test 仍可能与 train 时间相邻，不能当作严格时间外推。

历史 Model0.75 CNN53 的训练集合与新 UNet 不同，所以用它自己的 train 独立检索 Cref；
该族单列为 `model075_cnn_history`，不混入五组星系的当前 cache 并集。
CNN 目录只导出 out 预测，未伪造其已知运行 test 结果。

## 近邻与限制

每折使用自己的标准化输入。参考标尺由 train 每运行最多192个等时间锚点计算，
减去共同空间模板，再除以参考通道全局标准差；欧氏距离除特征数后开平方。
最近邻库为全部去重 train，排除同运行时间差小于0.25 Gyr的候选。

匹配不使用查询 Y/C，输出四种诊断：

|method|约束|
|---|---|
|x|仅输入距离|
|b05|abs(B-B_ref)<=max(0.10,0.05*abs(B))|
|b10|abs(B-B_ref)<=max(0.25,0.10*abs(B))|
|b20|abs(B-B_ref)<=max(0.50,0.20*abs(B))|

没有候选时 ref_uid 为空、Cref 为 NaN，原预测误差仍计入无条件统计。
`d_le_05` 是距离<=0.5的诊断子集，不是校准后的覆盖概率或物理域内证明。
baseline约束仅用于解释，不进入模型输入或超参数选择。
相近粗粒化张量不等于相同完整状态；配对不能直接识别缺失物理变量或因果机制。

主统计保留负闭合和所有尖峰；图使用 symlog 显示完整范围。
0.25/0.5 Gyr分组时间块bootstrap描述给定预测的时间块稳定性；
不覆盖训练种子和不同物理运行的不确定性。

## 验证与产物

检查正式完成状态、train/val/test/out cache 和 norm/metadata 的 SHA256、
manifest 的 UID/parent 隔离、预测身份、Y/B对齐、地图求和、candidate/最终ReLU和闭合恒等式。
CNN 检查自己的训练族、物理预测与导出闭合一致；旧 CSV 的 B 为float32，
统计使用同源 cache 的高精度 B。来源存于 `method.json`。

- `galaxy_true_closure.csv`：五组星系当前 cache 并集的真实闭合分位数、范围、均值和负值比例。
- `training_closure.csv`：各折、各训练星系及合并训练分布，包含去重与过采样权重两种口径。
- `closure_distributions.csv`：各 split/星系真实 C 与各模型预测 Chat 分布。
- `closure_relations.csv`：预测误差、闭合恢复比例、参考误差、Delta/R与alpha。
  无参考样本仍进入无条件C/Chat/误差；C_matched_mean、Chat_matched_mean和Bias_matched才与Cref使用同一有效样本。
- `delta_bins.csv`：固定物理闭合差分箱，便于比较正负偏移。
- `block_intervals.csv`：按时间块重采样的偏差、Delta/R及描述斜率区间。
- `*_references.csv.gz`、`predictions_and_references.csv.gz`：本地逐帧配对明细，不提交版本控制。
- `galaxy_closure_cdf.png`、`closure_reference_distributions.png`、`closure_reference_relations.png`：真实闭合与预测关系图。
- `training_and_prediction_closure.png`：训练真实C、留出C、参考Cref和预测Chat的范围与分位数。

实际三折分析结论见 `outputs/closure_comparison_20261111_verified/report.md`。
