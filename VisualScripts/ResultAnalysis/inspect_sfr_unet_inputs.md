# 异常snapshot及cache输入检查

入口：`Results_stat_Exp2_SFRUNet.ipynb`末尾的“异常帧及标准化输入检查”。本模块不读raw/HDF，
不重新标准化、不加载权重推理、不修改数据。支持Exp2 out；需要结果目录与对应train/out标量PT。
不需要逐格sidecar，因为预测文件已经包含逐格标签与预测。

## notebook配置

```python
INSPECT_INPUTS = True
INPUT_CACHE_DIR = PROJECT_ROOT / ".cache"
INPUT_TIMES = [8.99375, 9.405, 12.44125]
INPUT_NEIGHBORS = 5
INPUT_CELLS = None
INPUT_CHANNELS = None
```

默认关闭，避免普通结果分析额外加载较大cache。启用后运行新增配置、计算、展示三个cell。
INPUT_CELLS=None时每个事件采用该模型中心帧绝对SFR误差最大的格子；可显式设置
`[(6,0),(9,2),(9,1)]`以便不同模型用相同位置对照。索引顺序(theta,r)，从0开始。
INPUT_CHANNELS=None表示全部输入通道，也可以先用`["density","cooling_time"]`。
目标按时间匹配，默认容差1e-6 Gyr，找不到则报错；随后以UID确认cache对应关系。
邻帧是时间排序后前后各5帧，保留实际时间差，不假定缺口内存在输出。

## 函数调用

```python
from VisualScripts.ResultAnalysis.inspect_sfr_unet_inputs import inspect_inputs
review = inspect_inputs(RESULT_DIR, INPUT_CACHE_DIR, times=INPUT_TIMES,
                        neighbors=5, time_tolerance=1e-6,
                        cells=None, channels=None, plots=True)
```

output_dir默认结果下独立input_review随机目录，显式指定时也必须不存在。
train/out SHA256必须与训练contract一致，UID/parent顺序与manifest一致；检查通道、变换、时间和shape。
PT只用于可信自有文件。cache中x已是模型使用的标准化输入，不能再套归一化或误读为物理密度/温度。

## 输出与含义

- frames.csv：事件、实际时间、相对帧号、时间差、UID/parent、cache行号、选定格子、该格坐标和总Y/B/预测。
  cache行号不是原始snapshot编号，本模块不猜测raw文件名。
- input_points.csv：每个邻帧选定格子各通道值，以及训练同位置、3×3局部邻域、全局min/p01/p50/p99/max。
  同位置/局部percentile为经验秩（含并列）；全局percentile留空以避免每个查询重复扫描全训练张量。
- input_fields.csv：邻帧整个地图各通道的有限值极值、非有限值数。
- nearest_train.csv：中心帧全部通道同位置3×3 patch在训练集中的前5个近邻，给出UID、cache行、时间、RMS距离。
  边界patch截短；这是标准化输入的欧氏参考，不是物理相似性的充分条件或OOD概率。
- event*/通道_maps.png：中心与相邻前后帧共享色标，红叉为选定格子；非有限值遮罩，其数量见CSV。
- event*/input_history.png：选定格子的邻帧输入曲线，背景为训练同位置p01–p99。
- inspection.json：来源摘要、参数与解释边界。

参考统计仅使用train，按sample_uid去重以避免oversample重复影响经验分位；不会用out/val更新统计。
超过训练分位不自动等于错误，所有分量分别正常也不能证明联合状态常见。地图是格子索引，不是等体积网格。
默认格子由预测误差选择用于事后检查，不是输入因果归因。当前仅单模型入口，不自动聚合九个模型。
原始物理量、HDF字段与处理链追溯属于后续功能，此次未实现。

## 验证

`python -m unittest tests.test_sfr_unet_input_inspection tests.test_analyze_sfr_unet -q`

覆盖非排序时间的邻帧选择、时间不匹配/重复拒绝、分位与非有限参考、局部距离、
Exp2小型训练导出→cache输入检查及绘图、cache摘要篡改拒绝、旧分析与notebook兼容。


## 2026-11-07：完整输入默认存档

每次检查除CSV/PNG外，默认写入`input_windows.npz`和`artifacts.json`，无需额外开关。
NPZ包含所有输入通道的标准化数组、坐标、事件、时间、相对帧号、cache行与UID；
即使只绘制部分通道，存档仍保留全部通道。重叠时间窗口按事件分别保留条目。
使用`np.load(path, allow_pickle=False)`读取，x轴顺序为(entry,channel,theta,r)。
这不是原始物理场，也不是新训练cache。`inspection.json`保存其SHA256和选择的绘图通道，
`artifacts.json`列出本次已生成的文件摘要。输出目录不复用，不覆盖旧检查。
