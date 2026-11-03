# SFRUNet 结果分析与 Exp2 notebook

`analyze_sfr_unet.py` 直接读取正式训练已保存的物理预测文件；不加载网络、不访问 raw/HDF5/cache。
对应交互入口为 `Results_stat_Exp2_SFRUNet.ipynb`，每个代码 cell 均有中文总注释。
不修改旧 `analysis_utils.py/analyze_exp2.py` 和旧 notebook。

## 下载和运行

从服务器下载同一个结果目录的：

- `summary.json`、`contract.json`、`split_manifest.json`；
- `test_predictions.pt`；Exp2 还需 `out_predictions.pt`；
- 可选 `history.json`，用于训练/验证曲线。

不需要 best.pt 或 last.pt。只加载自己生成的可信 PyTorch 文件。

```powershell
python VisualScripts/ResultAnalysis/analyze_sfr_unet.py --result-dir Results/实际SFRUNet结果目录
```

也可打开 notebook，修改配置 cell 中的 `RESULT_DIR`，按顺序执行。
默认结果存到训练目录下新的 `analysis_sfr_unet_<北京时间>_<随机后缀>`；重复分析不会覆盖旧结果。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--result-dir` | 必填 | 已完成正式 SFRUNet 训练目录 |
| `--output-dir` | 自动新目录 | 显式输出目录，必须不存在 |
| `--plots/--no-plots` | 绘图 | 是否生成PNG；CSV/JSON始终生成 |
| `--gap-factor` | 5 | 相邻时间差超过该运行中位间隔的倍数时，积分不跨越该缺口；须>1 |
| `--burst-window` | 无 | 两个时间端点Gyr，属于snapshot时间+offset坐标，必须在out覆盖范围内 |
| `--time-offset` | 0 | burst分析时间平移；不自动套用松弛阶段长度 |
| `--burst-threshold` | 无 | 正阈值Msun/yr，启用事件开始/结束测量，需同时给窗口 |

## 核对与指标

检查训练完成且不是HPO、SFRUNet模型类型、Exp1/2分组、UID/parent跨split隔离、
预测文件身份、有限值、Y/B非负、B+C_hat恒等式、地图求和与总量一致。
负预测不裁剪，进入误差统计；零真值相对误差和常数真值R2返回空值。

`metrics.csv`：每个split/运行的Model和Baseline总量MAE/RMSE/R2/Bias/非零真值中位相对误差、
负总量预测比例，以及Cell_前缀的逐格同类指标。SFR均转为Msun/yr；Cell是每格积分量，不是密度。
不生成容易与标量closure输出混淆的变换域y_pred。

`test_predictions.csv` / `out_predictions.csv`：逐snapshot总量与误差，还包含逐格MAE/RMSE、
负格子比例、负SFR总幅度、sum|局部误差|。对比|sum局部误差|可识别抵消现象。

`integrated_metrics.csv` / `out_trajectory.csv`：仅对Exp2 out计算累计新形成质量及积分误差。
按真实Gyr时间排序，使用梯形积分，SFR乘1e9年/Gyr后得到Msun，等价于code SFR积分乘2.5e7Msun。
累计量从首帧设为0，不等同于日志NEWstar或模拟初始恒星质量。

`Model Error all time`、`Baseline Error all time` 定义为
`100*(预测积分-真实积分)/真实积分`，单位百分比；**负数低估、正数高估**。
积分范围仅限该缓存实际覆盖区间；缺口数量、覆盖时长与起止时间同时输出。
默认大缺口不积分，图中用灰色标记；单帧无法积分。时间重复直接报错，不静默平均。
阈值依赖采样分布，人工排除的细小缺帧可能仍被梯形近似，不能把它视为完整轨迹证明。

域内随机test是稀疏样本：展示散点/采样历史，不绘制或声称全运行累计质量。
预测文件没有几何坐标/体积；空间图使用r/theta索引。首帧、真实总SFR峰值、末帧各画一组图，
包含True/Baseline/Model、两个误差图与预测closure。物理场共享色标，误差/closure共享另一色标。
真实物理坐标图和SFR密度图需另行加载几何信息，当前不伪造。

## Burst 分析

默认关闭。用户需先核对星系的论文窗口及时间映射，再显式输入参数；本模块不带Model0.75或fiducial硬编码窗口。
可计算窗口内采样点MAE/RMSE、积分误差、峰值相对误差、峰值时刻偏差；峰值并列取最早点。
积分在窗口端点做线性插值，不外推，不跨明显缺口。mass_relative_error和peak_relative_error是比例，非百分数。

提供阈值时，返回窗口内首次阈值事件以及事件个数。开始/结束交叉时间线性插值；
事件未检出或对应窗口边界删失时，时刻误差留空。多个事件仅按首个比较，不声称完成事件匹配；
未加入自动背景估计、平滑或持续时间过滤。参数和事件定义全部记录在analysis.json。
使用目标out轨迹诊断可以，但不能将这些结果反过来选取同一Exp2的HPO参数。

图表包括SFR历史、逐时间误差、误差抵消、累计质量、散点、逐格RMSE和空间图。
`analysis.json` 保存单位、误差符号、积分/窗口参数及源文件SHA256，便于复查来源。

## 验证

```powershell
python -m unittest tests.test_analyze_sfr_unet -q
```

覆盖单位积分、缺口、重复时间、零真值、负预测、阈值交叉/删失、窗口端点、地图与总量一致性、
绘图、拒绝覆盖、notebook语法和注释。另用小型Exp2训练输出按顺序实际执行notebook。
未将smoke结果当作科学性能；未验证真实fiducial的论文burst窗口。
