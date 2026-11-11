# CNN与UNet四折统计图

## 组会图

- 六面板总览：[PNG](statistics_overview.png)、[PDF](statistics_overview.pdf)、[SVG](statistics_overview.svg)。
- 三种误差改善比例：[PNG](error_reduction.png)、[PDF](error_reduction.pdf)、[SVG](error_reduction.svg)。

总览同时展示CNN、UNet和MACER-LR baseline的MAE、RMSE、中位绝对相对误差、R²、Bias；
最后一格展示UNet相对CNN的MAE降低百分比。改善图将三种误差都换成相对CNN的改善百分比。

|留出星系|CNN MAE|UNet MAE|MAE降低|CNN R²|UNet R²|
|---|---:|---:|---:|---:|---:|
|Fiducial|1.412|0.163|88.5%|−1.728|0.929|
|Fiducial4|4.453|1.463|67.1%|0.158|0.901|
|Fiducial7|9.140|6.591|27.9%|0.292|0.674|
|Model0.75|3.574|1.162|67.5%|0.439|0.925|

主要观察：四折所列三种误差均降低，R²均提高。Fiducial的MAE/RMSE改善最大。
Fiducial4和Model0.75的MAE降低约三分之二。Fiducial7改善相对有限，
UNet平均偏差仍为−6.551 Msun/yr，中位绝对相对误差34.61%，表明持续低估仍需解决。

## 统计与来源

采用上一版history图完全相同的模型和共同帧：Fiducial CNN Trial43、UNet seed43；
Fiducial4 CNN为用户指定20260817历史实验；Fiducial7 CNN Trial68；Model0.75历史CNN Trial53；
后三组UNet均为已指定Final模型。精确目录见sources.json。
物理单位MAE/RMSE/Bias为Msun/yr；Bias=mean(预测−真值)，正值高估、负值低估。
中位绝对相对误差只对Y>0的帧计算，图中乘100呈百分数，指标取中位数。
误差降低=100×(1−UNet误差/CNN误差)。R²只比较原值及其差值。

数据读取自../cnn_unet_histories_baseline_20261111/metrics.csv，
该目录已完成单位、绝对时间、UID/parent或历史时间配对和逐帧真值核验。
本次新增核验：四折各三个方法的指标行唯一、帧数相同并与来源审计一致，指标有限且误差项非负。
六面板图和改善图均实际生成并目视检查。全部精度见comparison.csv和metrics.csv，来源及代码哈希见sources.json。

每个星系采用snapshot等权统计，时间积分另见history分析。保留四个星系的独立结果，
避免不同cadence和样本量导致某个运行支配合并指标。相邻帧相关，本图为固定模型的描述性点估计。
本次未估计时间块置信区间或多训练种子不确定性。
Fiducial4/Model0.75的CNN为历史实验，训练数据或归一化不同；模型之间监督、输入或优化设置也存在差异。
这些结果描述现有实验性能；单独的架构效应仍需受控实验测定。

## 复现

```powershell
D:/conda_envs/physics_ai/python.exe VisualScripts/ResultAnalysis/plot_cnn_unet_statistics.py --output-dir VisualScripts/ResultAnalysis/outputs/新的统计图目录
```

脚本拒绝覆盖已有目录。中文字体使用本机Microsoft YaHei；PNG为230 dpi，PDF/SVG为矢量格式。
