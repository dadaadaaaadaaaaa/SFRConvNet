# Model0.75 与 Fiducial7：Exp2 分布泛化比较

日期：2026-11-10。沿用2026-11-08 Fiducial的方法：去重train的SFR范围分层，以及train-only输入近邻覆盖分层。
证据来自实际cache、正式模型metadata和预测、重新计算的距离、测试；没有新训练、推理或按out选参。
全部954/7025帧保留，包括预测异常；表格单位都是总SFR的Msun/yr，不是变换目标空间。

## 主要结论

- Model0.75：UNet在输入覆盖内、外都改善历史CNN，累计形成质量误差从-17.18%减至-1.88%。
  但最远输入尾部的优势缩小，真实峰值仍低估25.1%。
- Fiducial7：UNet在标签范围内与高端范围外都改善CNN，但仍持续低估；高端MAE=23.35，
  真实峰时低估41.4%，观测区间累计质量误差-34.18%。不能把改善写成已经可靠恢复爆发幅值。
- 先前Fiducial主要暴露低SFR外推与某些种子的尖峰；新两折显示失败模式并不相同。
  特别是Fiducial7，错误无法主要由“全场输入距离超过阈值”解释。

## 对照身份与可比性

UNet来源为用户给定两个Final目录，均complete、hpo=false、seed42、ReLU Candidate1、100轮训练：

- Model0.75：`Exp2_LOO_Model075_SFRUNet_ReLU_Candidate1_Final_20261109_142102_519eb27d`，best epoch99。
- Fiducial7：`Exp2_LOO_Fiducial7_SFRUNet_ReLU_Candidate1_Final_20261110_031212_9b4f1351`，best epoch94。

Fiducial7使用同一v7 metadata的CNN Trial68。
Model0.75使用Train_history登记的CNN Trial53作为历史对照：测试954帧的UID/parent、时间、Y/B和坐标相同，
撤销标准化后的输入最大差3.64e-12；但train UID集合并不相同（旧15187、新15106、交集9192）。
因此Model0.75是相同out上的历史方案比较，不是控制训练数据的架构消融。
所有分层边界都来自各自新UNet的train，不能沿用Fiducial边界，也不拿历史CNN训练集混合校准。

## 一、标签SFR范围

|留出运行|去重train范围|out范围|范围内N|低端范围外N|高端范围外N|
|---|---:|---:|---:|---:|---:|
|Model0.75|0–84.3584|0.2461–54.0787|954|0|0|
|Fiducial7|0–54.0787|3.5277–84.3584|6493|0|532|

Model0.75在min–max和p01–p99定义下都没有标签域外帧，不能估计该折的标签范围外误差。
Fiducial7的532个高端帧位于绝对时间4.1325–5.0950 Gyr内的多个观测段，并非整个窗口持续越界。
它们不是532个独立事件；也不能根据后续缺口推断物理爆发已经结束。

|留出运行与类别|N|CNN MAE / RMSE|UNet MAE / RMSE|Baseline MAE / RMSE|
|---|---:|---:|---:|---:|
|Model0.75：范围内|954|3.574 / 5.599|1.162 / 2.048|6.189 / 8.192|
|Fiducial7：范围内|6493|6.953 / 9.467|5.218 / 6.619|11.542 / 13.315|
|Fiducial7：高端范围外|532|35.827 / 36.328|23.352 / 24.017|41.451 / 41.857|

Fiducial7的UNet MAE在范围内改善25.0%，高端范围外改善34.8%。
但高端组全部预测低于真值，中位绝对相对误差仍37.95%；按snapshot求和的相对偏低为36.01%。
改变训练标签分位数阈值后，改善方向保留，具体计数与边界见各折报告。

## 二、输入场覆盖

每个训练运行最多192个物理时间锚点；每组最多64个校准点，排除同运行±0.25Gyr的近邻。
在12个标准化输入通道的全场RMS近邻距离上，使用train校准p95阈值。
阈值Model0.75=0.856822、Fiducial7=0.921884；它们是各折独立的相对标尺。

|留出运行与类别|N|CNN MAE / RMSE|UNet MAE / RMSE|Baseline MAE / RMSE|
|---|---:|---:|---:|---:|
|Model0.75：输入覆盖内|842|3.247 / 5.186|0.930 / 1.636|5.784 / 7.765|
|Model0.75：输入覆盖外|112|6.036 / 8.056|2.908 / 3.951|9.235 / 10.881|
|Fiducial7：输入覆盖内|6901|9.090 / 13.501|6.574 / 9.170|13.777 / 17.212|
|Fiducial7：输入覆盖外|124|11.928 / 14.541|7.551 / 9.401|15.510 / 17.695|

Model0.75覆盖外UNet MAE改善51.8%，但仍是自身覆盖内误差的3.13倍。
将阈值改成p99，仅剩24个最远帧，CNN/UNet MAE=4.5019/4.2905，优势显著缩小。

Fiducial7覆盖外UNet MAE改善36.7%，但覆盖内仍有很大误差。
532个高SFR范围外帧中525个被全场距离判为覆盖内，内盘表示下也有520个被覆盖。
真实主峰的输入距离/阈值仅0.6191，而UNet幅值仍低估41.4%。
因此不能把“输入近邻覆盖”当成正确预测的保证，也不能直接据此证明完整物理场真正域内或LR→SFR非唯一。
平均距离、关键位置差异、有限参考库与闭合映射都可能影响判断，当前分析没有完成因果归因。

## 三、误差性质与积分

|留出运行|真实峰值|真实峰时UNet|真实峰时CNN|UNet峰时偏差|最大单帧占总平方误差|
|---|---:|---:|---:|---:|---:|
|Model0.75|54.0787|40.4918|22.3847|-25.1%|4.61%|
|Fiducial7|84.3584|49.4132|32.0499|-41.4%|0.231%|

Model0.75主要是峰值附近及部分后期状态难度增加，Fiducial7主要为长时持续低估。
两者都没有出现前述Fiducial个别ReLU种子那种一帧支配绝大多数平方误差的情况。

同一观测区间梯形积分、显式排除大缺口：

|留出运行|有效覆盖时长/Gyr|排除大缺口|CNN累计质量误差|UNet累计质量误差|Baseline累计质量误差|
|---|---:|---:|---:|---:|---:|
|Model0.75|11.9125|0|-17.18%|-1.88%|-79.26%|
|Fiducial7|8.7850|2|-47.65%|-34.18%|-72.03%|

这些积分不是日志NEWstar；Fiducial7不能声称覆盖完整0.620–12.4875 Gyr。
两折UNet的瞬时统计与积分均已和用户现有分析交叉核对一致。

## 统计边界与验证

0.25/0.5Gyr配对时间块bootstrap中，可估计各组的UNet−CNN MAE差区间方向均为改善。
Fiducial7高标签组仅4个0.25Gyr块或2个0.5Gyr块，少于预定5块，因此其置信区间留空。
这只是固定模型、单一留出轨迹的条件区间；新两折均只有一个UNet种子，不能解释为独立星系或种子总体置信度。
不同折的绝对MAE还受真值幅度差异影响，不能据此直接给星系泛化难度排序。

相关10项测试通过；两折全量CLI完成；原Fiducial完整回归与旧metrics逐数值一致。
新旧输出单位、UID、baseline、cache摘要、地图求和、B+C与ReLU契约已核验，PNG人工检查。
未修改数据/权重、用户Train_history及notebook；未验证新训练、其他种子或真实GIZMO。

## 产物与复算

- [Model0.75完整报告](model075_domains_20261110_final/report.md)、[轨迹图](model075_domains_20261110_final/domain_history.png)。
- [Fiducial7完整报告](fiducial7_domains_20261110_final/report.md)、[轨迹图](fiducial7_domains_20261110_final/domain_history.png)。
- 各目录的metrics、integrated_metrics、time_segments、paired_block_bootstrap和method保留数值及来源摘要。
- 逐帧predictions_and_domains.csv保留本地，不提交；可由下列命令复算。

```powershell
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_fiducial_domains.py --case model075 --output-dir VisualScripts/ResultAnalysis/outputs/新的Model075目录
C:/Users/augus/miniconda3/python.exe VisualScripts/ResultAnalysis/compare_fiducial_domains.py --case fiducial7 --output-dir VisualScripts/ResultAnalysis/outputs/新的Fiducial7目录
C:/Users/augus/miniconda3/python.exe -m unittest tests.test_fiducial_domains tests.test_analyze_sfr_unet -q
```
