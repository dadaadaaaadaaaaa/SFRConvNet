# 四折所有时间、所有粗格的SFR诊断

## 图像与数据

|图像|PNG|PDF|SVG|
|---|---|---|---|
|四组UNet密度汇总|[PNG](all_four_unet_density.png)|[PDF](all_four_unet_density.pdf)|[SVG](all_four_unet_density.svg)|
|Fiducial完整诊断|[PNG](fiducial_all_cells.png)|[PDF](fiducial_all_cells.pdf)|[SVG](fiducial_all_cells.svg)|
|Fiducial4完整诊断|[PNG](fiducial4_all_cells.png)|[PDF](fiducial4_all_cells.pdf)|[SVG](fiducial4_all_cells.svg)|
|Fiducial7完整诊断|[PNG](fiducial7_all_cells.png)|[PDF](fiducial7_all_cells.pdf)|[SVG](fiducial7_all_cells.svg)|
|Model0.75完整诊断|[PNG](model075_all_cells.png)|[PDF](model075_all_cells.pdf)|[SVG](model075_all_cells.svg)|

每个完整诊断包含UNet与baseline逐格计数密度、两套零/成星分类矩阵、真实SFR分箱恢复率、精确零值误差收支。
密度图全部计入零值及正值，使用统一symlog坐标和LogNorm计数色标，红虚线为预测=真值。
颜色是指定物理分箱中的样本计数，密度图分箱宽度随SFR变化；它不是单位线性面积的概率密度。
原点及极弱SFR共享首批线性区间，精确零值的数量由分类表另行给出。

全部图使用相同模型：Fiducial ReLU Candidate1 seed43，以及Fiducial4/Fiducial7/Model0.75指定Final。
逐格预测来自正式out_predictions.pt，基线为对应的MACER-LR baseline。

## 主要结果

### 零值识别的数量与SFR贡献

精确零值口径epsilon=0：

|留出星系|真实0预测为正：格子数|真实正预测为0：格子数|虚报SFR/真实总和|漏报损失/真实总和|
|---|---:|---:|---:|---:|
|Fiducial|249867|45572|6.440%|0.352%|
|Fiducial4|298045|63433|6.251%|0.039%|
|Fiducial7|366055|27840|0.506%|0.020%|
|Model0.75|34703|4194|3.408%|0.027%|

四组都出现更多的零值虚报格子，但这不能直接说明哪个错误主导总量。真实零值数量远大于正值数量。
例如Fiducial7精确零值虚报率24.16%，正值漏报率16.30%；相应SFR贡献分别仅为0.506%和0.020%。

主图采用固定epsilon=1e-4 Msun/yr，Active定义为Y>epsilon，Inactive定义为Y<=epsilon；预测类别同理。
矩阵每格同时标注绝对数量与按真实类别归一化的行百分比。此阈值用于弱成星诊断，不代表原始模拟的零值定义。
敏感性同时保存epsilon=0、1e-6、1e-4、1e-3。

|留出星系|虚报率（真实Inactive中的比例）|漏报率（真实Active中的比例），epsilon=1e-4|
|---|---:|---:|
|Fiducial|12.67%|22.81%|
|Fiducial4|15.28%|16.45%|
|Fiducial7|21.39%|10.77%|
|Model0.75|14.42%|17.72%|

对于epsilon>0，虚报SFR贡献采用sum(predicted−true)，漏报损失采用sum(true−predicted)，
保留阈值内的小正真值，避免直接把它当成精确零值。

### 高SFR格的幅度恢复

将真实逐格Y>=1 Msun/yr作为固定的高SFR诊断子集：

|留出星系|格子数|该子集真实SFR占比|预测总和/真实总和|低估格子比例|
|---|---:|---:|---:|---:|
|Fiducial|1281|21.02%|92.59%|44.96%|
|Fiducial4|25823|76.98%|90.50%|59.77%|
|Fiducial7|29218|87.69%|63.43%|85.96%|
|Model0.75|1653|83.31%|94.18%|60.13%|

这组结果直接支持Fiducial7的高成星格幅度不足：该子集贡献87.69%的真实SFR，预测仅恢复63.43%。
子集的净低估相当于全部真实SFR总和的32.07%，约占全体净低估34.18%的93.84%。
这是固定样本集合上的加性收支关系，不是物理因果贡献率。
Fiducial7的持续总量低估主要由已有成星格的幅度不足体现，精确正值被完全置零损失只有0.020%。

正标签按真实Y分箱，恢复率R_k=sum(predicted)/sum(true)。每箱N、真实/预测总和、
中位预测、逐格比值中位数与低估比例均写入positive_sfr_bins.csv。
主图展示Y>=1e-4的箱，极弱正值箱全部保存在CSV中，避免极小分母产生的巨大比值占据主图动态范围。
箱间不等样本量，尾部箱须结合N解释。未用held-out误差选择阈值或改变模型。

### 精确零值误差收支

将净误差拆成三部分，全部以所有真实SFR之和归一化：

|留出星系|零值虚报|正值格高估|正值格低估|净偏差|
|---|---:|---:|---:|---:|
|Fiducial|+6.440%|+12.609%|−11.423%|+7.626%|
|Fiducial4|+6.251%|+7.913%|−15.729%|−1.565%|
|Fiducial7|+0.506%|+1.388%|−36.072%|−34.177%|
|Model0.75|+3.408%|+7.905%|−13.177%|−1.864%|

Fiducial4和Model0.75净误差较小，同时存在方向相反的局部误差。误差总和接近零，仍需检查空间分配。

## 统计口径、验证与限制

- 帧数×240粗格：Fiducial 8429×240=2022960，Fiducial4 8770×240=2104800，
  Fiducial7 7025×240=1686000，Model0.75 954×240=228960。
- 地图标准顺序N1theta-r，形状N×1×12×20；从代码SFR乘0.025得到每粗格Msun/yr，格子量已含体积。
- 核验正式预测schema/单位、loss/ReLU契约、candidate组成、地图及总量、UID/parent、运行身份、有限值。
  四组文件摘要与既有正式分析一致；计数密度图核验全部样本进入分箱，误差三项之和严格等于净偏差。
- 每个时间—格子样本等权；所有SFR总和均为snapshot求和。这里没有按实际时间间隔积分，不能把这些和称为新形成恒星质量。
- coarse cells的体积不同，当前图比较每格积分SFR贡献；体积加权分布和SFR密度分布需要另行定义。
- 相邻时间和空间格相关；样本数仅表示计数，不作为独立模拟数。未计算置信区间，也未训练新模型。
- 三项单元测试通过：条件分类率与误差收支、近零阈值贡献定义、分箱零值排除与末端保留。
- 图像已实际生成并目视核验Fiducial7完整图和四折密度汇总。图形由Matplotlib绘制。

metrics.csv、classification_thresholds.csv、positive_sfr_bins.csv、high_sfr_cells.csv和error_budget.csv保存精确结果。
sources.json保存来源哈希、物理阈值和全部分箱边界。

## 复现

```powershell
D:/conda_envs/physics_ai/python.exe VisualScripts/ResultAnalysis/analyze_unet_all_cells.py --output-dir VisualScripts/ResultAnalysis/outputs/新的逐格目录
D:/conda_envs/physics_ai/python.exe -m unittest tests.test_all_cell_diagnostics -q
```

目录必须不存在。输出PNG、PDF及SVG，原正式结果保持原样。
