# CNN与UNet留出SFR历史组会图

```powershell
D:/conda_envs/physics_ai/python.exe VisualScripts/ResultAnalysis/plot_cnn_unet_histories.py --output-dir VisualScripts/ResultAnalysis/outputs/新的目录
D:/conda_envs/physics_ai/python.exe -m unittest tests.test_history_plot_alignment -q
```

只读正式预测，不加载权重或重新推理。输出目录必须不存在。

固定主图来源：Fiducial CNN Trial43、UNet ReLU seed43；Fiducial4 CNN为用户指定的
Exp2_LOOfiducial4_NoAug_20260817_172856；Fiducial7 CNN Trial68；Model0.75 CNN历史Trial53。
后三组UNet使用CASES里已核验的Final目录。Fiducial Trial7另外保留补图。
精确结果目录和文件SHA256写入sources.json，物理指标写入metrics.csv。

每组左CNN右UNet，共用横纵坐标。横轴统一snapshot绝对时间0.5–12.5 Gyr；
纵轴使用同组真值和两模型的联合范围加7%边距，不截断峰值。
不同星系各自行使纵轴范围，四行汇总图只在每行左右共享纵轴。
按用户参考图，主图为symlog坐标，共同采用1e-4 Msun/yr线性阈值，保留零点及10的幂次刻度；另存线性图。
黑色为真值、红色为各模型、绿色为MACER-LR baseline。两侧使用同一份UNet正式导出的权威物理baseline；
历史CNN的KS baseline定义不同，不混用为MACER-LR baseline。
所有图均导出PNG（220 dpi）、矢量PDF和SVG；不平滑、不抽稀。
大于5倍中位cadence的缺口断线并着灰，不把缺失区间连接为已预测轨迹。

UNet的out_predictions.pt经过既有load_predictions验证，地图/总量及ReLU契约一致。
当前CNN按UID与parent配对；Fiducial4旧CNN无UID，先根据RawDataConfig的
segmented_restart_absolute契约将旧编号时间还原为绝对时间，再按时间完整配对8770帧。
旧CNN额外46个早期帧不进入共同区间。所有配对逐帧核验真值；Fiducial4最大差约1.01e-5 Msun/yr。
所有旧CSV的sfr列统一乘0.025；Fiducial4虽标为Msun/yr，实际仍为代码值，
这一修正由当前权威Y逐帧核验，不对y_true/y_pred变换空间列作图。

Fiducial4历史CNN是3-run、8通道、直接SFR，UNet是5-run、12通道、逐格闭合监督；
Model0.75历史CNN训练cache与归一化也不同。图只比较已有实验性能，不能称为纯架构消融。
即使其他折同粗数据，监督信息和优化设置也不同。

最新含baseline成品位于outputs/cnn_unet_histories_baseline_20261111；此前不含baseline版本保留。
