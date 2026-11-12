# 时间残差与预测散点组合图

上方使用原sfr_history.png中的signed error时间曲线，横轴Snapshot time [Gyr]；
下方保留原scatter_and_cell_error.png的预测对真值散点，横轴True SFR [Msun/yr]。
上下采用独立横轴。蓝色model、橙色baseline，残差为预测减真值。

|留出星系|PNG|PDF|SVG|
|---|---|---|---|
|Fiducial|[PNG](fiducial_signed_error_and_scatter.png)|[PDF](fiducial_signed_error_and_scatter.pdf)|[SVG](fiducial_signed_error_and_scatter.svg)|
|Fiducial4|[PNG](fiducial4_signed_error_and_scatter.png)|[PDF](fiducial4_signed_error_and_scatter.pdf)|[SVG](fiducial4_signed_error_and_scatter.svg)|
|Fiducial7|[PNG](fiducial7_signed_error_and_scatter.png)|[PDF](fiducial7_signed_error_and_scatter.pdf)|[SVG](fiducial7_signed_error_and_scatter.svg)|
|Model0.75|[PNG](model075_signed_error_and_scatter.png)|[PDF](model075_signed_error_and_scatter.pdf)|[SVG](model075_signed_error_and_scatter.svg)|
|四组汇总|[PNG](all_four_signed_error_and_scatter.png)|[PDF](all_four_signed_error_and_scatter.pdf)|[SVG](all_four_signed_error_and_scatter.svg)|

沿用原绘图参数：时间曲线为默认plot线宽，零线gray、lw=0.6；
缺口按原analysis.json的gap_factor标灰，透明度0.2，保留原曲线在缺口中的连接方式。
下方s=5、alpha=0.5、灰色虚线y=x、线性坐标；150 dpi、bbox_inches=tight。
单图6.5×8英寸，上下高度比1:2；所有数据和峰值保留。

来源仍为Fiducial ReLU seed43及其余三组指定Final。逐帧CSV、身份、正式预测摘要与残差恒等式均已核验，
四组完整运行成功，Fiducial新版组合图已目视检查。精确来源和坐标范围见sources.json。
原图及上一版True SFR残差组合图均保留。

复现：

```powershell
D:/conda_envs/physics_ai/python.exe VisualScripts/ResultAnalysis/plot_unet_residual_scatter.py --output-dir VisualScripts/ResultAnalysis/outputs/新的目录
```

脚本默认使用时间残差。`--residual-x true_sfr`可复现上版共享True SFR横轴的布局。
