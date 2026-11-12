# 四折UNet残差与预测散点组合图

每图上方为signed error，下方为预测对真值散点；上下共享True SFR横轴。
signed error定义为预测减真值。蓝色model、橙色MACER-LR baseline。

|留出星系|PNG|PDF|SVG|
|---|---|---|---|
|Fiducial|[PNG](fiducial_signed_error_and_scatter.png)|[PDF](fiducial_signed_error_and_scatter.pdf)|[SVG](fiducial_signed_error_and_scatter.svg)|
|Fiducial4|[PNG](fiducial4_signed_error_and_scatter.png)|[PDF](fiducial4_signed_error_and_scatter.pdf)|[SVG](fiducial4_signed_error_and_scatter.svg)|
|Fiducial7|[PNG](fiducial7_signed_error_and_scatter.png)|[PDF](fiducial7_signed_error_and_scatter.pdf)|[SVG](fiducial7_signed_error_and_scatter.svg)|
|Model0.75|[PNG](model075_signed_error_and_scatter.png)|[PDF](model075_signed_error_and_scatter.pdf)|[SVG](model075_signed_error_and_scatter.svg)|
|四组汇总|[PNG](all_four_signed_error_and_scatter.png)|[PDF](all_four_signed_error_and_scatter.pdf)|[SVG](all_four_signed_error_and_scatter.svg)|

按照用户澄清，上方横轴采用True SFR，下方保留原scatter_and_cell_error.png的预测散点部分。
逐格RMSE随时间面板保留在原图中。原结果文件和图片均保持原样，组合图单独输出。

沿用analyze_sfr_unet.py绘图参数：s=5、alpha=0.5、默认蓝/橙配色、线性坐标、灰色虚线y=x、
灰色零误差线lw=0.6、150 dpi和bbox_inches=tight。上方残差采用相同散点样式，展示每个真实SFR对应的多帧误差。
单图figsize=(6.5,8)，上下高度比1:2；上下严格共享横轴范围和刻度。纵轴范围分别根据各自完整数据自动计算。
无平滑、分箱、抽稀或峰值裁剪。物理单位为Msun/yr。

结果来源沿用前面组会图：Fiducial为ReLU Candidate1 seed43；Fiducial4、Fiducial7、Model0.75为已指定Final。
读取其已有analysis目录的out_predictions.csv，核对out manifest的UID/parent、运行身份、有限值和残差恒等式。
analysis.json中的contract、summary、manifest与out_predictions.pt SHA256均与当前文件一致。
全部来源、源图摘要、精确实验目录、帧数与坐标范围记录于sources.json。

实际处理帧数：Fiducial 8429、Fiducial4 8770、Fiducial7 7025、Model0.75 954。
四组完整运行成功；Fiducial单图与四组汇总图已目视核验。仅调整图形组合，未重新训练或推理。

复现入口：

```powershell
D:/conda_envs/physics_ai/python.exe VisualScripts/ResultAnalysis/plot_unet_residual_scatter.py --output-dir VisualScripts/ResultAnalysis/outputs/新的组合图目录
```

输出目录必须不存在，PDF/SVG为矢量格式。
