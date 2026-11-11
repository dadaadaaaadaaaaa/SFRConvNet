# 组会用CNN与UNet SFR history对照

每图左CNN、右UNet，黑色真值、橙色CNN、蓝绿色UNet。相同星系的两幅图严格共享横纵轴范围与刻度。
横轴为snapshot绝对时间，统一0.5–12.5 Gyr；纵轴为Msun/yr，各星系按本组全部曲线联合范围确定。
灰色为较大数据缺口，不插值连接。未平滑、抽稀或裁剪峰值。

|线性主图|PNG|PDF|SVG|
|---|---|---|---|
|四组汇总|[PNG](all_four_cnn_vs_unet_linear.png)|[PDF](all_four_cnn_vs_unet_linear.pdf)|[SVG](all_four_cnn_vs_unet_linear.svg)|
|Fiducial|[PNG](fiducial_cnn_vs_unet_linear.png)|[PDF](fiducial_cnn_vs_unet_linear.pdf)|[SVG](fiducial_cnn_vs_unet_linear.svg)|
|Fiducial4|[PNG](fiducial4_cnn_vs_unet_linear.png)|[PDF](fiducial4_cnn_vs_unet_linear.pdf)|[SVG](fiducial4_cnn_vs_unet_linear.svg)|
|Fiducial7|[PNG](fiducial7_cnn_vs_unet_linear.png)|[PDF](fiducial7_cnn_vs_unet_linear.pdf)|[SVG](fiducial7_cnn_vs_unet_linear.svg)|
|Model0.75|[PNG](model075_cnn_vs_unet_linear.png)|[PDF](model075_cnn_vs_unet_linear.pdf)|[SVG](model075_cnn_vs_unet_linear.svg)|

同名`_symlog`版本用于辨认低SFR；其0.1 Msun/yr以下为线性区间，两侧采用相同变换。
`fiducial_trial7_supplement_*`为另一版CNN补充对照，主图固定用户选择的Trial43。
Fiducial UNet固定用户指定seed43，其他三组为对应Final训练。

|共同帧指标|CNN MAE|UNet MAE|CNN R²|UNet R²|
|---|---:|---:|---:|---:|
|Fiducial|1.412|0.163|−1.728|0.929|
|Fiducial4|4.453|1.463|0.158|0.901|
|Fiducial7|9.140|6.591|0.292|0.674|
|Model0.75|3.574|1.162|0.439|0.925|

Fiducial4旧CNN采用3-run、8通道、直接SFR，UNet为5-run、12通道、逐格闭合监督。
旧CNN的时间已按权威重启段映射还原，并与UNet8770帧逐帧真值核验；额外46帧早期数据不进入对照。
旧CSV虽写Msun/yr，实际SFR列仍需乘0.025；修正后最大真值差约1.01e-5 Msun/yr。
Model0.75历史CNN训练cache及归一化也不同。组会应称为已有模型效果比较，不能称为纯架构消融。

完整来源及对齐审计见sources.json；指标见metrics.csv。
四组主图和Trial7补图均实际生成，线性汇总图与symlog单图已目视核验；2项时间映射/标签对齐测试通过。
