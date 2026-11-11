# 含baseline的CNN/UNet历史对照

按用户参考图使用symlog纵轴，linthresh=1e-4 Msun/yr，保留零点。
同一星系左CNN右UNet，共享全部坐标范围和刻度；黑色真值、红色模型、绿色MACER-LR baseline。
baseline使用同一份正式UNet导出的物理baseline，不混入历史CNN不同定义的KS baseline。
全部数据来源、时间和标签对齐仍按上版核验，无平滑、抽稀或峰值裁剪；缺口断线显示。

|主图|PNG|PDF|SVG|
|---|---|---|---|
|四组汇总|[PNG](all_four_cnn_vs_unet_symlog.png)|[PDF](all_four_cnn_vs_unet_symlog.pdf)|[SVG](all_four_cnn_vs_unet_symlog.svg)|
|Fiducial|[PNG](fiducial_cnn_vs_unet_symlog.png)|[PDF](fiducial_cnn_vs_unet_symlog.pdf)|[SVG](fiducial_cnn_vs_unet_symlog.svg)|
|Fiducial4|[PNG](fiducial4_cnn_vs_unet_symlog.png)|[PDF](fiducial4_cnn_vs_unet_symlog.pdf)|[SVG](fiducial4_cnn_vs_unet_symlog.svg)|
|Fiducial7|[PNG](fiducial7_cnn_vs_unet_symlog.png)|[PDF](fiducial7_cnn_vs_unet_symlog.pdf)|[SVG](fiducial7_cnn_vs_unet_symlog.svg)|
|Model0.75|[PNG](model075_cnn_vs_unet_symlog.png)|[PDF](model075_cnn_vs_unet_symlog.pdf)|[SVG](model075_cnn_vs_unet_symlog.svg)|

另保留各图线性版本和Fiducial Trial7补图。Fiducial主图CNN为Trial43、UNet为指定seed43。
Fiducial4和Model0.75仍是历史CNN效果比较，训练数据、输入或监督不同，不能当纯架构消融。
物理指标见metrics.csv（新增Baseline），来源与哈希见sources.json。
四组及补图完整生成；Fiducial主图已目视核验，2项对齐测试通过，包含baseline随时间对齐检查。
