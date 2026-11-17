# 六组盘星系瞬时 SFR 同轴比较

入口：`plot_disk_sfr_comparison.py`。沿用 `SFREvolution.ipynb` 的瞬时 SFR
定义：`sum(Data-Set-11) * 0.025 Msun/yr`，不额外乘体积，不使用日志
`NEWstar` 的差分。六组为 Fiducial、Fiducial4、Fiducial7、Model0.25、
Model0.75、Model1.1；不包含 torus 核区环面或不存在的 Fiducial10 配置路径。

五组已有数据复用 `outputs/ssfr_verified_20261023/ssfr_verified.parquet`
中逐帧保存的原始 SFR：核对完整文件清单、每帧大小/修改时间、原始索引及当前
配置生成的时间，并对每组首、中、末三个 snapshot 重读 Data-Set-11。
大小/修改时间核对不是文件内容哈希；缓存来源的历史全量审计见
`plot_disk_ssfr_evolution.md`，本入口不声称重新全量读取了五组原始网格。

Model0.25 尚未登记在当前 RawDataConfig 中。本脚本仅为绘图使用
`DiskGalaxy/Model0.25` 和 `index * 0.0125 Gyr` 的独立分析映射；
1000 帧逐一核验 HDF `long_name` 的 SFR 语义及时间打印精度，再归约原始网格。
这不改变训练数据范围或 split。

横轴为绝对模拟时间（不减去 0.5625 Gyr），纵轴采用与 notebook 一致的
`symlog(linthresh=1e-4)`，保留零值；不平滑、不插值，超过正常 cadence
1.5 倍的时间间隔断线。Fiducial 组与 Model 组原始采样间隔不同，比较高频
起伏时应注意该差异。图表示模拟存储的全网格瞬时 SFR，不是累计恒星质量。
样式参考用户提供的 sSFR 图：仅显示细实线、不显示点标记，保留完整坐标边框、
浅色虚线网格和右上角双列图例；纵轴仍为本任务的 SFR。Model0.25 使用棕色。

```powershell
$env:MPLBACKEND = 'Agg'
python VisualScripts/DataAnalysis/plot_disk_sfr_comparison.py
```

需要 numpy、pandas、matplotlib、pyarrow、pyhdf，以及项目数据读取模块依赖。
默认输出至 `outputs/sfr_six_models_20261117/`，已有非空目录时拒绝覆盖；
可使用 `--output-dir <新目录>`。输出 PNG、PDF、CSV、Parquet 与来源核验 JSON。
后续绘图可直接读取该 Parquet 并调用 `plot_comparison(frame)`，无需再扫描原始文件。
原 notebook 已有用户改动，本次独立脚本不修改它。

本次实际运行得到 27362 帧：Fiducial 8475、Fiducial4 8816、Fiducial7 7071，
三个 Model 各 1000。六组时间均覆盖 0–12.4875 Gyr。已验证来源身份无重复、
有限值、绘图样本与原表逐值一致、零值保留及缺帧断线，且目视检查 PNG。
五组复用表未进行全文件内容哈希或重新全量网格归约。
