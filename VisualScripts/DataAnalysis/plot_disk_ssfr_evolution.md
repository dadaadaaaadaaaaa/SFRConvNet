# 五组盘星系的 sSFR 核验与绘图

分析入口为 `plot_disk_ssfr_evolution.py`，notebook 入口为 `SFREvolution.ipynb`
末尾的 sSFR 单元。只包含 Fiducial、Fiducial4、Fiducial7、Model0.75、Model1.1。

## 质量与速率定义

- 原始背景恒星质量：`sum(Data-Set-14)`，字段标签必须为 `Mstar_i`。
- 原始新恒星质量：`sum(Data-Set-12)`，字段标签必须为 `MNEWSTAR`。
- 总质量：上述两项相加，乘 `2.5e7 Msun`；这两个质量场不再乘体积。
- 原始瞬时 SFR：`sum(Data-Set-11) * 0.025 Msun/yr`，不再次乘体积。
- `sSFR = SFR / M_star(t)`，单位 `yr^-1`。

这是模拟存储的背景质量加新恒星质量口径，不另外补设质量损失、恒星迁移或观测孔径
修正，也不宣称直接复现论文中的全部分析口径。

原始 `MNEWSTAR` 中可能存在局部负数值残差。分析保留原值求和，不裁剪；要求全局
新恒星质量非负，并记录负单元数量及负质量占比。SFR 与初始恒星质量仍严格要求非负。

## 缓存精度核验

复现上一版 notebook 的分母：从已有 `fine_sfr_v7`（优先）或 `fine_sfr_v6` 中取
`sum(initial_stellar_density * geometry/cell_volume)`，加上日志 Parquet 的 `NEWstar`
对齐至 snapshot 时刻的值。日志重复时间保留最后一条，使用线性插值；端点只允许
`5e-4 Gyr` 的日志打印舍入误差，端点使用数目进入报告。

逐帧比较缓存与原始总恒星质量。对每个运行，最大相对误差不超过 `1e-3`（0.1%）则
整组复用缓存质量；否则整组使用原始质量。审计表同时保留两者、新恒星质量、来源文件
大小/修改时间、原始 snapshot 编号和相对误差。

时间由当前 `.config/RawDataConfig.json` 的编号映射重建，并逐帧核对 HDF 标签打印
精度。默认显示绝对模拟时间。生产时间选项减去 `0.5625 Gyr` 并排除松弛期；不通过
连续枚举压缩缺帧时间。曲线保留零值，时间间隔超过正常 cadence 的1.5倍时断线。

## 执行

```powershell
$env:MPLBACKEND = 'Agg'
python VisualScripts/DataAnalysis/plot_disk_ssfr_evolution.py
```

默认输出目录为 `VisualScripts/DataAnalysis/outputs/ssfr_verified_20261023/`：

- `stellar_mass_audit.json`：定义、容差、逐运行结果及来源。
- `ssfr_verified.parquet`：逐帧原始/缓存质量、SFR、最终 sSFR 和来源。
- `sSFREvolution.png/pdf`：五组分面演化图。
- `sSFR_comparison.png/pdf`：五组同轴对比图。

已有核验产品时 CLI 拒绝覆盖；重新扫描需指定新的 `--output-dir`。
本次目录的 `_audit_parts/` 保存已完成运行，异常退出后可恢复这些部分结果。
`--production-time` 选项绘制生产时间。训练数据、缓存及 checkpoint 不受该分析修改。

notebook 末尾单元优先复用内存中的五组 SFR，按星系和原始文件名关联，并与本次原始
SFR 核验一致性。没有该变量时直接使用本次审计保存的 SFR。两种入口均不会在绘图时
重新扫描原始文件。notebook 的 `SSFR_USE_PRODUCTION_TIME` 控制显示时间口径。

## 验证

```powershell
python -m unittest discover -s tests -p test_disk_ssfr_analysis.py -v
```

覆盖按最大质量误差切换来源、零质量分母拒绝、缺帧断线/零 SFR 保留和生产时间变换。
实际逐帧核验结果以本次 `stellar_mass_audit.json` 为准。

## 2026-10-23 全量核验记录

| 运行 | snapshot 数 | 总恒星质量最大相对误差 | 最终来源 |
|---|---:|---:|---|
| Fiducial | 8475 | 7.1763e-7 | fine 背景质量 + 日志 NEWstar |
| Fiducial4 | 8816 | 1.0720e-6 | 同上 |
| Fiducial7 | 7071 | 3.5973e-7 | 同上 |
| Model0.75 | 1000 | 3.5377e-5 | 同上 |
| Model1.1 | 1000 | 2.9701e-6 | 同上 |

总计26362帧，未因局部负残差裁剪或排除质量字段。所有运行均小于0.1%的复用阈值，
最坏差异约0.00354%。fine 背景质量与原始 `sum(Data-Set-14)` 的最大相对误差小于
8.35e-10。此记录证明与原始输出质量归约接近，不证明论文名义质量或观测存活质量口径。

实际执行上述 CLI，生成并目视检查了两张 PNG；同时导出 PDF。5项回归检查通过。
notebook 的核验表入口及内存 SFR 入口均用26362帧真实产品执行通过，确认质量归约与
`sSFR*M_star=SFR` 恒等式，并通过禁止原始 HDF4/Parquet 读取的 mock 验证内存复用路径。
