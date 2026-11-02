# 二维逐格 SFR baseline 与监督（方案 C 的数据阶段）

## 定义与范围

对二维原生 `(N_theta,N_r)` 网格：

\[
Y_i=\sum_{j\in i}{\tt DataSet11}_j,\quad
B_i=\mathcal S_i[\mathcal R_h(U_{HR})],\quad C_i^{true}=Y_i-B_i.
\]

三者均为**每格已积分的瞬时 SFR 贡献**，单位 `MACER_mass_code/Gyr`；乘 0.025 为 `Msun/yr`。
Data-Set-11 已含体积，归约时不再乘体积，也不平均。非均匀网格同样成立。
`B_i` 在 LR 气体上计算，不能把 HR baseline 求和当作 LR baseline。
保存 `data_compatible` 和 `paper_physical` 两套 baseline/closure；canonical 默认前者。
验证 `sum(Y_i)=Y`、`sum(B_i)=B`、`sum(C_i)=C` 和逐格 `Y_i=B_i+C_i`。
`Y_i/B_i` 非负，closure 可负；零 SFR 格子也是标签。

本阶段支持二维 `identity_fine` 与 `spatial_block/R_h`。R_m 尚未导出标签输运权重，
不能用规则分块代替；含逐格标签的 fine 仍可生成原 R_m 标量产品，但不生成 R_m 局部真值。
三维流程和原 CNN/标量训练行为不变。

## 新旧入口兼容

- 新 `build_fine_from_raw.py` 自动保留 Data-Set-11 的逐格数组。
- `build_coarse_from_fine.py` 在 fine 有标签时自动对 R_h 逐块求和。
- 旧 fine/coarse 仍可用于标量训练；缺失局部标签时不会填零或按 baseline 分配真值。
- 本脚本为旧 fine 或 R_h **另存一份**完整产品：保留气体、日志、UID、总量及现存 baseline，补齐局部真值。
- 源文件由 metadata 的 `source_series_id/source_file_id` 定位，验证 SHA256、权威体积、shape 和总量。
  不重新选择日志时刻，不重算 baseline，不修改时间映射。

使用新目录后缀 `_cell_v1`。这是 format-version 6 的可选监督扩展，保留已有 v7 时间修正，
不是新物理算子版本。

## 命令

在项目根目录先测试少量已有 coarse：

```powershell
python raw_data_processing/build_cell_sfr_supervision.py --galaxy disk_galaxy_fiducial --input-dirname coarse_rh_12x20_sfr_v7 --output-dirname coarse_rh_12x20_sfr_v7_cell_smoke --max-snapshots 3
```

正式批处理使用不同输出目录：

```powershell
python raw_data_processing/build_cell_sfr_supervision.py --galaxy disk_galaxy_fiducial --galaxy disk_galaxy_fiducial_4 --galaxy disk_galaxy_fiducial_7 --galaxy disk_galaxy_model_0_75 --galaxy disk_galaxy_model_1_1 --input-dirname coarse_rh_12x20_sfr_v7 --output-dirname coarse_rh_12x20_sfr_v7_cell_v1
```

若要尝试多个分辨率，先补齐 fine，后续不再重复读 raw：

```powershell
python raw_data_processing/build_cell_sfr_supervision.py --galaxy disk_galaxy_fiducial --input-dirname fine_sfr_v7 --output-dirname fine_sfr_v7_cell_v1
python raw_data_processing/build_coarse_from_fine.py --galaxy disk_galaxy_fiducial --fine-dirname fine_sfr_v7_cell_v1 --coarsener spatial_block --coarse-shape 12 20 --coarse-dirname coarse_rh_12x20_sfr_v7_cell_v1 --require-cell-supervision
```

输入目录名应与实际产品一致，例如时间迁移产品也可能在 `fine_sfr_v7_timefix`。
也可使用原 raw→fine 命令生成含逐格标签的新 fine。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--galaxy` | 必填，可重复 | 完整生成名 |
| `--input-dirname` | 必填 | 二维 fine 或 R_h 的单层子目录 |
| `--output-dirname` | 必填 | 不同于输入；已有输出文件即报错 |
| `--config-dir` | `.config` | 读取 BaseConfig.json |
| `--raw-dir` / `--data-dir` | BaseConfig 对应路径 | 支持服务器路径覆盖 |
| `--snapshot-index` | 全部，可重复 | 按 processed 文件名中的原始 index 选择 |
| `--max-snapshots` | 不限制 | 每组最多处理数量，必须为正 |

已完成输出不覆盖。中断后可选择未处理的 snapshot 或新输出目录；本版不自动续跑。
沿用 processed 已有排除结果，不额外创建缺失的原始帧。

## HDF5 与 Python 契约

原 `processed_format_version=6` 和标量 canonical 保持不变；新增可选
`metadata.cell_supervision`，独立 `schema_version=1`。

```text
sfr_supervision/canonical/Y_alpha_cell        # 新 (theta,r)
sfr_supervision/canonical/B_alpha_cell        # 新 (theta,r)
sfr_supervision/canonical/C_alpha_true_cell   # 新 (theta,r)
sfr_supervision/<mode>/cell_sfr_baseline      # 原有逐格 baseline
sfr_supervision/<mode>/cell_sfr_closure_true   # 新逐格 closure
```

原 `Y_alpha/B_alpha/C_alpha_true` 标量不变。metadata 记录单位、轴、shape、标签来源、归约、
baseline 定义、UID、各模式 physics_product_uid 和数组 SHA256；独立局部 product_uid 随标签/
baseline 变化。sample_uid 不变，便于复用 split。数组只在监督组，不进入 `snapshot` 或输入通道。

```python
from raw_data_processing.galaxy_data import GalData
g = GalData().load_h5(path)
assert g.has_cell_sfr_supervision
maps = g.cell_sfr_supervision()  # Y、B、C_true 的副本；默认 data_compatible
paper_maps = g.cell_sfr_supervision("paper_physical")
```

普通 `prepare_data.py` 仍只生成标量 cache。逐格训练数据准备见
[build_cell_supervision_cache.md](../Experiment/build_cell_supervision_cache.md)。
