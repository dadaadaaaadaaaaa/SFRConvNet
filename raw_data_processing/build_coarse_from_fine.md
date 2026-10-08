# `build_coarse_from_fine.py` 使用说明

## MACER3D 空间降采样（2026-09-10 新增）

```powershell
python raw_data_processing/build_coarse_from_fine.py --galaxy dwarf_galaxy_fullFB_inflow --coarsener spatial_3d --prediction-mode residual --fine-dirname fine_residual
python raw_data_processing/build_coarse_from_fine.py --galaxy dwarf_galaxy_fullFB_inflow --coarsener spatial_3d --prediction-mode direct --fine-dirname fine_residual --coarse-shape 16 16 32
```

三维默认 shape 为 `(32,16,32)`，参数顺序是 `r theta phi`。每个维度必须整除 fine 对应维度。
输出默认位于 `data/<galaxy>/coarse_spatial3d_<shape>_<mode>/`，可用 `--coarse-dirname` 指定单层目录名。
Residual fine 可以生成 Direct 或 Residual coarse；Direct fine 缺少冷却和引力，不能生成 Residual coarse。
三维默认 `--prediction-mode direct`，如要输出 baseline 与 closure 必须显式指定 `residual`。
本入口名为 `build_coarse_from_fine.py`，没有单独的 `build_raw_from_fine.py`。
二维仍默认 `8 8`，原空间及质量粗粒化行为保留；以下章节描述二维契约。

## 1. 功能与边界

`build_coarse_from_fine.py` 从已经生成的 processed fine HDF5 构建 coarse 数据：

```text
format-version 6 fine HDF5
              ↓ GalData.coarsen(...)
R_h / R_m-v0 / R_m-v1 coarse HDF5
```

它不再读取原始 HDFRA，也不重新解析日志。三种粗粒化方法仍由同一个入口调度：

- `spatial_block`：固定规则块 \(R_h\)；
- `axisymmetric_rm_v0`：top-hat/cubic-spline + Sinkhorn/IPFP 的质量自适应版本；
- `axisymmetric_rm_v1`：预计算邻居、自适应局部输运图、Wendland-C2 核与加速平衡版本。

详细的守恒变量、冷却率密度与加速度的处理规则见 `galaxy_data.md`、
`coarsening_common.py` 和各 `coarsening_*.py`。

## 2. 输入检查和输出契约

输入必须同时满足：

1. metadata 完整并能重新验证 `parent_hr_uid/sample_uid`；
2. `derivation.operator_id == identity_fine`，从而禁止误把 coarse 再当 fine 粗化；
3. 至少包含 `density`、`gas_energy`、`cooling_time`、`cooling_rate_density`、
   `radial_acceleration`；
4. 已保存 canonical (Y,B,C^{\rm true}) supervision。

默认输出目录为：

- \(R_h\)：`<data_dir>/<galaxy_name>/coarse/`；
- \(R_m\)：`<data_dir>/<galaxy_name>/coarse_<coarsener>/`。

可以用 `--coarse-dirname` 显式隔离不同网格尺寸、质量尺度或核函数。不同算子参数的文件
不应写入同一目录，否则 `GalDataSet` 的 processing-signature 校验会拒绝混合。

输出继续保存：

- 保守粗粒化后的物理场、坐标、真实输出体积；
- 逐量守恒误差和 \(R_m\) 平衡/支撑诊断；
- 与 fine 相同的 `parent_hr_uid`、由算子和输出几何决定的新 `sample_uid`；
- LR baseline \(B_\alpha=\mathcal S_\alpha[\mathcal R_\alpha(U_{\rm HR})]\)；
- 原 Data-Set-11 真值 \(Y_\alpha\) 与 signed closure
  \(C_\alpha^{\rm true}=Y_\alpha-B_\alpha\)。

## 3. 命令行参数

下列粗粒化参数名称和默认值与旧 `galaxy_data.py` 入口保持一致。

| 参数 | 默认值 | 可选值与含义 |
|---|---:|---|
| `--galaxy` | 全部 | 生成名，可重复 |
| `--coarsener` | `spatial_block` | `spatial_block`、`axisymmetric_rm_v0`、`axisymmetric_rm_v1` |
| `--coarse-shape` | `8 8` | 输出 `(N_theta,N_r)` |
| `--mass-filter` | `None` | \(R_m\) 必填，单位为 MACER 质量代码单位；\(R_h\) 必须省略 |
| `--kernel` | `None` | v0 可选 `top_hat`、`cubic_spline`；v1 固定 `wendland_c2` |
| `--coarse-dirname` | 自动 | 星系目录下的单层输出目录名 |
| `--initial-neighbors` | 算子默认 | \(R_m\) 初始邻居数，至少为 2 |
| `--balance-tolerance` | `1e-6` | 输运平衡容差 |
| `--balance-max-iterations` | `30000` | v0 IPFP 最大迭代数；v1 忽略 |
| `--v1-max-neighbors` | `4096` | v1 自适应邻居上限 |
| `--v1-max-runtime-seconds` | `60.0` | v1 单 snapshot 算子内部运行时限（秒） |
| `--v1-solver-max-iterations` | `300` | v1 对偶求解最大迭代数 |
| `--v1-polish-max-iterations` | `500` | v1 IPFP polish 最大迭代数 |

拆分入口新增的批处理控制参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--config-dir` | `.config` | JSON 配置目录 |
| `--fine-dirname` | `fine` | 输入 fine 子目录名 |
| `--snapshot-index` | 全部 | 只粗化指定 filename stem/index，可重复 |
| `--max-snapshots` | `None` | 每个星系最多处理前 (N) 个 fine 文件 |

## 4. 执行命令

查看帮助：

```powershell
python raw_data_processing/build_coarse_from_fine.py --help
```

\(R_h\) 8×8：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener spatial_block `
  --coarse-shape 8 8 `
  --coarse-dirname coarse_rh_8x8_sfr_v6
```

\(R_h\) 16×16：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener spatial_block `
  --coarse-shape 16 16 `
  --coarse-dirname coarse_rh_16x16_sfr_v6
```

\(R_m\)-v0：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener axisymmetric_rm_v0 `
  --coarse-shape 8 8 `
  --mass-filter 1.0 `
  --kernel top_hat `
  --coarse-dirname coarse_rm_v0_m1
```

\(R_m\)-v1：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener axisymmetric_rm_v1 `
  --coarse-shape 8 8 `
  --mass-filter 1.0 `
  --initial-neighbors 32 `
  --v1-max-neighbors 4096 `
  --v1-max-runtime-seconds 60 `
  --coarse-dirname coarse_rm_v1_m1
```

只做两个文件的 smoke test：

```powershell
python raw_data_processing/build_coarse_from_fine.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener spatial_block `
  --coarse-shape 8 8 `
  --max-snapshots 2 `
  --coarse-dirname coarse_rh_smoke
```

## 5. 与训练代码的衔接

新 coarse 文件不需要额外转换即可进入：

```text
GalDataSet.load_data(require_complete_metadata=True)
  → CachePt.to_pt(objective_mode="direct_sfr")
  → GalPTDataset(..., objective_mode="direct_sfr")
```

或：

```text
GalDataSet.load_data(require_complete_metadata=True)
  → CachePt.to_pt(objective_mode="signed_closure")
  → GalPTDataset(..., objective_mode="signed_closure")
  → 显式组成 B + C_hat
```

固定单一 \(R_h\) 使用 schema-v5 的 inactive `conditioning=none`，训练时不要启用
resolution FiLM。\(R_m\) 保持 `log10(mass_filter)` 条件语义并要求相应 FiLM 接口。

## 6. 可靠性检查建议

每次新增算子参数或输出网格形状，至少检查：

1. `metadata_completeness == complete`；
2. coarse 与来源 fine 的 `parent_hr_uid` 相同、`sample_uid` 不同；
3. `derivation.operator_id` 和全部 `operator_parameters` 正确；
4. `epsilon_conservation` 不超过指定容差；
5. 冷却功率、质量、动量和内能等逐量守恒诊断有限；
6. \(Y=B+C^{\rm true}\) 在浮点容差内成立；
7. 同目录所有文件坐标网格与 processing signature 一致；
8. direct-SFR 与 signed-closure 两种 CachePT/GalPTDataset 都能加载。

三维 `fine→coarse` 已经完整读取 fine 数组，因此 provenance 默认记录文件名、大小和
纳秒修改时间，不再为每个大型 fine HDF5 额外计算一次全文件 SHA-256。物理产品身份
仍由 fine 的 `sample_uid`、算子参数和输出几何共同确定。
