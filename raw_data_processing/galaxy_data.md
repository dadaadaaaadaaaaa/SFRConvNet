# `galaxy_data.py` 使用说明

## 1. 功能概述

`galaxy_data.py` 是 MACNet 数据预处理的总入口，负责把 MACER HDF4 snapshot、Parquet 日志、粗粒化算子和训练数据契约连接起来。其主要职责为：

1. 从 `hdfra.*` 读取 MACER 网格物理场；
2. 校验 `Data-Set-*` 的 `long_name`，防止数据集编号与物理语义错位；
3. 保存权威网格体积 `Data-Set-15`，计算网格气体质量；
4. 根据配置读取 `Data-Set-11`，得到全局瞬时 SFR 真值；
5. 调用 (R_h)、(R_m)-v0 或 (R_m)-v1 粗粒化；
6. 在 HR 或 LR 场上运行 MACER SFR 算子，构造 (Y_\alpha,B_\alpha,C_\alpha^{\rm true})；
7. 保存和校验 format-version 6 processed HDF5、metadata、UID 与守恒诊断；
8. 通过 `GalDataSet` 加载多个 snapshot，完成通道选择、标准化、分组与数据划分。

## 2. 核心数据语义

- `density`：质量密度；
- `gas_energy`：气体内能密度，不是单元总内能；
- `grid_volume`：MACER `Data-Set-15` 给出的单元体积；
- `gas_mass = density * grid_volume`；
- `sfr_target`：对 `Data-Set-11` 的所有网格值求和得到的 snapshot 全局瞬时 SFR；
- `sfr_macer_baseline`：在当前分辨率气体场上应用默认 MACER SFR 算子所得的总 SFR；
- `sfr_closure_true = sfr_target - sfr_macer_baseline`。

默认代码单位为：Gyr、kpc、`2.5e7 Msun`。

## 3. `GalData` 类

### 构造参数

```python
GalData(ndim=2, coordinate_mode="polar")
```

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `ndim` | `2` | 空间维数；当前 MACER2D 流程使用 2 |
| `coordinate_mode` | `polar` | `polar`：球极坐标网格；`cartesian`：直角坐标 |

### 重要属性

| 属性 | 含义 |
|---|---|
| `snapshot` | 字段名到二维数组的映射 |
| `r`、`theta`、`phi` | 极坐标；二维数据无独立 `phi` 网格 |
| `x`、`y`、`z` | 坐标转换结果；二维模式下访问 `z` 会报错 |
| `grid_volume` | 权威单元体积数组 |
| `gas_mass` | 每格气体质量数组 |
| `sfr_target` | 当前 snapshot 的全局瞬时 SFR |
| `sfr_macer_baseline` | canonical 数据兼容 SFR baseline |
| `sfr_closure_true` | signed closure 真值 |
| `sample_uid` | 当前 processed 产品身份 |
| `parent_hr_uid` | 来源 HR snapshot 身份 |

### `read_hdfra(...)`

```python
read_hdfra(
    hdfra_path,
    time,
    log_path,
    func_parse_log,
    scope,
    target_config=None,
    source_metadata=None,
)
```

读取 HDF4 snapshot 和对应日志上下文。`target_config` 决定训练目标数据集和归约方式；盘星系当前应为 `Data-Set-11` 的 `sum`。

### `compute_macer_sfr_products(time_step=None, context_selection=None)`

运行两套 SFR 算子并保存监督产品。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `time_step` | `None` | `None` 时优先使用日志 `dt`；缺失或非正时使用连续极限 `0` |
| `context_selection` | `None` | `None` 时根据日志诊断自动写入窗口平均/插值/端点语义 |

canonical baseline 使用 `data_compatible` 模式；`paper_physical` 作为物理对照保存。

### `coarsen(operator, new_shape, ...)`

统一粗粒化入口。

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `operator` | 无 | `spatial_block`/`rh`；`axisymmetric_rm_v0`/`rm_v0`；`axisymmetric_rm_v1`/`rm_v1` |
| `new_shape` | 无 | `(N_theta, N_r)`，例如 `(8,8)` |
| `conservation_rtol` | `1e-6` | 守恒相对容差 |
| `conservation_atol` | `1e-10` | 守恒绝对容差 |
| `operator_kwargs` | 空 | 传递给具体粗粒化算子的参数 |

兼容接口：

- `rescale(new_shape, ...)`：调用空间块 (R_h)；
- `rescale_mass_adaptive(...)`：调用 (R_m)-v0，默认 `kernel="top_hat"`、`initial_neighbors=32`、`balance_tolerance=1e-6`、`balance_max_iterations=5000`。

### HDF5 I/O

- `save_h5(file_path)`：保存 snapshot、坐标、体积、日志、粗粒化诊断、metadata 和 `/sfr_supervision`；默认拒绝不完整的物理产品；
- `load_h5(file_path)`：读取并重新验证 UID、字段语义和 (Y=B+C)。

## 4. `GalDataSet` 类

`GalDataSet` 用于把多个 processed HDF5 组合成机器学习样本集合。

### 主要方法及参数

| 方法 | 关键参数与默认值 | 含义 |
|---|---|---|
| `load_data(folder_conf, ...)` | 以目录和组名为主 | 按确定性文件顺序读取一个运行的全部 snapshot |
| `select_input_channels(include=None, exclude=None)` | 两者默认 `None` | 显式保留或排除输入通道；不能同时使用含义冲突的列表 |
| `drop_invalid(...)` | 见函数签名 | 删除 NaN/Inf 等无效样本；是否要求正 SFR 可配置 |
| `exclude_raw_snapshot_indices(indices)` | 无默认索引 | 按原始 snapshot index 精确排除初始化输出 |
| `filter(threshold)` | 无默认值 | 保留目标大于阈值的样本；不建议用它代替初始化索引排除 |
| `mirror_data()` | 无参数 | 旧版镜像兼容；MACER2D 正式 SFR 训练不应使用 |
| `split(train_size, validation_size=0.0, ...)` | `train_size` 必填；验证比例默认 0 | 支持分层和按 `parent_hr_uid` 绑定派生样本；`prepare_data.py` 当前显式传入 0.6/0.2，得到 60%/20%/20%；随机时间切分仍可能产生相邻帧泄漏 |
| `standardize(mean=None, std=None, log_transform=True)` | 自动用当前集合统计量 | 通道标准化；验证和测试必须复用训练集统计量 |
| `balance_groups(method="oversample")` | `oversample` | 对不同运行分组过采样平衡 |

`log_mass_filter` 只允许用于具有正有限 `mass_filter` 的完整 (R_m) 数据集。固定分辨率 (R_h) 不应伪造该量。

## 5. 命令行参数

脚本主程序会按 `.config` 批量把原始 HDF4 转为 fine 和 coarse HDF5。

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `--galaxy` | 全部 | 指定生成名，如 `disk_galaxy_fiducial`；可重复 |
| `--coarsener` | `spatial_block` | `spatial_block`、`axisymmetric_rm_v0`、`axisymmetric_rm_v1` |
| `--coarse-shape` | `8 8` | 输出 `(N_theta,N_r)` |
| `--mass-filter` | `None` | (R_m) 必填，单位为 MACER 质量代码单位 |
| `--kernel` | `None` | v0 可选 `top_hat`、`cubic_spline`；v1 固定 `wendland_c2` |
| `--coarse-dirname` | 自动 | (R_h) 默认 `coarse`；(R_m) 默认 `coarse_<operator>` |
| `--initial-neighbors` | 算子默认 | (R_m) 初始邻居数 |
| `--balance-tolerance` | `1e-6` | 输运平衡容差 |
| `--balance-max-iterations` | `30000` | v0 IPFP 最大迭代；v1 忽略 |
| `--v1-max-neighbors` | `4096` | v1 自适应邻居上限 |
| `--v1-max-runtime-seconds` | `60.0` | v1 单 snapshot 内部运行时限 |
| `--v1-solver-max-iterations` | `300` | v1 对偶求解最大迭代 |
| `--v1-polish-max-iterations` | `500` | v1 IPFP polish 最大迭代 |

## 6. 执行命令

固定空间块 (R_h)，生成 `8x8` 数据：

```powershell
python raw_data_processing/galaxy_data.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener spatial_block `
  --coarse-shape 8 8 `
  --coarse-dirname coarse_rh_8x8_sfr_v6
```

(R_m)-v0 示例：

```powershell
python raw_data_processing/galaxy_data.py `
  --galaxy disk_galaxy_fiducial `
  --coarsener axisymmetric_rm_v0 `
  --coarse-shape 8 8 `
  --mass-filter 1.0 `
  --kernel top_hat `
  --coarse-dirname coarse_rm_v0_m1
```

查看参数：

```powershell
python raw_data_processing/galaxy_data.py --help
```

## 7. 运行前检查

1. 必须从项目根目录运行，使 `.config` 可被正确发现；
2. HDF4 读取环境必须安装 `pyhdf`；
3. `Data-Set-5`、`10`、`15`、`17`、加速度字段及目标字段的 `long_name` 必须与配置一致；
4. 不要把不同粗粒化算子或不同物理参数的 HDF5 静默混入同一训练目录；
5. 批量生成前先对单 snapshot 检查守恒、SFR supervision 和 HDF5 round-trip。
