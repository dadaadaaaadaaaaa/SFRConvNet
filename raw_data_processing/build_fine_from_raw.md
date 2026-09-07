# `build_fine_from_raw.py` 使用说明

## 1. 功能与边界

`build_fine_from_raw.py` 是 MACER HDFRA 原始数据到 processed fine HDF5 的专用入口：

```text
raw hdfra.* + 已解析 Parquet 日志 + .config
                         ↓
        format-version 6 fine HDF5
```

本脚本只负责 `raw → fine`，不做空间或质量粗粒化。粗粒化应继续运行
`build_coarse_from_fine.py`。这样原始 HDF4 只读取一次；后续尝试不同输出分辨率、
不同 \(R_h/R_m\) 参数时直接复用 fine HDF5。

底层物理与 I/O 没有另写一份实现，而是调用 `galaxy_data.py` 中的：

- `GalData.set_corr()`：建立 `Data-Set-*` 到物理字段名的映射；
- `GalData.read_hdfra()`：读取 HDF4、日志和 Data-Set-11 目标；
- `GalData.set_coord()`：读取 `fakeDim2/fakeDim1` 坐标和 Data-Set-15 体积；
- `GalData.save_h5()`：保存字段、metadata、UID 和 SFR supervision。

因此 `density`、`gas_energy`、`cooling_time`、加速度、体积和 SFR 的物理语义仍以
`galaxy_data.md`、`macer_sfr_operator.py` 与 `.config/HdfraConfig.json` 为准。

## 2. 输出内容

默认输出到：

```text
<data_dir>/<galaxy_name>/fine/<snapshot_index:05d>.h5
```

每个新文件应包含：

- `/snapshot`：气体场、温度、冷却率密度和配置中声明的其他字段；
- `/coordinates`：(r,\theta) 中心及边界；
- `/geometry/cell_volume`：Data-Set-15 权威单元体积；
- `/log_data`：与 snapshot 时间匹配的日志上下文及 Data-Set-11 总 SFR；
- `/metadata/contract_json`：完整来源身份、几何、字段和 `identity_fine` 推导契约；
- `/sfr_supervision`：HR 上的 (Y,B,C^{\rm true}=Y-B) 与两套 MACER SFR 算子信息。

默认会覆盖同名输出，这与旧 `galaxy_data.py` 主入口一致。正式重建前应备份已有 fine
目录，抽样测试时建议用独立的 `--fine-dirname`。

## 3. 命令行参数

| 参数 | 默认值 | 可选值与含义 |
|---|---:|---|
| `--galaxy` | 全部 HDFRA 运行 | 生成名，例如 `disk_galaxy_fiducial`；可重复 |
| `--config-dir` | `.config` | 包含 `BaseConfig.json`、`RawDataConfig.json`、`HdfraConfig.json` 的目录 |
| `--fine-dirname` | `fine` | `<data_dir>/<galaxy_name>/` 下的单层输出目录名 |
| `--snapshot-index` | 全部 | 只处理指定逻辑/原始 snapshot index；可重复 |
| `--max-snapshots` | `None` | 每个星系最多处理前 (N) 个合格 snapshot；仅建议用于 smoke test |

`--snapshot-index` 使用 metadata 中的逻辑/原始 index，而不是当前目录枚举位置。
`filename_index` 模式会从 `hdfra.<index>` 恢复该 index，缺失文件不会让后续物理时间错位。

对于旧配置中需要跳过初始化阶段的运行，仍保持旧规则：除列入
`_STABLE_FROM_START` 的四个盘星系运行外，按排序位置排除前 46 个 snapshot。

## 4. 执行命令

查看帮助：

```powershell
python raw_data_processing/build_fine_from_raw.py --help
```

构建单个星系的全部 fine 数据：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_fiducial
```

只检查两个明确的 snapshot，并写入独立目录：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_fiducial `
  --snapshot-index 100 `
  --snapshot-index 5000 `
  --fine-dirname fine_smoke
```

每个选中运行只处理前两个合格 snapshot：

```powershell
python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_fiducial `
  --max-snapshots 2 `
  --fine-dirname fine_smoke
```

## 5. 运行前检查与失败语义

1. 使用安装了 `pyhdf`、`h5py`、`pandas` 和 Parquet 引擎的 Python 环境；
2. 先运行 `parse_log_file.py`，确保 `<data_dir>/<galaxy_name>.parquet` 已存在；
3. 配置中的目标 Data-Set 不得同时列入输入 `correspondence`，否则会直接报标签泄漏；
4. Data-Set-5/8/9/10/11/15/17 的 `long_name` 不符合预期时会停止，不猜测字段编号；
5. `Data-Set-15` 必须为有限正体积，且二维形状与气体场一致；
6. Windows 下即使 RawData 路径含中文或与 workspace 跨盘，代码也会临时进入源目录，
   只把 ASCII `hdfra.<index>` 文件名交给 pyhdf；打开后立即恢复原工作目录；
7. 完整 `filename_index` 批处理会拒绝输出目录中不属于当前输入序列的旧 HDF5；抽样模式不执行该目录级 stale 检查。

## 6. 与旧入口的关系

`galaxy_data.py` 仍保留类定义和旧的一体化命令行入口，以免破坏已有调用；新工作流推荐：

```text
build_fine_from_raw.py（只运行一次）
                  ↓
build_coarse_from_fine.py（可反复改变算子/分辨率）
```

这种拆分不会改变 fine 数值结果，但可以避免每次粗粒化实验都重新读取 HDF4、重新解析
日志和重新计算 HR metadata。
