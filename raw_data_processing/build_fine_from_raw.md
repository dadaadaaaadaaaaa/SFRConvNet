# `build_fine_from_raw.py` 使用说明

## MACER3D 入口（2026-09-10 新增）

已支持 `RawDataConfig.dwarf_galaxy.fullFB_inflow`，直接读取 ATHDF 的物理时间，不需要先 parse hst。
默认 `--prediction-mode direct`；Residual fine 保存冷却和静态径向引力，能同时用于构建两种 coarse。

```powershell
python raw_data_processing/build_fine_from_raw.py --galaxy dwarf_galaxy_fullFB_inflow --prediction-mode residual --fine-dirname fine_residual
```

抽样可加 `--snapshot-index 51 --snapshot-index 254 --snapshot-index 299`，或 `--max-snapshots 3`。
如仅需要 Direct，将模式改为 `direct` 并使用独立的 `fine_direct` 子目录。
不同模式/源哈希/参数签名不能覆盖同一个三维输出文件；相同签名允许重新生成。
三维文件使用 `macnet_3d_format=1`，不要将其直接投入二维 GalDataSet 或现有 CNN 训练。
完整字段与守恒规则见根目录 `3DSFRConvnet.md` 的第 20 节；下文保留原二维说明。

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

二维流程默认会覆盖同名输出，这与旧 `galaxy_data.py` 主入口一致。三维流程默认先做
轻量契约核对并跳过匹配产品；不匹配的同名文件不会被覆盖。正式重建前仍应备份已有
fine 目录，抽样测试时建议用独立的 `--fine-dirname`。

## 3. 命令行参数

| 参数 | 默认值 | 可选值与含义 |
|---|---:|---|
| `--galaxy` | 全部 HDFRA 运行 | 生成名，例如 `disk_galaxy_fiducial`；可重复 |
| `--config-dir` | `.config` | 包含 `BaseConfig.json`、`RawDataConfig.json`、`HdfraConfig.json` 的目录 |
| `--fine-dirname` | `fine` | `<data_dir>/<galaxy_name>/` 下的单层输出目录名 |
| `--prediction-mode` | `direct` | 三维产品使用 `direct` 或 `residual`；二维行为不变 |
| `--source-hash` | `stat` | 三维源身份使用 `stat`（大小+mtime，不额外通读）或显式 `sha256` |
| `--no-resume` | 未启用 | 禁止跳过已有三维产品；同名输出仍受产品签名检查保护 |
| `--snapshot-index` | 全部 | 只处理指定逻辑/原始 snapshot index；可重复 |
| `--exclude-snapshot` | 未提供 | 按逻辑/原始 snapshot index 排除，可重复；一旦提供，显式清单取代“跳过排序后前 46 帧”的默认规则 |
| `--max-snapshots` | `None` | 每个星系最多处理前 (N) 个合格 snapshot；仅建议用于 smoke test |

`--snapshot-index` 使用 metadata 中的逻辑/原始 index，而不是当前目录枚举位置。
`filename_index` 模式会从 `hdfra.<index>` 恢复该 index，缺失文件不会让后续物理时间错位。

三个分段扁平化的 Fiducial 系列还使用 `RawDataConfig.time_mapping`。其 index 0 是
时间为 0 的特殊初始帧；index 1 起包含 0.5625 Gyr 的已有松弛时间，并且原始重启段
每段推进 1000 个时间步、扁平文件名每段只容纳 999 个正编号。当前绝对时间为：

```text
index = 0:  t = 0
index > 0:  t = positive_index_offset
                 + [index + floor((index-1)/segment_cycle_size)
                    * extra_time_steps_per_segment] * snapshot_dt
```

该绝对时间用于选择日志中的 `mbh` 与内部流体 `dt`，并写入 fine metadata；它不同于
为了作图而选择的 production-time 零点。读取 HDF4 后，程序还会按 `long_name` 的实际
打印精度核验 `Data-Set-11` 的 `AT TIME=`，不一致时立即停止，不再静默生成错配产品。

三维批处理默认安全续跑：若目标 HDF5 已存在，只读取其 metadata 契约并核对星系、
snapshot、源文件名/大小、模式、物理参数、`input.txt` 和静态引力来源。匹配则跳过，
不加载三维数组；不匹配或文件不完整则停止，避免把不同产品混入同一目录。默认
`stat` 身份不会为每个约数百 MB 的 ATHDF 再做一次全文件 I/O。若数据需要跨机器
长期归档或怀疑内容被原地篡改，可在首次构建新目录时指定 `--source-hash sha256`。

二维 HDFRA 在未提供 `--exclude-snapshot` 时，默认按文件排序位置跳过前 46 个
snapshot。只要显式提供至少一个 `--exclude-snapshot`，默认规则就不再执行，脚本只排除
用户给出的原始/逻辑 snapshot index。包含清单与排除清单有交集时会停止，不会猜测优先级。
该参数在读取 HDF4 前生效；当前不用于 ATHDF 三维入口。

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

构建 torus fine 数据并仅排除已确认损坏的 11 帧：

```powershell
python raw_data_processing/parse_log_file.py `
  --galaxy disk_galaxy_torus_100yr

python raw_data_processing/build_fine_from_raw.py `
  --galaxy disk_galaxy_torus_100yr `
  --exclude-snapshot 747 `
  --exclude-snapshot 754 `
  --exclude-snapshot 755 `
  --exclude-snapshot 763 `
  --exclude-snapshot 764 `
  --exclude-snapshot 765 `
  --exclude-snapshot 766 `
  --exclude-snapshot 767 `
  --exclude-snapshot 768 `
  --exclude-snapshot 769 `
  --exclude-snapshot 770
```

此命令会保留 torus 的 0–45 帧；代码不会自动再生成“含/不含初始帧”两套产品。

## 5. 运行前检查与失败语义

1. 使用安装了 `pyhdf`、`h5py`、`pandas` 和 Parquet 引擎的 Python 环境；
2. 先运行 `parse_log_file.py`，确保 `<data_dir>/<galaxy_name>.parquet` 已存在；
3. 配置中的目标 Data-Set 不得同时列入输入 `correspondence`，否则会直接报标签泄漏；
4. Data-Set-5/8/9/10/11/15/17 的 `long_name` 不符合预期时会停止，不猜测字段编号；
5. Data-Set-11 的 HDF 时间标签与配置时间不一致时会停止，拒绝错误匹配日志；
6. `Data-Set-15` 必须为有限正体积，且二维形状与气体场一致；
7. Windows 下即使 RawData 路径含中文或与 workspace 跨盘，代码也会临时进入源目录，
   只把 ASCII `hdfra.<index>` 文件名交给 pyhdf；打开后立即恢复原工作目录；
8. 完整 `filename_index` 批处理会拒绝输出目录中不属于当前输入序列的旧 HDF5；抽样模式不执行该目录级 stale 检查。

## 6. 与旧入口的关系

`galaxy_data.py` 仍保留类定义和旧的一体化命令行入口，以免破坏已有调用；新工作流推荐：

```text
build_fine_from_raw.py（只运行一次）
                  ↓
build_coarse_from_fine.py（可反复改变算子/分辨率）
```

这种拆分不会改变 fine 数值结果，但可以避免每次粗粒化实验都重新读取 HDF4、重新解析
日志和重新计算 HR metadata。

## 7. 已有二维 fine 的时间修复迁移

若已有完整 format-version 6 fine，且问题仅为绝对时间和日志上下文错配，可以使用
[migrate_fine_time_context.py](migrate_fine_time_context.md) 迁移到新目录，不必全量重读
raw 物理数组。迁移仍需要原始目录清单和 Parquet 日志；保留物理场和真实 Y，重新匹配
日志并重算两套 B/C、context 和物理产品 UID。不能只改时间属性，也不能从旧 fine 直接
粗化来修复。迁移默认拒绝覆盖，并支持严格匹配的续跑。完整命令、适用前提和真实 smoke
记录见上述说明；旧 fine 不完整或物理数组有问题时仍应从 raw 重建。
