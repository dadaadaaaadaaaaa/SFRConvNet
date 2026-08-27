# `parse_log_file.py` 使用说明

## 1. 功能概述

`parse_log_file.py` 负责把 MACER 文本日志转换为 Python 可以稳定读取的表格，并为 snapshot 提供对应时刻的全局物理上下文。主要功能包括：

1. 识别普通科学计数法和 Fortran 风格科学计数法，例如 `1.0D+03`、`1.0+03`；
2. 按 `.config/HdfraConfig.json` 中的列名解析日志；
3. 可将解析结果缓存为 Parquet 文件；
4. 在目标 snapshot 时间附近提取日志上下文；
5. 盘星系日志窗口为空时，使用相邻日志点线性插值，并记录插值诊断；
6. 为椭圆星系构造日志 SFR，为盘星系保留累计新生恒星质量和黑洞质量等上下文。

注意：盘星系训练真值不由日志中的 `NEWstar` 给出，而是由 snapshot 的 `Data-Set-11` 给出。`NEWstar` 是累计新生恒星质量，不是瞬时 SFR。

## 2. 主要接口

### `detect_fortran_style(path, max_lines=100)`

检查文件前若干行是否出现 Fortran 风格指数。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `path` | 无 | 日志文件路径 |
| `max_lines` | `100` | 最多检查的行数 |

### `load_config(config_dir=".config")`

读取指定目录内的全部 JSON 文件，返回以文件名 stem 为键的字典。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `config_dir` | `.config` | 配置目录；一般应从项目根目录调用 |

### `parse_logfile(path, colnames, save_path=None, force_parse=False)`

解析日志的统一入口。

| 参数 | 默认值 | 可选值/含义 |
|---|---:|---|
| `path` | 无 | 原始文本日志路径 |
| `colnames` | 无 | 按日志列顺序排列的列名列表 |
| `save_path` | `None` | Parquet 输出路径；为 `None` 时不保存 |
| `force_parse` | `False` | `False`：若 Parquet 已存在则直接读取；`True`：忽略旧缓存并重新解析 |

### `mean_around_time(df, time, scope, time_col="time")`

选择闭区间 `[time-scope, time+scope]` 内的日志行，对数值列取平均，并把结果中的时间重设为目标 `time`。

### `mean_around_time_with_linear_fallback(df, time, scope, time_col="time")`

用于盘星系的更严格时间匹配：

- 窗口内有日志：取窗口平均；
- 窗口为空且目标时刻位于日志范围内：使用左右日志点线性插值；
- 目标时刻位于日志范围外：使用最近端点。

返回值中会加入：

- `log_context_selection_code`：`0` 为窗口平均，`1` 为线性插值，`2` 为范围外最近端点；
- `log_context_window_count`；
- `log_context_left_time`、`log_context_right_time`；
- `log_context_bracket_width`；
- `log_context_interpolation_weight`。

### `ll_eg(path, time, scope)`

读取椭圆星系 Parquet 日志，在时间窗口内平均，并令：

```text
sfr_target = sfr_tot
```

### `ll_dg(path, time, scope)`

读取盘星系 Parquet 日志，使用带插值回退的时间匹配，并令：

```text
cumulative_new_stellar_mass = NEWstar
```

它不设置训练用 `sfr_target`；盘星系目标由 HDF snapshot 的 `Data-Set-11` 独立读取。

## 3. 配置依赖

直接执行脚本时读取：

- `.config/BaseConfig.json`：原始数据和 processed 数据目录；
- `.config/RawDataConfig.json`：各模拟运行的文件夹名和日志文件名；
- `.config/HdfraConfig.json`：日志列名。

脚本目前把 `force_parse=True` 写在主程序中，因此直接执行时会重新解析已有日志并覆盖相应 Parquet 缓存。

## 4. 执行命令

从项目根目录执行全部配置中的日志转换：

```powershell
python raw_data_processing/parse_log_file.py
```

查看或调用单个接口：

```python
from raw_data_processing.parse_log_file import parse_logfile

df = parse_logfile(
    path=r"D:\path\to\global.log",
    colnames=["time", "dt", "mbh", "NEWstar"],
    save_path=r"data\disk_galaxy_example.parquet",
    force_parse=False,
)
```

该脚本没有命令行参数；需要改变是否强制解析、文件路径或列名时，应修改配置文件或通过 Python 接口调用。

## 5. 输出与注意事项

- 输出 Parquet 文件位于 `BaseConfig.data_dir/<galaxy_name>.parquet`；
- 时间、质量和速率沿用 MACER 日志的代码单位；
- 窗口平均会平滑快速变化的 `mbh`、`dt` 等量；若要复现某一内部时间步的瞬时量，必须评估这种平滑是否合适；
- 解析器会跳过无法转换为浮点数的无效行，正式批处理后应检查行数和时间覆盖范围。
