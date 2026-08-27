'''
1. 使fortran格式的日志文件与python兼容
2. 读取config文件，并保存为parquet文件
3. 实现物理量的平滑处理（平均值计算）
4. 统一输出为mdot_macer
'''
import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from typing import List, Iterable
from pathlib import Path


# ======== Low-level utilities ========：底层工具
_FORTRAN_EXP_RE = re.compile(r"(\d+(?:\.\d+)?)([+-]\d+)") # 许多代码用fortran编写，输出的格式与python默认识别的科学计数法格式不同，因此需要转化一下


# def fix_fortran_number(token: str) -> str:
#     """Convert Fortran-style scientific notation to standard float notation."""
#     token = token.replace("D", "E")
#     return _FORTRAN_EXP_RE.sub(r"\1E\2", token)


def detect_fortran_style(path: str, max_lines: int = 100) -> bool: # 快速判断一个日志文件是不是fortran格式
    """Detect whether a file uses Fortran-style scientific notation."""
    with open(path) as f:
        for _, line in zip(range(max_lines), f):
            if "D" in line or _FORTRAN_EXP_RE.search(line):
                return True
    return False


def load_config(config_dir: str = ".config"): # 加载并解析config文件
    config_path = Path(config_dir)
    configs = {}
    if not config_path.exists():
        warnings.warn(f"Cannot find folder {config_path}")

    for json_file in config_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                configs[json_file.stem] = json.load(f)
        except Exception as e:
            warnings.warn(f"Error reading file {json_file.name}: {e}")

    return configs


# ======== Parsing ========


def _parse_fortran_log(path: str, colnames: List[str]) -> pd.DataFrame: # 解析fortran格式日志文件
    with open(path, "r") as f:
        content = f.read()

    # Global replacement of Fortran-style numbers，全局fortran替换
    content = content.replace("D", "E")
    content = _FORTRAN_EXP_RE.sub(r"\1E\2", content)

    # Exist '=', it indicates a "header = values" format that may span multiple lines
    # 在某些代码中，每一行数据前面可能会带一个像 time= 0.123 mass= 4.56 这样的表头，代码用=来分割，把整个dataframe切成一块块segment
    if "=" in content:
        segments = content.split("=")
        rows = []
        ncols_expected = len(colnames)

        for i, seg in enumerate(segments):
            if i == 0:
                continue  # The part before the first '=' is the header, skip

            tokens = seg.split()
            # Take only the first ncols_expected tokens as data
            if len(tokens) >= ncols_expected:
                try:
                    rows.append([float(t) for t in tokens[:ncols_expected]])
                except ValueError:
                    continue
    else:
        # Not exist '=', it is a plain table format
        # 如果没有=，则跳过空行和#，然后按行切分
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(t) for t in line.split()])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid data rows found in {path}")

    ncols = max(len(r) for r in rows)
    columns = _normalize_colnames(colnames, ncols)

    return pd.DataFrame(rows, columns=columns)


def _parse_plain_log(path: str, colnames: List[str]) -> pd.DataFrame: # 解析普通格式日志文件
    return pd.read_csv(
        path,
        sep=r"\s+",
        names=colnames,
        comment="#",
        engine="python",
    )


def _normalize_colnames(colnames: List[str], ncols: int) -> List[str]: # 如果列名数量小于预期，则自动补齐
    if len(colnames) < ncols:
        return colnames + [f"col{i}" for i in range(len(colnames), ncols)]
    return colnames[:ncols]


def parse_logfile(
    path: str, colnames: List[str], save_path: str | None = None, force_parse: bool = False
) -> pd.DataFrame:
    """
    Parse a log file into a DataFrame, optionally cached as parquet.
    """
    if save_path and os.path.exists(save_path) and not force_parse: # 如果已经存在parquet缓存文件且没有要求强制重新解析，那么直接读取parquet文件。这样能够节省大量时间。
        return pd.read_parquet(save_path)

    if not os.path.exists(path): 
        raise FileNotFoundError(path)

    is_fortran = detect_fortran_style(path)
    parser = _parse_fortran_log if is_fortran else _parse_plain_log # 根据是否是fortran格式选择不同的解析方法
    df = parser(path, colnames)

    if save_path: # 如果要求保存parquet文件，则创建文件夹并保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)

    return df


# ======== Time-window Analysis ========
"""
在模拟中，黑洞吸积率往往是一个剧烈波动的物理量。如果只提取其中一个snapshot的值，可能会捕捉到异常的波峰或者波谷。因此需要对数据进行平滑处理。
对于指定的时间帧time，提取time-scope~time+scope范围内的所有数据，然后计算平均值
最后再强制把时间设回目标时间
"""


def mean_around_time(df: pd.DataFrame, time: float, scope: float, time_col: str = "time") -> pd.Series:
    if time_col not in df.columns:
        raise KeyError(f"Missing required column '{time_col}'")

    mask = (df[time_col] >= time - scope) & (df[time_col] <= time + scope)
    avg = df.loc[mask].mean()
    avg[time_col] = time
    return avg


LOG_CONTEXT_WINDOW_MEAN = 0
LOG_CONTEXT_LINEAR_INTERPOLATION = 1
LOG_CONTEXT_NEAREST_ENDPOINT = 2


def mean_around_time_with_linear_fallback(
    df: pd.DataFrame,
    time: float,
    scope: float,
    time_col: str = "time",
) -> pd.Series:
    """窗口非空时取均值，空窗口时对双侧日志做线性插值。

    该函数不会扩大物理时间窗口。只有目标时间落在日志范围以外时，才退化为
    最近端点；选择方式和时间支撑以纯数值诊断字段随结果返回，便于 HDF5 和
    supervision metadata 无损持久化。
    """

    if time_col not in df.columns:
        raise KeyError(f"Missing required column '{time_col}'")
    time = float(time)
    scope = float(scope)
    if not np.isfinite(time) or not np.isfinite(scope) or scope < 0:
        raise ValueError("time and scope must be finite, with scope >= 0.")

    log_time = pd.to_numeric(df[time_col], errors="coerce")
    finite_time = np.isfinite(log_time.to_numpy(dtype=np.float64))
    if not np.any(finite_time):
        raise ValueError("Log data contains no finite time values.")
    valid = df.loc[finite_time].copy()
    valid[time_col] = log_time.loc[finite_time].to_numpy(dtype=np.float64)

    mask = (valid[time_col] >= time - scope) & (valid[time_col] <= time + scope)
    window = valid.loc[mask]
    if not window.empty:
        result = window.mean(numeric_only=True)
        selected_time = window[time_col].to_numpy(dtype=np.float64)
        result[time_col] = time
        result["log_context_selection_code"] = LOG_CONTEXT_WINDOW_MEAN
        result["log_context_window_count"] = int(len(window))
        result["log_context_left_time"] = float(np.min(selected_time))
        result["log_context_right_time"] = float(np.max(selected_time))
        result["log_context_bracket_width"] = float(
            np.max(selected_time) - np.min(selected_time)
        )
        result["log_context_interpolation_weight"] = 0.0
        return result

    # 同一时刻可能有多条日志。空窗口回退只需要连续数值上下文，因此先按时间
    # 聚合重复记录，避免零宽 bracket 和不确定的左右行选择。
    numeric_columns = list(valid.select_dtypes(include=[np.number]).columns)
    if time_col not in numeric_columns:
        numeric_columns.append(time_col)
    timeline = (
        valid[numeric_columns]
        .groupby(time_col, as_index=False, sort=True)
        .mean(numeric_only=True)
        .sort_values(time_col, kind="mergesort")
        .reset_index(drop=True)
    )
    times = timeline[time_col].to_numpy(dtype=np.float64)
    right_index = int(np.searchsorted(times, time, side="left"))

    if 0 < right_index < len(timeline):
        left = timeline.iloc[right_index - 1]
        right = timeline.iloc[right_index]
        left_time = float(left[time_col])
        right_time = float(right[time_col])
        width = right_time - left_time
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Cannot interpolate log context across a non-positive time bracket.")
        weight = (time - left_time) / width
        result = left + weight * (right - left)
        result[time_col] = time
        result["log_context_selection_code"] = LOG_CONTEXT_LINEAR_INTERPOLATION
        result["log_context_window_count"] = 0
        result["log_context_left_time"] = left_time
        result["log_context_right_time"] = right_time
        result["log_context_bracket_width"] = width
        result["log_context_interpolation_weight"] = float(weight)
        return result

    endpoint_index = 0 if right_index <= 0 else len(timeline) - 1
    endpoint = timeline.iloc[endpoint_index].copy()
    endpoint_time = float(endpoint[time_col])
    endpoint[time_col] = time
    endpoint["log_context_selection_code"] = LOG_CONTEXT_NEAREST_ENDPOINT
    endpoint["log_context_window_count"] = 0
    endpoint["log_context_left_time"] = endpoint_time
    endpoint["log_context_right_time"] = endpoint_time
    endpoint["log_context_bracket_width"] = 0.0
    endpoint["log_context_interpolation_weight"] = 0.0
    return endpoint


def _require_columns(data: pd.Series | pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in data]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# ======== Physics-specific Helpers ========


# def ll_eg(path: str, time: float, scope: float) -> pd.Series: # 针对椭圆星系，黑洞吸积率=爱丁顿吸积率*吸积率比例
#     df = pd.read_parquet(path)

#     avg = mean_around_time(df, time, scope)
#     _require_columns(avg, ["mdot_edd", "mdot_ratio"])

#     avg["mdot_macer"] = avg["mdot_edd"] * avg["mdot_ratio"]
#     return avg


# def ll_dg(path: str, time: float, scope: float) -> pd.Series: # 针对盘星系，黑洞吸积率直接由日志文件平均后给出，同时计算了爱丁顿吸积率
#     df = pd.read_parquet(path)

#     avg = mean_around_time(df, time, scope)
#     _require_columns(avg, ["mdot_bh", "mbh"])

#     avg["mdot_macer"] = avg["mdot_bh"]

#     # constants
#     G = 112
#     m_p = 3.365e-65
#     eta = 0.1
#     c = 3.07e5
#     sigma_t = 6.99e-68

#     avg["mdot_edd"] = (4 * np.pi * G * avg["mbh"] * m_p) / (eta * c * sigma_t)

#     return avg

def ll_eg(path: str, time: float, scope: float) -> pd.Series: # 读取日志文件中已经计算好的恒星形成率SFR，直接平均后给出
    df = pd.read_parquet(path)
    avg = mean_around_time(df, time, scope)
    _require_columns(avg, ["sfr_tot"])
    avg["sfr_target"] = avg["sfr_tot"]
    return avg

def ll_dg(path: str, time: float, scope: float) -> pd.Series: 
    df = pd.read_parquet(path)
    avg = mean_around_time_with_linear_fallback(df, time, scope)
    _require_columns(avg, ["NEWstar", "mbh"])
    # NEWstar is the cumulative newly formed stellar mass, not an
    # instantaneous SFR. Keep it for diagnostics, while the training target is
    # read independently from the snapshot target configured in HdfraConfig.
    avg["cumulative_new_stellar_mass"] = avg["NEWstar"]
    return avg

if __name__ == "__main__":
    force_parse = True

    configs = load_config()
    raw_data_dir = configs["BaseConfig"]["raw_data_dir"]
    df = {}
    for gal_type, gal_group in configs["RawDataConfig"].items():
        print(f"Processing galaxy group: {gal_type}")
        for gal_nickname, gal_config in gal_group.items():
            gal_name = f"{gal_type}_{gal_nickname}"
            print(f"  Parsing galaxy: {gal_name}")

            path = os.path.join(raw_data_dir, gal_config["folder_name"], gal_config["log_file"])
            
            if not os.path.exists(path):
                print(f"  [Warning] Log file not found for {gal_name} at {path}. Skipping...")
                continue
                
            col_names = configs["HdfraConfig"][gal_type]["colnames"]
            save_path = os.path.join(configs["BaseConfig"]["data_dir"], f"{gal_name}.parquet")

            df[gal_name] = parse_logfile(path=path, colnames=col_names, save_path=save_path, force_parse=force_parse)
    print("Parsing completed.")
