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
    avg = mean_around_time(df, time, scope)
    _require_columns(avg, ["NEWstar", "mbh"])
    avg["sfr_target"] = avg["NEWstar"] 
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
