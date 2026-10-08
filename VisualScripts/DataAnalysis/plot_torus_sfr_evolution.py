"""绘制 torus_100yr 模拟的瞬时恒星形成率随时间演化。"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyhdf.SD import SD, SDC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = Path(
    r"E:\科研项目与原始数据\ai+MACERsimulation\MACNet-v1"
    r"\RawData\DiskGalaxy\torus_100yr\torus_100yr"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "torus_100yr_sfr"

SFR_DATASET = "Data-Set-11"
SFR_TO_MSUN_PER_YR = 0.025
SNAPSHOT_INTERVAL_YR = 100.0
CORRUPTED_SNAPSHOT_GROUPS = ((747, 747), (754, 755), (763, 770))
CORRUPTED_SNAPSHOTS = frozenset(
    index
    for start, end in CORRUPTED_SNAPSHOT_GROUPS
    for index in range(start, end + 1)
)
TIME_PATTERN = re.compile(
    r"AT\s+TIME\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从原始 HDF4 snapshot 绘制 torus 瞬时 SFR 演化。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="包含 hdfra.* 的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图和 CSV 的输出目录。",
    )
    return parser.parse_args()


def snapshot_index(path: Path) -> int:
    suffix = path.name.rsplit(".", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"无法从文件名读取 snapshot 编号：{path.name}")
    return int(suffix)


def open_hdf4(path: Path) -> SD:
    """规避 Windows HDF4 C 库无法稳定处理中文绝对路径的问题。"""
    previous_directory = Path.cwd()
    try:
        os.chdir(path.parent)
        return SD(path.name, SDC.READ)
    finally:
        os.chdir(previous_directory)


def parse_time_yr(long_name: str, path: Path) -> float:
    match = TIME_PATTERN.search(long_name)
    if match is None:
        raise ValueError(f"{path.name} 的 long_name 不含 HDF 时间戳：{long_name!r}")
    time_gyr = float(match.group(1).replace("D", "E").replace("d", "e"))
    return time_gyr * 1.0e9


def read_snapshot_sfr(path: Path) -> tuple[float, float]:
    handle = open_hdf4(path)
    try:
        dataset = handle.select(SFR_DATASET)
        long_name = str(dataset.attributes().get("long_name", ""))
        if "SFR" not in long_name.upper():
            raise ValueError(
                f"{path.name} 的 {SFR_DATASET} 语义异常：long_name={long_name!r}"
            )
        values = np.asarray(dataset[:], dtype=np.float64)
    finally:
        handle.end()

    sfr_msun_per_yr = float(np.sum(values, dtype=np.float64) * SFR_TO_MSUN_PER_YR)
    if not np.isfinite(sfr_msun_per_yr) or sfr_msun_per_yr < 0.0:
        raise ValueError(f"{path.name} 的总瞬时 SFR 非有限或为负：{sfr_msun_per_yr}")
    return parse_time_yr(long_name, path), sfr_msun_per_yr


def load_evolution(input_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    files = sorted(input_dir.glob("hdfra.*"), key=snapshot_index)
    if not files:
        raise FileNotFoundError(f"没有在 {input_dir} 中找到 hdfra.*")

    indexed_files = {snapshot_index(path): path for path in files}
    first_index = min(indexed_files)
    last_index = max(indexed_files)
    indices = np.arange(first_index, last_index + 1, dtype=int)
    time_yr = indices.astype(np.float64) * SNAPSHOT_INTERVAL_YR
    sfr = np.full(indices.shape, np.nan, dtype=np.float64)
    status = ["missing"] * indices.size

    for position, index in enumerate(indices):
        if index in CORRUPTED_SNAPSHOTS:
            status[position] = "corrupted_excluded"
            continue
        path = indexed_files.get(int(index))
        if path is None:
            continue
        hdf_time_yr, snapshot_sfr = read_snapshot_sfr(path)
        expected_time_yr = index * SNAPSHOT_INTERVAL_YR
        if not np.isclose(hdf_time_yr, expected_time_yr, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"{path.name} 的 HDF 时间 {hdf_time_yr:g} yr 与编号推断时间 "
                f"{expected_time_yr:g} yr 不一致。"
            )
        time_yr[position] = hdf_time_yr
        sfr[position] = snapshot_sfr
        status[position] = "valid"

    return time_yr, sfr, status


def save_csv(
    output_path: Path,
    time_yr: np.ndarray,
    sfr: np.ndarray,
    status: list[str],
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["snapshot_index", "time_yr", "sfr_msun_per_yr", "status"]
        )
        for index, (snapshot_time, snapshot_sfr, snapshot_status) in enumerate(
            zip(time_yr, sfr, status, strict=True)
        ):
            writer.writerow(
                [
                    index,
                    f"{snapshot_time:.8g}",
                    "" if np.isnan(snapshot_sfr) else f"{snapshot_sfr:.12g}",
                    snapshot_status,
                ]
            )


def plot_evolution(
    output_path: Path,
    time_yr: np.ndarray,
    sfr: np.ndarray,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 320,
        }
    )

    time_kyr = time_yr / 1.0e3
    fig, axis = plt.subplots(figsize=(12.0, 6.2))
    axis.plot(
        time_kyr,
        sfr,
        color="#1565C0",
        linewidth=1.05,
        solid_capstyle="round",
        zorder=3,
    )

    # 用半个 snapshot 间隔扩展阴影，使单帧剔除在图中仍然可辨认。
    half_step_kyr = SNAPSHOT_INTERVAL_YR / 2.0e3
    for start, end in CORRUPTED_SNAPSHOT_GROUPS:
        axis.axvspan(
            start * SNAPSHOT_INTERVAL_YR / 1.0e3 - half_step_kyr,
            end * SNAPSHOT_INTERVAL_YR / 1.0e3 + half_step_kyr,
            color="#9E9E9E",
            alpha=0.32,
            linewidth=0.0,
            zorder=1,
        )

    axis.set_xlim(float(time_kyr.min()), 100.0)
    axis.set_ylim(0.0, float(np.nanmax(sfr)) * 1.08)
    axis.set_xlabel("Simulation time [kyr]")
    axis.set_ylabel(r"Instantaneous SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    axis.set_title("Torus simulation: instantaneous star-formation rate")
    axis.grid(True, which="major", color="#B0BEC5", linestyle="--", alpha=0.55)
    axis.grid(True, which="minor", color="#CFD8DC", linestyle=":", alpha=0.42)
    axis.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], color="#1565C0", linewidth=1.4, label="Instantaneous SFR"),
        Patch(
            facecolor="#9E9E9E",
            alpha=0.32,
            edgecolor="none",
            label="Corrupted snapshots excluded",
        ),
    ]
    axis.legend(handles=legend_handles, loc="upper left", frameon=True)
    axis.text(
        0.99,
        0.02,
        r"$\sum$ Data-Set-11 $\times\,0.025$; no smoothing or interpolation",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#455A64",
        fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    time_yr, sfr, status = load_evolution(input_dir)
    csv_path = output_dir / "torus_instantaneous_sfr.csv"
    figure_path = output_dir / "torus_instantaneous_sfr_evolution.png"
    save_csv(csv_path, time_yr, sfr, status)
    plot_evolution(figure_path, time_yr, sfr)

    valid_sfr = sfr[np.isfinite(sfr)]
    print(f"有效 snapshot：{valid_sfr.size}")
    print(f"剔除损坏 snapshot：{len(CORRUPTED_SNAPSHOTS)}")
    print(
        "SFR [Msun/yr]："
        f"min={valid_sfr.min():.6g}, median={np.median(valid_sfr):.6g}, "
        f"max={valid_sfr.max():.6g}"
    )
    print(f"图：{figure_path}")
    print(f"数据：{csv_path}")


if __name__ == "__main__":
    main()
