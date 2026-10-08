"""分别绘制 torus_100yr 的气体密度和黑洞吸积率演化。"""

from __future__ import annotations

import argparse
import csv
import os
import re
from array import array
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
DEFAULT_LOG_PATH = DEFAULT_INPUT_DIR.parent / "torus_100yr.log"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results" / "torus_100yr_diagnostics"

DENSITY_DATASET = "Data-Set-5"
VOLUME_DATASET = "Data-Set-15"
SNAPSHOT_INTERVAL_YR = 100.0
MASS_DENSITY_TO_MSUN_PER_PC3 = 0.025
RATE_TO_MSUN_PER_YR = 0.025
CORRUPTED_SNAPSHOT_GROUPS = ((747, 747), (754, 755), (763, 770))
CORRUPTED_SNAPSHOTS = frozenset(
    index
    for start, end in CORRUPTED_SNAPSHOT_GROUPS
    for index in range(start, end + 1)
)
RADIAL_ZONES_KPC = (
    (0.001, 0.01, "1–10 pc"),
    (0.01, 0.1, "10–100 pc"),
    (0.1, 1.001, "100–1000 pc"),
)
TIME_PATTERN = re.compile(
    r"AT\s+TIME\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def snapshot_index(path: Path) -> int:
    suffix = path.name.rsplit(".", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"无法解析 snapshot 编号：{path.name}")
    return int(suffix)


def open_hdf4(path: Path) -> SD:
    """只把 ASCII 文件名传给 Windows HDF4 C 库。"""
    previous_directory = Path.cwd()
    try:
        os.chdir(path.parent)
        return SD(path.name, SDC.READ)
    finally:
        os.chdir(previous_directory)


def parse_hdf_time_yr(long_name: str, path: Path) -> float:
    match = TIME_PATTERN.search(long_name)
    if match is None:
        raise ValueError(f"{path.name} 的 density long_name 缺少时间戳：{long_name!r}")
    time_gyr = float(match.group(1).replace("D", "E").replace("d", "e"))
    return time_gyr * 1.0e9


def read_static_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    handle = open_hdf4(path)
    try:
        radius = np.asarray(handle.select("fakeDim2")[:], dtype=np.float64).squeeze()
        volume_dataset = handle.select(VOLUME_DATASET)
        volume_label = str(volume_dataset.attributes().get("long_name", ""))
        if "VOL" not in volume_label.upper():
            raise ValueError(f"{path.name} 的 {VOLUME_DATASET} 语义异常：{volume_label!r}")
        volume = np.asarray(volume_dataset[:], dtype=np.float64).squeeze()
    finally:
        handle.end()
    if np.any(volume <= 0.0) or not np.isfinite(volume).all():
        raise ValueError("网格体积必须为有限正数。")
    return radius, volume


def read_density(path: Path) -> tuple[float, np.ndarray]:
    handle = open_hdf4(path)
    try:
        dataset = handle.select(DENSITY_DATASET)
        long_name = str(dataset.attributes().get("long_name", ""))
        if "DENSITY" not in long_name.upper():
            raise ValueError(f"{path.name} 的 {DENSITY_DATASET} 语义异常：{long_name!r}")
        density = np.asarray(dataset[:], dtype=np.float64).squeeze()
    finally:
        handle.end()
    if np.any(density < 0.0) or not np.isfinite(density).all():
        raise ValueError(f"{path.name} 含非有限或负密度。")
    return parse_hdf_time_yr(long_name, path), density


def volume_weighted_mean(
    density: np.ndarray,
    volume: np.ndarray,
    radial_mask: np.ndarray | None = None,
) -> float:
    if radial_mask is None:
        selected_density = density
        selected_volume = volume
    else:
        selected_density = density[..., radial_mask]
        selected_volume = volume[..., radial_mask]
    return float(
        np.sum(selected_density * selected_volume, dtype=np.float64)
        / np.sum(selected_volume, dtype=np.float64)
    )


def load_density_evolution(
    input_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    files = sorted(input_dir.glob("hdfra.*"), key=snapshot_index)
    if not files:
        raise FileNotFoundError(f"没有在 {input_dir} 中找到 hdfra.*")
    indexed_files = {snapshot_index(path): path for path in files}
    indices = np.arange(min(indexed_files), max(indexed_files) + 1, dtype=int)
    time_yr = indices.astype(np.float64) * SNAPSHOT_INTERVAL_YR
    status = ["missing"] * indices.size

    radius, volume = read_static_grid(indexed_files[int(indices[0])])
    radial_masks = {
        label: (radius >= lower) & (radius < upper)
        for lower, upper, label in RADIAL_ZONES_KPC
    }
    density_series = {
        "1–1000 pc": np.full(indices.shape, np.nan, dtype=np.float64),
        **{
            label: np.full(indices.shape, np.nan, dtype=np.float64)
            for _, _, label in RADIAL_ZONES_KPC
        },
    }

    for position, index in enumerate(indices):
        if index in CORRUPTED_SNAPSHOTS:
            status[position] = "corrupted_excluded"
            continue
        path = indexed_files.get(int(index))
        if path is None:
            continue
        hdf_time_yr, density = read_density(path)
        expected_time_yr = index * SNAPSHOT_INTERVAL_YR
        if not np.isclose(hdf_time_yr, expected_time_yr, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"{path.name} 的 HDF 时间 {hdf_time_yr:g} yr 与编号时间 "
                f"{expected_time_yr:g} yr 不一致。"
            )
        time_yr[position] = hdf_time_yr
        density_series["1–1000 pc"][position] = volume_weighted_mean(
            density, volume
        )
        for label, mask in radial_masks.items():
            density_series[label][position] = volume_weighted_mean(
                density, volume, mask
            )
        status[position] = "valid"

    for values in density_series.values():
        values *= MASS_DENSITY_TO_MSUN_PER_PC3
    return time_yr, density_series, status


def load_bhar_log(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """逐行读取大日志，避免一次性载入约 440 MB 文本。"""
    time_values = array("d")
    rate_values = array("d")
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            _, separator, payload = line.partition("=")
            if not separator:
                continue
            tokens = payload.replace("D", "E").split()
            if len(tokens) < 13:
                continue
            try:
                time_gyr = float(tokens[1])
                mdot_bh = float(tokens[6])
            except ValueError as exc:
                raise ValueError(f"日志第 {line_number} 行无法解析。") from exc
            if not np.isfinite(time_gyr) or not np.isfinite(mdot_bh) or mdot_bh < 0:
                raise ValueError(f"日志第 {line_number} 行包含非法 time/mdot_bh。")
            time_values.append(time_gyr)
            rate_values.append(mdot_bh)

    if not time_values:
        raise ValueError(f"没有从 {log_path} 读到吸积率日志。")
    time_kyr = np.frombuffer(time_values, dtype=np.float64).copy() * 1.0e6
    rate = np.frombuffer(rate_values, dtype=np.float64).copy() * RATE_TO_MSUN_PER_YR
    order = np.argsort(time_kyr, kind="stable")
    return time_kyr[order], rate[order]


def bin_bhar_for_display(
    time_kyr: np.ndarray,
    rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 100 yr 分箱保留范围，并在 snapshot 时刻取最近的瞬时值。"""
    n_bins = 1000
    bin_width_kyr = SNAPSHOT_INTERVAL_YR / 1.0e3
    bin_index = np.floor(time_kyr / bin_width_kyr).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    minimum = np.full(n_bins, np.inf, dtype=np.float64)
    maximum = np.full(n_bins, -np.inf, dtype=np.float64)
    np.minimum.at(minimum, bin_index, rate)
    np.maximum.at(maximum, bin_index, rate)
    minimum[~np.isfinite(minimum)] = np.nan
    maximum[~np.isfinite(maximum)] = np.nan

    sample_time = np.arange(n_bins, dtype=np.float64) * bin_width_kyr
    right = np.searchsorted(time_kyr, sample_time, side="left")
    right = np.clip(right, 0, time_kyr.size - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(time_kyr[left] - sample_time) <= np.abs(
        time_kyr[right] - sample_time
    )
    nearest = np.where(use_left, left, right)
    sample_rate = rate[nearest]
    return sample_time, sample_rate, minimum, maximum


def shade_corrupted_snapshots(axis: plt.Axes) -> None:
    half_step_kyr = SNAPSHOT_INTERVAL_YR / 2.0e3
    for start, end in CORRUPTED_SNAPSHOT_GROUPS:
        axis.axvspan(
            start * SNAPSHOT_INTERVAL_YR / 1.0e3 - half_step_kyr,
            end * SNAPSHOT_INTERVAL_YR / 1.0e3 + half_step_kyr,
            color="#9E9E9E",
            alpha=0.28,
            linewidth=0.0,
            zorder=1,
        )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 320,
        }
    )


def plot_density(
    output_path: Path,
    time_yr: np.ndarray,
    density_series: dict[str, np.ndarray],
) -> None:
    colors = {
        "1–1000 pc": "#263238",
        "1–10 pc": "#D32F2F",
        "10–100 pc": "#F57C00",
        "100–1000 pc": "#1976D2",
    }
    styles = {"1–1000 pc": "--", "1–10 pc": "-", "10–100 pc": "-", "100–1000 pc": "-"}
    fig, axis = plt.subplots(figsize=(12.0, 6.2))
    time_kyr = time_yr / 1.0e3
    for label, values in density_series.items():
        axis.plot(
            time_kyr,
            values,
            color=colors[label],
            linestyle=styles[label],
            linewidth=1.15 if label != "1–1000 pc" else 1.35,
            label=label,
            zorder=3,
        )
    shade_corrupted_snapshots(axis)
    axis.set_yscale("log")
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Simulation time [kyr]")
    axis.set_ylabel(r"Volume-weighted gas density [$M_\odot\,\mathrm{pc}^{-3}$]")
    axis.set_title("Torus simulation: instantaneous gas-density evolution")
    axis.grid(True, which="major", color="#B0BEC5", linestyle="--", alpha=0.55)
    axis.grid(True, which="minor", color="#CFD8DC", linestyle=":", alpha=0.35)
    axis.legend(loc="best", ncols=2, frameon=True)
    axis.text(
        0.99,
        0.02,
        "Volume-weighted means; no temporal smoothing or interpolation",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#455A64",
        fontsize=9.3,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_bhar(
    output_path: Path,
    time_kyr: np.ndarray,
    rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_time, sample_rate, minimum, maximum = bin_bhar_for_display(time_kyr, rate)
    fig, axis = plt.subplots(figsize=(12.0, 6.2))
    valid_range = np.isfinite(minimum) & np.isfinite(maximum)
    axis.fill_between(
        sample_time[valid_range],
        minimum[valid_range],
        maximum[valid_range],
        color="#90CAF9",
        alpha=0.45,
        linewidth=0.0,
        label="Raw log range within each 100 yr",
        zorder=1,
    )
    axis.plot(
        sample_time,
        sample_rate,
        color="#0D47A1",
        linewidth=0.9,
        label="Nearest raw value at snapshot time",
        zorder=3,
    )
    axis.set_yscale("symlog", linthresh=1.0e-6, linscale=0.65)
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Simulation time [kyr]")
    axis.set_ylabel(r"Black-hole accretion rate [$M_\odot\,\mathrm{yr}^{-1}$]")
    axis.set_title("Torus simulation: instantaneous black-hole accretion rate")
    axis.grid(True, which="major", color="#B0BEC5", linestyle="--", alpha=0.55)
    axis.grid(True, which="minor", color="#CFD8DC", linestyle=":", alpha=0.35)
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.99,
        0.02,
        "Light band preserves the unsmoothed extrema of the adaptive-step log",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#455A64",
        fontsize=9.3,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return sample_time, sample_rate, minimum, maximum


def save_density_csv(
    path: Path,
    time_yr: np.ndarray,
    density_series: dict[str, np.ndarray],
    status: list[str],
) -> None:
    labels = list(density_series)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["snapshot_index", "time_yr", *[f"density_{label}_msun_pc3" for label in labels], "status"]
        )
        for index in range(time_yr.size):
            writer.writerow(
                [
                    index,
                    f"{time_yr[index]:.8g}",
                    *[
                        "" if np.isnan(density_series[label][index]) else f"{density_series[label][index]:.12g}"
                        for label in labels
                    ],
                    status[index],
                ]
            )


def save_bhar_csv(
    path: Path,
    sample_time: np.ndarray,
    sample_rate: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "snapshot_index",
                "time_kyr",
                "nearest_mdot_bh_msun_yr",
                "bin_min_mdot_bh_msun_yr",
                "bin_max_mdot_bh_msun_yr",
            ]
        )
        for index in range(sample_time.size):
            writer.writerow(
                [
                    index,
                    f"{sample_time[index]:.8g}",
                    f"{sample_rate[index]:.12g}",
                    "" if np.isnan(minimum[index]) else f"{minimum[index]:.12g}",
                    "" if np.isnan(maximum[index]) else f"{maximum[index]:.12g}",
                ]
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    density_time, density_series, density_status = load_density_evolution(
        args.input_dir.resolve()
    )
    density_figure = output_dir / "torus_gas_density_evolution.png"
    density_csv = output_dir / "torus_gas_density_evolution.csv"
    plot_density(density_figure, density_time, density_series)
    save_density_csv(density_csv, density_time, density_series, density_status)

    log_time, raw_bhar = load_bhar_log(args.log_path.resolve())
    bhar_figure = output_dir / "torus_black_hole_accretion_rate_evolution.png"
    bhar_csv = output_dir / "torus_black_hole_accretion_rate_evolution.csv"
    display_values = plot_bhar(bhar_figure, log_time, raw_bhar)
    save_bhar_csv(bhar_csv, *display_values)

    valid_full_density = density_series["1–1000 pc"][
        np.isfinite(density_series["1–1000 pc"])
    ]
    print(f"有效密度 snapshot：{valid_full_density.size}")
    print(
        "全域体积加权密度 [Msun/pc^3]："
        f"min={valid_full_density.min():.6g}, "
        f"median={np.median(valid_full_density):.6g}, "
        f"max={valid_full_density.max():.6g}"
    )
    print(f"吸积日志行数：{raw_bhar.size}")
    print(
        "BHAR [Msun/yr]："
        f"min={raw_bhar.min():.6g}, median={np.median(raw_bhar):.6g}, "
        f"max={raw_bhar.max():.6g}"
    )
    print(f"密度图：{density_figure}")
    print(f"吸积率图：{bhar_figure}")


if __name__ == "__main__":
    main()
