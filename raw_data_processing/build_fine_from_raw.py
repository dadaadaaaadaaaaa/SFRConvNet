"""把 MACER 原始 HDFRA snapshot 转换为带完整 metadata 的 fine HDF5。

该脚本只负责 ``raw -> fine``。物理字段解释、Data-Set 标签校验、坐标与
Data-Set-15 体积读取、SFR supervision 生成均复用 ``galaxy_data.GalData``，
从而避免新入口与既有数据契约产生两套实现。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

try:
    from raw_data_processing.galaxy_data import (
        GalData,
        _build_snapshot_schedule,
        _reject_stale_filename_index_outputs,
        _snapshot_output_name,
    )
    from raw_data_processing.metadata_schema import sha256_file
    from raw_data_processing.parse_log_file import load_config, ll_dg, ll_eg
except ModuleNotFoundError:  # 兼容从 raw_data_processing 目录直接运行。
    from galaxy_data import (
        GalData,
        _build_snapshot_schedule,
        _reject_stale_filename_index_outputs,
        _snapshot_output_name,
    )
    from metadata_schema import sha256_file
    from parse_log_file import load_config, ll_dg, ll_eg


_STABLE_FROM_START = {
    "disk_galaxy_fiducial",
    "disk_galaxy_supplement",
    "disk_galaxy_fiducial_4",
    "disk_galaxy_fiducial_7",
}


def _snapshot_sort_key(name: str) -> tuple[int, int | str]:
    """按 ``hdfra.<整数>`` 的数值后缀排序，非标准名称排在末尾。"""

    suffix = name.rsplit(".", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def _iter_galaxy_configs(configs: dict, selected: set[str]):
    """按配置顺序产生被选择的星系运行。"""

    for galaxy_type, galaxy_group in configs["RawDataConfig"].items():
        # dwarf_galaxy 使用 ATHDF，不属于本脚本的 HDFRA 输入契约。
        if galaxy_type not in configs["HdfraConfig"]:
            continue
        hdfra_config = configs["HdfraConfig"][galaxy_type]
        if str(hdfra_config.get("format", "hdfra")).lower() != "hdfra":
            continue
        for nickname, galaxy_config in galaxy_group.items():
            galaxy_name = f"{galaxy_type}_{nickname}"
            if selected and galaxy_name not in selected:
                continue
            yield galaxy_type, galaxy_name, galaxy_config, hdfra_config


def _select_snapshot_rows(
    rows: list[tuple[int, str, float, int]],
    *,
    requested_indices: set[int],
    max_snapshots: int | None,
) -> list[tuple[int, str, float, int]]:
    """按逻辑输出索引筛选 snapshot，并在最后应用数量上限。"""

    if requested_indices:
        available = {row[3] for row in rows}
        missing = sorted(requested_indices.difference(available))
        if missing:
            raise ValueError(f"请求的 snapshot index 不存在或已被稳定性规则排除：{missing}")
        rows = [row for row in rows if row[3] in requested_indices]
    if max_snapshots is not None:
        rows = rows[:max_snapshots]
    return rows


def build_fine_from_raw(
    *,
    config_dir: str | os.PathLike[str] = ".config",
    galaxies: Iterable[str] = (),
    snapshot_indices: Iterable[int] = (),
    max_snapshots: int | None = None,
    fine_dirname: str = "fine",
) -> list[Path]:
    """依据配置批量构建 fine HDF5，并返回本次写出的文件路径。

    参数名称与命令行含义一致。默认覆盖同名输出，以保持旧
    ``galaxy_data.py`` 主程序的行为；建议正式重建前先备份旧目录。
    """

    configs = load_config(str(config_dir))
    required_sections = {"BaseConfig", "RawDataConfig", "HdfraConfig"}
    missing_sections = required_sections.difference(configs)
    if missing_sections:
        raise KeyError(f"配置目录缺少 JSON 配置：{sorted(missing_sections)}")
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError("max_snapshots 必须为正整数或 None。")
    if not fine_dirname or Path(fine_dirname).name != fine_dirname:
        raise ValueError("fine_dirname 必须是单层目录名，不能包含路径分隔符。")

    selected = set(galaxies)
    requested_indices = {int(value) for value in snapshot_indices}
    if any(value < 0 for value in requested_indices):
        raise ValueError("snapshot_indices 不能包含负数。")
    configured_names = {
        galaxy_name
        for _, galaxy_name, _, _ in _iter_galaxy_configs(configs, set())
    }
    unknown = sorted(selected.difference(configured_names))
    if unknown:
        raise ValueError(f"--galaxy 与 HDFRA 配置中的生成名不匹配：{unknown}")
    raw_data_dir = Path(configs["BaseConfig"]["raw_data_dir"])
    data_dir = Path(configs["BaseConfig"]["data_dir"])
    written: list[Path] = []

    for galaxy_type, galaxy_name, galaxy_config, hdfra_config in _iter_galaxy_configs(
        configs, selected
    ):
        hdfra_folder = raw_data_dir / galaxy_config["folder_name"] / "data"
        if not hdfra_folder.exists():
            hdfra_folder = raw_data_dir / galaxy_config["folder_name"]
        if not hdfra_folder.exists():
            print(f"[跳过] {galaxy_name}：找不到原始目录 {hdfra_folder}")
            continue

        hdfra_names = sorted(
            (path.name for path in hdfra_folder.iterdir() if path.name.startswith("hdfra.")),
            key=_snapshot_sort_key,
        )
        if not hdfra_names:
            print(f"[跳过] {galaxy_name}：目录中没有 hdfra.* 文件")
            continue

        correspondence = hdfra_config["correspondence"]
        target_config = hdfra_config.get("target")
        if target_config is not None and target_config["dataset"] in correspondence:
            raise ValueError(
                f"{galaxy_type} 的目标 {target_config['dataset']!r} 同时出现在输入字段中，"
                "会造成标签泄漏。"
            )

        times, output_indices = _build_snapshot_schedule(hdfra_names, galaxy_config)
        stable_from_start = galaxy_name in _STABLE_FROM_START
        rows = [
            (position, name, float(times[position]), int(output_indices[position]))
            for position, name in enumerate(hdfra_names)
            if stable_from_start or position > 45
        ]
        rows = _select_snapshot_rows(
            rows,
            requested_indices=requested_indices,
            max_snapshots=max_snapshots,
        )

        output_dir = data_dir / galaxy_name / fine_dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        # 仅在完整批处理时执行旧输出污染检查；抽样 smoke test 不应把未选文件
        # 误判为 stale output。
        if (
            galaxy_config.get("mode") == "filename_index"
            and not requested_indices
            and max_snapshots is None
        ):
            expected_names = {
                _snapshot_output_name(index)
                for _, _, _, index in rows
            }
            _reject_stale_filename_index_outputs(str(output_dir), expected_names)

        log_path = data_dir / f"{galaxy_name}.parquet"
        parse_log = ll_eg if "elliptical" in galaxy_type else ll_dg
        scope = float(galaxy_config["snapshot_dt"]) * 0.5
        print(f"[{galaxy_name}] raw→fine：{len(rows)} 个 snapshot")
        for _, file_name, snapshot_time, output_index in rows:
            source_path = hdfra_folder / file_name
            output_path = output_dir / _snapshot_output_name(output_index)
            galaxy = GalData(ndim=2, coordinate_mode="polar")
            galaxy.set_corr(correspondence).read_hdfra(
                hdfra_path=str(source_path),
                time=snapshot_time,
                log_path=str(log_path),
                func_parse_log=parse_log,
                scope=scope,
                target_config=target_config,
                source_metadata={
                    "simulation_code": "MACER2D",
                    "galaxy_type": galaxy_type,
                    "galaxy_id": galaxy_name,
                    "run_id": galaxy_name,
                    "source_series_id": galaxy_config["folder_name"],
                    "raw_snapshot_index": output_index,
                    "source_file_id": file_name,
                    "source_file_sha256": sha256_file(source_path),
                    "source_file_size_bytes": int(source_path.stat().st_size),
                    "time": snapshot_time,
                    "time_units": "Gyr",
                },
            )
            galaxy.set_coord(["fakeDim2", "fakeDim1"])
            galaxy.save_h5(str(output_path))
            written.append(output_path)

    return written


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器；既有 ``--galaxy`` 参数名保持不变。"""

    parser = argparse.ArgumentParser(
        description="Convert raw MACER HDFRA snapshots to fine HDF5 files."
    )
    parser.add_argument(
        "--galaxy",
        action="append",
        default=[],
        help="只处理指定生成名（例如 disk_galaxy_fiducial），可重复。",
    )
    parser.add_argument("--config-dir", default=".config", help="JSON 配置目录。")
    parser.add_argument("--fine-dirname", default="fine", help="星系数据目录下的 fine 子目录名。")
    parser.add_argument(
        "--snapshot-index",
        action="append",
        default=[],
        type=int,
        help="只处理指定逻辑/原始 snapshot index，可重复。",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="每个选中星系最多处理前 N 个合格 snapshot；主要用于 smoke test。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        parser.error("--max-snapshots 必须为正整数。")
    if any(index < 0 for index in args.snapshot_index):
        parser.error("--snapshot-index 不能为负数。")
    try:
        written = build_fine_from_raw(
            config_dir=args.config_dir,
            galaxies=args.galaxy,
            snapshot_indices=args.snapshot_index,
            max_snapshots=args.max_snapshots,
            fine_dirname=args.fine_dirname,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"raw→fine 完成：写出 {len(written)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
