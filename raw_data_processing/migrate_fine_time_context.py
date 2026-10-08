"""迁移二维 fine 的绝对时间、日志上下文和 SFR 产品，不重读 raw 物理数组。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_data_processing.build_fine_from_raw import (
    _iter_galaxy_configs,
    _select_snapshot_rows,
    _snapshot_sort_key,
)
from raw_data_processing.build_coarse_from_fine import _validate_fine_input
from raw_data_processing.galaxy_data import (
    GalData,
    _build_snapshot_schedule,
    _reject_stale_filename_index_outputs,
    _snapshot_output_name,
    _validate_macer_snapshot_time,
)
from raw_data_processing.metadata_schema import metadata_from_json, sha256_file
from raw_data_processing.parse_log_file import ll_dg_from_frame, load_config

MIGRATION_VERSION = 1


class CachedDiskLog:
    """日志只加载一次；用有序时间索引截取窗口，再调用既有平均/插值函数。"""

    def __init__(self, frame: pd.DataFrame):
        if not {"time", "mbh", "NEWstar"}.issubset(frame.columns):
            raise ValueError("盘星系日志必须包含 time、mbh 和 NEWstar。")
        times = pd.to_numeric(frame["time"], errors="coerce")
        self.frame = frame.loc[np.isfinite(times.to_numpy(dtype=float))].copy()
        self.frame["time"] = times.loc[self.frame.index].to_numpy(dtype=float)
        self.frame = self.frame.sort_values("time", kind="mergesort").reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("日志没有有限物理时间。")
        self.times = self.frame["time"].to_numpy(dtype=float)
        self.unique_times = np.unique(self.times)

    def select(self, time: float, scope: float) -> dict:
        if not np.isfinite(time) or not np.isfinite(scope) or scope < 0:
            raise ValueError("time/scope 必须有限，scope 必须非负。")
        left = int(np.searchsorted(self.times, time - scope, side="left"))
        right = int(np.searchsorted(self.times, time + scope, side="right"))
        if left == right:
            # 保留最近两侧的全部重复时刻，确保空窗口回退仍由原函数完成。
            position = int(np.searchsorted(self.unique_times, time, side="left"))
            low = self.unique_times[max(0, position - 1)]
            high = self.unique_times[min(position, len(self.unique_times) - 1)]
            left = int(np.searchsorted(self.times, low, side="left"))
            right = int(np.searchsorted(self.times, high, side="right"))
        result = ll_dg_from_frame(self.frame.iloc[left:right], time=time, scope=scope)
        return dict(result.items())


def _dirname(value: str) -> str:
    if (not value or value in {".", ".."} or "/" in value or "\\" in value
            or ":" in value or Path(value).is_absolute()):
        raise ValueError("fine 目录名必须是普通单层子目录名。")
    return value


def _verify_raw_time(path: Path, time: float) -> None:
    """可选地重验 HDF4 的时间标签，不加载物理数组。"""
    probe = GalData()._load_snapshot_hdfra(str(path), time)
    try:
        _validate_macer_snapshot_time(probe._data_hdfra, time)
    finally:
        probe._data_hdfra.end()


def _validate_output_directory(
    path: Path, provenance: dict, time_mapping: dict, run_id: str,
    target_config: dict, correspondence: dict,
) -> None:
    """抽样和续跑也禁止向已有目录混写另一套日志或筛选契约。"""
    for existing in path.glob("*.h5"):
        with h5py.File(existing, "r") as handle:
            if "metadata/contract_json" not in handle:
                raise ValueError(f"已有目录包含无契约文件：{existing}")
            metadata = metadata_from_json(handle["metadata/contract_json"][()])
            correlation = handle["correlation"]
            stored_corr = {
                correlation.attrs[f"{i}_orig"]: correlation.attrs[f"{i}_new"]
                for i in range(int(correlation.attrs["length"]))
            }
        record = metadata.get("time_context_migration", {})
        source = metadata.get("source", {})
        if (any(record.get(key) != value for key, value in provenance.items())
                or source.get("time_mapping") != time_mapping or source.get("run_id") != run_id
                or stored_corr != correspondence
                or any(metadata.get("target", {}).get(k) != v for k, v in target_config.items())):
            raise ValueError(f"已有目录的迁移契约不匹配：{existing}")


def _assert_preserved(original: GalData, migrated: GalData) -> None:
    """迁移前后物理内容须逐元素相同；不能改变 Y 或 snapshot 身份。"""
    if original.parent_hr_uid != migrated.parent_hr_uid or original.sample_uid != migrated.sample_uid:
        raise ValueError("迁移改变了物理 snapshot/sample 身份。")
    if original.sfr_target != migrated.sfr_target or original._target_metadata != migrated._target_metadata:
        raise ValueError("迁移改变了 Data-Set-11 真值或单位契约。")
    if set(original.snapshot) != set(migrated.snapshot):
        raise ValueError("迁移改变了物理字段集合。")
    pairs = [(original.snapshot[k], migrated.snapshot[k]) for k in original.snapshot]
    pairs += [(original.grid_volume, migrated.grid_volume)]
    pairs += list(zip(original._axes, migrated._axes))
    pairs += list(zip(original._axis_edges, migrated._axis_edges))
    if any(not np.array_equal(a, b, equal_nan=True) for a, b in pairs):
        raise ValueError("迁移改变了物理数组、权威体积或坐标。")


def migrate_snapshot(
    input_path: Path,
    output_path: Path,
    *,
    source: dict,
    target_config: dict,
    correspondence: dict,
    log_reader: CachedDiskLog,
    scope: float,
    provenance: dict,
    resume: bool = True,
) -> bool:
    """迁移一帧，写出返回 True；已存在且严格匹配则返回 False。"""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("禁止原地迁移或覆盖旧 fine。")
    with h5py.File(input_path, "r") as handle:
        if handle.attrs.get("processed_format_version") != 6 or "macnet_3d_format" in handle.attrs:
            raise ValueError("只支持完整的二维 processed format-version 6 fine。")
    original = GalData().load_h5(str(input_path))
    _validate_fine_input(original, input_path)
    if original.ndim != 2 or original._coordinate_mode != "polar":
        raise ValueError("只支持 MACER2D polar fine。")
    old_source = original.metadata["source"]
    for key, value in source.items():
        if key not in {"time", "time_mapping"} and old_source.get(key) != value:
            raise ValueError(f"{input_path} 的来源 {key} 与 raw 清单/配置不一致。")
    if not np.isclose(original.time, old_source.get("time", np.nan), rtol=0, atol=1e-12):
        raise ValueError("旧 fine 根时间与 source.time 不一致。")
    if original._corr != correspondence:
        raise ValueError("旧 fine 的字段映射与当前配置不一致，不能仅修复时间。")
    for key, value in target_config.items():
        if original._target_metadata.get(key) != value:
            raise ValueError(f"旧 fine 的 target 契约 {key} 与当前配置不一致。")
    if target_config.get("dataset") != "Data-Set-11" or target_config.get("reduction") != "sum":
        raise ValueError("仅支持 Data-Set-11 sum 的瞬时 SFR 真值。")
    record = dict(provenance, input_fine_sha256=sha256_file(input_path), old_time_gyr=float(original.time))
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"拒绝覆盖已有文件：{output_path}")
        restored = GalData().load_h5(str(output_path))
        if restored.metadata.get("time_context_migration") != record or restored.metadata["source"] != source:
            raise ValueError(f"已有输出与本次迁移契约不匹配：{output_path}")
        _assert_preserved(original, restored)
        if float(restored.time) != float(source["time"]):
            raise ValueError("已有输出根时间不匹配。")
        return False

    migrated = GalData().load_h5(str(input_path))
    migrated.time = float(source["time"])
    migrated._data_log = log_reader.select(migrated.time, scope)
    migrated._data_log["log_context_scope"] = float(scope)
    # 日志量不能替代真实瞬时标签；显式恢复原 Data-Set-11 归约值及其别名。
    migrated._data_log["sfr_target"] = float(original.sfr_target)
    migrated._data_log[str(original._target_metadata["name"])] = float(original.sfr_target)
    migrated.attach_fine_source_metadata(source)
    migrated.compute_macer_sfr_products()
    migrated._metadata["time_context_migration"] = record
    _assert_preserved(original, migrated)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".timefix-", suffix=".h5", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        migrated.save_h5(str(temporary_path))
        restored = GalData().load_h5(str(temporary_path))
        _assert_preserved(original, restored)
        logs_equal = set(restored._data_log) == set(migrated._data_log) and all(
            np.array_equal(restored._data_log[k], migrated._data_log[k], equal_nan=True)
            for k in migrated._data_log
        )
        if restored.metadata != migrated.metadata or not logs_equal:
            raise ValueError("迁移产品 HDF5 round-trip 不一致。")
        # 同目录硬链接发布保证目标已存在时失败，不会覆盖并发任务的输出。
        os.link(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def migrate_fine_time_context(
    *, config_dir: str | Path = ".config", galaxies=(),
    input_fine_dirname: str = "fine_sfr_v6", fine_dirname: str = "fine_sfr_v7_timefix",
    snapshot_indices=(), exclude_snapshot_indices=None, max_snapshots: int | None = None,
    verify_raw_time: bool = False, resume: bool = True,
) -> list[Path]:
    """按新版 raw→fine 的原始文件清单、时间映射和筛选规则迁移。"""
    _dirname(input_fine_dirname)
    _dirname(fine_dirname)
    if input_fine_dirname == fine_dirname:
        raise ValueError("输入与输出目录必须不同。")
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError("max_snapshots 必须为正整数。")
    selected, requested = set(galaxies), set(snapshot_indices)
    explicit = exclude_snapshot_indices is not None
    excluded = set(exclude_snapshot_indices or ())
    if not selected:
        raise ValueError("必须显式指定至少一个 --galaxy，避免误迁移其他运行。")
    if any(i < 0 for i in requested | excluded):
        raise ValueError("snapshot index 不能为负数。")
    configs = load_config(str(config_dir))
    configured = {name for _, name, _, _ in _iter_galaxy_configs(configs, set())}
    if selected - configured:
        raise ValueError(f"未知 galaxy：{sorted(selected - configured)}")
    data_dir = Path(configs["BaseConfig"]["data_dir"])
    raw_dir = Path(configs["BaseConfig"]["raw_data_dir"])
    written = []
    for galaxy_type, name, config, fields in _iter_galaxy_configs(configs, selected):
        if galaxy_type != "disk_galaxy" or fields.get("format", "hdfra") != "hdfra" or config.get("mode") != "filename_index":
            raise ValueError("仅支持 filename_index 模式的二维盘星系 HDFRA 运行。")
        raw_folder = raw_dir / config["folder_name"] / "data"
        if not raw_folder.is_dir():
            raw_folder = raw_dir / config["folder_name"]
        names = sorted((p.name for p in raw_folder.glob("hdfra.*") if p.is_file()), key=_snapshot_sort_key)
        if not names:
            raise FileNotFoundError(f"原始文件清单不可用：{raw_folder}")
        times, indices = _build_snapshot_schedule(names, config)
        rows = [(pos, filename, float(times[pos]), int(indices[pos])) for pos, filename in enumerate(names)
                if explicit or pos > 45]
        rows = _select_snapshot_rows(rows, requested_indices=requested, excluded_indices=excluded, max_snapshots=max_snapshots)
        input_dir, output_dir = data_dir / name / input_fine_dirname, data_dir / name / fine_dirname
        if input_dir.resolve() == output_dir.resolve():
            raise ValueError("输入与输出解析到同一目录。")
        if not output_dir.resolve().is_relative_to((data_dir / name).resolve()):
            raise ValueError("输出目录解析到当前运行目录之外，拒绝写入。")
        missing = [index for _, _, _, index in rows if not (input_dir / _snapshot_output_name(index)).is_file()]
        if missing:
            raise FileNotFoundError(f"{name} 缺少待迁移 fine（前十项）：{missing[:10]}")
        if not requested and max_snapshots is None and output_dir.is_dir():
            _reject_stale_filename_index_outputs(str(output_dir), {_snapshot_output_name(i) for _, _, _, i in rows})
        log_path = data_dir / f"{name}.parquet"
        log_hash = sha256_file(log_path)
        reader = CachedDiskLog(pd.read_parquet(log_path))
        scope = float(config["snapshot_dt"]) * 0.5
        excluded_list = sorted(excluded if explicit else set(int(i) for i in indices[:46]))
        provenance = {
            "migration_version": MIGRATION_VERSION, "input_fine_dirname": input_fine_dirname,
            "log_file_sha256": log_hash, "log_context_scope_gyr": scope,
            "selection_rule": "explicit_exclusions" if explicit else "skip_first_46_sorted_raw_files",
            "excluded_raw_snapshot_indices": excluded_list,
            "raw_time_validation": "HDF4_label" if verify_raw_time else "not_repeated; configuration_mapping",
        }
        time_mapping = config.get("time_mapping", {
            "mode": "linear_index", "snapshot_dt": float(config["snapshot_dt"]),
            "offset": float(config.get("offset", 0)),
        })
        _validate_output_directory(output_dir, provenance, time_mapping, name, fields["target"], fields["correspondence"])
        print(f"[{name}] fine 时间迁移：{len(rows)} 帧（日志仅加载一次）", flush=True)
        skipped = 0
        for count, (_, filename, time, index) in enumerate(rows, start=1):
            input_path = input_dir / _snapshot_output_name(index)
            # 不重哈希 raw；只复用旧 fine 的哈希，并核验文件名、身份和当前大小。
            with h5py.File(input_path, "r") as handle:
                old_source = metadata_from_json(handle["metadata/contract_json"][()])["source"]
            source = deepcopy(old_source)
            source.update(simulation_code="MACER2D", galaxy_type=galaxy_type, galaxy_id=name, run_id=name,
                          source_series_id=config["folder_name"], raw_snapshot_index=index, source_file_id=filename,
                          source_file_size_bytes=(raw_folder / filename).stat().st_size, time=time, time_units="Gyr",
                          time_mapping=time_mapping)
            digest = source.get("source_file_sha256", "")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("旧 fine 缺少有效 raw SHA256，不能建立迁移来源链。")
            if verify_raw_time:
                _verify_raw_time(raw_folder / filename, time)
            output_path = output_dir / _snapshot_output_name(index)
            if migrate_snapshot(input_path, output_path, source=source, target_config=fields["target"],
                                correspondence=fields["correspondence"], log_reader=reader, scope=scope,
                                provenance=provenance, resume=resume):
                written.append(output_path)
            else:
                skipped += 1
            if count % 100 == 0 or count == len(rows):
                print(f"[{name}] {count}/{len(rows)}，本次匹配跳过 {skipped} 帧", flush=True)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="修复已有二维 fine 的时间/日志/SFR 产品；不覆盖旧数据。")
    parser.add_argument("--galaxy", action="append", required=True)
    parser.add_argument("--config-dir", default=".config")
    parser.add_argument("--input-fine-dirname", default="fine_sfr_v6")
    parser.add_argument("--fine-dirname", default="fine_sfr_v7_timefix")
    parser.add_argument("--snapshot-index", type=int, action="append", default=[])
    parser.add_argument("--exclude-snapshot", type=int, action="append", default=None)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--verify-raw-time", action="store_true", help="额外读取 HDF4 时间标签；需要 pyhdf。")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="已有输出一律拒绝，不提供覆盖选项。")
    args = parser.parse_args(argv)
    try:
        written = migrate_fine_time_context(config_dir=args.config_dir, galaxies=args.galaxy,
            input_fine_dirname=args.input_fine_dirname, fine_dirname=args.fine_dirname,
            snapshot_indices=args.snapshot_index, exclude_snapshot_indices=args.exclude_snapshot,
            max_snapshots=args.max_snapshots, verify_raw_time=args.verify_raw_time, resume=args.resume)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))
    print(f"迁移完成：新写出 {len(written)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
