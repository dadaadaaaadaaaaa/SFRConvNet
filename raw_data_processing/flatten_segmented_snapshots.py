#!/usr/bin/env python3
"""将覆盖循环快照文件夹合并到一个全局索引目录中。

旧版 MACER 输出会在名为 ``<series>_0``、``<series>_1``、... 的文件夹中重复使用诸如 ``hdfra.001`` 至
``hdfra.999`` 之类的本地名称。该
实用程序使用以下规则将这些本地名称转换为唯一的全局快照名称：

    段 0：全局索引 = 本地索引
    段 k ≥ 1：全局索引 = 循环大小 * k + 本地索引

默认循环大小为 999。允许存在缺失的文件甚至缺失的段文件夹：
它们将在全局索引中保留为空缺，而非导致后续
快照时间发生偏移。

默认情况下，该工具会创建硬链接，因此扁平化视图不会
生成快照数据的另一份完整副本。源数据绝不会被修改，
目标目录必须不存在，且构建过程会在临时
同级目录中进行，随后执行原子重命名操作。

示例（Fiducial4）：:

    python raw_data_processing/flatten_segmented_snapshots.py \
        --source-dir RawData/DiskGalaxy/Fiducial4 \
        --output-dir RawData/DiskGalaxy/Fiducial4_flat \
        --segment-prefix Fiducial4_new \
        --include-file Fiducial4_new.log \
        --expected-range 0 9541

请先使用 ``--dry-run`` 选项进行模拟运行，以验证映射关系，而不会实际生成任何文件。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


HDFRA_RE = re.compile(r"^hdfra\.(\d+)$")
GENERIC_SEGMENT_RE = re.compile(r"^(.+)_(\d+)$")


@dataclass(frozen=True)
class SnapshotRecord:
    segment_name: str
    segment_index: int
    local_file: str
    local_index: int
    global_index: int
    source_path: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def discover_segments(source_dir: Path, segment_prefix: str | None) -> tuple[str, list[tuple[int, Path]]]:
    """Return one unambiguous series prefix and its numerically sorted segments."""
    groups: dict[str, list[tuple[int, Path]]] = {}
    for child in source_dir.iterdir():
        if not child.is_dir():
            continue
        match = GENERIC_SEGMENT_RE.fullmatch(child.name)
        if match is None:
            continue
        prefix, index_text = match.groups()
        if segment_prefix is not None and prefix != segment_prefix:
            continue
        groups.setdefault(prefix, []).append((int(index_text), child))

    if segment_prefix is not None:
        segments = groups.get(segment_prefix, [])
        if not segments:
            raise ValueError(
                f"No segment folders matching {segment_prefix!r}_<integer> were found in {source_dir}"
            )
        selected_prefix = segment_prefix
    else:
        if not groups:
            raise ValueError(f"No folders ending in _<integer> were found in {source_dir}")
        if len(groups) != 1:
            candidates = ", ".join(sorted(groups))
            raise ValueError(
                "Multiple segment series were found; specify --segment-prefix. "
                f"Candidates: {candidates}"
            )
        selected_prefix, segments = next(iter(groups.items()))

    segments.sort(key=lambda item: item[0])
    indices = [index for index, _ in segments]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate segment indices found for prefix {selected_prefix!r}")
    return selected_prefix, segments


def build_manifest(
    segments: Iterable[tuple[int, Path]],
    *,
    cycle_size: int,
    expected_range: tuple[int, int] | None,
) -> list[SnapshotRecord]:
    """Build and validate the local-to-global snapshot mapping."""
    records: list[SnapshotRecord] = []
    global_sources: dict[int, Path] = {}

    for segment_index, segment_dir in segments:
        segment_records = 0
        for source_path in segment_dir.iterdir():
            if not source_path.is_file():
                continue
            match = HDFRA_RE.fullmatch(source_path.name)
            if match is None:
                continue

            local_index = int(match.group(1))
            if segment_index == 0:
                if not 0 <= local_index <= cycle_size:
                    raise ValueError(
                        f"Invalid local index {local_index} in {segment_dir.name}; "
                        f"segment 0 expects 0..{cycle_size}"
                    )
                global_index = local_index
            else:
                if not 1 <= local_index <= cycle_size:
                    raise ValueError(
                        f"Invalid local index {local_index} in {segment_dir.name}; "
                        f"later segments expect 1..{cycle_size}"
                    )
                global_index = cycle_size * segment_index + local_index

            if expected_range is not None:
                start, end = expected_range
                if not start <= global_index < end:
                    raise ValueError(
                        f"Global index {global_index} from {source_path} lies outside "
                        f"the expected range [{start}, {end})"
                    )

            previous = global_sources.get(global_index)
            if previous is not None:
                raise ValueError(
                    f"Global index collision at {global_index}: {previous} and {source_path}"
                )
            global_sources[global_index] = source_path
            records.append(
                SnapshotRecord(
                    segment_name=segment_dir.name,
                    segment_index=segment_index,
                    local_file=source_path.name,
                    local_index=local_index,
                    global_index=global_index,
                    source_path=source_path,
                )
            )
            segment_records += 1

        if segment_records == 0:
            raise ValueError(f"Segment folder contains no hdfra files: {segment_dir}")

    if not records:
        raise ValueError("No hdfra snapshot files were found")
    records.sort(key=lambda record: record.global_index)
    return records


def _validate_include_files(source_dir: Path, include_files: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for name in include_files:
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
            raise ValueError(f"--include-file must be a filename in the source root, got {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate --include-file value: {name!r}")
        seen.add(name)
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Included file does not exist: {path}")
        if HDFRA_RE.fullmatch(name):
            raise ValueError(f"Included auxiliary file cannot use an hdfra name: {name!r}")
        paths.append(path)
    return paths


def _materialize(source: Path, destination: Path, method: str) -> None:
    if method == "hardlink":
        os.link(source, destination)
    elif method == "copy":
        shutil.copy2(source, destination)
    else:  # Protected by argparse, retained for programmatic use.
        raise ValueError(f"Unsupported materialization method: {method}")


def _write_manifest(
    path: Path,
    records: Sequence[SnapshotRecord],
    snapshot_dt: Decimal,
    filename_width: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "segment",
                "segment_index",
                "local_file",
                "local_index",
                "global_index",
                "time_gyr",
                "target_file",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "segment": record.segment_name,
                    "segment_index": record.segment_index,
                    "local_file": record.local_file,
                    "local_index": record.local_index,
                    "global_index": record.global_index,
                    "time_gyr": str(snapshot_dt * record.global_index),
                    "target_file": f"hdfra.{record.global_index:0{filename_width}d}",
                }
            )


def flatten_snapshots(
    *,
    source_dir: Path,
    output_dir: Path,
    segment_prefix: str | None,
    cycle_size: int,
    snapshot_dt: Decimal,
    expected_range: tuple[int, int] | None,
    filename_width: int,
    include_files: Sequence[str],
    method: str,
    manifest_name: str,
    dry_run: bool,
) -> list[SnapshotRecord]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be the source directory or a child of it")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if filename_width < 1:
        raise ValueError("filename_width must be positive")
    if expected_range is not None:
        start, end = expected_range
        if start < 0 or start >= end:
            raise ValueError("expected range must satisfy 0 <= START < END")
    manifest_path = Path(manifest_name)
    if manifest_path.is_absolute() or len(manifest_path.parts) != 1:
        raise ValueError("manifest name must be a filename, not a path")

    selected_prefix, segments = discover_segments(source_dir, segment_prefix)
    records = build_manifest(
        segments,
        cycle_size=cycle_size,
        expected_range=expected_range,
    )
    auxiliary_paths = _validate_include_files(source_dir, include_files)

    indices = [record.global_index for record in records]
    print(f"Series prefix:       {selected_prefix}")
    print(f"Segments found:      {len(segments)} ({', '.join(str(i) for i, _ in segments)})")
    print(f"Snapshots found:     {len(records)}")
    print(f"Global index range:  {indices[0]}..{indices[-1]}")
    print(f"Physical time range: {snapshot_dt * indices[0]}..{snapshot_dt * indices[-1]} Gyr")
    if expected_range is not None:
        start, end = expected_range
        print(f"Missing in expected range [{start}, {end}): {(end - start) - len(records)}")
    print(f"Output directory:    {output_dir}")
    print(f"Materialization:     {method}")

    if dry_run:
        print("Dry run complete; no files were created.")
        return records

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        for record in records:
            target_name = f"hdfra.{record.global_index:0{filename_width}d}"
            _materialize(record.source_path, staging_dir / target_name, method)
        for auxiliary_path in auxiliary_paths:
            destination = staging_dir / auxiliary_path.name
            if destination.exists():
                raise FileExistsError(f"Auxiliary filename collides with output: {destination.name}")
            _materialize(auxiliary_path, destination, method)
        _write_manifest(staging_dir / manifest_path.name, records, snapshot_dt, filename_width)

        created_snapshots = sum(
            1 for path in staging_dir.iterdir() if path.is_file() and HDFRA_RE.fullmatch(path.name)
        )
        if created_snapshots != len(records):
            raise RuntimeError(
                f"Output validation failed: created {created_snapshots} snapshots, expected {len(records)}"
            )
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"Completed successfully: {len(records)} snapshots written to {output_dir}")
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten segmented MACER hdfra snapshots using their global snapshot indices."
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing <prefix>_<k> folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New flat output directory; it must not exist.")
    parser.add_argument(
        "--segment-prefix",
        help="Folder prefix before the final _<k>, for example Fiducial4_new. Auto-detected if unique.",
    )
    parser.add_argument("--cycle-size", type=_positive_int, default=999, help="Overwrite cycle size (default: 999).")
    parser.add_argument(
        "--snapshot-dt",
        type=_positive_decimal,
        default=Decimal("0.00125"),
        help="Time between global snapshots in Gyr (default: 0.00125).",
    )
    parser.add_argument(
        "--expected-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Optional expected half-open global index range, for example 0 9541.",
    )
    parser.add_argument("--filename-width", type=_positive_int, default=5, help="Zero-padded output index width.")
    parser.add_argument(
        "--include-file",
        action="append",
        default=[],
        help="Auxiliary file in the source root to link/copy; may be repeated (for example a log file).",
    )
    parser.add_argument(
        "--method",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Create hard links (default) or full file copies.",
    )
    parser.add_argument("--manifest-name", default="snapshot_manifest.csv", help="CSV manifest filename.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report the mapping without writing files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_range = tuple(args.expected_range) if args.expected_range is not None else None
    try:
        flatten_snapshots(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            segment_prefix=args.segment_prefix,
            cycle_size=args.cycle_size,
            snapshot_dt=args.snapshot_dt,
            expected_range=expected_range,
            filename_width=args.filename_width,
            include_files=args.include_file,
            method=args.method,
            manifest_name=args.manifest_name,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
