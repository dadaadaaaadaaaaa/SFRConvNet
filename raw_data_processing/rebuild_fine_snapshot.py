"""根据其原始 HDF4 源文件重建一个已处理的精细 MACER 快照。

该命令设计为仅处理单个快照，并且默认情况下拒绝覆盖。
这是将权威的 Data-Set-15网格体和坐标边界添加到旧版已处理 HDF5 文件的安全迁移路径。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from raw_data_processing.galaxy_data import (
        GalData,
        _build_snapshot_schedule,
        _snapshot_index_from_filename,
    )
    from raw_data_processing.parse_log_file import load_config, ll_dg, ll_eg
    from raw_data_processing.metadata_schema import sha256_file
except ModuleNotFoundError:
    from galaxy_data import GalData, _build_snapshot_schedule, _snapshot_index_from_filename
    from parse_log_file import load_config, ll_dg, ll_eg
    from metadata_schema import sha256_file


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild one fine MACER HDF5 file with exact cell volumes and axis edges."
    )
    parser.add_argument(
        "--galaxy",
        required=True,
        help="Generated config name, for example disk_galaxy_fiducial.",
    )
    parser.add_argument(
        "--snapshot-index",
        required=True,
        type=int,
        help="Integer suffix in the raw hdfra filename, for example 0 for hdfra.00000.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination .h5 file.")
    parser.add_argument(
        "--config-dir", type=Path, default=Path(".config"), help="Configuration directory."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    return parser.parse_args()


def _resolve_galaxy(configs: dict, generated_name: str):
    matches = []
    for galaxy_type, group in configs["RawDataConfig"].items():
        for nickname, galaxy_config in group.items():
            if f"{galaxy_type}_{nickname}" == generated_name:
                matches.append((galaxy_type, galaxy_config))
    if len(matches) != 1:
        available = sorted(
            f"{galaxy_type}_{nickname}"
            for galaxy_type, group in configs["RawDataConfig"].items()
            for nickname in group
        )
        raise KeyError(
            f"Expected exactly one configuration for {generated_name!r}; "
            f"available names are {available}."
        )
    return matches[0]


def main():
    args = _parse_args()
    if args.snapshot_index < 0:
        raise ValueError("--snapshot-index must be non-negative.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite explicitly."
        )

    configs = load_config(str(args.config_dir))
    galaxy_type, galaxy_config = _resolve_galaxy(configs, args.galaxy)
    raw_root = Path(configs["BaseConfig"]["raw_data_dir"])
    data_root = Path(configs["BaseConfig"]["data_dir"])
    raw_folder = raw_root / galaxy_config["folder_name"] / "data"
    if not raw_folder.is_dir():
        raw_folder = raw_root / galaxy_config["folder_name"]
    if not raw_folder.is_dir():
        raise FileNotFoundError(f"Raw snapshot directory does not exist: {raw_folder}")

    raw_paths = sorted(
        (path for path in raw_folder.iterdir() if path.name.startswith("hdfra.")),
        key=lambda path: _snapshot_index_from_filename(path.name),
    )
    if not raw_paths:
        raise FileNotFoundError(f"No hdfra snapshots found in {raw_folder}")
    times, physical_indices = _build_snapshot_schedule(
        [path.name for path in raw_paths], galaxy_config
    )
    positions = np.flatnonzero(physical_indices == args.snapshot_index)
    if positions.size != 1:
        raise FileNotFoundError(
            f"Could not identify exactly one raw snapshot with index {args.snapshot_index} "
            f"inside {raw_folder}."
        )
    position = int(positions[0])
    raw_path = raw_paths[position]
    time = float(times[position])

    hdfra_config = configs["HdfraConfig"][galaxy_type]
    log_path = data_root / f"{args.galaxy}.parquet"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"Parsed log file does not exist: {log_path}. Run parse_log_file.py first."
        )
    log_parser = ll_eg if galaxy_type == "elliptical_galaxy" else ll_dg
    scope = float(galaxy_config["snapshot_dt"]) * 0.5

    fine = GalData(ndim=2, coordinate_mode="polar")
    fine.set_corr(hdfra_config["correspondence"]).read_hdfra(
        hdfra_path=str(raw_path),
        time=time,
        log_path=str(log_path),
        func_parse_log=log_parser,
        scope=scope,
        target_config=hdfra_config.get("target"),
        source_metadata={
            "simulation_code": "MACER2D",
            "galaxy_type": galaxy_type,
            "galaxy_id": args.galaxy,
            "run_id": args.galaxy,
            "source_series_id": galaxy_config["folder_name"],
            "raw_snapshot_index": int(args.snapshot_index),
            "source_file_id": raw_path.name,
            "source_file_sha256": sha256_file(raw_path),
            "source_file_size_bytes": int(raw_path.stat().st_size),
            "time": time,
            "time_units": "Gyr",
            "time_mapping": galaxy_config.get(
                "time_mapping",
                {
                    "mode": "linear_index",
                    "snapshot_dt": float(galaxy_config["snapshot_dt"]),
                    "offset": float(galaxy_config.get("offset", 0.0)),
                },
            ),
        },
    )
    fine.set_coord(["fakeDim2", "fakeDim1"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fine.save_h5(args.output)

    restored = GalData().load_h5(args.output)
    density = np.asarray(restored.snapshot["density"], dtype=np.float64)
    volume = np.asarray(restored.grid_volume, dtype=np.float64)
    print(f"raw={raw_path}")
    print(f"output={args.output}")
    print(f"time={restored.time:.16g}, shape={density.shape}")
    print(
        f"volume_source={restored._volume_source}, "
        f"total_volume={volume.sum(dtype=np.float64):.16g}, "
        f"total_gas_mass={(density * volume).sum(dtype=np.float64):.16g}"
    )
    print(
        f"target={restored.sfr_target}, "
        f"format_version=5, parent_hr_uid={restored.parent_hr_uid}, "
        f"sample_uid={restored.sample_uid}, round_trip=OK"
    )


if __name__ == "__main__":
    main()
