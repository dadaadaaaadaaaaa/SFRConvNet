"""为已含权威网格体积的 processed fine HDF5 补充完整 metadata。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from raw_data_processing.galaxy_data import GalData, _build_snapshot_schedule
    from raw_data_processing.parse_log_file import load_config
    from raw_data_processing.metadata_schema import sha256_file
except ModuleNotFoundError:
    from galaxy_data import GalData, _build_snapshot_schedule
    from parse_log_file import load_config
    from metadata_schema import sha256_file


def _resolve_galaxy(configs: dict, generated_name: str):
    matches = []
    for galaxy_type, group in configs["RawDataConfig"].items():
        for nickname, galaxy_config in group.items():
            if f"{galaxy_type}_{nickname}" == generated_name:
                matches.append((galaxy_type, galaxy_config))
    if len(matches) != 1:
        raise KeyError(f"无法唯一识别 galaxy 配置: {generated_name!r}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Upgrade one processed fine HDF5 file to metadata schema v1."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--galaxy", required=True)
    parser.add_argument("--snapshot-index", required=True, type=int)
    parser.add_argument("--config-dir", type=Path, default=Path(".config"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖 {args.output}；如确需覆盖请传入 --overwrite。")
    configs = load_config(str(args.config_dir))
    galaxy_type, galaxy_config = _resolve_galaxy(configs, args.galaxy)
    raw_root = Path(configs["BaseConfig"]["raw_data_dir"])
    raw_folder = raw_root / galaxy_config["folder_name"] / "data"
    if not raw_folder.is_dir():
        raw_folder = raw_root / galaxy_config["folder_name"]
    raw_names = sorted(
        path.name for path in raw_folder.glob("hdfra.*") if path.is_file()
    )
    times, physical_indices = _build_snapshot_schedule(raw_names, galaxy_config)
    positions = np.flatnonzero(physical_indices == args.snapshot_index)
    if positions.size != 1:
        raise FileNotFoundError(
            f"无法在 {raw_folder} 中唯一定位 snapshot {args.snapshot_index}。"
        )
    position = int(positions[0])
    expected_time = float(times[position])
    source_file_id = raw_names[position]
    source_path = raw_folder / source_file_id

    fine = GalData().load_h5(args.input)
    if not np.isclose(float(fine.time), expected_time, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"输入 HDF5 时间 {fine.time} 与配置时间 {expected_time} 不一致。"
        )
    fine.attach_fine_source_metadata(
        {
            "simulation_code": "MACER2D",
            "galaxy_type": galaxy_type,
            "galaxy_id": args.galaxy,
            "run_id": args.galaxy,
            "source_series_id": galaxy_config["folder_name"],
            "raw_snapshot_index": int(args.snapshot_index),
            "source_file_id": source_file_id,
            "source_file_sha256": sha256_file(source_path),
            "source_file_size_bytes": int(source_path.stat().st_size),
            "time": expected_time,
            "time_units": "Gyr",
            "metadata_upgrade_source": "processed_format_v4_with_Data-Set-15",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fine.save_h5(args.output)
    restored = GalData().load_h5(args.output)
    print(f"output={args.output}")
    print(f"parent_hr_uid={restored.parent_hr_uid}")
    print(f"sample_uid={restored.sample_uid}")
    print("metadata_round_trip=OK")


if __name__ == "__main__":
    main()
