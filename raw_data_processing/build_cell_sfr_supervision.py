"""为已有二维 fine/R_h 产品补齐逐格 SFR 监督；另存目录，不重算气体和 baseline。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from raw_data_processing.coarsening_common import block_reduce_sum
from raw_data_processing.galaxy_data import GalData, _validate_macer_dataset_labels
from raw_data_processing.metadata_schema import sha256_file, validate_complete_metadata


def attach_from_raw(galaxy: GalData, raw_path: Path) -> GalData:
    """使用源 SHA256 和权威体积核验对应关系，再保守归约原始标签。"""
    from pyhdf.SD import SD, SDC

    metadata = galaxy.metadata
    validate_complete_metadata(metadata)
    if galaxy.ndim != 2 or metadata["source"].get("simulation_code") != "MACER2D":
        raise ValueError("此入口仅处理 MACER2D。")
    operator = metadata["derivation"]["operator_id"]
    if operator not in {"identity_fine", "spatial_block"}:
        raise ValueError("只能补齐 identity_fine/spatial_block；R_m 需实际输运权重。")
    expected_hash = metadata["source"].get("source_file_sha256")
    if not expected_hash or sha256_file(raw_path) != expected_hash:
        raise ValueError("原始文件 SHA256 与 processed source 身份不一致。")
    previous = os.getcwd()
    try:
        # HDF4 Windows 库不能可靠处理中文绝对路径，仅读取阶段临时切换目录。
        os.chdir(raw_path.resolve().parent)
        handle = SD(raw_path.name, SDC.READ)
        try:
            _validate_macer_dataset_labels(handle, {"Data-Set-11", "Data-Set-15"})
            fields = []
            for name in ("Data-Set-11", "Data-Set-15"):
                dataset = handle.select(name)
                try:
                    fields.append(np.asarray(dataset[:], dtype=np.float64).squeeze())
                finally:
                    dataset.endaccess()
        finally:
            handle.end()
    finally:
        os.chdir(previous)
    target, volume = fields
    if target.ndim != 2 or volume.shape != target.shape:
        raise ValueError("原始逐格标签与体积 shape 不一致。")
    if not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("原始逐格 SFR 非有限或为负，不能通过求和抵消这些错误。")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("原始网格体积必须有限且为正。")
    shape = galaxy.grid_volume.shape
    if operator == "identity_fine":
        if target.shape != shape:
            raise ValueError("fine 标签 shape 不一致。")
        mapped_target, mapped_volume = target, volume
    else:
        parameters = metadata["derivation"]["operator_parameters"]
        if (parameters.get("aggregation") != "non_overlapping_index_blocks"
                or list(target.shape) != parameters.get("source_shape")
                or list(shape) != parameters.get("output_shape")):
            raise ValueError("R_h 分块参数与原始/当前 shape 不一致。")
        mapped_target = block_reduce_sum(target, shape)
        mapped_volume = block_reduce_sum(volume, shape)
    if not np.allclose(mapped_volume, galaxy.grid_volume, rtol=1e-12, atol=1e-14):
        raise ValueError("原始体积映射与 processed 网格不一致。")
    return galaxy.attach_cell_sfr_target(mapped_target)


def augment_product(input_path, output_path, *, raw_root):
    """另存一份附有逐格标签的完整产品，原 scalar UID 和数值保持不变。"""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists() or input_path.resolve() == output_path.resolve():
        raise FileExistsError(f"拒绝覆盖已有产品：{output_path}")
    galaxy = GalData().load_h5(input_path)
    if not isinstance(galaxy, GalData):
        raise ValueError("逐格监督 v1 不支持三维产品。")
    source = galaxy.metadata["source"]
    root = Path(raw_root).resolve()
    folder = root / source["source_series_id"]
    candidates = [folder / source["source_file_id"], folder / "data" / source["source_file_id"]]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1 or not matches[0].resolve().is_relative_to(root):
        raise FileNotFoundError(f"无法唯一确定原始文件：{candidates}")
    attach_from_raw(galaxy, matches[0])
    galaxy._metadata["cell_supervision_migration"] = {
        "version": 1, "input_processed_sha256": sha256_file(input_path),
        "raw_sha256_verified": source["source_file_sha256"],
        "baseline_policy": "reuse_existing_cell_baseline_without_recomputation",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix=".h5", prefix=".cell_", dir=output_path.parent)
    os.close(descriptor)
    try:
        galaxy.save_h5(temporary)
        restored = GalData().load_h5(temporary)
        if restored.sample_uid != galaxy.sample_uid:
            raise ValueError("逐格扩展不应改变 sample_uid。")
        # 硬链接发布不覆盖已存在文件；源和目标位于同一文件系统。
        os.link(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--galaxy", action="append", required=True, help="完整星系生成名，可重复")
    parser.add_argument("--config-dir", default=".config")
    parser.add_argument("--raw-dir", help="覆盖 BaseConfig.raw_data_dir")
    parser.add_argument("--data-dir", help="覆盖 BaseConfig.data_dir")
    parser.add_argument("--input-dirname", required=True)
    parser.add_argument("--output-dirname", required=True)
    parser.add_argument("--snapshot-index", type=int, action="append")
    parser.add_argument("--max-snapshots", type=int)
    args = parser.parse_args(argv)
    base = json.loads((Path(args.config_dir) / "BaseConfig.json").read_text(encoding="utf-8"))
    raw_root = Path(args.raw_dir or base["raw_data_dir"])
    data_root = Path(args.data_dir or base["data_dir"])
    for name in [args.input_dirname, args.output_dirname, *args.galaxy]:
        if name in {".", "..", ""} or Path(name).name != name or "/" in name or "\\" in name:
            parser.error("星系及产品目录名必须是单层目录名。")
    if args.input_dirname == args.output_dirname:
        parser.error("输入与输出目录必须不同。")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        parser.error("--max-snapshots 必须为正。")
    jobs = []
    for galaxy in dict.fromkeys(args.galaxy):
        paths = sorted((data_root / galaxy / args.input_dirname).glob("*.h5"))
        if args.snapshot_index:
            requested = set(args.snapshot_index)
            paths = [p for p in paths if p.stem.isdigit() and int(p.stem) in requested]
            missing = requested - {int(p.stem) for p in paths}
            if missing:
                parser.error(f"{galaxy} 缺少 snapshot：{sorted(missing)}")
        if not paths:
            parser.error(f"{galaxy} 没有可处理产品。")
        if args.max_snapshots is not None:
            paths = paths[:args.max_snapshots]
        jobs.extend((p, data_root / galaxy / args.output_dirname / p.name) for p in paths)
    for _, output in jobs:
        if output.exists():
            parser.error(f"输出已存在，请使用新目录或选择尚未处理的 snapshot：{output}")
    for index, (source, output) in enumerate(jobs, 1):
        augment_product(source, output, raw_root=raw_root)
        print(f"[{index}/{len(jobs)}] 逐格监督：{output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
