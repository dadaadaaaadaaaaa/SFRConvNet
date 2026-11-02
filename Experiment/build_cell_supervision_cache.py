"""按现有 CachePT 的 UID/顺序导出独立逐格监督 sidecar，不改 split、输入或标量 cache。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from raw_data_processing.galaxy_data import GalData
from raw_data_processing.metadata_schema import metadata_from_json, sha256_file


def export_cell_cache(cache_path, product_dirs, output_path):
    """逐个 snapshot 对齐；过采样重复 UID 原样保留，不把网格拆成独立样本。"""
    cache_path, output_path = Path(cache_path), Path(output_path)
    if output_path.exists() or cache_path.resolve() == output_path.resolve():
        raise FileExistsError(f"拒绝覆盖 cache：{output_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("schema", {}).get("name") != "macnet_sfr_cache":
        raise ValueError("需要现有 MACNet 标量 CachePT，不能使用旧无身份 tensor cache。")
    identity = cache["sample_identity"]
    uids = list(identity["sample_uid"])
    parents = list(identity["parent_hr_uid"])
    tensors = cache["tensors"]
    if not uids or len(parents) != len(uids) or len(tensors["x"]) != len(uids):
        raise ValueError("cache UID 与样本轴不一致。")
    wanted = set(uids)
    paths = {}
    for folder in product_dirs:
        for path in sorted(Path(folder).glob("*.h5")):
            with h5py.File(path, "r") as handle:
                if handle.attrs.get("macnet_3d_format") == 1:
                    continue
                metadata = metadata_from_json(handle["metadata/contract_json"][()])
            uid = metadata.get("identity", {}).get("sample_uid")
            if uid in wanted:
                if uid in paths:
                    raise ValueError(f"产品目录中 sample_uid 重复：{uid}")
                paths[uid] = path
    missing = wanted - paths.keys()
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个 cache 对应产品，例如 {sorted(missing)[:3]}")
    shape = tuple(tensors["x"].shape[-2:])
    if len(tensors["x"].shape) != 4:
        raise ValueError("逐格监督 v1 只支持 (N,C,theta,r) 的二维原生输入。")
    output_names = {"Y": "target_sfr_map_physical", "B": "macer_sfr_baseline_map",
                    "C_true": "closure_true_map_physical"}
    maps = {name: np.empty((len(uids), *shape), dtype=np.float64) for name in output_names.values()}
    contracts, product_uids = {}, [None] * len(uids)
    indices = {}
    for i, uid in enumerate(uids):
        indices.setdefault(uid, []).append(i)
    # 每个源 HDF5 只读取一次，重复 snapshot 按原 cache 位置赋值。
    for count, (uid, rows) in enumerate(indices.items(), 1):
        galaxy = GalData().load_h5(paths[uid])
        if galaxy.grid_volume.shape != shape:
            raise ValueError("cache 与逐格产品 shape 不一致；禁止使用 legacy mirror cache。")
        values = galaxy.cell_sfr_supervision()
        for i in rows:
            if parents[i] != galaxy.parent_hr_uid:
                raise ValueError("cache parent_hr_uid 与 processed 不一致。")
            if not np.allclose(np.asarray(tensors["coord"][i]), galaxy.coordinate, rtol=1e-6, atol=1e-8):
                raise ValueError("cache 坐标与逐格标签不一致，不能套用翻转/镜像后的 cache。")
            if not np.isclose(float(tensors["time"][i]), galaxy.time, rtol=1e-6, atol=1e-9):
                raise ValueError("cache time 与 processed 不一致，请检查 v6/v7 迁移。")
            for key, total in (("target_sfr_physical", galaxy.sfr_target),
                               ("macer_sfr_baseline", galaxy.sfr_macer_baseline),
                               ("closure_true_physical", galaxy.sfr_closure_true)):
                if key not in tensors or not np.isclose(float(tensors[key][i]), total, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"cache {key} 与逐格产品总量不一致。")
        for key, name in output_names.items():
            maps[name][rows] = values[key]
        contract = galaxy.metadata["cell_supervision"]
        product_uid = contract["product_uid"]
        contracts[product_uid] = contract
        for i in rows:
            product_uids[i] = product_uid
        if count % 100 == 0 or count == len(indices):
            print(f"逐格 cache 对齐：{count}/{len(indices)} 个独立 snapshot", flush=True)
    payload = {
        "schema": {"name": "macnet_cell_supervision_cache", "version": 1},
        "source_cache": {"filename": cache_path.name, "sha256": sha256_file(cache_path)},
        "sample_identity": identity,
        "tensors": {name: torch.from_numpy(array) for name, array in maps.items()},
        "cell_product_uid": product_uids, "cell_contracts": contracts,
        "contract": {"axis_order": ["sample", "theta", "r"],
                     "units": "MACER_mass_code/Gyr", "to_msun_per_yr": 0.025,
                     "transformed": False, "is_model_input": False,
                     "split_policy": "exact_source_cache_order_including_oversampling",
                     "scalar_learning_contract_unchanged": True},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".cell_", suffix=".pt", dir=output_path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.link(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--product-dir", action="append", help="含逐格监督的产品目录，可重复")
    parser.add_argument("--config-dir", default=".config")
    parser.add_argument("--data-dir")
    parser.add_argument("--resolution-dirname", help="自动查找 data/*/<此目录>；与 --product-dir 二选一")
    args = parser.parse_args(argv)
    if bool(args.product_dir) == bool(args.resolution_dirname):
        parser.error("--product-dir 与 --resolution-dirname 必须且只能提供一个。")
    folders = args.product_dir
    if args.resolution_dirname:
        name = args.resolution_dirname
        if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            parser.error("--resolution-dirname 必须是单层目录名。")
        base = json.loads((Path(args.config_dir) / "BaseConfig.json").read_text(encoding="utf-8"))
        data_root = Path(args.data_dir or base["data_dir"])
        folders = sorted(p / name for p in data_root.iterdir() if (p / name).is_dir())
    path = export_cell_cache(args.cache_path, folders, args.output_path)
    print(f"已保存逐格监督 sidecar：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
