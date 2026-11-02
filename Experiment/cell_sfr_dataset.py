"""按 UID 绑定既有标量 cache 与逐格监督 sidecar，提供独立的二维空间训练数据接口。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch
from torch.utils.data import Dataset

from Experiment.train import GalPTDataset
from Experiment.cell_sfr_objective import fit_cell_loss_scales
from raw_data_processing.cell_sfr_supervision import array_digest, validate_cell_arrays
from raw_data_processing.metadata_schema import canonical_json, sha256_file


class CellSFRDataset(Dataset):
    """返回字典 batch，避免将逐格数组错误广播到原标量 loss。

    split 必须显式提供；只接受固定二维 R_h、原生网格及默认 data_compatible baseline。
    不重新划分、不重新归一化、不做 mirror/旋转。原 cache 可以来自 Direct 或 Closure 实验，
    但必须已经保存完整物理 Y/B/C 和无活动条件的输入契约。
    """

    def __init__(self, cache_path, cell_cache_path, *, split):
        if split not in {"train", "val", "test", "out"}:
            raise ValueError("split 必须为 train/val/test/out。")
        cache_path, cell_cache_path = Path(cache_path), Path(cell_cache_path)
        for known in ("train", "val", "test", "out"):
            if (cache_path.stem == known or cache_path.stem.endswith("_" + known)) and known != split:
                raise ValueError("显式 split 与源 cache 文件名不一致。")
        # 仅加载由用户自己生成的可信 PyTorch cache。
        cell = torch.load(cell_cache_path, map_location="cpu", weights_only=False)
        if cell.get("schema") != {"name": "macnet_cell_supervision_cache", "version": 1}:
            raise ValueError("逐格 sidecar schema 不匹配。")
        source_hash = sha256_file(cache_path)
        if cell.get("source_cache", {}).get("sha256") != source_hash:
            raise ValueError("逐格 sidecar 与源 cache SHA256 不匹配。")
        self.base = GalPTDataset(cache_path, objective_mode="signed_closure")
        self.split = split
        self.source_cache_sha256 = source_hash
        self.cell_cache_sha256 = sha256_file(cell_cache_path)
        if self.base.conditioning_active or self.base.conditioning_variable != "none":
            raise ValueError("逐格 U-Net 首版仅支持 conditioning=none 的固定 R_h。")
        if split in {"train", "val"} and any("torus" in name.lower() for name in self.base.group_names):
            raise ValueError("torus 不能进入逐格训练、验证或尺度估计。")
        self.sample_uid = self.base.sample_uid.tolist()
        self.parent_hr_uid = self.base.parent_hr_uid.tolist()
        identity = {"sample_uid": self.sample_uid, "parent_hr_uid": self.parent_hr_uid}
        if cell.get("sample_identity") != identity or not self.sample_uid or not all(self.sample_uid + self.parent_hr_uid):
            raise ValueError("逐格 sidecar 的 UID/顺序与源 cache 不一致。")
        contract = cell.get("contract", {})
        if (contract.get("units") != "MACER_mass_code/Gyr" or contract.get("transformed") is not False
                or contract.get("is_model_input") is not False
                or contract.get("axis_order") != ["sample", "theta", "r"]):
            raise ValueError("逐格 sidecar 物理量/轴顺序契约不匹配。")
        self.input_channels = self.base.input_channels
        if not self.input_channels or any(any(token in name.lower() for token in
            ("sfr", "closure", "baseline", "rhodotsf", "newstar")) for name in self.input_channels):
            raise ValueError("输入通道缺失或混入监督量。")
        n, _, h, w = self.base.x.shape
        for name, tensor in (("x", self.base.x), ("coord", self.base.coord)):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"源 cache {name} 含非有限值。")
        if self.base.coord.shape != (n, 2, h, w) or (self.base.coord[:, 0] <= 0).any():
            raise ValueError("坐标必须为 (N,2,H,W)，且球坐标半径为正。")
        maps = {}
        for name in ("target_sfr_map_physical", "macer_sfr_baseline_map", "closure_true_map_physical"):
            tensor = torch.as_tensor(cell["tensors"][name], dtype=torch.float64)
            if tensor.shape != (n, h, w):
                raise ValueError(f"{name} 与源 cache shape 不一致。")
            maps[name] = tensor.contiguous()
        y, b, c = (maps[name] for name in
                   ("target_sfr_map_physical", "macer_sfr_baseline_map", "closure_true_map_physical"))
        product_uids = cell.get("cell_product_uid", [])
        if len(product_uids) != n:
            raise ValueError("逐格 product UID 数量不匹配。")
        contracts = cell.get("cell_contracts", {})
        for i, product_uid in enumerate(product_uids):
            product = contracts.get(product_uid, {})
            declaration = {k: v for k, v in product.items() if k != "product_uid"}
            expected_uid = "macnetcell:" + sha256(canonical_json(declaration).encode()).hexdigest()
            if product.get("product_uid") != product_uid or expected_uid != product_uid:
                raise ValueError("逐格 product UID 与契约不一致。")
            if (product.get("identity") != {"sample_uid": self.sample_uid[i], "parent_hr_uid": self.parent_hr_uid[i]}
                    or product.get("operator_id") != "spatial_block"
                    or product.get("schema_name") != "macnet_cell_sfr_supervision"
                    or product.get("schema_version") != 1
                    or product.get("shape") != [h, w]
                    or product.get("canonical_mode") != "data_compatible"
                    or product.get("units") != "MACER_mass_code/Gyr"):
                raise ValueError("逐格产品身份、形状或算子不匹配。")
            arrays = [v[i].numpy() for v in (y, b, c)]
            mode = product["modes"]["data_compatible"]
            if [array_digest(a) for a in arrays] != [product["target_sha256"], mode["baseline_sha256"], mode["closure_sha256"]]:
                raise ValueError("逐格数组摘要不匹配，可能发生标签错位或损坏。")
            validate_cell_arrays(*arrays, shape=(h, w), totals=tuple(float(a.sum()) for a in arrays))
        for tensor, original in ((y, self.base.target_sfr_physical), (b, self.base.macer_sfr_baseline),
                                 (c, self.base.closure_true_physical)):
            # 原 GalPTDataset 将标量转为 FP32，比较时允许相应舍入误差。
            if not torch.allclose(tensor.sum((1, 2)), original.double().flatten(), rtol=2e-6, atol=1e-7):
                raise ValueError("逐格地图总和与原 cache 的标量监督不一致。")
        self.target = y.unsqueeze(1)
        self.baseline = b.unsqueeze(1)
        self.closure_true = c.unsqueeze(1)
        self.spatial_shape = (h, w)

    def __len__(self):
        return len(self.sample_uid)

    def __getitem__(self, index):
        return {"x": self.base.x[index], "coord": self.base.coord[index],
                "target_map": self.target[index], "baseline_map": self.baseline[index],
                "closure_true_map": self.closure_true[index],
                "sample_uid": self.sample_uid[index], "parent_hr_uid": self.parent_hr_uid[index],
                "group": self.base.groups[index], "time": self.base.time[index]}

    def fit_scales(self):
        """训练 cache 顺序（包括重复 UID）定义统计分布；验证/测试直接复用返回的契约。"""
        return fit_cell_loss_scales(self.target, self.baseline, split=self.split,
                                   provenance={"source_cache_sha256": self.source_cache_sha256,
                                               "cell_cache_sha256": self.cell_cache_sha256})


def validate_cell_splits(**datasets):
    """检查 split 间 UID/母帧隔离与相同通道/网格；训练内部过采样重复不报错。"""
    items = list(datasets.items())
    for name, dataset in items:
        if name != dataset.split:
            raise ValueError("split 键与数据对象声明不一致。")
        reference = items[0][1]
        if dataset.input_channels != reference.input_channels or dataset.spatial_shape != reference.spatial_shape:
            raise ValueError("各 split 的通道或空间尺寸不一致。")
    for index, (name, left) in enumerate(items):
        for other, right in items[index + 1:]:
            for key in ("sample_uid", "parent_hr_uid"):
                if set(getattr(left, key)).intersection(getattr(right, key)):
                    raise ValueError(f"{name}/{other} 的 {key} 有交叉。")
