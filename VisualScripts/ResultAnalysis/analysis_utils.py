"""MACNet 训练结果的共享加载、推理、单位换算、指标计算和绘图工具。

本模块兼容版本化 CachePT、processed HDF5、新旧模型 checkpoint 与 direct/closure
目标契约，并为结果 notebook 和 ``analyze_exp2.py`` 提供统一分析接口。
"""

import os
import sys
import gc
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import h5py


# ==========================================
# 1. Project Setup & Imports
# ==========================================
def setup_project_env():
    """Found project root and set up environment."""
    current_path = Path.cwd()
    project_root = None
    markers = ["Experiment", "raw_data_processing", ".config"]

    # Try to find root from CWD (Notebook execution context)
    cand_path = current_path
    while cand_path != cand_path.parent:
        if any((cand_path / m).exists() for m in markers):
            project_root = cand_path
            break
        cand_path = cand_path.parent

    # Fallback: relative to this file if it's imported
    if project_root is None:
        try:
            # If __file__ is available
            project_root = Path(__file__).resolve().parent.parent.parent
        except NameError:
            # Fallback for some interactive environments
            project_root = Path(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

    # Add to sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ensure CWD is project root (fixes .config discovery in prepare_data.py)
    try:
        if Path.cwd() != project_root:
            os.chdir(project_root)
            print(f"[INFO] Changed CWD to Project Root: {project_root}")
    except Exception as e:
        print(f"[WARN] Could not switch CWD: {e}")

    return project_root


# Initialize root immediately for imports in this file
PROJECT_ROOT = setup_project_env()

# Project Imports
try:
    from raw_data_processing.galaxy_data import GalDataSet
    from Experiment.target_transform import (
        inverse_transform_sfr,
        inverse_transform_signed_closure,
        transform_sfr,
    )
    from Experiment.models import (
        MACNetRes_mbh,
        AccretionTransformer,
        MACNetFiLM,
        FourierPositionalEncoding2D,
        FiLMConditioner,
        MLP,
        SFRConvNet,
    )

    # Import Preprocess paths if available
    try:
        from Experiment.prepare_data import Exp_eg_folder_paths, Exp_dg_folder_paths
    except ImportError:
        Exp_eg_folder_paths = []
        Exp_dg_folder_paths = []
except ImportError as e:
    print(f"[WARN] Project imports failed in analysis_utils: {e}")


# ==========================================
# 2. Legacy Classes (from Exp2)
# ==========================================
class LegacyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, p_drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.drop1 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, p_drop)

    def forward(self, x, gamma=None, beta=None):
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        if gamma is not None and beta is not None:
            x = gamma * x + beta
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x


class LegacyAccretionTransformer(nn.Module):
    def __init__(
        self, c_in, d_model=256, n_layers=8, n_heads=8, d_ff=1024, pos_num_bands=32, pos_max_freq=64.0, p_drop=0.1
    ):
        super().__init__()
        self.posenc = FourierPositionalEncoding2D(
            num_bands=pos_num_bands, max_frequency=pos_max_freq, include_input=True
        )
        d_pos = self.posenc.out_dim
        self.content_proj = nn.Conv2d(c_in, d_model // 2, kernel_size=1)
        self.pos_proj = nn.Conv2d(d_pos, d_model // 2, kernel_size=1)
        self.drop = nn.Dropout(p_drop)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([LegacyTransformerBlock(d_model, n_heads, d_ff, p_drop) for _ in range(n_layers)])
        self.film = FiLMConditioner(n_layers, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, r, theta, bh_mass):
        B, C, H, W = x.shape
        pos = self.posenc(r, theta, normalize=True)
        feat = self.content_proj(x) + self.pos_proj(pos)
        feat = torch.cat([feat, self.drop(feat)], dim=1)
        tokens = feat.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        gammas, betas = self.film(bh_mass)
        x_tok = tokens
        for i, blk in enumerate(self.blocks):
            gamma = gammas[i].expand(B, x_tok.size(1), -1)
            beta = betas[i].expand(B, x_tok.size(1), -1)
            x_tok = blk(x_tok, gamma=gamma, beta=beta)
        return self.head(x_tok[:, 0, :]).squeeze(-1)


# ==========================================
# 3. Helper Functions
# ==========================================
def ensure_tensor(arr, dtype=torch.float32):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.to(dtype=dtype).contiguous()
    return torch.as_tensor(arr, dtype=dtype).contiguous()


def ensure_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def get_baseline_column(df):
    if "y_macer_baseline" in df.columns:
        return "y_macer_baseline"
    if "y_baseline" in df.columns:
        return "y_baseline"
    return None


def get_baseline_error_column(df):
    if "macer_baseline_error" in df.columns:
        return "macer_baseline_error"
    if "baseline_error" in df.columns:
        return "baseline_error"
    return None


def _target_metadata_from(obj=None):
    """Return normalized target metadata from a dataset or prediction DataFrame."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        metadata = obj
    elif isinstance(obj, pd.DataFrame):
        metadata = obj.attrs.get("target_metadata", {})
    else:
        metadata = getattr(obj, "target_metadata", {})
    metadata = dict(metadata or {})
    normalized = {}
    for key, value in metadata.items():
        normalized[key.removeprefix("target_")] = value
    return normalized


def _physical_sfr(values, metadata):
    """Invert the configured training transform; legacy runs use log10(SFR+floor)."""
    metadata = dict(metadata or {})
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    physical = np.full(values.shape, np.nan, dtype=float)
    physical[finite] = inverse_transform_sfr(
        values[finite],
        transform=metadata.get("transform", "log10_floor"),
        scale=float(metadata.get("scale", 1.0)),
        floor=float(metadata.get("floor", 1e-9)),
    )
    return physical


def _transformed_sfr_allow_negative_predictions(values, metadata):
    """只对非负 SFR 做目标变换；非法负预测保留为 NaN 而不是静默裁剪。"""

    metadata = dict(metadata or {})
    values = np.asarray(values, dtype=float)
    transformed = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    if np.any(valid):
        transformed[valid] = transform_sfr(
            values[valid],
            transform=metadata.get("transform", "log10_floor"),
            scale=float(metadata.get("scale", 1.0)),
            floor=float(metadata.get("floor", 1e-9)),
        )
    return transformed


def get_target_label(df=None):
    metadata = _target_metadata_from(df)
    transform = metadata.get("transform")
    units = metadata.get("units", "unknown units")
    units_tex = str(units).replace("_", r"\_")
    if transform == "log10_1p":
        scale = float(metadata.get("scale", 1.0))
        return rf"$\log_{{10}}[1+\mathrm{{SFR}}/({scale:g}\;\mathrm{{{units_tex}}})]$"
    return rf"$\log_{{10}}(\mathrm{{SFR}}/[\mathrm{{{units_tex}}}])$"


def get_physical_sfr_label(df=None):
    units = _target_metadata_from(df).get("units", "unknown units")
    return f"SFR [{units}]"


def _sfr_time_integration_factor(df, explicit_factor=None):
    """返回把 SFR×Gyr 转成太阳质量的因子，并拒绝未知单位的静默换算。"""

    if explicit_factor is not None:
        return float(explicit_factor)
    units = _target_metadata_from(df).get("units")
    if units == "MACER_mass_code/Gyr":
        return 2.5e7
    if units in {"Msun/yr", "M_sun/yr"}:
        return 1.0e9
    if units in {"Msun/Gyr", "M_sun/Gyr"}:
        return 1.0
    raise ValueError(
        f"不知道如何把 SFR 单位 {units!r} 与 Gyr 时间积分为太阳质量；"
        "请显式传入 sfr_unit_scale。"
    )


def get_baseline_label(df=None):
    if isinstance(df, pd.DataFrame) and "sfr_macer_baseline" in df.columns:
        return r"Baseline $\mathcal{S}[\mathcal{R}_\alpha(U_{\rm HR})]$"
    return "Baseline (KS)"


def find_latest_model(results_root: Path, exp_prefix: str) -> Path:
    if not results_root.exists():
        return None
    candidates = list(results_root.glob(f"{exp_prefix}*/*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ==========================================
# 4. Datasets
# ==========================================
def _cache_prefix_from_split_path(path):
    path = Path(path)
    name = path.name
    for suffix in ("train.pt", "val.pt", "test.pt", "out.pt"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return None


def load_cache_metadata(path):
    """Load channel metadata associated with a cache split or cache prefix."""
    path = Path(path)
    prefix = _cache_prefix_from_split_path(path)
    if prefix is None:
        prefix = path
    metadata_path = Path(f"{prefix}metadata.json")
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_norm_stats(norm_path):
    """Load normalization arrays plus optional channel metadata."""
    with np.load(norm_path, allow_pickle=False) as data:
        result = {
            "mean": data["mean"],
            "std": data["std"],
            "input_channels": tuple(data["input_channels"].tolist()) if "input_channels" in data else (),
            "excluded_channels": tuple(data["excluded_channels"].tolist()) if "excluded_channels" in data else (),
        }
        scalar_keys = (
            "target_quantity", "target_name", "target_dataset", "target_reduction",
            "target_units", "target_transform", "target_scale", "target_floor",
            "includes_zero_targets",
        )
        for key in scalar_keys:
            if key in data:
                result[key] = data[key].item()
        return result


class CachedPTDataset:
    def __init__(self, pt_path: Path):
        self.path = pt_path
        if not pt_path.exists():
            raise FileNotFoundError(f"Cached dataset not found: {pt_path}")

        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
        items = list(loaded) if isinstance(loaded, (list, tuple)) else None

        self.coord = None
        self.y_baseline = None
        self.baseline = None
        self.groups = None
        self.n_groups = None
        self.time = None
        self.type = "cached"
        self.sample_uid = None
        self.parent_hr_uid = None
        self.objective_mode = "direct_sfr"
        self.closure_transform = None
        self.closure_scale = None
        self.target_sfr_physical = None
        self.macer_sfr_baseline = None
        self.closure_true_physical = None
        self.closure_true_transformed = None
        self.conditioning_active = None
        self.conditioning_contract = None
        conditioning_value = None
        payload_conditioning_variable = None
        channel_metadata = load_cache_metadata(pt_path)
        self.input_channels = tuple(channel_metadata.get("input_channels", ()))
        self.excluded_channels = tuple(channel_metadata.get("excluded_channels", ()))
        self.target_metadata = {
            key.removeprefix("target_"): value
            for key, value in channel_metadata.items()
            if key.startswith("target_")
        }
        if "includes_zero_targets" in channel_metadata:
            self.target_metadata["includes_zero_targets"] = bool(channel_metadata["includes_zero_targets"])

        # 版本化缓存把 KS baseline、物理 baseline 与模型条件明确分开。
        if isinstance(loaded, dict):
            schema = loaded.get("schema", {})
            schema_version = int(schema.get("version", -1))
            if schema.get("name") != "macnet_sfr_cache" or schema_version not in {3, 4, 5}:
                raise RuntimeError(f"Unsupported cache schema in '{pt_path}': {schema!r}")
            tensors = loaded["tensors"]
            self.x = tensors["x"]
            self.coord = tensors["coord"]
            self.y = tensors["target_sfr"]
            self.y_baseline = tensors["sfr_ks_baseline"]
            self.sfr_ks_transformed = tensors["sfr_ks_baseline"]
            if schema_version >= 5:
                conditioning_value = tensors["conditioning_scalar"]
                conditioning_contract = loaded.get("model_input_contract", {}).get(
                    "conditioning", {}
                )
                self.conditioning_contract = dict(conditioning_contract)
                payload_conditioning_variable = conditioning_contract.get("name")
                self.conditioning_active = bool(
                    conditioning_contract.get("active", False)
                )
            else:
                conditioning_value = tensors["log_mass_filter"]
                payload_conditioning_variable = "log_mass_filter"
                self.conditioning_active = True
            self.groups = tensors["groups"]
            self.time = tensors["time"]
            self.target_sfr_physical = tensors.get("target_sfr_physical")
            self.macer_sfr_baseline = tensors.get("macer_sfr_baseline")
            self.closure_true_physical = tensors.get("closure_true_physical")
            self.closure_true_transformed = tensors.get("closure_true")
            learning_contract = loaded.get("learning_contract", {})
            self.objective_mode = learning_contract.get(
                "prepared_objective", learning_contract.get("objective_mode", "direct_sfr")
            )
            self.closure_transform = learning_contract.get("closure_transform")
            raw_closure_scale = learning_contract.get("closure_scale")
            self.closure_scale = (
                float(raw_closure_scale) if raw_closure_scale is not None else None
            )
            identity = loaded.get("sample_identity", {})
            self.sample_uid = np.asarray(identity.get("sample_uid", ()), dtype=str)
            self.parent_hr_uid = np.asarray(identity.get("parent_hr_uid", ()), dtype=str)
            self.kind = "vit"
        # Legacy SFR cache format:
        # (x, coord, target_sfr, transformed_ks_sfr, groups, n_groups, time)
        elif len(items) >= 7 and isinstance(items[1], (torch.Tensor, np.ndarray)) and np.asarray(items[1]).ndim >= 2:
            self.x, self.coord, self.y, self.y_baseline, self.groups, self.n_groups, self.time = items[:7]
            self.kind = "vit"
        elif len(items) >= 5 and isinstance(items[1], (torch.Tensor, np.ndarray)) and np.asarray(items[1]).ndim >= 2:
            self.x, self.coord, self.y, self.y_baseline, self.groups = items[:5]
            self.kind = "vit"
        elif len(items) >= 5:
            self.x, self.y, self.y_baseline, self.groups, self.n_groups = items[:5]
            self.kind = "macnet"
        else:
            self.x, self.y, self.y_baseline = items[:3]
            self.kind = "macnet"

        self.x = ensure_tensor(self.x)
        if self.input_channels and len(self.input_channels) != self.x.shape[1]:
            raise RuntimeError(
                f"Cache '{pt_path}' has {self.x.shape[1]} tensor channels but metadata lists "
                f"{len(self.input_channels)} channels: {self.input_channels}"
            )
        self.coord = ensure_tensor(self.coord)
        self.y = ensure_tensor(self.y).view(-1, 1)
        self.y_baseline = ensure_tensor(self.y_baseline).view(-1, 1) if self.y_baseline is not None else None
        self.baseline = self.y_baseline
        self.conditioning_scalar = (
            ensure_tensor(conditioning_value).view(-1, 1)
            if conditioning_value is not None
            else self.y_baseline
        )
        self.conditioning_variable = (
            payload_conditioning_variable
            or channel_metadata.get("conditioning_variable")
        )
        if self.conditioning_variable is None:
            # Compatibility with instantaneous-SFR caches created before this
            # field was recorded explicitly.
            if channel_metadata.get("target_quantity") == "instantaneous_sfr":
                self.conditioning_variable = "sfr_ks"
            else:
                self.conditioning_variable = "unknown"
        if self.conditioning_variable == "sfr_ks":
            self.sfr_ks_transformed = self.conditioning_scalar
        if self.conditioning_active is False and torch.count_nonzero(
            self.conditioning_scalar
        ).item() != 0:
            raise RuntimeError(
                f"Cache '{pt_path}' 声明 inactive conditioning，但占位张量并非全零。"
            )
        self.time = ensure_tensor(self.time) if self.time is not None else None

        for field_name in (
            "target_sfr_physical",
            "macer_sfr_baseline",
            "closure_true_physical",
            "closure_true_transformed",
        ):
            value = getattr(self, field_name)
            if value is not None:
                tensor_value = ensure_tensor(value).view(-1, 1)
                # 早期 v3 cache 用全 NaN 作为“尚无该物理产品”的占位，不能误判为
                # 一个不完整的 signed-closure 契约。
                if field_name != "target_sfr_physical" and torch.isnan(tensor_value).all():
                    tensor_value = None
                setattr(self, field_name, tensor_value)
        closure_product_fields = (
            self.macer_sfr_baseline,
            self.closure_true_physical,
            self.closure_true_transformed,
        )
        if any(value is not None for value in closure_product_fields):
            if self.target_sfr_physical is None or not all(
                value is not None for value in closure_product_fields
            ):
                raise RuntimeError(f"Cache '{pt_path}' 的 closure payload 不完整。")
            reconstructed = self.macer_sfr_baseline + self.closure_true_physical
            if not torch.allclose(
                reconstructed, self.target_sfr_physical, rtol=1e-5, atol=1e-7
            ):
                raise RuntimeError(f"Cache '{pt_path}' 不满足 Y=B+C。")
            if self.closure_transform not in {"signed_asinh", "identity"}:
                raise RuntimeError(
                    f"Cache '{pt_path}' 的 closure transform 非法：{self.closure_transform!r}。"
                )
            if self.closure_scale is None or not np.isfinite(self.closure_scale) or self.closure_scale <= 0:
                raise RuntimeError(
                    f"Cache '{pt_path}' 的 closure scale 非法：{self.closure_scale!r}。"
                )

        if self.groups is not None:
            self.group_names = None
            groups_arr = np.asarray(self.groups)
            if groups_arr.dtype.kind in ("S", "U"):
                # 如果 group 本身就是字符串 numpy 数组
                self.group_names = groups_arr
                _, indices = np.unique(groups_arr, return_inverse=True)
                self.groups = torch.as_tensor(indices, dtype=torch.long)
            elif isinstance(self.groups, (list, tuple)) and len(self.groups) > 0 and isinstance(self.groups[0], str):
                # 如果 group 是一组字符串列表
                self.group_names = np.array(self.groups)
                _, indices = np.unique(self.group_names, return_inverse=True)
                self.groups = torch.as_tensor(indices, dtype=torch.long)
            else:
                try:
                    self.groups = torch.as_tensor(self.groups)
                except Exception as e:
                    print(f"Warning: Could not automatically convert groups to tensor. {e}")

    def __len__(self):
        return len(self.y)


def _resolve_mirror_mode(mirror, native_shape, reference_shape=None):
    """根据缓存张量与原生 HDF5 网格的空间形状决定是否执行旧版 mirror。"""
    native_shape = tuple(int(value) for value in native_shape)
    reference_shape = (
        tuple(int(value) for value in reference_shape)
        if reference_shape is not None
        else None
    )
    if isinstance(mirror, (bool, np.bool_)):
        return bool(mirror)
    if mirror != "match_reference":
        raise ValueError(
            "mirror 必须是布尔值或 'match_reference'，"
            f"实际得到 {mirror!r}。"
        )
    if reference_shape is None:
        raise ValueError("mirror='match_reference' 需要 reference_dataset。")
    if reference_shape == native_shape:
        return False
    mirrored_shape = (native_shape[0], native_shape[1] * 2)
    if reference_shape == mirrored_shape:
        return True
    raise RuntimeError(
        "无法使原生 HDF5 空间形状与缓存形状匹配："
        f"native={native_shape}, reference={reference_shape}。"
    )


class HDF5Dataset:
    def __init__(
        self,
        hdf5_conf,
        resolution="coarse",
        mean=None,
        std=None,
        log_transform=True,
        mirror=False,
        balance=None,
        input_channels=None,
        exclude_channels=None,
        reference_dataset=None,
        target_metadata=None,
        exclude_raw_snapshot_indices=None,
    ):
        if isinstance(hdf5_conf, (list, tuple)):
            self.folder_path = hdf5_conf[0]
        else:
            self.folder_path = hdf5_conf

        self.type = os.path.basename(os.path.dirname(self.folder_path)).lower()

        loaded = GalDataSet().load_data((self.folder_path, 0))
        loaded.drop_invalid(require_positive_targets=False, require_positive_ks=False)
        if exclude_raw_snapshot_indices:
            loaded.exclude_raw_snapshot_indices(exclude_raw_snapshot_indices)

        if reference_dataset is not None:
            if input_channels is None:
                input_channels = getattr(reference_dataset, "input_channels", None) or None
            if exclude_channels is None:
                exclude_channels = getattr(reference_dataset, "excluded_channels", None) or None
            if target_metadata is None:
                target_metadata = _target_metadata_from(reference_dataset)
        if input_channels is not None and exclude_channels is not None:
            # An explicit include list is sufficient and avoids ambiguous rules.
            exclude_channels = None
        loaded.select_input_channels(include=input_channels, exclude=exclude_channels)
        self.input_channels = loaded.input_channels
        self.excluded_channels = tuple(exclude_channels or ())

        native_shape = tuple(loaded.x.shape[-2:])
        reference_shape = (
            tuple(reference_dataset.x.shape[-2:])
            if reference_dataset is not None
            else None
        )
        self.mirror_applied = _resolve_mirror_mode(
            mirror,
            native_shape,
            reference_shape,
        )
        if self.mirror_applied:
            loaded.mirror_data()
        if balance is not None:
            loaded.balance_groups(method=balance)

        if mean is not None and std is not None:
            mean_shape = np.asarray(mean).shape
            norm_channels = int(mean_shape[0] if len(mean_shape) == 1 else mean_shape[1])
            if norm_channels != len(self.input_channels):
                raise ValueError(
                    f"Normalization has {norm_channels} channels but HDF5 input selection has "
                    f"{len(self.input_channels)} ({self.input_channels}). Pass input_channels, "
                    "exclude_channels, or reference_dataset from the matching cached experiment."
                )
            self.x = loaded.standardize(mean, std, log_transform=log_transform)
        else:
            self.x, _, _ = loaded.standardize(log_transform=log_transform)
        self.coord = loaded.coordinate
        if reference_shape is not None and tuple(self.x.shape[-2:]) != reference_shape:
            raise RuntimeError(
                "Raw HDF5 预处理后的空间形状与缓存不一致："
                f"raw={tuple(self.x.shape[-2:])}, reference={reference_shape}, "
                f"mirror_applied={self.mirror_applied}。"
            )
        raw_target_metadata = loaded.target_metadata
        if raw_target_metadata.get("quantity") != "instantaneous_sfr":
            raise ValueError(
                f"HDF5 target is {raw_target_metadata.get('quantity')!r}, expected 'instantaneous_sfr'. "
                "Regenerate the processed HDF5 snapshots."
            )
        self.target_metadata = dict(target_metadata or raw_target_metadata)
        self.target_metadata.setdefault("transform", "log10_1p")
        self.target_metadata.setdefault("scale", 1.0)
        self.target_metadata.setdefault("floor", 1e-9)
        for key in ("quantity", "dataset", "reduction", "units"):
            if key in raw_target_metadata and key in self.target_metadata:
                if raw_target_metadata[key] != self.target_metadata[key]:
                    raise ValueError(
                        f"Raw HDF5 target {key}={raw_target_metadata[key]!r} does not match "
                        f"reference target {self.target_metadata[key]!r}."
                    )
        transform_options = {
            "transform": self.target_metadata["transform"],
            "scale": float(self.target_metadata["scale"]),
            "floor": float(self.target_metadata["floor"]),
        }
        self.y = transform_sfr(loaded.sfr_target, **transform_options)
        self.target_sfr_physical = np.asarray(loaded.sfr_target, dtype=np.float64)
        self.macer_sfr_baseline = np.asarray(
            loaded.sfr_macer_baseline, dtype=np.float64
        )
        self.closure_true_physical = np.asarray(
            loaded.sfr_closure_true, dtype=np.float64
        )
        reference_objective = getattr(reference_dataset, "objective_mode", "direct_sfr")
        self.objective_mode = reference_objective
        self.closure_transform = getattr(reference_dataset, "closure_transform", None)
        self.closure_scale = getattr(reference_dataset, "closure_scale", None)
        if reference_objective == "signed_closure":
            if not np.all(np.isfinite(self.macer_sfr_baseline)) or not np.all(
                np.isfinite(self.closure_true_physical)
            ):
                raise RuntimeError(
                    "signed-closure 分析要求所有 processed HDF5 都保存 MACER baseline 与 closure。"
                )
            if not np.allclose(
                self.macer_sfr_baseline + self.closure_true_physical,
                self.target_sfr_physical,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise RuntimeError("Processed HDF5 不满足 Y=B+C。")

        # KS 经验基线始终作为独立对照保存，不能冒充 MACER-LR 物理基线。
        self.y_baseline = transform_sfr(loaded.sfr_ks, **transform_options)
        self.baseline = self.y_baseline
        self.sfr_ks_transformed = self.y_baseline
        reference_conditioning = getattr(reference_dataset, "conditioning_contract", None)
        if reference_conditioning and not bool(reference_conditioning.get("active", False)):
            self.conditioning_scalar = np.zeros(len(self.y), dtype=np.float32)
            self.conditioning_variable = "none"
            self.conditioning_active = False
            self.conditioning_contract = dict(reference_conditioning)
        else:
            # 旧 SFR 模型继续使用变换后的 KS 标量作为 FiLM 条件。
            self.conditioning_scalar = self.y_baseline
            self.conditioning_variable = "sfr_ks"
            self.conditioning_active = True
            self.conditioning_contract = {
                "name": "sfr_ks",
                "active": True,
                "transform": self.target_metadata["transform"],
                "source": "legacy_sfr_conditioning",
                "dimension": 1,
            }
        self.time = loaded.time
        self.mdot_edd = np.array(loaded._raw_data["mdot_edd"]) if "mdot_edd" in loaded._raw_data else None
        self.sfr_target = loaded.sfr_target

        self.kind = "vit" if self.coord is not None else "macnet"

    def __len__(self):
        return len(self.y)


# ==========================================
# 5. Model Loading
# ==========================================
def load_analysis_model(model_path, device, datasets_dict=None):
    """Refactored model loading logic."""
    model_path = Path(model_path)
    state = torch.load(model_path, map_location=device, weights_only=False)
    raw_state = state.get("model", state.get("state_dict", state))
    keys = list(raw_state.keys())

    # Check for hyperparams FIRST
    config_path = model_path.parent / "hyperparameters.json"
    loaded_params = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_params = json.load(f)
        except:
            pass
    conditioning_variable = (
        state.get("conditioning_variable") if isinstance(state, dict) else None
    ) or loaded_params.get("conditioning_variable")
    objective_mode = (
        state.get("objective_mode") if isinstance(state, dict) else None
    ) or loaded_params.get("objective_mode", "direct_sfr")
    closure_transform = (
        state.get("closure_transform") if isinstance(state, dict) else None
    ) or loaded_params.get("closure_transform")
    closure_scale = (
        state.get("closure_scale") if isinstance(state, dict) else None
    )
    if closure_scale is None:
        closure_scale = loaded_params.get("closure_scale")
    if objective_mode not in {"direct_sfr", "signed_closure"}:
        raise RuntimeError(
            f"Checkpoint 声明了未知 objective_mode={objective_mode!r}。"
        )
    if objective_mode == "signed_closure":
        if closure_transform not in {"signed_asinh", "identity"}:
            raise RuntimeError(
                f"Signed-closure checkpoint 的 transform 非法：{closure_transform!r}。"
            )
        if closure_scale is None or not np.isfinite(float(closure_scale)) or float(closure_scale) <= 0:
            raise RuntimeError(
                f"Signed-closure checkpoint 的 scale 非法：{closure_scale!r}。"
            )
        closure_scale = float(closure_scale)
    if conditioning_variable is None and loaded_params.get("target_quantity") == "instantaneous_sfr":
        # Compatibility with SFR checkpoints created before this metadata was
        # saved explicitly. Their PT fourth field is transformed KS-law SFR.
        conditioning_variable = "sfr_ks"

    # Infer Type
    model_type = "macnet"
    if loaded_params and "model_type" in loaded_params:
        model_type = loaded_params["model_type"]
    else:
        if any(k.startswith("film_generator") for k in keys):
            model_type = "resfilm"
        elif any(k.startswith("cls_token") or k.startswith("posenc") for k in keys) or any(
            k.startswith("blocks") for k in keys
        ):
            if any(k.startswith("stem.") for k in keys):
                model_type = "convnet"
            else:
                model_type = "vit"

    # Infer Input Channels. New checkpoints/cache metadata are authoritative;
    # legacy runs still fall back to the supplied analysis dataset.
    checkpoint_channels = tuple(state.get("input_channels", ())) if isinstance(state, dict) else ()
    checkpoint_n_channels = state.get("n_input_channels") if isinstance(state, dict) else None
    configured_n_channels = loaded_params.get("n_input_channels") if loaded_params else None
    in_ch = int(checkpoint_n_channels or configured_n_channels or 14)
    dataset_channels = ()
    checkpoint_target_metadata = _target_metadata_from(
        state.get("target_metadata", {}) if isinstance(state, dict) else {}
    )
    if not checkpoint_target_metadata and loaded_params:
        checkpoint_target_metadata = _target_metadata_from(loaded_params.get("target_metadata", {}))
    if datasets_dict:
        ref_ds = next((ds for ds in datasets_dict.values() if ds is not None), None)
        if ref_ds:
            dataset_n_channels = int(ref_ds.x.shape[1])
            if (checkpoint_n_channels or configured_n_channels) and dataset_n_channels != in_ch:
                raise ValueError(
                    f"Model expects {in_ch} input channels but analysis dataset has "
                    f"{dataset_n_channels}."
                )
            dataset_channels = tuple(getattr(ref_ds, "input_channels", ()))
            if checkpoint_channels and dataset_channels and checkpoint_channels != dataset_channels:
                raise ValueError(
                    f"Model channels {checkpoint_channels} do not match dataset channels "
                    f"{dataset_channels}."
                )
            in_ch = dataset_n_channels
            dataset_target_metadata = _target_metadata_from(ref_ds)
            if bool(checkpoint_target_metadata) != bool(dataset_target_metadata):
                raise ValueError(
                    "Model and analysis dataset do not both provide target metadata. "
                    "Do not mix a legacy cumulative-SFR run with an instantaneous-SFR cache."
                )
            for key in ("quantity", "dataset", "reduction", "transform", "scale", "floor"):
                if key in checkpoint_target_metadata or key in dataset_target_metadata:
                    if checkpoint_target_metadata.get(key) != dataset_target_metadata.get(key):
                        raise ValueError(
                            f"Model target {key}={checkpoint_target_metadata.get(key)!r} does not match "
                            f"dataset target {dataset_target_metadata.get(key)!r}."
                        )
            dataset_objective = getattr(ref_ds, "objective_mode", "direct_sfr")
            if objective_mode == "signed_closure" and dataset_objective != "signed_closure":
                raise ValueError(
                    "Signed-closure checkpoint 必须配合包含 B、C 的 signed-closure cache。"
                )
            if objective_mode == "signed_closure":
                if getattr(ref_ds, "closure_transform", None) != closure_transform or not np.isclose(
                    float(getattr(ref_ds, "closure_scale", np.nan)), closure_scale
                ):
                    raise ValueError(
                        "Checkpoint 与分析 cache 的 closure transform/scale 不一致。"
                    )

    # Initialize
    model = None
    if model_type == "vit":
        # Default Params
        d_model = 256
        n_layers = 8
        n_heads = 4
        d_ff = 1024
        pos_num_bands = 32

        # Override from JSON
        if loaded_params:
            d_model = loaded_params.get("d_model", d_model)
            n_layers = loaded_params.get("n_layers", n_layers)
            n_heads = loaded_params.get("n_heads", n_heads)
            d_ff = loaded_params.get("d_ff", d_ff)
            pos_num_bands = loaded_params.get("pos_num_bands", pos_num_bands)
        else:
            # Fallback Inference
            if "cls_token" in raw_state:
                d_model = raw_state["cls_token"].shape[-1]
            block_ids = [int(k.split(".")[1]) for k in keys if k.startswith("blocks.")]
            n_layers = max(block_ids) + 1 if block_ids else 8
            for k in keys:
                if "mlp.fc1.weight" in k:
                    d_ff = raw_state[k].shape[0]
                    break
            n_heads = max(1, d_model // 32)
            if "posenc.freq_bands" in raw_state:
                pos_num_bands = raw_state["posenc.freq_bands"].shape[0]

        # Check Legacy
        is_legacy = False
        if "content_proj.weight" in raw_state:
            cp_shape = raw_state["content_proj.weight"].shape
            if cp_shape[0] == d_model // 2:
                is_legacy = True

        if is_legacy:
            model = LegacyAccretionTransformer(
                c_in=in_ch, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands
            ).to(device)
        else:
            model = AccretionTransformer(
                c_in=in_ch, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, pos_num_bands=pos_num_bands
            ).to(device)

    elif model_type == "convnet":
        base_ch = 64
        stage_depths = (2, 2, 2, 2)
        pos_num_bands = 32
        pos_max_freq = 64.0
        p_drop = 0.1
        se_reduction = 4
        use_film = False

        if loaded_params:
            base_ch = loaded_params.get("base_ch", base_ch)
            sd_val = loaded_params.get("stage_depths", "2,2,2,2")
            if isinstance(sd_val, str):
                stage_depths = tuple(int(x) for x in sd_val.split(","))
            elif isinstance(sd_val, (list, tuple)):
                stage_depths = tuple(sd_val)
            pos_num_bands = loaded_params.get("pos_num_bands", pos_num_bands)
            pos_max_freq = loaded_params.get("pos_max_freq", pos_max_freq)
            p_drop = loaded_params.get("p_drop", p_drop)
            se_reduction = loaded_params.get("se_reduction", se_reduction)
            use_film = loaded_params.get("use_film", use_film)

        if use_film and conditioning_variable not in {"sfr_ks", "log_mass_filter"}:
            raise RuntimeError(
                "SFRConvNet checkpoint 启用了 FiLM，但没有声明活动的 KS 或 "
                f"R_m 分辨率条件（得到 {conditioning_variable!r}）。"
            )

        model = SFRConvNet(
            c_in=in_ch,
            base_ch=base_ch,
            stage_depths=stage_depths,
            pos_num_bands=pos_num_bands,
            pos_max_freq=pos_max_freq,
            p_drop=p_drop,
            se_reduction=se_reduction,
            use_film=use_film,
            prediction_mode=objective_mode,
        ).to(device)

    elif model_type == "resfilm":
        if conditioning_variable == "sfr_ks":
            raise RuntimeError(
                "MACNetFiLM is a legacy BH-mass-conditioned model and cannot be "
                "loaded with an SFR cache whose conditioning variable is 'sfr_ks'."
            )
        model = MACNetFiLM(in_channels=in_ch).to(device)
    else:
        if conditioning_variable == "sfr_ks":
            raise RuntimeError(
                "MACNetRes_mbh is a legacy BH-mass-conditioned model and cannot be "
                "loaded with an SFR cache whose conditioning variable is 'sfr_ks'."
            )
        model = MACNetRes_mbh(in_channels=in_ch).to(device)

    try:
        model.load_state_dict(raw_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{model_path}' is not structurally compatible with the inferred {model_type} model."
        ) from exc

    model.eval()
    model.conditioning_variable = conditioning_variable or "unknown"
    model.objective_mode = objective_mode
    model.prediction_mode = objective_mode
    model.closure_transform = closure_transform
    model.closure_scale = closure_scale
    model.input_channels = checkpoint_channels or dataset_channels
    model.target_metadata = checkpoint_target_metadata
    return model, model_type


# ==========================================
# 6. Prediction
# ==========================================
@torch.no_grad()
def predict_dataset(ds, model, device, model_type, batch_size=32, return_feats=False):
    if ds is None:
        return (None, None, None) if return_feats else None

    x_all = ensure_tensor(ds.x)
    if x_all is None or len(x_all) == 0:
        empty_df = pd.DataFrame(columns=["y_true", "y_pred", "error", "type"])
        return (empty_df, None, None) if return_feats else empty_df

    x_data = ensure_numpy(x_all)
    conditioning_all = ensure_tensor(
        getattr(ds, "conditioning_scalar", getattr(ds, "y_baseline", None))
    )
    dataset_conditioning = getattr(ds, "conditioning_variable", "unknown")
    model_conditioning = getattr(model, "conditioning_variable", "unknown")
    if (
        dataset_conditioning != "unknown"
        and model_conditioning != "unknown"
        and dataset_conditioning != model_conditioning
    ):
        raise RuntimeError(
            "Model and dataset conditioning semantics differ: "
            f"model={model_conditioning!r}, dataset={dataset_conditioning!r}."
        )
    coord_all = ensure_tensor(ds.coord) if getattr(ds, "coord", None) is not None else None

    total = len(ds)
    preds = []
    feats_list = []

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    # ViT remains conservative because attention memory grows quickly. ConvNet
    # inference uses the caller-provided value; gradients are disabled here, so
    # it can normally use at least the training batch size.
    bs = 8 if model_type == "vit" else batch_size

    # Hook for features if requested
    hook_handle = None
    if return_feats:
        if model_type == "vit":
            def get_activation(storage):
                def hook(model, input, output):
                    storage.append(output[:, 0, :].detach().cpu())
                return hook

            if hasattr(model, "blocks") and len(model.blocks) > 0:
                hook_handle = model.blocks[-1].register_forward_hook(get_activation(feats_list))
        elif model_type == "convnet":
            def get_activation_convnet(storage):
                def hook(model, input, output):
                    storage.append(output.reshape(output.shape[0], -1).detach().cpu())
                return hook

            if hasattr(model, "head") and len(model.head) >= 3:
                hook_handle = model.head[2].register_forward_hook(get_activation_convnet(feats_list))

    try:
        for start in range(0, total, bs):
            end = min(start + bs, total)
            x_batch = x_all[start:end].to(device)
            conditioning_batch = (
                conditioning_all[start:end].to(device).view(-1)
                if conditioning_all is not None
                else None
            )

            if model_type in ["vit", "convnet"]:
                if coord_all is None:
                    raise ValueError(f"Model type '{model_type}' requires coordinate inputs, but dataset has no coords.")
                c_batch = coord_all[start:end].to(device)
                y_out = model(x_batch, c_batch[:, 0:1], c_batch[:, 1:2], conditioning_batch)
            else:
                y_out = model(x_batch, conditioning_batch)
            preds.append(y_out.cpu().view(-1))
    finally:
        if hook_handle:
            hook_handle.remove()

    model_output = torch.cat(preds).numpy()
    y_true = ensure_tensor(ds.y).view(-1).numpy()
    if not np.all(np.isfinite(model_output)):
        raise RuntimeError("Model predictions contain NaN or Inf.")
    feats = torch.cat(feats_list, dim=0).numpy() if feats_list else None

    target_metadata = _target_metadata_from(ds)
    model_target_metadata = _target_metadata_from(model)
    if model_target_metadata and target_metadata and model_target_metadata != target_metadata:
        raise ValueError("Model and dataset target metadata differ; refusing to mix target definitions.")
    objective_mode = getattr(model, "objective_mode", "direct_sfr")
    target_physical_data = getattr(ds, "target_sfr_physical", None)
    sfr_true = (
        ensure_numpy(target_physical_data).reshape(-1)
        if target_physical_data is not None
        else _physical_sfr(y_true, target_metadata)
    )
    if objective_mode == "signed_closure":
        baseline_data = getattr(ds, "macer_sfr_baseline", None)
        if baseline_data is None:
            raise RuntimeError(
                "Signed-closure 推理缺少 macer_sfr_baseline，无法构造 B+C_hat。"
            )
        macer_baseline = ensure_numpy(baseline_data).reshape(-1)
        closure_transform = getattr(model, "closure_transform", None)
        closure_scale = getattr(model, "closure_scale", None)
        closure_pred = inverse_transform_signed_closure(
            model_output,
            transform=closure_transform,
            scale=float(closure_scale),
        )
        sfr_pred = macer_baseline + closure_pred
        # 负最终 SFR 是模型诊断信号，不能通过 clamp 篡改；目标变换坐标中记为 NaN。
        y_pred = _transformed_sfr_allow_negative_predictions(sfr_pred, target_metadata)
    else:
        closure_pred = None
        sfr_pred = _physical_sfr(model_output, target_metadata)
        y_pred = model_output

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "model_output_raw": model_output,
            "sfr_true": sfr_true,
            "sfr_pred": sfr_pred,
        }
    )
    df.attrs["target_metadata"] = target_metadata
    df.attrs["objective_mode"] = objective_mode
    df.attrs["closure_transform"] = getattr(model, "closure_transform", None)
    df.attrs["closure_scale"] = getattr(model, "closure_scale", None)

    if objective_mode == "signed_closure":
        df["closure_pred_physical"] = closure_pred
        closure_true_data = getattr(ds, "closure_true_physical", None)
        if closure_true_data is None:
            raise RuntimeError("Signed-closure 分析缺少 closure_true_physical。")
        df["closure_true_physical"] = ensure_numpy(closure_true_data).reshape(-1)

    macer_data = getattr(ds, "macer_sfr_baseline", None)
    if macer_data is not None:
        df["sfr_macer_baseline"] = ensure_numpy(macer_data).reshape(-1)
        df["y_macer_baseline"] = _transformed_sfr_allow_negative_predictions(
            df["sfr_macer_baseline"].to_numpy(), target_metadata
        )
        df["macer_baseline_error"] = df["y_macer_baseline"] - df["y_true"]
        # 兼容旧绘图接口，但语义明确指向 MACER-LR baseline。
        df["sfr_baseline"] = df["sfr_macer_baseline"]
        df["y_baseline"] = df["y_macer_baseline"]
        df["baseline_error"] = df["macer_baseline_error"]

    ks_data = getattr(ds, "y_baseline", None)
    if ks_data is not None:
        df["y_ks_baseline"] = ensure_tensor(ks_data).view(-1).numpy()
        df.loc[np.isclose(df["y_ks_baseline"], -100), "y_ks_baseline"] = np.nan
        df["ks_baseline_error"] = df["y_ks_baseline"] - df["y_true"]
        df["sfr_ks_baseline"] = _physical_sfr(
            df["y_ks_baseline"].to_numpy(), target_metadata
        )
        if "y_baseline" not in df.columns:
            df["y_baseline"] = df["y_ks_baseline"]
            df["baseline_error"] = df["ks_baseline_error"]
            df["sfr_baseline"] = df["sfr_ks_baseline"]

    if hasattr(ds, "time") and ds.time is not None:
        df["time"] = ensure_numpy(ds.time).reshape(-1)
    if hasattr(ds, "mdot_edd") and ds.mdot_edd is not None:
        df["mdot_edd"] = ensure_numpy(ds.mdot_edd).reshape(-1)

    df["error"] = df["y_pred"] - df["y_true"]
    df["sfr_error"] = df["sfr_pred"] - df["sfr_true"]

    if hasattr(ds, "group_names") and ds.group_names is not None and len(ds.group_names) == len(df):
        df["type"] = ds.group_names
    elif hasattr(ds, "type"):
        df["type"] = ds.type
    elif hasattr(ds, "groups") and ds.groups is not None:
        df["type"] = "unknown_group"
    else:
        df["type"] = "test_set"

    del x_all, conditioning_all, coord_all
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if return_feats:
        return df, feats, x_data
    return df


def save_to_h5(path, df, feats, x_data, ds_type):
    with h5py.File(path, "w") as f:
        f.attrs["dataset_type"] = ds_type
        f.attrs["objective_mode"] = df.attrs.get("objective_mode", "direct_sfr")
        if df.attrs.get("closure_transform") is not None:
            f.attrs["closure_transform"] = df.attrs["closure_transform"]
        if df.attrs.get("closure_scale") is not None:
            f.attrs["closure_scale"] = float(df.attrs["closure_scale"])
        for key, value in _target_metadata_from(df).items():
            f.attrs[f"target_{key}"] = value
        f.create_dataset("x", data=x_data, compression="gzip")
        f.create_dataset("y_true", data=df["y_true"].values)
        f.create_dataset("y_pred", data=df["y_pred"].values)
        for column in (
            "model_output_raw",
            "sfr_true",
            "sfr_pred",
            "sfr_baseline",
            "sfr_macer_baseline",
            "sfr_ks_baseline",
            "closure_true_physical",
            "closure_pred_physical",
        ):
            if column in df.columns:
                f.create_dataset(column, data=df[column].values)
        if "type" in df.columns:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("label", data=df["type"].values.astype(object), dtype=dt)
        if feats is not None:
            f.create_dataset("latent", data=feats)
        if "y_baseline" in df.columns:
            f.create_dataset("y_baseline", data=df["y_baseline"].values)
        for column in ("y_macer_baseline", "y_ks_baseline"):
            if column in df.columns:
                f.create_dataset(column, data=df[column].values)
        if "time" in df.columns:
            f.create_dataset("time", data=df["time"].values)
        if "mdot_edd" in df.columns:
            f.create_dataset("mdot_edd", data=df["mdot_edd"].values)


# ==========================================
# 7. Visualization
# ==========================================
COLORS = ["#4B8D69", "#55658B", "#c1c16f", "#685587", "#B353CB", "#D78E5A", "#F06969"]


def save_plot(plt_obj, name, fig_dir, fmt="png"):
    path = fig_dir / f"{name}.{fmt}"
    plt_obj.savefig(path, dpi=150, bbox_inches="tight")


def plot_scatter(df, title, fig_dir, limit_range=True):
    baseline_col = get_baseline_column(df)
    has_baseline = baseline_col is not None
    cols_to_check = ["y_true", "y_pred"]
    if has_baseline:
        cols_to_check.append(baseline_col)

    vmin = df[cols_to_check].min().min() - 0.5
    vmax = df[cols_to_check].max().max() + 0.5

    def _plot_base(ax, show_baseline=False):
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.5, label="Ideal", zorder=5)
        ax.plot(
            [vmin, vmax], [vmin - 1, vmax - 1], "k--", lw=0.8,
            alpha=0.5, label=r"$\pm$1 target-transform unit"
        )
        ax.plot([vmin, vmax], [vmin + 1, vmax + 1], "k--", lw=0.8, alpha=0.5)

        groups = df["type"].unique()
        for i, g in enumerate(groups):
            sub = df[df["type"] == g]
            ax.scatter(
                sub["y_true"],
                sub["y_pred"],
                s=8,
                alpha=0.7,
                color=COLORS[i % len(COLORS)],
                label=f"{g}",
                edgecolors="none",
            )

        # Only plot valid baseline values to avoid alignment issues after NaN filtering.
        if show_baseline and baseline_col is not None:
            valid_mask = df[baseline_col].notna()
            ax.scatter(
                df.loc[valid_mask, "y_true"],
                df.loc[valid_mask, baseline_col],
                s=15,
                marker="x",
                alpha=0.4,
                color="gray",
                label=get_baseline_label(df),
                zorder=1,
            )

        ax.set_xlabel(f"True {get_target_label(df)}")
        ax.set_ylabel(f"Predicted {get_target_label(df)}")
        if limit_range:
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
        ax.grid(True, linestyle=":", alpha=0.3)

    if has_baseline:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120, sharex=True, sharey=True)
        _plot_base(axes[0], show_baseline=False)
        axes[0].set_title(f"{title} (Model Only)")
        _plot_base(axes[1], show_baseline=True)
        axes[1].set_title(f"{title} (With Baseline)")
        axes[0].legend(loc="upper left")
        axes[1].legend(loc="upper left")
        fig.tight_layout()
        save_plot(plt, f"scatter_{title.replace(' ', '_').lower()}", fig_dir)
        plt.show()
    else:
        plt.figure(figsize=(6, 6), dpi=120)
        ax = plt.gca()
        _plot_base(ax, show_baseline=False)
        ax.set_title(title)
        ax.legend(loc="upper left")
        save_plot(plt, f"scatter_{title.replace(' ', '_').lower()}", fig_dir)
        plt.show()


def plot_error_dist(df, title, fig_dir, model_label="Model"):
    plt.figure(figsize=(8, 5), dpi=120)
    plot_data = []
    tmp = df[["error", "type"]].copy()
    tmp["Method"] = model_label
    tmp.rename(columns={"error": "Error"}, inplace=True)
    plot_data.append(tmp)
    baseline_error_col = get_baseline_error_column(df)
    if baseline_error_col is not None:
        tmp2 = df[[baseline_error_col, "type"]].dropna().copy()
        tmp2["Method"] = get_baseline_label(df).replace("Baseline ", "")
        tmp2.rename(columns={baseline_error_col: "Error"}, inplace=True)
        plot_data.append(tmp2)
    # 模型与 baseline 都沿用原 DataFrame 的样本索引；纵向拼接后必须重建唯一索引，
    # 否则新版 seaborn 在内部 reindex 时会因重复标签报错。
    final_df = pd.concat(plot_data, ignore_index=True)
    split = final_df["Method"].nunique() == 2
    sns.violinplot(data=final_df, x="type", y="Error", hue="Method", split=split, inner="quartile", palette="muted")
    plt.axhline(0, color="r", linestyle="--", alpha=0.5)
    plt.axhline(1, color="gray", linestyle=":", alpha=0.5)
    plt.axhline(-1, color="gray", linestyle=":", alpha=0.5)
    plt.title(f"Error Distribution: {title}")
    plt.ylabel("Error (target-transform units)")
    plt.xticks(rotation=45, ha="right")
    save_plot(plt, f"error_dist_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_kde_contour(df, title, fig_dir):
    plt.figure(figsize=(6, 6), dpi=120)
    sns.kdeplot(
        x=df["y_true"],
        y=df["y_pred"],
        hue=df["type"],
        fill=True,
        alpha=0.3,
        levels=5,
        palette=COLORS[: len(df["type"].unique())],
    )
    vmin, vmax = df["y_true"].min(), df["y_true"].max()
    plt.plot([vmin, vmax], [vmin, vmax], "r--", lw=1)
    plt.title(f"Density: {title}")
    save_plot(plt, f"kde_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_physical_sfr_scatter(df, title, fig_dir):
    """比较最终物理 SFR；signed-closure 的负预测不会被裁剪。"""

    required = {"sfr_true", "sfr_pred"}
    if not required.issubset(df.columns):
        return
    true = df["sfr_true"].to_numpy(dtype=float)
    pred = df["sfr_pred"].to_numpy(dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not np.any(valid):
        return
    values = np.concatenate([true[valid], pred[valid]])
    max_abs = max(float(np.max(np.abs(values))), 1.0)
    positive = np.abs(values[np.nonzero(values)])
    linthresh = max(float(np.quantile(positive, 0.01)) if positive.size else 1e-6, 1e-9)

    plt.figure(figsize=(6.5, 6), dpi=120)
    plt.scatter(true[valid], pred[valid], s=7, alpha=0.55, edgecolors="none")
    plt.plot([-max_abs, max_abs], [-max_abs, max_abs], "r--", lw=1.2, label="Ideal")
    baseline_col = "sfr_macer_baseline" if "sfr_macer_baseline" in df.columns else None
    if baseline_col is not None:
        base = df[baseline_col].to_numpy(dtype=float)
        base_valid = np.isfinite(true) & np.isfinite(base)
        plt.scatter(
            true[base_valid], base[base_valid], s=10, marker="x", alpha=0.3,
            color="gray", label=get_baseline_label(df)
        )
    plt.xscale("symlog", linthresh=linthresh)
    plt.yscale("symlog", linthresh=linthresh)
    plt.xlabel("True physical SFR [MACER mass code / Gyr]")
    plt.ylabel("Predicted physical SFR [MACER mass code / Gyr]")
    plt.title(f"Physical SFR: {title}")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend()
    save_plot(plt, f"physical_sfr_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_closure_scatter(df, title, fig_dir):
    """绘制 signed closure 的物理真值与预测值。"""

    required = {"closure_true_physical", "closure_pred_physical"}
    if not required.issubset(df.columns):
        return
    true = df["closure_true_physical"].to_numpy(dtype=float)
    pred = df["closure_pred_physical"].to_numpy(dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not np.any(valid):
        return
    values = np.concatenate([true[valid], pred[valid]])
    bound = max(float(np.max(np.abs(values))), 1.0)
    nonzero = np.abs(values[np.nonzero(values)])
    linthresh = max(float(np.quantile(nonzero, 0.01)) if nonzero.size else 1e-6, 1e-9)

    plt.figure(figsize=(6.5, 6), dpi=120)
    plt.scatter(true[valid], pred[valid], s=7, alpha=0.55, edgecolors="none")
    plt.plot([-bound, bound], [-bound, bound], "r--", lw=1.2)
    plt.xscale("symlog", linthresh=linthresh)
    plt.yscale("symlog", linthresh=linthresh)
    plt.xlim(-bound, bound)
    plt.ylim(-bound, bound)
    plt.xlabel(r"True closure $C_\alpha$")
    plt.ylabel(r"Predicted closure $\widehat C_\alpha$")
    plt.title(f"Signed closure: {title}")
    plt.grid(True, linestyle=":", alpha=0.3)
    save_plot(plt, f"closure_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_mass_evolution(df, title, fig_dir, model_label="Model"):
    """
    Backward-compatible wrapper.
    Older notebooks call this function for BH growth analysis; in the SFR
    pipeline we reinterpret it as SFR history versus time.
    """
    return plot_sfr_history(df, title, fig_dir, model_label=model_label)


def _sfr_series(df, physical_column, transformed_column):
    if physical_column in df.columns:
        return df[physical_column].to_numpy(dtype=float)
    return _physical_sfr(df[transformed_column].to_numpy(dtype=float), _target_metadata_from(df))


def _cumulative_trapezoid(values, times, time_unit_to_years):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if len(values) != len(times):
        raise ValueError("SFR and time arrays must have the same length.")
    if len(values) == 0:
        return values
    dt = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(dt < 0):
        raise ValueError("Times must be finite and sorted in non-decreasing order.")
    increments = 0.5 * (values[:-1] + values[1:]) * dt * time_unit_to_years
    return np.concatenate(([0.0], np.cumsum(increments)))


def plot_cumulative_sfr(df, title, fig_dir, model_label="Model", sfr_unit_scale=None):
    """
    Plot cumulative stellar mass formed from the reconstructed SFR history.

    True/prediction/KS are instantaneous SFR. The plotted cumulative mass is
    derived with trapezoidal time integration; it is not a separately trained target.
    `time` 必须以 Gyr 表示；SFR 单位到太阳质量的换算由 target metadata 决定。
    """
    if "time" not in df.columns:
        return

    df_plot = df.sort_values("time").copy()
    times = df_plot["time"].values
    integration_factor = _sfr_time_integration_factor(
        df_plot, explicit_factor=sfr_unit_scale
    )
    sfr_true = _sfr_series(df_plot, "sfr_true", "y_true")
    sfr_pred = _sfr_series(df_plot, "sfr_pred", "y_pred")
    mass_true = _cumulative_trapezoid(sfr_true, times, integration_factor)
    mass_pred = _cumulative_trapezoid(sfr_pred, times, integration_factor)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(times, mass_true, "k-", lw=2, label="True", alpha=0.8)
    plt.plot(times, mass_pred, "r--", lw=2, label=f"Predicted ({model_label})", alpha=0.8)

    baseline_col = get_baseline_column(df_plot)
    if baseline_col is not None:
        sfr_baseline = _sfr_series(df_plot, "sfr_baseline", baseline_col)
        mass_baseline = _cumulative_trapezoid(sfr_baseline, times, integration_factor)
        plt.plot(times, mass_baseline, "g:", lw=1.5, label=get_baseline_label(df_plot), alpha=0.7)

    plt.yscale("log")

    plt.title(f"Cumulative Stellar Mass Formed: {title}")
    plt.xlabel("Time (Gyr)")
    plt.ylabel(r"$M_{\star,\mathrm{formed}}$ [$M_{\odot}$]")
    plt.legend()
    save_plot(plt, f"cum_sfr_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def plot_sfr_history(df, title, fig_dir, model_label="Model"):
    """
    Plot the reconstructed SFR history versus time.

    Values are inverted with the transform metadata saved beside the cache.
    `symlog` is used so genuine zero-SFR snapshots remain visible.
    """
    if "time" not in df.columns:
        return

    df_plot = df.sort_values("time").copy()
    times = df_plot["time"].values
    sfr_true = _sfr_series(df_plot, "sfr_true", "y_true")
    sfr_pred = _sfr_series(df_plot, "sfr_pred", "y_pred")

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(times, sfr_true, "k-", lw=2, label="True", alpha=0.8)
    plt.plot(times, sfr_pred, "r--", lw=2, label=f"Predicted ({model_label})", alpha=0.8)

    baseline_col = get_baseline_column(df_plot)
    if baseline_col is not None:
        sfr_baseline = _sfr_series(df_plot, "sfr_baseline", baseline_col)
        plt.plot(times, sfr_baseline, "g:", lw=1.5, label=get_baseline_label(df_plot), alpha=0.7)

    positive = np.concatenate([sfr_true[sfr_true > 0], sfr_pred[sfr_pred > 0]])
    linthresh = max(float(np.nanmin(positive)) if positive.size else 1e-9, 1e-12)
    plt.yscale("symlog", linthresh=linthresh)
    plt.title(f"SFR History: {title}")
    plt.xlabel("Time (Gyr)")
    plt.ylabel(get_physical_sfr_label(df_plot))
    plt.legend()
    save_plot(plt, f"sfr_history_{title.replace(' ', '_').lower()}", fig_dir)
    plt.show()


def get_metrics_summary(y_true, y_pred, name=""):
    try:
        from sklearn.metrics import r2_score
    except:

        def r2_score(t, p):
            residual = np.sum((t - p) ** 2)
            total = np.sum((t - np.mean(t)) ** 2)
            if total == 0:
                return 1.0 if residual == 0 else 0.0
            return 1 - residual / total

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return {
            "Dataset": name,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "OutlierFrac (|target error|>1)": np.nan,
            "N_valid": 0,
        }
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    d = y_true - y_pred
    return {
        "Dataset": name,
        "RMSE": np.sqrt(np.mean(d**2)),
        "MAE": np.mean(np.abs(d)),
        "R2": r2_score(y_true, y_pred),
        "OutlierFrac (|target error|>1)": np.mean(np.abs(d) > 1.0),
        "N_valid": int(valid.sum()),
    }


def get_physical_sfr_metrics(df, prediction_column="sfr_pred", name="Model"):
    """在物理 SFR 空间计算指标，并保留负最终预测的诊断比例。"""

    if "sfr_true" not in df.columns or prediction_column not in df.columns:
        raise KeyError(
            f"Physical metrics require 'sfr_true' and {prediction_column!r}."
        )
    true = df["sfr_true"].to_numpy(dtype=float)
    pred = df[prediction_column].to_numpy(dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    if not np.any(valid):
        return {
            "Dataset": name,
            "Physical_RMSE": np.nan,
            "Physical_MAE": np.nan,
            "Physical_R2": np.nan,
            "NegativePredFrac": np.nan,
            "N_valid": 0,
        }
    true_valid = true[valid]
    pred_valid = pred[valid]
    try:
        from sklearn.metrics import r2_score
        r2 = r2_score(true_valid, pred_valid)
    except Exception:
        residual = np.sum((true_valid - pred_valid) ** 2)
        total = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2 = 1.0 - residual / total if total > 0 else float(residual == 0)
    error = pred_valid - true_valid
    nonzero_true = np.abs(true_valid) > 0
    median_relative = (
        float(
            np.median(
                np.abs(error[nonzero_true]) / np.abs(true_valid[nonzero_true])
            )
        )
        if np.any(nonzero_true)
        else np.nan
    )
    return {
        "Dataset": name,
        "Physical_RMSE": float(np.sqrt(np.mean(error**2))),
        "Physical_MAE": float(np.mean(np.abs(error))),
        "Physical_R2": float(r2),
        "MedianAbsRelativeError_nonzero": median_relative,
        "NegativePredFrac": float(np.mean(pred_valid < 0)),
        "N_valid": int(valid.sum()),
    }
