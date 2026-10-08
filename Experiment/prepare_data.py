"""把 processed MACER HDF5 转换为版本化 PyTorch 训练缓存。

本模块负责输入通道选择、无效/初始化样本排除、分组划分、训练组平衡、
训练集标准化，以及 direct-SFR 或 signed-closure 监督量和完整 metadata 的保存。
固定分辨率 R_h 使用无活动条件标量，R_m 使用 ``log10(M_filter)``，两种语义
不得在同一 cache 中静默混合。
"""

import argparse
import json
import os
import re
import sys
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_data_processing.galaxy_data import GalDataSet
from raw_data_processing.geometry_features import (
    SUPPORTED_GEOMETRY_INPUT_MODES,
    geometry_input_contract,
)
from raw_data_processing.parse_log_file import load_config
from Experiment.target_transform import (
    SUPPORTED_CLOSURE_TRANSFORMS,
    SUPPORTED_TARGET_TRANSFORMS,
    transform_sfr,
    transform_signed_closure,
)
from Experiment.geometry_contract import validate_geometry_contract


def _optional_sample_array(data, names):
    """从 GalDataSet 的正式属性或 raw_data 中读取可选逐样本物理量。"""

    for name in names:
        if hasattr(type(data), name):
            value = getattr(data, name)
            if callable(value):
                value = value()
            return np.asarray(value, dtype=np.float64).reshape(-1)
    raw_data = getattr(data, "raw_data", {})
    for name in names:
        if name in raw_data:
            return np.asarray(raw_data[name], dtype=np.float64).reshape(-1)
    return None


_RH_OPERATOR_IDS = {"spatial_block", "identity_fine"}
_RM_OPERATOR_IDS = {
    "adaptive_mass_axisymmetric_v0",
    "adaptive_mass_axisymmetric_v1",
    "axisymmetric_rm_v0",
    "axisymmetric_rm_v1",
}

_ZOU_FIVE_GROUPS = (
    "disk_galaxy_fiducial",
    "disk_galaxy_fiducial_4",
    "disk_galaxy_fiducial_7",
    "disk_galaxy_model_0_75",
    "disk_galaxy_model_1_1",
)
_TORUS_TEST_GROUP = "disk_galaxy_torus_100yr"


def _resolve_conditioning_contract(data):
    """按粗粒化算子的物理语义构造统一条件标量。

    固定分辨率 R_h 不需要逐样本分辨率条件，因此使用 inactive 的零占位；
    R_m 保持原有 ``log10(M_filter)`` 数值和字段语义。不同算子不得在同一
    cache 中静默共享一个含义不一致的标量。
    """

    n_samples = int(np.asarray(data.sfr_target).reshape(-1).size)
    raw_data = getattr(data, "raw_data", {})
    operator_values = raw_data.get("operator_id")
    if operator_values is not None:
        operator_ids = tuple(
            sorted({str(value) for value in np.asarray(operator_values).reshape(-1)})
        )
    else:
        operator_ids = ()

    if operator_ids and set(operator_ids).issubset(_RH_OPERATOR_IDS):
        return np.zeros(n_samples, dtype=np.float32), {
            "name": "none",
            "active": False,
            "transform": "identity",
            "units": "dimensionless",
            "source": "fixed_rh_resolution",
            "dimension": 1,
            "operator_ids": list(operator_ids),
        }

    if operator_ids and set(operator_ids).issubset(_RM_OPERATOR_IDS):
        log_mass_filter = np.asarray(data.log_mass_filter, dtype=np.float32)
        return log_mass_filter, {
            "name": "log_mass_filter",
            "active": True,
            "transform": "log10",
            "units": "MACER_mass_code",
            "source": "coarsening.mass_filter",
            "dimension": 1,
            "operator_ids": list(operator_ids),
        }

    if len(operator_ids) > 1:
        raise ValueError(
            "一个 CachePT 暂不允许混合具有不同条件语义的 R_h/R_m 算子；"
            f"检测到 operator_id={list(operator_ids)}。请分别生成 cache。"
        )

    # 兼容不含 operator_id 的旧测试夹具和外部 GalDataSet 实现。只有在其
    # 明确提供合法 log_mass_filter 时，才按旧 R_m 契约处理。
    try:
        log_mass_filter = np.asarray(data.log_mass_filter, dtype=np.float32)
    except (AttributeError, KeyError, TypeError, ValueError):
        log_mass_filter = None
    if (
        log_mass_filter is not None
        and log_mass_filter.shape == (n_samples,)
        and np.all(np.isfinite(log_mass_filter))
    ):
        return log_mass_filter, {
            "name": "log_mass_filter",
            "active": True,
            "transform": "log10",
            "units": "MACER_mass_code",
            "source": "legacy_explicit_log_mass_filter",
            "dimension": 1,
            "operator_ids": list(operator_ids),
        }

    raise ValueError(
        "无法确定训练条件语义。新 processed HDF5 必须提供一致的 operator_id；"
        "固定 R_h 使用 spatial_block，R_m 必须同时提供正有限 mass_filter。"
    )


class CachePt: # 特征过滤、数据集划分、镜像扩增、样本平衡、计算均值和方差、归一化、保存为.pt文件
    def __init__(self, dataset: GalDataSet = None):
        self._dataset = dataset
        self._last_cache_schema_version = None
        self._last_conditioning_contract = None
        self._last_log_transform_mask = None
        self._last_geometry_input_contract = None

    # ------------ Properties -----------

    @property
    def dataset(self):
        return self._dataset

    @property
    def input_channels(self):
        return self._dataset.input_channels

    @property
    def last_cache_schema_version(self):
        return self._last_cache_schema_version

    @property
    def last_conditioning_contract(self):
        if self._last_conditioning_contract is None:
            return None
        return dict(self._last_conditioning_contract)

    @property
    def last_log_transform_mask(self):
        if self._last_log_transform_mask is None:
            return None
        return np.asarray(self._last_log_transform_mask, dtype=bool).copy()

    @property
    def last_geometry_input_contract(self):
        return self._last_geometry_input_contract

    # ------------ Dunder Methods ------------

    def __len__(self): # 通过调用time，返回数据集一共有多少个snapshot
        return len(self._dataset.raw_data["time"])

    def __add__(self, other: "CachePt") -> "CachePt": # 拼接dataset
        if not isinstance(other, CachePt):
            raise TypeError("Can only add CachePt to another CachePt")
        combined_dataset = self._dataset + other._dataset
        return CachePt(combined_dataset)

    # ------------ Methods ------------

    def load(
        self,
        path: str,
        *,
        include_channels=None,
        exclude_channels=None,
        include_zero_targets: bool = True,
        require_target_quantity: str | None = "instantaneous_sfr",
        allow_mixed_coarsening: bool = False,
        require_complete_metadata: bool = False,
        exclude_raw_snapshot_indices=(),
        show_progress: bool = False,
        geometry_input_mode: str = "none",
    ):
        self._dataset = GalDataSet().load_data(
            path,
            allow_mixed_coarsening=allow_mixed_coarsening,
            require_complete_metadata=require_complete_metadata,
            show_progress=show_progress,
            geometry_input_mode=geometry_input_mode,
        )
        target_quantity = self._dataset.target_metadata.get("quantity")
        if require_target_quantity and target_quantity != require_target_quantity:
            raise ValueError(
                f"Processed data target is {target_quantity!r}, expected {require_target_quantity!r}. "
                "Regenerate fine/coarse HDF5 files from the raw hdfra snapshots before preparing caches."
            )
        self._dataset.exclude_raw_snapshot_indices(exclude_raw_snapshot_indices)
        self._dataset.select_input_channels(include=include_channels, exclude=exclude_channels)
        self._dataset.drop_invalid(
            require_positive_targets=not include_zero_targets,
            require_positive_ks=not include_zero_targets,
        )
        return self

    def split( # 用statify把数据划分为训练集、验证集和测试集
        self,
        train_size: float,
        validation_size: float,
        stratify: str | np.ndarray | None = None,
        random_state: int | None = None,
        group_by_parent_uid: bool = True,
    ):
        train_set, val_set, test_set = self._dataset.split(
            train_size=train_size,
            validation_size=validation_size,
            stratify=stratify,
            random_state=random_state,
            group_by_parent_uid=group_by_parent_uid,
        )
        return CachePt(train_set), CachePt(val_set), CachePt(test_set)

    def process(self, threshold: float = None, mirror: bool = False, balance: str = None):
        data = self._dataset
        if threshold is not None:
            data.filter(threshold) # 过滤掉恒星形成率小于threshold的样本
        if mirror:
            data.mirror_data() # mirror，镜像扩增数据
        if balance is not None: # balance，过采样或欠采样平衡数据
            if balance not in ["oversample", "undersample"]:
                raise ValueError("Balance method must be 'oversample' or 'undersample'")
            data.balance_groups(method=balance)
        return CachePt(data)

    def to_pt(
        self,
        save_path: str,
        mean: np.ndarray | None = None, # 均值
        std: np.ndarray | None = None, # 方差，用于归一化
        *,
        target_transform: str = "log10_1p",
        target_scale: float = 1.0,
        target_floor: float = 1e-9,
        objective_mode: str = "direct_sfr",
        closure_transform: str = "signed_asinh",
        closure_scale: float = 1.0,
        log_transform_mask=None,
    ):
        data = self._dataset
        geometry_contract = getattr(
            data, "geometry_input_contract", geometry_input_contract("none")
        )
        geometry_active = geometry_contract["mode"] != "none"
        if geometry_active and os.path.exists(save_path):
            raise FileExistsError(f"拒绝覆盖已有几何消融 cache：{save_path}")
        if mean is not None and std is not None: # 验证集&测试集
            x, resolved_log_mask = data.standardize(
                mean,
                std,
                log_transform_mask=log_transform_mask,
                return_log_transform_mask=True,
            )
        else: # 训练集
            x, mean, std, resolved_log_mask = data.standardize(
                log_transform_mask=log_transform_mask,
                return_log_transform_mask=True,
            )
        self._last_log_transform_mask = np.asarray(
            resolved_log_mask, dtype=bool
        ).reshape(-1)
        validate_geometry_contract(
            geometry_contract, getattr(data, "input_channels", ()),
            self._last_log_transform_mask,
        )
        target_sfr_physical = np.asarray(data.sfr_target, dtype=np.float64).reshape(-1)
        y = transform_sfr(
            target_sfr_physical,
            transform=target_transform,
            scale=target_scale,
            floor=target_floor,
        )
        # KS 经验基线继续保存用于评估，但不再充当模型条件。
        sfr_ks_transformed = transform_sfr(
            data.sfr_ks,
            transform=target_transform,
            scale=target_scale,
            floor=target_floor,
        )
        conditioning_scalar, conditioning_contract = _resolve_conditioning_contract(data)
        is_legacy_rm_contract = conditioning_contract["name"] == "log_mass_filter"
        if geometry_active and is_legacy_rm_contract:
            raise ValueError("当前几何输入消融只支持固定 R_h，不复用 R_m FiLM 条件。")
        baseline_physical = _optional_sample_array(
            data,
            ("sfr_macer_baseline", "sfr_baseline", "macer_sfr_baseline", "baseline_sfr"),
        )
        closure_physical = _optional_sample_array(
            data,
            ("sfr_closure_true", "closure_true", "true_closure", "closure_sfr"),
        )
        # 旧 processed HDF5 会由 GalDataSet 以全 NaN 占位；它表示字段缺失，不是
        # 一个可训练的物理 baseline。部分缺失则属于不允许静默混合的数据错误。
        if baseline_physical is not None and np.all(np.isnan(baseline_physical)):
            baseline_physical = None
        elif baseline_physical is not None and not np.all(np.isfinite(baseline_physical)):
            raise ValueError("MACER-LR baseline is only partially available or non-finite.")
        if closure_physical is not None and np.all(np.isnan(closure_physical)):
            closure_physical = None
        elif closure_physical is not None and not np.all(np.isfinite(closure_physical)):
            raise ValueError("True closure is only partially available or non-finite.")
        if objective_mode not in {"direct_sfr", "signed_closure"}:
            raise ValueError(
                "objective_mode must be 'direct_sfr' or 'signed_closure', "
                f"got {objective_mode!r}."
            )
        if objective_mode == "signed_closure" and baseline_physical is None:
            raise ValueError(
                "signed_closure cache requires the MACER-LR physical baseline B_alpha. "
                "Regenerate processed HDF5 files with sfr_baseline/macer_sfr_baseline."
            )

        tensors = {
            "x": x,
            "coord": data.coordinate,
            "target_sfr": y,
            "target_sfr_physical": target_sfr_physical.astype(np.float64),
            "sfr_ks_baseline": sfr_ks_transformed,
            "h_eff": data.h_eff,
            "n_eff": data.n_eff,
            "groups": data.groups,
            "time": data.time,
        }
        if is_legacy_rm_contract:
            # 保持既有 R_m schema v3/v4 的字段和值完全不变。
            tensors["log_mass_filter"] = conditioning_scalar
        else:
            tensors["conditioning_scalar"] = conditioning_scalar
        cache_version = 3
        if baseline_physical is not None:
            if baseline_physical.shape != target_sfr_physical.shape:
                raise ValueError("MACER-LR baseline length does not match target_sfr.")
            if not np.all(np.isfinite(baseline_physical)) or np.any(baseline_physical < 0):
                raise ValueError("MACER-LR baseline must contain finite non-negative SFR values.")
            computed_closure = target_sfr_physical - baseline_physical
            if closure_physical is not None and not np.allclose(
                closure_physical,
                computed_closure,
                rtol=1e-6,
                atol=1e-8,
            ):
                max_error = float(np.max(np.abs(closure_physical - computed_closure)))
                raise ValueError(
                    "Stored closure is inconsistent with Y_alpha - B_alpha; "
                    f"maximum absolute error is {max_error:.6e}."
                )
            closure_physical = computed_closure
            tensors.update(
                {
                    "macer_sfr_baseline": baseline_physical.astype(np.float64),
                    "closure_true_physical": closure_physical.astype(np.float64),
                    "closure_true": transform_signed_closure(
                        closure_physical,
                        transform=closure_transform,
                        scale=closure_scale,
                    ),
                }
            )
            cache_version = 4
        if not is_legacy_rm_contract:
            # v3/v4 的 conditioning 语义已经固定为 log_mass_filter；R_h 使用
            # v5，避免在不提升版本的情况下改变历史 schema 含义。
            cache_version = 5
        if geometry_active:
            cache_version = 6
        has_closure_payload = baseline_physical is not None
        self._last_cache_schema_version = cache_version
        self._last_conditioning_contract = dict(conditioning_contract)
        self._last_geometry_input_contract = geometry_contract
        resolution_features = (
            ["log_mass_filter"] if is_legacy_rm_contract else []
        )
        torch.save(
            {
                "schema": {"name": "macnet_sfr_cache", "version": cache_version},
                "tensors": tensors,
                "sample_identity": {
                    "sample_uid": data.sample_uid.tolist(),
                    "parent_hr_uid": data.parent_hr_uid.tolist(),
                },
                "preprocessing_contract": {
                    "input_transform": (
                        "physical_log1p_and_fixed_reference_log10_geometry_then_standardize"
                        if geometry_active else "per_channel_log1p_then_standardize"
                    ),
                    "log_transform_mask": self._last_log_transform_mask.tolist(),
                    "log_transform_mask_source": "training_split",
                    "normalization_source": "training_split",
                },
                "model_input_contract": {
                    "physical_fields": "x",
                    "conditioning": conditioning_contract,
                    "resolution_features": resolution_features,
                    "reserved_resolution_features": ["h_eff", "n_eff"],
                    "baseline_is_model_input": False,
                    "baseline_is_output_composition": has_closure_payload,
                    "geometry_inputs": geometry_contract,
                    "input_channels": list(getattr(data, "input_channels", ())),
                },
                "learning_contract": {
                    "available_objectives": (
                        ["direct_sfr", "signed_closure"]
                        if has_closure_payload
                        else ["direct_sfr"]
                    ),
                    "prepared_objective": objective_mode,
                    "final_prediction": (
                        "macer_sfr_baseline + closure_hat_physical"
                        if has_closure_payload
                        else "direct_sfr_prediction"
                    ),
                    "closure_transform": closure_transform,
                    "closure_scale": float(closure_scale),
                    "conditioning": conditioning_contract,
                },
            },
            save_path,
        )
        return mean, std


def _save_norm_stats(
    save_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    input_channels=(),
    excluded_channels=(),
    target_metadata=None,
    target_transform="log10_1p",
    target_scale=1.0,
    target_floor=1e-9,
    objective_mode="direct_sfr",
    closure_transform="signed_asinh",
    closure_scale=1.0,
    cache_schema_version=None,
    conditioning_contract=None,
    includes_zero_targets=True,
    oversample_groups=True,
    split_metadata=None,
    closure_scale_selection=None,
    log_transform_mask=None,
    geometry_inputs=None,
): # 接收CachePt处理训练集时算出来的mean和std，打包保存成npz文件
    norm_path = f"{save_path}norm.npz"
    channel_names = np.asarray(tuple(input_channels), dtype=str)
    excluded_names = np.asarray(tuple(excluded_channels), dtype=str)
    target_metadata = dict(target_metadata or {})
    resolved_cache_schema_version = int(
        cache_schema_version
        if cache_schema_version is not None
        else (4 if objective_mode == "signed_closure" else 3)
    )
    if conditioning_contract is None:
        conditioning_contract = {
            "name": "log_mass_filter",
            "active": True,
            "transform": "log10",
            "units": "MACER_mass_code",
            "source": "legacy_cache",
            "dimension": 1,
            "operator_ids": [],
        }
    else:
        conditioning_contract = dict(conditioning_contract)
    conditioning_name = str(conditioning_contract.get("name", "unknown"))
    conditioning_active = bool(conditioning_contract.get("active", False))
    conditioning_transform = str(
        conditioning_contract.get("transform", "identity")
    )
    resolution_features = (
        [conditioning_name]
        if conditioning_active and conditioning_name != "none"
        else []
    )
    metadata = {
        "format_version": 3 if resolved_cache_schema_version >= 6 else 2,
        "input_channels": channel_names.tolist(),
        "excluded_channels": excluded_names.tolist(),
        "n_input_channels": int(len(channel_names)),
        "target_quantity": target_metadata.get("quantity", "instantaneous_sfr"),
        "target_name": target_metadata.get("name", "sfr"),
        "target_dataset": target_metadata.get("dataset", "Data-Set-11"),
        "target_reduction": target_metadata.get("reduction", "sum"),
        "target_units": target_metadata.get("units", "Msun/yr"),
        "target_transform": target_transform,
        "target_scale": float(target_scale),
        "target_floor": float(target_floor),
        "includes_zero_targets": bool(includes_zero_targets),
        "oversample_groups": bool(oversample_groups),
        "group_balance": "oversample" if oversample_groups else "none",
        "cache_schema_version": resolved_cache_schema_version,
        "objective_mode": objective_mode,
        "prepared_objective": objective_mode,
        "available_objectives": (
            ["direct_sfr", "signed_closure"]
            if resolved_cache_schema_version >= 4
            else ["direct_sfr"]
        ),
        "closure_transform": closure_transform,
        "closure_scale": float(closure_scale),
        "closure_scale_selection": dict(closure_scale_selection or {
            "mode": "explicit",
            "source_split": None,
            "source_tensor": None,
            "method": "user_supplied",
            "resolved_value": float(closure_scale),
        }),
        "closure_definition": "target_sfr_physical-macer_sfr_baseline",
        "final_prediction": "macer_sfr_baseline+closure_hat_physical",
        "conditioning_variable": conditioning_name,
        "conditioning_active": conditioning_active,
        "conditioning_transform": conditioning_transform,
        "conditioning_contract": conditioning_contract,
        "baseline_variable": "sfr_ks",
        "baseline_transform": target_transform,
        "resolution_features": resolution_features,
        "reserved_resolution_features": ["h_eff", "n_eff"],
        "log_transform_mask": (
            np.asarray(log_transform_mask, dtype=bool).reshape(-1).tolist()
            if log_transform_mask is not None
            else []
        ),
        "log_transform_mask_source": "training_split",
        "geometry_input_contract": (
            geometry_inputs if geometry_inputs is not None
            else geometry_input_contract("none")
        ),
    }
    if split_metadata:
        overlap = set(split_metadata.get("train_groups", ())).intersection(
            split_metadata.get("held_out_groups", ())
        )
        if overlap:
            raise ValueError(
                "Refusing to save invalid split metadata: train_groups and "
                f"held_out_groups overlap: {sorted(overlap)}"
            )
        metadata.update(split_metadata)
    np.savez(
        norm_path,
        mean=mean,
        std=std,
        input_channels=channel_names,
        excluded_channels=excluded_names,
        target_quantity=np.asarray(metadata["target_quantity"]),
        target_name=np.asarray(metadata["target_name"]),
        target_dataset=np.asarray(metadata["target_dataset"]),
        target_reduction=np.asarray(metadata["target_reduction"]),
        target_units=np.asarray(metadata["target_units"]),
        target_transform=np.asarray(metadata["target_transform"]),
        target_scale=np.asarray(metadata["target_scale"]),
        target_floor=np.asarray(metadata["target_floor"]),
        includes_zero_targets=np.asarray(metadata["includes_zero_targets"]),
        oversample_groups=np.asarray(metadata["oversample_groups"]),
        group_balance=np.asarray(metadata["group_balance"]),
        conditioning_variable=np.asarray(metadata["conditioning_variable"]),
        conditioning_active=np.asarray(metadata["conditioning_active"]),
        conditioning_transform=np.asarray(metadata["conditioning_transform"]),
        objective_mode=np.asarray(metadata["objective_mode"]),
        closure_transform=np.asarray(metadata["closure_transform"]),
        closure_scale=np.asarray(metadata["closure_scale"]),
        closure_scale_mode=np.asarray(metadata["closure_scale_selection"]["mode"]),
        closure_scale_method=np.asarray(metadata["closure_scale_selection"]["method"]),
        cache_schema_version=np.asarray(metadata["cache_schema_version"]),
        log_transform_mask=np.asarray(metadata["log_transform_mask"], dtype=bool),
        geometry_input_contract_json=np.asarray(
            json.dumps(metadata["geometry_input_contract"], sort_keys=True)
        ),
    )
    metadata_path = f"{save_path}metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved normalization stats: {norm_path}")
    print(f"[INFO] Saved channel metadata: {metadata_path}")


def _parse_channel_list(value: str):
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _parse_bool(value):
    """Parse explicit command-line booleans such as --oversample=false."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value (true/false), got {value!r}."
    )


def _parse_closure_scale(value):
    """解析正浮点数或 ``auto`` 形式的闭合项尺度。"""

    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        scale = float(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--closure-scale 需要正浮点数或 'auto'，实际得到 {value!r}。"
        ) from exc
    if not np.isfinite(scale) or scale <= 0:
        raise argparse.ArgumentTypeError(
            f"--closure-scale 必须为正有限数，实际得到 {value!r}。"
        )
    return scale


def _estimate_closure_scale_from_pt(temporary_train_path):
    """只从临时训练 cache 的物理闭合项估计稳健尺度。

    主定义是 ``median(abs(C_true))``。如果训练集中至少一半的闭合项恰为
    零而导致该统计量退化，则回退到非零绝对值的中位数。这里不读取验证集或
    测试集，从数据流上避免尺度估计造成信息泄漏。
    """

    payload = torch.load(temporary_train_path, map_location="cpu", weights_only=False)
    try:
        closure = payload["tensors"]["closure_true_physical"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "closure-scale auto 需要临时 train.pt 包含 "
            "tensors/closure_true_physical；请使用带 MACER-LR baseline 的数据。"
        ) from exc

    values = np.asarray(closure, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("无法从空训练集估计 closure scale。")
    if not np.all(np.isfinite(values)):
        bad_count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            "训练集 closure_true_physical 含有非有限值，"
            f"无法自动估计尺度（{bad_count}/{values.size}）。"
        )

    absolute = np.abs(values)
    scale = float(np.median(absolute))
    method = "median_abs"
    positive = absolute[absolute > 0.0]
    if not np.isfinite(scale) or scale <= 0.0:
        if positive.size == 0:
            raise ValueError(
                "训练集 closure_true_physical 全部为零；自动尺度没有定义，"
                "请检查 baseline/真值，或显式指定 --closure-scale。"
            )
        scale = float(np.median(positive))
        method = "median_nonzero_abs_fallback"

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"自动估计得到非法 closure scale：{scale!r}。")

    statistics = {
        "mode": "auto",
        "source_split": "train",
        "source_tensor": "closure_true_physical",
        "method": method,
        "resolved_value": scale,
        "sample_count": int(values.size),
        "zero_count": int(np.count_nonzero(absolute == 0.0)),
        "negative_count": int(np.count_nonzero(values < 0.0)),
        "median_abs": float(np.median(absolute)),
        "p90_abs": float(np.percentile(absolute, 90.0)),
        "p99_abs": float(np.percentile(absolute, 99.0)),
        "max_abs": float(np.max(absolute)),
    }
    return scale, statistics


def _write_train_cache_with_resolved_closure_scale(
    train_cache,
    save_path,
    transform_options,
):
    """写训练 cache，并在请求 ``auto`` 时执行临时 cache 两阶段流程。"""

    resolved_options = dict(transform_options)
    requested_scale = resolved_options.get("closure_scale", 1.0)
    final_train_path = f"{save_path}train.pt"

    if requested_scale != "auto":
        resolved_options["closure_scale"] = float(requested_scale)
        mean, std = train_cache.to_pt(final_train_path, **resolved_options)
        resolved_options["log_transform_mask"] = train_cache.last_log_transform_mask
        selection = {
            "mode": "explicit",
            "source_split": None,
            "source_tensor": None,
            "method": "user_supplied",
            "resolved_value": float(requested_scale),
        }
        return mean, std, resolved_options, selection

    temporary_train_path = f"{save_path}train.auto_scale_tmp.pt"
    provisional_options = dict(resolved_options)
    # 临时 cache 中的物理闭合项与变换尺度无关；1.0 仅用于满足现有 schema。
    provisional_options["closure_scale"] = 1.0
    if os.path.exists(temporary_train_path):
        os.remove(temporary_train_path)

    print(f"[INFO] 写入临时训练 cache 以估计 closure scale: {temporary_train_path}")
    try:
        train_cache.to_pt(temporary_train_path, **provisional_options)
        resolved_options["log_transform_mask"] = train_cache.last_log_transform_mask
        scale, selection = _estimate_closure_scale_from_pt(temporary_train_path)
    finally:
        if os.path.exists(temporary_train_path):
            os.remove(temporary_train_path)

    resolved_options["closure_scale"] = scale
    print(
        "[INFO] closure-scale auto：仅由训练集 "
        f"{selection['method']} 得到 {scale:.10g}；正在重新生成正式 cache。"
    )
    mean, std = train_cache.to_pt(final_train_path, **resolved_options)
    return mean, std, resolved_options, selection


def _parse_experiment_tokens(values):
    """Normalize CLI values such as ``exp1 exp2`` or ``exp1,exp2``."""
    normalized = []
    for value in values or ():
        for token in str(value).split(","):
            token = token.strip().lower()
            if not token:
                continue
            match = re.fullmatch(r"(?:exp)?(1|2|3|4|8)", token)
            if not match:
                raise argparse.ArgumentTypeError(
                    f"Unsupported experiment {token!r}; choose exp1, exp2, exp3, exp4, or exp8."
                )
            name = f"exp{match.group(1)}"
            if name not in normalized:
                normalized.append(name)
    return tuple(normalized)


def _slugify_group_name(group_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(group_name)).strip("_")
    if not slug:
        raise ValueError(f"Cannot construct a cache name from group {group_name!r}.")
    return slug


def _valid_group_paths(folder_paths):
    """Return configured groups that currently contain at least one HDF5 cache."""
    valid = []
    skipped = []
    for entry in folder_paths:
        path, group_name = entry
        has_h5 = os.path.isdir(path) and any(
            name.lower().endswith(".h5") for name in os.listdir(path)
        )
        if has_h5:
            valid.append([os.path.abspath(path), str(group_name)])
        else:
            skipped.append([os.path.abspath(path), str(group_name)])
    return valid, skipped


def _normalize_held_out_names(requested_names, valid_group_names):
    """Resolve nicknames (``fiducial_4``) or full group names unambiguously."""
    valid_group_names = tuple(valid_group_names)
    resolved = []
    for requested in requested_names or ():
        requested = str(requested).strip()
        if not requested:
            continue
        candidates = [requested]
        if not requested.startswith("disk_galaxy_"):
            candidates.append(f"disk_galaxy_{requested}")
        match = next((name for name in candidates if name in valid_group_names), None)
        if match is None:
            raise ValueError(
                f"Requested held-out galaxy {requested!r} is not available. "
                f"Valid groups: {list(valid_group_names)}"
            )
        if match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _cache_groups(cache: CachePt) -> tuple[str, ...]:
    return tuple(sorted({str(group) for group in cache.dataset.groups.tolist()}))


def _assert_logo_partition(train_cache, val_cache, test_cache, out_cache, *, train_groups, held_out_group):
    """Adversarial split check using the actual samples, not only requested paths."""
    expected_train = set(train_groups)
    expected_out = {held_out_group}
    actual = {
        "train": set(_cache_groups(train_cache)),
        "val": set(_cache_groups(val_cache)),
        "test": set(_cache_groups(test_cache)),
        "out": set(_cache_groups(out_cache)),
    }
    for split in ("train", "val", "test"):
        if actual[split] != expected_train:
            raise RuntimeError(
                f"Invalid leave-one-galaxy-out fold: {split} groups {sorted(actual[split])} "
                f"do not equal expected in-domain groups {sorted(expected_train)}."
            )
    if actual["out"] != expected_out:
        raise RuntimeError(
            f"Invalid leave-one-galaxy-out fold: out groups {sorted(actual['out'])} "
            f"do not equal held-out group {held_out_group!r}."
        )
    overlap = expected_train.intersection(actual["out"])
    if overlap:
        raise RuntimeError(
            f"Data leakage detected: held-out groups overlap training groups: {sorted(overlap)}"
        )
    return actual


def _cache_prefix(experiment_number: int, cache_tag: str) -> str:
    tag = cache_tag.strip().strip("_")
    return f"Exp{experiment_number}_{tag}_" if tag else f"Exp{experiment_number}_"


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare SFR training caches.")
    parser.add_argument(
        "--exclude-channels",
        default="initial_stellar_density",
        help="Comma-separated input channels to exclude, for example: initial_stellar_density",
    )
    parser.add_argument(
        "--cache-tag",
        default="InstantSFR_NoInitialStellarDensity",
        help="Tag inserted in cache prefixes. Use InstantSFR with an empty exclusion list for the retained-channel control.",
    )
    parser.add_argument(
        "--resolution-dirname",
        default="coarse",
        help=(
            "每个 galaxy 数据目录下要读取的 processed HDF5 子目录。"
            "例如固定 8x8 R_h 可使用 coarse_rh_8x8_sfr_v6，默认保持 coarse。"
        ),
    )
    parser.add_argument(
        "--geometry-input-mode",
        choices=SUPPORTED_GEOMETRY_INPUT_MODES,
        default="none",
        help=("none 保持原输入；absolute_radius 为 Exp3.1；"
              "absolute_radius_widths 为 Exp3.2。几何 log10 通道只做训练集标准化。"),
    )
    parser.add_argument(
        "--target-transform",
        choices=SUPPORTED_TARGET_TRANSFORMS,
        default="log10_1p",
        help="Invertible transform applied identically to true SFR and the KS baseline.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=1.0,
        help="log10_1p 的 SFR 尺度；单位必须与 processed HDF5 target_units 一致。",
    )
    parser.add_argument(
        "--target-floor",
        type=float,
        default=1e-9,
        help="Positive floor used only by log10_floor.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("direct_sfr", "signed_closure"),
        default="direct_sfr",
        help=(
            "direct_sfr 保留历史直接预测；signed_closure 需要 processed HDF5 "
            "提供 MACER-LR baseline，并训练 C_alpha=Y_alpha-B_alpha。"
        ),
    )
    parser.add_argument(
        "--closure-transform",
        choices=SUPPORTED_CLOSURE_TRANSFORMS,
        default="signed_asinh",
        help="可正可负闭合项使用的可逆变换。",
    )
    parser.add_argument(
        "--closure-scale",
        type=_parse_closure_scale,
        default=1.0,
        help=(
            "signed_asinh 的物理 SFR 尺度，单位与 target_units 相同。"
            "可传正浮点数，或传 auto 以只从临时 train.pt 的 "
            "closure_true_physical 自动估计。"
        ),
    )
    parser.add_argument(
        "--drop-zero-targets",
        action="store_true",
        help="Drop zero-SFR samples. By default zero-SFR snapshots are retained.",
    )
    parser.add_argument(
        "--exclude-raw-snapshot-index",
        action="append",
        type=int,
        default=[],
        metavar="INDEX",
        help=(
            "精确排除指定的原始 snapshot index，可重复。盘星系初始化输出通常使用 "
            "--exclude-raw-snapshot-index 0；这不会误删晚期物理零 SFR。"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for dataset splitting.")
    parser.add_argument(
        "--only-exp1",
        action="store_true",
        help="Deprecated compatibility alias for --experiments exp1.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        metavar="EXP",
        help=(
            "Experiments to prepare, for example --experiments exp3 or "
            "--experiments exp1 exp2. Defaults to exp1-exp4 plus exp8."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=None,
        metavar="EXP",
        help="Repeatable singular alias, for example --experiment exp3 --experiment exp8.",
    )
    parser.add_argument(
        "--held-out-galaxy",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Limit Exp2 folds to one or more galaxies. Accepts a nickname such as "
            "fiducial_4 or a full name such as disk_galaxy_fiducial_4. "
            "Omit to generate one fold for every available disk galaxy."
        ),
    )
    parser.add_argument(
        "--oversample",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help=(
            "Balance training groups by oversampling (default: true). "
            "Accepts --oversample=true or --oversample=false."
        ),
    )
    parser.add_argument(
        "--progress",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        metavar="BOOL",
        help=(
            "显示 processed HDF5 加载进度（默认 true）。"
            "可用 --progress=false 关闭，便于非交互日志环境。"
        ),
    )
    args = parser.parse_args()
    args.exclude_channels = _parse_channel_list(args.exclude_channels)
    if args.geometry_input_mode != "none" and args.cache_tag == "InstantSFR_NoInitialStellarDensity":
        parser.error("几何消融必须使用独立 --cache-tag，不能覆盖原 Exp3 cache。")
    if args.target_scale <= 0:
        parser.error("--target-scale must be positive.")
    if args.target_floor <= 0:
        parser.error("--target-floor must be positive.")
    if args.exclude_channels and not args.cache_tag:
        parser.error("--cache-tag is required when --exclude-channels is used, to avoid overwriting old caches.")
    try:
        requested = _parse_experiment_tokens((args.experiments or []) + (args.experiment or []))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.only_exp1:
        if requested and requested != ("exp1",):
            parser.error("--only-exp1 cannot be combined with experiments other than exp1.")
        requested = ("exp1",)
    args.experiments = requested or ("exp1", "exp2", "exp3", "exp4", "exp8")
    return args


def get_folder_paths(
    configs_path,
    gal_type,
    resolution,
    *,
    include_roles=("train",),
): # 读取config文件，返回指定类型和分辨率的文件夹绝对路径列表。这样代码路径就直接从config文件中读取，不需要硬编码在代码里了。
    configs = load_config(configs_path)
    data_dir = configs["BaseConfig"]["data_dir"]
    raw_data_config = configs["RawDataConfig"]

    folder_abs_paths = []
    allowed_roles = None if include_roles is None else set(include_roles)
    for gal_nickname, gal_config in raw_data_config[gal_type].items():
        role = str(gal_config.get("role", "train"))
        if allowed_roles is not None and role not in allowed_roles:
            continue
        gal_name = f"{gal_type}_{gal_nickname}"
        folder_abs_path = os.path.join(data_dir, gal_name, resolution)
        folder_abs_paths.append([folder_abs_path, gal_name])
    return folder_abs_paths


def _prepare_random_snapshot_experiment(
    experiment_number,
    folder_paths,
    *,
    args,
    load_options,
    transform_options,
    train_balance,
    split_label="random_snapshot_within_group",
):
    """Prepare the legacy in-domain 60/20/20 reference split."""
    valid_paths, skipped_paths = _valid_group_paths(folder_paths)
    if not valid_paths:
        raise ValueError(f"Experiment {experiment_number} has no valid HDF5 groups.")
    if skipped_paths:
        print(f"  [INFO] Ignoring {len(skipped_paths)} configured groups without processed HDF5 data.")

    cache = CachePt().load(valid_paths, **load_options)
    train_cache, val_cache, test_cache = cache.split(
        train_size=0.6,
        validation_size=0.2,
        stratify="groups",
        random_state=args.seed,
    )
    np.random.seed(args.seed + int(experiment_number) * 1000)
    train_cache = train_cache.process(mirror=False, balance=train_balance)
    val_cache = val_cache.process(mirror=False)
    test_cache = test_cache.process(mirror=False)

    group_names = [entry[1] for entry in valid_paths]
    split_groups = {
        "train": list(_cache_groups(train_cache)),
        "val": list(_cache_groups(val_cache)),
        "test": list(_cache_groups(test_cache)),
    }
    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Testing set size:    {len(test_cache)}")

    save_path = os.path.join(".cache", _cache_prefix(experiment_number, args.cache_tag))
    mean, std, resolved_transform_options, scale_selection = (
        _write_train_cache_with_resolved_closure_scale(
            train_cache,
            save_path,
            transform_options,
        )
    )
    _save_norm_stats(
        save_path,
        mean,
        std,
        input_channels=train_cache.input_channels,
        excluded_channels=args.exclude_channels,
        target_metadata=train_cache.dataset.target_metadata,
        includes_zero_targets=not args.drop_zero_targets,
        oversample_groups=args.oversample,
        cache_schema_version=train_cache.last_cache_schema_version,
        conditioning_contract=train_cache.last_conditioning_contract,
        geometry_inputs=train_cache.last_geometry_input_contract,
        closure_scale_selection=scale_selection,
        split_metadata={
            "experiment": f"exp{experiment_number}",
            "split_strategy": split_label,
            "split_seed": int(args.seed),
            "oversample_seed": int(args.seed + int(experiment_number) * 1000),
            "available_groups": group_names,
            "train_groups": group_names,
            "held_out_groups": [],
            "split_groups": split_groups,
            "group_paths": {entry[1]: entry[0] for entry in valid_paths},
            "sample_counts": {
                "train": len(train_cache),
                "val": len(val_cache),
                "test": len(test_cache),
                "out": 0,
            },
            "cache_prefix": os.path.basename(save_path),
        },
        **resolved_transform_options,
    )
    val_cache.to_pt(save_path + "val.pt", mean, std, **resolved_transform_options)
    test_cache.to_pt(save_path + "test.pt", mean, std, **resolved_transform_options)
    return save_path


def _require_group_entries(folder_paths, required_groups, *, purpose):
    """按显式 allowlist 解析 processed 目录；缺少任何一组时立即停止。"""

    by_name = {str(entry[1]): entry for entry in folder_paths}
    missing_config = [name for name in required_groups if name not in by_name]
    if missing_config:
        raise ValueError(
            f"{purpose} 缺少数据配置：{missing_config}。"
        )
    selected = [by_name[name] for name in required_groups]
    valid, missing_data = _valid_group_paths(selected)
    if missing_data:
        details = {entry[1]: entry[0] for entry in missing_data}
        raise ValueError(
            f"{purpose} 缺少 processed HDF5；请先生成对应 resolution：{details}"
        )
    return valid


def _assert_disjoint_cache_identities(caches):
    """检查各 split 的样本和母 snapshot 身份均无交叉。"""

    for identity_name in ("sample_uid", "parent_hr_uid"):
        values = {}
        for split_name, cache in caches.items():
            array = np.asarray(
                getattr(cache.dataset, identity_name), dtype=str
            ).reshape(-1)
            if np.any(array == ""):
                raise RuntimeError(
                    f"Exp3 {split_name} 缺少 {identity_name}，无法验证数据隔离。"
                )
            values[split_name] = set(array.tolist())
        names = tuple(values)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1:]:
                overlap = values[left].intersection(values[right])
                if overlap:
                    preview = sorted(overlap)[:3]
                    raise RuntimeError(
                        f"Exp3 {left}/{right} 的 {identity_name} 存在交叉：{preview}"
                    )


def _prepare_exp3_torus_ood(
    train_folder_paths,
    test_only_folder_paths,
    *,
    args,
    load_options,
    transform_options,
):
    """Zou 五组随机 75/25 train/val，torus 固定为唯一测试域。"""

    train_entries = _require_group_entries(
        train_folder_paths,
        _ZOU_FIVE_GROUPS,
        purpose="Exp3 Zou train/validation",
    )
    test_entries = _require_group_entries(
        test_only_folder_paths,
        (_TORUS_TEST_GROUP,),
        purpose="Exp3 torus test",
    )

    cache_in = CachePt().load(train_entries, **load_options)
    train_cache, val_cache, unused_test_cache = cache_in.split(
        train_size=0.75,
        validation_size=0.25,
        stratify="groups",
        random_state=args.seed,
    )
    if len(unused_test_cache) != 0:
        raise RuntimeError(
            "Exp3 的 75/25 二路划分意外产生了未分配的 Zou test 样本。"
        )
    # Exp3 不在 CachePT 中复制少样本运行。正式训练由
    # run_time_balanced sampler 实现运行等权和运行内物理时间加权。
    train_cache = train_cache.process(mirror=False, balance=None)
    val_cache = val_cache.process(mirror=False)
    test_cache = CachePt().load(test_entries, **load_options).process(mirror=False)

    expected_train_groups = set(_ZOU_FIVE_GROUPS)
    actual_groups = {
        "train": set(_cache_groups(train_cache)),
        "val": set(_cache_groups(val_cache)),
        "test": set(_cache_groups(test_cache)),
    }
    for split_name in ("train", "val"):
        if actual_groups[split_name] != expected_train_groups:
            raise RuntimeError(
                f"Exp3 {split_name} groups={sorted(actual_groups[split_name])}，"
                f"预期为 {sorted(expected_train_groups)}。"
            )
    if actual_groups["test"] != {_TORUS_TEST_GROUP}:
        raise RuntimeError(
            f"Exp3 test groups={sorted(actual_groups['test'])}，"
            f"预期仅为 {_TORUS_TEST_GROUP!r}。"
        )
    if train_cache.input_channels != test_cache.input_channels:
        raise RuntimeError(
            "Exp3 Zou 与 torus 的输入通道不一致："
            f"{train_cache.input_channels} != {test_cache.input_channels}"
        )
    if train_cache.dataset.target_metadata != test_cache.dataset.target_metadata:
        raise RuntimeError(
            "Exp3 Zou 与 torus 的 SFR target 语义或单位不一致："
            f"{train_cache.dataset.target_metadata} != "
            f"{test_cache.dataset.target_metadata}"
        )
    _assert_disjoint_cache_identities(
        {"train": train_cache, "val": val_cache, "test": test_cache}
    )

    print(f"  Training set size:   {len(train_cache)}")
    print(f"  Validation set size: {len(val_cache)}")
    print(f"  Torus test set size: {len(test_cache)}")
    print("  [PASS] Exp3 train/val/test groups and UID isolation verified.")

    save_path = os.path.join(".cache", _cache_prefix(3, args.cache_tag))
    if train_cache.dataset.geometry_input_contract["mode"] != "none":
        for suffix in ("train.pt", "val.pt", "test.pt", "norm.npz", "metadata.json", "train.auto_scale_tmp.pt"):
            if os.path.exists(save_path + suffix):
                raise FileExistsError(f"拒绝覆盖已有消融产物：{save_path + suffix}")
    mean, std, resolved_transform_options, scale_selection = (
        _write_train_cache_with_resolved_closure_scale(
            train_cache,
            save_path,
            transform_options,
        )
    )
    group_paths = {
        entry[1]: entry[0] for entry in train_entries + test_entries
    }
    split_metadata = {
        "experiment": "exp3",
        "split_strategy": "random_snapshot_75_25_torus_test",
        "split_seed": int(args.seed),
        "available_groups": list(_ZOU_FIVE_GROUPS) + [_TORUS_TEST_GROUP],
        "train_groups": list(_ZOU_FIVE_GROUPS),
        "validation_groups": list(_ZOU_FIVE_GROUPS),
        "test_groups": [_TORUS_TEST_GROUP],
        "held_out_groups": [_TORUS_TEST_GROUP],
        "test_only_groups": [_TORUS_TEST_GROUP],
        "split_groups": {
            key: sorted(value) for key, value in actual_groups.items()
        },
        "group_paths": group_paths,
        "sample_counts": {
            "train": len(train_cache),
            "val": len(val_cache),
            "test": len(test_cache),
            "out": 0,
        },
        "train_fraction": 0.75,
        "validation_fraction": 0.25,
        "cache_group_balance": "none",
        "training_sampling": "run_time_balanced",
        "cache_prefix": os.path.basename(save_path),
    }
    _save_norm_stats(
        save_path,
        mean,
        std,
        input_channels=train_cache.input_channels,
        excluded_channels=args.exclude_channels,
        target_metadata=train_cache.dataset.target_metadata,
        includes_zero_targets=not args.drop_zero_targets,
        oversample_groups=False,
        cache_schema_version=train_cache.last_cache_schema_version,
        conditioning_contract=train_cache.last_conditioning_contract,
        geometry_inputs=train_cache.last_geometry_input_contract,
        closure_scale_selection=scale_selection,
        split_metadata=split_metadata,
        **resolved_transform_options,
    )
    val_cache.to_pt(save_path + "val.pt", mean, std, **resolved_transform_options)
    test_cache.to_pt(save_path + "test.pt", mean, std, **resolved_transform_options)
    return save_path


def _prepare_exp2_logo_folds(
    folder_paths,
    *,
    args,
    load_options,
    transform_options,
    train_balance,
):
    """Generate one cache quartet per held-out disk galaxy."""
    valid_paths, skipped_paths = _valid_group_paths(folder_paths)
    valid_groups = tuple(entry[1] for entry in valid_paths)
    if skipped_paths:
        print("  Ignoring configured groups without processed HDF5 data:")
        for path, group in skipped_paths:
            print(f"    - {group}: {path}")
    if len(valid_paths) < 2:
        raise ValueError(
            "Exp2 leave-one-galaxy-out requires at least two available processed galaxies; "
            f"found {list(valid_groups)}."
        )

    requested = _normalize_held_out_names(args.held_out_galaxy, valid_groups)
    held_out_groups = requested or valid_groups
    base_prefix = _cache_prefix(2, args.cache_tag)
    folds = []

    for held_out_group in held_out_groups:
        held_out_entry = next(entry for entry in valid_paths if entry[1] == held_out_group)
        in_entries = [entry for entry in valid_paths if entry[1] != held_out_group]
        in_groups = tuple(entry[1] for entry in in_entries)
        fold_index = valid_groups.index(held_out_group)
        fold_name = f"LOO_{_slugify_group_name(held_out_group)}"
        save_path = os.path.join(".cache", f"{base_prefix}{fold_name}_")

        print("\n" + "=" * 72)
        print(f"Processing Exp2 fold: {fold_name}")
        print(f"  In-domain groups: {list(in_groups)}")
        print(f"  Held-out group:   {held_out_group}")

        cache_in = CachePt().load(in_entries, **load_options)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        # A fold-specific seed makes manual single-fold preparation identical
        # to the same fold produced as part of an exhaustive run.
        oversample_seed = int(args.seed + fold_index)
        np.random.seed(oversample_seed)
        train_cache = train_cache.process(mirror=False, balance=train_balance)
        val_cache = val_cache.process(mirror=False)
        test_cache = test_cache.process(mirror=False)
        cache_out = CachePt().load([held_out_entry], **load_options).process(mirror=False)

        actual_groups = _assert_logo_partition(
            train_cache,
            val_cache,
            test_cache,
            cache_out,
            train_groups=in_groups,
            held_out_group=held_out_group,
        )
        print("  [PASS] Actual train/val/test groups are disjoint from the held-out group.")
        print(f"  Training set size:                 {len(train_cache)}")
        print(f"  Validation set size:               {len(val_cache)}")
        print(f"  In-distribution test set size:      {len(test_cache)}")
        print(f"  Held-out galaxy test set size:      {len(cache_out)}")

        mean, std, resolved_transform_options, scale_selection = (
            _write_train_cache_with_resolved_closure_scale(
                train_cache,
                save_path,
                transform_options,
            )
        )
        split_metadata = {
            "experiment": "exp2",
            "split_strategy": "leave_one_galaxy_out",
            "split_seed": int(args.seed),
            "fold_name": fold_name,
            "fold_index": fold_index,
            "n_folds": len(valid_groups),
            "oversample_seed": oversample_seed,
            "available_groups": list(valid_groups),
            "train_groups": list(in_groups),
            "in_domain_groups": list(in_groups),
            "in_distribution_test_groups": list(in_groups),
            "held_out_group": held_out_group,
            "held_out_groups": [held_out_group],
            "out_groups": [held_out_group],
            "split_groups": {key: sorted(value) for key, value in actual_groups.items()},
            "group_paths": {entry[1]: entry[0] for entry in valid_paths},
            "raw_in_domain_groups": list(in_groups),
            "raw_held_out_groups": [held_out_group],
            "sample_counts": {
                "train": len(train_cache),
                "val": len(val_cache),
                "test": len(test_cache),
                "out": len(cache_out),
            },
            "cache_prefix": os.path.basename(save_path),
        }
        _save_norm_stats(
            save_path,
            mean,
            std,
            input_channels=train_cache.input_channels,
            excluded_channels=args.exclude_channels,
            target_metadata=train_cache.dataset.target_metadata,
            includes_zero_targets=not args.drop_zero_targets,
            oversample_groups=args.oversample,
            cache_schema_version=train_cache.last_cache_schema_version,
            conditioning_contract=train_cache.last_conditioning_contract,
            geometry_inputs=train_cache.last_geometry_input_contract,
            closure_scale_selection=scale_selection,
            split_metadata=split_metadata,
            **resolved_transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **resolved_transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **resolved_transform_options)
        cache_out.to_pt(save_path + "out.pt", mean, std, **resolved_transform_options)
        folds.append(split_metadata)

    manifest_path = os.path.join(".cache", f"{base_prefix}folds.json")
    manifest = {
        "format_version": 1,
        "experiment": "exp2",
        "split_strategy": "leave_one_galaxy_out",
        "available_groups": list(valid_groups),
        "requested_held_out_groups": list(requested),
        "fold_count": len(folds),
        "folds": [
            {
                "fold_name": fold["fold_name"],
                "cache_prefix": fold["cache_prefix"],
                "train_groups": fold["train_groups"],
                "held_out_groups": fold["held_out_groups"],
            }
            for fold in folds
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"\n[INFO] Saved Exp2 fold manifest: {manifest_path}")
    return manifest_path


# Define global paths for external imports (e.g. by notebooks)
try:
    _resolution = "coarse"
    _configs_dir = ".config"
    Exp_eg_folder_paths = get_folder_paths(_configs_dir, "elliptical_galaxy", _resolution) # 特定分辨率的椭圆星系的路径
    Exp_dg_folder_paths = get_folder_paths(_configs_dir, "disk_galaxy", _resolution) # 特定分辨率的盘状星系的路径
except Exception as e:
    # This might happen if CWD is not project root during import
    Exp_eg_folder_paths = []
    Exp_dg_folder_paths = []


if __name__ == "__main__":
    args = _parse_args()
    excluded_channels = args.exclude_channels
    include_zero_targets = not args.drop_zero_targets
    load_options = {
        "exclude_channels": excluded_channels,
        "include_zero_targets": include_zero_targets,
        "require_target_quantity": "instantaneous_sfr",
        "exclude_raw_snapshot_indices": tuple(args.exclude_raw_snapshot_index),
        "show_progress": args.progress,
        "geometry_input_mode": args.geometry_input_mode,
    }
    transform_options = {
        "target_transform": args.target_transform,
        "target_scale": args.target_scale,
        "target_floor": args.target_floor,
        "objective_mode": args.objective_mode,
        "closure_transform": args.closure_transform,
        "closure_scale": args.closure_scale,
    }
    train_balance = "oversample" if args.oversample else None
    np.random.seed(args.seed)
    os.makedirs(".cache", exist_ok=True)
    resolution = args.resolution_dirname
    configs_dir = ".config"

    # 命令行可选择独立 processed 子目录；模块级默认路径仅保留给旧 notebook。
    exp_eg_folder_paths = get_folder_paths(
        configs_dir, "elliptical_galaxy", resolution
    )
    exp_dg_folder_paths = get_folder_paths(
        configs_dir, "disk_galaxy", resolution, include_roles=("train",)
    )
    exp_dg_test_only_paths = get_folder_paths(
        configs_dir, "disk_galaxy", resolution, include_roles=("test_only",)
    )

    print(f"Requested experiments: {', '.join(args.experiments)}")

    # Exp1 remains the in-domain random-snapshot reference experiment.
    if "exp1" in args.experiments:
        print("\nProcessing Experiment 1 (in-domain random snapshot split)...")
        _prepare_random_snapshot_experiment(
            1,
            exp_dg_folder_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
        )

    # Exp2 is now exhaustive leave-one-galaxy-out over available HDF5 groups.
    if "exp2" in args.experiments:
        print("\nProcessing Experiment 2 (leave-one-galaxy-out)...")
        _prepare_exp2_logo_folds(
            exp_dg_folder_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
        )

    # Exp3: Zou 五组只进入随机 75/25 train/validation，torus 是唯一 test。
    if "exp3" in args.experiments:
        print("\nProcessing Experiment 3 (Zou 75/25 train/val; torus test-only)...")
        _prepare_exp3_torus_ood(
            exp_dg_folder_paths,
            exp_dg_test_only_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
        )

    # 旧 Exp3（删除 low 后随机 60/20/20）迁移为 Exp8。
    if "exp8" in args.experiments:
        print("\nProcessing Experiment 8 (legacy Exp3 semantics)...")
        exp8_paths = [
            entry for entry in exp_dg_folder_paths
            if entry[1] != "disk_galaxy_low"
        ]
        _prepare_random_snapshot_experiment(
            8,
            exp8_paths,
            args=args,
            load_options=load_options,
            transform_options=transform_options,
            train_balance=train_balance,
            split_label="random_snapshot_without_low_resolution_group",
        )

    # Preserve legacy Exp4's configured third-group holdout. Exp2 should be
    # preferred for new multi-fold generalization studies.
    if "exp4" in args.experiments:
        print("\nProcessing Experiment 4 (legacy single holdout)...")
        valid_nolow, _ = _valid_group_paths(
            [entry for entry in exp_dg_folder_paths if entry[1] != "disk_galaxy_low"]
        )
        if len(valid_nolow) < 3:
            raise ValueError(
                "Legacy Exp4 requires at least three available non-low-resolution groups."
            )
        legacy_args = argparse.Namespace(**vars(args))
        legacy_args.held_out_galaxy = [valid_nolow[2][1]]
        print(
            "  [INFO] Exp4 retains its historical third-available-group holdout. "
            "Use Exp2 for exhaustive leave-one-galaxy-out."
        )
        # Exp4 keeps its historical prefix, so prepare its one fold explicitly.
        held = valid_nolow[2]
        in_entries = [entry for entry in valid_nolow if entry[1] != held[1]]
        cache_in = CachePt().load(in_entries, **load_options)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6,
            validation_size=0.2,
            stratify="groups",
            random_state=args.seed,
        )
        train_cache = train_cache.process(mirror=False, balance=train_balance)
        val_cache = val_cache.process(mirror=False)
        test_cache = test_cache.process(mirror=False)
        cache_out = CachePt().load([held], **load_options).process(mirror=False)
        actual_groups = _assert_logo_partition(
            train_cache,
            val_cache,
            test_cache,
            cache_out,
            train_groups=[entry[1] for entry in in_entries],
            held_out_group=held[1],
        )
        save_path = os.path.join(".cache", _cache_prefix(4, args.cache_tag))
        mean, std, resolved_transform_options, scale_selection = (
            _write_train_cache_with_resolved_closure_scale(
                train_cache,
                save_path,
                transform_options,
            )
        )
        _save_norm_stats(
            save_path,
            mean,
            std,
            input_channels=train_cache.input_channels,
            excluded_channels=excluded_channels,
            target_metadata=train_cache.dataset.target_metadata,
            includes_zero_targets=include_zero_targets,
            oversample_groups=args.oversample,
            cache_schema_version=train_cache.last_cache_schema_version,
            conditioning_contract=train_cache.last_conditioning_contract,
            geometry_inputs=train_cache.last_geometry_input_contract,
            closure_scale_selection=scale_selection,
            split_metadata={
                "experiment": "exp4",
                "split_strategy": "legacy_single_group_holdout",
                "split_seed": int(args.seed),
                "available_groups": [entry[1] for entry in valid_nolow],
                "train_groups": [entry[1] for entry in in_entries],
                "held_out_group": held[1],
                "held_out_groups": [held[1]],
                "out_groups": [held[1]],
                "split_groups": {key: sorted(value) for key, value in actual_groups.items()},
                "group_paths": {entry[1]: entry[0] for entry in valid_nolow},
                "sample_counts": {
                    "train": len(train_cache), "val": len(val_cache),
                    "test": len(test_cache), "out": len(cache_out),
                },
                "cache_prefix": os.path.basename(save_path),
            },
            **resolved_transform_options,
        )
        val_cache.to_pt(save_path + "val.pt", mean, std, **resolved_transform_options)
        test_cache.to_pt(save_path + "test.pt", mean, std, **resolved_transform_options)
        cache_out.to_pt(save_path + "out.pt", mean, std, **resolved_transform_options)
