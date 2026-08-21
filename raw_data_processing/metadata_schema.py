"""MACNet 处理后 snapshot 的 metadata 契约与稳定 UID 工具。"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METADATA_SCHEMA_NAME = "macnet_galdata"
METADATA_SCHEMA_VERSION = 1
UID_HASH_LENGTH = 20


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """流式计算源文件 SHA-256，避免把整个 snapshot 一次读入内存。"""

    digest = sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """把 NumPy 标量和数组转换为稳定、严格的 JSON 值。"""

    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata 不允许包含 NaN 或无穷浮点数。")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"metadata 包含不可序列化类型: {type(value)!r}")


def canonical_json(value: Any) -> str:
    """生成跨平台稳定的规范 JSON 字符串。"""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def metadata_to_json(metadata: Mapping[str, Any]) -> str:
    """序列化 metadata；公开函数便于 HDF5 层统一使用。"""

    return canonical_json(metadata)


def metadata_from_json(payload: str | bytes) -> dict[str, Any]:
    """读取并验证 JSON metadata 的顶层结构。"""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    value = json.loads(str(payload))
    if not isinstance(value, dict):
        raise ValueError("metadata JSON 的顶层必须是对象。")
    return value


def _uid(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:UID_HASH_LENGTH]}"


def build_parent_hr_uid(source: Mapping[str, Any]) -> str:
    """由物理源身份构造与文件路径无关的 HR snapshot UID。"""

    required = (
        "simulation_code",
        "galaxy_id",
        "run_id",
        "source_series_id",
        "raw_snapshot_index",
    )
    missing = [key for key in required if source.get(key) in (None, "")]
    if missing:
        raise ValueError(f"构造 parent_hr_uid 缺少字段: {missing}")
    identity = {key: source[key] for key in required}
    return _uid("macerhr", identity)


def geometry_fingerprint(
    axes: Sequence[np.ndarray],
    axis_edges: Sequence[np.ndarray],
    *,
    coordinate_mode: str,
) -> str:
    """对坐标数组内容生成短指纹，区分同形状但不同物理网格。"""

    digest = sha256()
    digest.update(str(coordinate_mode).encode("utf-8"))
    for collection in (axes, axis_edges):
        for values in collection:
            array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()[:UID_HASH_LENGTH]


def build_sample_uid(
    parent_hr_uid: str,
    *,
    operator_id: str,
    operator_version: int | str,
    operator_parameters: Mapping[str, Any],
    output_geometry_fingerprint: str,
) -> str:
    """构造一个确定性衍生数据产品的 UID。"""

    if not parent_hr_uid:
        raise ValueError("构造 sample_uid 需要非空 parent_hr_uid。")
    payload = {
        "parent_hr_uid": str(parent_hr_uid),
        "operator_id": str(operator_id),
        "operator_version": operator_version,
        "operator_parameters": operator_parameters,
        "output_geometry_fingerprint": str(output_geometry_fingerprint),
    }
    return _uid("macnetsample", payload)


def build_physics_product_uid(
    sample_uid: str,
    *,
    operator_id: str,
    operator_version: int | str,
    operator_parameters: Mapping[str, Any],
    context: Mapping[str, Any],
) -> str:
    """区分同一粗粒化样本上的不同物理基线产品。"""

    if not sample_uid:
        raise ValueError("构造 physics_product_uid 需要非空 sample_uid。")
    return _uid(
        "macnetphysics",
        {
            "sample_uid": str(sample_uid),
            "operator_id": str(operator_id),
            "operator_version": operator_version,
            "operator_parameters": operator_parameters,
            "context": context,
        },
    )


def default_field_schema(snapshot_fields: Sequence[str]) -> dict[str, Any]:
    """返回目前已明确的 MACER 字段语义；未知单位不作猜测。"""

    known = {
        "density": {
            "kind": "intensive_scalar",
            "quantity": "gas_density",
            "units": "2.5e7_Msun/kpc^3",
        },
        "gas_energy": {
            "kind": "intensive_scalar",
            "quantity": "gas_internal_energy_density",
            "source_dataset": "Data-Set-10",
            "units": "2.5e7_Msun/(kpc*Gyr^2)",
        },
        "temperature": {
            "kind": "intensive_scalar",
            "quantity": "gas_temperature",
            "units": "K",
        },
        "v1": {"kind": "vector_component", "basis": "spherical", "component": "r", "units": "kpc/Gyr"},
        "v2": {"kind": "vector_component", "basis": "spherical", "component": "theta", "units": "kpc/Gyr"},
        "v3": {"kind": "vector_component", "basis": "spherical", "component": "phi", "units": "kpc/Gyr"},
        "initial_stellar_density": {
            "kind": "intensive_scalar",
            "quantity": "initial_stellar_mass_density",
            "units": "2.5e7_Msun/kpc^3",
        },
        "radial_acceleration": {
            "kind": "vector_component",
            "basis": "spherical",
            "component": "r",
            "quantity": "gravitational_acceleration",
            "source_dataset": "Data-Set-8",
            "units": "kpc/Gyr^2",
        },
        "theta_acceleration": {
            "kind": "vector_component",
            "basis": "spherical",
            "component": "theta",
            "quantity": "gravitational_acceleration",
            "source_dataset": "Data-Set-9",
            "units": "kpc/Gyr^2",
        },
        "cooling_time": {
            "kind": "intensive_scalar",
            "quantity": "gas_cooling_time",
            "source_dataset": "Data-Set-17",
            "units": "Gyr",
        },
        "cooling_rate_density": {
            "kind": "derived_intensive_scalar",
            "quantity": "gas_cooling_rate_density",
            "definition": "gas_energy/cooling_time",
            "units": "2.5e7_Msun/(kpc*Gyr^3)",
        },
    }
    fields = {}
    for name in snapshot_fields:
        schema = dict(known.get(name, {"kind": "unclassified"}))
        schema.setdefault("units", "MACER_code_units_unspecified")
        fields[str(name)] = schema
    fields["grid_volume"] = {
        "kind": "geometric_extensive",
        "quantity": "axisymmetric_cell_volume",
        "source_dataset": "Data-Set-15",
        "units": "kpc^3",
    }
    fields["gas_mass"] = {
        "kind": "derived_extensive",
        "definition": "density*grid_volume",
        "units": "2.5e7_Msun",
    }
    return fields


def build_fine_metadata(
    *,
    source: Mapping[str, Any],
    snapshot_fields: Sequence[str],
    axes: Sequence[np.ndarray],
    axis_edges: Sequence[np.ndarray],
    coordinate_mode: str,
    ndim: int,
    target_metadata: Mapping[str, Any],
    volume_source: str,
) -> dict[str, Any]:
    """创建一个完整 fine snapshot 的版本化 metadata。"""

    source_dict = _json_value(source)
    parent_hr_uid = build_parent_hr_uid(source_dict)
    geometry_hash = geometry_fingerprint(
        axes, axis_edges, coordinate_mode=coordinate_mode
    )
    operator_parameters = {
        "source_shape": [len(np.asarray(axis)) for axis in reversed(axes)]
        if int(ndim) == 2
        else [len(np.asarray(axis)) for axis in axes],
        "coordinate_mode": str(coordinate_mode),
    }
    sample_uid = build_sample_uid(
        parent_hr_uid,
        operator_id="identity_fine",
        operator_version=1,
        operator_parameters=operator_parameters,
        output_geometry_fingerprint=geometry_hash,
    )
    return {
        "schema_name": METADATA_SCHEMA_NAME,
        "schema_version": METADATA_SCHEMA_VERSION,
        "metadata_completeness": "complete",
        "identity": {
            "parent_hr_uid": parent_hr_uid,
            "sample_uid": sample_uid,
        },
        "source": source_dict,
        "geometry": {
            "coordinate_mode": str(coordinate_mode),
            "ndim": int(ndim),
            "coordinate_axis_order": ["r", "theta"] if int(ndim) == 2 else [],
            "spatial_axis_order": ["theta", "r"] if int(ndim) == 2 else [],
            "vector_basis": "spherical" if coordinate_mode == "polar" else "cartesian",
            "volume_source": str(volume_source),
            "geometry_fingerprint": geometry_hash,
        },
        "fields": default_field_schema(snapshot_fields),
        "target": _json_value(target_metadata),
        "derivation": {
            "operator_id": "identity_fine",
            "operator_version": 1,
            "operator_parameters": operator_parameters,
        },
    }


def build_derived_metadata(
    source_metadata: Mapping[str, Any],
    *,
    operator_metadata: Mapping[str, Any],
    axes: Sequence[np.ndarray],
    axis_edges: Sequence[np.ndarray],
    coordinate_mode: str,
    volume_source: str,
) -> dict[str, Any]:
    """继承 HR 身份并为一个粗粒化结果生成新的 sample_uid。"""

    metadata = deepcopy(dict(source_metadata))
    identity = metadata.get("identity", {})
    parent_hr_uid = identity.get("parent_hr_uid")
    if not parent_hr_uid:
        return {
            "schema_name": METADATA_SCHEMA_NAME,
            "schema_version": METADATA_SCHEMA_VERSION,
            "metadata_completeness": "legacy_missing_source_identity",
        }

    operator_id = str(operator_metadata.get("operator", "unknown"))
    operator_version = operator_metadata.get("operator_version", 0)
    operator_parameters = operator_metadata.get("operator_parameters")
    if not isinstance(operator_parameters, Mapping):
        raise ValueError(
            f"粗粒化算子 {operator_id!r} 未提供结构化 operator_parameters。"
        )
    geometry_hash = geometry_fingerprint(
        axes, axis_edges, coordinate_mode=coordinate_mode
    )
    sample_uid = build_sample_uid(
        str(parent_hr_uid),
        operator_id=operator_id,
        operator_version=operator_version,
        operator_parameters=operator_parameters,
        output_geometry_fingerprint=geometry_hash,
    )
    metadata["identity"] = {
        "parent_hr_uid": str(parent_hr_uid),
        "sample_uid": sample_uid,
    }
    metadata["metadata_completeness"] = "complete"
    metadata["geometry"] = {
        **dict(metadata.get("geometry", {})),
        "volume_source": str(volume_source),
        "geometry_fingerprint": geometry_hash,
    }
    metadata["derivation"] = {
        "operator_id": operator_id,
        "operator_version": operator_version,
        "operator_parameters": _json_value(operator_parameters),
    }
    return metadata


def validate_complete_metadata(metadata: Mapping[str, Any]) -> None:
    """对新生成样本执行最小但严格的身份契约检查。"""

    if metadata.get("schema_name") != METADATA_SCHEMA_NAME:
        raise ValueError("metadata schema_name 不匹配。")
    if int(metadata.get("schema_version", -1)) != METADATA_SCHEMA_VERSION:
        raise ValueError("metadata schema_version 不匹配。")
    if metadata.get("metadata_completeness") != "complete":
        raise ValueError("样本缺少完整源身份，不能用于多分辨率实验。")
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("metadata 缺少 identity 对象。")
    for key in ("parent_hr_uid", "sample_uid"):
        if not identity.get(key):
            raise ValueError(f"metadata.identity 缺少 {key}。")
    expected_parent_uid = build_parent_hr_uid(metadata.get("source", {}))
    if identity["parent_hr_uid"] != expected_parent_uid:
        raise ValueError("parent_hr_uid 与 source 身份字段不一致。")
    derivation = metadata.get("derivation", {})
    geometry = metadata.get("geometry", {})
    expected_sample_uid = build_sample_uid(
        expected_parent_uid,
        operator_id=derivation.get("operator_id", ""),
        operator_version=derivation.get("operator_version", ""),
        operator_parameters=derivation.get("operator_parameters", {}),
        output_geometry_fingerprint=geometry.get("geometry_fingerprint", ""),
    )
    if identity["sample_uid"] != expected_sample_uid:
        raise ValueError("sample_uid 与 derivation/geometry 字段不一致。")
    canonical_json(metadata)
