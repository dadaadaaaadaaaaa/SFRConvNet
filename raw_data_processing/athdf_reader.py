"""MACER3D/Athena++ ``.athdf`` 快照的严格选择性读取器。

本模块只负责三件事：验证文件结构、按 ``LogicalLocations`` 重建全局网格，
以及从真实 cell faces 计算球坐标单元体积。它不负责降采样，也不替调用方
猜测字段的物理语义。

项目内部统一采用物理轴顺序 ``(r, theta, phi)``。Athena++ 的 block 数据
通常采用 ``(x3, x2, x1) == (phi, theta, r)``，读取时会显式转置。
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np


PHYSICAL_AXIS_ORDER = ("r", "theta", "phi")
ATHENA_BLOCK_ARRAY_ORDER = ("phi", "theta", "r")

# 模型输入、baseline 依赖与监督字段在常量层面明确分开，避免调用方把
# baseline 中间量或标签误当成神经网络输入通道。
DEFAULT_MODEL_INPUT_FIELDS = (
    "rho",
    "press",
    "vel1",
    "vel2",
    "vel3",
)
DEFAULT_BASELINE_FIELDS = ("cool_rate",)
DEFAULT_TARGET_FIELD = "rhodotsf"

# 兼容读取器初版导出的名称；它现在严格等价于模型输入，不含 cool_rate。
DEFAULT_INPUT_FIELDS = DEFAULT_MODEL_INPUT_FIELDS

# 这是读取一份训练样本通常需要从原始文件取出的最小集合，不代表其中所有
# 字段都会进入模型输入。
DEFAULT_REQUIRED_FIELDS = (
    *DEFAULT_MODEL_INPUT_FIELDS,
    *DEFAULT_BASELINE_FIELDS,
    DEFAULT_TARGET_FIELD,
)


@dataclass(frozen=True)
class AthdfVariableLocation:
    """一个 Athena++ 变量在 HDF5 数据集中的位置。"""

    dataset_name: str
    local_index: int


@dataclass(frozen=True)
class AthdfMetadata:
    """不读取物理场即可获得的快照结构元数据。"""

    path: Path
    time: float
    coordinate_system: str
    root_grid_size: tuple[int, int, int]
    mesh_block_size: tuple[int, int, int]
    max_level: int
    num_mesh_blocks: int
    dataset_names: tuple[str, ...]
    num_variables: tuple[int, ...]
    variable_names: tuple[str, ...]
    variable_locations: Mapping[str, AthdfVariableLocation]
    dataset_shapes: Mapping[str, tuple[int, ...]]
    dataset_dtypes: Mapping[str, str]
    physical_axis_order: tuple[str, str, str] = PHYSICAL_AXIS_ORDER
    athena_block_array_order: tuple[str, str, str] = ATHENA_BLOCK_ARRAY_ORDER


@dataclass(frozen=True)
class AthdfGeometry:
    """按 ``(r, theta, phi)`` 排列的球坐标几何。"""

    centers: tuple[np.ndarray, np.ndarray, np.ndarray]
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    cell_volume: np.ndarray
    volume_source: str = "athdf_cell_faces:spherical_exact"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(values.size) for values in self.centers)


@dataclass(frozen=True)
class AthdfSnapshot:
    """一个经过结构验证、按需读取的 MACER3D 快照。"""

    metadata: AthdfMetadata
    geometry: AthdfGeometry
    fields: Mapping[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.geometry.shape


def _import_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境。
        raise ModuleNotFoundError(
            "读取 .athdf 需要 h5py。请在 MACNet 使用的 Python 环境中安装 h5py。"
        ) from exc
    return h5py


def _decode_scalar(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _decode_sequence(values: Any) -> tuple[str, ...]:
    return tuple(_decode_scalar(value) for value in np.atleast_1d(values))


def _integer_tuple(values: Any, *, name: str, length: int) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.shape != (length,):
        raise ValueError(f"ATHDF 属性 {name!r} 应为长度 {length}，实际为 {array.shape}。")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"ATHDF 属性 {name!r} 必须为整数。")
    result = tuple(int(value) for value in array)
    if any(value <= 0 for value in result):
        raise ValueError(f"ATHDF 属性 {name!r} 必须全部为正数，实际为 {result}。")
    return result


def _require_attributes(handle: Any, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in handle.attrs]
    if missing:
        raise ValueError(f"ATHDF 缺少必要属性：{missing}。")


def _validate_logical_locations(
    handle: Any,
    *,
    root_grid_size: tuple[int, int, int],
    mesh_block_size: tuple[int, int, int],
    max_level: int,
) -> np.ndarray:
    if max_level != 0:
        raise NotImplementedError(
            "当前 ATHDF 读取器仅支持 MaxLevel == 0 的均匀网格；"
            f"该文件 MaxLevel={max_level}。AMR 数据需要单独的保守限制算法。"
        )
    for root_size, block_size in zip(root_grid_size, mesh_block_size):
        if root_size % block_size != 0:
            raise ValueError(
                "RootGridSize 必须能被 MeshBlockSize 整除："
                f"RootGridSize={root_grid_size}, MeshBlockSize={mesh_block_size}。"
            )

    if "LogicalLocations" not in handle or "Levels" not in handle:
        raise ValueError("ATHDF 缺少 LogicalLocations 或 Levels 数据集。")
    logical_locations = np.asarray(handle["LogicalLocations"][()], dtype=np.int64)
    if logical_locations.ndim != 2 or logical_locations.shape[1] != 3:
        raise ValueError(
            "LogicalLocations 应具有 (num_mesh_blocks, 3) 形状，"
            f"实际为 {logical_locations.shape}。"
        )
    levels = np.asarray(handle["Levels"][()], dtype=np.int64)
    if levels.shape != (logical_locations.shape[0],):
        raise ValueError("Levels 的长度与 LogicalLocations 不一致。")
    if np.any(levels != 0):
        raise NotImplementedError("当前读取器不支持包含非零 refinement level 的 ATHDF。")

    block_grid_size = tuple(
        root_size // block_size
        for root_size, block_size in zip(root_grid_size, mesh_block_size)
    )
    expected_locations = set(itertools.product(*(range(size) for size in block_grid_size)))
    actual_locations = {tuple(int(value) for value in row) for row in logical_locations}
    if len(actual_locations) != logical_locations.shape[0]:
        raise ValueError("LogicalLocations 包含重复 MeshBlock 位置。")
    if actual_locations != expected_locations:
        missing = sorted(expected_locations.difference(actual_locations))
        extra = sorted(actual_locations.difference(expected_locations))
        raise ValueError(
            "LogicalLocations 未完整覆盖均匀全局网格："
            f"missing={missing[:5]}, extra={extra[:5]}。"
        )
    return logical_locations


def _metadata_from_handle(handle: Any, path: Path) -> tuple[AthdfMetadata, np.ndarray]:
    _require_attributes(
        handle,
        (
            "Coordinates",
            "RootGridSize",
            "MeshBlockSize",
            "MaxLevel",
            "Time",
            "DatasetNames",
            "NumVariables",
            "VariableNames",
        ),
    )
    coordinate_system = _decode_scalar(handle.attrs["Coordinates"])
    if coordinate_system != "spherical_polar":
        raise ValueError(
            "当前读取器只支持 spherical_polar 坐标，"
            f"实际为 {coordinate_system!r}。"
        )

    root_grid_size = _integer_tuple(
        handle.attrs["RootGridSize"], name="RootGridSize", length=3
    )
    mesh_block_size = _integer_tuple(
        handle.attrs["MeshBlockSize"], name="MeshBlockSize", length=3
    )
    max_level = int(handle.attrs["MaxLevel"])
    logical_locations = _validate_logical_locations(
        handle,
        root_grid_size=root_grid_size,
        mesh_block_size=mesh_block_size,
        max_level=max_level,
    )

    dataset_names = _decode_sequence(handle.attrs["DatasetNames"])
    num_variables_array = np.asarray(handle.attrs["NumVariables"])
    if num_variables_array.ndim != 1 or not np.issubdtype(
        num_variables_array.dtype, np.integer
    ):
        raise ValueError("NumVariables 必须是一维整数数组。")
    num_variables = tuple(int(value) for value in num_variables_array)
    variable_names = _decode_sequence(handle.attrs["VariableNames"])
    if len(dataset_names) != len(num_variables):
        raise ValueError("DatasetNames 与 NumVariables 的长度不一致。")
    if sum(num_variables) != len(variable_names):
        raise ValueError(
            "NumVariables 之和与 VariableNames 数量不一致："
            f"{sum(num_variables)} != {len(variable_names)}。"
        )
    if len(set(variable_names)) != len(variable_names):
        raise ValueError("VariableNames 包含重复名称，无法建立无歧义字段映射。")

    num_mesh_blocks = int(logical_locations.shape[0])
    expected_block_array_shape = tuple(reversed(mesh_block_size))
    variable_locations: dict[str, AthdfVariableLocation] = {}
    dataset_shapes: dict[str, tuple[int, ...]] = {}
    dataset_dtypes: dict[str, str] = {}
    offset = 0
    for dataset_name, count in zip(dataset_names, num_variables):
        if dataset_name not in handle:
            raise ValueError(f"DatasetNames 声明的数据集 {dataset_name!r} 不存在。")
        dataset = handle[dataset_name]
        expected_shape = (count, num_mesh_blocks, *expected_block_array_shape)
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(
                f"数据集 {dataset_name!r} 形状错误："
                f"实际 {tuple(dataset.shape)}，预期 {expected_shape}。"
            )
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError(f"数据集 {dataset_name!r} 不是数值类型。")
        dataset_shapes[dataset_name] = tuple(int(value) for value in dataset.shape)
        dataset_dtypes[dataset_name] = str(dataset.dtype)
        for local_index in range(count):
            variable_name = variable_names[offset + local_index]
            variable_locations[variable_name] = AthdfVariableLocation(
                dataset_name=dataset_name,
                local_index=local_index,
            )
        offset += count

    metadata = AthdfMetadata(
        path=path,
        time=float(handle.attrs["Time"]),
        coordinate_system=coordinate_system,
        root_grid_size=root_grid_size,
        mesh_block_size=mesh_block_size,
        max_level=max_level,
        num_mesh_blocks=num_mesh_blocks,
        dataset_names=dataset_names,
        num_variables=num_variables,
        variable_names=variable_names,
        variable_locations=MappingProxyType(variable_locations),
        dataset_shapes=MappingProxyType(dataset_shapes),
        dataset_dtypes=MappingProxyType(dataset_dtypes),
    )
    return metadata, logical_locations


def inspect_athdf(path: str | Path) -> AthdfMetadata:
    """验证一个 ATHDF 文件并返回不含物理场数组的元数据。"""

    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"ATHDF 文件不存在：{resolved_path}")
    with h5py.File(resolved_path, "r") as handle:
        metadata, _ = _metadata_from_handle(handle, resolved_path)
    return metadata


def _assemble_axis(
    handle: Any,
    logical_locations: np.ndarray,
    *,
    axis_index: int,
    root_size: int,
    block_size: int,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    face_name = f"x{axis_index + 1}f"
    center_name = f"x{axis_index + 1}v"
    if face_name not in handle or center_name not in handle:
        raise ValueError(f"ATHDF 缺少坐标数据集 {face_name!r} 或 {center_name!r}。")
    faces = handle[face_name]
    centers = handle[center_name]
    expected_face_shape = (logical_locations.shape[0], block_size + 1)
    expected_center_shape = (logical_locations.shape[0], block_size)
    if tuple(faces.shape) != expected_face_shape:
        raise ValueError(
            f"{face_name} 形状错误：实际 {tuple(faces.shape)}，"
            f"预期 {expected_face_shape}。"
        )
    if tuple(centers.shape) != expected_center_shape:
        raise ValueError(
            f"{center_name} 形状错误：实际 {tuple(centers.shape)}，"
            f"预期 {expected_center_shape}。"
        )

    num_axis_blocks = root_size // block_size
    face_segments: list[np.ndarray | None] = [None] * num_axis_blocks
    center_segments: list[np.ndarray | None] = [None] * num_axis_blocks
    for block_index, location in enumerate(logical_locations):
        logical_axis_index = int(location[axis_index])
        local_faces = np.asarray(faces[block_index], dtype=np.float64)
        local_centers = np.asarray(centers[block_index], dtype=np.float64)
        old_faces = face_segments[logical_axis_index]
        old_centers = center_segments[logical_axis_index]
        if old_faces is None:
            face_segments[logical_axis_index] = local_faces
            center_segments[logical_axis_index] = local_centers
        elif not np.allclose(old_faces, local_faces, rtol=rtol, atol=atol):
            raise ValueError(
                f"相同逻辑 {face_name} block 在其他方向上给出了不一致的坐标。"
            )
        elif not np.allclose(old_centers, local_centers, rtol=rtol, atol=atol):
            raise ValueError(
                f"相同逻辑 {center_name} block 在其他方向上给出了不一致的坐标。"
            )

    if any(values is None for values in face_segments + center_segments):
        raise ValueError(f"{face_name}/{center_name} 未覆盖全部逻辑 block。")

    global_faces = np.empty(root_size + 1, dtype=np.float64)
    global_centers = np.empty(root_size, dtype=np.float64)
    for logical_axis_index in range(num_axis_blocks):
        local_faces = face_segments[logical_axis_index]
        local_centers = center_segments[logical_axis_index]
        assert local_faces is not None and local_centers is not None
        start = logical_axis_index * block_size
        if logical_axis_index > 0 and not np.isclose(
            global_faces[start], local_faces[0], rtol=rtol, atol=atol
        ):
            raise ValueError(f"相邻 {face_name} block 的公共边界不连续。")
        global_faces[start : start + block_size + 1] = local_faces
        global_centers[start : start + block_size] = local_centers

    if not np.all(np.isfinite(global_faces)) or np.any(np.diff(global_faces) <= 0):
        raise ValueError(f"全局 {face_name} 必须有限且严格递增。")
    if not np.all(np.isfinite(global_centers)) or np.any(np.diff(global_centers) <= 0):
        raise ValueError(f"全局 {center_name} 必须有限且严格递增。")
    if np.any(global_centers <= global_faces[:-1]) or np.any(
        global_centers >= global_faces[1:]
    ):
        raise ValueError(f"{center_name} 中存在不位于相应 cell faces 之间的中心。")
    return global_centers, global_faces


def _read_geometry(
    handle: Any,
    metadata: AthdfMetadata,
    logical_locations: np.ndarray,
) -> AthdfGeometry:
    centers: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    for axis_index, (root_size, block_size) in enumerate(
        zip(metadata.root_grid_size, metadata.mesh_block_size)
    ):
        axis_centers, axis_edges = _assemble_axis(
            handle,
            logical_locations,
            axis_index=axis_index,
            root_size=root_size,
            block_size=block_size,
        )
        centers.append(axis_centers)
        edges.append(axis_edges)

    radius_edges, theta_edges, phi_edges = edges
    if np.any(radius_edges <= 0):
        raise ValueError("球坐标径向 cell faces 必须严格为正。")
    if theta_edges[0] < 0 or theta_edges[-1] > np.pi:
        raise ValueError("theta cell faces 必须位于 [0, pi]。")
    phi_span = float(phi_edges[-1] - phi_edges[0])
    if not np.isclose(phi_span, 2.0 * np.pi, rtol=1e-6, atol=1e-7):
        raise ValueError(
            "当前读取器要求完整的 2pi 方位角区域，"
            f"实际跨度为 {phi_span:.16g}。"
        )

    radial_factor = (radius_edges[1:] ** 3 - radius_edges[:-1] ** 3) / 3.0
    polar_factor = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    azimuthal_factor = np.diff(phi_edges)
    cell_volume = (
        radial_factor[:, None, None]
        * polar_factor[None, :, None]
        * azimuthal_factor[None, None, :]
    )
    if cell_volume.shape != metadata.root_grid_size:
        raise RuntimeError("内部错误：球坐标体积数组形状与 RootGridSize 不一致。")
    if not np.all(np.isfinite(cell_volume)) or np.any(cell_volume <= 0):
        raise ValueError("由 ATHDF cell faces 计算出的体积必须有限且严格为正。")

    analytic_total = (
        (radius_edges[-1] ** 3 - radius_edges[0] ** 3)
        / 3.0
        * (np.cos(theta_edges[0]) - np.cos(theta_edges[-1]))
        * phi_span
    )
    if not np.isclose(
        cell_volume.sum(dtype=np.float64), analytic_total, rtol=1e-12, atol=0.0
    ):
        raise ValueError("逐 cell 体积总和与球坐标区域解析体积不一致。")

    return AthdfGeometry(
        centers=(centers[0], centers[1], centers[2]),
        edges=(edges[0], edges[1], edges[2]),
        cell_volume=cell_volume,
    )


def _normalize_requested_fields(
    fields: str | Iterable[str] | None,
) -> tuple[str, ...]:
    if fields is None:
        return DEFAULT_REQUIRED_FIELDS
    if isinstance(fields, str):
        requested = (fields,)
    else:
        requested = tuple(str(field) for field in fields)
    if any(not field for field in requested):
        raise ValueError("请求字段名不能为空字符串。")
    if len(set(requested)) != len(requested):
        raise ValueError("请求字段列表包含重复名称。")
    return requested


def _assemble_field(
    handle: Any,
    metadata: AthdfMetadata,
    logical_locations: np.ndarray,
    field_name: str,
    *,
    dtype: np.dtype | None,
    validate_finite: bool,
) -> np.ndarray:
    location = metadata.variable_locations[field_name]
    dataset = handle[location.dataset_name]
    block_values = np.asarray(dataset[location.local_index, ...])
    output_dtype = block_values.dtype if dtype is None else dtype
    output = np.empty(metadata.root_grid_size, dtype=output_dtype)
    block_r, block_theta, block_phi = metadata.mesh_block_size

    for block_index, (logical_r, logical_theta, logical_phi) in enumerate(
        logical_locations
    ):
        r_start = int(logical_r) * block_r
        theta_start = int(logical_theta) * block_theta
        phi_start = int(logical_phi) * block_phi
        output[
            r_start : r_start + block_r,
            theta_start : theta_start + block_theta,
            phi_start : phi_start + block_phi,
        ] = np.transpose(block_values[block_index], (2, 1, 0))

    if validate_finite and not np.all(np.isfinite(output)):
        raise ValueError(f"字段 {field_name!r} 包含 NaN 或 Inf。")
    return output


def read_athdf(
    path: str | Path,
    fields: str | Iterable[str] | None = None,
    *,
    dtype: np.dtype | str | type | None = np.float32,
    validate_finite: bool = True,
) -> AthdfSnapshot:
    """按需读取并重建一个无 AMR 的球坐标 ATHDF 快照。

    Parameters
    ----------
    path
        ``.athdf`` 文件路径。
    fields
        要读取的字段名。传入 ``None`` 时使用
        :data:`DEFAULT_REQUIRED_FIELDS`；传入空元组可只读取几何。
    dtype
        输出物理场的数据类型。几何与体积始终使用 ``float64``；传入
        ``None`` 时保留 HDF5 中的原始字段类型。
    validate_finite
        是否拒绝包含 NaN/Inf 的请求字段。
    """

    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"ATHDF 文件不存在：{resolved_path}")
    requested_fields = _normalize_requested_fields(fields)
    output_dtype = None if dtype is None else np.dtype(dtype)
    if output_dtype is not None and not np.issubdtype(output_dtype, np.floating):
        raise ValueError("ATHDF 物理场输出 dtype 必须是浮点类型或 None。")

    with h5py.File(resolved_path, "r") as handle:
        metadata, logical_locations = _metadata_from_handle(handle, resolved_path)
        missing = [
            field
            for field in requested_fields
            if field not in metadata.variable_locations
        ]
        if missing:
            raise KeyError(
                f"请求字段不存在：{missing}；可用字段为 {metadata.variable_names}。"
            )
        geometry = _read_geometry(handle, metadata, logical_locations)
        loaded_fields = {
            field_name: _assemble_field(
                handle,
                metadata,
                logical_locations,
                field_name,
                dtype=output_dtype,
                validate_finite=validate_finite,
            )
            for field_name in requested_fields
        }

    return AthdfSnapshot(
        metadata=metadata,
        geometry=geometry,
        fields=MappingProxyType(loaded_fields),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查或选择性读取 MACER3D ATHDF 快照。")
    parser.add_argument("path", type=Path, help="一个 .athdf 文件路径。")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=None,
        help="要实际读取的字段；省略时只打印元数据。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    metadata = inspect_athdf(args.path)
    print(f"path={metadata.path}")
    print(f"time={metadata.time}")
    print(f"coordinates={metadata.coordinate_system}")
    print(f"root_grid_size={metadata.root_grid_size}  # (r, theta, phi)")
    print(f"mesh_block_size={metadata.mesh_block_size}  # (r, theta, phi)")
    print(f"num_mesh_blocks={metadata.num_mesh_blocks}")
    print(f"variables={metadata.variable_names}")
    if args.fields is not None:
        snapshot = read_athdf(args.path, fields=args.fields)
        print(f"loaded_fields={tuple(snapshot.fields)}")
        print(f"field_shape={snapshot.shape}  # (r, theta, phi)")
        print(
            "total_volume="
            f"{snapshot.geometry.cell_volume.sum(dtype=np.float64):.16g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
