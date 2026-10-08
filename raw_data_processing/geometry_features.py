"""二维球极网格的显式物理尺度输入，不改变物理场、体积或监督量。"""

from __future__ import annotations

import numpy as np


SUPPORTED_GEOMETRY_INPUT_MODES = (
    "none",
    "absolute_radius",
    "absolute_radius_widths",
)
GEOMETRY_INPUT_CHANNELS = (
    "log10_radius_kpc",
    "log10_radial_width_kpc",
    "log10_polar_arc_width_kpc",
)


def geometry_input_channels(mode: str) -> tuple[str, ...]:
    """返回各消融模式追加到 x 的有序通道名。"""

    if mode not in SUPPORTED_GEOMETRY_INPUT_MODES:
        raise ValueError(
            f"未知 geometry_input_mode={mode!r}；允许值为 {SUPPORTED_GEOMETRY_INPUT_MODES}。"
        )
    if mode == "none":
        return ()
    if mode == "absolute_radius":
        return GEOMETRY_INPUT_CHANNELS[:1]
    return GEOMETRY_INPUT_CHANNELS


def geometry_input_contract(mode: str) -> dict:
    """返回可序列化的尺度输入契约；参考长度始终固定为 1 kpc。"""

    channels = geometry_input_channels(mode)
    definitions = {
        "log10_radius_kpc": {
            "formula": "log10(r/(1 kpc))",
            "source": "coordinates/axis_0",
        },
        "log10_radial_width_kpc": {
            "formula": "log10((r_plus-r_minus)/(1 kpc))",
            "source": "coordinates/axis_0_edges",
        },
        "log10_polar_arc_width_kpc": {
            "formula": "log10(r*(theta_plus-theta_minus)/(1 kpc))",
            "source": "coordinates/axis_0;coordinates/axis_1_edges",
        },
    }
    return {
        "version": 1,
        "mode": mode,
        "channels": list(channels),
        "coordinate_mode": "polar" if channels else None,
        "array_order": "theta,r" if channels else None,
        "length_units": "kpc",
        "angle_units": "rad",
        "reference_length_kpc": 1.0,
        "channel_units": "dimensionless",
        "transform": "log10_fixed_reference",
        "additional_log1p": False,
        "normalization": "training_split_mean_std_only",
        "normalization_source": "training_split",
        "source": "processed_hdf5_coordinates",
        "per_snapshot_minmax": False,
        "fields": {name: definitions[name] for name in channels},
        "authoritative_volume_unchanged": True,
        "original_fourier_encoding_unchanged": True,
    }


def build_geometry_input_features(
    *,
    axes,
    axis_edges,
    coordinate_mode: str,
    ndim: int,
    shape,
    mode: str,
) -> dict[str, np.ndarray]:
    """从保存的坐标中心和真实边界计算 (N_theta,N_r) 尺度通道。

    不从中心间距重新推断边界，也不以几何公式覆盖 Data-Set-15 体积。
    宽度是输出粗网格的几何宽度，不等同于 R_m 的输运核支撑尺度。
    """

    channels = geometry_input_channels(mode)
    if not channels:
        return {}
    if ndim != 2 or coordinate_mode != "polar":
        raise ValueError("显式半径/宽度输入目前仅支持二维 polar 球极网格。")
    if len(axes) != 2 or len(axis_edges) != 2:
        raise ValueError("二维几何输入必须保存 r、theta 两个坐标轴及其边界。")
    radius, theta = (np.asarray(axis, dtype=np.float64) for axis in axes)
    radial_edges, theta_edges = (
        np.asarray(edges, dtype=np.float64) for edges in axis_edges
    )
    if any(axis.ndim != 1 or axis.size == 0 for axis in (radius, theta)):
        raise ValueError("r、theta 必须是非空的一维坐标轴。")
    expected_shape = (theta.size, radius.size)
    if tuple(shape) != expected_shape:
        raise ValueError(
            f"几何输入 shape 必须按 (theta,r) 对齐：{tuple(shape)} != {expected_shape}。"
        )
    for name, axis, edges in (
        ("r", radius, radial_edges),
        ("theta", theta, theta_edges),
    ):
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} 坐标中心必须有限且严格递增。")
        if edges.shape != (axis.size + 1,) or not np.all(np.isfinite(edges)):
            raise ValueError(f"{name} 边界必须有限且包含 len(axis)+1 个值。")
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"{name} 边界必须严格递增，以保证正单元宽度。")
        if np.any(axis < edges[:-1]) or np.any(axis > edges[1:]):
            raise ValueError(f"{name} 坐标中心必须位于对应单元的保存边界内。")
    if np.any(radius <= 0) or np.any(radial_edges < 0):
        raise ValueError("半径中心必须严格为正，径向边界不能为负。")
    angular_tolerance = 1.0e-6
    if theta_edges[0] < -angular_tolerance or theta_edges[-1] > np.pi + angular_tolerance:
        raise ValueError("球极角边界必须处于 [0,pi] rad；不能将方位角误作 theta。")

    radius_field = np.broadcast_to(radius[None, :], expected_shape)
    fields = {"log10_radius_kpc": np.log10(radius_field)}
    if mode == "absolute_radius_widths":
        radial_width = np.broadcast_to(np.diff(radial_edges)[None, :], expected_shape)
        polar_width = radius_field * np.diff(theta_edges)[:, None]
        if np.any(radial_width <= 0) or np.any(polar_width <= 0):
            raise ValueError("物理单元宽度必须严格为正。")
        fields["log10_radial_width_kpc"] = np.log10(radial_width)
        fields["log10_polar_arc_width_kpc"] = np.log10(polar_width)
    if any(not np.all(np.isfinite(field)) for field in fields.values()):
        raise ValueError("几何对数通道含非有限值；拒绝截断或默默替换无效几何。")
    return {name: np.array(fields[name], dtype=np.float64, copy=True) for name in channels}
