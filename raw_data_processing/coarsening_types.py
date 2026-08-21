"""MACER粗粒化方案共用的仅数组数据契约。
这些粗粒化实现有意未导入 :class:`GalData`。将该边界与HDF4/HDF5及训练相关问题隔离，既可避免循环
导入，又能使每个数值运算符都能独立进行测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CoarseningInput:
    """Immutable references to the arrays needed by a coarsening operator.

    Snapshot arrays use MACER storage order.  For current MACER2D polar data
    this is ``(theta, r)``, whereas ``axes`` and ``axis_edges`` use coordinate
    order ``(r, theta)``.
    """

    snapshot: Mapping[str, np.ndarray]
    grid_volume: np.ndarray
    axes: tuple[np.ndarray, ...]
    axis_edges: tuple[np.ndarray, ...]
    coordinate_mode: str
    ndim: int


@dataclass(frozen=True)
class CoarseningResult:
    """Pure-array result returned by every coarse-graining implementation."""

    snapshot: Mapping[str, np.ndarray]
    grid_volume: np.ndarray
    axes: tuple[np.ndarray, ...]
    axis_edges: tuple[np.ndarray, ...]
    volume_source: str
    metadata: Mapping[str, object]
    diagnostics: Mapping[str, np.ndarray]
    runtime: Mapping[str, float] = field(default_factory=dict)
