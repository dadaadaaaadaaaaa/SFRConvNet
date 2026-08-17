"""Array-only data contracts shared by MACER coarse-graining schemes.

The coarsening implementations deliberately do not import :class:`GalData`.
Keeping this boundary free of HDF4/HDF5 and training concerns avoids circular
imports and makes every numerical operator independently testable.
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
