"""在多样化 snapshot 样本上审计轴对称质量自适应粗粒化。

审计明确区分以下两种含义不同的质量：

``source_enclosed_mass``
    每个重叠自适应 source 支撑域内的质量。它需要达到 ``mass_filter``，
    允许离散单元造成超调，但不表示互斥分区。

``output_cell_mass``
    沉积到每个规则 CNN 输出格的气体质量。它只要求全局守恒，
    不要求等于 ``mass_filter``。

存储字段 ``gas_energy`` 是气体内能密度，因此审计的广延量是气体内能
``sum(gas_energy * volume)``。解析动能单独报告；Data-Set-10 不是总能量。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import queue
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from raw_data_processing.coarsening_common import (
        temperature_from_internal_energy_density,
    )
    from raw_data_processing.galaxy_data import GalData
except ModuleNotFoundError:  # Direct execution from raw_data_processing/.
    from coarsening_common import temperature_from_internal_energy_density
    from galaxy_data import GalData


CONTINUOUS_SELECTION_FEATURES = (
    "time_rank_within_run",
    "log1p_sfr",
    "log10_total_gas_mass",
    "log10_density_p50_mass_weighted",
    "log10_density_p90_mass_weighted",
    "log10_density_p99_mass_weighted",
    "log10_density_max",
    "cold_gas_mass_fraction",
    "central_gas_mass_fraction",
    "mass_weighted_speed_rms",
    "inflowing_gas_mass_fraction",
    "mass_weighted_cylindrical_radius",
    "mass_weighted_abs_z",
    "meridional_axis_ratio",
)


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _infer_run(path: Path) -> str:
    parent = path.parent
    if parent.name.lower() in {"fine", "processed", "snapshots"}:
        return parent.parent.name
    return parent.name


def _sample_uid(path: Path, run: str) -> str:
    raw = f"{run}|{path.resolve()}".encode("utf-8")
    return f"{run}:{path.stem}:{hashlib.sha1(raw).hexdigest()[:10]}"


def discover_snapshot_paths(inputs: Sequence[str | Path]) -> list[dict[str, str]]:
    """Resolve files/directories into deterministic, de-duplicated descriptors."""
    descriptors: list[dict[str, str]] = []
    seen: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(path.glob("*.h5"))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
        for snapshot_path in paths:
            resolved = snapshot_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = _infer_run(resolved)
            descriptors.append(
                {
                    "path": str(resolved),
                    "run": run,
                    "sample_uid": _sample_uid(resolved, run),
                }
            )
    if not descriptors:
        raise FileNotFoundError("No .h5 snapshots were found in the requested inputs.")
    return descriptors


def load_snapshot_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load an explicit TXT, CSV, or JSON list of snapshots.

    CSV accepts columns ``path`` and optional ``run``.  JSON may be a list of
    path strings, a list of objects, or ``{"snapshots": [...]}``.  Relative
    paths are resolved relative to the manifest file.
    """
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("snapshots")
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list or contain a 'snapshots' list.")
        raw_rows = [({"path": item} if isinstance(item, str) else item) for item in payload]
    elif suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
    else:
        raw_rows = [
            {"path": line.strip()}
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    result: list[dict[str, str]] = []
    seen: set[Path] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or not row.get("path"):
            raise ValueError(f"Manifest row {index + 1} has no path: {row!r}")
        snapshot_path = Path(str(row["path"])).expanduser()
        if not snapshot_path.is_absolute():
            snapshot_path = manifest_path.parent / snapshot_path
        snapshot_path = snapshot_path.resolve()
        if snapshot_path in seen:
            continue
        seen.add(snapshot_path)
        run = str(row.get("run") or _infer_run(snapshot_path))
        result.append(
            {
                "path": str(snapshot_path),
                "run": run,
                "sample_uid": str(row.get("sample_uid") or _sample_uid(snapshot_path, run)),
            }
        )
    if not result:
        raise ValueError(f"Manifest contains no snapshots: {manifest_path}")
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return math.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights, dtype=np.float64)
    threshold = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, threshold, side="left"), values.size - 1)])


def _infer_selection_edges(axis: np.ndarray, *, radial: bool) -> np.ndarray:
    """Infer edges for *selection features only*, never for conservation audit."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
        raise ValueError("Legacy coordinate axes must be increasing 1D arrays.")
    if radial:
        if np.any(axis <= 0):
            raise ValueError("Legacy radial centres must be positive.")
        interior = np.sqrt(axis[:-1] * axis[1:])
        first, last = axis[0] ** 2 / interior[0], axis[-1] ** 2 / interior[-1]
    else:
        interior = 0.5 * (axis[:-1] + axis[1:])
        first = axis[0] - (interior[0] - axis[0])
        last = axis[-1] + (axis[-1] - interior[-1])
    return np.concatenate(([first], interior, [last]))


def _selection_feature_payload(
    snapshot: Mapping[str, np.ndarray],
    axes: Sequence[np.ndarray],
    volume: np.ndarray,
    *,
    time: float,
    sfr_target: float,
    volume_source: str,
    cold_temperature_threshold: float,
    central_radius: float,
) -> dict[str, Any]:
    density = np.asarray(snapshot["density"], dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    mass = density * volume
    total_mass = float(mass.sum(dtype=np.float64))
    if total_mass <= 0 or not math.isfinite(total_mass):
        raise ValueError("Snapshot has no finite positive selection-weight gas mass.")
    # Data-Set-10 与 density 唯一决定温度；不把缓存中的派生温度当成独立状态量。
    temperature = temperature_from_internal_energy_density(
        snapshot["gas_energy"], density
    )
    r, theta = np.meshgrid(*axes)
    cylindrical_r = r * np.sin(theta)
    z = r * np.cos(theta)
    velocity_fields = [
        np.asarray(snapshot.get(name, np.zeros_like(density)), dtype=np.float64)
        for name in ("v1", "v2", "v3")
    ]
    speed2 = sum(np.square(component) for component in velocity_fields)
    vr = velocity_fields[0]
    mean_r = float(np.sum(mass * cylindrical_r) / total_mass)
    mean_abs_z = float(np.sum(mass * np.abs(z)) / total_mass)
    var_r = float(np.sum(mass * np.square(cylindrical_r - mean_r)) / total_mass)
    z_mean = float(np.sum(mass * z) / total_mass)
    var_z = float(np.sum(mass * np.square(z - z_mean)) / total_mass)
    return {
        "selection_status": "ok",
        "selection_volume_source": volume_source,
        "time": _finite_float(time),
        "sfr_target": _finite_float(sfr_target, default=0.0),
        "log1p_sfr": float(np.log1p(max(_finite_float(sfr_target, 0.0), 0.0))),
        "selection_weight_total_gas_mass": total_mass,
        "log10_total_gas_mass": float(np.log10(total_mass)),
        "log10_density_p50_mass_weighted": float(
            np.log10(max(_weighted_quantile(density, mass, 0.50), np.finfo(float).tiny))
        ),
        "log10_density_p90_mass_weighted": float(
            np.log10(max(_weighted_quantile(density, mass, 0.90), np.finfo(float).tiny))
        ),
        "log10_density_p99_mass_weighted": float(
            np.log10(max(_weighted_quantile(density, mass, 0.99), np.finfo(float).tiny))
        ),
        "log10_density_max": float(
            np.log10(max(float(np.max(density)), np.finfo(float).tiny))
        ),
        "cold_gas_mass_fraction": float(
            mass[temperature < cold_temperature_threshold].sum(dtype=np.float64) / total_mass
        ),
        "central_gas_mass_fraction": float(
            mass[r < central_radius].sum(dtype=np.float64) / total_mass
        ),
        "mass_weighted_speed_rms": float(np.sqrt(np.sum(mass * speed2) / total_mass)),
        "inflowing_gas_mass_fraction": float(mass[vr < 0].sum(dtype=np.float64) / total_mass),
        "mass_weighted_cylindrical_radius": mean_r,
        "mass_weighted_abs_z": mean_abs_z,
        "meridional_axis_ratio": float(
            np.sqrt(max(var_z, 0.0) / max(var_r, np.finfo(float).tiny))
        ),
    }


def _characterize_legacy_h5(
    path: Path,
    *,
    cold_temperature_threshold: float,
    central_radius: float,
) -> dict[str, Any]:
    """Read legacy HDF5 for sample selection without pretending it is audit-ready."""
    import h5py

    with h5py.File(path, "r") as handle:
        if "snapshot" not in handle or "coordinates" not in handle:
            raise ValueError("Legacy file has no snapshot/coordinates groups.")
        snapshot = {name: dataset[()] for name, dataset in handle["snapshot"].items()}
        axes = (
            np.asarray(handle["coordinates/axis_0"][()], dtype=np.float64),
            np.asarray(handle["coordinates/axis_1"][()], dtype=np.float64),
        )
        radial_edges = _infer_selection_edges(axes[0], radial=True)
        theta_edges = _infer_selection_edges(axes[1], radial=False)
        # Approximate geometry is permitted only for diverse-sample selection.
        # The conservation/coarsening worker will refuse this legacy file.
        volume = (
            (2.0 * np.pi / 3.0)
            * (radial_edges[1:] ** 3 - radial_edges[:-1] ** 3)[None, :]
            * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))[:, None]
        )
        time = handle.attrs.get("time", math.nan)
        sfr_target = 0.0
        if "log_data/sfr_target" in handle:
            sfr_target = handle["log_data/sfr_target"][()]
        elif "log_data/sfr" in handle:
            sfr_target = handle["log_data/sfr"][()]
        payload = _selection_feature_payload(
            snapshot,
            axes,
            volume,
            time=float(time),
            sfr_target=float(sfr_target),
            volume_source="approximate_inferred_axisymmetric_volume_for_selection_only",
            cold_temperature_threshold=cold_temperature_threshold,
            central_radius=central_radius,
        )
        payload.update(
            {
                "processed_format_version": int(handle.attrs.get("format_version", 0)),
                "audit_ready": False,
                "rebuild_required": True,
                "audit_readiness_reason": "missing authoritative geometry/cell_volume and/or coordinate edges",
            }
        )
        return payload


def characterize_snapshot(
    descriptor: Mapping[str, str],
    *,
    cold_temperature_threshold: float,
    central_radius: float,
) -> dict[str, Any]:
    """Extract inexpensive selection features without applying a coarsener."""
    record: dict[str, Any] = dict(descriptor)
    path = Path(descriptor["path"])
    try:
        try:
            data = GalData().load_h5(path)
            record.update(
                _selection_feature_payload(
                    data.snapshot,
                    data.raw_axes,
                    data.grid_volume,
                    time=data.time,
                    sfr_target=data.sfr_target,
                    volume_source="authoritative_Data-Set-15",
                    cold_temperature_threshold=cold_temperature_threshold,
                    central_radius=central_radius,
                )
            )
            record.update(
                {
                    "processed_format_version": 4,
                    "audit_ready": True,
                    "rebuild_required": False,
                    "audit_readiness_reason": "authoritative volume and stored coordinate edges available",
                }
            )
        except Exception as modern_error:
            record.update(
                _characterize_legacy_h5(
                    path,
                    cold_temperature_threshold=cold_temperature_threshold,
                    central_radius=central_radius,
                )
            )
            record["modern_load_error"] = f"{type(modern_error).__name__}: {modern_error}"
    except Exception as exc:
        record.update(
            {
                "selection_status": "failed",
                "selection_error_type": type(exc).__name__,
                "selection_error_message": str(exc),
            }
        )
    return record


def _rank_within_groups(records: list[dict[str, Any]], key: str, output_key: str) -> None:
    runs = sorted({str(record["run"]) for record in records})
    for run in runs:
        indices = [i for i, record in enumerate(records) if str(record["run"]) == run]
        ordered = sorted(indices, key=lambda i: (_finite_float(records[i].get(key)), records[i]["path"]))
        denominator = max(1, len(ordered) - 1)
        for rank, index in enumerate(ordered):
            records[index][output_key] = rank / denominator


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return np.minimum(n_bins - 1, (ranks * n_bins) // max(1, len(values)))


def _robust_standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    median = np.nanmedian(matrix, axis=0)
    filled = np.where(np.isfinite(matrix), matrix, median)
    q25, q75 = np.nanpercentile(filled, [25, 75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(filled, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(fallback > 1e-12, fallback, 1.0))
    return (filled - median) / scale


def select_diverse_snapshots(
    records: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int = 42,
    n_time_bins: int = 4,
    n_sfr_bins: int = 4,
) -> list[dict[str, Any]]:
    """Run-aware farthest-point sampling over physical and stratification features."""
    valid = [dict(record) for record in records if record.get("selection_status", "ok") == "ok"]
    if not valid:
        raise ValueError("No valid characterized snapshots are available for selection.")
    if count <= 0:
        raise ValueError("count must be positive.")
    if count >= len(valid):
        return sorted(valid, key=lambda record: (record["run"], record["path"]))

    _rank_within_groups(valid, "time", "time_rank_within_run")
    time_bins = np.minimum(
        n_time_bins - 1,
        (np.asarray([r["time_rank_within_run"] for r in valid]) * n_time_bins).astype(int),
    )
    sfr_bins = _quantile_bin(np.asarray([r["log1p_sfr"] for r in valid]), n_sfr_bins)
    runs = sorted({str(record["run"]) for record in valid})
    run_to_index = {run: index for index, run in enumerate(runs)}
    continuous = _robust_standardize(
        np.asarray([[record.get(key, math.nan) for key in CONTINUOUS_SELECTION_FEATURES] for record in valid])
    )
    run_one_hot = np.zeros((len(valid), len(runs)), dtype=np.float64)
    time_one_hot = np.zeros((len(valid), n_time_bins), dtype=np.float64)
    sfr_one_hot = np.zeros((len(valid), n_sfr_bins), dtype=np.float64)
    for index, record in enumerate(valid):
        run_one_hot[index, run_to_index[str(record["run"])]] = 2.0
        time_one_hot[index, time_bins[index]] = 1.5
        sfr_one_hot[index, sfr_bins[index]] = 1.5
        record["time_quantile_bin"] = int(time_bins[index])
        record["sfr_quantile_bin"] = int(sfr_bins[index])
    features = np.concatenate((continuous, run_one_hot, time_one_hot, sfr_one_hot), axis=1)

    rng = np.random.default_rng(seed)
    tie_noise = rng.uniform(0.0, 1e-12, size=len(valid))
    selected: list[int] = []
    # Guarantee run coverage first, choosing a physically extreme representative
    # from each run.  Remaining points are standard farthest-point samples.
    for run in runs:
        candidates = np.asarray([i for i, record in enumerate(valid) if str(record["run"]) == run])
        if len(selected) >= count:
            break
        norms = np.sum(np.square(features[candidates]), axis=1) + tie_noise[candidates]
        selected.append(int(candidates[int(np.argmax(norms))]))
    min_distance2 = np.full(len(valid), np.inf, dtype=np.float64)
    for index in selected:
        min_distance2 = np.minimum(
            min_distance2, np.sum(np.square(features - features[index]), axis=1)
        )
    min_distance2[selected] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(min_distance2 + tie_noise))
        selected.append(next_index)
        min_distance2 = np.minimum(
            min_distance2, np.sum(np.square(features - features[next_index]), axis=1)
        )
        min_distance2[selected] = -np.inf

    output = [valid[index] for index in selected]
    for order, record in enumerate(output):
        record["selection_order"] = order
        record["selection_method"] = "run_seeded_stratified_farthest_point"
        record["selection_seed"] = seed
    return output


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any(values < 0):
        return math.nan
    total = values.sum(dtype=np.float64)
    if total <= 0:
        return 0.0
    ordered = np.sort(values)
    indices = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum((2.0 * indices - ordered.size - 1.0) * ordered) / (ordered.size * total))


def _distribution_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_{name}": math.nan for name in ("min", "p05", "median", "mean", "p95", "max", "std", "cv", "gini")}
    mean = float(np.mean(finite))
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_p05": float(np.quantile(finite, 0.05)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_mean": mean,
        f"{prefix}_p95": float(np.quantile(finite, 0.95)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_cv": float(np.std(finite) / abs(mean)) if mean != 0 else math.nan,
        f"{prefix}_gini": _gini(finite),
    }


def _extensive_state(data: GalData) -> dict[str, Any]:
    snapshot = data.snapshot
    volume = np.asarray(data.grid_volume, dtype=np.float64)
    density = np.asarray(snapshot["density"], dtype=np.float64)
    mass = density * volume
    r, theta = np.meshgrid(*data.raw_axes)
    zero = np.zeros_like(mass)
    vr = np.asarray(snapshot.get("v1", zero), dtype=np.float64)
    vtheta = np.asarray(snapshot.get("v2", zero), dtype=np.float64)
    vphi = np.asarray(snapshot.get("v3", zero), dtype=np.float64)
    numeric_momenta = {name: mass * velocity for name, velocity in zip(("v1", "v2", "v3"), (vr, vtheta, vphi))}
    pz_terms = mass * (vr * np.cos(theta) - vtheta * np.sin(theta))
    lz_terms = mass * (r * np.sin(theta)) * vphi
    resolved_kinetic_terms = 0.5 * mass * (vr * vr + vtheta * vtheta + vphi * vphi)
    state = {
        "volume_terms": volume,
        "gas_mass_terms": mass,
        "stored_internal_energy_terms": np.asarray(snapshot["gas_energy"], dtype=np.float64) * volume,
        "pz_terms": pz_terms,
        "lz_terms": lz_terms,
        "resolved_kinetic_terms": resolved_kinetic_terms,
        **{f"numeric_{name}_momentum_terms": terms for name, terms in numeric_momenta.items()},
    }
    if "initial_stellar_density" in snapshot:
        state["initial_stellar_mass_terms"] = (
            np.asarray(snapshot["initial_stellar_density"], dtype=np.float64) * volume
        )
    if "cooling_rate_density" in snapshot:
        state["cooling_power_terms"] = (
            np.asarray(snapshot["cooling_rate_density"], dtype=np.float64) * volume
        )
    if {"radial_acceleration", "theta_acceleration"}.issubset(snapshot):
        acceleration_r = np.asarray(snapshot["radial_acceleration"], dtype=np.float64)
        acceleration_theta = np.asarray(
            snapshot["theta_acceleration"], dtype=np.float64
        )
        state["gravitational_force_R_terms"] = mass * (
            acceleration_r * np.sin(theta) + acceleration_theta * np.cos(theta)
        )
        state["gravitational_force_z_terms"] = mass * (
            acceleration_r * np.cos(theta) - acceleration_theta * np.sin(theta)
        )
    return state


def _comparison(prefix: str, fine_terms: np.ndarray, coarse_terms: np.ndarray, atol: float) -> dict[str, float]:
    fine_total = float(np.sum(fine_terms, dtype=np.float64))
    coarse_total = float(np.sum(coarse_terms, dtype=np.float64))
    delta = coarse_total - fine_total
    scale = max(float(np.sum(np.abs(fine_terms), dtype=np.float64)), float(atol))
    return {
        f"{prefix}_fine": fine_total,
        f"{prefix}_coarse": coarse_total,
        f"{prefix}_delta": delta,
        f"{prefix}_scaled_relative_error": abs(delta) / scale,
        f"{prefix}_reference_l1_scale": scale,
    }


def _connected_components(mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask)
    components = 0
    largest = 0
    for row, column in zip(*np.nonzero(mask)):
        if visited[row, column]:
            continue
        components += 1
        size = 0
        stack = [(int(row), int(column))]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            size += 1
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and mask[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        largest = max(largest, size)
    return components, largest


def _dominant_assignment_connectivity(
    dominant: np.ndarray | None,
    output_shape: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if dominant is None:
        return {}, []
    dominant = np.asarray(dominant)
    per_cell: list[dict[str, Any]] = []
    disconnected = 0
    empty = 0
    component_total = 0
    largest_fractions: list[float] = []
    for destination in range(int(np.prod(output_shape))):
        mask = dominant == destination
        assigned = int(np.count_nonzero(mask))
        components, largest = _connected_components(mask)
        largest_fraction = largest / assigned if assigned else 0.0
        if assigned == 0:
            empty += 1
        if components > 1:
            disconnected += 1
        component_total += components
        largest_fractions.append(largest_fraction)
        per_cell.append(
            {
                "dominant_assigned_source_count": assigned,
                "dominant_component_count": components,
                "dominant_largest_component_fraction": largest_fraction,
            }
        )
    summary = {
        "connectivity_semantics": "4-neighbor connectivity of hard argmax destination; soft supports overlap",
        "dominant_partition_component_total": component_total,
        "dominant_partition_disconnected_count": disconnected,
        "dominant_partition_empty_count": empty,
        "dominant_partition_largest_component_fraction_min": min(largest_fractions),
        "dominant_partition_largest_component_fraction_mean": float(np.mean(largest_fractions)),
    }
    return summary, per_cell


def audit_galdata(
    fine: GalData,
    *,
    operator: str,
    new_shape: tuple[int, int],
    mass_filter: float,
    operator_kwargs: Mapping[str, Any] | None = None,
    conservation_rtol: float = 1e-6,
    conservation_atol: float = 1e-10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply one operator and return one summary plus per-output-cell records."""
    kwargs = dict(operator_kwargs or {})
    kwargs["mass_filter"] = float(mass_filter)
    started = perf_counter()
    coarse = fine.coarsen(
        operator,
        tuple(new_shape),
        conservation_rtol=conservation_rtol,
        conservation_atol=conservation_atol,
        **kwargs,
    )
    wall_seconds = perf_counter() - started
    metadata = coarse.coarsening_metadata
    diagnostics = coarse.coarsening_diagnostics
    summary: dict[str, Any] = {
        "status": "ok",
        "operator_requested": operator,
        "operator": metadata.get("operator", operator),
        "operator_version": metadata.get("operator_version", ""),
        "output_shape_theta": int(new_shape[0]),
        "output_shape_r": int(new_shape[1]),
        "mass_filter": float(mass_filter),
        "source_enclosed_mass_semantics": "overlapping adaptive support mass; thresholded by mass_filter; not a partition mass",
        "output_cell_mass_semantics": "regular CNN grid gas mass; globally conserved; not constrained to mass_filter",
        "runtime_wall_seconds": wall_seconds,
        "runtime_operator_total_seconds": metadata.get("runtime_total_seconds", math.nan),
        "runtime_neighbor_kernel_seconds": metadata.get("runtime_neighbor_kernel_seconds", math.nan),
        "runtime_neighbor_graph_seconds": metadata.get("runtime_neighbor_graph_seconds", math.nan),
        "runtime_balance_seconds": metadata.get("runtime_balance_seconds", math.nan),
        "balance_error": metadata.get("transport_balance_error", math.nan),
        "balance_tolerance": metadata.get("transport_balance_tolerance", math.nan),
        "balance_iterations": metadata.get("transport_balance_iterations", math.nan),
        "kernel_nnz": metadata.get("kernel_nnz", math.nan),
        "kernel": metadata.get("kernel", ""),
    }
    for name in (
        "knn_backend",
        "knn_cache_hit",
        "knn_cached_k",
        "knn_max_row_k",
        "neighbor_expansions",
        "graph_repair_rounds",
        "graph_initial_cross_scc_edges",
        "graph_pruned_cross_scc_edges",
        "graph_pruned_kernel_weight",
        "graph_scc_count",
        "kernel_nnz_before_prune",
        "lbfgs_iterations",
        "ipfp_polish_iterations",
        "lbfgs_success",
        "lbfgs_message",
    ):
        if name in metadata:
            summary[name] = metadata[name]

    fine_state, coarse_state = _extensive_state(fine), _extensive_state(coarse)
    required_stored = ["volume", "gas_mass", "stored_internal_energy"]
    for name in ("numeric_v1_momentum", "numeric_v2_momentum", "numeric_v3_momentum"):
        if f"{name}_terms" in fine_state:
            required_stored.append(name)
    if "initial_stellar_mass_terms" in fine_state:
        required_stored.append("initial_stellar_mass")
    for name in (
        "cooling_power",
        "gravitational_force_R",
        "gravitational_force_z",
    ):
        if f"{name}_terms" in fine_state:
            required_stored.append(name)
    for name in required_stored + ["pz", "lz"]:
        fine_key = f"{name}_terms"
        if fine_key in fine_state and fine_key in coarse_state:
            summary.update(_comparison(name, fine_state[fine_key], coarse_state[fine_key], conservation_atol))

    fine_kinetic = float(np.sum(fine_state["resolved_kinetic_terms"], dtype=np.float64))
    coarse_kinetic = float(np.sum(coarse_state["resolved_kinetic_terms"], dtype=np.float64))
    kinetic_loss = fine_kinetic - coarse_kinetic
    summary.update(
        {
            "resolved_kinetic_energy_fine": fine_kinetic,
            "resolved_kinetic_energy_coarse": coarse_kinetic,
            "resolved_kinetic_energy_loss": kinetic_loss,
            "resolved_kinetic_energy_loss_fraction": kinetic_loss / fine_kinetic if fine_kinetic > 0 else math.nan,
            "energy_statement": "gas internal energy = sum(gas internal-energy density * cell volume); Data-Set-10 is not total energy",
        }
    )
    stored_pass = True
    stored_scaled_errors: list[float] = []
    for name in required_stored:
        scale = summary[f"{name}_reference_l1_scale"]
        delta = abs(summary[f"{name}_delta"])
        stored_scaled_errors.append(float(summary[f"{name}_scaled_relative_error"]))
        stored_pass &= delta <= conservation_atol + conservation_rtol * scale
    summary["stored_extensive_conservation_pass"] = bool(stored_pass)
    summary["epsilon_conservation_max"] = max(stored_scaled_errors, default=math.nan)
    summary["epsilon_balance"] = summary["balance_error"]
    summary["n_balance_iterations"] = summary["balance_iterations"]

    source_enclosed = diagnostics.get("source_enclosed_mass")
    if source_enclosed is not None:
        source_enclosed = np.asarray(source_enclosed, dtype=np.float64)
        summary.update(_distribution_stats("source_enclosed_mass", source_enclosed))
        source_ratio = source_enclosed / mass_filter
        summary.update(_distribution_stats("source_enclosed_mass_to_filter", source_ratio))
        summary["source_enclosed_mass_below_filter_count"] = int(
            np.count_nonzero(source_enclosed < mass_filter * (1.0 - 1e-12))
        )

    output_mass = np.asarray(coarse.snapshot["density"], dtype=np.float64) * np.asarray(
        coarse.grid_volume, dtype=np.float64
    )
    summary.update(_distribution_stats("output_cell_mass", output_mass))
    output_ratio = output_mass / mass_filter
    summary.update(_distribution_stats("output_cell_mass_to_filter", output_ratio))
    summary["output_cell_mass_ratio_rmse_from_one"] = float(
        np.sqrt(np.mean(np.square(output_ratio - 1.0)))
    )
    summary["output_cell_mass_ratio_mean_absolute_deviation_from_one"] = float(
        np.mean(np.abs(output_ratio - 1.0))
    )
    for diagnostic_name, output_prefix in (
        ("effective_support_scale", "h_eff"),
        ("effective_source_count", "n_eff"),
    ):
        if diagnostic_name in diagnostics:
            summary.update(_distribution_stats(output_prefix, diagnostics[diagnostic_name]))

    connectivity_summary, connectivity_cells = _dominant_assignment_connectivity(
        diagnostics.get("dominant_destination"), tuple(new_shape)
    )
    summary.update(connectivity_summary)
    h_eff = np.asarray(diagnostics.get("effective_support_scale", np.full(new_shape, np.nan)))
    n_eff = np.asarray(diagnostics.get("effective_source_count", np.full(new_shape, np.nan)))
    cells: list[dict[str, Any]] = []
    for row in range(new_shape[0]):
        for column in range(new_shape[1]):
            flat_index = row * new_shape[1] + column
            cell = {
                "output_cell_index": flat_index,
                "output_theta_index": row,
                "output_r_index": column,
                "output_cell_mass": float(output_mass[row, column]),
                "output_cell_mass_to_filter": float(output_ratio[row, column]),
                "effective_support_scale": float(h_eff[row, column]),
                "effective_source_count": float(n_eff[row, column]),
                "mass_semantics": "regular output cell mass, not fixed-mass support",
            }
            if flat_index < len(connectivity_cells):
                cell.update(connectivity_cells[flat_index])
            cells.append(cell)
    return summary, cells


def _audit_worker(payload: Mapping[str, Any], result_queue) -> None:
    descriptor = dict(payload["descriptor"])
    try:
        fine = GalData().load_h5(descriptor["path"])
        summary, cells = audit_galdata(
            fine,
            operator=payload["operator"],
            new_shape=tuple(payload["new_shape"]),
            mass_filter=payload["mass_filter"],
            operator_kwargs=payload["operator_kwargs"],
            conservation_rtol=payload["conservation_rtol"],
            conservation_atol=payload["conservation_atol"],
        )
        result_queue.put({"summary": summary, "cells": cells})
    except BaseException as exc:  # Worker must report numerical failures and continue the audit.
        result_queue.put(
            {
                "summary": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                },
                "cells": [],
            }
        )


def run_snapshot_with_timeout(payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    """Run one snapshot in a killable process and return failure instead of aborting."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_audit_worker, args=(dict(payload), result_queue))
    started = perf_counter()
    process.start()
    # Read before join: a worker result containing 64 per-cell records can
    # exceed the OS pipe buffer. Joining first lets the child block forever in
    # Queue.put(), which falsely turns every successful audit into a timeout.
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        elapsed = perf_counter() - started
        process.terminate()
        process.join(5.0)
        result_queue.close()
        return {
            "summary": {
                "status": "timeout",
                "error_type": "SnapshotTimeout",
                "error_message": f"Exceeded per-snapshot timeout of {timeout_seconds:g} seconds.",
                "runtime_wall_seconds": elapsed,
            },
            "cells": [],
        }
    process.join(5.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
    result_queue.close()
    return result


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _powershell_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_rebuild_artifacts(
    output_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    *,
    rebuilt_root: Path,
    preprocess_python: str,
) -> list[dict[str, Any]]:
    """Write a safe rebuild script and a manifest pointing at future v4 files."""
    rebuilt_manifest: list[dict[str, Any]] = []
    commands = [
        "$ErrorActionPreference = 'Stop'",
        "# Generated by audit_mass_coarsening.py; existing files are not overwritten.",
    ]
    for record in selected:
        rebuilt = dict(record)
        if bool(record.get("audit_ready", False)):
            rebuilt_manifest.append(
                {"path": record["path"], "run": record["run"], "sample_uid": record["sample_uid"]}
            )
            continue
        source_path = Path(str(record["path"]))
        try:
            snapshot_index = int(source_path.stem)
        except ValueError:
            rebuilt["rebuild_error"] = "snapshot filename stem is not an integer"
            rebuilt_manifest.append(rebuilt)
            continue
        destination = (rebuilt_root / str(record["run"]) / source_path.name).resolve()
        command = (
            f"& {_powershell_quote(preprocess_python)} "
            f"{_powershell_quote('raw_data_processing/rebuild_fine_snapshot.py')} "
            f"--galaxy {_powershell_quote(record['run'])} "
            f"--snapshot-index {snapshot_index} --output {_powershell_quote(destination)}"
        )
        commands.append(command)
        rebuilt_manifest.append(
            {
                "path": str(destination),
                "run": record["run"],
                "sample_uid": record["sample_uid"],
                "legacy_source_path": record["path"],
            }
        )
    script_path = output_dir / "rebuild_selected_snapshots.ps1"
    script_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    _write_csv(output_dir / "selected_rebuilt_manifest.csv", rebuilt_manifest)
    return rebuilt_manifest


def _parse_operator_kwargs(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--operator-kwargs-json must decode to a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a reproducible, physically diverse fine-snapshot sample and audit "
            "axisymmetric Rm-v0/v1 conservation, transport, mass dispersion, and connectivity."
        )
    )
    parser.add_argument("--input", nargs="*", default=[], help="One or more fine HDF5 files/directories.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Explicit TXT/CSV/JSON snapshot list. This bypasses automatic 30-100 selection.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for audit CSV files.")
    parser.add_argument(
        "--sample-count", type=int, default=60,
        help="Automatic diverse sample size; must be between 30 and 100 (default: 60).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic selection seed.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Optional sorted candidate cap for a smoke run.")
    parser.add_argument(
        "--operator", action="append", choices=("axisymmetric_rm_v0", "axisymmetric_rm_v1"),
        help="Operator to audit; repeat to audit both. Default: axisymmetric_rm_v0.",
    )
    parser.add_argument("--shape", nargs=2, type=int, default=(8, 8), metavar=("N_THETA", "N_R"))
    parser.add_argument("--mass-filter", required=True, type=float, help="Fixed support-mass budget in density*volume units.")
    parser.add_argument("--kernel", default=None, help="Explicit kernel override; v0 defaults to top_hat, v1 uses its own default.")
    parser.add_argument("--initial-neighbors", type=int, default=None)
    parser.add_argument("--balance-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--balance-max-iterations", type=int, default=30000,
        help="Maximum v0 IPFP/Sinkhorn iterations (not passed to v1).",
    )
    parser.add_argument("--v1-solver-max-iterations", type=int, default=None)
    parser.add_argument("--v1-polish-max-iterations", type=int, default=None)
    parser.add_argument(
        "--v1-max-runtime-seconds", type=float, default=None,
        help="Internal v1 deadline; defaults below the outer per-snapshot timeout.",
    )
    parser.add_argument(
        "--operator-kwargs-json", default=None,
        help="Additional operator kwargs as inline JSON or a JSON-file path.",
    )
    parser.add_argument("--conservation-rtol", type=float, default=1e-6)
    parser.add_argument("--conservation-atol", type=float, default=1e-10)
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Kill one stalled snapshot/operator job after this time.")
    parser.add_argument("--cold-temperature-threshold", type=float, default=2.0e4)
    parser.add_argument("--central-radius", type=float, default=1.0)
    parser.add_argument("--selection-only", action="store_true", help="Write selection tables without applying a coarsener.")
    parser.add_argument(
        "--rebuilt-fine-root",
        type=Path,
        default=None,
        help="Destination root used in the generated legacy-to-v4 rebuild manifest.",
    )
    parser.add_argument(
        "--preprocess-python",
        default=sys.executable,
        help="Python executable written into rebuild_selected_snapshots.ps1 (must provide pyhdf).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input and args.manifest is None:
        parser.error("Provide --input and/or --manifest.")
    if args.manifest is None and not 30 <= args.sample_count <= 100:
        parser.error("Automatic --sample-count must be between 30 and 100.")
    if args.mass_filter <= 0 or not math.isfinite(args.mass_filter):
        parser.error("--mass-filter must be positive and finite.")
    if any(value <= 0 for value in args.shape):
        parser.error("--shape values must be positive.")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if args.balance_max_iterations <= 0:
        parser.error("--balance-max-iterations must be positive.")
    for option_name in ("v1_solver_max_iterations", "v1_polish_max_iterations"):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be positive.")
    if args.v1_max_runtime_seconds is not None and args.v1_max_runtime_seconds <= 0:
        parser.error("--v1-max-runtime-seconds must be positive.")
    if args.candidate_limit is not None and args.candidate_limit <= 0:
        parser.error("--candidate-limit must be positive.")

    operators = args.operator or ["axisymmetric_rm_v0"]
    descriptors = (
        load_snapshot_manifest(args.manifest)
        if args.manifest is not None
        else discover_snapshot_paths(args.input)
    )
    if args.candidate_limit is not None:
        descriptors = descriptors[: args.candidate_limit]
    characterized = [
        characterize_snapshot(
            descriptor,
            cold_temperature_threshold=args.cold_temperature_threshold,
            central_radius=args.central_radius,
        )
        for descriptor in descriptors
    ]
    if args.manifest is None:
        selected = select_diverse_snapshots(characterized, args.sample_count, seed=args.seed)
    else:
        selected = characterized
        for index, record in enumerate(selected):
            record["selection_order"] = index
            record["selection_method"] = "explicit_manifest"
            record["selection_seed"] = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "selection_candidates.csv", characterized)
    _write_csv(args.output_dir / "selected_snapshots.csv", selected)
    rebuilt_root = (args.rebuilt_fine_root or (args.output_dir / "fine_v4")).resolve()
    write_rebuild_artifacts(
        args.output_dir,
        selected,
        rebuilt_root=rebuilt_root,
        preprocess_python=args.preprocess_python,
    )
    rebuild_count = sum(not bool(record.get("audit_ready", False)) for record in selected)
    if rebuild_count:
        print(
            f"{rebuild_count}/{len(selected)} selected snapshots are legacy files and cannot be "
            "conservation-audited yet. See rebuild_selected_snapshots.ps1 and "
            "selected_rebuilt_manifest.csv."
        )
    if args.selection_only:
        print(f"Selected {len(selected)} snapshots; selection-only audit complete.")
        return 0

    base_kwargs = _parse_operator_kwargs(args.operator_kwargs_json)
    base_kwargs["balance_tolerance"] = args.balance_tolerance

    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    total_jobs = len(selected) * len(operators)
    job_index = 0
    for descriptor in selected:
        identity = {
            key: descriptor.get(key, "")
            for key in ("sample_uid", "run", "path", "time", "sfr_target", "selection_order")
        }
        for operator in operators:
            job_index += 1
            if not bool(descriptor.get("audit_ready", False)):
                summary_rows.append(
                    {
                        **identity,
                        "operator_requested": operator,
                        "status": "rebuild_required",
                        "error_type": "LegacyProcessedHDF5",
                        "error_message": (
                            "Authoritative Data-Set-15 volume and coordinate edges are absent. "
                            "Run rebuild_selected_snapshots.ps1, then audit "
                            "selected_rebuilt_manifest.csv."
                        ),
                    }
                )
                _write_csv(args.output_dir / "audit_summary.csv", summary_rows)
                _write_csv(args.output_dir / "audit_output_cells.csv", cell_rows)
                print(f"[{job_index}/{total_jobs}] {identity['sample_uid']} {operator}: rebuild_required")
                continue
            operator_kwargs = dict(base_kwargs)
            if args.initial_neighbors is not None:
                operator_kwargs["initial_neighbors"] = args.initial_neighbors
            if operator == "axisymmetric_rm_v0":
                operator_kwargs["balance_max_iterations"] = args.balance_max_iterations
                operator_kwargs["kernel"] = args.kernel or "top_hat"
            else:
                if args.v1_solver_max_iterations is not None:
                    operator_kwargs["solver_max_iterations"] = args.v1_solver_max_iterations
                if args.v1_polish_max_iterations is not None:
                    operator_kwargs["polish_max_iterations"] = args.v1_polish_max_iterations
                operator_kwargs.setdefault(
                    "max_runtime_seconds",
                    args.v1_max_runtime_seconds
                    if args.v1_max_runtime_seconds is not None
                    else max(1.0, 0.85 * args.timeout_seconds),
                )
            payload = {
                "descriptor": descriptor,
                "operator": operator,
                "new_shape": tuple(args.shape),
                "mass_filter": args.mass_filter,
                "operator_kwargs": operator_kwargs,
                "conservation_rtol": args.conservation_rtol,
                "conservation_atol": args.conservation_atol,
            }
            print(f"[{job_index}/{total_jobs}] {identity['sample_uid']} {operator}", flush=True)
            result = run_snapshot_with_timeout(payload, args.timeout_seconds)
            summary = {**identity, "operator_requested": operator, **result["summary"]}
            summary_rows.append(summary)
            for cell in result["cells"]:
                cell_rows.append({**identity, "operator_requested": operator, **cell})
            # Checkpoint after every job so interruption does not discard completed work.
            _write_csv(args.output_dir / "audit_summary.csv", summary_rows)
            _write_csv(args.output_dir / "audit_output_cells.csv", cell_rows)

    succeeded = sum(row.get("status") == "ok" for row in summary_rows)
    timed_out = sum(row.get("status") == "timeout" for row in summary_rows)
    failed = total_jobs - succeeded - timed_out
    print(f"Audit complete: jobs={total_jobs}, ok={succeeded}, timeout={timed_out}, failed={failed}")
    print(f"Summary: {args.output_dir / 'audit_summary.csv'}")
    print(f"Per-cell masses: {args.output_dir / 'audit_output_cells.csv'}")
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
