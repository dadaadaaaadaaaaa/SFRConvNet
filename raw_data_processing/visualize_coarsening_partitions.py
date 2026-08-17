"""Visual diagnostics for mass-adaptive coarse graining.

The ``dominant_destination`` image produced here is a *pseudo-partition*: it
hardens the largest weight of an overlapping soft transport plan after the
fact.  It is not an exclusive physical tessellation and its component masses
are not expected to equal ``mass_filter``.  Four-colour rendering is used only
to distinguish adjacent connected components.  If DSATUR requests more than
four colours, the tool reports the failure and does not fabricate a four-colour
map.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PSEUDO_PARTITION_WARNING = (
    "dominant_destination is an argmax hardening of overlapping soft transport; "
    "it is not an exclusive physical equal-mass partition"
)


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: float("nan") for name in ("min", "q05", "median", "q95", "max", "mean")}
    return {
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _read_inputs(fine_path: Path, coarse_path: Path):
    with h5py.File(fine_path, "r") as fine_file:
        if "geometry/cell_volume" not in fine_file or "snapshot/density" not in fine_file:
            raise ValueError("Fine HDF5 must contain geometry/cell_volume and snapshot/density.")
        fine_volume = np.asarray(fine_file["geometry/cell_volume"][()], dtype=np.float64)
        density = np.asarray(fine_file["snapshot/density"][()], dtype=np.float64)
        if fine_volume.shape != density.shape:
            raise ValueError("Fine density and cell_volume shapes differ.")
        if np.any(fine_volume <= 0) or np.any(density < 0):
            raise ValueError("Fine volumes must be positive and density must be non-negative.")
        fine_mass = density * fine_volume

    with h5py.File(coarse_path, "r") as coarse_file:
        if "coarsening" not in coarse_file:
            raise ValueError("Coarse HDF5 has no coarsening diagnostics group.")
        group = coarse_file["coarsening"]
        metadata = {key: _decode(value) for key, value in group.attrs.items()}
        diagnostics = {name: np.asarray(group[name][()]) for name in group}
        if "snapshot/density" not in coarse_file or "geometry/cell_volume" not in coarse_file:
            raise ValueError("Coarse HDF5 must contain density and cell_volume.")
        coarse_shape = tuple(int(value) for value in coarse_file["snapshot/density"].shape)
        coarse_mass = (
            np.asarray(coarse_file["snapshot/density"][()], dtype=np.float64)
            * np.asarray(coarse_file["geometry/cell_volume"][()], dtype=np.float64)
        )
    return fine_mass, coarse_mass, coarse_shape, metadata, diagnostics


def connected_components_four_neighbor(
    labels: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[set[int]]]:
    """Split each pseudo-label into four-neighbour connected components."""

    labels = np.asarray(labels)
    mass = np.asarray(mass, dtype=np.float64)
    if labels.ndim != 2 or labels.shape != mass.shape:
        raise ValueError("labels and mass must be two-dimensional arrays with equal shape.")
    if np.any(~np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError("Component masses must be finite and non-negative.")

    n_theta, n_radius = labels.shape
    component_map = np.full(labels.shape, -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for row in range(n_theta):
        for column in range(n_radius):
            if component_map[row, column] >= 0:
                continue
            component_id = len(components)
            destination = int(labels[row, column])
            queue = deque([(row, column)])
            component_map[row, column] = component_id
            cells: list[tuple[int, int]] = []
            while queue:
                current_row, current_column = queue.popleft()
                cells.append((current_row, current_column))
                for delta_row, delta_column in directions:
                    neighbor_row = current_row + delta_row
                    neighbor_column = current_column + delta_column
                    if not (0 <= neighbor_row < n_theta and 0 <= neighbor_column < n_radius):
                        continue
                    if component_map[neighbor_row, neighbor_column] >= 0:
                        continue
                    if int(labels[neighbor_row, neighbor_column]) != destination:
                        continue
                    component_map[neighbor_row, neighbor_column] = component_id
                    queue.append((neighbor_row, neighbor_column))

            perimeter = 0
            component_mass = 0.0
            for cell_row, cell_column in cells:
                component_mass += float(mass[cell_row, cell_column])
                for delta_row, delta_column in directions:
                    neighbor_row = cell_row + delta_row
                    neighbor_column = cell_column + delta_column
                    if not (0 <= neighbor_row < n_theta and 0 <= neighbor_column < n_radius):
                        perimeter += 1
                    elif component_map[neighbor_row, neighbor_column] != component_id:
                        # A same-label neighbour not visited yet is necessarily
                        # queued in this component, so checking the source label
                        # avoids counting its edge as a perimeter edge.
                        if int(labels[neighbor_row, neighbor_column]) != destination:
                            perimeter += 1
            cell_count = len(cells)
            compactness = (
                float(4.0 * np.pi * cell_count / perimeter**2) if perimeter > 0 else 0.0
            )
            components.append(
                {
                    "component_id": component_id,
                    "destination": destination,
                    "cell_count": cell_count,
                    "mass": component_mass,
                    "perimeter_edges_index_space": perimeter,
                    "compactness_index_4pi_area_over_perimeter2": compactness,
                }
            )

    adjacency = [set() for _ in components]
    for row in range(n_theta):
        for column in range(n_radius):
            current = int(component_map[row, column])
            for delta_row, delta_column in ((1, 0), (0, 1)):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if neighbor_row >= n_theta or neighbor_column >= n_radius:
                    continue
                neighbor = int(component_map[neighbor_row, neighbor_column])
                if current != neighbor:
                    adjacency[current].add(neighbor)
                    adjacency[neighbor].add(current)
    return component_map, components, adjacency


def dsatur_greedy_four_coloring(adjacency: list[set[int]]) -> dict[str, Any]:
    """Run deterministic DSATUR and explicitly report use of more than 4 colours."""

    count = len(adjacency)
    colours = np.full(count, -1, dtype=np.int16)
    degrees = np.asarray([len(neighbours) for neighbours in adjacency], dtype=np.int32)
    for _ in range(count):
        uncoloured = np.flatnonzero(colours < 0)
        if uncoloured.size == 0:
            break
        best = max(
            uncoloured.tolist(),
            key=lambda node: (
                len({int(colours[n]) for n in adjacency[node] if colours[n] >= 0}),
                int(degrees[node]),
                -int(node),
            ),
        )
        used = {int(colours[neighbor]) for neighbor in adjacency[best] if colours[neighbor] >= 0}
        colour = 0
        while colour in used:
            colour += 1
        colours[best] = colour
    required = int(colours.max()) + 1 if count else 0
    return {
        "success": required <= 4,
        "colours_required_by_dsatur_greedy": required,
        "colours": colours,
        "warning": None
        if required <= 4
        else "DSATUR greedy requested more than four colours; no pseudo-partition colour map was rendered.",
    }


def strongly_connected_components(flow: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return SCC labels for native-cell -> destination-cell directed flow."""

    flow = np.asarray(flow, dtype=np.float64)
    if flow.ndim != 2 or flow.shape[0] != flow.shape[1]:
        raise ValueError("Coarse directed-flow diagnostics must be square.")
    node_count = flow.shape[0]
    scale = float(np.max(flow)) if flow.size else 0.0
    absolute_threshold = float(threshold * scale)
    adjacency = [set() for _ in range(node_count)]
    reverse = [set() for _ in range(node_count)]
    for destination, native in np.argwhere(flow > absolute_threshold):
        destination = int(destination)
        native = int(native)
        if destination == native:
            continue
        adjacency[native].add(destination)
        reverse[destination].add(native)

    visited = np.zeros(node_count, dtype=bool)
    order: list[int] = []
    for start in range(node_count):
        if visited[start]:
            continue
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    stack.append((neighbor, False))

    labels = np.full(node_count, -1, dtype=np.int32)
    component_sizes: list[int] = []
    for start in reversed(order):
        if labels[start] >= 0:
            continue
        component_id = len(component_sizes)
        size = 0
        stack = [start]
        labels[start] = component_id
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in reverse[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        component_sizes.append(size)
    edge_count = sum(len(neighbors) for neighbors in adjacency)
    return {
        "labels": labels,
        "component_count": len(component_sizes),
        "component_sizes": component_sizes,
        "largest_component_node_fraction": (
            float(max(component_sizes) / node_count) if node_count else 0.0
        ),
        "directed_edge_count": edge_count,
        "relative_threshold": float(threshold),
        "absolute_threshold": absolute_threshold,
        "adjacency": adjacency,
    }


def _partition_report(
    labels: np.ndarray,
    fine_mass: np.ndarray,
    components: list[dict[str, Any]],
    colouring: dict[str, Any],
) -> dict[str, Any]:
    destinations = sorted(int(value) for value in np.unique(labels))
    destination_report: dict[str, Any] = {}
    largest_ratios = []
    for destination in destinations:
        selected = [component for component in components if component["destination"] == destination]
        total_mass = float(sum(component["mass"] for component in selected))
        largest_mass = float(max((component["mass"] for component in selected), default=0.0))
        ratio = largest_mass / total_mass if total_mass > 0 else 0.0
        largest_ratios.append(ratio)
        destination_report[str(destination)] = {
            "connected_component_count": len(selected),
            "assigned_mass": total_mass,
            "largest_component_mass_fraction_within_destination": ratio,
            "mean_index_compactness": float(
                np.mean(
                    [component["compactness_index_4pi_area_over_perimeter2"] for component in selected]
                )
            ),
        }
    component_masses = np.asarray([component["mass"] for component in components], dtype=np.float64)
    compactness = np.asarray(
        [component["compactness_index_4pi_area_over_perimeter2"] for component in components],
        dtype=np.float64,
    )
    total_mass = float(np.sum(fine_mass))
    return {
        "semantics": PSEUDO_PARTITION_WARNING,
        "component_count": len(components),
        "destination_count": len(destinations),
        "fragmented_destination_count": int(
            sum(item["connected_component_count"] > 1 for item in destination_report.values())
        ),
        "largest_component_mass_fraction_of_all_source_mass": (
            float(component_masses.max() / total_mass)
            if component_masses.size and total_mass > 0
            else 0.0
        ),
        "largest_component_mass_fraction_within_destination_summary": _summary(
            np.asarray(largest_ratios)
        ),
        "component_mass_summary": _summary(component_masses),
        "index_compactness_summary": _summary(compactness),
        "four_colouring": {
            key: value
            for key, value in colouring.items()
            if key != "colours"
        },
        "by_destination": destination_report,
    }


def _draw_field(axis, values: np.ndarray, title: str, *, cmap: str = "viridis"):
    image = axis.imshow(np.asarray(values), origin="lower", aspect="auto", cmap=cmap)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("radial fine-cell index")
    axis.set_ylabel("theta fine-cell index")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _draw_flow(axis, flow: np.ndarray, coarse_shape: tuple[int, int], scc: dict[str, Any]):
    node_count = int(np.prod(coarse_shape))
    if flow.shape != (node_count, node_count):
        raise ValueError(
            f"Flow matrix shape {flow.shape} is incompatible with coarse shape {coarse_shape}."
        )
    positions = np.column_stack(
        (np.tile(np.arange(coarse_shape[1]), coarse_shape[0]), np.repeat(np.arange(coarse_shape[0]), coarse_shape[1]))
    )
    labels = np.asarray(scc["labels"])
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        c=labels,
        cmap="tab20",
        s=35,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )
    edges = []
    threshold = float(scc["absolute_threshold"])
    for destination, native in np.argwhere(flow > threshold):
        if destination != native:
            edges.append((float(flow[destination, native]), int(native), int(destination)))
    edges.sort(reverse=True)
    plotted_edges = edges[:200]
    maximum = max((edge[0] for edge in plotted_edges), default=1.0)
    for value, native, destination in plotted_edges:
        start = positions[native]
        end = positions[destination]
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "0.25",
                "alpha": 0.15 + 0.55 * value / maximum,
                "lw": 0.3 + 1.2 * value / maximum,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=1,
        )
    axis.set_title(
        f"coarse directed transport; SCC={scc['component_count']}\n"
        f"showing {len(plotted_edges)}/{len(edges)} off-diagonal edges",
        fontsize=9,
    )
    axis.set_xlabel("radial coarse-cell index")
    axis.set_ylabel("theta coarse-cell index")
    axis.set_xlim(-0.5, coarse_shape[1] - 0.5)
    axis.set_ylim(-0.5, coarse_shape[0] - 0.5)
    axis.set_aspect("equal")


def visualize_coarsening_partitions(
    fine_path: str | Path,
    coarse_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    hm: float | None = None,
    flow_relative_threshold: float = 1e-6,
) -> dict[str, Any]:
    # Keep graph/connected-component diagnostics usable in the lightweight
    # preprocessing environment, where matplotlib is intentionally optional.
    global plt, ListedColormap
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    """Render diagnostics and return/write a machine-readable JSON report.

    ``H_m`` defaults to the median positive ``source_support_scale`` and is
    only a plotting normalization.  It is not a new physical mass scale.
    """

    fine_path = Path(fine_path)
    coarse_path = Path(coarse_path)
    output_path = Path(output_path)
    if flow_relative_threshold < 0 or not np.isfinite(flow_relative_threshold):
        raise ValueError("flow_relative_threshold must be finite and non-negative.")
    fine_mass, coarse_mass, coarse_shape, metadata, diagnostics = _read_inputs(
        fine_path, coarse_path
    )
    required = {
        "source_support_scale",
        "source_enclosed_mass",
        "source_transport_entropy",
        "dominant_fraction",
        "dominant_destination",
    }
    missing = sorted(required.difference(diagnostics))
    if missing:
        raise ValueError(f"Coarse diagnostics are missing required arrays: {missing}.")
    support = np.asarray(diagnostics["source_support_scale"], dtype=np.float64)
    enclosed = np.asarray(diagnostics["source_enclosed_mass"], dtype=np.float64)
    entropy = np.asarray(diagnostics["source_transport_entropy"], dtype=np.float64)
    dominant_fraction = np.asarray(diagnostics["dominant_fraction"], dtype=np.float64)
    dominant_destination = np.asarray(diagnostics["dominant_destination"], dtype=np.int64)
    for name, values in {
        "source_support_scale": support,
        "source_enclosed_mass": enclosed,
        "source_transport_entropy": entropy,
        "dominant_fraction": dominant_fraction,
        "dominant_destination": dominant_destination,
    }.items():
        if values.shape != fine_mass.shape:
            raise ValueError(f"Diagnostic {name} has shape {values.shape}, expected {fine_mass.shape}.")

    mass_filter = float(metadata.get("mass_filter", np.nan))
    if not np.isfinite(mass_filter) or mass_filter <= 0:
        raise ValueError("Coarsening metadata must provide a positive finite mass_filter.")
    positive_support = support[np.isfinite(support) & (support > 0)]
    hm_was_supplied = hm is not None
    if hm is None:
        if positive_support.size == 0:
            raise ValueError("source_support_scale has no positive finite values.")
        hm = float(np.median(positive_support))
    else:
        hm = float(hm)
    if not np.isfinite(hm) or hm <= 0:
        raise ValueError("H_m normalization must be positive and finite.")

    component_map, components, adjacency = connected_components_four_neighbor(
        dominant_destination, fine_mass
    )
    colouring = dsatur_greedy_four_coloring(adjacency)
    partition_report = _partition_report(
        dominant_destination, fine_mass, components, colouring
    )

    flow_key = None
    for candidate in (
        "transport_mass_by_destination_native",
        "transport_volume_by_destination_native",
    ):
        if candidate in diagnostics:
            flow_key = candidate
            break
    scc = None
    flow = None
    if flow_key is not None:
        flow = np.asarray(diagnostics[flow_key], dtype=np.float64)
        scc = strongly_connected_components(flow, flow_relative_threshold)

    report: dict[str, Any] = {
        "fine_file": str(fine_path),
        "coarse_file": str(coarse_path),
        "operator": metadata.get("operator", "unknown"),
        "partition_semantics": PSEUDO_PARTITION_WARNING,
        "H_m_definition": "user-supplied plotting normalization"
        if hm_was_supplied
        else "median positive source_support_scale for this snapshot",
        "H_m": hm,
        "mass_filter": mass_filter,
        "source_support_scale_over_H_m": _summary(support / hm),
        "source_enclosed_mass_over_mass_filter": _summary(enclosed / mass_filter),
        "source_transport_entropy": _summary(entropy),
        "dominant_fraction": _summary(dominant_fraction),
        "coarse_output_mass": _summary(coarse_mass),
        "coarse_output_mass_over_mass_filter": _summary(coarse_mass / mass_filter),
        "pseudo_partition": partition_report,
        "coarse_transport": None,
    }
    if scc is not None:
        report["coarse_transport"] = {
            "diagnostic": flow_key,
            "scc_count": scc["component_count"],
            "scc_sizes": scc["component_sizes"],
            "largest_scc_node_fraction": scc["largest_component_node_fraction"],
            "directed_edge_count": scc["directed_edge_count"],
            "relative_threshold": scc["relative_threshold"],
            "absolute_threshold": scc["absolute_threshold"],
        }

    figure, axes = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    _draw_field(axes[0, 0], support / hm, r"source support $H_k/H_m$")
    _draw_field(axes[0, 1], enclosed / mass_filter, r"enclosed mass / $M_{filter}$", cmap="magma")
    _draw_field(axes[0, 2], entropy, "source transport entropy", cmap="cividis")
    _draw_field(axes[0, 3], dominant_fraction, "dominant transport fraction", cmap="viridis")

    if colouring["success"]:
        component_colours = np.asarray(colouring["colours"], dtype=np.int16)
        colour_image = component_colours[component_map]
        axes[1, 0].imshow(
            colour_image,
            origin="lower",
            aspect="auto",
            cmap=ListedColormap(["#4477AA", "#EE6677", "#228833", "#CCBB44"]),
            vmin=0,
            vmax=3,
        )
        axes[1, 0].set_title(
            "pseudo-partition components (display-only 4-colour)\nNOT a physical tessellation",
            fontsize=9,
        )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            colouring["warning"],
            ha="center",
            va="center",
            wrap=True,
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("pseudo-partition colouring NOT rendered", fontsize=9)
    axes[1, 0].set_xlabel("radial fine-cell index")
    axes[1, 0].set_ylabel("theta fine-cell index")

    _draw_field(
        axes[1, 1],
        dominant_destination,
        "dominant destination ID (argmax only)",
        cmap="turbo",
    )
    if flow is not None and scc is not None:
        _draw_flow(axes[1, 2], flow, coarse_shape, scc)
    else:
        axes[1, 2].text(
            0.5,
            0.5,
            "coarse transport-flow diagnostic unavailable",
            ha="center",
            va="center",
            transform=axes[1, 2].transAxes,
        )
        axes[1, 2].set_title("coarse directed flow / SCC", fontsize=9)
    axes[1, 3].axis("off")
    summary_text = (
        f"pseudo-components: {partition_report['component_count']}\n"
        f"destinations: {partition_report['destination_count']}\n"
        f"fragmented destinations: {partition_report['fragmented_destination_count']}\n"
        f"DSATUR colours: {colouring['colours_required_by_dsatur_greedy']}\n"
        f"four-colour rendered: {colouring['success']}\n"
        f"largest component / all source mass: "
        f"{partition_report['largest_component_mass_fraction_of_all_source_mass']:.4g}\n\n"
        f"{PSEUDO_PARTITION_WARNING}."
    )
    axes[1, 3].text(0.0, 1.0, summary_text, ha="left", va="top", wrap=True, fontsize=9)
    figure.suptitle(
        f"Mass-adaptive coarse-graining diagnostics: {coarse_path.name}\n"
        "Pseudo-partition colours are display diagnostics, not physical cells",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    if report_path is None:
        report_path = output_path.with_suffix(".json")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize mass-adaptive soft-transport diagnostics. The coloured pseudo-partition "
            "is not an exclusive physical equal-mass tessellation."
        )
    )
    parser.add_argument("--fine", required=True, type=Path, help="Fine format-v4 HDF5 snapshot.")
    parser.add_argument("--coarse", required=True, type=Path, help="Coarse HDF5 with diagnostics.")
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--report", type=Path, default=None, help="Output JSON path.")
    parser.add_argument(
        "--hm",
        type=float,
        default=None,
        help="Optional H_m plotting normalization; default is median positive source support.",
    )
    parser.add_argument(
        "--flow-relative-threshold",
        type=float,
        default=1e-6,
        help="Ignore directed-flow edges below this fraction of the maximum flow.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    report = visualize_coarsening_partitions(
        args.fine,
        args.coarse,
        args.output,
        report_path=args.report,
        hm=args.hm,
        flow_relative_threshold=args.flow_relative_threshold,
    )
    print(f"figure={args.output}")
    print(f"pseudo_components={report['pseudo_partition']['component_count']}")
    print(
        "four_colour_success="
        f"{report['pseudo_partition']['four_colouring']['success']}"
    )
    if report["coarse_transport"] is not None:
        print(f"coarse_scc={report['coarse_transport']['scc_count']}")


if __name__ == "__main__":
    main()
