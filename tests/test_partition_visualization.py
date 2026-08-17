import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from raw_data_processing.visualize_coarsening_partitions import (
    PSEUDO_PARTITION_WARNING,
    connected_components_four_neighbor,
    dsatur_greedy_four_coloring,
    strongly_connected_components,
    visualize_coarsening_partitions,
)


def _write_synthetic_pair(directory: Path) -> tuple[Path, Path]:
    fine_path = directory / "fine.h5"
    coarse_path = directory / "coarse.h5"
    fine_shape = (4, 4)
    coarse_shape = (2, 2)
    density = 1.0 + np.arange(16, dtype=np.float64).reshape(fine_shape) / 10.0
    volume = np.ones(fine_shape, dtype=np.float64)
    with h5py.File(fine_path, "w") as file:
        snapshot = file.create_group("snapshot")
        snapshot.create_dataset("density", data=density)
        geometry = file.create_group("geometry")
        geometry.create_dataset("cell_volume", data=volume)

    # Destination 0 has two disconnected components, deliberately testing
    # that hard labels are split before display colouring.
    destinations = np.asarray(
        [
            [0, 1, 1, 0],
            [2, 1, 1, 3],
            [2, 2, 3, 3],
            [0, 2, 3, 0],
        ],
        dtype=np.int32,
    )
    flow = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ]
    )
    with h5py.File(coarse_path, "w") as file:
        snapshot = file.create_group("snapshot")
        snapshot.create_dataset("density", data=np.ones(coarse_shape))
        geometry = file.create_group("geometry")
        geometry.create_dataset("cell_volume", data=np.full(coarse_shape, 4.0))
        coarsening = file.create_group("coarsening")
        coarsening.attrs["operator"] = "synthetic_soft_transport"
        coarsening.attrs["mass_filter"] = 2.0
        coarsening.create_dataset(
            "source_support_scale", data=np.linspace(1.0, 2.0, 16).reshape(fine_shape)
        )
        coarsening.create_dataset(
            "source_enclosed_mass", data=np.linspace(2.0, 4.0, 16).reshape(fine_shape)
        )
        coarsening.create_dataset(
            "source_transport_entropy", data=np.linspace(0.0, 1.0, 16).reshape(fine_shape)
        )
        coarsening.create_dataset(
            "dominant_fraction", data=np.linspace(0.5, 1.0, 16).reshape(fine_shape)
        )
        coarsening.create_dataset("dominant_destination", data=destinations)
        coarsening.create_dataset("transport_mass_by_destination_native", data=flow)
    return fine_path, coarse_path


class PartitionVisualizationTests(unittest.TestCase):
    def test_component_split_and_four_colour(self):
        labels = np.asarray([[0, 1, 0], [2, 1, 2], [0, 1, 0]])
        mass = np.ones_like(labels, dtype=np.float64)
        component_map, components, adjacency = connected_components_four_neighbor(labels, mass)
        self.assertEqual(component_map.shape, labels.shape)
        self.assertGreater(len(components), len(np.unique(labels)))
        colouring = dsatur_greedy_four_coloring(adjacency)
        self.assertTrue(colouring["success"])
        self.assertLessEqual(colouring["colours_required_by_dsatur_greedy"], 4)

    def test_directed_flow_scc(self):
        flow = np.asarray(
            [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        )
        result = strongly_connected_components(flow, threshold=1e-12)
        self.assertEqual(result["component_count"], 2)
        self.assertEqual(sorted(result["component_sizes"]), [2, 2])

    def test_more_than_four_colours_is_reported(self):
        # K5 is intentionally non-planar; this exercises the explicit failure
        # path rather than pretending that four display colours succeeded.
        adjacency = [set(range(5)).difference({node}) for node in range(5)]
        colouring = dsatur_greedy_four_coloring(adjacency)
        self.assertFalse(colouring["success"])
        self.assertEqual(colouring["colours_required_by_dsatur_greedy"], 5)
        self.assertIn("no pseudo-partition colour map", colouring["warning"])

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib") is not None,
        "matplotlib is optional in the preprocessing environment",
    )
    def test_end_to_end_png_and_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            fine_path, coarse_path = _write_synthetic_pair(directory)
            output_path = directory / "diagnostics.png"
            report_path = directory / "diagnostics.json"
            report = visualize_coarsening_partitions(
                fine_path,
                coarse_path,
                output_path,
                report_path=report_path,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 1000)
            self.assertTrue(report_path.is_file())
            with report_path.open("r", encoding="utf-8") as handle:
                restored = json.load(handle)
            self.assertEqual(restored["partition_semantics"], PSEUDO_PARTITION_WARNING)
            self.assertGreater(restored["pseudo_partition"]["component_count"], 4)
            self.assertTrue(restored["pseudo_partition"]["four_colouring"]["success"])
            self.assertEqual(restored["coarse_transport"]["scc_count"], 2)
            self.assertEqual(report["operator"], "synthetic_soft_transport")


if __name__ == "__main__":
    unittest.main()
