import json
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # pragma: no cover - 允许缺少可选依赖的环境跳过。
    h5py = None

from raw_data_processing.athdf_reader import (
    DEFAULT_BASELINE_FIELDS,
    DEFAULT_INPUT_FIELDS,
    DEFAULT_MODEL_INPUT_FIELDS,
    DEFAULT_REQUIRED_FIELDS,
    DEFAULT_TARGET_FIELD,
    inspect_athdf,
    read_athdf,
)


class DwarfGalaxyConfigTests(unittest.TestCase):
    def test_dwarf_field_roles_match_reader_defaults(self):
        config_path = Path(__file__).resolve().parents[1] / ".config" / "HdfraConfig.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)["dwarf_galaxy"]

        roles = config["field_roles"]
        self.assertEqual(tuple(roles["model_inputs"]), DEFAULT_MODEL_INPUT_FIELDS)
        self.assertEqual(tuple(roles["baseline_only"]), DEFAULT_BASELINE_FIELDS)
        self.assertEqual(roles["target"], DEFAULT_TARGET_FIELD)

        configured_fields = set(config["correspondence"])
        self.assertTrue(set(DEFAULT_REQUIRED_FIELDS).issubset(configured_fields))
        self.assertTrue(set(roles["model_inputs"]).isdisjoint(roles["baseline_only"]))
        self.assertNotIn(roles["target"], roles["model_inputs"])
        self.assertNotIn(roles["target"], roles["baseline_only"])
        self.assertFalse(config["baseline"]["gas_self_gravity_included"])
        self.assertTrue(config["baseline"]["requires_gravitational_potential"])


@unittest.skipIf(h5py is None, "h5py is not installed")
class AthdfReaderTests(unittest.TestCase):
    def _write_synthetic_athdf(self, path: Path, *, max_level: int = 0) -> dict[str, np.ndarray]:
        root_size = (4, 2, 4)
        block_size = (2, 1, 2)
        block_grid_size = tuple(
            root // block for root, block in zip(root_size, block_size)
        )
        locations = list(itertools.product(*(range(size) for size in block_grid_size)))
        # 故意打乱 block 顺序，确保读取器不会依赖文件中的物理排列。
        locations = [locations[index] for index in (6, 1, 4, 3, 0, 7, 2, 5)]

        radius_edges = np.asarray([1.0, 1.5, 2.5, 4.0, 7.0], dtype=np.float32)
        theta_edges = np.asarray([0.2, 1.1, 2.8], dtype=np.float32)
        phi_edges = np.linspace(0.0, 2.0 * np.pi, 5, dtype=np.float32)
        global_edges = (radius_edges, theta_edges, phi_edges)
        global_centers = tuple(
            0.5 * (edges[:-1] + edges[1:]) for edges in global_edges
        )

        rr, tt, pp = np.meshgrid(
            np.arange(root_size[0]),
            np.arange(root_size[1]),
            np.arange(root_size[2]),
            indexing="ij",
        )
        expected = {
            "rho": (100.0 * rr + 10.0 * tt + pp).astype(np.float32),
            "press": (1000.0 + 100.0 * rr + 10.0 * tt + pp).astype(np.float32),
            "rhodotsf": (2000.0 + 100.0 * rr + 10.0 * tt + pp).astype(np.float32),
        }

        num_blocks = len(locations)
        hydro = np.empty((2, num_blocks, 2, 1, 2), dtype=np.float32)
        user = np.empty((1, num_blocks, 2, 1, 2), dtype=np.float32)
        coordinate_faces = [
            np.empty((num_blocks, block_size[axis] + 1), dtype=np.float32)
            for axis in range(3)
        ]
        coordinate_centers = [
            np.empty((num_blocks, block_size[axis]), dtype=np.float32)
            for axis in range(3)
        ]
        for block_index, location in enumerate(locations):
            slices = tuple(
                slice(logical * block, (logical + 1) * block)
                for logical, block in zip(location, block_size)
            )
            hydro[0, block_index] = np.transpose(expected["rho"][slices], (2, 1, 0))
            hydro[1, block_index] = np.transpose(expected["press"][slices], (2, 1, 0))
            user[0, block_index] = np.transpose(
                expected["rhodotsf"][slices], (2, 1, 0)
            )
            for axis, (logical, block) in enumerate(zip(location, block_size)):
                start = logical * block
                coordinate_faces[axis][block_index] = global_edges[axis][
                    start : start + block + 1
                ]
                coordinate_centers[axis][block_index] = global_centers[axis][
                    start : start + block
                ]

        with h5py.File(path, "w") as handle:
            handle.attrs["Coordinates"] = np.bytes_("spherical_polar")
            handle.attrs["RootGridSize"] = np.asarray(root_size, dtype=np.int32)
            handle.attrs["MeshBlockSize"] = np.asarray(block_size, dtype=np.int32)
            handle.attrs["MaxLevel"] = np.int32(max_level)
            handle.attrs["Time"] = np.float32(2.5)
            handle.attrs["DatasetNames"] = np.asarray(
                [b"hydro", b"user"], dtype="S21"
            )
            handle.attrs["NumVariables"] = np.asarray([2, 1], dtype=np.int32)
            handle.attrs["VariableNames"] = np.asarray(
                [b"rho", b"press", b"rhodotsf"], dtype="S21"
            )
            handle.create_dataset("LogicalLocations", data=np.asarray(locations))
            handle.create_dataset(
                "Levels", data=np.full(num_blocks, max_level, dtype=np.int32)
            )
            handle.create_dataset("hydro", data=hydro)
            handle.create_dataset("user", data=user)
            for axis in range(3):
                handle.create_dataset(f"x{axis + 1}f", data=coordinate_faces[axis])
                handle.create_dataset(f"x{axis + 1}v", data=coordinate_centers[axis])
        return expected

    def test_inspection_maps_variables_across_hdf_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.athdf"
            self._write_synthetic_athdf(path)
            metadata = inspect_athdf(path)

        self.assertEqual(metadata.root_grid_size, (4, 2, 4))
        self.assertEqual(metadata.mesh_block_size, (2, 1, 2))
        self.assertEqual(metadata.variable_names, ("rho", "press", "rhodotsf"))
        self.assertEqual(metadata.variable_locations["rho"].dataset_name, "hydro")
        self.assertEqual(metadata.variable_locations["press"].local_index, 1)
        self.assertEqual(metadata.variable_locations["rhodotsf"].dataset_name, "user")
        self.assertEqual(metadata.variable_locations["rhodotsf"].local_index, 0)

    def test_selective_read_reconstructs_global_physical_axis_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.athdf"
            expected = self._write_synthetic_athdf(path)
            snapshot = read_athdf(path, fields=("rho", "rhodotsf"))

        self.assertEqual(snapshot.shape, (4, 2, 4))
        self.assertEqual(tuple(snapshot.fields), ("rho", "rhodotsf"))
        np.testing.assert_array_equal(snapshot.fields["rho"], expected["rho"])
        np.testing.assert_array_equal(
            snapshot.fields["rhodotsf"], expected["rhodotsf"]
        )
        self.assertNotIn("press", snapshot.fields)

    def test_exact_spherical_volume_uses_cell_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.athdf"
            self._write_synthetic_athdf(path)
            snapshot = read_athdf(path, fields=())

        radius_edges, theta_edges, phi_edges = snapshot.geometry.edges
        expected_total = (
            (radius_edges[-1] ** 3 - radius_edges[0] ** 3)
            / 3.0
            * (np.cos(theta_edges[0]) - np.cos(theta_edges[-1]))
            * (phi_edges[-1] - phi_edges[0])
        )
        self.assertAlmostEqual(
            float(snapshot.geometry.cell_volume.sum()), float(expected_total), places=10
        )
        self.assertEqual(snapshot.geometry.cell_volume.shape, (4, 2, 4))
        self.assertTrue(np.all(snapshot.geometry.cell_volume > 0))

    def test_missing_requested_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.athdf"
            self._write_synthetic_athdf(path)
            with self.assertRaisesRegex(KeyError, "请求字段不存在"):
                read_athdf(path, fields=("not_a_field",))

    def test_amr_file_is_rejected_instead_of_silently_misread(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_amr.athdf"
            self._write_synthetic_athdf(path, max_level=1)
            with self.assertRaisesRegex(NotImplementedError, "MaxLevel == 0"):
                inspect_athdf(path)

    def test_default_field_set_keeps_target_separate_by_name(self):
        self.assertEqual(DEFAULT_TARGET_FIELD, "rhodotsf")
        self.assertEqual(DEFAULT_INPUT_FIELDS, DEFAULT_MODEL_INPUT_FIELDS)
        self.assertNotIn(DEFAULT_TARGET_FIELD, DEFAULT_INPUT_FIELDS)
        self.assertEqual(DEFAULT_BASELINE_FIELDS, ("cool_rate",))
        self.assertNotIn("cool_rate", DEFAULT_MODEL_INPUT_FIELDS)
        self.assertIn(DEFAULT_TARGET_FIELD, DEFAULT_REQUIRED_FIELDS)
        self.assertIn("cool_rate", DEFAULT_REQUIRED_FIELDS)
        self.assertNotIn("newstar", DEFAULT_REQUIRED_FIELDS)
        self.assertNotIn("metallicity", DEFAULT_REQUIRED_FIELDS)


if __name__ == "__main__":
    unittest.main()
