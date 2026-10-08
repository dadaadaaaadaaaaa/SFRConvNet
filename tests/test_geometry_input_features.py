"""Exp3.1/3.2 的几何输入、标准化与原物理产品不变性回归测试。"""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from raw_data_processing.galaxy_data import GalData, GalDataSet
from raw_data_processing.geometry_features import (
    GEOMETRY_INPUT_CHANNELS,
    build_geometry_input_features,
    geometry_input_contract,
)


def _features(mode="absolute_radius_widths", **overrides):
    arguments = {
        "axes": (np.array([1.0, 3.0, 7.0]), np.array([0.4, 1.4])),
        "axis_edges": (np.array([0.5, 2.0, 5.0, 10.0]), np.array([0.1, 0.8, 2.0])),
        "coordinate_mode": "polar",
        "ndim": 2,
        "shape": (2, 3),
        "mode": mode,
    }
    arguments.update(overrides)
    return build_geometry_input_features(**arguments)


def _galaxy(index=1, target=3.0):
    """使用非规则边界；几何不依赖标签，体积故意不由几何公式重算。"""

    galaxy = GalData(ndim=2, coordinate_mode="polar")
    radius = np.array([10.0, 20.0, 40.0, 80.0])
    theta = np.array([0.3, 0.8, 1.3, 1.8])
    galaxy._axes = [radius, theta]
    galaxy._axis_edges = [
        np.array([8.0, 15.0, 30.0, 60.0, 100.0]),
        np.array([0.1, 0.55, 1.05, 1.55, 2.1]),
    ]
    galaxy._coord = tuple(np.meshgrid(*galaxy._axes))
    galaxy._grid_volume = np.arange(1.0, 17.0).reshape(4, 4)
    galaxy._volume_source = "synthetic_authoritative_volume"
    density = np.full((4, 4), 2.0)
    energy = density * 100.0 / 93.0
    zeros = np.zeros((4, 4))
    galaxy._data_hdfra = {
        "density": density,
        "gas_energy": energy,
        "temperature": 93.0 * energy / density,
        "v1": zeros,
        "v2": zeros,
        "v3": zeros,
        "initial_stellar_density": zeros,
        "cooling_time": np.full((4, 4), 0.05),
        "radial_acceleration": np.full((4, 4), 100.0),
        "theta_acceleration": zeros,
    }
    galaxy._corr = {}
    galaxy._data_log = {"mbh": 2.0, "dt": 1.0e-6, "sfr": target, "sfr_target": target}
    galaxy._target_metadata = {
        "dataset": "Data-Set-11",
        "name": "sfr",
        "reduction": "sum",
        "quantity": "instantaneous_sfr",
        "units": "MACER_mass_code/Gyr",
        "to_msun_per_yr": 0.025,
    }
    galaxy.time = float(index)
    galaxy.attach_fine_source_metadata(
        {
            "simulation_code": "MACER2D",
            "galaxy_id": "synthetic",
            "run_id": "synthetic",
            "source_series_id": "synthetic",
            "raw_snapshot_index": index,
        }
    )
    return galaxy.compute_macer_sfr_products()


class GeometryInputFeatureTests(unittest.TestCase):
    def test_exact_stored_edge_widths_and_array_order(self):
        features = _features()
        np.testing.assert_allclose(features["log10_radius_kpc"], np.log10([[1, 3, 7]] * 2))
        np.testing.assert_allclose(
            features["log10_radial_width_kpc"], np.log10([[1.5, 3, 5]] * 2)
        )
        np.testing.assert_allclose(
            features["log10_polar_arc_width_kpc"],
            np.log10(np.array([0.7, 1.2])[:, None] * np.array([1, 3, 7])[None, :]),
        )
        self.assertEqual(tuple(features), GEOMETRY_INPUT_CHANNELS)

    def test_absolute_channels_track_uniform_physical_scaling(self):
        original = _features()
        scaled = _features(
            axes=(np.array([100, 300, 700]), np.array([0.4, 1.4])),
            axis_edges=(np.array([50, 200, 500, 1000]), np.array([0.1, 0.8, 2.0])),
        )
        for name in GEOMETRY_INPUT_CHANNELS:
            np.testing.assert_allclose(scaled[name] - original[name], 2.0)
        radius_only = _features("absolute_radius")
        self.assertEqual(tuple(radius_only), ("log10_radius_kpc",))
        np.testing.assert_array_equal(radius_only["log10_radius_kpc"], original["log10_radius_kpc"])

    def test_invalid_geometry_and_unknown_mode_fail_closed(self):
        malformed = (
            {"mode": "unknown"},
            {"coordinate_mode": "cartesian"},
            {"ndim": 3},
            {"shape": (3, 2)},
            {"axes": (np.array([0, 3, 7]), np.array([0.4, 1.4]))},
            {"axes": (np.array([1, np.nan, 7]), np.array([0.4, 1.4]))},
            {"axis_edges": (np.array([0.5, 2, 5]), np.array([0.1, 0.8, 2]))},
            {"axis_edges": (np.array([0.5, 2, 2, 10]), np.array([0.1, 0.8, 2]))},
            {"axis_edges": (np.array([-0.5, 2, 5, 10]), np.array([0.1, 0.8, 2]))},
            {"axis_edges": (np.array([0.5, 2, 5, 10]), np.array([0.1, 0.8, 7]))},
        )
        for arguments in malformed:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _features(**arguments)
        self.assertEqual(_features("none", ndim=3), {})

    def test_contract_has_fixed_reference_and_no_extra_log(self):
        none = geometry_input_contract("none")
        self.assertEqual(none["mode"], "none")
        self.assertEqual(none["channels"], [])
        contract = geometry_input_contract("absolute_radius_widths")
        self.assertEqual(contract["reference_length_kpc"], 1.0)
        self.assertEqual(contract["angle_units"], "rad")
        self.assertFalse(contract["additional_log1p"])
        self.assertFalse(contract["per_snapshot_minmax"])
        self.assertEqual(contract["fields"]["log10_radial_width_kpc"]["source"], "coordinates/axis_0_edges")

    def test_hdf5_opt_in_preserves_fields_coordinates_labels_and_uids(self):
        with tempfile.TemporaryDirectory() as folder:
            _galaxy(target=0.0).save_h5(Path(folder) / "00001.h5")
            default = GalDataSet().load_data(folder)
            explicit_none = GalDataSet().load_data(folder, geometry_input_mode="none")
            enhanced = GalDataSet().load_data(folder, geometry_input_mode="absolute_radius_widths")
            np.testing.assert_array_equal(default.x, explicit_none.x)
            self.assertFalse(set(default.input_channels).intersection(GEOMETRY_INPUT_CHANNELS))
            self.assertEqual(enhanced.input_channels, default.input_channels + GEOMETRY_INPUT_CHANNELS)
            np.testing.assert_array_equal(enhanced.x[:, :default.x.shape[1]], default.x)
            np.testing.assert_array_equal(enhanced.coordinate, default.coordinate)
            for name in ("sfr_target", "sfr_macer_baseline", "sfr_closure_true", "sample_uid", "parent_hr_uid"):
                np.testing.assert_array_equal(getattr(enhanced, name), getattr(default, name))
            np.testing.assert_array_equal(enhanced.raw_data["gas_mass"], default.raw_data["gas_mass"])
            np.testing.assert_array_equal(enhanced.raw_data["grid_volume"], default.raw_data["grid_volume"])
            np.testing.assert_allclose(enhanced.sfr_target, enhanced.sfr_macer_baseline + enhanced.sfr_closure_true)
            self.assertLess(enhanced.sfr_closure_true[0], 0.0)
            self.assertFalse(set(enhanced.input_channels).intersection({"sfr_target", "sfr_macer_baseline", "sfr_closure_true"}))
            with h5py.File(Path(folder) / "00001.h5") as handle:
                self.assertFalse(set(handle["snapshot"]).intersection(GEOMETRY_INPUT_CHANNELS))

    def test_prelogged_geometry_is_never_logged_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            _galaxy().save_h5(Path(folder) / "00001.h5")
            enhanced = GalDataSet().load_data(folder, geometry_input_mode="absolute_radius_widths")
            # 所有几何通道都是正值，此时按数值符号推断会错误启用 log1p。
            self.assertTrue(np.all(enhanced.x[:, -3:] > 0))
            x, mean, std, mask = enhanced.standardize(return_log_transform_mask=True)
            self.assertFalse(np.any(mask[-3:]))
            raw_geometry = enhanced.x[:, -3:]
            np.testing.assert_allclose(mean[:, -3:], np.mean(raw_geometry, axis=(0, 2, 3), keepdims=True))
            np.testing.assert_allclose(x[:, -3:], (raw_geometry - mean[:, -3:]) / std[:, -3:])
            supplied_mask = mask.copy()
            supplied_mask[-1] = True
            with self.assertRaisesRegex(ValueError, "不能再次应用 log1p"):
                enhanced.standardize(mean, std, log_transform_mask=supplied_mask)
            np.testing.assert_array_equal(
                enhanced.standardize(mean, std, log_transform_mask=mask), x
            )

    def test_labels_do_not_affect_geometry_and_test_uses_train_normalization(self):
        with tempfile.TemporaryDirectory() as train_folder, tempfile.TemporaryDirectory() as test_folder:
            train_galaxy = _galaxy(target=3.0)
            test_galaxy = _galaxy(target=999.0)
            train_galaxy.save_h5(Path(train_folder) / "00001.h5")
            test_galaxy.save_h5(Path(test_folder) / "00001.h5")
            train = GalDataSet().load_data(train_folder, geometry_input_mode="absolute_radius_widths")
            test = GalDataSet().load_data(test_folder, geometry_input_mode="absolute_radius_widths")
            np.testing.assert_array_equal(train.x, test.x)
            x, mean, std, mask = train.standardize(return_log_transform_mask=True)
            np.testing.assert_array_equal(test.standardize(mean, std, log_transform_mask=mask), x)

    def test_split_and_add_preserve_geometry_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            for index in range(8):
                _galaxy(index=index).save_h5(Path(folder) / f"{index:05d}.h5")
            dataset = GalDataSet().load_data(folder, geometry_input_mode="absolute_radius_widths")
            train, val, empty = dataset.split(train_size=0.75, validation_size=0.25, random_state=42)
            for part in (train, val, empty, train + val):
                self.assertEqual(part.geometry_input_contract, dataset.geometry_input_contract)
                self.assertEqual(part.geometry_input_mode, "absolute_radius_widths")
            self.assertEqual(set(train.parent_hr_uid).intersection(val.parent_hr_uid), set())
            self.assertEqual(len(train), 6)
            self.assertEqual(len(val), 2)
            default = GalDataSet().load_data(folder)
            with self.assertRaises(ValueError):
                _ = train + default
            dataset.select_input_channels(exclude=["log10_polar_arc_width_kpc"])
            self.assertIn("log10_polar_arc_width_kpc", dataset.geometry_input_contract["channels"])
            self.assertNotIn("log10_polar_arc_width_kpc", dataset.input_channels)


if __name__ == "__main__":
    unittest.main()
