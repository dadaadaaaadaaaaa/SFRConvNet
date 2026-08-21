import unittest

import numpy as np

from raw_data_processing.macer_sfr_operator import (
    DATA_COMPATIBLE_MODE,
    LIMITER_COOLING,
    LIMITER_FREE_FALL,
    LIMITER_INELIGIBLE,
    LIMITER_ROTATION,
    MACER_G_CODE,
    PAPER_PHYSICAL_MODE,
    calculate_macer_sfr,
    exp_factor_sf,
)


class ExpFactorSFTests(unittest.TestCase):
    def test_matches_all_fortran_branches(self):
        tau = np.asarray([2.0, 2.0, 2.0])
        dt = np.asarray([0.0, 0.01, 0.2])
        result = exp_factor_sf(dt, tau)
        expected = np.asarray(
            [
                1.0 / 2.0,
                (1.0 - 0.5 * 0.01 / 2.0) / 2.0,
                (1.0 - np.exp(-0.2 / 2.0)) / 0.2,
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0.0)

    def test_rejects_nonphysical_inputs(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            exp_factor_sf(-1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            exp_factor_sf(0.1, 0.0)


class MacerSFROperatorTests(unittest.TestCase):
    def _fields(self):
        shape = (2, 3)
        density = np.full(shape, 2.0)
        # T=930 K，明显低于源码阈值。
        energy = density * 10.0
        return {
            "density": density,
            "internal_energy_density": energy,
            "cooling_time": np.full(shape, 1.0e-6),
            "cell_volume": np.arange(1.0, 7.0).reshape(shape),
            "radius": np.asarray([1.0, 2.0, 3.0]),
            "time_step": 0.0,
        }

    def test_data_compatible_mode_matches_closed_form_and_total(self):
        fields = self._fields()
        result = calculate_macer_sfr(
            **fields,
            mode=DATA_COMPATIBLE_MODE,
            black_hole_mass=2.0,
        )
        expected_ff = np.sqrt(3.0 * np.pi / (32.0 * MACER_G_CODE * 2.0))
        expected_rot = 2.0 * np.pi * np.sqrt(
            np.asarray([1.0, 8.0, 27.0]) / (MACER_G_CODE * 2.0)
        )
        expected_tau = np.minimum(expected_ff, expected_rot)
        expected_density = 0.01 * 2.0 / expected_tau

        np.testing.assert_allclose(result.tau_free_fall, expected_ff)
        np.testing.assert_allclose(result.tau_rotation[0], expected_rot)
        np.testing.assert_allclose(result.sfr_density[0], expected_density)
        np.testing.assert_allclose(
            result.cell_sfr, result.sfr_density * fields["cell_volume"]
        )
        self.assertAlmostEqual(
            result.total_sfr,
            float(result.cell_sfr.sum(dtype=np.float64)),
        )
        self.assertEqual(result.parameters["cstar"], 0.01)

    def test_paper_mode_uses_absolute_local_acceleration(self):
        fields = self._fields()
        acceleration = np.asarray([-4.0, 0.0, 9.0])
        result = calculate_macer_sfr(
            **fields,
            mode=PAPER_PHYSICAL_MODE,
            radial_acceleration=acceleration,
        )
        expected = np.asarray(
            [
                2.0 * np.pi * np.sqrt(1.0 / 4.0),
                np.inf,
                2.0 * np.pi * np.sqrt(3.0 / 9.0),
            ]
        )
        np.testing.assert_allclose(result.tau_rotation[0], expected)
        self.assertEqual(result.parameters["cstar"], 0.1)

    def test_spherical_acceleration_makes_rotation_timescales_identical(self):
        fields = self._fields()
        mass = 3.5
        radius = np.asarray(fields["radius"])
        acceleration = MACER_G_CODE * mass / radius**2
        data_result = calculate_macer_sfr(
            **fields,
            mode=DATA_COMPATIBLE_MODE,
            black_hole_mass=mass,
        )
        paper_result = calculate_macer_sfr(
            **fields,
            mode=PAPER_PHYSICAL_MODE,
            radial_acceleration=acceleration,
        )
        np.testing.assert_allclose(
            data_result.tau_rotation, paper_result.tau_rotation, rtol=1e-14
        )
        # 当时标与阈值相同，两模式仅因约定的 cstar 相差十倍。
        np.testing.assert_allclose(
            paper_result.sfr_density, 10.0 * data_result.sfr_density
        )

    def test_strict_source_thresholds_and_zero_density(self):
        density = np.asarray([[1.0 / 1.02, 2.0, 0.0, 2.0]])
        temperature = np.asarray([[100.0, 4.0e4, 100.0, 100.0]])
        energy = np.zeros_like(density)
        positive = density > 0
        energy[positive] = temperature[positive] * density[positive] / 93.0
        result = calculate_macer_sfr(
            density=density,
            internal_energy_density=energy,
            cooling_time=np.full_like(density, 0.1),
            cell_volume=np.ones_like(density),
            radius=np.ones_like(density),
            black_hole_mass=1.0,
        )
        np.testing.assert_array_equal(
            result.eligible_mask, np.asarray([[False, False, False, True]])
        )
        np.testing.assert_array_equal(
            result.limiter_code[:,:3],
            np.full((1, 3), LIMITER_INELIGIBLE, dtype=np.int8),
        )
        self.assertEqual(result.sfr_density[0, 0], 0.0)
        self.assertEqual(result.sfr_density[0, 1], 0.0)
        self.assertEqual(result.sfr_density[0, 2], 0.0)
        self.assertTrue(np.isinf(result.tau_free_fall[0, 2]))
        self.assertEqual(result.finite_step_factor[0, 2], 0.0)

    def test_zero_density_and_zero_acceleration_remain_safe(self):
        result = calculate_macer_sfr(
            density=np.zeros((1, 1)),
            internal_energy_density=np.zeros((1, 1)),
            cooling_time=np.zeros((1, 1)),
            cell_volume=np.ones((1, 1)),
            radius=np.ones((1, 1)),
            radial_acceleration=np.zeros((1, 1)),
            mode=PAPER_PHYSICAL_MODE,
        )
        self.assertEqual(result.total_sfr, 0.0)
        self.assertTrue(np.isinf(result.tau_free_fall[0, 0]))
        self.assertTrue(np.isinf(result.tau_rotation[0, 0]))
        self.assertTrue(np.isinf(result.tau_star_formation[0, 0]))

    def test_limiter_diagnostics_cover_cooling_freefall_and_rotation(self):
        density = np.full((1, 3), 2.0)
        energy = density.copy()
        radius = np.asarray([[1.0, 1.0, 1.0]])
        # 第三个格点用很小的等效中心质量，使旋转时标短于自由落体时标。
        masses = np.asarray([[1.0, 1.0, 1.0e-4]])
        ff = np.sqrt(3.0 * np.pi / (32.0 * MACER_G_CODE * 2.0))
        rot_third = 2.0 * np.pi * np.sqrt(1.0 / (MACER_G_CODE * 1.0e-4))
        self.assertGreater(rot_third, ff)
        # 要让旋转成为 min，需要增大中心质量。
        masses[0, 2] = 1.0e8
        result = calculate_macer_sfr(
            density=density,
            internal_energy_density=energy,
            cooling_time=np.asarray([[10.0, 1.0e-9, 1.0e-9]]),
            cell_volume=np.ones((1, 3)),
            radius=radius,
            black_hole_mass=masses,
        )
        np.testing.assert_array_equal(
            result.limiter_code,
            np.asarray(
                [[LIMITER_COOLING, LIMITER_FREE_FALL, LIMITER_ROTATION]],
                dtype=np.int8,
            ),
        )

    def test_finite_step_is_applied_to_sfr(self):
        fields = self._fields()
        continuous = calculate_macer_sfr(
            **fields,
            black_hole_mass=2.0,
        )
        fields["time_step"] = 0.1
        finite = calculate_macer_sfr(
            **fields,
            black_hole_mass=2.0,
        )
        self.assertTrue(np.all(finite.sfr_density < continuous.sfr_density))

    def test_rejects_missing_mode_specific_input_and_bad_shapes(self):
        fields = self._fields()
        with self.assertRaisesRegex(ValueError, "black_hole_mass"):
            calculate_macer_sfr(**fields)
        with self.assertRaisesRegex(ValueError, "radial_acceleration"):
            calculate_macer_sfr(**fields, mode=PAPER_PHYSICAL_MODE)
        with self.assertRaisesRegex(ValueError, "expected"):
            calculate_macer_sfr(
                **{**fields, "cell_volume": np.ones((1, 3))},
                black_hole_mass=1.0,
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            calculate_macer_sfr(
                **{**fields, "radius": np.asarray([0.0, 1.0, 2.0])},
                black_hole_mass=1.0,
            )


if __name__ == "__main__":
    unittest.main()
