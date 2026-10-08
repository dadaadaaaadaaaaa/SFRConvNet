"""三维 SFR 的量纲、时标分支、阈值与退化输入测试。"""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from raw_data_processing.macer_sfr_operator_3d import (
    SFR3DParameters, calculate_macer_sfr_3d, validate_gravity_geometry,
)


class SFR3DTests(unittest.TestCase):
    def fields(self):
        shape = (1, 1, 3)
        return dict(density=np.full(shape, 2.), pressure=np.full(shape, 200.),
                    metallicity=np.full(shape, .00134), cooling_rate=np.full(shape, 1.e8),
                    cell_volume=np.array([[[1., 2., 3.]]]), radius=np.ones(shape),
                    radial_acceleration=-np.ones(shape))

    def test_free_fall_closed_form_and_units(self):
        f = self.fields()
        result = calculate_macer_sfr_3d(**f)
        par = SFR3DParameters()
        expected = .03 * 2 / np.sqrt(3*np.pi/(32*112.4766*2))
        np.testing.assert_allclose(result.sfr_density, expected)
        self.assertAlmostEqual(result.total_sfr, 6*expected)
        # 由物理长度单位转换出的数密度，不能把 code mass 当成克。
        mu = 1/(1.625-.25*.00134)
        n_physical = 2 / (mu*par.hydrogen_mass_code*par.length_unit_cm**3)
        np.testing.assert_allclose(result.number_density, n_physical)
        np.testing.assert_allclose(result.temperature, 100*mu*par.hydrogen_mass_code/par.boltzmann_code)
        np.testing.assert_array_equal(result.limiter_code, 2)

    def test_rotation_factor_and_limiting_branches(self):
        f = self.fields()
        f['radial_acceleration'] = np.array([[[-1., -1., -1.e8]]])
        f['cooling_rate'][0, 0, 0] = 1.
        result = calculate_macer_sfr_3d(**f)
        np.testing.assert_array_equal(result.limiter_code, [[[1, 2, 3]]])
        alt = calculate_macer_sfr_3d(**f, parameters=replace(SFR3DParameters(), rotation_factor=1.))
        self.assertAlmostEqual(alt.sfr_density[0,0,2]/result.sfr_density[0,0,2], 2*np.pi)
        self.assertEqual(alt.sfr_density[0,0,1], result.sfr_density[0,0,1])

    def test_zero_cooling_and_vacuum_are_safe(self):
        f = self.fields()
        f['cooling_rate'][:] = 0
        f['density'][0,0,0] = 0
        result = calculate_macer_sfr_3d(**f)
        self.assertEqual(result.total_sfr, 0)
        self.assertTrue(np.all(np.isinf(result.tau_cool)))
        self.assertFalse(result.eligible_mask[0,0,0])
        self.assertTrue(np.isfinite(result.cell_sfr).all())

    def test_threshold_and_density_definitions(self):
        f = self.fields()
        f['density'][:] = .7
        f['pressure'][:] = 70.
        total = calculate_macer_sfr_3d(**f)
        hydrogen = calculate_macer_sfr_3d(**f, parameters=replace(SFR3DParameters(), number_density_mode='hydrogen_nuclei'))
        self.assertGreater(total.total_sfr, 0)
        self.assertEqual(hydrogen.total_sfr, 0)
        # 将上限设置为算出的精确温度，等号不通过严格阈值。
        threshold = replace(SFR3DParameters(), temperature_ceiling=float(total.temperature[0,0,0]))
        self.assertEqual(calculate_macer_sfr_3d(**f, parameters=threshold).total_sfr, 0)

    def test_rejects_bad_fields(self):
        for key, value in [('cooling_rate', -1.), ('density', np.nan), ('cell_volume', 0.), ('metallicity', 2.)]:
            with self.subTest(key=key):
                f = self.fields()
                f[key][:] = value
                with self.assertRaises(ValueError):
                    calculate_macer_sfr_3d(**f)
        f = self.fields()
        f['pressure'] = np.ones((3,))
        with self.assertRaises(ValueError):
            calculate_macer_sfr_3d(**f)

    def test_geometry_mismatch_rejected(self):
        def snapshot(offset):
            return SimpleNamespace(geometry=SimpleNamespace(
                centers=(np.array([1.+offset]),)*3, edges=(np.array([0.,2.]),)*3))
        validate_gravity_geometry(snapshot(0), snapshot(0))
        with self.assertRaises(ValueError):
            validate_gravity_geometry(snapshot(0), snapshot(.1))


if __name__ == '__main__':
    unittest.main()
