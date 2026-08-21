import unittest

import numpy as np

from raw_data_processing.coarsening_common import (
    temperature_from_internal_energy_density,
    validate_input_fields,
)
from raw_data_processing.coarsening_types import CoarseningInput
from raw_data_processing.galaxy_data import GalData


def _thermodynamic_input(*, temperature_offset: float = 0.0) -> CoarseningInput:
    density = np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float64)
    internal_energy_density = np.asarray(
        [[3.0, 5.0], [7.0, 11.0]], dtype=np.float64
    )
    temperature = (
        temperature_from_internal_energy_density(
            internal_energy_density, density
        )
        + temperature_offset
    )
    return CoarseningInput(
        snapshot={
            "density": density,
            "gas_energy": internal_energy_density,
            "temperature": temperature,
        },
        grid_volume=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        axes=(
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.asarray([0.5, 1.5], dtype=np.float64),
        ),
        axis_edges=(
            np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        ),
        coordinate_mode="polar",
        ndim=2,
    )


class ThermodynamicSemanticsTests(unittest.TestCase):
    def test_temperature_is_derived_from_internal_energy_density(self):
        density = np.asarray([2.0, 5.0], dtype=np.float64)
        internal_energy_density = np.asarray([6.0, 20.0], dtype=np.float64)
        np.testing.assert_allclose(
            temperature_from_internal_energy_density(
                internal_energy_density, density
            ),
            93.0 * internal_energy_density / density,
        )

    def test_rejects_dataset6_as_gas_energy(self):
        with self.assertRaisesRegex(ValueError, "Data-Set-6 is temperature"):
            GalData().set_corr(
                {
                    "Data-Set-5": "density",
                    "Data-Set-6": "gas_energy",
                }
            )

    def test_accepts_dataset10_as_internal_energy_density(self):
        galaxy = GalData().set_corr(
            {
                "Data-Set-5": "density",
                "Data-Set-10": "gas_energy",
            }
        )
        self.assertEqual(galaxy._corr["Data-Set-10"], "gas_energy")

    def test_rejects_temperature_inconsistent_with_macer_relation(self):
        with self.assertRaisesRegex(ValueError, "T = 93"):
            validate_input_fields(_thermodynamic_input(temperature_offset=1.0))

    def test_accepts_consistent_thermodynamic_fields(self):
        self.assertEqual(validate_input_fields(_thermodynamic_input()), (2, 2))


if __name__ == "__main__":
    unittest.main()
