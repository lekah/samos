# -*- coding: utf-8 -*-
"""Checks on UNIT_SYSTEMS, in particular the stress factors.

s_conv used to be 1.0 everywhere while read_lammps_dump advertised that
'units' sets every conversion factor, so stresses came through in
whatever unit the dump used.  These tests pin the factors against
first-principles arithmetic rather than against themselves.
"""

import unittest

from samos.utils.units import UNIT_SYSTEMS

# 1 eV / 1 A^3, in Pa.
EV_PER_ANG3_IN_PA = 1.602176634e-19 / 1e-30

# LAMMPS fixes the pressure unit of each system independently; see
# https://docs.lammps.org/units.html
PRESSURE_UNIT_IN_PA = {
    'real': 101325.0,       # atm
    'metal': 1e5,           # bar
    'si': 1.0,              # Pa
    'cgs': 0.1,             # dyne/cm^2
    'electron': 1.0,        # Pa
    'micro': 1e3,           # pg/(um*us^2)
    'nano': 1e6,            # ag/(nm*ns^2)
}

# The systems whose pressure unit is also their energy-per-volume unit.
ENERGY_PER_VOLUME = ('si', 'cgs', 'micro', 'nano')


class TestStressConversion(unittest.TestCase):

    def test_every_system_has_a_stress_factor(self):
        for name, factors in UNIT_SYSTEMS.items():
            self.assertIn('s_conv', factors, name)
            self.assertGreater(factors['s_conv'], 0.0, name)

    def test_factors_match_the_pressure_units(self):
        self.assertEqual(set(PRESSURE_UNIT_IN_PA), set(UNIT_SYSTEMS))
        for name, in_pa in PRESSURE_UNIT_IN_PA.items():
            expected = in_pa / EV_PER_ANG3_IN_PA
            self.assertAlmostEqual(
                UNIT_SYSTEMS[name]['s_conv'] / expected, 1.0, places=4,
                msg='{}: {} vs {}'.format(
                    name, UNIT_SYSTEMS[name]['s_conv'], expected))

    def test_energy_per_volume_systems_are_self_consistent(self):
        """For these four, pressure is energy over volume, so the
        stress factor must fall out of the energy and length factors.
        The other three (atm, bar, Pa) deliberately do not."""
        for name in ENERGY_PER_VOLUME:
            f = UNIT_SYSTEMS[name]
            derived = f['e_conv'] / f['l_conv']**3
            self.assertAlmostEqual(f['s_conv'] / derived, 1.0, places=4,
                                   msg=name)

    def test_the_other_three_are_not_energy_per_volume(self):
        """Guards against someone 'simplifying' the table by deriving
        all seven factors from e_conv and l_conv."""
        for name in ('real', 'metal', 'electron'):
            f = UNIT_SYSTEMS[name]
            derived = f['e_conv'] / f['l_conv']**3
            self.assertNotAlmostEqual(f['s_conv'] / derived, 1.0, places=1,
                                      msg=name)


class TestReaderUsesTheFactor(unittest.TestCase):

    def test_apply_unit_conversion_scales_stress(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        t = Trajectory(atoms=Atoms('H2'), timestep=1.)
        t.set_positions(np.zeros((3, 2, 3)))
        stress = np.arange(18, dtype=float).reshape(3, 6)
        t.set_stress(stress.copy())
        t.apply_unit_conversion(s_conv=UNIT_SYSTEMS['metal']['s_conv'])
        np.testing.assert_allclose(
            t.get_stress(),
            stress * UNIT_SYSTEMS['metal']['s_conv'])


if __name__ == '__main__':
    unittest.main()
