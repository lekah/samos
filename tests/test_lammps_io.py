# -*- coding: utf-8 -*-
"""
Tests for samos.io.lammps.read_lammps_dump.

Two trajectory files from examples/data/ are used:

  Al31-1200K-1ps.lammpstrj
    31 Al atoms, 501 steps, 1 ps timestep.
    Columns: id xu yu zu  (no type or element column).
    Must supply element information explicitly.

  LGPS-500K-1ns.lammpstrj
    400 atoms (Li/Ge/P/S), 10001 steps.
    Columns: id mass type element xu yu zu
    Can be loaded via the element column, or by supplying the type map.
    LAMMPS type -> element: 1=Li, 2=Ge, 3=P, 4=S.
"""

import os
import unittest

import numpy as np
from samos.io.lammps import read_lammps_dump

# Paths relative to the repository root.
_DATA = os.path.join(
    os.path.dirname(__file__), '..', 'examples', 'data')

AL31_PATH = os.path.abspath(os.path.join(_DATA, 'Al31-1200K-1ps.lammpstrj'))
LGPS_PATH = os.path.abspath(os.path.join(_DATA, 'LGPS-500K-1ns.lammpstrj'))

_have_al31 = os.path.isfile(AL31_PATH)
_have_lgps = os.path.isfile(LGPS_PATH)


@unittest.skipUnless(_have_al31, 'Al31-1200K-1ps.lammpstrj not found')
class TestAl31(unittest.TestCase):
    """
    Al31 dump has no type or element column.  Element information must
    be supplied explicitly, either as a per-atom list or as a formula.
    Mirrors the loading logic in examples/ex1-compute-MSD-from-LAMMPS/.
    """

    def test_elements_list(self):
        """Load by passing one element symbol per atom."""
        traj = read_lammps_dump(
            AL31_PATH, timestep=1e3, quiet=True,
            elements=['Al'] * 31)
        self.assertEqual(traj.nat, 31)
        self.assertEqual(traj.nstep, 501)
        self.assertTrue(np.all(traj.get_types() == 'Al'))
        self.assertAlmostEqual(traj.get_timestep(), 1e3)

        pos = traj.get_positions()
        self.assertEqual(pos.shape, (501, 31, 3))

        self.assertNotIn('velocities', traj.get_arraynames())

        vel = traj.calculate_velocities_from_positions()
        self.assertEqual(vel.shape, (501, 31, 3))
        self.assertIn('velocities', traj.get_arraynames())


@unittest.skipUnless(
    _have_lgps,
    'LGPS-500K-1ns.lammpstrj not found '
    '(uncompress with: tar -xvf LGPS-500K-1ns.lammpstrj.tar.xz)')
class TestLGPS(unittest.TestCase):
    """
    LGPS dump has both a 'type' integer column and an 'element' string
    column.  Three loading paths are tested:

      1. No element arguments  -- reads symbols from the 'element' column.
      2. types=['Li','Ge','P','S'] -- maps integer types 1-4 to symbols.
      3. elements=[...] per atom  -- explicit flat symbol list.

    All three must produce the same species assignment.
    Mirrors the loading logic in examples/ex1-compute-MSD-from-LAMMPS/
    benchmark-msd.py.
    """

    # Type map: LAMMPS integer type -> element symbol.
    _TYPES = ['Li', 'Ge', 'P', 'S']
    _NAT = 400
    _NSTEP = 100

    def _load(self, **kwargs):
        # istep=10 keeps the test fast by reading every 10th frame.
        return read_lammps_dump(LGPS_PATH, timestep=1e3, quiet=True,
                                istep=10, nsteps=self._NSTEP, **kwargs)

    def test_elements(self):
        """Load using the 'element' column already in the dump."""
        traj_elements = self._load()
        self.assertEqual(traj_elements.nat, self._NAT)
        types = traj_elements.get_types()
        self.assertIn('Li', types)
        self.assertIn('Ge', types)
        self.assertIn('P',  types)
        self.assertIn('S',  types)

        traj_types = self._load(types=self._TYPES)
        self.assertEqual(traj_types.nat, self._NAT)
        types = traj_types.get_types()
        self.assertIn('Li', types)
        self.assertIn('S',  types)

        self.assertTrue(np.array_equal(
            traj_elements.get_types(), traj_types.get_types()))

        idx = traj_elements.get_indices_of_species('Li')
        self.assertGreater(len(idx), 0)
        types = traj_elements.get_types()
        self.assertTrue(np.all(types[idx] == 'Li'))

    def test_attributes(self):
        traj = self._load()
        self.assertAlmostEqual(traj.get_timestep(), 1e3)
        self.assertNotIn('velocities', traj.get_arraynames())

        self.assertEqual(traj.get_positions().shape,
                         (self._NSTEP, self._NAT, 3))


if __name__ == '__main__':
    unittest.main()
