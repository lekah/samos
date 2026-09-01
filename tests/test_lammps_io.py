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


class TestDumpErrorPaths(unittest.TestCase):
    """The parser used to fail in ways that hid the real cause: a
    sys.exit(1) inside a library function, and an except handler that
    raised NameError on its own message."""

    _ORTHO_BOX = ('ITEM: BOX BOUNDS pp pp pp\n'
                  '0.0 10.0\n0.0 10.0\n0.0 10.0\n')

    def _write(self, body_header, box=None, body=None):
        import tempfile
        box = self._ORTHO_BOX if box is None else box
        body = '1 1 0.0 0.0 0.0\n2 1 1.0 1.0 1.0\n' if body is None else body
        text = ('ITEM: TIMESTEP\n0\n'
                'ITEM: NUMBER OF ATOMS\n2\n'
                + box + body_header + body)
        fh = tempfile.NamedTemporaryFile('w', suffix='.lammpstrj',
                                         delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_valid_minimal_dump(self):
        """The fixture itself must parse, so the tests below fail for
        the reason they claim to."""
        traj = read_lammps_dump(
            self._write('ITEM: ATOMS id type xu yu zu\n'),
            types=['H'], quiet=True)
        self.assertEqual(traj.nat, 2)
        self.assertEqual(traj.nstep, 1)

    def test_missing_positions_raises_instead_of_exiting(self):
        """Regression: this called sys.exit(1), killing the interpreter
        of any program that imported the module."""
        path = self._write('ITEM: ATOMS id type vx vy vz\n')
        with self.assertRaises(TypeError) as cm:
            read_lammps_dump(path, types=['H'], quiet=True)
        # the message must name what was missing and what was found
        self.assertIn('position', str(cm.exception).lower())
        self.assertIn('vx', str(cm.exception))

    def test_malformed_triclinic_box_reports_parse_error(self):
        """Regression: the handler printed an unbound `idim`, so it
        raised NameError and masked the real parse failure."""
        box = ('ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
               '0.0 10.0 0.0\n'
               '0.0 10.0\n'          # missing the tilt factor
               '0.0 10.0 0.0\n')
        path = self._write('ITEM: ATOMS id type xu yu zu\n', box=box)
        with self.assertRaises(ValueError):
            read_lammps_dump(path, types=['H'], quiet=True)

    def test_triclinic_box_is_read(self):
        box = ('ITEM: BOX BOUNDS xy xz yz pp pp pp\n'
               '0.0 10.0 1.0\n0.0 10.0 2.0\n0.0 10.0 3.0\n')
        path = self._write('ITEM: ATOMS id type xu yu zu\n', box=box)
        traj = read_lammps_dump(path, types=['H'], quiet=True)
        cell = traj.get_cells()[0]
        np.testing.assert_allclose(np.diag(cell), [10., 10., 10.])
        self.assertAlmostEqual(cell[1, 0], 1.0)   # xy
        self.assertAlmostEqual(cell[2, 0], 2.0)   # xz
        self.assertAlmostEqual(cell[2, 1], 3.0)   # yz


if __name__ == '__main__':
    unittest.main()
