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
import tempfile
import unittest

import numpy as np
from samos.io.lammps import (
    ChangingAtomsInDump,
    _build_parser as lammps_parser,
    main as lammps_main,
    read_lammps_dump,
)
from samos.structurelist import StructureList
from samos.trajectory import Trajectory

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


class TestDumpParsingRobustness(unittest.TestCase):
    """The float pattern demanded a decimal point and a lowercase e, and
    a truncated file surfaced as numpy's inhomogeneous-shape error."""

    def _write(self, text):
        import tempfile
        fh = tempfile.NamedTemporaryFile('w', suffix='.lammpstrj',
                                         delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def _dump(self, box, body='1 1 0.0 0.0 0.0\n2 1 1.0 1.0 1.0\n'):
        return self._write('ITEM: TIMESTEP\n0\n'
                           'ITEM: NUMBER OF ATOMS\n2\n'
                           + box
                           + 'ITEM: ATOMS id type xu yu zu\n' + body)

    def test_integer_box_bounds(self):
        """LAMMPS can write bounds with no decimal point at all."""
        path = self._dump('ITEM: BOX BOUNDS pp pp pp\n'
                          '0 10\n0 10\n0 10\n')
        traj = read_lammps_dump(path, types=['H'], quiet=True)
        np.testing.assert_allclose(np.diag(traj.get_cells()[0]),
                                   [10., 10., 10.])

    def test_uppercase_exponent_box_bounds(self):
        path = self._dump('ITEM: BOX BOUNDS pp pp pp\n'
                          '0.0E+00 1.0E+01\n'
                          '0.0E+00 1.0E+01\n'
                          '0.0E+00 1.0E+01\n')
        traj = read_lammps_dump(path, types=['H'], quiet=True)
        np.testing.assert_allclose(np.diag(traj.get_cells()[0]),
                                   [10., 10., 10.])

    def test_truncated_frame_names_the_line(self):
        path = self._dump('ITEM: BOX BOUNDS pp pp pp\n'
                          '0.0 10.0\n0.0 10.0\n0.0 10.0\n',
                          body='1 1 0.0 0.0 0.0\n')   # second atom missing
        with self.assertRaises(ValueError) as cm:
            read_lammps_dump(path, types=['H'], quiet=True)
        self.assertIn('atom lines', str(cm.exception))

    def test_ragged_frame_names_the_line(self):
        path = self._dump('ITEM: BOX BOUNDS pp pp pp\n'
                          '0.0 10.0\n0.0 10.0\n0.0 10.0\n',
                          body='1 1 0.0 0.0 0.0\n2 1 1.0 1.0\n')
        with self.assertRaises(ValueError) as cm:
            read_lammps_dump(path, types=['H'], quiet=True)
        self.assertIn('columns', str(cm.exception))


class TestChangingAtoms(unittest.TestCase):
    """
    A dump whose frames hold different atoms used to be truncated at
    the first such frame, with a message that only appeared when the
    read was not quiet.  What is done with it is now the caller's
    choice, and silence is not one of the options.
    """

    _BOX = ('ITEM: BOX BOUNDS pp pp pp\n'
            '0.0 10.0\n0.0 10.0\n0.0 10.0\n')

    def _dump(self, frames, header='ITEM: ATOMS id type xu yu zu\n'):
        """
        Write a dump of several frames.

        :param frames: One list of atom lines per frame, without the
            newlines.  The atom count written in each frame's header is
            taken from the lines given, so a frame that grows or
            shrinks needs nothing else said about it.
        """
        import tempfile
        text = ''
        for istep, body in enumerate(frames):
            text += ('ITEM: TIMESTEP\n{}\n'
                     'ITEM: NUMBER OF ATOMS\n{}\n'.format(
                         istep, len(body))
                     + self._BOX + header
                     + ''.join(line + '\n' for line in body))
        fh = tempfile.NamedTemporaryFile('w', suffix='.lammpstrj',
                                         delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def _growing(self):
        """Two frames, the second holding one atom more."""
        return self._dump([
            ['1 1 0.0 0.0 0.0', '2 1 1.0 1.0 1.0'],
            ['1 1 0.1 0.0 0.0', '2 1 1.1 1.0 1.0', '3 1 2.0 2.0 2.0'],
        ])

    def _reacting(self):
        """Two frames of two atoms, one of which changes species."""
        return self._dump([
            ['1 1 0.0 0.0 0.0', '2 1 1.0 1.0 1.0'],
            ['1 1 0.1 0.0 0.0', '2 2 1.1 1.0 1.0'],
        ])

    def _uniform(self):
        return self._dump([
            ['1 1 0.0 0.0 0.0', '2 1 1.0 1.0 1.0'],
            ['1 1 0.1 0.0 0.0', '2 1 1.1 1.0 1.0'],
        ])

    def test_a_changing_count_is_refused_by_default(self):
        with self.assertRaises(ChangingAtomsInDump) as cm:
            read_lammps_dump(self._growing(), types=['H', 'He'],
                             quiet=True)
        message = str(cm.exception)
        self.assertIn('Frame 1', message)
        self.assertIn('3 atoms', message)
        self.assertIn('structures', message)

    def test_a_changing_species_is_refused_by_default(self):
        """The atom count alone does not say whether the atoms are the
        same ones: a reacting run can swap species and keep its size."""
        with self.assertRaises(ChangingAtomsInDump) as cm:
            read_lammps_dump(self._reacting(), types=['H', 'He'],
                             quiet=True)
        self.assertIn('species', str(cm.exception))
        self.assertIn('H -> He', str(cm.exception))

    def test_truncate_keeps_the_frames_before_the_change(self):
        for path in (self._growing(), self._reacting()):
            with self.subTest(path=path):
                with self.assertWarns(UserWarning) as cm:
                    traj = read_lammps_dump(
                        path, types=['H', 'He'], quiet=True,
                        on_changing_atoms='truncate')
                self.assertIsInstance(traj, Trajectory)
                self.assertEqual(traj.nstep, 1)
                self.assertEqual(traj.nat, 2)
                self.assertIn('Frame 1', str(cm.warning))

    def test_structures_reads_every_frame(self):
        structures = read_lammps_dump(
            self._growing(), types=['H', 'He'], quiet=True,
            on_changing_atoms='structures')
        self.assertIsInstance(structures, StructureList)
        self.assertEqual(structures.nstep, 2)
        self.assertEqual(structures.get_frame_natoms(0), 2)
        self.assertEqual(structures.get_frame_natoms(1), 3)
        self.assertFalse(structures.has_uniform_composition)
        np.testing.assert_array_equal(
            structures.get_frame_positions(1),
            [[0.1, 0., 0.], [1.1, 1., 1.], [2., 2., 2.]])
        np.testing.assert_array_equal(
            structures.get_frame_cell(0), np.eye(3) * 10.)

    def test_structures_follows_a_changing_species(self):
        structures = read_lammps_dump(
            self._reacting(), types=['H', 'He'], quiet=True,
            on_changing_atoms='structures')
        self.assertEqual(list(structures.get_frame_types(0)), ['H', 'H'])
        self.assertEqual(list(structures.get_frame_types(1)), ['H', 'He'])

    def test_structures_reads_an_element_column_per_frame(self):
        """The column every reactive dump writes: the symbols come from
        the frame being read, not from the first frame."""
        path = self._dump(
            [['1 H 0.0 0.0 0.0', '2 O 1.0 1.0 1.0'],
             ['1 H 0.1 0.0 0.0', '2 O 1.1 1.0 1.0', '3 H 2.0 2.0 2.0']],
            header='ITEM: ATOMS id element xu yu zu\n')
        structures = read_lammps_dump(path, quiet=True,
                                      on_changing_atoms='structures')
        self.assertEqual(list(structures.get_frame_types(0)), ['H', 'O'])
        self.assertEqual(list(structures.get_frame_types(1)),
                         ['H', 'O', 'H'])
        self.assertEqual(sorted(structures.get_species()), ['H', 'O'])

    def test_a_uniform_dump_is_a_trajectory_under_every_policy(self):
        for policy in ('error', 'truncate', 'structures'):
            with self.subTest(policy=policy):
                traj = read_lammps_dump(
                    self._uniform(), types=['H', 'He'], quiet=True,
                    on_changing_atoms=policy)
                self.assertIsInstance(traj, Trajectory)
                self.assertEqual(traj.nstep, 2)
                self.assertEqual(list(traj.types), ['H', 'H'])

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            read_lammps_dump(self._uniform(), types=['H'], quiet=True,
                             on_changing_atoms='carry on')
        self.assertIn('truncate', str(cm.exception))

    def test_an_explicit_element_list_needs_a_fixed_count(self):
        """The list is positional, so there is no honest way to apply a
        two-entry list to a frame of three atoms."""
        with self.assertRaises(ChangingAtomsInDump) as cm:
            read_lammps_dump(self._growing(), elements=['H', 'H'],
                             quiet=True, on_changing_atoms='structures')
        self.assertIn('positional', str(cm.exception))

    def test_an_explicit_element_list_survives_truncation(self):
        """The frames that are kept are all of the size the list
        describes, so the frame that is not is never named."""
        with self.assertWarns(UserWarning):
            traj = read_lammps_dump(self._growing(), elements=['H', 'H'],
                                    quiet=True,
                                    on_changing_atoms='truncate')
        self.assertEqual(traj.nstep, 1)
        self.assertEqual(list(traj.types), ['H', 'H'])

    def test_velocities_are_kept_frame_by_frame(self):
        path = self._dump(
            [['1 1 0.0 0.0 0.0 0.1 0.0 0.0',
              '2 1 1.0 1.0 1.0 0.0 0.1 0.0'],
             ['1 1 0.1 0.0 0.0 0.1 0.0 0.0',
              '2 1 1.1 1.0 1.0 0.0 0.1 0.0',
              '3 1 2.0 2.0 2.0 0.0 0.0 0.1']],
            header='ITEM: ATOMS id type xu yu zu vx vy vz\n')
        structures = read_lammps_dump(
            path, types=['H'], quiet=True,
            on_changing_atoms='structures')
        np.testing.assert_allclose(
            structures.get_frame_array('velocities', 1),
            [[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1]])
        np.testing.assert_allclose(
            structures.get_frame_atoms(1).get_velocities(),
            [[0.1, 0., 0.], [0., 0.1, 0.], [0., 0., 0.1]])

    def test_the_type_map_must_cover_the_types_in_the_dump(self):
        """A type the map does not reach used to index past its end, or
        silently wrap round to the last symbol for type 0."""
        with self.assertRaises(ValueError) as cm:
            read_lammps_dump(self._reacting(), types=['H'], quiet=True,
                             on_changing_atoms='structures')
        self.assertIn('symbol', str(cm.exception))


class TestRaggedExport(unittest.TestCase):
    """
    The export a vented run needs: a dump that loses atoms has to
    reach extxyz carrying everything a dump of constant size carries,
    ids above all, since once atoms leave the box an index says
    nothing about which atom it is.
    """

    _BOX = ('ITEM: BOX BOUNDS pp pp pp\n'
            '0.0 10.0\n0.0 10.0\n0.0 10.0\n')
    _HEADER = 'ITEM: ATOMS id type xu yu zu vx vy vz v_gam\n'

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _atom(self, iat, itype, step, gamma):
        """One atom line, its position walking with the step."""
        return '{} {} {} {} {} 0.1 0.2 0.3 {}'.format(
            iat, itype, 0.5 * iat + 0.01 * step, 0.5 * iat, 0.5, gamma)

    def _dump(self, name='verlet.trj', nframe=4, nat=5, lost_after=2):
        """
        A dump of *nat* atoms, the last of which leaves after frame
        *lost_after*, as fix evaporate would take it out.
        """
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as handle:
            for step in range(nframe):
                here = nat if step <= lost_after else nat - 1
                body = [self._atom(iat + 1, 1 if iat else 2, step,
                                   0.01 * (iat + step))
                        for iat in range(here)]
                handle.write(
                    'ITEM: TIMESTEP\n{}\n'
                    'ITEM: NUMBER OF ATOMS\n{}\n'.format(
                        100 * step, len(body))
                    + self._BOX + self._HEADER
                    + ''.join(line + '\n' for line in body))
        return path

    def _thermo(self, name='verlet-1.evp', nframe=4):
        """A thermo file with one row per frame of the dump."""
        path = os.path.join(self.tmpdir, name)
        with open(path, 'w') as handle:
            handle.write('# a comment line\n')
            handle.write('# TimeStep v_maxgam\n')
            for step in range(nframe):
                handle.write('{} {}\n'.format(100 * step,
                                              1.1 + 0.01 * step))
        return path

    def _export(self, extra=(), **kwargs):
        """Run the module's own command line and read the result back."""
        from ase.io import read as ase_read
        out = os.path.join(self.tmpdir, 'out.extxyz')
        argv = [kwargs.pop('dump', None) or self._dump(),
                '--thermo-file', self._thermo(),
                '-t', 'He', 'B', '-o', out,
                '--changing-atoms', 'structures',
                '--ignore-forces', '-a', 'id', 'v_gam',
                '-tk', 'v_maxgam', '-q']
        argv.extend(extra)
        lammps_main(argv)
        return ase_read(out, index=':'), out

    def test_every_frame_is_written(self):
        frames, _ = self._export()
        self.assertEqual(len(frames), 4)
        self.assertEqual([len(atoms) for atoms in frames], [5, 5, 5, 4])

    def test_ids_say_which_atom_left(self):
        frames, _ = self._export()
        self.assertEqual(list(frames[0].arrays['id']), [1, 2, 3, 4, 5])
        self.assertEqual(list(frames[3].arrays['id']), [1, 2, 3, 4])
        # An id is a label, so it has to survive as one: written as a
        # float it comes back as 1.0 and no longer indexes anything.
        self.assertTrue(
            np.issubdtype(frames[0].arrays['id'].dtype, np.integer))

    def test_velocities_and_columns_reach_the_file(self):
        frames, path = self._export()
        np.testing.assert_allclose(frames[3].get_velocities(),
                                   [[0.1, 0.2, 0.3]] * 4)
        np.testing.assert_allclose(frames[3].arrays['v_gam'],
                                   [0.03, 0.04, 0.05, 0.06])
        with open(path) as handle:
            handle.readline()
            comment = handle.readline()
        for column in ('species:S:1', 'pos:R:3', 'momenta:R:3',
                       'v_gam:R:1', 'id:I:1'):
            self.assertIn(column, comment)

    def test_thermo_keywords_reach_each_frames_info(self):
        frames, _ = self._export()
        for step, atoms in enumerate(frames):
            self.assertAlmostEqual(atoms.info['v_maxgam'],
                                   1.1 + 0.01 * step)

    @staticmethod
    def _frames_text(path):
        """An extxyz file split into one block of text per frame."""
        blocks = []
        with open(path) as handle:
            while True:
                count = handle.readline()
                if not count.strip():
                    return blocks
                lines = [count, handle.readline()]
                lines += [handle.readline()
                          for _ in range(int(count))]
                blocks.append(''.join(lines))

    def test_a_frame_of_a_vented_run_looks_like_any_other(self):
        """
        The contract itself: the frames a vented run shares with a
        constant one have to come out byte for byte the same, whether
        they were written as a trajectory or as a structure list, or
        everything downstream needs two code paths.
        """
        dump = self._dump()
        common = [dump, '--thermo-file', self._thermo(),
                  '-t', 'He', 'B', '--ignore-forces',
                  '-a', 'id', 'v_gam', '-tk', 'v_maxgam', '-q']
        paths = {}
        for policy in ('truncate', 'structures'):
            paths[policy] = os.path.join(
                self.tmpdir, '{}.extxyz'.format(policy))
            argv = common + ['-o', paths[policy],
                             '--changing-atoms', policy]
            if policy == 'truncate':
                with self.assertWarns(UserWarning):
                    result = lammps_main(argv)
                self.assertIsInstance(result, Trajectory)
            else:
                result = lammps_main(argv)
                self.assertIsInstance(result, StructureList)

        as_trajectory = self._frames_text(paths['truncate'])
        as_structures = self._frames_text(paths['structures'])
        self.assertEqual(len(as_trajectory), 3)
        self.assertEqual(len(as_structures), 4)
        self.assertEqual(as_trajectory, as_structures[:3])

    def test_the_policy_reaches_the_reader_from_the_command_line(self):
        args = lammps_parser().parse_args(
            ['dump.trj', '--changing-atoms', 'truncate'])
        self.assertEqual(args.on_changing_atoms, 'truncate')

    def test_the_default_policy_is_to_refuse(self):
        args = lammps_parser().parse_args(['dump.trj'])
        self.assertEqual(args.on_changing_atoms, 'error')


if __name__ == '__main__':
    unittest.main()
