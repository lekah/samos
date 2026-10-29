# -*- coding: utf-8 -*-
"""
Tests for ADF over a StructureList, i.e. frames that need not share
their atoms.

Three things are being checked.  First, that nothing moved for a fixed
composition: the per-species lookup and the normalisation now happen
inside the frame loop, and a StructureList whose frames hold the same
atoms has to give what the equivalent Trajectory gives.  Second, that
varying composition works at all, that the atom indices really are
frame-local, and that the result is normalised the way this package
decided to normalise it -- averaged over every sampled frame, including
the ones holding none of the triplet's center species.  Third, that the
three sources of explicit bond topology, which are lists of global atom
indices, are still refused for frames that give an index no fixed
meaning.
"""

import os
import tempfile
import unittest

import numpy as np
from ase import Atoms

from samos.analysis.rdf import ADF
from samos.structurelist import StructureList
from samos.trajectory import Trajectory

# A right angle at the center atom, which is what every geometry below
# is built from: two bonds of 1.5 A along x and y.
CELL = 10.0
BONDS = {'Si-O': (1.0, 2.0)}


def _frame(symbols, positions, cell=CELL):
    """One ase.Atoms in a cubic cell."""
    return Atoms(symbols,
                 positions=np.array(positions, dtype=float),
                 cell=np.eye(3) * float(cell), pbc=True)


def _right_angle(symbols=('O', 'Si', 'O'), origin=(2.0, 5.0, 5.0),
                 cell=CELL):
    """
    A single O-Si-O unit subtending exactly 90 degrees at the Si.

    *symbols* gives the order the three atoms are stored in, so the
    same geometry can be written down with the center atom in any
    position -- which is how the frame-local indexing is tested.
    """
    x, y, z = origin
    place = {'Si': (x, y, z), 'O': None}
    legs = [(x - 1.5, y, z), (x, y + 1.5, z)]
    positions = []
    for symbol in symbols:
        if symbol == 'Si':
            positions.append(place['Si'])
        else:
            positions.append(legs.pop(0))
    return _frame(list(symbols), positions, cell=cell)


def _linear(origin=(2.0, 5.0, 5.0), cell=CELL):
    """
    A single O-Si-O unit subtending exactly 180 degrees at the Si.

    Needed next to the right angle so that a per-frame average and a
    pooled one are told apart: equal counts at two different angles
    come out differently from the two weightings.
    """
    x, y, z = origin
    return _frame(['O', 'Si', 'O'],
                  [(x - 1.5, y, z), (x, y, z), (x + 1.5, y, z)],
                  cell=cell)


def _gas(symbols, seed, cell=CELL):
    """
    A frame of randomly placed atoms, holding no Si.

    Takes a chemical formula, so the atom count comes from the built
    Atoms rather than from the length of the string.
    """
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=np.eye(3) * float(cell), pbc=True)
    atoms.set_positions(rng.random((len(atoms), 3)) * float(cell))
    return atoms


def _adf(frames, **kwargs):
    """Run an ADF over *frames* with per-frame bond detection."""
    kwargs.setdefault('bonds', BONDS)
    kwargs.setdefault('nbins', 180)
    analyzer = ADF(structures=StructureList.from_atoms(frames),
                   verbosity=0)
    return analyzer.run(**kwargs)


def _integral(res, label='O_Si_O'):
    """
    Angles counted per center atom per frame.

    The arrays are densities per degree, so summing them over the bins
    needs the bin width back; with 180 bins over 180 degrees that is
    1.0, but the test should not depend on the choice.
    """
    angles = res.get_array('angles_{}'.format(label))
    binsize = float(angles[1] - angles[0])
    return float(res.get_array('adf_{}'.format(label)).sum()) * binsize


def _peak(res, label='O_Si_O'):
    """The bin center holding the most weight."""
    adf = res.get_array('adf_{}'.format(label))
    return float(res.get_array('angles_{}'.format(label))[
        int(np.argmax(adf))])


class TestStructureListIsAccepted(unittest.TestCase):
    """
    ADF used to refuse a bare StructureList at construction, because
    its bond topology is a list of global atom indices.  Cutoff
    detection rebuilds that list from each frame's own atoms, so the
    requirement belongs to explicit topology alone.
    """

    def test_construction_no_longer_refuses_one(self):
        frames = [_gas('H8', seed=11), _gas('H10', seed=12)]
        analyzer = ADF(structures=StructureList.from_atoms(frames),
                       verbosity=0)
        self.assertEqual(analyzer.structures.nstep, 2)

    def test_a_trajectory_is_still_accepted(self):
        traj = Trajectory()
        traj.set_atoms(_right_angle())
        traj.set_positions(
            _right_angle().get_positions()[np.newaxis])
        self.assertIsNotNone(ADF(structures=traj, verbosity=0).structures)


class TestAgreesWithTrajectory(unittest.TestCase):
    """
    A StructureList whose frames do hold the same atoms is the same
    thing as a Trajectory, so it has to give the same answer; this is
    the check that moving the bookkeeping into the frame loop changed
    nothing.
    """

    def _both(self, nstep=4, seed=21):
        rng = np.random.default_rng(seed)
        atoms = _right_angle()
        base = atoms.get_positions()
        # Jitter small enough to keep every bond inside the cutoffs.
        offsets = 0.05 * rng.standard_normal((nstep, len(atoms), 3))
        traj = Trajectory()
        traj.set_atoms(atoms)
        traj.set_positions(base[np.newaxis] + offsets)
        frames = [_frame(atoms.get_chemical_symbols(), base + offset)
                  for offset in offsets]
        return traj, frames

    def test_same_adf(self):
        traj, frames = self._both()
        from_traj = ADF(structures=traj, verbosity=0).run(
            bonds=BONDS, nbins=180)
        from_list = _adf(frames)
        np.testing.assert_allclose(
            from_list.get_array('adf_O_Si_O'),
            from_traj.get_array('adf_O_Si_O'))
        np.testing.assert_allclose(
            from_list.get_array('angles_O_Si_O'),
            from_traj.get_array('angles_O_Si_O'))

    def test_same_triplets_reported(self):
        traj, frames = self._both()
        self.assertEqual(
            sorted(ADF(structures=traj, verbosity=0).run(
                bonds=BONDS).get_arraynames()),
            sorted(_adf(frames).get_arraynames()))


class TestFrameLocalIndices(unittest.TestCase):
    """
    The atom indices the bond detection returns index one frame's own
    atoms.  Storing the same geometry in a different order per frame is
    what tells a frame-local lookup from one carried over from an
    earlier frame: a global index would pick an O where the Si is.
    """

    def test_reordered_frames_give_the_same_angle(self):
        frames = [_right_angle(('O', 'Si', 'O')),
                  _right_angle(('Si', 'O', 'O')),
                  _right_angle(('O', 'O', 'Si'))]
        res = _adf(frames)
        self.assertAlmostEqual(_peak(res), 90.5, places=6)
        # One angle per Si per frame, and every frame has exactly one.
        self.assertAlmostEqual(_integral(res), 1.0, places=10)

    def test_a_shuffled_frame_is_not_read_through_the_first_one(self):
        ordered = _adf([_right_angle(('O', 'Si', 'O'))] * 3)
        shuffled = _adf([_right_angle(('O', 'Si', 'O')),
                         _right_angle(('Si', 'O', 'O')),
                         _right_angle(('O', 'O', 'Si'))])
        np.testing.assert_allclose(shuffled.get_array('adf_O_Si_O'),
                                   ordered.get_array('adf_O_Si_O'))


class TestPerFrameNormalisation(unittest.TestCase):
    """
    Each frame is divided by its own center count and the frames are
    averaged, so a frame with twice as many identical units counts the
    same as one with a single unit.  The old normalisation used one
    center count for the whole run, which is simply the wrong number as
    soon as the frames differ.
    """

    def test_center_count_may_differ_between_frames(self):
        # One right angle, then two straight ones far enough apart not
        # to bond to each other.
        bent = _right_angle(origin=(2.0, 2.0, 2.0))
        straight = _linear(origin=(2.0, 2.0, 2.0))
        straight.extend(_linear(origin=(2.0, 7.0, 2.0)))
        res = _adf([bent, straight])

        adf = res.get_array('adf_O_Si_O')
        angles = res.get_array('angles_O_Si_O')
        binsize = float(angles[1] - angles[0])
        bent_bin = int(np.argmin(np.abs(angles - 90.5)))
        straight_bin = int(np.argmin(np.abs(angles - 179.5)))

        # Each frame contributes one angle per center atom and the two
        # frames are averaged, so the weight splits evenly even though
        # the second frame holds twice the atoms.  Pooling the counts
        # over the whole run instead -- three angles, three centers --
        # would put a third of the weight on the right angle and
        # two thirds on the straight one.
        self.assertAlmostEqual(adf[bent_bin] * binsize, 0.5, places=10)
        self.assertAlmostEqual(adf[straight_bin] * binsize, 0.5,
                               places=10)
        self.assertAlmostEqual(_integral(res), 1.0, places=10)

    def test_a_frame_without_the_center_still_counts(self):
        res = _adf([_right_angle(), _gas('H8', seed=31)])
        # Half the frames hold an Si, so the documented convention is
        # half an angle per center per frame -- not one, which would be
        # the conditional distribution over the frames holding Si.
        self.assertAlmostEqual(_integral(res), 0.5, places=10)

    def test_a_center_absent_from_the_first_frame_is_reported(self):
        # Reading the center count off frame 0, as the old code did,
        # drops this triplet from the output altogether.
        res = _adf([_gas('H8', seed=41), _right_angle()])
        self.assertIn('adf_O_Si_O', res.get_arraynames())
        self.assertAlmostEqual(_integral(res), 0.5, places=10)

    def test_an_absent_triplet_is_left_out(self):
        # Asked for explicitly, so that it is dropped for want of a
        # center atom rather than never requested in the first place.
        res = _adf([_gas('H8', seed=51), _gas('H6', seed=52)],
                   species_triplets=[('O', 'Si', 'O')])
        self.assertEqual(res.get_arraynames(), [])


class TestVaryingCell(unittest.TestCase):
    """
    The minimum-image reduction is hoisted out of the frame loop only
    when the cell is the same in every frame, which a structure list
    makes no promise about.
    """

    def test_a_different_cell_per_frame(self):
        res = _adf([_right_angle(cell=10.0), _right_angle(cell=12.0),
                    _right_angle(cell=14.0)])
        self.assertAlmostEqual(_peak(res), 90.5, places=6)
        self.assertAlmostEqual(_integral(res), 1.0, places=10)

    def test_a_fixed_cell_gives_the_same_answer(self):
        varying = _adf([_right_angle(cell=10.0), _right_angle(cell=12.0)])
        fixed = _adf([_right_angle(cell=10.0), _right_angle(cell=10.0)])
        # The geometry is identical and well inside both cells, so the
        # hoisted and the per-frame reduction have to agree.
        np.testing.assert_allclose(varying.get_array('adf_O_Si_O'),
                                   fixed.get_array('adf_O_Si_O'))


class TestExplicitTopologyStillNeedsStableIndices(unittest.TestCase):
    """
    set_bonds, load_bonds_lammps and static_bonds all identify atoms by
    a global index, so each has to refuse frames that give an index no
    fixed meaning -- and accept frames that do, whether those come as a
    Trajectory or as a StructureList that happens to be uniform.
    """

    def _mixed(self):
        return StructureList.from_atoms(
            [_gas('H8', seed=61), _gas('H10', seed=62)])

    def _uniform(self):
        return StructureList.from_atoms([_right_angle()] * 3)

    def _bonds_file(self):
        handle = tempfile.NamedTemporaryFile(
            'w', suffix='.data', delete=False)
        handle.write('Bonds\n\n1 1 1 2\n2 1 2 3\n')
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_set_bonds_is_refused(self):
        analyzer = ADF(structures=self._mixed(), verbosity=0)
        with self.assertRaises(TypeError) as ctx:
            analyzer.set_bonds([[0, 1]])
        message = str(ctx.exception)
        self.assertIn('same atoms in the same order', message)
        self.assertIn('global index', message)

    def test_load_bonds_lammps_is_refused(self):
        analyzer = ADF(structures=self._mixed(), verbosity=0)
        with self.assertRaises(TypeError):
            analyzer.load_bonds_lammps(self._bonds_file())

    def test_static_bonds_is_refused(self):
        analyzer = ADF(structures=self._mixed(), verbosity=0)
        with self.assertRaises(TypeError) as ctx:
            analyzer.run(bonds={'H-H': (0.0, 3.0)}, static_bonds=True)
        self.assertIn('static_bonds', str(ctx.exception))

    def test_bonds_set_first_then_mixed_frames_is_refused(self):
        # The two can be set in either order, so the check has to fire
        # from whichever side comes second.
        analyzer = ADF(bonds=[[0, 1]], verbosity=0)
        with self.assertRaises(TypeError):
            analyzer.set_structures(self._mixed())

    def test_something_that_is_not_frames_at_all_still_type_errors(self):
        analyzer = ADF(bonds=[[0, 1]], verbosity=0)
        with self.assertRaises(TypeError):
            analyzer.set_structures('not a structure list')

    def test_a_uniform_structure_list_is_accepted(self):
        analyzer = ADF(structures=self._uniform(), verbosity=0)
        analyzer.set_bonds([[0, 1], [1, 2]])
        res = analyzer.run(nbins=180)
        self.assertAlmostEqual(_peak(res), 90.5, places=6)

    def test_static_bonds_works_on_a_uniform_structure_list(self):
        analyzer = ADF(structures=self._uniform(), verbosity=0)
        res = analyzer.run(bonds=BONDS, nbins=180, static_bonds=True)
        self.assertAlmostEqual(_integral(res), 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
