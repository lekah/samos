# -*- coding: utf-8 -*-
"""
Tests for RDF over a StructureList, i.e. frames that need not share
their atoms.

Two things are being checked.  First, that nothing moved: the
per-species bookkeeping now happens inside the frame loop instead of
once before it, and for a fixed composition that has to come out the
same, which the stored fixture pins down.  Second, that varying
composition now works at all, and that it is normalised the way this
package decided to normalise it -- averaged over every sampled frame,
including the ones that hold none of the species in question.
"""

import json
import os
import unittest

import numpy as np
from ase import Atoms

from samos.analysis.rdf import ADF, RDF
from samos.structurelist import StructureList
from samos.trajectory import Trajectory

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'ref', 'rdf_regression.json')


def _traj(symbols, nstep, cell, seed, vary_cell=False):
    """Matches the generator that produced the stored fixture."""
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=cell, pbc=True)
    traj = Trajectory(atoms=atoms)
    traj.set_positions(rng.random((nstep, len(atoms), 3))
                       @ np.asarray(cell))
    if vary_cell:
        scales = 1.0 + 0.05 * np.arange(nstep)
        traj.set_cells(np.array([np.asarray(cell) * s for s in scales]))
    return traj


def _fixture_cases():
    return {
        'fixed_ortho': (
            _traj('Li8O16', 12, np.diag([10.0, 10.0, 10.0]), 1),
            dict(radius=4.0, nbins=40)),
        'varying_cell': (
            _traj('Li8O16', 8, np.diag([10.0, 11.0, 12.0]), 2,
                  vary_cell=True),
            dict(radius=3.5, nbins=25)),
        'skewed': (
            _traj('H6He6', 6,
                  [[9.0, 0.0, 0.0], [2.5, 9.0, 0.0], [1.0, 1.5, 9.0]], 3),
            dict(radius=3.0, nbins=20)),
        'strided_subset': (
            _traj('Li8O16', 11, np.diag([10.0, 10.0, 10.0]), 4),
            dict(radius=4.0, nbins=30, istart=1, istop=10, stepsize=4,
                 species_pairs=[('Li', 'O'), ('O', 'O')])),
    }


def _gas_frames(nstep, symbols, length, seed=0):
    """Frames of an ideal gas, as a list of ase.Atoms."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(nstep):
        atoms = Atoms(symbols, cell=np.eye(3) * length, pbc=True)
        atoms.set_positions(rng.random((len(atoms), 3)) * length)
        frames.append(atoms)
    return frames


def _plateau(res, label, rmin):
    radii = res.get_array('radii_{}'.format(label))
    return res.get_array('rdf_{}'.format(label))[radii > rmin].mean()


class TestFixedCompositionIsUnchanged(unittest.TestCase):
    """
    The guard against silent normalisation drift.  The fixture was
    generated from the version of RDF.run that worked its species
    bookkeeping out once, before the frame loop.
    """

    @classmethod
    def setUpClass(cls):
        with open(REF_PATH) as handle:
            cls.ref = json.load(handle)

    def test_integrals_are_bit_identical(self):
        # int_ and radii_ involve no per-frame division, so these must
        # match exactly, not merely closely.
        for name, (traj, kwargs) in _fixture_cases().items():
            res = RDF(structures=traj, verbosity=0).run(**kwargs)
            for key, expected in self.ref[name]['arrays'].items():
                if key.startswith('rdf_'):
                    continue
                np.testing.assert_array_equal(
                    res.get_array(key), np.array(expected),
                    err_msg='{}/{}'.format(name, key))

    def test_g_of_r_matches_to_rounding(self):
        # The ideal-gas reference now divides inside the frame loop
        # rather than once at the end, so the last couple of digits
        # move.  Nothing else should.
        for name, (traj, kwargs) in _fixture_cases().items():
            res = RDF(structures=traj, verbosity=0).run(**kwargs)
            for key, expected in self.ref[name]['arrays'].items():
                if not key.startswith('rdf_'):
                    continue
                np.testing.assert_allclose(
                    res.get_array(key), np.array(expected),
                    rtol=1e-12, atol=0.0,
                    err_msg='{}/{}'.format(name, key))

    def test_reported_pairs_and_distances_are_unchanged(self):
        for name, (traj, kwargs) in _fixture_cases().items():
            res = RDF(structures=traj, verbosity=0).run(**kwargs)
            expected = self.ref[name]['attrs']
            self.assertEqual(
                [list(pair) for pair in res.get_attr('species_pairs')],
                expected['species_pairs'], msg=name)
            for key, value in expected.items():
                if key.startswith(('shortest_distance', 'n_pairs_')):
                    self.assertAlmostEqual(res.get_attr(key), value,
                                           places=12,
                                           msg='{}/{}'.format(name, key))

    def test_n_data_now_counts_every_sampled_frame(self):
        # Deliberate fix, not drift.  n_data used to be
        # n_pairs * ((istop - istart) // stepsize), which floors where
        # the frame list ceils.  istart=1, istop=10, stepsize=4 samples
        # frames 1, 5 and 9 -- three of them -- but the old expression
        # said two.
        traj, kwargs = _fixture_cases()['strided_subset']
        res = RDF(structures=traj, verbosity=0).run(**kwargs)
        old = self.ref['strided_subset']['attrs']
        self.assertEqual(old['n_data_Li_O'], 256)
        self.assertEqual(res.get_attr('n_data_Li_O'),
                         res.get_attr('n_pairs_Li_O') * 3)
        self.assertEqual(res.get_attr('n_data_Li_O'), 384)

    def test_unstrided_n_data_is_untouched(self):
        for name in ('fixed_ortho', 'varying_cell', 'skewed'):
            traj, kwargs = _fixture_cases()[name]
            res = RDF(structures=traj, verbosity=0).run(**kwargs)
            for key, value in self.ref[name]['attrs'].items():
                if key.startswith('n_data_'):
                    self.assertEqual(res.get_attr(key), value,
                                     msg='{}/{}'.format(name, key))


class TestTrajectoryAndStructureListAgree(unittest.TestCase):
    """
    The bridge.  Same frames, same numbers, whichever layout holds
    them -- which is the whole claim of the shared frame interface.
    """

    def _both(self, **kwargs):
        traj = _traj('Li8O16', 10, np.diag([10.0, 10.0, 10.0]), 7)
        # Trajectory is iterable over its frames, so this is the same
        # structures read back out through the shared interface.
        structures = StructureList.from_atoms(list(traj))
        return (RDF(structures=traj, verbosity=0).run(**kwargs),
                RDF(structures=structures, verbosity=0).run(**kwargs))

    def test_same_g_of_r_and_integrals(self):
        from_traj, from_list = self._both(radius=4.0, nbins=40)
        self.assertEqual(sorted(from_traj.get_arraynames()),
                         sorted(from_list.get_arraynames()))
        for key in from_traj.get_arraynames():
            np.testing.assert_allclose(
                from_list.get_array(key), from_traj.get_array(key),
                rtol=1e-12, atol=1e-15, err_msg=key)

    def test_same_reported_attributes(self):
        from_traj, from_list = self._both(radius=4.0, nbins=40)
        for key, value in from_traj.get_attrs().items():
            if key == 'species_pairs':
                self.assertEqual(
                    [tuple(p) for p in from_list.get_attr(key)],
                    [tuple(p) for p in value])
            else:
                self.assertAlmostEqual(from_list.get_attr(key), value,
                                       places=12, msg=key)

    def test_skew_path_agrees_too(self):
        from_traj, from_list = self._both(radius=3.0, nbins=20,
                                          method='skew')
        np.testing.assert_allclose(from_list.get_array('rdf_Li_O'),
                                   from_traj.get_array('rdf_Li_O'),
                                   rtol=1e-12, atol=1e-15)


class TestFramesThatDiffer(unittest.TestCase):
    """Frames with different atom counts, species and cells."""

    # Each frame is its own ideal gas normalised by its own density, so
    # the average over frames is still a plateau at one however much
    # the frames differ.  Enough frames to make that a real check
    # rather than a coin toss: a handful of frames of a few dozen atoms
    # carries tens of percent of counting noise on its own.
    NFRAME = 24

    def test_varying_atom_count(self):
        rng = np.random.default_rng(31)
        frames = []
        for index in range(self.NFRAME):
            nat = (60, 90, 75, 110)[index % 4]
            atoms = Atoms('H' * nat, cell=np.eye(3) * 16.0, pbc=True)
            atoms.set_positions(rng.random((nat, 3)) * 16.0)
            frames.append(atoms)
        res = RDF(structures=StructureList.from_atoms(frames),
                  verbosity=0).run(radius=6.0, nbins=30)
        self.assertAlmostEqual(_plateau(res, 'H_H', 4.0), 1.0, delta=0.05)

    def test_varying_species(self):
        rng = np.random.default_rng(37)
        frames = []
        for index in range(self.NFRAME):
            symbols = ('Li40O40', 'Li20O60', 'Li60O20',
                       'Li50O50')[index % 4]
            atoms = Atoms(symbols, cell=np.eye(3) * 18.0, pbc=True)
            atoms.set_positions(rng.random((len(atoms), 3)) * 18.0)
            frames.append(atoms)
        res = RDF(structures=StructureList.from_atoms(frames),
                  verbosity=0).run(radius=6.0, nbins=30)
        for label in ('Li_Li', 'Li_O', 'O_O'):
            self.assertAlmostEqual(_plateau(res, label, 4.0), 1.0,
                                   delta=0.05, msg=label)

    def test_varying_cell(self):
        rng = np.random.default_rng(41)
        frames = []
        for index in range(self.NFRAME):
            edge = (14.0, 16.0, 18.0, 15.0)[index % 4]
            atoms = Atoms('H80', cell=np.eye(3) * edge, pbc=True)
            atoms.set_positions(rng.random((80, 3)) * edge)
            frames.append(atoms)
        structures = StructureList.from_atoms(frames)
        self.assertFalse(structures.has_fixed_cell)
        res = RDF(structures=structures, verbosity=0).run(
            radius=6.0, nbins=30)
        self.assertAlmostEqual(_plateau(res, 'H_H', 4.0), 1.0, delta=0.05)

    def test_species_pairs_default_is_the_union_over_frames(self):
        # Al appears in one frame only and Si in the others, so the
        # default pair list is built from the union of all three
        # frames rather than from any single one.
        rng = np.random.default_rng(43)
        frames = []
        for symbols in ('Si8O16', 'Si8O16', 'Al8O16'):
            atoms = Atoms(symbols, cell=np.eye(3) * 12.0, pbc=True)
            atoms.set_positions(rng.random((len(atoms), 3)) * 12.0)
            frames.append(atoms)
        res = RDF(structures=StructureList.from_atoms(frames),
                  verbosity=0).run(radius=4.0, nbins=20)
        reported = sorted('{}_{}'.format(a, b)
                          for a, b in res.get_attr('species_pairs'))
        # Al_Si is in that union but no frame holds both, so it has no
        # pairs anywhere and is dropped like any other empty pair.
        self.assertEqual(reported,
                         ['Al_Al', 'Al_O', 'O_O', 'O_Si', 'Si_Si'])
        self.assertNotIn('rdf_Al_Si', res.get_arraynames())

    def test_selecting_no_frames_says_so(self):
        # The per-frame normalisation would otherwise hand back empty
        # arrays; the old whole-trajectory prefactor divided by zero.
        structures = StructureList.from_atoms(
            _gas_frames(5, 'H10', 10.0, seed=79))
        with self.assertRaises(ValueError) as ctx:
            RDF(structures=structures, verbosity=0).run(
                radius=3.0, istart=4, istop=2)
        self.assertIn('No frames selected', str(ctx.exception))

    def test_pair_absent_from_every_frame_is_dropped(self):
        frames = _gas_frames(4, 'H20', 12.0, seed=47)
        res = RDF(structures=StructureList.from_atoms(frames),
                  verbosity=0).run(
            radius=4.0, nbins=20,
            species_pairs=[('H', 'H'), ('H', 'Xe'), ('Xe', 'Xe')])
        self.assertEqual([tuple(p) for p in res.get_attr('species_pairs')],
                         [('H', 'H')])
        self.assertNotIn('rdf_H_Xe', res.get_arraynames())


class TestAveragingOverEveryFrame(unittest.TestCase):
    """
    The normalisation convention, written down as a test.

    A frame holding none of a pair's species contributes nothing and
    still counts in the average, so g(r) tends to the fraction of
    frames that contain the pair rather than to one.  Change this test
    if that convention ever changes -- it is the only place the choice
    is visible.
    """

    LENGTH = 20.0

    def _mixed(self, n_with, n_without, seed=53):
        rng = np.random.default_rng(seed)
        frames = []
        for index in range(n_with + n_without):
            symbols = 'Li50O50' if index < n_with else 'Li50'
            atoms = Atoms(symbols, cell=np.eye(3) * self.LENGTH, pbc=True)
            atoms.set_positions(
                rng.random((len(atoms), 3)) * self.LENGTH)
            frames.append(atoms)
        return StructureList.from_atoms(frames)

    def test_half_the_frames_gives_half_the_plateau(self):
        res = RDF(structures=self._mixed(20, 20), verbosity=0).run(
            radius=6.0, nbins=30)
        self.assertAlmostEqual(_plateau(res, 'Li_O', 4.0), 0.5, delta=0.05)

    def test_a_quarter_of_the_frames_gives_a_quarter(self):
        res = RDF(structures=self._mixed(10, 30), verbosity=0).run(
            radius=6.0, nbins=30)
        self.assertAlmostEqual(_plateau(res, 'Li_O', 4.0), 0.25,
                               delta=0.05)

    def test_a_species_present_throughout_is_unaffected(self):
        # Li is in every frame, so its own g(r) still plateaus at one.
        res = RDF(structures=self._mixed(20, 20), verbosity=0).run(
            radius=6.0, nbins=30)
        self.assertAlmostEqual(_plateau(res, 'Li_Li', 4.0), 1.0,
                               delta=0.05)


class TestIndexSpeciesNeedFixedAtoms(unittest.TestCase):
    """
    An integer species spec is an atom index, which picks a different
    atom in each frame once the frames stop agreeing on their atoms.
    """

    def test_rejected_on_a_non_uniform_structure_list(self):
        frames = [Atoms('H4', positions=np.zeros((4, 3)),
                        cell=np.eye(3) * 9.0, pbc=True),
                  Atoms('H6', positions=np.zeros((6, 3)),
                        cell=np.eye(3) * 9.0, pbc=True)]
        with self.assertRaises(ValueError) as ctx:
            RDF(structures=StructureList.from_atoms(frames),
                verbosity=0).run(radius=3.0, species_pairs=[(0, 'H')])
        self.assertIn('atom index', str(ctx.exception))

    def test_still_allowed_on_a_trajectory(self):
        traj = _traj('Li8O16', 5, np.diag([10.0, 10.0, 10.0]), 59)
        res = RDF(structures=traj, verbosity=0).run(
            radius=4.0, nbins=20, species_pairs=[(0, 'O')])
        self.assertIn('rdf_atom0_O', res.get_arraynames())

    def test_still_allowed_on_a_uniform_structure_list(self):
        frames = _gas_frames(5, 'Li8O16', 10.0, seed=61)
        structures = StructureList.from_atoms(frames)
        self.assertTrue(structures.has_uniform_composition)
        res = RDF(structures=structures, verbosity=0).run(
            radius=4.0, nbins=20, species_pairs=[(0, 'O')])
        self.assertIn('rdf_atom0_O', res.get_arraynames())

    def test_an_index_spec_no_longer_labels_itself_none(self):
        # Regression: get_label had no branch for an int, so it fell
        # through to one that printed the type and returned None, and
        # the output arrays came out named 'rdf_None_O'.
        traj = _traj('Li8O16', 4, np.diag([10.0, 10.0, 10.0]), 73)
        res = RDF(structures=traj, verbosity=0).run(
            radius=4.0, nbins=10, species_pairs=[(0, 'O')])
        self.assertNotIn('rdf_None_O', res.get_arraynames())


class TestADFStillNeedsATrajectory(unittest.TestCase):
    """
    ADF is deferred.  Its bond topology is a list of global atom
    indices, which only means anything if the atoms are the same in
    every frame, so it has to refuse a bare StructureList rather than
    failing somewhere inside run().
    """

    def test_structure_list_is_refused_with_a_reason(self):
        frames = _gas_frames(3, 'H8', 10.0, seed=67)
        with self.assertRaises(TypeError) as ctx:
            ADF(structures=StructureList.from_atoms(frames), verbosity=0)
        message = str(ctx.exception)
        self.assertIn('atom indices', message)
        self.assertIn('not implemented yet', message)

    def test_a_trajectory_is_still_accepted(self):
        traj = _traj('H8', 4, np.diag([10.0, 10.0, 10.0]), 71)
        self.assertIsNotNone(ADF(structures=traj, verbosity=0).structures)


if __name__ == '__main__':
    unittest.main()
