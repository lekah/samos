# -*- coding: utf-8 -*-
"""
Tests for samos.structurelist.StructureList and for the interface it
gives samos.trajectory.Trajectory.

The point of the class is that frames need not share their atoms, so
most of what is checked here is that a frame's atom count, species and
cell are read back per frame rather than assumed constant.  The
Trajectory half checks the other side of the same coin: that its
rectangular storage answers the same questions, with no copying and
with nothing about its own behaviour changed.
"""

import tempfile
import unittest

import numpy as np
from ase import Atoms

from samos.structurelist import StructureList
from samos.trajectory import Trajectory


def _mixed_frames():
    """Three frames that differ in atom count, species and cell."""
    rng = np.random.default_rng(11)
    return [
        Atoms('Si2O4', positions=rng.random((6, 3)) * 5.0,
              cell=np.eye(3) * 5.0, pbc=True),
        Atoms('Si3O6', positions=rng.random((9, 3)) * 6.0,
              cell=np.eye(3) * 6.0, pbc=True),
        Atoms('Al2O3', positions=rng.random((5, 3)) * 4.0,
              cell=np.eye(3) * 4.0, pbc=True),
    ]


def _uniform_frames(nframe=4, nat=5):
    rng = np.random.default_rng(3)
    return [Atoms('H' * nat, positions=rng.random((nat, 3)) * 5.0,
                  cell=np.eye(3) * 5.0, pbc=True)
            for _ in range(nframe)]


class TestStructureListFromAtoms(unittest.TestCase):
    """Building a list out of frames that do not match."""

    def setUp(self):
        self.frames = _mixed_frames()
        self.sl = StructureList.from_atoms(self.frames)

    def test_frame_count(self):
        self.assertEqual(self.sl.nstep, 3)
        self.assertEqual(len(self.sl), 3)

    def test_atom_counts_are_per_frame(self):
        self.assertEqual(
            [self.sl.get_frame_natoms(i) for i in range(3)], [6, 9, 5])

    def test_positions_are_read_back_per_frame(self):
        for i, atoms in enumerate(self.frames):
            np.testing.assert_allclose(
                self.sl.get_frame_positions(i), atoms.get_positions())

    def test_types_are_read_back_per_frame(self):
        for i, atoms in enumerate(self.frames):
            self.assertEqual(list(self.sl.get_frame_types(i)),
                             atoms.get_chemical_symbols())

    def test_cells_are_read_back_per_frame(self):
        for i, atoms in enumerate(self.frames):
            np.testing.assert_allclose(self.sl.get_frame_cell(i),
                                       np.asarray(atoms.cell))

    def test_a_frame_is_a_view_not_a_copy(self):
        # The reason for storing flat rather than as a list of Atoms.
        frame = self.sl.get_frame_positions(1)
        self.assertTrue(np.shares_memory(
            frame, self.sl.get_array('positions')))

    def test_species_is_the_union_over_frames(self):
        # Al appears only in the last frame, Si only in the first two.
        self.assertEqual(list(self.sl.get_species()), ['Al', 'O', 'Si'])

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            StructureList.from_atoms([])

    def test_non_atoms_entry_names_the_offender(self):
        with self.assertRaises(TypeError) as ctx:
            StructureList.from_atoms([self.frames[0], 'not atoms'])
        self.assertIn('Entry 1', str(ctx.exception))


class TestStructureListFrameIndexing(unittest.TestCase):

    def setUp(self):
        self.frames = _mixed_frames()
        self.sl = StructureList.from_atoms(self.frames)

    def test_negative_index_counts_from_the_end(self):
        np.testing.assert_allclose(
            self.sl.get_frame_positions(-1),
            self.frames[-1].get_positions())

    def test_index_past_the_end_raises(self):
        with self.assertRaises(IndexError):
            self.sl.get_frame_positions(3)

    def test_index_before_the_start_raises(self):
        with self.assertRaises(IndexError):
            self.sl.get_frame_positions(-4)

    def test_getitem_gives_one_frame(self):
        atoms = self.sl[1]
        self.assertEqual(atoms.get_chemical_symbols(),
                         self.frames[1].get_chemical_symbols())

    def test_getitem_with_a_slice_raises(self):
        with self.assertRaises(TypeError):
            self.sl[0:2]

    def test_iteration_yields_every_frame(self):
        counts = [len(atoms) for atoms in self.sl]
        self.assertEqual(counts, [6, 9, 5])


class TestStructureListAtomsRoundTrip(unittest.TestCase):
    """get_frame_atoms should give back what went in."""

    def test_round_trip_of_every_frame(self):
        frames = _mixed_frames()
        sl = StructureList.from_atoms(frames)
        for original, restored in zip(frames, sl):
            self.assertEqual(restored.get_chemical_symbols(),
                             original.get_chemical_symbols())
            np.testing.assert_allclose(restored.get_positions(),
                                       original.get_positions())
            np.testing.assert_allclose(np.asarray(restored.cell),
                                       np.asarray(original.cell))
            np.testing.assert_array_equal(restored.pbc, original.pbc)

    def test_non_periodic_frames_stay_non_periodic(self):
        # pbc is stored rather than assumed: a cluster that came in
        # aperiodic must not come back out periodic.
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]], pbc=False)
        sl = StructureList.from_atoms([atoms])
        self.assertFalse(sl.get_frame_atoms(0).pbc.any())


class TestStructureListSaveAndLoad(unittest.TestCase):
    """
    The payoff of flat numpy storage: AttributedArray's tarball
    save/load applies with no new code.
    """

    def test_round_trip_through_a_file(self):
        frames = _mixed_frames()
        sl = StructureList.from_atoms(frames)
        sl.set_attr('source', 'test')
        with tempfile.NamedTemporaryFile() as handle:
            sl.save(handle.name)
            loaded = StructureList.load_file(handle.name)

        self.assertEqual(loaded.nstep, 3)
        self.assertEqual(loaded.get_attr('source'), 'test')
        self.assertEqual(list(loaded.get_species()), ['Al', 'O', 'Si'])
        for i, atoms in enumerate(frames):
            np.testing.assert_allclose(loaded.get_frame_positions(i),
                                       atoms.get_positions())
            self.assertEqual(list(loaded.get_frame_types(i)),
                             atoms.get_chemical_symbols())
            np.testing.assert_allclose(loaded.get_frame_cell(i),
                                       np.asarray(atoms.cell))


class TestStructureListUniformityFlags(unittest.TestCase):
    """
    has_fixed_cell and has_uniform_composition are what an analyzer
    checks before hoisting work out of its frame loop.
    """

    def test_mixed_frames_are_neither(self):
        sl = StructureList.from_atoms(_mixed_frames())
        self.assertFalse(sl.has_fixed_cell)
        self.assertFalse(sl.has_uniform_composition)

    def test_uniform_frames_are_both(self):
        sl = StructureList.from_atoms(_uniform_frames())
        self.assertTrue(sl.has_fixed_cell)
        self.assertTrue(sl.has_uniform_composition)

    def test_same_atoms_changing_cell(self):
        frames = _uniform_frames()
        frames[2].set_cell(np.eye(3) * 7.0)
        sl = StructureList.from_atoms(frames)
        self.assertFalse(sl.has_fixed_cell)
        self.assertTrue(sl.has_uniform_composition)

    def test_same_count_but_different_species(self):
        # Equal atom counts, so the offsets alone cannot tell these
        # apart -- the symbols have to be compared too.
        rng = np.random.default_rng(5)
        frames = [Atoms('H2O', positions=rng.random((3, 3)),
                        cell=np.eye(3) * 5.0, pbc=True),
                  Atoms('HLiO', positions=rng.random((3, 3)),
                        cell=np.eye(3) * 5.0, pbc=True)]
        sl = StructureList.from_atoms(frames)
        self.assertTrue(sl.has_fixed_cell)
        self.assertFalse(sl.has_uniform_composition)

    def test_a_single_frame_is_uniform(self):
        sl = StructureList.from_atoms(_mixed_frames()[:1])
        self.assertTrue(sl.has_uniform_composition)


class TestTrajectoryIsAStructureList(unittest.TestCase):
    """
    Trajectory answers the same questions from rectangular storage.
    """

    def setUp(self):
        rng = np.random.default_rng(19)
        self.nstep, self.nat = 4, 6
        self.pos = rng.random((self.nstep, self.nat, 3)) * 5.0
        self.traj = Trajectory(
            atoms=Atoms('Li2O4', cell=np.eye(3) * 5.0, pbc=True),
            positions=self.pos)

    def test_it_is_an_instance(self):
        # This is what lets an analyzer keep a real type check rather
        # than sniffing for methods.
        self.assertIsInstance(self.traj, StructureList)

    def test_frame_positions_match_the_rectangular_array(self):
        for i in range(self.nstep):
            np.testing.assert_array_equal(
                self.traj.get_frame_positions(i), self.pos[i])

    def test_a_frame_is_a_view_not_a_copy(self):
        frame = self.traj.get_frame_positions(2)
        self.assertTrue(np.shares_memory(
            frame, self.traj.get_positions()))

    def test_frame_types_and_counts(self):
        for i in range(self.nstep):
            self.assertEqual(list(self.traj.get_frame_types(i)),
                             ['Li', 'Li', 'O', 'O', 'O', 'O'])
            self.assertEqual(self.traj.get_frame_natoms(i), self.nat)

    def test_species_and_uniformity(self):
        self.assertEqual(list(self.traj.get_species()), ['Li', 'O'])
        self.assertTrue(self.traj.has_uniform_composition)

    def test_len_and_iteration(self):
        self.assertEqual(len(self.traj), self.nstep)
        frames = list(self.traj)
        self.assertEqual(len(frames), self.nstep)
        np.testing.assert_allclose(frames[1].get_positions(),
                                   self.pos[1])

    def test_negative_and_out_of_range_indices(self):
        np.testing.assert_array_equal(
            self.traj.get_frame_positions(-1), self.pos[-1])
        with self.assertRaises(IndexError):
            self.traj.get_frame_positions(self.nstep)

    def test_frame_atoms_carry_symbols_cell_and_pbc(self):
        atoms = self.traj.get_frame_atoms(0)
        self.assertEqual(atoms.get_chemical_symbols(),
                         ['Li', 'Li', 'O', 'O', 'O', 'O'])
        np.testing.assert_allclose(np.asarray(atoms.cell),
                                   np.eye(3) * 5.0)
        self.assertTrue(atoms.pbc.all())


class TestTrajectoryCellSources(unittest.TestCase):
    """
    A trajectory takes its cell either from a per-step array or from
    the reference atoms, and get_frame_cell has to cover both.
    """

    def _traj(self, cells=None):
        rng = np.random.default_rng(23)
        traj = Trajectory(atoms=Atoms('H3', cell=np.eye(3) * 5.0,
                                      pbc=True),
                          positions=rng.random((3, 3, 3)))
        if cells is not None:
            traj.set_cells(cells)
        return traj

    def test_fixed_cell_comes_from_the_atoms(self):
        traj = self._traj()
        self.assertTrue(traj.has_fixed_cell)
        for i in range(3):
            np.testing.assert_allclose(traj.get_frame_cell(i),
                                       np.eye(3) * 5.0)

    def test_per_step_cells_are_read_per_frame(self):
        cells = np.array([np.eye(3) * edge for edge in (5.0, 6.0, 7.0)])
        traj = self._traj(cells)
        self.assertFalse(traj.has_fixed_cell)
        for i, edge in enumerate((5.0, 6.0, 7.0)):
            np.testing.assert_allclose(traj.get_frame_cell(i),
                                       np.eye(3) * edge)

    def test_per_step_cells_that_never_change_count_as_fixed(self):
        cells = np.array([np.eye(3) * 5.0] * 3)
        self.assertTrue(self._traj(cells).has_fixed_cell)

    def test_frame_cell_without_atoms_or_cells_explains_itself(self):
        traj = Trajectory(types=['H'] * 3)
        traj.set_positions(np.zeros((2, 3, 3)))
        with self.assertRaises(ValueError) as ctx:
            traj.get_frame_cell(0)
        self.assertIn('no per-step', str(ctx.exception))


class TestTrajectoryUnchanged(unittest.TestCase):
    """
    The new base class must not have disturbed what Trajectory already
    did.  These repeat the load-bearing bits against a trajectory that
    also carries velocities, an extra array and a timestep.
    """

    def _traj(self):
        rng = np.random.default_rng(29)
        traj = Trajectory(atoms=Atoms('H4', cell=np.eye(3) * 5.0,
                                      pbc=True),
                          timestep=2.0)
        traj.set_positions(rng.random((6, 4, 3)))
        traj.set_velocities(rng.random((6, 4, 3)))
        traj.set_array('extra', rng.random(6))
        return traj

    def test_nstep_still_comes_from_the_arrays(self):
        self.assertEqual(self._traj().nstep, 6)

    def test_save_and_load_round_trip(self):
        traj = self._traj()
        with tempfile.NamedTemporaryFile() as handle:
            traj.save(handle.name)
            loaded = Trajectory.load_file(handle.name)
        np.testing.assert_array_equal(loaded.get_positions(),
                                      traj.get_positions())
        np.testing.assert_array_equal(loaded.get_velocities(),
                                      traj.get_velocities())
        self.assertEqual(loaded.nstep, 6)
        self.assertEqual(loaded.get_timestep(), 2.0)

    def test_slice_steps_still_works(self):
        traj = self._traj()
        sliced = traj.slice_steps(slice(0, 6, 2))
        self.assertEqual(sliced.nstep, 3)
        np.testing.assert_array_equal(sliced.get_positions(),
                                      traj.get_positions()[0:6:2])
        self.assertEqual(sliced.get_timestep(), 4.0)

    def test_get_step_atoms_still_works(self):
        traj = self._traj()
        atoms = traj.get_step_atoms(3)
        np.testing.assert_allclose(atoms.get_positions(),
                                   traj.get_positions()[3])

    def test_unknown_keyword_is_still_rejected(self):
        # StructureList deliberately has no __init__, so the keyword
        # check at the end of the chain must still be reached.
        with self.assertRaises(TypeError) as ctx:
            Trajectory(positionz=np.zeros((2, 3, 3)))
        self.assertIn('positionz', str(ctx.exception))

    def test_structures_is_not_a_trajectory_keyword(self):
        # from_atoms on a Trajectory means the rectangular build, so
        # the flat setter must not be reachable by keyword here.
        with self.assertRaises(TypeError):
            Trajectory(structures=_mixed_frames())


class TestStructureListSliceSteps(unittest.TestCase):
    """
    Backs the CLI's --index for frames that differ, so the atom counts
    have to be re-derived rather than assumed.
    """

    def setUp(self):
        self.frames = _mixed_frames()
        self.sl = StructureList.from_atoms(self.frames)

    def test_selected_frames_are_kept_in_order(self):
        sliced = self.sl.slice_steps(slice(0, 3, 2))
        self.assertEqual(sliced.nstep, 2)
        self.assertEqual(
            [sliced.get_frame_natoms(i) for i in range(2)], [6, 5])
        for new, old in enumerate((0, 2)):
            np.testing.assert_allclose(
                sliced.get_frame_positions(new),
                self.frames[old].get_positions())
            self.assertEqual(list(sliced.get_frame_types(new)),
                             self.frames[old].get_chemical_symbols())
            np.testing.assert_allclose(
                sliced.get_frame_cell(new),
                np.asarray(self.frames[old].cell))

    def test_offsets_are_rebuilt_not_carried_over(self):
        # The whole point: frame 1 of the slice must start where frame
        # 0 of the slice ends, not where it started in the original.
        sliced = self.sl.slice_steps(slice(1, 3))
        np.testing.assert_array_equal(
            sliced.get_array('offsets'), [0, 9, 14])
        self.assertEqual(len(sliced.get_array('positions')), 14)
        self.assertEqual(len(sliced.get_array('types')), 14)

    def test_negative_and_open_ended_slices(self):
        self.assertEqual(self.sl.slice_steps(slice(None, -1)).nstep, 2)
        self.assertEqual(self.sl.slice_steps(slice(-2, None)).nstep, 2)
        self.assertEqual(self.sl.slice_steps(slice(None, None, -1)).nstep,
                         3)

    def test_reversed_slice_reverses_the_frames(self):
        reversed_ = self.sl.slice_steps(slice(None, None, -1))
        np.testing.assert_allclose(reversed_.get_frame_positions(0),
                                   self.frames[-1].get_positions())

    def test_pbc_is_sliced_with_the_frames(self):
        frames = _mixed_frames()
        frames[1].pbc = False
        sliced = StructureList.from_atoms(frames).slice_steps(slice(1, 3))
        self.assertFalse(sliced.get_frame_pbc(0).any())
        self.assertTrue(sliced.get_frame_pbc(1).all())

    def test_attributes_are_carried_over(self):
        self.sl.set_attr('source', 'test')
        self.assertEqual(
            self.sl.slice_steps(slice(0, 2)).get_attr('source'), 'test')

    def test_original_is_unchanged(self):
        self.sl.slice_steps(slice(0, 1))
        self.assertEqual(self.sl.nstep, 3)
        self.assertEqual(len(self.sl.get_array('positions')), 20)

    def test_empty_slice_raises(self):
        with self.assertRaises(ValueError):
            self.sl.slice_steps(slice(2, 2))

    def test_non_slice_raises(self):
        with self.assertRaises(TypeError):
            self.sl.slice_steps(1)

    def test_an_array_of_unknown_kind_raises(self):
        # Neither per-atom nor per-frame, so there is no right way to
        # cut it.  Failing beats guessing.
        self.sl.set_array('mystery', np.arange(4.0))
        with self.assertRaises(TypeError) as ctx:
            self.sl.slice_steps(slice(0, 2))
        self.assertIn('mystery', str(ctx.exception))

    def test_a_trajectory_still_uses_its_own_slicing(self):
        traj = Trajectory(atoms=Atoms('H4', cell=np.eye(3) * 5.0, pbc=True),
                          timestep=2.0)
        traj.set_positions(np.zeros((6, 4, 3)))
        sliced = traj.slice_steps(slice(None, None, 2))
        self.assertIsInstance(sliced, Trajectory)
        # The timestep rescaling is Trajectory's, and must survive the
        # base class gaining a slice_steps of its own.
        self.assertEqual(sliced.get_timestep(), 4.0)


class TestStructureListTransformSpecies(unittest.TestCase):

    def test_every_atom_of_every_frame_is_relabelled(self):
        sl = StructureList.from_atoms(_mixed_frames())
        sl.transform_species('H')
        self.assertEqual(list(sl.get_species()), ['H'])
        for i in range(sl.nstep):
            self.assertEqual(set(sl.get_frame_types(i)), {'H'})

    def test_atom_counts_are_untouched(self):
        sl = StructureList.from_atoms(_mixed_frames())
        sl.transform_species('H')
        self.assertEqual([sl.get_frame_natoms(i) for i in range(3)],
                         [6, 9, 5])


class TestUnitConversion(unittest.TestCase):
    """
    Lengths are scaled by StructureList; a Trajectory extends that with
    the arrays only it has.
    """

    def test_structure_list_scales_positions_and_cells(self):
        frames = _mixed_frames()
        sl = StructureList.from_atoms(frames)
        sl.apply_unit_conversion(l_conv=2.0)
        np.testing.assert_allclose(sl.get_frame_positions(1),
                                   frames[1].get_positions() * 2.0)
        np.testing.assert_allclose(sl.get_frame_cell(1),
                                   np.asarray(frames[1].cell) * 2.0)

    def test_a_factor_of_one_changes_nothing(self):
        frames = _mixed_frames()
        sl = StructureList.from_atoms(frames)
        sl.apply_unit_conversion(l_conv=1.0)
        np.testing.assert_array_equal(sl.get_frame_positions(0),
                                      frames[0].get_positions())

    def test_trajectory_still_converts_velocities_and_lengths(self):
        rng = np.random.default_rng(83)
        pos = rng.random((4, 3, 3))
        vel = rng.random((4, 3, 3))
        traj = Trajectory(atoms=Atoms('H3', cell=np.eye(3) * 5.0, pbc=True))
        traj.set_positions(pos)
        traj.set_velocities(vel)
        traj.apply_unit_conversion(l_conv=2.0, v_conv=3.0)
        np.testing.assert_allclose(traj.get_positions(), pos * 2.0)
        np.testing.assert_allclose(traj.get_velocities(), vel * 3.0)


if __name__ == '__main__':
    unittest.main()
