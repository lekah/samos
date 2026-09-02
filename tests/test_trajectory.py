# -*- coding: utf-8 -*-

import unittest


class TestTrajectory(unittest.TestCase):

    def test_creation(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        np.random.seed(2345)
        pos = np.random.random((10, 10, 3))
        frc = np.random.random((10, 10, 3))
        vel = np.random.random((10, 10, 3))

        t = Trajectory()
        t.set_atoms(Atoms('H'*10))
        t.set_positions(pos)
        t.set_velocities(vel)
        t.set_forces(frc)

        self.assertTrue(np.array_equal(pos, t.get_positions()))
        self.assertTrue(np.array_equal(vel, t.get_velocities()))
        self.assertTrue(np.array_equal(frc, t.get_forces()))

        atoms_step_3 = t.get_step_atoms(3)

        self.assertTrue(np.array_equal(atoms_step_3.get_positions(), pos[3]))
        # if not np.array_equal(atoms_step_3.get_velocities(), vel[3]):
        #     print(atoms_step_3.get_velocities())
        #     print(vel[3])
        self.assertTrue(((atoms_step_3.get_velocities() - vel[3])**2
                         ).sum() < 1e-6)
        # this doesn't work well for some reason:
        # self.assertTrue(np.array_equal(
        #   atoms_step_3.get_velocities(), vel[3]))

    def test_store_and_reload(self):
        import numpy as np
        import tempfile
        from ase import Atoms
        from samos.trajectory import Trajectory
        pos = np.random.random((10, 10, 3))
        vel = np.random.random((10, 10, 3))
        frc = np.random.random((10, 10, 3))
        xtr = np.random.random(10)
        t = Trajectory()
        t.set_atoms(Atoms('H'*10))
        t.set_positions(pos)
        t.set_velocities(vel)
        t.set_forces(frc)
        t.set_array('extra', xtr)
        with tempfile.NamedTemporaryFile() as f:
            t.save(f.name)
            tnew = Trajectory.load_file(f.name)

        self.assertTrue(np.array_equal(pos, tnew.get_positions()))
        self.assertTrue(np.array_equal(vel, tnew.get_velocities()))
        self.assertTrue(np.array_equal(frc, tnew.get_forces()))
        self.assertTrue(np.array_equal(xtr, tnew.get_array('extra')))

    def test_compatibility(self):
        from samos.trajectory import (
            Trajectory, check_trajectory_compatibility,
            IncompatibleTrajectoriesException)
        from ase import Atoms
        atoms1 = Atoms('H'*10+'O')
        atoms2 = Atoms('H'*11+'O')
        atoms3 = Atoms('O'+'H'*10)
        t1 = Trajectory(atoms=atoms1, timestep=1.)
        t2 = Trajectory(atoms=atoms2, timestep=1.)
        t3 = Trajectory(atoms=atoms3, timestep=1.)
        t4 = Trajectory(atoms=atoms3.copy(), timestep=1.)

        with self.assertRaises(IncompatibleTrajectoriesException):
            check_trajectory_compatibility([t1, t2])
        with self.assertRaises(IncompatibleTrajectoriesException):
            check_trajectory_compatibility([t2, t3])
        with self.assertRaises(IncompatibleTrajectoriesException):
            check_trajectory_compatibility([t1, t3])
        with self.assertRaises(TypeError):
            check_trajectory_compatibility([t1, t3, 3])
        self.assertTrue(check_trajectory_compatibility([t3, t4]))
        t4.set_timestep(3)
        with self.assertRaises(IncompatibleTrajectoriesException):
            self.assertTrue(check_trajectory_compatibility([t3, t4]))

    def test_store_and_reload_preserves_array_names(self):
        """Regression: load_file used rstrip('.npy'), which strips a
        character set rather than a suffix, so 'potential_energy.npy'
        came back as 'potential_energ'."""
        import numpy as np
        import tempfile
        from ase import Atoms
        from samos.trajectory import Trajectory
        t = Trajectory()
        t.set_atoms(Atoms('H'*4))
        t.set_positions(np.random.random((10, 4, 3)))
        t.set_pot_energies(np.arange(10.0))
        # names ending in any of '.npy' are the ones rstrip mangles
        t.set_array('energy', np.arange(10.0))
        t.set_array('nanny', np.arange(10.0))
        with tempfile.NamedTemporaryFile() as f:
            t.save(f.name)
            tnew = Trajectory.load_file(f.name)

        self.assertEqual(t.get_arraynames(), tnew.get_arraynames())
        self.assertTrue(np.array_equal(
            t.get_array(t._POT_ENER_KEY), tnew.get_array(t._POT_ENER_KEY)))

    def test_set_array_check_existing_raises_valueerror(self):
        """Regression: the duplicate-name branch called '...'.formamt(),
        so it raised AttributeError instead of ValueError."""
        import numpy as np
        from samos.utils.attributed_array import AttributedArray
        a = AttributedArray()
        a.set_array('x', np.arange(5))
        with self.assertRaises(ValueError):
            a.set_array('x', np.arange(5), check_existing=True)


class TestArrayShapeGuards(unittest.TestCase):
    """The wanted_shape_1 / wanted_shape_2 guards indexed into
    array.shape before confirming the rank, so a too-flat array raised a
    bare IndexError from the guard rather than a message saying what was
    wrong."""

    def setUp(self):
        from samos.utils.attributed_array import AttributedArray
        self.a = AttributedArray()

    def test_rank_too_low_reports_the_rank(self):
        import numpy as np
        with self.assertRaises(IndexError) as cm:
            self.a.set_array('x', np.arange(5), wanted_shape_1=3)
        msg = str(cm.exception)
        self.assertIn('dimension', msg)
        self.assertIn('1', msg)

    def test_wrong_size_still_rejected(self):
        import numpy as np
        with self.assertRaises(IndexError):
            self.a.set_array('x', np.zeros((4, 2)), wanted_shape_1=3)

    def test_correct_shape_accepted(self):
        import numpy as np
        self.a.set_array('x', np.zeros((4, 3, 3)), wanted_shape_len=3,
                         wanted_shape_1=3, wanted_shape_2=3)
        self.assertIn('x', self.a.get_arraynames())

    def test_wrong_rank_reports_both_ranks(self):
        import numpy as np
        with self.assertRaises(TypeError) as cm:
            self.a.set_array('x', np.zeros((4, 3)), wanted_shape_len=3)
        self.assertIn('3', str(cm.exception))


class TestAttributedArrayRoundTrip(unittest.TestCase):
    """Regression: load_file read cls._ATOMS_FILENAME, which only exists
    on Trajectory, so no plain AttributedArray or TimeSeries result
    could ever be reloaded.  Atoms handling now lives in a Trajectory
    _load_extra hook mirroring _save_atoms."""

    def _round_trip(self, obj):
        import tempfile
        with tempfile.NamedTemporaryFile() as f:
            obj.save(f.name)
            return type(obj).load_file(f.name)

    def test_attributed_array_round_trip(self):
        import numpy as np
        from samos.utils.attributed_array import AttributedArray
        a = AttributedArray()
        a.set_array('x', np.arange(5.))
        a.set_attr('note', 'hello')
        b = self._round_trip(a)
        self.assertTrue(np.array_equal(a.get_array('x'), b.get_array('x')))
        self.assertEqual(b.get_attr('note'), 'hello')

    def test_time_series_round_trip(self):
        import numpy as np
        from samos.analysis.dynamics import TimeSeries
        ts = TimeSeries()
        ts.set_array('msd_isotropic_H_mean', np.linspace(0., 1., 7))
        ts.set_attr('timestep_fs', 1.0)
        back = self._round_trip(ts)
        self.assertTrue(np.array_equal(
            ts.get_array('msd_isotropic_H_mean'),
            back.get_array('msd_isotropic_H_mean')))
        self.assertEqual(back.get_attr('timestep_fs'), 1.0)

    def test_trajectory_still_restores_atoms(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        t = Trajectory()
        t.set_atoms(Atoms('H2O2'))
        t.set_positions(np.random.random((6, 4, 3)))
        back = self._round_trip(t)
        self.assertIsNotNone(back.get_atoms())
        self.assertEqual(list(back.get_types()), list(t.get_types()))
        self.assertTrue(np.array_equal(
            t.get_positions(), back.get_positions()))


class TestSingleBohrConstant(unittest.TestCase):
    """bohr_to_ang was defined in four modules with two different
    values; samos.utils.constants is now the only definition."""

    def test_all_modules_agree(self):
        from samos.utils.constants import bohr_to_ang
        from samos.io.xsf import bohr_to_ang as xsf_value
        from samos.utils.units import UNIT_SYSTEMS
        self.assertEqual(xsf_value, bohr_to_ang)
        self.assertEqual(UNIT_SYSTEMS['electron']['l_conv'], bohr_to_ang)

    def test_no_module_redefines_it(self):
        import os
        import re
        root = os.path.join(os.path.dirname(__file__), '..', 'samos')
        offenders = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                if os.path.samefile(
                        path, os.path.join(root, 'utils', 'constants.py')):
                    continue
                with open(path) as fh:
                    for line in fh:
                        if re.match(r'\s*bohr_to_ang\s*=', line):
                            offenders.append(path)
        self.assertEqual(offenders, [])


class TestSliceSteps(unittest.TestCase):
    """Trajectory.slice_steps, which backs the CLI's --index option."""

    def _traj(self, nstep=12, nat=6):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        rng = np.random.default_rng(7)
        t = Trajectory(atoms=Atoms('H' * nat), timestep=2.0)
        t.set_positions(rng.random((nstep, nat, 3)))
        t.set_velocities(rng.random((nstep, nat, 3)))
        return t

    def test_arrays_are_sliced(self):
        import numpy as np
        t = self._traj()
        sliced = t.slice_steps(slice(2, 10, 3))
        np.testing.assert_array_equal(
            sliced.get_positions(), t.get_positions()[2:10:3])
        np.testing.assert_array_equal(
            sliced.get_velocities(), t.get_velocities()[2:10:3])
        self.assertEqual(sliced.nstep, 3)

    def test_types_are_per_atom_and_not_sliced(self):
        # nstep equals nat here, so a blanket slice over every array
        # would go unnoticed on shape alone.
        t = self._traj(nstep=6, nat=6)
        t.set_types(['H', 'H', 'H', 'Li', 'Li', 'Li'])
        sliced = t.slice_steps(slice(None, None, 2))
        self.assertEqual(len(sliced.get_types()), 6)
        self.assertEqual(list(sliced.get_types()), list(t.get_types()))
        self.assertEqual(sliced.nstep, 3)

    def test_stride_scales_the_timestep(self):
        t = self._traj()
        self.assertEqual(t.slice_steps(slice(None, None, 4)).get_timestep(),
                         8.0)
        # A negative stride keeps the spacing, only the order changes.
        self.assertEqual(t.slice_steps(slice(None, None, -2)).get_timestep(),
                         4.0)

    def test_contiguous_slice_keeps_the_timestep(self):
        t = self._traj()
        self.assertEqual(t.slice_steps(slice(3, 9)).get_timestep(), 2.0)

    def test_original_is_unchanged(self):
        t = self._traj()
        t.slice_steps(slice(None, None, 2))
        self.assertEqual(t.nstep, 12)
        self.assertEqual(t.get_timestep(), 2.0)

    def test_atoms_are_carried_over(self):
        t = self._traj()
        sliced = t.slice_steps(slice(0, 4))
        self.assertEqual(len(sliced.atoms), len(t.atoms))
        self.assertEqual(sliced.atoms.get_chemical_symbols(),
                         t.atoms.get_chemical_symbols())

    def test_empty_slice_raises(self):
        t = self._traj()
        with self.assertRaises(ValueError):
            t.slice_steps(slice(100, 200))

    def test_non_slice_raises(self):
        t = self._traj()
        with self.assertRaises(TypeError):
            t.slice_steps(3)


if __name__ == '__main__':
    unittest.main()
