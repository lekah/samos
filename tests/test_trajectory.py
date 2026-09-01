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


if __name__ == '__main__':
    unittest.main()
