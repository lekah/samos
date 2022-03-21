# -*- coding: utf-8 -*-

import unittest

class TestTrajectory(unittest.TestCase):

    def test_creation(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        pos = np.random.random((10,10,3))
        frc = np.random.random((10,10,3))
        vel = np.random.random((10,10,3))

        t = Trajectory()
        t.set_atoms(Atoms('H'*10))
        t.set_positions(pos)
        t.set_velocities(vel)
        t.set_forces(frc)

        np.testing.assert_almost_equal(pos, t.get_positions(), decimal=12)
        np.testing.assert_almost_equal(vel, t.get_velocities(), decimal=12)
        np.testing.assert_almost_equal(frc, t.get_forces(), decimal=12)


        atoms_step_3 = t.get_step_atoms(3)

        np.testing.assert_almost_equal(atoms_step_3.get_positions(), pos[3], decimal=12)
        np.testing.assert_almost_equal(atoms_step_3.get_velocities(), vel[3], decimal=12)


    def test_store_and_reload(self):
        import numpy as np
        import tempfile
        from ase import Atoms
        from samos.trajectory import Trajectory
        pos = np.random.random((10,10,3))
        vel = np.random.random((10,10,3))
        frc = np.random.random((10,10,3))
        xtr = np.random.random(10)
        t = Trajectory()
        t.set_atoms(Atoms('H'*10))
        t.set_positions(pos)
        t.set_velocities(vel)
        t.set_forces(frc)
        t.set_array('extra', xtr)
        with tempfile.NamedTemporaryFile() as f:
            t.save(f.name)
            tnew =  Trajectory.load_file(f.name)

        np.testing.assert_almost_equal(pos, tnew.get_positions(), decimal=12)
        np.testing.assert_almost_equal(vel, tnew.get_velocities(), decimal=12)
        np.testing.assert_almost_equal(frc, tnew.get_forces(), decimal=12)
        np.testing.assert_almost_equal(xtr, tnew.get_array('extra'), decimal=12)

    def test_compatibility(self):
        from samos.trajectory import Trajectory, check_trajectory_compatibility, IncompatibleTrajectoriesException
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
        self.assertTrue(check_trajectory_compatibility([t3,t4]))
        t4.set_timestep(3)
        with self.assertRaises(IncompatibleTrajectoriesException):
            self.assertTrue(check_trajectory_compatibility([t3,t4]))

if __name__ == '__main__':
    unittest.main()
