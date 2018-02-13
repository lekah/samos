import unittest

class TestTrajectory(unittest.TestCase):

    def test_creation(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        pos = np.random.random((10,10,3))
        vel = np.random.random((10,10,3))
        frc = np.random.random((10,10,3))
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
        self.assertTrue(np.array_equal(atoms_step_3.get_velocities(), vel[3]))


    def test_store_and_reload(self):
        import numpy as np
        import tempfile
        from ase import Atoms
        from samos.trajectory import Trajectory
        pos = np.random.random((10,10,3))
        vel = np.random.random((10,10,3))
        frc = np.random.random((10,10,3))
        t = Trajectory()
        t.set_atoms(Atoms('H'*10))
        t.set_positions(pos)
        t.set_velocities(vel)
        t.set_forces(frc)
        with tempfile.NamedTemporaryFile() as f:
            print t.save(f.name)
        #~ t.save(


if __name__ == '__main__':
    unittest.main()
