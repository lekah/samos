import unittest

class TestDiffusion(unittest.TestCase):

    def test_1(self):
        from samos.trajectory import Trajectory
        from samos.analysis.get_diffusion import DiffusionAnalyzer
        t = Trajectory.load_file('data/H2O-64-300K.tar.gz')
        d = DiffusionAnalyzer(verbosity=2)
        
        d.set_trajectories(t)
        d.get_msd_isotropic(t_start_fit_fs=2000., t_end_fit_fs=4000., nr_of_blocks=12)
        #~ get_msd_isotropic(t)
if __name__ == '__main__':
    unittest.main()
