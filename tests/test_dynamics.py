import unittest

class TestDynamics(unittest.TestCase):

    def test_1(self):
        from samos.trajectory import Trajectory
        from samos.analysis.dynamics import DynamicsAnalyzer
        from samos.plotting.plot_dynamics import plot_msd_isotropic
        import json
        t = Trajectory.load_file('data/H2O-64-300K.tar.gz')
        t.recenter()
        d = DynamicsAnalyzer(verbosity=0)
        
        d.set_trajectories(t)

        vaf = d.get_vaf(t_start_fit_fs=2000.,  t_end_fit_fs=4000., nr_of_blocks=12)


        msd_iso = d.get_msd(
                t_start_fit_fs=2000., 
                t_end_fit_fs=4000., 
                nr_of_blocks=12,)
        if 1:
            msd_iso_dec = d.get_msd(
                    t_start_fit_fs=2000., 
                    t_end_fit_fs=4000., 
                    nr_of_blocks=12, decomposed=True)
        
        attrs = msd_iso.get_attrs()
        with open('ref/msd_iso_H2O-64-300K.json', 'r') as f:
            ref_attrs = json.load(f)
        for k in ('H', 'O'):
            self.assertEqual(ref_attrs[k], attrs[k])

        plot_msd_isotropic(msd_iso, show=True)
if __name__ == '__main__':
    unittest.main()
