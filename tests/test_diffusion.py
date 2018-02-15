import unittest

class TestDiffusion(unittest.TestCase):

    def test_1(self):
        from samos.trajectory import Trajectory
        from samos.analysis.get_diffusion import DiffusionAnalyzer
        from samos.plotting.plot_diffusion import plot_msd_isotropic
        import json
        t = Trajectory.load_file('data/H2O-64-300K.tar.gz')
        t.recenter()
        d = DiffusionAnalyzer(verbosity=0)
        
        d.set_trajectories(t)
        msd_iso = d.get_msd_isotropic(
                t_start_fit_fs=2000., 
                t_end_fit_fs=4000., 
                nr_of_blocks=12,)
        msd_iso_com = d.get_msd_isotropic(
                t_start_fit_fs=2000., 
                t_end_fit_fs=4000., 
                nr_of_blocks=12, species_of_interest='O', do_com=True)
        #~ vaf_iso = d.get_vaf_isotropic(t_start_fit_fs=2000., 
                #~ t_end_fit_fs=4000., 
                #~ nr_of_blocks=12,)
            
        attrs = msd_iso.get_attrs()
        with open('ref/msd_iso_H2O-64-300K.json', 'r') as f:
            #~ json.dump(attrs, f)
            ref_attrs = json.load(f)
        self.assertEqual(ref_attrs, attrs)


        plot_msd_isotropic(msd_iso, show=True)
if __name__ == '__main__':
    unittest.main()
