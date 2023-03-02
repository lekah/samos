# -*- coding: utf-8 -*-

import unittest


class TestDynamics(unittest.TestCase):

    def test_1(self):
        from samos.trajectory import Trajectory
        from samos.analysis.dynamics import DynamicsAnalyzer
        from samos.utils.constants import bohr_to_ang
        import json
        t = Trajectory.load_file('data/H2O-64-300K.tar.gz')
        t.recenter()
        t.rescale_array(t._VELOCITIES_KEY, bohr_to_ang)
        t.rescale_array(t._POSITIONS_KEY, bohr_to_ang)
        d = DynamicsAnalyzer(verbosity=0)

        d.set_trajectories(t)

        pws = d.get_power_spectrum(smothening=1, nr_of_blocks=6)

        vaf = d.get_vaf(t_start_fit_fs=2000.,
                        stepsize_tau=20, t_end_fit_fs=4000.,
                        nr_of_blocks=12, species_of_interest=['O', 'H'])

        msd_iso = d.get_msd(
            t_start_fit_fs=2000.,
            t_end_fit_fs=4000.,
            # ~ nr_of_blocks=12,)
            block_length_dt=640, species_of_interest=['O', 'H']
        )
        msd_iso_dec = d.get_msd(
            t_start_fit_fs=2000.,
            t_end_fit_fs=4000., stepsize_tau=20,
            nr_of_blocks=12, decomposed=True)

        for attributed_array, name in ((msd_iso, 'msd_iso'),
                                       (msd_iso_dec, 'msd_iso_dec'),
                                       (vaf, 'vaf'), (pws, 'pws')):
            attrs = attributed_array.get_attrs()
            # ~ with open('ref/{}_H2O-64-300K.json'.format(name), 'w') as f:
            # ~ json.dump(attrs , f)
            with open('ref/{}_H2O-64-300K.json'.format(name), 'r') as f:
                ref_attrs = json.load(f)
            self.assertEqual(ref_attrs, attrs)

        # Uncomment to test plot:
        # ~ from matplotlib import pyplot as plt
        # ~ fig = plt.figure(figsize=(12,7))
        # ~ plt.suptitle(r'Diffusion TIP4P-$H_2O$ at 300K', fontsize=18)
        # ~ plot_msd_isotropic(msd_iso, fig.add_subplot(3,1,1))
        # ~ plot_vaf_isotropic(vaf, fig.add_subplot(3,1,2))
        # ~ plot_power_spectrum(pws, fig.add_subplot(3,1,3), )
        # ~ plt.show()


if __name__ == '__main__':
    unittest.main()
