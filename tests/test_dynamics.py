# -*- coding: utf-8 -*-

import unittest
import numpy as np
from ase import Atoms
from samos.trajectory import Trajectory


class TestRecenter(unittest.TestCase):
    """Tests for Trajectory.recenter(), which replaced the Fortran
    recenter_positions / recenter_velocities routines."""

    def _make_trajectory(self, seed=42):
        rng = np.random.default_rng(seed)
        # 5 atoms (3 H, 2 O), 20 steps
        atoms = Atoms('H3O2')
        pos = rng.random((20, 5, 3))
        vel = rng.random((20, 5, 3))
        t = Trajectory(atoms=atoms, timestep=1.0)
        t.set_positions(pos)
        t.set_velocities(vel)
        return t

    def _weighted_com(self, array, rel_masses):
        """array: (nstep, nat, 3), rel_masses: (nat,) normalised"""
        return np.einsum('a,sac->sc', rel_masses, array)  # (nstep, 3)

    def test_recenter_full_com_is_zero(self):
        t = self._make_trajectory()
        masses = t.atoms.get_masses()
        rel_masses = masses / masses.sum()

        t.recenter()

        com_pos = self._weighted_com(t.get_positions(), rel_masses)
        com_vel = self._weighted_com(t.get_velocities(), rel_masses)
        np.testing.assert_allclose(com_pos, 0.0, atol=1e-12)
        np.testing.assert_allclose(com_vel, 0.0, atol=1e-12)

    def test_recenter_geometric_com_is_zero(self):
        t = self._make_trajectory()
        nat = len(t.atoms)
        rel_masses = np.ones(nat) / nat

        t.recenter(mode='geometric')

        com_pos = self._weighted_com(t.get_positions(), rel_masses)
        np.testing.assert_allclose(com_pos, 0.0, atol=1e-12)

    def test_recenter_sublattice_com_is_zero(self):
        """Only the O sublattice COM should be zeroed."""
        t = self._make_trajectory()
        masses = t.atoms.get_masses()
        # O atoms are indices 3 and 4 in 'H3O2'
        factors = np.array([0, 0, 0, 1, 1], dtype=float)
        rel_masses = (factors * masses) / (factors * masses).sum()

        t.recenter(sublattice=['O'])

        com_pos = self._weighted_com(t.get_positions(), rel_masses)
        np.testing.assert_allclose(com_pos, 0.0, atol=1e-12)

    def test_recenter_matches_fortran(self):
        """Cross-check numpy result against the original Fortran routines.
        Skipped automatically if the Fortran extension is not compiled."""
        try:
            from samos.lib.mdutils import recenter_positions, recenter_velocities
        except ImportError:
            self.skipTest("Fortran mdutils extension not available")

        t = self._make_trajectory()
        masses = t.atoms.get_masses().astype(float)
        factors = np.ones(len(t.atoms), dtype=int)

        pos = t.get_positions()
        vel = t.get_velocities()

        pos_fortran = recenter_positions(pos, masses, factors)
        vel_fortran = recenter_velocities(vel, masses, factors)

        t.recenter()

        np.testing.assert_allclose(t.get_positions(), pos_fortran, atol=1e-12)
        np.testing.assert_allclose(t.get_velocities(), vel_fortran, atol=1e-12)


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
