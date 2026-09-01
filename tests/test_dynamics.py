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
            from samos.lib.mdutils import (
                recenter_positions, recenter_velocities)
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

        np.testing.assert_allclose(t.get_positions(),
                                   pos_fortran, atol=1e-12)
        np.testing.assert_allclose(t.get_velocities(),
                                   vel_fortran, atol=1e-12)


class TestDynamics(unittest.TestCase):
    def test_1(self):
        def compare_values(val1, val2, label):
            # print("Comparing {}: {} vs {}".format(label, val1, val2))
            # check if they are floats and compare approximately:
            if isinstance(val1, float) or isinstance(val2, float):
                if not np.isclose(val1, val2, atol=1e-6):
                    print(f"Float mismatch at '{label}': {val1} != {val2}")
                    return False
            elif isinstance(val1, (list, tuple, np.ndarray)
                            ) and isinstance(
                                val2, (list, tuple, np.ndarray)):
                if len(val1) != len(val2):
                    print(
                        "Length mismatch at "
                        f"'{label}': {len(val1)} != {len(val2)}")
                    return False
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    if not compare_values(v1, v2, f"{label}[{i}]"):
                        return False
            elif val1 != val2:
                print(f"Value mismatch at '{label}': {val1} != {val2}")
                return False
            return True

        def compare_dicts(d1, d2, name, path=''):
            """Recursively compare two dictionaries and print differences."""
            for key in d1:
                if key not in d2:
                    print(f"Key '{path + key}' missing in second dict")
                    return False
                val1 = d1[key]
                val2 = d2[key]
                if isinstance(val1, dict) and isinstance(val2, dict):
                    if not compare_dicts(val1, val2, name, path + key + '.'):
                        return False
                # not lists but iterable:
                elif hasattr(val1, '__iter__') and hasattr(val2, '__iter__'):
                    # loop and use compare_values for each element:
                    for i, (v1, v2) in enumerate(zip(val1, val2
                                                     )):
                        if not compare_values(v1, v2,
                                              f"{path + key}[{i}] in {name}"):
                            return False
                else:
                    if not compare_values(val1, val2,
                                          f"{path + key} in {name}"):
                        return False
                # check if these are floats and compare approximately:
            for key in d2:
                if key not in d1:
                    print(f"Key '{path + key}' missing in first dict")
                    return False
            return True
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

        pws = d.get_power_spectrum(smoothing=1, nr_of_blocks=6)

        vaf = d.get_vaf(t_start_fit=2., t_end_fit=4., t_unit='ps',
                        stepsize_tau=20,
                        nr_of_blocks=12, species_of_interest=['O', 'H'])

        msd_iso = d.get_msd(
            t_start_fit=2., t_end_fit=4.,
            # block_length_dt=640 @ 12.5 fs/step = 8 ps
            block_length=8., t_unit='ps',
            species_of_interest=['O', 'H'],
            backend='fortran')

        msd_iso_dec = d.get_msd(
            t_start_fit=2., t_end_fit=4., t_unit='ps',
            stepsize_tau=20,
            nr_of_blocks=12, decomposed=True,
            backend='fortran')

        for attributed_array, name in ((msd_iso, 'msd_iso'),
                                       (msd_iso_dec, 'msd_iso_dec'),
                                       (vaf, 'vaf'), (pws, 'pws')):
            attrs = attributed_array.get_attrs()
            # ~ with open('ref/{}_H2O-64-300K.json'.format(name), 'w') as f:
            # ~ json.dump(attrs , f)
            with open('ref/{}_H2O-64-300K.json'.format(name), 'r') as f:
                ref_attrs = json.load(f)
            if ref_attrs != attrs:
                # compare key by key and value by value.
                # Since the dictionary is nested, I need a
                # recursive function to compare them.
                result = compare_dicts(ref_attrs, attrs, name)
                if not result:
                    self.fail(f"Attributes of {name} do not match reference.")

        msd_iso = d.get_msd(
            t_start_fit=2., t_end_fit=4.,
            block_length=8., t_unit='ps',
            species_of_interest=['O', 'H'],
            backend='cpp')

        msd_iso_dec = d.get_msd(
            t_start_fit=2., t_end_fit=4., t_unit='ps',
            stepsize_tau=20,
            nr_of_blocks=12, decomposed=True,
            backend='cpp')

        for attributed_array, name in ((msd_iso, 'msd_iso'),
                                       (msd_iso_dec, 'msd_iso_dec')):
            attrs = attributed_array.get_attrs()
            with open('ref/{}_H2O-64-300K.json'.format(name), 'r') as f:
                ref_attrs = json.load(f)
            for key in ref_attrs.keys():
                try:
                    self.assertEqual(ref_attrs[key], attrs[key])
                except AssertionError:
                    # the c++ values do not match bit for bit because
                    # omp ordering can have slightly different
                    # rounding, but it should still be extremely
                    # close, so use numpy testing that can use tolerances
                    for subkey in ref_attrs[key]:
                        np.testing.assert_allclose(
                            attrs[key][subkey], ref_attrs[key][subkey],
                            rtol=1e-12)


class TestMSDSingleBlock(unittest.TestCase):
    """A species with a single block has no spread to report, so the
    std/sem arrays are filled with NaN.  Regression: that path used
    np.NaN, which NumPy 2.0 removed, so any single-block MSD raised
    AttributeError.  Every other test here uses several blocks."""

    def _make_trajectory(self, seed=7, nstep=200, nat=4):
        rng = np.random.default_rng(seed)
        atoms = Atoms('H2O2')
        pos = np.cumsum(rng.normal(0., 0.1, (nstep, nat, 3)), axis=0)
        return Trajectory(atoms=atoms, positions=pos, timestep=1.)

    def test_single_block_msd_fills_nan(self):
        from samos.analysis.dynamics import DynamicsAnalyzer
        d = DynamicsAnalyzer(trajectories=[self._make_trajectory()],
                             verbosity=0)
        msd = d.get_msd(t_end_fit=100, t_unit='dt', nr_of_blocks=1)

        for species in ('H', 'O'):
            std = msd.get_array('msd_isotropic_{}_std'.format(species))
            sem = msd.get_array('msd_isotropic_{}_sem'.format(species))
            self.assertTrue(np.all(np.isnan(std)))
            self.assertTrue(np.all(np.isnan(sem)))
            mean = msd.get_array('msd_isotropic_{}_mean'.format(species))
            self.assertTrue(np.all(np.isfinite(mean)))
            attrs = msd.get_attr(species)
            self.assertTrue(np.isfinite(attrs['diffusion_mean_cm2_s']))
            self.assertTrue(np.isnan(attrs['diffusion_std_cm2_s']))
            self.assertTrue(np.isnan(attrs['diffusion_sem_cm2_s']))


class TestCenterOfMassAndKineticEnergies(unittest.TestCase):
    """Regression: DynamicsAnalyzer read self._atoms, which
    set_trajectories never assigns, so do_com and get_kinetic_energies
    both raised AttributeError -- reported as 'please call
    set_trajectories', which the caller had already done."""

    def _make_trajectory(self, seed=11, nstep=400):
        rng = np.random.default_rng(seed)
        # 2 H (light, fast) and 2 O (heavy, slow)
        atoms = Atoms('H2O2')
        steps = rng.normal(0., 1., (nstep, 4, 3))
        steps[:, :2, :] *= 0.20   # H
        steps[:, 2:, :] *= 0.05   # O
        pos = np.cumsum(steps, axis=0)
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(pos)
        t.calculate_velocities_from_positions()
        return t

    def _analyzer(self, **kwargs):
        from samos.analysis.dynamics import DynamicsAnalyzer
        return DynamicsAnalyzer(trajectories=[self._make_trajectory()],
                                verbosity=0, **kwargs)

    def test_do_com_msd_runs(self):
        d = self._analyzer()
        msd = d.get_msd(t_end_fit=100, t_unit='dt', nr_of_blocks=4,
                        do_com=True)
        for species in ('H', 'O'):
            mean = msd.get_array('msd_isotropic_{}_mean'.format(species))
            self.assertTrue(np.all(np.isfinite(mean)))

    def test_do_com_msd_is_species_resolved(self):
        """The COM must be built from the species being analysed.  With
        factors=[1]*nat it was the COM of the whole system, so every
        species differed only by a constant prefactor."""
        d = self._analyzer()
        msd = d.get_msd(t_end_fit=100, t_unit='dt', nr_of_blocks=4,
                        do_com=True)
        h = msd.get_array('msd_isotropic_H_mean')
        o = msd.get_array('msd_isotropic_O_mean')
        # Same atom count, so a whole-system COM would make these two
        # curves identical.  H diffuses faster here by construction.
        self.assertFalse(np.allclose(h, o))
        self.assertGreater(h[-1], o[-1])

    def test_do_com_vaf_runs(self):
        d = self._analyzer()
        vaf = d.get_vaf(t_end_fit=50, t_end=100, t_unit='dt',
                        nr_of_blocks=4, do_com=True)
        for species in ('H', 'O'):
            mean = vaf.get_array('vaf_isotropic_{}_mean'.format(species))
            self.assertTrue(np.all(np.isfinite(mean)))

    def test_kinetic_energies_species_decomposition(self):
        """Regression: the species branch stored the system-level kinE
        array under 'species_kinetic_energy_*'."""
        d = self._analyzer()
        ke = d.get_kinetic_energies(stepsize=10, decompose_species=True)
        system = ke.get_array('system_kinetic_energy_0')
        species = ke.get_array('species_kinetic_energy_0')
        self.assertEqual(species.ndim, 2)
        self.assertEqual(species.shape, (len(system), 2))
        self.assertTrue(np.all(np.isfinite(species)))
        # each column is a distinct species, not a copy of the system array
        self.assertFalse(np.allclose(species[:, 0], species[:, 1]))

    def test_missing_trajectories_message(self):
        from samos.analysis.dynamics import DynamicsAnalyzer
        from samos.utils.exceptions import InputError
        d = DynamicsAnalyzer(verbosity=0)
        with self.assertRaises(InputError) as cm:
            d.get_msd(t_end_fit=100, t_unit='dt')
        self.assertIn('set_trajectories', str(cm.exception))


class TestBlockLayoutAndSingleBlockStatistics(unittest.TestCase):
    """Block layouts that do not fit used to reach the compute kernels
    with a zero block length (the guard only tested `< 0`), and the VAF
    divided by `nblocks - 1` without a guard."""

    def _make_trajectory(self, seed=13, nstep=300, nat=4):
        rng = np.random.default_rng(seed)
        atoms = Atoms('H2O2')
        pos = np.cumsum(rng.normal(0., 0.1, (nstep, nat, 3)), axis=0)
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(pos)
        t.calculate_velocities_from_positions()
        return t

    def _analyzer(self):
        from samos.analysis.dynamics import DynamicsAnalyzer
        return DynamicsAnalyzer(trajectories=[self._make_trajectory()],
                                verbosity=0)

    def test_too_many_blocks_msd_raises(self):
        from samos.utils.exceptions import InputError
        d = self._analyzer()
        # 300 steps, t_end_dt = 100 -> 200 usable, so 500 blocks of
        # length 0.  This used to be handed to the kernel as-is.
        with self.assertRaises(InputError) as cm:
            d.get_msd(t_end_fit=100, t_unit='dt', nr_of_blocks=500)
        self.assertIn('block', str(cm.exception).lower())

    def test_too_many_blocks_vaf_raises(self):
        from samos.utils.exceptions import InputError
        d = self._analyzer()
        with self.assertRaises(InputError):
            d.get_vaf(t_end_fit=50, t_end=100, t_unit='dt',
                      nr_of_blocks=500)

    def test_block_length_longer_than_trajectory_raises(self):
        from samos.utils.exceptions import InputError
        d = self._analyzer()
        with self.assertRaises(InputError):
            d.get_msd(t_end_fit=100, t_unit='dt', block_length=1000)

    def test_nr_of_blocks_zero_raises(self):
        from samos.utils.exceptions import InputError
        d = self._analyzer()
        with self.assertRaises(InputError):
            d.get_msd(t_end_fit=100, t_unit='dt', nr_of_blocks=0)

    def test_single_block_vaf_fills_nan_without_warning(self):
        import warnings
        d = self._analyzer()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            vaf = d.get_vaf(t_end_fit=50, t_end=100, t_unit='dt',
                            nr_of_blocks=1)

        for species in ('H', 'O'):
            mean = vaf.get_array('vaf_isotropic_{}_mean'.format(species))
            std = vaf.get_array('vaf_isotropic_{}_std'.format(species))
            sem = vaf.get_array('vaf_isotropic_{}_sem'.format(species))
            self.assertTrue(np.all(np.isfinite(mean)))
            self.assertTrue(np.all(np.isnan(std)))
            self.assertTrue(np.all(np.isnan(sem)))
            attrs = vaf.get_attr(species)
            self.assertTrue(np.isfinite(attrs['diffusion_mean_cm2_s']))
            self.assertTrue(np.isnan(attrs['diffusion_std_cm2_s']))
            self.assertTrue(np.isnan(attrs['diffusion_sem_cm2_s']))


class TestPowerSpectrumSmoothing(unittest.TestCase):
    """Regression: the smoothing kernel was np.ones((nblocks, N)), a 2-D
    kernel that also convolved over the block axis, so each block's
    spectrum absorbed its neighbours' and the std/sem were meaningless.
    The existing reference test uses smoothing=1 and cannot see this."""

    def _analyzer(self, seed=17, nstep=600, nat=4):
        import numpy as np
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(seed)
        atoms = Atoms('H2O2')
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(np.cumsum(rng.normal(0., .1, (nstep, nat, 3)),
                                  axis=0))
        t.calculate_velocities_from_positions()
        return DynamicsAnalyzer(trajectories=[t], verbosity=0)

    def test_smoothing_preserves_each_block_integral(self):
        """A running mean over frequency bins must not move weight
        between blocks."""
        d = self._analyzer()
        raw = d.get_power_spectrum(nr_of_blocks=4, smoothing=1)
        smooth = d.get_power_spectrum(nr_of_blocks=4, smoothing=5)
        a = raw.get_array('periodogram_H_0')
        b = smooth.get_array('periodogram_H_0')
        self.assertEqual(a.shape, b.shape)
        # per-block sums survive smoothing up to the zero-padded edges
        np.testing.assert_allclose(a.sum(axis=1), b.sum(axis=1), rtol=5e-2)

    def test_blocks_stay_independent(self):
        """Scaling one block's velocities must not change the smoothed
        spectrum of the other blocks."""
        d = self._analyzer()
        base = d.get_power_spectrum(
            nr_of_blocks=2, smoothing=5).get_array('periodogram_H_0')

        traj = d._trajectories[0]
        vel = traj.get_velocities().copy()
        vel[:len(vel) // 2] *= 10.0          # perturb block 0 only
        traj.set_velocities(vel)
        perturbed = d.get_power_spectrum(
            nr_of_blocks=2, smoothing=5).get_array('periodogram_H_0')

        self.assertFalse(np.allclose(base[0], perturbed[0]))
        np.testing.assert_allclose(base[1], perturbed[1], rtol=1e-10)


class TestKineticEnergiesVectorized(unittest.TestCase):
    """get_kinetic_energies ran a Python loop over steps x atoms x
    polarizations.  The einsum replacement must reproduce it exactly."""

    def _analyzer(self, seed=23, nstep=40, stepsize=3):
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(seed)
        atoms = Atoms('H2O2')
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(rng.random((nstep, 4, 3)))
        t.set_velocities(rng.normal(0., 0.01, (nstep, 4, 3)))
        self.stepsize = stepsize
        self.traj = t
        return DynamicsAnalyzer(trajectories=[t], verbosity=0)

    def _reference(self, kind):
        """The original triple-nested loop, kept here as the oracle."""
        from samos.utils.constants import amu_kg, kB
        prefactor = amu_kg * 1e10 / kB
        masses = self.traj.atoms.get_masses()
        vel_array = self.traj.get_velocities()
        nstep, nat, _ = vel_array.shape
        steps = list(range(0, nstep, self.stepsize))
        if kind == 'system':
            out = np.zeros(len(steps))
            for i0, istep in enumerate(steps):
                for iat in range(nat):
                    for ipol in range(3):
                        out[i0] += (prefactor * masses[iat]
                                    * vel_array[istep, iat, ipol]**2)
            out /= nat * 3
            return out
        out = np.zeros((len(steps), nat))
        for i0, istep in enumerate(steps):
            for iat in range(nat):
                for ipol in range(3):
                    out[i0, iat] += (prefactor * masses[iat]
                                     * vel_array[istep, iat, ipol]**2 / 3.)
        return out

    def test_system_matches_the_loop(self):
        d = self._analyzer()
        ke = d.get_kinetic_energies(stepsize=self.stepsize)
        np.testing.assert_allclose(
            ke.get_array('system_kinetic_energy_0'),
            self._reference('system'), rtol=1e-12)

    def test_atoms_matches_the_loop(self):
        d = self._analyzer()
        ke = d.get_kinetic_energies(stepsize=self.stepsize,
                                    decompose_system=False,
                                    decompose_atoms=True)
        np.testing.assert_allclose(
            ke.get_array('atoms_kinetic_energy_0'),
            self._reference('atoms'), rtol=1e-12)

    def test_species_is_the_mean_over_its_atoms(self):
        d = self._analyzer()
        ke = d.get_kinetic_energies(stepsize=self.stepsize,
                                    decompose_species=True)
        per_atom = self._reference('atoms')
        species = ke.get_array('species_kinetic_energy_0')
        for ityp, sym in enumerate(ke.get_attr('species_of_interest')):
            idx = self.traj.get_indices_of_species(sym, start=0)
            np.testing.assert_allclose(
                species[:, ityp], per_atom[:, idx].mean(axis=1),
                rtol=1e-12)


class TestSmoothingRename(unittest.TestCase):
    """'smothening' was the public spelling; 'smoothing' replaces it and
    the old name is accepted with a DeprecationWarning."""

    def _analyzer(self):
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(31)
        t = Trajectory(atoms=Atoms('H2O2'), timestep=1.)
        t.set_positions(np.cumsum(rng.normal(0., .1, (200, 4, 3)), axis=0))
        t.calculate_velocities_from_positions()
        return DynamicsAnalyzer(trajectories=[t], verbosity=0)

    def test_old_spelling_warns_but_works(self):
        import warnings
        d = self._analyzer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            old = d.get_power_spectrum(nr_of_blocks=2, smothening=5)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning)
                            for w in caught))
        new = d.get_power_spectrum(nr_of_blocks=2, smoothing=5)
        np.testing.assert_allclose(new.get_array('periodogram_H_0'),
                                   old.get_array('periodogram_H_0'))

    def test_too_many_blocks_rejected(self):
        from samos.utils.exceptions import InputError
        with self.assertRaises(InputError):
            self._analyzer().get_power_spectrum(nr_of_blocks=500)


class TestMSDPlottingFitWindows(unittest.TestCase):
    """get_msd accepts list-valued t_start_fit/t_end_fit, but the
    plotters divided those lists by stepsize (TypeError) and indexed the
    window axis of slopes_intercepts_* as if it were the block axis."""

    def _msd(self, decomposed=False, **fit):
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(41)
        t = Trajectory(atoms=Atoms('H2O2'), timestep=1.)
        t.set_positions(np.cumsum(rng.normal(0., .1, (400, 4, 3)), axis=0))
        d = DynamicsAnalyzer(trajectories=[t], verbosity=0)
        return d.get_msd(t_unit='dt', nr_of_blocks=3,
                         decomposed=decomposed, **fit)

    def _axes(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        return fig.add_subplot(1, 1, 1)

    def test_single_window_still_plots(self):
        from samos.plotting.plot_dynamics import plot_msd_isotropic
        msd = self._msd(t_start_fit=10, t_end_fit=100)
        self.assertFalse(msd.get_attr('multiple_params_fit'))
        plot_msd_isotropic(msd, ax=self._axes())

    def test_multiple_windows_plot(self):
        from samos.plotting.plot_dynamics import plot_msd_isotropic
        msd = self._msd(t_start_fit=[10, 50], t_end_fit=[100, 150])
        self.assertTrue(msd.get_attr('multiple_params_fit'))
        ax = self._axes()
        plot_msd_isotropic(msd, ax=ax)
        # one dashed fit line per (window, block) per species
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == '--']
        self.assertEqual(len(dashed), 2 * 3 * 2)

    def test_multiple_windows_plot_decomposed(self):
        from samos.plotting.plot_dynamics import plot_msd_anisotropic
        msd = self._msd(decomposed=True,
                        t_start_fit=[10, 50], t_end_fit=[100, 150])
        plot_msd_anisotropic(msd, ax=self._axes(), diagonal_only=True)


if __name__ == '__main__':
    unittest.main()
