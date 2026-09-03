# -*- coding: utf-8 -*-

import unittest
import numpy as np
from ase import Atoms
from samos.trajectory import Trajectory


class TestRecenter(unittest.TestCase):
    """Tests for Trajectory.recenter().

    It replaced the Fortran recenter_positions / recenter_velocities,
    which have since been deleted.  The cross-check against them went
    too; what is left states the property directly, which is the
    stronger test anyway: the centre of mass must come out at zero."""

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
            species_of_interest=['O', 'H'])

        msd_iso_dec = d.get_msd(
            t_start_fit=2., t_end_fit=4., t_unit='ps',
            stepsize_tau=20,
            nr_of_blocks=12, decomposed=True)

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

    def test_do_com_ignores_atom_indices(self):
        """atom_indices picks out individual atoms, and do_com leaves
        none -- the species has been collapsed onto a single centre of
        mass.  The filter used to run anyway and emptied the selection
        whenever atom_indices did not happen to contain index 1, which
        handed the Fortran kernel zero atoms."""
        d = self._analyzer()
        kw = dict(t_end_fit=100, t_unit='dt', nr_of_blocks=4, do_com=True)
        plain = d.get_msd(**kw)
        filtered = d.get_msd(atom_indices=[2, 3], **kw)
        for species in ('H', 'O'):
            key = 'msd_isotropic_{}_mean'.format(species)
            np.testing.assert_array_equal(
                plain.get_array(key), filtered.get_array(key))

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

    def test_scalar_window_keeps_its_result_shape(self):
        """The fit is built one way now, with a leading window axis.
        For a scalar fit window that axis is taken off again, so the
        stored shapes stay what a scalar window always produced."""
        iso = self._msd(t_start_fit=10, t_end_fit=100)
        self.assertEqual(
            iso.get_array('slopes_intercepts_isotropic_H_0').shape,
            (3, 2))
        dec = self._msd(decomposed=True, t_start_fit=10, t_end_fit=100)
        self.assertEqual(
            dec.get_array('slopes_intercepts_decomposed_H_0').shape,
            (3, 3, 3, 2))
        # One number per species, not a one-element array.
        self.assertEqual(
            np.shape(iso.get_attr('H')['diffusion_mean_cm2_s']), ())

    def test_list_window_adds_a_leading_axis(self):
        iso = self._msd(t_start_fit=[10, 50], t_end_fit=[100, 150])
        self.assertEqual(
            iso.get_array('slopes_intercepts_isotropic_H_0').shape,
            (2, 3, 2))
        dec = self._msd(decomposed=True,
                        t_start_fit=[10, 50], t_end_fit=[100, 150])
        self.assertEqual(
            dec.get_array('slopes_intercepts_decomposed_H_0').shape,
            (2, 3, 3, 3, 2))
        self.assertEqual(
            len(iso.get_attr('H')['diffusion_mean_cm2_s']), 2)


class TestPowerSpectrumSpread(unittest.TestCase):
    """
    The standard error was computed as std/sqrt(nblocks - 1), so the
    default single-block run divided by zero: numpy emitted a
    RuntimeWarning and the stored sem array was entirely NaN.  get_msd
    and get_vaf already filled NaN explicitly in that case.
    """

    def _spectrum(self, nr_of_blocks):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(3)
        t = Trajectory(atoms=Atoms('H' * 4), timestep=1.0)
        t.set_velocities(rng.random((200, 4, 3)))
        return DynamicsAnalyzer(trajectories=[t]).get_power_spectrum(
            nr_of_blocks=nr_of_blocks)

    def test_single_block_does_not_warn(self):
        # Warnings are recorded rather than raised: the division sits
        # inside a bare `except Exception: print(e)`, which would
        # swallow an escalated warning before the test could see it.
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self._spectrum(1)
        divides = [w for w in caught
                   if issubclass(w.category, RuntimeWarning)
                   and 'divide' in str(w.message)]
        self.assertEqual(divides, [])

    def test_single_block_reports_no_spread(self):
        import numpy as np
        ps = self._spectrum(1)
        mean = ps.get_array('periodogram_H_mean')
        self.assertTrue(np.isfinite(mean).all())
        for name in ('periodogram_H_std', 'periodogram_H_sem'):
            spread = ps.get_array(name)
            self.assertEqual(spread.shape, mean.shape)
            self.assertTrue(np.isnan(spread).all(), name)

    def test_several_blocks_report_a_finite_spread(self):
        import numpy as np
        ps = self._spectrum(4)
        for name in ('periodogram_H_std', 'periodogram_H_sem'):
            self.assertTrue(np.isfinite(ps.get_array(name)).all(), name)
        np.testing.assert_allclose(
            ps.get_array('periodogram_H_sem'),
            ps.get_array('periodogram_H_std') / np.sqrt(3))


class TestSlicedTrajectoryTimeAxis(unittest.TestCase):
    """
    Striding a trajectory has to carry through to the analysis: the MSD
    builds its time axis from timestep_fs alone, so if slice_steps did
    not scale the timestep, every lag of a strided trajectory would be
    reported at a fraction of its real value.
    """

    def _traj(self, nstep=200, nat=4):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        rng = np.random.default_rng(11)
        walk = np.cumsum(rng.normal(scale=0.1, size=(nstep, nat, 3)), axis=0)
        t = Trajectory(atoms=Atoms('H' * nat), timestep=1.0)
        t.set_positions(walk)
        return t

    def _t_list(self, traj):
        from samos.analysis.dynamics import DynamicsAnalyzer
        dyn = DynamicsAnalyzer(trajectories=[traj])
        msd = dyn.get_msd(t_start_fit=10., t_end_fit=50., t_unit='fs',
                          nr_of_blocks=1)
        return msd.get_array('t_list_fs')

    def test_stride_widens_the_lag_spacing(self):
        import numpy as np
        full = self._traj()
        sliced = full.slice_steps(slice(None, None, 2))
        self.assertEqual(sliced.get_timestep(), 2.0)

        t_full = self._t_list(full)
        t_sliced = self._t_list(sliced)
        # Lags are twice as far apart, and both axes are in the same
        # femtoseconds and start at the same place.
        self.assertAlmostEqual(np.diff(t_sliced)[0],
                               2.0 * np.diff(t_full)[0])
        self.assertAlmostEqual(t_sliced[0], t_full[0])
        # Every lag of the strided run coincides with one of the full run.
        np.testing.assert_allclose(t_sliced, t_full[::2][:len(t_sliced)])


class TestFitWindowRequirement(unittest.TestCase):
    """get_power_spectrum used to parse block_length/nr_of_blocks by
    itself because _get_running_params insists on a fit window.  That
    requirement is now driven by require_fitting, so the periodogram
    can share the parsing and the block layout with its siblings."""

    def _analyzer(self, nstep=600, nat=4, seed=17):
        import numpy as np
        from samos.analysis.dynamics import DynamicsAnalyzer
        rng = np.random.default_rng(seed)
        t = Trajectory(atoms=Atoms('H2O2'), timestep=1.)
        t.set_positions(np.cumsum(rng.normal(0., .1, (nstep, nat, 3)),
                                  axis=0))
        t.calculate_velocities_from_positions()
        return DynamicsAnalyzer(trajectories=[t], verbosity=0)

    def test_msd_still_requires_a_fit_window(self):
        from samos.utils.exceptions import InputError
        with self.assertRaises(InputError) as cm:
            self._analyzer().get_msd(nr_of_blocks=2)
        self.assertIn('t_end_fit', str(cm.exception))

    def test_vaf_still_requires_a_fit_window(self):
        from samos.utils.exceptions import InputError
        with self.assertRaises(InputError) as cm:
            self._analyzer().get_vaf(nr_of_blocks=2)
        self.assertIn('t_end_fit', str(cm.exception))

    def test_params_without_fitting_reserve_no_lag_window(self):
        d = self._analyzer()
        p = d._get_running_params(1., nr_of_blocks=3,
                                  require_fitting=False)
        self.assertIsNone(p.t_start_fit_dt)
        self.assertIsNone(p.t_end_fit_dt)
        # t_end_dt drives how many steps _resolve_blocks holds back.
        self.assertEqual(p.t_end_dt, 0)
        self.assertEqual(p.nr_of_blocks, 3)

    def test_fit_window_rejected_rather_than_ignored(self):
        from samos.utils.exceptions import InputError
        d = self._analyzer()
        with self.assertRaises(InputError):
            d._get_running_params(1., t_end_fit=10, t_unit='dt',
                                  require_fitting=False)

    def test_blocks_tile_the_whole_trajectory(self):
        """No lag window is reserved, so a block is nstep // nblocks
        steps long and the one-sided periodogram has half as many
        frequency bins plus one."""
        nstep = 600
        d = self._analyzer(nstep=nstep)
        for nr_of_blocks in (1, 2, 7, 13):
            pws = d.get_power_spectrum(nr_of_blocks=nr_of_blocks)
            expected = (nstep // nr_of_blocks) // 2 + 1
            self.assertEqual(len(pws.get_array('frequency_0')), expected,
                             'nr_of_blocks={}'.format(nr_of_blocks))
            self.assertEqual(
                pws.get_array('periodogram_H_0').shape,
                (nr_of_blocks, expected))

    def test_nr_of_blocks_and_block_length_agree(self):
        d = self._analyzer(nstep=600)
        by_count = d.get_power_spectrum(nr_of_blocks=4)
        by_length = d.get_power_spectrum(block_length=150, t_unit='dt')
        for name in ('frequency_0', 'periodogram_H_0', 'periodogram_O_0'):
            np.testing.assert_array_equal(by_count.get_array(name),
                                          by_length.get_array(name))

    def test_zero_blocks_rejected(self):
        """_resolve_blocks validates the request; the inline layout the
        periodogram used to carry did not."""
        from samos.utils.exceptions import InputError
        with self.assertRaises(InputError):
            self._analyzer().get_power_spectrum(nr_of_blocks=0)

    def test_too_many_blocks_message_omits_the_lag_hint(self):
        """Advising a smaller t_end would send the caller after a knob
        that does nothing for a periodogram."""
        from samos.utils.exceptions import InputError
        with self.assertRaises(InputError) as cm:
            self._analyzer().get_power_spectrum(nr_of_blocks=5000)
        self.assertNotIn('t_end', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
