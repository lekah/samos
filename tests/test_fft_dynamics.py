# -*- coding: utf-8 -*-
"""Tests for samos.analysis._fft_dynamics, the numpy/FFT replacement
for samos.lib.mdutils (issue #1).

Checked against the mathematical definition directly via independent
brute-force (nested-loop) reference implementations here -- not
against the Fortran, which has its own bugs (see test_dynamics.py's
TestEngineAgreement and README.md's "MSD time axis" entry). This file
exercises the kernels in isolation; TestEngineAgreement in
test_dynamics.py exercises them through the public DynamicsAnalyzer
API, including the block-resolution and do_com/do_long wiring around
them.
"""

import unittest
import warnings

import numpy as np

from samos.analysis import _fft_dynamics as fd


def _brute_msd(positions, idx0, stepsize_t, block_length_dt, nr_of_blocks,
               nr_of_t):
    nat_of_interest = len(idx0)
    msd = np.zeros((nr_of_blocks, nr_of_t))
    for ib in range(nr_of_blocks):
        start = ib * block_length_dt
        for t in range(nr_of_t):
            shift = stepsize_t * t
            total = 0.0
            for iat in idx0:
                for tau in range(start, start + block_length_dt):
                    d = positions[tau + shift, iat] - positions[tau, iat]
                    total += np.dot(d, d)
            msd[ib, t] = total
    return msd / (block_length_dt * nat_of_interest)


def _brute_msd_max_stats(positions, idx0, stepsize_t, nr_of_t):
    n = positions.shape[0]
    msd = np.zeros(nr_of_t)
    for t in range(nr_of_t):
        shift = stepsize_t * t
        total = 0.0
        count = 0
        for iat in idx0:
            for tau in range(0, n - shift):
                d = positions[tau + shift, iat] - positions[tau, iat]
                total += np.dot(d, d)
                count += 1
        msd[t] = total / count
    return msd


def _brute_decompose(positions, idx0, stepsize_t, block_length_dt,
                     nr_of_blocks, nr_of_t):
    nat_of_interest = len(idx0)
    msd = np.zeros((nr_of_blocks, nr_of_t, 3, 3))
    for ib in range(nr_of_blocks):
        start = ib * block_length_dt
        for t in range(nr_of_t):
            shift = stepsize_t * t
            for ipol in range(3):
                for jpol in range(3):
                    total = 0.0
                    for iat in idx0:
                        for tau in range(start, start + block_length_dt):
                            di = (positions[tau + shift, iat, ipol]
                                  - positions[tau, iat, ipol])
                            dj = (positions[tau + shift, iat, jpol]
                                  - positions[tau, iat, jpol])
                            total += di * dj
                    msd[ib, t, ipol, jpol] = total
    return msd / (block_length_dt * nat_of_interest)


def _brute_vaf(velocities, idx0, stepsize_t, block_length_dt, nr_of_blocks,
               nr_of_t):
    nat_of_interest = len(idx0)
    vaf = np.zeros((nr_of_blocks, nr_of_t))
    for ib in range(nr_of_blocks):
        start = ib * block_length_dt
        for t in range(nr_of_t):
            shift = stepsize_t * t
            total = 0.0
            for iat in idx0:
                for tau in range(start, start + block_length_dt):
                    total += np.dot(velocities[tau + shift, iat],
                                    velocities[tau, iat])
            vaf[ib, t] = total
    return vaf / (block_length_dt * nat_of_interest)


class TestAgainstBruteForce(unittest.TestCase):
    """Every kernel checked against a plain nested-loop implementation
    of the same definition -- deliberately not derived from, or
    compared against, the Fortran, so a shared misunderstanding can't
    hide in both."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.nstep, self.nat = 60, 5
        self.positions = np.cumsum(
            rng.normal(0, 0.3, (self.nstep, self.nat, 3)), axis=0)
        self.velocities = rng.normal(0, 1.0, (self.nstep, self.nat, 3))
        self.idx0 = np.array([0, 2, 3])
        self.idx1 = self.idx0 + 1  # 1-based, the public convention
        self.stepsize_t = 2
        self.block_length_dt = 10
        self.nr_of_blocks = 3
        self.nr_of_t = 5
        self.nat_of_interest = len(self.idx0)

    def test_msd(self):
        got = fd.calculate_msd_specific_atoms(
            self.positions, self.idx1, self.stepsize_t, 1,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        want = _brute_msd(
            self.positions, self.idx0, self.stepsize_t,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_msd_lag_zero_is_the_first_column(self):
        # The fixed convention (see module docstring): index 0 is lag
        # 0, trivially zero -- unlike the old Fortran, which never
        # computed lag 0 at all.
        got = fd.calculate_msd_specific_atoms(
            self.positions, self.idx1, self.stepsize_t, 1,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        np.testing.assert_allclose(got[:, 0], 0.0, atol=1e-10)

    def test_msd_max_stats(self):
        got = fd.calculate_msd_specific_atoms_max_stats(
            self.positions, self.idx1, self.stepsize_t, 1, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        want = _brute_msd_max_stats(
            self.positions, self.idx0, self.stepsize_t, self.nr_of_t)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_decompose_d(self):
        got = fd.calculate_msd_specific_atoms_decompose_d(
            self.positions, self.idx1, self.stepsize_t, 1,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        want = _brute_decompose(
            self.positions, self.idx0, self.stepsize_t,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_decompose_d_trace_matches_isotropic_msd(self):
        # Sum of the diagonal is the same quantity calculate_msd_
        # specific_atoms reports directly -- a cross-check between the
        # two routines, not just against the brute-force reference.
        decomposed = fd.calculate_msd_specific_atoms_decompose_d(
            self.positions, self.idx1, self.stepsize_t, 1,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        isotropic = fd.calculate_msd_specific_atoms(
            self.positions, self.idx1, self.stepsize_t, 1,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t,
            self.nstep, self.nat, self.nat_of_interest)
        trace = decomposed[:, :, 0, 0] + decomposed[:, :, 1, 1] \
            + decomposed[:, :, 2, 2]
        np.testing.assert_allclose(trace, isotropic, rtol=1e-10, atol=1e-12)

    def test_vaf(self):
        got, _ = fd.calculate_vaf_specific_atoms(
            self.velocities, self.idx1, self.stepsize_t, 1, self.nr_of_t,
            self.nr_of_blocks, self.block_length_dt, 0.5, 'trapezoid',
            self.nstep, self.nat, self.nat_of_interest)
        want = _brute_vaf(
            self.velocities, self.idx0, self.stepsize_t,
            self.block_length_dt, self.nr_of_blocks, self.nr_of_t)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_vaf_integral_trapezoid(self):
        vaf, got_int = fd.calculate_vaf_specific_atoms(
            self.velocities, self.idx1, self.stepsize_t, 1, self.nr_of_t,
            self.nr_of_blocks, self.block_length_dt, 0.5, 'trapezoid',
            self.nstep, self.nat, self.nat_of_interest)
        want_int = np.zeros_like(vaf)
        want_int[:, 0] = 0.5 * 0.5 * vaf[:, 0]
        for t in range(1, self.nr_of_t):
            if t == self.nr_of_t - 1:
                want_int[:, t] = 0.5 * 0.5 * vaf[:, t] + want_int[:, t - 1]
            else:
                want_int[:, t] = 0.5 * vaf[:, t] + want_int[:, t - 1]
        np.testing.assert_allclose(got_int, want_int, rtol=1e-10, atol=1e-12)


class TestComKernels(unittest.TestCase):

    def test_get_com_positions_matches_weighted_mean(self):
        rng = np.random.default_rng(3)
        nstep, nat = 15, 6
        positions = rng.normal(size=(nstep, nat, 3))
        masses = rng.uniform(1, 20, nat)
        factors = np.array([1, 0, 1, 1, 0, 1])
        got = fd.get_com_positions(positions, masses, factors)
        sel = factors.astype(bool)
        want = (np.einsum('a,sac->sc', masses[sel], positions[:, sel, :])
                / masses[sel].sum())
        self.assertEqual(got.shape, (nstep, 1, 3))
        np.testing.assert_allclose(got[:, 0, :], want, rtol=1e-12)

    def test_get_com_velocities_same_formula(self):
        rng = np.random.default_rng(4)
        nstep, nat = 10, 4
        velocities = rng.normal(size=(nstep, nat, 3))
        masses = rng.uniform(1, 20, nat)
        factors = np.array([1, 1, 0, 1])
        got_pos_kernel = fd.get_com_positions(velocities, masses, factors)
        got_vel_kernel = fd.get_com_velocities(velocities, masses, factors)
        np.testing.assert_array_equal(got_pos_kernel, got_vel_kernel)


class TestStepsizeTauIgnored(unittest.TestCase):
    """stepsize_tau no longer buys the caller anything (issue #6): the
    numpy engine always uses every origin, warns if asked to do
    otherwise, and gives the identical result regardless of the value
    passed."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.positions = np.cumsum(
            rng.normal(0, 0.3, (40, 4, 3)), axis=0)
        self.velocities = rng.normal(size=(40, 4, 3))
        self.idx1 = np.array([1, 2, 3, 4])

    def test_msd_warns_and_is_unaffected(self):
        kwargs = dict(block_length_dt=8, nr_of_blocks=3, nr_of_t=4,
                      nstep=40, nat=4, nat_of_interest=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            at_one = fd.calculate_msd_specific_atoms(
                self.positions, self.idx1, 1, 1, **kwargs)
            self.assertEqual(len(caught), 0)
            at_ten = fd.calculate_msd_specific_atoms(
                self.positions, self.idx1, 1, 10, **kwargs)
            self.assertEqual(len(caught), 1)
            self.assertIn('stepsize_tau', str(caught[0].message))
        np.testing.assert_array_equal(at_one, at_ten)

    def test_vaf_warns_and_is_unaffected(self):
        kwargs = dict(nr_of_t=4, nr_of_blocks=3, block_length_dt=8,
                      deltaT=0.5, integration_method='trapezoid',
                      nstep=40, nat=4, nat_of_interest=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            vaf_one, int_one = fd.calculate_vaf_specific_atoms(
                self.velocities, self.idx1, 1, 1, **kwargs)
            vaf_ten, int_ten = fd.calculate_vaf_specific_atoms(
                self.velocities, self.idx1, 1, 10, **kwargs)
            self.assertEqual(len(caught), 1)
        np.testing.assert_array_equal(vaf_one, vaf_ten)
        np.testing.assert_array_equal(int_one, int_ten)

    def test_max_stats_never_warns(self):
        # The Fortran body never used stepsize_tau either -- there was
        # never a slow path here for it to speed up, so passing it is
        # not a behaviour change and shouldn't warn.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fd.calculate_msd_specific_atoms_max_stats(
                self.positions, self.idx1, 1, 10, 4, 40, 4, 4)
            self.assertEqual(len(caught), 0)


if __name__ == '__main__':
    unittest.main()
