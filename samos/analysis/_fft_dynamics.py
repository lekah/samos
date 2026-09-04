# -*- coding: utf-8 -*-
"""
Pure numpy/scipy replacement for samos/lib/mdutils.f90 (issue #1).
That Fortran file is no longer compiled -- dynamics.py's get_msd and
get_vaf call these functions unconditionally -- but it is kept in the
repository for reference, with a note at its top pointing back here.

Every function has the same positional signature its Fortran
counterpart had (minus the intent(out) argument, which becomes the
return value), which is why dynamics.py's call sites needed no change
beyond the import. See issues.md issue #1 for the identity these are
built on, and the "new math" writeup in the implementation plan
(conversation history) for the per-routine derivations.

Lag convention: index t = 0 .. nr_of_t-1 corresponds to raw lag
``stepsize_t * t``, t=0 included. The Fortran MSD routines instead
looped ``t = 1, nr_of_t`` (lag 0 never computed, everything shifted
one slot from what t_list_fs claims), a pre-existing bug independent
of this rewrite -- fixed here rather than reproduced. VAF's Fortran
loop already started at t=0, so its behaviour is unchanged.

stepsize_tau: accepted for signature compatibility, but every origin
in a block is now used regardless of its value -- with an O(N log N)
correlation there is no cost to using all of them, so subsampling
buys nothing (see issue #6). A value other than 1 is warned about and
ignored, except for ..._max_stats, whose Fortran body never used the
parameter to begin with.
"""

from warnings import warn

import numpy as np
from scipy.signal import correlate


def _warn_if_subsampled(stepsize_tau):
    if stepsize_tau != 1:
        warn(
            'stepsize_tau={} was requested, but the numpy engine uses '
            'every time origin regardless (it is no longer the '
            'expensive part) -- see issue #6. Pass stepsize_tau=1, or '
            'nothing, to silence this.'.format(stepsize_tau),
            stacklevel=3)


def _window_sums(x, start, length, max_shift):
    """
    ``sum(x[start+shift : start+length+shift])`` for shift=0..max_shift,
    for a 1-D array *x*, via one prefix sum -- O(1) per shift.
    """
    prefix = np.concatenate(([0.0], np.cumsum(x)))
    shifts = np.arange(max_shift + 1)
    return prefix[start + length + shifts] - prefix[start + shifts]


def _block_correlate(target, origin, start, length, max_shift):
    """
    ``sum(target[start:start+length+shift] * origin[start:start+length])``
    aligned so index k of the result is
    ``sum_tau target[tau+k] * origin[tau]`` for tau in the block --
    for shift=0..max_shift, via one FFT-based correlation.
    """
    g = origin[start:start + length]
    h = target[start:start + length + max_shift]
    return correlate(h, g, mode='valid', method='fft')


def get_com_positions(positions, masses, factors):
    """Mass-weighted centre of mass, restricted to atoms with a
    nonzero *factors* entry. Shape (nstep, 1, 3), matching the
    Fortran's declared output."""
    rel_masses = factors * masses
    rel_masses = rel_masses / rel_masses.sum()
    com = np.einsum('a,sac->sc', rel_masses, positions)
    return com[:, None, :]


def get_com_velocities(velocities, masses, factors):
    """Same as get_com_positions, for velocities."""
    return get_com_positions(velocities, masses, factors)


def calculate_msd_specific_atoms(positions, indices_of_interest,
                                 stepsize_t, stepsize_tau, block_length_dt,
                                 nr_of_blocks, nr_of_t, nstep, nat,
                                 nat_of_interest):
    """MSD(iblock, t), isotropic (summed over x/y/z, not averaged)."""
    _warn_if_subsampled(stepsize_tau)
    idx0 = np.asarray(indices_of_interest, dtype=int) - 1
    max_shift = stepsize_t * (nr_of_t - 1)
    msd = np.zeros((nr_of_blocks, nr_of_t))

    for iat in idx0:
        traj = positions[:, iat, :]  # (nstep, 3)
        s = np.einsum('sc,sc->s', traj, traj)  # |r(tau)|^2
        for ib in range(nr_of_blocks):
            start = ib * block_length_dt
            sw = _window_sums(s, start, block_length_dt, max_shift)
            c = np.zeros(max_shift + 1)
            for ipol in range(3):
                c += _block_correlate(
                    traj[:, ipol], traj[:, ipol], start,
                    block_length_dt, max_shift)
            term = sw[0] + sw - 2.0 * c  # index by raw shift
            msd[ib, :] += term[::stepsize_t]

    msd /= block_length_dt * nat_of_interest
    return msd


def calculate_msd_specific_atoms_max_stats(positions, indices_of_interest,
                                           stepsize_t, stepsize_tau,
                                           nr_of_t, nstep, nat,
                                           nat_of_interest):
    """
    Same identity as calculate_msd_specific_atoms, no blocks: every
    origin over the whole trajectory. *stepsize_tau* is accepted only
    for signature compatibility -- the Fortran body never used it
    either (there was never a slow path here to speed up).

    Every lag needs a different origin-window length here (there is
    no fixed block to restrict to), so this uses one 'full' -- not
    'valid' -- correlation per atom/component: it hands back every
    lag from 0 to n-1 in a single FFT call, of which only the first
    ``max_shift+1`` are read off.
    """
    idx0 = np.asarray(indices_of_interest, dtype=int) - 1
    max_shift = stepsize_t * (nr_of_t - 1)
    shifts = np.arange(0, max_shift + 1, stepsize_t)  # length nr_of_t
    msd_sum = np.zeros(nr_of_t)
    count = np.zeros(nr_of_t)

    for iat in idx0:
        traj = positions[:, iat, :]
        n = traj.shape[0]
        s = np.einsum('sc,sc->s', traj, traj)
        prefix = np.concatenate(([0.0], np.cumsum(s)))
        c = np.zeros(n)
        for ipol in range(3):
            full = correlate(traj[:, ipol], traj[:, ipol],
                             mode='full', method='fft')
            # full is length 2n-1; index n-1 is lag 0, and lags
            # 0..n-1 (positive-shift half) follow it in order.
            c += full[n - 1:]
        sw0 = prefix[n - shifts]
        sw_shift = prefix[n] - prefix[shifts]
        msd_sum += sw0 + sw_shift - 2.0 * c[shifts]
        count += (n - shifts)

    return msd_sum / count


def calculate_msd_specific_atoms_decompose_d(positions, indices_of_interest,
                                             stepsize_t, stepsize_tau,
                                             block_length_dt, nr_of_blocks,
                                             nr_of_t, nstep, nat,
                                             nat_of_interest):
    """MSD(iblock, t, i, j), the (3,3) tensor generalisation."""
    _warn_if_subsampled(stepsize_tau)
    idx0 = np.asarray(indices_of_interest, dtype=int) - 1
    max_shift = stepsize_t * (nr_of_t - 1)
    msd = np.zeros((nr_of_blocks, nr_of_t, 3, 3))

    for iat in idx0:
        traj = positions[:, iat, :]  # (nstep, 3)
        for ib in range(nr_of_blocks):
            start = ib * block_length_dt
            for ipol in range(3):
                for jpol in range(ipol, 3):
                    p = traj[:, ipol] * traj[:, jpol]
                    sw = _window_sums(p, start, block_length_dt, max_shift)
                    x_ij = _block_correlate(
                        traj[:, ipol], traj[:, jpol], start,
                        block_length_dt, max_shift)
                    if ipol == jpol:
                        x_ji = x_ij
                    else:
                        x_ji = _block_correlate(
                            traj[:, jpol], traj[:, ipol], start,
                            block_length_dt, max_shift)
                    term = (sw[0] + sw - x_ij - x_ji)[::stepsize_t]
                    msd[ib, :, ipol, jpol] += term
                    if ipol != jpol:
                        msd[ib, :, jpol, ipol] += term

    msd /= block_length_dt * nat_of_interest
    return msd


def calculate_vaf_specific_atoms(velocities, indices_of_interest,
                                 stepsize_t, stepsize_tau, nr_of_t,
                                 nr_of_blocks, block_length_dt, deltaT,
                                 integration_method, nstep, nat,
                                 nat_of_interest):
    """VAF(iblock, t) and its running integral."""
    _warn_if_subsampled(stepsize_tau)
    idx0 = np.asarray(indices_of_interest, dtype=int) - 1
    max_shift = stepsize_t * (nr_of_t - 1)
    vaf = np.zeros((nr_of_blocks, nr_of_t))

    for iat in idx0:
        traj = velocities[:, iat, :]
        for ib in range(nr_of_blocks):
            start = ib * block_length_dt
            c = np.zeros(max_shift + 1)
            for ipol in range(3):
                c += _block_correlate(
                    traj[:, ipol], traj[:, ipol], start,
                    block_length_dt, max_shift)
            vaf[ib, :] += c[::stepsize_t]

    vaf /= block_length_dt * nat_of_interest

    vaf_integral = np.zeros_like(vaf)
    if integration_method == 'trapezoid':
        vaf_integral[:, 0] = 0.5 * deltaT * vaf[:, 0]
        for t in range(1, nr_of_t):
            if t == nr_of_t - 1:
                vaf_integral[:, t] = (
                    0.5 * deltaT * vaf[:, t] + vaf_integral[:, t - 1])
            else:
                vaf_integral[:, t] = (
                    deltaT * vaf[:, t] + vaf_integral[:, t - 1])
    elif integration_method == 'trapezoid-simple':
        for t in range(1, nr_of_t):
            vaf_integral[:, t] = vaf_integral[:, t - 1] + (
                0.5 * deltaT * (vaf[:, t] + vaf[:, t - 1]))
    elif integration_method == 'simpson':
        vaf_integral[:, 0] = deltaT / 3.0 * vaf[:, 0]
        for t in range(1, nr_of_t):
            if t == nr_of_t - 1:
                vaf_integral[:, t] = (
                    deltaT / 3.0 * vaf[:, t] + vaf_integral[:, t - 1])
            elif t % 2 == 0:
                vaf_integral[:, t] = (
                    4.0 * deltaT / 3.0 * vaf[:, t] + vaf_integral[:, t - 1])
            else:
                vaf_integral[:, t] = (
                    2.0 * deltaT / 3.0 * vaf[:, t] + vaf_integral[:, t - 1])
    else:
        raise ValueError(
            'Unknown integration_method {!r}'.format(integration_method))

    return vaf, vaf_integral
