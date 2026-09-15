# -*- coding: utf-8 -*-

from collections import namedtuple
from warnings import warn

import numpy as np
from scipy.stats import linregress
from scipy.signal import convolve
from samos.trajectory import check_trajectory_compatibility, Trajectory
from samos.utils.attributed_array import AttributedArray
from samos.utils.constants import ANG2_FS_TO_CM2_S
from samos.utils.exceptions import InputError
from samos.utils.time_units import parse_time


# Return type of _get_running_params.  Named fields prevent the silent
# breakage that a positional tuple causes when fields are added or
# reordered.
RunningParams = namedtuple('RunningParams', [
    'species_of_interest',
    'nr_of_blocks',
    't_start_dt',
    't_end_dt',
    't_start_fit_dt',
    't_end_fit_dt',
    'nr_of_t',
    'stepsize_t',
    'stepsize_tau',
    'block_length_dt',
    'do_com',
    'do_long',
    't_long_end_dt',
    't_long_factor',
])


def _check_deprecated_time_kwargs(kwargs):
    """
    Raise InputError if *kwargs* contains old-style unit-suffix time
    parameters (e.g. ``t_end_fit_ps``, ``block_length_dt``).

    These were removed when the API was redesigned to accept a plain
    numeric value plus a single ``t_unit`` argument per call.  Any
    remaining unrecognised keyword is also rejected here so callers do
    not need a second guard.

    :param dict kwargs: The ``**kwargs`` dict from the calling method.
    :raises InputError: Always, when *kwargs* is non-empty.
    """
    _TIME_PARAMS = (
        't_start', 't_end', 't_start_fit', 't_end_fit',
        'block_length', 't_long_end',
    )
    _OLD_SUFFIXES = ('_fs', '_ps', '_dt')
    old = [
        k for k in kwargs
        if any(k == p + s for p in _TIME_PARAMS for s in _OLD_SUFFIXES)
    ]
    if old:
        raise InputError(
            "Unrecognized keyword argument(s): {}.\n"
            "The time-parameter API changed: unit suffixes (_fs, _ps, "
            "_dt) are no longer supported.\n"
            "Pass a plain numeric value and set t_unit instead, e.g.:\n"
            "  get_msd(t_end_fit=10, t_unit='ps')   "
            "# was: t_end_fit_ps=10\n"
            "  get_msd(t_end_fit=10000, t_unit='fs')  "
            "# was: t_end_fit_fs=10000\n"
            "  get_msd(t_end_fit=800, t_unit='dt')   "
            "# was: t_end_fit_dt=800\n"
            "See README.md for the full list of changes.".format(
                ', '.join(old))
        )
    if kwargs:
        raise InputError(
            "Unrecognized keyword argument(s): {}.".format(
                ', '.join(kwargs))
        )


class TimeSeries(AttributedArray):
    pass


def _get_masses(trajectory):
    """
    Return the atomic masses (amu) of *trajectory*.

    :raises InputError:
        If no :class:`ase.Atoms` has been attached, in which case masses
        are unknown -- ``set_types`` alone is not enough.
    """
    atoms = trajectory.atoms
    if atoms is None:
        raise InputError(
            'Mass-weighted analysis needs atomic masses, but this '
            'trajectory has no ase.Atoms attached. Call '
            'Trajectory.set_atoms() before running it.')
    return atoms.get_masses()


def _resolve_blocks(nstep, params):
    """
    Resolve the block layout for one trajectory of *nstep* steps.

    Blocks are cut from the first ``nstep - t_end_dt`` steps, so that
    every block start still has ``t_end_dt`` steps of trajectory ahead
    of it for the longest lag.  ``t_end_dt == 0`` marks an analysis
    with no lag window (the periodogram), whose blocks tile the whole
    trajectory.

    Exactly one of ``params.nr_of_blocks`` / ``params.block_length_dt``
    is set by :meth:`DynamicsAnalyzer._get_running_params`.

    :returns: ``(block_length_dt, nr_of_blocks)`` for this trajectory.
    :raises InputError:
        If the requested layout does not fit in the trajectory.  A zero
        block length or count used to pass the old ``< 0`` test and
        reach the Fortran/C++ kernels, which index without bounds
        checking.
    """
    usable = nstep - params.t_end_dt
    if params.nr_of_blocks is not None:
        nr_of_blocks = params.nr_of_blocks
        if nr_of_blocks < 1:
            raise InputError(
                'nr_of_blocks must be at least 1, got {}.'.format(
                    nr_of_blocks))
        block_length_dt = usable // nr_of_blocks
    elif params.block_length_dt is not None:
        block_length_dt = params.block_length_dt
        if block_length_dt < 1:
            raise InputError(
                'block_length must be at least one timestep, got {} '
                'dt.'.format(block_length_dt))
        nr_of_blocks = usable // block_length_dt
    else:  # pragma: no cover -- _get_running_params always sets one
        raise RuntimeError(
            'Neither nr_of_blocks nor block_length_dt was specified')

    if block_length_dt < 1 or nr_of_blocks < 1:
        if params.t_end_dt:
            reason = (
                'only {} step(s) are available as block starts '
                '(nstep - t_end_dt, with t_end_dt={}). Use fewer '
                'blocks, a shorter block_length, or a smaller '
                't_end/t_end_fit.'.format(usable, params.t_end_dt))
        else:
            # Nothing is held back here, so advising a smaller t_end
            # would send the caller after a knob that does nothing.
            reason = 'use fewer blocks or a shorter block_length.'
        raise InputError(
            'Cannot fit {} block(s) of {} step(s) into a trajectory of '
            '{} steps: {}'.format(
                nr_of_blocks, block_length_dt, nstep, reason))
    return block_length_dt, nr_of_blocks


def _species_factors(trajectory, atomic_species):
    """
    Return the 0/1 selector array that restricts a centre-of-mass
    calculation to *atomic_species*.

    The Fortran/C++ ``get_com_positions`` and ``get_com_velocities``
    kernels weight atom *i* by ``factors[i] * masses[i]``, so passing
    all ones would yield the centre of mass of the entire system -- the
    same near-constant quantity for every species, and identically zero
    for a recentered trajectory.

    :returns: ``(factors, nat_of_species)``
    """
    indices = trajectory.get_indices_of_species(atomic_species, start=0)
    factors = np.zeros(len(trajectory.types), dtype=int)
    factors[indices] = 1
    return factors, len(indices)


def _com_array(trajectory, atomic_species, array, kernel):
    """
    Collapse *array* onto the centre of mass of *atomic_species*.

    This is what ``do_com`` does for collective (charge) diffusion: the
    quantity of interest is the motion of a species as a whole, not of
    its individual atoms.  *kernel* is ``get_com_positions`` or
    ``get_com_velocities``; they take the same arguments, so the choice
    of kernel is the only thing that differs between MSD and VAF.

    :returns:
        ``(com_array, indices_of_interest, prefactor)``.  The result
        holds a single pseudo-atom, hence the fixed index list, and
        *prefactor* rescales by the number of atoms that went into it.
    """
    masses = _get_masses(trajectory)
    factors, prefactor = _species_factors(trajectory, atomic_species)
    return kernel(array, masses, factors), [1], prefactor


def _fit_block(block, fit, t_list_fit_fs, decomposed):
    """
    Least-squares fit of one MSD block over one fit window.

    Four copies of this used to sit inline, nested eight levels deep
    inside :meth:`DynamicsAnalyzer.get_msd`, where there was no room
    left on the line to write them legibly.

    :param block: One block of the MSD, indexed by lag time; shape
        ``(nr_of_t,)``, or ``(nr_of_t, 3, 3)`` when *decomposed*.
    :param slice fit: Lag-time range to fit over.
    :param t_list_fit_fs: Times in fs, one per entry that *fit* selects.
    :param bool decomposed: Whether *block* holds the (3, 3) tensor.
    :returns: Slope and intercept: shape ``(2,)``, or ``(3, 3, 2)``.
    """
    if not decomposed:
        slope, intercept, _, _, _ = linregress(t_list_fit_fs, block[fit])
        return np.array([slope, intercept])
    out = np.empty((3, 3, 2))
    for ipol in range(3):
        for jpol in range(3):
            slope, intercept, _, _, _ = linregress(
                t_list_fit_fs, block[fit, ipol, jpol])
            out[ipol, jpol] = (slope, intercept)
    return out


class DynamicsAnalyzer:
    """
    This class serves as the main analyzer
    for dynamic properties (MSD, VAF, etc)
    """

    def __init__(self, *, trajectories=None, species_of_interest=None,
                 verbosity=None, **kwargs):
        """
        :param list trajectories: Trajectories to analyse; a single
            :class:`~samos.trajectory.Trajectory` is also accepted.
        :param species_of_interest: Chemical symbol, or list of them,
            to restrict every analysis to.
        :param int verbosity: 0 silences the progress printing.
        :raises TypeError: On an unrecognised keyword argument.
        """
        self._species_of_interest = None
        self._verbosity = 1
        if kwargs:
            raise TypeError(
                '{} got unexpected keyword argument(s): {}'.format(
                    type(self).__name__, ', '.join(sorted(kwargs))))
        if trajectories is not None:
            self.set_trajectories(trajectories)
        if species_of_interest is not None:
            self.set_species_of_interest(species_of_interest)
        if verbosity is not None:
            self.set_verbosity(verbosity)

    def set_trajectories(self, trajectories):
        """
        Expects a list of trajectories
        """
        if isinstance(trajectories, Trajectory):
            trajectories = [trajectories]
        # I check the compatibility. Implicitly,
        # also checks if trajectories are valid instances.
        types, self._timestep_fs = check_trajectory_compatibility(
            trajectories)
        self._types = np.array(types)
        # Setting as attribute of self for analysis
        self._trajectories = trajectories

    def set_species_of_interest(self, species_of_interest):
        """
        :param list species_of_interest:
            To set a global list
            of species of interest for all the analysis
        :todo: Check the species whether they are valid
        """
        if isinstance(species_of_interest, str):
            self._species_of_interest = [species_of_interest]
        elif isinstance(species_of_interest, (tuple, set, list)):
            self._species_of_interest = list(species_of_interest)
        else:
            raise TypeError(
                'Species of interest has to be a list of'
                ' strings with the atomic symbol')

    def set_verbosity(self, verbosity):
        if not isinstance(verbosity, int):
            raise TypeError('Verbosity is an integer')
        self._verbosity = verbosity

    def get_species_of_interest(self):
        # Also a good way to check if atoms have been set
        types = self._types
        if self._species_of_interest is None:
            return sorted(set(types))
        else:
            return self._species_of_interest

    def _require_trajectories(self):
        """
        Return ``(timestep_fs, trajectories)`` set by
        :meth:`set_trajectories`.

        Deliberately limited to those two attributes: this used to be a
        ``try/except AttributeError`` wrapped around much more code, so
        an unrelated AttributeError raised inside an analysis was
        reported to the user as "please call set_trajectories".

        :raises InputError: If :meth:`set_trajectories` was never called.
        """
        if not hasattr(self, '_trajectories'):
            raise InputError(
                'No trajectories have been set. Use the set_trajectories '
                'method, or pass trajectories=... to the constructor, '
                'before running an analysis.')
        return self._timestep_fs, self._trajectories

    def _get_running_params(self, timestep_fs, t_unit='ps',
                            species_of_interest=None,
                            stepsize_t=1, stepsize_tau=1,
                            t_start=0, t_end=None,
                            t_start_fit=0, t_end_fit=None,
                            block_length=None, nr_of_blocks=None,
                            do_com=False, do_long=False,
                            t_long_end=None, t_long_factor=None,
                            require_fitting=True):
        """
        Parse and validate analysis parameters, converting all time
        values to integer timesteps.

        All time arguments (*t_start*, *t_end*, *t_start_fit*,
        *t_end_fit*, *t_long_end*, *block_length*) are plain numbers
        interpreted in the unit given by *t_unit*.

        :param float timestep_fs: Trajectory timestep in femtoseconds.
        :param str t_unit:
            Time unit for all time arguments.  One of 'fs', 'ps', 'dt'
            (default 'ps').  'dt' means the value is already an integer
            number of trajectory timesteps.  New units can be added by
            extending samos.utils.time_units._FS_PER_UNIT.
        :param list species_of_interest:
            Species to calculate.  Defaults to all species present.
        :param int stepsize_t: Outer-loop stride (default 1).
        :param int stepsize_tau: Inner-loop stride (default 1).
        :param float t_start:
            Start of the sliding window (default 0).
        :param float t_end:
            End of the sliding window.  Defaults to *t_end_fit* when
            ``None``.
        :param float t_start_fit:
            Start of the fitting window (default 0).  May be a
            list/array to compute fits over multiple windows in one
            call; must match the length of *t_end_fit* in that case.
        :param float t_end_fit:
            End of the fitting window (required).  May be a list/array;
            see *t_start_fit*.
        :param float block_length:
            Length of each trajectory block expressed in *t_unit*.
            Mutually exclusive with *nr_of_blocks*.
        :param int nr_of_blocks:
            Number of trajectory blocks (default 1).
            Mutually exclusive with *block_length*.
        :param bool do_com:
            Calculate centre-of-mass diffusion (default False).
        :param bool do_long:
            Include a maximum-statistics pass over the whole trajectory
            (default False).
        :param float t_long_end:
            End of the long-time window, in *t_unit*.
            Mutually exclusive with *t_long_factor*.
        :param float t_long_factor:
            Fraction of the trajectory length to use as the long-time
            window (dimensionless).  Mutually exclusive with *t_long_end*.
        :param bool require_fitting:
            Whether this analysis fits a time series (default True).
            :meth:`get_msd` and :meth:`get_vaf` pass True and therefore
            require *t_end_fit*.  Analyses without a lag window -- the
            periodogram of :meth:`get_power_spectrum` -- pass False:
            the fit window is then rejected rather than required,
            ``t_start_fit_dt``/``t_end_fit_dt`` come back as ``None``,
            and ``t_end_dt`` is 0 so that blocks tile the whole
            trajectory.
        :returns: :class:`RunningParams` namedtuple.
        :raises InputError: For invalid parameter combinations.
        """
        if species_of_interest is None:
            species_of_interest = self.get_species_of_interest()

        stepsize_t = int(stepsize_t)
        stepsize_tau = int(stepsize_tau)

        if require_fitting:
            # --- t_start_fit (scalar or list; default 0) ---
            t_start_fit_dt = parse_time(t_start_fit, t_unit, timestep_fs)
            if not np.all(np.asarray(t_start_fit_dt) >= 0):
                raise InputError('t_start_fit must be >= 0')

            # --- t_end_fit (required; scalar or list) ---
            if t_end_fit is None:
                raise InputError(
                    'Provide a time to end fitting the time series '
                    '(t_end_fit=<value>, t_unit={!r}).'.format(t_unit))
            t_end_fit_dt = parse_time(t_end_fit, t_unit, timestep_fs)

            if not np.all(
                    np.asarray(t_end_fit_dt) > np.asarray(t_start_fit_dt)):
                raise InputError(
                    't_end_fit must be greater than t_start_fit')

            # Both must be the same kind (scalar or array) and, if
            # arrays, the same length.
            fit_start_scalar = isinstance(t_start_fit_dt, int)
            fit_end_scalar = isinstance(t_end_fit_dt, int)
            if fit_start_scalar != fit_end_scalar:
                raise InputError(
                    't_start_fit and t_end_fit must both be scalars or '
                    'both be lists of the same length')
            if not fit_start_scalar:
                if len(t_start_fit_dt) != len(t_end_fit_dt):
                    raise InputError(
                        't_start_fit and t_end_fit lists must have the '
                        'same length')
        else:
            # Rejected rather than ignored: silently dropping a fit
            # window the caller asked for is worse than refusing it.
            if t_end_fit is not None:
                raise InputError(
                    'This analysis does not fit a time series, so '
                    't_end_fit has no meaning here.')
            t_start_fit_dt = None
            t_end_fit_dt = None

        # --- t_start (default 0) ---
        t_start_dt = parse_time(t_start, t_unit, timestep_fs)
        if t_start_dt < 0:
            raise InputError('t_start must be >= 0')
        if t_start_dt > 0:
            raise NotImplementedError('t_start > 0 is not implemented')

        # --- t_end (default: max of t_end_fit) ---
        if t_end is not None:
            t_end_dt = parse_time(t_end, t_unit, timestep_fs)
        elif require_fitting:
            t_end_dt = int(np.max(t_end_fit_dt))
        else:
            # _resolve_blocks holds back t_end_dt steps at the end of
            # the trajectory for the longest lag; an analysis without a
            # lag window needs none held back.
            t_end_dt = 0
        if require_fitting:
            if t_end_dt <= t_start_dt:
                raise InputError('t_end must be greater than t_start')
            if t_end_dt < int(np.max(t_end_fit_dt)):
                raise InputError('t_end must be >= t_end_fit')

        # 0 when there is no lag window, in which case nothing reads it.
        nr_of_t = (t_end_dt - t_start_dt) // stepsize_t

        # --- block_length / nr_of_blocks (mutually exclusive) ---
        if block_length is not None and nr_of_blocks is not None:
            raise InputError(
                'block_length and nr_of_blocks are mutually exclusive')
        if block_length is not None:
            block_length_dt = parse_time(block_length, t_unit, timestep_fs)
            nr_of_blocks = None
        elif nr_of_blocks is not None:
            block_length_dt = None
            nr_of_blocks = int(nr_of_blocks)
        else:
            nr_of_blocks = 1
            block_length_dt = None

        # --- t_long_end / t_long_factor (mutually exclusive) ---
        if t_long_end is not None and t_long_factor is not None:
            raise InputError(
                't_long_end and t_long_factor are mutually exclusive')
        if t_long_end is not None:
            t_long_end_dt = parse_time(t_long_end, t_unit, timestep_fs)
            t_long_factor = None
        elif t_long_factor is not None:
            t_long_end_dt = None
            t_long_factor = float(t_long_factor)
        else:
            # Both left as None; adapted to trajectory length at runtime.
            t_long_end_dt = None
            t_long_factor = None

        return RunningParams(
            species_of_interest=species_of_interest,
            nr_of_blocks=nr_of_blocks,
            t_start_dt=t_start_dt,
            t_end_dt=t_end_dt,
            t_start_fit_dt=t_start_fit_dt,
            t_end_fit_dt=t_end_fit_dt,
            nr_of_t=nr_of_t,
            stepsize_t=stepsize_t,
            stepsize_tau=stepsize_tau,
            block_length_dt=block_length_dt,
            do_com=do_com,
            do_long=do_long,
            t_long_end_dt=t_long_end_dt,
            t_long_factor=t_long_factor,
        )

    def get_msd(self, decomposed=False, atom_indices=None,
                t_unit='ps',
                species_of_interest=None,
                stepsize_t=1, stepsize_tau=1,
                t_start=0, t_end=None,
                t_start_fit=0, t_end_fit=None,
                block_length=None, nr_of_blocks=None,
                do_com=False, do_long=False,
                t_long_end=None, t_long_factor=None,
                **kwargs):
        """
        Calculate the mean-square displacement (MSD).

        #.  Calculate the MSD for each block.
        #.  Calculate the mean and standard deviation of the slope.
        #.  Calculate the diffusion coefficient, including error propagation.

        :param bool decomposed:
            Compute the (3,3) MSD tensor by calculating each Cartesian
            component independently (default False).
        :param list atom_indices:
            Subset of atom indices to include.  The intersection with
            *species_of_interest* is used, so this narrows the selection.
            Ignored when *do_com* is set, which leaves no individual
            atoms to select from.
        :param str t_unit:
            Time unit for all time arguments
            (``'fs'``, ``'ps'``, or ``'dt'``; default ``'ps'``).
        :param list species_of_interest:
            Species to analyse, e.g. ``['O', 'H']``.
            Defaults to all species in the trajectory.
        :param int stepsize_t: Outer-loop stride (default 1).
        :param int stepsize_tau: Inner-loop stride (default 1).
        :param float t_start:
            Start of the sliding window (default 0, in *t_unit*).
        :param float t_end:
            End of the sliding window (in *t_unit*).
            Defaults to *t_end_fit* when ``None``.
        :param float t_start_fit:
            Start of the linear-fit window (default 0, in *t_unit*).
            May be a list/array; slopes are then computed for each
            (t_start_fit, t_end_fit) pair.
        :param float t_end_fit:
            End of the linear-fit window (required, in *t_unit*).
            May be a list/array; see *t_start_fit*.
        :param float block_length:
            Length of each trajectory block (in *t_unit*).
            Mutually exclusive with *nr_of_blocks*.
        :param int nr_of_blocks:
            Number of trajectory blocks (default 1).
            Mutually exclusive with *block_length*.
        :param bool do_com:
            Calculate centre-of-mass diffusion instead of tracer
            diffusion (default False).
        :param bool do_long:
            Also compute a maximum-statistics MSD using the whole
            trajectory with no blocks (default False).
        :param float t_long_end:
            End of the long-time window (in *t_unit*).
            Mutually exclusive with *t_long_factor*.
        :param float t_long_factor:
            Fraction of the trajectory length to use as the long-time
            window (dimensionless). Mutually exclusive with *t_long_end*.
        :raises InputError:
            If old-style unit-suffix kwargs are passed (e.g.
            ``t_end_fit_ps``). These were removed in the API redesign;
            pass a plain value and ``t_unit`` instead.
        """
        _check_deprecated_time_kwargs(kwargs)

        from samos.analysis._fft_dynamics import (
            calculate_msd_specific_atoms,
            calculate_msd_specific_atoms_decompose_d,
            calculate_msd_specific_atoms_max_stats,
            get_com_positions)
        timestep_fs, trajectories = self._require_trajectories()

        p = self._get_running_params(
            timestep_fs, t_unit=t_unit,
            species_of_interest=species_of_interest,
            stepsize_t=stepsize_t, stepsize_tau=stepsize_tau,
            t_start=t_start, t_end=t_end,
            t_start_fit=t_start_fit, t_end_fit=t_end_fit,
            block_length=block_length, nr_of_blocks=nr_of_blocks,
            do_com=do_com, do_long=do_long,
            t_long_end=t_long_end, t_long_factor=t_long_factor,
            require_fitting=True,
        )
        # A scalar fit window is a list of one window from here on, so
        # that the fitting below has a single code path.  The extra
        # leading axis is taken off again where the results are stored,
        # to keep the shape a scalar window has always produced.
        multiple_params_fit = not isinstance(p.t_start_fit_dt, int)
        if multiple_params_fit:
            fit_starts = np.asarray(p.t_start_fit_dt)
            fit_ends = np.asarray(p.t_end_fit_dt)
        else:
            fit_starts = np.array([p.t_start_fit_dt])
            fit_ends = np.array([p.t_end_fit_dt])

        msd = TimeSeries()
        # list of t at which MSD will be computed
        t_list_fs = timestep_fs * p.stepsize_t * \
            (p.t_start_dt + np.arange(p.nr_of_t))
        msd.set_array('t_list_fs', t_list_fs)

        results_dict = {atomic_species: {}
                        for atomic_species in p.species_of_interest}
        nr_of_t_long_list = []
        t_list_long_fs = []
        # Setting params for calculation of MSD and conductivity
        # Future: Maybe allow for element specific parameter settings?

        for atomic_species in p.species_of_interest:
            msd_this_species = []  # Here I collect the trajectories
            slopes = []  # That's where I collect slopes
            # for the final estimate of diffusion

            for itraj, trajectory in enumerate(trajectories):
                positions = trajectory.get_positions()
                if p.do_com:
                    positions, indices_of_interest, prefactor = _com_array(
                        trajectory, atomic_species, positions,
                        get_com_positions)
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1
                    # Only meaningful per atom: do_com has already
                    # replaced the atoms by a single pseudo-atom, so
                    # there is nothing left for atom_indices to select.
                    if atom_indices is not None:
                        indices_of_interest = [
                            i for i in indices_of_interest
                            if i in atom_indices]

                # make blocks
                nstep, nat, _ = positions.shape
                block_length_dt_this_traj, nr_of_blocks_this_traj = (
                    _resolve_blocks(nstep, p))

                nat_of_interest = len(indices_of_interest)

                #
                # compute MSD (using defined blocks, nstep, nr_of_t, ...)
                if self._verbosity > 0:
                    print((
                        '\n    ! Calculating MSD for atomic species {} '
                        'in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      I will calculate {} block(s) of size {} '
                        '({} ps)\n'
                        '      I will fit from {} ({} ps) to {} ({} ps)\n'
                        '      Outer stepsize is {}, inner is {}\n'
                        ''.format(
                            atomic_species, itraj, nat_of_interest,
                            atomic_species, nr_of_blocks_this_traj,
                            block_length_dt_this_traj,
                            block_length_dt_this_traj * timestep_fs / 1e3,
                            p.t_start_fit_dt,
                            p.t_start_fit_dt * timestep_fs / 1e3,
                            p.t_end_fit_dt,
                            p.t_end_fit_dt * timestep_fs / 1e3,
                            p.stepsize_t, p.stepsize_tau)
                    ))
                if decomposed:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms_decompose_d(
                            positions, indices_of_interest, p.stepsize_t,
                            p.stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj,
                            p.nr_of_t, nstep, nat, nat_of_interest)
                    )
                else:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms(
                            positions, indices_of_interest, p.stepsize_t,
                            p.stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj, p.nr_of_t, nstep,
                            nat, nat_of_interest)
                    )
                if self._verbosity > 0:
                    print('      Done\n')

                for iblock, block in enumerate(msd_this_species_this_traj):
                    msd_this_species.append(block)
                msd.set_array('msd_{}_{}_{}'.format(
                    'decomposed' if decomposed else 'isotropic',
                    atomic_species, itraj), msd_this_species_this_traj)

                #
                # linear regression of MSD
                per_window_shape = (3, 3, 2) if decomposed else (2,)
                slopes_intercepts = np.empty(
                    (len(fit_starts), nr_of_blocks_this_traj)
                    + per_window_shape)
                for istart, (start_dt, end_dt) in enumerate(
                        zip(fit_starts, fit_ends)):
                    # Neither varies per block, so both are hoisted out
                    # of the loop below.
                    fit = slice(
                        (start_dt - p.t_start_dt) // p.stepsize_t,
                        end_dt // p.stepsize_t)
                    t_list_fit_fs = (
                        timestep_fs * p.stepsize_t
                        * np.arange(start_dt // p.stepsize_t,
                                    end_dt // p.stepsize_t))
                    for iblock, block in enumerate(
                            msd_this_species_this_traj):
                        slopes_intercepts[istart, iblock] = _fit_block(
                            block, fit, t_list_fit_fs, decomposed)

                # Ellipsis spans the (3, 3) tensor axes when decomposed
                # and nothing otherwise, so one expression covers both.
                for iblock in range(nr_of_blocks_this_traj):
                    block_slopes = slopes_intercepts[:, iblock, ..., 0]
                    slopes.append(block_slopes if multiple_params_fit
                                  else block_slopes[0])

                msd.set_array('slopes_intercepts_{}_{}_{}'.format(
                    'decomposed' if decomposed else 'isotropic',
                    atomic_species, itraj),
                    slopes_intercepts if multiple_params_fit
                    else slopes_intercepts[0])

                #
                # compute MSD with maximal statistics (whole trajectory,
                # no blocks)
                if p.do_long:
                    # nr_of_t_long may change with the length of the traj
                    # (if not set by user)
                    if p.t_long_end_dt is not None:
                        nr_of_t_long = p.t_long_end_dt // p.stepsize_t
                    elif p.t_long_factor is not None:
                        nr_of_t_long = int(
                            p.t_long_factor * nstep / p.stepsize_t)
                    else:
                        nr_of_t_long = (nstep - 1) // p.stepsize_t
                    if nr_of_t_long > nstep:
                        raise RuntimeError(
                            't_long_end_dt is bigger than the '
                            'trajectory length')
                    nr_of_t_long_list.append(nr_of_t_long)
                    t_list_long_fs.append(
                        timestep_fs * p.stepsize_t
                        * np.arange(nr_of_t_long))
                    msd_this_species_this_traj_max_stats = (
                        prefactor * calculate_msd_specific_atoms_max_stats(
                            positions, indices_of_interest, p.stepsize_t,
                            p.stepsize_tau, nr_of_t_long, nstep, nat,
                            nat_of_interest))
                    msd.set_array('msd_long_{}_{}'.format(atomic_species,
                                                          itraj),
                                  msd_this_species_this_traj_max_stats)
            #
            # end of trajectories loop

            # Calculating the mean, std, sem for each point in time
            # (averaging over trajectories)
            msd_mean = np.mean(msd_this_species, axis=0)
            if (len(msd_this_species) > 1):
                msd_std = np.std(msd_this_species, axis=0)
                msd_sem = msd_std / np.sqrt(len(msd_this_species) - 1)
            else:
                msd_std = np.full(msd_mean.shape, np.nan)
                msd_sem = np.full(msd_mean.shape, np.nan)
            msd.set_array('msd_{}_{}_mean'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_mean)
            msd.set_array('msd_{}_{}_std'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_std)
            msd.set_array('msd_{}_{}_sem'.format(
                'decomposed' if decomposed else 'isotropic',
                atomic_species),
                msd_sem)

            slopes = np.array(slopes)  # 0th axis
            results_dict[atomic_species]['slope_msd_mean'] = np.mean(
                slopes, axis=0)
            if (len(msd_this_species) > 1):
                results_dict[atomic_species]['slope_msd_std'] = np.std(
                    slopes, axis=0)
                results_dict[atomic_species]['slope_msd_sem'] = (
                    results_dict[atomic_species]['slope_msd_std']
                    / np.sqrt(len(slopes)-1))
            else:
                results_dict[atomic_species]['slope_msd_std'] = np.full(
                    results_dict[atomic_species]['slope_msd_mean'].shape,
                    np.nan)
                results_dict[atomic_species]['slope_msd_sem'] = np.full(
                    results_dict[atomic_species]['slope_msd_mean'].shape,
                    np.nan)

            if decomposed:
                dimensionality_factor = 2.
            else:
                dimensionality_factor = 6.
            # D = slope / dimensionality_factor, with the slope in
            # A^2/fs and D reported in cm^2/s.
            results_dict[atomic_species]['diffusion_mean_cm2_s'] = (
                ANG2_FS_TO_CM2_S / dimensionality_factor
                * results_dict[atomic_species]['slope_msd_mean'])
            if (len(msd_this_species) > 1):
                results_dict[atomic_species]['diffusion_std_cm2_s'] = (
                    ANG2_FS_TO_CM2_S / dimensionality_factor
                    * results_dict[atomic_species]['slope_msd_std'])
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = (
                    ANG2_FS_TO_CM2_S / dimensionality_factor
                    * results_dict[atomic_species]['slope_msd_sem'])
            else:
                results_dict[atomic_species]['diffusion_std_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.nan)
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.nan)

            # I need to transform to lists, numpy are not json serializable:
            for k in ('slope_msd_mean', 'slope_msd_std', 'slope_msd_sem',
                      'diffusion_mean_cm2_s', 'diffusion_std_cm2_s',
                      'diffusion_sem_cm2_s'):
                if isinstance(results_dict[atomic_species][k],
                              np.ndarray):
                    results_dict[atomic_species][k] = results_dict[
                        atomic_species][k].tolist()
            if self._verbosity > 1:
                print('      Done, these are the results for {}:'.format(
                    atomic_species))
                for key, val in results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print('          {:<20} {}'.format(key,  val))
        # end of species_of_interest loop
        #

        results_dict.update({
            't_start_fit_dt': (p.t_start_fit_dt.tolist()
                               if multiple_params_fit
                               else p.t_start_fit_dt),
            't_end_fit_dt':   (p.t_end_fit_dt.tolist()
                               if multiple_params_fit
                               else p.t_end_fit_dt),
            't_start_dt':             p.t_start_dt,
            't_end_dt':               p.t_end_dt,
            'nr_of_trajectories':     len(trajectories),
            'stepsize_t':             p.stepsize_t,
            'species_of_interest':    p.species_of_interest,
            'timestep_fs':            timestep_fs,
            'nr_of_t':                p.nr_of_t,
            'decomposed':             decomposed,
            'do_long':                p.do_long,
            'multiple_params_fit':    multiple_params_fit,
        })
        if p.do_long:
            results_dict['nr_of_t_long_list'] = nr_of_t_long_list
            msd.set_array('t_list_long_fs', t_list_long_fs)
        for k, v in results_dict.items():
            msd.set_attr(k, v)
        return msd

    def get_vaf(self, arrayname=None, integration='trapezoid',
                remove_angular_momentum=False, t_unit='ps',
                species_of_interest=None,
                stepsize_t=1, stepsize_tau=1,
                t_start=0, t_end=None,
                t_start_fit=0, t_end_fit=None,
                block_length=None, nr_of_blocks=None,
                do_com=False, **kwargs):

        _check_deprecated_time_kwargs(kwargs)

        from samos.analysis._fft_dynamics import (
            calculate_vaf_specific_atoms,
            get_com_velocities)
        timestep_fs, trajectories = self._require_trajectories()

        p = self._get_running_params(
            timestep_fs, t_unit=t_unit,
            species_of_interest=species_of_interest,
            stepsize_t=stepsize_t, stepsize_tau=stepsize_tau,
            t_start=t_start, t_end=t_end,
            t_start_fit=t_start_fit, t_end_fit=t_end_fit,
            block_length=block_length, nr_of_blocks=nr_of_blocks,
            do_com=do_com,
            require_fitting=True,
        )
        if p.do_long:
            raise NotImplementedError('Do_long is not implemented for VAF')

        vaf_time_series = TimeSeries()

        results_dict = dict()

        for atomic_species in p.species_of_interest:

            vaf_this_species = []
            vaf_integral_this_species = []
            fitted_means_of_integral = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    velocities = trajectory.get_array(arrayname)
                else:
                    velocities = trajectory.get_velocities(
                        remove_angular_momentum=remove_angular_momentum)
                if p.do_com:
                    velocities, indices_of_interest, prefactor = _com_array(
                        trajectory, atomic_species, velocities,
                        get_com_velocities)
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1

                nstep, nat, _ = velocities.shape
                block_length_dt_this_traj, nr_of_blocks_this_traj = (
                    _resolve_blocks(nstep, p))

                nat_of_interest = len(indices_of_interest)

                if self._verbosity > 0:
                    print((
                        '\n    ! Calculating VAF for atomic species {} '
                        'in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      I will calculate {} block(s)'
                        ''.format(atomic_species, itraj,
                                  nat_of_interest, atomic_species,
                                  p.nr_of_blocks)
                    ))

                vaf, vaf_integral = calculate_vaf_specific_atoms(
                    velocities, indices_of_interest,
                    p.stepsize_t, p.stepsize_tau,
                    p.nr_of_t, nr_of_blocks_this_traj,
                    block_length_dt_this_traj,
                    timestep_fs * p.stepsize_t,
                    integration, nstep, nat, nat_of_interest)
                # The integral is in A^2/fs; divide by the three
                # Cartesian dimensions to get D in cm^2/s.
                vaf_integral *= ANG2_FS_TO_CM2_S / 3. * prefactor

                for iblock in range(nr_of_blocks_this_traj):
                    vaf_this_species.append(vaf[iblock])
                    vaf_integral_this_species.append(vaf_integral[iblock])
                    data_ = vaf_integral[
                        iblock,
                        p.t_start_fit_dt // p.stepsize_t:
                        p.t_end_fit_dt // p.stepsize_t]
                    fitted_means_of_integral.append(data_.mean())

                vaf_time_series.set_array(
                    'vaf_isotropic_{}_{}'.format(atomic_species, itraj),
                    vaf)
                vaf_time_series.set_array(
                    'vaf_integral_isotropic_{}_{}'.format(
                        atomic_species, itraj),
                    vaf_integral)

            for arr, name in ((vaf_this_species, 'vaf_isotropic'),
                              (vaf_integral_this_species,
                               'vaf_integral_isotropic')):
                arr = np.array(arr)

                arr_mean = np.mean(arr, axis=0)
                if arr.shape[0] > 1:
                    arr_std = np.std(arr, axis=0)
                    arr_sem = arr_std / np.sqrt(arr.shape[0] - 1)
                else:
                    # A single block carries no spread; reporting 0/0
                    # here produced a RuntimeWarning and a silent NaN.
                    # get_msd fills NaN in the same situation.
                    arr_std = np.full(arr_mean.shape, np.nan)
                    arr_sem = np.full(arr_mean.shape, np.nan)
                vaf_time_series.set_array(
                    '{}_{}_mean'.format(name, atomic_species),
                    arr_mean)
                vaf_time_series.set_array(
                    '{}_{}_std'.format(name, atomic_species),
                    arr_std)
                vaf_time_series.set_array(
                    '{}_{}_sem'.format(name, atomic_species),
                    arr_sem)

            fitted_means_of_integral = np.array(fitted_means_of_integral)
            results_dict[atomic_species] = dict(
                diffusion_mean_cm2_s=fitted_means_of_integral.mean())
            if len(fitted_means_of_integral) > 1:
                results_dict[atomic_species]['diffusion_std_cm2_s'] = (
                    fitted_means_of_integral.std())
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = (
                    results_dict[atomic_species]['diffusion_std_cm2_s']
                    / np.sqrt(len(fitted_means_of_integral) - 1))
            else:
                results_dict[atomic_species]['diffusion_std_cm2_s'] = np.nan
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = np.nan

            if self._verbosity > 1:
                print(
                    ('      Done, these are the results for {}:'.format(
                        atomic_species)))
                for key, val in results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print(('          {:<20} {}'.format(key,  val)))

        results_dict.update({
            't_start_fit_dt':     p.t_start_fit_dt,
            't_end_fit_dt':       p.t_end_fit_dt,
            't_start_dt':         p.t_start_dt,
            't_end_dt':           p.t_end_dt,
            'nr_of_trajectories': len(trajectories),
            'stepsize_t':         p.stepsize_t,
            'species_of_interest': p.species_of_interest,
            'timestep_fs':        timestep_fs,
            'nr_of_t':            p.nr_of_t,
        })

        for k, v in results_dict.items():
            vaf_time_series.set_attr(k, v)
        return vaf_time_series

    def get_kinetic_energies(self, stepsize=1, decompose_system=True,
                             decompose_atoms=False,
                             decompose_species=False):
        from samos.utils.constants import amu_kg, kB

        timestep_fs, trajectories = self._require_trajectories()

        prefactor = amu_kg * 1e10 / kB
        # * 1.06657254018667

        if decompose_atoms and decompose_species:
            raise Exception('Cannot decompose atoms and decompose species')

        kinetic_energies_series = TimeSeries()
        kinetic_energies_series.set_attr('stepsize', stepsize)
        kinetic_energies_series.set_attr('timestep_fs', timestep_fs)

        for itraj, t in enumerate(trajectories):
            masses = _get_masses(t)
            vel_array = t.get_velocities()
            nstep, nat, _ = vel_array.shape
            steps = list(range(0, nstep, stepsize))
            # (nsteps, nat, 3), the same steps the loops used to visit
            vel = vel_array[::stepsize]

            if decompose_system:
                # sum_i m_i * |v_i|^2 for every step, vectorized over
                # atoms and polarizations
                kinE = prefactor * np.einsum(
                    'i,sic,sic->s', masses, vel, vel)
                kinE[:] /= nat*3  # I divide by the degrees of freedom!
                kinetic_energies_series.set_array(
                    'system_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'mean_system_kinetic_energy_{}'.format(itraj), kinE.mean())
            if decompose_species:
                species_of_interest = self.get_species_of_interest()
                ntyp = len(species_of_interest)
                steps = list(range(0, nstep, stepsize))
                kinE_species = np.zeros((len(steps), ntyp))
                for ityp, atomic_species in enumerate(species_of_interest):
                    indices_of_interest = t.get_indices_of_species(
                        atomic_species, start=0)
                    vel_this = vel[:, indices_of_interest]
                    kinE_species[:, ityp] = prefactor * np.einsum(
                        'i,sic,sic->s', masses[indices_of_interest],
                        vel_this, vel_this)
                    kinE_species[:, ityp] /= float(len(indices_of_interest)*3)

                kinetic_energies_series.set_array(
                    'species_kinetic_energy_{}'.format(itraj), kinE_species)
                kinetic_energies_series.set_attr(
                    'species_of_interest', species_of_interest)
                kinetic_energies_series.set_attr(
                    'mean_species_kinetic_energy_{}'.format(itraj),
                    kinE_species.mean(axis=0).tolist())

            if decompose_atoms:
                # per atom, so only the polarizations are summed
                kinE = (prefactor / 3.) * masses[None, :] * np.einsum(
                    'sic,sic->si', vel, vel)

                kinetic_energies_series.set_array(
                    'atoms_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'mean_atoms_kinetic_energy_{}'.format(itraj),
                    kinE.mean(axis=0).tolist())

        return kinetic_energies_series

    def get_power_spectrum(self, arrayname=None,
                           remove_angular_momentum=False,
                           t_unit='ps',
                           species_of_interest=None,
                           smoothing=1,
                           block_length=None,
                           nr_of_blocks=None,
                           **kwargs):
        """
        Calculate the power spectrum.

        :param str arrayname:
            Name of a custom velocity array stored on the trajectory.
            If ``None``, velocities are read via
            :meth:`~samos.trajectory.Trajectory.get_velocities`.
        :param bool remove_angular_momentum:
            Remove the rigid-body rotational contribution from
            velocities before computing the spectrum (default False).
        :param str t_unit:
            Time unit for *block_length* (``'fs'``, ``'ps'``, or
            ``'dt'``; default ``'ps'``).
        :param list species_of_interest:
            Species to analyse.  Defaults to all species present.
        :param int smoothing:
            Smooth the power spectrum by convolving with a uniform
            kernel of this width in frequency bins (default 1, no
            smoothing).  The old misspelling ``smothening`` still works
            but raises a DeprecationWarning.
        :param float block_length:
            Length of each trajectory block expressed in *t_unit*.
            Mutually exclusive with *nr_of_blocks*.
        :param int nr_of_blocks:
            Number of trajectory blocks (default 1).
            Mutually exclusive with *block_length*.
        """

        from scipy import signal

        if 'smothening' in kwargs:
            warn("'smothening' is a misspelling and is deprecated, use "
                 "'smoothing' instead", DeprecationWarning, stacklevel=2)
            smoothing = kwargs.pop('smothening')
        _check_deprecated_time_kwargs(kwargs)

        timestep_fs, trajectories = self._require_trajectories()
        # Sampling frequency of the trajectory in THz (inverse ps)
        sampling_frequency_THz = 1e3 / timestep_fs

        # require_fitting=False: a periodogram has neither a fit nor a
        # lag window, so t_end_dt comes back 0 and _resolve_blocks lets
        # the blocks tile the whole trajectory.
        p = self._get_running_params(
            timestep_fs, t_unit=t_unit,
            species_of_interest=species_of_interest,
            block_length=block_length, nr_of_blocks=nr_of_blocks,
            require_fitting=False,
        )
        species_of_interest = p.species_of_interest
        smoothing = int(smoothing)

        power_spectrum = TimeSeries()

        for index_of_species, atomic_species in enumerate(species_of_interest):
            periodogram_this_species = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    vel_array = trajectory.get_array(
                        arrayname)[:, trajectory.get_indices_of_species(
                            atomic_species, start=0), :]
                else:
                    vel_array = trajectory.get_velocities(
                        remove_angular_momentum=remove_angular_momentum)[
                        :, trajectory.get_indices_of_species(atomic_species,
                                                             start=0), :]
                nstep, _, _ = vel_array.shape
                block_length_dt_this_traj, nr_of_blocks_this_traj = (
                    _resolve_blocks(nstep, p))

                # I need to have blocks of equal length, and use the
                # split method I need the length of the array to be
                # a multiple of nr_of_blocks_this_traj
                blocks = np.array(np.split(
                    vel_array[:nr_of_blocks_this_traj
                              * block_length_dt_this_traj],
                    nr_of_blocks_this_traj, axis=0))
                nblocks = len(blocks)
                if self._verbosity > 0:
                    print('nblocks = {}, blocks.shape = {},'
                          ' block_length_fs = {}'.format(
                              nblocks, blocks.shape,
                              blocks.shape[1]*timestep_fs))

                freq, pd = signal.periodogram(blocks,
                                              fs=sampling_frequency_THz,
                                              axis=1, return_onesided=True)
                # I mean over all atoms of this species and directions
                # In the future, maybe consider having a
                # direction resolved periodogram?
                pd_this_species_this_traj = pd.mean(axis=(2, 3))
                # Smothening the array:
                if smoothing > 1:
                    # Running mean over frequency bins.  The kernel must
                    # be one row high: a (nblocks, smoothing) kernel
                    # convolves over the block axis as well, so every
                    # block's spectrum picked up its neighbours' and the
                    # std/sem computed from them were meaningless.
                    kernel = np.ones((1, smoothing)) / smoothing
                    pd_this_species_this_traj = convolve(
                        pd_this_species_this_traj,
                        kernel, mode='same')

                power_spectrum.set_array('periodogram_{}_{}'.format(
                    atomic_species, itraj), pd_this_species_this_traj)
                if not index_of_species:
                    # I need to save the frequencies only once,
                    # so I save them only for the first species.
                    # I do not see any problem here, but maybe I
                    # missed something.
                    power_spectrum.set_array(
                        'frequency_{}'.format(itraj), freq)
                for block in pd_this_species_this_traj:
                    periodogram_this_species.append(block)
            try:
                length_last_block = len(block)
                for pd in periodogram_this_species:
                    if len(pd) != length_last_block:
                        raise Exception(
                            'Cannot calculate mean signal '
                            ' because of different lengths')
                periodogram_this_species = np.array(
                    periodogram_this_species)
                mean = periodogram_this_species.mean(axis=0)
                if len(periodogram_this_species) > 1:
                    std = periodogram_this_species.std(axis=0)
                    sem = std / np.sqrt(len(periodogram_this_species) - 1)
                else:
                    # A single block carries no spread; reporting 0/0
                    # here produced a RuntimeWarning and a silent NaN.
                    # get_msd and get_vaf fill NaN in the same situation.
                    std = np.full(mean.shape, np.nan)
                    sem = np.full(mean.shape, np.nan)
                power_spectrum.set_array(
                    'periodogram_{}_mean'.format(atomic_species), mean)
                power_spectrum.set_array(
                    'periodogram_{}_std'.format(atomic_species), std)
                power_spectrum.set_array(
                    'periodogram_{}_sem'.format(atomic_species), sem)
            except Exception as e:
                # Not the end of the world, I just don't calculate the mean
                print(e)

        for k, v in (('species_of_interest', species_of_interest),
                     ('nr_of_trajectories', len(trajectories)),):
            power_spectrum.set_attr(k, v)
        return power_spectrum
