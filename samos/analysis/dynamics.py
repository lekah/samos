# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
from scipy.stats import linregress
from scipy.signal import convolve
from samos.trajectory import check_trajectory_compatibility, Trajectory
from samos.utils.attributed_array import AttributedArray
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


class DynamicsAnalyzer(object):
    """
    This class serves as the main analyzer
    for dynamic properties (MSD, VAF, etc)
    """

    def __init__(self, **kwargs):
        self._species_of_interest = None
        self._verbosity = 1
        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

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

    def _get_running_params(self, timestep_fs, t_unit='ps',
                            species_of_interest=None,
                            stepsize_t=1, stepsize_tau=1,
                            t_start=0, t_end=None,
                            t_start_fit=0, t_end_fit=None,
                            block_length=None, nr_of_blocks=None,
                            do_com=False, do_long=False,
                            t_long_end=None, t_long_factor=None):
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
        :returns: :class:`RunningParams` namedtuple.
        :raises InputError: For invalid parameter combinations.
        """
        if species_of_interest is None:
            species_of_interest = self.get_species_of_interest()

        stepsize_t = int(stepsize_t)
        stepsize_tau = int(stepsize_tau)

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

        # Both must be the same kind (scalar or array) and, if arrays,
        # the same length.
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

        # --- t_start (default 0) ---
        t_start_dt = parse_time(t_start, t_unit, timestep_fs)
        if t_start_dt < 0:
            raise InputError('t_start must be >= 0')
        if t_start_dt > 0:
            raise NotImplementedError('t_start > 0 is not implemented')

        # --- t_end (default: max of t_end_fit) ---
        if t_end is not None:
            t_end_dt = parse_time(t_end, t_unit, timestep_fs)
        else:
            t_end_dt = int(np.max(t_end_fit_dt))
        if t_end_dt <= t_start_dt:
            raise InputError('t_end must be greater than t_start')
        if t_end_dt < int(np.max(t_end_fit_dt)):
            raise InputError('t_end must be >= t_end_fit')

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
                backend='fortran', num_threads=None,
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
        :param str backend:
            Compute kernel: ``'fortran'`` (default) or ``'cpp'`` (OpenMP).
        :param int num_threads:
            Number of OpenMP threads for the C++ backend. Only used when
            backend='cpp'. Defaults to the current OMP_NUM_THREADS setting.
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

        if backend == 'cpp':
            from samos.lib.mdutils_cpp_omp import (
                calculate_msd_specific_atoms,
                calculate_msd_specific_atoms_decompose_d,
                calculate_msd_specific_atoms_max_stats,
                get_com_positions, set_num_threads)
            if num_threads is not None:
                set_num_threads(int(num_threads))
        elif backend == 'fortran':
            from samos.lib.mdutils import (
                calculate_msd_specific_atoms,
                calculate_msd_specific_atoms_decompose_d,
                calculate_msd_specific_atoms_max_stats,
                get_com_positions)
        else:
            raise ValueError("backend must be 'cpp' or 'fortran'")
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        p = self._get_running_params(
            timestep_fs, t_unit=t_unit,
            species_of_interest=species_of_interest,
            stepsize_t=stepsize_t, stepsize_tau=stepsize_tau,
            t_start=t_start, t_end=t_end,
            t_start_fit=t_start_fit, t_end_fit=t_end_fit,
            block_length=block_length, nr_of_blocks=nr_of_blocks,
            do_com=do_com, do_long=do_long,
            t_long_end=t_long_end, t_long_factor=t_long_factor,
        )
        multiple_params_fit = not isinstance(p.t_start_fit_dt, int)

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
                    # I replace the array positions with the COM!
                    # Getting the massesfor recentering
                    masses = self._atoms.get_masses()
                    factors = [1]*len(masses)
                    positions = get_com_positions(positions, masses, factors)
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_indices_of_species(
                        atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1
                if atom_indices is not None:
                    indices_of_interest = [
                        i for i in indices_of_interest if i in atom_indices]

                # make blocks
                nstep, nat, _ = positions.shape
                if p.nr_of_blocks:
                    block_length_dt_this_traj = (
                        nstep - p.t_end_dt) // p.nr_of_blocks
                    nr_of_blocks_this_traj = p.nr_of_blocks
                elif p.block_length_dt is not None and p.block_length_dt > 0:
                    block_length_dt_this_traj = p.block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - p.t_end_dt) // p.block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_dt '
                        'was specified')
                if (
                    (nr_of_blocks_this_traj < 0)
                        or (block_length_dt_this_traj < 0)):
                    raise RuntimeError(
                        't_end_dt (or t_end_fit_dt) is bigger'
                        ' than the trajectory length')

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
                            atomic_species,
                            nr_of_blocks_this_traj,
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
                if multiple_params_fit:
                    # using lists of (t_start_fit_dt, t_end_fit_dt):
                    # we will loop over them
                    if decomposed:
                        slopes_intercepts = np.empty(
                            (len(p.t_start_fit_dt),
                             nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (len(p.t_start_fit_dt),
                             nr_of_blocks_this_traj, 2))
                    for istart, (current_t_start_fit_dt,
                                 current_t_end_fit_dt) in enumerate(
                            zip(p.t_start_fit_dt, p.t_end_fit_dt)):
                        current_t_list_fit_fs = (
                            timestep_fs * p.stepsize_t *
                            np.arange(
                                current_t_start_fit_dt // p.stepsize_t,
                                current_t_end_fit_dt // p.stepsize_t))
                        for iblock, block in enumerate(
                                msd_this_species_this_traj):
                            if decomposed:
                                for ipol in range(3):
                                    for jpol in range(3):
                                        data = block[
                                            current_t_start_fit_dt
                                            // p.stepsize_t:
                                            current_t_end_fit_dt
                                            // p.stepsize_t,
                                            ipol, jpol]
                                        slope, intercept, _, _, _ = (
                                            linregress(
                                                current_t_list_fit_fs,
                                                data))
                                        slopes_intercepts[
                                            istart, iblock,
                                            ipol, jpol, 0] = slope
                                        slopes_intercepts[
                                            istart, iblock,
                                            ipol, jpol, 1] = intercept
                            else:
                                data = block[
                                    (current_t_start_fit_dt
                                     - p.t_start_dt) // p.stepsize_t:
                                    current_t_end_fit_dt // p.stepsize_t]
                                slope, intercept, _, _, _ = linregress(
                                    current_t_list_fit_fs, data)
                                slopes_intercepts[istart, iblock, 0] = slope
                                slopes_intercepts[
                                    istart, iblock, 1] = intercept
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            slopes.append(
                                slopes_intercepts[:, iblock, :, :, 0])
                        else:
                            slopes.append(slopes_intercepts[:, iblock, 0])
                else:
                    # just one value of (t_start_fit_dt, t_end_fit_dt)
                    # TODO: we could avoid this special case by defining
                    # t_start_fit_dt as a length-1 list instead of int.
                    if decomposed:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 2))
                    t_list_fit_fs = (
                        timestep_fs * p.stepsize_t
                        * np.arange(
                            p.t_start_fit_dt // p.stepsize_t,
                            p.t_end_fit_dt // p.stepsize_t))
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            for ipol in range(3):
                                for jpol in range(3):
                                    slope, intercept, _, _, _ = linregress(
                                        t_list_fit_fs,
                                        block[
                                            p.t_start_fit_dt // p.stepsize_t:
                                            p.t_end_fit_dt // p.stepsize_t,
                                            ipol, jpol])
                                    slopes_intercepts[
                                        iblock, ipol, jpol, 0] = slope
                                    slopes_intercepts[
                                        iblock, ipol, jpol, 1] = intercept
                            slopes.append(slopes_intercepts[iblock, :, :, 0])
                        else:
                            slope, intercept, _, _, _ = linregress(
                                t_list_fit_fs,
                                block[
                                    (p.t_start_fit_dt - p.t_start_dt)
                                    // p.stepsize_t:
                                    p.t_end_fit_dt // p.stepsize_t])
                            slopes_intercepts[iblock, 0] = slope
                            slopes_intercepts[iblock, 1] = intercept
                            slopes.append(slopes_intercepts[iblock, 0])

                msd.set_array('slopes_intercepts_{}_{}_{}'.format(
                    'decomposed' if decomposed else 'isotropic',
                    atomic_species, itraj),
                    slopes_intercepts)

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
                msd_std = np.full(msd_mean.shape, np.NaN)
                msd_sem = np.full(msd_mean.shape, np.NaN)
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
                    np.NaN)
                results_dict[atomic_species]['slope_msd_sem'] = np.full(
                    results_dict[atomic_species]['slope_msd_mean'].shape,
                    np.NaN)

            if decomposed:
                dimensionality_factor = 2.
            else:
                dimensionality_factor = 6.
            results_dict[atomic_species]['diffusion_mean_cm2_s'] = 1e-1 / \
                dimensionality_factor * \
                results_dict[atomic_species]['slope_msd_mean']
            if (len(msd_this_species) > 1):
                results_dict[atomic_species]['diffusion_std_cm2_s'] = 1e-1 / \
                    dimensionality_factor * \
                    results_dict[atomic_species]['slope_msd_std']
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = 1e-1 / \
                    dimensionality_factor * \
                    results_dict[atomic_species]['slope_msd_sem']
            else:
                results_dict[atomic_species]['diffusion_std_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.NaN)
                results_dict[atomic_species]['diffusion_sem_cm2_s'] = np.full(
                    results_dict[atomic_species]['diffusion_mean_cm2_s'].shape,
                    np.NaN)

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

        from samos.lib.mdutils import (
            calculate_vaf_specific_atoms,
            get_com_velocities)
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories '
                'method to set trajectories'
                '\n{}\n'.format(e)
            )

        p = self._get_running_params(
            timestep_fs, t_unit=t_unit,
            species_of_interest=species_of_interest,
            stepsize_t=stepsize_t, stepsize_tau=stepsize_tau,
            t_start=t_start, t_end=t_end,
            t_start_fit=t_start_fit, t_end_fit=t_end_fit,
            block_length=block_length, nr_of_blocks=nr_of_blocks,
            do_com=do_com,
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
                    # I replace the array positions with the COM!
                    # Getting the masses for recentering:
                    masses = self._atoms.get_masses()
                    factors = [1]*len(masses)
                    velocities = get_com_velocities(
                        velocities, masses, factors)
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_indices_of_species(
                        atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_indices_of_species(
                        atomic_species, start=1)
                    prefactor = 1

                nstep, nat, _ = velocities.shape
                if p.nr_of_blocks:
                    block_length_dt_this_traj = (
                        nstep - p.t_end_dt) // p.nr_of_blocks
                    nr_of_blocks_this_traj = p.nr_of_blocks
                elif (p.block_length_dt is not None
                        and p.block_length_dt > 0):
                    block_length_dt_this_traj = p.block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - p.t_end_dt) // p.block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_dt is '
                        'specified')

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
                # transforming A^2/fs -> cm^2 /s, dividing by three to get D
                vaf_integral *= 0.1/3. * prefactor

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
                arr_std = np.std(arr, axis=0)
                arr_sem = arr_std / np.sqrt(arr.shape[0] - 1)
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
                diffusion_mean_cm2_s=fitted_means_of_integral.mean(),
                diffusion_std_cm2_s=fitted_means_of_integral.std())

            results_dict[atomic_species]['diffusion_sem_cm2_s'] = (
                results_dict[atomic_species]['diffusion_std_cm2_s']
                / np.sqrt(len(fitted_means_of_integral) - 1))

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

        try:
            timestep_fs = self._timestep_fs
            atoms = self._atoms
            masses = atoms.get_masses()
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        prefactor = amu_kg * 1e10 / kB
        # * 1.06657254018667

        if decompose_atoms and decompose_species:
            raise Exception('Cannot decompose atoms and decompose species')

        kinetic_energies_series = TimeSeries()
        kinetic_energies_series.set_attr('stepsize', stepsize)
        kinetic_energies_series.set_attr('timestep_fs', timestep_fs)

        for itraj, t in enumerate(trajectories):
            vel_array = t.get_velocities()
            nstep, nat, _ = vel_array.shape
            steps = list(range(0, nstep, stepsize))

            if decompose_system:

                kinE = np.zeros(len(steps))
                for istep0, istep in enumerate(steps):
                    for iat in range(nat):
                        for ipol in range(3):
                            kinE[istep0] += (
                                prefactor * masses[iat]
                                * vel_array[istep, iat, ipol]**2)
                kinE[:] /= nat*3  # I devide by the degrees of freedom!
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
                    for istep0, istep in enumerate(steps):
                        for _, iat in enumerate(indices_of_interest):
                            for ipol in range(3):
                                kinE_species[istep0, ityp] += (
                                    prefactor * masses[iat] *
                                    vel_array[istep, iat, ipol]**2)

                    kinE_species[:, ityp] /= float(len(indices_of_interest)*3)

                kinetic_energies_series.set_array(
                    'species_kinetic_energy_{}'.format(itraj), kinE)
                kinetic_energies_series.set_attr(
                    'species_of_interest', species_of_interest)
                kinetic_energies_series.set_attr(
                    'mean_species_kinetic_energy_{}'.format(itraj),
                    kinE_species.mean(axis=0).tolist())

            if decompose_atoms:
                kinE = np.zeros((len(steps), nat))
                for istep0, istep in enumerate(steps):
                    # ~ print istep0
                    for iat in range(nat):
                        for ipol in range(3):
                            kinE[istep0, iat] += prefactor * masses[iat] * \
                                vel_array[istep, iat, ipol]**2 / 3.

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
                           smothening=1,
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
        :param int smothening:
            Smooth the power spectrum by convolving with a uniform
            kernel of this width in frequency bins (default 1, no
            smoothing).
        :param float block_length:
            Length of each trajectory block expressed in *t_unit*.
            Mutually exclusive with *nr_of_blocks*.
        :param int nr_of_blocks:
            Number of trajectory blocks (default 1).
            Mutually exclusive with *block_length*.
        """

        from scipy import signal

        _check_deprecated_time_kwargs(kwargs)

        try:
            trajectories = self._trajectories
            timestep_fs = self._timestep_fs
            # Calculating the sampling frequency of the
            # trajectory in THz (the inverse of a picosecond)
            sampling_frequency_THz = 1e3 / timestep_fs
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

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

        if species_of_interest is None:
            species_of_interest = self.get_species_of_interest()
        smothening = int(smothening)

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

                if nr_of_blocks:
                    nr_of_blocks_this_traj = nr_of_blocks
                    # Use the number of blocks specified by user
                    split_number = nstep // nr_of_blocks_this_traj
                elif block_length_dt is not None and block_length_dt > 0:
                    nr_of_blocks_this_traj = nstep // block_length_dt
                    # Use the precise length specified by user
                    split_number = block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_dt '
                        'is specified')

                # I need to have blocks of equal length, and use the
                # split method I need the length of the array to be
                # a multiple of nr_of_blocks_this_traj
                blocks = np.array(np.split(
                    vel_array[:nr_of_blocks_this_traj*split_number],
                    nr_of_blocks_this_traj, axis=0))
                nblocks = len(blocks)
                if self._verbosity > 0:
                    print('nblocks = {}, blocks.shape = {},'
                          ' block_length_ps = {}'.format(
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
                if smothening > 1:
                    # Applying a simple convolution to get the mean
                    kernel = np.ones((nblocks, smothening)) / smothening
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
                power_spectrum.set_array('periodogram_{}_mean'.format(
                    atomic_species), periodogram_this_species.mean(axis=0))
                std = periodogram_this_species.std(axis=0)
                power_spectrum.set_array(
                    'periodogram_{}_std'.format(atomic_species), std)
                power_spectrum.set_array('periodogram_{}_sem'.format(
                    atomic_species),
                    std/np.sqrt(len(periodogram_this_species)-1))
            except Exception as e:
                # Not the end of the world, I just don't calculate the mean
                print(e)

        for k, v in (('species_of_interest', species_of_interest),
                     ('nr_of_trajectories', len(trajectories)),):
            power_spectrum.set_attr(k, v)
        return power_spectrum


def util_msd(trajectory_path, stepsize=1, species=None,
             plot=True, savefig=None, t_start_fit_ps=5,
             t_end_fit_ps=10, timestep=None, nblocks=None):
    if trajectory_path.endswith('.extxyz'):
        from ase.io import read
        aselist = read(trajectory_path, format='extxyz', index=':')
        traj = Trajectory.from_atoms(aselist)

    else:
        traj = Trajectory.load_file(trajectory_path)
    if timestep:
        traj.set_timestep(timestep)
    dyn = DynamicsAnalyzer(trajectories=[traj])
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))
    print(traj.get_timestep(), species)
    print(t_start_fit_ps, t_end_fit_ps, timestep)
    msd = dyn.get_msd(stepsize_t=stepsize,
                      species_of_interest=species,
                      t_end_fit=t_end_fit_ps,
                      t_start_fit=t_start_fit_ps,
                      nr_of_blocks=nblocks,
                      t_unit='ps')
    if plot or savefig:
        from samos.plotting.plot_dynamics import plot_msd_isotropic
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1, left=0.18, right=0.95, bottom=0.18, top=0.95)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(gs[0])
        plot_msd_isotropic(msd, ax=ax)
        if plot:
            plt.show()
        elif savefig:
            plt.savefig(savefig, dpi=240)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('analysis/plot of a MSD, given a trajectory')
    parser.add_argument('trajectory_path')
    parser.add_argument('-s', '--stepsize', type=int,
                        help='Stepsize over the trajectory, defaults to 1',
                        default=1)
    parser.add_argument('--species', nargs='+',)
    parser.add_argument('--plot', action='store_true',
                        help='Plot the MSD to screen')
    parser.add_argument('-te', '--t-end-fit-ps',
                        help='End of the fit in ps, defaults to 100',
                        type=float, default=10)
    parser.add_argument('-ts', '--t-start-fit-ps',
                        help='End of the fit in ps, defaults to 50',
                        type=float, default=5)
    parser.add_argument('--timestep', type=float,
                        help='Timestep in fs, defaults to 1',
                        default=1)
    parser.add_argument('-n', '--nblocks', type=int,
                        default=1, help='Number of blocks to use')
    parser.add_argument(
        '--savefig',
        help='Where to save figure (will otherwise show on screen)')
    args = parser.parse_args()
    kwargs = vars(args)
    util_msd(**kwargs)
