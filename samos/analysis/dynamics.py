# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import linregress
from scipy.signal import convolve
from samos.trajectory import check_trajectory_compatibility, Trajectory
from samos.utils.attributed_array import AttributedArray
from samos.utils.exceptions import InputError


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
        (self._atoms,
         self._timestep_fs) = check_trajectory_compatibility(trajectories)
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

    @property
    def atoms(self):
        try:
            return self._atoms
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method'
                ' to set trajectories, and I will get the atoms from there.'
                '\n{}\n'.format(e)
            )

    def set_verbosity(self, verbosity):
        if not isinstance(verbosity, int):
            raise TypeError('Verbosity is an integer')
        self._verbosity = verbosity

    def get_species_of_interest(self):
        # Also a good way to check if atoms have been set
        atoms = self.atoms
        if self._species_of_interest is None:
            return sorted(set(atoms.get_chemical_symbols()))
        else:
            return self._species_of_interest

    def _get_running_params(self, timestep_fs, **kwargs):
        """
        Utility function to get a number of parameters.
        :param list species_of_interest: The species to calculate.
        :param int stepsize_t: Integer value of the outer-loop stepsize.
            Setting this to higher than 1 will decrease the resolution.
            Defaults to 1
        :param int stepsize_tau: Integer value of the inner loop stepsize.
            If higher than 1, the sliding window will be moved more
            sparsely through the block. Defaults to 1.
        :param float t_start_fs:
            Minimum value of the sliding window
            in femtoseconds.
        :param float t_start_ps:
            Minimum value of
            the sliding window in picoseconds.
        :param int t_start_dt:
            Minimum value of the sliding window in
            multiples of the trajectory timestep.
        :param float t_end_fs:
            Maximum value of the sliding window
            in femtoseconds.
        :param float t_end_ps:
            Maximum value of the
            sliding window in picoseconds.
        :param int t_end_dt:
            Maximum value of the sliding window in
            multiples of the trajectory timestep.
        :param float block_length_fs:
            Block size for trajectory blocking in fs.
        :param float block_length_ps:
            Block size for trajectory blocking in picoseconds.
        :param int block_length_dt:
            Block size for trajectory blocking in
            multiples of the trajectory timestep.
        :param int nr_of_blocks:
            Nr of blocks that the trajectory should
            be split in (excludes setting of block_length).
            If nothing else is set, defaults to 1.
        :param float t_start_fit_fs:
            Time to start the fitting of the time series in femtoseconds.
        :param float t_start_fit_ps:
            Time to start the fitting of the time series in picoseconds.
        :param int t_start_fit_dt:
            Time to end the fitting of the time series
            in multiples of the trajectory timestep.
        :param float t_end_fit_fs:
            Time to end the fitting of
            the time series in femtoseconds.
        :param float t_end_fit_ps:
            Time to end the fitting of the time series in picoseconds.
        :param int t_end_fit_dt:
            Time to end the fitting of the time series
            in multiples of the trajectory timestep.
        :param bool do_long:
            whether to perform a maximum-statistics MSD calculation,
            using the whole trajectory and no blocks.
        :param float t_long_end_fs:
            Maximum value of the sliding window in
            femtoseconds used in maximum-statistics calculation.
        :param float t_long_end_ps:
            Maximum value of the sliding window in
            picoseconds used in maximum-statistics calculation.
        :param int t_long_end_dt:
            Maximum value of the sliding window in multiples
            of the trajectory timestep used in
            maximum-statistics calculation.
        :param bool do_com:
            whether to calculate center-of-mass
            diffusion instead of tracer diffusion.
        """

        species_of_interest = kwargs.pop(
            'species_of_interest', self.get_species_of_interest())

        stepsize_t = int(kwargs.pop('stepsize_t', 1))
        stepsize_tau = int(kwargs.pop('stepsize_tau', 1))

        keywords_provided = list(kwargs.keys())
        for mutually_exclusive_keys in (
                ('t_start_fs', 't_start_ps', 't_start_dt'),
                ('t_end_fs', 't_end_ps', 't_end_dt'),
                ('block_length_fs', 'block_length_ps',
                 'block_length_dt', 'nr_of_blocks'),
                ('t_start_fit_fs', 't_start_fit_ps', 't_start_fit_dt'),
                ('t_end_fit_fs', 't_end_fit_ps', 't_end_fit_dt'),
                ('t_long_end_fs', 't_long_end_ps',
                 't_long_end_dt', 't_long_factor'),
        ):
            keys_provided_this_group = [
                k for k in mutually_exclusive_keys if k in keywords_provided]
            if len(keys_provided_this_group) > 1:
                raise InputError(
                    'This keywords are mutually exclusive: {}'.format(
                        ', '.join(keys_provided_this_group)))

        if 't_start_fit_fs' in keywords_provided:
            arg = kwargs.pop('t_start_fit_fs')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.rint(
                    np.array(arg, dtype=float) / timestep_fs).astype(int)
            else:
                t_start_fit_dt = int(float(arg) / timestep_fs)
        elif 't_start_fit_ps' in keywords_provided:
            arg = kwargs.pop('t_start_fit_ps')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.rint(
                    1000 * np.array(arg, dtype=float) / timestep_fs
                ).astype(int)
            else:
                t_start_fit_dt = int(1000 * float(arg) / timestep_fs)
        elif 't_start_fit_dt' in keywords_provided:
            arg = kwargs.pop('t_start_fit_dt')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_start_fit_dt = np.array(arg, dtype=int)
            else:
                t_start_fit_dt = int(arg)
        else:
            t_start_fit_dt = 0

        if not np.all(np.array(t_start_fit_dt >= 0)):
            raise InputError('t_start_fit_dt is not positive or 0')

        if 't_end_fit_fs' in keywords_provided:
            arg = kwargs.pop('t_end_fit_fs')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.rint(
                    np.array(arg, dtype=float) / timestep_fs).astype(int)
            else:
                t_end_fit_dt = int(float(arg) / timestep_fs)
        elif 't_end_fit_ps' in keywords_provided:
            arg = kwargs.pop('t_end_fit_ps')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.rint(
                    1000 * np.array(arg, dtype=float)
                    / timestep_fs
                ).astype(int)
            else:
                t_end_fit_dt = int(1000 * float(arg) / timestep_fs)
        elif 't_end_fit_dt' in keywords_provided:
            arg = kwargs.pop('t_end_fit_dt')
            if isinstance(arg, (list, tuple, np.ndarray)):
                t_end_fit_dt = np.array(arg, dtype=int)
            else:
                t_end_fit_dt = int(arg)
        else:
            raise InputError('Provide a time to end fitting the time series')

        if not np.all(t_end_fit_dt > t_start_fit_dt):
            raise InputError('t_end_fit_dt must be larger than '
                             't_start_fit_dt')

        if not isinstance(t_start_fit_dt, int):
            if isinstance(t_end_fit_dt, int):
                raise InputError(
                    't_start_fit_dt and t_end_fit_dt'
                    ' must be both integers or lists')
            elif (len(t_start_fit_dt) != len(t_end_fit_dt)):
                raise InputError(
                    't_start_fit_dt and t_end_fit_dt must '
                    'be of the same size')
        elif not isinstance(t_end_fit_dt, int):
            raise InputError(
                't_start_fit_dt and t_end_fit_dt must be both '
                'integers or lists')

        if 't_start_fs' in keywords_provided:
            t_start_dt = int(float(kwargs.pop('t_start_fs')) / timestep_fs)
        elif 't_start_ps' in keywords_provided:
            t_start_dt = int(
                1000 * float(kwargs.pop('t_start_ps')) / timestep_fs)
        elif 't_start_dt' in keywords_provided:
            t_start_dt = int(kwargs.pop('t_start_dt'))
        else:
            # By default I create the time series from the start
            t_start_dt = 0

        if not (t_start_dt >= 0):
            raise InputError('t_start_dt is not positive or 0')
        if t_start_dt > 0:
            raise NotImplementedError('t_start has not been implemented yet!')

        if 't_end_fs' in keywords_provided:
            t_end_dt = int(float(kwargs.pop('t_end_fs')) / timestep_fs)
        elif 't_end_ps' in keywords_provided:
            t_end_dt = int(1000 * float(kwargs.pop('t_end_ps')) / timestep_fs)
        elif 't_end_dt' in keywords_provided:
            t_end_dt = int(kwargs.pop('t_end_dt'))
        else:
            t_end_dt = int(np.max(t_end_fit_dt))

        if not (t_end_dt > t_start_dt):
            raise InputError('t_end_dt is not larger than t_start_dt')
        if not (t_end_dt >= np.max(t_end_fit_dt)):
            raise InputError('t_end_dt must be larger than t_end_fit_dt')

        # The number of timesteps I will calculate:
        nr_of_t = (t_end_dt - t_start_dt) // stepsize_t

        # Checking if I have to partition the trajectory into blocks
        # (By default just 1 block)
        if 'block_length_fs' in keywords_provided:
            block_length_dt = int(
                float(kwargs.pop('block_length_fs')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_ps' in keywords_provided:
            block_length_dt = int(
                1000 * float(kwargs.pop('block_length_ps')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_dt' in keywords_provided:
            block_length_dt = int(kwargs.pop('block_length_dt'))
            nr_of_blocks = None
        elif 'nr_of_blocks' in keywords_provided:
            nr_of_blocks = int(kwargs.pop('nr_of_blocks'))
            block_length_dt = None
        else:
            nr_of_blocks = 1
            block_length_dt = None

        # Asking whether to calculate COM diffusion
        do_com = kwargs.pop('do_com', False)

        # Asking whether to calculate for every trajectory
        # a time series with maximal statistics:
        do_long = kwargs.pop('do_long', False)
        if 't_long_end_fs' in keywords_provided:
            t_long_end_dt = int(
                float(kwargs.pop('t_long_end_fs')) / timestep_fs)
            t_long_factor = None
        elif 't_long_end_ps' in keywords_provided:
            t_long_end_dt = int(
                1000 * float(kwargs.pop('t_long_end_ps')) / timestep_fs)
            t_long_factor = None
        elif 't_long_end_dt' in keywords_provided:
            t_long_end_dt = int(kwargs.pop('t_long_end_dt'))
            t_long_factor = None
        elif 't_long_factor' in keywords_provided:
            t_long_factor = float(kwargs.pop('t_long_factor'))
            t_long_end_dt = None
        else:
            t_long_end_dt = None  # will be adapted to trajectory length!!
            t_long_factor = None  # will be adapted to trajectory length!!

        # Irrespective of whether do_long is false or true,
        # I see whether factors are calculated:

        if kwargs:
            raise InputError(
                'Uncrecognized keywords: {}'.format(list(kwargs.keys())))

        return (species_of_interest, nr_of_blocks, t_start_dt,
                t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
                stepsize_t, stepsize_tau, block_length_dt, do_com,
                do_long, t_long_end_dt, t_long_factor)

    def get_msd(self, decomposed=False, atom_indices=None, **kwargs):
        """
        Calculates the mean square discplacement (MSD),

        #.  Calculate the MSD for each block
        #.  Calculate the mean and the standard deviation of the slope
        #.  Calculate the conductivity, including error propagation.

        :param bool decomposed:
            Compute the (3,3) MSD matrix by computing
            each Cartesian component independently.
        :param list species_of_interest:
            The species of interest for which to calculate
            the MSD, for example ["O", "H"]
        :param list atom_indices:
            The indices of interest for which to calculate the MSD,
            for example [0, 1, 2, 5].
            The intersection of atom_indices and species_of_interest
            is taken, so aotm_indices can be used to narrow the list of atoms
        :param **kwargs:
            All other parameters required by
            DynamicsAnalyzer._get_running_params()

        For this function t_start_fit_* and t_end_fit_* can also
        be lists/arrays. The slope and conductivities
        will be computed for each (t_start_fit, t_end_fit) pair.
        """
        from samos.lib.mdutils import (
            calculate_msd_specific_atoms,
            calculate_msd_specific_atoms_decompose_d,
            calculate_msd_specific_atoms_max_stats, get_com_positions)
        try:
            timestep_fs = self._timestep_fs
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                '\n\n\n'
                'Please use the set_trajectories method to set trajectories'
                '\n{}\n'.format(e)
            )

        (
            species_of_interest, nr_of_blocks, t_start_dt, t_end_dt,
            t_start_fit_dt, t_end_fit_dt, nr_of_t, stepsize_t,
            stepsize_tau, block_length_dt, do_com, do_long, t_long_end_dt,
            t_long_factor) = self._get_running_params(timestep_fs, **kwargs)
        multiple_params_fit = not isinstance(t_start_fit_dt, int)

        msd = TimeSeries()
        # list of t at which MSD will be computed
        t_list_fs = timestep_fs * stepsize_t * \
            (t_start_dt + np.arange(nr_of_t))
        msd.set_array('t_list_fs', t_list_fs)

        results_dict = {atomic_species: {}
                        for atomic_species in species_of_interest}
        nr_of_t_long_list = []
        t_list_long_fs = []
        # Setting params for calculation of MSD and conductivity
        # Future: Maybe allow for element specific parameter settings?

        for atomic_species in species_of_interest:
            msd_this_species = []  # Here I collect the trajectories
            slopes = []  # That's where I collect slopes
            # for the final estimate of diffusion

            for itraj, trajectory in enumerate(trajectories):
                positions = trajectory.get_positions()
                if do_com:
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
                if nr_of_blocks:
                    block_length_dt_this_traj = (
                        nstep - t_end_dt) // nr_of_blocks
                    nr_of_blocks_this_traj = nr_of_blocks
                elif block_length_dt > 0:
                    block_length_dt_this_traj = block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - t_end_dt) // block_length_dt
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
                            t_start_fit_dt, t_start_fit_dt * timestep_fs / 1e3,
                            t_end_fit_dt, t_end_fit_dt * timestep_fs / 1e3,
                            stepsize_t, stepsize_tau)
                    ))
                if decomposed:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms_decompose_d(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj,
                            nr_of_t, nstep, nat, nat_of_interest)
                    )
                else:
                    msd_this_species_this_traj = (
                        prefactor * calculate_msd_specific_atoms(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, block_length_dt_this_traj,
                            nr_of_blocks_this_traj, nr_of_t, nstep,
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
                            (len(t_start_fit_dt),
                             nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (len(t_start_fit_dt), nr_of_blocks_this_traj, 2))
                    for istart, (current_t_start_fit_dt,
                                 current_t_end_fit_dt) in enumerate(
                            zip(t_start_fit_dt, t_end_fit_dt)):
                        current_t_list_fit_fs = (
                            timestep_fs * stepsize_t *
                            np.arange(current_t_start_fit_dt//stepsize_t,
                                      current_t_end_fit_dt//stepsize_t))
                        for iblock, block in enumerate(
                                msd_this_species_this_traj):
                            if decomposed:
                                for ipol in range(3):
                                    for jpol in range(3):
                                        data = block[
                                            current_t_start_fit_dt//stepsize_t:
                                            current_t_end_fit_dt//stepsize_t,
                                            ipol, jpol]
                                        slope, intercept, _, _, _ = linregress(
                                            current_t_list_fit_fs, data)
                                        slopes_intercepts[istart, iblock, ipol,
                                                          jpol, 0] = slope
                                        slopes_intercepts[istart, iblock, ipol,
                                                          jpol, 1] = intercept
                                # slopes.append(slopes_intercepts[istart,
                                # iblock, :, :, 0])
                            else:
                                data = block[
                                    (current_t_start_fit_dt-t_start_dt)//stepsize_t:  # noqa: E501
                                    current_t_end_fit_dt//stepsize_t]
                                slope, intercept, _, _, _ = linregress(
                                    current_t_list_fit_fs, data)
                                slopes_intercepts[istart, iblock, 0] = slope
                                slopes_intercepts[istart,
                                                  iblock, 1] = intercept
                                # slopes.append(slope)
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            slopes.append(
                                slopes_intercepts[:, iblock, :, :, 0])
                        else:
                            slopes.append(slopes_intercepts[:, iblock, 0])
                else:
                    # just one value of (t_start_fit_dt, t_end_fit_dt)
                    # TODO: we could avoid this special case by defining
                    # t_start_fit_dt as a lenght-1 list, instead of int.
                    # We keep it for backward-compatibility.
                    if decomposed:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 3, 3, 2))
                    else:
                        slopes_intercepts = np.empty(
                            (nr_of_blocks_this_traj, 2))
                    t_list_fit_fs = timestep_fs * stepsize_t * \
                        np.arange(t_start_fit_dt//stepsize_t,
                                  t_end_fit_dt//stepsize_t)
                    for iblock, block in enumerate(msd_this_species_this_traj):
                        if decomposed:
                            for ipol in range(3):
                                for jpol in range(3):
                                    slope, intercept, _, _, _ = linregress(
                                        t_list_fit_fs,
                                        block[t_start_fit_dt//stepsize_t:
                                              t_end_fit_dt//stepsize_t,
                                              ipol, jpol])
                                    slopes_intercepts[iblock,
                                                      ipol, jpol, 0] = slope
                                    slopes_intercepts[iblock, ipol,
                                                      jpol, 1] = intercept
                            slopes.append(slopes_intercepts[iblock, :, :, 0])
                        else:
                            slope, intercept, _, _, _ = linregress(
                                t_list_fit_fs,
                                block[(t_start_fit_dt-t_start_dt)//stepsize_t:
                                      t_end_fit_dt//stepsize_t])
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
                if do_long:
                    # nr_of_t_long may change with the length of the traj
                    # (if not set by user)
                    if t_long_end_dt is not None:
                        nr_of_t_long = t_long_end_dt // stepsize_t
                    elif t_long_factor is not None:
                        nr_of_t_long = int(t_long_factor *
                                           nstep / stepsize_t)
                    else:
                        nr_of_t_long = (nstep - 1) // stepsize_t
                    if nr_of_t_long > nstep:
                        raise RuntimeError(
                            't_long_end_dt is bigger than the '
                            'trajectory length')
                    nr_of_t_long_list.append(nr_of_t_long)
                    t_list_long_fs.append(
                        timestep_fs * stepsize_t * np.arange(nr_of_t_long))
                    msd_this_species_this_traj_max_stats = (
                        prefactor * calculate_msd_specific_atoms_max_stats(
                            positions, indices_of_interest, stepsize_t,
                            stepsize_tau, nr_of_t_long, nstep, nat,
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
            't_start_fit_dt': (t_start_fit_dt.tolist()
                               if multiple_params_fit
                               else t_start_fit_dt),
            't_end_fit_dt': (t_end_fit_dt.tolist()
                             if multiple_params_fit
                             else t_end_fit_dt),
            't_start_dt':   t_start_dt,
            't_end_dt':   t_end_dt,
            'nr_of_trajectories':   len(trajectories),
            'stepsize_t':   stepsize_t,
            'species_of_interest':   species_of_interest,
            'timestep_fs':   timestep_fs,
            'nr_of_t':   nr_of_t,
            'decomposed':   decomposed,
            'do_long':   do_long,
            'multiple_params_fit':   multiple_params_fit,
        })
        if do_long:
            results_dict['nr_of_t_long_list'] = nr_of_t_long_list
            msd.set_array('t_list_long_fs', t_list_long_fs)
        for k, v in results_dict.items():
            msd.set_attr(k, v)
        return msd

    def get_vaf(self, arrayname=None, integration='trapezoid', **kwargs):

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

        (species_of_interest, nr_of_blocks, t_start_dt, t_end_dt,
         t_start_fit_dt, t_end_fit_dt, nr_of_t,
            stepsize_t, stepsize_tau, block_length_dt,
            do_com, do_long, t_long_end_dt,
            _) = self._get_running_params(timestep_fs, **kwargs)
        if do_long:
            raise NotImplementedError('Do_long is not implemented for VAF')

        vaf_time_series = TimeSeries()

        results_dict = dict()

        for atomic_species in species_of_interest:

            vaf_this_species = []
            vaf_integral_this_species = []
            fitted_means_of_integral = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    velocities = trajectory.get_array(arrayname)
                else:
                    velocities = trajectory.get_velocities()
                if do_com:
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
                if nr_of_blocks > 0:
                    block_length_dt_this_traj = (
                        nstep - t_end_dt) // nr_of_blocks
                    nr_of_blocks_this_traj = nr_of_blocks
                elif block_length_dt > 0:
                    block_length_dt_this_traj = block_length_dt
                    nr_of_blocks_this_traj = (
                        nstep - t_end_dt) // block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_ft is '
                        'specified')

                # slopes_intercepts = np.empty((nr_of_blocks_this_traj, 2))

                nat_of_interest = len(indices_of_interest)

                if self._verbosity > 0:
                    print((
                        '\n    ! Calculating VAF for atomic species {} '
                        'in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      I will calculate {} block(s)'
                        ''.format(atomic_species, itraj,
                                  nat_of_interest, atomic_species,
                                  nr_of_blocks)
                    ))

                vaf, vaf_integral = calculate_vaf_specific_atoms(
                    velocities, indices_of_interest, stepsize_t, stepsize_tau,
                    nr_of_t, nr_of_blocks_this_traj, block_length_dt_this_traj,
                    timestep_fs*stepsize_t,
                    integration, nstep, nat, nat_of_interest)
                # transforming A^2/fs -> cm^2 /s, dividing by three to get D
                vaf_integral *= 0.1/3. * prefactor

                for iblock in range(nr_of_blocks_this_traj):

                    # ~ D =  0.1 / 3. * prefactor * vaf_integral[iblock]
                    vaf_this_species.append(vaf[iblock])
                    # ~ print vaf[iblock,0]
                    vaf_integral_this_species.append(vaf_integral[iblock])
                    data_ = vaf_integral[
                        iblock,
                        t_start_fit_dt//stepsize_t:t_end_fit_dt//stepsize_t]
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
                # ~ print name, arr_mean.shape
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
            't_start_fit_dt':   t_start_fit_dt,
            't_end_fit_dt':   t_end_fit_dt,
            't_start_dt':   t_start_dt,
            't_end_dt':   t_end_dt,

            'nr_of_trajectories':   len(trajectories),

            'stepsize_t':   stepsize_t,
            'species_of_interest':   species_of_interest,
            'timestep_fs':   timestep_fs,
            'nr_of_t':   nr_of_t, })

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

    def get_power_spectrum(self, arrayname=None, **kwargs):
        """
        Calculate the power spectrum.
        :param int smothening:
            Smothen the power spectrum by
            taking a mean every N steps.
        """

        from scipy import signal

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

        keywords_provided = list(kwargs.keys())
        for mutually_exclusive_keys in (
                ('block_length_fs', 'block_length_ps',
                 'block_length_dt', 'nr_of_blocks'),):
            keys_provided_this_group = [k for k
                                        in mutually_exclusive_keys
                                        if k in keywords_provided]
            if len(keys_provided_this_group) > 1:
                raise InputError(
                    'This keywords are mutually exclusive: '
                    '{}'.format(', '.join(keys_provided_this_group)))
        if 'block_length_fs' in keywords_provided:
            block_length_dt = int(
                float(kwargs.pop('block_length_fs')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_ps' in keywords_provided:
            block_length_dt = int(
                1000*float(kwargs.pop('block_length_ps')) / timestep_fs)
            nr_of_blocks = None
        elif 'block_length_dt' in keywords_provided:
            block_length_dt = int(kwargs.pop('block_length_dt'))
            nr_of_blocks = None
        elif 'nr_of_blocks' in keywords_provided:
            nr_of_blocks = kwargs.pop('nr_of_blocks')
            block_length_dt = None
        else:
            nr_of_blocks = 1
            block_length_dt = None
        species_of_interest = kwargs.pop(
            'species_of_interest', None) or self.get_species_of_interest()
        smothening = int(kwargs.pop('smothening', 1))
        if kwargs:
            raise InputError(
                'Uncrecognized keywords: {}'.format(list(kwargs.keys())))

        power_spectrum = TimeSeries()

        for index_of_species, atomic_species in enumerate(species_of_interest):
            periodogram_this_species = []

            for itraj, trajectory in enumerate(trajectories):
                if arrayname:
                    vel_array = trajectory.get_array(
                        arrayname)[:, trajectory.get_indices_of_species(
                            atomic_species, start=0), :]
                else:
                    vel_array = trajectory.get_velocities()[
                        :, trajectory.get_indices_of_species(atomic_species,
                                                             start=0), :]
                nstep, _, _ = vel_array.shape

                if nr_of_blocks > 0:
                    nr_of_blocks_this_traj = nr_of_blocks
                    # Use the number of blocks specified by user
                    split_number = nstep // nr_of_blocks_this_traj
                elif block_length_dt > 0:
                    nr_of_blocks_this_traj = nstep // block_length_dt
                    # Use the precise length specified by user
                    split_number = block_length_dt
                else:
                    raise RuntimeError(
                        'Neither nr_of_blocks nor block_length_ft '
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
