
from ase import Atoms
import numpy as np
from scipy.stats import linregress
from scipy.stats import sem as standard_error_of_mean
from samos.trajectory import check_trajectory_compatibility, Trajectory
from samos.utils.attributed_array import AttributedArray


class TimeSeries(AttributedArray):
    pass




class DynamicsAnalyzer(object):
    """
    This class
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
        # I check the compatibility. Implicitly, also checks if trajectories are valid instances.
        self._atoms, self._timestep_fs = check_trajectory_compatibility(trajectories)
        # Setting as attribute of self for analysis
        self._trajectories = trajectories

    def set_species_of_interest(self, species_of_interest):
        """
        :param list species_of_interest: To set a global list of species of interest for all the analysis
        :todo: Check the species whether they are valid
        """
        if isinstance(species_of_interest, basestring):
            self._species_of_interest = [species_of_interest]
        elif isinstance(species_of_interest, (tuple, set, list)):
            self._species_of_interest = list(species_of_interest)
        else:
            raise TypeError("Species of interest has to be a list of strings with the atomic symbol")
    @property
    def atoms(self):
        try:
            return self._atoms
        except AttributeError as e:
            raise Exception(
                "\n\n\n"
                "Please use the set_trajectories method to set trajectories, and I will get the atoms from there."
                "\n{}\n".format(e)
            )

    def set_verbosity(self, verbosity):
        if not isinstance(verbosity, int):
            raise TypeError("Verbosity is an integer")
        self._verbosity = verbosity

    def get_species_of_interest(self):
        atoms = self.atoms # Also a good way to check if atoms have been set
        if self._species_of_interest is None:
            return sorted(set(atoms.get_chemical_symbols()))
        else:
            return self._species_of_interst

    def _get_running_params(self, timestep_fs, **kwargs):
        """
        Utility function to get a number of parameters
        """


        species_of_interest = kwargs.pop("species_of_interest", self.get_species_of_interest())
    
        stepsize_t  = kwargs.pop('stepsize_t', 1)
        stepsize_tau  = kwargs.pop('stepsize_tau', 1)
    
        t_start_fit_fs = kwargs.pop('t_start_fit_fs', None)
        t_start_fit_dt = kwargs.pop('t_start_fit_dt', None)
    
    
        if t_start_fit_fs and t_start_fit_dt:
            raise Exception("You cannot set both 't_start_fit_fs' and 't_start_fit_dt'")
        elif isinstance(t_start_fit_fs, (float, int)):
            t_start_fit_dt  = int(float(t_start_fit_fs) / timestep_fs)
        elif isinstance(t_start_fit_dt, int):
            block_length_fs  = t_start_fit_dt * timestep_fs
        else:
            raise Exception(
                "Please set the time to start the fit with keyword t_start_fit_fs or t_start_fit_dt"
            )
    
    
        t_start_fs = kwargs.pop('t_start_fs', None)
        t_start_dt = kwargs.pop('t_start_dt', 0)
    
        if isinstance(t_start_fs, float):
            t_start_dt = int(t_start_fs / timestep_fs)
        elif isinstance(t_start_dt, int):
            t_start_fs = t_start_dt*timestep_fs
        else:
            raise Exception(
                "\n\n\n"
                "Set the time that you consider the\n"
                "VAF to have converged as a float with\n"
                "keyword t_start_fs or t_start_dt"
                "\n\n\n"
            )
    
    
    
        t_end_fit_fs = kwargs.pop('t_end_fit_fs', None)
        t_end_fit_dt = kwargs.pop('t_end_fit_dt', None)
    
        if t_end_fit_fs and t_end_fit_dt:
            raise Exception("You cannot set both 't_end_fit_fs' and 't_end_fit_dt'")
        elif isinstance(t_end_fit_fs, (int,float)):
            t_end_fit_dt  = int(float(t_end_fit_fs) / timestep_fs)
        elif isinstance(t_end_fit_dt, int):
            block_length_fs  = t_end_fit_dt * timestep_fs
        else:
            raise Exception(
                "Please set the time to start the fit with keyword t_end_fit_fs or t_end_fit_dt"
            )
    
    
        t_end_fs = kwargs.pop('t_end_fs', None)
        t_end_dt = kwargs.pop('t_end_dt', None)
    
        if t_end_fs and t_end_dt:
            raise Exception("You cannot set both 't_end_fs' and 't_end_dt'")
        if isinstance(t_end_fs, float):
            t_end_dt = int(t_end_fs / timestep_fs)
        elif isinstance(t_end_dt, int):
            t_end_fs = timestep_fs * t_end_dt
        else:
            t_end_dt = t_end_fit_dt
            #~ t_end_fs = t_end_fit_dt
    
        # The number of timesteps I will calculate:
        nr_of_t = (t_end_dt - t_start_dt) / stepsize_t # In principle I could start at t_start_dt, but for the
        # the plotting I will calculate also the part between 0 adn t_start.
    
        # Checking if I have to partition the trajectory into blocks (By default just 1 block)
        block_length_fs = kwargs.pop('block_length_fs', None)
        block_length_dt = kwargs.pop('block_length_dt', None)
    
        nr_of_blocks = kwargs.pop('nr_of_blocks', None)
    
    
        # Asking whether to calculate COM diffusion
        do_com = kwargs.pop('do_com', False)
    
    
        if kwargs:
            raise Exception("Uncrecognized keywords: {}".format(kwargs.keys()))
        if block_length_fs and block_length_dt :
           raise Exception("You cannot set both 'block_length_fs' and 'block_length_dt'")
        elif ( block_length_dt or block_length_fs ) and nr_of_blocks:
            raise Exception("You cannot set both 'a block length' and 'nr_of_blocks'") #
            # Maybe allow this in the future, at the cost of not the whole trajectory being used:
        elif isinstance(block_length_fs, float):
            block_length_dt  = int(block_length_fs / timestep_fs)
        elif isinstance(block_length_dt, int):
            block_length_fs  = block_length_dt * timestep_fs
        elif isinstance(nr_of_blocks, int):
            nr_of_blocks = nr_of_blocks
        else:
            nr_of_blocks=1
        return (species_of_interest, nr_of_blocks, t_start_dt, t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
            stepsize_t, stepsize_tau, block_length_dt, do_com)

    def get_msd(self, decomposed=False, **kwargs):
        """
        Calculates the mean square discplacement (MSD),
    
        #.  Calculate the MSD for each block
        #.  Calculate the mean and the standard deviation of the slope
        #.  Calculate the conductivity, including error propagation.
    
        :param list species_of_interest:
            The species of interest for which to calculate the MSD, for example ["O", "H"]
        :param int stepsize_t:
            This tells me whether I will have a stepsize larger than 1 (the default)
            when looping over the trajectory.
        """
        from samos.lib.mdutils import calculate_msd_specific_atoms, calculate_msd_specific_atoms_decompose_d, get_com_positions
        try:
            timestep_fs = self._timestep_fs
            atoms = self._atoms
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                "\n\n\n"
                "Please use the set_trajectories method to set trajectories"
                "\n{}\n".format(e)
            )


        (species_of_interest, nr_of_blocks, t_start_dt, t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
            stepsize_t, stepsize_tau, block_length_dt, do_com) = self._get_running_params(timestep_fs, **kwargs)

        msd = TimeSeries()

        results_dict = {atomic_species: {}
                for atomic_species
                in species_of_interest
            }
        msd_all_species = []
        #self.msd_averaged = []
        # Setting params for calculation of MSD and conductivity
        # Future: Maybe allow for element specific parameter settings?
    
        for atomic_species in species_of_interest:
            msd_this_species = []

            if decomposed:
                slopes_intercepts = np.empty((len(trajectories), nr_of_blocks, 3,3, 2))
            else:
                slopes_intercepts = np.empty((len(trajectories), nr_of_blocks, 2))
    
            for itraj, trajectory in enumerate(trajectories):

                positions = trajectory.get_positions()

                if do_com:
                    # I replace the array positions with the COM!
                    masses = self._atoms.get_masses() # Getting the masses for recentering
                    factors = [1]*len(masses)
                    #~ nstep, nat, _ = positions.shape
                    positions = get_com_positions(positions, masses, factors)
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_incides_of_species(atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_incides_of_species(atomic_species, start=1)
                    prefactor = 1

                nstep, nat, _= positions.shape
                block_length_dt = (nstep - t_end_dt)  / nr_of_blocks


                nat_of_interest = len(indices_of_interest)

                if self._verbosity > 0:
                    print(
                            '\n    ! Calculating MSD for atomic species {} in trajectory {}\n'
                            '      Structure contains {} atoms of type {}\n'
                            '      I will calculate {} block(s)'
                            ''.format(atomic_species, itraj, nat_of_interest, atomic_species, nr_of_blocks)
                        )
                if decomposed:
                    msd_this_species_this_traj = prefactor*calculate_msd_specific_atoms_decompose_d(
                            positions, indices_of_interest, stepsize_t, stepsize_tau, block_length_dt,
                            nr_of_blocks, nr_of_t, nstep, nat, nat_of_interest)
                else:
                    msd_this_species_this_traj = prefactor*calculate_msd_specific_atoms(
                            positions, indices_of_interest, stepsize_t, stepsize_tau, block_length_dt,
                            nr_of_blocks, nr_of_t, nstep, nat, nat_of_interest)

                if self._verbosity > 0:
                    print('      Done\n')

                range_for_t = timestep_fs*stepsize_t*np.arange(t_start_fit_dt/stepsize_t, t_end_fit_dt/stepsize_t)
    
                for iblock, block in enumerate(msd_this_species_this_traj):
                    if decomposed:
                        for ipol in range(3):
                            for jpol in range(3):
                                slope, intercept, _, _, _ = linregress(range_for_t, block[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t,ipol, jpol])
                                slopes_intercepts[itraj, iblock, ipol, jpol, 0] = slope
                                slopes_intercepts[itraj, iblock, ipol, jpol, 1] = intercept
                        msd.set_array('msd_decompsed_{}_{}_{}'.format(atomic_species, itraj, iblock), block)
                    else:
                        slope, intercept, _, _, _ = linregress(range_for_t, block[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])
                        slopes_intercepts[itraj, iblock, 0] = slope
                        slopes_intercepts[itraj, iblock, 1] = intercept
                        msd.set_array('msd_isotropic_{}_{}_{}'.format(atomic_species, itraj, iblock), block)
                    msd_this_species.append(block)

            # Calculating the average sem/std for each point in time:
            msd_mean = np.mean(msd_this_species, axis=0)
            msd_std = np.std(msd_this_species, axis=0)
            msd_sem = msd_std / np.sqrt(len(msd_this_species) - 1)

            msd.set_array('msd_{}_{}_mean'.format('decomposed' if decomposed else 'isotropic', atomic_species), msd_mean)
            msd.set_array('msd_{}_{}_std'.format('decomposed' if decomposed else 'isotropic',atomic_species), msd_std)
            msd.set_array('msd_{}_{}_sem'.format('decomposed' if decomposed else 'isotropic',atomic_species), msd_sem)

            if decomposed:
                results_dict[atomic_species].update({
                    'slope_msd_mean':np.mean(slopes_intercepts[:,:,:,:,0], axis=(0,1)),
                    'slope_msd_std':np.std(slopes_intercepts[:,:,:,:,0], axis=(0,1)),
                    'slopes_intercepts':slopes_intercepts.tolist()})
                results_dict[atomic_species]['slope_msd_sem'] = results_dict[atomic_species]['slope_msd_std'] / np.sqrt(len(trajectories)*nr_of_blocks-1)
            else:
                results_dict[atomic_species].update({
                    'slope_msd_mean':np.mean(slopes_intercepts[:,:,0]),
                    'slope_msd_std':np.std(slopes_intercepts[:,:,0]),
                    'slopes_intercepts':slopes_intercepts.tolist()})
                results_dict[atomic_species]['slope_msd_sem'] = results_dict[atomic_species]['slope_msd_std'] / np.sqrt(len(trajectories)*nr_of_blocks-1)

            results_dict[atomic_species]['diffusion_mean_cm2_s'] =  1e-1 / 6.* results_dict[atomic_species]['slope_msd_mean']
            results_dict[atomic_species]['diffusion_std_cm2_s']  =  1e-1 / 6.* results_dict[atomic_species]['slope_msd_std']
            results_dict[atomic_species]['diffusion_sem_cm2_s']  =  1e-1 / 6.* results_dict[atomic_species]['slope_msd_sem']


            if decomposed:
                # I need to transform to lists, numpy are not json serializable:
                for k in ('slope_msd_mean', 'slope_msd_std', 'slope_msd_sem',
                    'diffusion_mean_cm2_s', 'diffusion_std_cm2_s','diffusion_sem_cm2_s'):
                    results_dict[atomic_species][k] = results_dict[atomic_species][k].tolist()
            if self._verbosity > 1:
                print('      Done, these are the results for {}:'.format(atomic_species))
                for key, val in results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print(  '          {:<20} {}'.format(key,  val))

        results_dict.update({
            't_start_fit_dt'        :   t_start_fit_dt,
            't_end_fit_dt'          :   t_end_fit_dt,
            't_start_dt'            :   t_start_dt,
            't_end_dt'              :   t_end_dt,
            'nr_of_blocks'          :   nr_of_blocks,
            'nr_of_trajectories'    :   len(trajectories),
            'block_length_dt'       :   block_length_dt,
            'stepsize_t'            :   stepsize_t,
            'species_of_interest'   :   species_of_interest,
            'timestep_fs'           :   timestep_fs,
            'nr_of_t'               :   nr_of_t,
            'decomposed'            :   decomposed,
        })
        for k,v in results_dict.items():
            msd.set_attr(k,v)
        
        return msd


    def get_vaf(self, integration='trapezoid', **kwargs):

        from samos.lib.mdutils import calculate_vaf_specific_atoms, get_com_velocities
        try:
            timestep_fs = self._timestep_fs
            atoms = self._atoms
            trajectories = self._trajectories
        except AttributeError as e:
            raise Exception(
                "\n\n\n"
                "Please use the set_trajectories method to set trajectories"
                "\n{}\n".format(e)
            )


        (species_of_interest, nr_of_blocks, t_start_dt, t_end_dt, t_start_fit_dt, t_end_fit_dt, nr_of_t,
            stepsize_t, stepsize_tau, block_length_dt, do_com) = self._get_running_params(timestep_fs, **kwargs)

        vaf = TimeSeries()


        results_dict = dict()
        vaf_all_species = []

        #~ self.D_from_vaf_averaged = []
        for atomic_species in species_of_interest:



            #~ nat_of_interest = len(indices_of_interest)
            vaf_this_species = []
            slopes_n_intercepts = np.empty((len(trajectories), nr_of_blocks, 2))
            #~ means_of_integral = []
            for itraj, trajectory in enumerate(trajectories):
                if do_com:
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_incides_of_species(atomic_species))
                else:
                    indices_of_interest = trajectory.get_incides_of_species(atomic_species, start=1)
                    prefactor = 1

                velocities = trajectory.get_velocities()

                total_steps = len(vel_array)
                total_time = total_steps*self.timestep_in_fs
                if nr_of_blocks:
                    block_length_dt = (total_steps - t_end_vaf_dt)  / nr_of_blocks
                    block_length_fs = block_length_dt * timestep_in_fs
                else:
                    nr_of_blocks   = ( total_steps - t_end_vaf_dt ) / block_length_dt

                if do_com:
                    factors = self._get_factors(atomic_species)
                    vel_array = get_com_velocities(vel_array, self._masses, factors, nstep, nat)
                    nstep, nat, _= vel_array.shape

                if self._verbosity > 0:
                    log.write(
                            '\n    ! Calculating VAF for atomic species {}\n'
                            '      Structure contains {} atoms of type {}\n'
                            '      Max time (fs)     = {}\n'
                            '      Max time (dt)     = {}\n'
                            '      Stepsize for t    = {}\n'
                            '      stepsize for tau  = {}\n'
                            '      nr of timesteps   = {}\n'
                            '      nr of blocks      = {}\n'
                            '      Block length (fs) = {}\n'
                            '      Block length (dt) = {}\n'
                            '      Calculating VAF with fortran subroutine fortvaf.calculate_vaf_specific_atoms\n'
                            ''.format(
                                atomic_species, nat_of_interest, atomic_species,
                                t_end_vaf_fs, t_end_vaf_dt, stepsize_t, stepsize_tau, nr_of_t,
                                nr_of_blocks, block_length_fs, block_length_dt
                            )
                        )

                res = calculate_vaf_specific_atoms(
                    vel_array,
                    indices_of_interest,
                    stepsize_t,
                    stepsize_tau,
                    nr_of_t,
                    nr_of_blocks,
                    block_length_dt,
                    timestep_in_fs*stepsize_t, # Integrating with timestep*0.1 / 3.
                    integration,
                    nstep,
                    nat,
                    nat_of_interest
                )

                range_for_t = timestep_in_fs*stepsize_t*np.arange(t_start_fit_dt/stepsize_t, t_end_fit_dt/stepsize_t)

                #~ for iblock, block in enumerate(msd_this_species_this_traj, start=1):


                for block_vaf, integrated_vaf in zip(*res):
                    D =  0.1 / 3. *prefactor* integrated_vaf  # transforming A^2/fs -> cm^2 /s, dividing by three to get D
                    vaf_this_species.append((block_vaf, D))
                    slope, intercept, _, _, _ = linregress(
                            range_for_t,
                            D[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t]
                        )
                    slopes_n_intercepts.append((slope, intercept))
                    means_of_integral.append(D[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t].mean())

            D_vaf = zip(*vaf_this_species)[1]
            D_mean = np.mean(D_vaf, axis=0)
            D_std  = np.std(D_vaf, axis=0)
            D_sem  = D_std / np.sqrt(len(D_vaf) - 1)

            D_upper_sem = D_mean + D_sem
            D_lower_sem = D_mean - D_sem
            D_upper_std = D_mean + D_std
            D_lower_std = D_mean - D_std

            upper_bound_sem = D_upper_sem[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t].mean()
            lower_bound_sem = D_lower_sem[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t].mean()
            upper_bound_std = D_upper_std[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t].mean()
            lower_bound_std = D_lower_std[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t].mean()
            self.D_from_vaf_averaged.append((
                    D_mean, D_sem, D_std,
                    0.5*(upper_bound_sem + lower_bound_sem),
                    0.5*(upper_bound_sem - lower_bound_sem),
                    0.5*(upper_bound_std + lower_bound_std),
                    0.5*(upper_bound_std - lower_bound_std)
                ))
            self.vaf_results_dict[atomic_species] = dict(
                    slopes_n_intercepts=slopes_n_intercepts,
                    means_of_integral=means_of_integral,
                    means_of_means=np.mean(means_of_integral),
                    std_of_means=np.std(means_of_integral),
                    sem_of_means=standard_error_of_mean(means_of_integral)
                )
            self.vaf_all_species.append(vaf_this_species)

        results_dict.update({
            't_start_fit_dt'        :   t_start_fit_dt,
            't_end_fit_dt'          :   t_end_fit_dt,
            't_start_dt'            :   t_start_dt,
            't_end_dt'              :   t_end_dt,
            'nr_of_blocks'          :   nr_of_blocks,
            'nr_of_trajectories'    :   len(trajectories),
            'block_length_dt'       :   block_length_dt,
            'stepsize_t'            :   stepsize_t,
            'species_of_interest'   :   species_of_interest,
            'timestep_fs'           :   timestep_fs,
            'nr_of_t'               :   nr_of_t,})

        return self.vaf_results_dict, np.array(self.vaf_all_species)


