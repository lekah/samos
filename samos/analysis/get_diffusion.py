
from ase import Atoms
import numpy as np
from scipy.stats import linregress
from scipy.stats import sem as standard_error_of_mean
from samos.trajectory import check_trajectory_compatibility, Trajectory
from samos.utils.attributed_array import AttributedArray


class TimeSeries(AttributedArray):
    pass




class DiffusionAnalyzer(object):
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

        self._atoms, self._timestep_fs = check_trajectory_compatibility(trajectories)
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

    def _get_running_params(**kwargs):
        """
        Utility function to get a number of parameters
        """
        pass # TODO
    def get_msd_isotropic(self, **kwargs):
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
        from samos.lib.mdutils import calculate_msd_specific_atoms, get_com_positions
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
    
    
        t_start_msd_fs = kwargs.pop('t_start_msd_fs', None)
        t_start_msd_dt = kwargs.pop('t_start_msd_dt', 0)
    
        if isinstance(t_start_msd_fs, float):
            t_start_msd_dt = int(t_start_msd_fs / timestep_fs)
        elif isinstance(t_start_msd_dt, int):
            t_start_msd_fs = t_start_msd_dt*timestep_fs
        else:
            raise Exception(
                "\n\n\n"
                "Set the time that you consider the\n"
                "VAF to have converged as a float with\n"
                "keyword t_start_msd_fs or t_start_msd_dt"
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
    
    
        t_end_msd_fs = kwargs.pop('t_end_msd_fs', None)
        t_end_msd_dt = kwargs.pop('t_end_msd_dt', None)
    
        if t_end_msd_fs and t_end_msd_dt:
            raise Exception("You cannot set both 't_end_msd_fs' and 't_end_msd_dt'")
        if isinstance(t_end_msd_fs, float):
            t_end_msd_dt = int(t_end_msd_fs / timestep_fs)
        elif isinstance(t_end_msd_dt, int):
            t_end_msd_fs = timestep_fs * t_end_msd_dt
        else:
            t_end_msd_dt = t_end_fit_dt
            #~ t_end_msd_fs = t_end_fit_dt
    
        # The number of timesteps I will calculate:
        nr_of_t = (t_end_msd_dt - t_start_msd_dt) / stepsize_t # In principle I could start at t_start_msd_dt, but for the
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


        msd = TimeSeries()

        msd_results_dict = {atomic_species: {}
                for atomic_species
                in species_of_interest
            }
        msd_isotrop_all_species = []
        #self.msd_averaged = []
        # Setting params for calculation of MSD and conductivity
        # Future: Maybe allow for element specific parameter settings?
    
        for atomic_species in species_of_interest:
            msd_isotrop_this_species = []
            slopes_intercepts = np.empty((len(trajectories), nr_of_blocks, 2))
            slopes = []
    
            for itraj, trajectory in enumerate(trajectories):
                if do_com:
                    indices_of_interest = [1]
                    prefactor = len(trajectory.get_incides_of_species(atomic_species, start=0))
                else:
                    indices_of_interest = trajectory.get_incides_of_species(atomic_species, start=1)
                    prefactor = 1
                positions = trajectory.get_positions()
        
                nat_of_interest = len(indices_of_interest)
                nstep, nat, _= positions.shape
                total_time = nstep*timestep_fs
    
                if nr_of_blocks:
                    block_length_dt = (nstep -t_end_msd_dt)  / nr_of_blocks
                    block_length_fs = block_length_dt*timestep_fs
                else:
                    nr_of_blocks   = (nstep - t_end_msd_dt) / block_length_dt
    
                if do_com:
                    factors = self._get_factors(atomic_species)
                    trajectory = get_com_positions(positions, self._atoms.get_masses(), factors, nstep, nat)
                    nstep, nat, _= trajectory.shape
    
                if self._verbosity > 0:
                    print(
                        '\n    ! Calculating MSD for atomic species {} in trajectory {}\n'
                        '      Structure contains {} atoms of type {}\n'
                        '      Assuming convergence of VAF at {}\n'
                        '      Block length for calculation is {} fs ({} dt)\n'
                        '      I will calculate {} block(s)'
                        ''.format(
                            atomic_species, itraj, nat_of_interest, atomic_species,
                            t_start_msd_fs, block_length_fs, block_length_dt, nr_of_blocks
                        )
                    )
    
                msd_isotrop_this_species_this_traj = prefactor*calculate_msd_specific_atoms(
                            positions,
                            indices_of_interest,
                            stepsize_t,
                            stepsize_tau,
                            block_length_dt,
                            nr_of_blocks,
                            nr_of_t,
                            nstep,
                            nat,
                            nat_of_interest
                        )
    
                if self._verbosity > 0:
                    print('      Done\n')
                range_for_t = timestep_fs*stepsize_t*np.arange(t_start_fit_dt/stepsize_t, t_end_fit_dt/stepsize_t)
    
                for iblock, block in enumerate(msd_isotrop_this_species_this_traj):
                    slope, intercept, _, _, _ = linregress(range_for_t, block[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])
                    slopes_intercepts[itraj, iblock, 0] = slope
                    slopes.append(slope)
                    slopes_intercepts[itraj, iblock, 1] = intercept
                    msd.set_array('msd_isotropic_{}_{}_{}'.format(atomic_species, itraj, iblock), block)
                    msd_isotrop_this_species.append(block)

    
    
            # Calculating the average sem/std for each point in time:
            msd_mean = np.mean(msd_isotrop_this_species, axis=0)
            msd_std = np.std(msd_isotrop_this_species, axis=0)
            msd_sem = msd_std / np.sqrt(len(msd_isotrop_this_species) - 1)
            msd.set_array('msd_isotropic_{}_mean'.format(atomic_species), msd_mean)
            msd.set_array('msd_isotropic_{}_std'.format(atomic_species), msd_std)
            msd.set_array('msd_isotropic_{}_sem'.format(atomic_species), msd_sem)

            #~ upper_bound_sem_slope,_,_,_,_ = linregress(range_for_t, (msd_mean+msd_sem)[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])
            #~ lower_bound_sem_slope,_,_,_,_ = linregress(range_for_t, (msd_mean-msd_sem)[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])
            #~ upper_bound_std_slope,_,_,_,_ = linregress(range_for_t, (msd_mean+msd_std)[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])
            #~ lower_bound_std_slope,_,_,_,_ = linregress(range_for_t, (msd_mean-msd_std)[t_start_fit_dt/stepsize_t:t_end_fit_dt/stepsize_t])

            # The factor 1e-5 comes from the conversion of A**2 / fs -> m**2/ s
            #~ self.msd_averaged.append((
                    #~ msd_mean, msd_sem, msd_std,
                    #~ 1e-5 / 6.* 0.5*(upper_bound_sem_slope+lower_bound_sem_slope), # For convenience I put here the mean and sem that I calculated for the diffusion
                    #~ 1e-5 / 6.* 0.5*(upper_bound_sem_slope-lower_bound_sem_slope),
                    #~ 1e-5 / 6.* 0.5*(upper_bound_std_slope+lower_bound_std_slope), # For convenience I put here the mean and sem that I calculated for the diffusion
                    #~ 1e-5 / 6.* 0.5*(upper_bound_std_slope-lower_bound_std_slope),
                #~ ))
    
            msd_results_dict[atomic_species].update({
                'slope_msd_mean':np.mean(slopes),
                'slope_msd_std':np.std(slopes),
                'slope_msd_sem':standard_error_of_mean(slopes),
                'slopes_intercepts':slopes_intercepts.tolist(),
                #~ 'labels':labels,
            })
            diffusion_mean_SI = 1e-5 / 6.* msd_results_dict[atomic_species]['slope_msd_mean']
            diffusion_std_SI = 1e-5 / 6.* msd_results_dict[atomic_species]['slope_msd_std']
            diffusion_sem_SI = 1e-5 / 6.* msd_results_dict[atomic_species]['slope_msd_sem']

            msd_results_dict[atomic_species]['diffusion_mean_cm2_s'] =  1e4*diffusion_mean_SI
            msd_results_dict[atomic_species]['diffusion_std_cm2_s']  =  1e4*diffusion_std_SI
            msd_results_dict[atomic_species]['diffusion_sem_cm2_s']  =  1e4*diffusion_sem_SI
    
    
            if self._verbosity > 1:
                print('      Done, these are the results for {}:'.format(atomic_species))
                for key, val in msd_results_dict[atomic_species].items():
                    if not isinstance(val, (tuple, list, dict)):
                        print(  '          {:<20} {}'.format(key,  val))

            #~ self.msd_isotrop_all_species.append(msd_isotrop_this_species)
    
        msd_results_dict.update({
            't_start_fit_dt'        :   t_start_fit_dt,
            't_end_fit_dt'          :   t_end_fit_dt,
            't_start_msd_dt'        :   t_start_msd_dt,
            't_end_msd_dt'          :   t_end_msd_dt,
            'nr_of_blocks'          :   nr_of_blocks,
            'nr_of_trajectories'    :   len(trajectories),
            'block_length_dt'       :   block_length_dt,
            'stepsize_t'            :   stepsize_t,
            'species_of_interest'   :   species_of_interest,
            'timestep_fs'        :   timestep_fs,
            'nr_of_t'               :   nr_of_t,
        })
        for k,v in msd_results_dict.items():
            msd.set_attr(k,v)
        
        return msd
    
    
    
    def get_msd_decomposed(self, **kwargs):
        """
        Calculates the mean square discplacement (MSD),
        Using the fortran module in lib.difflib.
    
        .. figure:: /images/fort_python_msd.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center
    
            Comparison between results achieved with the fortran implementation and python
            showing perfect agreement with respect to each other.
    
        Velocity autocorrelation function (VAF) and conductivity from the trajectory
        passed in :func:`set_trajectory`
        and the structure set in :func:`set_structure`.
        This procedes in several steps:
    
        #.  Calculate the MSD for each block, calculate the slope from the VAF0 estimate to block end
        #.  Calculate the mean and the standard deviation of the slope
        #.  Calculate the conductivity, including error propagation.
    
        :param list species_of_interest:
            The species of interest for which to calculate the MSD, for example ["O", "H"]
        :param int stepsize:
            This tells me whether I will have a stepsize larger than 1 (the default)
            when looping over the trajectory. I.e. when set to 10, will consider every
            10th timestep of the trajectory
        """
        from difflib import calculate_msd_specific_atoms_decompose_d
        try:
    
            #~ trajectory = self.trajectory
            timestep_fs = self.timestep_fs
    
        except AttributeError as e:
            raise Exception(
                "\n\n\n"
                "Please use the set_trajectories method to set trajectories"
                "\n{}\n".format(e)
            )
        try:
            structure = self.structure
            self._chemical_symbols
            species_of_interest = self.species_of_interest
        except AttributeError:
            raise Exception(
                "\n\n\n"
                "Please use the set_structure method to set structure"
                "\n\n\n"
            )
        log = self.log # set in __init__
    
    
        # Getting parameters for calculations
    
        #~ recenter = kwargs.pop('recenter', False)
    
        # Here I give the possibility to override species of interest just for
        # this calculation
        species_of_interest = kwargs.pop("species_of_interest", species_of_interest)
        stepsize  = kwargs.pop('stepsize', 1)
        stepsize_tau  = kwargs.pop('stepsize_tau', 1)
    
        t_start_fit_fs = kwargs.pop('t_start_fit_fs', None)
        t_start_fit_dt = kwargs.pop('t_start_fit_dt', None)
    
        if t_start_fit_fs and t_start_fit_dt:
            raise Exception("You cannot set both 't_start_fit_fs' and 't_start_fit_dt'")
        elif isinstance(t_start_fit_fs, float):
            t_start_fit_dt  = int(t_start_fit_fs / timestep_fs)
        elif isinstance(t_start_fit_dt, int):
            block_length_fs  = t_start_fit_dt * timestep_fs
        else:
            raise Exception(
                "Please set the time to start the fit with keyword t_start_fit_fs or t_start_fit_dt"
            )
    
        only_means = kwargs.pop('only_means', False)
    
        t_start_msd_fs = kwargs.pop('t_start_msd_fs', None)
        t_start_msd_dt = kwargs.pop('t_start_msd_dt', 0)
    
        if isinstance(t_start_msd_fs, float):
            t_start_msd_dt = int(t_start_msd_fs / timestep_fs)
        elif isinstance(t_start_msd_dt, int):
            t_start_msd_fs = t_start_msd_dt*timestep_fs
        else:
            raise Exception(
                "\n\n\n"
                "Set the time that you consider the\n"
                "VAF to have converged as a float with\n"
                "keyword t_start_msd_fs or t_start_msd_dt"
                "\n\n\n"
            )
    
    
    
        t_end_fit_fs = kwargs.pop('t_end_fit_fs', None)
        t_end_fit_dt = kwargs.pop('t_end_fit_dt', None)
    
        if t_end_fit_fs and t_end_fit_dt:
            raise Exception("You cannot set both 't_end_fit_fs' and 't_end_fit_dt'")
        elif isinstance(t_end_fit_fs, float):
            t_end_fit_dt  = int(t_end_fit_fs / timestep_fs)
        elif isinstance(t_end_fit_dt, int):
            block_length_fs  = t_end_fit_dt * timestep_fs
        else:
            raise Exception(
                "Please set the time to start the fit with keyword t_end_fit_fs or t_end_fit_dt"
            )
    
    
        t_end_msd_fs = kwargs.pop('t_end_msd_fs', None)
        t_end_msd_dt = kwargs.pop('t_end_msd_dt', None)
    
        if t_end_msd_fs and t_end_msd_dt:
            raise Exception("You cannot set both 't_end_msd_fs' and 't_end_msd_dt'")
        if isinstance(t_end_msd_fs, float):
            t_end_msd_dt = int(t_end_msd_fs / timestep_fs)
        elif isinstance(t_end_msd_dt, int):
            t_end_msd_fs = timestep_fs * t_end_msd_dt
        else:
            t_end_msd_dt = t_end_fit_dt
            #~ t_end_msd_fs = t_end_fit_dt
    
        # The number of timesteps I will calculate:
        nr_of_t = (t_end_msd_dt - t_start_msd_dt) / stepsize # In principle I could start at t_start_msd_dt, but for the
        # the plotting I will calculate also the part between 0 adn t_start.
    
        # Checking if I have to partition the trajectory into blocks (By default just 1 block)
        block_length_fs = kwargs.pop('block_length_fs', None)
        block_length_dt = kwargs.pop('block_length_dt', None)
    
        nr_of_blocks = kwargs.pop('nr_of_blocks', None)
    
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
            #~ raise Exception(
                #~ "Please set the block length in femtoseconds (kw block_length_fs or block_length_dt) or define the\n"
                #~ "number of blocks (kw nr_of_blocks)"
            #~ )
    
    
        assert nr_of_blocks > 0, 'Number of blocks is not positive'
    
    
        # Make this not overwrite results within the loop:
        self.msd_results_dict_decomposed = {atomic_species: {}
                for atomic_species
                in species_of_interest
            }
    
        self.msd_decomposed = [] # That's where I'm storing the MSD decomposed as a function of time.
        # Each MSD is a 3x3xT matrix, T is time! 3x3 from the 3 coordinates
    
    
    
        for ityp, atomic_species in enumerate(species_of_interest):
    
            self.msd_results_dict_decomposed[atomic_species]['slopes'] = []
            self.msd_results_dict_decomposed[atomic_species]['intercepts'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['blocks'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['diffusions'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['diffusion_mean'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['diffusion_std'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['diffusion_sem'] = []
            #~ self.msd_results_dict_decomposed[atomic_species]['labels'] = []
    
            indices_of_interest = self._get_indices_of_interest(atomic_species, start=1)
    
            nat_of_interest = len(indices_of_interest)
    
            msd_decomposed_this_species = []
    
            ntraj = len(self._positions)
    
            # Here I'm storing the slopes of each block over all trajectories:
            slopes = []
    
            for itraj, trajectory in enumerate(self._positions):
    
                nstep, nat, _= trajectory.shape
                total_time = nstep*self.timestep_fs
                if nr_of_blocks:
                    block_length_dt = (nstep -t_end_msd_dt)  / nr_of_blocks
                    block_length_fs = block_length_dt*timestep_fs
                else:
                    nr_of_blocks   = (nstep -max_time_in_dt) / block_length_dt
    
    
                #~ labels_dec_m = np.chararray((nr_of_blocks, 3,3), itemsize=25)
                slope_dec_m = np.empty((nr_of_blocks, 3,3))
                intercept_dec_m = np.empty((nr_of_blocks, 3,3))
                #~ diffusion_dec_m = np.empty((nr_of_blocks, 3,3))
                #~ diffusion_dec_mean = np.empty((3,3))
                #~ diffusion_dec_std = np.empty((3,3))
                #~ diffusion_dec_sem = np.empty((3,3))
                #~ block_dec_m = np.empty((nr_of_blocks, nr_of_t, 3,3))
    
    
                log.write(
                    '\n    ! Calculating MSD for atomic species {} in trajectory {}\n'
                    '      Structure contains {} atoms of type {}\n'
                    '      Assuming convergence of VAF at {}\n'
                    '      Block length for calculation is {} fs ({} dt)\n'
                    '      I will calculate {} block(s)\n'
                    ''.format(
                        atomic_species, itraj, nat_of_interest, atomic_species,
                        t_start_msd_fs, block_length_fs, block_length_dt, nr_of_blocks
                    )
                )
    
                msd_decomposed_this_species_this_traj = calculate_msd_specific_atoms_decompose_d(
                        trajectory,
                        indices_of_interest,
                        stepsize,
                        stepsize_tau,
                        block_length_dt,
                        nr_of_blocks,
                        nr_of_t,
                        nstep,
                        nat,
                        nat_of_interest
                    )
                log.write('      Done\n')
                range_for_t = timestep_fs*stepsize*np.arange(t_start_fit_dt/stepsize, t_end_fit_dt/stepsize)
    
                # Storing the MSD decomposed as a function of time here:
                msd_decomposed_this_species.append(msd_decomposed_this_species_this_traj)
    
    
                for iblock, block in enumerate(msd_decomposed_this_species_this_traj):
                    for ipol in range(3):
                        for jpol in range(3):
                            slope, intercept, _, _, _ = linregress(range_for_t, block[t_start_fit_dt/stepsize:t_end_fit_dt/stepsize,ipol, jpol])
                            slope_dec_m[iblock, ipol, jpol] = slope
                            intercept_dec_m[iblock, ipol, jpol] = intercept
    
                self.msd_results_dict_decomposed[atomic_species]['slopes'].append(slope_dec_m.tolist())
                self.msd_results_dict_decomposed[atomic_species]['intercepts'].append(intercept_dec_m.tolist())
    
    
            # Here I'm calculating the mean for all blocks and trajectories
            all_slopes = np.array(self.msd_results_dict_decomposed[atomic_species]['slopes'])
            if only_means:
                self.msd_results_dict_decomposed[atomic_species].pop('slopes')
                self.msd_results_dict_decomposed[atomic_species].pop('intercepts')
            #~ print 0.5e-1*all_slopes.mean(axis=0)[0]
            
            # 1e-1 to cgs units
            #~ print all_slopes[:,0,1,2].mean()
            
            slopes_dict = {
                'slope_msd_mean':all_slopes.mean(axis=0)[0],
                'slope_msd_std':all_slopes.std(axis=0)[0],
                'slope_msd_sem':standard_error_of_mean(all_slopes, axis=0)[0],
            }
            #~ print slopes_dict['slope_msd_std']
            #~ print slopes_dict['slope_msd_sem']
            diffusion_dict = dict(
                diffusion_mean_cm2_s = 0.5e-1*slopes_dict['slope_msd_mean'],
                diffusion_std_cm2_s = 0.5e-1*slopes_dict['slope_msd_std'],
                diffusion_sem_cm2_s = 0.5e-1*slopes_dict['slope_msd_sem'],
            )
    
            for d in (slopes_dict, diffusion_dict):
                self.msd_results_dict_decomposed[atomic_species].update({k:v.tolist() for k,v in d.items()})
    
    
            self.msd_decomposed.append(msd_decomposed_this_species)
        self.msd_results_dict_decomposed.update({
            't_start_fit_dt'        :   t_start_fit_dt,
            't_end_fit_dt'          :   t_end_fit_dt,
            't_start_msd_dt'        :   t_start_msd_dt,
            't_end_msd_dt'          :   t_end_msd_dt,
            'nr_of_blocks'          :   nr_of_blocks,
            'nr_of_trajectories'    :   nr_of_trajectories,
            'block_length_dt'       :   block_length_dt,
            'stepsize'              :   stepsize,
            'species_of_interest'   :   species_of_interest,
            'timestep_fs'        :   timestep_fs,
            'nr_of_t'               :   nr_of_t,
            'ntraj'                 :   itraj+1,
        })
        return self.msd_results_dict_decomposed, np.array(self.msd_decomposed)
    
    
    
