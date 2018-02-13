import numpy as np


class Trajectory(object):
    """
    Class defining our trajectories.
    A trajectory is a sequence of time-ordered points in phase space.
    The internal units of a trajectory:
    *   Femtoseconds for times
    *   Angstrom for coordinates
    *   eV for energies
    *   Masses and cells are set via the _atoms member, an ase.Atoms instance and units as in ase are used. 
    """
    _POSITIONS_KEY = 'positions'
    _VELOCITIES_KEY = 'velocities'
    _FORCES_KEY = 'forces'
    def __init__(self, **kwargs):
        """
        Instantiating a trajectory class.
        Optional keyword-arguments are everything with a set-method.
        """
        self._atoms = None
        self._arrays = {}
        self._nstep = None
        self._timestep_fs = None
        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

    def set_timestep(self, timestep_fs):
        """
        :param timestep: expects value of the timestep in femtoseconds
        """
        self._timestep_fs = float(timestep_fs)


    def get_timestep(self):
        """
        raises: ValueError if timestep has not been set
        """
        if isinstance(self._timestep_fs, float):
            return self._timestep_fs
        else:
            raise ValueError("Timestep has not been set")

    def set_atoms(self, atoms):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            raise ValueError("You have to pass an instance of ase.Atoms")
        self._atoms = atoms

    def get_atoms(self):
        if self._atoms:
            return self._atoms
        else:
            raise ValueError("Atoms have not been set")
    @property
    def atoms(self):
        return self.get_atoms()
    @property
    def cell(self):
        return self.atoms.cell

    def set_array(self, name, array, check_existing=False, check_nstep=True, check_nat=True):
        """
        Method to set an array with a name to reference it.
        :param str name: A name to reference that array
        :param array: A valid numpy array or an object that can be converted with a call to numpy.array
        :param bool check_existing:
            Check for an array of that name existing, and raise if it exists.
            Defaults to False.
        :param book check_nstep:
            Check if the number of steps, which is the first dimension of the array, is commensurate
            with other arrays stored. Defaults to True
        :param bool check_nat:
            If the array is of rank 3 or higher, the second dimension is interpreted as the number of atoms.
            If this flag is True, I will check for arrays with rank 3 or higher. Defaults  to True.
            Requires that the atoms have been set
        """
        # First, I call np.array to ensure it's a valid array
        array = np.array(array)
        if not isinstance(name, basestring):
            raise TypeError("Name has to be a string")
        if check_existing:
            if name in self._arrays.keys():
                raise ValueError("Name {} already exists".formamt(name))
        if check_nstep:
            for other_name, other_array in self._arrays.items():
                assert array.shape[0] == other_array.shape[0], (
                    'Number of steps in this array is not compliant with array {}'.format(other_name))
        if check_nat and len(array.shape) > 2:
            if array.shape[1] != len(self.atoms):
                raise ValueError("Second dimension of array does not match the number of atoms")
        self._arrays[name] = array


    def get_array(self, name):
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError("An array with that name ( {} ) has not been set.".format(name))

    def get_incides_of_species(self, species, start=0):
        """
        Convenience function to get all indices of a species.
        :param species:
            The identifier of a species. If this is a string, I assume the chemical symbol (abbreviation).
            I.e. Li for lithium, H for hydrogen.
            If it's an integer, I assume the atomic number.
        :param int start:
            The start of indexing, defaults to 0. For fortran indexing, set to 1.
        :return: A numpy array of indices
        """
        assert isinstance(start, int), "Start is not an integer"
        if isinstance(species, str):
            array_to_index = self.atoms.get_chemical_symbols()
        elif isinstance(species, int):
            array_to_index = self.atoms.get_atomic_numbers()
        else:
            raise TypeError("species  has  to be an integer or a string, I got {}".format(type(species)))

        return np.array([i for i, s in enumerate(array_to_search, start=start) if s==species])


    @property
    def nstep(self):
        """
        :returns: The number of trajectory steps
        :raises: ValueError if no unique number of steps can be determined.
        """
        nstep_set = set([array.shape[0] for array in self._arrays.values()])
        if len(nstep_set) == 0:
            raise ValueError("No arrays have been set, yet")
        elif len(nstep_set) > 1:
            raise ValueError("Incommensurate arrays")
        else:
            return nstep_set.pop()

    def set_positions(self, array, check_existing=False):
        """
        Set the positions of the trajectory.
        :param array: A numpy array with the positions in absolute values in units of angstrom
        :param bool check_exising: Check if the positions have been set, and raise in such case. Defaults to False.
        """
        self.set_array(self._POSITIONS_KEY, array, check_existing=check_existing, check_nat=True, check_nstep=True)

    def get_positions(self):
        return self.get_array(self._POSITIONS_KEY)

    def set_velocities(self, array, check_existing=False):
        """
        Set the velocites of the trajectory.
        :param array: A numpy array with the velocites in absolute values in units of angstrom/femtoseconds
        :param bool check_exising: Check if the velocities have been set, and raise in such case. Defaults to False.
        """
        self.set_array(self._VELOCITIES_KEY, array, check_existing=check_existing, check_nat=True, check_nstep=True)

    def get_velocities(self):
        return self.get_array(self._VELOCITIES_KEY)

    def set_forces(self, array, check_existing=False):
        """
        Set the forces of the trajectory.
        :param array: A numpy array with the forces in absolute values in units of eV/angstrom
        :param bool check_exising: Check if the forces have been set, and raise in such case. Defaults to False.
        """
        self.set_array(self._FORCES_KEY, array, check_existing=check_existing, check_nat=True, check_nstep=True)

    def get_forces(self):
        return self.get_array(self._FORCES_KEY)
        
    def get_step_atoms(self, index):
        """
        For a set stepindex, returns an atoms instance with all the settings from that step.
        :param int index: The index of the step
        :returns: an ase.Atoms instance from the trajectory at step
        """
        assert isinstance(index, int)
        atoms = self.atoms.copy()
        for k,v in self._arrays.items():
            try:
                getattr(atoms, 'set_{}'.format(k))(v[index])
            except AttributeError:
                pass
        return atoms

    def save(self, filename):
        """
        Saves the trajectory instance to tarfile.
        :param str filename: The filename. Won't be checked or modified with extension!
        """
        
