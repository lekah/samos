import numpy as np
from samos.utils.attributed_array import AttributedArray


class IncompatibleTrajectoriesException(Exception):
    pass


def check_trajectory_compatibility(trajectories):
    """
    Check whether the trajectories passed are compatible.
    They are compatible if they have the same order of atoms, and the same cell, and store the same arrays
    """

    assert len(trajectories) >= 1, 'No trajectories passed'
    for t in trajectories:
        if not isinstance(t, Trajectory):
            raise TypeError("{} is not an instance of Trajectory".format(t))
    array_names_set = set()
    chemical_symbols_set = set()
    timestep_set = set()
    for t in trajectories:
        array_names_set.add(frozenset(t.get_arraynames()))
        atoms = t.atoms
        chemical_symbols_set.add(tuple(atoms.get_chemical_symbols()))
        timestep = t.get_timestep()
        timestep_set.add(timestep)

    if len(array_names_set) > 1:
        raise IncompatibleTrajectoriesException("Different arrays are set")
    if len(chemical_symbols_set) > 1:
        raise IncompatibleTrajectoriesException("Different chemical symbols in trajectory")
    return atoms, timestep


class Trajectory(AttributedArray):
    """
    Class defining our trajectories.
    A trajectory is a sequence of time-ordered points in phase space.
    The internal units of a trajectory:
    *   Femtoseconds for times
    *   Angstrom for coordinates
    *   eV for energies
    *   Masses and cells are set via the _atoms member, an ase.Atoms instance and units as in ase are used. 
    """
    _TIMESTEP_KEY = 'timestep_fs'
    _POSITIONS_KEY = 'positions'
    _VELOCITIES_KEY = 'velocities'
    _FORCES_KEY = 'forces'
    _ATOMS_FILENAME = 'atoms.traj'
    def __init__(self, **kwargs):
        """
        Instantiating a trajectory class.
        Optional keyword-arguments are everything with a set-method.
        """
        self._atoms = None
        super(Trajectory, self).__init__(**kwargs)

    def _save_atoms(self, folder_name):
        from io.path import join
        if self._atoms:
            from ase.io import write
            write(join(folder_name, self._ATOMS_FILENAME), self._atoms)

    def get_timestep(self):
        return self.get_attr(self._TIMESTEP_KEY)

    def set_timestep(self, timestep_fs):
        """
        :param timestep: expects value of the timestep in femtoseconds
        """
        self.set_attr(self._TIMESTEP_KEY, float(timestep_fs))


    def get_atoms(self):
        if self._atoms:
            return self._atoms
        else:
            raise ValueError("Atoms have not been set")
    @property
    def atoms(self):
        return self.get_atoms()

    def set_atoms(self, atoms):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            raise ValueError("You have to pass an instance of ase.Atoms")
        self._atoms = atoms

    @property
    def cell(self):
        return self.atoms.cell


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

        return np.array([i for i, s in enumerate(array_to_index, start=start) if s==species])


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


    def recenter(self):
        """
        Recenter positions and velocities in-place
        """
        from samos.lib.mdutils import recenter_positions, recenter_velocities
        
        
