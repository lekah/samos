# -*- coding: utf-8 -*-

import numpy as np
from samos.utils.attributed_array import AttributedArray


class IncompatibleTrajectoriesException(Exception):
    pass


def check_trajectory_compatibility(trajectories):
    """
    Check whether the trajectories passed are compatible.
    They are compatible if they have the same order of atoms,
    and the same cell, and store the same arrays
    """

    assert len(trajectories) >= 1, 'No trajectories passed'
    for t in trajectories:
        if not isinstance(t, Trajectory):
            raise TypeError('{} is not an instance of Trajectory'.format(t))
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
        raise IncompatibleTrajectoriesException(
            'Different arrays are set')
    if len(chemical_symbols_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different chemical symbols in trajectories')
    if len(timestep_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different timesteps in trajectories')
    return atoms, timestep


class Trajectory(AttributedArray):
    """
    Class defining our trajectories.
    A trajectory is a sequence of time-ordered points in phase space.
    The internal units of a trajectory:
    *   Femtoseconds for times
    *   Angstrom for coordinates
    *   eV for energies
    *   Masses and cells are set via the _atoms member, an ase.Atoms
        instance and units as in ase are used.
    """
    _TIMESTEP_KEY = 'timestep_fs'
    _POSITIONS_KEY = 'positions'
    _VELOCITIES_KEY = 'velocities'
    _STRESS_KEY = 'stress'
    _CELL_KEY = 'cells'
    _FORCES_KEY = 'forces'
    _POT_ENER_KEY = 'potential_energy'
    _ATOMS_FILENAME = 'atoms.traj'

    def __init__(self, **kwargs):
        """
        Instantiating a trajectory class.
        Optional keyword-arguments are everything with a set-method.
        """
        self._atoms = None
        super(Trajectory, self).__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms_list, timestep_fs=None):
        """
        Instantiate a new class instance given a set of atoms
        """
        from ase import Atoms
        chem_sym_set = set()
        for atoms in atoms_list:
            if not isinstance(atoms, Atoms):
                raise TypeError("I have to receive a list/iterable over "
                                "{}".format(Atoms))
            chem_sym_set.add(tuple(atoms.get_chemical_symbols()))
        if len(chem_sym_set) < 1:
            raise ValueError("Empty list provided")
        elif len(chem_sym_set) > 1:
            raise ValueError("The chemical_symbols list of provided atoms "
                             "are not the same for all, cannot proceed")

        positions = np.array([atoms.get_positions() for atoms in atoms_list])
        velocities = np.array([atoms.get_velocities() for atoms in atoms_list])
        try:
            forces = np.array([atoms.get_forces() for atoms in atoms_list])
        except Exception:
            forces = None
        cells = np.array([atoms.cell for atoms in atoms_list])
        new = cls(atoms=atoms_list[0])
        new.set_positions(positions)
        try:
            if (velocities**2).sum() > 1e-12:
                new.set_velocities(velocities)
        except TypeError:
            pass  # velocities are returned as none if not existen
        if forces is not None and (forces**2).sum() > 1e-12:
            new.set_forces(forces)
        if (cells.std(axis=0).sum()) > 1e-12:
            new.set_cells(cells)
        if timestep_fs is not None:
            new.set_timestep(timestep_fs)
        return new

    def _save_atoms(self, folder_name):
        from os.path import join
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
            raise ValueError('Atoms have not been set')

    @property
    def atoms(self):
        return self.get_atoms()

    def set_atoms(self, atoms):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            raise ValueError('You have to pass an instance of ase.Atoms')
        self._atoms = atoms

    @property
    def cell(self):
        return self.atoms.cell

    def set_cells(self, array, check_existing=False):
        self.set_array(self._CELL_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_1=3,
                       wanted_shape_2=3)

    def get_cells(self):
        if self._CELL_KEY in self.get_arraynames():
            return self.get_array(self._CELL_KEY)
        return None

    def get_volumes(self):
        cells = self.get_cells()
        if cells is None:
            volume = self.atoms.get_volume()
            return np.array([volume]*self.nstep)
        volumes = [np.linalg.det(cell) for cell in cells]
        return np.array(volumes)

    def get_indices_of_species(self, species, start=0):
        """
        Convenience function to get all indices of a species.
        :param species:
            The identifier of a species. If this is a string,
            I assume the chemical symbol (abbreviation).
            I.e. Li for lithium, H for hydrogen.
            If it's an integer, I assume the atomic number.
        :param int start:
            The start of indexing, defaults to 0.
            For fortran indexing, set to 1.
        :return: A numpy array of indices
        """
        assert isinstance(start, int), 'Start is not an integer'
        if isinstance(species, str):
            array_to_index = self.atoms.get_chemical_symbols()
        elif isinstance(species, int):
            array_to_index = self.atoms.get_atomic_numbers()
        else:
            raise TypeError('species  has  to be an integer or a string, '
                            'I got {}'.format(type(species)))

        return np.array([i for i, s
                         in enumerate(array_to_index, start=start)
                         if s == species])

    @property
    def nstep(self):
        """
        :returns: The number of trajectory steps
        :raises: ValueError if no unique number of steps can be determined.
        """
        nstep_set = set([array.shape[0]
                        for array in list(self._arrays.values())])
        if len(nstep_set) == 0:
            raise ValueError('No arrays have been set, yet')
        elif len(nstep_set) > 1:
            raise ValueError('Incommensurate arrays')
        else:
            return nstep_set.pop()

    def set_positions(self, array, check_existing=False):
        """
        Set the positions of the trajectory.
        :param array:
            A numpy array with the positions in absolute
            values in units of angstrom
        :param bool check_exising:
            Check if the positions have been set, and
            raise in such case. Defaults to False.
        """
        self.set_array(self._POSITIONS_KEY, array,
                       check_existing=check_existing,
                       check_nat=True, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_2=3)

    def get_positions(self):
        return self.get_array(self._POSITIONS_KEY)

    def set_velocities(self, array, check_existing=False):
        """
        Set the velocites of the trajectory.
        :param array:
            A numpy array with the velocites in absolute
            values in units of angstrom/femtoseconds
        :param bool check_exising:
            Check if the velocities have been set, and
            raise in such case. Defaults to False.
        """
        self.set_array(self._VELOCITIES_KEY, array,
                       check_existing=check_existing, check_nat=True,
                       check_nstep=True, wanted_shape_len=3, wanted_shape_2=3)

    def calculate_velocities_from_positions(self, overwrite=False):
        """
        Using positions-verlet update formula to infer
        velocities from positions
        """
        if self._VELOCITIES_KEY in self.get_arraynames():
            if not overwrite:
                raise Exception("I am overwriting an existing velocity array"
                                "Pass overwrite=True to allow")
        pos = self.get_positions()
        timestep_fs = self.get_timestep()
        vel_first = (pos[1] - pos[0]) / timestep_fs
        vel_last = (pos[-1] - pos[-2]) / timestep_fs
        vel_intermediate = (pos[2:] - pos[:-2]) / (2*timestep_fs)
        vel = np.vstack(([vel_first], vel_intermediate, [vel_last]))
        self.set_velocities(vel)
        return vel.copy()

    def get_velocities(self):
        return self.get_array(self._VELOCITIES_KEY)

    def set_forces(self, array, check_existing=False):
        """
        Set the forces of the trajectory.
        :param array:
            A numpy array with the forces in absolute
            values in units of eV/angstrom
        :param bool check_exising:
            Check if the forces have been set, and raise in
            such case. Defaults to False.
        """
        self.set_array(self._FORCES_KEY, array, check_existing=check_existing,
                       check_nat=True, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_2=3)

    def set_stress(self, array, order='voigt', check_existing=False):
        """
        order voigt expects keys ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')
        """
        if order == 'voigt':
            self.set_array(self._STRESS_KEY, array,
                           check_existing=check_existing, check_nstep=True,
                           wanted_shape_1=6, wanted_shape_len=2)
        else:
            raise ValueError("Not implemented order {}".format(order))

    def get_stress(self):
        return self.get_array(self._STRESS_KEY)

    def get_forces(self):
        return self.get_array(self._FORCES_KEY)

    def set_pot_energies(self, array, check_existing=False):

        self.set_array(self._POT_ENER_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=1)

    def get_step_atoms(self, index, ignore_calculated=False,
                       warnings=True):
        """
        For a set stepindex, returns an atoms instance with all
        the settings from that step.
        :param int index: The index of the step
        :param bool ignore_calculated:
            ignore the calculated values (forces, energies, stress)
        :returns: an ase.Atoms instance from the trajectory at step
        """
        assert isinstance(index, (int, np.int64)
                          ), "step index has to be an integer"

        need_calculator = False
        if not ignore_calculated:
            for key in (self._FORCES_KEY, self._POT_ENER_KEY):
                if key in self.get_arraynames():
                    need_calculator = True
                    break
        atoms = self.atoms.copy()

        if need_calculator:
            from ase.calculators.singlepoint import SinglePointCalculator
            calc_kwargs = {}

        for k, v in list(self._arrays.items()):
            if k == self._CELL_KEY:
                atoms.set_cell(v[index])
            elif k == self._FORCES_KEY:
                if need_calculator:
                    calc_kwargs['forces'] = v[index]
            elif k == self._POT_ENER_KEY:
                if need_calculator:
                    calc_kwargs['energy'] = v[index]
            elif k == self._STRESS_KEY:
                if need_calculator:
                    calc_kwargs['stress'] = v[index]
            else:
                try:
                    getattr(atoms, 'set_{}'.format(k))(v[index])
                except AttributeError as e:
                    if warnings:
                        print(e)
        if need_calculator:
            calc = SinglePointCalculator(atoms, **calc_kwargs)
            atoms.set_calculator(calc)  # this seems to be deprecated,
            # replace with atoms.calc = calc at somepoint

        return atoms

    def get_ase_trajectory(self, start=0, end=None, stepsize=1):
        """
        :param int stepsize: A step size, defaults to 1
        :param int start: The start step, defaults to 0
        :param int end: The last step defaults to length of trajectory

        :returns: a list of atoms instances of this trajectory
        """
        if end is None:
            end = self.nstep
        assert isinstance(
            start, int) and start >= 0, "start has to be a positive integer"
        assert isinstance(
            end, int) and end >= 0, "end has to be a positive integer"
        assert isinstance(
            stepsize, int
        ) and stepsize >= 0, "stepsize has to be a positive integer"
        if end > self.nstep:
            raise ValueError(
                "End > nsteps, leave None and it will be set to nstep")
        indices = np.arange(start, end, stepsize)

        if len(indices) < 1:
            raise ValueError("No indices for trajectory")
        assert isinstance(
            start, int) and start >= 0, "start has to be a positive integer"
        atomslist = [self.get_step_atoms(idx) for idx in indices]

        return atomslist

    def recenter(self, sublattice=None, mode=None):
        """
        Recenter positions and velocities in-place
        :param tuple sublattice:
            A tuple or list of element names or indices that
            define a sublattice of the structure.
            If given, the trajectory will be centered on the
            center of mass of that sublattice.
        """
        from samos.lib.mdutils import recenter_positions, recenter_velocities
        if mode == 'geometric':
            masses = [1.0] * len(masses)
        else:
            masses = self.atoms.get_masses()
        if sublattice is not None:
            if not isinstance(sublattice, (tuple, list, set)):
                raise TypeError(
                    'You have to pass a tuple/list/set as sublattice')
            factors = [0]*len(masses)
            for item in sublattice:
                if isinstance(item, int):
                    try:
                        factors[item] = 1
                    except IndexError:
                        raise IndexError(
                            'You passed an integer for the sublattice, '
                            'but it is out of range')
                elif isinstance(item, str):
                    for index in self.get_indices_of_species(item):
                        factors[index] = 1
                else:
                    raise TypeError(
                        'You passed {} {} as a sublattice specifier, '
                        'this is not recognized'.format(type(item), item))
        else:
            factors = [1] * len(masses)
        self.set_positions(recenter_positions(self.get_positions(), masses, factors))
        if 'velocities' in self:
            self.set_velocities(recenter_velocities(
                self.get_velocities(), masses, factors))
