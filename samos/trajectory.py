# -*- coding: utf-8 -*-

import numpy as np
from samos.utils.attributed_array import AttributedArray


class IncompatibleTrajectoriesException(Exception):
    pass


def check_trajectory_compatibility(trajectories):
    """
    Check whether the trajectories passed are compatible.
    They are compatible if they store the same arrays, the same
    chemical symbols in the same order, and the same timestep.
    The cells are not compared.
    """

    assert len(trajectories) >= 1, 'No trajectories passed'
    for t in trajectories:
        if not isinstance(t, Trajectory):
            raise TypeError('{} is not an instance of Trajectory'.format(t))
    array_names_set = set()
    types_set = set()
    timestep_set = set()
    for t in trajectories:
        array_names_set.add(frozenset(t.get_arraynames()))
        types_set.add(tuple(t.types))
        timestep_set.add(t.get_timestep())

    if len(array_names_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different arrays are set')
    if len(types_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different chemical symbols in trajectories')
    if len(timestep_set) > 1:
        raise IncompatibleTrajectoriesException(
            'Different timesteps in trajectories')
    return np.array(types_set.pop()), timestep_set.pop()


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
    _TYPES_KEY = 'types'

    def __init__(self, **kwargs):
        """
        Instantiating a trajectory class.
        Optional keyword-arguments are everything with a set-method.
        """
        self._atoms = None
        super().__init__(**kwargs)

    @classmethod
    def from_atoms(cls, atoms_list, timestep_fs=None, add_arrays=None):
        """
        Instantiate a new class instance given a set of atoms
        """
        from ase import Atoms
        chem_sym_set = set()
        for atoms in atoms_list:
            if not isinstance(atoms, Atoms):
                raise TypeError('I have to receive a list/iterable over '
                                '{}'.format(Atoms))
            chem_sym_set.add(tuple(atoms.get_chemical_symbols()))
        if len(chem_sym_set) < 1:
            raise ValueError('Empty list provided')
        elif len(chem_sym_set) > 1:
            # let's try to fix that by reordering the atoms:
            chem_sym_set = set()
            for atoms in atoms_list:
                order = np.argsort(atoms.get_atomic_numbers())
                atoms.set_atomic_numbers(atoms.get_atomic_numbers()[order])
                atoms.set_positions(atoms.get_positions()[order])
                if atoms.get_velocities() is not None:
                    atoms.set_velocities(atoms.get_velocities()[order])
                try:
                    atoms.set_forces(atoms.get_forces()[order])
                except Exception:
                    pass
                chem_sym_set.add(tuple(atoms.get_chemical_symbols()))
            if len(chem_sym_set) > 1:
                raise ValueError('The chemical_symbols list of provided atoms '
                                 'are not the same for all, cannot proceed')

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
            pass  # velocities are returned as None if not existent
        if forces is not None and (forces**2).sum() > 1e-12:
            new.set_forces(forces)
        if (cells.std(axis=0).sum()) > 1e-12:
            new.set_cells(cells)
        if timestep_fs is not None:
            new.set_timestep(timestep_fs)
        if add_arrays is not None:
            for key in add_arrays:
                array = np.array([atoms.get_array(key) for atoms
                                  in atoms_list])
                new.set_array(key, array,
                              check_existing=False,
                              check_nstep=True)
        return new

    def _save_atoms(self, folder_name):
        from os.path import join
        if self._atoms:
            from ase.io import write
            write(join(folder_name, self._ATOMS_FILENAME), self._atoms)

    def _load_extra(self, folder_name, files_in_tar):
        """Restore the ase.Atoms written by :meth:`_save_atoms`."""
        from os.path import join
        if self._ATOMS_FILENAME in files_in_tar:
            from ase.io import read
            self.set_atoms(read(join(folder_name, self._ATOMS_FILENAME)))
            files_in_tar.remove(self._ATOMS_FILENAME)

    def get_timestep(self):
        """
        Return the MD timestep in femtoseconds.

        :raises KeyError: If the timestep has not been set.
        """
        return self.get_attr(self._TIMESTEP_KEY)

    def set_timestep(self, timestep_fs):
        """
        :param timestep: expects value of the timestep in femtoseconds
        """
        self.set_attr(self._TIMESTEP_KEY, float(timestep_fs))

    def get_atoms(self):
        """
        Return a copy of the reference :class:`~ase.Atoms` object, or
        None if atoms have not been set.

        A copy is returned so that modifications by the caller do not
        affect the stored atoms.
        """
        if self._atoms is not None:
            return self._atoms.copy()
        else:
            return None
            # raise ValueError('Atoms have not been set')

    def set_types(self, types):
        """
        Override the chemical-symbol list stored in this trajectory.

        Normally :meth:`get_types` reads symbols from the ASE
        :class:`~ase.Atoms` object set via :meth:`set_atoms`.  Calling
        this method stores a separate override array, which takes
        priority.  Useful when the trajectory has been relabelled (e.g.
        via :meth:`transform_species`) without changing the underlying
        atoms object.

        :param array-like types: 1-D sequence of chemical symbols,
            one per atom.
        """
        types = np.array(types, dtype=str)
        self.set_array(self._TYPES_KEY, types,
                       check_existing=False,
                       check_nat=False, check_nstep=False,)

    def get_types(self):
        """
        Return a 1-D array of chemical symbols, one per atom.

        Priority: the override array set by :meth:`set_types` is
        returned if present; otherwise symbols are taken from the ASE
        atoms object; returns None if neither is available.
        """
        if self._TYPES_KEY in self.get_arraynames():
            return self.get_array(self._TYPES_KEY)
        elif self._atoms is not None:
            return np.array(self._atoms.get_chemical_symbols())
        else:
            return None

    @property
    def types(self):
        return self.get_types()

    @property
    def atoms(self):
        return self.get_atoms()

    def set_atoms(self, atoms):
        from ase import Atoms
        if not isinstance(atoms, Atoms):
            raise ValueError('You have to pass an instance of ase.Atoms')
        self._atoms = atoms

    @property
    def nat(self):
        types = self.get_types()
        if types is None:
            raise ValueError('Types have not been set')
        else:
            return len(types)

    @property
    def cell(self):
        return self.atoms.cell

    def set_cells(self, array, check_existing=False):
        # make sure a list is converted to an array before
        # calling shape:

        array = np.array(array)
        if array.shape == (3, 3):
            array = np.tile(array, (self.nstep, 1, 1))
        self.set_array(self._CELL_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_1=3,
                       wanted_shape_2=3)

    def get_cells(self):
        """
        Return the per-step cell array of shape (nstep, 3, 3) in
        Angstrom, or None if no time-varying cell has been stored.

        When the cell is constant throughout the trajectory it is not
        stored here; use :attr:`cell` (from the reference atoms object)
        instead.
        """
        if self._CELL_KEY in self.get_arraynames():
            return self.get_array(self._CELL_KEY)
        return None

    def get_volumes(self):
        """
        Return the cell volume at each step as a 1-D array of length
        nstep, in Angstrom^3.

        If no time-varying cell is stored, the constant volume from the
        reference atoms object is broadcast across all steps.
        """
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
            If it's an integer, I that only this integer is wanted.
        :param int start:
            The start of indexing, defaults to 0.
            For fortran indexing, set to 1.
        :return: A numpy array of indices
        """
        assert isinstance(start, int), 'Start is not an integer'
        types = self.get_types()
        if isinstance(species, str):
            msk = types == species
        elif isinstance(species, int):
            msk = np.zeros(len(types), dtype=bool)
            msk[species] = True
        else:
            raise TypeError('species  has  to be an integer or a string, '
                            'I got {}'.format(type(species)))
        return np.where(msk)[0] + start

    def set_positions(self, array, check_existing=False):
        """
        Set the positions of the trajectory.
        :param array:
            A numpy array with the positions in absolute
            values in units of angstrom
        :param bool check_existing:
            Check if the positions have been set, and
            raise in such case. Defaults to False.
        """
        self.set_array(self._POSITIONS_KEY, array,
                       check_existing=check_existing,
                       check_nat=self.nat, check_nstep=True,
                       wanted_shape_len=3, wanted_shape_2=3)

    def get_positions(self):
        return self.get_array(self._POSITIONS_KEY)

    def set_velocities(self, array, check_existing=False):
        """
        Set the velocites of the trajectory.
        :param array:
            A numpy array with the velocites in absolute
            values in units of angstrom/femtoseconds
        :param bool check_existing:
            Check if the velocities have been set, and
            raise in such case. Defaults to False.
        """
        self.set_array(self._VELOCITIES_KEY, array,
                       check_existing=check_existing, check_nat=self.nat,
                       check_nstep=True, wanted_shape_len=3, wanted_shape_2=3)

    def calculate_velocities_from_positions(self, overwrite=False):
        """
        Using positions-verlet update formula to infer
        velocities from positions
        """
        if self._VELOCITIES_KEY in self.get_arraynames():
            if not overwrite:
                raise Exception('I am overwriting an existing velocity array'
                                'Pass overwrite=True to allow')
        pos = self.get_positions()
        timestep_fs = self.get_timestep()
        vel_first = (pos[1] - pos[0]) / timestep_fs
        vel_last = (pos[-1] - pos[-2]) / timestep_fs
        vel_intermediate = (pos[2:] - pos[:-2]) / (2*timestep_fs)
        vel = np.vstack(([vel_first], vel_intermediate, [vel_last]))
        self.set_velocities(vel)
        return vel.copy()

    def get_velocities(self, remove_angular_momentum=False):
        """
        Return the velocity array (nstep, nat, 3) in Angstrom/fs.

        :param bool remove_angular_momentum:
            If True, subtract the rigid-body rotational velocity
            ``omega(t) x r_i(t)`` from each atom at each timestep
            before returning, without modifying the stored array.

            The correction requires positions and masses and must be
            applied after any centre-of-mass translation has been
            removed (e.g. via :meth:`recenter`), because the inertia
            tensor and angular momentum are both defined relative to
            the centre of mass.  If COM recentering has not been
            applied the result is still correct but omega will include
            a spurious contribution from the translational offset.

            An in-place variant was not chosen here because angular
            momentum removal is only meaningful for velocity-based
            analysis (VAF, VDOS).  Modifying the stored array would
            silently corrupt any subsequent position-based analysis
            (MSD) that re-derives velocities from positions, and would
            prevent the user from toggling the correction without
            reloading the trajectory.
        """
        vel = self.get_array(self._VELOCITIES_KEY)
        if not remove_angular_momentum:
            return vel

        pos = self.get_positions()        # (nstep, nat, 3)
        masses = self.atoms.get_masses()  # (nat,)
        vel = vel.copy()                  # do not modify the stored array

        # Inertia tensor per step: shape (nstep, 3, 3).
        # I[s,a,b] = sum_i m_i * (|r_i|^2 * delta_ab - r_i[a] * r_i[b])
        # Written as the outer-product form to avoid explicit delta_ab:
        #   I = sum_i m_i * (|r_i|^2 * eye(3) - outer(r_i, r_i))
        r2 = np.einsum('sia,sia->si', pos, pos)       # (nstep, nat)
        I = (np.einsum('si,ab->sab', masses * r2, np.eye(3))  # noqa: E741
             - np.einsum('i,sia,sib->sab', masses, pos, pos))

        # Angular momentum per step: L = sum_i m_i * (r_i x v_i)
        L = np.einsum('i,sia->sa', masses,
                      np.cross(pos, vel))             # (nstep, 3)

        # Solve I @ omega = L for omega at each step.
        # numpy.linalg.solve accepts batched (nstep, 3, 3) input since
        # NumPy 1.14, already required by this package.
        omega = np.linalg.solve(I, L)                 # (nstep, 3)

        # Subtract rigid-body rotational contribution v_rot = omega x r_i.
        vel -= np.cross(omega[:, np.newaxis, :], pos)  # broadcast over nat

        return vel

    def set_forces(self, array, check_existing=False):
        """
        Set the forces of the trajectory.
        :param array:
            A numpy array with the forces in absolute
            values in units of eV/angstrom
        :param bool check_existing:
            Check if the forces have been set, and raise in
            such case. Defaults to False.
        """
        self.set_array(self._FORCES_KEY, array, check_existing=check_existing,
                       check_nat=self.nat, check_nstep=True,
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
            raise ValueError('Not implemented order {}'.format(order))

    def get_stress(self):
        return self.get_array(self._STRESS_KEY)

    def get_forces(self):
        return self.get_array(self._FORCES_KEY)

    def set_pot_energies(self, array, check_existing=False):

        self.set_array(self._POT_ENER_KEY, array,
                       check_existing=check_existing,
                       check_nat=False, check_nstep=True,
                       wanted_shape_len=1)

    def apply_unit_conversion(self, l_conv=1.0, v_conv=1.0, f_conv=1.0,
                              e_conv=1.0, s_conv=1.0):
        """
        Scale stored arrays to samos internal units in-place.

        Only arrays present in the trajectory are touched; missing arrays
        are silently skipped.

        :param float l_conv: length factor (multiply to get Angstrom);
            applied to both positions and cell vectors
        :param float v_conv: velocity factor (multiply to get A/fs)
        :param float f_conv: force factor (multiply to get eV/A)
        :param float e_conv: energy factor (multiply to get eV)
        :param float s_conv: stress factor
        """
        names = self.get_arraynames()
        if l_conv != 1.0:
            if self._POSITIONS_KEY in names:
                self._arrays[self._POSITIONS_KEY] *= l_conv
            if self._CELL_KEY in names:
                self._arrays[self._CELL_KEY] *= l_conv
        if v_conv != 1.0 and self._VELOCITIES_KEY in names:
            self._arrays[self._VELOCITIES_KEY] *= v_conv
        if f_conv != 1.0 and self._FORCES_KEY in names:
            self._arrays[self._FORCES_KEY] *= f_conv
        if e_conv != 1.0 and self._POT_ENER_KEY in names:
            self._arrays[self._POT_ENER_KEY] *= e_conv
        if s_conv != 1.0 and self._STRESS_KEY in names:
            self._arrays[self._STRESS_KEY] *= s_conv

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
                          ), 'step index has to be an integer'

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
                shape = v[index].shape
                if len(shape) > 0:
                    if shape[0] == len(atoms):
                        try:
                            if hasattr(atoms, 'set_{}'.format(k)):
                                getattr(atoms, 'set_{}'.format(k))(v[index])
                            else:
                                atoms.set_array(k, v[index])
                        except Exception:
                            pass
                    else:
                        print(f"Warning. Can't pass array {k} to atoms")
                else:
                    # this must be a single number
                    atoms.info[k] = v[index]

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
            start, int) and start >= 0, 'start has to be a positive integer'
        assert isinstance(
            end, int) and end >= 0, 'end has to be a positive integer'
        assert isinstance(
            stepsize, int
        ) and stepsize >= 0, 'stepsize has to be a positive integer'
        if end > self.nstep:
            raise ValueError(
                'End > nsteps, leave None and it will be set to nstep')
        indices = np.arange(start, end, stepsize)

        if len(indices) < 1:
            raise ValueError('No indices for trajectory')
        assert isinstance(
            start, int) and start >= 0, 'start has to be a positive integer'
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
        # masses are either set to 1.0 (all) or to the actual masses
        # of the atoms
        # Setting to 1 means that the mass is not accounted for and the
        # the center is purely geometric.
        if mode == 'geometric':
            masses = [1.0] * len(self.atoms)
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
            factors = [1] * len(self.atoms)

        rel_masses = np.array(factors, dtype=float
                              ) * np.array(masses, dtype=float)
        rel_masses /= rel_masses.sum()
        # com shape: (nstep, 3)
        com = np.einsum('a,sac->sc', rel_masses, self.get_positions())
        self.set_positions(self.get_positions() - com[:, np.newaxis, :])
        if 'velocities' in self.get_arraynames():
            com_vel = np.einsum('a,sac->sc', rel_masses, self.get_velocities())
            self.set_velocities(
                self.get_velocities() - com_vel[:, np.newaxis, :])

    def transform_species(self, target):
        """
        Relabel every atom in the trajectory as *target* in-place.

        This is useful when the analysis should treat a multi-species
        trajectory as a single-component system - for example, computing
        a collective MSD for a sublattice without filtering by element.

        Two storage locations must be kept in sync:

        * ``self._atoms`` the ASE :class:`~ase.Atoms` object that
          holds chemical symbols and masses.  Setting new symbols here
          also updates per-atom masses via ASE's internal table, which
          matters for mass-weighted operations such as :meth:`recenter`.
        * The ``_TYPES_KEY`` array, an optional override stored in the
          :class:`~samos.utils.attributed_array.AttributedArray` that
          :meth:`get_types` checks before falling back to ``_atoms``.
          If present it must be overwritten; otherwise callers going
          through :meth:`get_types` would still see the old labels.

        :param str target:
            Chemical symbol of the target species (e.g. ``'H'``).
        """
        nat = len(self._atoms)
        self._atoms.set_chemical_symbols([target] * nat)
        if self._TYPES_KEY in self.get_arraynames():
            self.set_types([target] * nat)
