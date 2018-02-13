import numpy as np


class Trajectory(object):
    """
    The main class. A trajectory is a sequence of time-ordered points in phase space.
    """
    def __init__(self, **kwargs):
        #~ self._positions = None
        #~ self._velocities = None
        #~ self._forces = None
        self._cell = None
        self._arrays = {}
        #~ self._cells = None
        self._symbols = None
        self._nat = None

        for key, val in kwargs.items():
            #~ try:
            getattr(self, 'set_{}'.format(key))(val)
            #~ except Exception as e:

    def set_timestep(self, timestep):
        """
        :param timestep: expects value of the timestep
        """
        raise NotImplemented("instance of units?")

    def set_symbols(self, symbols_list):
        """
        :param list symbols_list: A list of chemical symbols, defining the order of elements in single-particle array given.
        """
        if not isinstance(symbols_list, (tuple, list)):
            raise ValueError("Input to set_symbols has to be a list of symbols")
        self._symbols = symbols_list[:]
        self._nat = len(self._symbols)
    def set_cell(self, cell):
        _cell = np.array(cell)
        assert _cell.shape == (3,3), "Cell needs to be of shape  3,3"
        self._cell = _cell

    def set_array(self, name, array):
        array = np.array(array)
        for other_name, other_array in self._arrays.items():
            assert array.shape[0] == other_array.shape[0], (
            'Number of steps in this array is not compliant with array {}'.format(other_name))
                
        self._arrays[name] = array

    def get_array(self, name):
        return self._arrays[name]

    def get_incides_of_species(self, species, start=0):
        return np.array([i for i, s in enumerate(self._symbols, start=start) if s==species])

    @property
    def cell(self):
        if self._cell is None:
            raise ValueError("Cell not set, yet")
        return self._cell
    @property
    def symbols(self):
        if not self._symbols:
            raise ValueError("symbols not set, yet")
        return self._symbols
    @property
    def nstep(self):
        nstep_set = set([array.shape[0] for array in self._arrays.values()])
        if len(nstep_set) == 0:
            raise ValueError("No arrays have been set, yet")
        elif len(nstep_set) > 1:
            raise ValueError("Incommensurate arrays")
        else:
            return nstep_set.pop()
    @property
    def nat(self):
        return len(self.symbols)

    def set_positions(self, array):
        self.set_array('positions', array)

    def get_positions(self):
        return self.get_array('positions')

    def set_velocities(self, array):
        self.set_array('velocities', array)

    def get_velocities(self):
        return self.get_array('velocities')

    def set_forces(self, array):
        self.set_array('forces', array)

    def get_forces(self):
        return self.get_array('forces')
        
    def get_step_atoms(self, index):
        from ase import Atoms
        if not self._symbols:
            raise Exception("You need to first set the symbols")
        # ase expects in angstrom:
        atoms = Atoms(self._symbols)
        return atoms

