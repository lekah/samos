# -*- coding: utf-8 -*-

from ase import Atoms
import numpy as np
from samos.utils.attributed_array import AttributedArray

# The stored array names that do not become a column of their own.
# Both containers use these same names, which is what lets one
# function below serve a frame of either.
_CELLS = 'cells'
_CALCULATOR_KEYS = {'forces': 'forces',
                    'potential_energy': 'energy',
                    'stress': 'stress'}
# Only these two make a frame worth a calculator.  Stress on its own
# never did, and reading it back out of an extxyz file written without
# the other two has not been asked for.
_NEED_CALCULATOR_KEYS = ('forces', 'potential_energy')


def attach_arrays(atoms, arrays, ignore_calculated=False):
    """
    Put one frame's arrays onto *atoms*.

    The single rule both containers follow, so that a frame of a
    trajectory and a frame of a structure list reach a writer in the
    same shape: an array with one row per atom becomes a column of its
    own, a single number becomes an info entry, and forces, energy and
    stress become a calculator.  What an extxyz file's ``Properties``
    line says is decided here and nowhere else.

    :param atoms: The frame as an :class:`ase.Atoms`, already holding
        its symbols, positions, cell and boundary conditions.
    :param dict arrays: The stored arrays, each already cut down to
        this one frame.
    :param bool ignore_calculated: Leave forces, energy and stress off.
    :returns: *atoms*, which is modified in place.
    """
    need_calculator = (not ignore_calculated
                       and any(key in arrays
                               for key in _NEED_CALCULATOR_KEYS))
    calc_kwargs = {}

    for name, value in arrays.items():
        if name == _CELLS:
            atoms.set_cell(value)
        elif name in _CALCULATOR_KEYS:
            if need_calculator:
                calc_kwargs[_CALCULATOR_KEYS[name]] = value
        else:
            shape = np.shape(value)
            if len(shape) > 0:
                if shape[0] == len(atoms):
                    try:
                        if hasattr(atoms, 'set_{}'.format(name)):
                            getattr(atoms, 'set_{}'.format(name))(value)
                        else:
                            atoms.set_array(name, value)
                    except Exception:
                        pass
                else:
                    print("Warning. Can't pass array {} to atoms".format(
                        name))
            else:
                # this must be a single number
                atoms.info[name] = value

    if need_calculator:
        from ase.calculators.singlepoint import SinglePointCalculator
        atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
    return atoms


class StructureList(AttributedArray):
    """
    An ordered list of structures that need not share their atoms.

    Frames are stored flat: every atom of every frame goes into one
    ``(total_atoms, ...)`` array, and an ``offsets`` array of length
    ``nstep + 1`` records where each frame starts and ends.  A frame is
    therefore a slice of that array -- a view, not a copy.

    A list of :class:`ase.Atoms` was the obvious alternative and was
    rejected.  Flat numpy arrays mean the save/load machinery of
    :class:`~samos.utils.attributed_array.AttributedArray` applies with
    no new code, and any further per-atom quantity (forces, charges,
    scattering weights) needs no bookkeeping beyond the same offsets.

    :class:`~samos.trajectory.Trajectory` is the special case where
    every frame holds the same atoms in the same order, which is what
    lets it keep rectangular ``(nstep, nat, 3)`` storage.  It overrides
    only the few methods below that touch the layout -- above all
    :meth:`_frame_slice` and :meth:`_flat`, through which every
    per-atom access is funnelled -- and inherits everything built on
    them.
    """
    _POSITIONS_KEY = 'positions'
    _TYPES_KEY = 'types'
    _CELL_KEY = 'cells'
    _PBC_KEY = 'pbc'
    _OFFSETS_KEY = 'offsets'
    _VELOCITIES_KEY = 'velocities'
    _FORCES_KEY = 'forces'
    _POT_ENER_KEY = 'potential_energy'
    _STRESS_KEY = 'stress'
    _ARRAY_KINDS_KEY = 'array_kinds'

    # Which stored arrays are indexed by atom and which by frame.
    # Slicing has to know: a per-atom array is cut with the offsets, a
    # per-frame array with the frame numbers, and getting it the wrong
    # way round produces a plausible-looking but corrupt result rather
    # than an error.
    #
    # These four are the layout itself and are always known.  Anything
    # else -- velocities, forces, a column read from a dump, a number
    # per frame read from a thermo file -- is declared by the setter
    # that stores it, see :meth:`set_per_atom_array`.
    _PER_ATOM_KEYS = (_POSITIONS_KEY, _TYPES_KEY)
    _PER_FRAME_KEYS = (_CELL_KEY, _PBC_KEY)

    # No __init__ of its own, deliberately.  Holding no state is what
    # lets Trajectory inherit from this class without its rectangular
    # arrays having to coexist with a second, unused layout.

    @classmethod
    def from_atoms(cls, atoms_list):
        """
        Build a structure list from an iterable of :class:`ase.Atoms`.

        Unlike :meth:`~samos.trajectory.Trajectory.from_atoms`, the
        frames may differ in atom count, species and cell.

        :param atoms_list: Iterable of :class:`ase.Atoms`.
        :returns: A new :class:`StructureList`.
        """
        new = cls()
        new.set_structures(atoms_list)
        return new

    @classmethod
    def from_frames(cls, positions, types, cells, pbc=True):
        """
        Build a structure list from per-atom and per-frame arrays.

        The route for a reader, which holds the arrays already and may
        carry a label that is not a chemical element at all: nothing
        here goes through :class:`ase.Atoms`, so no label is checked
        against the periodic table and no per-frame object is built for
        a file of many frames.

        :param positions: One ``(nat, 3)`` array per frame, in Angstrom.
        :param types: One sequence of per-atom labels per frame.
        :param cells: One ``(3, 3)`` cell per frame, in Angstrom.
        :param pbc: The boundary conditions, as one flag, one ``(3,)``
            set of flags shared by every frame, or one entry per frame.
        :returns: A new :class:`StructureList`.
        """
        new = cls()
        new.set_frames(positions, types, cells, pbc=pbc)
        return new

    def set_frames(self, positions, types, cells, pbc=True):
        """
        Replace the stored frames with the arrays given.

        The one place the flat layout is written; both
        :meth:`from_frames` and :meth:`set_structures` end up here.

        :param positions: One ``(nat, 3)`` array per frame, in Angstrom.
        :param types: One sequence of per-atom labels per frame.
        :param cells: One ``(3, 3)`` cell per frame, in Angstrom.
        :param pbc: The boundary conditions, as one flag, one ``(3,)``
            set of flags shared by every frame, or one entry per frame.
        :raises ValueError: If no frames are given, if the three
            sequences differ in length, or if a frame's positions and
            labels do not agree in number.
        """
        positions = [np.asarray(p, dtype=float) for p in positions]
        types = [list(t) for t in types]
        if not positions:
            raise ValueError('No frames given, nothing to store')
        if not len(positions) == len(types) == len(cells):
            raise ValueError(
                'Got {} frames of positions, {} of types and {} '
                'cells'.format(len(positions), len(types), len(cells)))
        for index, (pos, labels) in enumerate(zip(positions, types)):
            if len(pos) != len(labels):
                raise ValueError(
                    'Frame {} holds {} positions but {} type '
                    'labels'.format(index, len(pos), len(labels)))

        counts = [len(pos) for pos in positions]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(int)
        flat_positions = np.concatenate(
            [pos.reshape(len(pos), 3) for pos in positions])
        # Built from one flat list of Python strings rather than by
        # concatenating per-frame arrays: an empty frame would
        # contribute a float64 array of size zero and poison the dtype
        # of the concatenation.
        flat_types = []
        for labels in types:
            flat_types.extend(labels)
        pbc = np.asarray(pbc, dtype=bool)
        if pbc.ndim == 0:
            pbc = np.full((len(positions), 3), bool(pbc))
        elif pbc.ndim == 1:
            pbc = np.tile(pbc, (len(positions), 1))

        self.set_array(self._OFFSETS_KEY, offsets, wanted_shape_len=1)
        self.set_array(self._POSITIONS_KEY, flat_positions,
                       wanted_shape_len=2, wanted_shape_1=3)
        self.set_array(self._TYPES_KEY, np.array(flat_types, dtype=str),
                       wanted_shape_len=1)
        self.set_array(self._CELL_KEY,
                       np.array([np.asarray(cell, dtype=float)
                                 for cell in cells]),
                       wanted_shape_len=3,
                       wanted_shape_1=3, wanted_shape_2=3)
        self.set_array(self._PBC_KEY, pbc, wanted_shape_len=2,
                       wanted_shape_1=3)

    def _array_kinds(self):
        """
        Every stored array that is known to be per-atom or per-frame.

        The layout's own arrays, plus whatever the setters below have
        declared.  Kept in the attributes rather than on the instance
        so that it is written out and read back with the arrays it
        describes; a file whose extra arrays came back undeclared
        could not be sliced.

        :returns: A dict of array name to ``'atom'`` or ``'frame'``.
        """
        kinds = {name: 'atom' for name in self._PER_ATOM_KEYS}
        kinds.update({name: 'frame' for name in self._PER_FRAME_KEYS})
        kinds.update(self._attrs.get(self._ARRAY_KINDS_KEY, {}))
        return kinds

    def _declare_array(self, name, kind):
        """Record that *name* is indexed by atom or by frame."""
        kinds = dict(self._attrs.get(self._ARRAY_KINDS_KEY, {}))
        kinds[name] = kind
        self.set_attr(self._ARRAY_KINDS_KEY, kinds)

    def set_per_atom_array(self, name, values):
        """
        Store an array that has one row per atom, frame by frame.

        Velocities, forces and any column read alongside the positions
        go in this way.  The frames have to be in place already, since
        it is their atom counts the entries are checked against.

        :param str name: The array name, also the name it is written
            under when a frame becomes an :class:`ase.Atoms`.
        :param values: One ``(nat, ...)`` array per frame.
        :raises ValueError: If the number of entries or the length of
            one of them does not match the frames stored.
        """
        counts = np.diff(self._arrays[self._OFFSETS_KEY])
        values = [np.asarray(value) for value in values]
        if len(values) != len(counts):
            raise ValueError(
                'Got {} entries for {} frame(s)'.format(
                    len(values), len(counts)))
        for index, (value, count) in enumerate(zip(values, counts)):
            if len(value) != count:
                raise ValueError(
                    'Entry {} of {} holds {} row(s), frame {} holds {} '
                    'atom(s)'.format(index, name, len(value), index,
                                     count))
        self.set_array(name, np.concatenate(values))
        self._declare_array(name, 'atom')

    def set_per_frame_array(self, name, values):
        """
        Store an array that has one entry per frame.

        A cell, a potential energy, a number read from a thermo file.

        :param str name: The array name, also the key it is written
            under in a frame's ``info`` when it is a single number.
        :param values: One entry per frame.
        :raises ValueError: If the number of entries does not match the
            frames stored.
        """
        values = np.asarray(values)
        if len(values) != self.nstep:
            raise ValueError(
                'Got {} entries for {} frame(s)'.format(
                    len(values), self.nstep))
        self.set_array(name, values)
        self._declare_array(name, 'frame')

    def get_frame_arrays(self, frame):
        """
        Every declared array of *frame*, beyond positions and species.

        The cells and the boundary flags are left out as well: a frame
        carries those itself, and they are not columns.

        :returns: A dict of array name to this frame's values.
        """
        skip = (self._POSITIONS_KEY, self._TYPES_KEY, self._CELL_KEY,
                self._PBC_KEY, self._OFFSETS_KEY)
        arrays = {}
        for name, kind in self._array_kinds().items():
            if name in skip or name not in self._arrays:
                continue
            if kind == 'atom':
                arrays[name] = self.get_frame_array(name, frame)
            else:
                arrays[name] = self._arrays[name][self._frame_index(frame)]
        return arrays

    def set_structures(self, atoms_list):
        """
        Replace the stored frames with *atoms_list*.

        :param atoms_list: Iterable of :class:`ase.Atoms`.
        :raises ValueError: If the iterable is empty.
        :raises TypeError: If an entry is not an :class:`ase.Atoms`.
        """
        atoms_list = list(atoms_list)
        if not atoms_list:
            raise ValueError(
                'Empty list of structures, nothing to store')
        for index, atoms in enumerate(atoms_list):
            if not isinstance(atoms, Atoms):
                raise TypeError(
                    'Entry {} is a {}, not an {}'.format(
                        index, type(atoms).__name__, Atoms))

        self.set_frames(
            [atoms.get_positions() for atoms in atoms_list],
            [atoms.get_chemical_symbols() for atoms in atoms_list],
            [np.asarray(atoms.cell, dtype=float) for atoms in atoms_list],
            pbc=np.array([np.asarray(atoms.pbc, dtype=bool)
                          for atoms in atoms_list]))

    @property
    def nstep(self):
        """
        The number of frames, or 0 if none have been stored.

        Read off the offsets rather than from
        :class:`~samos.utils.attributed_array.AttributedArray`'s step
        counter, whose rule -- the first axis of a stored array --
        counts atoms in this layout, not frames.
        """
        if self._OFFSETS_KEY not in self._arrays:
            return 0
        return len(self._arrays[self._OFFSETS_KEY]) - 1

    def _frame_index(self, frame):
        """
        Normalise a possibly negative frame index and bounds-check it.

        :raises IndexError: If *frame* is out of range.
        """
        nstep = self.nstep
        index = int(frame)
        if index < 0:
            index += nstep
        if not 0 <= index < nstep:
            raise IndexError(
                'frame {} is out of range for {} frame(s)'.format(
                    frame, nstep))
        return index

    def _frame_slice(self, frame):
        """
        The rows of a per-atom array that belong to *frame*.

        One of the two methods a differently laid out subclass has to
        override; see the class docstring.
        """
        index = self._frame_index(frame)
        offsets = self._arrays[self._OFFSETS_KEY]
        return slice(int(offsets[index]), int(offsets[index + 1]))

    def _flat(self, name):
        """
        The per-atom array *name* as ``(total_atoms, ...)``.

        Already flat in this class; the other method a differently laid
        out subclass has to override.
        """
        return self.get_array(name)

    def get_frame_array(self, name, frame):
        """
        The rows of the per-atom array *name* belonging to *frame*.

        Per-atom only -- a per-frame array such as the cells is one
        entry per frame and is not sliced this way.

        :returns: A view into the stored array, not a copy.
        """
        return self._flat(name)[self._frame_slice(frame)]

    def get_frame_positions(self, frame):
        """Positions of *frame*, shape ``(nat, 3)``, in Angstrom."""
        return self.get_frame_array(self._POSITIONS_KEY, frame)

    def get_frame_types(self, frame):
        """Chemical symbols of *frame*, one per atom."""
        return self.get_frame_array(self._TYPES_KEY, frame)

    def get_frame_natoms(self, frame):
        """The number of atoms in *frame*."""
        index = self._frame_index(frame)
        offsets = self._arrays[self._OFFSETS_KEY]
        return int(offsets[index + 1] - offsets[index])

    def get_cells(self):
        """
        The per-frame cells, shape ``(nstep, 3, 3)``, or None if none
        are stored.
        """
        if self._CELL_KEY not in self._arrays:
            return None
        return self._arrays[self._CELL_KEY]

    def get_frame_cell(self, frame):
        """The cell of *frame*, shape ``(3, 3)``, in Angstrom."""
        return self.get_cells()[self._frame_index(frame)]

    def get_frame_pbc(self, frame):
        """The periodic boundary flags of *frame*, shape ``(3,)``."""
        return self._arrays[self._PBC_KEY][self._frame_index(frame)]

    def get_frame_atoms(self, frame, ignore_calculated=False):
        """
        *frame* as a fresh :class:`ase.Atoms`.

        Symbols, positions, cell and boundary conditions, and then
        every declared array through :func:`attach_arrays`, the same
        function :meth:`~samos.trajectory.Trajectory.get_step_atoms`
        ends in -- so a frame of either container reaches a writer in
        the same shape.

        The masses are the standard ones for the symbols, there being
        no reference structure here to take them from.  Velocities
        pass through them on their way to momenta, so a frame whose
        masses were set by hand is the one case where the two
        containers part ways.

        :param bool ignore_calculated: Leave forces, energy and stress
            off the frame.
        """
        atoms = Atoms(symbols=list(self.get_frame_types(frame)),
                      positions=self.get_frame_positions(frame),
                      cell=self.get_frame_cell(frame),
                      pbc=self.get_frame_pbc(frame))
        return attach_arrays(atoms, self.get_frame_arrays(frame),
                             ignore_calculated=ignore_calculated)

    def get_species(self):
        """Sorted unique chemical symbols over every frame."""
        return np.unique(self.get_array(self._TYPES_KEY))

    @property
    def has_fixed_cell(self):
        """Whether every frame shares the same cell."""
        cells = self.get_cells()
        if cells is None:
            return True
        return bool(np.allclose(cells, cells[0]))

    @property
    def has_uniform_composition(self):
        """
        Whether every frame holds the same species in the same order.

        Analyzers use this to hoist per-species index lookups out of
        their frame loop, which is worth a lot on a long trajectory and
        is exactly what varying composition makes impossible.
        """
        counts = np.diff(self._arrays[self._OFFSETS_KEY])
        if len(counts) < 2:
            return True
        if not np.all(counts == counts[0]):
            return False
        blocks = self.get_array(self._TYPES_KEY).reshape(
            len(counts), counts[0])
        return bool(np.all(blocks == blocks[0]))

    def slice_steps(self, index):
        """
        Return a new structure list holding only the frames selected by
        *index*.  The instance this is called on is left unchanged.

        Per-atom arrays are rebuilt by concatenating the selected
        frames' slices, and the offsets are rebuilt from the selected
        atom counts, so the result is as compact as if it had been
        built from those frames in the first place.

        :param slice index: The frames to keep, e.g. ``slice(0, 50, 2)``.
        :returns: A new :class:`StructureList`.
        :raises TypeError: If *index* is not a slice, or an array is
            stored that is neither per-atom nor per-frame.
        :raises ValueError: If the slice selects no frames.
        """
        if not isinstance(index, slice):
            raise TypeError(
                'index has to be a slice, got {}'.format(type(index)))
        frames = range(self.nstep)[index]
        if not len(frames):
            raise ValueError(
                'Slicing {} frames with {} leaves nothing to '
                'analyse'.format(self.nstep, index))

        kinds = self._array_kinds()
        unknown = set(self._arrays) - set(kinds) - {self._OFFSETS_KEY}
        if unknown:
            raise TypeError(
                'Cannot slice {}: not known to be per-atom or '
                'per-frame. Store it with set_per_atom_array or '
                'set_per_frame_array.'.format(', '.join(sorted(unknown))))

        new = self.__class__()
        slices = [self._frame_slice(frame) for frame in frames]
        counts = [self.get_frame_natoms(frame) for frame in frames]
        new.set_array(self._OFFSETS_KEY,
                      np.concatenate(([0], np.cumsum(counts))).astype(int))
        for name, kind in kinds.items():
            if name not in self._arrays:
                continue
            if kind == 'atom':
                flat = self._flat(name)
                new.set_array(name, np.concatenate(
                    [flat[where] for where in slices]))
            else:
                new.set_array(name, self._arrays[name][list(frames)])
        for key, value in self._attrs.items():
            new.set_attr(key, value)
        return new

    def transform_species(self, target):
        """
        Relabel every atom in every frame as *target* in-place.

        Useful when an analysis should treat the whole set as a single
        component -- an RDF of everything against everything, say --
        without filtering by element.

        :param str target: Chemical symbol, e.g. ``'H'``.
        """
        types = self.get_array(self._TYPES_KEY)
        self.set_array(self._TYPES_KEY,
                       np.full(len(types), target, dtype=str))

    def apply_unit_conversion(self, l_conv=1.0, v_conv=1.0, f_conv=1.0,
                              e_conv=1.0, s_conv=1.0):
        """
        Scale the stored arrays to samos internal units in-place.

        Only the arrays present are touched, and only those whose
        meaning is known: a column read from a dump under its own name
        is left alone, since nothing here can say what it measures.
        The flat layout makes no difference, the factor multiplying
        every element either way, which is why
        :class:`~samos.trajectory.Trajectory` inherits this rather
        than reimplementing it.

        :param float l_conv: length factor (multiply to get Angstrom);
            applied to both positions and cell vectors
        :param float v_conv: velocity factor (multiply to get A/fs)
        :param float f_conv: force factor (multiply to get eV/A)
        :param float e_conv: energy factor (multiply to get eV)
        :param float s_conv: stress factor to eV/Angstrom^3
        """
        factors = ((l_conv, (self._POSITIONS_KEY, self._CELL_KEY)),
                   (v_conv, (self._VELOCITIES_KEY,)),
                   (f_conv, (self._FORCES_KEY,)),
                   (e_conv, (self._POT_ENER_KEY,)),
                   (s_conv, (self._STRESS_KEY,)))
        for factor, names in factors:
            if factor == 1.0:
                continue
            for name in names:
                if name in self._arrays:
                    self._arrays[name] = self._arrays[name] * factor

    def __len__(self):
        return self.nstep

    def __iter__(self):
        for frame in range(self.nstep):
            yield self.get_frame_atoms(frame)

    def __getitem__(self, frame):
        if isinstance(frame, slice):
            raise TypeError(
                'Indexing gives one frame as an ase.Atoms; a slice is '
                'not supported here')
        return self.get_frame_atoms(frame)
