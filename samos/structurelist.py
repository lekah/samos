# -*- coding: utf-8 -*-

from ase import Atoms
import numpy as np
from samos.utils.attributed_array import AttributedArray


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

    # Which stored arrays are indexed by atom and which by frame.
    # Slicing has to know: a per-atom array is cut with the offsets, a
    # per-frame array with the frame numbers, and getting it the wrong
    # way round produces a plausible-looking but corrupt result rather
    # than an error.
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

        counts = [len(atoms) for atoms in atoms_list]
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(int)
        positions = np.concatenate(
            [atoms.get_positions() for atoms in atoms_list])
        # Built from one flat list of Python strings rather than by
        # concatenating per-frame arrays: an empty frame would
        # contribute a float64 array of size zero and poison the dtype
        # of the concatenation.
        symbols = []
        for atoms in atoms_list:
            symbols.extend(atoms.get_chemical_symbols())
        types = np.array(symbols, dtype=str)
        cells = np.array([np.asarray(atoms.cell, dtype=float)
                          for atoms in atoms_list])
        pbc = np.array([np.asarray(atoms.pbc, dtype=bool)
                        for atoms in atoms_list])

        self.set_array(self._OFFSETS_KEY, offsets, wanted_shape_len=1)
        self.set_array(self._POSITIONS_KEY, positions,
                       wanted_shape_len=2, wanted_shape_1=3)
        self.set_array(self._TYPES_KEY, types, wanted_shape_len=1)
        self.set_array(self._CELL_KEY, cells, wanted_shape_len=3,
                       wanted_shape_1=3, wanted_shape_2=3)
        self.set_array(self._PBC_KEY, pbc, wanted_shape_len=2,
                       wanted_shape_1=3)

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

    def get_frame_atoms(self, frame):
        """
        *frame* as a fresh :class:`ase.Atoms`.

        Positions, symbols, cell and boundary conditions only.  On
        :class:`~samos.trajectory.Trajectory`,
        :meth:`~samos.trajectory.Trajectory.get_step_atoms` is the
        richer version that also attaches masses and a calculator.
        """
        return Atoms(symbols=list(self.get_frame_types(frame)),
                     positions=self.get_frame_positions(frame),
                     cell=self.get_frame_cell(frame),
                     pbc=self.get_frame_pbc(frame))

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

        unknown = (set(self._arrays)
                   - set(self._PER_ATOM_KEYS)
                   - set(self._PER_FRAME_KEYS)
                   - {self._OFFSETS_KEY})
        if unknown:
            raise TypeError(
                'Cannot slice {}: not known to be per-atom or '
                'per-frame. Add it to _PER_ATOM_KEYS or '
                '_PER_FRAME_KEYS.'.format(', '.join(sorted(unknown))))

        new = self.__class__()
        slices = [self._frame_slice(frame) for frame in frames]
        counts = [self.get_frame_natoms(frame) for frame in frames]
        new.set_array(self._OFFSETS_KEY,
                      np.concatenate(([0], np.cumsum(counts))).astype(int))
        for name in self._PER_ATOM_KEYS:
            if name in self._arrays:
                flat = self._flat(name)
                new.set_array(name, np.concatenate(
                    [flat[where] for where in slices]))
        for name in self._PER_FRAME_KEYS:
            if name in self._arrays:
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

    def apply_unit_conversion(self, l_conv=1.0):
        """
        Scale the stored lengths to samos internal units in-place.

        Only the arrays present are touched.  The flat layout makes no
        difference here: the factor multiplies every element either
        way, which is why
        :class:`~samos.trajectory.Trajectory` can extend this rather
        than reimplement it.

        :param float l_conv: length factor (multiply to get Angstrom);
            applied to both positions and cell vectors
        """
        if l_conv == 1.0:
            return
        for name in (self._POSITIONS_KEY, self._CELL_KEY):
            if name in self._arrays:
                self._arrays[name] = self._arrays[name] * l_conv

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
