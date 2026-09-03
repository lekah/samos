# -*- coding: utf-8 -*-

import numpy as np
from ase.geometry import minkowski_reduce
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from samos.trajectory import Trajectory
from samos.utils.attributed_array import AttributedArray

import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from warnings import warn


def get_cell(trajectory, frame=None):
    """
    Cell of *frame*, or the trajectory's fixed cell if it stores none
    per frame.

    ``ase.cell.Cell`` only grew its ``.array`` attribute in ase 3.18;
    ``.copy()`` is the fallback for anything older.
    """
    cells = trajectory.get_cells()
    if cells is not None and frame is not None:
        return cells[frame]
    atoms = trajectory.get_atoms()
    try:
        return atoms.cell.array
    except AttributeError:
        return atoms.cell.copy()


class MinimumImage:
    """
    Shortest periodic distances and displacement vectors for one cell.

    Two wrong-but-tempting schemes preceded this one, and both used to
    live in this module:

    * Wrapping each fractional component to (-0.5, 0.5] picks one
      periodic image without ever comparing it to the others.  It is
      exact only for a rectangular cell; for an fcc primitive cell it
      gets a quarter of all pairs wrong, by up to 1.8 A.
    * Testing the eight corners of the cell does compare, and is exact
      for a reduced cell, but a strongly skewed cell puts the nearest
      image further than one cell vector away, out of reach of those
      eight.

    Minkowski-reducing the cell first makes the nearest image one of
    the 27 immediate neighbours, so testing those is exact for any
    cell.  The reduction is the expensive part, which is why it happens
    once here rather than once per call.  See ase.geometry.find_mic,
    which is correct in the same way but redoes the reduction on every
    call and so runs several times slower in a loop.

    Like any minimum-image scheme, this is only meaningful for
    distances below half the cell's narrowest width.
    """

    # Offsets of the 27 immediate neighbour cells, origin included.
    _OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

    def __init__(self, cell):
        cell = np.asarray(cell, dtype=float)
        reduced, _ = minkowski_reduce(cell)
        self._cell = reduced
        self._cellI = np.linalg.inv(reduced)
        self._images = self._OFFSETS @ reduced
        lengths = np.linalg.norm(self._images, axis=1)
        self._shortest_lattice_vector = lengths[lengths > 0.0].min()

    @property
    def max_radius(self):
        """
        Largest distance at which a minimum-image analysis is unbiased.

        Two periodic copies of the same atom are separated by a lattice
        vector t, so both can sit within r of another atom as soon as
        |t| <= 2r.  Beyond half the shortest lattice vector, a pair
        therefore has more than one image in range while minimum-image
        counting keeps only the nearest, and histograms come out low.

        The shortest lattice vector is read off the Minkowski-reduced
        basis, so this is a property of the lattice and not of the cell
        the caller happened to supply.  It is a weaker limit than the
        familiar half-narrowest-width rule, which exists to keep naive
        wrapping honest rather than to keep counting honest.
        """
        return 0.5 * self._shortest_lattice_vector

    def _wrapped(self, diff):
        return ((np.asarray(diff) @ self._cellI) % 1.0) @ self._cell

    def distances(self, diff):
        """Shortest periodic distance for each row of *diff*."""
        return cdist(self._wrapped(diff), self._images).min(axis=1)

    def vectors(self, diff):
        """
        Shortest periodic displacement vector for each row of *diff*.

        Ties -- a displacement of exactly half a cell vector -- resolve
        to the first of the equidistant images, which is the one with
        the most negative offset.
        """
        wrapped = self._wrapped(diff)
        closest = cdist(wrapped, self._images).argmin(axis=1)
        return wrapped - self._images[closest]


def is_orthorhombic(cell, rtol=1e-10):
    """
    Whether *cell* is diagonal with positive edge lengths.

    A cell that came out of a relaxation carries rounding junk off the
    diagonal, so an exact test would send almost every "cubic" cell
    down the slow path.  *rtol* is relative to the largest entry of the
    cell: treating an off-diagonal of that size as zero displaces a
    position by at most ``rtol`` times the cell size, which is 5e-9 A
    for the default on a 50 A cell.
    """
    cell = np.asarray(cell, dtype=float)
    off_diagonal = cell - np.diag(np.diag(cell))
    scale = np.abs(cell).max()
    return bool(np.all(np.abs(off_diagonal) <= rtol * scale)
                and np.all(np.diag(cell) > 0.0))


def _wrap_into_box(positions, edges):
    """
    Fold *positions* into ``[0, edge)`` along each axis.

    ``positions % edges`` is not enough on its own: for a coordinate a
    hair below zero the remainder rounds up to exactly the edge, and
    :class:`scipy.spatial.cKDTree` then rejects the whole array with
    "some input data are greater than the size of the periodic box".
    Unwrapped MD coordinates hit this routinely.
    """
    wrapped = np.asarray(positions, dtype=float) % edges
    wrapped[wrapped >= edges] = 0.0
    return wrapped


def pairs_within(positions_1, positions_2, radius, algorithm,
                 cell=None, mic=None, wrapped=False):
    """
    Every periodic pair closer than *radius*, and how far apart it is.

    Two algorithms, giving the same answer:

    ``'ortho'``
        A pair of k-d trees over the periodic box.  Only builds the
        pairs that are actually within *radius*, so it costs O(N) in
        both time and memory, but it needs an orthorhombic *cell*.
    ``'skew'``
        Every pair against all 27 neighbour images via *mic*.  Exact
        for any cell shape, and O(N^2) in both time and memory.

    Both report the distance to the *nearest* periodic image only, so
    both are biased low beyond ``MinimumImage.max_radius`` and in the
    same way -- see :meth:`BaseAnalyzer._check_radius`.

    :param bool wrapped:
        For ``'ortho'`` only: skip folding *positions_1* and
        *positions_2* into the box, because the caller already did so.
        A caller that queries the same frame for several species pairs
        can fold every atom once and pass the same wrapped array in
        for each pair, rather than this function folding the same
        atoms again for every pair that touches them.  Ignored for
        ``'skew'``, which never wraps -- :class:`MinimumImage` finds
        the nearest periodic image directly from unwrapped positions.

    :returns:
        ``(i, j, d)``.  *i* indexes *positions_1* and *j* indexes
        *positions_2*, both locally; *d* holds the distances.  Pairs of
        an atom with itself are not filtered out here, because whether
        two rows are the same atom is the caller's bookkeeping.
    """
    if algorithm == 'ortho':
        edges = np.diag(np.asarray(cell, dtype=float))
        p1 = positions_1 if wrapped else _wrap_into_box(positions_1, edges)
        p2 = positions_2 if wrapped else _wrap_into_box(positions_2, edges)
        tree_1 = cKDTree(p1, boxsize=edges)
        tree_2 = cKDTree(p2, boxsize=edges)
        found = tree_1.sparse_distance_matrix(
            tree_2, radius, output_type='ndarray')
        return found['i'], found['j'], found['v']

    if algorithm != 'skew':
        raise ValueError(
            "algorithm must be 'ortho' or 'skew', got {!r}".format(algorithm))

    n_1, n_2 = len(positions_1), len(positions_2)
    i, j = np.divmod(np.arange(n_1 * n_2), n_2)
    d = mic.distances(np.asarray(positions_2)[j] - np.asarray(positions_1)[i])
    close = d <= radius
    return i[close], j[close], d[close]


class BaseAnalyzer(metaclass=ABCMeta):
    def __init__(self, *, trajectory=None, **kwargs):
        """
        :param Trajectory trajectory: The trajectory to analyse.
        :raises TypeError: On an unrecognised keyword argument.
        """
        self._trajectory = None
        if kwargs:
            raise TypeError(
                '{} got unexpected keyword argument(s): {}'.format(
                    type(self).__name__, ', '.join(sorted(kwargs))))
        if trajectory is not None:
            self.set_trajectory(trajectory)

    def set_trajectory(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError(
                'You need to pass a {} as trajectory'.format(Trajectory))
        self._trajectory = trajectory

    @property
    def trajectory(self):
        """
        The trajectory set with :meth:`set_trajectory`.

        :raises ValueError:
            If none was set.  Reading ``self._trajectory`` directly used
            to give an AttributeError from deep inside :meth:`run`.
        """
        if self._trajectory is None:
            raise ValueError(
                'No trajectory has been set. Use the set_trajectory '
                'method, or pass trajectory=... to the constructor.')
        return self._trajectory

    def _check_radius(self, max_radius, radius, what='radius'):
        """
        Warn, at most once per run, if *radius* is too large for the
        cell to support unbiased minimum-image counting.

        A warning rather than an error: the result is biased low past
        this point, not meaningless, and callers have been computing
        RDFs out to it for years.

        Takes *max_radius* as a number rather than a
        :class:`MinimumImage`, because the orthorhombic path knows it
        is half the shortest cell edge and so never builds one.
        """
        if radius <= max_radius:
            return
        if getattr(self, '_radius_warned', False):
            return
        self._radius_warned = True
        warn('{} of {:.3f} A exceeds half the shortest lattice vector '
             '({:.3f} A). Beyond that, an atom has more than one '
             'periodic image in range and only the nearest is counted, '
             'so the result is biased low. Use a larger cell or a '
             'smaller radius.'.format(what, radius, max_radius),
             stacklevel=3)

    @staticmethod
    def _choose_algorithm(cells, frames, method):
        """
        Pick ``'ortho'`` or ``'skew'`` for each sampled frame.

        The choice is made from the cells alone, before any distance is
        computed, so ``method='ortho'`` on a skewed cell fails at once
        rather than partway through a long trajectory.

        :param cells: ``(nstep, 3, 3)`` array, or a single ``(3, 3)``
            cell that applies to every frame.
        :param frames: Indices of the frames that will be sampled.
        :param str method: ``'auto'``, ``'ortho'`` or ``'skew'``.
        :returns:
            ``(algorithms, message)`` -- one algorithm name per entry of
            *frames*, and the line to print saying what was chosen.
        """
        if method not in ('auto', 'ortho', 'skew'):
            raise ValueError(
                "method must be 'auto', 'ortho' or 'skew', got "
                "{!r}".format(method))

        if method == 'skew':
            # No need to look at the cells at all.
            return (np.full(len(frames), 'skew'),
                    'using the slower skew algorithm, because it was '
                    'requested.')

        cells = np.asarray(cells, dtype=float)
        if cells.ndim == 2:
            # One cell for the whole trajectory, so one test for it.
            ortho = np.full(len(frames), is_orthorhombic(cells))
        else:
            ortho = np.array([is_orthorhombic(cells[frame])
                              for frame in frames])
        if method == 'ortho':
            if not ortho.all():
                raise ValueError(
                    "method='ortho' needs a cell that is diagonal with "
                    'positive edges, but {} of the {} sampled frames do '
                    'not have one. Use method=\'auto\' or '
                    "method='skew'.".format(
                        int((~ortho).sum()), len(frames)))
            return (np.full(len(frames), 'ortho'),
                    'using the faster ortho algorithm, because it was '
                    'requested.')

        algorithms = np.where(ortho, 'ortho', 'skew')
        if ortho.all():
            message = ('using the faster ortho algorithm, because the '
                       'cell was detected as orthorhombic.')
        elif not ortho.any():
            message = ('using the slower skew algorithm, because the '
                       'cell is not orthorhombic.')
        else:
            message = ('using the faster ortho algorithm for {} of {} '
                       'frames and the slower skew algorithm for the '
                       'rest, because the cell shape changes.'.format(
                           int(ortho.sum()), len(frames)))
        return algorithms, message

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class RDF(BaseAnalyzer):
    def run(self, radius, species_pairs=None,
            istart=0, istop=None, stepsize=1, nbins=100, method='auto'):
        """
        Calculate a RDF, also searching periodic images.

        Each sampled frame is normalised by its own number density and
        the results are averaged, so g(r) is the mean of the per-frame
        g(r).  For a fixed cell this is the same as normalising the
        summed histogram once; for a variable cell it is not, and the
        alternatives (normalising by the mean volume, or by the mean
        density) differ from this one by the covariance between the
        pair count and the volume.

        The ideal-gas reference an atom is compared against excludes
        the atom itself, so a species paired with itself is normalised
        by N-1 rather than N.  Without that, g(r) of a like pair tends
        to (N-1)/N at large r instead of 1.

        The ``int_*`` arrays are running neighbour counts and carry no
        volume factor, so they are unaffected by either of the above.
        Each entry sums whole bins, so it is the count inside the outer
        edge of that bin, half a binsize beyond the matching entry of
        ``radii_*``, which holds bin centres.

        Distances come from :func:`pairs_within`, which reports the
        distance to the nearest periodic image whichever algorithm it
        uses.  A *radius* beyond ``MinimumImage.max_radius`` biases
        g(r) low -- see there -- and warns.

        :param float radius:
            The radius for the calculation of the RDF
        :param int istart: where to start sampling
        :param int istop: where to stop sampling (trajectory index)
        :param int stepsize:
            Sampling steps to take, defaults to 1.
            A stepsize=10 takes every 10th step for trajectory calculation.
        :param in binsize:
            Number of bins to use in histogram, defaults to 100.
            Increasing this means finer resolution but also more noise.
        :param str method:
            Which pair-finding algorithm to use: ``'auto'`` (default)
            takes ``'ortho'`` wherever the cell allows it, ``'ortho'``
            demands it and raises if the cell is skewed, and ``'skew'``
            forces the exact-for-any-cell path even when the cell is
            orthorhombic.  The two agree to rounding, so ``'skew'`` is
            useful as a check on ``'ortho'``.  See :func:`pairs_within`.
        :raises ValueError:
            On an unknown *method*, or ``method='ortho'`` with a cell
            that is not diagonal with positive edges.
        """
        def get_indices(spec, chem_sym):
            """
            get the indices for specification spec
            """
            if isinstance(spec, str):
                return np.where(chem_sym == spec)[0].tolist()
            elif isinstance(spec, int):
                return [spec]
            elif isinstance(spec, (tuple, list)):
                list_ = []
                for item in spec:
                    list_ += get_indices(item, chem_sym)
                return list_
            else:
                raise TypeError(
                    '{} can not be transformed to index'.format(spec))

        def get_label(spec, ispec):
            """
            Get a good label for specification spec. If none can be found
            give one based on iteration counter ispec
            """
            if isinstance(spec, str):
                return spec
            elif isinstance(spec, (tuple, list)):
                return 'spec_{}'.format(ispec)
            else:
                print(type(spec))

        positions = self.trajectory.get_positions()
        types = self.trajectory.get_types()
        cells = self.trajectory.get_cells()
        self._radius_warned = False
        fixed_cell = cells is None
        if fixed_cell:
            fixed_volume = self.trajectory.atoms.get_volume()

        if istop is None:
            istop = len(positions)
        elif istop > len(positions):
            raise ValueError('Istop ({}) is higher than the number of '
                             'positions ({})'.format(
                                 istop, len(positions)))
        frames = np.arange(istart, istop, stepsize)

        algorithms, message = self._choose_algorithm(
            get_cell(self.trajectory) if fixed_cell else cells,
            frames, method)
        print('RDF: ' + message)

        if species_pairs is None:
            species_pairs = sorted(list(
                itertools.combinations_with_replacement(
                    sorted(set(types)), 2)))
        indices_pairs = []
        labels = []
        species_pairs_pruned = []
        for ispec, (spec1, spec2) in enumerate(species_pairs):
            ind_spec1, ind_spec2 = (get_indices(spec1, types),
                                    get_indices(spec2, types))
            # special situation if there's only one atom of a species
            # and we're making the RDF of that species with itself.
            # there will be empty pairs_of_atoms and the
            # code below would crash!
            if ind_spec1 == ind_spec2 and len(ind_spec1) == 1:
                continue
            indices_pairs.append((ind_spec1, ind_spec2))
            labels.append('{}_{}'.format(
                get_label(spec1, ispec), get_label(spec2, ispec)))
            species_pairs_pruned.append((spec1, spec2))
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs_pruned)
        binsize = float(radius)/nbins
        bin_edges = np.histogram([], bins=nbins, range=(0, radius))[1]

        # Everything that does not change from frame to frame, worked
        # out once per species pair.  n_pairs is counted rather than
        # enumerated: listing every index pair is what used to make
        # this function quadratic in memory even before any distance
        # was computed.
        plan = []
        for label, (ind1, ind2) in zip(labels, indices_pairs):
            same_list = ind1 == ind2
            if same_list:
                # Each unordered pair once, then counted from both
                # ends by the factor of two.
                n_pairs = len(ind1) * (len(ind1) - 1) // 2
                pair_factor = 2.0
            else:
                n_pairs = (len(ind1) * len(ind2)
                           - len(set(ind1) & set(ind2)))
                pair_factor = 1.0
            if not n_pairs:
                # e.g. a species that is absent from the trajectory.
                # Skipping keeps the remaining pairs computable; the
                # histogram below would be empty and its normalisation
                # a division by zero.
                print('Warning: no atom pairs for {}, skipping'
                      ''.format(label))
                continue
            # An atom of species 1 that is itself one of the species-2
            # atoms is not its own neighbour, so the ideal-gas count it
            # is compared against is len(ind2) minus the chance of that
            # coincidence.  For a pair of a species with itself this is
            # the familiar N-1; for disjoint species it is len(ind2).
            n_neighbours_ideal = (
                len(ind2)
                - len(set(ind1) & set(ind2)) / float(len(ind1)))
            if n_neighbours_ideal <= 0:
                print('Warning: no ideal-gas reference for {}, skipping'
                      ''.format(label))
                continue
            # normalize the histogram, by the number of steps taken,
            # and the number of species1
            prefactor = pair_factor / float(len(frames)) / float(len(ind1))
            plan.append(dict(
                label=label, ind1=np.asarray(ind1), ind2=np.asarray(ind2),
                same_list=same_list, n_pairs=n_pairs, prefactor=prefactor,
                n_neighbours_ideal=n_neighbours_ideal,
                hist=np.zeros(nbins, dtype=float),
                # Second accumulator, weighted by each frame's volume,
                # so that the g(r) below is the mean of the per-frame
                # g(r) rather than one histogram divided by a single
                # volume.  hist itself stays a plain neighbour count,
                # which is what the running integral reports.
                hist_by_density=np.zeros(nbins, dtype=float),
                shortest=np.inf))

        # Frames outermost so that the cell work -- the Minkowski
        # reduction above all, which the MinimumImage docstring calls
        # the expensive part -- happens once per frame instead of once
        # per frame per species pair.
        def prepare_cell(cell, algorithm):
            """Cell-dependent work: the reduction, and the radius check.

            Only the skew path needs a MinimumImage; for an
            orthorhombic cell the shortest lattice vector is the
            shortest edge, so the ortho path gets its radius limit
            without reducing anything.
            """
            mic = None if algorithm == 'ortho' else MinimumImage(cell)
            max_radius = (0.5 * np.diag(cell).min() if algorithm == 'ortho'
                          else mic.max_radius)
            self._check_radius(max_radius, radius, 'RDF radius')
            return mic

        if fixed_cell:
            cell, volume = get_cell(self.trajectory), fixed_volume
            mic = prepare_cell(cell, algorithms[0])

        for iframe, index in enumerate(frames):
            algorithm = algorithms[iframe]
            if not fixed_cell:
                cell = cells[index]
                volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                mic = prepare_cell(cell, algorithm)

            # Every species pair that shares a species would otherwise
            # fold that species' atoms into the box again for each
            # pair it appears in.  Folding the whole frame once here
            # instead means every atom is wrapped exactly once per
            # frame, however many pairs it takes part in.
            if algorithm == 'ortho':
                wrapped_positions = _wrap_into_box(
                    positions[index], np.diag(cell))

            for entry in plan:
                ind1, ind2 = entry['ind1'], entry['ind2']
                if algorithm == 'ortho':
                    pos1 = wrapped_positions[ind1]
                    pos2 = wrapped_positions[ind2]
                else:
                    pos1 = positions[index, ind1, :]
                    pos2 = positions[index, ind2, :]
                i, j, distances = pairs_within(
                    pos1, pos2, radius, algorithm, cell=cell, mic=mic,
                    wrapped=(algorithm == 'ortho'))
                # An atom is not its own neighbour.  For a species
                # against itself, i < j additionally keeps each
                # unordered pair once, which pair_factor doubles back.
                global_i, global_j = ind1[i], ind2[j]
                keep = (global_i < global_j if entry['same_list']
                        else global_i != global_j)
                distances = distances[keep]
                if len(distances):
                    entry['shortest'] = min(entry['shortest'],
                                            distances.min())
                counts = entry['prefactor'] * np.histogram(
                    distances, bins=nbins, range=(0, radius))[0]
                entry['hist'] += counts
                entry['hist_by_density'] += counts * volume

        radii = 0.5*(bin_edges[:-1]+bin_edges[1:])
        shortest_distance_all = np.inf
        for entry in plan:
            label = entry['label']
            rdf = (entry['hist_by_density']
                   / (4.0 * np.pi * radii**2 * binsize)
                   / entry['n_neighbours_ideal'])
            integral = np.cumsum(entry['hist'])

            rdf_res.set_array('rdf_{}'.format(label), rdf)
            rdf_res.set_array('int_{}'.format(label), integral)
            rdf_res.set_array('radii_{}'.format(label), radii)
            rdf_res.set_attr('n_pairs_{}'.format(label), entry['n_pairs'])
            rdf_res.set_attr('n_data_{}'.format(label),
                             entry['n_pairs'] * ((istop-istart)//stepsize))
            rdf_res.set_attr('shortest_distance_{}'.format(label),
                             float(entry['shortest']))
            shortest_distance_all = min(shortest_distance_all,
                                        entry['shortest'])
        # One global value, after every pair has contributed.  This used
        # to be written inside the loop, so it held the running minimum
        # over the pairs seen so far under a name that reads per-pair.
        rdf_res.set_attr('shortest_distance', float(shortest_distance_all))
        return rdf_res


class BondAnalyzer(BaseAnalyzer):
    """
    Base class for analyzers that need bond topology.

    Bond topology can come from three sources, checked in priority order
    by get_bonds():

      1. Explicit list set via set_bonds() or load_bonds_lammps() --
         static, used as-is for every frame.
      2. Per-frame detection via a 'bonds' cutoff dict passed to run() --
         re-evaluated each frame by default.
      3. One-time detection from the first sampled frame when
         static_bonds=True is passed to run() -- result is frozen via
         set_bonds() before the main loop.
    """

    def __init__(self, *, bonds=None, **kwargs):
        """
        :param array-like bonds: Explicit bond topology, shape
            (N_bonds, 2), 0-based atom indices.  See :meth:`set_bonds`.
        """
        self._bonds = None
        super().__init__(**kwargs)
        if bonds is not None:
            self.set_bonds(bonds)

    def set_bonds(self, bonds):
        """
        Set explicit bond topology.

        :param array-like bonds:
            Shape (N_bonds, 2), 0-based atom indices.  Canonical ordering
            (i < j) is enforced and duplicate entries are removed.
        """
        arr = np.asarray(bonds, dtype=int)
        arr = np.sort(arr, axis=1)
        self._bonds = np.unique(arr, axis=0)

    def load_bonds_lammps(self, path):
        """
        Read bond topology from the Bonds section of a LAMMPS data file.

        Atom indices in the file are 1-based; stored here as 0-based.
        Bond type (column 2 of the Bonds section) is ignored.

        :param str path: path to the LAMMPS data file.
        """
        bonds = []
        in_bonds = False
        with open(path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                # Section header: first token is all letters.
                first = stripped.split()[0]
                if first.isalpha():
                    if stripped.lower().startswith('bonds'):
                        in_bonds = True
                    elif in_bonds:
                        break  # different section -- Bonds block ended
                    continue
                if not in_bonds:
                    continue
                parts = stripped.split()
                if len(parts) >= 4:
                    # Format: bond_id  bond_type  atom_i  atom_j
                    i = int(parts[2]) - 1
                    j = int(parts[3]) - 1
                    bonds.append((min(i, j), max(i, j)))
        if bonds:
            self._bonds = np.array(sorted(set(bonds)), dtype=int)
        else:
            self._bonds = np.empty((0, 2), dtype=int)

    def get_bonds(self, cutoffs=None, frame=None):
        """
        Return bond list as an (N_bonds, 2) integer array.

        Explicit stored bonds take priority over cutoff detection.

        :param dict cutoffs:
            {'Spec1-Spec2': (r_min, r_max)} -- required when no explicit
            bonds have been set.
        :param int frame:
            Trajectory frame index for on-the-fly detection.  Required
            when cutoffs is not None and no stored bonds are available.
        """
        if self._bonds is not None:
            return self._bonds
        if cutoffs is not None:
            if frame is None:
                raise ValueError(
                    "frame must be provided for cutoff-based bond "
                    "detection.")
            return self._detect_bonds(cutoffs, frame)
        raise ValueError(
            "No bond topology available.  Call set_bonds(), "
            "load_bonds_lammps(), or pass bonds= cutoffs to run().")

    @staticmethod
    def _parse_cutoffs(cutoffs):
        """
        Normalize cutoff dict from 'A-B' string keys to ('A','B') tuples.
        """
        result = {}
        for key, val in cutoffs.items():
            parts = key.split('-')
            if len(parts) != 2:
                raise ValueError(
                    "Bond key must be 'Spec1-Spec2', "
                    "got '{}'.".format(key))
            if len(val) != 2:
                raise ValueError(
                    "Cutoff for '{}' must be (r_min, r_max), "
                    "got {}.".format(key, val))
            result[tuple(parts)] = (float(val[0]), float(val[1]))
        return result

    @staticmethod
    def _lookup_cutoff(cutoffs_parsed, spec1, spec2):
        """
        Return (r_min, r_max) for bond spec1-spec2, accepting either
        ordering.  Raises ValueError if the bond type is undefined.
        """
        if (spec1, spec2) in cutoffs_parsed:
            return cutoffs_parsed[(spec1, spec2)]
        if (spec2, spec1) in cutoffs_parsed:
            return cutoffs_parsed[(spec2, spec1)]
        raise ValueError(
            "No cutoff defined for bond '{}-{}'.  "
            "Available: {}.".format(
                spec1, spec2,
                ', '.join('-'.join(k) for k in cutoffs_parsed)))

    def _detect_bonds(self, cutoffs, frame):
        """
        Detect bonds in a single frame via distance cutoffs.

        Returns an (N_bonds, 2) int array with canonical i < j ordering.
        A set is used internally to deduplicate same-species pairs.
        """
        cutoffs_parsed = self._parse_cutoffs(cutoffs)
        positions = self.trajectory.get_positions()[frame]
        mic = MinimumImage(get_cell(self.trajectory, frame))
        self._check_radius(mic.max_radius,
                           max(r for _, r in cutoffs_parsed.values()),
                           'Bond cutoff')
        types = self.trajectory.get_types()

        bond_set = set()
        for (sp1, sp2), (r_min, r_max) in cutoffs_parsed.items():
            ind1 = np.where(types == sp1)[0]
            ind2 = np.where(types == sp2)[0]
            for i in ind1:
                dists = mic.distances(positions[ind2] - positions[i])
                for j in ind2[(dists >= r_min) & (dists <= r_max)]:
                    ji = int(j)
                    if ji != i:
                        bond_set.add((min(i, ji), max(i, ji)))

        if bond_set:
            return np.array(sorted(bond_set), dtype=int)
        return np.empty((0, 2), dtype=int)

    @staticmethod
    def _build_adjacency(bond_array):
        """
        Build {atom_idx: set(bonded_atom_indices)} from a canonical
        (i < j) bond array.  Adjacency is stored bidirectionally so
        both directions can be looked up in O(1).
        """
        adjacency = defaultdict(set)
        for i, j in bond_array:
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))
        return adjacency


class ADF(BondAnalyzer):
    """
    Angular distribution function.

    Computes the distribution of bond angles A-B-C where B is the center
    atom.  Neighbors are identified via bond topology (explicit or
    cutoff-detected); the angle is the one subtended at B.
    """

    def run(self, species_triplets=None, centers=None,
            istart=0, istop=None, stepsize=1, nbins=100,
            bonds=None, static_bonds=False):
        """
        Compute the angular distribution function.

        :param list species_triplets:
            List of (spec_left, spec_center, spec_right) tuples.
            Mutually exclusive with centers.
        :param list centers:
            Center species symbols.  Expands to all (*,center,*)
            combinations for species present in the trajectory.
            Mutually exclusive with species_triplets.
        :param int istart: first frame index (inclusive).
        :param int istop:
            Last frame index (exclusive).  Defaults to trajectory end.
        :param int stepsize: frame stride.
        :param int nbins: number of bins over [0, 180] degrees.
        :param dict bonds:
            Per-bond cutoffs {'Spec1-Spec2': (r_min, r_max)}.  Used for
            on-the-fly neighbor detection; ignored if explicit topology
            was already set via set_bonds() or load_bonds_lammps().
        :param bool static_bonds:
            When True and bonds cutoffs are given (no explicit topology
            set), detect topology once from the first sampled frame and
            reuse it for all frames.
        :return:
            AttributedArray with arrays adf_A_B_C (normalized, per
            center atom per frame per degree) and angles_A_B_C (bin
            centres in degrees) for each triplet.
        :rtype: AttributedArray
        """
        if species_triplets is not None and centers is not None:
            raise ValueError(
                "Pass species_triplets or centers, not both.")

        self._radius_warned = False
        positions = self.trajectory.get_positions()
        types = self.trajectory.get_types()
        cells = self.trajectory.get_cells()

        if istop is None:
            istop = len(positions)
        frames = np.arange(istart, istop, stepsize)
        if not len(frames):
            raise ValueError(
                'No frames selected: istart={}, istop={}, stepsize={} '
                'over a trajectory of {} step(s).'.format(
                    istart, istop, stepsize, len(positions)))

        # One-time detection: freeze topology from the first sampled frame.
        if static_bonds and bonds is not None and self._bonds is None:
            self.set_bonds(self._detect_bonds(bonds, int(frames[0])))

        use_static = self._bonds is not None
        if use_static:
            static_adjacency = self._build_adjacency(self._bonds)
        elif bonds is None:
            raise ValueError(
                "No bond topology available.  Provide bonds= cutoffs "
                "or call set_bonds() / load_bonds_lammps() first.")

        # Build species triplet list.
        all_species = sorted(set(types))
        if centers is not None:
            species_triplets = [
                (s1, c, s2)
                for c in centers
                for s1, s2 in itertools.combinations_with_replacement(
                    all_species, 2)
            ]
        elif species_triplets is None:
            species_triplets = [
                (s1, c, s2)
                for c in all_species
                for s1, s2 in itertools.combinations_with_replacement(
                    all_species, 2)
            ]

        # Precompute per-species index arrays (same for every frame).
        triplet_idx = {
            (sl, sc, sr): (
                np.where(types == sl)[0],
                np.where(types == sc)[0],
                np.where(types == sr)[0],
            )
            for sl, sc, sr in species_triplets
        }

        binsize = 180.0 / nbins
        bin_centers = (np.arange(nbins) + 0.5) * binsize
        hists = {t: np.zeros(nbins, dtype=float) for t in species_triplets}

        # Reduce the cell only once for fixed-cell trajectories.
        if cells is None:
            fixed_cell = True
            mic = MinimumImage(get_cell(self.trajectory))
        else:
            fixed_cell = False

        for frame in frames:
            if not fixed_cell:
                mic = MinimumImage(cells[frame])
            pos = positions[frame]

            # Bond detection is per-frame for dynamic mode; static mode
            # reuses the frozen adjacency built before the loop.
            if use_static:
                adjacency = static_adjacency
            else:
                adjacency = self._build_adjacency(
                    self._detect_bonds(bonds, int(frame)))

            for sl, sc, sr in species_triplets:
                idx_l, idx_c, idx_r = triplet_idx[(sl, sc, sr)]
                if not len(idx_c):
                    continue
                same = (sl == sr)

                for j in idx_c:
                    bonded = adjacency[j]
                    left = [i for i in idx_l
                            if i in bonded and i != j]
                    right = (left if same else
                             [k for k in idx_r
                              if k in bonded and k != j])
                    if not left or not right:
                        continue

                    r_l = mic.vectors(pos[left] - pos[j])
                    r_r = r_l if same else mic.vectors(
                        pos[right] - pos[j])

                    nl = np.linalg.norm(r_l, axis=1)
                    nr = np.linalg.norm(r_r, axis=1)
                    if np.any(nl == 0) or np.any(nr == 0):
                        continue

                    cos_t = (r_l @ r_r.T) / (
                        nl[:, None] * nr[None, :])
                    np.clip(cos_t, -1.0, 1.0, out=cos_t)
                    theta = np.degrees(np.arccos(cos_t))

                    if same:
                        n = len(left)
                        # Upper triangle avoids counting (i,k) and (k,i)
                        # as separate pairs for same-species neighbors.
                        mask = np.triu(
                            np.ones((n, n), dtype=bool), k=1)
                        theta = theta[mask]

                    hists[(sl, sc, sr)] += np.histogram(
                        theta.ravel(), bins=nbins,
                        range=(0.0, 180.0))[0]

        res = AttributedArray()
        res.set_attr('species_triplets', species_triplets)
        n_frames = len(frames)

        for sl, sc, sr in species_triplets:
            _, idx_c, _ = triplet_idx[(sl, sc, sr)]
            if not len(idx_c):
                continue
            norm = float(len(idx_c)) * float(n_frames) * binsize
            label = '{}_{}_{}'.format(sl, sc, sr)
            res.set_array('adf_{}'.format(label),
                          hists[(sl, sc, sr)] / norm)
            res.set_array('angles_{}'.format(label), bin_centers)

        return res


class TorsionAnalyzer(BondAnalyzer):
    """Torsion (dihedral) angle distribution.  Not yet implemented."""

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "TorsionAnalyzer is not yet implemented.")


def pairs_with_other_species(trajectory, species):
    """
    Build the ``(requested, other)`` species pairs for an RDF.

    Every entry of *species* is paired with every distinct species
    present in *trajectory* that was not itself requested.

    Deduplicating matters: both call sites used to iterate the per-atom
    symbol list rather than the set of species, so a hundred-atom cell
    produced a hundred identical copies of each pair, every one of them
    computed from scratch.

    :param trajectory: :class:`~samos.trajectory.Trajectory` to inspect.
    :param list species: Chemical symbols of interest.
    :returns: List of ``(spec, other)`` tuples, deterministically ordered.
    """
    others = sorted(set(trajectory.get_types()) - set(species))
    return [(spec, other) for spec in species for other in others]
