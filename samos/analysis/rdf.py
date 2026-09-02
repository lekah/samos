# -*- coding: utf-8 -*-

import numpy as np
from ase.geometry import minkowski_reduce
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


class BaseAnalyzer(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._trajectory = None
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)

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

    def _check_radius(self, mic, radius, what='radius'):
        """
        Warn, at most once per run, if *radius* is too large for the
        cell to support unbiased minimum-image counting.

        A warning rather than an error: the result is biased low past
        this point, not meaningless, and callers have been computing
        RDFs out to it for years.
        """
        if radius <= mic.max_radius:
            return
        if getattr(self, '_radius_warned', False):
            return
        self._radius_warned = True
        warn('{} of {:.3f} A exceeds half the shortest lattice vector '
             '({:.3f} A). Beyond that, an atom has more than one '
             'periodic image in range and only the nearest is counted, '
             'so the result is biased low. Use a larger cell or a '
             'smaller radius.'.format(what, radius, mic.max_radius),
             stacklevel=3)

    @abstractmethod
    def run(self, *args, **kwargs):
        pass


class RDF(BaseAnalyzer):
    def run_fort(self, radius=None, species_pairs=None, istart=0,
                 istop=None, stepsize=1, nbins=100):
        """
        :param float radius:
            The radius for the calculation of the RDF
        """
        if 1:
            raise NotImplementedError('This is not fully implemented')
        from samos.lib.rdf import calculate_rdf
        atoms = self.trajectory.atoms
        volume = atoms.get_volume()
        positions = self.trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(
                set(atoms.get_chemical_symbols()), 2))
        # Transposed, unlike AngularSpectrum below -- one of the two
        # is wrong, but run_fort raises before reaching here.
        cell = get_cell(self.trajectory).T
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        for spec1,  spec2 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[
                0] + 1  # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            density = float(len(ind2)) / volume
            rdf, integral, radii = calculate_rdf(
                positions, istart, istop, stepsize,
                radius, density, cell,
                cellI, ind1, ind2, nbins)
            rdf_res.set_array('rdf_{}_{}'.format(spec1, spec2), rdf)
            rdf_res.set_array('int_{}_{}'.format(spec1, spec2), integral)
            rdf_res.set_array('radii_{}_{}'.format(spec1, spec2), radii)
        return rdf_res

    def run(self, radius, species_pairs=None,
            istart=0, istop=None, stepsize=1, nbins=100):
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

        Distances use :class:`MinimumImage`, which is exact for any
        cell shape.  A *radius* beyond ``MinimumImage.max_radius``
        still biases g(r) low -- see there -- and warns.

        TODO: Implement orthorhombic case to gain efficiency

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
        if cells is None:
            fixed_cell = True
            volume = self.trajectory.atoms.get_volume()
            mic = MinimumImage(get_cell(self.trajectory))
            self._check_radius(mic, radius, 'RDF radius')
        else:
            fixed_cell = False

        if istop is None:
            istop = len(positions)
        elif istop > len(positions):
            raise ValueError('Istop ({}) is higher than the number of '
                             'positions ({})'.format(
                                 istop, len(positions)))
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

        # wrapping the positions:
        shortest_distance_all = np.inf
        for label, (ind1, ind2) in zip(labels, indices_pairs):
            shortest_distance_this_pair = np.inf
            if ind1 == ind2:
                # lists are equal, I will therefore not double calculate
                pairs_of_atoms = [(i, j) for i in ind1
                                  for j in ind2 if i < j]
                pair_factor = 2.0
            else:
                pairs_of_atoms = [(i, j) for i in ind1
                                  for j in ind2 if i != j]
                pair_factor = 1.0
            if not pairs_of_atoms:
                # e.g. a species that is absent from the trajectory.
                # Skipping keeps the remaining pairs computable; the
                # unpacking below would raise an opaque
                # 'not enough values to unpack' instead.
                print('Warning: no atom pairs for {}, skipping'
                      ''.format(label))
                continue
            ind_pair1, ind_pair2 = list(zip(*pairs_of_atoms))

            # doinng a loop in time to avoid memory explosion
            # this also makes it easier to deal with cell changes
            hist, bin_edges = np.histogram([], bins=nbins, range=(0, radius))
            hist = hist.astype(float)
            # Second accumulator, weighted by each frame's volume, so
            # that the g(r) below is the mean of the per-frame g(r)
            # rather than one histogram divided by a single volume.
            # hist itself stays a plain neighbour count, which is what
            # the running integral reports.
            hist_by_density = np.zeros(nbins, dtype=float)
            # normalize the histogram, by the number of steps taken,
            # and the number of species1
            prefactor = (
                pair_factor
                / float(len(np.arange(istart, istop, stepsize)))
                / float(len(ind1)))
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
            for index in np.arange(istart, istop, stepsize):
                if not fixed_cell:
                    cell = cells[index]
                    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                    mic = MinimumImage(cell)
                    self._check_radius(mic, radius, 'RDF radius')
                shortest_distances = mic.distances(
                    positions[index, ind_pair2, :]
                    - positions[index, ind_pair1, :])
                shortest_distance_this_pair = min(
                    shortest_distance_this_pair, shortest_distances.min())
                counts = prefactor * (
                    np.histogram(shortest_distances, bins=nbins,
                                 range=(0, radius))[0]).astype(float)
                hist += counts
                hist_by_density += counts * volume

            radii = 0.5*(bin_edges[:-1]+bin_edges[1:])

            rdf = (hist_by_density
                   / (4.0 * np.pi * radii**2 * binsize)
                   / n_neighbours_ideal)
            integral = np.empty(len(rdf))
            sum_ = 0.0
            for i in range(len(integral)):
                sum_ += hist[i]
                integral[i] = sum_

            rdf_res.set_array('rdf_{}'.format(label), rdf)
            rdf_res.set_array('int_{}'.format(label), integral)
            rdf_res.set_array('radii_{}'.format(label), radii)
            rdf_res.set_attr('n_pairs_{}'.format(label), len(pairs_of_atoms))
            rdf_res.set_attr('n_data_{}'.format(label),
                             len(pairs_of_atoms) * ((istop-istart)//stepsize))
            rdf_res.set_attr('shortest_distance_{}'.format(label),
                             float(shortest_distance_this_pair))
            shortest_distance_all = min(shortest_distance_all,
                                        shortest_distance_this_pair)
        # One global value, after every pair has contributed.  This used
        # to be written inside the loop, so it held the running minimum
        # over the pairs seen so far under a name that reads per-pair.
        rdf_res.set_attr('shortest_distance', float(shortest_distance_all))
        return rdf_res


class AngularSpectrum(BaseAnalyzer):
    def run(self, radius=None, species_pairs=None,
            istart=1, istop=None, stepsize=1, nbins=100):
        """
        :param float radius: The radius for the calculation of the RDF
        """
        from samos.lib.rdf import calculate_angular_spec
        atoms = self.trajectory.atoms
        positions = self.trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(
                set(atoms.get_chemical_symbols()), 3))
        # The Fortran kernel does its own, naive, minimum image; it
        # does not go through MinimumImage.  See issues.md.
        cell = get_cell(self.trajectory)
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        # These are triplets, not pairs; the attribute used to be called
        # 'species_pairs', which no plotting function could tell apart
        # from an RDF result.
        rdf_res.set_attr('species_triplets', species_pairs)
        for spec1,  spec2, spec3 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[
                0] + 1  # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            ind3 = np.where(chem_sym == spec3)[0] + 1
            angular_spec, angles = calculate_angular_spec(
                positions, istart, istop, stepsize,
                radius, cell, cellI, ind1, ind2, ind3, nbins)
            rdf_res.set_array('aspec_{}_{}_{}'.format(
                spec1, spec2, spec3), angular_spec)
            rdf_res.set_array('angles_{}_{}_{}'.format(
                spec1, spec2, spec3), angles)
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

    def __init__(self, **kwargs):
        self._bonds = None
        super().__init__(**kwargs)

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
        self._check_radius(mic, max(r for _, r in cutoffs_parsed.values()),
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
