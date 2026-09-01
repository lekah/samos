# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

from samos.trajectory import Trajectory
from samos.utils.attributed_array import AttributedArray

import itertools
from abc import ABCMeta, abstractmethod
from collections import defaultdict


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
        cell = np.array(atoms.cell.T)
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
        TODO:
            Improve algorithm because it can actually fail in
            very acute cell systems
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
        range_ = list(range(0, 2))
        if cells is None:
            fixed_cell = True
            atoms = self.trajectory.atoms
            volume = atoms.get_volume()
            try:
                cell = atoms.cell.array
            except AttributeError:
                cell = atoms.cell.copy()
            cellI = np.linalg.inv(cell)
            a, b, c = cell
            corners = [i*a+j*b + k *
                       c for i in range_ for j in range_ for k in range_]
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
            # normalize the histogram, by the number of steps taken,
            # and the number of species1
            prefactor = (
                pair_factor
                / float(len(np.arange(istart, istop, stepsize)))
                / float(len(ind1)))
            for index in np.arange(istart, istop, stepsize):
                if not fixed_cell:
                    cell = cells[index]
                    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                    cellI = np.linalg.inv(cell)
                    a, b, c = cell
                    corners = np.array([i*a+j*b + k*c
                                        for i in range_
                                        for j in range_
                                        for k in range_])
                diff_real_unwrapped = (
                    positions[index, ind_pair2, :]
                    - positions[index, ind_pair1, :])
                diff_crystal_wrapped = (diff_real_unwrapped@cellI) % 1.0
                diff_real_wrapped = np.dot(diff_crystal_wrapped, cell)
                # in diff_real_wrapped I have all positions wrapped
                #  into periodic cell
                shortest_distances = cdist(
                    diff_real_wrapped, corners).min(axis=1)
                shortest_distance_this_pair = min(
                    shortest_distance_this_pair, shortest_distances.min())
                hist += prefactor * \
                    (np.histogram(shortest_distances, bins=nbins,
                     range=(0, radius))[0]).astype(float)

            radii = 0.5*(bin_edges[:-1]+bin_edges[1:])

            rdf = hist / (4.0 * np.pi * radii**2 * binsize) / \
                (len(ind2)/volume)
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
        cell = np.array(atoms.cell)
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

    @staticmethod
    def _pbc_wrap(diff, cellI, cell):
        """
        Wrap displacement vectors diff (shape N x 3) to minimum-image
        vectors.  Projects to fractional coordinates, wraps each
        component to (-0.5, 0.5], and projects back to Cartesian.
        """
        frac = (diff @ cellI) % 1.0
        frac[frac > 0.5] -= 1.0
        return frac @ cell

    def _detect_bonds(self, cutoffs, frame):
        """
        Detect bonds in a single frame via distance cutoffs.

        Returns an (N_bonds, 2) int array with canonical i < j ordering.
        A set is used internally to deduplicate same-species pairs.
        """
        cutoffs_parsed = self._parse_cutoffs(cutoffs)
        positions = self.trajectory.get_positions()[frame]
        cells = self.trajectory.get_cells()
        if cells is not None:
            cell = cells[frame]
        else:
            try:
                cell = self.trajectory.get_atoms().cell.array
            except AttributeError:
                cell = self.trajectory.get_atoms().cell.copy()
        cellI = np.linalg.inv(cell)
        types = self.trajectory.get_types()

        bond_set = set()
        for (sp1, sp2), (r_min, r_max) in cutoffs_parsed.items():
            ind1 = np.where(types == sp1)[0]
            ind2 = np.where(types == sp2)[0]
            for i in ind1:
                diffs = self._pbc_wrap(
                    positions[ind2] - positions[i], cellI, cell)
                dists = np.linalg.norm(diffs, axis=1)
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

        # Invert cell only once for fixed-cell trajectories.
        if cells is None:
            fixed_cell = True
            try:
                cell = self.trajectory.get_atoms().cell.array
            except AttributeError:
                cell = self.trajectory.get_atoms().cell.copy()
            cellI = np.linalg.inv(cell)
        else:
            fixed_cell = False

        for frame in frames:
            if not fixed_cell:
                cell = cells[frame]
                cellI = np.linalg.inv(cell)
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

                    r_l = self._pbc_wrap(
                        pos[left] - pos[j], cellI, cell)
                    r_r = r_l if same else self._pbc_wrap(
                        pos[right] - pos[j], cellI, cell)

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


def util_rdf_and_plot(trajectory_path, radius=5.0, stepsize=1, bins=100,
                      species_pairs=None, species=None, savefig=None,
                      plot=False, printrdf=False, no_int=False, index=':'):
    if trajectory_path.endswith('.extxyz'):
        from ase.io import read
        aselist = read(trajectory_path, format='extxyz', index=index)
        traj = Trajectory.from_atoms(aselist)
    else:
        traj = Trajectory.load_file(trajectory_path)
    print('Read trajectory of shape {}'.format(traj.get_positions().shape))
    if species_pairs:
        species_pairs_ = []
        for spec in species_pairs:
            species_pairs_.append(spec.split('-'))
    elif species:
        species_pairs_ = pairs_with_other_species(traj, species)
    else:
        species_pairs_ = None
    rdf = RDF(trajectory=traj)
    res = rdf.run(radius=radius, stepsize=stepsize,
                  nbins=bins, species_pairs=species_pairs_)
    print('Shortest distance: {}'.format(
        res.get_attr('shortest_distance')))
    if plot or savefig:
        from samos.plotting.plot_rdf import plot_rdf
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(4, 3))
        gs = GridSpec(1, 1, top=0.99, right=0.83, left=0.14, bottom=0.16)
        ax = fig.add_subplot(gs[0])
        plot_rdf(res, ax=ax, no_int=no_int)
        ax.set_xlim(-0.2, radius)
        if savefig:
            plt.savefig(savefig, dpi=250)
        if plot:
            plt.show()

    if printrdf:
        species_pairs = res.get_attr('species_pairs')
        for spec1, spec2 in species_pairs:
            try:
                rdf = res.get_array('rdf_{}_{}'.format(spec1, spec2))
            except KeyError:
                print(
                    'Warning: RDF for {}-{} was not calculated, skipping'
                    ''.format(spec1, spec2))
                continue
            integral = res.get_array('int_{}_{}'.format(spec1, spec2))
            radii = res.get_array('radii_{}_{}'.format(spec1, spec2))
            name = '{}-{}-{}.dat'.format(printrdf, spec1, spec2)
            np.savetxt(name, np.array([radii, rdf, integral]).T,
                       header='radius    rdf     integral')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser('analysis/plot of a RDF, given a trajectory')
    parser.add_argument('trajectory_path')
    parser.add_argument('-r', '--radius', required=False, type=float,
                        default=5.0,
                        help='The radius (max) of the RDF, defaults to 5.0')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins, defaults to 100', default=100)
    parser.add_argument('-s', '--stepsize', type=int,
                        help='Stepsize over the trajectory, defaults to 1',
                        default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--species-pairs', nargs='+',
        help=('species pairs separated by a space between pairs and a'
              ' dash within the pair, e.g., '
              '--species C-O O-O'))
    group.add_argument(
        '--species', nargs='+',
        help=('species separated by a space, e.g., --species C O'))
    parser.add_argument('--printrdf',
                        help='Print the RDF to a file as a csv',)
    parser.add_argument('--plot', action='store_true',
                        help='Plot the RDF to screen')
    parser.add_argument('--no-int', action='store_true',
                        help='dont plot integral')
    parser.add_argument('-i', '--index', type=str, default=':',
                        help='index to read from trajectory, default is all')
    parser.add_argument(
        '--savefig',
        help='Where to save figure (will otherwise show on screen)')
    args = parser.parse_args()
    kwargs = vars(args)
    util_rdf_and_plot(**kwargs)
