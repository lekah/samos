# -*- coding: utf-8 -*-

import numpy as  np
from scipy.spatial.distance import cdist

from samos.trajectory import Trajectory
from samos.utils.attributed_array import AttributedArray

import itertools
from abc import ABCMeta, abstractmethod


class BaseAnalyzer(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)
    def set_trajectory(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError('You need ot pass a {} as trajectory'.format(Trajectory))
        self._trajectory = trajectory
    @abstractmethod
    def run(*args, **kwargs):
        pass

class RDF(BaseAnalyzer):
    def run_fort(self, radius=None, species_pairs=None, istart=0, istop=None, stepsize=1, nbins=100):
        """
        :param float radius: The radius for the calculation of the RDF
        :param float density: The grid density. The number of bins is given by radius/density
        """
        raise NotImplemented('This is not fully implemented')
        from samos.lib.rdf import calculate_rdf, calculate_angular_spec
        atoms = self._trajectory.atoms
        volume = atoms.get_volume()
        positions = self._trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(set(atoms.get_chemical_symbols()), 2))
        cell = np.array(atoms.cell.T)
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        for spec1,  spec2 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[0] + 1 # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            density = float(len(ind2)) / volume
            rdf, integral, radii = calculate_rdf(positions, istart, istop, stepsize,
                radius, density, cell,
                cellI, ind1, ind2, nbins)
            rdf_res.set_array('rdf_{}_{}'.format(spec1, spec2), rdf)
            rdf_res.set_array('int_{}_{}'.format(spec1, spec2), integral)
            rdf_res.set_array('radii_{}_{}'.format(spec1, spec2), radii)
        return rdf_res

    def run(self, radius=None, species_pairs=None, istart=0, istop=None, stepsize=1, nbins=100):
        """
        Calculate a RDF also search periodic images.
        TODO: Improve algorithm because it can actually fail in very acute cell systems
        TODO: Implement orthorhombic case to gain efficiency
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
                raise TypeError('{} can not be transformed to index'.format(spec))
        def get_label(spec, ispec):
            """
            Get a good label for scpecification spec. If none can befound
            give on based on iteration counter ispec
            """
            if isinstance(spec,str):
                return spec
            elif isinstance(spec, (tuple, list)):
                return 'spec_{}'.format(ispec)
            else:
                print( type(spec))

        atoms = self._trajectory.atoms
        volume = atoms.get_volume()
        positions = self._trajectory.get_positions()
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        cells = self._trajectory.get_cells()
        range_ = list(range(0,2))
        if cells is None:
            fixed_cell = True
            try:
                cell = atoms.cell.array
            except AttributeError:
                cell = atoms.cell.copy()
            cellI = np.linalg.inv(cell)
            a, b, c = cell
            corners = [i*a+j*b + k*c for i in range_ for j in range_ for k in range_]
        else:
            fixed_cell = False

        if istop is None:
            istop = len(positions)
        elif istop >= len(positions):
            raise ValueError("Istop ({}) is higher (or equal) than number of positions ({})".format(
                istop, len(positions)))
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(set(atoms.get_chemical_symbols()), 2))
        indices_pairs = []
        labels = []
        species_pairs_pruned = []
        for ispec, (spec1, spec2) in enumerate(species_pairs):
            ind_spec1, ind_spec2 = get_indices(spec1, chem_sym), get_indices(spec2, chem_sym)
            # special situation if there's only one atom of a species
            # and we're making the RDF of that species with itself.
            # there will be empty pairs_of_atoms and the code below would crash!
            if ind_spec1==ind_spec2 and len(ind_spec1) ==1:
                continue
            indices_pairs.append((ind_spec1, ind_spec2))
            labels.append('{}_{}'.format(get_label(spec1, ispec), get_label(spec2, ispec)))
            species_pairs_pruned.append((spec1, spec2))
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs_pruned)
        binsize=float(radius)/nbins

        # wrapping the positions:
        for label, (ind1, ind2) in zip(labels, indices_pairs):
            if ind1==ind2:
                # lists are equal, I will therefore not double calculate
                pairs_of_atoms = [(i,j) for i in ind1 for j in ind2 if i<j]
                pair_factor = 2.0
            else:
                pairs_of_atoms = [(i,j) for i in ind1 for j in ind2 if i!=j]
                pair_factor = 1.0
            # It can happen that pairs_of_atoms
            ind_pair1, ind_pair2 = list(zip(*pairs_of_atoms))

            # doinng a loop in time to avoid memory explosion
            # this also makes it easier to deal with cell changes
            hist, bin_edges = np.histogram([], bins=nbins, range=(0, radius))
            hist = hist.astype(float)
            # normalize the histogram, by the number of steps taken, and the number of species1
            prefactor = pair_factor /  float(len(np.arange(istart, istop, stepsize))) / float(len(ind1))
            for index in np.arange(istart, istop, stepsize):
                if not fixed_cell:
                    cell = cells[index]
                    cellI = np.linalg.inv(cell)
                    a, b, c = cell
                    corners = np.array([i*a+j*b + k*c 
                                for i in range_
                                for j in range_
                                for k in range_])
                diff_real_unwrapped = positions[index, ind_pair2, :] - positions[index, ind_pair1, :]
                diff_crystal_wrapped = (diff_real_unwrapped@cellI) % 1.0
                diff_real_wrapped = np.dot(diff_crystal_wrapped, cell)
                # in diff_real_wrapped I have all positions wrapped into periodic cell
                shortest_distances = cdist(diff_real_wrapped, corners).min(axis=1)
                hist +=  prefactor * (np.histogram(shortest_distances, bins=nbins, range=(0,radius))[0]).astype(float)

            radii = 0.5*(bin_edges[:-1]+bin_edges[1:])

            rdf = hist / (4.0 * np.pi * radii**2 * binsize )  / (len(ind2)/volume)
            integral = np.empty(len(rdf))
            sum_ = 0.0
            for i in range(len(integral)):
                sum_ += hist[i]
                integral[i] = sum_

            rdf_res.set_array('rdf_{}'.format(label), rdf)
            rdf_res.set_array('int_{}'.format(label), integral)
            rdf_res.set_array('radii_{}'.format(label), radii)

        return rdf_res

class AngularSpectrum(BaseAnalyzer):
    def run(self, radius=None, species_pairs=None, istart=1, istop=None, stepsize=1, nbins=100):
        """
        :param float radius: The radius for the calculation of the RDF
        :param float density: The grid density. The number of bins is given by radius/density
        """
        atoms = self._trajectory.atoms
        volume = atoms.get_volume()
        positions = self._trajectory.get_positions()
        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(set(atoms.get_chemical_symbols()), 3))
        cell = np.array(atoms.cell)
        cellI = np.linalg.inv(cell)
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        for spec1,  spec2, spec3 in species_pairs:
            ind1 = np.where(chem_sym == spec1)[0] + 1 # +1 for fortran indexing
            ind2 = np.where(chem_sym == spec2)[0] + 1
            ind3 = np.where(chem_sym == spec3)[0] + 1
            angular_spec, angles = calculate_angular_spec(positions, istart, istop, stepsize,
                radius, cell, cellI, ind1, ind2, ind3, nbins)
            rdf_res.set_array('aspec_{}_{}_{}'.format(spec1, spec2, spec3), angular_spec)
            rdf_res.set_array('angles_{}_{}_{}'.format(spec1, spec2, spec3), angles)
        return rdf_res

def util_rdf_and_plot(trajectory_path, radius=5.0, stepsize=1, bins=100,
        species_pairs=None, savefig=None):
    from samos.plotting.plot_rdf import plot_rdf
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    traj = Trajectory.load_file(trajectory_path)
    print("Read trajectory of shape {}".format(traj.get_positions().shape))
    if species_pairs:
        species_pairs_ = []
        for spec in species_pairs:
            species_pairs_.append(spec.split('-'))
    else:
        species_pairs_=None
    rdf = RDF(trajectory=traj)
    res = rdf.run(radius=radius, stepsize=stepsize, nbins=bins,species_pairs=species_pairs_)
    fig = plt.figure(figsize=(4,3))
    gs = GridSpec(1,1, top=0.99, right=0.86, left=0.14, bottom=0.16)
    ax = fig.add_subplot(gs[0])
    plot_rdf(res, ax=ax)
    if savefig:
        plt.savefig(savefig, dpi=250)
    else:
        plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("analysis/plot of a RDF, given a trajectory")
    parser.add_argument('trajectory_path')
    parser.add_argument('-r', '--radius', required=False, type=float, default=5.0,
                help='The radius (max) of the RDF, defaults to 5.0')
    parser.add_argument('-b', '--bins', type=int, help='Number of bins, defaults to 100', default=100)
    parser.add_argument('-s', '--stepsize', type=int,
            help='Stepsize over the trajectory, defaults to 1', default=1)
    parser.add_argument('--species-pairs', nargs='+',
            help='species pairs separated by a dash, e.g., --species-pairs C-O O-O')
    parser.add_argument('--savefig', help='Where to save figure (will otherwise show on screen)')
    args = parser.parse_args()
    kwargs = vars(args)
    util_rdf_and_plot(**kwargs)
