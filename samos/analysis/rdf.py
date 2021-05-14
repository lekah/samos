import numpy as  np
from scipy.spatial.distance import cdist
from ase import Atoms
from samos.trajectory import Trajectory
from samos.utils.attributed_array import AttributedArray
from samos.lib.rdf import calculate_rdf, calculate_angular_spec
import itertools
from abc import ABCMeta, abstractmethod


class BaseAnalyzer(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)
    def set_trajectory(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError("You need ot pass a {} as trajectory".format(Trajectory))
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
        raise NotImplemented("This is not fully implemented")
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
                return "spec_{}".format(ispec)
        atoms = self._trajectory.atoms
        volume = atoms.get_volume()
        positions = self._trajectory.get_positions()
        chem_sym = np.array(atoms.get_chemical_symbols(), dtype=str)
        cell = np.array(atoms.cell)
        a, b, c = cell
        range_ = list(range(0,2))
        corners = [i*a+j*b + k*c for i in range_ for j in range_ for k in range_]
        cellI = np.linalg.inv(cell)

        if istop is None:
            istop = len(positions)
        if species_pairs is None:
            species_pairs = list(itertools.combinations_with_replacement(set(atoms.get_chemical_symbols()), 2))
        indices_pairs = []
        labels = []
        for ispec, (spec1, spec2) in enumerate(species_pairs):
            indices_pairs.append((get_indices(spec1, chem_sym), get_indices(spec2, chem_sym)))
            spec1_label = get_label(spec1, ispec)
            spec2_label = get_label(spec2, ispec)
            labels.append('{}_{}'.format(spec1_label, spec2_label))

        rdf_res = AttributedArray()
        rdf_res.set_attr('species_pairs', species_pairs)
        binsize=float(radius)/nbins
        for label, (ind1, ind2) in zip(labels, indices_pairs):
            if ind1==ind2:
                # lists are equal, I will therefore not double calculate
                pairs_of_atoms = [(i,j) for i in ind1 for j in ind2 if i<j]
                pair_factor = 2.0
            else:
                pairs_of_atoms = [(i,j) for i in ind1 for j in ind2 if i!=j]
                pair_factor = 1.0

            ind_pair1, ind_pair2 = list(zip(*pairs_of_atoms))
            diff_real_unwrapped = (positions[istart:istop:stepsize, ind_pair2, :] -  positions[istart:istop:stepsize, ind_pair1, :]).reshape(-1, 3)
            density = float(len(ind2)) / volume
            diff_crystal_wrapped = np.dot(diff_real_unwrapped, cellI) % 1.0
            diff_real_wrapped = np.dot(diff_crystal_wrapped, cell)
            
            shortest_distances = cdist(diff_real_wrapped, corners).min(axis=1)

            hist, bin_edges = np.histogram(shortest_distances, bins=nbins, range=(0,radius))
            radii = 0.5*(bin_edges[:-1]+bin_edges[1:])

            # now I need to normalize the histogram, by the number of steps taken, and the number of species1
            hist = hist *pair_factor /  float(len(np.arange(istart, istop, stepsize))) / float(len(ind1))

            rdf = hist / (4.0 * np.pi * radii**2 * binsize )  / (len(ind2)/volume)
            integral = np.empty(len(rdf))
            sum_ = 0.0 
            for i in range(len(integral)):
                sum_ += hist[i]
                integral[i] = sum_

            rdf_res.set_array('rdf_{}'.format(label), rdf)
            rdf_res.set_array('int_{}'.format(label), integral)
            rdf_res.set_array('radii_{}'.format(label), radii)


            #~ rdf_res.set_array('int_{}_{}'.format(spec1, spec2), integral)
            #~ rdf_res.set_array('radii_{}_{}'.format(spec1, spec2), radii)
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

    
