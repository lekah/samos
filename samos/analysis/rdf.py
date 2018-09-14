import numpy as  np
from ase import Atoms
from samos.trajectory import Trajectory
from samos.utils.attributed_array import AttributedArray
from samos.lib.rdf import calculate_rdf
import itertools


class RDF(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

    # ~ def set_atoms(self, atoms):
        # ~ if not isinstance(atoms, Atoms):
            # ~ raise TypeError("You need to  pass an {} instance as atoms".format(Atoms))
        # ~ self.atoms = atoms
    def set_trajectory(self, trajectory):
        if not isinstance(trajectory, Trajectory):
            raise TypeError("You need ot pass a {} as trajectory".format(Trajectory))
        self._trajectory = trajectory
    def run(self, radius=None, species_pairs=None, istart=0, istop=None, stepsize=1, nbins=100):
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
            species_pairs = list(itertools.combinations_with_replacement(set(atoms.get_chemical_symbols()), 2))
        cell = np.array(atoms.cell)
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
            
