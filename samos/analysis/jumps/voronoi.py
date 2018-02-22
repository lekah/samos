

"""
This module has all the implementations to do a Voronoi decomposition on a
structure.
"""
# misc
import re, os, sys, time, copy, json
from subprocess import check_output as cout

import tempfile
# stats and mathematical ops:
import numpy as np
from scipy.spatial import ConvexHull  # Voronoi #check out how that works
from scipy.stats import linregress

# ASE
from ase.io import read, write
from ase import Atom, Atoms
from ase.visualize import view
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers, covalent_radii, atomic_names


#hashing
from hashlib import sha224 # seriously?

#PLOTTING
from matplotlib import pyplot as plt
from matplotlib import pylab as plb
from matplotlib import gridspec
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation
from matplotlib import colors
from matplotlib import colorbar
from colorsys import rgb_to_hls, hls_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
#my own:
#~ from imports import structure_wizard

# from carlo.utils import flatten
bohr_to_ang = 0.52917720859

IMAGES3D = [
        (i,j,k)
        for i in range(-1,2)
        for j in range(-1,2)
        for k in range(-1,2)
    ]

IMAGES2D = [
        (i,j)
        for i
        in range(-1,2)
        for j in range(-1,2)
    ]

class MissingVolume(Exception):
    pass

class SurplusVolume(Exception):
    pass


class VoronoiNetwork():
    """
    This is the main class to perform to analyze structure and MD-trajectories
    using the site tesselation algorithm
    based on:

    - Voronoi decomposition
    - Delaunay triangulation

    A typical analysis has three main steps:

    - The decomposition of a structure, either a ground state structure the first snapshot of the trajectory
      A Voronoi Decomposition is done on the atomic position of non-diffusing species.
      A network of connected sites is created (VoronoiNetwork)
    - The projection of a continuous MD-trajectory on the discrete sites,
      producing a sequence of visited sites for each site.
    - The analysis of that `site-trajectory`, returns quantities such as the observed rates of ionic exchange.
    """


    _strategies = {'precise',  'neighbors'}	 #all possible strategies currently implemented
    @classmethod
    def strategies(cls):
        return cls._strategies

    def __init__(self, verbosity=False, **kwargs):
        """
        Initializes the VoronoiNetwork.

        :param bool verbosity: Verbosity of output
        :param log: Buffer to print output
        :param bool correct_drift:
            Whether to correct for the rigid sublattice drifting
            out of the unit cell. This was observed for long trajectories.
            Default is True.
        """
        self._verbosity = verbosity
        self._log = log
        self._correct_drift = correct_drift

    def set_atoms(self, atoms, host=None):
        """
        Set the atomic structure.
        :param atoms: an instance of ase.Atoms
        :param host:
            The host atoms that define the decomposition.
            They can be a list of indices or element names in the structure
        """
        def set_indices(spec):
            """
            Check the specification.
            If the specification is a string (or a list of strings), returns
            all the indices of the structure where the symbol matches the string.
            If it's an integer or a list of integers, returns the list of integers
            """
            if isinstance(spec, int):
                return [spec]
            elif isinstance(spec, basestring):
                spec = [spec] # make a list:
            chemical_symbols = atoms.get_chemical_symbols()
            if isinstance(spec, (tuple, list)):
                indices = []
                for it in spec:
                    if isinstance(it, int):
                        indices.append(it)
                    elif isinstance(it, basestring):
                        # find all indices that match the symbol:
                        [indices.append(idx) for idx, s in enumerate(chemical_symbols) if s==it]
                    else:
                        raise InputValidationError("I don't know what to do with {} {}\n"
                            "as a specification in set_atoms".format(type(it), it))
            else:
                raise ValueError("I don't know what to do with {} {}\n"
                            "as a specification in set_atoms".format(type(spec), spec))
            return sorted(set(indices))

        if not isinstance(atoms, Atoms):
            raise InputValidationError("atoms has to be an instance of {}".format(Atoms))

        atoms.set_positions([collapse_into_unit_cell(p, atoms.cell) for p in atoms.positions])
        # What about the sublattice drift?
        self._atoms = atoms
        #~ self._track = set_indices(track)
        self._host_indices =  set_indices(host)

        if not(len(self._host_indices)):
           raise ValueError("I have an empty list of host ions")

        #Create the host structure:
        self._host_structure = Atoms(cell=self._atoms.cell)
        #~ self._lattice = Lattice(self._atoms.cell)
        #Append the atoms and make a mapping from the indices in the host structure to the
        # corresponding atom in the original structure:
        
        [self._host_structure.append(self._atoms[idx]) for idx in self._host_indices]


    def set_trajectory(self, trajectory, timestep_fs=None):
        """
        Loads/initializes an MD trajectory.

        :param array: An instance of numpy.array of dimension [numberOfSteps, NumberOfAtoms, 3], with positions given in angstrom
        :param float timestep_in_fs: The timestep in fs
        """
        if not isinstance(trajectory, Trajectory):
            raise InputValidationError("The trajectory you gave ({}) is not an instance of {}".format(trajectory, Trajectory))
        self._trajectory = trajectory

    def __len__(self):
        """
        Return length of my list of nodes.
        """
        try:
            return len(self._nodes)
        except:
            return 0

    @property
    def nodes(self):
        return self._nodes.values()

    def decompose_qhull(self, add_images=True, accuracy=1e-3):
        """
        Performs:
        - A Voronoi Decomposion
        - A Delaunay Triangulation

        using a Qhull wrapper. Then select the hulls that belong to the unit
        cell and discard the others. The total volume of the site hulls select
        has to be (disregarding numeric errors) exactly the volume of the unit
        cell. If this is not the case, return an error...
        """
        def read_qhull_output(text):
            """
            Read the output of qhull with argument FF to print full information 
            on facets. Retuns a list with an integer, the position of the
            voronoi node and the constituting vertices for each node input 
            (for one facets looks like):

            - f3264
                - flags: bottom simplicial
                - normal:  -0.07825  -0.0801   0.8416  -0.5284
                - offset:  -2.143951
                - center: -0.5093569446837466 -0.5214288628296513 5.478512151086883
                - vertices: p68(v162) p15(v161) p17(v158) p31(v75) # So in pII(vJJ) II is the index of the point in the input positions
                - neighboring facets: f3210 f3243 f3263 f3265
            the output for above facet is:
            [3264, -0.5093569446837466, -0.5214288628296513, 5.478512151086883, 0, 68,15, 17, 31]
            """
            facets_regex = re.compile(
                """
                -[ \t](?P<facetkey>f[0-9]+)  [\n]
                [ \t]*-[ ]flags: .* [\n]
                [ \t]*-[ ]normal: .* [\n]
                [ \t]*-[ ]offset: .* [\n]
                [ \t]*-[ ]center:(?P<center>([ ][\-]?[0-9]*[\.]?[0-9]*(e[-?[0-9]+)?){3}) [ \t] [\n]
                [ \t]*-[ ]vertices:(?P<vertices>([ ]p[0-9]+\(v[0-9]+\))+) [ \t]? [\n]
                [ \t]*-[ ]neighboring[ ]facets:(?P<neighbors>([ ]f[0-9]+)+)
                """, re.X | re.M)   # [ \t]*-[ ]center\:[ ](?P<center>[\-]?[0-9]*[\.]?[0-9]*[ ]) # [\n]

            verticesp = re.compile('(?<=p)[0-9]+')

            facet_list = [
                {
                    'id':match.group('facetkey').strip('f'),
                    'center':map(float,match.group('center').split()),
                    'vertices':map(int, verticesp.findall(match.group('vertices')))
                } for match
                in facets_regex.finditer(text)
            ]
            return facet_list


        self._log.write('   Decomposing with qhull\n')
        self._log.write('      Making the structures\n')
        #~ make_my_structures()
        host_supercell = self._host_structure.repeat([3,3,3])
        for c in self._host_structure.cell:
            host_supercell.translate(-c)

        for atoms, name in (
                (self._atoms, 'Original'),
                (self._host_structure, 'Host'),
                (host_supercell, 'Replicated')
            ):
            self._log.write(
                    '         {:<15} length: {:<4} contains: {}\n'
                    ''.format(
                            name, len(atoms),
                            ', '.join(list(set([atom.symbol for atom in atoms])))
                    )
                )
        # Now i run qhull by command line in shell and capture the output
        # Both happen inf files since the output might be too big
        # to be capture in a subprocess (larger than shell buffer)
        # Same for the input for large structures
        with tempfile.NamedTemporaryFile(
                'w',prefix = 'qvor', suffix='.in',
                delete=True) as inputf:
            inputf.write('3\n{}\n'.format(len(host_supercell)))
            [
                inputf.write('{} {} {}\n'.format(x,y,z))
                for x,y,z
                in host_supercell.positions
            ]
            inputf.flush()
            
            with tempfile.NamedTemporaryFile(
                    'rw',prefix = 'qvor',suffix='.out', delete=True
                ) as outputf:
                cout('cat "{}" |  qvoronoi FF QJ > {}'.format(
                            inputf.name, outputf.name), shell=True)
                qhull_out=outputf.read()

        vertices = read_qhull_output(qhull_out)  #These are the vertices
        pp = [[i,j,k] for i in (-1,2) for j in  (-1,2) for k in  (-1,2)]
        #any Voronoi node outside the hull from these points is very far and can
        # be excluded. Here i get the vertices of the cell:
        points = [
                np.dot(np.array(self._atoms.cell).T, np.array([i,j,k]))
                for i,j,k
                in pp
            ]
        # This is my listof nodes:
        self._nodes = {}
        self._log.write(
                '      Qhull gave me {} vertices\n'.format(len(vertices))
            )
        # Qhull gave me huge number of vertices, I need to filter the once that
        # are important to me
        for vertice_spec in vertices:
            center = vertice_spec['center']
            vertices_in_host_supercell  = vertice_spec['vertices']
            #~ if set(vertices_in_host_supercell).difference(set(range(52,56))):
                #~ continue
            #~ print vertices_in_host_supercell
            if not is_in_unit_cell(center, self._atoms.cell):
                # If the node is not in the unit cell, I will not add it to my
                # nodes
                continue
            vertices_in_host = [v%len(self._host_structure) for v in vertices_in_host_supercell]
            #~ view(host_supercell)
            #~ view(host_supercell[vertices_in_host_supercell])
            #~ print vertices_in_host_supercell
            #~ print vertices_in_host
            # these are the translations!
            translations = [
                    host_supercell.positions[v1] - self._host_structure.positions[v2]
                    for v1, v2 in zip(vertices_in_host_supercell, vertices_in_host)
                ]

            # I need to map the indices in the host structure to the vertices
            # Don't sort, translations need to be in the same order
            vertices_in_atoms =  [self._host_indices[v] for v in vertices_in_host]
            
            if len(set(vertices_in_atoms)) < 4:
                continue
                raise Exception("The number of vertices for this node is less than 4\n",
                    "Your unit cell is too small")
            node = VoronoiNode(center=center, vertices=vertices_in_atoms, translations=translations)
            
            try:
                # maybe I am unable to create the hull because there
                # is no volume (might happen)
                node.get_hull(self._atoms.positions)
            except Exception as e:
                self._log.write( ' {} \n'.format(e))
                continue
            self._nodes[sha224(str(vertices_in_atoms))] = node

        self._log.write('      I reduced that to {} vertices\n'.format(len(self)))
        #Calculate the combined volumes of the sites
        comb = np.sum(self.get_volumes(self._atoms.positions))
        cellv = self.get_volume()
        self._log.write( '      Checking volumes\n'
                        '         Unit cell volume: {}\n'
                        '         Combined hull volume: {}\n'.format(cellv, comb))
        if add_images:
            if comb/cellv < 1.0 - accuracy:
                raise MissingVolume(
                    'Missing volume (Comb: {}, Unit cell: {})'
                    ''.format(comb, cellv)
                )
            elif comb/cellv > 1.0 + accuracy:
                raise SurplusVolume(
                    'Surplus volume (Comb: {}, Unit cell: {})'
                    ''.format(comb, cellv)
                )


    def track_ions(self,
            max_ion_losses=5, volume_increase_threshold=1.25,
            #~ strategy='neighbors', update_host=True):
            strategy='precise', update_host=True):

        unitcellvolume = self.get_volume()
        ion_losses = 0

        self._log.write('   Initializing Convex Hulls and checking volume\n')

        [node.initialize_hull() for node in self.nodes]
        comb = np.sum(self.get_volumes())
        self._log.write('   Sum of volumes is {}\n'.format(comb))
        self._log.write(
                '   Starting to track {} ions in {} possible sites\n'
                '   in trajectory of shape {}\n'
                ''.format(
                        len(self._track), len(self),
                        self._trajectory.positions.shape
                    )
            )

        visited_sites = [-1]*len(self._track)
        for positions_t in self._trajectory.positions:
            if strategy == 'precise':
                for track_idx, at in enumerate(self._track):
                    # position of this ion at time t:
                    pos_at_t = positions_t[at][:]
                    count = 0
                    for node_idx, node in enumerate(self.nodes):
                        if node.inside(pos_at_t):
                            count += 1
                            visited_sites[track_idx] = node_idx
                print ' '.join(map('{:<3}'.format, visited_sites))
                return
                        
                visited_sites = []
                for ion in res:
                    counts = ion.count(True)
                    if counts == 1:
                        val = ion.index(True)
                    elif counts == 0:
                        if be_hypersensitive:
                            raise Exception('Ion not found in any site')
                        self._log.write('WARNING: ION WAS NOT FOUND IN ANY SITE\n')
                        val = -1
                    else:
                        if be_hypersensitive:
                            raise Exception('Ion found in more than one site')
                        self._log.write(
                            'WARNING: ION WAS FOUND IN {} SITES'
                            '\n'.format(counts)
                        )
                        val = - counts
                    visited_sites.append(val)
            elif strategy == 'neighbors':
                print
                return
                for idx in enumerate(tracklist):
                    
                    if prev_visit == -1:
                        self._log.write('Lost an ion\n')
                        self.ionlosses += 1
                    for site_id in self.nodes[prev_visit].get_distance_neighbors():
                        if self.nodes[site_id].inside(pos):
                            return site_id
                for site_id in self.nodes[prev_visit].get_neighbors():
                    if self.nodes[site_id].inside(pos):
                        return site_id
                [search_the_neighbors(prev_visit,
                                               pos,
                                               dist_threshold,
                                               normalize_by_center)
                          for prev_visit, pos in
                          [(self.visited_sites[nodeindex],
                            collapse_into_unit_cell(
                                positions[atomindex].tolist(),
                                self.atoms.cell))]]



    def track_ions_in_traj(self, max_ion_losses=5,
            volume_increase_threshold=1.25,
            is_flipper_level_1=False,
        ):
        """
        Tracks atoms in  tracklist (default: all remove atoms, default: all Li)
        giving as result the site each ion can be associated with, strategy as
        defined by self.strategy. See details in find_ion
        """
        def find_ion_precise(positions, timestep=False, be_hypersensitive=False):
            """
            Find the ion tracking in a precise way
            """

            res = [
                    [
                        site.inside(positions[item][:].tolist())
                        for site
                        in self.nodes
                    ]
                    for item
                    in tracklist
                ]
            visited_sites = []
            for ion in res:
                counts = ion.count(True)
                if counts == 1:
                    val = ion.index(True)
                elif counts == 0:
                    if be_hypersensitive:
                        raise Exception('Ion not found in any site')
                    self._log.write('WARNING: ION WAS NOT FOUND IN ANY SITE\n')
                    val = -1
                else:
                    if be_hypersensitive:
                        raise Exception('Ion found in more than one site')
                    self._log.write(
                        'WARNING: ION WAS FOUND IN {} SITES'
                        '\n'.format(counts)
                    )
                    val = - counts
                visited_sites.append(val)
            self.visited_sites = visited_sites

        def find_ion_by_distance(positions):
            try:
                self.visited_sites = [
                        update_by_distance(
                            self.visited_sites[i],
                            positions[atom].tolist()
                        )
                        for i,atom
                        in enumerate(tracklist)
                    ]
            except Exception as e:
                self._log.write(
                        '      Diverging to precise tracking for first timestep'
                        '\n'
                    )
                find_ion_precise(positions, be_hypersensitive = True)


        def find_ion_using_neighbors_track_volume(timestep):
            """
            Given the positions at timestep t, return a list of indices for each
            Li-ion. The indices refer to the index of the site that the ion is
            in at this timestep. This algorithm is orders of magnite faster
            than find_ions_precise because:

            *   It remembers the site the ion was previously found in and
                searches there first.
            *   If the ion is not found there, the algorithm search through the
                list of sites sorted by distance to site it was previously found.
            *   It does not check whether the ions was found in more than site.
                This can happen if the rigid sublattice has melted and anions
                changed positions.
            *   The algorithm stops if the number of times that an ion was not
                found anywhere (index=-1) exceeds preset max_ion_losses.
            *   This means that something is not right, that the rigid
                sublattice has moved by a lot or even melted.
            *   The algorithm can also stop after the volume ratio exceeds a
                certain value. The volume ratio
                (sum of combined volumes of hulls / unit cell volume)
                is a good indicator for melting

            The algorithm also calculates the volume of the sites.
            It makes sense to do the tracking and volume evolution together since that safes time in total.
            The updating of site-hulls is a time intensive step.

            """
            def update_hull(node):
                node.update_hull(positions)

            def search_the_neighbors(prev_visit, pos):
                """
                Expects the first argument to be the previous visit, the second
                is  a position
                """
                if prev_visit is None or prev_visit == -1:
                    self._log.write('Lost an ion\n')
                    self.ionlosses += 1
                    for site_id in self.nodes[prev_visit].get_distance_neighbors():
                        if self.nodes[site_id].inside(pos):
                            return site_id
                for site_id in self.nodes[prev_visit].get_neighbors():
                    if self.nodes[site_id].inside(pos):
                        return site_id
                return -1
            positions  = self._trajectory.positions[timestep]
            #~ if self.correct_sublattice_drift:
            if not is_flipper_level_1:
                positions = normalize_positions(
                        positions,
                        self._atoms.cell,
                        tracklist
                    )
                # update the positions of cage-constituting atoms for each site:
                [node.update_hull(positions) for node in self.nodes]  ## Faster
                #~ map(update_hull, self.nodes) ###slower!!

            try:
                # self.visited_sites = map(
                #         search_the_neighbors,
                #         *zip(*[
                #             (
                #                 self.visited_sites[nodeindex],
                #                 collapse_into_unit_cell(
                #                         positions[atomindex].tolist(),
                #                         self.atoms.cell
                #                     )
                #             )
                #             for nodeindex, atomindex
                #             in enumerate(tracklist)
                #         ])
                #     )
                # print tracklist
                #~ if not get_distances:
                self.visited_sites = [search_the_neighbors(prev_visit,
                                                               pos,
                                                               dist_threshold,
                                                               normalize_by_center)
                                          for prev_visit, pos in
                                          [(self.visited_sites[nodeindex],
                                            collapse_into_unit_cell(
                                                positions[atomindex].tolist(),
                                                self.atoms.cell))
                                           for nodeindex, atomindex
                                           in enumerate(tracklist)]]

                #~ else:
                    #~ self.visited_sites_with_dist = [search_the_neighbors_with_distances(prev_visit,
                                                                                    #~ pos,
                                                                                    #~ dist_threshold,
                                                                                    #~ normalize_by_center)
                                                #~ for prev_visit, pos in
                                                #~ [(self.visited_sites_with_dist[nodeindex][0],
                                                  #~ collapse_into_unit_cell(
                                                      #~ positions[atomindex].tolist(),
                                                      #~ self.atoms.cell))
                                                 #~ for nodeindex, atomindex
                                                 #~ in enumerate(tracklist)]]
#~ 
                    #~ self.visited_sites, self.cation_distances, self.center_distances, self.cat_center_distance, self.cat_vertex_distances = zip(*self.visited_sites_with_dist)

            except AttributeError as e:
                self._log.write(
                        '      Diverging to precise tracking for first timestep'
                        '\n'
                    )
                find_ion_precise(positions)

            volumes = self.get_volumes()
            volume = np.sum(volumes)
            volume_ratio = volume/unitcellvolume

            if max_ion_losses and max_ion_losses == self.ionlosses:
                raise Exception(
                        'Number of ions lost reached threshold of {}'
                        ''.format(max_ion_losses)
                    )
            if volume_increase_threshold and volume_ratio >= volume_increase_threshold:
                raise Exception(
                    'Volume of site hulls ({}) exceeded threshold \n'
                    '({} = {} x {}'.format(
                        volume,
                        volume_increase_threshold*unitcellvolume,
                        volume_increase_threshold,
                        unitcellvolume
                    )
                )
            if self.verbosity:
                self._log.write(
                    't: {:<5} V: {:<5} L:{:<3} OccS: {}'
                    '\n'.format(
                            timestep,
                            '{:.5f}'.format(volume_ratio),
                            self.ionlosses,
                            ' '.join([
                                    '{:>3}'.format(s)
                                    for s
                                    in self.visited_sites
                                ])
                        )
                    )
            if get_distances:
                return self.visited_sites, volumes, volume, self.cation_distances, self.center_distances, self.cat_center_distance, self.cat_vertex_distances
            else:
                return self.visited_sites, volumes, volume

        def is_melted(slope, intercept, rvalue, pvalue, stdr):
            """
            Melting criteria on the linregression of the volumes implemented here
            """
            return slope > 0 and rvalue > 0.2


        unitcellvolume = self.get_volume()
        self.ionlosses = 0
        tracklist  = self._track[:] #  [val for key, val in sorted(self.tracked_ions.items())]
        self._log.write('   Initializing Convex Hulls and checking volume\n')
        [node.initialize_hull() for node in self.nodes]
        comb = np.sum(self.get_volumes())
        self._log.write('   Sum of volumes is {}\n'.format(comb))
        self._log.write(
                '   Starting to track {} ions in {} possible sites\n'
                '   in trajectory of shape {}\n'
                ''.format(
                        len(tracklist), len(self),
                        self._trajectory.positions.shape
                    )
            )


        res = [find_ion_using_neighbors_track_volume(i) for i in range(len(self._trajectory.positions))]

        #~ if not get_distances:
        self.site_traj, self.volumes, self.total_volume = [np.array(i) for i in zip(*res)]

        #~ else:
            #~ self.site_traj, self.volumes, self.total_volume, self.all_cation_dist, self.all_center_dist, self.all_cat_center_dist, self.all_cat_vertex_dist = [i for i in zip(*res)]
#~ 
            #~ self.site_traj = np.array(self.site_traj)
            #~ self.volumes = np.array(self.volumes)
            #~ self.total_volume = np.array(self.total_volume)

        self._log.write('   Has the structure melted?\n')
        initial_volume = self.total_volume[0]
        self._log.write('   Initial volume was {}\n'.format(initial_volume))
        final_volume = self.total_volume[-1]
        self._log.write('   Final   volume was {}\n'.format(final_volume))

        X = range(len(self.site_traj))
        slope, intercept, rvalue, pvalue, std = linregress(X, self.total_volume)
        self._log.write('   Linear Regression results:\n')
        for key, val in [
                ('slope',slope),
                ('intercept', intercept),
                ('r-value', rvalue),
                ('p-value',pvalue),
                ('stderr', std)
            ]:
            self._log.write('      {:<10}{}\n'.format(key, val))
        melted = is_melted(slope, intercept, rvalue, pvalue, std)
        self._log.write(
                '   I classify that as {}melted\n'
                ''.format('' if melted else 'not ')
            )
        self.track_results = dict(
            timestep_in_fs  = self.timestep_in_fs,
            volume_reg      = dict(
                    slope=slope, intercept=intercept,
                    rvalue=rvalue, pvalue=pvalue, std=std
                ),
            melted          = int(melted),
            initial_volume  = initial_volume,
            final_volume    = final_volume,
            unitcellvolume  = unitcellvolume,
            ionlosses       = self.ionlosses,
        )

        return np.array(self.site_traj), np.array(self.volumes), self.track_results


    def make_distances(self):
        """
        Calculate all the distances in the Voronoi network
        """
        self.distances = {}
        for it1, node1 in enumerate(self.nodes):
            self.distances[it1] = {}
            for it2, node2 in enumerate(self.nodes):
                if it1 >= it2: continue
                self.distances[it1][it2] = np.linalg.norm(
                        node1.xyz - find_closest_periodic_image(
                                node2.xyz, node1.xyz, self.atoms.cell
                            )[1]
                    )


    def get_distances(self, i1, i2):
        if i1 < i2: return self.distances[i1][i2]
        elif i1>i2: return self.distances[i2][i1]
        else: return 0.0

    def get_volume(self):
        """Return volume of unit cell calculated by ASE (checks?)"""
        return self._atoms.get_volume()

    def get_volumes(self,positions):
        """Return the volume of the simplicial hulls"""
        return [node.get_volume(positions) for node in self.nodes]

    def view_sites(self, savefig=None, alpha=0.5):
        """
        Creating a 3d plot where voronoi nodes and voronoi cages are shown for
        the structure as dots and colored tetrahedra


        .. figure:: /images/vd_Nb4O24Li28.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center

            Decomposition of a structure


        """

        #~ try:
            #~ self.nodes[0].cage_positions
        #~ except AttributeError:
            #~ [node.initialize_hull() for node in self.nodes]
        gs = gridspec.GridSpec(
                2,1, height_ratios=[10,1],
                left=0, right=1, top=1, bottom=0
            )
        fig = plt.figure(facecolor='white', figsize = (8,6))
        ax = fig.add_subplot(gs[0], projection='3d')
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_axis_off()

        try:
            pp = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
            for p1 in pp:
                for p2 in pp:
                    counts = [p1[i] == p2[i] for i in range(3)]
                    if not counts.count(True) == 2: continue
                    ax.plot(
                            *zip(np.dot([p1], self._atoms.cell).tolist()[0],
                            np.dot([p2], self._atoms.cell).tolist()[0]),
                            color=(0,0,0)
                        )
        except Exception as e:
            self._log.write('   Cannot print cell lines due to {}\n'.format(e))
        try:
            for indeks, atom in enumerate(self._atoms):
                ele = atom.symbol
                # I want color as used in ASE
                color = jmol_colors[atomic_numbers[ele]]
                pos = [[i] for i in atom.position]  #patch
                ax.scatter(*pos, color = color, s = 50)
        except Exception as e:
            self._log.write(' ERROR: {}\n'.format(e))

        positions = self._atoms.positions
        for node in self.nodes:
            #~ print node._vertices
            points = node._get_hull_points(positions)
            try:
                colors = [
                    jmol_colors[atomic_numbers[self._atoms[vertice].symbol]]
                    for vertice
                    in node._vertices
                ]
            except Exception as e:
                self._log.write(
                        '   ERROR: {}\n   Setting color to red\n'.format(e)
                    )
                colors = 'red'
            if len(points) < 4:
                raise Exception('less then 4 points, cannot construct complex hull')
            simplices = ConvexHull(points).simplices
            try:
                poly3d = [
                    [
                        points[simplices[ix][iy]]
                        for iy
                        in range(len(simplices[0]))
                    ]
                    for ix in range(len(simplices))]
            except Exception as e:
                print e

            collection = Poly3DCollection(
                    poly3d, linewidths=1,
                    alpha=alpha,
                )
            face_color = np.random.rand(3).tolist()
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)
            xyz =  [[i] for i in node._center]
            #~ ax.scatter(*xyz, s = 30 , marker = '^', c = 'blue')
            ax.scatter(*zip(*points), c=colors, s=50)

        try:
            ax2 = fig.add_subplot(gs[1])
            ax2.set_axis_off()
            elements = set(self._atoms.get_chemical_symbols()) # if element != 'Li'
            colors = [
                jmol_colors[
                    atomic_numbers[element]
                ]
                for element in elements
            ]
            ls_n_labels = [
                (plt.scatter([],[], c=color, s=50), element)
                for color, element
                in zip(colors, elements)
            ]
            #~ ls_n_labels.append((
                    #~ plt.scatter([],[],s=30, marker='^', c='blue'),
                    #~ 'interstitial site'
                #~ ))
            ls, labels = zip(*ls_n_labels)
            leg = plt.legend(
                    ls, labels, ncol=len(ls_n_labels),
                    frameon=False, handlelength=2, loc='lower center',
                    scatterpoints=1, borderpad=3, handletextpad=1,
                    title='', fontsize=16
                )

            at_nrs = self._atoms.get_atomic_numbers().tolist()
            rev_at_nrs = {v:k for k,v in atomic_numbers.items()}
            composition = sorted([
                    (at_nr , at_nrs.count(at_nr))
                    for at_nr
                    in set(at_nrs)
                ])
            composition = [(rev_at_nrs[at_nr], count) for at_nr, count in composition]
            name_of_structure = ''.join([
                '{}_{{{}}}'.format(ele, count)
                for ele, count in composition
            ])
            plt.suptitle( r'Decomposition of ${}$'.format(name_of_structure), fontsize = 20)
        except Exception as e:
            self._log.write('{}\n'.format(e))
        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()
        plt.close(fig)

    def view_trajectory(self, save_animation = False, name = None):
        """
        View the trajectory in an animation, including the movement of the hulls.
        """
        def get_timestep(
                num, ax, ax2, trajectory, site_traj, atoms,
                tetrahedra, bar, progress_title
            ):
            timestep = trajectory[num]
            if self.correct_sublattice_drift:
            # Center the positions on the center of the delithiated structure
                timestep =  normalize_positions (
                                timestep,
                                self.atoms.cell,
                                self.tracked_ions.values()
                            )

            [node.update_hull(timestep) for node in self.nodes]
            for index, t in enumerate(tetrahedra):
                simplices = self.nodes[index].simplices()
                points = self.nodes[index].cage_positions
                #~ print simplices
                poly3d = [
                        [
                                points[simplices[ix][iy]]
                                for iy
                                in range(len(simplices[0]))
                        ]
                        for ix
                        in range(len(simplices))
                    ]
                t.set_verts(poly3d)
                if index in site_traj[num]:
                    t.set_facecolor('red')
                    alpha = min([0.03, t.get_alpha() + 0.001])
                else:
                    alpha = max([0.0, t.get_alpha() - 0.001])
                t.set_alpha(alpha)

            for indeks, atom in enumerate(atoms):
                position = [[i] for i in timestep[indeks]]
                atom.set_data(*position[:2])
                atom.set_3d_properties(position[2])
                pos_hack = [[i] for i in position]
                if indeks in tracklist:
                    ax.plot(
                            *position, linestyle='points', marker='.',
                            markersize=3, color=atom.get_color()
                        )
            if num:
                bar.set_width(num)
            plt.setp(
                progress_title,
                text='progress: {:.2f}%, @Timestep: {} , @Time: {:.2f} fs'.format(
                    100.* num / len(self.trajectory),
                    num,
                    self.timestep_in_fs*num
                )
            )

            ############# END get_timestep ####################################

        tracklist  = [val for key, val in sorted(self.tracked_ions.items())]
        if self.correct_sublattice_drift:
            # Center the positions on the center of the delithiated structure
            self.atoms.set_positions(
                            normalize_positions (
                                np.array(self.atoms.positions),
                                self.atoms.cell,
                                tracklist
                            )
                        )

        cmap = plt.cm.BuPu
        cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        gs = gridspec.GridSpec(2,1, height_ratios = [50,1])

        fig = plt.figure(facecolor='white', figsize = (16,9))

        #~ ax = p3.Axes3D(fig)
        ax = fig.add_subplot(gs[0], projection = '3d')
        ax2 = fig.add_subplot(gs[1])
        #~ ax.title(r"$Li_4$ at 400K")
        if name is not None:
            plt.suptitle(name, fontsize = 20)
        ax.set_axis_off()
        # draw the cell
        pp = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
        for p1 in pp:
            for p2 in pp:
                #skip unnecessary points:
                counts = [p1[i] == p2[i] for i in range(3)]
                if not counts.count(True) == 2:
                    continue
                ax.plot(
                    *zip(
                            np.dot([p1], self.atoms.cell).tolist()[0],
                            np.dot([p2], self.atoms.cell).tolist()[0]
                        ),
                    color=(0,0,0)
                )

        # list atoms is the spheres in the visualization, dont confuse with
        # self.atoms, the ASE instance
        atoms = []
        label_dict = {}
        for indeks, pos in enumerate(self.atoms.positions):
            ele = self.atoms[indeks].symbol
            atomic_number = atomic_numbers[ele]
            color = jmol_colors[atomic_number]
            size_to_give = 20. * covalent_radii[atomic_number]
            handle = ax.plot(
                    *zip(pos), marker='o', linestyle='points',
                    color=color, markersize=size_to_give
                )[0]
            atoms.append(handle)
            label_dict.update({ele: handle})
        tetrahedra = []
        for index, node in enumerate(self.nodes):
            try:
                points = list(node.cage_positions)
            except:
                node.initialize_hull()
                points = list(node.cage_positions)
            assert len(points) == 4
            simplices = ConvexHull(points).simplices
            poly3d = [
                    [
                        points[simplices[ix][iy]]
                        for iy
                        in range(len(simplices[0]))
                    ]
                    for ix
                    in range(len(simplices))
                ]
            collection = Poly3DCollection(poly3d, linewidths=0.015, alpha=0.00)
            face_color = 'white' #scalarMap.to_rgba(np.random.random())
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)
            tetrahedra.append(collection)
        ax.legend(
                label_dict.values(), label_dict.keys(),
                frameon=False, loc=3, numpoints=1,
                ncol=len(label_dict)
            )

        bar = ax2.barh([0.2], [1], color = 'green')[0]
        plt.xticks(range(0, len(self.trajectory), len(self.trajectory)/10))
        plt.yticks([])
        ax2.set_axis_off()
        progress_title = plt.title('Progress', fontsize = 15)
        ax.autoscale(False)

        fps = 20

        timesteps_to_play = 30*fps if save_animation else len(self.trajectory)
        ani = animation.FuncAnimation(
                fig, get_timestep, timesteps_to_play,
                fargs=(
                        ax, ax2, self.trajectory[::25],
                        self.site_traj[::25],
                        atoms, tetrahedra, bar,
                        progress_title
                    ),
                blit=False
            )

        if save_animation:
            folder = '{}/Videos/videos_pub/lithium_animations'.format(os.getenv('HOME'))
            i = 1
            while True:
                filepath = '{}/{}_{}.mp4'.format(folder, save_animation, i)
                if not os.path.exists(filepath):
                    break
                i +=1
            print 'writing to', filepath
            ani.save(filepath, fps=fps, dpi = 200)
        else:
            plt.show()

    def plot_track_results(
            self, plot_on_screen=False, name='', block=False, **kwargs
        ):
        """
        Plot the results of the track calculations.
        Assumes that this instance of VoronoiNetwork either just ran
        :func:`~carlo.codes.voronoi.decomposition.VoronoiNetwork.track_ions_in_traj`
        or that the results of a previous run of
        :func:`~carlo.codes.voronoi.decomposition.VoronoiNetwork.track_ions_in_traj`
        were passed to
        :func:`~carlo.codes.voronoi.decomposition.VoronoiNetwork.set_site_traj`.

        .. figure:: /images/mp-560104_Li1Ta1Ge1O5-temp-800.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center

            Site trajectory of a bad conductor

        .. figure::  /images/mp-37399_Li7Nb1O6-temp-800_site_traj.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center

            Site trajectory of a good conductor
            
        Plot the results of the track calculations concerning the volume of each
        site during the trajectory. Assumes that this instance of VoronoiNetwork
        either just ran a track_ions_in_traj or that the volumes were passed  to
        :func:`~carlo.codes.voronoi.decomposition.VoronoiNetwork.set_volumes`.
        Creates two subplots:

        #.  The volume evolution for each site
        #.  The total volume over time, the linear regression and the unit cell
            volume for visual analysis to detect melting.

        .. figure:: /images/mp-754060_Li7Bi1O6-temp-800_volume_evolution.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center

            Volume evolution standard run

        .. figure::  /images/mp-645317_Li4C1O4-temp-800_volume_evolution.pdf
            :figwidth: 100 %
            :width: 100 %
            :align: center

            Volume evolution melted material
        """
        def myrepl(matchobj):
            return '_{{{}}}'.format(matchobj.group(0))
        gs = gridspec.GridSpec(3,1, hspace = 0.4)
        X = range(len(self.site_traj))
        unitcellvolume = self.track_results['unitcellvolume']
        color = '#ED8E8D' if self.track_results['melted'] else '#8EED8D'
        slope = self.track_results['volume_reg']['slope']
        intercept = self.track_results['volume_reg']['intercept']

        fig = plt.figure(figsize = (16,9))


        ################### PLOT 1 SITE TRAJECTORY #############################
        ax0 = fig.add_subplot(gs[0])
        try:
            formula = re.search(
                    '([A-Z][a-z]?[0-9]{1,2})+(?=-temp)',
                    name
                ).group(0)
            formula = re.sub('\d+', myrepl, formula)
            temp = re.search('(?<=temp-)\d+', name).group(0)
            plt.title(
                    'Site trajectory of ${}$ at {}K'.format(formula, temp),
                    fontsize=18
                )
        except:
            plt.title('Site trajectory of {}'.format(name), fontsize=20)
        zipped_list = zip(*self.site_traj)
        plt.ylabel(r"Site ID", fontsize = 16)
        plt.xlabel('Time [dt]', fontsize = 16)
        for tick in ax0.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax0.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        for t in zipped_list:
            ax0.plot(t)


        ################### PLOT 2 INDIVIDUAL VOLUMES ##########################
        ax1 = fig.add_subplot(gs[1])
        for z in zip(*self.volumes):
            ax1.plot(z)
        plt.ylabel(r"Volume $[\AA^3]$", fontsize = 16)
        plt.xlabel(r"Timesteps [dt]", fontsize = 16)
        plt.title ('Volume evolution individual site hull', fontsize = 20)


        ################### PLOT 3 TOTAL VOLUMES ###############################
        ax2 = fig.add_subplot(gs[2], axisbg = color)

        Y = [slope*x+intercept for x in X]
        ax2.plot(X,Y, label = 'Lin. Regression')
        ax2.plot(X, self.total_volume, label = 'Total volume of cages')
        ax2.plot(X, [unitcellvolume]*len(X), label = 'Supercell volume')
        ax2.legend(loc = 2)

        plt.ylabel(r"Volume $[\AA^3]$", fontsize = 16)
        plt.xlabel(r"Timesteps [dt]", fontsize = 16)
        plt.title (
                'Volume evolution - sum over all site hulls in unit cell',
                fontsize=20
            )
        plt.suptitle(name, fontsize = 24)
        plt.show()


    def view(self, repeat = [1,1,1]):
        """
        Visualization of input structure and the site hulls (one by one)
        """
        print 'Showing you the atoms you gave me'
        view(self.atoms)
        atoms = self.atoms.copy()
        try:
            for node in self.nodes:
                atoms.append(Atom('H', node.xyz))
            print(
                    'Showing you the voronoi {} nodes'
                    ''.format(
                        len([atom for atom in atoms if atom.symbol == 'H'])
                    )
                )
            view(atoms.repeat(repeat))
        except Exception as e:
            print e
            print 'No nodes to show'



    def reduce_by_volume(self, min_ratio):
        """
        Reduces the number of nodes by volume intersection.
        Each node is seen as a sphere centered at the Voronoi node with radius
        the Voronoi radius (distance to vertices). If the ratio between the
        overlap of the spheres and the volume of smaller sphere is larger than
        min_ratio, the nodes are collapsed into one node
        """
        ########################################################################
        def get_sphere_intersection(pos_1,  pos_2,rad_1, rad_2):
            # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
            # V =(pi(R+r-d)^2(d^2+2dr-3r^2+2dR+6rR-3R^2))/(12d).
            # pi is set to 1 here
            d = np.linalg.norm(np.array(pos_1) - np.array(pos_2))
            # 0th case, same position:
            if not d:
                return 1.0
            #first case, spheres do not touch:
            if d > rad_1 + rad_2:
                return 0.0
            r_min  = min([rad_1, rad_2])
            r_max = max([rad_1, rad_2])
            #second case, one sphere encapsulates the other:
            if r_max > r_min + d:
                return 1.0
            V_intersect = \
                (rad_1+rad_2 - d)**2 * \
                (d ** 2  + 6*rad_1*rad_2+2* d*(rad_1 +rad_2) - \
                3 * (rad_1**2 + rad_2**2)) /(12 * d)
            V_sphere = 4.0 / 3.0 * r_min**3
            return V_intersect / V_sphere
        ########################################################################

        nodes_to_delete = set()
        new_nodes = []
        for i1 in range(len(self.nodes)):
            for i2 in range(i1+1,len(self.nodes)):
                if get_sphere_intersection(
                        self.nodes[i1].xyz,
                        self.nodes[i2].xyz,
                        self.nodes[i1].radius,
                        self.nodes[i2].radius
                    ) > min_ratio:
                    nodes_to_delete.add(i1)
                    nodes_to_delete.add(i2)
                    added_to = None
                    for nodesetindex, nodeset in enumerate(new_nodes):
                        if i1 in nodeset:
                            nodeset.add(i1)
                            nodeset.add(i2)
                            if added_to is None:
                                added_to = nodesetindex
                            else:
                                nodeset = nodeset.union(new_nodes.pop(added_to))
                    if added_to is None:
                        new_nodes.append(set([i1,i2]))
        for set_of_nodes in new_nodes:
            new_xyz = [
                    sum(pos)/len(set_of_nodes)
                    for pos in
                    zip(*[
                            self.nodes[nodeindex].xyz
                            for nodeindex
                            in set_of_nodes
                        ])
                ]
            #union of vertices of old nodes gives new nodes'vertices
            new_vertices =  list(set(flatten([
                    self.nodes[nodeindex].vertice_list
                    for nodeindex
                    in set_of_nodes
                ])))
            new_node = VoronoiNode(
                    vertices=new_vertices, xyz=new_xyz,
                    atoms=self.atoms, network=self
                )
            new_node.initialize_hull()
            self.nodes.append(new_node)

        for i in sorted(nodes_to_delete, reverse = True):
            self.nodes.pop(i)
        self._log.write(
                '   {} old nodes will be collapsed into {} new nodes\n'
                ''.format(len(nodes_to_delete), len(new_nodes))
            )
        self._log.write(
                '   Combined hull volume:  {}\n'
                ''.format(sum(self.get_volumes()))
            )

    def reduce_by_radius(self, min_radius):
        """Reduces number of nodes by deleting every Voronoi cell below certain
        threshold radius. DEPRECATED"""
        raise DeprecationWarning
        self.nodes = [node for node in self.nodes if node.size > min_radius]

    def reduce_by_collapse(self, threshold=0.0, check_plain=False):
        """
        Reduces number of nodes by collapsing 2 nodes closer than threshold into
        one node. Method of choice, since until now the only method that can
        make a union of the cage atoms, therefore theoretically keeping the
        volume of the voronoi network constant, however this seems not to work
        always
        """
        while True:
            i_collapsed = False
            for i1, n1 in enumerate(self.nodes):
                for i2,n2 in enumerate(self.nodes):
                    if i1==i2 or np.linalg.norm(n1.xyz - n2.xyz) > threshold:
                        # they are either the same atoms, have distance above
                        # threshold
                        continue
                    if check_plain and len(n1.shell.union(n2.shell))<3:
                        # If wanted, nodes are only collapsed if their voronoi
                        # cells have a common plane
                        continue
                    i_collapsed =  True
                    new_node = VoronoiNode(
                            node_id=n1.node_id,
                            xyz=0.5*(n1.xyz+n2.xyz),
                            vertices=list(n1.vertice_set.union(n2.vertice_set)),
                            network=self,
                            atoms=self.atoms, images=list(set(n1.images).union(set(n2.images)))
                       )
                    new_node.update_hull()
                    new_node.initialize_hull()
                    self.nodes[i1] = new_node
            if not i_collapsed:
                break


    def reduce_by_crowding(self, threshold = 0.7):
        """
        Reduces number of nodes by deleting nodes that are closer than threshold
        to another node with larger sphere. DEPRECATED
        """
        raise DeprecationWarning
        help_atoms = Atoms(cell = self.atoms.cell[:])
        [help_atoms.append(Atom('H', node.xyz)) for node in self.nodes]
        distances = [
                (
                    atom_distance(
                        help_atoms[i],
                        help_atoms[j],
                        help_atoms.cell[:]
                    ),
                    i,
                    j
                )
                for i in range(len(help_atoms))
                for j in range(len(help_atoms))
                if i<j
            ]
        deleted_set = set()
        for d,i,j in sorted(distances):
            if d > threshold: break
            if i in deleted_set or j in deleted_set:
                continue #This distance should not exist
            rad_i = self.nodes[i].size
            rad_j = self.nodes[j].size
            if rad_i > rad_j: deleted_set.add(j)
            else: deleted_set.add(i)
        print deleted_set
        self.nodes = [
                node
                for index, node
                in enumerate(self.nodes)
                if index not in deleted_set
            ]
        help_atoms = self.delithiated.copy()
        [help_atoms.append(Atom('H', node.xyz)) for node in self.nodes]




    def grid_check(self, grid_density, plot = False):
        storage_dir = '/home/kahle/temp/grid_check'
        storage = '{}/{}'.format(storage_dir, 'grid_check_GRIDDENSITY_TIMESTEP.npy')
        killfile = '{}/{}'.format(storage_dir, 'killfile')
        A =  np.array(self.atoms.cell[0])
        B =  np.array(self.atoms.cell[1])
        C =  np.array(self.atoms.cell[2])
        gridpoints = [
            (float(i)/grid_density *A +float(j)/grid_density *B +float(k)/grid_density *C).tolist()
            for i in range(grid_density)
            for j in range(grid_density)
            for k in range(grid_density)
        ]
        [node.initialize_hull() for node in self.nodes]
        grid_size = grid_density ** 3

        for index, positions in enumerate(self.trajectory):
            print 'AT TIMESTEP', index
            filename = storage.replace('TIMESTEP', str(index)).replace('GRIDDENSITY', str(grid_density))
            try:
                count_list =  np.load(filename)
                if len(count_list) != grid_size:
                    raise Exception
            except:
                if plot:
                    break
                f = open(killfile)
                if 'KILL' in f.read():
                    break
                f.close()
                print '   checking decomposition with grid of {0} x {0} x {0}'.format(grid_density)
                if index:
                    [node.update_hull(positions) for node in self.nodes]
                count_list = np.array([[node.inside(point) for node in self.nodes].count(True) for point in gridpoints], dtype = int)
                np.save(filename, count_list)
            double_count = len([val for val in count_list if val >1])
            not_found_count = len([val for val in count_list if val == 0])
            print '   double    instances: {0}'.format(double_count)
            print '   not-found instances: {0}'.format(not_found_count)

        nr_of_timesteps = index

        if not plot:
            return

        def get_timestep(trajectory_index):
            print 'AT TIMESTEP', trajectory_index
            timestep = self.trajectory[trajectory_index]
            filename = storage.replace('TIMESTEP', str(trajectory_index)).replace('GRIDDENSITY', str(grid_density))
            count_list =  np.load(filename)
            #~ print len(count_list)
            #~ print len(points_in_plot)
            #~ print points_in_plot
            new_points = []
            for index,count in enumerate(count_list):
                if count == 1:
                    continue
                elif count == 0:
                    new_points.append(ax.plot(
                            *[[i] for i in gridpoints[index]],
                            color='red', markersize=10, alpha=1,
                            marker='o', linestyle='points'
                        ))
                else:
                    ax.plot(*[[i] for i in gridpoints[index]], color='green')

            for indeks, atom in enumerate(atoms):
                newdata = [[i] for i in timestep[indeks]]
                atom.set_data(*newdata[:2])
                atom.set_3d_properties(newdata[2])


        fig = plt.figure(figsize = (16, 9))
        ax = p3.Axes3D(fig)
        ax.set_axis_off()
        pp = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
        for p1 in pp:
            for p2 in pp:
                #skip unnecessary points:
                counts = [p1[i] == p2[i] for i in range(3)]
                if not counts.count(True) == 2:
                    continue
                ax.plot(*zip(
                        np.dot([p1], self.atoms.cell).tolist()[0],
                        np.dot([p2], self.atoms.cell).tolist()[0]
                    ),
                    color = (0,0,0)
                )


        atoms = []
        for indeks, pos in enumerate(self.atoms.positions):
            ele = self.atoms[indeks].symbol
            color = jmol_colors[atomic_numbers[ele]]
            atoms.append(ax.plot(
                    *zip(pos), marker='o', linestyle='points',
                    color=color, markersize=10, alpha =0.7
                )[0]
            )

        tetrahedra = []
        for index, node in enumerate(self.nodes):
            try:
                points = list(node.cage_positions)
            except:
                node.initialize_hull()
                points = list(node.cage_positions)
            assert len(points) == 4
            simplices = ConvexHull(points).simplices
            poly3d = [
                [
                    points[simplices[ix][iy]]
                    for iy in range(len(simplices[0]))
                ]
                for ix
                in range(len(simplices))
            ]
            collection = Poly3DCollection(poly3d, linewidths=0.3, alpha=0)
            face_color = 'white' #scalarMap.to_rgba(np.random.random())
            collection.set_facecolor(face_color)
            ax.add_collection3d(collection)
            tetrahedra.append(collection)

        conspicuous_points = []


        ani = animation.FuncAnimation(
            fig,get_timestep,nr_of_timesteps,interval=10)
        plt.show()


    #### ANALYSIS SECTION ######################

    def get_site_adjacency_matrix(self):
        return [
            [
                int(node1.is_neighbor_of(node2))
                for node1
                in self.nodes]
            for node2
            in self.nodes
        ]

    def get_vertice_adjacency_matrix(self):

        for vertice1  in  self.mapping.values():
            for vertice2 in self.mapping.values():
                if vertice1 == vertice2:
                    continue
                print int(any([node.contains(vertice1, vertice2) for node in self.nodes])),
            print
        #~ return [[int(node1.is_neighbor_of(node2)) for node1 in self.nodes] for node2 in self.nodes]


    def find_neighbors(self):
        """
        Find the neighbors of each of my nodes
        """
        [node.find_neighbors() for node in self.nodes]

    def volume_explored(self, log = sys.stdout, check = False):
        """
        This functions measures the volume explored of the ions in a site
        """

        log.write('   Starting to measure the volume explored in each occupied site\n')
        [node.initialize_volume_list() for node in self.nodes]
        #~ print 1, self.tracked_ions
        if check:
            for timestep, positions in enumerate(self.trajectory):
                [
                    self.nodes[nodeindex].add_volume_point(
                        positions[self.tracked_ions[str(ionindex)]],
                        positions
                    )
                    for ionindex, nodeindex
                    in enumerate(self.site_traj[timestep])
                ]
                [
                    node.update_hull(positions)
                    for index, node
                    in enumerate(self.nodes)
                    if index in self.site_traj[timestep]
                ]
                if not(all([
                        self.nodes[nodeindex].inside(
                            positions[self.tracked_ions[str(ionindex)]]
                        )
                        for ionindex, nodeindex
                        in enumerate(self.site_traj[timestep])])):
                    print 'Ion not inside'

        else:
            [
                [
                    self.nodes[nodeindex].add_volume_point(
                        positions[self.tracked_ions[str(ionindex)]],
                        positions
                    )
                    for ionindex, nodeindex
                    in enumerate(self.site_traj[timestep])
                ]
                for timestep, positions
                in enumerate(self.trajectory)
            ]
        return [node.get_volume_explored() for node in self.nodes]

    def get_structure_similarity(self):
        """Return the similarity with 'perfect' substructures for each node"""
        return [node.get_structure_similarity() for node in self.nodes]

    def get_squeze_param(self):
        """
        Get the space (area) that the ion has to squeeze through the phase to
        its neighboring nodes
        """
        return [node.get_squeze_param() for node in self.nodes]



    def jump_analysis(self, **kwargs):
        """
        The main function to analyze the jumps happened during a
        simulation as returned by
        :func:`VoronoiNetwork.track_ions_in_traj`
        For each jump, the following quantities are measured:

        *   The observed average mean lifetime matrix :math:`M^{\\tau}` in
            femtoseconds, stored in *self.jump_lifetime_mean_fs_matrix*.
            That one is phenomenologically created by observing how long an ion
            stayed at site A before jumping to site B
        *   The count matrix :math:`M^c`, storing as integers the number of jumps
            observed from A to B, stored in *self.count_matrix*.
            Created with a counter in a loop going through all observed jumps.
        *   The stochastic matrix :math:`M^s`, storing as floats all the
            probability that if ion is at A it will jump to B.
            Is calculated from the count matrix:

            #.  For each site A (stored in rows), get the total number of jumps
                (:math:`N_a`) out of site A.
            #.  For each jump A to B, populate :math:`M^s` with
                :math:`M_{a,b}^s = M^c_{a,b} / N_a`

            This matrix is stored in *self.stochastic_matrix*
        *   The probablity matrix *P* is defined as
            :math:`P_{a,b} = M^s_{a,b} \cdot M_a / N_a`
            where :math:`M_a / N_a` is the average time an ion spends at A
            before jumping out
            (:math:`M_a = \sum_b M^{\\tau}_{a,b} \\cdot M^s_{a,b}`).
            Stored in *self.propability_fs_matrix* 
            (probability given in :math:`\\frac{1}{fs}`)
        *   the probablity to jump to the neighbor

        Anything that is site specific should be
        stored in a vector / a list.
        Anything that is jump and therefore site-pair
        specific should be stored in a matrix

        .. figure:: /images/jumpanalysis.pdf

            :align: center

            The standard jump analysis
        """
        #~
        #~ self.symmetry_matrix            = np.array(symmetry_matrix)
        #~ self.stochastic_matrix          = np.array(stochastic_matrix)
        #~ self.detailed_balance_fs_matrix = np.array(detailed_balance_fs_matrix)
        #~ self.propability_fs_matrix      = np.array(propability_fs_matrix)
        #~ self.jump_lifetime_mean_fs_matrix = np.array(jump_lifetime_mean_fs_matrix)
        #~ self.count_matrix               = np.array(count_matrix)
        #~ self.propability_fs_matrix      = np.array(propability_fs_matrix)
        #~ self.visited_nodes              = visited_nodes

        enforce_dtbalanced = kwargs.get('enforce_dtbalanced', False)
        if enforce_dtbalanced:
            site_trajectory_to_analyze = np.array(
                    self.site_traj.tolist() + self.site_traj.tolist()[::-1]
                )
        else:
            site_trajectory_to_analyze = np.array(self.site_traj)
        simulation_time_dt = len(site_trajectory_to_analyze)
        simulation_time_fs = simulation_time_dt*self.timestep_in_fs

        # Detect all the jumps that occur
        self.jumps = detect_jumps(site_trajectory_to_analyze)
        self.nr_of_jumps = sum([len(sp_jumps) - 1 for sp_jumps in self.jumps])
        self._log.write(
            '   Site trajectory to analyze of shape:    {} \n'
            '   Jump Detector detected jumps per ion:   {} \n'
            '   Total jumps:{}\n'.format(
                site_trajectory_to_analyze.shape,
                ', '.join(
                    [str(len(jumptraj)-1) for jumptraj in self.jumps]
                ),
                self.nr_of_jumps
            )
        )

        # Now, get simple statistics of jumps.
        # See which jumps actually occured and
        # find all the sites that were visited:
        visited_sites_set = set()
        [map(visited_sites_set.add, zip(*ion_traj)[0]) for ion_traj in self.jumps]

        # Make a list of the visited sets
        visited_sites_list = sorted(visited_sites_set)

        #make a map old_index -> new index
        visited_sites_map = {
                val:index
                for index, val
                in enumerate(visited_sites_set)
            }

        #all the nodes that were visited:
        visited_nodes = [self.nodes[index] for index in visited_sites_list]
        nr_of_visited_nodes = len(visited_nodes)

        ### VECTORS AND STATISTICS ON SITES (NOT STORED IN THE END) ####
        #~ counts_vector = [0] * nr_of_visited_nodes

        # The visittimes_mean_fs_vector is a vector of the lifetime that
        # the ion spend at the corresponding site
        # It should never be 0 / nan because every site is visited at least once
        # TODO: Check, is that correct??
        #~ visittimes_mean_fs_vector = map(get_mean_or_0,  visittimes_fs_vector)


        ######################### COUNT MATRIX #########################
        # First I count the number of jump occurences,
        # that is how many times an ion jumped from a site A to a site B.
        count_matrix = np.zeros(
                [nr_of_visited_nodes, nr_of_visited_nodes]
            )
        for jumps_this_ion in self.jumps:
            for index, val in enumerate(jumps_this_ion[1:]): #avoid index error
                # site from is the last visit in jumps_this_ion,
                # since index gives the value in jumps_this_ion one before val
                site_from = visited_sites_map[jumps_this_ion[index][0]]
                site_to = visited_sites_map[val[0]]
                count_matrix[site_from][site_to] += 1

        ##################### LIFETIMES MATRIX ########################
        # Now I count the lifetimes of each jump.
        # Lifetime is defined as the time of an ion entering the site to
        # an ion leaving the site.
        # The lifetime_matrix contains every single of these events,
        # so that later we can calculate mean and standard deviation

        # instantiate matrix as 2d rank tensor of lists
        lifetimes_matrix = [
            [[] for j in range(nr_of_visited_nodes)]
            for i in range(nr_of_visited_nodes)
        ]
        # populate it by counting the lifetimes
        for jumps_this_ion in self.jumps:
            for index, val in enumerate(jumps_this_ion[1:]): #avoid index error
                site_from = visited_sites_map[jumps_this_ion[index][0]]
                site_to = visited_sites_map[val[0]]
                lifetimes_matrix[site_from][site_to].append(
                        (val[1]-jumps_this_ion[index][1])*self.timestep_in_fs
                    )

        # Now we calculate the mean of each list in lifetimes_matrix.
        # Be aware that if no jump occured, np.mean will return NaN which
        # cannot be stored in a JSON.
        # We will set this value to 0 for now, but technically it should be
        # infinity!
        jump_lifetime_mean_fs_matrix = [
                map(get_mean_or_0, row)
                for row
                in lifetimes_matrix
            ]


        ############# STOCHASTIC MATRIX, PROBABILITY MATRIX ############
        # Let's look at the propabilities that an ion at site i will be found at 
        # sound j after:
        # We call this matrix stochastic_matrix
        # let's produce another matrix, the propability_matrix
        # This gives the propabilities (per fs) that a jump will occur
        # from A to B if ion is at A.

        stochastic_matrix = [None]* nr_of_visited_nodes
        propability_fs_matrix = [None]* nr_of_visited_nodes
        for site_from, counts_out_of_site_from in enumerate(count_matrix):
            total_nr_of_jumps = sum(counts_out_of_site_from) # Given by
            stochastic_matrix[
                site_from
            ] = counts_out_of_site_from / total_nr_of_jumps
            propability_fs_matrix[
                site_from
            ] = stochastic_matrix[
                    site_from
                ] / np.sum(
                    jump_lifetime_mean_fs_matrix[
                            site_from
                        ]*stochastic_matrix[site_from]
                    )

        ################### DETAILED BALANCE ###########################
        # Lets look at detailed balance:
        # If P_i is the propability that a site is occupied during a trajectory,
        # and p_ij the propability that an ion will jump from i to j in one unit
        # of time then: P_i p_ij = P_j p_ji
        #~ detailed_balance_fs_matrix = [
            #~ [
                #~ p*total_visittime_fs_vector[index]/simulation_time_fs
                #~ for p in row
            #~ ]
            #~ for index, row in enumerate(propability_fs_matrix)
        #~ ]
        visittimes_fs_vector = [[] for i in range(nr_of_visited_nodes)]
        for jumps_this_ion in self.jumps:
            for index, jump in enumerate(jumps_this_ion[:-1]):
                site, timestep = jump
                #~ counts_vector[visited_sites_map[site]] += 1
                visittimes_fs_vector[visited_sites_map[site]].append(
                        self.timestep_in_fs*float(
                            jumps_this_ion[index+1][1]-timestep
                        )
                    )
        total_visittime_fs_vector = map(sum, visittimes_fs_vector)
        detailed_balance_fs_matrix = np.array(
                map(np.dot, propability_fs_matrix,total_visittime_fs_vector)
            ) / simulation_time_fs


        ################## SYMMETRY MATRIX ####################################
        symmetry_matrix=detailed_balance_fs_matrix-detailed_balance_fs_matrix.T

        ############ DONE let's print out information###########################
        if self.verbosity:
            labels = (
                    ' stochastic_matrix, propability_dt_matrix,'
                    'detailed_balance_dt_matrix, symmetry_check'
                    ''.split(',')
                )
            for ii, matrix in enumerate([
                    stochastic_matrix,propability_fs_matrix,
                    detailed_balance_fs_matrix, symmetry_matrix
                ]):
                self._log.write('\n\n{}\n\n'.format( labels[ii].upper()))
                self._log.write(
                        '    {}\n'
                        ''.format(' '.join([
                                '{:<8}'.format(index)
                                for index
                                in range(len(stochastic_matrix))
                            ])
                        )
                    )
                for index, row in enumerate(matrix):
                    self._log.write(
                            '{:<4}{}\n'.format(
                                index, ' '.join([
                                        '{:<8}'.format('{:.3}'.format(i))
                                        for i
                                        in row
                                ])
                            )
                        )
                self._log.write('\n\n\n')

        self.symmetry_matrix            = np.array(symmetry_matrix)
        self.stochastic_matrix          = np.array(stochastic_matrix)
        self.detailed_balance_fs_matrix = np.array(detailed_balance_fs_matrix)
        self.propability_fs_matrix      = np.array(propability_fs_matrix)
        self.jump_lifetime_mean_fs_matrix=np.array(jump_lifetime_mean_fs_matrix)
        self.count_matrix               = np.array(count_matrix)
        self.propability_fs_matrix      = np.array(propability_fs_matrix)
        self.visited_nodes              = visited_nodes


        self.visited_sites_dict =  {
            'sites' : [node.getlist() for node in self.visited_nodes],
            #~ 'tracked_ions': self.tracked_ions,
            'size'  : len(self.visited_nodes),
            'cell'  : self.atoms.cell.tolist(),
            'atoms' : self.atoms.get_chemical_symbols(),
            'positions' : self.atoms.positions.tolist(),
        }
        return dict(
            symmetry = self.symmetry_matrix,
            stochastic = self.stochastic_matrix,
            detailed_balance = self.detailed_balance_fs_matrix,
            probability = self.propability_fs_matrix,
            jump = self.jump_lifetime_mean_fs_matrix,
            count = self.count_matrix,
        ), self.visited_sites_dict

    @staticmethod
    def get_jump_analysis_array_names():
        return (
            'symmetry', 'stochastic', 'detailed_balance',
            'probability', 'jump', 'count'
        )

    def set_jump_analysis_results(
            self, symmetry, stochastic, detailed_balance,
            probability, jump, count, visited_sites_dict
        ):
        self.symmetry_matrix = symmetry
        self.stochastic_matrix = stochastic
        self.detailed_balance_fs_matrix = detailed_balance
        self.propability_fs_matrix = probability
        self.jump_lifetime_mean_fs_matrix = jump
        self.count_matrix = count
        self.visited_sites_dict = visited_sites_dict



    def plot_jump_analysis(self, name=None, block=True, aminice=None, **kwargs):

        atoms = self.visited_sites_dict['atoms']
        positions = self.visited_sites_dict['positions']
        cell = self.visited_sites_dict['cell']
        self._log.write('Making figure\n\n')
        nr_of_visited_nodes = len(self.count_matrix)

        gs = gridspec.GridSpec(
                2,4, height_ratios=[2,1],
                left=0.05, right=0.9,
                hspace=0.25, wspace=0.4
            )

        #~ fig = plt.figure(figsize = (25,16))
        fig = plt.figure(figsize = (16,9))
        #~ if name:
            #~ plt.suptitle('Analysis of {}'.format(name), fontsize = 20)

        #~ ax0 = fig.add_subplot(gs[0,:3])
        ax0 = fig.add_subplot(gs[0,:2])

        plotlist = [
            [
                site_from, site_to,lifetime,
                self.stochastic_matrix[site_from][site_to],
                self.count_matrix[site_from][site_to]
            ]
            for site_from, row in enumerate(self.jump_lifetime_mean_fs_matrix)
            for site_to, lifetime in enumerate(row)
        ]

        x,y, lifetimes, connectivity, count = zip(*plotlist)
        maxlife = max(lifetimes)
        maxcon = max(connectivity)
        maxcount = float(max(count))
        #scale connectivity so that highest value is 50
        conscale = 1000.0/float(maxcount)

        largest_scatterpoint_size = 100.0
        countscale = largest_scatterpoint_size / maxcount
        count_scaled = [countscale*c for c in count]
        cmap = plt.cm.rainbow

        plt.ylabel("Jump destination [id]", fontsize = 14)
        plt.xlabel('Jump departure site [id]', fontsize = 14)
        plt.title("Transition counts and lifetimes", fontsize = 16)
        plt.xlim(-1,nr_of_visited_nodes)
        plt.ylim(-1,nr_of_visited_nodes)
        plt.xticks([])
        #range(len(visited_nodes)))
        plt.yticks([]) #range(len(visited_nodes)))

        f = ax0.scatter(x,y, cmap  = cmap, c = lifetimes, s =  count_scaled)

        #patch to also show legend explaining size

        ls_n_labels = [
                (
                    plt.scatter(
                            [],[], s=float(i)/4*countscale*maxcount,
                            edgecolors='none'
                    ),
                    int(float(i)/4*maxcount)
                )
                for i
                in range(1,5)
            ]
        ls, labels = zip(*ls_n_labels)
        #~ .subplots_adjust(top=0.9, bottom = 0.12)
        leg = ax0.legend(
                ls, labels, ncol=5, frameon=True, fontsize=10, handlelength=2,
                loc='lower center', borderpad=0.4, handletextpad=0.4,
                title='Observed jumps', scatterpoints=1, fancybox=True
            )
        leg.get_frame().set_alpha(0.3)

        # plotting the right side of top row, connectivity graph
        
        ax1 = fig.add_subplot(gs[0,2:], projection='3d')
        
        if aminice is None:
            niceness = ''
            bgcolor = "white"
        elif aminice:
            niceness = "Nice "
            bgcolor = "#BDF7A4"
        else:
            niceness = "Bad "
            bgcolor = "#F7A4A4"
        ax1.set_axis_bgcolor(bgcolor)
        plt.title(r"{}Structure (${}$) and pathways".format(
                niceness, 
                ''.join([
                    '{}_{{{}}}'.format(atom, atoms.count(atom))
                    for atomindex, atom
                    in sorted([
                            (atomic_numbers[atom], atom)
                            for atom
                            in set(atoms)
                        ])
                ])
            ),
            fontsize = 18
        )
        ax1.set_axis_off()

        maxcon  = max([max(row) for row in self.detailed_balance_fs_matrix])
        conscale = 5. / maxcon
        maxact  = max([sum(row) for row in self.detailed_balance_fs_matrix])
        actscale = 30. /maxact
        handles = []
        #initializing figure instance

        #draw the atoms as balls
        for i, atom in enumerate(atoms):
            #~ ele = atom.symbol
            #~ color = list(rgb_to_hls(*jmol_colors[atomic_numbers[ele]]))
            #~ color[1] = 0.5  #I am setting down the lightness
            #~ color =  hls_to_rgb(*color) #back to rgb
            #~ print atom, color
            # I want color as used in ASE
            color = jmol_colors[atomic_numbers[atom]]
            pos = [[j] for j in positions[i]]
            ax1.plot(
                    *pos, marker='o', linestyle='points',
                    color=color, markersize=10
                )
        # Legend for atoms:
        [
            ax1.plot(
                    [],[],[],
                    marker= 'o', linestyle='points',
                    color=jmol_colors[atomic_numbers[atom]],
                    markersize=10, label=atom
            )
            for atom in set(atoms)
        ]
        ax1.plot(
                [],[],[],
                marker='^', linestyle='points', markersize=6,
                color='b', label='Site'
            )
        plt.legend(
                loc = 'lower left',
                #~ loc = 'lower center',
                numpoints=1,
                #~ ncol = len(ls_n_labels)+1
            )
        divider = make_axes_locatable(ax0)
        cax = divider.append_axes("right", size="1%", pad=0.01)
        cb = plt.colorbar(f, cax)
        cb.set_label('Lifetimes [fs]', fontsize = 14)

        #get black lines for the cell
        pp = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
        for p1 in pp:
            for p2 in pp:
                counts = [p1[i] == p2[i] for i in range(3)]
                if not counts.count(True) == 2: continue
                ax1.plot(
                        *zip(
                                np.dot([p1], cell).tolist()[0],
                                np.dot([p2], cell).tolist()[0]
                            ),
                            color = (0,0,0)
                    )

        #drawing the connections:
        cmap = plt.cm.autumn
        cNorm  = colors.Normalize(vmin=0, vmax=100)
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
        plotted_sites = set()
        for site_from, row in enumerate(self.detailed_balance_fs_matrix):
            pos1 = np.array(self.visited_sites_dict['sites'][site_from][1:4])
            for site_to, con in enumerate(row):
                if not con:
                    continue
                pos2 = np.array(self.visited_sites_dict['sites'][site_to][1:4])
                pos2 = find_closest_periodic_image(
                        pos2, pos1, np.array(cell)
                    )[1].tolist()

                lifetime = self.jump_lifetime_mean_fs_matrix[site_from][site_to]

                handles,  = ax1.plot(
                        *zip(pos1, pos2),
                        linewidth=conscale*con,
                        color=scalarMap.to_rgba(lifetime)
                    )
                scalarMap.set_array(lifetime)
                if str(pos2) in plotted_sites:
                    continue
                ax1.scatter(
                        *pos2, s=actscale*np.sum(
                                self.detailed_balance_fs_matrix[:, site_to]
                            ),
                        marker='^'
                    )
                plotted_sites.add(str(pos))

        # making the bottom row
        matrices_n_labels = (
            #~ (self.count_matrix, 'Count matrix'),
            (self.jump_lifetime_mean_fs_matrix, 'Lifetime matrix'),
            (self.stochastic_matrix, 'Stochastic matrix'),
            (self.propability_fs_matrix, 'Probability matrix'),
            (self.detailed_balance_fs_matrix, 'Detailed balance matrix'),
            #~ (self.symmetry_matrix, 'Symmetry check'),
        )
        index = 0

        for matrix, label in matrices_n_labels:
            newax = fig.add_subplot(gs[1, index])
            plt.title(label, fontsize = 18)
            newax.matshow(matrix[::-1], cmap  = plt.cm.summer)
            newax.set_xticklabels([])
            newax.set_yticklabels([])
            index +=1


        plt.show(block=block)


    def get_distances(self):
        for outerkey, outerval in self.jump_dict.items():
            for innerkey, innerval in outerval.items():
                innerval['distance'] = find_closest_periodic_image(
                        self.site_dict[innerkey]['position'],
                        self.site_dict[outerkey]['position'],
                        self.cell
                    )[0]



class VoronoiNode(object):
    def __init__(self, center, vertices, translations, hash_=None):
        self._center = np.array(center)
        self._vertices = vertices
        if len(self._vertices) < 4:
            raise Exception('Invalid shell for hull')
        if hash_ is None:
            self._hash = sha224(str(sorted(self._vertices)))
        else:
            self._hash =hash_
        self._translations = translations

    def _get_hull_points(self, positions):
        hull_points = [positions[v] + self._translations[i] for i, v in enumerate(self._vertices)]
        return hull_points

    def get_hull(self, positions):
        hull_points = self._get_hull_points(positions)
        return ConvexHull(hull_points)


    def old_():
        distance_pos_image_list = [
                find_closest_periodic_image(
                        np.array(self._network._atoms.positions[i]),
                        self._center,
                        self._network._atoms.cell[:]
                    ) for i in self._vertices
            ]

        distances, self._cage_positions, self._images=zip(*distance_pos_image_list)
        self._radius = np.mean(distances)
        if np.std(distances) > 0.1 * self._radius:
            raise Exception(
                'Too large discrepancy in the distances: {}'
                ''.format(distances)
            )

        #create a list of all the tranlations there exist:
        helplist = [list(set(i)) for i in zip(*self._images)]
        #these are the images I care about:
        self._important_images = [
                (i,j,k)
                for i in helplist[0]
                for j in helplist[1]
                for k in helplist[2]
            ]

        self._hull = ConvexHull(self._cage_positions)
    def __repr__(self):
        return 'Node {}'.format(self._hash)

    def getlist(self):
        return [self.node_id]+ list(self.xyz) + [self.radius] + self.vertice_list

    def simplices(self): return self.hull.simplices

    def set_index(self, index):
        self.node_id = index
        #~ print self.content

    def initialize_hull(self):
        """
        Initialize the hull and find the correct periodic images of the your vertices
        """
        distance_pos_image_list = [
                find_closest_periodic_image(
                        np.array(self._network._atoms.positions[i]),
                        self._center,
                        self._network._atoms.cell[:]
                    ) for i in self._vertices
            ]
        distances, self._cage_positions, self._images=zip(*distance_pos_image_list)
        self._radius = np.mean(distances)
        if np.std(distances) > 0.1 * self._radius:
            raise Exception(
                'Too large discrepancy in the distances: {}'
                ''.format(distances)
            )

        #create a list of all the tranlations there exist:
        helplist = [list(set(i)) for i in zip(*self._images)]
        #these are the images I care about:
        self._important_images = [
                (i,j,k)
                for i in helplist[0]
                for j in helplist[1]
                for k in helplist[2]
            ]

        self._hull = ConvexHull(self._cage_positions)

    def update_hull(self, positions):
        """
        Updates the hull by receiving the positions of a step in the trajectory,
        it will select the relevant positions and update the hull
        """
        self._cage_positions = np.array([positions[i] for i in self._vertices
            ])+np.dot(np.array(self._images), np.array(self._network._atoms.cell[:]))
        self.hull = ConvexHull(self._cage_positions)

        return self._cage_positions #new, maybe problems?

    def get_radius(self):
        try:
            return self.radius
        except:
            self.radius = np.linalg.norm(np.array(self.cage_positions[0]) - np.array(self.xyz))
            return self.radius

    def is_neighbor_of(self, other_node):
        return len(self.vertice_set.intersection(other_node.vertice_set)) == 3

    def check(self):
        if self.in_reduced_cell: return True
        unit_cell_volume =  self.delithiated.get_volume()
        pp = [[i,j,k] for i in range(2) for j in  range(2) for k in range(2)]
        # any Voronoi node outside the hull from these points is very far and can
        # be excluded
        points = [
                np.dot(np.array(self.delithiated.cell).T, np.array([i,j,k]))
                for i,j,k
                in pp
            ]
        qh = ConvexHull(points)
        #Is there any vertice in the unit cell:
        for pos in self.cage_positions:
            help_hull = ConvexHull(qh.points.tolist() + [pos])
            a = set(map(frozenset,help_hull.simplices.tolist())).difference(set(map(frozenset,qh.simplices.tolist())))
            if not a: return True
        #is the this hull too far to have an intersection with unit cell:
        return False
        radius = self.get_radius()
        cell = np.array(self.delithiated.cell)
        p2 = cell[0] + cell[1] + cell[2]
        p1 = 0*p2
        pos = np.array(self.xyz)
        for sign, point in [1,p1], [-1,p2]:
            for vector in cell:
                vector  = sign*vector
                try:
                    discriminant = np.linalg.norm(np.dot(vector, point - pos))**2 - np.linalg.norm(point -pos)**2 + radius**2
                    #~ print discriminant
                    if discrimant < 0: raise Exception
                    sol1 = (-np.dot(vector, point- pos) + np.sqrt(discriminant)) / np.linalg.norm(vector)**2
                    sol2 = (-np.dot(vector, point- pos) - np.sqrt(discriminant)) / np.linalg.norm(vector)**2
                    if 0 < sol1 < 1: return True
                    if 0 < sol2 < sol1: return True
                except Exception as e: pass

        return False


    def inside(self, point, lattice):
        """
        :param point: position of the ion as a list [xyz], collapsed into unit
            cell
        :returns: Boolean value, true if that point is inside the hull, else False
        """
        # Assuming point in unit cell, should be done beforehand
        # We need the closest periodic image to voronoi node, could be in
        # neighboring cell
        point = find_closest_periodic_image(
                point,
                self._center,
                cell=self._network._atoms.cell,
                images=self._important_images
            )[1].tolist()
        # this is the convex hull of the vertices of site with the point included
        help_hull = ConvexHull(self.hull.points.tolist()+[point])
        # is there a difference between the two hulls
        if set(
                map(frozenset,help_hull.simplices.tolist())
            ).difference(
                set(map(frozenset,self.hull.simplices.tolist()))
            ):
            return False #there is a difference, point is not inside
        #~ self.view(point)
        else:
            # point is inside convex hull, therefore on this site, if no
            # difference was found
            return True

    def find_neighbors_by_distance(self):
        my_pos =  np.array(self.xyz)
        distance_site_list = [
                (
                    np.linalg.norm(
                        my_pos - find_closest_periodic_image(
                                site.xyz,
                                my_pos,
                                self.atoms.cell
                            )[1]
                    ),
                    indeks
                )
                for indeks, site
                in enumerate(self.network.nodes)
            ]
        self.distances, self.distance_neighbors = zip(*sorted(distance_site_list))


    def get_distance_neighbors(self):
        try:
            return self.distance_neighbors
        except:
            self.find_neighbors_by_distance()
            return self.distance_neighbors

    def get_neighbors(self):
        return [
                self.node_id
            ] + self.get_face_sharers(
            ).keys() + self.get_edge_sharers(
            ).keys() + self.get_vertice_sharers(
            ).keys()


    def get_volume(self, positions):
        #~ if positions is not None:
            #~ self._cage_positions = np.array(
                    #~ [positions[i] for i in self.vertice_list]
                #~ )+np.dot(
                    #~ np.array(self.images), np.array(self.atoms.cell[:])
                #~ )
        cage_positions = self._get_hull_points(positions)
        if len(cage_positions) == 4:
            vec1 = cage_positions[1] - cage_positions[0]
            vec2 = cage_positions[2] - cage_positions[0]
            vec3 = cage_positions[3] - cage_positions[0]
            vol = np.dot(vec3.T, np.cross(vec1, vec2))
            # why the division by 6?
            volume = abs(vol)/6.0
        else:
            qhull_inp = 'echo "\n3\n{}\n{}\n" | qhull FA Pp'.format(
                    len(self._vertices),
                    '\n'.join([
                            ' '.join([str(p) for p in pos])
                            for pos
                            in cage_positions
                    ])
                )
            qhull_out = cout(qhull_inp, shell=True)
            volume = float(
                    re.search('volume:\s*[0-9.]*', qhull_out
                ).group().split()[1])
        return volume

    def initialize_volume_list(self):
        self.volume_point_list = []
    def add_volume_point(self, point, positions):
        self.cage_positions = np.array([
                positions[i]
                for i
                in self.vertice_list
            ])+np.dot(np.array(self.images), np.array(self.atoms.cell[:]))
        point = collapse_into_unit_cell(point, self.atoms.cell[:])
        center = np.array([0,0,0])
        fac = 1.0/len(self.cage_positions)
        for i in self.cage_positions: center = center + fac*i
        point = find_closest_periodic_image(point, center, self.atoms.cell[:])[1]
        self.volume_point_list.append(point-center)



    def get_volume_explored(self):
        l = self.volume_point_list
        mean_pos = np.array([np.mean(coords) for coords in zip(*l)])
        if len(mean_pos):
            MSD = np.array([
                [
                    np.mean([
                        (pos[row] - mean_pos[row])*(pos[column] - mean_pos[column])
                        for pos
                        in l
                    ])
                    for row
                    in range(3)
                ]
                for column
                in range(3)
            ]).real
            eigvals, eigvecs = np.linalg.eig(MSD)
            diagonal_MSD = np.dot(np.dot(eigvecs.T, MSD), eigvecs).real
            volume_explored =  4.0/3.0 * np.pi *  np.linalg.det(diagonal_MSD)
        else:
            MSD = None
            diagonal_MSD = None
            volume_explored = None
        return mean_pos, MSD, diagonal_MSD, volume_explored
        if len(mean_pos):
            print 'mean: {:<40}, MSD: \n{}'.format(mean_pos, MSD)

    def get_structure_similarity(self):
        def normalize(vec):
            return vec/np.linalg.norm(vec)
        def get_tetrahedrality():
            return np.std(center_to_face_norms_sorted)/radius
        def get_cubidity():
            return np.std(self.get_distance_neighbors()[:6])/radius
        def get_octahedrality():
            """
            An octahedron is decomposed into 4 chainsaw structures, therefore
            2 faces are very close, two faces are far away, and 3 neighbors are
            very close. In the last part I include distance to site itself,
            since that is 0 and encodes the 'very close' information
            """
            return (
                    np.std(
                        center_to_face_norms_sorted[:2]+[0.0]
                    ) + np.std(
                        center_to_face_norms_sorted[2:]
                    ) + np.std(
                        self.get_distance_neighbors()[:4]
                    )
                ) / radius

        def get_trigonal_bipyramidility():
            """
            A trignal bipyramide can be composed in two ways:

            *   Just line an octahedron, but here there are only three close
                neighbors, the forth one has to be far
            *   by two stacked tetrahedra, whose nodes are quite close together
            """
            chainsaw_res = (
                    np.std(
                            center_to_face_norms_sorted[:2]+[0.0]
                        ) + np.std(
                            center_to_face_norms_sorted[2:]
                        )
                    ) / radius + np.std(
                        self.distances[:3]
                    ) / np.std(
                        self.distances[:4]
                    )
            tet_res = center_to_face_norms_sorted[0] / radius + np.std(
                    center_to_face_norms_sorted[1:4]
                ) / radius
            return min(chainsaw_res, tet_res)

        center_to_atom_list = [
                np.array(pos) - np.array(self.xyz)
                for pos
                in self.cage_positions
            ]
        radius = np.mean([
                np.linalg.norm(vec)
                for vec
                in center_to_atom_list
            ])

        center_to_face_norms = [
                abs(np.dot(
                        center_to_atom_list[i],
                        normalize(np.cross(
                            center_to_atom_list[i] - center_to_atom_list[j],
                            center_to_atom_list[i] - center_to_atom_list[k]
                        ))
                ))
                for i,j,k
                in [(0,1,2), (1,2,3), (0,1,3), (0,2,3)]
            ]

        center_to_face_norms_sorted = sorted(center_to_face_norms)
        #~ print center_to_face_norms_sorted[:2]
        similarity = {}
        similarity['tetrahedron'] = get_tetrahedrality()
        similarity['octahedron'] = get_octahedrality()
        similarity['trigonal_bipyramid'] = get_trigonal_bipyramidility()
        similarity['cube'] = get_cubidity()
        return similarity

    def get_face_sharers(self):
        """Find out which of the other nodes share a plane with me"""
        try:
            return self.face_sharers
        except:
            self.face_sharers = {
                node.node_id : list(self.vertice_set.intersection(node.vertice_set))
                for node in self.network.nodes
                if len(self.vertice_set.intersection(node.vertice_set)) ==3
            }
            return self.face_sharers

    def get_edge_sharers(self):
        """Find out which of the other nodes share an edge with me"""
        try:
            return self.edge_sharers
        except:
            self.edge_sharers = {
                node.node_id : list(self.vertice_set.intersection(node.vertice_set))
                for node
                in self.network.nodes
                if len(self.vertice_set.intersection(node.vertice_set)) ==2
            }
            return self.edge_sharers

    def get_vertice_sharers(self):
        """Find out which of the other nodes share a plane with me"""
        try:
            return self.vertice_sharers
        except:
            self.vertice_sharers = {
                    node.node_id : list(self.vertice_set.intersection(node.vertice_set))
                    for node
                    in self.network.nodes
                    if len(self.vertice_set.intersection(node.vertice_set)) ==1
                }
            return self.vertice_sharers

    def contains(self, *vertices):
        """
        Returns True if all the vertices are among my_vertices
        """
        return not( set(vertices).difference(self.vertice_set))

    def get_squeze_param(self):
        self.get_face_sharers()
        #~ self.reverse_map  = {j:i for i,j in self.mapping.items()}
        #~ self.shared_planes_dict = {}
        return_dict = {}
        #~ sys.exit()
        for neighbor, shared_vertices in self.face_sharers.items():
            positions = [
                position
                for index, position
                in enumerate(self.cage_positions)
                if self.vertice_list[index] in shared_vertices
            ]
            space = np.linalg.norm(
                np.cross(
                        positions[2] - positions[0],
                        positions[1] - positions[0]
                    )
                )
            return_dict[neighbor] = space
        return return_dict


    def view(self, atoms, point=None, block=True):
        help_atoms = Atoms(cell=atoms.cell[:])
        [help_atoms.append(atoms[j]) for  j in self._vertices]
        help_atoms.set_positions(self._get_hull_points(atoms.positions))
        help_atoms.append(Atom('H', self._center))
        if point is not None:
            help_atoms.append(Atom('Li', point))
        view(help_atoms)
        if block:
            raw_input('Enter to view next')


def collapse_into_unit_cell(point, cell):
    """
    Applies linear transformation to coordinate system based on crystal
    lattice, vectors. The inverse of that inverse transformation matrix with the
    point given results in the point being given as a multiples of lattice vectors
    Than take the integer of the rows to find how many times you have to shift
    the point back"""
    invcell = np.matrix(cell).T.I
    # point in crystal coordinates
    points_in_crystal = np.dot(invcell,point).tolist()[0]
    #point collapsed into unit cell
    points_in_unit_cell = [i%1 for i in points_in_crystal]
    return np.dot(cell.T, points_in_unit_cell).tolist()

def is_in_unit_cell(point, cell):
    """
    Applies linear transformation to coordinate system based on crystal lattice
    vectors. The inverse of that inverse transformation matrix with the point
    given results in the point being given as a multiples of lattice vectors
    Than take the integer of the rows to find how many times you have to shift 
    the point back
    """
    invcell = np.matrix(cell).T.I
     # point in crystal coordinates:
    points_in_crystal = np.dot(invcell,point).tolist()[0]
    return all([0<= c <= 1 for c in points_in_crystal])

def get_color(thisval, maxval = 50):
    #~ return np.arctan(mean)/(0.5*np.pi), 0,0
    return hls_to_rgb(thisval/maxval, 0.5, 1)

def find_min_atom_distance(pos1,pos2, cell):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    a,b,c = [np.array(i) for i in cell] #lattice vectors are a, b, c
    return min([
            np.linalg.norm(i*a+j*b+k*c + pos1 -pos2)
            for i in range(-1,2)
            for j in range(-1,2)
            for k in range(-1,2)
        ])




def find_closest_periodic_image(
        periodic_image_point,fixed_point,cell,
        images=None
    ):
    if images is None:
        # Taking all possible images
        images = IMAGES3D
    a,b,c = [np.array(i) for i in cell] #lattice vectors are a, b, c
    per_images = [i*a+j*b+k*c+periodic_image_point for (i,j,k) in images]
    distances = [np.linalg.norm(im - fixed_point) for im in per_images]
    indeks = distances.index(min(distances))
    return distances[indeks], per_images[indeks], images[indeks]

def normalize_positions(positions, cell, indices_to_ignore):
    """
    Normalize the positions of the atoms so that they are centered in the cell.
    This is important for systems in which the sublattices move with respect
    to each other. Centers only by position, masses are not take into account.

    :param positions: A numpy array of positions
    :param cell: the cell as a list 
    :param indices_to_ignore: the indices that should be not considered when
        centering, e.g. indices of the diffusing species

    :returns: centered positions (numerically centered)
    """
    a,b,c = [np.array(i) for i in cell]
    center_of_cell = 0.5*(a+b+c)
    center_of_atoms = np.array([
            np.mean([
                c
                for index, c 
                in enumerate(components)
                if index not in indices_to_ignore
            ])
            for components
            in positions.T
        ])
    shift = center_of_atoms - center_of_cell
    newpositions = positions - shift
    return newpositions


def find_closest_periodic_image2d(periodic_image_point,fixed_point, cel):
    a,b = [np.array(i) for i in cell] #lattice vectors are a, b, c
    per_images = [i*a+j*b+periodic_image_point for (i,j) in IMAGES2D]
    distances = [np.linalg.norm(im - fixed_point) for im in per_images]
    indeks = distances.index(min(distances))
    #~ for i in zip(IMAGES2D, per_images, distances): print i
    return distances[indeks], per_images[indeks], IMAGES2D[indeks]



def detect_jumps(site_trajectory_to_analyze):
    """
    Calculates all jumps that happened according to voronoi trajectory
    Needs a voronoi trajectory to have been calculated.
    Returns a list of each jump recorded for each Lithium ion:
        [(new_site, timestep), (new_site, timestep)....]
    """
    def single_particle_jump_detection(singe_particle_site_trajectory):
        """
        Find out all the jumps that happened in an MDrun by receiving as input
        the voronoi_trajectory for single ion
        """
        single_p_hops = []
        old_id = -1
        for timestep, site_id in enumerate(singe_particle_site_trajectory):
            if site_id < 0 or site_id == old_id: continue
            old_id = site_id
            single_p_hops.append([site_id, timestep])
        return single_p_hops


    jumps = [single_particle_jump_detection(i) for i in zip(*site_trajectory_to_analyze)]
    return jumps


def get_mean_or_0(l):
    if l: return np.mean(l)
    return 0
def get_std_or_0(l):
    s = np.std(l)
    if np.isnan(s):
        return 0
    return s


if __name__ == '__main__':
    from argparse import ArgumentParser
    from ase.io.trajectory import TrajectoryReader
    parser = ArgumentParser()
    parser.add_argument('trajfile', help='Required trajectory file')
    parser.add_argument('-e', '--element', nargs='+',help='Element to track', default=['Li'])
    parser.add_argument('-v', '--verbosity', action='store_true', help='Switch on verbosity')
    parsed_args = parser.parse_args(sys.argv[1:])
    reader = TrajectoryReader(parsed_args.trajfile)
    structure = reader[0]

    trajectory = Trajectory()
    trajectory.set_positions(np.array([atoms.positions for atoms in reader]), 'angstrom')
    vn = VoronoiNetwork(verbosity=parsed_args.verbosity) # I am instantiating a VoronoiNetwork instance, with verbosity and atoms to track acc. to userinput
    vn.set_atoms(structure, track=parsed_args.element)  # setting the atoms her e
    vn.set_trajectory(trajectory) # setting the trajectory (I am setting the timestep to 1 fs
    vn.decompose_qhull() # Decomposing the structure I gave (first timestep of trajectory)
    vn.track_ions() # Tracking the ions to track in the voronoi network
    #~ vn.plot_track_results() # plots
    #~ vn.jump_analysis() # jump analysis, but that should be done only for long trajectories
    #~ vn.plot_jump_analysis() #Plotting the results of jump analysis

    
    
    
    
