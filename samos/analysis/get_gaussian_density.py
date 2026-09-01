#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from samos.io.xsf import write_xsf
from samos.utils.terminal import get_terminal_width


def get_gaussian_density(trajectory, element=None, outputfile='out.xsf',
                         sigma=0.3, n_sigma=3.0, density=0.1,
                         istart=1, istop=None, stepsize=1,
                         indices_i_care=None, indices_exclude_from_plot=None):
    """
    Write the gaussian-broadened probability density of an atomic
    species to an xsf file.

    :param trajectory:
        The :class:`~samos.trajectory.Trajectory` to read positions and
        the cell from. Requires an ase.Atoms to be set.
    :param str element:
        The species to calculate the density for, has to be present in
        the chemical symbols. Ignored if indices_i_care is given.
    :param str outputfile:
        The xsf outputfile, '.xsf' is appended if missing.
    :param float sigma: The gaussian broadening to apply, in angstrom
    :param float n_sigma:
        the multiple of sigma for which to
        create the bounding box.
    :param float density: The grid spacing in angstrom
    :param int istart: Index to start reading positions
    :param int istop: Index to stop reading positions
    :param int stepsize: Take every stepsize-th step of the trajectory
    :param indices_i_care:
        The atom indices to accumulate the density over, 1-based for
        fortran. Defaults to all atoms of element, or to all atoms.
    :param indices_exclude_from_plot:
        The atom indices not written to the xsf file, 1-based.
        Defaults to indices_i_care, so that the mobile species is not
        drawn on top of its own density.
    """
    from samos.lib.gaussian_density import make_gaussian_density

    cell = trajectory.cell
    positions = trajectory.get_positions()

    nstep, nat, _ = positions.shape
    symbols = trajectory.atoms.get_chemical_symbols()
    starting_pos = trajectory.atoms.get_positions()

    if not outputfile.endswith('.xsf'):
        outputfile += '.xsf'

    # indices_i_care are used to calculate the density
    if indices_i_care is None:
        if element:
            indices_i_care = trajectory.get_indices_of_species(
                element, start=1)
        else:
            indices_i_care = np.array(list(range(1, nat+1)))

    print('(get_gaussian_density) indices_i_care:', indices_i_care)
    if not len(indices_i_care):
        raise Exception(
            'Element {} not found in symbols {}'.format(element, symbols))

    nat_this_species = len(indices_i_care)

    if istop is None:
        istop = nstep

    try:
        termwidth = get_terminal_width()
        pbar_frequency = int((istop - istart) / termwidth)
    except Exception as e:
        print('Warning Could not get progressbar ({})'.format(e))
        pbar_frequency = int((istop - istart) / 30)

    pbar_frequency = max([pbar_frequency, 1])
    a, b, c = [np.linalg.norm(cell[i]) for i in range(3)]
    n1, n2, n3 = [int(celldim/density)+1 for celldim in (a, b, c)]

    print('Grid is {} x {} x {}'.format(n1, n2, n3))
    print('Box is  {} x {} x {}'.format(a, b, c))
    print('Writing xsf file to', format(outputfile))
    if indices_exclude_from_plot is None:
        indices_exclude_from_plot = indices_i_care
    print(
        '(get_gaussian_density) We do not show these atoms in the xsf file: '
        f'{indices_exclude_from_plot}')
    # data=None writes the header and grid dimensions only;
    # make_gaussian_density appends the grid itself below.
    write_xsf(
        [s for i, s in enumerate(symbols, start=1)
         if i not in indices_exclude_from_plot],
        [p for i, p in enumerate(starting_pos, start=1)
         if i not in indices_exclude_from_plot],
        cell, data=None, outfilename=outputfile, shape=(n1, n2, n3))

    S = np.diag([1., 1., 1., -(sigma*n_sigma/density)**2])
    cellT = cell.T
    cellTI = np.linalg.inv(cellT)
    #  I describe the move from atomic to crystal
    # coordinates with an affine transformation M:
    M = np.r_[np.c_[cellTI, np.zeros(3)], [[0., 0., 0., 1.]]]
    # Q is a check, but not used. Check is orthogonality
    # Q is the sphere transformed by transformation M
    # Q = M.I.T @ S @ M.I
    # Now, as defined in the source, I calculate R = Q^(-1)
    R = M @ np.linalg.inv(S) @ M.T
    # The boundaries are given by:
    # ~ xmax = (R[0,3] - np.sqrt(R[0,3]**2 - R[0,0]*R[3,3])) / R[3,3]
    # ~ xmin = (R[0,3] + np.sqrt(R[0,3]**2 - R[0,0]*R[3,3])) / R[3,3]
    # ~ ymax = (R[1,3] - np.sqrt(R[1,3]**2 - R[1,1]*R[3,3])) / R[3,3]
    # ~ ymin = (R[1,3] + np.sqrt(R[1,3]**2 - R[1,1]*R[3,3])) / R[3,3]
    # ~ zmax = (R[2,3] - np.sqrt(R[2,3]**2 - R[2,2]*R[3,3])) / R[3,3]
    # ~ zmin = (R[2,3] + np.sqrt(R[2,3]**2 - R[2,2]*R[3,3])) / R[3,3]
    # The size of the bounding box is given by (max - min)
    # for each dimension.
    # I want this to be expressed as integer values in the grid,
    # though, for convenience.
    # In  plain terms, bx,by,bz tell me how many grid point
    # I have to walk up/down in x/y/z
    # maximally to be sure that I contain all the points that lie
    # with n_sigma*sigma from the origin!
    # Of course, of main importance is the density!
    # I add to be sure, since int cuts of floating points!
    b1 = int(np.abs(
        (R[0, 3] - np.sqrt(R[0, 3]**2 - R[0, 0]*R[3, 3]))
        / R[3, 3]) / density) + 1
    # Normally I would have to do 0.5 (xmax - xmin) from above, but I know that
    # I'm at the origin R[0,3] is 0
    b2 = int(
        abs((R[1, 3] - np.sqrt(R[1, 3]**2 - R[1, 1]*R[3, 3]))
            / R[3, 3]) / density)+1
    b3 = int(
        abs((R[2, 3] - np.sqrt(R[2, 3]**2 - R[2, 2]*R[3, 3]))
            / R[3, 3]) / density)+1

    make_gaussian_density(
        positions, outputfile, n1, n2, n3, b1, b2, b3, istart, istop, stepsize,
        sigma, cell, cellTI, indices_i_care, pbar_frequency, nstep, nat,
        nat_this_species
    )


if __name__ == '__main__':
    # Defining the command line arguments:
    from argparse import ArgumentParser
    from ase.io import read
    from samos.trajectory import Trajectory
    from samos.io.ase_io import read_positions_with_ase
    ap = ArgumentParser()

    ap.add_argument('cif', help='Cif file with structure')
    ap.add_argument('positions', help='a trajectory file to read')

    ap.add_argument('-n', '--n-sigma', type=int, default=3)
    ap.add_argument('-d', '--density', type=float, default=0.1,
                    help='nr of grid points per angstrom')
    # ap.add_argument('-r', '--recenter', action='store_true')  N
    ap.add_argument('--istart', help='starting point', type=int, default=1)
    ap.add_argument('--istop', help='ending point', type=int, default=None)
    ap.add_argument('-i', '--stepsize',
                    help='stepsize in trajectory', default=1)
    ap.add_argument('-s', '--sigma',
                    help='Value of sigma in ANGSTROM', type=float, default=0.3)

    ap.add_argument('-e', '--element',
                    help='Density of this atom-type', type=str, default='Li')
    ap.add_argument('-o', '--outputfile', help='outputfile', default='out.xsf')

    # Parsing the arguments:
    parsed_args = vars(ap.parse_args(sys.argv[1:]))

    t = Trajectory()
    t.set_atoms(read(parsed_args.pop('cif')))
    t.set_array(t._POSITIONS_KEY, read_positions_with_ase(
        parsed_args.pop('positions')), check_nat=False)
    get_gaussian_density(t, **parsed_args)
