#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from samos.utils.terminal import get_terminal_width

bohr_to_ang = 0.52917720859


def write_xsf_header(
        atoms, positions, cell, data,
        vals_per_line=6, outfilename=None, **kwargs):
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise Exception('No file')
    if data is not None:
        xdim, ydim, zdim = data.shape
    else:
        xdim = kwargs.get('xdim')
        ydim = kwargs.get('ydim')
        zdim = kwargs.get('zdim')
    f.write(' CRYSTAL\n PRIMVEC\n')
    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(r) for r in row])))
    f.write('PRIMCOORD\n       {}        1\n'.format(len(atoms)))
    for atom, pos in zip(atoms, positions):
        f.write('{:<3}     {}\n'.format(
            atom, '   '.join(['{:.9f}'.format(v) for v in pos])))

    f.write("""BEGIN_BLOCK_DATAGRID_3D
3D_PWSCF
DATAGRID_3D_UNKNOWN
        {}         {}         {}
  0.000000  0.000000  0.000000
""".format(*[i+1 for i in (xdim, ydim, zdim)]))

    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(item) for item in row])))

    if data is not None:
        col = 1
        for z in range(zdim+1):
            for y in range(ydim+1):
                for x in range(xdim+1):
                    f.write('  {:0.4E}'.format(
                        data[x % xdim, y % ydim, z % zdim]))
                    if col < vals_per_line:
                        col += 1
                    else:
                        f.write('\n')
                        col = 1
        if col:
            f.write('\n')
        f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n')
    f.close()


def get_gaussian_density(trajectory, element=None, outputfile='out.xsf',
                         sigma=0.3, n_sigma=3.0, density=0.1,
                         istart=1, istop=None, stepsize=1,
                         indices_i_care=None, indices_exclude_from_plot=None):
    """
    :param str positionsf: Where to read the positions from.
    :param str pos_units:
        The units of the positions (implemented so far: angstrom, bohr).
    :param str outputfile: The xsf outputfile
    :param bool with_symbols:
        Whether symbols are printed in front
        of positions (will be ignored)
    :param list cell: the 3x3 cell,
    :param str element:
        The to calculate the density for,
        has to be present in symbols
    :param int nat:
        The number of atoms written in the positionsfile
        per atomic step
    :param float sigma: The gaussian broadening to apply
    :param float n_sigma:
        the multiple of sigma for which to
        create the bounding box.
    :param float density: The density for the grid
    :param int istart: Index to start reading positions
    :param int istop: Index to stop reading positions
    :param bool recenter: Whether to recenter
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
    write_xsf_header(
        [s for i, s in enumerate(symbols, start=1)
         if i not in indices_exclude_from_plot],
        [p for i, p in enumerate(starting_pos, start=1)
         if i not in indices_exclude_from_plot],
        cell, None, outfilename=outputfile, xdim=n1, ydim=n2, zdim=n3)

    S = np.matrix(np.diag([1, 1, 1, -(sigma*n_sigma/density)**2]))
    cellT = cell.T
    cellTI = np.matrix(cellT).I
    #  I describe the move from atomic to crystal
    # coordinates with an affine transformation M:
    M = np.matrix(np.r_[np.c_[np.matrix(cellTI), np.zeros(3)], [[0, 0, 0, 1]]])
    # Q is a check, but not used. Check is orthogonality
    # Q is the sphere transformed by transformation M
    # Q = M.I.T * S * M.I
    # Now, as defined in the source, I calculate R = Q^(-1)
    R = M * S.I * M.T
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
