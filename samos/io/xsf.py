#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

from samos.utils.constants import bohr_to_ang


def read_xsf(filename, fold_positions=False):
    finished = False
    skip_lines = 0
    reading_grid = False
    reading_dims = False
    reading_structure = False
    reading_nat = False
    reading_cell = False
    x, y, z, xdim, ydim, zdim = 0, 0, 0, 0, 0, 0
    rho_of_r, atoms, positions, cell = [], [], [], []
    with open(filename) as f:
        finished = False
        for line in f.readlines():
            if reading_grid:
                try:
                    for value in line.split():
                        if x != xdim-1 and y != ydim-1 and z != zdim-1:
                            rho_of_r[x, y, z] = float(value)
                        # Move on to next gridpoint
                        x += 1
                        if x == xdim:
                            x = 0
                            y += 1
                        if y == ydim:
                            z += 1
                            y = 0
                        if z == zdim-1:
                            finished = True
                            break

                except ValueError:
                    break
            elif skip_lines:
                skip_lines -= 1
            elif reading_structure:
                pos = list(map(float, line.split()[1:]))
                if len(pos) != 3:
                    reading_structure = False
                else:
                    atoms.append(line.split()[0])
                    positions.append(pos)
            elif reading_nat:
                nat, _ = list(map(int, line.split()))
                reading_nat = False
                reading_structure = True
            elif reading_cell:
                cell.append(list(map(float, line.split())))
                if len(cell) == 3:
                    reading_cell = False
                    reading_grid = True
            elif reading_dims:
                xdim, ydim, zdim = list(map(int, line.split()))
                rho_of_r = np.zeros([xdim-1, ydim-1, zdim-1])
                # ~ data2 = np.empty(xdim*ydim*zdim)
                # ~ iiii=0
                reading_dims = False
                reading_cell = True
                skip_lines = 1
            elif 'DATAGRID_3D_UNKNOWN' in line:
                x = 0
                y = 0
                z = 0
                cell = []
                reading_dims = True
            elif 'PRIMCOORD' in line:
                atoms = []
                positions = []
                reading_nat = True
            if finished:
                break

    try:
        volume_ang = np.dot(np.cross(cell[0], cell[1]), cell[2])
    except UnboundLocalError:
        raise Exception('No cell was read in XSF file, stopping')
    volume_au = volume_ang / bohr_to_ang**3

    if fold_positions:
        cell = np.array(cell)
        invcell = np.linalg.inv(cell.T)
        for idx, pos in enumerate(positions):
            # point in crystal coordinates
            points_in_crystal = invcell @ pos
            # point collapsed into unit cell
            points_in_unit_cell = points_in_crystal % 1.0
            positions[idx] = cell.T @ points_in_unit_cell

    return dict(
        data=rho_of_r, volume_ang=volume_ang, volume_au=volume_au,
        atoms=atoms, positions=positions, cell=cell
    )


def write_xsf(
        atoms, positions, cell, data=None,
        vals_per_line=6, outfilename=None,
        is_flattened=False, shape=None,):
    """
    Write a structure and, optionally, a 3D grid to an xsf file.

    :param atoms: The chemical symbols, one per position
    :param positions: The absolute positions in angstrom
    :param cell: The 3x3 cell in angstrom
    :param data:
        The grid, of shape given by shape if is_flattened. Pass None to
        write the header and grid dimensions only, leaving the datagrid
        block open for a writer that appends to the same file (that is
        what samos.lib.gaussian_density.make_gaussian_density does).
        shape is required in that case.
    :param int vals_per_line: Values written per line of the datagrid
    :param str outfilename: The file to write to, sys.stdout if None
    :param bool is_flattened: Whether data is a flat array
    :param shape: The (xdim, ydim, zdim) of the grid
    """
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise TypeError(
            'outfilename must be a path or None, got {}'.format(
                type(outfilename)))

    if is_flattened or data is None:
        try:
            xdim, ydim, zdim = shape
        except (TypeError, ValueError):
            raise ValueError(
                'shape is required when data is flattened or None, '
                'got {}'.format(shape))
    else:
        xdim, ydim, zdim = data.shape
        shape = data.shape
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
""".format(*[i+1 for i in shape]))
    for row in cell:
        f.write('    {}\n'.format('    '.join(
            ['{:.9f}'.format(r) for r in row])))
    if data is None:
        # A caller appends the grid and the END markers itself.
        if f is not sys.stdout:
            f.close()
        return
    col = 1
    if is_flattened:
        for val in data:
            f.write('  {:0.4E}'.format(val))
            if col < vals_per_line:
                col += 1
            else:
                f.write('\n')
                col = 1
    else:
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
    # Close only what this function opened.  An unconditional
    # close() here shut sys.stdout, which is what
    # outfilename=None -- the documented default -- selects.
    if f is not sys.stdout:
        f.close()


def write_grid(data, outfilename=None, vals_per_line=5,):
    xdim, ydim, zdim = data.shape
    if isinstance(outfilename, str):
        f = open(outfilename, 'w')
    elif outfilename is None:
        f = sys.stdout
    else:
        raise TypeError(
            'outfilename must be a path or None, got {}'.format(
                type(outfilename)))

    xdim, ydim, zdim = data.shape
    f.write('3         {}         {}         {}\n'.format(
        *[i+1 for i in data.shape]))
    col = 0
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                f.write('  {:0.4E}'.format(data[x, y, z]))
                if col < vals_per_line:
                    col += 1
                else:
                    f.write('\n')
                    col = 0
    if col:
        f.write('\n')
    # Close only what this function opened.  An unconditional
    # close() here shut sys.stdout, which is what
    # outfilename=None -- the documented default -- selects.
    if f is not sys.stdout:
        f.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(
        'Reads and writes an XSF file or a data file.\n'
        'python temp.xsf -o grid.xyz')
    p.add_argument('file', type=str)
    p.add_argument('--format', choices=['xsf', 'grid', 'none'],
                   default='grid',
                   help='whether to print the output in xsf or grid format')
    p.add_argument('-o', '--output',
                   help='The name of the output file, default to sys.out')
    p.add_argument(
        '--min', help='print minimum grid value and exit', action='store_true')
    p.add_argument(
        '--max', help='print maximum grid value and exit', action='store_true')

    pa = p.parse_args(sys.argv[1:])
    r = read_xsf(filename=pa.file)
    if pa.min:
        print(r['data'].min())
    elif pa.max:
        print(r['data'].max())
    elif pa.format == 'grid':
        write_grid(outfilename=pa.output, **r)
    elif pa.format == 'xsf':
        write_xsf(outfilename=pa.output, **r)
    elif pa.format == 'none':
        pass
    else:
        raise Exception('unknown format {}'.format(pa.format))
