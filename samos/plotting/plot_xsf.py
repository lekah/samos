#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mayavi import mlab
import sys, numpy as np, re
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers, covalent_radii, chemical_symbols
from ase import Atoms
from tvtk.api import tvtk

from ase.visualize.mlab import plot

from aiida_scripts.charges.io_xsf import read_xsf

covalent_radii[3] *= 0.66

EPSILON = 1e-8
bohr_to_ang = 0.52917720859

POS_REGEX_DECOMPOSED = re.compile("""
^                                                   # Linestart
[ \t]*                                              # Optional white space
(?P<sym>[A-Za-z]+[A-Za-z0-9]*)                      # get the symbol
(?P<vals>(\s+ ([\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)? ))+)
""", re.X | re.M)


POS_BLOCK_REGEX_DECOMPOSED = re.compile("""
([A-Za-z]+[A-Za-z0-9]*\s+([ \t]+ [\-|\+]?  ( \d*[\.]\d+  | \d+[\.]?\d* )  ([E | e][+|-]?\d+)?)+\s*)+
""", re.X | re.M)



def plot_charge(
        files, forces_list=None, fscale_factor=1.0, title=None, colormap='cool', color=None, invert_colors=False,
        do_isosurface=False, contours=[0.0001, 0.001], only_total=False, size=(1280, 720), azimuth=155, elevation=70,
        opacity=0.15, shift=None, savefig=None, atoms_of_interest=None, no_legend=False, log_rho=False,
        no_cell=False, repeat=(1,1,1), base_unit='bohr', legend_title=None):
    #~ x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    #~ s = np.sin(x*y*z)/(x*y*z)
    if len(files) > 2:
        raise Exception("Can't deal with more that 2 files")
    opacity_atoms = 0.9
    n1,n2,n3 = repeat
    nr_unit_cells=n1*n2*n3
    for idx, fname in enumerate(files):
        res = read_xsf(filename=fname, fold_positions=True)
        if not idx:

            rho = res.get('data')

            if log_rho:
                rho = np.log10(rho)
                rho = np.where(np.isnan(rho), -120, rho)
            positions = np.array(res.get('positions'))
            symbols = res.get('atoms')
            cell = res.get('cell')
            delta_tau = np.zeros((nr_unit_cells*positions.shape[0], positions.shape[1]))

        else:
            rho -= res.get('data')
            delta_tau = res.get('positions') - positions


    atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
    print(atoms.get_volume())

    atoms = atoms.repeat(repeat)
    print(len(atoms))


    figure=mlab.figure(1, bgcolor=(1, 1, 1), size=size)  # make a white figure
    #~ if title is not None:
        #~ mlab.title(title)
    # Plot the atoms as spheres:
    figure.scene.disable_render = True

    opacity_atoms *= 0.7
    opacity_atoms = 1

    if atoms_of_interest is None:
        atoms_of_interest = set(symbols)
    A = atoms.cell
    for pos, Z, dpos in zip(atoms.positions, atoms.numbers, delta_tau):
        if chemical_symbols[Z] in atoms_of_interest:

            atoms_color=tuple(jmol_colors[Z])
            mlab.points3d(*pos,
                          scale_factor=covalent_radii[Z],
                          resolution=20,
                          color=atoms_color,
                          opacity=opacity_atoms
                        )
            if abs(np.linalg.norm(dpos)) > 0.01:
                x,y,z = pos
                u,v,w = dpos
                cyls = mlab.quiver3d(
                            x,y,z, u,v,w,
                            color=atoms_color,
                            scale_factor=2  ,
                            mode='arrow',
                    )
                cyls.glyph.glyph_source.glyph_source.shaft_radius = 0.2
                cyls.glyph.glyph_source.glyph_source.tip_radius = 0.4

                # t = mlab.text3d(pos[0], pos[1], pos[2], chemical_symbols[Z], color=(0,0,0), scale=.5)

    if forces_list is not None:
        opacity=0.7
        line_width=200.
        if only_total:
            colors = iter(((0,0,0), (0,1,0), (0,0,1)))

        for forces in forces_list:
            if only_total:
                color = next(colors)
            for pos, vecs in zip(atoms.positions, forces):
                x,y,z = pos
                if not only_total:
                    colors = iter(( (0.1,0,0), (1,0,0), (0,1,0), (0,0,1), (1., 0.84,0)))
                for u,v,w in vecs:
                    if not only_total:
                        color =  next(colors)
                    cyls = mlab.quiver3d(
                            x,y,z, u,v,w,
                            color=color, line_width=line_width,
                            mode='arrow',
                            #~ mode='2darrow',
                            #~ mode='cylinder',
                            scale_factor=fscale_factor,
                            opacity=opacity
                        )
                    #~ print cyls.glyph.glyph_source.glyph_source.__dict__.keys()

                    cyls.glyph.glyph_source.glyph_source.shaft_radius = 0.2*min([np.sqrt(1./(u**2+v**2+w**2)), 4.5])
                    cyls.glyph.glyph_source.glyph_source.tip_radius = 0.4*min([np.sqrt(1./(u**2+v**2+w**2)), 4.5])
                    #~ cyls.glyph.glyph_source.glyph_source.shaft_radius = np.sqrt(1./(u**2+v**2+w**2))
                    #~ cyls.glyph.glyph_source.glyph_source.shaft_radius = 0.3
                    # total is the first 3 columns, so I break out of loop if I do not want to show the rest.
                    if only_total:
                        break


            opacity *= 0.6
            #~ line_width *= 0.5

    # Draw the unit cell:

    if not(no_cell):
        for i1, a in enumerate(A):
            i2 = (i1 + 1) % 3
            i3 = (i1 + 2) % 3
            for b in [np.zeros(3), A[i2]]:
                for c in [np.zeros(3), A[i3]]:
                    p1 = b + c
                    p2 = p1 + a
                    mlab.plot3d([p1[0], p2[0]],
                                [p1[1], p2[1]],
                                [p1[2], p2[2]],
                                tube_radius=0.1)

    if do_isosurface:
        # Here I am calculating the total charge as multiples of electrons
        # Based on trial and error, I find that if I integrate over each
        # element of rho with dV = volume in bohr of the grid point, I get the
        # correct number of electrons:
        total_charge = rho.sum() * atoms.get_volume() / rho.size
        if base_unit == 'angstrom':
            pass
        elif base_unit in ('bohr', 'atomic'):
            total_charge *= bohr_to_ang**(-3)
        else:
            raise NotImplemented
        print('The total charge sums to {:6.3f} electrons'.format(total_charge))
        mean_charge_value = rho.sum() / rho.size
        print('Mean charge value is {:18.16f}'.format(mean_charge_value))
        if shift is not None:
            print(shift)
            rho=rho-shift
        # Now I replicate
        rho = np.concatenate([rho]*n1, axis=0)
        rho = np.concatenate([rho]*n2, axis=1)
        rho = np.concatenate([rho]*n3, axis=2)

        src = mlab.pipeline.scalar_field(rho)
        if contours is None:
            mean=rho.mean()
            std = rho.std()
            #~ contours=[mean-std, mean, mean+std]
            contours=[mean-std, mean+std]
            print('Choosing contour lines myself:', contours)

        isos = mlab.pipeline.iso_surface(
                src, opacity=opacity, colormap=colormap,
                vmin=min(contours), vmax=max(contours),
                contours=contours, color=color )

        if invert_colors:
            lut = isos.module_manager.scalar_lut_manager.lut.table.to_array()
            ilut = lut[::-1]
            # putting LUT back in the surface object
            isos.module_manager.scalar_lut_manager.lut.table = ilut
        #~ isos.module_manager.scalar_lut_manager.show_scalar_bar = True
        polydata = isos.actor.mapper.input  #__dict__.keys()
        try:
            pts = np.array(polydata.points) - 1
        except TypeError:
            raise Exception(
                'I guess the contours you chose are not valid\n'
                'Valid range is {} to {}'.format(rho.min(), rho.max()))
        # Transform the points to the unit cell:

        polydata.points = np.dot(pts, A / np.array(rho.shape)[:, np.newaxis])
        if not no_legend:
            cb = mlab.colorbar(orientation='horizontal', nb_labels=len(contours))
            cb._label_text_property.color = (0,0,0)

            cb.scalar_bar_representation.position = [0.05, 0.00]
            cb.scalar_bar_representation.position2 = [0.9, 0.14]
        if legend_title:
#        legend_title = "{} {}-density".format('log' if log_rho else '', '-'.join(atoms_of_interest))
            width = len(legend_title)*size[0]*0.00003
            mlab.text(0.5*(1-width), 0.0025, legend_title, line_width=10, width=width, color=(0,0,0))
    figure.scene.disable_render = False
    if title is not None:
        mlab.title(title, color=(0,0,0), height=0.97, size=8e-4*size[1])
    #~ print mlab.title.__doc__
    mlab.view(azimuth=azimuth, elevation=elevation, distance='auto')
    # Show the 3d plot:


    if savefig:
        mlab.savefig(savefig)
    else:
        mlab.show()


def read_forces(files, atoms_of_interest, take_difference=False):
    #~ if len(files)==1:
        #~ print 'Forces detected'
    #~ elif len(files)==2:
        #~ print '2 forces files detected, taking difference'
    #~ else:
        #~ raise Exception("unsupported number of files ({}) for forces".format(len(files)))
    forces = []
    for fname in files:

        with open(fname) as f:

                forces.append(np.array(
                        [
                            list(map(float, pos_match.group('vals').split()))
                            for match in POS_BLOCK_REGEX_DECOMPOSED.finditer(f.read())
                            for pos_match in POS_REGEX_DECOMPOSED.finditer(match.group(0))
                            if pos_match.group('sym') in atoms_of_interest

                        ]
                    )
                )

    for f in forces:
        print(f.shape)
        #~ for lin in f:
            #~ print lin
        #~ raw_input()
    returnlist = []
    if take_difference:
        for f in forces[1:]:
            try:
                returnlist.append(forces[0] - f)
            except ValueError:
                shapediff = f.shape[1] - forces[0].shape[1]
                if shapediff > 0:
                    ff = np.concatenate((forces[0], np.zeros((forces[0].shape[0], shapediff))), axis=1)
                    return ff - forces[1]
                else:
                    raise Exception()
    else:
        returnlist = forces
    return returnlist


if __name__ == '__main__':
    def logarithmize_vector(vector):
        norm = np.linalg.norm(vector)
        if norm < EPSILON:
            return vector
        return np.log(norm+2)/norm * vector
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('files', type=str, nargs='+')
    p.add_argument('-v', '--verbosity', action='store_true')
    p.add_argument('-f', '--forces', type=str, nargs='+', help='some force files')
    p.add_argument( '--fscale',type=float, default=1.)
    p.add_argument('--fstretch',type=float, default=1.)
    p.add_argument('--flog',action='store_true', help='logarithmize the forces')
    p.add_argument('--no-legend',action='store_true', help='no legend')
    p.add_argument('--no-iso',action='store_true', help='Do not show the isosurface')
    p.add_argument('--no-cell',action='store_true', help='Do not show the cell')
    #~ p.add_argument('--diff', action='store_true', help='Show difference in forces')
    p.add_argument('-a', '--atoms-of-interest', type=str, nargs='+') #, default=['Li'])
    p.add_argument('-m', '--colormap', type=str,  default='spring')
    p.add_argument('--color', nargs=3, type=float, help='Colors as RGB tuple in range 0->1 (overrides colormap setting)')
    p.add_argument('--invert-colors', help='invert the colormap', action='store_true')
    p.add_argument('--log-rho', help='Convert the field to its logarithm', action='store_true')
    p.add_argument('-c', '--contours', type=float, nargs='+')
    p.add_argument('-s', '--size', type=int, nargs=2, default=(1280, 720))
    p.add_argument('-t', '--title', type=str,help='title')
    p.add_argument( '--legend-title', type=str,help='title')
    p.add_argument('-o', '--opacity', type=float,help='opacity of iso surface', default=0.15)
    p.add_argument('-d', '--diff', action='store_true',help='take difference (with respect to first force provided)')
    p.add_argument('-r', '--repeat',  type=int, nargs=3, default=(1,1,1))
    p.add_argument('--total', action='store_true',help='Show only total force')
    p.add_argument('--shift', type=float)
    p.add_argument('--elevation', type=float, default=70)
    p.add_argument('--azimuth', type=float, default=155)
    p.add_argument('--savefig', type=str, default=None)
    p.add_argument('--base-units', type=str,choices=('bohr', 'atomic', 'angstrom'), help='units on which the density is based', default='bohr')

    #~ p.add_argument('--histo', action='store_true')
    pa = p.parse_args(sys.argv[1:])

    if pa.forces is not None:
        forces_list = []
        for forces in read_forces(pa.forces, pa.atoms_of_interest, take_difference=pa.diff):
            forces = pa.fstretch*forces
            nat, idim = forces.shape

            forces = forces.reshape(nat, idim/3, 3)
            if pa.flog:
                forces = [
                        list(map(logarithmize_vector, vecs))
                        for vecs in forces
                    ]
            forces_list.append(forces)
    else:
        forces_list = None

    plot_charge(pa.files, forces_list=forces_list, fscale_factor=pa.fscale, colormap=pa.colormap, color=tuple(pa.color) if pa.color else None, invert_colors=pa.invert_colors,
            title=pa.title, do_isosurface=not(pa.no_iso), contours=pa.contours, only_total=pa.total,
            size=pa.size, opacity=pa.opacity, shift=pa.shift, savefig=pa.savefig, atoms_of_interest=pa.atoms_of_interest,
            no_legend=pa.no_legend, log_rho=pa.log_rho, no_cell=pa.no_cell, repeat=pa.repeat, base_unit=pa.base_units, legend_title=pa.legend_title,
            azimuth=pa.azimuth,elevation=pa.elevation
        )
    #~ plot_charge(read_charge(filename=pa.file))
