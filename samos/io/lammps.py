import numpy as np
import os, sys
from ase.io import read
import re
from ase import Atoms
from ase.visualize import view

from samos.trajectory import Trajectory

integer_regex = re.compile('(?P<int>\d+)') # only positvie integers, actually
float_regex = re.compile('(?P<float>[\-]?\d+\.\d+(e[+\-]\d+)?)')


def get_indices(header_list, prefix="", postfix=""):
    try:
        idc = np.array([header_list.index(f'{prefix}{dim}{postfix}') for dim in 'xyz'])
    except Exception as e:
        return False, None
    return True, idc

def get_position_indices(header_list):
    # Unwrapped positions u, # scaled positions s, wrapped positions given as x y z
    for postfix in ("u", "s", ""):
        found, idc = get_indices(header_list, prefix='', postfix=postfix)
        if found:
            if postfix in ("s", ""):
                print("Warning: I am not unwrapping positions, this is not yet implementd")
            return postfix, idc
    if 'xsu' in header_list:
        raise NotImplementedError("Do not support scaled unwrapped coordinates")

    raise TypeError("No position indices found")



    #     return 'u', np.array([header_list.index(f'{dim}u') for dim in 'xyz'])
    # elif 'xs' in header_list:
    #     # scaled coordinates
    #     return 's',np.array([header_list.index(f'{dim}s') for dim in 'xyz'])
    # elif 'x' in header_list:
    #     return 'w', np.array([header_list.index(f'{dim}') for dim in 'xyz'])
    # elif 'xsu' in header_list:

def read_step_info(lines, lidx=0, start=False):
    assert len(lines) == 9
    assert lines[0].startswith("ITEM: TIMESTEP"), "Not a valid lammps dump file?"
    try:
        timestep = int(integer_regex.search(lines[1]).group('int'))
    except Exception as e:
        print(f"Timestep is not an integer or was not found in line {lidx+1} ({lines[1]})")
        raise e
    assert lines[2].startswith("ITEM: NUMBER OF ATOMS"), "Not a valid lammps dump file, expected NUMBER OF ATOMS"
    try:
        nat = int(integer_regex.search(lines[3]).group('int'))
    except Exception as e:
        print("Could not read number of atoms")
        raise e
    cell = np.zeros((3,3))
    if lines[4].startswith("ITEM: BOX BOUNDS pp pp pp")
        try:
            for idim in range(3):
                d1, d2 = [float(m.group('float')) for m in float_regex.finditer(lines[5+idim])]
                cell[idim, idim] = d2 - d1
        except Exception as e:
            print(f"Could not read cell dimension {idim}")
            raise e
    elif lines[4].startswith("BOX BOUNDS xy xz yz pp pp pp"):
        try:
            # see https://docs.lammps.org/dump.html
            # and https://docs.lammps.org/Howto_triclinic.html
            xlo, xhi, xy = [float(m.group('float')) for m in float_regex.finditer(lines[5])]
            ylo, yhi, xz = [float(m.group('float')) for m in float_regex.finditer(lines[6])]
            zlo, zhi, yz = [float(m.group('float')) for m in float_regex.finditer(lines[7])]
            cell[0, 0] = xhi - xlo
            cell[1, 1] = yhi - ylo
            cell[2, 2] = zhi - zlo
            cell[1, 0] = xy
            cell[2, 0] = xz
            cell[2, 1] = yz
        except Exception as e:
            print(f"Could not read cell dimension {idim}")
            raise e
    else:
        raise ValueError("unsupported lammps dump file, expected  BOX BOUNDS pp pp pp or BOX BOUNDS xy xz yz pp pp pp")
    if start:
        print(f"Read starting cell as:\n{cell}")
        assert lines[8].startswith("ITEM: ATOMS"), "Not a supported format, expected ITEM: ATOMS"
        header_list = lines[8].lstrip("ITEM: ATOMS").split()

        atomid_idx = header_list.index('id')
        print(f'Atom ID index: {atomid_idx}')
        try:
            element_idx = header_list.index('element')
            print("Element found at index {element_idx}")
        except ValueError as e:
            element_idx = None
        try:
            type_idx = header_list.index('type')
            print("type found at index {type_idx}")
        except ValueError as e:
            type_idx = None
        try:
            postype, posids = get_position_indices(header_list)
        except Exception as e:
            print("Abandoning because positions are not given")
            sys.exit(1)
        print("Positions are given as: {}".format({'u':"unwrapped", 's':"Scaled (wrapped)", "":"Wrapped"}[postype]))
        print("Position indices are: {}".format(posids))
        has_vel, velids = get_indices(header_list, 'v')
        if has_vel:
            print("Velocities found at indices: {}".format(velids))
        else:
            print("Velocities were not found")
        has_frc, frcids = get_indices(header_list, 'f')
        if has_frc:
            print("Forces found at indices: {}".format(frcids))
        else:
            print("Forces were not found")
        return nat, atomid_idx, element_idx, type_idx, postype, posids, has_vel, velids, has_frc, frcids
    else:
        return nat, timestep, cell


def pos_2_absolute(cell, pos, postype):
    """
    Transforming positions to absolute positions
    """
    if postype in ("u", "w", ""):
        return pos
    elif postype == 's':
        return pos.dot(cell)
    else:
        raise RuntimeError(f"Unknown postype '{postype}'")

def get_thermo_props(fname):
    with open(fname) as f:
        f.readline() # first line
        header = f.readline().lstrip('#').strip().split()
    # header = [h.lstrip('v_').lstrip('c_') for h in header]
    arr = np.loadtxt(fname, skiprows=2)
    ts_index = header.index('TimeStep')
    return header, arr, ts_index


def read_lammps_dump(filename, elements=None,
            elements_file=None, types=None,  timestep=None,
            thermo_file=None, thermo_pe=None, thermo_stress=None,
            #thermo_ke=None #thermo_te=None,
            save_extxyz=False, outfile=None,
            ignore_forces=False, ignore_velocities=False,
            skip=0, f_conv=1.0, e_conv=1.0, s_conv=1.0
        ):
    """
    Read a filedump from lammps. It expects atomid to be printed, and positions to be given in scaled or unwrapped coordinates
    """
    # opening a first time to check file and get indices of positions, velocities etc
    with open(filename) as f:
        # first doing a check
        lines = [next(f) for _ in range(9)]
        (nat_must,atomid_idx, element_idx, type_idx,
         postype, posids, has_vel, velids, has_frc, frcids) = read_step_info(lines, lidx=0, start=True)

        if ignore_forces:
            has_frc = False
        if ignore_velocities:
            has_vel = False
        body = np.array([f.readline().split() for _ in range(nat_must)]) # these are read as strings
        atomids = np.array(body[:, atomid_idx], dtype=int)
        sorting_key = atomids.argsort()
        if type_idx is not None and types is not None:
            types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int)
            types_in_body -= 1 # 1-based to 0-based indexing
            symbols = np.array(types, dtype=str)[types_in_body]
        elif element_idx is not None:
            # readingin elements frmo body
            symbols = np.array(body[:,element_idx])[sorting_key]
            # print(elements)
            # print(len(elements))
        elif elements is not None:
            assert len(elements) == nat_must
            symbols = elements[:]
            # raise ValueError("elements have to be given in LAMMPS or provided to function")
        elif elements_file is not None:
            with open(elements_file) as f:
                for line in f:
                    if line:
                        break
                elements = line.strip().split()
                if len(elements) != nat_must:
                    raise ValueError("length of list of elements ({}) is not equal number of atoms ({})".format(
                                len(elements), nat_must)
                    )
                symbols = elements[:]
        else:
            symbols = ['H']*nat_must




    positions = []
    timesteps = []
    cells = []
    if has_vel:
        velocities = []
    if has_frc:
        forces = []

    lidx = 0
    iframe = 0
    with open(filename) as f:
        while True:
            step_info = [f.readline() for _ in range(9)]
            if ''.join(step_info) == '':
                print(f"End reached at line {lidx}, stopping")
                break
            nat, timestep, cell = read_step_info(step_info, lidx=lidx, start=False)
            lidx += 9
            if nat != nat_must:
                print("Changing number of atoms is not supported, breaking")
                break


            body = np.array([f.readline().split() for _ in range(nat_must)]) # these are read as strings
            lidx += nat_must
            atomids = np.array(body[:, atomid_idx], dtype=int)
            sorting_key = atomids.argsort()
            pos = np.array(body[:, posids], dtype=float)[sorting_key]
            if iframe >= skip:
                positions.append(pos_2_absolute(cell, pos, postype))
                timesteps.append(timestep)
                cells.append(cell)
                if has_vel:
                    velocities.append(np.array(body[:, velids], dtype=float)[sorting_key])
                if has_frc:
                    forces.append(f_conv*np.array(body[:, frcids], dtype=float)[sorting_key])
            iframe += 1

    print(f"Read trajectory of length {iframe}\nCreating Trajectory of length {len(timesteps)}")


    atoms = Atoms(symbols, positions[0], cell=cells[0], pbc=True)
    traj = Trajectory(atoms=atoms,
            positions=positions, cells=cells )
    if has_vel:
        traj.set_velocities(velocities)
    if has_frc:
        traj.set_forces(forces)
    if timestep:
        traj.set_timestep(timestep)

    if thermo_file:
        header, arr, ts_index = get_thermo_props(thermo_file)
        timesteps_thermo = np.array(arr[:, ts_index], dtype=int).tolist()
        indices = []
        for ts in timesteps:
            try:
                indices.append(timesteps_thermo.index(ts))
            except ValueError:
                raise ValueError(f"Index {ts} is not in thermo file")
        indices = np.array(indices, dtype=int)
        # if thermo_te:
        #     colidx = header.index(thermo_te)
        #     traj.set_total_energies(arr[indices, colidx])
        if thermo_pe:
            colidx = header.index(thermo_pe)
            traj.set_pot_energies(e_conv*arr[indices, colidx])
        if thermo_stress:
            stressall = []
            keys = ('xx', 'yy', 'zz', 'yz', 'xz', 'xy') # voigt notation for stress
            # first diagonal terms:
            for key in keys:
                fullkey = thermo_stress + key
                colidx = header.index(fullkey)
                stressall.append(s_conv*arr[indices, colidx])
            traj.set_stress(np.array(stressall).T)
        # if thermo_ke:
        #     colidx = header.index(thermo_ke)
        #     traj.set_kinetic_energies(arr[indices, colidx])
    if save_extxyz:
        path_to_save = outfile or filename + '.extxyz'
        from ase.io import write
        asetraj = traj.get_ase_trajectory()
        write(path_to_save, asetraj, format='extxyz')
    elif outfile:
        path_to_save = outfile or filename +'.traj'
        traj.save(path_to_save)
    return traj


if __name__== '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("""Reads lammps input and returns/saves a Trajectory instance""")
    parser.add_argument('filename', help='The filename/path of the lammps trajectory')
    parser.add_argument('-o', '--outfile', help='The filename/path to save trajectory at')
    parser.add_argument('-t', '--types', nargs='+', help='list of types, will be matched with types given in lammps')
    parser.add_argument('-e', '--elements', nargs='+', help='list of elements')
    parser.add_argument('--elements-file',
            help='A file containing the elements as space-separated strings')
    parser.add_argument('--timestep', type=float, help='The timestep of the trajectory printed')
    parser.add_argument('--f-conv', type=float, help='The conversion factor for forces', default=1.0)
    parser.add_argument('--e-conv', type=float, help='The conversion factor for energies', default=1.0)
    parser.add_argument('--s-conv', type=float, help='The conversion factor for stresses', default=1.0)

    parser.add_argument('--thermo-file', help='File path to equivalent thermo-file')
    parser.add_argument('--thermo-pe', help='Thermo keyword for potential energy',)
    parser.add_argument('--thermo-stress', help='Thermo keyword for stress without the xx/yy/xz..',)
    # parser.add_argument('--thermo-te', help='Thermo keyword for total energy',)
    # parser.add_argument('--thermo-ke', help='Thermo keyword for kinetic energy',)
    parser.add_argument('--save-extxyz', action='store_true', help='save extxyz instead of traj')
    parser.add_argument('--ignore-velocities', action='store_true',
                help='Ignore velocities in dump file')
    parser.add_argument('--ignore-forces', action='store_true',
                help='Ignore forces in dump file')
    parser.add_argument('--skip', type=int, default=0,
                help='Skip this many first steps')
    args = parser.parse_args()
    read_lammps_dump(**vars(args))