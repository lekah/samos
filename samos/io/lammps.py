# -*- coding: utf-8 -*-
import numpy as np

import re
from warnings import warn
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from samos.structurelist import StructureList
from samos.trajectory import Trajectory
from samos.utils.units import UNIT_SYSTEMS as _LAMMPS_UNITS

#  only matches positive integers
integer_regex = re.compile(r'(?P<int>\d+)')  # noqa: W605
# Accepts 0, 0.0, .5, 1e3, -1.0E+01, 2.5d0 -- LAMMPS writes box
# bounds in several of these forms depending on how it was built.
float_regex = re.compile(
    r'(?P<float>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][+-]?\d+)?)')

# What to do with a dump whose frames do not all hold the same atoms.
# 'error' is the default because the alternative this reader shipped
# with was to stop at the first such frame and return the frames before
# it, which loses data without saying so.
CHANGING_ATOMS_POLICIES = ('error', 'truncate', 'structures')


class ChangingAtomsInDump(ValueError):
    """
    A dump file's frames do not all hold the same atoms.

    A :class:`ValueError`, so that a caller who only needs to know the
    file is not a trajectory needs no import of its own, while
    :mod:`samos.cli` can catch this one case to say what reads such a
    file instead.
    """


def get_indices(header_list, prefix='', postfix=''):
    try:
        idc = np.array([header_list.index(f'{prefix}{dim}{postfix}')
                        for dim in 'xyz'])
    except Exception:
        return False, None
    return True, idc


def get_position_indices(header_list, quiet=False):
    # Unwrapped positions u, # scaled positions s
    # wrapped positions given as x y z
    for postfix in ('u', 's', ''):
        found, idc = get_indices(header_list, prefix='',
                                 postfix=postfix)
        if found:
            if postfix in ('s', '') and not quiet:
                print('Warning: I am not unwrapping positions,'
                      ' this is not yet implemented')
            return postfix, idc
    if 'xsu' in header_list:
        raise NotImplementedError('Do not support scaled'
                                  ' unwrapped coordinates')

    raise TypeError(
        'No position columns found in the dump header ({}). Expected '
        'x/y/z (wrapped), xs/ys/zs (scaled) or xu/yu/zu '
        '(unwrapped).'.format(header_list))


def read_step_info(lines, lidx=0, start=False, additional_kw=[], quiet=False):
    assert len(lines) == 9
    if not lines[0].startswith('ITEM: TIMESTEP'):
        raise Exception("Did not start with 'ITEM: TIMESTEP'\n"
                        'Is this not a valid lammps dump file?')
    try:
        timestep = int(integer_regex.search(lines[1]).group('int'))
    except Exception as e:
        if not quiet:
            print('Timestep is not an integer or was not found in '
                  f'line {lidx+1} ({lines[1]})')
        raise e
    if not lines[2].startswith('ITEM: NUMBER OF ATOMS'):
        raise Exception('Not a valid lammps dump file, '
                        'expected NUMBER OF ATOMS')
    try:
        nat = int(integer_regex.search(lines[3]).group('int'))
    except Exception as e:
        if not quiet:
            print('Could not read number of atoms')
        raise e
    # Only the box lengths are kept, the origin (xlo/ylo/zlo) is
    # dropped. Everything downstream works on displacements, where the
    # origin cancels; scaled coordinates are converted with pos.dot(cell)
    # and so end up relative to the origin as well.
    cell = np.zeros((3, 3))
    if lines[4].startswith('ITEM: BOX BOUNDS pp pp pp'):
        try:
            for idim in range(3):
                d1, d2 = [float(m.group('float'))
                          for m in float_regex.finditer(lines[5+idim])]
                cell[idim, idim] = d2 - d1
        except Exception as e:
            if not quiet:
                print(f'Could not read cell dimension {idim}')
            raise e
    elif lines[4].startswith('ITEM: BOX BOUNDS xy xz yz pp pp pp'):
        try:
            # see https://docs.lammps.org/dump.html
            # and https://docs.lammps.org/Howto_triclinic.html
            xlo, xhi, xy = [float(m.group('float'))
                            for m in float_regex.finditer(lines[5])]
            ylo, yhi, xz = [float(m.group('float'))
                            for m in float_regex.finditer(lines[6])]
            zlo, zhi, yz = [float(m.group('float'))
                            for m in float_regex.finditer(lines[7])]
            cell[0, 0] = xhi - xlo
            cell[1, 1] = yhi - ylo
            cell[2, 2] = zhi - zlo
            cell[1, 0] = xy
            cell[2, 0] = xz
            cell[2, 1] = yz
        except Exception as e:
            if not quiet:
                print('Could not read triclinic cell bounds from:\n{}'
                      .format(''.join(lines[5:8])))
            raise e
    else:
        raise ValueError('unsupported lammps dump file, '
                         'expected  BOX BOUNDS pp pp pp or '
                         'BOX BOUNDS xy xz yz pp pp pp')
    if start:
        if not quiet:
            print(f'Read starting cell as:\n{cell}')
        if not lines[8].startswith('ITEM: ATOMS'):
            raise Exception('Not a supported format, expected ITEM: ATOMS')
        header_list = lines[8].removeprefix('ITEM: ATOMS').split()

        atomid_idx = header_list.index('id')
        if not quiet:
            print(f'Atom ID index: {atomid_idx}')
        try:
            element_idx = header_list.index('element')
            if not quiet:
                print(f'Element found at index {element_idx}')
        except ValueError:
            element_idx = None
        try:
            type_idx = header_list.index('type')
            if not quiet:
                print(f'type found at index {type_idx}')
        except ValueError:
            type_idx = None
        # Let the error propagate: this used to call sys.exit(1), which
        # kills the interpreter of any program importing this module.
        postype, posids = get_position_indices(header_list, quiet=quiet)
        if not quiet:
            print('Positions are given as: {}'.format(
                {'u': 'unwrapped', 's': 'Scaled (wrapped)',
                 '': 'Wrapped'}[postype]))
        if not quiet:
            print('Position indices are: {}'.format(posids))
        has_vel, velids = get_indices(header_list, 'v')
        if has_vel:
            if not quiet:
                print('Velocities found at indices: {}'.format(velids))
        else:
            if not quiet:
                print('Velocities were not found')
        has_frc, frcids = get_indices(header_list, 'f')
        if has_frc:
            if not quiet:
                print('Forces found at indices: {}'.format(frcids))
        else:
            if not quiet:
                print('Forces were not found')
        additional_ids = []
        if additional_kw:
            for kw in additional_kw:
                if kw not in header_list:
                    raise ValueError(f'Keyword {kw} not found in '
                                     f'header ({header_list})')
                additional_ids.append(header_list.index(kw))

        return (nat, atomid_idx, element_idx, type_idx, postype,
                posids, has_vel, velids, has_frc, frcids, additional_ids)
    else:
        return nat, timestep, cell


def read_body(f, nat, lidx=0):
    """
    Read *nat* atom lines from *f* and return them as a string array.

    :param f: The open dump file, positioned at the first atom line
    :param int nat: The number of atom lines to read
    :param int lidx: Lines consumed so far, used for the error message
    :returns: A (nat, ncolumns) array of strings
    :raises ValueError:
        If the file ends early or a line has a different number of
        columns than the first. numpy would otherwise report this as an
        inhomogeneous-shape error naming neither the frame nor the line.
    """
    rows = []
    ncol = None
    for iat in range(nat):
        line = f.readline()
        if not line:
            raise ValueError(
                'File ended after {} of {} atom lines (line {})'.format(
                    iat, nat, lidx + iat + 1))
        row = line.split()
        if ncol is None:
            ncol = len(row)
        elif len(row) != ncol:
            raise ValueError(
                'Line {} has {} columns, the frame started with {}'.format(
                    lidx + iat + 1, len(row), ncol))
        rows.append(row)
    return np.array(rows)


def pos_2_absolute(cell, pos, postype):
    """
    Transforming positions to absolute positions
    """
    if postype in ('u', 'w', ''):
        return pos
    elif postype == 's':
        return pos.dot(cell)
    else:
        raise RuntimeError(f"Unknown postype '{postype}'")


def get_thermo_props(fname):
    with open(fname) as f:
        f.readline()  # first line
        header = f.readline().lstrip('#').strip().split()
    # header = [h.lstrip('v_').lstrip('c_') for h in header]
    arr = np.loadtxt(fname, skiprows=2)
    if len(arr.shape) == 1:
        # special case if only one line is present
        arr = np.array([arr])
    ts_index = header.index('TimeStep')
    return header, arr, ts_index


def _symbols_from_type_column(type_idx, type_map, quiet, also_print=None):
    """
    Name the atoms of a frame by its integer type column.

    :param int type_idx: Column holding the LAMMPS type.
    :param type_map: Chemical symbol per type, in type order.
    :param str also_print: A further line to show with the type report.
    :returns: ``symbols_for(body, sorting_key)``.
    """
    announced = []

    def symbols_for(body, sorting_key):
        types_in_body = np.array(body[:, type_idx][sorting_key], dtype=int)
        if not quiet and not announced:
            # Said once, not once per frame: the dump command fixes the
            # columns for the whole file, so every frame would repeat it.
            print('types in body: {}'.format(', '.join(
                sorted(map(str, set(types_in_body))))))
            if also_print is not None:
                print(also_print)
            announced.append(True)
        if types_in_body.min() < 1 or types_in_body.max() > len(type_map):
            raise ValueError(
                'The dump holds types {} to {}, but {} symbol(s) were '
                'given to map them to'.format(
                    types_in_body.min(), types_in_body.max(),
                    len(type_map)))
        return type_map[types_in_body - 1]  # 1-based to 0-based indexing

    return symbols_for


def _symbols_fixed(symbols, nat, what):
    """
    Name the atoms of a frame from a list the caller supplied.

    The list is positional, one entry per slot in atom-id order, so it
    only means anything while the atom count stays put.  A frame of a
    different size has no slot for each entry, which is raised rather
    than guessed at.

    :param symbols: One chemical symbol per atom.
    :param int nat: The atom count the list was checked against.
    :param str what: What the list came from, for the error messages.
    :returns: ``symbols_for(body, sorting_key)``.
    """
    if len(symbols) != nat:
        raise ValueError(
            'length of {} ({}) is not equal number of atoms '
            '({})'.format(what, len(symbols), nat))

    def symbols_for(body, sorting_key):
        if len(body) != nat:
            raise ChangingAtomsInDump(
                'A frame holds {} atoms, but {} names {} of them. An '
                'explicit list of elements is positional, so it can '
                'only be used on a dump whose atom count does not '
                'change.'.format(len(body), what, nat))
        return symbols

    return symbols_for


def _symbols_from_masses(mass_types, quiet):
    """
    The chemical symbol of every LAMMPS type, by closest atomic mass.

    :param str mass_types: A LAMMPS input file holding a Masses block.
    :returns: One chemical symbol per type, in type order.
    :raises ValueError: If the types in the block are not consecutive.
    """
    with open(mass_types) as fmass:
        masses = []
        reading_masses = False
        for line in fmass:
            line = line.strip()
            if not line:
                continue
            if reading_masses:
                try:
                    typ, mass = line.split()[:2]
                    masses.append((int(typ), float(mass)))
                except ValueError:
                    break
            elif line.startswith('Masses'):
                reading_masses = True
            else:
                pass
    type_indices, masses = zip(*masses)
    masses = np.array(masses, dtype=float)
    type_indices = np.array(type_indices, dtype=int)
    # small check whether all types are present
    if not np.all(np.arange(1, type_indices.max()+1) == type_indices):
        raise ValueError('Types are not consecutive')
    symbols = []
    for mass in masses:
        # finding the closest element based on its mass
        idx = np.argmin(np.abs(atomic_masses - mass))
        symbols.append(chemical_symbols[idx])
    return symbols


def _make_symbol_source(nat, types=None, element_idx=None, type_idx=None,
                        elements=None, elements_file=None, mass_types=None,
                        quiet=False):
    """
    Build the function that names the atoms of one frame.

    Called once, from the header pass; the function it returns is
    called per frame.  That split is what lets a dump whose species or
    atom count change be read at all: the symbols of a frame come from
    that frame's own body, not from the first frame's.

    The branches are tried in the order the keywords are documented in
    :func:`read_lammps_dump`, the first one given winning.

    :param int nat: The atom count of the first frame, which the
        positional branches are checked against.
    :returns: ``symbols_for(body, sorting_key)``, giving one chemical
        symbol per atom of that frame, in atom-id order.
    :raises ValueError: If a branch needs a column the dump lacks.
    """
    if types is not None:
        if type_idx is None:
            raise ValueError('types specified but not found in file')
        return _symbols_from_type_column(
            type_idx, np.array(types, dtype=str), quiet)
    if element_idx is not None:
        def symbols_for(body, sorting_key):
            return np.array(body[:, element_idx])[sorting_key]
        return symbols_for
    if elements is not None:
        return _symbols_fixed(np.array(elements, dtype=str), nat,
                              'list of elements')
    if elements_file is not None:
        with open(elements_file) as f:
            from_file = f.readline().strip().split()
        return _symbols_fixed(np.array(from_file, dtype=str), nat,
                              'list of elements')
    if mass_types is not None:
        if type_idx is None:
            raise ValueError('types specified but not found in file')
        symbols = _symbols_from_masses(mass_types, quiet)
        return _symbols_from_type_column(
            type_idx, np.array(symbols, dtype=str), quiet,
            also_print='symbols: {}'.format(', '.join(symbols)))

    # last resort, setting everything to Hydrogen
    def symbols_for(body, sorting_key):
        return np.full(len(body), 'H', dtype=str)
    return symbols_for


def _describe_count_change(nat, nat_first):
    """A clause naming a frame's atom count against the first frame's."""
    return 'it holds {} atoms, the first frame held {}'.format(
        nat, nat_first)


def _describe_species_change(symbols, symbols_first):
    """
    Why a frame's atoms are not the species the first frame's were.

    :param symbols: Chemical symbols of this frame, in atom-id order.
    :param symbols_first: The same for the first frame.
    :returns: A clause naming the difference, or None if there is none.
    """
    changed = np.flatnonzero(np.asarray(symbols) !=
                             np.asarray(symbols_first))
    if len(changed):
        return ('the species of {} atom(s) changed, the first at '
                'position {} in atom-id order ({} -> {})'.format(
                    len(changed), changed[0] + 1,
                    symbols_first[changed[0]], symbols[changed[0]]))
    return None


def _build_structure_list(filename, positions, types, cells, dropped,
                          l_conv=1.0, thermo_file=None, outfile=None,
                          save_extxyz=False, quiet=False):
    """
    Assemble the frames of a dump that is not a trajectory.

    :param str filename: The dump that was read, used to name the
        output file when one is written and *outfile* is not given.
    :param dropped: Names of the arrays read from the dump that a
        structure list has nowhere to put, warned about so that they
        are not silently missing from the result.
    :returns: A :class:`~samos.structurelist.StructureList`.
    :raises ValueError: If *thermo_file* was given, whose per-frame
        columns a structure list does not store.
    """
    if thermo_file is not None:
        raise ValueError(
            'A thermo file cannot be read for a dump whose frames hold '
            'different atoms: the frames are returned as a '
            'StructureList, which stores no energies or stress.')
    if dropped:
        warn('The frames of this dump do not all hold the same atoms, '
             'so they are returned as a StructureList, which stores '
             'positions and species only. Dropped: {}.'.format(
                 ', '.join(dropped)))
    structures = StructureList.from_frames(positions, types, cells,
                                           pbc=True)
    structures.apply_unit_conversion(l_conv=l_conv)
    if not quiet:
        print('Creating StructureList of length {}'.format(
            structures.nstep))
    if save_extxyz or (outfile is not None and outfile.endswith('extxyz')):
        from ase.io import write
        write(outfile or filename + '.extxyz', list(structures),
              format='extxyz')
    elif outfile:
        structures.save(outfile)
    return structures


def read_lammps_dump(filename, elements=None,
                     elements_file=None, types=None, timestep=None,
                     mass_types=None,
                     thermo_file=None, thermo_pe=None, thermo_stress=None,
                     thermo_keywords=[], save_extxyz=False, outfile=None,
                     ignore_forces=False, ignore_velocities=False,
                     skip=0, l_conv=1.0, f_conv=1.0, e_conv=1.0,
                     s_conv=1.0, v_conv=1.0,
                     additional_keywords_dump=[], quiet=False,
                     istep=1, nsteps=None, units=None,
                     on_changing_atoms='error'):
    """
    Read a filedump from lammps.
    It expects atomid to be printed, and positions
    to be given in scaled or unwrapped coordinates
    :param filename: lammps dump file to read
    :param elements: list of elements to use
    :param elements_file:
        file containing elements (separated by space),
        instead of elements
    :param types:
        list of types if elements are not specified
        and type is a column
    :param timestep: timestep of dump (in fs)
    :param thermo_file: file containing thermo output in case required
    :param thermo_pe:
        potential energy column in thermo file (as given in header)
    :param thermo_stress:
        stress column in thermo file (as given in header,
        will do the _xx/_yy etc)
    :param thermo_keywords:
        The columns to read that are not pe or stress from the thermo dump
    :param save_extxyz:
        save to extxyz file
        (or if outfile is given with .extxyz)
    :param outfile: output file name, will write trajectory by default
    :param ignore_forces: ignore forces even if written in dump
    :param ignore_velocities: ignore velocities even if written in dump
    :param skip: skip first n steps
    :param l_conv: length conversion factor to Angstrom (positions, cells)
    :param f_conv: force conversion factor to eV/Angstrom
    :param e_conv: energy conversion factor to eV
    :param s_conv: stress conversion factor to eV/Angstrom^3
    :param v_conv:
        Velocity conversion factor to Angstrom/femtosecond.
        Ignored when *units* is set.
    :param units:
        LAMMPS unit system name.  When given, sets *l_conv*, *v_conv*,
        *f_conv*, *e_conv*, and *s_conv* automatically.  Cannot be
        combined with any of those individual conversion factors.
        Supported values: ``'real'``, ``'metal'``, ``'si'``, ``'cgs'``,
        ``'electron'``, ``'micro'``, ``'nano'``.
    :param additional_keywords_dump:
        additional keywords to be added read form dump and
        to be added as array. The column name is used both
        as key but also as arrayname
    :param nsteps:
        Stop after this many frames have been stored (i.e. after
        applying *skip* and *istep*).  If None, read until end of file.
    :param on_changing_atoms:
        What to do with a frame that does not hold the atoms the first
        frame held, either because the atom count changed or because an
        atom changed species.

        *   ``'error'`` (the default) raises
            :class:`ChangingAtomsInDump`.
        *   ``'truncate'`` returns the frames before that one as a
            :class:`~samos.trajectory.Trajectory`, warning that it did.
        *   ``'structures'`` returns every frame as a
            :class:`~samos.structurelist.StructureList`, which holds
            positions, species and cells but no velocities, forces or
            further columns.

        A dump whose frames do all hold the same atoms is read as a
        trajectory whichever of the three is given.
    :raises ChangingAtomsInDump:
        If the frames differ and *on_changing_atoms* is ``'error'``.
    """
    if on_changing_atoms not in CHANGING_ATOMS_POLICIES:
        raise ValueError(
            "Unknown policy '{}' for on_changing_atoms. Supported: "
            '{}.'.format(on_changing_atoms,
                         ', '.join(CHANGING_ATOMS_POLICIES)))
    if units is not None:
        if units not in _LAMMPS_UNITS:
            raise ValueError(
                "Unknown unit system '{}'. Supported: {}.".format(
                    units, ', '.join(sorted(_LAMMPS_UNITS))))
        if any(c != 1.0 for c in (l_conv, v_conv, f_conv, e_conv, s_conv)):
            raise ValueError(
                "Cannot combine 'units' with individual conversion "
                "factors (l_conv, v_conv, f_conv, e_conv, s_conv).")
        _u = _LAMMPS_UNITS[units]
        l_conv = _u['l_conv']
        v_conv = _u['v_conv']
        f_conv = _u['f_conv']
        e_conv = _u['e_conv']
        s_conv = _u['s_conv']

    # opening a first time to check file and get
    # indices of positions, velocities etc...
    with open(filename) as f:
        # first doing a check
        lines = [next(f) for _ in range(9)]
        (nat_must, atomid_idx, element_idx, type_idx,
         postype, posids, has_vel, velids,
         has_frc, frcids, additional_ids_dump) = read_step_info(
            lines, lidx=0, start=True,
            additional_kw=additional_keywords_dump, quiet=quiet)

        if ignore_forces:
            has_frc = False
        if ignore_velocities:
            has_vel = False
        # The body of the first frame is not read here any more: it
        # served only to name the atoms, and that now happens frame by
        # frame in the loop below, from each frame's own body.
        symbols_for = _make_symbol_source(
            nat_must, types=types, element_idx=element_idx,
            type_idx=type_idx, elements=elements,
            elements_file=elements_file, mass_types=mass_types,
            quiet=quiet)

    positions = []
    timesteps = []
    cells = []
    if has_vel:
        velocities = []
    if has_frc:
        forces = []

    lidx = 0
    iframe = 0
    symbols_first = None
    ragged = False
    # Filled from the first frame that differs onwards.  Every frame
    # kept before it held the atoms of the first frame, and that gap is
    # filled in below, so a dump that never changes pays nothing for
    # the possibility that it might.
    symbols_kept = []
    # dealing with additional keywords
    additional_arrays = {kw: [] for kw in additional_keywords_dump}

    with open(filename) as f:
        while True:
            step_info = [f.readline() for _ in range(9)]
            if ''.join(step_info) == '':
                if not quiet:
                    print(f'End reached at line {lidx}, stopping')
                break
            nat, timestep_, cell = read_step_info(
                step_info, lidx=lidx, start=False, quiet=quiet)
            lidx += 9

            # these are read as strings
            body = read_body(f, nat, lidx=lidx)
            lidx += nat
            atomids = np.array(body[:, atomid_idx], dtype=int)
            sorting_key = atomids.argsort()
            symbols = None
            if symbols_first is None:
                symbols = symbols_for(body, sorting_key)
                symbols_first = symbols
                mismatch = None
            elif nat != len(symbols_first):
                mismatch = _describe_count_change(nat, len(symbols_first))
                # The atoms of this frame are left unnamed until the
                # policy has had its say: a positional list of elements
                # cannot describe a frame of another size, and under
                # 'error' and 'truncate' this frame is not kept, so it
                # is never asked to.
            else:
                symbols = symbols_for(body, sorting_key)
                mismatch = _describe_species_change(symbols, symbols_first)
            if mismatch is not None:
                if on_changing_atoms == 'error':
                    raise ChangingAtomsInDump(
                        'Frame {} of {} does not hold the atoms the '
                        'first frame held: {}. Pass '
                        "on_changing_atoms='structures' to read the "
                        'frames as a StructureList, or '
                        "'truncate' to stop at this frame.".format(
                            iframe, filename, mismatch))
                elif on_changing_atoms == 'truncate':
                    # Warned rather than printed: dropping the rest of
                    # the file is worth saying even when the caller
                    # asked for a quiet read.
                    warn('Frame {} of {} does not hold the atoms the '
                         'first frame held: {}. Stopping there, so the '
                         'rest of the file is not read and the '
                         'trajectory returned holds {} frame(s).'.format(
                             iframe, filename, mismatch, len(positions)))
                    break
                else:
                    ragged = True
                    if symbols is None:
                        symbols = symbols_for(body, sorting_key)

            pos = np.array(body[:, posids], dtype=float)[sorting_key]
            if iframe >= skip and iframe % istep == 0:
                positions.append(pos_2_absolute(cell, pos, postype))
                timesteps.append(timestep_)
                cells.append(cell)
                if ragged:
                    symbols_kept.append(symbols)
                if has_vel:
                    velocities.append(
                        np.array(body[:, velids],
                                 dtype=float)[sorting_key])
                if has_frc:
                    forces.append(
                        np.array(body[:, frcids],
                                 dtype=float)[sorting_key])
                for kw, idx_add in zip(additional_keywords_dump,
                                       additional_ids_dump):
                    additional_arrays[kw].append(
                        np.array(body[:, idx_add],
                                 dtype=float)[sorting_key])
                if nsteps is not None and len(positions) >= nsteps:
                    if not quiet:
                        print(f'Reached nsteps={nsteps}, stopping')
                    break
            iframe += 1
    if not quiet:
        print(f'Read trajectory of length {iframe}')
    if ragged:
        dropped = []
        if has_vel:
            dropped.append('velocities')
        if has_frc:
            dropped.append('forces')
        dropped.extend(additional_keywords_dump)
        return _build_structure_list(
            filename, positions,
            [symbols_first] * (len(positions) - len(symbols_kept))
            + symbols_kept,
            cells, dropped, l_conv=l_conv, thermo_file=thermo_file,
            outfile=outfile, save_extxyz=save_extxyz, quiet=quiet)
    if not quiet:
        print(f'Creating Trajectory of length {len(timesteps)}')
    symbols = symbols_first
    try:
        atoms = Atoms(symbols, positions[0], cell=cells[0], pbc=True)
        traj = Trajectory(atoms=atoms,
                          positions=positions, cells=cells)
    except KeyError:
        traj = Trajectory(types=symbols, cells=cells)
        traj.set_positions(positions)
    if has_vel:
        traj.set_velocities(velocities)
    if has_frc:
        traj.set_forces(forces)
    if timestep:
        traj.set_timestep(timestep)
    for key, arr in additional_arrays.items():
        traj.set_array(key, np.array(arr))
    if thermo_file:
        header, arr, ts_index = get_thermo_props(thermo_file)
        timesteps_thermo = np.array(arr[:, ts_index], dtype=int).tolist()
        indices = []
        for ts in timesteps:
            try:
                indices.append(timesteps_thermo.index(ts))
            except ValueError:
                raise ValueError(f'Index {ts} is not in thermo file')
        indices = np.array(indices, dtype=int)
        # if thermo_te:
        #     colidx = header.index(thermo_te)
        #     traj.set_total_energies(arr[indices, colidx])
        headers_to_do = set(header)
        if thermo_pe:
            colidx = header.index(thermo_pe)
            traj.set_pot_energies(arr[indices, colidx])
            headers_to_do.remove(thermo_pe)
        if thermo_stress:
            stressall = []
            # voigt notation for stress:
            keys = ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')
            # first diagonal terms:
            for key in keys:
                fullkey = thermo_stress + key
                headers_to_do.remove(fullkey)
                colidx = header.index(fullkey)
                stressall.append(arr[indices, colidx])
            traj.set_stress(np.array(stressall).T)
        for kw in thermo_keywords:
            try:
                traj.set_array(kw, arr[indices, header.index(kw)])
            except Exception as e:
                print(e)
        # if thermo_ke:
        #     colidx = header.index(thermo_ke)
        #     traj.set_kinetic_energies(arr[indices, colidx])
    traj.apply_unit_conversion(
        l_conv=l_conv, v_conv=v_conv, f_conv=f_conv,
        e_conv=e_conv, s_conv=s_conv)
    if save_extxyz or (outfile is not None and outfile.endswith('extxyz')):
        from ase.io import write
        path_to_save = outfile or filename + '.extxyz'
        asetraj = traj.get_ase_trajectory()
        write(path_to_save, asetraj, format='extxyz')
    elif outfile:
        path_to_save = outfile or filename + '.traj'
        traj.save(path_to_save)
    return traj


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        """Reads lammps input and returns/saves a Trajectory instance""")
    parser.add_argument('filename',
                        help='The filename/path of the lammps trajectory')
    parser.add_argument('-o', '--outfile',
                        help='The filename/path to save trajectory at')
    parser.add_argument('-t', '--types', nargs='+',
                        help=('list of types, will be matched with'
                              ' types given in lammps'))
    parser.add_argument('-e', '--elements', nargs='+',
                        help='list of elements')
    parser.add_argument('--elements-file',
                        help=('A file containing the elements '
                              'as space-separated strings'))
    parser.add_argument('--mass-types',
                        help=('The input file of lammps containing '
                              'the masses of the types'))
    parser.add_argument('--timestep', type=float,
                        help='The timestep of the trajectory printed')
    parser.add_argument(
        '--units', default=None, choices=sorted(_LAMMPS_UNITS),
        help='LAMMPS unit system. Sets all conversion factors automatically. '
             'Cannot be combined with --v-conv, --f-conv, --e-conv, or '
             '--s-conv.')
    parser.add_argument('--l-conv', type=float, default=1.0,
                        help='Conversion factor for lengths to Angstrom '
                             '(positions and cell vectors).')
    parser.add_argument('--f-conv', type=float,
                        help='Conversion factor for forces to eV/A.',
                        default=1.0)
    parser.add_argument('--e-conv', type=float,
                        help='Conversion factor for energies to eV.',
                        default=1.0)
    parser.add_argument('--s-conv', type=float,
                        help='Conversion factor for stresses to eV/A^3.',
                        default=1.0)
    parser.add_argument('--v-conv', type=float, default=1.0,
                        help='Conversion factor for velocities to A/fs.')
    parser.add_argument(
        '-i', '--istep', type=int, default=1,
        help='Just take this frequency of steps from the trajectory')
    parser.add_argument(
        '--nsteps', type=int, default=None,
        help='Stop after storing this many frames (after skip and istep).')
    parser.add_argument(
        '--thermo-file', help='File path to equivalent thermo-file')
    parser.add_argument(
        '--thermo-pe',
        help='Thermo keyword for potential energy',)
    parser.add_argument('--thermo-stress',
                        help=('Thermo keyword for stress '
                              'without the xx/yy/xz..'))
    parser.add_argument('-tk', '--thermo-keywords',
                        nargs='*', default=[],
                        help='The keywords to read from the thermo file')
    parser.add_argument('--save-extxyz',
                        action='store_true',
                        help='save extxyz instead of traj')
    parser.add_argument('--ignore-velocities',
                        action='store_true',
                        help='Ignore velocities in dump file')
    parser.add_argument('--ignore-forces', action='store_true',
                        help='Ignore forces in dump file')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip this many first steps')
    parser.add_argument('-a', '--additional-keywords-dump', nargs='+',
                        help=('Additional keywords to be read from dump file'),
                        default=[])
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not print anything')
    args = parser.parse_args()
    if args.units is not None:
        manual = [f'--{k.replace("_", "-")}'
                  for k, default in (('l_conv', 1.0), ('v_conv', 1.0),
                                     ('f_conv', 1.0), ('e_conv', 1.0),
                                     ('s_conv', 1.0))
                  if getattr(args, k) != default]
        if manual:
            parser.error(
                '--units cannot be combined with individual conversion '
                'flags: ' + ', '.join(manual))
    read_lammps_dump(**vars(args))
