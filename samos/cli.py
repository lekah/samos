#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for samos.

Every analysis is a command of its own, installed both as a standalone
executable and as a sub-command of the ``samos`` dispatcher::

    samos-msd TRAJECTORY [options]
    samos msd TRAJECTORY [options]

Each command builds a single parser from the shared parent parsers in
this module, so its options may be given in any order.  The dispatcher
takes the command name as its first argument and passes the rest
through unchanged.

Commands: msd, vaf, vdos, rdf, adf.
"""

import sys
from argparse import ArgumentParser

import numpy as np
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from samos.trajectory import Trajectory
from samos.analysis.dynamics import DynamicsAnalyzer
from samos.analysis.rdf import RDF, ADF, pairs_with_other_species
from samos.utils.units import UNIT_SYSTEMS
from samos.plotting.plot_dynamics import (
    plot_msd_isotropic,
    plot_power_spectrum,
    plot_vaf_isotropic,
)
from samos.plotting.plot_rdf import plot_adf, plot_rdf


def _expand_elements(values):
    """
    Expand *values* (a list of strings from argparse) to a flat list of
    chemical symbols, one per atom.

    Two forms are accepted:

    * **Formula string** -- a single token that contains at least one
      digit, e.g. ``['Al31']`` or ``['Li10GeP2S12']``.  The formula is
      parsed with :class:`ase.formula.Formula` and expanded to the
      corresponding flat symbol list.
    * **Explicit list** -- multiple tokens (or a single token with no
      digit), e.g. ``['Al', 'Al', 'Al']``.  Returned unchanged.

    :param list values: Raw string list from ``--lammps-elements``.
    :returns: Flat list of chemical symbol strings.
    :raises ValueError: If a single-token formula cannot be parsed.
    """
    if len(values) == 1 and any(c.isdigit() for c in values[0]):
        from ase.formula import Formula
        return list(Formula(values[0]))
    return values


def _write_csv(filename, x_col, x_label, y_cols, y_labels):
    """
    Write a two-dimensional dataset to a CSV file.

    The first column is the x axis (*x_label*); subsequent columns are
    the per-species y values (*y_labels*).  All arrays must have the
    same length.

    :param str filename: Output file path.
    :param array x_col: 1-D x-axis array.
    :param str x_label: Header label for the x column.
    :param list y_cols: List of 1-D y arrays, one per species.
    :param list y_labels: Corresponding header labels.
    """
    header = ','.join([x_label] + y_labels)
    data = np.column_stack([x_col] + y_cols)
    np.savetxt(filename, data, delimiter=',', header=header, comments='')


def _make_axes(margins=None):
    """
    Create the single-axes figure every sub-command plots into.

    :param dict margins:
        GridSpec margins, defaults to the layout used by the time-series
        plots. The RDF passes its own, since its twin axis needs room on
        the right.
    :returns: the matplotlib Axes
    """
    if margins is None:
        margins = dict(left=0.18, right=0.95, bottom=0.18, top=0.95)
    gs = GridSpec(1, 1, **margins)
    fig = plt.figure(figsize=(4, 3))
    return fig.add_subplot(gs[0])


def _finish_plot(savefig):
    """
    Save the current figure to *savefig*, or show it when that is None.
    """
    if savefig:
        plt.savefig(savefig, dpi=240)
    else:
        plt.show()


def load_trajectory(trajectory_path, timestep=None, lammps_types=None,
                    lammps_elements=None, lammps=False, units=None):
    """
    Load a trajectory from *trajectory_path* and return a
    :class:`~samos.trajectory.Trajectory` instance.

    :param str trajectory_path:
        Path to the trajectory file. When *lammps* is True,
        *lammps_types* is given, or *lammps_elements* is given, the
        file is read as a LAMMPS dump; the ``.extxyz`` format is read
        via ASE; all other formats are passed to
        :meth:`~samos.trajectory.Trajectory.load_file`.
    :param float timestep:
        If given, override the timestep stored in the file (femtoseconds).
    :param list lammps_types:
        Chemical symbols in LAMMPS integer-type order, one entry per
        type (e.g. ``['Li', 'P', 'S']`` maps type 1->Li, 2->P, 3->S).
        Use this when the dump stores a ``type`` column but no
        ``element`` column.
    :param list lammps_elements:
        Explicit chemical symbol for every atom, length must equal the
        number of atoms in the dump (e.g. ``['Al']*31``).  Use this
        when neither a ``type`` column with a type map nor an
        ``element`` column is available.
    :param bool lammps:
        Read the file as a LAMMPS dump without supplying element
        information.  Use this when the dump already contains an
        ``element`` column so the symbols can be read directly.
    :param str units:
        Named unit system to convert from (e.g. ``'metal'``, ``'real'``).
        For LAMMPS dumps the conversion is applied during reading; for
        all other formats it is applied after loading via
        :meth:`~samos.trajectory.Trajectory.apply_unit_conversion`.
    :returns: :class:`~samos.trajectory.Trajectory`
    """
    if lammps_types is not None or lammps_elements is not None or lammps:
        from samos.io.lammps import read_lammps_dump
        traj = read_lammps_dump(trajectory_path,
                                types=lammps_types,
                                elements=lammps_elements,
                                timestep=timestep,
                                units=units)
        return traj
    try:
        traj = Trajectory.load_file(trajectory_path)
    except Exception:
        aselist = read(trajectory_path, format='extxyz', index=':')
        traj = Trajectory.from_atoms(aselist)
    if timestep is not None:
        traj.set_timestep(timestep)
    if units is not None:
        traj.apply_unit_conversion(**UNIT_SYSTEMS[units])
    return traj


def run_msd(traj, stepsize=1, species=None, plot=False, savefig=None,
            t_start_fit=5., t_end_fit=10., t_unit='ps', nblocks=1,
            backend='fortran', num_threads=None, write=None):
    """
    Compute the MSD for *traj* and optionally display or save a plot.

    :param traj: Pre-loaded trajectory.
    :type traj: :class:`~samos.trajectory.Trajectory`
    :param int stepsize:
        Outer-loop step size over trajectory frames (default 1).
    :param list species:
        Chemical symbols to analyse, e.g. ``['Li', 'O']``.
        If ``None``, all species present in the trajectory are used.
    :param bool plot:
        Show the MSD plot interactively (requires a display).
    :param str savefig:
        File path to save the plot. Mutually exclusive with *plot*.
    :param float t_start_fit:
        Start of the linear-fit window (in *t_unit*; default 5).
    :param float t_end_fit:
        End of the linear-fit window (in *t_unit*; default 10).
    :param str t_unit:
        Time unit for *t_start_fit* and *t_end_fit*
        (``'fs'``, ``'ps'``, or ``'dt'``; default ``'ps'``).
    :param int nblocks:
        Number of blocks to split the trajectory into (default 1).
    :param str backend:
        Compute kernel: ``'fortran'`` (default) or ``'cpp'`` (OpenMP).
    :param int num_threads:
        Number of OpenMP threads for the C++ backend. Ignored when
        backend is ``'fortran'``.
    :param str write:
        If given, write the mean MSD for each species to this CSV file.
        Columns: ``t_fs``, then ``msd_{species}_A2`` per species.
    """
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))

    dyn = DynamicsAnalyzer(trajectories=[traj])
    msd = dyn.get_msd(
        stepsize_t=stepsize,
        species_of_interest=species,
        t_start_fit=t_start_fit,
        t_end_fit=t_end_fit,
        t_unit=t_unit,
        nr_of_blocks=nblocks,
        backend=backend,
        num_threads=num_threads,
    )

    if write:
        t = msd.get_array('t_list_fs')
        _write_csv(
            write,
            x_col=t,
            x_label='t_fs',
            y_cols=[
                msd.get_array('msd_isotropic_{}_mean'.format(s))
                for s in species
            ],
            y_labels=['msd_{}_A2'.format(s) for s in species],
        )

    if plot or savefig:
        ax = _make_axes()
        plot_msd_isotropic(msd, ax=ax)

        _finish_plot(savefig)

    return msd


def run_vaf(traj, stepsize=1, species=None, plot=False, savefig=None,
            t_start_fit=5., t_end_fit=10., t_end=None, t_unit='ps',
            nblocks=1, integration='trapezoid',
            remove_angular_momentum=False, write=None):
    """
    Compute the VAF and its running integral (diffusion coefficient)
    for *traj*, and optionally display or save a plot.

    :param traj: Pre-loaded trajectory.
    :type traj: :class:`~samos.trajectory.Trajectory`
    :param int stepsize:
        Outer-loop step size over trajectory frames (default 1).
    :param list species:
        Chemical symbols to analyse, e.g. ``['Li', 'O']``.
        If ``None``, all species present in the trajectory are used.
    :param bool plot:
        Show the VAF plot interactively (requires a display).
    :param str savefig:
        File path to save the plot. Mutually exclusive with *plot*.
    :param float t_start_fit:
        Start of the integral-averaging window (in *t_unit*; default 5).
    :param float t_end_fit:
        End of the integral-averaging window (in *t_unit*; default 10).
    :param float t_end:
        Maximum lag time of the VAF (in *t_unit*).
        Defaults to *t_end_fit* when not set.
    :param str t_unit:
        Time unit for all time arguments
        (``'fs'``, ``'ps'``, or ``'dt'``; default ``'ps'``).
    :param int nblocks:
        Number of blocks to split the trajectory into (default 1).
    :param str integration:
        Integration method passed to :meth:`DynamicsAnalyzer.get_vaf`;
        ``'trapezoid'`` (default) or ``'simpson'``.
    :param str write:
        If given, write the mean VAF for each species to this CSV file.
        Columns: ``t_fs``, then ``vaf_{species}_A2fs-2`` per species.
    """
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))

    dyn = DynamicsAnalyzer(trajectories=[traj])
    vaf = dyn.get_vaf(
        integration=integration,
        remove_angular_momentum=remove_angular_momentum,
        stepsize_t=stepsize,
        species_of_interest=species,
        t_start_fit=t_start_fit,
        t_end_fit=t_end_fit,
        t_end=t_end,
        t_unit=t_unit,
        nr_of_blocks=nblocks,
    )

    if write:
        # Reconstruct the time axis from stored attributes; the VAF
        # result does not carry a pre-built t_list array like the MSD.
        attrs = vaf.get_attrs()
        ts = attrs['timestep_fs'] * attrs.get('stepsize_t', 1)
        t = ts * np.arange(
            attrs['t_start_dt'] / attrs.get('stepsize_t', 1),
            attrs['t_end_dt'] / attrs.get('stepsize_t', 1),
        )
        _write_csv(
            write,
            x_col=t,
            x_label='t_fs',
            y_cols=[
                vaf.get_array('vaf_isotropic_{}_mean'.format(s))
                for s in species
            ],
            y_labels=['vaf_{}_A2fs-2'.format(s) for s in species],
        )

    if plot or savefig:
        ax = _make_axes()
        plot_vaf_isotropic(vaf, ax=ax)

        _finish_plot(savefig)

    return vaf


def run_vdos(traj, species=None, plot=False, savefig=None,
             nblocks=1, smoothing=1,
             remove_angular_momentum=False, write=None):
    """
    Compute the vibrational density of states (power spectrum via
    Welch periodogram) for *traj*, and optionally display or save a plot.

    :param traj: Pre-loaded trajectory.
    :type traj: :class:`~samos.trajectory.Trajectory`
    :param list species:
        Chemical symbols to analyse, e.g. ``['Li', 'O']``.
        If ``None``, all species present in the trajectory are used.
    :param bool plot:
        Show the power spectrum interactively (requires a display).
    :param str savefig:
        File path to save the plot. Mutually exclusive with *plot*.
    :param int nblocks:
        Number of blocks to split the trajectory into (default 1).
    :param int smoothing:
        Smoothing kernel width in frequency bins (default 1, no
        smoothing).
    :param str write:
        If given, write the mean power spectrum for each species to this
        CSV file.  Columns: ``frequency_THz``, then one column per
        species named ``vdos_{species}``.
    """
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))

    dyn = DynamicsAnalyzer(trajectories=[traj])
    vdos = dyn.get_power_spectrum(
        species_of_interest=species,
        nr_of_blocks=nblocks,
        smoothing=smoothing,
        remove_angular_momentum=remove_angular_momentum,
    )

    if write:
        # Frequencies are stored per trajectory; trajectory 0 is used
        # as the reference since all trajectories share the same
        # sampling frequency and block length.
        freq = vdos.get_array('frequency_0')
        _write_csv(
            write,
            x_col=freq,
            x_label='frequency_THz',
            y_cols=[
                vdos.get_array('periodogram_{}_mean'.format(s))
                for s in species
            ],
            y_labels=['vdos_{}'.format(s) for s in species],
        )

    if plot or savefig:
        ax = _make_axes()
        plot_power_spectrum(vdos, ax=ax)

        _finish_plot(savefig)

    return vdos


def run_rdf(traj, stepsize=1, species=None, species_pairs=None,
            radius=5.0, bins=100, no_int=False,
            plot=False, savefig=None, write=None):
    """
    Compute the radial distribution function for *traj* and optionally
    display, save, or write a plot.

    :param traj: Pre-loaded trajectory.
    :type traj: :class:`~samos.trajectory.Trajectory`
    :param int stepsize:
        Step size over trajectory frames (default 1).
    :param list species:
        Compute all pairs between these species and the remaining
        species in the trajectory. Mutually exclusive with
        *species_pairs*.
    :param list species_pairs:
        Explicit list of ``'A-B'`` pair strings to compute,
        e.g. ``['Li-O', 'O-O']``. Mutually exclusive with *species*.
    :param float radius:
        Maximum radius of the RDF in Angstrom (default 5.0).
    :param int bins:
        Number of histogram bins (default 100).
    :param bool no_int:
        Suppress the running integral from the plot.
    :param bool plot:
        Show the RDF plot interactively (requires a display).
    :param str savefig:
        File path to save the plot. Mutually exclusive with *plot*.
    :param str write:
        If given, write the RDF and integral for each species pair to
        this CSV file. Columns: ``radius_A``, then
        ``rdf_{A}_{B}`` and ``int_{A}_{B}`` for each pair.
    """
    if species_pairs is not None and species is not None:
        raise ValueError(
            '--species-pairs and --species are mutually exclusive')

    if species_pairs is not None:
        pairs = [sp.split('-') for sp in species_pairs]
    elif species is not None:
        pairs = pairs_with_other_species(traj, species)
    else:
        pairs = None

    rdf_analyzer = RDF(trajectory=traj)
    res = rdf_analyzer.run(
        radius=radius, stepsize=stepsize, nbins=bins,
        species_pairs=pairs)

    if write:
        computed_pairs = res.get_attr('species_pairs')
        # All pairs share the same radii array; use the first pair's
        # radii as the x column.
        s1, s2 = computed_pairs[0]
        x = res.get_array('radii_{}_{}'.format(s1, s2))
        y_cols, y_labels = [], []
        for sp1, sp2 in computed_pairs:
            key = '{}_{}'.format(sp1, sp2)
            y_cols.append(res.get_array('rdf_{}'.format(key)))
            y_labels.append('rdf_{}'.format(key))
            y_cols.append(res.get_array('int_{}'.format(key)))
            y_labels.append('int_{}'.format(key))
        _write_csv(write, x_col=x, x_label='radius_A',
                   y_cols=y_cols, y_labels=y_labels)

    if plot or savefig:
        ax = _make_axes(dict(top=0.99, right=0.83,
                             left=0.14, bottom=0.16))
        plot_rdf(res, ax=ax, no_int=no_int)
        ax.set_xlim(-0.2, radius)

        _finish_plot(savefig)

    return res


def run_adf(traj, stepsize=1, centers=None, species_triplets=None,
            bonds=None, bonds_file=None, radius=None, static_bonds=False,
            bins=180, plot=False, savefig=None, write=None,
            species=None):
    """
    Compute the angular distribution function for *traj* and optionally
    display, save, or write results.

    :param traj: Pre-loaded trajectory.
    :type traj: :class:`~samos.trajectory.Trajectory`
    :param int stepsize:
        Step size over trajectory frames (default 1).
    :param list centers:
        Center species symbols.  Expands to all (*,center,*) triplets
        for species present in the trajectory.
        Mutually exclusive with *species_triplets*.
    :param list species_triplets:
        Explicit triplet strings ``'A-B-C'`` (center in the middle),
        e.g. ``['O-Si-O', 'O-Al-O']``.
        Mutually exclusive with *centers*.
    :param list bonds:
        Per-bond cutoff strings ``'SPEC:RMIN:RMAX'`` in Angstrom,
        e.g. ``['Si-O:1.4:2.0']``.  Mutually exclusive with
        *bonds_file* and *radius*.
    :param str bonds_file:
        Path to a LAMMPS data file whose ``Bonds`` section provides
        explicit bond topology.  Mutually exclusive with *bonds* and
        *radius*.
    :param float radius:
        Global cutoff in Angstrom: all atom pairs within this distance
        are treated as bonded.  Expands to ``(0, radius)`` for every
        species pair present in the trajectory.  Mutually exclusive
        with *bonds* and *bonds_file*.
    :param bool static_bonds:
        When using *bonds* cutoffs, detect topology once from the first
        frame and reuse it for all frames.
    :param int bins:
        Number of angle bins over [0, 180] degrees (default 180).
    :param bool plot:
        Show the ADF plot interactively (requires a display).
    :param str savefig:
        File path to save the plot.
    :param str write:
        If given, write the ADF for each triplet to this CSV file.
        Columns: ``angle_deg``, then ``adf_A_B_C`` per triplet.
    :param list species:
        Fallback center species when neither *centers* nor
        *species_triplets* is given (mapped from the top-level
        ``--species`` flag).
    """
    # Parse bond cutoffs: ['Si-O:1.4:2.0', ...] -> {'Si-O': (1.4, 2.0)}
    bonds_dict = None
    if bonds is not None:
        bonds_dict = {}
        for token in bonds:
            parts = token.split(':')
            if len(parts) != 3:
                raise ValueError(
                    "Bond cutoff must be SPEC:RMIN:RMAX "
                    "(e.g. Si-O:1.4:2.0), got '{}'.".format(token))
            bonds_dict[parts[0]] = (float(parts[1]), float(parts[2]))
    elif radius is not None:
        # Expand to all species pairs present in the trajectory.
        all_species = sorted(set(traj.get_types()))
        bonds_dict = {}
        for i, s1 in enumerate(all_species):
            for s2 in all_species[i:]:
                bonds_dict['{}-{}'.format(s1, s2)] = (0.0, radius)

    # Parse triplet strings: ['O-Si-O', ...] -> [('O','Si','O'), ...]
    triplets = None
    if species_triplets is not None:
        triplets = []
        for t in species_triplets:
            parts = t.split('-')
            if len(parts) != 3:
                raise ValueError(
                    "Triplet must be A-B-C with the center species in "
                    "the middle (e.g. O-Si-O), got '{}'.".format(t))
            triplets.append(tuple(parts))

    # Species selection priority: triplets > centers > species (top-level)
    effective_centers = centers
    if triplets is None and effective_centers is None:
        effective_centers = species  # None -> ADF computes all triplets

    adf_analyzer = ADF(trajectory=traj)
    if bonds_file is not None:
        adf_analyzer.load_bonds_lammps(bonds_file)

    res = adf_analyzer.run(
        species_triplets=triplets,
        centers=effective_centers,
        stepsize=stepsize,
        nbins=bins,
        bonds=bonds_dict,
        static_bonds=static_bonds)

    computed_triplets = res.get_attr('species_triplets')

    if write:
        if not computed_triplets:
            print('Warning: no triplets computed; not writing output.')
        else:
            sl, sc, sr = computed_triplets[0]
            x = res.get_array('angles_{}_{}_{}'.format(sl, sc, sr))
            y_cols, y_labels = [], []
            for sl, sc, sr in computed_triplets:
                key = '{}_{}_{}'.format(sl, sc, sr)
                if 'adf_{}'.format(key) in res:
                    y_cols.append(res.get_array('adf_{}'.format(key)))
                    y_labels.append('adf_{}'.format(key))
            if y_cols:
                _write_csv(write, x_col=x, x_label='angle_deg',
                           y_cols=y_cols, y_labels=y_labels)

    if plot or savefig:
        plot_adf(res, ax=_make_axes())

        _finish_plot(savefig)

    return res


# Sub-command name -> one-line summary, used by the samos dispatcher.
COMMAND_SUMMARIES = (
    ('msd', 'Mean-square displacement and the diffusion coefficient.'),
    ('vaf', 'Velocity autocorrelation function and its integral.'),
    ('vdos', 'Vibrational density of states (Welch periodogram).'),
    ('rdf', 'Radial distribution function and its running integral.'),
    ('adf', 'Angular distribution function over bond triplets.'),
)


def _traj_parser():
    """
    Build the parser holding the options every command accepts: which
    file to read, how to interpret it, what to do to it before the
    analysis, and where the output goes.

    Returned as a parent parser (``add_help=False``) so that every
    command parser gets one flat namespace from a single
    ``parse_args`` call.  There is therefore no ordering rule between
    these options and the command-specific ones.

    :returns: :class:`argparse.ArgumentParser` to be used as a parent.
    """
    p = ArgumentParser(add_help=False)
    p.add_argument(
        'trajectory_path',
        help='Path to the trajectory file (.extxyz or native samos format).')
    p.add_argument(
        '--timestep', type=float, default=None, metavar='FS',
        help='Override the trajectory timestep in femtoseconds.')
    p.add_argument(
        '--lammps-types', nargs='+', metavar='SYMBOL',
        dest='lammps_types',
        help='Read the trajectory as a LAMMPS dump file and map LAMMPS '
             'integer types to these chemical symbols in type order '
             '(e.g. Li P S for types 1 2 3). Use when the dump stores '
             'a "type" column but no "element" column.')
    p.add_argument(
        '--lammps-elements', nargs='+', metavar='SYMBOL',
        dest='lammps_elements',
        help='Read the trajectory as a LAMMPS dump file and assign '
             'chemical symbols explicitly, one per atom in atom-id order. '
             'Accepts either a formula string (e.g. Al31 or Li10GeP2S12) '
             'or a space-separated list (e.g. Al Al Al). Use when '
             'neither a "type" column with a type map nor an "element" '
             'column is available.')
    p.add_argument(
        '--lammps', action='store_true', default=False,
        help='Read the trajectory as a LAMMPS dump file. Use this when '
             'the dump already contains an "element" column so no element '
             'list needs to be supplied.')
    p.add_argument(
        '--species', nargs='+', metavar='SYMBOL',
        help='Chemical symbols to analyse (default: all species).')
    p.add_argument(
        '--write', metavar='FILE',
        help='Write results to FILE as CSV (one column per species).')
    p.add_argument(
        '--recenter', action='store_true',
        help='Recenter positions and velocities before analysis.')
    p.add_argument(
        '--compute-velocities', action='store_true',
        dest='compute_velocities',
        help='Compute velocities from positions using the Verlet finite-'
             'difference formula before analysis. Required for VAF and '
             'VDOS when the trajectory does not store velocities.')
    p.add_argument(
        '--transform-species', metavar='SYMBOL', default=None,
        dest='transform_species',
        help='Relabel all atoms as SYMBOL before analysis.')
    p.add_argument(
        '--units', default=None, metavar='SYSTEM',
        choices=sorted(UNIT_SYSTEMS),
        help='Convert trajectory arrays to samos internal units '
             '(velocities -> A/fs, forces -> eV/A, energies -> eV). '
             'For LAMMPS dumps conversion is applied during reading; '
             'for all other formats it is applied after loading. '
             'Choices: ' + ', '.join(sorted(UNIT_SYSTEMS)) + '.')

    plot_group = p.add_mutually_exclusive_group()
    plot_group.add_argument(
        '--plot', action='store_true', help='Show the plot interactively.')
    plot_group.add_argument(
        '--savefig', metavar='FILE',
        help='Save the plot to FILE instead of showing it.')
    return p


def _block_parser():
    """
    Build the parent parser for block averaging, which only the
    time-correlation commands (msd, vaf, vdos) support.  It used to be
    a global option that rdf and adf accepted and silently ignored.

    :returns: :class:`argparse.ArgumentParser` to be used as a parent.
    """
    p = ArgumentParser(add_help=False)
    p.add_argument(
        '-n', '--nblocks', type=int, default=1, metavar='N',
        help='Number of blocks to split the trajectory into (default: 1).')
    return p


def _make_parser(command, description, blocks=True):
    """
    Build the parser for one command, with the shared parents attached.

    :param str command: The command name, e.g. ``'msd'``.
    :param str description: Shown at the top of ``--help``.
    :param bool blocks: Whether the command supports ``-n/--nblocks``.
    :returns: :class:`argparse.ArgumentParser`
    """
    parents = [_traj_parser()]
    if blocks:
        parents.append(_block_parser())
    return ArgumentParser(prog='samos-{}'.format(command),
                          description=description, parents=parents)


def _add_fit_window(p, what):
    """
    Add the ``-ts``/``-te``/``--t-unit`` trio shared by msd and vaf.

    :param p: The parser to add to.
    :param str what: What the window is used for, spliced into the help.
    """
    p.add_argument(
        '-ts', '--t-start-fit', type=float, default=5., metavar='T',
        dest='t_start_fit',
        help='Start of the {} (default: 5, unit: --t-unit).'.format(what))
    p.add_argument(
        '-te', '--t-end-fit', type=float, default=10., metavar='T',
        dest='t_end_fit',
        help='End of the {} (default: 10, unit: --t-unit).'.format(what))
    p.add_argument(
        '--t-unit', default='ps', choices=['fs', 'ps', 'dt'],
        dest='t_unit', metavar='UNIT',
        help='Time unit for the time arguments: fs, ps, or dt '
             '(default: ps).')


def _add_stepsize(p):
    """Add the frame stride shared by every command except vdos."""
    p.add_argument(
        '-s', '--stepsize', type=int, default=1, metavar='N',
        help='Step size over trajectory frames (default: 1).')


def _add_angular_momentum(p):
    """Add the angular-momentum removal flag shared by vaf and vdos."""
    p.add_argument(
        '-a', '--remove-angular-momentum', action='store_true',
        dest='remove_angular_momentum',
        help='Remove rigid-body rotational contribution from velocities.')


def _parser_msd():
    p = _make_parser('msd', 'Calculate and optionally plot the MSD.')
    _add_stepsize(p)
    _add_fit_window(p, 'linear-fit window')
    p.add_argument(
        '--backend', default='fortran', choices=['fortran', 'cpp'],
        help='Compute kernel: fortran (default) or cpp (OpenMP).')
    p.add_argument(
        '-j', '--num-threads', type=int, default=None, metavar='N',
        dest='num_threads',
        help='OpenMP thread count for the cpp backend '
             '(default: OMP_NUM_THREADS).')
    return p


def _parser_vaf():
    p = _make_parser(
        'vaf', 'Calculate the VAF and its integral (diff. coefficient).')
    _add_stepsize(p)
    _add_fit_window(p, 'integral-averaging window')
    p.add_argument(
        '--t-end', type=float, default=None, metavar='T',
        dest='t_end',
        help='Maximum lag time of the VAF (unit: --t-unit; '
             'default: t-end-fit).')
    p.add_argument(
        '--integration', default='trapezoid', metavar='METHOD',
        choices=['trapezoid', 'simpson'],
        help='Integration method: trapezoid (default) or simpson.')
    _add_angular_momentum(p)
    return p


def _parser_vdos():
    p = _make_parser(
        'vdos',
        'Calculate the VDOS via a Welch periodogram of atomic velocities.')
    p.add_argument(
        '-sm', '--smoothing', type=int, default=1, metavar='N',
        help='Smoothing kernel width in frequency bins '
             '(default: 1, no smoothing).')
    _add_angular_momentum(p)
    return p


def _parser_rdf():
    p = _make_parser('rdf', 'Calculate the RDF and its running integral.',
                     blocks=False)
    _add_stepsize(p)
    p.add_argument(
        '-r', '--radius', type=float, default=5.0, metavar='A',
        help='Maximum radius of the RDF in Angstrom (default: 5.0).')
    p.add_argument(
        '-b', '--bins', type=int, default=100, metavar='N',
        help='Number of histogram bins (default: 100).')
    # --species and --species-pairs are mutually exclusive; the check is
    # in run_rdf because --species comes from the shared parent parser.
    p.add_argument(
        '--species-pairs', nargs='+', metavar='A-B', dest='species_pairs',
        help='Species pairs to compute, e.g. Li-O O-O. '
             'Mutually exclusive with --species.')
    p.add_argument(
        '--no-int', action='store_true', dest='no_int',
        help='Suppress the running integral from the plot.')
    return p


def _parser_adf():
    p = _make_parser(
        'adf',
        'Calculate the ADF for bond triplets A-B-C (B is the center atom).',
        blocks=False)
    _add_stepsize(p)
    p.add_argument(
        '-b', '--bins', type=int, default=180, metavar='N',
        help='Number of angle bins over [0, 180] degrees (default: 180).')
    p.add_argument(
        '--static-bonds', action='store_true', dest='static_bonds',
        help='When using --bonds cutoffs, detect topology once from the '
             'first frame and reuse it for all frames (default: detect '
             'per frame).')

    bond_src = p.add_mutually_exclusive_group(required=True)
    bond_src.add_argument(
        '-r', '--radius', type=float, metavar='A',
        help='Global neighbor cutoff in Angstrom: all pairs within this '
             'distance are treated as bonded.  Applied to every species '
             'pair present in the trajectory.')
    bond_src.add_argument(
        '--bonds', nargs='+', metavar='SPEC:RMIN:RMAX',
        help='Per-bond distance cutoffs for neighbor detection.  Each '
             'token has the form SPEC:RMIN:RMAX in Angstrom, e.g. '
             'Si-O:1.4:2.0 Al-O:1.6:2.2.')
    bond_src.add_argument(
        '--bonds-file', metavar='FILE', dest='bonds_file',
        help='LAMMPS data file containing a Bonds section with explicit '
             'bond topology.')

    triplet_sel = p.add_mutually_exclusive_group()
    triplet_sel.add_argument(
        '--centers', nargs='+', metavar='SYMBOL',
        help='Center species for the ADF.  Computes all (*,center,*) '
             'triplets.  Mutually exclusive with --species-triplets.')
    triplet_sel.add_argument(
        '--species-triplets', nargs='+', metavar='A-B-C',
        dest='species_triplets',
        help='Explicit triplets to compute, center in the middle, '
             'separated by dashes (e.g. O-Si-O O-Al-O).  '
             'Mutually exclusive with --centers.')
    return p


def _prepare(args):
    """
    Load the trajectory named on the command line and apply the
    preprocessing requested by the shared options.

    :param args: The parsed :class:`argparse.Namespace`.
    :returns: :class:`~samos.trajectory.Trajectory`
    """
    lammps_elements = (
        _expand_elements(args.lammps_elements)
        if args.lammps_elements is not None else None
    )
    traj = load_trajectory(args.trajectory_path, timestep=args.timestep,
                           lammps_types=args.lammps_types,
                           lammps_elements=lammps_elements,
                           lammps=args.lammps,
                           units=args.units)

    if args.transform_species:
        traj.transform_species(args.transform_species)

    if args.recenter:
        traj.recenter()

    if args.compute_velocities:
        traj.calculate_velocities_from_positions()

    return traj


def _output_kwargs(args):
    """The species/output options every run_* function takes."""
    return dict(species=args.species, plot=args.plot,
                savefig=args.savefig, write=args.write)


def main_msd(argv=None):
    """Entry point of the ``samos-msd`` command."""
    args = _parser_msd().parse_args(argv)
    run_msd(_prepare(args), stepsize=args.stepsize, nblocks=args.nblocks,
            t_start_fit=args.t_start_fit, t_end_fit=args.t_end_fit,
            t_unit=args.t_unit, backend=args.backend,
            num_threads=args.num_threads, **_output_kwargs(args))


def main_vaf(argv=None):
    """Entry point of the ``samos-vaf`` command."""
    args = _parser_vaf().parse_args(argv)
    run_vaf(_prepare(args), stepsize=args.stepsize, nblocks=args.nblocks,
            t_start_fit=args.t_start_fit, t_end_fit=args.t_end_fit,
            t_end=args.t_end, t_unit=args.t_unit,
            integration=args.integration,
            remove_angular_momentum=args.remove_angular_momentum,
            **_output_kwargs(args))


def main_vdos(argv=None):
    """Entry point of the ``samos-vdos`` command."""
    args = _parser_vdos().parse_args(argv)
    run_vdos(_prepare(args), nblocks=args.nblocks, smoothing=args.smoothing,
             remove_angular_momentum=args.remove_angular_momentum,
             **_output_kwargs(args))


def main_rdf(argv=None):
    """Entry point of the ``samos-rdf`` command."""
    args = _parser_rdf().parse_args(argv)
    run_rdf(_prepare(args), stepsize=args.stepsize,
            species_pairs=args.species_pairs, radius=args.radius,
            bins=args.bins, no_int=args.no_int, **_output_kwargs(args))


def main_adf(argv=None):
    """Entry point of the ``samos-adf`` command."""
    args = _parser_adf().parse_args(argv)
    run_adf(_prepare(args), stepsize=args.stepsize, centers=args.centers,
            species_triplets=args.species_triplets, bonds=args.bonds,
            bonds_file=args.bonds_file, radius=args.radius,
            static_bonds=args.static_bonds, bins=args.bins,
            **_output_kwargs(args))


MAINS = {
    'msd': main_msd,
    'vaf': main_vaf,
    'vdos': main_vdos,
    'rdf': main_rdf,
    'adf': main_adf,
}


def _usage(stream):
    """Write the dispatcher's command listing to *stream*."""
    stream.write(
        'usage: samos COMMAND [options]\n\n'
        'Every command is also installed as a standalone executable, so\n'
        '"samos msd ..." and "samos-msd ..." are the same thing.\n\n'
        'Commands:\n')
    for name, summary in COMMAND_SUMMARIES:
        stream.write('  {:<6}{}\n'.format(name, summary))
    stream.write(
        "\nRun 'samos COMMAND --help' for the options of one command.\n")


def main(argv=None):
    """
    Entry point of the ``samos`` command, which dispatches on its first
    argument and passes everything after it through unchanged.  The
    command name is always ``argv[0]``, so no option can be mistaken
    for it and no option needs to precede it.

    :param list argv: Argument list, defaults to ``sys.argv[1:]``.
    :returns: The process exit status.
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _usage(sys.stderr)
        return 2
    command = argv[0]
    if command in ('-h', '--help'):
        _usage(sys.stdout)
        return 0
    if command not in MAINS:
        sys.stderr.write(
            "samos: error: unknown command '{}' (choose from {})\n".format(
                command, ', '.join(name for name, _ in COMMAND_SUMMARIES)))
        return 2
    return MAINS[command](argv[1:])


if __name__ == '__main__':
    sys.exit(main())
