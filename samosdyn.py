#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
samosdyn.py — command-line interface for samos dynamics analysis.

Usage
-----
  samosdyn.py TRAJECTORY [--timestep FS] COMMAND [options]

Currently supported sub-commands
---------------------------------
msd    Calculate the mean-square displacement (MSD) and optionally plot it.
vaf    Calculate the velocity autocorrelation function and optionally plot it.
vdos   Calculate the vibrational density of states (power spectrum) and
       optionally plot it.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from samos.trajectory import Trajectory
from samos.analysis.dynamics import DynamicsAnalyzer
from samos.plotting.plot_dynamics import (
    plot_msd_isotropic,
    plot_power_spectrum,
    plot_vaf_isotropic,
)


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


def load_trajectory(trajectory_path, timestep=None):
    """
    Load a trajectory from *trajectory_path* and return a
    :class:`~samos.trajectory.Trajectory` instance.

    :param str trajectory_path:
        Path to the trajectory file. The ``.extxyz`` format is read via
        ASE; all other formats are passed to
        :meth:`~samos.trajectory.Trajectory.load_file`.
    :param float timestep:
        If given, override the timestep stored in the file (femtoseconds).
    :returns: :class:`~samos.trajectory.Trajectory`
    """
    try:
        traj = Trajectory.load_file(trajectory_path)
    except Exception:
        aselist = read(trajectory_path, format='extxyz', index=':')
        traj = Trajectory.from_atoms(aselist)
    if timestep is not None:
        traj.set_timestep(timestep)

    return traj


def run_msd(traj, stepsize=1, species=None, plot=False, savefig=None,
            t_start_fit_ps=5., t_end_fit_ps=10., nblocks=1, write=None):
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
    :param float t_start_fit_ps:
        Start of the linear-fit window in picoseconds.
    :param float t_end_fit_ps:
        End of the linear-fit window in picoseconds.
    :param int nblocks:
        Number of blocks to split the trajectory into (default 1).
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
        t_start_fit_ps=t_start_fit_ps,
        t_end_fit_ps=t_end_fit_ps,
        nr_of_blocks=nblocks,
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
        gs = GridSpec(1, 1, left=0.18, right=0.95,
                      bottom=0.18, top=0.95)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(gs[0])
        plot_msd_isotropic(msd, ax=ax)

        if savefig:
            plt.savefig(savefig, dpi=240)
        else:
            plt.show()

    return msd


def run_vaf(traj, stepsize=1, species=None, plot=False, savefig=None,
            t_start_fit_ps=5., t_end_fit_ps=10., t_end_ps=None,
            nblocks=1, integration='trapezoid', write=None):
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
    :param float t_start_fit_ps:
        Start of the integral-averaging window in picoseconds.
    :param float t_end_fit_ps:
        End of the integral-averaging window in picoseconds.
    :param float t_end_ps:
        Maximum lag time of the VAF in picoseconds. Defaults to
        *t_end_fit_ps* when not set.
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

    kwargs = dict(
        stepsize_t=stepsize,
        species_of_interest=species,
        t_start_fit_ps=t_start_fit_ps,
        t_end_fit_ps=t_end_fit_ps,
        nr_of_blocks=nblocks,
    )
    if t_end_ps is not None:
        kwargs['t_end_ps'] = t_end_ps

    dyn = DynamicsAnalyzer(trajectories=[traj])
    vaf = dyn.get_vaf(integration=integration, **kwargs)

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
        gs = GridSpec(1, 1, left=0.18, right=0.95,
                      bottom=0.18, top=0.95)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(gs[0])
        plot_vaf_isotropic(vaf, ax=ax)

        if savefig:
            plt.savefig(savefig, dpi=240)
        else:
            plt.show()

    return vaf


def run_vdos(traj, species=None, plot=False, savefig=None,
             nblocks=1, smoothing=1, write=None):
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
        smoothing). Passed as ``smothening`` to
        :meth:`DynamicsAnalyzer.get_power_spectrum` to work around the
        typo in the underlying API.
    :param str write:
        If given, write the mean power spectrum for each species to this
        CSV file.  Columns: ``frequency_THz``, then one column per
        species named ``vdos_{species}``.
    """
    if species is None:
        species = sorted(set(traj.atoms.get_chemical_symbols()))

    dyn = DynamicsAnalyzer(trajectories=[traj])
    # 'smothening' is the misspelled kwarg in get_power_spectrum;
    # we expose a correctly-spelled parameter and translate it here.
    vdos = dyn.get_power_spectrum(
        species_of_interest=species,
        nr_of_blocks=nblocks,
        smothening=smoothing,
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
        gs = GridSpec(1, 1, left=0.18, right=0.95,
                      bottom=0.18, top=0.95)
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(gs[0])
        plot_power_spectrum(vdos, ax=ax)

        if savefig:
            plt.savefig(savefig, dpi=240)
        else:
            plt.show()

    return vdos


def _build_parser():
    parser = ArgumentParser(
        prog='samosdyn',
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Trajectory arguments are shared by all sub-commands and are
    # placed before the sub-command name so they appear in the top-level
    # help and do not need to be repeated in every sub-parser.
    parser.add_argument(
        'trajectory_path',
        help='Path to the trajectory file '
             '(.extxyz or native samos format).',
    )
    parser.add_argument(
        '--timestep',
        type=float, default=None, metavar='FS',
        help='Override the trajectory timestep in femtoseconds.',
    )
    parser.add_argument(
        '--species',
        nargs='+', metavar='SYMBOL',
        help='Chemical symbols to analyse '
             '(default: all species in trajectory).',
    )
    parser.add_argument(
        '-n', '--nblocks',
        type=int, default=1, metavar='N',
        help='Number of blocks to split the trajectory into (default: 1).',
    )

    parser.add_argument(
        '--write',
        metavar='FILE',
        help='Write results to FILE as CSV (one column per species).',
    )

    parser.add_argument(
        '--recenter',
        action='store_true',
        help='Recenter positions and velocities before analysis.',
    )
    parser.add_argument(
        '--transform-species',
        metavar='SYMBOL', default=None, dest='transform_species',
        help='Relabel all atoms as SYMBOL before analysis.',
    )

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        '--plot',
        action='store_true',
        help='Show the plot interactively.',
    )
    plot_group.add_argument(
        '--savefig',
        metavar='FILE',
        help='Save the plot to FILE instead of showing it.',
    )

    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    # ------------------------------------------------------------------
    # msd sub-command
    # ------------------------------------------------------------------
    p_msd = sub.add_parser(
        'msd',
        help='Calculate and optionally plot the mean-square displacement.',
        description='Calculate and optionally plot the MSD.',
    )
    p_msd.add_argument(
        '-s', '--stepsize',
        type=int, default=1, metavar='N',
        help='Step size over trajectory frames (default: 1).',
    )
    p_msd.add_argument(
        '-ts', '--t-start-fit-ps',
        type=float, default=5., metavar='PS',
        dest='t_start_fit_ps',
        help='Start of the linear-fit window in picoseconds (default: 5).',
    )
    p_msd.add_argument(
        '-te', '--t-end-fit-ps',
        type=float, default=10., metavar='PS',
        dest='t_end_fit_ps',
        help='End of the linear-fit window in picoseconds (default: 10).',
    )
    # ------------------------------------------------------------------
    # vaf sub-command
    # ------------------------------------------------------------------
    p_vaf = sub.add_parser(
        'vaf',
        help='Calculate and optionally plot the velocity autocorrelation '
             'function.',
        description='Calculate the VAF and its running integral '
                    '(diffusion coefficient).',
    )
    p_vaf.add_argument(
        '-s', '--stepsize',
        type=int, default=1, metavar='N',
        help='Step size over trajectory frames (default: 1).',
    )
    p_vaf.add_argument(
        '-ts', '--t-start-fit-ps',
        type=float, default=5., metavar='PS',
        dest='t_start_fit_ps',
        help='Start of the integral-averaging window in picoseconds '
             '(default: 5).',
    )
    p_vaf.add_argument(
        '-te', '--t-end-fit-ps',
        type=float, default=10., metavar='PS',
        dest='t_end_fit_ps',
        help='End of the integral-averaging window in picoseconds '
             '(default: 10).',
    )
    p_vaf.add_argument(
        '--t-end-ps',
        type=float, default=None, metavar='PS',
        dest='t_end_ps',
        help='Maximum lag time of the VAF in picoseconds '
             '(default: t-end-fit-ps).',
    )
    p_vaf.add_argument(
        '--integration',
        default='trapezoid', metavar='METHOD',
        choices=['trapezoid', 'simpson'],
        help='Integration method for the VAF integral '
             '(trapezoid or simpson, default: trapezoid).',
    )
    # ------------------------------------------------------------------
    # vdos sub-command
    # ------------------------------------------------------------------
    p_vdos = sub.add_parser(
        'vdos',
        help='Calculate and optionally plot the vibrational density of '
             'states (power spectrum).',
        description='Calculate the vibrational density of states via a '
                    'Welch periodogram of the atomic velocities.',
    )
    p_vdos.add_argument(
        '--smoothing',
        type=int, default=1, metavar='N',
        help='Smoothing kernel width in frequency bins '
             '(default: 1, no smoothing).',
    )
    return parser


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()

    traj = load_trajectory(args.trajectory_path, timestep=args.timestep)

    if args.transform_species:
        traj.transform_species(args.transform_species)

    if args.recenter:
        traj.recenter()

    if args.command == 'msd':
        run_msd(
            traj,
            stepsize=args.stepsize,
            species=args.species,
            plot=args.plot,
            savefig=args.savefig,
            t_start_fit_ps=args.t_start_fit_ps,
            t_end_fit_ps=args.t_end_fit_ps,
            nblocks=args.nblocks,
            write=args.write,
        )
    elif args.command == 'vaf':
        run_vaf(
            traj,
            stepsize=args.stepsize,
            species=args.species,
            plot=args.plot,
            savefig=args.savefig,
            t_start_fit_ps=args.t_start_fit_ps,
            t_end_fit_ps=args.t_end_fit_ps,
            t_end_ps=args.t_end_ps,
            nblocks=args.nblocks,
            integration=args.integration,
            write=args.write,
        )
    elif args.command == 'vdos':
        run_vdos(
            traj,
            species=args.species,
            plot=args.plot,
            savefig=args.savefig,
            nblocks=args.nblocks,
            smoothing=args.smoothing,
            write=args.write,
        )
