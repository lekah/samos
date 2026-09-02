# -*- coding: utf-8 -*-
"""
Tests for samos.cli, the command-line interface.

Every analysis is its own command built from a single parser, so these
tests cover both that options are accepted in any order and that each
command runs end to end and writes the CSV it promises.
"""

import os
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

import matplotlib
matplotlib.use('Agg')

from samos import cli  # noqa: E402


def _write_extxyz(path, nstep=120, seed=0):
    """
    Write a small diffusive Li4O4 trajectory to *path*.

    The CLI reads from disk, so the fixture has to be a real file
    rather than an in-memory Trajectory.  Positions are a random walk
    so that the MSD is monotonic and the fit windows below are not
    degenerate.
    """
    rng = np.random.default_rng(seed)
    cell = np.eye(3) * 6.0
    start = rng.random((8, 3)) * 6.0
    walk = start + np.cumsum(
        rng.normal(scale=0.05, size=(nstep, 8, 3)), axis=0)
    images = []
    for positions in walk:
        atoms = Atoms('Li4O4', cell=cell, pbc=True)
        atoms.set_positions(positions)
        images.append(atoms)
    ase_write(path, images, format='extxyz')


class CLITestCase(unittest.TestCase):
    """Base class providing a trajectory file in a temporary directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.traj_path = os.path.join(self.tmpdir, 'traj.extxyz')
        _write_extxyz(self.traj_path)

    def tearDown(self):
        self._tmp.cleanup()

    def out(self, name):
        """Path to an output file inside the temporary directory."""
        return os.path.join(self.tmpdir, name)

    def assert_csv(self, path, header, min_rows=2):
        """Assert that *path* is a CSV with *header* and some data."""
        self.assertTrue(os.path.exists(path), '{} not written'.format(path))
        with open(path) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], header)
        self.assertGreaterEqual(len(lines) - 1, min_rows)


class TestOptionOrder(CLITestCase):
    """
    Each command has one parser, so an option may appear anywhere after
    the trajectory path.  The sub-command parser this replaced forced
    every shared option to precede the command name, and accepted the
    same short flag with a different meaning on either side of it.
    """

    def test_order_does_not_matter(self):
        parser = cli._parser_msd()
        orders = (
            [self.traj_path, '-n', '3', '-s', '2', '--backend', 'cpp'],
            [self.traj_path, '--backend', 'cpp', '-n', '3', '-s', '2'],
            ['-n', '3', self.traj_path, '--backend', 'cpp', '-s', '2'],
            ['-n', '3', '-s', '2', '--backend', 'cpp', self.traj_path],
        )
        namespaces = [vars(parser.parse_args(a)) for a in orders]
        for other in namespaces[1:]:
            self.assertEqual(namespaces[0], other)

    def test_list_option_must_not_precede_the_path(self):
        """
        The one ordering rule left: a list-valued option immediately
        before the trajectory path swallows it, because argparse cannot
        tell where an nargs='+' list ends and a positional begins.
        Writing the path first -- the natural order -- always works.
        """
        parser = cli._parser_rdf()
        args = parser.parse_args([self.traj_path, '--species', 'Li', 'O'])
        self.assertEqual(args.species, ['Li', 'O'])
        self.assertEqual(args.trajectory_path, self.traj_path)
        with self.assertRaises(SystemExit):
            parser.parse_args(['--species', 'Li', 'O', self.traj_path])


class TestFlags(CLITestCase):
    """The short flags that used to collide across parsers."""

    def test_n_is_nblocks_and_j_is_threads(self):
        args = cli._parser_msd().parse_args(
            [self.traj_path, '-n', '7', '-j', '4'])
        self.assertEqual(args.nblocks, 7)
        self.assertEqual(args.num_threads, 4)

    def test_msd_backend_has_no_short_flag(self):
        # -b used to mean --backend here and --bins elsewhere.
        with self.assertRaises(SystemExit):
            cli._parser_msd().parse_args([self.traj_path, '-b', 'cpp'])
        args = cli._parser_msd().parse_args(
            [self.traj_path, '--backend', 'cpp'])
        self.assertEqual(args.backend, 'cpp')

    def test_b_is_bins(self):
        rdf = cli._parser_rdf().parse_args([self.traj_path, '-b', '33'])
        self.assertEqual(rdf.bins, 33)
        adf = cli._parser_adf().parse_args(
            [self.traj_path, '-r', '3', '-b', '44'])
        self.assertEqual(adf.bins, 44)

    def test_integration_has_no_short_flag(self):
        # -i is reserved for the trajectory slice option.
        with self.assertRaises(SystemExit):
            cli._parser_vaf().parse_args([self.traj_path, '-i', 'simpson'])
        args = cli._parser_vaf().parse_args(
            [self.traj_path, '--integration', 'simpson'])
        self.assertEqual(args.integration, 'simpson')

    def test_only_correlation_commands_take_nblocks(self):
        for parser in (cli._parser_msd(), cli._parser_vaf(),
                       cli._parser_vdos()):
            self.assertEqual(parser.parse_args([self.traj_path]).nblocks, 1)
        # Blocking is meaningless for rdf and adf, which used to accept
        # --nblocks as a global option and silently ignore it.
        for parser in (cli._parser_rdf(), cli._parser_adf()):
            with self.assertRaises(SystemExit):
                parser.parse_args([self.traj_path, '-r', '3', '-n', '2'])


class TestDispatcher(CLITestCase):
    """The samos command, which dispatches on its first argument."""

    def test_no_arguments_is_an_error(self):
        self.assertEqual(cli.main([]), 2)

    def test_help_succeeds(self):
        self.assertEqual(cli.main(['--help']), 0)

    def test_unknown_command_is_an_error(self):
        self.assertEqual(cli.main(['msdd']), 2)

    def test_every_listed_command_is_dispatchable(self):
        listed = [name for name, _ in cli.COMMAND_SUMMARIES]
        self.assertEqual(sorted(listed), sorted(cli.MAINS))

    def test_dispatch_equals_direct_call(self):
        common = ['--timestep', '1', '-r', '4', '-b', '20']
        cli.main(['rdf', self.traj_path, '--write', self.out('a.csv')]
                 + common)
        cli.main_rdf([self.traj_path, '--write', self.out('b.csv')] + common)
        with open(self.out('a.csv')) as f:
            a = f.read()
        with open(self.out('b.csv')) as f:
            b = f.read()
        self.assertEqual(a, b)


class TestCommands(CLITestCase):
    """Each command end to end, from argv to the written CSV."""

    def test_msd(self):
        cli.main_msd([self.traj_path, '--timestep', '1', '-n', '2',
                      '--t-unit', 'dt', '-ts', '5', '-te', '20',
                      '--write', self.out('msd.csv')])
        self.assert_csv(self.out('msd.csv'), 't_fs,msd_Li_A2,msd_O_A2')

    def test_vaf(self):
        cli.main_vaf([self.traj_path, '--timestep', '1',
                      '--compute-velocities', '--t-unit', 'dt',
                      '-ts', '5', '-te', '20',
                      '--write', self.out('vaf.csv')])
        self.assert_csv(self.out('vaf.csv'),
                        't_fs,vaf_Li_A2fs-2,vaf_O_A2fs-2')

    def test_vdos(self):
        cli.main_vdos([self.traj_path, '--timestep', '1',
                       '--compute-velocities', '-sm', '3',
                       '--write', self.out('vdos.csv')])
        self.assert_csv(self.out('vdos.csv'),
                        'frequency_THz,vdos_Li,vdos_O')

    def test_rdf(self):
        cli.main_rdf([self.traj_path, '--timestep', '1', '-r', '4',
                      '-b', '20', '--species-pairs', 'Li-O',
                      '--write', self.out('rdf.csv')])
        self.assert_csv(self.out('rdf.csv'),
                        'radius_A,rdf_Li_O,int_Li_O')

    def test_adf(self):
        cli.main_adf([self.traj_path, '--timestep', '1', '-r', '3',
                      '-b', '20', '--species-triplets', 'O-Li-O',
                      '--write', self.out('adf.csv')])
        self.assert_csv(self.out('adf.csv'), 'angle_deg,adf_O_Li_O')

    def test_savefig(self):
        # The plotting path is otherwise untouched by --write.
        cli.main_msd([self.traj_path, '--timestep', '1', '--t-unit', 'dt',
                      '-ts', '5', '-te', '20',
                      '--savefig', self.out('msd.png')])
        self.assertTrue(os.path.exists(self.out('msd.png')))

    def test_rdf_rejects_species_and_species_pairs(self):
        # --species comes from the shared parser and --species-pairs from
        # the rdf parser, so argparse cannot enforce this for us.
        with self.assertRaises(ValueError):
            cli.main_rdf([self.traj_path, '--timestep', '1', '-r', '4',
                          '--species', 'Li', '--species-pairs', 'Li-O'])


if __name__ == '__main__':
    unittest.main()
