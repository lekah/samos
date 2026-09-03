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
from argparse import ArgumentTypeError
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

    def test_rdf_method_defaults_to_auto(self):
        args = cli._parser_rdf().parse_args([self.traj_path])
        self.assertEqual(args.method, 'auto')
        for choice in ('auto', 'ortho', 'skew'):
            args = cli._parser_rdf().parse_args(
                [self.traj_path, '--method', choice])
            self.assertEqual(args.method, choice)
        with self.assertRaises(SystemExit):
            cli._parser_rdf().parse_args([self.traj_path, '--method', 'fast'])

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

    def test_rdf_method_reaches_the_analyzer(self):
        """The two algorithms agree, so the same CSV must come out
        whichever one the flag selects."""
        for choice in ('ortho', 'skew'):
            cli.main_rdf([self.traj_path, '--timestep', '1', '-r', '4',
                          '-b', '20', '--species-pairs', 'Li-O',
                          '--method', choice,
                          '--write', self.out(choice + '.csv')])
        with open(self.out('ortho.csv')) as f:
            ortho = f.read()
        with open(self.out('skew.csv')) as f:
            self.assertEqual(ortho, f.read())

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


class TestIndex(CLITestCase):
    """The -i/--index option and the slice parsing behind it."""

    def test_valid_slices(self):
        cases = {
            ':': slice(None, None, None),
            '::10': slice(None, None, 10),
            ':1000': slice(None, 1000, None),
            '100:': slice(100, None, None),
            '100:500': slice(100, 500, None),
            '500:1500:2': slice(500, 1500, 2),
            '-1000:': slice(-1000, None, None),
            ' 10 : 20 ': slice(10, 20, None),
        }
        for text, expected in cases.items():
            self.assertEqual(cli._parse_index(text), expected, text)

    def test_rejected_slices(self):
        # A bare frame number is refused rather than silently reducing
        # the trajectory to a single frame.
        for text in ('5', '1:2:3:4', 'a:b', '::0', '1.5:3'):
            with self.assertRaises(ArgumentTypeError, msg=text):
                cli._parse_index(text)

    def test_parser_converts_the_slice(self):
        args = cli._parser_rdf().parse_args(
            [self.traj_path, '-i', '10:90:2'])
        self.assertEqual(args.index, slice(10, 90, 2))
        self.assertIsNone(
            cli._parser_rdf().parse_args([self.traj_path]).index)

    def test_stride_widens_the_written_time_axis(self):
        # 120 frames at 1 fs; every second frame is 2 fs apart.
        for index, spacing, name in ((None, 1.0, 'full.csv'),
                                     ('::2', 2.0, 'strided.csv')):
            argv = [self.traj_path, '--timestep', '1', '--t-unit', 'dt',
                    '-ts', '5', '-te', '20', '--write', self.out(name)]
            if index is not None:
                argv += ['-i', index]
            cli.main_msd(argv)
            t = np.loadtxt(self.out(name), delimiter=',', skiprows=1)[:, 0]
            self.assertAlmostEqual(t[1] - t[0], spacing)

    def test_slice_changes_the_result(self):
        # Averaging over a third of the frames gives a different RDF;
        # identical output would mean --index never reached the analysis.
        common = ['--timestep', '1', '-r', '4', '-b', '20',
                  '--species-pairs', 'Li-O']
        cli.main_rdf([self.traj_path] + common
                     + ['--write', self.out('all.csv')])
        cli.main_rdf([self.traj_path, '-i', '0:40'] + common
                     + ['--write', self.out('part.csv')])
        full = np.loadtxt(self.out('all.csv'), delimiter=',', skiprows=1)
        part = np.loadtxt(self.out('part.csv'), delimiter=',', skiprows=1)
        np.testing.assert_allclose(full[:, 0], part[:, 0])
        self.assertFalse(np.allclose(full[:, 1], part[:, 1]))

    def test_warns_when_striding_and_deriving_velocities(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main_vdos([self.traj_path, '--timestep', '1',
                           '--compute-velocities', '-i', '::3',
                           '--write', self.out('vdos.csv')])
        self.assertIn('Warning', buf.getvalue())
        self.assertIn('every 3th frame', buf.getvalue())

    def test_no_warning_without_a_stride(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main_vdos([self.traj_path, '--timestep', '1',
                           '--compute-velocities', '-i', '0:100',
                           '--write', self.out('vdos.csv')])
        self.assertNotIn('Warning', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
