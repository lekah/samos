# -*- coding: utf-8 -*-
"""
Tests for samos.analysis.rdf.BondAnalyzer and ADF.

All tests use synthetic trajectories built in-memory; no external data
files are required.
"""

import os
import tempfile
import unittest

import numpy as np
from ase import Atoms

from samos.trajectory import Trajectory
from samos.analysis.rdf import ADF, BondAnalyzer, TorsionAnalyzer


def _make_traj(symbols, positions, cell):
    """
    Build a single-frame Trajectory.

    :param list symbols: chemical symbol per atom, e.g. ['Si', 'O', 'O']
    :param array-like positions: shape (n_atoms, 3), Cartesian
    :param array-like cell: shape (3, 3) or scalar for a cubic cell
    """
    if np.ndim(cell) == 0:
        cell = np.eye(3) * float(cell)
    atoms = Atoms(symbols, cell=cell, pbc=True)
    traj = Trajectory()
    traj.set_atoms(atoms)
    # set_positions expects shape (n_frames, n_atoms, 3)
    traj.set_positions(np.array(positions, dtype=float)[np.newaxis])
    return traj


def _bare_ba(traj=None):
    """
    Return a BondAnalyzer (via ADF) with no bonds set.
    BondAnalyzer is abstract (inherits run from BaseAnalyzer); ADF is the
    concrete subclass used here to exercise the shared infrastructure.
    """
    ba = ADF.__new__(ADF)
    ba._bonds = None
    ba._trajectory = traj
    return ba


# ---------------------------------------------------------------------------
# BondAnalyzer -- static helpers (no trajectory needed)
# ---------------------------------------------------------------------------

class TestPbcWrap(unittest.TestCase):

    def setUp(self):
        self.cell = np.eye(3) * 10.0
        self.cellI = np.linalg.inv(self.cell)

    def test_no_wrap_needed(self):
        diff = np.array([[3.0, 0.0, 0.0]])
        result = BondAnalyzer._pbc_wrap(diff, self.cellI, self.cell)
        np.testing.assert_allclose(result, [[3.0, 0.0, 0.0]])

    def test_wraps_large_positive(self):
        # 9.0 A in a 10 A cell -> minimum image is -1.0 A
        diff = np.array([[9.0, 0.0, 0.0]])
        result = BondAnalyzer._pbc_wrap(diff, self.cellI, self.cell)
        np.testing.assert_allclose(result, [[-1.0, 0.0, 0.0]])

    def test_exactly_half_not_wrapped(self):
        # frac = 0.5 is NOT > 0.5, so no wrap applied
        diff = np.array([[5.0, 0.0, 0.0]])
        result = BondAnalyzer._pbc_wrap(diff, self.cellI, self.cell)
        np.testing.assert_allclose(result, [[5.0, 0.0, 0.0]])

    def test_multiple_vectors(self):
        diff = np.array([[9.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        result = BondAnalyzer._pbc_wrap(diff, self.cellI, self.cell)
        np.testing.assert_allclose(
            result, [[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])


class TestSetBonds(unittest.TestCase):

    def test_canonical_ordering(self):
        ba = _bare_ba()
        ba.set_bonds([[3, 1], [5, 2]])
        np.testing.assert_array_equal(ba._bonds, [[1, 3], [2, 5]])

    def test_deduplication(self):
        ba = _bare_ba()
        ba.set_bonds([[1, 2], [2, 1], [1, 2]])
        self.assertEqual(len(ba._bonds), 1)
        np.testing.assert_array_equal(ba._bonds, [[1, 2]])

    def test_empty_array(self):
        ba = _bare_ba()
        ba.set_bonds(np.empty((0, 2), dtype=int))
        self.assertEqual(ba._bonds.shape, (0, 2))


class TestParseCutoffs(unittest.TestCase):

    def test_normalizes_string_key(self):
        result = BondAnalyzer._parse_cutoffs({'Si-O': (1.4, 2.0)})
        self.assertIn(('Si', 'O'), result)
        self.assertEqual(result[('Si', 'O')], (1.4, 2.0))

    def test_bad_key_raises(self):
        with self.assertRaises(ValueError):
            BondAnalyzer._parse_cutoffs({'SiO': (1.4, 2.0)})

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            BondAnalyzer._parse_cutoffs({'Si-O': (1.4,)})


class TestLookupCutoff(unittest.TestCase):

    def setUp(self):
        self.parsed = {('Si', 'O'): (1.4, 2.0)}

    def test_forward_order(self):
        r = BondAnalyzer._lookup_cutoff(self.parsed, 'Si', 'O')
        self.assertEqual(r, (1.4, 2.0))

    def test_reverse_order(self):
        r = BondAnalyzer._lookup_cutoff(self.parsed, 'O', 'Si')
        self.assertEqual(r, (1.4, 2.0))

    def test_missing_raises(self):
        with self.assertRaises(ValueError):
            BondAnalyzer._lookup_cutoff(self.parsed, 'Al', 'O')


class TestBuildAdjacency(unittest.TestCase):

    def test_bidirectional(self):
        bonds = np.array([[0, 1], [1, 2]])
        adj = BondAnalyzer._build_adjacency(bonds)
        self.assertIn(1, adj[0])
        self.assertIn(0, adj[1])
        self.assertIn(2, adj[1])
        self.assertIn(1, adj[2])

    def test_empty_bonds(self):
        adj = BondAnalyzer._build_adjacency(np.empty((0, 2), dtype=int))
        self.assertEqual(len(adj), 0)


# ---------------------------------------------------------------------------
# BondAnalyzer -- methods that require a Trajectory
# ---------------------------------------------------------------------------

class TestDetectBonds(unittest.TestCase):
    """
    3-atom system: O at x=0.5, Si at x=2.0, O at x=3.5 (cell 10 A).
    Si-O bond length = 1.5 A.
    """

    def setUp(self):
        self.traj = _make_traj(
            ['O', 'Si', 'O'],
            [[0.5, 5.0, 5.0],
             [2.0, 5.0, 5.0],
             [3.5, 5.0, 5.0]],
            10.0)

    def _ba(self):
        return _bare_ba(self.traj)

    def test_finds_both_bonds_within_cutoff(self):
        bonds = self._ba()._detect_bonds({'Si-O': (1.0, 2.0)}, frame=0)
        self.assertEqual(len(bonds), 2)

    def test_misses_bonds_outside_cutoff(self):
        # cutoff too small to reach either O
        bonds = self._ba()._detect_bonds({'Si-O': (2.0, 3.0)}, frame=0)
        self.assertEqual(len(bonds), 0)

    def test_pbc_bond_detected(self):
        # Si at x=0.5, O at x=9.5 -- distance 1.0 A via PBC (cell=10)
        traj = _make_traj(
            ['Si', 'O'],
            [[0.5, 5.0, 5.0],
             [9.5, 5.0, 5.0]],
            10.0)
        bonds = _bare_ba(traj)._detect_bonds(
            {'Si-O': (0.5, 1.5)}, frame=0)
        self.assertEqual(len(bonds), 1)

    def test_canonical_bond_ordering(self):
        bonds = self._ba()._detect_bonds({'Si-O': (1.0, 2.0)}, frame=0)
        for b in bonds:
            self.assertLessEqual(b[0], b[1])


class TestGetBonds(unittest.TestCase):

    def test_stored_bonds_take_priority(self):
        ba = _bare_ba()
        ba.set_bonds([[0, 1]])
        # Even with valid cutoffs, stored bonds are returned unchanged.
        result = ba.get_bonds(cutoffs={'X-Y': (1.0, 2.0)}, frame=0)
        np.testing.assert_array_equal(result, [[0, 1]])

    def test_no_source_raises(self):
        with self.assertRaises(ValueError):
            _bare_ba().get_bonds()

    def test_cutoffs_without_frame_raises(self):
        with self.assertRaises(ValueError):
            _bare_ba().get_bonds(cutoffs={'Si-O': (1.0, 2.0)})


# ---------------------------------------------------------------------------
# BondAnalyzer -- LAMMPS data file reader
# ---------------------------------------------------------------------------

_LAMMPS_DATA = """\
LAMMPS data file

3 atoms
2 bonds

Masses

1 28.086
2 16.000

Atoms

1 1 0.0 0.0 0.0
2 2 1.0 0.0 0.0
3 2 2.0 0.0 0.0

Bonds

1 1 2 1
2 1 2 3
"""


class TestLoadBondsLammps(unittest.TestCase):

    def setUp(self):
        fh = tempfile.NamedTemporaryFile(
            mode='w', suffix='.lammps', delete=False)
        fh.write(_LAMMPS_DATA)
        fh.close()
        self.path = fh.name

    def tearDown(self):
        os.unlink(self.path)

    def _load(self):
        ba = _bare_ba()
        ba.load_bonds_lammps(self.path)
        return ba

    def test_correct_bond_count(self):
        self.assertEqual(len(self._load()._bonds), 2)

    def test_zero_based_indices(self):
        # File has atoms 2-1 and 2-3 (1-based) -> (0,1) and (1,2) (0-based)
        bonds = [tuple(b) for b in self._load()._bonds]
        self.assertIn((0, 1), bonds)
        self.assertIn((1, 2), bonds)

    def test_canonical_ordering(self):
        for b in self._load()._bonds:
            self.assertLessEqual(b[0], b[1])


# ---------------------------------------------------------------------------
# ADF
# ---------------------------------------------------------------------------

class TestADFInputValidation(unittest.TestCase):

    def _adf(self):
        traj = _make_traj(
            ['Si', 'O', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            20.0)
        return ADF(trajectory=traj)

    def test_centers_and_triplets_mutually_exclusive(self):
        a = self._adf()
        with self.assertRaises(ValueError):
            a.run(centers=['Si'],
                  species_triplets=[('O', 'Si', 'O')],
                  bonds={'Si-O': (0.5, 1.5)})

    def test_no_topology_raises(self):
        a = self._adf()
        with self.assertRaises(ValueError):
            a.run(species_triplets=[('O', 'Si', 'O')])

    def test_torsion_not_implemented(self):
        t = TorsionAnalyzer.__new__(TorsionAnalyzer)
        with self.assertRaises(NotImplementedError):
            t.run()


class TestADFAngles(unittest.TestCase):
    """
    Known-geometry tests: angle values are analytically determined.
    nbins=180 gives 1-degree bins; bin centre of bin k is (k+0.5) degrees.
    """

    def _run(self, symbols, positions, cutoffs, triplet, cell=20.0):
        traj = _make_traj(symbols, positions, cell)
        a = ADF(trajectory=traj)
        return a.run(species_triplets=[triplet],
                     bonds=cutoffs, nbins=180)

    def test_linear_angle(self):
        # O-Si-O collinear along x -> angle = 180 degrees
        res = self._run(
            ['O', 'Si', 'O'],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            {'Si-O': (1.0, 2.0)},
            ('O', 'Si', 'O'))
        adf = res.get_array('adf_O_Si_O')
        angles = res.get_array('angles_O_Si_O')
        self.assertAlmostEqual(angles[np.argmax(adf)], 179.5, places=0)

    def test_right_angle(self):
        # O-Si-O perpendicular -> angle = 90 degrees
        res = self._run(
            ['O', 'Si', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            {'Si-O': (0.5, 1.5)},
            ('O', 'Si', 'O'))
        adf = res.get_array('adf_O_Si_O')
        angles = res.get_array('angles_O_Si_O')
        self.assertAlmostEqual(angles[np.argmax(adf)], 90.0, delta=1.0)

    def test_same_species_no_double_counting(self):
        # 3 O neighbors at 90 degrees to each other -> C(3,2) = 3 angles.
        # Normalization: n_center=1, n_frames=1, binsize=1 -> adf.sum() = 3.
        traj = _make_traj(
            ['Si', 'O', 'O', 'O'],
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            20.0)
        a = ADF(trajectory=traj)
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (0.5, 1.5)}, nbins=180)
        # adf.sum() * binsize = total_angles / n_center / n_frames
        total = res.get_array('adf_O_Si_O').sum() * 1.0  # binsize = 1 deg
        self.assertAlmostEqual(total, 3.0, places=5)

    def test_pbc_angle(self):
        # Si at (1,5,5); O1 at (0.1,5,5) [r=0.9]; O2 at (9.9,5,5) [r=1.1 via PBC].
        # Both bond vectors point in the -x direction -> angle near 0 degrees.
        res = self._run(
            ['Si', 'O', 'O'],
            [[1.0, 5.0, 5.0], [0.1, 5.0, 5.0], [9.9, 5.0, 5.0]],
            {'Si-O': (0.5, 1.5)},
            ('O', 'Si', 'O'),
            cell=10.0)
        adf = res.get_array('adf_O_Si_O')
        angles = res.get_array('angles_O_Si_O')
        self.assertLess(angles[np.argmax(adf)], 10.0)


class TestADFOutputFormat(unittest.TestCase):

    def setUp(self):
        self.traj = _make_traj(
            ['O', 'Si', 'O'],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
            20.0)

    def test_array_keys_present(self):
        a = ADF(trajectory=self.traj)
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (1.0, 2.0)}, nbins=90)
        names = res.get_arraynames()
        self.assertIn('adf_O_Si_O', names)
        self.assertIn('angles_O_Si_O', names)

    def test_array_shapes(self):
        a = ADF(trajectory=self.traj)
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (1.0, 2.0)}, nbins=90)
        self.assertEqual(res.get_array('adf_O_Si_O').shape, (90,))
        self.assertEqual(res.get_array('angles_O_Si_O').shape, (90,))

    def test_bin_centres(self):
        # nbins=36 -> binsize=5; centres = 2.5, 7.5, ..., 177.5
        a = ADF(trajectory=self.traj)
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (1.0, 2.0)}, nbins=36)
        angles = res.get_array('angles_O_Si_O')
        self.assertAlmostEqual(angles[0], 2.5)
        self.assertAlmostEqual(angles[-1], 177.5)

    def test_species_triplets_attr(self):
        a = ADF(trajectory=self.traj)
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (1.0, 2.0)}, nbins=36)
        self.assertIn(('O', 'Si', 'O'), res.get_attr('species_triplets'))


class TestADFStaticBonds(unittest.TestCase):
    """
    Two-frame trajectory: atoms bonded in frame 0, out of range in frame 1.
    static_bonds=True freezes topology from frame 0 and applies it to both
    frames, doubling the angle count relative to dynamic detection.
    """

    def setUp(self):
        cell = np.eye(3) * 20.0
        atoms = Atoms(['Si', 'O', 'O'], cell=cell, pbc=True)
        pos = np.array([
            # frame 0: Si-O bonds at 1.0 A
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            # frame 1: atoms far apart, outside cutoff
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        ], dtype=float)
        traj = Trajectory()
        traj.set_atoms(atoms)
        traj.set_positions(pos)
        self.traj = traj

    def _run(self, static):
        a = ADF(trajectory=self.traj)
        return a.run(species_triplets=[('O', 'Si', 'O')],
                     bonds={'Si-O': (0.5, 1.5)},
                     static_bonds=static, nbins=180)

    def test_static_finds_angles_in_both_frames(self):
        adf = self._run(static=True).get_array('adf_O_Si_O')
        self.assertGreater(adf.sum(), 0.0)

    def test_static_doubles_count_vs_dynamic(self):
        # static: topology applied to both frames -> 2 counts, norm=2 -> 1.0
        # dynamic: frame 1 has no bonds -> 1 count, norm=2 -> 0.5
        # ratio = 2.0
        s = self._run(static=True).get_array('adf_O_Si_O').sum()
        d = self._run(static=False).get_array('adf_O_Si_O').sum()
        self.assertAlmostEqual(s, 2.0 * d, places=5)


class TestADFCentersExpansion(unittest.TestCase):

    def test_centers_produces_results(self):
        # centers=['Si'] should expand to all (*,Si,*) triplets;
        # with one O species present that is just ('O','Si','O').
        traj = _make_traj(
            ['Si', 'O', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            20.0)
        a = ADF(trajectory=traj)
        res = a.run(centers=['Si'], bonds={'Si-O': (0.5, 1.5)}, nbins=90)
        self.assertIn('adf_O_Si_O', res.get_arraynames())

    def test_centers_and_triplets_give_same_result(self):
        traj = _make_traj(
            ['Si', 'O', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            20.0)
        a1 = ADF(trajectory=traj)
        a2 = ADF(trajectory=traj)
        r1 = a1.run(centers=['Si'],
                    bonds={'Si-O': (0.5, 1.5)}, nbins=90)
        r2 = a2.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (0.5, 1.5)}, nbins=90)
        np.testing.assert_array_equal(
            r1.get_array('adf_O_Si_O'),
            r2.get_array('adf_O_Si_O'))


class TestADFExplicitBonds(unittest.TestCase):

    def test_set_bonds_used_over_cutoffs(self):
        # Provide explicit bonds that differ from what cutoffs would find.
        # Explicit bonds: only bond (0,1), skipping (0,2).
        # Angle triplet (O,Si,O) needs 2 neighbors -> no angle possible.
        traj = _make_traj(
            ['Si', 'O', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            20.0)
        a = ADF(trajectory=traj)
        a.set_bonds([[0, 1]])  # only one Si-O bond
        res = a.run(species_triplets=[('O', 'Si', 'O')],
                    bonds={'Si-O': (0.5, 1.5)},  # ignored
                    nbins=180)
        # Only 1 neighbor -> no angle can be formed
        self.assertAlmostEqual(
            res.get_array('adf_O_Si_O').sum(), 0.0)


class TestADFEmptyFrameSelection(unittest.TestCase):
    """An empty frame selection used to reach frames[0] (IndexError) with
    static_bonds, and to divide by n_frames=0 without it."""

    def _analyzer(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        from samos.analysis.rdf import ADF
        atoms = Atoms('OSiO', cell=np.eye(3) * 10., pbc=True)
        pos = np.tile(np.array([[0., 0., 0.],
                                [1.6, 0., 0.],
                                [3.2, 0., 0.]]), (5, 1, 1))
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(pos)
        return ADF(trajectory=t)

    def test_empty_frame_range_raises(self):
        adf = self._analyzer()
        with self.assertRaises(ValueError) as cm:
            adf.run(istart=5, istop=5, bonds={'Si-O': (1.0, 2.0)})
        self.assertIn('No frames selected', str(cm.exception))

    def test_empty_frame_range_raises_with_static_bonds(self):
        adf = self._analyzer()
        with self.assertRaises(ValueError):
            adf.run(istart=10, istop=None, bonds={'Si-O': (1.0, 2.0)},
                    static_bonds=True)


class TestPlotRDFColors(unittest.TestCase):
    """plot_rdf's second-axis colour selection had `if 'color': pass`
    followed by an independent `if/else`, so an explicitly supplied
    colour was overwritten by the g(r) line's colour."""

    def _result(self):
        import numpy as np
        from samos.utils.attributed_array import AttributedArray
        res = AttributedArray()
        res.set_attr('species_pairs', [('Li', 'O')])
        res.set_array('rdf_Li_O', np.linspace(0., 2., 10))
        res.set_array('int_Li_O', np.linspace(0., 5., 10))
        res.set_array('radii_Li_O', np.linspace(0., 5., 10))
        return res

    def _axes(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        return fig.add_subplot(1, 1, 1)

    def test_explicit_integral_color_is_kept(self):
        from samos.plotting.plot_rdf import plot_rdf
        handles = plot_rdf(self._result(), ax=self._axes(),
                           plot_params={'color': 'red'},
                           plot_params2={'color': 'black'})
        rdf_line, int_line = handles
        self.assertEqual(rdf_line.get_color(), 'red')
        self.assertEqual(int_line.get_color(), 'black')

    def test_integral_defaults_to_the_rdf_color(self):
        from samos.plotting.plot_rdf import plot_rdf
        handles = plot_rdf(self._result(), ax=self._axes(),
                           plot_params={'color': 'red'})
        rdf_line, int_line = handles
        self.assertEqual(int_line.get_color(), rdf_line.get_color())


class TestRDFValidationAndAttributes(unittest.TestCase):
    """RDF.run had an impossible default (radius=None), rejected the
    istop its own default resolves to, crashed opaquely on a species
    that is absent, and wrote a cross-pair running minimum under a
    per-pair-looking attribute name."""

    def _trajectory(self, seed=3, nstep=8):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        rng = np.random.default_rng(seed)
        atoms = Atoms('H2O2', cell=np.eye(3) * 10., pbc=True)
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(rng.random((nstep, 4, 3)) * 10.)
        return t

    def _rdf(self):
        from samos.analysis.rdf import RDF
        return RDF(trajectory=self._trajectory())

    def test_radius_is_required(self):
        with self.assertRaises(TypeError):
            self._rdf().run()

    def test_istop_equal_to_length_is_accepted(self):
        """len(positions) is exactly what istop=None resolves to, so it
        must not be rejected."""
        res = self._rdf().run(radius=4.0, istop=8)
        self.assertIn('rdf_H_H', res.get_arraynames())

    def test_istop_beyond_length_still_rejected(self):
        with self.assertRaises(ValueError):
            self._rdf().run(radius=4.0, istop=9)

    def test_absent_species_is_skipped_not_crashed(self):
        res = self._rdf().run(radius=4.0, species_pairs=[('H', 'N'),
                                                         ('H', 'O')])
        # the impossible pair is dropped, the valid one survives
        self.assertNotIn('rdf_H_N', res.get_arraynames())
        self.assertIn('rdf_H_O', res.get_arraynames())

    def test_shortest_distance_is_per_pair_and_global(self):
        import numpy as np
        res = self._rdf().run(radius=4.0)
        per_pair = [res.get_attr('shortest_distance_{}'.format(lbl))
                    for lbl in ('H_H', 'H_O', 'O_O')]
        self.assertTrue(all(np.isfinite(d) for d in per_pair))
        # the global value is the minimum over every pair, not the
        # running minimum at whichever pair happened to be last
        self.assertAlmostEqual(res.get_attr('shortest_distance'),
                               min(per_pair))


class TestSpeciesPairHelper(unittest.TestCase):
    """Both call sites built these pairs from the per-atom symbol list,
    so each pair appeared once per atom."""

    def _trajectory(self):
        import numpy as np
        from ase import Atoms
        from samos.trajectory import Trajectory
        # 6 atoms but only 3 species
        atoms = Atoms('H3OFe2', cell=np.eye(3) * 10., pbc=True)
        t = Trajectory(atoms=atoms, timestep=1.)
        t.set_positions(np.zeros((2, 6, 3)))
        return t

    def test_pairs_are_deduplicated(self):
        from samos.analysis.rdf import pairs_with_other_species
        pairs = pairs_with_other_species(self._trajectory(), ['H'])
        self.assertEqual(pairs, [('H', 'Fe'), ('H', 'O')])

    def test_requested_species_excluded_from_others(self):
        from samos.analysis.rdf import pairs_with_other_species
        pairs = pairs_with_other_species(self._trajectory(), ['H', 'O'])
        self.assertEqual(sorted(pairs),
                         [('H', 'Fe'), ('O', 'Fe')])


class TestWritersLeaveStdoutOpen(unittest.TestCase):
    """write_xsf / write_grid / write_xsf_header default to
    outfilename=None, i.e. sys.stdout, and then closed it."""

    def _capture(self, func, *args, **kwargs):
        import io
        import sys
        buf = io.StringIO()
        real = sys.stdout
        sys.stdout = buf
        try:
            func(*args, **kwargs)
        finally:
            closed = sys.stdout.closed
            sys.stdout = real
        return buf, closed

    def _grid(self):
        import numpy as np
        return np.arange(8.).reshape(2, 2, 2)

    def test_write_xsf_leaves_stdout_open(self):
        import numpy as np
        from samos.io.xsf import write_xsf
        buf, closed = self._capture(
            write_xsf, ['H'], [[0., 0., 0.]], np.eye(3), self._grid())
        self.assertFalse(closed)
        self.assertIn('BEGIN_BLOCK_DATAGRID_3D', buf.getvalue())
        # the embedded header literal must stay flush left
        self.assertIn('\n3D_PWSCF\n', buf.getvalue())

    def test_write_grid_leaves_stdout_open(self):
        from samos.io.xsf import write_grid
        _, closed = self._capture(write_grid, self._grid())
        self.assertFalse(closed)

    def test_bad_outfilename_type_rejected(self):
        import numpy as np
        from samos.io.xsf import write_xsf
        with self.assertRaises(TypeError):
            write_xsf(['H'], [[0., 0., 0.]], np.eye(3), self._grid(),
                      outfilename=42)

    def test_named_file_is_closed(self):
        import numpy as np
        import tempfile
        import os
        from samos.io.xsf import write_grid
        fd, path = tempfile.mkstemp(suffix='.xsf')
        os.close(fd)
        self.addCleanup(os.unlink, path)
        write_grid(self._grid(), outfilename=path)
        with open(path) as fh:
            self.assertTrue(fh.read().strip())


class TestBaseAnalyzerWithoutTrajectory(unittest.TestCase):
    """BaseAnalyzer never initialised _trajectory, so calling run()
    before set_trajectory() raised AttributeError from deep inside."""

    def test_rdf_without_trajectory(self):
        from samos.analysis.rdf import RDF
        with self.assertRaises(ValueError) as cm:
            RDF().run(radius=4.0)
        self.assertIn('set_trajectory', str(cm.exception))

    def test_adf_without_trajectory(self):
        from samos.analysis.rdf import ADF
        with self.assertRaises(ValueError) as cm:
            ADF().run(bonds={'Si-O': (1.0, 2.0)})
        self.assertIn('set_trajectory', str(cm.exception))

    def test_set_trajectory_type_checked(self):
        from samos.analysis.rdf import RDF
        with self.assertRaises(TypeError):
            RDF(trajectory='not a trajectory')


if __name__ == '__main__':
    unittest.main()
