# -*- coding: utf-8 -*-
"""
Tests for samos.analysis.rdf.RDF.

The normalisation is the subject here: g(r) is defined as the mean of
the per-frame g(r), and the ideal-gas reference an atom is compared
against excludes the atom itself.
"""

import io
import unittest
import warnings
from contextlib import redirect_stdout

import numpy as np
from ase import Atoms

from samos.trajectory import Trajectory
from samos.analysis.rdf import RDF, MinimumImage


def _gas(nstep, symbols, length, seed=0, cells=None):
    """
    Build an ideal gas: positions drawn uniformly in the cell, so that
    g(r) is 1 everywhere and any deviation is a normalisation error.

    :param int nstep: Number of frames.
    :param str symbols: Chemical formula, e.g. ``'H100He100'``.
    :param float length: Edge of the cubic cell, in Angstrom.
    :param cells: Optional (nstep, 3, 3) array for a variable cell; the
        positions are then drawn in each frame's own cell.
    """
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=np.eye(3) * length, pbc=True)
    traj = Trajectory(atoms=atoms)
    frac = rng.random((nstep, len(atoms), 3))
    if cells is None:
        traj.set_positions(frac * length)
    else:
        traj.set_positions(np.einsum('sac,scd->sad', frac, cells))
        traj.set_cells(cells)
    return traj


def _breathing_cells(nstep, first, last):
    """Cubic cells whose edge grows linearly from *first* to *last*."""
    return np.array([np.eye(3) * length
                     for length in np.linspace(first, last, nstep)])


def _plateau(res, label, rmin):
    """Mean of g(r) beyond *rmin*, which should be 1 for an ideal gas."""
    radii = res.get_array('radii_{}'.format(label))
    return res.get_array('rdf_{}'.format(label))[radii > rmin].mean()


class TestVariableCellNormalisation(unittest.TestCase):
    """
    The density used to come from whichever frame happened to be
    sampled last, so reversing the frame order of an NPT trajectory
    rescaled g(r) by the ratio of the two end volumes.
    """

    def _traj(self, order):
        cells = _breathing_cells(40, 12.0, 18.0)
        traj = _gas(40, 'H30He30', 12.0, cells=cells)
        sliced = traj.slice_steps(order)
        return RDF(trajectory=sliced, verbosity=0).run(radius=5.0, nbins=50)

    def test_frame_order_does_not_change_the_rdf(self):
        forward = self._traj(slice(None))
        reverse = self._traj(slice(None, None, -1))
        for label in ('H_He', 'H_H', 'He_He'):
            np.testing.assert_allclose(
                forward.get_array('rdf_{}'.format(label)),
                reverse.get_array('rdf_{}'.format(label)),
                err_msg=label)

    def test_result_is_the_mean_of_the_per_frame_rdfs(self):
        # Two frames whose volumes differ by a factor of eight.  This is
        # what picks option C out of the alternatives: normalising the
        # summed histogram by the mean volume, or by the mean density,
        # gives a different answer here.
        cells = _breathing_cells(2, 10.0, 20.0)
        traj = _gas(2, 'H20He20', 10.0, cells=cells)
        both = RDF(trajectory=traj, verbosity=0).run(radius=4.0, nbins=40)
        singles = [
            RDF(trajectory=traj.slice_steps(slice(i, i + 1)), verbosity=0).run(
                radius=4.0, nbins=40)
            for i in (0, 1)
        ]
        np.testing.assert_allclose(
            both.get_array('rdf_H_He'),
            0.5 * (singles[0].get_array('rdf_H_He')
                   + singles[1].get_array('rdf_H_He')))


class TestIdealGasReference(unittest.TestCase):
    """
    An atom is not its own neighbour, so a species paired with itself
    has to be normalised by N-1.  Using N sent g(r) to (N-1)/N at large
    r -- a 1 % error for 100 atoms and 11 % for 10.
    """

    RADIUS, NBINS, LENGTH = 6.0, 60, 20.0

    def _reference(self, res, label):
        """
        Recover the ideal-gas neighbour count the normalisation used.

        ``rdf = counts / (4 pi r^2 dr) / (n_ideal / V)`` and ``counts``
        is the bin-by-bin difference of the running integral, so
        n_ideal follows exactly.  Reading it back is deterministic,
        unlike watching where g(r) plateaus, which needs far more
        sampling than a unit test can afford to resolve 1 %.
        """
        radii = res.get_array('radii_{}'.format(label))
        rdf = res.get_array('rdf_{}'.format(label))
        integral = res.get_array('int_{}'.format(label))
        counts = np.diff(np.concatenate(([0.0], integral)))
        binsize = self.RADIUS / self.NBINS
        shell = 4.0 * np.pi * radii**2 * binsize
        occupied = rdf > 0
        n_ideal = (counts[occupied] * self.LENGTH**3
                   / (shell[occupied] * rdf[occupied]))
        # One constant, whatever the bin.
        np.testing.assert_allclose(n_ideal, n_ideal[0])
        return n_ideal[0]

    def _run(self, symbols, **kwargs):
        traj = _gas(20, symbols, self.LENGTH)
        return RDF(trajectory=traj, verbosity=0).run(
            radius=self.RADIUS, nbins=self.NBINS, **kwargs)

    def test_like_pair_excludes_the_atom_itself(self):
        res = self._run('H100')
        self.assertAlmostEqual(self._reference(res, 'H_H'), 99.0, places=6)

    def test_unlike_pair_counts_every_atom(self):
        res = self._run('H100He80')
        self.assertAlmostEqual(self._reference(res, 'H_He'), 80.0, places=6)

    def test_overlapping_species_groups(self):
        # ind1 is H and He together, ind2 is He alone: they overlap
        # without being equal, so the reference is neither 100 nor 99
        # but 100 - 100/200, the chance that the centre atom is itself
        # one of the He atoms.
        res = self._run('H100He100',
                        species_pairs=[(('H', 'He'), 'He')])
        label = [name[len('rdf_'):] for name in res.get_arraynames()
                 if name.startswith('rdf_')][0]
        self.assertAlmostEqual(self._reference(res, label), 99.5, places=6)

    def test_ideal_gas_plateau_is_one(self):
        # The physics the above adds up to, at the sampling a unit test
        # can afford: loose, and only a smoke test.
        res = RDF(trajectory=_gas(60, 'H100He100', self.LENGTH),
                  verbosity=0).run(radius=self.RADIUS, nbins=self.NBINS)
        for label in ('H_H', 'H_He'):
            self.assertAlmostEqual(_plateau(res, label, 4.0), 1.0,
                                   delta=0.02)


class TestRunningIntegral(unittest.TestCase):
    """
    The int_* arrays are neighbour counts and carry no volume factor,
    so neither the per-frame normalisation nor the N-1 reference
    touches them.
    """

    def test_integral_counts_ideal_gas_neighbours(self):
        n, length = 100, 20.0
        res = RDF(trajectory=_gas(60, 'H{}'.format(n), length),
                  verbosity=0).run(radius=6.0, nbins=60)
        radii = res.get_array('radii_H_H')
        integral = res.get_array('int_H_H')
        # <n(r)> = rho * 4/3 pi r^3 with rho = (N - 1) / V.  The sum
        # includes every bin whole, so it is the count inside the outer
        # edge of the bin, half a binsize beyond the centre in radii.
        binsize = 6.0 / 60
        edges = radii + 0.5 * binsize
        expected = ((n - 1) / length**3) * (4. / 3.) * np.pi * edges**3
        np.testing.assert_allclose(integral[radii > 3], expected[radii > 3],
                                   rtol=0.03)


class TestMinimumImageInRDF(unittest.TestCase):
    """RDF.run used to test only the eight corners of the cell, which
    misses the nearest image when the basis is not reduced."""

    # The same simple cubic lattice, described twice.  Shearing b by
    # three lattice vectors changes no atom position and no volume, so
    # every physical result has to come out identical.
    CUBIC = np.eye(3) * 6.0
    SHEARED = np.array([[6., 0., 0.], [18., 6., 0.], [0., 0., 6.]])

    def _traj(self, cell, nstep=10, seed=12):
        rng = np.random.default_rng(seed)
        atoms = Atoms('H20He20', cell=self.CUBIC, pbc=True)
        positions = rng.random((nstep, len(atoms), 3)) @ self.CUBIC
        atoms.set_cell(cell)
        traj = Trajectory(atoms=atoms)
        traj.set_positions(positions)
        return traj

    def test_rdf_is_independent_of_the_cell_basis(self):
        """With the eight-corner scheme the sheared basis lost about a
        sixth of all pairs, because their nearest image lay further
        than one cell vector away."""
        # 2.9 A is inside max_radius (3.0 A) for both, so neither run
        # trips the radius guard.
        cubic = RDF(trajectory=self._traj(self.CUBIC), verbosity=0).run(
            radius=2.9, nbins=30)
        sheared = RDF(trajectory=self._traj(self.SHEARED), verbosity=0).run(
            radius=2.9, nbins=30)
        for name in cubic.get_arraynames():
            np.testing.assert_allclose(
                sheared.get_array(name), cubic.get_array(name),
                err_msg=name)

    def test_the_two_bases_really_are_the_same_lattice(self):
        """Guards the fixture itself: if the shear stopped being a
        lattice-preserving one, the test above would prove nothing."""
        self.assertAlmostEqual(np.linalg.det(self.CUBIC),
                               np.linalg.det(self.SHEARED), places=9)
        # Every sheared vector is an integer combination of cubic ones.
        coeffs = self.SHEARED @ np.linalg.inv(self.CUBIC)
        np.testing.assert_allclose(coeffs, np.round(coeffs), atol=1e-12)


class TestRadiusGuard(unittest.TestCase):
    """Past half the shortest lattice vector a pair has more than one
    image in range and only the nearest is counted, so g(r) is biased
    low.  That used to happen silently."""

    def _traj(self, length=10.0, nstep=5, seed=1):
        return _gas(nstep, 'H8He8', length, seed=seed)

    def test_radius_within_the_cell_is_quiet(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RDF(trajectory=self._traj(), verbosity=0).run(radius=4.0, nbins=20)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_radius_beyond_half_the_lattice_vector_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RDF(trajectory=self._traj(), verbosity=0).run(radius=6.0, nbins=20)
        self.assertEqual(len(caught), 1)
        self.assertIn('biased low', str(caught[0].message))

    def test_warning_is_emitted_once_per_run(self):
        """The check runs per frame for a variable cell; the user
        should not get one warning per frame."""
        cells = _breathing_cells(20, 10.0, 12.0)
        traj = _gas(20, 'H8He8', 10.0, cells=cells)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RDF(trajectory=traj, verbosity=0).run(radius=6.0, nbins=20)
        self.assertEqual(len(caught), 1)

    def test_limit_is_the_lattice_not_the_supplied_basis(self):
        """An fcc primitive cell of a=8 has 4.6 A perpendicular widths
        but 5.66 A lattice vectors, so the limit is 2.83 A."""
        cell = 8.0*np.array([[0., .5, .5], [.5, 0., .5], [.5, .5, 0.]])
        self.assertAlmostEqual(MinimumImage(cell).max_radius,
                               0.5 * 8.0 / np.sqrt(2.), places=10)


def _gas_in_cell(nstep, symbols, cell, seed=0, cells=None):
    """
    Ideal gas in an arbitrary *cell*, given as a 3x3 matrix or as three
    edge lengths.  Positions are drawn uniformly in fractional
    coordinates, so they fill whatever cell shape is asked for.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape == (3,):
        cell = np.diag(cell)
    rng = np.random.default_rng(seed)
    atoms = Atoms(symbols, cell=cell, pbc=True)
    traj = Trajectory(atoms=atoms)
    frac = rng.random((nstep, len(atoms), 3))
    if cells is None:
        traj.set_positions(np.einsum('sac,cd->sad', frac, cell))
    else:
        traj.set_positions(np.einsum('sac,scd->sad', frac, cells))
        traj.set_cells(cells)
    return traj


class TestOrthoSkewAgreement(unittest.TestCase):
    """
    The ortho algorithm (a periodic k-d tree) and the skew one (every
    pair against all 27 neighbour images) compute the same quantity:
    the distance to the nearest periodic image.  Everything the two
    produce therefore has to match, not just the histogram, so these
    tests compare every array and every attribute.
    """

    TRICLINIC = np.array([[10., 0., 0.], [3., 10., 0.], [2., 4., 10.]])

    def _assert_same_result(self, traj, **kwargs):
        """Run both algorithms and compare everything they return."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ortho = RDF(trajectory=traj, verbosity=0).run(
                method='ortho', **kwargs)
            skew = RDF(trajectory=traj, verbosity=0).run(
                method='skew', **kwargs)

        self.assertEqual(sorted(ortho.get_arraynames()),
                         sorted(skew.get_arraynames()))
        self.assertEqual(sorted(ortho.get_attrs()), sorted(skew.get_attrs()))
        # An empty result would pass every comparison below.
        self.assertTrue(ortho.get_arraynames())

        for name in ortho.get_arraynames():
            np.testing.assert_allclose(
                ortho.get_array(name), skew.get_array(name),
                rtol=1e-12, atol=0.0,
                err_msg='array {} differs'.format(name))
        for name, value in ortho.get_attrs().items():
            other = skew.get_attr(name)
            if isinstance(value, float):
                self.assertAlmostEqual(
                    value, other, places=10,
                    msg='attribute {} differs'.format(name))
            else:
                self.assertEqual(value, other,
                                 'attribute {} differs'.format(name))
        return ortho

    def test_cubic_cell(self):
        """Three species, so the pairs cover a species with itself, two
        different species, and a species that is in both lists."""
        traj = _gas_in_cell(15, 'H20He20Li10', [12.0] * 3)
        self._assert_same_result(traj, radius=5.0)

    def test_orthorhombic_with_unequal_edges(self):
        """Catches anywhere a single box length is assumed."""
        traj = _gas_in_cell(15, 'H20He20', [9.0, 13.0, 17.0])
        self._assert_same_result(traj, radius=4.0)

    def test_overlapping_species_groups(self):
        """A pair whose two index lists share atoms exercises the
        self-pair mask, which is what stops an atom being counted as
        its own neighbour at r=0."""
        traj = _gas_in_cell(10, 'H20He20Li10', [12.0] * 3)
        res = self._assert_same_result(
            traj, radius=5.0,
            species_pairs=[(('H', 'Li'), 'H'), ('He', ('H', 'He'))])
        # No spike in the first bin: an atom is not its own neighbour.
        self.assertEqual(res.get_array('rdf_spec_0_H')[0], 0.0)

    def test_variable_cell(self):
        traj = _gas_in_cell(12, 'H20He20', [12.0] * 3,
                            cells=_breathing_cells(12, 12.0, 18.0))
        self._assert_same_result(traj, radius=5.0)

    def test_sampling_window(self):
        traj = _gas_in_cell(30, 'H20He20', [12.0] * 3)
        self._assert_same_result(traj, radius=5.0,
                                 istart=3, istop=27, stepsize=4)

    def test_radius_beyond_the_minimum_image_limit(self):
        """Past half the shortest lattice vector both algorithms count
        only the nearest image and so are biased low.  They have to be
        biased identically, and the warning has to still fire."""
        traj = _gas_in_cell(10, 'H20He20', [10.0] * 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            RDF(trajectory=traj, verbosity=0).run(radius=8.0, method='ortho')
        self.assertEqual(len(caught), 1)
        self.assertIn('biased low', str(caught[0].message))
        self._assert_same_result(traj, radius=8.0)

    def test_atoms_on_the_cell_boundary(self):
        """A coordinate a hair below zero folds to exactly the edge
        under a plain modulo, which cKDTree rejects outright.  Atoms
        sitting on 0 and on the far face must also not be doubled."""
        length = 10.0
        positions = np.array([[[-1e-17, 1.0, 1.0],
                               [0.0, 5.0, 5.0],
                               [length - 1e-13, 2.0, 2.0],
                               [-1e-9, 8.0, 8.0],
                               [3.0, 3.0, 3.0],
                               [7.0, 7.0, 7.0]]])
        atoms = Atoms('H3He3', cell=np.eye(3) * length, pbc=True)
        traj = Trajectory(atoms=atoms)
        traj.set_positions(positions)
        self._assert_same_result(traj, radius=4.0)

    def test_nearly_orthorhombic_cell_still_takes_ortho(self):
        """Rounding junk off the diagonal must not push a cubic cell
        onto the slow path, and must not change the answer either."""
        cell = np.eye(3) * 12.0
        cell[0, 1] = 1e-14
        traj = _gas_in_cell(10, 'H20He20', cell)
        buffer = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect_stdout(buffer):
                RDF(trajectory=traj).run(radius=4.0)
        self.assertIn('faster ortho algorithm', buffer.getvalue())
        self._assert_same_result(traj, radius=4.0)


class TestAlgorithmSelection(unittest.TestCase):
    """Which algorithm gets picked, and what is said about it."""

    TRICLINIC = np.array([[10., 0., 0.], [3., 10., 0.], [2., 4., 10.]])

    def _run_capturing_output(self, traj, **kwargs):
        buffer = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect_stdout(buffer):
                RDF(trajectory=traj).run(radius=4.0, **kwargs)
        return buffer.getvalue()

    def test_auto_picks_ortho_for_an_orthorhombic_cell(self):
        traj = _gas_in_cell(5, 'H10He10', [12.0] * 3)
        out = self._run_capturing_output(traj)
        self.assertIn('faster ortho algorithm', out)
        self.assertIn('detected as orthorhombic', out)

    def test_auto_picks_skew_for_a_triclinic_cell(self):
        traj = _gas_in_cell(5, 'H10He10', self.TRICLINIC)
        out = self._run_capturing_output(traj)
        self.assertIn('slower skew algorithm', out)
        self.assertIn('not orthorhombic', out)

    def test_forcing_skew_says_it_was_requested(self):
        traj = _gas_in_cell(5, 'H10He10', [12.0] * 3)
        out = self._run_capturing_output(traj, method='skew')
        self.assertIn('slower skew algorithm', out)
        self.assertIn('requested', out)

    def test_mixed_cell_shapes_report_both(self):
        """Half the frames orthorhombic, half not."""
        cells = np.array([np.eye(3) * 12.0] * 3
                         + [self.TRICLINIC] * 2)
        traj = _gas_in_cell(5, 'H10He10', np.eye(3) * 12.0, cells=cells)
        out = self._run_capturing_output(traj)
        self.assertIn('for 3 of 5 frames', out)
        self.assertIn('cell shape changes', out)

    def test_ortho_refuses_a_triclinic_cell(self):
        traj = _gas_in_cell(5, 'H10He10', self.TRICLINIC)
        with self.assertRaises(ValueError) as ctx:
            RDF(trajectory=traj, verbosity=0).run(radius=4.0, method='ortho')
        self.assertIn('5 sampled frames', str(ctx.exception))

    def test_unknown_method_is_rejected(self):
        traj = _gas_in_cell(5, 'H10He10', [12.0] * 3)
        with self.assertRaises(ValueError) as ctx:
            RDF(trajectory=traj, verbosity=0).run(radius=4.0, method='fast')
        self.assertIn("'auto', 'ortho' or 'skew'", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
