# -*- coding: utf-8 -*-
"""Tests for get_gaussian_density's pure-Python density grid.

samos.lib.gaussian_density (Fortran) has been deleted. Its coordinate
handling turned out to carry two bugs, both fixed here rather than
ported over:

* It folded an atom into the cell with ``inv(cell.T)`` but converted
  back with ``cell`` instead of ``cell.T`` -- a round trip that only
  returns to where it started for a symmetric (in particular,
  orthorhombic) cell.
* Every grid candidate's weight was computed one grid step further
  along each axis than the storage cell it was written to. Present
  for every cell shape, not just skewed ones -- invisible only
  because a stored gaussian is usually broader than one grid spacing.

Because of the second bug, the old routine cannot serve as a precise
ground truth even for an orthorhombic cell. The tests below mostly
check the new implementation's own physics directly (vertex
placement, normalisation, periodicity, collision handling), plus one
comparison against a captured Fortran reference with its known
one-cell shift undone.
"""

import json
import os
import tempfile
import unittest

import numpy as np
from ase import Atoms

from samos.trajectory import Trajectory
from samos.analysis.get_gaussian_density import (
    _compute_density_grid, get_gaussian_density)
from samos.io.xsf import read_xsf

REF_DIR = os.path.join(os.path.dirname(__file__), 'ref')


class TestExactVertexPlacement(unittest.TestCase):
    """An atom sitting exactly on a grid vertex must peak there, with
    everything else negligible -- the simplest check that a real
    position lands on the matching storage cell, for any cell shape.
    """

    def test_orthorhombic_vertex(self):
        cell = np.diag([4.0, 4.0, 4.0])
        n1 = n2 = n3 = 5
        positions = np.array([[[1.6, 2.4, 0.8]]])  # vertex (2, 3, 1)
        grid = _compute_density_grid(
            positions, cell, [1], 0.05, n1, n2, n3, 1, 1, 1, 1, 1, 1)
        peak = np.unravel_index(np.argmax(grid), grid.shape)
        self.assertEqual(peak, (2, 3, 1))
        self.assertGreater(grid[2, 3, 1], 0.99 * grid.sum())

    def test_skewed_vertex(self):
        cell = np.array([[4.0, 0.0, 0.0],
                         [0.6, 3.5, 0.0],
                         [0.3, -0.4, 3.0]])
        n1 = n2 = n3 = 6
        frac_vertex = np.array([2, 4, 1]) / 6.0
        real = frac_vertex @ cell
        positions = np.array([[real]])
        grid = _compute_density_grid(
            positions, cell, [1], 0.05, n1, n2, n3, 1, 1, 1, 1, 1, 1)
        peak = np.unravel_index(np.argmax(grid), grid.shape)
        self.assertEqual(peak, (2, 4, 1))
        self.assertGreater(grid[2, 4, 1], 0.99 * grid.sum())


class TestNormalization(unittest.TestCase):
    """The grid must integrate to exactly the number of atoms it was
    built from, for any cell shape, atom subset and frame count."""

    def test_orthorhombic_multi_atom_multi_frame(self):
        rng = np.random.default_rng(0)
        cell = np.diag([5.0, 6.0, 4.0])
        nat, nstep = 6, 4
        positions = rng.uniform(-2, 8, size=(nstep, nat, 3))
        n1, n2, n3 = 12, 14, 10
        grid = _compute_density_grid(
            positions, cell, list(range(1, nat + 1)), 0.3,
            n1, n2, n3, 3, 3, 3, 1, nstep, 1)
        dV = abs(np.linalg.det(cell)) / (n1 * n2 * n3)
        self.assertAlmostEqual(grid.sum() * dV, nat, places=8)

    def test_skewed_subset_of_atoms(self):
        rng = np.random.default_rng(1)
        cell = np.array([[5.0, 0.5, -0.3],
                         [0.2, 6.0, 0.4],
                         [-0.1, 0.3, 4.5]])
        nat, nstep = 5, 3
        positions = rng.uniform(-2, 8, size=(nstep, nat, 3))
        indices_i_care = [1, 3, 4]
        n1, n2, n3 = 10, 12, 9
        grid = _compute_density_grid(
            positions, cell, indices_i_care, 0.3,
            n1, n2, n3, 3, 3, 3, 1, nstep, 1)
        dV = abs(np.linalg.det(cell)) / (n1 * n2 * n3)
        total = grid.sum() * dV
        self.assertAlmostEqual(total, len(indices_i_care), places=6)


class TestPeriodicity(unittest.TestCase):
    """Shifting an atom by exactly one lattice vector is the same
    physical position, so it must leave the grid unchanged."""

    def test_shift_by_lattice_vector(self):
        cell = np.array([[5.0, 0.3, -0.2],
                         [0.1, 6.0, 0.4],
                         [-0.3, 0.2, 4.0]])
        n1, n2, n3 = 11, 13, 9
        pos = np.array([1.3, 2.7, 0.9])
        grid_a = _compute_density_grid(
            np.array([[pos]]), cell, [1], 0.3,
            n1, n2, n3, 4, 4, 4, 1, 1, 1)
        shifted_pos = pos + cell[0] - 2 * cell[1]
        grid_b = _compute_density_grid(
            np.array([[shifted_pos]]), cell, [1], 0.3,
            n1, n2, n3, 4, 4, 4, 1, 1, 1)
        np.testing.assert_allclose(grid_a, grid_b, rtol=0, atol=1e-12)


class TestWraparoundCollision(unittest.TestCase):
    """A local box wider than half the grid revisits the same wrapped
    cell more than once from a single atom. Summing must catch every
    visit -- a naive fancy-index += would silently keep only one."""

    def test_box_wraps_onto_itself(self):
        cell = np.diag([3.0, 3.0, 3.0])
        n = 3     # tiny grid
        b = 2     # box wider than the grid itself
        sigma = 0.8
        positions = np.array([[[0.0, 0.0, 0.0]]])
        grid = _compute_density_grid(
            positions, cell, [1], sigma, n, n, n, b, b, b, 1, 1, 1)

        # Independent brute-force reference: one candidate at a time,
        # explicitly summing every collision.
        invcell = np.linalg.inv(cell)
        frac = (positions[0, 0] @ invcell) % 1.0
        real = frac @ cell
        base = np.floor(frac * n).astype(int)
        expected = np.zeros((n, n, n))
        for o1 in range(-b, b + 1):
            for o2 in range(-b, b + 1):
                for o3 in range(-b, b + 1):
                    idx = base + np.array([o1, o2, o3])
                    gp = (idx / n) @ cell
                    w = np.exp(-np.sum((gp - real) ** 2) / (2 * sigma**2))
                    expected[idx[0] % n, idx[1] % n, idx[2] % n] += w
        dV = abs(np.linalg.det(cell)) / n**3
        expected /= expected.sum() * dV
        np.testing.assert_allclose(grid, expected, rtol=1e-10)


class TestAgreementWithOldFortran(unittest.TestCase):
    """Cross-check against a reference captured from the deleted
    Fortran routine: orthorhombic cell, multiple atoms and frames, box
    wide enough that truncating the gaussian's tail is not a factor.
    The known one-grid-step placement bug (see module docstring) is
    undone with a plain np.roll before comparing."""

    def test_matches_captured_fortran_reference_after_shift(self):
        with open(os.path.join(
                REF_DIR, 'gaussian_density_ortho.json')) as f:
            fx = json.load(f)
        cell = np.array(fx['cell'])
        positions = np.array(fx['positions'])
        n1, n2, n3 = fx['grid_shape']
        grid_old = np.array(fx['grid']).reshape(n1, n2, n3)

        grid_new = _compute_density_grid(
            positions, cell, fx['indices_i_care'], fx['sigma'],
            n1, n2, n3, fx['b1'], fx['b2'], fx['b3'],
            fx['istart'], fx['istop'], fx['stepsize'])

        old_shifted = np.roll(grid_old, shift=1, axis=(0, 1, 2))
        np.testing.assert_allclose(
            grid_new, old_shifted, rtol=1e-3, atol=1e-4)


class TestGetGaussianDensityEndToEnd(unittest.TestCase):
    """The public function still writes a readable, normalised xsf
    file, exercising the grid-size / box-size geometry and write_xsf
    plumbing around _compute_density_grid."""

    def test_writes_a_readable_normalised_xsf(self):
        cell = np.diag([5.0, 5.0, 5.0])
        atoms = Atoms('LiCl', cell=cell, pbc=True,
                      positions=[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
        t = Trajectory(atoms=atoms, timestep=1.0)
        rng = np.random.default_rng(2)
        pos = atoms.get_positions()[None, :, :] + rng.uniform(
            -0.3, 0.3, size=(5, 2, 3))
        t.set_positions(pos)

        with tempfile.TemporaryDirectory() as d:
            outfile = os.path.join(d, 'out.xsf')
            get_gaussian_density(
                t, element='Li', outputfile=outfile,
                sigma=0.3, n_sigma=3.0, density=0.4,
                istart=1, istop=5, stepsize=1, verbosity=0)
            result = read_xsf(outfile)

        # places=3, not 6: the file round-trips through write_xsf's
        # %.4E text format (4 significant digits per grid point), so
        # summing ~2000 points accumulates more rounding than the
        # in-memory grid itself carries -- see TestNormalization for
        # a tight check on the array before it goes through text.
        dV = result['volume_ang'] / result['data'].size
        self.assertAlmostEqual(result['data'].sum() * dV, 1, places=3)
        # Li (index 1) is the species the density was built from, so
        # it's excluded from the plotted structure by default.
        self.assertEqual(result['atoms'], ['Cl'])


if __name__ == '__main__':
    unittest.main()
