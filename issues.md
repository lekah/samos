# Issues

Findings from the language- and implementation-mix review of
2026-09-03, ordered by importance (#1 = most important). Work from
the top.

Each issue carries a **Fix difficulty** score from 1 to 10:

* **1** -- very easy; a one- or two-line change, and if the fix is
  wrong an existing (or trivially added) test catches it immediately.
* **5** -- a contained refactor of one function or a small API change;
  breakage is noticeable but needs a deliberate test.
* **10** -- large cross-cutting refactor; if the fix is wrong the
  damage is silent, numerically subtle, and hard to trace back.

**When an issue is fixed and verified, delete its entry from this
file.** Renumbering the remaining entries is not required -- the
numbers are labels, not an ordering contract.

---

## Inventory

| file | lines | what it does | reached from |
|---|---|---|---|
| `samos/lib/mdutils.f90` | 374 | MSD x3, VAF, centre of mass x2 | `get_msd`, `get_vaf` |
| `samos/lib/mdutils_cpp_omp.cpp` | 181 | 4 of those 6, with OpenMP | `get_msd(backend='cpp')` |
| `samos/lib/gaussian_density.f90` | 182 | paints gaussians on a 3-D grid | `get_gaussian_density` |
| `setup.py` | 71 | hand-rolled build for both toolchains | every install |

737 lines of compiled code, about 10% of the project. Everything else
-- RDF, ADF, LAMMPS reading, CLI, plotting -- is pure Python.

All timings below were measured on this machine against synthetic
trajectories, comparing the shipped kernel with a prototype
replacement. "agreement" is the largest relative difference between
the two results, i.e. floating-point rounding only.

---

## 1. The MSD and VAF kernels are quadratic where they need not be

**Fix difficulty: 7**

`samos/lib/mdutils.f90` (whole file), `samos/analysis/dynamics.py`

Every MSD and VAF routine loops over each pair of time origins, which
costs O(N^2) in trajectory length. The same quantity can be had from
an FFT in O(N log N) -- the Fast Correlation Algorithm, standard in
this field. The identity is

```
sum_tau |r(tau+t) - r(tau)|^2
    = sum_tau |r(tau+t)|^2 + sum_tau |r(tau)|^2
      - 2 * sum_tau r(tau+t) . r(tau)
```

The first two terms are sliding-window sums (a cumulative sum), the
third is a cross-correlation (one FFT). Measured against the shipped
Fortran, same estimator, not an approximation:

| routine | Fortran | numpy/scipy | speedup | agreement |
|---|---|---|---|---|
| MSD blocked, 20k steps | 2.8 s | 0.17 s | 16x | 5e-14 |
| MSD blocked, 100k steps | 108 s | 1.8 s | 61x | 2e-13 |
| MSD decomposed 3x3 | 10.5 s | 0.80 s | 13x | 4e-14 |
| MSD max-stats (`do_long`) | 21.9 s | 0.26 s | 83x | 1e-13 |
| VAF | 51.3 s | 0.60 s | 85x | 2e-14 |
| `get_com_positions` | 3.4 s | 0.33 s | 10x | exact |

The gap grows with trajectory length, because one side is quadratic
and the other is not. For comparison, the C++ backend on 8 threads
buys 6.8x over Fortran -- less than pure single-threaded numpy.

The centre-of-mass routines are not even a special case: they are a
mass-weighted average over atoms, which `np.einsum` hands to BLAS and
gets bit-identical numbers 10x faster.

**Fix:** reimplement all six routines in `dynamics.py` using
`numpy.fft` / `scipy.signal.fftconvolve`, and delete `mdutils.f90`.
Roughly 80 lines of Python replace 374 lines of Fortran.

Two things to plan for:

* **Memory.** The FFT holds an array of `2 * nstep * nat_of_interest`
  doubles. For a very long trajectory with many atoms that must be
  chunked over atoms; the correlation is a plain sum over atoms, so
  batching is exact.
* **`stepsize_tau`.** See issue #6 -- it stops making sense.

Verify by capturing every result array for the existing test
trajectories before the change and comparing after. The tolerance
should be roundoff (1e-13 relative), not a loose fit.

---

## 2. The C++ backend is a partial duplicate of the Fortran

**Fix difficulty: 3**

`samos/lib/mdutils_cpp_omp.cpp`, `samos/analysis/dynamics.py:579-594`,
`samos/cli.py:783-788`, `tests/test_dynamics.py:169-197`,
`tests/test_cli.py:83-122`, `setup.py`

`mdutils_cpp_omp.cpp` re-implements 4 of the 6 Fortran routines. It
has no `get_com_velocities` and no `calculate_vaf_specific_atoms`, so
`backend='cpp'` quietly applies to MSD only -- a user who sets it
expecting a faster VAF gets the Fortran path with no warning.

Two implementations of the same maths have to be kept in step, and
they have already drifted. The header comment claims
`calculate_msd_specific_atoms_max_stats` "uses Welford's online mean
... rather than the Fortran accumulate-then-divide approach", but the
Fortran uses a running mean too (`mdutils.f90:127-128`). The comment
describes a difference that does not exist.

The cost of the switch itself: about 36 lines across `dynamics.py`,
`cli.py` and the tests, plus a duplicated test block that has to
compare with a tolerance rather than exactly, because OpenMP
reduction order changes the rounding.

**Fix:** delete the file, the `backend=` and `num_threads=` arguments,
the `--backend` CLI flag, and the duplicated tests. In `setup.py`,
drop the `Pybind11Extension` entry, the `from pybind11.setup_helpers`
import, and the whole `cpp_exts` / `f2py_exts` split in
`CombinedBuild` -- with only f2py extensions left, `run()` no longer
needs to partition anything. Drop `pybind11>=2.6` from
`[build-system] requires` in `pyproject.toml`.

The `extra_compile_args=['-O3', '-fopenmp']` and
`extra_link_args=['-fopenmp']` belong to the `Pybind11Extension`
constructor and go with it. The f2py path never sees them: `setup.py`
shells out to `python -m numpy.f2py -c ...` with no compiler flags at
all, so removing the C++ extension removes the only OpenMP dependency
in the project. That also fixes issue #5.

Worth doing whether or not #1 happens, but if #1 lands first this is
a straight deletion with nothing to replace.

---

## 4. `plot_xsf.py` is dead and imports an undeclared dependency

**Fix difficulty: 1**

`samos/plotting/plot_xsf.py:4`

`from mayavi import mlab` at module import. Mayavi is not in
`pyproject.toml` dependencies, is not installed in the development
environment, is imported by nothing else in samos, and has no tests.
380 lines that cannot run as shipped.

**Fix:** delete it, or move it to `examples/` as a standalone script
with its own install note. If it is kept in the package, mayavi has
to become a declared optional dependency and the import has to move
inside the function that needs it, so that importing
`samos.plotting` does not fail.

---

## 5. The macOS build is broken, and the metadata claims it works

**Fix difficulty: 2**

`setup.py:66-67`, `pyproject.toml` classifiers

`extra_compile_args=['-O3', '-fopenmp']` fails with Apple's clang,
which ships no libomp unless the user installs it separately. The
`Operating System :: MacOS` classifier says the package supports
macOS. CI only builds on `ubuntu-latest`, so nothing catches it.

**Fix:** issue #2 removes the flag entirely and the problem with it.
Until then, either make OpenMP optional (probe for it and fall back
to a serial build) or drop the macOS classifier. Adding
`macos-latest` to the CI matrix would have surfaced this.

---

## 6. `stepsize_tau` is a speed knob that costs statistics

**Fix difficulty: 2**

`samos/analysis/dynamics.py:342`, `:541`

Documented only as "Inner-loop stride". It subsamples the time origins
so the quadratic loop does less work, at the cost of averaging over
fewer samples. It buys nothing physically.

Once #1 lands, every time origin is used for free, so the knob only
makes results noisier.

**Fix:** deprecate it after #1. Warn if it is passed with a value
other than 1 and ignore it, rather than emulating the subsampling --
emulating it would mean keeping a slow path alive for a parameter
nobody wants.

---

## 7. `constants.py` redefines what ase already provides

**Fix difficulty: 1**

`samos/utils/constants.py`

`bohr_to_ang = 0.52917721092` and `kB_ev = 8.6173303e-5` duplicate
`ase.units.Bohr` and `ase.units.kB`, with slightly older CODATA
values. `ase` is already a hard dependency.

**Fix:** import from `ase.units` and keep only the constants ase does
not carry (`ANG2_FS_TO_CM2_S`). Note that the values change in the
12th digit, so any reference JSON in `tests/ref/` that was produced
through `bohr_to_ang` needs re-checking, not blind regeneration.

---

## 8. `samos/lib/__init__.py` imports itself

**Fix difficulty: 1**

`samos/lib/__init__.py:8`

```python
from . import *  # noqa: F403
```

A package importing itself with a star. It does nothing beyond what
the two lines above it already did, and the `# noqa` markers hide
that the linter noticed.

The star imports also mean that touching `samos.lib` at all pulls in
both compiled modules, so a user with no Fortran compiler cannot
import the package even to reach a routine that does not need it.
(In practice `dynamics.py` and `get_gaussian_density.py` import the
submodules directly, so nothing depends on the star imports.)

**Fix:** reduce the file to a docstring. If #1 lands, `mdutils` is
gone and only `gaussian_density` is left to mention.

---

## 9. The Fortran writes the output file that Python opened

**Fix difficulty: 4**

`samos/lib/gaussian_density.f90:144-161`,
`samos/analysis/get_gaussian_density.py:140`

Python writes the xsf header through `write_xsf`, then
`make_gaussian_density` opens Fortran unit 21 on the same path with
`status='old', access='append'` and writes the data grid itself,
including the closing `END_DATAGRID_3D` / `END_BLOCK_DATAGRID_3D`
lines. Two languages take turns appending to one file, and the format
is defined half in each.

This is the one routine where compiled code genuinely earns its keep.
It is a scatter-add of gaussians onto a grid, which numpy is bad at: a
vectorised numpy prototype (batched over atoms and frames,
`np.bincount` for the scatter) came out **2.5x slower** than the
Fortran -- 0.9 s vs 2.4 s for 200 frames x 20 atoms on a 120^3 grid.
That is a real loss, unlike the 10-85x win in issue #1.

**Fix:** keep the Fortran maths, but have it return the grid as an
array and let Python write it. `make_gaussian_density` already
allocates `counted(n1,n2,n3)` -- make it `intent(out)` instead of
writing it to disk, and pass it to the existing `write_xsf`. That
removes the filename argument, the append-mode coupling, and the
duplicated xsf format knowledge, and makes the routine testable
without a temporary file.

**Alternative worth considering:** accept the 2.5x and go pure Python.
That deletes `setup.py` entirely, drops gfortran, meson and ninja from
the build, and turns samos into a pure-Python wheel that installs
anywhere in a second -- including Windows and macOS, which currently
need a working Fortran toolchain. The density is a niche feature: it
is not exposed through the CLI and appears in one example notebook.
This is a judgement call about whether the install experience is worth
more than 2.5x on a rarely used routine.

---

## Summary of what could go

Deleting #2 and #4 loses nothing at all: 561 lines, one language, and
the pybind11 and OpenMP dependencies.

Doing #1 as well replaces 374 lines of Fortran with about 80 lines of
Python that runs 10-85x faster, and leaves `gaussian_density.f90` as
the only compiled file in the project.

Doing the alternative under #9 on top of that removes `setup.py`, the
compiler requirement, and the whole f2py build path.
