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
| `setup.py` | 54 | f2py build | every install |

374 lines of compiled code, down from 737 now that
`gaussian_density.f90` (issue #9) and `mdutils_cpp_omp.cpp` (issue #2)
are gone. Everything else -- RDF, ADF, LAMMPS reading, CLI, plotting
-- is pure Python.

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

## Summary of what could go

`plot_xsf.py` (issue #4, dead mayavi import), `gaussian_density.f90`
(issue #9) and `mdutils_cpp_omp.cpp` (issue #2) are all gone now.
Deleting #2 also took the pybind11 and OpenMP dependencies with it --
including the `-fopenmp` flag that made the macOS classifier (issue
#5) a lie in the first place, so #5 is fixed for real now rather than
papered over by dropping the classifier as an interim fix.

Only `mdutils.f90` remains compiled. Doing #1 replaces its 374 lines
with about 80 lines of Python that runs 10-85x faster, and at that
point removes the compiler requirement entirely: there would be no
compiled code left, and `setup.py`, gfortran, meson and ninja all drop
out of the build.
