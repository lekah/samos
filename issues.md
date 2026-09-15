# Issues

Findings from the language- and implementation-mix review of
2026-09-03, ordered by importance (#1 = most important) while open.
Every issue found is now fixed and removed below; see the Summary at
the end for what changed.

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
| `setup.py` | 56 | build entry point, nothing to build | every install |

Zero lines of compiled code in active use, down from 737 at the start
of this review. Everything -- RDF, ADF, LAMMPS reading, CLI, dynamics,
plotting -- is pure Python. `mdutils.f90`, `mdutils_cpp_omp.cpp` and
`gaussian_density.f90` are kept in `samos/lib` for reference, each
with a header pointing at what replaced it, but none is built or
imported by anything.

---

## Summary of what could go

Every issue this review raised has been addressed. `plot_xsf.py`
(issue #4, dead mayavi import) is gone; `gaussian_density.f90` (issue
#9), `mdutils_cpp_omp.cpp` (issue #2) and `mdutils.f90` (issue #1) are
all disabled, kept only as reference; `stepsize_tau` (issue #6) is
deprecated -- warned about and ignored, since an O(N log N)
correlation has no use for subsampling. Deleting #2 also took
pybind11 and the `-fopenmp` flag with it, so the macOS classifier
issue (#5) is fixed for real rather than papered over.

Replacing `mdutils.f90`'s six O(N^2) routines with the Fast
Correlation Algorithm (`samos/analysis/_fft_dynamics.py`) measured
20-200x faster end to end, growing with trajectory length as expected
for O(N^2) vs O(N log N). It also surfaced a real, independent bug
along the way: `get_msd`'s reported time axis was off by one step,
confirmed with a constant-velocity trajectory whose MSD at every lag
is exactly predictable -- fixed rather than carried over, see
README.md's "`mdutils.f90` is disabled".

No compiler, `gfortran`, `meson`, `ninja` or pybind11 is needed to
install samos any more.
