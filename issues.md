# Issues

Findings from the code review of 2026-09-01, ordered by importance
(#1 = most important). Work from the top.

Each issue carries a **Fix difficulty** score from 1 to 10:

* **1** -- very easy; a one- or two-line change, and if the fix is wrong
  an existing (or trivially added) test catches it immediately.
* **5** -- a contained refactor of one function or a small API change;
  breakage is noticeable but needs a deliberate test.
* **10** -- large cross-cutting refactor; if the fix is wrong the damage
  is silent, numerically subtle, and hard to trace back.

This file supersedes the old `TODO.md`, whose open items were folded in
here.

**Everything that could be fixed without a decision has been.** The
three entries left over need something this file cannot supply:

* **A decision about scope** -- #41 (whether the Fortran
  `AngularSpectrum` is worth repairing now that `ADF` supersedes it).
* **A large refactor with silent failure modes** -- #33.
  #26 is now a small, optional tidy-up.

**When an issue is fixed and verified, delete its entry from this file.**
Renumbering the remaining entries is not required -- the numbers are
labels, not an ordering contract.

---

## 41. The Fortran AngularSpectrum reports wrong angles across the cell boundary

**Fix difficulty: 4**

`samos/lib/rdf.f90:129-135` and `:150-156` (`calculate_angular_spec`),
reached through `AngularSpectrum.run`

Both wrap loops do this for a fractional component above one half:

```fortran
distance_crystal(idim) = 1.0D0 - distance_crystal(idim)
```

That mirrors the component instead of translating it; the periodic
image is at `f - 1`, not `1 - f`.  The magnitude survives, so distances
in a cubic cell still come out right, but the *vector* is reflected and
the angle built from it is not.

Reproduced in a plain cubic cell: an O with two H neighbours whose
bonds cross the boundary has a true angle of 180 degrees and is
reported at 45.  The same geometry placed in the middle of the cell,
where nothing wraps, comes out correct at 90 degrees.

The surrounding scheme is the naive one that `MinimumImage` replaced on
the Python side, so it is also wrong for non-orthogonal cells even
without the sign error.

`AngularSpectrum` is not reachable from the CLI and is superseded by
`ADF`, which is correct.  Nothing in the examples or the README uses
it.

**Fix:** decide whether `AngularSpectrum` earns its keep.  Deleting it
in favour of `ADF` costs nothing that is documented.  Keeping it means
correcting the Fortran and rebuilding, which needs gfortran and cannot
be verified by the Python tests alone.

---


## 26. `get_msd` and `get_vaf` repeat the `do_com` call sequence

**Fix difficulty: 3**

`samos/analysis/dynamics.py:564-577` and `:923-937`

The block-parameter half of this issue is fixed: `get_power_spectrum`
now goes through `_get_running_params(require_fitting=False)` and
`_resolve_blocks` like its siblings, so there is one copy of the
argument parsing and one copy of the layout arithmetic.

What is left is that the `do_com` branches of `get_msd` and `get_vaf`
still mirror each other. They already share the `_get_masses` /
`_species_factors` helpers, so only the call sequence is repeated --
fetch masses, get species factors, call `get_com_positions` /
`get_com_velocities`, set `indices_of_interest = [1]`.

**Fix:** a helper returning `(array, indices_of_interest, prefactor)`,
parameterised by the kernel to call. This sits in the numerical inner
loop, so check MSD and VAF numbers are unchanged before and after.

## 33. `AttributedArray.__init__` dispatch depends on keyword order

**Fix difficulty: 7**

`samos/utils/attributed_array.py:16-17`, `samos/analysis/rdf.py:17-18`,
`samos/analysis/dynamics.py:178-179`

`for key, val in kwargs.items(): getattr(self, 'set_{}'.format(key))(val)`
makes correctness depend on `**kwargs` insertion order:
`Trajectory(atoms=..., positions=...)` works, the reverse raises,
because `set_positions` needs `self.nat`. `samos/io/lammps.py:450-453`
already relies on that ordering. Bad keyword names produce
`AttributeError: 'Trajectory' object has no attribute 'set_typo'`
rather than a useful message.

**Fix:** explicit keyword arguments with a defined application order,
or an ordered whitelist of setters applied in dependency order. Touches
every constructor call site, hence the score.
