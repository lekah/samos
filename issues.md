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
two entries left over need something this file cannot supply:

* **A large refactor with silent failure modes** -- #33.
* **An optional tidy-up** -- #26.

**When an issue is fixed and verified, delete its entry from this file.**
Renumbering the remaining entries is not required -- the numbers are
labels, not an ordering contract.

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
