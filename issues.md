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
four entries left over need something this file cannot supply:

* **A physics decision** -- #21 (what samos's internal stress unit is),
  #25 (which minimum-image convention wins, given that switching
  changes published numbers for non-orthogonal cells).
* **A large refactor with silent failure modes** -- #26, #33.

**When an issue is fixed and verified, delete its entry from this file.**
Renumbering the remaining entries is not required -- the numbers are
labels, not an ordering contract.

---

## 21. `s_conv` is 1.0 for every unit system while the API claims otherwise

**Fix difficulty: 4**

`samos/utils/units.py` (all seven entries), documented at
`samos/io/lammps.py:227-232` and in the `--units` CLI help

Every unit system sets `'s_conv': 1.0`, including `si` (Pa), `cgs`
(dyne/cm^2), `real` (atm) and `metal` (bar). The docstring says `units`
"sets *all* conversion factors automatically". Stresses come out
unconverted while the API states the opposite -- a silent, physically
wrong result for anyone reading stress from a LAMMPS dump.

`samos/utils/units.py` also does not define an internal stress unit at
all (the module docstring lists positions, velocities, forces, energy,
time).

**Fix:** decide on an internal stress unit (eV/A^3 would match the rest),
fill in the table, and add it to the module docstring -- or, if stress
conversion is deliberately out of scope, document `s_conv` as a no-op
placeholder and say so in the `--units` help. This needs a physics
decision, not just code.

---

## 25. Two incompatible minimum-image implementations in one module

**Fix difficulty: 6**

`samos/analysis/rdf.py:220-232` (`RDF.run`, 8-corner `cdist` scheme)
versus `samos/analysis/rdf.py:433-445` (`BondAnalyzer._pbc_wrap`,
fractional wrapping)

The two disagree for acute cells, and only `RDF.run` carries the
"can actually fail in very acute cell systems" caveat. Whichever is
chosen, both RDF and ADF/bond detection should use the same one, so that
a bond present in the ADF is a bond visible in the RDF.

Related duplication in the same file: the
`try: cell.array / except AttributeError: cell.copy()` idiom appears at
`:124`, `:423` and `:560`, and the fixed/variable-cell setup block is
written twice.

**Fix:** one `minimum_image(diff, cell, cellI)` helper plus one
`get_cell(trajectory, frame)` helper, used by both analyzers. Changing
the RDF's scheme changes published numbers for non-orthogonal cells --
verify against a reference RDF before switching.

---

## 26. `get_power_spectrum` bypasses `_get_running_params`

**Fix difficulty: 5**

`samos/analysis/dynamics.py:1125-1180`

`get_msd` and `get_vaf` share `_resolve_blocks`; `get_power_spectrum`
still parses `block_length` / `nr_of_blocks` inline rather than calling
`_get_running_params`, and computes its own layout with a different
formula (`nstep // nr_of_blocks`, no `t_end_dt` term). That formula is
correct for a periodogram, which has no lag window, and it now validates
its result -- but the parsing is still duplicated and will drift.

The `do_com` branches of `get_msd` and `get_vaf` also still mirror each
other, though they share the `_get_masses` / `_species_factors` helpers,
so only the call sequence is repeated.

**Fix:** give `_resolve_blocks` a flag for whether to reserve `t_end_dt`
steps, and have `get_power_spectrum` call `_get_running_params` like its
siblings. A `_com_positions(trajectory, array)` helper would fold the
two remaining `do_com` branches together.

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
