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

This file supersedes the old `TODO.md`: its open items were folded
in here (issues #22, #33, #36, #37, #38, #39), and its two open
bugs -- the `get_kinetic_energies` species array and the `get_vaf`
single-block division -- have since been fixed.

**When an issue is fixed and verified, delete its entry from this file.**
Renumbering the remaining entries is not required -- the numbers are
labels, not an ordering contract.

---

## 11. RDF normalises variable-cell trajectories with the last frame's volume

**Fix difficulty: 3**

`samos/analysis/rdf.py:217-218`

In the `fixed_cell=False` branch `volume` is reassigned every frame
inside the sampling loop, but only the final value survives to the
normalisation `hist / (4 pi r^2 dr) / (len(ind2) / volume)`. For NPT
trajectories the density prefactor is wrong by the ratio of the last
frame's volume to the mean. Fixed-cell runs are unaffected, which is why
it goes unnoticed.

**Fix:** accumulate the volume over sampled frames and normalise with the
mean (or normalise each frame's histogram contribution by that frame's
volume -- decide which definition you want and document it). Needs a
test with a deliberately varying cell.

---

## 12. Multi-window MSD fits cannot be plotted

**Fix difficulty: 5**

`samos/plotting/plot_dynamics.py:67-70` and `:172-175`

`attrs.get('t_start_fit_dt') // stepsize` -> `TypeError` when
`multiple_params_fit` is true, because the attribute is then a list.
Past that, `slopes_intercepts_this_traj[iblock]` indexes the fit-window
axis rather than the block axis -- the stored array is
`(n_windows, nblocks, 2)` in that mode (`dynamics.py:604-610`).

`get_msd` accepts list-valued `t_start_fit`/`t_end_fit`, but nothing can
plot the result unless `no_block_fits=True` is passed.

**Fix:** branch on `multiple_params_fit` in both plot functions and draw
one fit line per (window, block) pair. Consider representing the scalar
case as a length-1 list so the two code paths collapse into one -- that would fix this at the source and remove the
duplicated special case in `get_msd`.

---

## 14. Dead entry points

**Fix difficulty: 3**

* `samos/analysis/get_gaussian_density.py:200` imports
  `samos.io.ase_io`, **which does not exist** -- `samos/io/` contains
  only `lammps.py` and `xsf.py`. The `__main__` block cannot run.
* `samos/io/xsf.py:217-219`: `write_grid(outfilename=..., **r)` and
  `write_xsf(outfilename=..., **r)` splat a dict containing
  `volume_ang` / `volume_au`, which neither function accepts ->
  `TypeError` on both CLI paths.

**Fix:** either restore the missing reader (or route through
`ase.io.read` + `Trajectory.from_atoms`), and filter the `read_xsf`
result dict down to the keys the writers accept. Decide whether these
`__main__` blocks should exist at all now that `scripts/samos` is the
CLI -- see issue #21.

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

`samos/analysis/rdf.py:196-203` (`RDF.run`, 8-corner `cdist` scheme)
versus `samos/analysis/rdf.py:400-408` (`BondAnalyzer._pbc_wrap`,
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

## 28. `AngularSpectrum` / `ADF` result formats are incompatible, and `ADF` has no plotter

**Fix difficulty: 4**

`samos/analysis/rdf.py:255` stores *triplets* under the attribute name
`species_pairs`, and `samos/plotting/plot_rdf.py:94` reads that key.
`ADF` uses `species_triplets` with `adf_*` / `angles_*` arrays, so
`plot_angular_spec` cannot plot `ADF` output. `scripts/samos:575-590`
works around this by inlining its own ADF plot; there is no `plot_adf`
in `samos.plotting`.

**Fix:** rename `AngularSpectrum`'s attribute to `species_triplets`,
add `plot_adf` to `samos/plotting/plot_rdf.py`, and have `scripts/samos`
call it instead of its inline copy.

---

## 33. `AttributedArray.__init__` dispatch depends on keyword order

**Fix difficulty: 7**

`samos/utils/attributed_array.py:12-16`, `samos/analysis/rdf.py:15-17`,
`samos/analysis/dynamics.py:173-176`

`for key, val in kwargs.items(): getattr(self, 'set_{}'.format(key))(val)`
makes correctness depend on `**kwargs` insertion order:
`Trajectory(atoms=..., positions=...)` works, the reverse raises,
because `set_positions` needs `self.nat`. `samos/io/lammps.py:411`
already relies on that ordering. Bad keyword names produce
`AttributeError: 'Trajectory' object has no attribute 'set_typo'`
rather than a useful message.

**Fix:** explicit keyword arguments with a defined application order,
or an ordered whitelist of setters applied in dependency order. Touches
every constructor call site, hence the score.

---

## 35. `util_msd` / `util_rdf_and_plot` duplicate the `scripts/samos` CLI

**Fix difficulty: 3**

`samos/analysis/dynamics.py:util_msd` and
`samos/analysis/rdf.py:util_rdf_and_plot`, plus the `__main__` argument
parsers below them, now overlap almost entirely with the `msd` and `rdf`
sub-commands of `scripts/samos`. The repeated figure/save/show
boilerplate has been factored into `_make_axes` / `_finish_plot` inside
`scripts/samos`, but the two `util_*` functions still carry their own
copies.

**Needs a decision:** whether the per-module `__main__` blocks should
exist at all now that `scripts/samos` is the entry point. Deleting them
removes `python -m samos.analysis.rdf ...` as a usage, which may be in
someone's scripts. Ask before removing.

