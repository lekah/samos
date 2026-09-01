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

## 17. Unvalidated / fragile LAMMPS dump parsing

**Fix difficulty: 4**

`samos/io/lammps.py:12` (`float_regex`), `:273` and `:379` (body read)

* `float_regex = r'[\-]?\d+\.\d+(e[+\-]\d+)?'` demands a literal decimal
  point and a lowercase `e`. Box bounds written as `0 10`, `1e+01` or
  `-1.0E+00` produce the wrong number of matches and an opaque unpack
  error.
* `np.array([f.readline().split() for _ in range(nat_must)])` on a
  truncated file builds a ragged list -> numpy "inhomogeneous shape"
  error rather than "file ended early at frame N".
* The `xlo`/`ylo`/`zlo` origin is discarded (`cell[i,i] = d2 - d1`).
  Harmless for displacement-based analysis, but scaled coordinates are
  converted with `pos.dot(cell)` and never re-offset -- document the
  assumption or apply the origin.

**Fix:** widen the float regex (or use `float()` on `split()` tokens and
catch `ValueError`); check the line count while reading the body and
raise a message naming the frame and line.

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

## 22. `smothening` typo in the public API

**Fix difficulty: 3**

`samos/analysis/dynamics.py:1086, 1105, 1140, 1193, 1195`

Worth noting how far the cost has spread: `scripts/samos`
carries a translation shim and an apologetic comment
(`scripts/samos:302, 319-321, 333, 338`), `tests/test_dynamics.py:156`
uses the typo, and `examples/ex2-compute-VAF-from-extxyz/compute-vaf.py`
propagates it.

**Fix:** rename to `smoothing`, accept `smothening` as a deprecated
alias for one release (emit `DeprecationWarning`), update all four call
sites and remove the shim in `scripts/samos`.

---

## 23. `bohr_to_ang` defined four times, with two different values

**Fix difficulty: 2**

* `samos/io/xsf.py:6` -- `0.52917720859`
* `samos/plotting/plot_xsf.py:18` -- `0.52917720859`
* `samos/analysis/get_gaussian_density.py:8` -- `0.52917720859`
* `samos/utils/constants.py:8` -- `0.52917721092`  <- the module meant
  to hold it, imported by none of the above
* `samos/utils/units.py` -- a third truncation, `0.529177`, inline in
  the `electron` entry

**Fix:** import from `samos.utils.constants` everywhere, pick one value
(CODATA 2018: `0.529177210903`), and cite the source in `constants.py`.

---

## 24. `write_xsf_header` duplicates `write_xsf`

**Fix difficulty: 3**

`samos/analysis/get_gaussian_density.py:11-61` versus
`samos/io/xsf.py:102-164`

Near-verbatim copies: same CRYSTAL/PRIMVEC header, same `PRIMCOORD`
block, same `data[x % xdim, y % ydim, z % zdim]` wrap loop, same
`if col:` no-op, and now the same guarded-close fix applied twice. The only real
difference is that the copy can emit a header with no data block.

**Fix:** give `samos.io.xsf.write_xsf` a `data=None` mode and delete the
copy.

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

## 26. Block-length dispatch written three times; `get_power_spectrum` bypasses `_get_running_params`

**Fix difficulty: 5**

`samos/analysis/dynamics.py:1125-1171` (`get_power_spectrum`)

`get_msd` and `get_vaf` now share `_resolve_blocks`, but
`get_power_spectrum` still re-implements the `block_length` /
`nr_of_blocks` parsing from `_get_running_params` inline rather than
calling it, with a *different* formula (`nstep // nr_of_blocks`, with no
`t_end_dt` term) and no validation at all:

```
get_power_spectrum(nr_of_blocks=500) on a 300-step trajectory
-> periodogram of shape (500, 0), silently, no error
```

That is 500 empty spectra presented as a result.

The `do_com` branches of `get_msd` and `get_vaf` still mirror each
other, though they now share the `_get_masses` /
`_species_factors` helpers, so only the call sequence is repeated.

**Fix:** have `get_power_spectrum` call `_get_running_params` and
`_resolve_blocks` like its siblings -- decide first whether its
block layout should keep ignoring `t_end_dt` (it has no lag window, so
it probably should) and give `_resolve_blocks` a flag for that. A
`_com_positions(trajectory, array)` helper would fold the two remaining
`do_com` branches together.

---

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

## 31. Documentation that contradicts the code

**Fix difficulty: 1**

* `samos/trajectory.py:139` -- `get_timestep` docstring says "or None if
  not set"; it delegates to `get_attr`, which raises `KeyError`.
* `samos/trajectory.py:15` -- `check_trajectory_compatibility` docstring
  claims it checks "the same cell"; it checks array names, types and
  timestep, never the cell.
* `samos/analysis/get_gaussian_density.py:68-91` -- the docstring
  documents a different function: `positionsf`, `pos_units`,
  `with_symbols`, `cell`, `nat`, `recenter` are not parameters, while
  `trajectory`, `stepsize`, `indices_i_care` and
  `indices_exclude_from_plot` are undocumented.
* `samos/utils/attributed_array.py:34` -- `:param book check_nstep:`;
  `:42` -- `check_nat` "Defaults to True" (it defaults to `False`).
* `samos/utils/attributed_array.py:169` -- refers to
  "`Trajectore.store`"; the method is `save`.
* `samos/trajectory.py:294, 312, 399` -- `check_exising`.
* `samos/analysis/dynamics.py:1181` -- prints `block_length_ps = {}`
  with a value in **fs**.
* `samos/io/lammps.py:109, 115` -- `print('Element found at index
  {element_idx}')` missing the `f` prefix; prints the literal braces.
* Spelling in user-visible strings: `'You need ot pass'` (rdf.py:22),
  `'scpecification'` / `'befound'` (rdf.py:106), `'frmo'`
  (lammps.py:287), `'kwywods'` (lammps.py:361), `'keywrods'`
  (lammps.py:531), `'not existen'` (trajectory.py:116), `'I devide'`
  (dynamics.py:1035).
* `README.md` expands SAMOS as "Suite for Analysis of Molecular
  Simulations"; `pyproject.toml` says "Package for Analysis and Tricks
  for MOlecular Simulations". Pick one.

---

## 32. `np.matrix` is deprecated and pending removal

**Fix difficulty: 3**

`samos/io/xsf.py:87`, `samos/analysis/get_gaussian_density.py:148-158`

`np.matrix` has been discouraged since NumPy 1.15 and is slated for
removal. The `get_gaussian_density` bounding-box derivation leans on
`*` meaning matrix multiplication and on `.I`, so the translation to
`ndarray` (`@`, `np.linalg.inv`) must be done carefully.

**Fix:** convert to `ndarray` with `@` and `np.linalg.inv`; verify the
resulting `b1, b2, b3` bounding box is unchanged for a triclinic cell
before and after.

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

## 34. `mmap_mode='r'` in `load_file` is defeated by `set_array`

**Fix difficulty: 2**

`samos/utils/attributed_array.py:204-205` loads with `mmap_mode='r'`,
but `set_array` immediately does `array = np.array(array)`, which
copies. The mmap buys nothing and the intent is misleading (it also
makes the temp-directory cleanup safe, so the copy cannot simply be
removed).

**Fix:** drop `mmap_mode`, or add a documented lazy path that keeps the
extracted files alive. Decide and comment which. Note `load_file` now
delegates non-array members to the `_load_extra` hook, so the array
loop is the only remaining caller of `np.load` here.

---

## 35. `scripts/samos` plotting boilerplate repeated five times

**Fix difficulty: 3**

`scripts/samos:197-205, 285-294, 351-360, 425-434, 574-590`

The same `GridSpec(...) / plt.figure(figsize=(4,3)) / add_subplot /
savefig-or-show` block appears in `run_msd`, `run_vaf`, `run_vdos`,
`run_rdf` and `run_adf`, and twice more in
`samos/analysis/dynamics.py:util_msd` and
`samos/analysis/rdf.py:util_rdf_and_plot`.

`util_msd` and `util_rdf_and_plot` (plus their `__main__` argument
parsers) now overlap almost entirely with `scripts/samos`.

**Fix:** one `_make_axes()` / `_finish(fig, plot, savefig)` pair in
`scripts/samos`; then decide whether the `util_*` functions and the
per-module `__main__` blocks should be deleted in favour of the single
CLI (see also issue #14).

---

## 36. `util_msd` still uses the old unit-suffix parameter names

**Fix difficulty: 1**

`samos/analysis/dynamics.py:1238-1240`

`util_msd(t_start_fit_ps=..., t_end_fit_ps=...)` keeps the pre-redesign
naming in its own signature, even though its internal call to `get_msd`
is already updated. Low priority -- `util_msd` is
not part of the main public API and may be deleted entirely under
issue #35.

---

## 37. `get_kinetic_energies` runs a Python triple-nested loop

**Fix difficulty: 3**

`samos/analysis/dynamics.py:1020-1080`

Loops over steps x atoms x polarisations in pure Python; unusable for
large trajectories.

**Fix:** `np.einsum('i,sic,sic->s', masses, vel, vel)` and equivalents
for the species/atom decompositions. Assert the vectorised result
matches the current loop on a small trajectory before replacing it --
`tests/test_dynamics.py::TestCenterOfMassAndKineticEnergies` covers
the species decomposition.

---

## 38. `class DynamicsAnalyzer(object)` and `def run(*args, **kwargs)` without `self`

**Fix difficulty: 1**

`samos/analysis/dynamics.py:167`, `samos/analysis/rdf.py:14, 25-27`

Python-2 era `(object)` base, and `BaseAnalyzer.run`'s abstract signature
omits `self`. `BaseAnalyzer` also never initialises `self._trajectory`,
so calling `run()` before `set_trajectory()` gives `AttributeError`
rather than a clear message.

**Fix:** drop `(object)`, add `self`, initialise `self._trajectory =
None` in `BaseAnalyzer.__init__` and raise a clear error when unset.

---

## 39. Magic numbers in the diffusion prefactors lack a unit comment

**Fix difficulty: 1**

`samos/analysis/dynamics.py:777, 922`

`1e-1 / dimensionality_factor` in `get_msd` and `0.1/3.` in `get_vaf`
are A^2/fs -> cm^2/s conversions.

**Fix:** name the constant (`ANG2_FS_TO_CM2_S = 1e-1`) in
`samos/utils/constants.py` and reference it from both sites.
