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
one entry left over is a large refactor with silent failure modes,
#33, which touches every constructor call site in the package.

**When an issue is fixed and verified, delete its entry from this file.**
Renumbering the remaining entries is not required -- the numbers are
labels, not an ordering contract.

---

## 33. `AttributedArray.__init__` dispatch depends on keyword order

**Fix difficulty: 7**

`samos/utils/attributed_array.py:16-17`, `samos/analysis/rdf.py:115-116`,
`samos/analysis/dynamics.py:207-208`

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
