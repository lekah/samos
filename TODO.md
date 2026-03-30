# TODO

## samos/analysis/dynamics.py

### Bugs

- [ ] **`get_kinetic_energies` — wrong array stored for species decomposition** (line ~935)
  `kinE` (system-level) is stored under `'species_kinetic_energy_*'`; should be `kinE_species`.

- [ ] **`get_vaf` — division by zero for single block**
  `arr_sem = arr_std / np.sqrt(arr.shape[0] - 1)` divides by zero when
  there is only one block. Guard with `if arr.shape[0] > 1` as `get_msd` does.

### Performance

- [ ] **`get_kinetic_energies` — Python triple-nested loop**
  Inner loops over steps/atoms/polarizations should be replaced with
  vectorized numpy operations (e.g. `np.einsum`). Critical for large trajectories.

### Design

- [ ] **`__init__` dynamic dispatch via `getattr(self, 'set_*')`**
  Produces cryptic errors on bad kwargs. Consider explicit keyword arguments.

### Code hygiene

- [ ] **`smothening` typo in public API**
  `smothening` → `smoothing` (parameter name in `get_power_spectrum` and its
  docstring). Requires updating all call sites, including `scripts/samos`.

- [ ] **Python 2 class declaration** (line 15)
  `class DynamicsAnalyzer(object):` → `class DynamicsAnalyzer:`

- [ ] **Magic numbers lack unit-conversion explanation**
  `0.1 / 3.` in `get_vaf` and `1e-1 / dimensionality_factor` in `get_msd`
  are Å²/fs → cm²/s conversions. Add comments explaining the formula.

- [ ] **`util_msd` — legacy parameter names in signature**
  `util_msd(t_start_fit_ps=..., t_end_fit_ps=...)` uses the old unit-suffix
  style in its own signature (though the internal call to `get_msd` is
  already updated). Low priority since `util_msd` is not part of the main
  public API.

---

## Completed

- [x] **`_get_running_params` — repeated unit-conversion logic**
  Replaced ~8 copy-pasted fs/ps/dt if/elif chains with a single
  `parse_time(value, unit, timestep_fs)` utility in `samos/utils/time_units.py`.

- [x] **`_get_running_params` — 14-element positional return tuple**
  Replaced with `RunningParams` namedtuple; callers use named field access.

- [x] **`get_power_spectrum` — duplicates block-length parsing**
  Block-length parsing now uses `parse_time` directly; duplication removed.

- [x] **`get_vaf` — TypeError when `block_length` is used**
  Fixed: block/nr_of_blocks dispatch now uses falsy test consistently with
  `get_msd`.

- [x] **`**kwargs` in public API obscures accepted parameters**
  `get_msd`, `get_vaf`, and `get_power_spectrum` now have fully explicit
  signatures. `_get_running_params` is also fully explicit. No `**kwargs`
  anywhere in the public analysis API.

- [x] **Unit-suffix kwargs (`t_end_fit_ps`, `block_length_dt`, etc.)**
  Old-style kwargs removed; all time parameters now take a plain numeric
  value and a single `t_unit` argument per call ('fs', 'ps', or 'dt').

- [x] **Typos: `'Uncrecognized'`, `'block_length_ft'`, `'aotm_indices'`**
  All corrected during the refactor.

- [x] **Dead commented-out code in `get_vaf`**
  Removed `# ~` lines (old print statements and commented-out assignments).

- [x] **Misleading comment in `_get_running_params`**
  "I see whether factors are calculated" comment — gone with the rewrite.
