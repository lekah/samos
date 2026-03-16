# TODO

## samos/analysis/dynamics.py

### Bugs

- [ ] **`get_kinetic_energies` — wrong array stored for species decomposition** (line ~935)
  `kinE` (system-level) is stored under `'species_kinetic_energy_*'`; should be `kinE_species`.

- [ ] **`get_vaf` — TypeError when `block_length_*` is used** (line ~762)
  `if nr_of_blocks > 0:` raises `TypeError` when `nr_of_blocks` is `None`.
  Fix: use `if nr_of_blocks:` (falsy test), consistent with `get_msd`.

- [ ] **`get_vaf` — division by zero for single block** (line ~831)
  `arr_sem = arr_std / np.sqrt(arr.shape[0] - 1)` divides by zero when
  there is only one block. Guard with `if arr.shape[0] > 1` as `get_msd` does.

### Performance

- [ ] **`get_kinetic_energies` — Python triple-nested loop** (lines ~893, ~910, ~925)
  Inner loops over steps/atoms/polarizations should be replaced with
  vectorized numpy operations (e.g. `np.einsum`). Critical for large trajectories.

### Design

- [ ] **`_get_running_params` — repeated unit-conversion logic**
  The fs/ps/dt resolution pattern is copy-pasted ~8 times. Extract a helper
  `_to_dt(kwargs, fs_key, ps_key, dt_key, timestep_fs, default=None)`.

- [ ] **`_get_running_params` — 14-element positional return tuple**
  Fragile: adding/reordering a value silently breaks all call sites.
  Replace with a `namedtuple` or `dataclass`.

- [ ] **`get_power_spectrum` — duplicates block-length parsing**
  Reimplements the block-length logic from `_get_running_params` instead of
  delegating to it. Any fix must be manually mirrored.

- [ ] **`__init__` dynamic dispatch via `getattr(self, 'set_*')`**
  Produces cryptic errors on bad kwargs. Consider explicit keyword arguments.

### Code hygiene

- [ ] **Typos in public API and error messages**
  - `smothening` → `smoothing` (parameter name + docstring in `get_power_spectrum`)
  - `'Uncrecognized keywords'` → `'Unrecognized keywords'` (appears twice)
  - `'block_length_ft'` → `'block_length_dt'` in `RuntimeError` in `get_vaf`
  - `'aotm_indices'` → `'atom_indices'` in `get_msd` docstring

- [ ] **Python 2 class declaration** (line 15)
  `class DynamicsAnalyzer(object):` → `class DynamicsAnalyzer:`

- [ ] **Dead commented-out code in `get_vaf`**
  Remove `# ~` lines (old print statements and commented-out assignments).

- [ ] **Magic numbers lack unit-conversion explanation**
  `0.1 / 3.` in `get_vaf` and `1e-1 / dimensionality_factor` in `get_msd`
  are Å²/fs → cm²/s conversions. Add comments explaining the formula.

- [ ] **Misleading comment above dead code in `_get_running_params`** (line ~303)
  Comment says "I see whether factors are calculated" but nothing follows.
  The actual adaptation to trajectory length happens later in `get_msd`.
