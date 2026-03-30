# SAMOS (Suite for Analysis of Molecular Simulations)

This software, distributed openly and free of charge (see LICENCE), analyzes
molecular dynamics simulations.  Currently implemented are:

* Tracer and charge diffusion coefficients from the mean-square displacement
  (MSD) or the integral of the velocity autocorrelation function (VAF)
* Vibrational density of states (power spectrum / VDOS)
* Atomic probability densities
* Radial distribution functions
* Several plotting utilities

The computationally intensive parts are written in Fortran 90 (wrapped with
f2py) and optionally in C++ with OpenMP parallelisation.

---

## Installation

```bash
pip install numpy       # or: conda install numpy
pip install .
```

Other dependencies: `ase`, `scipy`, `matplotlib`.

---

## Command-line usage

The `samos` script accepts a trajectory path followed by a sub-command.
Time values are plain floats; the unit is set once per sub-command with
`--t-unit` (choices: `fs`, `ps`, `dt`; default: `ps`).

### MSD

```bash
# Fit window 2-4 ps, 6 blocks, Li and O species
samos traj.extxyz --species Li O -n 6 msd --t-start-fit 2 --t-end-fit 4

# Same but specify fit window in femtoseconds
samos traj.extxyz msd --t-start-fit 2000 --t-end-fit 4000 --t-unit fs

# Use the C++ backend with 4 threads; write results to CSV
samos traj.extxyz msd -b cpp -n 4 --write msd.csv

# Fit window in timesteps (dt)
samos traj.extxyz msd --t-start-fit 160 --t-end-fit 320 --t-unit dt
```

### VAF

```bash
# 12 blocks, integral-averaging window 2-4 ps, max lag time 5 ps
samos traj.extxyz -n 12 vaf --t-start-fit 2 --t-end-fit 4 --t-end 5

# Remove rigid-body angular momentum before computing the VAF
samos traj.extxyz vaf --remove-angular-momentum --t-unit fs \
    --t-start-fit 2000 --t-end-fit 4000

# Write VAF to CSV and save plot
samos traj.extxyz vaf --write vaf.csv --savefig vaf.png
```

### VDOS (vibrational density of states)

```bash
# 4 blocks, smoothing kernel width of 3 bins
samos traj.extxyz -n 4 vdos --smoothing 3

# Plot interactively
samos traj.extxyz vdos --plot
```

### RDF

```bash
# All pairs involving Li, radius 6 A
samos traj.extxyz --species Li rdf --radius 6

# Explicit pairs only
samos traj.extxyz rdf --species-pairs Li-O O-O --bins 200
```

### Global options

| Flag | Description |
|------|-------------|
| `--timestep FS` | Override the trajectory timestep (fs) |
| `--species SYM ...` | Restrict analysis to these chemical symbols |
| `-n N` / `--nblocks N` | Split trajectory into N blocks |
| `--recenter` | Subtract centre-of-mass motion before analysis |
| `--transform-species SYM` | Relabel all atoms as SYM |
| `--write FILE` | Write results to a CSV file |
| `--plot` | Show the plot interactively |
| `--savefig FILE` | Save the plot to FILE |

---

## Python API

```python
from samos.trajectory import Trajectory
from samos.analysis.dynamics import DynamicsAnalyzer

traj = Trajectory.load_file('traj.extxyz')
traj.recenter()

d = DynamicsAnalyzer(trajectories=[traj])

# MSD — fit window and block length both in ps
msd = d.get_msd(
    species_of_interest=['Li', 'O'],
    t_start_fit=2., t_end_fit=4., t_unit='ps',
    block_length=8., nr_of_blocks=None)

# MSD with C++ backend
msd_cpp = d.get_msd(
    t_start_fit=2., t_end_fit=4., t_unit='ps',
    nr_of_blocks=6, backend='cpp', num_threads=4)

# VAF
vaf = d.get_vaf(
    species_of_interest=['Li'],
    t_start_fit=2., t_end_fit=4., t_end=5., t_unit='ps',
    nr_of_blocks=12)

# Power spectrum / VDOS
pws = d.get_power_spectrum(
    species_of_interest=['Li', 'O'],
    nr_of_blocks=4, smothening=3)
```

Time parameters (`t_start`, `t_end`, `t_start_fit`, `t_end_fit`,
`block_length`, `t_long_end`) all accept a plain numeric value; the
unit is set once per call via `t_unit` (`'fs'`, `'ps'`, or `'dt'`).
`stepsize_t` and `stepsize_tau` are always plain integer timestep counts.

The C++ MSD backend supports OpenMP parallelisation.  Pass
`backend='cpp'` and optionally `num_threads=N`.  The Fortran backend is
the default.

---

## Backwards-incompatible changes

The following changes break existing call sites.

### Time-parameter API (dynamics module)

Old code used per-parameter unit suffixes:

```python
# OLD — no longer accepted; raises InputError
d.get_msd(t_start_fit_ps=2., t_end_fit_ps=4., block_length_dt=640)
d.get_vaf(t_end_fs=5000., nr_of_blocks=12)
```

New code passes a plain value and a single `t_unit` per call:

```python
# NEW
d.get_msd(t_start_fit=2., t_end_fit=4., t_unit='ps',
          block_length=8., nr_of_blocks=None)
d.get_vaf(t_end=5., t_unit='ps', nr_of_blocks=12)
```

Passing an old-style suffix kwarg (e.g. `t_end_fit_ps`) now raises an
`InputError` with an explicit migration message.

### CLI time flags

Old per-parameter flags have been replaced:

| Removed flag | Replacement |
|---|---|
| `--t-start-fit-ps` | `--t-start-fit` (with `--t-unit ps`) |
| `--t-end-fit-ps` | `--t-end-fit` (with `--t-unit ps`) |
| `--t-end-ps` (VAF) | `--t-end` (with `--t-unit ps`) |

The `--t-unit` flag is per sub-command and defaults to `ps`.

### `_get_running_params` return value

Previously returned a 14-element positional tuple.  Now returns a
`RunningParams` namedtuple; callers must access fields by name
(e.g. `rp.t_start_fit_dt` instead of `rp[5]`).
