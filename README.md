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

The build compiles Fortran extensions with f2py and a C++ extension with
pybind11 + OpenMP.  A Fortran compiler (gfortran) and a C++ compiler must
be present before running `pip install .`.

```bash
# conda
conda install -c conda-forge gfortran

# apt (Debian/Ubuntu)
sudo apt install gfortran
```

### Troubleshooting build failures

**`numpy.f2py` returns non-zero exit status during `pip install`**

The pip output usually truncates the real compiler error.  Run the
failing f2py command directly to see it:

```bash
cd samos/lib
python -m numpy.f2py -c gaussian_density.f90 -m gaussian_density
```

Common root causes:

| Symptom | Fix |
|---------|-----|
| `gfortran: command not found` | Install gfortran (see above) |
| `meson: command not found` | `pip install meson ninja` |
| Errors about `numpy.distutils` on Python 3.12+ | Upgrade numpy to 2.x (`pip install "numpy>=2.0"`) and ensure meson/ninja are installed |

**C++ extension fails to compile (`-fopenmp` not found)**

The C++ backend uses OpenMP.  On macOS with Apple Clang, `-fopenmp` is
not available by default.  Install LLVM via Homebrew and set the
compiler environment variables, or install gcc:

```bash
brew install gcc
CC=gcc-14 CXX=g++-14 pip install .
```

The Fortran backend works without OpenMP and is used by default; the
C++ backend is optional.

---

## Examples

Working examples are in the `examples/` directory:

| Folder | What it shows |
|--------|---------------|
| `ex1-compute-MSD-from-LAMMPS/` | MSD from a LAMMPS dump; Fortran vs C++ benchmark |
| `ex2-compute-VAF-from-extxyz/` | VAF and VDOS from an extxyz file |
| `ex3-compute-RDF/` | Radial distribution function |
| `ex4-compute-ionic-density/` | Ionic probability densities |
| `ex5-using-the-script/` | All of the above reproduced with the `samos` CLI |

Run `bash examples/ex5-using-the-script/run.sh` from the repository root
to see every sub-command in action.

---

## Command-line usage

The `samos` script accepts a trajectory path followed by a sub-command.
Global options (trajectory format, preprocessing, output) come before the
sub-command; sub-command-specific options come after it.

```
samos TRAJECTORY [global options] COMMAND [command options]
```

Time values are plain floats; the unit is set once per sub-command with
`--t-unit` (choices: `fs`, `ps`, `dt`; default: `ps`).

### Reading LAMMPS dump files

LAMMPS dump files are not auto-detected.  Use one of these flags to
tell `samos` how to resolve element names:

```bash
# Dump has an 'element' column -- no element list needed
samos traj.lammpstrj --lammps --timestep 2 msd ...

# Dump has a 'type' column -- supply symbols in LAMMPS type order
samos traj.lammpstrj --lammps-types Li Ge P S --timestep 2 msd ...

# No type or element column -- supply one symbol per atom
# Accepts a formula string or a space-separated list
samos traj.lammpstrj --lammps-elements Li10GeP2S12 --timestep 2 msd ...
samos traj.lammpstrj --lammps-elements Al Al Al --timestep 2 msd ...
```

### Preprocessing

```bash
# Subtract centre-of-mass motion before analysis
samos traj.extxyz --recenter msd ...

# Derive velocities from positions (required for VAF/VDOS when the
# trajectory file does not store velocities)
samos traj.extxyz --compute-velocities vaf ...

# Relabel all atoms as a single species before analysis
samos traj.lammpstrj --lammps-types Li Ge P S \
    --transform-species Li msd ...
```

### MSD

```bash
# Fit window 2-4 ps, 6 blocks, Li and O species
samos traj.extxyz --species Li O -n 6 msd --t-start-fit 2 --t-end-fit 4

# Specify fit window in femtoseconds
samos traj.extxyz msd --t-start-fit 2000 --t-end-fit 4000 --t-unit fs

# Fit window in timesteps
samos traj.extxyz msd --t-start-fit 160 --t-end-fit 320 --t-unit dt

# C++ backend with 4 OpenMP threads; write results to CSV
samos traj.extxyz msd -b cpp -n 4 --write msd.csv --savefig msd.png

# LAMMPS dump, Li species, 3 blocks
samos traj.lammpstrj --lammps-types Li Ge P S --timestep 1000 \
    --species Li -n 3 \
    msd --t-start-fit 50 --t-end-fit 100
```

### VAF

VAF and VDOS require velocities.  If the trajectory file does not store
them, add `--compute-velocities` to derive them from positions, if every timestep was stored (otherwise velocities cannot be re-computed)

```bash
# 12 blocks, integral-averaging window 2-4 ps, max lag time 5 ps
samos traj.extxyz -n 12 vaf --t-start-fit 2 --t-end-fit 4 --t-end 5

# extxyz without velocities: derive from positions first
samos traj.extxyz --compute-velocities --timestep 2 \
    -n 4 vaf --t-start-fit 1 --t-end-fit 5

# Remove rigid-body angular momentum
samos traj.extxyz --compute-velocities \
    vaf --remove-angular-momentum --t-start-fit 1 --t-end-fit 5

# Write VAF to CSV and save plot
samos traj.extxyz --compute-velocities \
    vaf --write vaf.csv --savefig vaf.png
```

### VDOS (vibrational density of states)

```bash
# 4 blocks, smoothing kernel width of 3 bins
samos traj.extxyz --compute-velocities -n 4 vdos --smoothing 3

# Plot interactively
samos traj.extxyz --compute-velocities vdos --plot

# Save to file
samos traj.extxyz --compute-velocities vdos --savefig vdos.png
```

### RDF

```bash
# All pairs involving Li, radius 6 A
samos traj.extxyz --species Li rdf --radius 6

# Explicit pairs, custom bin count
samos traj.extxyz rdf --species-pairs Li-O O-O --bins 200

# LAMMPS dump
samos traj.lammpstrj --lammps-types Li Ge P S --timestep 1000 \
    rdf --radius 6 --savefig rdf.png
```

### Global options

| Flag | Description |
|------|-------------|
| `--timestep FS` | Override the trajectory timestep (fs) |
| `--lammps` | Read as LAMMPS dump (file must have an `element` column) |
| `--lammps-types SYM ...` | Read as LAMMPS dump; map integer types to symbols in order |
| `--lammps-elements SYM...\|FORMULA` | Read as LAMMPS dump; assign symbols per atom or via formula |
| `--species SYM ...` | Restrict analysis to these chemical symbols |
| `-n N` / `--nblocks N` | Split trajectory into N blocks |
| `--recenter` | Subtract centre-of-mass motion before analysis |
| `--compute-velocities` | Derive velocities from positions (Verlet formula) |
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

# MSD -- fit window and block length both in ps
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
# OLD -- no longer accepted; raises InputError
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
