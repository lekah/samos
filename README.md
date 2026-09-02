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
| `ex5-using-the-script/` | All of the above reproduced with the command-line tools |

Run `bash examples/ex5-using-the-script/run.sh` from the repository root
to see every command in action.

---

## Command-line usage

Every analysis is its own command, taking a trajectory path followed by
its options:

```
samos-msd TRAJECTORY [options]
```

| Command | What it computes |
|---------|------------------|
| `samos-msd` | Mean-square displacement and the diffusion coefficient |
| `samos-vaf` | Velocity autocorrelation function and its integral |
| `samos-vdos` | Vibrational density of states (Welch periodogram) |
| `samos-rdf` | Radial distribution function and its running integral |
| `samos-adf` | Angular distribution function over bond triplets |

`samos` is a dispatcher for the same commands, so that `samos` on its
own lists what is available:

```bash
samos            # list the commands
samos msd ...    # identical to samos-msd ...
```

Options may be given in any order, with one exception: **put the
trajectory path first**.  A list-valued option such as `--species Li O`
placed immediately before the path swallows it, because argparse cannot
tell where the list ends and the path begins.

Time values are plain floats; the unit is set with `--t-unit`
(choices: `fs`, `ps`, `dt`; default: `ps`).

### Options accepted by every command

| Flag | Description |
|------|-------------|
| `--timestep FS` | Override the trajectory timestep (fs) |
| `--lammps` | Read as LAMMPS dump (file must have an `element` column) |
| `--lammps-types SYM ...` | Read as LAMMPS dump; map integer types to symbols in order |
| `--lammps-elements SYM...\|FORMULA` | Read as LAMMPS dump; assign symbols per atom or via formula |
| `--units SYSTEM` | Convert arrays from a named unit system to samos internal units |
| `-i/--index SLICE` | Analyse only these frames, e.g. `::10` or `500:1500:2` |
| `--species SYM ...` | Restrict analysis to these chemical symbols |
| `--recenter` | Subtract centre-of-mass motion before analysis |
| `--compute-velocities` | Derive velocities from positions (Verlet formula) |
| `--transform-species SYM` | Relabel all atoms as SYM |
| `--write FILE` | Write results to a CSV file |
| `--plot` | Show the plot interactively |
| `--savefig FILE` | Save the plot to FILE |

`-n/--nblocks N` splits the trajectory into N blocks and is accepted by
`samos-msd`, `samos-vaf` and `samos-vdos`.  It is not accepted by
`samos-rdf` and `samos-adf`, which do not block-average.

### Reading LAMMPS dump files

LAMMPS dump files are not auto-detected.  Use one of these flags to
tell samos how to resolve element names:

```bash
# Dump has an 'element' column -- no element list needed
samos-msd traj.lammpstrj --lammps --timestep 2

# Dump has a 'type' column -- supply symbols in LAMMPS type order
samos-msd traj.lammpstrj --lammps-types Li Ge P S --timestep 2

# No type or element column -- supply one symbol per atom
# Accepts a formula string or a space-separated list
samos-msd traj.lammpstrj --lammps-elements Li10GeP2S12 --timestep 2
samos-msd traj.lammpstrj --lammps-elements Al Al Al --timestep 2
```

Preprocessing applies to every command:

```bash
# Remove centre-of-mass drift
samos-msd traj.extxyz --recenter

# Derive velocities from positions (needed for VAF and VDOS when the
# trajectory does not store them)
samos-vaf traj.extxyz --compute-velocities
```

### Selecting frames

`-i/--index` takes a Python slice and restricts the analysis to those
frames:

```bash
# First 1000 frames
samos-msd traj.extxyz --index :1000

# Every 10th frame
samos-rdf traj.extxyz --index ::10 --radius 6

# Every 2nd frame between 500 and 1500
samos-msd traj.extxyz --index 500:1500:2
```

A stride multiplies the timestep by the same factor, so time axes stay
in real femtoseconds: `--index ::10` on a 2 fs trajectory yields lags
spaced 20 fs apart, not 2 fs.

Combining a stride with `--compute-velocities` derives velocities from
the retained frames only, across the wider spacing.  That is consistent
but coarser than differencing every frame, and the commands warn when
you do it.

### MSD

```bash
# Fit window 2-4 ps, 6 blocks, Li and O species
samos-msd traj.extxyz --species Li O -n 6 --t-start-fit 2 --t-end-fit 4

# Specify fit window in femtoseconds
samos-msd traj.extxyz --t-start-fit 2000 --t-end-fit 4000 --t-unit fs

# Fit window in timesteps
samos-msd traj.extxyz --t-start-fit 160 --t-end-fit 320 --t-unit dt

# C++ backend with 4 OpenMP threads; write results to CSV
samos-msd traj.extxyz --backend cpp -j 4 --write msd.csv --savefig msd.png

# LAMMPS dump, Li species, 3 blocks
samos-msd traj.lammpstrj --lammps-types Li Ge P S --timestep 1000 \
    --species Li -n 3 --t-start-fit 50 --t-end-fit 100
```

Command options: `-s/--stepsize N`, `-ts/--t-start-fit T`,
`-te/--t-end-fit T`, `--t-unit UNIT`, `--backend {fortran,cpp}`,
`-j/--num-threads N`.

### VAF

VAF and VDOS require velocities.  If the trajectory file does not store
them, add `--compute-velocities` to derive them from positions -- which
is only possible if every timestep was written.

```bash
# 12 blocks, integral-averaging window 2-4 ps, max lag time 5 ps
samos-vaf traj.extxyz -n 12 --t-start-fit 2 --t-end-fit 4 --t-end 5

# extxyz without velocities: derive from positions first
samos-vaf traj.extxyz --compute-velocities --timestep 2 \
    -n 4 --t-start-fit 1 --t-end-fit 5

# Remove rigid-body angular momentum
samos-vaf traj.extxyz --compute-velocities \
    --remove-angular-momentum --t-start-fit 1 --t-end-fit 5

# Write VAF to CSV and save plot
samos-vaf traj.extxyz --compute-velocities --write vaf.csv --savefig vaf.png
```

Command options: `-s/--stepsize N`, `-ts/--t-start-fit T`,
`-te/--t-end-fit T`, `--t-end T`, `--t-unit UNIT`,
`--integration {trapezoid,simpson}`, `-a/--remove-angular-momentum`.

### VDOS (vibrational density of states)

```bash
# 4 blocks, smoothing kernel width of 3 bins
samos-vdos traj.extxyz --compute-velocities -n 4 --smoothing 3

# Plot interactively
samos-vdos traj.extxyz --compute-velocities --plot

# Save to file
samos-vdos traj.extxyz --compute-velocities --savefig vdos.png
```

Command options: `-sm/--smoothing N`, `-a/--remove-angular-momentum`.

### RDF

```bash
# All pairs involving Li, radius 6 A
samos-rdf traj.extxyz --species Li --radius 6

# Explicit pairs, custom bin count
samos-rdf traj.extxyz --species-pairs Li-O O-O --bins 200

# LAMMPS dump
samos-rdf traj.lammpstrj --lammps-types Li Ge P S --timestep 1000 \
    --radius 6 --savefig rdf.png
```

Command options: `-s/--stepsize N`, `-r/--radius A`, `-b/--bins N`,
`--species-pairs A-B ...`, `--no-int`.  `--species` and
`--species-pairs` are mutually exclusive.

### ADF

The ADF needs a bond definition: a global cutoff, per-pair cutoffs, or
an explicit topology from a LAMMPS data file.

```bash
# Every pair within 3.2 A counts as a bond
samos-adf traj.extxyz --radius 3.2 --centers Si

# Per-bond cutoffs, explicit triplets (centre species in the middle)
samos-adf traj.extxyz --bonds Si-O:1.4:2.0 Al-O:1.6:2.2 \
    --species-triplets O-Si-O O-Al-O --write adf.csv

# Topology from a LAMMPS data file, detected once from the first frame
samos-adf traj.lammpstrj --lammps --bonds-file system.data --static-bonds
```

Command options: `-s/--stepsize N`, `-b/--bins N`, `-r/--radius A`,
`--bonds SPEC:RMIN:RMAX ...`, `--bonds-file FILE`, `--static-bonds`,
`--centers SYM ...`, `--species-triplets A-B-C ...`.  The three bond
sources are mutually exclusive and one is required; `--centers` and
`--species-triplets` are mutually exclusive.

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

### RDF normalisation

`RDF.run` now normalises each sampled frame by its own number density
and averages the result, so `g(r)` is the mean of the per-frame `g(r)`.
It previously divided the summed histogram by whichever frame happened
to be sampled last, which left the RDF of a variable-cell (NPT)
trajectory dependent on the frame order.  Fixed-cell results are
unchanged by this.

The ideal-gas reference now excludes the centre atom, so a species
paired with itself is normalised by `N-1` instead of `N`.  Like-pair
RDFs therefore rise by a factor `N/(N-1)` -- 1 % for 100 atoms of a
species, 11 % for 10 -- and now tend to 1 at large `r` rather than to
`(N-1)/N`.  Unlike-pair RDFs and all `int_*` running integrals are
unaffected.

### CLI structure

The single `samos TRAJECTORY [global options] COMMAND [command options]`
script has been replaced by one command per analysis.  Options no longer
have to be split across the command name:

```bash
# OLD
samos traj.extxyz --species Li -n 6 msd --t-start-fit 2 --t-end-fit 4

# NEW (samos msd ... is equivalent)
samos-msd traj.extxyz --species Li -n 6 --t-start-fit 2 --t-end-fit 4
```

Two short flags changed meaning, because they used to denote different
things on either side of the command name:

| Old | New |
|---|---|
| `msd -n N` (OpenMP threads) | `-j/--num-threads N`; `-n` is always `--nblocks` |
| `msd -b BACKEND` | `--backend BACKEND`; `-b` is always `--bins` |
| `vaf -i METHOD` | `--integration METHOD` |

`-n/--nblocks` is no longer accepted by `samos-rdf` and `samos-adf`,
which never used it.

The `run_msd`, `run_vaf`, `run_vdos`, `run_rdf` and `run_adf` functions
that back these commands now live in the importable `samos.cli` module,
together with `load_trajectory`.

### CLI time flags

Old per-parameter flags have been replaced:

| Removed flag | Replacement |
|---|---|
| `--t-start-fit-ps` | `--t-start-fit` (with `--t-unit ps`) |
| `--t-end-fit-ps` | `--t-end-fit` (with `--t-unit ps`) |
| `--t-end-ps` (VAF) | `--t-end` (with `--t-unit ps`) |

The `--t-unit` flag is per command and defaults to `ps`.

### `_get_running_params` return value

Previously returned a 14-element positional tuple.  Now returns a
`RunningParams` namedtuple; callers must access fields by name
(e.g. `rp.t_start_fit_dt` instead of `rp[5]`).
