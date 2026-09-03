#!/usr/bin/env bash
# run.sh - reproduce examples 1-3 using the samos command-line tools.
#
# Each block mirrors the corresponding Python example:
#   ex1  MSD from a LAMMPS dump (Al, 1 ps sampling timestep)
#   ex2  VAF and VDOS from an extxyz file (Al, 2 fs timestep)
#   ex3  RDF from a LAMMPS dump (Al, 1 ps sampling timestep)
#
# Every analysis is its own command; options may be given in any order
# after the trajectory path.  "samos msd ..." is the same as
# "samos-msd ...".
#
# Run from this directory:
#   bash run.sh
#
# Results (PNG figures and CSV files) are written to this directory.

set -euo pipefail

DATA=../data

# ------------------------------------------------------------------
# ex1 - MSD
# ------------------------------------------------------------------
echo "=== MSD (ex1) ==="

# Mirrors compute-msd.py: Al, 4 blocks, fit window 50-100 ps.
# --lammps-elements selects read_lammps_dump and sets the element list;
# --timestep overrides the value stored in the file.
samos-msd "$DATA/Al31-1200K-1ps.lammpstrj" \
    --timestep 1000 \
    --lammps-elements Al31 \
    -n 4 \
    --savefig msd.png \
    --write msd.csv \
    --t-start-fit 50 --t-end-fit 100 --t-unit ps

# Mirrors benchmark-msd.py: Li in LGPS, 3 blocks, C++ backend.
# Uncompress the archive first if the file is missing:
#   tar -xvf "$DATA/LGPS-500K-1ns.lammpstrj.tar.xz" -C "$DATA"
if [ -f "$DATA/LGPS-500K-1ns.lammpstrj" ]; then
    echo "=== MSD LGPS (ex1 benchmark, fortran) ==="
    samos-msd "$DATA/LGPS-500K-1ns.lammpstrj" \
        --lammps-types Li Ge P S \
        --timestep 1000 \
        --species Li \
        -n 3 \
        --savefig msd-lgps-fortran.png \
        --write msd-lgps.csv \
        --t-start-fit 50 --t-end-fit 100 --t-unit ps

    echo "=== MSD LGPS (ex1 benchmark, C++ 4 threads) ==="
    # -n is the block count; OpenMP threads are -j.
    samos-msd "$DATA/LGPS-500K-1ns.lammpstrj" \
        --lammps-types Li Ge P S \
        --timestep 1000 \
        --species Li \
        -n 3 \
        --savefig msd-lgps-cpp.png \
        --t-start-fit 50 --t-end-fit 100 --t-unit ps \
        --backend cpp -j 4
else
    echo "LGPS trajectory not found, skipping (see comment above)."
fi

# ------------------------------------------------------------------
# ex2 - VAF and VDOS
# ------------------------------------------------------------------
echo "=== VAF (ex2) ==="

# Mirrors compute-vaf.py: Al, 4 blocks, fit window 1-5 ps.
# The extxyz file has no velocities stored; --compute-velocities
# derives them from positions via the Verlet finite-difference formula.
samos-vaf "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    -n 4 \
    --savefig vaf.png \
    --write vaf.csv \
    --t-start-fit 1 --t-end-fit 5 --t-unit ps

echo "=== VDOS (ex2) ==="

# No smoothing.
samos-vdos "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    --savefig vdos.png \
    --write vdos.csv

# With smoothing (kernel width 10 bins), no CSV needed.
samos-vdos "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    --savefig vdos-smoothed.png \
    --smoothing 10

# ------------------------------------------------------------------
# ex3 - RDF
# ------------------------------------------------------------------
echo "=== RDF (ex3) ==="

# Mirrors compute-rdf.py: Al-Al pair, radius 6 A.
samos-rdf "$DATA/Al31-1200K-1ps.lammpstrj" \
    --lammps-types Al \
    --timestep 1000 \
    --savefig rdf.png \
    --write rdf.csv \
    --radius 6

echo "=== Done. ==="
