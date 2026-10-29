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

# What each command is about to do is printed rather than left in a
# comment, so that a run can be followed as it happens: the analyzers
# print progress of their own in between, and an unlabelled wall of it
# says little about which analysis produced which figure.
say() {
    echo
    echo "=== $1 ==="
    shift
    printf '%s\n' "$@"
    echo
}

# ------------------------------------------------------------------
# ex1 - MSD
# ------------------------------------------------------------------
say "MSD (ex1)" \
    "Mean-squared displacement of 31 Al atoms at 1200 K, from a LAMMPS" \
    "dump. --lammps-elements names the elements and is also what" \
    "selects the LAMMPS reader; --timestep is in fs, so 1000 says the" \
    "frames are 1 ps apart, overriding whatever the file claims." \
    "The slope is fitted between 50 and 100 ps, and the spread over 4" \
    "blocks gives the error bar." \
    "Writes msd.png and msd.csv."

samos-msd "$DATA/Al31-1200K-1ps.lammpstrj" \
    --timestep 1000 \
    --lammps-elements Al31 \
    -n 4 \
    --savefig msd.png \
    --write msd.csv \
    --t-start-fit 50 --t-end-fit 100 --t-unit ps

if [ -f "$DATA/LGPS-500K-1ns.lammpstrj" ]; then
    say "MSD of Li in LGPS" \
        "The same analysis on a real solid electrolyte: 1 ns of" \
        "Li10GeP2S12 at 500 K. This dump stores numeric atom types" \
        "rather than element names, so --lammps-types gives the" \
        "symbols in type order. --species Li restricts the analysis to" \
        "the mobile ion; the framework does not diffuse and would only" \
        "dilute the result." \
        "Writes msd-lgps.png and msd-lgps.csv."

    samos-msd "$DATA/LGPS-500K-1ns.lammpstrj" \
        --lammps-types Li Ge P S \
        --timestep 1000 \
        --species Li \
        -n 3 \
        --savefig msd-lgps.png \
        --write msd-lgps.csv \
        --t-start-fit 50 --t-end-fit 100 --t-unit ps

else
    say "MSD of Li in LGPS - skipped" \
        "That trajectory ships compressed. To include this part:" \
        "  tar -xvf \"$DATA/LGPS-500K-1ns.lammpstrj.tar.xz\" -C \"$DATA\""
fi

# ------------------------------------------------------------------
# ex2 - VAF and VDOS
# ------------------------------------------------------------------
say "VAF (ex2)" \
    "Velocity autocorrelation function of the same Al system, this" \
    "time from an extxyz file sampled every 2 fs -- fine enough to" \
    "resolve the vibrations. The file stores positions only, so" \
    "--compute-velocities derives velocities from them with the Verlet" \
    "finite-difference formula. Integrating the VAF over 1-5 ps gives" \
    "a diffusion coefficient arrived at independently of the MSD" \
    "above; the two should agree." \
    "Writes vaf.png and vaf.csv."

samos-vaf "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    -n 4 \
    --savefig vaf.png \
    --write vaf.csv \
    --t-start-fit 1 --t-end-fit 5 --t-unit ps

say "VDOS (ex2)" \
    "The vibrational density of states: the Fourier transform of that" \
    "VAF, so its peaks are the vibrational modes of the system." \
    "Run twice on purpose -- first raw, then smoothed with a kernel 10" \
    "bins wide, which trades resolution for a readable curve." \
    "Writes vdos.png, vdos.csv and vdos-smoothed.png."

samos-vdos "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    --savefig vdos.png \
    --write vdos.csv

samos-vdos "$DATA/Al31-1200K-2fs.extxyz" \
    --timestep 2 \
    --compute-velocities \
    --savefig vdos-smoothed.png \
    --smoothing 10

# ------------------------------------------------------------------
# ex3 - RDF
# ------------------------------------------------------------------
say "RDF (ex3)" \
    "Radial distribution function out to 4 A, with the running" \
    "neighbour count alongside it. --lammps reads the element names" \
    "from the dump's own element column, and --transform-species Al" \
    "relabels every atom so that one all-to-all g(r) comes out rather" \
    "than a curve per species pair. No timestep is needed, or" \
    "accepted: the RDF treats every frame on its own." \
    "Writes rdf.png and rdf.csv."

samos-rdf "$DATA/Al31-1200K-1ps.lammpstrj" \
    --lammps --transform-species Al \
    --savefig rdf.png \
    --write rdf.csv \
    --radius 4

say "Done." \
    "The figures and CSV files are in this directory."
