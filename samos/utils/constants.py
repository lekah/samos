# -*- coding: utf-8 -*-

from ase.units import Bohr, kB as kB_ev

# constants
# kB (SI, J/K) and amu_kg have no public name in ase.units -- only the
# private ase.units._k and ._amu -- so they stay defined here rather
# than reaching into another package's internals.
kB = 1.38064852e-23
kB_au = 3.166810800209422e-06
amu_kg = 1.660539040e-27
bohr_to_ang = Bohr

# Unit conversion for diffusion coefficients:
# 1 A^2/fs = (1e-8 cm)^2 / 1e-15 s = 1e-1 cm^2/s
ANG2_FS_TO_CM2_S = 1e-1
