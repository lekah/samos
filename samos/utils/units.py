# -*- coding: utf-8 -*-
"""
Physical unit conversion factors to samos internal units.

Internal units
--------------
  positions/lengths -- Angstrom (A)
  velocities        -- A/fs
  forces            -- eV/A
  energy            -- eV
  time              -- fs  (see also time_units.py)

``UNIT_SYSTEMS`` maps named unit system strings to dicts of conversion
factors.  Multiplying a raw value by the corresponding factor gives the
value in samos internal units.

The names follow LAMMPS convention but the table is format-agnostic and
applies to any trajectory file that stores data in those units.

Physical constants used
-----------------------
  1 eV  = 23.0609 kcal/mol  (NIST 2018)
  1 eV  = 27.2114 Ha
  1 Bohr = 0.52917721092 A  (samos.utils.constants.bohr_to_ang)
  1 atu  = 0.0241888 fs  (Hartree atomic time unit)

Sources: https://docs.lammps.org/units.html
"""

from samos.utils.constants import bohr_to_ang

UNIT_SYSTEMS = {
    # real: distance A, time fs, energy kcal/mol, velocity A/fs
    'real': {
        'l_conv': 1.0,
        'v_conv': 1.0,
        'f_conv': 1.0 / 23.0609,
        'e_conv': 1.0 / 23.0609,
        's_conv': 1.0,
    },
    # metal: distance A, time ps, energy eV, velocity A/ps
    'metal': {
        'l_conv': 1.0,
        'v_conv': 1e-3,
        'f_conv': 1.0,
        'e_conv': 1.0,
        's_conv': 1.0,
    },
    # si: distance m, time s, energy J, velocity m/s
    'si': {
        'l_conv': 1e10,         # m -> A
        'v_conv': 1e-5,         # m/s -> A/fs: 1e10 A / 1e15 fs
        'f_conv': 6.2415e8,     # N -> eV/A: 6.2415e18 eV / 1e10 A
        'e_conv': 6.2415e18,    # J -> eV
        's_conv': 1.0,
    },
    # cgs: distance cm, time s, energy erg, velocity cm/s
    'cgs': {
        'l_conv': 1e8,          # cm -> A
        'v_conv': 1e-7,         # cm/s -> A/fs: 1e8 A / 1e15 fs
        'f_conv': 6.2415e3,     # dyne -> eV/A: 1e-5 N * 6.2415e8
        'e_conv': 6.2415e11,    # erg -> eV: 1e-7 J * 6.2415e18
        's_conv': 1.0,
    },
    # electron: distance Bohr, time atu, energy Ha, velocity Bohr/atu
    'electron': {
        'l_conv': bohr_to_ang,               # Bohr -> A
        'v_conv': bohr_to_ang / 0.0241888,   # Bohr/atu -> A/fs ~ 21.877
        'f_conv': 27.2114 / bohr_to_ang,     # Ha/Bohr -> eV/A ~ 51.42
        'e_conv': 27.2114,                   # Ha -> eV
        's_conv': 1.0,
    },
    # micro: distance micron, time microsecond, velocity micron/us = m/s
    'micro': {
        'l_conv': 1e4,          # micron -> A
        'v_conv': 1e-5,         # micron/us = m/s -> A/fs
        'f_conv': 6.2415e-1,    # pg*micron/us^2 = 1e-9 N -> eV/A
        'e_conv': 6.2415e3,     # pg*micron^2/us^2 = 1e-15 J -> eV
        's_conv': 1.0,
    },
    # nano: distance nm, time ns, velocity nm/ns = m/s
    'nano': {
        'l_conv': 10.0,         # nm -> A
        'v_conv': 1e-5,         # nm/ns = m/s -> A/fs
        'f_conv': 6.2415e-4,    # ag*nm/ns^2 = 1e-12 N -> eV/A
        'e_conv': 6.2415e-3,    # ag*nm^2/ns^2 = 1e-21 J -> eV
        's_conv': 1.0,
    },
}
