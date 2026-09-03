# -*- coding: utf-8 -*-
"""
Physical unit conversion factors to samos internal units.

Internal units
--------------
  positions/lengths -- Angstrom (A)
  velocities        -- A/fs
  forces            -- eV/A
  energy            -- eV
  stress            -- eV/A^3  (1 eV/A^3 = 160.2177 GPa)
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
  1 Pa   = 6.2415e-12 eV/A^3  (inverse of 1.602177e11 Pa per eV/A^3)

A note on ``s_conv``, because the numbers look inconsistent at a glance:
LAMMPS picks the pressure unit of each system independently of its
energy and length units.  For 'si', 'cgs', 'micro' and 'nano' the
pressure unit happens to be that system's energy-per-volume, so
``s_conv == e_conv / l_conv**3``.  For 'real' (atm), 'metal' (bar) and
'electron' (Pa) it does not, and deriving those three from e_conv and
l_conv would be wrong by factors between 1e5 and 1e12.  Hence every
value is written out rather than computed.

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
        's_conv': 6.3242e-7,      # atm -> eV/A^3: 101325 Pa
    },
    # metal: distance A, time ps, energy eV, velocity A/ps
    'metal': {
        'l_conv': 1.0,
        'v_conv': 1e-3,
        'f_conv': 1.0,
        'e_conv': 1.0,
        's_conv': 6.2415e-7,      # bar -> eV/A^3: 1e5 Pa
    },
    # si: distance m, time s, energy J, velocity m/s
    'si': {
        'l_conv': 1e10,         # m -> A
        'v_conv': 1e-5,         # m/s -> A/fs: 1e10 A / 1e15 fs
        'f_conv': 6.2415e8,     # N -> eV/A: 6.2415e18 eV / 1e10 A
        'e_conv': 6.2415e18,    # J -> eV
        's_conv': 6.2415e-12,     # Pa -> eV/A^3
    },
    # cgs: distance cm, time s, energy erg, velocity cm/s
    'cgs': {
        'l_conv': 1e8,          # cm -> A
        'v_conv': 1e-7,         # cm/s -> A/fs: 1e8 A / 1e15 fs
        'f_conv': 6.2415e3,     # dyne -> eV/A: 1e-5 N * 6.2415e8
        'e_conv': 6.2415e11,    # erg -> eV: 1e-7 J * 6.2415e18
        's_conv': 6.2415e-13,     # dyne/cm^2 -> eV/A^3: 0.1 Pa
    },
    # electron: distance Bohr, time atu, energy Ha, velocity Bohr/atu
    'electron': {
        'l_conv': bohr_to_ang,               # Bohr -> A
        'v_conv': bohr_to_ang / 0.0241888,   # Bohr/atu -> A/fs ~ 21.877
        'f_conv': 27.2114 / bohr_to_ang,     # Ha/Bohr -> eV/A ~ 51.42
        'e_conv': 27.2114,                   # Ha -> eV
        's_conv': 6.2415e-12,     # Pa -> eV/A^3 (not Ha/Bohr^3)
    },
    # micro: distance micron, time microsecond, velocity micron/us = m/s
    'micro': {
        'l_conv': 1e4,          # micron -> A
        'v_conv': 1e-5,         # micron/us = m/s -> A/fs
        'f_conv': 6.2415e-1,    # pg*micron/us^2 = 1e-9 N -> eV/A
        'e_conv': 6.2415e3,     # pg*micron^2/us^2 = 1e-15 J -> eV
        's_conv': 6.2415e-9,      # pg/(um*us^2) = 1e3 Pa -> eV/A^3
    },
    # nano: distance nm, time ns, velocity nm/ns = m/s
    'nano': {
        'l_conv': 10.0,         # nm -> A
        'v_conv': 1e-5,         # nm/ns = m/s -> A/fs
        'f_conv': 6.2415e-4,    # ag*nm/ns^2 = 1e-12 N -> eV/A
        'e_conv': 6.2415e-3,    # ag*nm^2/ns^2 = 1e-21 J -> eV
        's_conv': 6.2415e-6,      # ag/(nm*ns^2) = 1e6 Pa -> eV/A^3
    },
}
