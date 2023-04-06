# -*- coding: utf-8 -*-

from ase.data.colors import jmol_colors, cpk_colors
from ase.data import atomic_numbers

CUSTOM_COLORS = {
    'H': (0, 0, 0)
}


def get_color(chemical_symbol, scheme='jmol'):
    if chemical_symbol in CUSTOM_COLORS:
        return CUSTOM_COLORS[chemical_symbol]
    elif scheme == 'jmol':
        return jmol_colors[atomic_numbers[chemical_symbol]]
    elif scheme == 'cpk':
        return cpk_colors[atomic_numbers[chemical_symbol]]
    else:
        raise ValueError('Unknown scheme {}'.format(scheme))
