# -*- coding: utf-8 -*-
"""
Utilities for converting user-supplied time values to integer timesteps.

The unit registry (_FS_PER_UNIT) maps unit names to their equivalent
number of femtoseconds.  Adding a new unit requires only one new entry
here; no changes to call sites are needed.

The special pseudo-unit 'dt' is not in the registry because it requires
no conversion: the value is already expressed as an integer number of
trajectory timesteps.
"""

import numpy as np

from samos.utils.exceptions import InputError


# Registry: unit name -> femtoseconds per unit.
# Extend here to support new units (e.g. 'ns': 1e6).
_FS_PER_UNIT = {
    'fs': 1.0,
    'ps': 1000.0,
}

# All valid unit strings, including the special 'dt' pseudo-unit.
VALID_UNITS = sorted(_FS_PER_UNIT) + ['dt']


def parse_time(value, unit, timestep_fs):
    """
    Convert a time *value* expressed in *unit* to an integer number of
    trajectory timesteps.

    Supports scalar values as well as lists/arrays so that multi-window
    fit parameters (e.g. a list of t_end_fit values) are handled
    uniformly.

    :param value:
        Numeric time value.  May be a scalar, list, or array.
    :param str unit:
        Time unit.  One of the keys in VALID_UNITS ('fs', 'ps', 'dt').
        'dt' is a pseudo-unit meaning the value is already in timesteps;
        no conversion is applied.
    :param float timestep_fs:
        Trajectory timestep in femtoseconds.  Only used for 'fs' and
        'ps'; ignored for 'dt'.
    :returns:
        Integer timestep count (``int``) for scalar input, or a
        ``numpy.ndarray`` of dtype ``int`` for array input.
    :raises InputError:
        If *unit* is not one of the recognised unit strings.

    To add a new unit (e.g. nanoseconds), add ``'ns': 1e6`` to
    ``_FS_PER_UNIT``.  This function requires no further modification.
    """
    is_scalar = np.ndim(value) == 0

    if unit == 'dt':
        if is_scalar:
            return int(value)
        return np.asarray(value, dtype=int)

    if unit not in _FS_PER_UNIT:
        raise InputError(
            "Unknown time unit '{}'. Valid units are: {}.".format(
                unit, ', '.join(VALID_UNITS)))

    value_fs = _FS_PER_UNIT[unit] * np.asarray(value, dtype=float)
    result = np.rint(value_fs / timestep_fs).astype(int)

    if is_scalar:
        return int(result)
    return result
