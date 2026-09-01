# -*- coding: utf-8 -*-

from json import dumps
import numpy as np
import shutil


class AttributedArray:
    _ATTRIBUTE_FILENAME = 'attributes.json'

    def __init__(self, **kwargs):
        self._arrays = {}
        self._attrs = {}

        self._nstep = None
        for key, val in list(kwargs.items()):
            getattr(self, 'set_{}'.format(key))(val)

    def set_array(
        self, name, array, check_existing=False,
        check_nstep=False, check_nat=False,
            wanted_shape_len=None, wanted_shape_1=None,
            wanted_shape_2=None):
        """
        Method to set an array with a name to reference it.
        :param str name: A name to reference that array
        :param array:
            A valid numpy array or an object that can
            be converted with a call to numpy.array
        :param bool check_existing:
            Check for an array of that name existing,
            and raise if it exists.
            Defaults to False.
        :param bool check_nstep:
            Check if the number of steps, which is the first
            dimension of the array, is commensurate
            with other arrays stored. Defaults to False
        :param check_nat:
            The expected number of atoms, or False to skip the check.
            If the array is of rank 3 or higher, the second
            dimension is interpreted as the number of atoms and
            compared against this value. Defaults to False.
        """
        # First, I call np.array to ensure it's a valid array
        array = np.array(array)
        if not isinstance(name, str):
            raise TypeError('Name has to be a string')
        if check_existing:
            if name in list(self._arrays.keys()):
                raise ValueError('Name {} already exists'.format(name))
        if wanted_shape_len is not None:
            if len(array.shape) != wanted_shape_len:
                raise TypeError(
                    f'array {name} is of wrong type, has to be of '
                    f'dimension {wanted_shape_len}, got '
                    f'{len(array.shape)}')
        # Check the rank before indexing into shape, so that a rank-1
        # array reports what is wrong instead of raising a bare
        # IndexError from the guard itself.
        for idim, wanted in ((1, wanted_shape_1), (2, wanted_shape_2)):
            if wanted is None:
                continue
            if len(array.shape) <= idim:
                raise IndexError(
                    f'array {name} has {len(array.shape)} dimension(s), '
                    f'so dimension {idim} cannot be checked '
                    f'against {wanted}')
            if array.shape[idim] != wanted:
                raise IndexError(
                    f'{idim}. dimension of array {name} has to '
                    f'be {wanted}, got {array.shape[idim]}')
        if check_nstep:
            if self._nstep is None:
                self._nstep = array.shape[0]
            elif self._nstep != array.shape[0]:
                raise ValueError(
                    'Number of steps in array {} ({}) is not '
                    'compliant with number of steps in previous '
                    'arrays ({})'.format(name, array.shape[0],
                                         self._nstep))
        if check_nat and len(array.shape) > 2:
            if not isinstance(check_nat, int):
                raise TypeError(
                    'If check_nat is not False, it has to be an integer')
            if array.shape[1] != check_nat:
                raise ValueError(
                    'Second dimension of array does not '
                    'match the number of atoms')
        self._arrays[name] = array

    def __contains__(self, arrayname):
        return arrayname in self._arrays

    @property
    def nstep(self):
        """
        :returns: The number of trajectory steps
        :raises: ValueError if no unique number of steps can be determined.
        """
        # if self._nstep is None:
        #     raise ValueError('Number of steps has not been set')
        return self._nstep

    def get_array(self, name):
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError(
                'An array with that name ( {} ) has not '
                'been set.'.format(name))

    def get_arraynames(self):
        return sorted(self._arrays.keys())

    def get_attrs(self):
        return self._attrs

    def get_attr(self, key):
        return self._attrs[key]

    def set_attr(self, key, value):
        # Testing whether this is valid:
        dumps(value)
        self._attrs[key] = value

    def rescale_array(self, arrayname, value):
        """
        Rescale the array by a certain value
        """
        self._arrays[arrayname] *= float(value)

    def save(self, filename):
        """
        Saves the trajectory instance to tarfile.
        :param str filename:
            The filename. Won't be checked or modified with extension!
        """
        import tarfile
        import tempfile
        from inspect import getmembers, ismethod

        temp_folder = tempfile.mkdtemp()
        for funcname, func in getmembers(self, predicate=ismethod):
            if funcname.startswith('_save_'):
                func(temp_folder)

        with tarfile.open(filename, 'w:gz',
                          format=tarfile.PAX_FORMAT) as tar:
            tar.add(temp_folder, arcname='')

    def _save_arrays(self, folder_name):
        from os.path import join
        for arrayname, array in list(self._arrays.items()):
            np.save(join(folder_name, '{}.npy'.format(arrayname)), array)

    def _load_extra(self, folder_name, files_in_tar):
        """
        Hook for subclasses to restore members that are not arrays or
        attributes, mirroring their ``_save_*`` methods.

        Implementations must discard the names they consumed from
        *files_in_tar*; whatever remains is loaded as a ``.npy`` array.

        :param str folder_name: The unpacked archive directory.
        :param set files_in_tar: Names still to be accounted for.
        """
        pass

    def remove_array(self, arrayname):
        if arrayname not in self._arrays:
            raise KeyError(f'{arrayname} is not one of arrays')
        del self._arrays[arrayname]

    def _save_attributes(self, folder_name):
        from os.path import join
        import json

        with open(join(folder_name, self._ATTRIBUTE_FILENAME), 'w') as f:
            json.dump(self._attrs, f)

    @classmethod
    def load_file(cls, filename):
        """
        Given a filename, try to load the trajectories and
        return a new instance of the class.
        The filename should ideally be created with the
        save method.
        If created by hand, it has to be a valid tar.gz
        compressed tar.
        """
        import tarfile
        import tempfile
        import json
        import os
        from os.path import join
        temp_folder = tempfile.mkdtemp()

        try:
            with tarfile.open(filename, 'r:gz',
                              format=tarfile.PAX_FORMAT) as tar:
                # filter='data' refuses members with absolute paths or
                # paths escaping temp_folder.  It became the default in
                # Python 3.14; tarfile.data_filter is present exactly
                # when the argument is supported (3.12, backported to
                # 3.10.12 and 3.11.4).
                if hasattr(tarfile, 'data_filter'):
                    tar.extractall(temp_folder, filter='data')
                else:
                    tar.extractall(temp_folder)

            files_in_tar = set(os.listdir(temp_folder))

            with open(join(temp_folder, cls._ATTRIBUTE_FILENAME)) as f:
                attributes = json.load(f)
            files_in_tar.remove(cls._ATTRIBUTE_FILENAME)
            new = cls()
            for k, v in list(attributes.items()):
                new.set_attr(k, v)

            new._load_extra(temp_folder, files_in_tar)

            for array_file in files_in_tar:
                if not array_file.endswith('.npy'):
                    raise Exception(
                        'Unrecognized file in trajectory export: {}'
                        ''.format(array_file))
                new.set_array(array_file.removesuffix('.npy'), np.load(
                    join(temp_folder, array_file), mmap_mode='r'))
        except Exception as e:
            shutil.rmtree(temp_folder)
            raise e
        shutil.rmtree(temp_folder)
        return new
