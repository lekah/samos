from json import dumps
import numpy as np


class AttributedArray(object):
    _ATTRIBUTE_FILENAME = 'attributes.json'

    def __init__(self, **kwargs):
        self._arrays = {}
        self._attrs = {}

        for key, val in kwargs.items():
            getattr(self, 'set_{}'.format(key))(val)

    def set_array(self, name, array, check_existing=False, check_nstep=False, check_nat=False):
        """
        Method to set an array with a name to reference it.
        :param str name: A name to reference that array
        :param array: A valid numpy array or an object that can be converted with a call to numpy.array
        :param bool check_existing:
            Check for an array of that name existing, and raise if it exists.
            Defaults to False.
        :param book check_nstep:
            Check if the number of steps, which is the first dimension of the array, is commensurate
            with other arrays stored. Defaults to False
        :param bool check_nat:
            If the array is of rank 3 or higher, the second dimension is interpreted as the number of atoms.
            If this flag is True, I will check for arrays with rank 3 or higher. Defaults  to True.
            Requires that the atoms have been set
        """
        # First, I call np.array to ensure it's a valid array
        array = np.array(array)
        if not isinstance(name, basestring):
            raise TypeError("Name has to be a string")
        if check_existing:
            if name in self._arrays.keys():
                raise ValueError("Name {} already exists".formamt(name))
        if check_nstep:
            for other_name, other_array in self._arrays.items():
                assert array.shape[0] == other_array.shape[0], (
                    'Number of steps in array {} ({}) is not compliant with array {} ({})'.format(
                            name, array.shape[0],other_name,  other_array.shape[0]))
        if check_nat and len(array.shape) > 2:
            if array.shape[1] != len(self.atoms):
                raise ValueError("Second dimension of array does not match the number of atoms")
        self._arrays[name] = array


    def get_array(self, name):
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError("An array with that name ( {} ) has not been set.".format(name))

    def get_arraynames(self):
        return sorted(self._arrays.keys())


    def get_attrs(self):
        return self._attrs

    def get_attr(self, key):
        return self._attrs[key]

    def set_attr(self, key, value):
        #Testing whether this is valid:
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
        :param str filename: The filename. Won't be checked or modified with extension!
        """
        import tarfile, tempfile
        from inspect import getmembers, ismethod
        
        temp_folder = tempfile.mkdtemp()
        for funcname, func in getmembers(self, predicate=ismethod):
            if funcname.startswith('_save_'):
                func(temp_folder)

        with tarfile.open(filename, "w:gz", format=tarfile.PAX_FORMAT) as tar:
            tar.add(temp_folder, arcname="")

    def _save_arrays(self, folder_name):
        from os.path import join
        for arrayname, array in self._arrays.items():
            np.save(join(folder_name, '{}.npy'.format(arrayname)), array)

    


    def _save_attributes(self, folder_name):
        from os.path import join
        import json

        with open(join(folder_name, self._ATTRIBUTE_FILENAME), 'w') as f:
            json.dump(self._attrs, f)

    @classmethod
    def load_file(cls, filename):
        """
        Given a filename, try to load the trajectories and return a new instance of the class.
        The filename should ideally be created with the Trajectore.store method.
        If created by hand, it has to be a valid tar.gz compressed tar.
        """
        import tarfile, tempfile, json, os
        from os.path import join
        temp_folder = tempfile.mkdtemp()
        with tarfile.open(filename, "r:gz", format=tarfile.PAX_FORMAT) as tar:
            tar.extractall(temp_folder)

        files_in_tar = set(os.listdir(temp_folder))

        with open(join(temp_folder, cls._ATTRIBUTE_FILENAME)) as f:
            attributes = json.load(f)
        files_in_tar.remove(cls._ATTRIBUTE_FILENAME)
        new = cls()
        for k,v in attributes.items():
            new.set_attr(k, v)

        if cls._ATOMS_FILENAME in files_in_tar:
            from ase.io import read
            new.set_atoms(read(join(temp_folder, cls._ATOMS_FILENAME)))
            files_in_tar.remove(cls._ATOMS_FILENAME)

        for array_file in files_in_tar:
            if not array_file.endswith('.npy'):
                raise Exception("Unrecognized file in trajectory export: {}".format(array_file))
            new.set_array(array_file.rstrip('.npy'), np.load(join(temp_folder, array_file)))
        return new

