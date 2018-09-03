from setuptools import find_packages
from numpy.distutils.core import setup, Extension
from json import load as json_load

ext1 = Extension(
        name = 'samos.lib.gaussian_density',
        sources = ['samos/lib/gaussian_density.f90'],
    )
ext2 = Extension(
        name = 'samos.lib.mdutils',
        sources = ['samos/lib/mdutils.f90'],
    )
ext3 = Extension(
        name = 'samos.lib.rdf',
        sources = ['samos/lib/rdf.f90'],
    )

if __name__ == '__main__':
    with open('setup.json', 'r') as info:
        kwargs = json_load(info)
    setup(
        include_package_data=True,
        packages=find_packages(),
        package_data = {'': ['*.f90']},
        ext_modules = [ext1, ext2, ext3],
        **kwargs
    )
