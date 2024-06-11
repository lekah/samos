# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup, Extension
from json import load as json_load
from setuptools.command.build_ext import build_ext

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

class f2py_Extension(Extension):

    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = [os.path.abspath(sourcedir) for sourcedir in sourcedirs]
        self.dirs = sourcedirs

class f2py_Build(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        for index, to_compile in enumerate(ext.sourcedirs):
            module_loc = os.path.split(ext.dirs[index])[0]
            module_name = os.path.split(to_compile)[1].split('.')[0]
            os.system('cd %s;f2py -c %s -m %s' % (module_loc,to_compile,module_name))

if __name__ == '__main__':
    with open('setup.json', 'r') as info:
        kwargs = json_load(info)
    setup(
        include_package_data=True,
        packages=find_packages(),
        package_data = {'': ['*.f90']},
        ## Following inexplicably works even though there is single name for all 3 f90 files
        ext_modules = [f2py_Extension('fortran_lib', ['samos/lib/gaussian_density.f90', 'samos/lib/mdutils.f90', 'samos/lib/rdf.f90'])],
        cmdclass=dict(build_ext=f2py_Build),
        **kwargs
    )
