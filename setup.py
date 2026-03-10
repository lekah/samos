# -*- coding: utf-8 -*-
# Metadata is declared in pyproject.toml.
# This file handles only the C++ (pybind11) and Fortran (f2py) extension builds.

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension


class f2py_Extension(Extension):

    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = [os.path.abspath(s) for s in sourcedirs]
        self.dirs = sourcedirs


class CombinedBuild(build_ext):
    """Builds f2py (Fortran) and pybind11 (C++) extensions in one pass."""

    def run(self):
        f2py_exts = [e for e in self.extensions if isinstance(e, f2py_Extension)]
        cpp_exts  = [e for e in self.extensions if not isinstance(e, f2py_Extension)]

        # Standard setuptools path initialises the compiler; run for C++ only.
        self.extensions = cpp_exts
        build_ext.run(self)

        # Python < 3.12: numpy.distutils is used by f2py but the stdlib env var
        # bypasses that by using Python's own distutils instead of setuptools.
        # Python >= 3.12: distutils is removed, numpy.f2py uses the meson backend.
        if sys.version_info >= (3, 12):
            env_prefix = ''
        else:
            env_prefix = 'SETUPTOOLS_USE_DISTUTILS=stdlib '

        for ext in f2py_exts:
            for i, src in enumerate(ext.sourcedirs):
                module_loc = os.path.split(ext.dirs[i])[0]
                module_name = os.path.split(src)[1].split('.')[0]
                os.system(
                    'cd %s; %s%s -m numpy.f2py -c %s -m %s'
                    % (module_loc, env_prefix, sys.executable, src, module_name)
                )

        self.extensions = cpp_exts + f2py_exts


setup(
    ext_modules=[
        f2py_Extension('fortran_lib', [
            'samos/lib/gaussian_density.f90',
            'samos/lib/mdutils.f90',
            'samos/lib/rdf.f90',
        ]),
        Pybind11Extension(
            'samos.lib.mdutils_cpp_omp',
            ['samos/lib/mdutils_cpp_omp.cpp'],
            extra_compile_args=['-O3', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass=dict(build_ext=CombinedBuild),
)
